import path from 'node:path';
import type { RepomixConfigMerged } from '../config/configSchema.js';
import { logger } from '../shared/logger.js';
import { logMemoryUsage, withMemoryLogging } from '../shared/memoryUtils.js';
import { getProcessConcurrency, getWorkerThreadCount } from '../shared/processConcurrency.js';
import type { RepomixProgressCallback } from '../shared/types.js';
import { collectFiles, type SkippedFileInfo } from './file/fileCollect.js';
import { sortPaths } from './file/filePathSort.js';
import { processFiles } from './file/fileProcess.js';
import { searchFiles } from './file/fileSearch.js';
import type { FilesByRoot } from './file/fileTreeGenerate.js';
import type { ProcessedFile } from './file/fileTypes.js';
import { getGitDiffs } from './git/gitDiffHandle.js';
import { getGitLogs } from './git/gitLogHandle.js';
import { calculateMetrics, createMetricsTaskRunner } from './metrics/calculateMetrics.js';
import { prefetchFileChangeCounts } from './output/outputSort.js';
import { copyToClipboardIfEnabled } from './packager/copyToClipboardIfEnabled.js';
import { writeOutputToDisk } from './packager/writeOutputToDisk.js';
import { filterOutUntrustedFiles } from './security/filterOutUntrustedFiles.js';
import { createSecurityTaskRunner, runSecurityCheck, type SuspiciousFileResult } from './security/securityCheck.js';
import { validateFileSafety } from './security/validateFileSafety.js';

// Lazy-load output generation modules to reduce startup critical path.
// These modules import Handlebars (~25ms), template styles (~15ms), and other
// output-specific dependencies that are not needed until output generation begins.
// By deferring them from the static import graph, the defaultAction module preload
// completes ~40ms faster, eliminating the gap between cliRun and the preload.
// The dynamic import starts at the beginning of pack() and overlaps with searchFiles
// (~75ms) and collectFiles (~230ms), hiding the ~40ms import cost entirely.
type GenerateOutputFn = typeof import('./output/outputGenerate.js')['generateOutput'];
type ProduceOutputFn = typeof import('./packager/produceOutput.js')['produceOutput'];
type PackSkillFn = typeof import('./skill/packSkill.js')['packSkill'];

export interface PackResult {
  totalFiles: number;
  totalCharacters: number;
  totalTokens: number;
  fileCharCounts: Record<string, number>;
  fileTokenCounts: Record<string, number>;
  gitDiffTokenCount: number;
  gitLogTokenCount: number;
  outputFiles?: string[];
  suspiciousFilesResults: SuspiciousFileResult[];
  suspiciousGitDiffResults: SuspiciousFileResult[];
  suspiciousGitLogResults: SuspiciousFileResult[];
  processedFiles: ProcessedFile[];
  safeFilePaths: string[];
  skippedFiles: SkippedFileInfo[];
}

const defaultDeps = {
  searchFiles,
  collectFiles,
  processFiles,
  validateFileSafety,
  filterOutUntrustedFiles,
  createSecurityTaskRunner,
  runSecurityCheck,
  produceOutput: null as ProduceOutputFn | null,
  generateOutput: null as GenerateOutputFn | null,
  writeOutputToDisk,
  copyToClipboardIfEnabled,
  calculateMetrics,
  createMetricsTaskRunner,
  sortPaths,
  getGitDiffs,
  getGitLogs,
  packSkill: null as PackSkillFn | null,
  prefetchFileChangeCounts,
};

export interface PackOptions {
  skillName?: string;
  skillDir?: string;
  skillProjectName?: string;
  skillSourceUrl?: string;
}

export const pack = async (
  rootDirs: string[],
  config: RepomixConfigMerged,
  progressCallback: RepomixProgressCallback = () => {},
  overrideDeps: Partial<typeof defaultDeps> = {},
  explicitFiles?: string[],
  options: PackOptions = {},
): Promise<PackResult> => {
  const deps = {
    ...defaultDeps,
    ...overrideDeps,
  };

  logMemoryUsage('Pack - Start');

  // Lazy-load output generation modules (Handlebars, templates, etc.) during I/O-bound phases.
  // These modules add ~40ms to the static import graph that would otherwise block the
  // defaultAction preload at startup. By deferring them to a dynamic import here, the
  // preload completes faster and the import overlaps with searchFiles and collectFiles I/O.
  // When all three output deps are provided via overrideDeps (tests), the import is skipped.
  const needsOutputModules = !deps.generateOutput || !deps.produceOutput || !deps.packSkill;
  const outputModulePromise = needsOutputModules
    ? Promise.all([
        import('./output/outputGenerate.js'),
        import('./packager/produceOutput.js'),
        import('./skill/packSkill.js'),
      ]).then(([outputGenMod, produceOutputMod, packSkillMod]) => {
        // Only set deps that weren't provided by overrideDeps (e.g., test mocks)
        if (!deps.generateOutput) deps.generateOutput = outputGenMod.generateOutput;
        if (!deps.produceOutput) deps.produceOutput = produceOutputMod.produceOutput;
        if (!deps.packSkill) deps.packSkill = packSkillMod.packSkill;
      })
    : null;

  // Pre-initialize metrics worker pool and start warmup BEFORE searchFiles.
  // Each worker thread loads the gpt-tokenizer encoding module (~200ms) on its first task.
  // By starting warmup before search, the CPU-intensive tokenizer initialization overlaps
  // with the I/O-bound file search (git ls-files, ~75ms) and the beginning of file collection,
  // reducing the total warmup contention window during file I/O.
  //
  // Adapt metrics thread count based on the expected tokenization workload:
  // - tokenCountTree: true → all files tokenized (~1000 files, ~20 batches) → need full threads
  // - tokenCountTree: number > 0 or false → ~50 files sampled (1 batch) → 1 thread suffices
  //
  // After the sample size cap (commit a6f43cc), the threshold/default paths only send ~50
  // files in 1 batch, so extra metrics threads sit idle. Reducing to 1 thread for these
  // paths frees CPU cores for the security worker (which runs ~300ms cold on 1 thread).
  // This eliminates the CPU contention that previously existed between 3 idle metrics
  // threads and the security scanner during the speculative execution phase.
  //
  // For the "all files" path (tokenCountTree: true), use processConcurrency - 2 threads
  // to reserve cores for 1 security thread and the main thread (output generation).
  // The TASKS_PER_THREAD threshold in getWorkerThreadCount is 100.
  const tokenCountTreeValue = config.output.tokenCountTree;
  const allFilesMode = tokenCountTreeValue === true || tokenCountTreeValue === 'true';
  const metricsMaxThreads = allFilesMode ? Math.max(1, getProcessConcurrency() - 2) : 1;
  const estimatedTasks = metricsMaxThreads * 100;
  const metricsTaskRunner = deps.createMetricsTaskRunner(estimatedTasks);
  const warmupTask = { content: '', encoding: config.tokenCount.encoding };
  const warmupThreadCount = getWorkerThreadCount(estimatedTasks).maxThreads;
  const warmupPromises = Array.from({ length: warmupThreadCount }, () =>
    metricsTaskRunner.run(warmupTask).catch(() => 0),
  );
  const warmupPromise = Promise.all(warmupPromises).then(() => 0); // Suppress unhandled rejection

  // Pre-initialize security worker pool and warm up secretlint BEFORE searchFiles.
  // The secretlint module + rule preset loading on the worker thread takes ~150-200ms.
  // By creating the pool and sending a warmup task here, this cold start overlaps with
  // the I/O-bound file search (git ls-files, ~75ms) and file collection (~120ms),
  // so the worker is already initialized when real security tasks arrive after collection.
  // Without this, the cold start happens sequentially on the critical path after collectFiles.
  const securityTaskRunner = config.security.enableSecurityCheck ? deps.createSecurityTaskRunner() : null;
  if (securityTaskRunner) {
    // Use a valid filePath so the warmup task exercises the full secretlint scanning code
    // path (not just module loading). This JIT-compiles the secretlint rule regex matching
    // and lintSource function before real tasks arrive, reducing first-batch latency.
    // An empty filePath causes secretlint to throw before scanning, leaving the scanning
    // code path un-optimized.
    securityTaskRunner.run({ filePath: 'warmup.js', content: '', type: 'file' as const }).catch((error) => {
      logger.debug('Security worker warmup failed (non-fatal):', error);
    });
  }

  // Start git operations early so they overlap with searchFiles and sort,
  // completing before collectFiles begins. Previously, git ops ran in parallel
  // with collectFiles, causing I/O contention between git subprocesses (diff,
  // log, file change count queries) and the concurrent file reads. By starting
  // them here, the git index is read while searchFiles is also reading it
  // (both are git-based, benefiting from shared page cache), and they finish
  // before the I/O-intensive file reading phase begins.
  const gitDiffPromise = Promise.resolve(deps.getGitDiffs(rootDirs, config));
  const gitLogPromise = Promise.resolve(deps.getGitLogs(rootDirs, config));
  const prefetchPromise = Promise.resolve(deps.prefetchFileChangeCounts(config)).catch(() => {});
  // Suppress unhandled rejection if searchFiles throws before these are awaited in the try block
  gitDiffPromise.catch(() => {});
  gitLogPromise.catch(() => {});

  progressCallback('Searching for files...');
  const searchResults = await withMemoryLogging('Search Files', async () =>
    Promise.all(
      rootDirs.map(async (rootDir) => ({ rootDir, ...(await deps.searchFiles(rootDir, config, explicitFiles)) })),
    ),
  );
  const filePathsByDir = searchResults.map(({ rootDir, filePaths }) => ({ rootDir, filePaths }));
  const emptyDirPaths = searchResults.flatMap((r) => r.emptyDirPaths);

  // Sort file paths
  progressCallback('Sorting files...');
  const allFilePaths = filePathsByDir.flatMap(({ filePaths }) => filePaths);
  const sortedFilePaths = deps.sortPaths(allFilePaths);

  // Regroup sorted file paths by rootDir using Set for O(1) membership checks
  const filePathSetByDir = new Map(filePathsByDir.map(({ rootDir, filePaths }) => [rootDir, new Set(filePaths)]));
  const sortedFilePathsByDir = rootDirs.map((rootDir) => ({
    rootDir,
    filePaths: sortedFilePaths.filter((filePath) => filePathSetByDir.get(rootDir)?.has(filePath) ?? false),
  }));

  try {
    // Pipeline file collection with security checking to reduce critical path latency.
    // Previously, security only started after ALL files were collected, creating a
    // sequential dependency: collectFiles(149ms) → security(224ms). By splitting
    // collection into two stages, the security worker begins processing the first batch
    // while the second batch is still being read from disk.
    //
    // Timeline (4-core, ~1000 files):
    //   Before: ──[collect 149ms]──[security 224ms]──
    //   After:  ──[collect½ 75ms]──[security starts]──────────────────
    //                              ──[collect½ 75ms]──[process+output]──
    //
    // This gives security a ~75ms head start, reducing the critical path by that amount.
    progressCallback('Collecting files...');

    // Await git operations (started before searchFiles, should already be resolved)
    const [gitDiffResult, gitLogResult] = await Promise.all([gitDiffPromise, gitLogPromise, prefetchPromise]);

    // Split file paths into two halves for pipelined collection+security
    const firstHalfByDir = sortedFilePathsByDir.map(({ rootDir, filePaths }) => ({
      rootDir,
      filePaths: filePaths.slice(0, Math.ceil(filePaths.length / 2)),
    }));
    const secondHalfByDir = sortedFilePathsByDir.map(({ rootDir, filePaths }) => ({
      rootDir,
      filePaths: filePaths.slice(Math.ceil(filePaths.length / 2)),
    }));

    // Stage 1: Collect first half of files
    const firstCollectResults = await withMemoryLogging('Collect Files (first half)', async () =>
      Promise.all(
        firstHalfByDir.map(({ rootDir, filePaths }) => deps.collectFiles(filePaths, rootDir, config, progressCallback)),
      ),
    );
    const firstRawFiles = firstCollectResults.flatMap((curr) => curr.rawFiles);

    // Start security on first batch immediately (overlaps with second half collection).
    // Use runSecurityCheck directly (not validateFileSafety) to send file tasks without
    // git diff/log, which will be included in the second batch.
    let firstSecurityPromise: Promise<SuspiciousFileResult[]> = Promise.resolve([]);
    if (config.security.enableSecurityCheck && securityTaskRunner) {
      progressCallback('Running security check...');
      firstSecurityPromise = deps.runSecurityCheck(firstRawFiles, progressCallback, undefined, undefined, undefined, {
        taskRunner: securityTaskRunner,
      });
    }

    // Stage 2: Collect second half in parallel with security processing first batch
    const secondCollectResults = await withMemoryLogging('Collect Files (second half)', async () =>
      Promise.all(
        secondHalfByDir.map(({ rootDir, filePaths }) =>
          deps.collectFiles(filePaths, rootDir, config, progressCallback),
        ),
      ),
    );
    const secondRawFiles = secondCollectResults.flatMap((curr) => curr.rawFiles);

    const rawFiles = [...firstRawFiles, ...secondRawFiles];
    const allSkippedFiles = [
      ...firstCollectResults.flatMap((curr) => curr.skippedFiles),
      ...secondCollectResults.flatMap((curr) => curr.skippedFiles),
    ];

    // Start security on second batch + git diff/log content.
    // The security worker thread processes first-batch tasks while these are queued,
    // so the second batch begins immediately after the first batch finishes.
    let secondSecurityPromise: Promise<SuspiciousFileResult[]> = Promise.resolve([]);
    if (config.security.enableSecurityCheck && securityTaskRunner) {
      secondSecurityPromise = deps.runSecurityCheck(
        secondRawFiles,
        progressCallback,
        gitDiffResult,
        gitLogResult,
        undefined,
        { taskRunner: securityTaskRunner },
      );
    }

    // Suppress unhandled rejection if one batch fails while the other is still running.
    // Errors re-surface when securityPromise is awaited below.
    firstSecurityPromise.catch(() => {});
    secondSecurityPromise.catch(() => {});

    // Combined security promise that merges both batches and separates by type
    const securityPromise = Promise.all([firstSecurityPromise, secondSecurityPromise]).then(
      ([firstResults, secondResults]) => {
        const allResults = [...firstResults, ...secondResults];
        const suspiciousGitDiffResults = allResults.filter((r) => r.type === 'gitDiff');
        const suspiciousGitLogResults = allResults.filter((r) => r.type === 'gitLog');

        // Log warnings for suspicious git content (mirrors validateFileSafety behavior)
        if (suspiciousGitDiffResults.length > 0) {
          logger.warn('Security issues found in Git diffs, but they will still be included in the output');
          for (const result of suspiciousGitDiffResults) {
            logger.warn(
              `  - ${result.filePath}: ${result.messages.length} ${result.messages.length === 1 ? 'issue' : 'issues'} detected`,
            );
          }
        }
        if (suspiciousGitLogResults.length > 0) {
          logger.warn('Security issues found in Git logs, but they will still be included in the output');
          for (const result of suspiciousGitLogResults) {
            logger.warn(
              `  - ${result.filePath}: ${result.messages.length} ${result.messages.length === 1 ? 'issue' : 'issues'} detected`,
            );
          }
        }

        return {
          suspiciousFilesResults: allResults.filter((r) => r.type === 'file'),
          suspiciousGitDiffResults,
          suspiciousGitLogResults,
        };
      },
    );
    const processPromise = withMemoryLogging('Process Files', () =>
      deps.processFiles(rawFiles, config, progressCallback),
    );

    // Build filePathsByRoot for multi-root tree generation
    // Use directory basename as the label for each root
    // Fallback to rootDir if basename is empty (e.g., filesystem root "/")
    const filePathsByRoot: FilesByRoot[] = sortedFilePathsByDir.map(({ rootDir, filePaths }) => ({
      rootLabel: path.basename(rootDir) || rootDir,
      files: filePaths,
    }));

    // Chain speculative output generation and metrics calculation from file processing.
    // For non-split output: output generation → metrics runs on the metrics worker pool
    // while security check runs on a separate worker pool. Both proceed in parallel.
    // This overlaps the entire output generation + metrics pipeline with the security check,
    // so only the longer of the two determines the stage duration.
    // For split output mode, fall back to the existing produceOutput flow after security.
    const isSplitOutput = config.output.splitOutput !== undefined;

    // For non-split output, chain: process → [output || metrics] (all speculative).
    // Output generation runs on the main thread while file token counting runs on
    // worker threads. By passing the output as a promise to calculateMetrics, file
    // and git token counting start immediately without waiting for output generation.
    // calculateMetrics only awaits the output promise when it needs the character count
    // for the final token estimation or full output tokenization.
    //
    // The output promise is exposed separately from the metrics promise so that output
    // writing can start as soon as the output is generated, overlapping disk I/O with
    // the remaining metrics worker computation. This hides ~42ms of disk write time
    // and ~10ms of clipboard copy behind the metrics computation phase.
    // Errors are suppressed here; they re-surface when the promise is awaited in the
    // common path, or are harmlessly discarded in the rare path (suspicious files found).
    let speculativeOutputPromise: Promise<string> | undefined;
    let speculativeMetricsOnlyPromise: Promise<Awaited<ReturnType<typeof deps.calculateMetrics>>> | undefined;
    let speculativeWritePromise: Promise<void> | undefined;
    let speculativeClipboardPromise: Promise<void> | undefined;

    if (!isSplitOutput) {
      // Chain from processPromise: generate output, then start metrics with output as a promise
      speculativeOutputPromise = processPromise.then(async (processed) => {
        await warmupPromise;
        if (outputModulePromise) await outputModulePromise;
        progressCallback('Generating output...');
        // deps.generateOutput is guaranteed set after outputModulePromise resolves
        return (deps.generateOutput as NonNullable<typeof deps.generateOutput>)(
          rootDirs,
          config,
          processed,
          allFilePaths,
          gitDiffResult,
          gitLogResult,
          filePathsByRoot,
          emptyDirPaths,
        );
      });

      // Start metrics as soon as file processing completes, passing the output as a promise
      // so file/git token counting begins immediately without waiting for output generation.
      speculativeMetricsOnlyPromise = processPromise.then(async (processed) => {
        await warmupPromise;
        return withMemoryLogging('Calculate Metrics', () =>
          deps.calculateMetrics(
            processed,
            speculativeOutputPromise as Promise<string>,
            progressCallback,
            config,
            gitDiffResult,
            gitLogResult,
            {
              taskRunner: metricsTaskRunner,
            },
          ),
        );
      });

      // Start writing output to disk and copying to clipboard as soon as output generation
      // completes, overlapping disk I/O with the remaining security check.
      //
      // Previously, writeOutputToDisk and copyToClipboardIfEnabled waited until after the
      // security check resolved, adding their latency sequentially to the critical path.
      // The security check (secretlint on worker threads) is the pipeline bottleneck after
      // file collection, typically taking 200-400ms. By chaining write/clipboard from the
      // output promise, they execute during the security check window and are already
      // complete (or nearly so) when the security check finishes.
      //
      // In the common case (no suspicious files), the speculative write is the final result.
      // In the rare case (suspicious files found), the output file is overwritten with
      // the corrected output in the fallback path, so the speculative write is harmless.
      //
      // Skip speculative write for stdout mode since stdout output cannot be retracted
      // if the security check later finds suspicious files.
      if (config.output.stdout !== true) {
        speculativeWritePromise = speculativeOutputPromise.then(async (output) => {
          progressCallback('Writing output file...');
          await deps.writeOutputToDisk(output, config);
        });
        speculativeClipboardPromise = speculativeOutputPromise.then(async (output) => {
          await deps.copyToClipboardIfEnabled(output, progressCallback, config);
        });
      }

      // Suppress unhandled rejection for paths that don't await these promises
      // (skill generation, split output, or rare suspicious-files case)
      speculativeOutputPromise.catch(() => {});
      speculativeMetricsOnlyPromise.catch(() => {});
      speculativeWritePromise?.catch(() => {});
      speculativeClipboardPromise?.catch(() => {});
    }

    const [{ suspiciousFilesResults, suspiciousGitDiffResults, suspiciousGitLogResults }, processedFilesSpeculative] =
      await Promise.all([securityPromise, processPromise]);

    // Use speculative result if no files were flagged, otherwise re-process safe subset
    let processedFiles: ProcessedFile[];
    let safeFilePaths: string[];
    if (suspiciousFilesResults.length === 0) {
      processedFiles = processedFilesSpeculative;
      safeFilePaths = rawFiles.map((file) => file.path);
    } else {
      const safeRawFiles = deps.filterOutUntrustedFiles(rawFiles, suspiciousFilesResults);
      safeFilePaths = safeRawFiles.map((file) => file.path);
      processedFiles = await deps.processFiles(safeRawFiles, config, progressCallback);
    }

    // Check if skill generation is requested
    if (config.skillGenerate !== undefined && options.skillDir) {
      // Await warmup to ensure graceful worker shutdown (avoid terminating WASM-loading thread)
      await warmupPromise;
      if (outputModulePromise) await outputModulePromise;

      const result = await (deps.packSkill as NonNullable<typeof deps.packSkill>)({
        rootDirs,
        config,
        options,
        processedFiles,
        allFilePaths,
        gitDiffResult,
        gitLogResult,
        suspiciousFilesResults,
        suspiciousGitDiffResults,
        suspiciousGitLogResults,
        safeFilePaths,
        skippedFiles: allSkippedFiles,
        progressCallback,
      });

      logMemoryUsage('Pack - End');
      return result;
    }

    let outputFiles: string[] | undefined;

    if (isSplitOutput) {
      // Split output mode: use produceOutput for both generation and writing
      await warmupPromise;
      if (outputModulePromise) await outputModulePromise;
      progressCallback('Generating output...');
      const outputPromise = (deps.produceOutput as NonNullable<typeof deps.produceOutput>)(
        rootDirs,
        config,
        processedFiles,
        allFilePaths,
        gitDiffResult,
        gitLogResult,
        progressCallback,
        filePathsByRoot,
        emptyDirPaths,
      );
      const outputForMetrics = outputPromise.then((r) => r.outputForMetrics);

      const [produceResult, metrics] = await Promise.all([
        outputPromise,
        withMemoryLogging('Calculate Metrics', () =>
          deps.calculateMetrics(
            processedFiles,
            outputForMetrics,
            progressCallback,
            config,
            gitDiffResult,
            gitLogResult,
            {
              taskRunner: metricsTaskRunner,
            },
          ),
        ),
      ]);
      outputFiles = produceResult.outputFiles;

      const result = {
        ...metrics,
        ...(outputFiles && { outputFiles }),
        suspiciousFilesResults,
        suspiciousGitDiffResults,
        suspiciousGitLogResults,
        processedFiles,
        safeFilePaths,
        skippedFiles: allSkippedFiles,
      };

      logMemoryUsage('Pack - End');
      return result;
    }

    // Non-split output: use speculative metrics if security check passed,
    // otherwise regenerate output and recalculate metrics.
    let metrics: Awaited<ReturnType<typeof deps.calculateMetrics>>;

    if (suspiciousFilesResults.length === 0 && speculativeOutputPromise && speculativeMetricsOnlyPromise) {
      // Common case: speculative results are valid (may already be resolved).
      // Write/clipboard were started speculatively during the security check phase,
      // so they are already complete or nearly complete by now.
      if (speculativeWritePromise) {
        // Non-stdout: speculative write/clipboard already in progress
        const [, , speculativeMetrics] = await Promise.all([
          speculativeWritePromise,
          speculativeClipboardPromise,
          speculativeMetricsOnlyPromise,
        ]);
        metrics = speculativeMetrics;
      } else {
        // stdout mode: write was deferred to avoid irreversible output before security check
        const output = await speculativeOutputPromise;
        progressCallback('Writing output file...');
        const [, , speculativeMetrics] = await Promise.all([
          deps.writeOutputToDisk(output, config),
          deps.copyToClipboardIfEnabled(output, progressCallback, config),
          speculativeMetricsOnlyPromise,
        ]);
        metrics = speculativeMetrics;
      }
    } else {
      // Rare case: suspicious files found, regenerate with safe files
      await warmupPromise;
      if (outputModulePromise) await outputModulePromise;
      progressCallback('Generating output...');
      const output = await (deps.generateOutput as NonNullable<typeof deps.generateOutput>)(
        rootDirs,
        config,
        processedFiles,
        allFilePaths,
        gitDiffResult,
        gitLogResult,
        filePathsByRoot,
        emptyDirPaths,
      );

      const metricsPromise = withMemoryLogging('Calculate Metrics', () =>
        deps.calculateMetrics(processedFiles, output, progressCallback, config, gitDiffResult, gitLogResult, {
          taskRunner: metricsTaskRunner,
        }),
      );

      progressCallback('Writing output file...');
      await deps.writeOutputToDisk(output, config);
      await deps.copyToClipboardIfEnabled(output, progressCallback, config);

      metrics = await metricsPromise;
    }

    const result = {
      ...metrics,
      suspiciousFilesResults,
      suspiciousGitDiffResults,
      suspiciousGitLogResults,
      processedFiles,
      safeFilePaths,
      skippedFiles: allSkippedFiles,
    };

    logMemoryUsage('Pack - End');

    return result;
  } finally {
    // Fire-and-forget: don't block pack() return on worker termination.
    // All tasks are complete; workers will be terminated when the process exits.
    // This saves ~70ms that pool.destroy() spends signaling worker threads.
    metricsTaskRunner.cleanup().catch(() => {});
    securityTaskRunner?.cleanup().catch(() => {});
  }
};
