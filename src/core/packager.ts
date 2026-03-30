import path from 'node:path';
import type { RepomixConfigMerged } from '../config/configSchema.js';
import { logMemoryUsage, withMemoryLogging } from '../shared/memoryUtils.js';
import { getWorkerThreadCount } from '../shared/processConcurrency.js';
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
import { generateOutput } from './output/outputGenerate.js';
import { prefetchFileChangeCounts } from './output/outputSort.js';
import { copyToClipboardIfEnabled } from './packager/copyToClipboardIfEnabled.js';
import { produceOutput } from './packager/produceOutput.js';
import { writeOutputToDisk } from './packager/writeOutputToDisk.js';
import { filterOutUntrustedFiles } from './security/filterOutUntrustedFiles.js';
import type { SuspiciousFileResult } from './security/securityCheck.js';
import { validateFileSafety } from './security/validateFileSafety.js';
import { packSkill } from './skill/packSkill.js';

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
  produceOutput,
  generateOutput,
  writeOutputToDisk,
  copyToClipboardIfEnabled,
  calculateMetrics,
  createMetricsTaskRunner,
  sortPaths,
  getGitDiffs,
  getGitLogs,
  packSkill,
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

  // Pre-initialize metrics worker pool and warm up ALL threads to overlap tiktoken initialization
  // with subsequent pipeline stages (file collection, security check, file processing).
  // Each thread loads the gpt-tokenizer encoding module (~170ms) on its first task.
  // By submitting one warmup task per thread, all threads initialize in parallel during
  // file collection, so they're ready when speculative metrics starts.
  const metricsTaskRunner = deps.createMetricsTaskRunner(allFilePaths.length);
  const warmupTask = { content: '', encoding: config.tokenCount.encoding };
  const threadCount = getWorkerThreadCount(allFilePaths.length).maxThreads;
  const warmupPromises = Array.from({ length: threadCount }, () => metricsTaskRunner.run(warmupTask).catch(() => 0));
  const warmupPromise = Promise.all(warmupPromises).then(() => 0); // Suppress unhandled rejection

  try {
    // Run file collection, git operations, and sort data prefetch in parallel since
    // they are independent. Pre-fetching git file change counts here populates the
    // module-level cache in outputSort.ts, so sortOutputFiles (called later during
    // output generation) returns instantly from cache instead of spawning a git subprocess.
    progressCallback('Collecting files...');
    const [collectResults, gitDiffResult, gitLogResult] = await Promise.all([
      withMemoryLogging(
        'Collect Files',
        async () =>
          await Promise.all(
            sortedFilePathsByDir.map(({ rootDir, filePaths }) =>
              deps.collectFiles(filePaths, rootDir, config, progressCallback),
            ),
          ),
      ),
      deps.getGitDiffs(rootDirs, config),
      deps.getGitLogs(rootDirs, config),
      deps.prefetchFileChangeCounts(config).catch(() => {}),
    ]);

    const rawFiles = collectResults.flatMap((curr) => curr.rawFiles);
    const allSkippedFiles = collectResults.flatMap((curr) => curr.skippedFiles);

    // Run security check and file processing in parallel using speculative execution.
    // Security check runs on worker threads (secretlint). File processing runs lightweight
    // transforms on the main thread by default; when compress/removeComments is enabled,
    // it uses a separate worker pool. Both paths benefit from the overlap.
    // In the common case (no suspicious files), the speculative processing result is used directly.
    // If suspicious files are found (rare), file processing is re-run on the filtered safe subset.
    progressCallback('Running security check...');
    const securityPromise = withMemoryLogging('Security Check', () =>
      deps.validateFileSafety(rawFiles, progressCallback, config, gitDiffResult, gitLogResult),
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
    // Errors are suppressed here; they re-surface when the promise is awaited in the
    // common path, or are harmlessly discarded in the rare path (suspicious files found).
    const speculativeMetricsPromise = !isSplitOutput
      ? processPromise.then(async (processed) => {
          // Ensure tiktoken is initialized before submitting metrics tasks
          await warmupPromise;
          progressCallback('Generating output...');
          const outputPromise = deps.generateOutput(
            rootDirs,
            config,
            processed,
            allFilePaths,
            gitDiffResult,
            gitLogResult,
            filePathsByRoot,
            emptyDirPaths,
          );
          // Start metrics immediately — file token counting on worker threads overlaps
          // with output generation on the main thread.
          const metricsPromise = withMemoryLogging('Calculate Metrics', () =>
            deps.calculateMetrics(processed, outputPromise, progressCallback, config, gitDiffResult, gitLogResult, {
              taskRunner: metricsTaskRunner,
            }),
          );
          const [output, metricsResult] = await Promise.all([outputPromise, metricsPromise]);
          return { output, metrics: metricsResult };
        })
      : undefined;

    // Suppress unhandled rejection for paths that don't await speculativeMetricsPromise
    // (skill generation, split output, or rare suspicious-files case)
    speculativeMetricsPromise?.catch(() => {});

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

      const result = await deps.packSkill({
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
      progressCallback('Generating output...');
      const outputPromise = deps.produceOutput(
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

    if (suspiciousFilesResults.length === 0 && speculativeMetricsPromise) {
      // Common case: speculative results are valid (may already be resolved)
      const { output, metrics: speculativeMetrics } = await speculativeMetricsPromise;

      // Write output to disk and clipboard
      progressCallback('Writing output file...');
      await deps.writeOutputToDisk(output, config);
      await deps.copyToClipboardIfEnabled(output, progressCallback, config);

      metrics = speculativeMetrics;
    } else {
      // Rare case: suspicious files found, regenerate with safe files
      await warmupPromise;
      progressCallback('Generating output...');
      const output = await deps.generateOutput(
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
    await metricsTaskRunner.cleanup();
  }
};
