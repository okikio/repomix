import type { RepomixConfigMerged } from '../../config/configSchema.js';
import { initTaskRunner, type TaskRunner } from '../../shared/processConcurrency.js';
import type { RepomixProgressCallback } from '../../shared/types.js';
import type { ProcessedFile } from '../file/fileTypes.js';
import type { GitDiffResult } from '../git/gitDiffHandle.js';
import type { GitLogResult } from '../git/gitLogHandle.js';
import { buildSplitOutputFilePath } from '../output/outputSplit.js';
import { calculateGitDiffMetrics } from './calculateGitDiffMetrics.js';
import { calculateGitLogMetrics } from './calculateGitLogMetrics.js';
import { calculateOutputMetrics } from './calculateOutputMetrics.js';
import { calculateSelectiveFileMetrics } from './calculateSelectiveFileMetrics.js';
import type { TokenCountTask } from './workers/calculateMetricsWorker.js';

export interface CalculateMetricsResult {
  totalFiles: number;
  totalCharacters: number;
  totalTokens: number;
  fileCharCounts: Record<string, number>;
  fileTokenCounts: Record<string, number>;
  gitDiffTokenCount: number;
  gitLogTokenCount: number;
}

/**
 * Create a metrics task runner that can be pre-initialized to overlap
 * tiktoken WASM loading with other pipeline stages.
 */
export const createMetricsTaskRunner = (numOfTasks: number): TaskRunner<TokenCountTask, number> => {
  return initTaskRunner<TokenCountTask, number>({
    numOfTasks,
    workerType: 'calculateMetrics',
    runtime: 'worker_threads',
  });
};

const defaultDeps = {
  calculateSelectiveFileMetrics,
  calculateOutputMetrics,
  calculateGitDiffMetrics,
  calculateGitLogMetrics,
  taskRunner: undefined as TaskRunner<TokenCountTask, number> | undefined,
};

export const calculateMetrics = async (
  processedFiles: ProcessedFile[],
  output: string | string[] | Promise<string | string[]>,
  progressCallback: RepomixProgressCallback,
  config: RepomixConfigMerged,
  gitDiffResult: GitDiffResult | undefined,
  gitLogResult: GitLogResult | undefined,
  overrideDeps: Partial<typeof defaultDeps> = {},
): Promise<CalculateMetricsResult> => {
  const deps = { ...defaultDeps, ...overrideDeps };

  progressCallback('Calculating metrics...');

  // Initialize a single task runner for all metrics calculations
  const taskRunner =
    deps.taskRunner ??
    initTaskRunner<TokenCountTask, number>({
      numOfTasks: processedFiles.length,
      workerType: 'calculateMetrics',
      runtime: 'worker_threads',
    });

  try {
    const topFilesLength = config.output.topFilesLength;
    const tokenCountTreeValue = config.output.tokenCountTree;
    const shouldCalculateAllFiles = !!tokenCountTreeValue;

    // When tokenCountTree is a positive number (threshold mode), files below the threshold
    // won't be displayed in the tree. We can skip BPE-encoding them on worker threads and
    // instead estimate their token counts from the chars-per-token ratio of larger files.
    // This reduces worker tasks from ~1000 to ~50-60, cutting metrics time by ~80%.
    //
    // Conservative char threshold: use 2.5 chars/token (code averages ~3.5-4.0 chars/token).
    // Files below this char count can't possibly reach the token threshold.
    const minTokenCount = typeof tokenCountTreeValue === 'number' && tokenCountTreeValue > 0 ? tokenCountTreeValue : 0;
    const useThresholdOptimization = shouldCalculateAllFiles && minTokenCount > 0;

    // Determine which files to send to workers for exact BPE token counting:
    // - tokenCountTree with threshold: tokenize files potentially above threshold + top N for ratio
    // - tokenCountTree true/0: tokenize all files for exact tree display
    // - tokenCountTree false: tokenize top files by size for ratio estimation
    let metricsTargetPaths: string[];

    if (useThresholdOptimization) {
      const charThreshold = Math.floor(minTokenCount * 2.5);
      const sortedBySize = [...processedFiles].sort((a, b) => b.content.length - a.content.length);
      const totalChars = sortedBySize.reduce((sum, f) => sum + f.content.length, 0);

      // Build target set: files potentially above the token threshold
      const targetSet = new Set<string>();
      for (const f of sortedBySize) {
        if (f.content.length >= charThreshold) {
          targetSet.add(f.path);
        }
      }

      // Add files by size for ratio estimation, capped by both coverage and count.
      // The largest files provide the most representative chars-per-token ratio per
      // tokenization cost. A small cap (10 files) is sufficient because the largest files
      // cover the most content per BPE encoding cost, and code files have consistent
      // chars-per-token ratios (~3.5-4.5). Reducing from 50 to 10 files cuts BPE encoding
      // time by ~65% (176ms → 63ms) while keeping the ratio estimate within ~8% of the
      // 50-file baseline. Since the output token count is already an estimate (not exact),
      // this trade-off is acceptable for the displayed summary metric.
      const ratioSampleCap = Math.max(topFilesLength, 10);
      const aboveThresholdCount = targetSet.size;
      const maxTargetFiles = aboveThresholdCount + ratioSampleCap;
      const coverageTarget = totalChars * 0.5;
      let coveredChars = 0;
      for (const f of sortedBySize) {
        if (coveredChars >= coverageTarget || targetSet.size >= maxTargetFiles) {
          break;
        }
        targetSet.add(f.path);
        coveredChars += f.content.length;
      }

      metricsTargetPaths = Array.from(targetSet);
    } else if (shouldCalculateAllFiles) {
      metricsTargetPaths = processedFiles.map((file) => file.path);
    } else {
      metricsTargetPaths = [...processedFiles]
        .sort((a, b) => b.content.length - a.content.length)
        .slice(0, Math.min(processedFiles.length, Math.max(topFilesLength, 10)))
        .map((file) => file.path);
    }

    // Start file metrics and git metrics immediately - these don't depend on the output.
    // When output is passed as a promise, this overlaps file/git token counting with
    // output generation, allowing worker threads to process files while the main thread
    // renders the output template.
    const fileMetricsPromise = deps.calculateSelectiveFileMetrics(
      processedFiles,
      metricsTargetPaths,
      config.tokenCount.encoding,
      progressCallback,
      { taskRunner },
    );
    const gitDiffMetricsPromise = deps.calculateGitDiffMetrics(config, gitDiffResult, { taskRunner });
    const gitLogMetricsPromise = deps.calculateGitLogMetrics(config, gitLogResult, { taskRunner });

    // Estimate total output tokens from file token counts using the character ratio,
    // eliminating the expensive full-output token counting pass entirely.
    //
    // The output consists of ~97% file content and ~3% template overhead (headers, tree,
    // XML/Markdown tags). By applying the chars-per-token ratio derived from counted files
    // to the total output character count, we get an accurate estimate without re-tokenizing
    // the full output string on worker threads.

    // Await output character count in parallel with worker-based file/git metrics
    const [selectiveFileMetrics, gitDiffTokenCount, gitLogTokenCount, resolvedOutput] = await Promise.all([
      fileMetricsPromise,
      gitDiffMetricsPromise,
      gitLogMetricsPromise,
      Promise.resolve(output),
    ]);
    const outputParts = Array.isArray(resolvedOutput) ? resolvedOutput : [resolvedOutput];

    const totalFiles = processedFiles.length;
    const totalCharacters = outputParts.reduce((sum, part) => sum + part.length, 0);

    // Build character counts for all files and token counts for counted files
    const fileCharCounts: Record<string, number> = {};
    const fileTokenCounts: Record<string, number> = {};
    let fileTokenSum = 0;
    let fileCharSum = 0;

    for (const file of processedFiles) {
      fileCharCounts[file.path] = file.content.length;
      // When all files are tokenized (no threshold optimization), sum chars from all files
      if (shouldCalculateAllFiles && !useThresholdOptimization) {
        fileCharSum += file.content.length;
      }
    }
    for (const file of selectiveFileMetrics) {
      fileTokenCounts[file.path] = file.tokenCount;
      fileTokenSum += file.tokenCount;
      // When using threshold optimization or selective mode, sum chars from counted files only
      if (!shouldCalculateAllFiles || useThresholdOptimization) {
        fileCharSum += file.charCount;
      }
    }

    // Compute chars-per-token ratio from counted files
    const charsPerToken = fileCharSum > 0 && fileTokenSum > 0 ? fileCharSum / fileTokenSum : 1;

    // For threshold optimization: estimate token counts for non-tokenized files
    // using the observed ratio. These files are below the display threshold,
    // but having estimated counts allows the tree to show accurate directory totals.
    if (useThresholdOptimization) {
      for (const file of processedFiles) {
        if (fileTokenCounts[file.path] === undefined) {
          fileTokenCounts[file.path] = Math.round(file.content.length / charsPerToken);
        }
      }
    }

    // Estimate total output tokens from the observed chars-per-token ratio
    const totalTokens = fileCharSum > 0 ? Math.round((fileTokenSum / fileCharSum) * totalCharacters) : totalCharacters;

    return {
      totalFiles,
      totalCharacters,
      totalTokens,
      fileCharCounts,
      fileTokenCounts,
      gitDiffTokenCount: gitDiffTokenCount,
      gitLogTokenCount: gitLogTokenCount.gitLogTokenCount,
    };
  } finally {
    // Cleanup the task runner after all calculations are complete (only if we created it)
    if (!deps.taskRunner) {
      await taskRunner.cleanup();
    }
  }
};
