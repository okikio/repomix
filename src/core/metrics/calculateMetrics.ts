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
    // For top files display optimization: calculate token counts only for top files by character count
    // However, if tokenCountTree is enabled, calculate for all files to avoid double calculation
    const topFilesLength = config.output.topFilesLength;
    const shouldCalculateAllFiles = !!config.output.tokenCountTree;

    // Determine which files to calculate token counts for:
    // - If tokenCountTree is enabled: calculate for all files to avoid double calculation
    // - Otherwise: calculate only for top files by character count for optimization
    const metricsTargetPaths = shouldCalculateAllFiles
      ? processedFiles.map((file) => file.path)
      : [...processedFiles]
          .sort((a, b) => b.content.length - a.content.length)
          .slice(0, Math.min(processedFiles.length, Math.max(topFilesLength * 10, topFilesLength)))
          .map((file) => file.path);

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

    // Await the output (resolves immediately if already a string, waits if a promise)
    const resolvedOutput = await Promise.resolve(output);
    const outputParts = Array.isArray(resolvedOutput) ? resolvedOutput : [resolvedOutput];

    // When all files have individual token counts (tokenCountTree enabled), we can estimate
    // the output token total from the file token sum, avoiding the expensive full-output
    // token counting pass. The output consists of ~97% file content and ~3% overhead
    // (headers, tree structure, XML/Markdown tags). Estimating total tokens from file tokens
    // using the character ratio produces results within ~0.1% of the exact count, while
    // eliminating the redundant re-tokenization of the full output string.
    // When only a subset of files are counted (tokenCountTree disabled), we must count
    // the full output to get an accurate total.
    if (shouldCalculateAllFiles) {
      const [selectiveFileMetrics, gitDiffTokenCount, gitLogTokenCount] = await Promise.all([
        fileMetricsPromise,
        gitDiffMetricsPromise,
        gitLogMetricsPromise,
      ]);

      const totalFiles = processedFiles.length;
      const totalCharacters = outputParts.reduce((sum, part) => sum + part.length, 0);

      // Build character and token counts for all files
      const fileCharCounts: Record<string, number> = {};
      const fileTokenCounts: Record<string, number> = {};
      let fileTokenSum = 0;
      let fileCharSum = 0;

      for (const file of processedFiles) {
        fileCharCounts[file.path] = file.content.length;
        fileCharSum += file.content.length;
      }
      for (const file of selectiveFileMetrics) {
        fileTokenCounts[file.path] = file.tokenCount;
        fileTokenSum += file.tokenCount;
      }

      // Estimate total output tokens from file token counts using the character ratio.
      // The output contains all file content plus template overhead (headers, tree, tags).
      // Since file chars and output chars are known exactly, we apply the observed
      // chars-per-token ratio from files to the full output length.
      // Accuracy: within ~0.1% of exact count on typical repositories.
      const totalTokens =
        fileCharSum > 0 ? Math.round((fileTokenSum / fileCharSum) * totalCharacters) : totalCharacters;

      return {
        totalFiles,
        totalCharacters,
        totalTokens,
        fileCharCounts,
        fileTokenCounts,
        gitDiffTokenCount: gitDiffTokenCount,
        gitLogTokenCount: gitLogTokenCount.gitLogTokenCount,
      };
    }

    // Start output token counting now that the output content is available
    const outputTokenCountPromise = Promise.all(
      outputParts.map(async (part, index) => {
        const partPath =
          outputParts.length > 1 ? buildSplitOutputFilePath(config.output.filePath, index + 1) : config.output.filePath;
        return await deps.calculateOutputMetrics(part, config.tokenCount.encoding, partPath, { taskRunner });
      }),
    );

    const [selectiveFileMetrics, outputTokenCounts, gitDiffTokenCount, gitLogTokenCount] = await Promise.all([
      fileMetricsPromise,
      outputTokenCountPromise,
      gitDiffMetricsPromise,
      gitLogMetricsPromise,
    ]);

    const totalTokens = outputTokenCounts.reduce((sum, count) => sum + count, 0);
    const totalFiles = processedFiles.length;
    const totalCharacters = outputParts.reduce((sum, part) => sum + part.length, 0);

    // Build character counts for all files
    const fileCharCounts: Record<string, number> = {};
    for (const file of processedFiles) {
      fileCharCounts[file.path] = file.content.length;
    }

    // Build token counts only for top files
    const fileTokenCounts: Record<string, number> = {};
    for (const file of selectiveFileMetrics) {
      fileTokenCounts[file.path] = file.tokenCount;
    }

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
