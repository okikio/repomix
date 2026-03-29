import pc from 'picocolors';
import { logger } from '../../shared/logger.js';
import type { TaskRunner } from '../../shared/processConcurrency.js';
import type { RepomixProgressCallback } from '../../shared/types.js';
import type { ProcessedFile } from '../file/fileTypes.js';
import type { TokenEncoding } from './TokenCounter.js';
import type { TokenCountBatchTask, TokenCountTask } from './workers/calculateMetricsWorker.js';
import type { FileMetrics } from './workers/types.js';

// Number of files per batch sent to each worker.
// Larger batches reduce per-task overhead (message passing, structured cloning, promise resolution)
// while still distributing work evenly across the pool.
const FILES_PER_BATCH = 50;

export const calculateSelectiveFileMetrics = async (
  processedFiles: ProcessedFile[],
  targetFilePaths: string[],
  tokenCounterEncoding: TokenEncoding,
  progressCallback: RepomixProgressCallback,
  deps: { taskRunner: TaskRunner<TokenCountTask, number> },
): Promise<FileMetrics[]> => {
  const targetFileSet = new Set(targetFilePaths);
  const filesToProcess = processedFiles.filter((file) => targetFileSet.has(file.path));

  if (filesToProcess.length === 0) {
    return [];
  }

  try {
    const startTime = process.hrtime.bigint();
    logger.trace(`Starting selective metrics calculation for ${filesToProcess.length} files using worker pool`);

    // Batch files to reduce per-task overhead. Each batch is sent as a single worker task,
    // avoiding the message passing and promise resolution overhead of individual tasks.
    const batches: ProcessedFile[][] = [];
    for (let i = 0; i < filesToProcess.length; i += FILES_PER_BATCH) {
      batches.push(filesToProcess.slice(i, i + FILES_PER_BATCH));
    }

    let completedFiles = 0;
    const batchResults = await Promise.all(
      batches.map(async (batch) => {
        const batchTask: TokenCountBatchTask = {
          contents: batch.map((f) => f.content),
          encoding: tokenCounterEncoding,
          paths: batch.map((f) => f.path),
        };

        const tokenCounts = await deps.taskRunner.runNamed!<TokenCountBatchTask, number[]>(
          'countTokensBatch',
          batchTask,
        );

        return batch.map((file, i) => {
          completedFiles++;
          progressCallback(`Calculating metrics... (${completedFiles}/${filesToProcess.length}) ${pc.dim(file.path)}`);
          logger.trace(`Calculating metrics... (${completedFiles}/${filesToProcess.length}) ${file.path}`);
          return {
            path: file.path,
            charCount: file.content.length,
            tokenCount: tokenCounts[i],
          };
        });
      }),
    );

    const results = batchResults.flat();

    const endTime = process.hrtime.bigint();
    const duration = Number(endTime - startTime) / 1e6;
    logger.trace(`Selective metrics calculation completed in ${duration.toFixed(2)}ms`);

    return results;
  } catch (error) {
    logger.error('Error during selective metrics calculation:', error);
    throw error;
  }
};
