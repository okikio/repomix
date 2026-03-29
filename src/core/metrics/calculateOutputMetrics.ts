import { logger } from '../../shared/logger.js';
import type { TaskRunner } from '../../shared/processConcurrency.js';
import type { TokenEncoding } from './TokenCounter.js';
import type { TokenCountBatchTask, TokenCountTask } from './workers/calculateMetricsWorker.js';

const NUM_CHUNKS = 1000;
const MIN_CONTENT_LENGTH_FOR_PARALLEL = 1_000_000; // 1000KB

// Number of batch tasks to distribute output chunks across the worker pool.
// Each batch contains multiple chunks counted in a single worker round-trip,
// reducing per-task overhead while maintaining good load balancing.
const NUM_OUTPUT_BATCHES = 20;

export const calculateOutputMetrics = async (
  content: string,
  encoding: TokenEncoding,
  path: string | undefined,
  deps: { taskRunner: TaskRunner<TokenCountTask, number> },
): Promise<number> => {
  const shouldRunInParallel = content.length > MIN_CONTENT_LENGTH_FOR_PARALLEL;

  try {
    logger.trace(`Starting output token count for ${path || 'output'}`);
    const startTime = process.hrtime.bigint();

    let result: number;

    if (shouldRunInParallel) {
      // Split content into small chunks for accurate BPE token counting
      const chunkSize = Math.ceil(content.length / NUM_CHUNKS);
      const chunks: string[] = [];

      for (let i = 0; i < content.length; i += chunkSize) {
        chunks.push(content.slice(i, i + chunkSize));
      }

      // Group chunks into batches to reduce per-task overhead.
      // Each batch is sent as a single worker task, avoiding the message passing
      // and promise resolution overhead of individual tasks.
      const batchSize = Math.ceil(chunks.length / NUM_OUTPUT_BATCHES);
      const batchResults = await Promise.all(
        Array.from({ length: Math.ceil(chunks.length / batchSize) }, (_, i) => {
          const batchChunks = chunks.slice(i * batchSize, (i + 1) * batchSize);
          return deps.taskRunner.runNamed!<TokenCountBatchTask, number[]>('countTokensBatch', {
            contents: batchChunks,
            encoding,
          });
        }),
      );

      // Sum up all token counts from all batches
      result = batchResults.flat().reduce((sum, count) => sum + count, 0);
    } else {
      // Process small content directly
      result = await deps.taskRunner.run({
        content,
        encoding,
        path,
      });
    }

    const endTime = process.hrtime.bigint();
    const duration = Number(endTime - startTime) / 1e6;
    logger.trace(`Output token count completed in ${duration.toFixed(2)}ms`);

    return result;
  } catch (error) {
    logger.error('Error during token count:', error);
    throw error;
  }
};
