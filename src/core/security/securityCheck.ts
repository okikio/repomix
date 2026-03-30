import pc from 'picocolors';
import { logger } from '../../shared/logger.js';
import { initTaskRunner, type TaskRunner } from '../../shared/processConcurrency.js';
import type { RepomixProgressCallback } from '../../shared/types.js';
import type { RawFile } from '../file/fileTypes.js';
import type { GitDiffResult } from '../git/gitDiffHandle.js';
import type { GitLogResult } from '../git/gitLogHandle.js';
import type { SecurityCheckBatchTask, SecurityCheckTask, SecurityCheckType } from './workers/securityCheckWorker.js';

export interface SuspiciousFileResult {
  filePath: string;
  messages: string[];
  type: SecurityCheckType;
}

// Number of files per batch sent to each worker.
// Larger batches reduce per-task overhead (message passing, structured cloning, promise resolution)
// while still distributing work evenly across the pool.
const FILES_PER_BATCH = 50;

// Cap the worker thread count for security checks to minimize CPU contention with the
// metrics worker pool (which runs the CPU-intensive token counting on all cores).
// Security runs off the critical path in the pipeline: security (~200ms with 1 thread)
// overlaps with process (~80ms) → output (~50ms) → metrics (~340ms), so security
// always finishes before metrics. Using fewer threads avoids starving the metrics pool
// of CPU time, which otherwise adds ~140ms of contention on a 4-core machine.
// With batching (50 files/batch), 1 thread processes ~1000 files in ~200ms, well within
// the pipeline's tolerance.
const SECURITY_MAX_TASKS_FOR_THREAD_CALC = 100;

export const runSecurityCheck = async (
  rawFiles: RawFile[],
  progressCallback: RepomixProgressCallback = () => {},
  gitDiffResult?: GitDiffResult,
  gitLogResult?: GitLogResult,
  deps = {
    initTaskRunner,
  },
  options: {
    taskRunner?: TaskRunner<SecurityCheckTask, SuspiciousFileResult | null>;
  } = {},
): Promise<SuspiciousFileResult[]> => {
  const gitDiffTasks: SecurityCheckTask[] = [];
  const gitLogTasks: SecurityCheckTask[] = [];

  // Add Git diff content for security checking if available
  if (gitDiffResult) {
    if (gitDiffResult.workTreeDiffContent) {
      gitDiffTasks.push({
        filePath: 'Working tree changes',
        content: gitDiffResult.workTreeDiffContent,
        type: 'gitDiff',
      });
    }

    if (gitDiffResult.stagedDiffContent) {
      gitDiffTasks.push({
        filePath: 'Staged changes',
        content: gitDiffResult.stagedDiffContent,
        type: 'gitDiff',
      });
    }
  }

  // Add Git log content for security checking if available
  if (gitLogResult) {
    if (gitLogResult.logContent) {
      gitLogTasks.push({
        filePath: 'Git log history',
        content: gitLogResult.logContent,
        type: 'gitLog',
      });
    }
  }

  // Use pre-created task runner if provided (pool lifecycle managed by caller),
  // otherwise create a new one with a capped thread count to reduce CPU contention.
  const taskRunner =
    options.taskRunner ??
    deps.initTaskRunner<SecurityCheckTask, SuspiciousFileResult | null>({
      numOfTasks: Math.min(
        rawFiles.length + gitDiffTasks.length + gitLogTasks.length,
        SECURITY_MAX_TASKS_FOR_THREAD_CALC,
      ),
      workerType: 'securityCheck',
      runtime: 'worker_threads',
    });
  const ownsTaskRunner = !options.taskRunner;

  const fileTasks: SecurityCheckTask[] = rawFiles.map((file) => ({
    filePath: file.path,
    content: file.content,
    type: 'file',
  }));

  // Combine file tasks, Git diff tasks, and Git log tasks
  const allTasks = [...fileTasks, ...gitDiffTasks, ...gitLogTasks];

  try {
    logger.trace(`Starting security check for ${allTasks.length} files/content`);
    const startTime = process.hrtime.bigint();

    // Batch tasks to reduce per-task overhead. Each batch is sent as a single worker task,
    // avoiding the message passing and promise resolution overhead of individual tasks.
    const batches: SecurityCheckTask[][] = [];
    for (let i = 0; i < allTasks.length; i += FILES_PER_BATCH) {
      batches.push(allTasks.slice(i, i + FILES_PER_BATCH));
    }

    let completedFiles = 0;
    const totalFiles = allTasks.length;

    const batchResults = await Promise.all(
      batches.map(async (batch) => {
        const batchResult = await taskRunner.runNamed!<SecurityCheckBatchTask, (SuspiciousFileResult | null)[]>(
          'runSecurityCheckBatch',
          { tasks: batch },
        );

        for (const task of batch) {
          completedFiles++;
          progressCallback(`Running security check... (${completedFiles}/${totalFiles}) ${pc.dim(task.filePath)}`);
          logger.trace(`Running security check... (${completedFiles}/${totalFiles}) ${task.filePath}`);
        }

        return batchResult;
      }),
    );

    const results = batchResults.flat();

    const endTime = process.hrtime.bigint();
    const duration = Number(endTime - startTime) / 1e6;
    logger.trace(`Security check completed in ${duration.toFixed(2)}ms`);

    return results.filter((result): result is SuspiciousFileResult => result !== null);
  } catch (error) {
    logger.error('Error during security check:', error);
    throw error;
  } finally {
    // Only cleanup worker pool if we created it (not externally managed).
    // Fire-and-forget: all tasks are complete, workers terminate on process exit.
    if (ownsTaskRunner) {
      taskRunner.cleanup().catch(() => {});
    }
  }
};
