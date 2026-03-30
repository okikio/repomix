import { describe, expect, it, type Mock, vi } from 'vitest';
import type { ProcessedFile } from '../../../src/core/file/fileTypes.js';
import type { GitDiffResult } from '../../../src/core/git/gitDiffHandle.js';
import { calculateMetrics } from '../../../src/core/metrics/calculateMetrics.js';
import { calculateSelectiveFileMetrics } from '../../../src/core/metrics/calculateSelectiveFileMetrics.js';
import type { RepomixProgressCallback } from '../../../src/shared/types.js';
import { createMockConfig } from '../../testing/testUtils.js';

vi.mock('../../../src/core/metrics/TokenCounter.js', () => {
  return {
    TOKEN_ENCODINGS: ['o200k_base', 'cl100k_base', 'p50k_base', 'p50k_edit', 'r50k_base'],
    TokenCounter: vi.fn().mockImplementation(() => ({
      countTokens: vi.fn().mockReturnValue(10),
      free: vi.fn(),
    })),
  };
});
vi.mock('../../../src/core/metrics/aggregateMetrics.js');
vi.mock('../../../src/core/metrics/calculateSelectiveFileMetrics.js', () => ({
  calculateSelectiveFileMetrics: vi.fn(),
}));

describe('calculateMetrics', () => {
  it('should calculate metrics and return the result', async () => {
    const processedFiles: ProcessedFile[] = [
      { path: 'file1.txt', content: 'a'.repeat(100) },
      { path: 'file2.txt', content: 'b'.repeat(200) },
    ];
    const output = 'a'.repeat(300);
    const progressCallback: RepomixProgressCallback = vi.fn();

    const fileMetrics = [
      { path: 'file1.txt', charCount: 100, tokenCount: 10 },
      { path: 'file2.txt', charCount: 200, tokenCount: 20 },
    ];
    (calculateSelectiveFileMetrics as unknown as Mock).mockResolvedValue(fileMetrics);

    const aggregatedResult = {
      totalFiles: 2,
      totalCharacters: 300,
      totalTokens: 30,
      fileCharCounts: {
        'file1.txt': 100,
        'file2.txt': 200,
      },
      fileTokenCounts: {
        'file1.txt': 10,
        'file2.txt': 20,
      },
      gitDiffTokenCount: 0,
      gitLogTokenCount: 0,
    };

    const config = createMockConfig();

    const gitDiffResult: GitDiffResult | undefined = undefined;

    const mockTaskRunner = {
      run: vi.fn(),
      cleanup: vi.fn(),
    };

    const result = await calculateMetrics(processedFiles, output, progressCallback, config, gitDiffResult, undefined, {
      calculateSelectiveFileMetrics,
      calculateOutputMetrics: async () => 30,
      calculateGitDiffMetrics: () => Promise.resolve(0),
      calculateGitLogMetrics: () => Promise.resolve({ gitLogTokenCount: 0 }),
      taskRunner: mockTaskRunner,
    });

    expect(progressCallback).toHaveBeenCalledWith('Calculating metrics...');
    expect(calculateSelectiveFileMetrics).toHaveBeenCalledWith(
      processedFiles,
      ['file2.txt', 'file1.txt'], // sorted by character count desc
      'o200k_base',
      progressCallback,
      expect.objectContaining({
        taskRunner: expect.any(Object),
      }),
    );
    expect(result).toEqual(aggregatedResult);
  });

  it('should accept output as a Promise and start file metrics before output resolves', async () => {
    const processedFiles: ProcessedFile[] = [{ path: 'file1.txt', content: 'a'.repeat(100) }];
    const progressCallback: RepomixProgressCallback = vi.fn();

    const fileMetrics = [{ path: 'file1.txt', charCount: 100, tokenCount: 10 }];
    (calculateSelectiveFileMetrics as unknown as Mock).mockResolvedValue(fileMetrics);

    const config = createMockConfig();

    // Track call order to verify file metrics starts before output resolves
    const callOrder: string[] = [];
    const mockCalculateSelectiveFileMetrics = vi.fn().mockImplementation(async (...args: unknown[]) => {
      callOrder.push('fileMetrics:start');
      const result = await (calculateSelectiveFileMetrics as unknown as Mock)(...args);
      callOrder.push('fileMetrics:end');
      return result;
    });

    // Create a promise that resolves after a microtask to simulate async output generation
    const outputPromise = Promise.resolve().then(() => {
      callOrder.push('output:resolved');
      return 'output content';
    });

    const result = await calculateMetrics(
      processedFiles,
      outputPromise,
      progressCallback,
      config,
      undefined,
      undefined,
      {
        calculateSelectiveFileMetrics: mockCalculateSelectiveFileMetrics,
        calculateOutputMetrics: async () => 15,
        calculateGitDiffMetrics: () => Promise.resolve(0),
        calculateGitLogMetrics: () => Promise.resolve({ gitLogTokenCount: 0 }),
        taskRunner: { run: vi.fn(), cleanup: vi.fn() },
      },
    );

    // File metrics should have been called (started before output resolved)
    expect(mockCalculateSelectiveFileMetrics).toHaveBeenCalled();
    // Verify file metrics started before output resolved
    expect(callOrder.indexOf('fileMetrics:start')).toBeLessThan(callOrder.indexOf('output:resolved'));
    // Token count is now estimated from file token ratio (10 tokens / 100 chars * 14 output chars = 1)
    expect(result.totalTokens).toBe(Math.round((10 / 100) * 'output content'.length));
    expect(result.totalCharacters).toBe('output content'.length);
  });

  it('should estimate output tokens from file tokens when tokenCountTree is enabled', async () => {
    const processedFiles: ProcessedFile[] = [
      { path: 'file1.txt', content: 'a'.repeat(100) },
      { path: 'file2.txt', content: 'b'.repeat(200) },
    ];
    // Output is all file content (300 chars) plus 30 chars of overhead (330 total)
    const output = 'a'.repeat(100) + 'b'.repeat(200) + '<overhead>_padding_</overhead>';
    const progressCallback: RepomixProgressCallback = vi.fn();

    const fileMetrics = [
      { path: 'file1.txt', charCount: 100, tokenCount: 10 },
      { path: 'file2.txt', charCount: 200, tokenCount: 20 },
    ];
    (calculateSelectiveFileMetrics as unknown as Mock).mockResolvedValue(fileMetrics);

    const config = createMockConfig({ output: { tokenCountTree: 50000 } });

    const mockCalculateOutputMetrics = vi.fn();

    const result = await calculateMetrics(processedFiles, output, progressCallback, config, undefined, undefined, {
      calculateSelectiveFileMetrics,
      calculateOutputMetrics: mockCalculateOutputMetrics,
      calculateGitDiffMetrics: () => Promise.resolve(0),
      calculateGitLogMetrics: () => Promise.resolve({ gitLogTokenCount: 0 }),
      taskRunner: { run: vi.fn(), cleanup: vi.fn() },
    });

    // calculateOutputMetrics should NOT be called when tokenCountTree is enabled
    expect(mockCalculateOutputMetrics).not.toHaveBeenCalled();
    // totalTokens is estimated from file token ratio: (30/300) * 330 = 33
    expect(result.totalTokens).toBe(33);
    expect(result.totalCharacters).toBe(output.length);
    expect(result.fileTokenCounts).toEqual({ 'file1.txt': 10, 'file2.txt': 20 });
  });

  it('should estimate output tokens with promise output when tokenCountTree is enabled', async () => {
    const processedFiles: ProcessedFile[] = [{ path: 'file1.txt', content: 'a'.repeat(200) }];
    const outputPromise = Promise.resolve('a'.repeat(200) + '<header/>');
    const progressCallback: RepomixProgressCallback = vi.fn();

    const fileMetrics = [{ path: 'file1.txt', charCount: 200, tokenCount: 50 }];
    (calculateSelectiveFileMetrics as unknown as Mock).mockResolvedValue(fileMetrics);

    const config = createMockConfig({ output: { tokenCountTree: true } });
    const mockCalculateOutputMetrics = vi.fn();

    const result = await calculateMetrics(
      processedFiles,
      outputPromise,
      progressCallback,
      config,
      undefined,
      undefined,
      {
        calculateSelectiveFileMetrics,
        calculateOutputMetrics: mockCalculateOutputMetrics,
        calculateGitDiffMetrics: () => Promise.resolve(0),
        calculateGitLogMetrics: () => Promise.resolve({ gitLogTokenCount: 0 }),
        taskRunner: { run: vi.fn(), cleanup: vi.fn() },
      },
    );

    expect(mockCalculateOutputMetrics).not.toHaveBeenCalled();
    // (50/200) * 209 = 52.25 → 52
    expect(result.totalTokens).toBe(52);
  });

  it('should estimate output tokens from top files when tokenCountTree is disabled', async () => {
    const processedFiles: ProcessedFile[] = [
      { path: 'file1.txt', content: 'a'.repeat(100) },
      { path: 'file2.txt', content: 'b'.repeat(200) },
    ];
    const output = 'full output string';
    const progressCallback: RepomixProgressCallback = vi.fn();

    // Only top file by size is counted (file2.txt)
    const fileMetrics = [{ path: 'file2.txt', charCount: 200, tokenCount: 20 }];
    (calculateSelectiveFileMetrics as unknown as Mock).mockResolvedValue(fileMetrics);

    const config = createMockConfig({ output: { tokenCountTree: false } });
    const mockCalculateOutputMetrics = vi.fn().mockResolvedValue(42);

    const result = await calculateMetrics(processedFiles, output, progressCallback, config, undefined, undefined, {
      calculateSelectiveFileMetrics,
      calculateOutputMetrics: mockCalculateOutputMetrics,
      calculateGitDiffMetrics: () => Promise.resolve(0),
      calculateGitLogMetrics: () => Promise.resolve({ gitLogTokenCount: 0 }),
      taskRunner: { run: vi.fn(), cleanup: vi.fn() },
    });

    // Should estimate tokens from file ratio, not count full output
    expect(mockCalculateOutputMetrics).not.toHaveBeenCalled();
    // Estimation: 20 tokens / 200 chars * 18 output chars = 2
    expect(result.totalTokens).toBe(Math.round((20 / 200) * output.length));
  });

  it('should estimate token counts for non-tokenized files in threshold mode', async () => {
    // 4 files: 2 large (tokenized by coverage), 2 small (estimated)
    const processedFiles: ProcessedFile[] = [
      { path: 'large1.ts', content: 'a'.repeat(500) },
      { path: 'large2.ts', content: 'b'.repeat(400) },
      { path: 'small1.md', content: 'c'.repeat(50) },
      { path: 'small2.md', content: 'd'.repeat(50) },
    ];
    const output = 'x'.repeat(1000) + '<overhead/>';
    const progressCallback: RepomixProgressCallback = vi.fn();

    // Mock only returns metrics for the 2 large files (workers only tokenized those)
    const fileMetrics = [
      { path: 'large1.ts', charCount: 500, tokenCount: 100 },
      { path: 'large2.ts', charCount: 400, tokenCount: 80 },
    ];
    (calculateSelectiveFileMetrics as unknown as Mock).mockResolvedValue(fileMetrics);

    const config = createMockConfig({ output: { tokenCountTree: 10000 } });

    const result = await calculateMetrics(processedFiles, output, progressCallback, config, undefined, undefined, {
      calculateSelectiveFileMetrics,
      calculateOutputMetrics: vi.fn(),
      calculateGitDiffMetrics: () => Promise.resolve(0),
      calculateGitLogMetrics: () => Promise.resolve({ gitLogTokenCount: 0 }),
      taskRunner: { run: vi.fn(), cleanup: vi.fn() },
    });

    // Large files should have exact counts
    expect(result.fileTokenCounts['large1.ts']).toBe(100);
    expect(result.fileTokenCounts['large2.ts']).toBe(80);

    // Small files should have estimated counts from ratio: chars / (900/180) = chars * 0.2
    // charsPerToken = 900/180 = 5.0, so estimated = Math.round(50 / 5.0) = 10
    expect(result.fileTokenCounts['small1.md']).toBe(10);
    expect(result.fileTokenCounts['small2.md']).toBe(10);

    // All 4 files should have token counts
    expect(Object.keys(result.fileTokenCounts)).toHaveLength(4);
  });

  it('should handle empty files gracefully with tokenCountTree enabled', async () => {
    const processedFiles: ProcessedFile[] = [{ path: 'empty.txt', content: '' }];
    const output = '<header>no files</header>';
    const progressCallback: RepomixProgressCallback = vi.fn();

    const fileMetrics = [{ path: 'empty.txt', charCount: 0, tokenCount: 0 }];
    (calculateSelectiveFileMetrics as unknown as Mock).mockResolvedValue(fileMetrics);

    const config = createMockConfig({ output: { tokenCountTree: 50000 } });

    const result = await calculateMetrics(processedFiles, output, progressCallback, config, undefined, undefined, {
      calculateSelectiveFileMetrics,
      calculateOutputMetrics: vi.fn(),
      calculateGitDiffMetrics: () => Promise.resolve(0),
      calculateGitLogMetrics: () => Promise.resolve({ gitLogTokenCount: 0 }),
      taskRunner: { run: vi.fn(), cleanup: vi.fn() },
    });

    // When all files are empty (fileCharSum=0), fall back to totalCharacters
    expect(result.totalTokens).toBe(output.length);
    expect(result.totalCharacters).toBe(output.length);
  });
});
