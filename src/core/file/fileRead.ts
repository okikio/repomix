import * as fs from 'node:fs/promises';
import { logger } from '../../shared/logger.js';

// Lazy-load binary detection modules to avoid their combined ~13ms import cost.
// is-binary-path (~8ms) and isbinaryfile (~5ms) are only needed during file
// collection, but their static imports add to the defaultAction preload chain.
// By deferring them, the preload completes ~13ms faster, which eliminates the
// gap where the main thread waits for the preload after cliRun finishes.
let _isBinaryPath: ((filePath: string) => boolean) | undefined;
let _isBinaryFile: ((bytes: Buffer, size?: number) => Promise<boolean>) | undefined;
const loadBinaryDeps = async () => {
  if (!_isBinaryPath || !_isBinaryFile) {
    const [bpMod, bfMod] = await Promise.all([import('is-binary-path'), import('isbinaryfile')]);
    _isBinaryPath = bpMod.default;
    _isBinaryFile = bfMod.isBinaryFile;
  }
  // biome-ignore lint/style/noNonNullAssertion: guaranteed assigned in the if-block above
  return { isBinaryPath: _isBinaryPath!, isBinaryFile: _isBinaryFile! };
};

/**
 * Pre-warm binary detection modules so they're ready when file collection starts.
 * Called from pack() at the start, overlapping the ~13ms import with searchFiles I/O.
 */
export const prewarmBinaryDeps = (): Promise<unknown> => loadBinaryDeps();

// Lazy-load jschardet and iconv-lite to avoid their combined ~19ms import cost.
// They're only needed for non-UTF-8 files (~1% of source code), so the common
// UTF-8 fast path never triggers the import.
let _jschardet: { detect: (buffer: Buffer) => { encoding: string } | null } | undefined;
let _iconv:
  | {
      decode: (buffer: Buffer, encoding: string, options?: object) => string;
      encodingExists: (encoding: string) => boolean;
    }
  | undefined;
const loadEncodingDeps = async () => {
  if (!_jschardet || !_iconv) {
    const [jschardetMod, iconvMod] = await Promise.all([import('jschardet'), import('iconv-lite')]);
    _jschardet = jschardetMod.default ?? jschardetMod;
    _iconv = (iconvMod as { default?: typeof _iconv }).default ?? iconvMod;
  }
  // biome-ignore lint/style/noNonNullAssertion: guaranteed assigned in the if-block above
  return { jschardet: _jschardet!, iconv: _iconv! };
};

export type FileSkipReason = 'binary-extension' | 'binary-content' | 'size-limit' | 'encoding-error';

export interface FileReadResult {
  content: string | null;
  skippedReason?: FileSkipReason;
}

/**
 * Read a file and return its text content
 * @param filePath Path to the file
 * @param maxFileSize Maximum file size in bytes
 * @returns File content as string and skip reason if file was skipped
 */
export const readRawFile = async (filePath: string, maxFileSize: number): Promise<FileReadResult> => {
  try {
    const { isBinaryPath, isBinaryFile } = await loadBinaryDeps();

    // Check binary extension first (no I/O needed) to skip read for binary files
    if (isBinaryPath(filePath)) {
      logger.debug(`Skipping binary file: ${filePath}`);
      return { content: null, skippedReason: 'binary-extension' };
    }

    logger.trace(`Reading file: ${filePath}`);

    // Read file directly and check buffer length instead of calling fs.stat() first.
    // This eliminates one libuv threadpool operation per file. Node's readFile internally
    // does open+fstat+read+close; the separate stat() added a redundant open+stat+close
    // that doubled the libuv queue depth during file collection.
    const buffer = await fs.readFile(filePath);

    if (buffer.length > maxFileSize) {
      const sizeKB = (buffer.length / 1024).toFixed(1);
      const maxSizeKB = (maxFileSize / 1024).toFixed(1);
      logger.trace(`File exceeds size limit: ${sizeKB}KB > ${maxSizeKB}KB (${filePath})`);
      return { content: null, skippedReason: 'size-limit' };
    }

    if (await isBinaryFile(buffer)) {
      logger.debug(`Skipping binary file (content check): ${filePath}`);
      return { content: null, skippedReason: 'binary-content' };
    }

    // Fast path: Try UTF-8 decoding first (covers ~99% of source code files)
    // This skips the expensive jschardet.detect() which scans the entire buffer
    // through multiple encoding probers with frequency table lookups
    try {
      let content = new TextDecoder('utf-8', { fatal: true }).decode(buffer);
      if (content.charCodeAt(0) === 0xfeff) {
        content = content.slice(1); // strip UTF-8 BOM
      }
      return { content };
    } catch {
      // Not valid UTF-8, fall through to encoding detection
    }

    // Slow path: Detect encoding with jschardet for non-UTF-8 files (e.g., Shift-JIS, EUC-KR)
    const { jschardet, iconv } = await loadEncodingDeps();
    const { encoding: detectedEncoding } = jschardet.detect(buffer) ?? {};
    const encoding = detectedEncoding && iconv.encodingExists(detectedEncoding) ? detectedEncoding : 'utf-8';
    const content = iconv.decode(buffer, encoding, { stripBOM: true });

    if (content.includes('\uFFFD')) {
      logger.debug(`Skipping file due to encoding errors (detected: ${encoding}): ${filePath}`);
      return { content: null, skippedReason: 'encoding-error' };
    }

    return { content };
  } catch (error) {
    logger.warn(`Failed to read file: ${filePath}`, error);
    return { content: null, skippedReason: 'encoding-error' };
  }
};
