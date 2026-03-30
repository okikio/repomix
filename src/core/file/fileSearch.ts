import type { Stats } from 'node:fs';
import fs from 'node:fs/promises';
import path from 'node:path';
import { type Options as GlobbyOptions, globby } from 'globby';
import { minimatch } from 'minimatch';
import picomatch from 'picomatch';
import type { RepomixConfigMerged } from '../../config/configSchema.js';
import { defaultIgnoreList } from '../../config/defaultIgnore.js';
import { RepomixError } from '../../shared/errorHandle.js';
import { logger } from '../../shared/logger.js';
import { execGitLsFiles } from '../git/gitCommand.js';
import { sortPaths } from './filePathSort.js';

import { checkDirectoryPermissions, PermissionError } from './permissionCheck.js';

export interface FileSearchResult {
  filePaths: string[];
  emptyDirPaths: string[];
}

const findEmptyDirectories = async (
  rootDir: string,
  directories: string[],
  ignorePatterns: string[],
): Promise<string[]> => {
  const emptyDirs: string[] = [];

  for (const dir of directories) {
    const fullPath = path.join(rootDir, dir);
    try {
      const entries = await fs.readdir(fullPath);
      const hasVisibleContents = entries.some((entry) => !entry.startsWith('.'));

      if (!hasVisibleContents) {
        // This checks if the directory itself matches any ignore patterns
        const shouldIgnore = ignorePatterns.some((pattern) => minimatch(dir, pattern) || minimatch(`${dir}/`, pattern));

        if (!shouldIgnore) {
          emptyDirs.push(dir);
        }
      }
    } catch (error) {
      logger.debug(`Error checking directory ${dir}:`, error);
    }
  }

  return emptyDirs;
};

// Check if a path is a git worktree reference file
const isGitWorktreeRef = async (gitPath: string): Promise<boolean> => {
  try {
    const stats = await fs.stat(gitPath);
    if (!stats.isFile()) {
      return false;
    }

    const content = await fs.readFile(gitPath, 'utf8');
    return content.startsWith('gitdir:');
  } catch {
    return false;
  }
};

/**
 * Escapes special characters in glob patterns to handle paths with parentheses.
 * Example: "src/(categories)" -> "src/\\(categories\\)"
 */
export const escapeGlobPattern = (pattern: string): string => {
  // First escape backslashes
  const escapedBackslashes = pattern.replace(/\\/g, '\\\\');
  // Then escape special characters () and [], but NOT {}
  return escapedBackslashes.replace(/[()[\]]/g, '\\$&');
};

/**
 * Normalizes glob patterns by removing trailing slashes and ensuring consistent directory pattern handling.
 * Makes "**\/folder", "**\/folder/", and "**\/folder/**\/*" behave identically.
 *
 * @param pattern The glob pattern to normalize
 * @returns The normalized pattern
 */
export const normalizeGlobPattern = (pattern: string): string => {
  // Remove trailing slash but preserve patterns that end with "**/"
  if (pattern.endsWith('/') && !pattern.endsWith('**/')) {
    return pattern.slice(0, -1);
  }

  // Convert **/folder to **/folder/** for consistent ignore pattern behavior
  if (pattern.startsWith('**/') && !pattern.includes('/**')) {
    return `${pattern}/**`;
  }

  return pattern;
};

/**
 * Collects ignore patterns from .repomixignore and .ignore files
 * found in the git ls-files output. Only reads files that are known
 * to exist (present in the file listing), avoiding unnecessary I/O.
 */
const collectIgnoreFilePatterns = async (
  rootDir: string,
  config: RepomixConfigMerged,
  allFiles: string[],
): Promise<string[]> => {
  const ignoreFileNames = new Set<string>(['.repomixignore']);
  if (config.ignore.useDotIgnore) {
    ignoreFileNames.add('.ignore');
  }

  // Find ignore files in the git output (avoids per-directory I/O probes)
  const ignoreFiles = allFiles.filter((filePath) => {
    const basename = path.basename(filePath);
    return ignoreFileNames.has(basename);
  });

  if (ignoreFiles.length === 0) {
    return [];
  }

  const results = await Promise.all(
    ignoreFiles.map(async (ignoreFilePath) => {
      try {
        const content = await fs.readFile(path.join(rootDir, ignoreFilePath), 'utf8');
        const filePatterns = parseIgnoreContent(content);
        const dir = path.dirname(ignoreFilePath);
        return filePatterns.map((pattern) => (dir === '.' ? pattern : `${dir}/${pattern}`));
      } catch {
        return [];
      }
    }),
  );

  return results.flat();
};

/**
 * Fast file search using `git ls-files` for git repositories.
 * Lists both tracked and untracked files (respecting .gitignore),
 * then applies repomix ignore/include patterns locally.
 * Returns null if git ls-files is not available or not applicable.
 *
 * Uses non-normalized ignore patterns because normalizeGlobPattern transforms
 * patterns for globby semantics (e.g. **\/folder → **\/folder/**) which break
 * minimatch file matching.
 */
const searchFilesGit = async (
  rootDir: string,
  config: RepomixConfigMerged,
  ignorePatterns: string[],
  includePatterns: string[],
  deps = { execGitLsFiles },
): Promise<string[] | null> => {
  try {
    const allFiles = await deps.execGitLsFiles(rootDir);

    logger.debug(`[git ls-files] Found ${allFiles.length} files`);

    // Collect patterns from .repomixignore and .ignore files
    // (git ls-files respects .gitignore but not these repomix-specific files)
    const ignoreFileExtraPatterns = await collectIgnoreFilePatterns(rootDir, config, allFiles);
    const allIgnorePatterns = [...ignorePatterns, ...ignoreFileExtraPatterns];

    // Expand bare directory patterns (e.g., "node_modules", "build/")
    // to also match children (e.g., "node_modules/**"), matching globby/gitignore semantics.
    // Strip trailing slashes before appending /** to avoid double-slash patterns.
    const expandedIgnore: string[] = [];
    for (const p of allIgnorePatterns) {
      expandedIgnore.push(p);
      if (!p.includes('*') && !p.endsWith('/**')) {
        const base = p.endsWith('/') ? p.slice(0, -1) : p;
        expandedIgnore.push(`${base}/**`);
      }
    }

    // Compile all patterns into single matcher functions using picomatch.
    // picomatch compiles patterns into optimized regexes, which is much faster
    // than checking each pattern individually with minimatch.
    const picoOpts = { dot: true };
    const isIgnored = picomatch(expandedIgnore, picoOpts);
    const hasCustomIncludes = !(includePatterns.length === 1 && includePatterns[0] === '**/*');
    const isIncluded = hasCustomIncludes ? picomatch(includePatterns, picoOpts) : null;

    const filteredFiles = allFiles.filter((filePath) => {
      if (isIncluded && !isIncluded(filePath)) {
        return false;
      }
      if (isIgnored(filePath)) {
        return false;
      }
      return true;
    });

    logger.debug(`[git ls-files] After filtering: ${filteredFiles.length} files`);

    return filteredFiles;
  } catch {
    logger.debug('[git ls-files] Not available, falling back to globby');
    return null;
  }
};

// Get all file paths considering the config
export const searchFiles = async (
  rootDir: string,
  config: RepomixConfigMerged,
  explicitFiles?: string[],
): Promise<FileSearchResult> => {
  // Check if the path exists and get its type
  let pathStats: Stats;
  try {
    pathStats = await fs.stat(rootDir);
  } catch (error) {
    if (error instanceof Error && 'code' in error) {
      const errorCode = (error as NodeJS.ErrnoException).code;
      if (errorCode === 'ENOENT') {
        throw new RepomixError(`Target path does not exist: ${rootDir}`);
      }
      if (errorCode === 'EPERM' || errorCode === 'EACCES') {
        throw new PermissionError(
          `Permission denied while accessing path. Please check folder access permissions for your terminal app. path: ${rootDir}`,
          rootDir,
          errorCode,
        );
      }
      // Handle other specific error codes with more context
      throw new RepomixError(`Failed to access path: ${rootDir}. Error code: ${errorCode}. ${error.message}`);
    }
    // Preserve original error stack trace for debugging
    const repomixError = new RepomixError(
      `Failed to access path: ${rootDir}. Reason: ${error instanceof Error ? error.message : JSON.stringify(error)}`,
    );
    repomixError.cause = error;
    throw repomixError;
  }

  // Check if the path is a directory
  if (!pathStats.isDirectory()) {
    throw new RepomixError(
      `Target path is not a directory: ${rootDir}. Please specify a directory path, not a file path.`,
    );
  }

  // Now check directory permissions
  const permissionCheck = await checkDirectoryPermissions(rootDir);

  if (permissionCheck.details?.read !== true) {
    if (permissionCheck.error instanceof PermissionError) {
      throw permissionCheck.error;
    }
    throw new RepomixError(
      `Target directory is not readable or does not exist. Please check folder access permissions for your terminal app.\npath: ${rootDir}`,
    );
  }

  try {
    const { adjustedIgnorePatterns, rawIgnorePatterns, ignoreFilePatterns } = await prepareIgnoreContext(
      rootDir,
      config,
    );

    logger.trace('Ignore patterns:', adjustedIgnorePatterns);
    logger.trace('Ignore file patterns:', ignoreFilePatterns);

    // Start with configured include patterns
    let includePatterns = config.include.map((pattern) => escapeGlobPattern(pattern));

    // If explicit files are provided, add them to include patterns
    if (explicitFiles) {
      if (explicitFiles.length === 0) {
        logger.warn('[stdin mode] No files received from stdin. Will search all files matching include patterns.');
      } else {
        logger.debug(`[stdin mode] Processing ${explicitFiles.length} explicit files`);
        logger.trace('[stdin mode] Explicit files (absolute):', explicitFiles);

        const relativePaths = explicitFiles.map((filePath) => {
          const relativePath = path.relative(rootDir, filePath);
          // Escape the path to handle special characters
          return escapeGlobPattern(relativePath);
        });

        logger.trace('[stdin mode] Explicit files (relative, escaped):', relativePaths);
        logger.trace('[stdin mode] Include patterns before merge:', includePatterns);

        includePatterns = [...includePatterns, ...relativePaths];

        logger.debug(`[stdin mode] Total include patterns after merge: ${includePatterns.length}`);
      }
    }

    // If no include patterns at all, default to all files
    if (includePatterns.length === 0) {
      includePatterns = ['**/*'];
    }

    logger.trace('Include patterns with explicit files:', includePatterns);
    logger.trace('Ignore patterns:', adjustedIgnorePatterns);
    logger.trace('Ignore file patterns (for globby):', ignoreFilePatterns);

    // Try fast git ls-files path for git repositories.
    // git ls-files is ~10-20x faster than globby for file listing because it reads
    // from git's index instead of traversing the filesystem.
    // Only used when gitignore is enabled (git ls-files respects .gitignore natively)
    // and no explicit files are provided (stdin mode needs globby's pattern matching).
    let filePaths: string[] | null = null;
    let usedGitPath = false;
    if (config.ignore.useGitignore && !explicitFiles) {
      const gitStartTime = Date.now();
      filePaths = await searchFilesGit(rootDir, config, rawIgnorePatterns, includePatterns);
      if (filePaths !== null) {
        usedGitPath = true;
        const gitElapsedTime = Date.now() - gitStartTime;
        logger.debug(`[git ls-files] Completed in ${gitElapsedTime}ms, found ${filePaths.length} files`);
      }
    }

    // Fall back to globby if git ls-files is not available
    if (filePaths === null) {
      logger.debug('[globby] Starting file search...');
      const globbyStartTime = Date.now();

      filePaths = await globby(includePatterns, {
        ...createBaseGlobbyOptions(rootDir, config, adjustedIgnorePatterns, ignoreFilePatterns),
        onlyFiles: true,
      }).catch((error: unknown) => {
        // Handle EPERM errors specifically
        const code = (error as NodeJS.ErrnoException | { code?: string })?.code;
        if (code === 'EPERM' || code === 'EACCES') {
          throw new PermissionError(
            `Permission denied while scanning directory. Please check folder access permissions for your terminal app. path: ${rootDir}`,
            rootDir,
          );
        }
        throw error;
      });

      const globbyElapsedTime = Date.now() - globbyStartTime;
      logger.debug(`[globby] Completed in ${globbyElapsedTime}ms, found ${filePaths.length} files`);
    }

    let emptyDirPaths: string[] = [];
    if (config.output.includeEmptyDirectories) {
      logger.debug('[empty dirs] Searching for empty directories...');
      const emptyDirStartTime = Date.now();

      // When git path was used, derive directories from the file list to avoid globby.
      // Extract unique parent directory paths, then check which are "empty"
      // (no visible non-dot entries). Falls back to globby for non-git repos.
      let directories: string[];
      if (usedGitPath) {
        // Derive directories from git file paths (which always use forward slashes).
        // Use string splitting instead of path.dirname to preserve forward slashes on Windows.
        const dirSet = new Set<string>();
        for (const filePath of filePaths) {
          const parts = filePath.split('/');
          for (let i = 1; i < parts.length; i++) {
            const dir = parts.slice(0, i).join('/');
            dirSet.add(dir);
          }
        }
        directories = Array.from(dirSet);
      } else {
        directories = await globby(includePatterns, {
          ...createBaseGlobbyOptions(rootDir, config, adjustedIgnorePatterns, ignoreFilePatterns),
          onlyDirectories: true,
        });
      }

      const emptyDirElapsedTime = Date.now() - emptyDirStartTime;
      logger.debug(`[empty dirs] Found ${directories.length} directories in ${emptyDirElapsedTime}ms`);

      const filterStartTime = Date.now();
      emptyDirPaths = await findEmptyDirectories(rootDir, directories, adjustedIgnorePatterns);
      const filterTime = Date.now() - filterStartTime;
      logger.debug(`[empty dirs] Filtered to ${emptyDirPaths.length} empty directories in ${filterTime}ms`);
    }

    logger.debug(`[result] Total files: ${filePaths.length}, empty directories: ${emptyDirPaths.length}`);
    logger.trace(`Filtered ${filePaths.length} files`);

    return {
      filePaths: sortPaths(filePaths),
      emptyDirPaths: sortPaths(emptyDirPaths),
    };
  } catch (error: unknown) {
    // Re-throw PermissionError as is
    if (error instanceof PermissionError) {
      throw error;
    }

    if (error instanceof Error) {
      logger.error('Error filtering files:', error.message);
      throw new Error(`Failed to filter files in directory ${rootDir}. Reason: ${error.message}`);
    }

    logger.error('An unexpected error occurred:', error);
    throw new Error('An unexpected error occurred while filtering files.');
  }
};

export const parseIgnoreContent = (content: string): string[] => {
  if (!content) return [];

  return content.split('\n').reduce<string[]>((acc, line) => {
    const trimmedLine = line.trim();
    if (trimmedLine && !trimmedLine.startsWith('#')) {
      acc.push(trimmedLine);
    }
    return acc;
  }, []);
};

/**
 * Prepares ignore context including patterns and file patterns with git worktree handling.
 * This logic is shared across searchFiles, listDirectories, and listFiles.
 *
 * @param rootDir The root directory to search
 * @param config The merged configuration
 * @returns Object containing adjusted ignore patterns and ignore file patterns
 */
const prepareIgnoreContext = async (
  rootDir: string,
  config: RepomixConfigMerged,
): Promise<{ adjustedIgnorePatterns: string[]; rawIgnorePatterns: string[]; ignoreFilePatterns: string[] }> => {
  const [ignorePatterns, ignoreFilePatterns] = await Promise.all([
    getIgnorePatterns(rootDir, config),
    getIgnoreFilePatterns(config),
  ]);

  // Keep raw patterns for git ls-files path (minimatch needs original glob semantics)
  const rawIgnorePatterns = [...ignorePatterns];

  // Normalize ignore patterns to handle trailing slashes consistently (for globby)
  const normalizedIgnorePatterns = ignorePatterns.map(normalizeGlobPattern);

  // Check if .git is a worktree reference
  const gitPath = path.join(rootDir, '.git');
  const isWorktree = await isGitWorktreeRef(gitPath);

  // Modify ignore patterns for git worktree
  const adjustedIgnorePatterns = [...normalizedIgnorePatterns];
  if (isWorktree) {
    // Remove '.git/**' pattern and add '.git' to ignore the reference file
    const gitIndex = adjustedIgnorePatterns.indexOf('.git/**');
    if (gitIndex !== -1) {
      adjustedIgnorePatterns.splice(gitIndex, 1);
      adjustedIgnorePatterns.push('.git');
    }
  }

  // Apply worktree adjustment to raw patterns for git ls-files path
  if (isWorktree) {
    const gitIndex = rawIgnorePatterns.indexOf('.git/**');
    if (gitIndex !== -1) {
      rawIgnorePatterns.splice(gitIndex, 1);
      rawIgnorePatterns.push('.git');
    }
  }

  return { adjustedIgnorePatterns, rawIgnorePatterns, ignoreFilePatterns };
};

/**
 * Creates base globby options with common ignore patterns.
 * Returns options that can be extended with specific settings like onlyFiles or onlyDirectories.
 */
const createBaseGlobbyOptions = (
  rootDir: string,
  config: RepomixConfigMerged,
  ignorePatterns: string[],
  ignoreFilePatterns: string[],
): Omit<GlobbyOptions, 'onlyFiles' | 'onlyDirectories'> => ({
  cwd: rootDir,
  ignore: ignorePatterns,
  gitignore: config.ignore.useGitignore,
  ignoreFiles: ignoreFilePatterns,
  absolute: false,
  dot: true,
  followSymbolicLinks: false,
});

export const getIgnoreFilePatterns = async (config: RepomixConfigMerged): Promise<string[]> => {
  const ignoreFilePatterns: string[] = [];

  // Note: When ignore files are found in nested directories, files in deeper
  // directories have higher priority, following the behavior of ripgrep and fd.
  // For example, `src/.ignore` patterns override `./.ignore` patterns.
  //
  // Multiple ignore files in the same directory (.gitignore, .ignore, .repomixignore)
  // are all merged together. The order in this array does not affect priority.
  //
  // .gitignore files are handled by globby's gitignore option (not ignoreFiles)
  // to properly respect parent directory .gitignore files, matching Git's behavior.

  if (config.ignore.useDotIgnore) {
    ignoreFilePatterns.push('**/.ignore');
  }

  ignoreFilePatterns.push('**/.repomixignore');

  return ignoreFilePatterns;
};

export const getIgnorePatterns = async (rootDir: string, config: RepomixConfigMerged): Promise<string[]> => {
  const ignorePatterns = new Set<string>();

  // Add default ignore patterns
  if (config.ignore.useDefaultPatterns) {
    logger.trace('Adding default ignore patterns');
    for (const pattern of defaultIgnoreList) {
      ignorePatterns.add(pattern);
    }
  }

  // Add repomix output file
  if (config.output.filePath) {
    const absoluteOutputPath = path.resolve(config.cwd, config.output.filePath);
    const relativeToTargetPath = path.relative(rootDir, absoluteOutputPath);

    logger.trace('Adding output file to ignore patterns:', relativeToTargetPath);

    ignorePatterns.add(relativeToTargetPath);
  }

  // Add custom ignore patterns
  if (config.ignore.customPatterns) {
    logger.trace('Adding custom ignore patterns:', config.ignore.customPatterns);
    for (const pattern of config.ignore.customPatterns) {
      ignorePatterns.add(pattern);
    }
  }

  // Add patterns from .git/info/exclude if useGitignore is enabled
  if (config.ignore.useGitignore) {
    // Read .git/info/exclude file
    const excludeFilePath = path.join(rootDir, '.git', 'info', 'exclude');
    try {
      const excludeFileContent = await fs.readFile(excludeFilePath, 'utf8');
      const excludePatterns = parseIgnoreContent(excludeFileContent);

      for (const pattern of excludePatterns) {
        ignorePatterns.add(pattern);
      }
    } catch (error) {
      // File might not exist or might not be accessible, which is fine
      logger.trace('Could not read .git/info/exclude file:', error instanceof Error ? error.message : String(error));
    }
  }

  return Array.from(ignorePatterns);
};

/**
 * Lists all directories in the given root directory, respecting ignore patterns.
 * This function does not apply include patterns - it returns the full directory set subject to ignore rules.
 *
 * @param rootDir The root directory to scan
 * @param config The merged configuration
 * @returns Array of directory paths relative to rootDir
 */
export const listDirectories = async (rootDir: string, config: RepomixConfigMerged): Promise<string[]> => {
  const { adjustedIgnorePatterns, ignoreFilePatterns } = await prepareIgnoreContext(rootDir, config);

  const directories = await globby(['**/*'], {
    ...createBaseGlobbyOptions(rootDir, config, adjustedIgnorePatterns, ignoreFilePatterns),
    onlyDirectories: true,
  });

  return sortPaths(directories);
};

/**
 * Lists all files in the given root directory, respecting ignore patterns.
 * This function does not apply include patterns - it returns the full file set subject to ignore rules.
 *
 * @param rootDir The root directory to scan
 * @param config The merged configuration
 * @returns Array of file paths relative to rootDir
 */
export const listFiles = async (rootDir: string, config: RepomixConfigMerged): Promise<string[]> => {
  const { adjustedIgnorePatterns, ignoreFilePatterns } = await prepareIgnoreContext(rootDir, config);

  const files = await globby(['**/*'], {
    ...createBaseGlobbyOptions(rootDir, config, adjustedIgnorePatterns, ignoreFilePatterns),
    onlyFiles: true,
  });

  return sortPaths(files);
};
