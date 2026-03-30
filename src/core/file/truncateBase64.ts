// Constants for base64 detection and truncation
const MIN_BASE64_LENGTH_DATA_URI = 40;
const MIN_BASE64_LENGTH_STANDALONE = 256;
const TRUNCATION_LENGTH = 32;
const MIN_CHAR_DIVERSITY = 10;
const MIN_CHAR_TYPE_COUNT = 3;

// Pre-compiled regex patterns (avoid re-creation per file)
const dataUriPattern = new RegExp(
  `data:([a-zA-Z0-9\\/\\-\\+]+)(;[a-zA-Z0-9\\-=]+)*;base64,([A-Za-z0-9+/=]{${MIN_BASE64_LENGTH_DATA_URI},})`,
  'g',
);
const standaloneBase64Pattern = new RegExp(`([A-Za-z0-9+/]{${MIN_BASE64_LENGTH_STANDALONE},}={0,2})`, 'g');

/**
 * Fast check: does the content contain any line (segment between newlines) that is at least
 * `minLen` characters long? Uses String.indexOf which is SIMD-optimized in V8,
 * making it ~10x faster than a character-by-character JavaScript loop.
 */
const hasLongLine = (content: string, minLen: number): boolean => {
  let prev = -1;
  for (;;) {
    const next = content.indexOf('\n', prev + 1);
    if (next === -1) {
      return content.length - prev - 1 >= minLen;
    }
    if (next - prev - 1 >= minLen) {
      return true;
    }
    prev = next;
  }
};

/**
 * Scan for a contiguous run of `minLen`+ base64 characters ([A-Za-z0-9+/]).
 * Only called on the subset of files that pass the fast `hasLongLine` check,
 * so the total bytes scanned is small.
 */
const hasLongBase64Run = (content: string, minLen: number): boolean => {
  let run = 0;
  for (let i = 0; i < content.length; i++) {
    const c = content.charCodeAt(i);
    // A-Z: 65-90, a-z: 97-122, 0-9: 48-57, +: 43, /: 47
    if ((c >= 48 && c <= 57) || (c >= 65 && c <= 90) || (c >= 97 && c <= 122) || c === 43 || c === 47) {
      if (++run >= minLen) {
        return true;
      }
    } else {
      run = 0;
    }
  }
  return false;
};

/**
 * Truncates base64 encoded data in content to reduce file size.
 * Detects common base64 patterns like data URIs and standalone base64 strings.
 *
 * Uses a layered pre-check strategy to skip the expensive global regex on files
 * that cannot contain base64 data:
 *   1. Short files (< 40 chars) are returned immediately.
 *   2. Data URI check: `content.includes('base64,')` — V8 SIMD-optimized, ~1ms for 1000 files.
 *   3. Standalone check: `hasLongLine` (SIMD indexOf) then `hasLongBase64Run` (char scan)
 *      on the ~18% of files that have lines >= 256 chars.
 *   4. Full regex only on the ~0.5% of files that pass all pre-checks.
 *
 * @param content The content to process
 * @returns Content with base64 data truncated
 */
export const truncateBase64Content = (content: string): string => {
  // Fast path: file too short for any base64 pattern
  if (content.length < MIN_BASE64_LENGTH_DATA_URI) {
    return content;
  }

  // Layered pre-checks using fast V8-native string operations.
  // Data URIs always contain the literal substring "base64,".
  const mayHaveDataUri = content.includes('base64,');

  // Standalone base64 requires a 256+ char run of [A-Za-z0-9+/].
  // First check if any line is long enough (SIMD indexOf, ~2ms for 1000 files),
  // then verify with a character scan (only on the ~18% with long lines).
  const mayHaveStandalone =
    hasLongLine(content, MIN_BASE64_LENGTH_STANDALONE) && hasLongBase64Run(content, MIN_BASE64_LENGTH_STANDALONE);

  if (!mayHaveDataUri && !mayHaveStandalone) {
    return content;
  }

  // Reset lastIndex since patterns are global and reused across calls
  dataUriPattern.lastIndex = 0;
  standaloneBase64Pattern.lastIndex = 0;

  let processedContent = content;

  // Replace data URIs (only if pre-check detected the "base64," substring)
  if (mayHaveDataUri) {
    processedContent = processedContent.replace(dataUriPattern, (_match, mimeType, params, base64Data) => {
      const preview = base64Data.substring(0, TRUNCATION_LENGTH);
      return `data:${mimeType}${params || ''};base64,${preview}...`;
    });
  }

  // Replace standalone base64 strings (only if pre-check detected a long run)
  if (mayHaveStandalone) {
    processedContent = processedContent.replace(standaloneBase64Pattern, (match, base64String) => {
      // Check if this looks like actual base64 (not just a long string)
      if (isLikelyBase64(base64String)) {
        const preview = base64String.substring(0, TRUNCATION_LENGTH);
        return `${preview}...`;
      }
      return match;
    });
  }

  return processedContent;
};

/**
 * Checks if a string is likely to be base64 encoded data
 *
 * @param str The string to check
 * @returns True if the string appears to be base64 encoded
 */
function isLikelyBase64(str: string): boolean {
  // Check for valid base64 characters only
  if (!/^[A-Za-z0-9+/]+=*$/.test(str)) {
    return false;
  }

  // Check for reasonable distribution of characters (not all same char)
  const charSet = new Set(str);
  if (charSet.size < MIN_CHAR_DIVERSITY) {
    return false;
  }

  // Additional check: base64 encoded binary data typically has good character distribution
  // Must have at least MIN_CHAR_TYPE_COUNT of the 4 character types (numbers, uppercase, lowercase, special)
  const hasNumbers = /[0-9]/.test(str);
  const hasUpperCase = /[A-Z]/.test(str);
  const hasLowerCase = /[a-z]/.test(str);
  const hasSpecialChars = /[+/]/.test(str);

  // Real base64 encoded binary data virtually always contains digits
  if (!hasNumbers) {
    return false;
  }

  const charTypeCount = [hasNumbers, hasUpperCase, hasLowerCase, hasSpecialChars].filter(Boolean).length;

  return charTypeCount >= MIN_CHAR_TYPE_COUNT;
}
