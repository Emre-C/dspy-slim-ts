/**
 * Golden / benchmark answer comparison for registry metrics (exact_match_numeric,
 * exact_match_normalized, exact_match_normalized_contains, default string equality).
 */

const NUMBER_WORDS = new Map<string, string>([
  ['zero', '0'],
  ['one', '1'],
  ['two', '2'],
  ['three', '3'],
  ['four', '4'],
  ['five', '5'],
  ['six', '6'],
  ['seven', '7'],
  ['eight', '8'],
  ['nine', '9'],
  ['ten', '10'],
  ['eleven', '11'],
  ['twelve', '12'],
  ['thirteen', '13'],
  ['fourteen', '14'],
  ['fifteen', '15'],
  ['sixteen', '16'],
  ['seventeen', '17'],
  ['eighteen', '18'],
  ['nineteen', '19'],
  ['twenty', '20'],
]);

function extractLastNumeric(value: unknown): string {
  const matches = String(value).match(/-?\d[\d,]*\.?\d*/g);
  if (!matches || matches.length === 0) {
    return String(value);
  }
  return matches.at(-1)!.replaceAll(',', '');
}

function normalizeMetricText(value: unknown): string {
  const normalized = String(value)
    .normalize('NFD')
    .toLowerCase()
    .replace(/[!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]/g, '')
    .replace(/\b(a|an|the)\b/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  return normalized
    .split(' ')
    .filter((token) => token.length > 0)
    .map((token) => NUMBER_WORDS.get(token) ?? token)
    .join(' ');
}

export function answersMatch(actual: unknown, expected: unknown, metric: string): boolean {
  if (metric === 'exact_match_numeric') {
    return extractLastNumeric(actual) === extractLastNumeric(expected);
  }
  if (metric === 'exact_match_normalized') {
    return normalizeMetricText(actual) === normalizeMetricText(expected);
  }
  if (metric === 'exact_match_normalized_contains') {
    const normalizedActual = normalizeMetricText(actual);
    const normalizedExpected = normalizeMetricText(expected);
    return normalizedActual === normalizedExpected
      || normalizedActual.includes(normalizedExpected)
      || normalizedExpected.includes(normalizedActual);
  }
  return String(actual) === String(expected);
}
