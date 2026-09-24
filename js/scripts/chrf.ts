// chrF (character n-gram F-score, beta=2) between two SRT/text files.
// Used to measure how far simultaneous output drifts from the offline
// (whole-utterance) translation of the same audio.
import { readFileSync } from 'node:fs';

const text = (p: string) =>
  readFileSync(p, 'utf8')
    .split('\n')
    .filter((l) => l && !/^\d+$/.test(l) && !l.includes('-->'))
    .join(' ')
    .replace(/\s+/g, ' ')
    .trim();

function ngrams(s: string, n: number) {
  const m = new Map<string, number>();
  const t = s.replace(/ /g, '');
  for (let i = 0; i + n <= t.length; i++) m.set(t.slice(i, i + n), (m.get(t.slice(i, i + n)) ?? 0) + 1);
  return m;
}

export function chrf(hyp: string, ref: string, N = 6, beta = 2) {
  let p = 0;
  let r = 0;
  for (let n = 1; n <= N; n++) {
    const h = ngrams(hyp, n);
    const g = ngrams(ref, n);
    let match = 0;
    for (const [k, c] of h) match += Math.min(c, g.get(k) ?? 0);
    const hs = [...h.values()].reduce((a, b) => a + b, 0);
    const gs = [...g.values()].reduce((a, b) => a + b, 0);
    p += hs ? match / hs : 0;
    r += gs ? match / gs : 0;
  }
  p /= N;
  r /= N;
  return p + r ? ((1 + beta * beta) * p * r) / (beta * beta * p + r) * 100 : 0;
}

if (import.meta.main ?? process.argv[1]?.endsWith('chrf.ts')) {
  const [hyp, ref] = process.argv.slice(2);
  console.log(chrf(text(hyp), text(ref)).toFixed(1));
}
