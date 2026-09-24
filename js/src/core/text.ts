// Output hygiene for Whisper on stream audio.

// Phrases Whisper produces from music / silence (subtitle-credit style text from its training data).
const HALLUCINATIONS = [
  /^thank(s| you) (so much )?for watching[.!]*$/i,
  /^please (like and )?subscribe[.!]*$/i,
  /^(see you|bye)[,.!]* ?(next time|in the next video)?[.!]*$/i,
  /subtitles? (by|created by)/i,
  /^(\.|\s|♪|music)+$/i,
  /^thank you[.!]*$/i,
];

export function isHallucination(text: string, avgLogprob: number, noSpeech: number, seconds: number): boolean {
  const t = text.trim();
  if (!t) return false;
  // Low confidence on an utterance the model itself flags as non-speech.
  if (noSpeech > 0.5 && avgLogprob < -0.8) return true;
  // Canned outro lines on short/uncertain audio.
  if (HALLUCINATIONS.some((re) => re.test(t)) && (seconds < 4 || avgLogprob < -0.5)) return true;
  return false;
}

/** True when the tail of `ids` repeats a short n-gram 4+ times. */
export function hasLoop(ids: number[]): boolean {
  const n = ids.length;
  for (let k = 1; k <= 8; k++) {
    const reps = k <= 2 ? 6 : 4;
    if (n < k * reps) continue;
    let ok = true;
    for (let r = 1; r < reps && ok; r++) {
      for (let i = 0; i < k; i++) {
        if (ids[n - 1 - i] !== ids[n - 1 - i - r * k]) {
          ok = false;
          break;
        }
      }
    }
    if (ok) return true;
  }
  return false;
}
