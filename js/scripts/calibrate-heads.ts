// Ranks the 40 decoder cross-attention heads (2 layers x 20) by how well they
// track the audio: for each VAD segment we decode fully and, per head, take the
// argmax frame of every text token. A good alignment head is monotonic
// (Spearman rho vs. token index), spans the utterance, and keeps its mass on
// real audio rather than the zero padding.
//
//   node scripts/calibrate-heads.ts ../samples/*.16k.wav
import { readWav, setupNode } from '../src/node/env.ts';
import { AudioPipeline, EVENTS } from '../src/core/wasm.ts';
import { SileroVad } from '../src/core/vad.ts';
import { TOK, WhisperRunner } from '../src/core/whisper.ts';

await setupNode();
const files = process.argv.slice(2);
const H = 40;
const F = 1500;
const r = await WhisperRunner.load('kotoba-whisper-bilingual-v1.0-xattn', {
  device: process.env.DEVICE ?? 'webgpu',
  dtype: { encoder: 'fp16', decoder: 'fp16' },
  heads: Array.from({ length: H }, (_, i) => i),
});
const vad = await SileroVad.load('silero-vad', 'cpu');

function ranks(v: number[]) {
  const idx = v.map((x, i) => [x, i]).sort((a, b) => a[0] - b[0]);
  const out = new Array(v.length);
  idx.forEach(([, i], k) => (out[i] = k));
  return out;
}
function spearman(a: number[]) {
  const n = a.length;
  const ra = ranks(a);
  let d2 = 0;
  for (let i = 0; i < n; i++) d2 += (ra[i] - i) ** 2;
  return 1 - (6 * d2) / (n * (n * n - 1));
}

const stats = { transcribe: new Map<number, number[][]>(), translate: new Map<number, number[][]>() };
for (const s of Object.values(stats)) for (let h = 0; h < H; h++) s.set(h, []);

async function run(mel: Float32Array, content: number, task: number, lang: number, key: 'transcribe' | 'translate') {
  const enc = await r.encode(mel);
  const rows: Float32Array[] = []; // per generated token: [H][F]
  let out = await r.decode([TOK.SOT, lang, task, TOK.NO_TIMESTAMPS], enc);
  const lastRow = (o: typeof out) => {
    const x = new Float32Array(H * F);
    for (let h = 0; h < H; h++) x.set(o.xattn.subarray((h * o.rows + o.rows - 1) * F, (h * o.rows + o.rows) * F), h * F);
    return x;
  };
  const ids: number[] = [];
  while (ids.length < 120) {
    const next = r.argmax(out.logits);
    if (next === TOK.EOT) break;
    rows.push(lastRow(out));
    ids.push(next);
    out = await r.decode([next], enc, out.cache);
  }
  r.releaseCache(out.cache);
  enc.dispose();
  if (ids.length < 5) return '';
  for (let h = 0; h < H; h++) {
    const am = rows.map((x) => {
      let b = 0;
      for (let f = 1; f < F; f++) if (x[h * F + f] > x[h * F + b]) b = f;
      return b;
    });
    let inContent = 0;
    for (const x of rows) for (let f = 0; f < Math.min(content, F); f++) inContent += x[h * F + f];
    const cover = (Math.max(...am) - Math.min(...am)) / content;
    stats[key].get(h)!.push([spearman(am), Math.min(1, cover), inContent / rows.length]);
  }
  return r.decodeText(ids);
}

let segs = 0;
for (const file of files) {
  const { rate, pcm } = readWav(file);
  const p = new AudioPipeline(rate, 80, 300, 128);
  p.configure_vad(0.5, 0.35, 500, 100, 10000, 24000, 250);
  p.push_input(pcm);
  const mel = new Float32Array(p.mel_size());
  while (p.vad_frame_ready()) {
    const ev = p.push_vad_prob(await vad.prob(p.vad_input()));
    if (ev & EVENTS.END && !(ev & EVENTS.DISCARD) && p.segment_len() > 16000 * 1.5) {
      const content = p.mel(mel);
      const ja = await run(mel, content, TOK.TRANSCRIBE, TOK.JA, 'transcribe');
      const en = await run(mel, content, TOK.TRANSLATE, TOK.EN, 'translate');
      segs++;
      if (process.env.VERBOSE) console.log(`${(p.segment_start() / 16000).toFixed(1)}s | ${ja} | ${en}`);
    }
    if (ev & EVENTS.END) p.clear_segment();
  }
}

const mean = (xs: number[][], k: number) => xs.reduce((a, x) => a + x[k], 0) / Math.max(1, xs.length);
const table = [...Array(H).keys()].map((h) => {
  const a = stats.transcribe.get(h)!;
  const t = stats.translate.get(h)!;
  const asr = mean(a, 0) * mean(a, 1) * mean(a, 2);
  return { h, layer: Math.floor(h / 20), head: h % 20, rhoAsr: mean(a, 0), cover: mean(a, 1), mass: mean(a, 2), rhoTr: mean(t, 0), score: asr };
});
table.sort((x, y) => y.score - x.score);
console.log(`${segs} segments`);
console.log('id  L.h    rho_asr cover  mass   rho_tr score');
for (const t of table)
  console.log(
    `${String(t.h).padStart(2)}  ${t.layer}.${String(t.head).padEnd(2)}   ${t.rhoAsr.toFixed(3)}  ${t.cover.toFixed(2)}   ${t.mass.toFixed(2)}   ${t.rhoTr.toFixed(3)}  ${t.score.toFixed(3)}`,
  );
