// Prints VAD segments of a WAV (sanity check for the Silero wrapper + segmenter).
import { readWav, setupNode } from '../src/node/env.ts';
import { AudioPipeline, EVENTS } from '../src/core/wasm.ts';
import { SileroVad } from '../src/core/vad.ts';

await setupNode();
const file = process.argv[2] ?? '../samples/8RSIoTCrhSM.16k.wav';
const { rate, pcm } = readWav(file);
const vad = await SileroVad.load('silero-vad', 'cpu');
const p = new AudioPipeline(rate, 80, 300, 128);
p.configure_vad(0.5, 0.35, 500, 100, 10000, 24000, 250);
p.push_input(pcm);
const t0 = performance.now();
let frames = 0;
while (p.vad_frame_ready()) {
  const ev = p.push_vad_prob(await vad.prob(p.vad_input()));
  frames++;
  if (ev & EVENTS.END) {
    const s = p.segment_start() / 16000;
    console.log(`${s.toFixed(2)}\t${(s + p.segment_len() / 16000).toFixed(2)}${ev & EVENTS.DISCARD ? '\tdiscard' : ''}`);
    p.clear_segment();
  }
  if (ev & EVENTS.OVERFLOW) console.log('overflow at', (p.processed() / 16000).toFixed(2));
}
console.error(`${frames} frames in ${(performance.now() - t0).toFixed(0)} ms`);
