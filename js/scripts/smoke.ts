// Sanity checks: wasm mel == transformers.js mel; full (non-streaming) translation runs.
import { WhisperFeatureExtractor } from '@huggingface/transformers';
import { readWav, setupNode } from '../src/node/env.ts';
import { AudioPipeline } from '../src/core/wasm.ts';
import { TOK, WhisperRunner } from '../src/core/whisper.ts';

await setupNode();
const [file = '../samples/8RSIoTCrhSM.16k.wav', from = '0', to = '12'] = process.argv.slice(2);
const { pcm } = readWav(file);
const audio = pcm.subarray(Math.round(+from * 16000), Math.round(+to * 16000));

// Mel parity: feed the clip as one "segment" through the pipeline with VAD forced on.
const p = new AudioPipeline(16000, 0, 0, 128);
p.push_input(audio);
while (p.vad_frame_ready()) p.push_vad_prob(1);
const mel = new Float32Array(p.mel_size());
const frames = p.mel(mel);
const fe = await WhisperFeatureExtractor.from_pretrained('kotoba-whisper-bilingual-v1.0-xattn');
const segAudio = p.segment_audio();
const ref = (await fe(segAudio)).input_features.data as Float32Array;
let maxErr = 0;
for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(ref[i] - mel[i]));
console.log(`mel: ${frames} content frames, max |wasm - transformers.js| = ${maxErr.toExponential(2)}`);

const device = process.env.DEVICE ?? 'cuda';
const dt = process.env.DTYPE ?? 'fp16';
const t0 = performance.now();
const r = await WhisperRunner.load('kotoba-whisper-bilingual-v1.0-xattn', {
  device,
  dtype: { encoder: dt, decoder: dt },
  heads: Array.from({ length: 40 }, (_, i) => i),
});
console.log(`load ${device}/${dt}: ${(performance.now() - t0).toFixed(0)} ms`);

for (const [task, lang] of [[TOK.TRANSLATE, TOK.EN], [TOK.TRANSCRIBE, TOK.JA]] as const) {
  for (let rep = 0; rep < 2; rep++) {
    const t1 = performance.now();
    const enc = await r.encode(mel);
    const t2 = performance.now();
    const ids: number[] = [];
    let out = await r.decode([TOK.SOT, lang, task, TOK.NO_TIMESTAMPS], enc, undefined, 0);
    const noSpeech = Math.exp(WhisperRunner.logProb(out.extraRow!, TOK.NO_SPEECH));
    while (ids.length < 200) {
      const next = r.argmax(out.logits);
      if (next === TOK.EOT) break;
      ids.push(next);
      out = await r.decode([next], enc, out.cache);
    }
    r.releaseCache(out.cache);
    enc.dispose();
    const t3 = performance.now();
    if (rep === 1)
      console.log(
        `${task === TOK.TRANSLATE ? 'translate' : 'transcribe'}: encoder ${(t2 - t1).toFixed(0)} ms, ` +
          `decoder ${ids.length} tok ${(t3 - t2).toFixed(0)} ms, no_speech=${noSpeech.toFixed(3)}, xattn=${out.xattn.length}\n  ${r.decodeText(ids)}`,
      );
  }
}
