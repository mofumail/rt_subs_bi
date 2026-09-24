# rt_subs_bi

Real-time Japanese → English subtitles for YouTube streams, on-device.

```
YouTube <video> ─► Web Audio tap ─► [Rust/Wasm] resample 16 kHz · high-pass · 32 ms frames
                                              │
                         Silero VAD (2 MB) ◄──┘  speech gate + utterance segmentation
                                              │
                   [Rust/Wasm] incremental Whisper log-mel (128 × 3000)
                                              │
        kotoba-whisper-bilingual encoder ─► 2-layer decoder (JA speech → EN text directly)
                                              │ cross-attention of 6 alignment heads
                      [Rust/Wasm] AlignAtt policy: emit token / wait for more audio
                                              │
                     overlay: committed text (stable) + predicted continuation (dimmed)
```

## Approach

**Policy: AlignAtt on Whisper's cross-attention** (Papi et al. 2023, used by
SimulStreaming, the IWSLT 2025 simultaneous-track winner). The segment
buffer is re-encoded as audio arrives. The decoder then emits tokens greedily
and stops as soon as the next token's cross-attention lands in the last
`frameThreshold` × 20 ms of received audio. At that point the model is
"reading the frontier" and needs more speech. Emitted text is never revised.

**Prediction.** Past the stop point the decoder keeps going for up to 12
tokens. Those tokens are its guess at what comes next, shown dimmed and
replaced on every update. This gives you the "predict the sentence" effect
without a second model. I didn't use a separate LLM predictor (profile/buffer
+ LLM) because it would add a second multi-GB model, and Japanese is
verb-final: the part that decides the English sentence ("…ない", "…たい",
"…と思う") arrives last. A wrong prediction that gets committed can't be
taken back, while a dimmed guess costs nothing.

**Silero VAD** gates the model and does not remove music. It keeps music-only
stretches away from Whisper, which is where "Thank you for watching"
hallucinations come from, and its pauses give clean utterance boundaries.
Speech mixed with BGM goes to Whisper as is. The high-pass filter only
removes bass/kick below 80 Hz. Whisper handles BGM better than it handles
denoiser artifacts. Long monologues without pauses are handled in two ways:
the silence needed to end a segment shrinks from 500 ms to 100 ms after
10 s, and at 24 s the segment is cut behind the audio that the committed
text is aligned to.

**The model** is `kotoba-tech/kotoba-whisper-bilingual-v1.0`: a
distil-large-v3 encoder (32 layers) with a 2-layer decoder, trained for
JA speech → EN text. The published ONNX export does not expose
cross-attention, so `scripts/patch_xattn.py` adds a `cross_attentions`
output (plus an `xattn_heads` input that selects the heads) to the merged
decoder. It is a graph edit, not a re-export, and works on every quantized
variant. The generation config's `alignment_heads` belong to large-v3's
32-layer decoder and don't exist here. `scripts/calibrate-heads.ts` ranked
the 40 real heads: layer-1 heads 14, 5, 3, 9, 8 and 19 track the audio
(Spearman ρ between 0.7 and 0.98). Layer 0 is positional.

The CTranslate2 `-faster` variant can't run in a browser, so the ONNX export
is used instead.

### Split of work

| Rust → Wasm (`rt_subs/`, 330 KB) | TypeScript (`js/src/core/`) |
|---|---|
| polyphase resampler (any rate → 16 kHz) | model loading / caching (transformers.js) |
| biquad high-pass | Silero + encoder + decoder calls (ORT via transformers.js) |
| incremental Whisper log-mel (matches transformers.js to 1.6e-5) | greedy decode loop, KV cache |
| VAD segmenter (hysteresis, adaptive silence, pre-roll) | line events, hallucination filter |
| AlignAtt scoring (z-score, median filter, head mean) | Node CLI, local engine, Firefox extension |

## Measured (RTX 4070 Ti, Node + WebGPU/Dawn, 3 min VTuber chat with BGM)

"Lag" is wall time from a word's audio arriving to its translation appearing
as committed text.

| mode | encoder / step | median lag | p90 lag |
|---|---|---|---|
| offline (translate each whole utterance) | — | 1.89 s | 3.38 s |
| AlignAtt, fp16 | 154 ms | 0.99 s | 1.94 s |
| AlignAtt, q4f16 (≈530 MB) | 98 ms | 1.32 s | 2.39 s |

Latency ↔ quality via the frame threshold (fp16). chrF measures agreement
with the offline translation of the same audio, i.e. how much quality the
simultaneous policy gives up:

| frame threshold | median lag | p90 lag | chrF vs offline |
|---|---|---|---|
| 10 (200 ms) | 0.69 s | 1.92 s | 52.0 |
| **25 (500 ms, default)** | **0.99 s** | **1.93 s** | **70.8** |
| 40 (800 ms) | 1.31 s | 2.21 s | 83.0 |

## Use

### 1. Build and fetch the model

```sh
cd js && npm install
npm run wasm                       # Rust → js/pkg/rt_subs
python -m venv .venv && .venv/bin/pip install onnx
.venv/bin/python ../scripts/patch_xattn.py --dtypes q4f16 fp16 q4 --out ../models/kotoba-whisper-bilingual-v1.0-xattn
mkdir -p ../models/silero-vad/onnx && curl -L -o ../models/silero-vad/onnx/model.onnx \
  https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx
```

### 2a. Terminal

```sh
node src/node/cli.ts 'https://www.youtube.com/watch?v=…'          # live or VOD, paced at 1×
node src/node/cli.ts clip.webm --srt out.srt --prompt "Pekora, Hololive"
node src/node/cli.ts clip.webm --offline --srt ref.srt              # whole-utterance baseline
```

### 2b. Firefox

```sh
npm run build:ext                  # → js/dist/firefox
npm run engine                     # local engine + model server on 127.0.0.1:8765
npx web-ext run -s dist/firefox    # or about:debugging → Load Temporary Add-on → manifest.json
```

Open a YouTube video or stream, click the toolbar button, and press
**Start subtitles**.

- **Engine: Auto** runs in the browser on WebGPU when Firefox exposes it,
  and otherwise streams audio to `npm run engine`. On Linux you may need
  `dom.webgpu.enabled` in `about:config`. Without WebGPU the 640M-parameter
  encoder is too slow on CPU Wasm for live use.
- Adapters without `shader-f16` load the `q4` model instead of `q4f16`.
- To run fully in-browser with no local process, upload the patched model
  directory to your own HF repo:
  `huggingface-cli upload <you>/kotoba-whisper-bilingual-v1.0-xattn models/kotoba-whisper-bilingual-v1.0-xattn`.
  Then set **Model base URL** to `https://huggingface.co/` and **Model id**
  to that repo.
- **Context prompt** is fed to the decoder as previous text. Streamer names
  and recurring terms there fix most name mistakes.

## Limits

- One speaker at a time. Overlapping collab chatter will merge.
- Tapping the `<video>` element means muting YouTube also mutes the tap.
- The encoder always processes a 30 s window. That fixed cost per step sets
  the floor on how often the policy can update. On weaker GPUs, steps get
  larger by themselves and lag grows. Pick **Small** and raise the frame
  threshold.
- Translation quality is the model's: loose, sometimes wrong on slang and
  game terms. Streaming costs little on top of that (see the chrF column).

## Layout

```
rt_subs/               Rust DSP core (cargo test; wasm-pack → js/pkg)
scripts/patch_xattn.py add cross-attention outputs to the ONNX decoder
js/src/core/           pipeline shared by Node and the extension
js/src/node/           cli.ts, server.ts (local engine), env.ts
js/src/extension/      Firefox MV2: background (engine), content (tap + overlay), popup
js/scripts/            calibrate-heads, smoke (mel parity), vad-segments, chrf, build-extension
```
