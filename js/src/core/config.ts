/** Tunables. Defaults target a single speaker over stream BGM. */
export interface SubtitlerOptions {
  /** HF repo id or local path of the xattn-patched kotoba-whisper-bilingual ONNX export. */
  model: string;
  vadModel: string;
  device: 'webgpu' | 'cuda' | 'cpu' | 'wasm';
  /** Weights per session; the encoder dominates cost and memory. */
  dtype: { encoder: string; decoder: string };
  /**
   * Cross-attention heads used for alignment, as flat ids `layer * 20 + head`
   * over the 2-layer distilled decoder (see scripts/calibrate-heads.ts).
   */
  alignmentHeads: number[];

  /** AlignAtt: stop emitting when attention is this close to the audio end (20 ms frames). */
  frameThreshold: number;
  /** Minimum new audio before another policy step (ms). */
  minStepMs: number;
  /** Speculative continuation shown after the committed text (tokens, 0 = off). */
  tentativeTokens: number;
  /** Previous-output tokens fed back as `<|startofprev|>` context. */
  contextTokens: number;
  /** Fixed prompt text: streamer name, recurring terms, style. */
  staticPrompt: string;
  /** Segments with a no-speech probability above this and no text are dropped. */
  noSpeechThreshold: number;

  vad: {
    threshold: number;
    negThreshold: number;
    minSilenceMs: number;
    minSilenceFloorMs: number;
    softMaxMs: number;
    hardMaxMs: number;
    minSpeechMs: number;
    preRollMs: number;
  };
  highpassHz: number;
}

export const DEFAULTS: SubtitlerOptions = {
  model: 'rt-subs/kotoba-whisper-bilingual-v1.0-xattn',
  vadModel: 'onnx-community/silero-vad',
  device: 'webgpu',
  dtype: { encoder: 'q4f16', decoder: 'q4f16' },
  alignmentHeads: [],
  frameThreshold: 25,
  minStepMs: 300,
  tentativeTokens: 12,
  contextTokens: 64,
  staticPrompt: '',
  noSpeechThreshold: 0.6,
  vad: {
    threshold: 0.5,
    negThreshold: 0.35,
    minSilenceMs: 500,
    minSilenceFloorMs: 100,
    softMaxMs: 10_000,
    hardMaxMs: 24_000,
    minSpeechMs: 250,
    preRollMs: 300,
  },
  highpassHz: 80,
};

export function withDefaults(o: Partial<SubtitlerOptions> = {}): SubtitlerOptions {
  return {
    ...DEFAULTS,
    ...o,
    dtype: { ...DEFAULTS.dtype, ...o.dtype },
    vad: { ...DEFAULTS.vad, ...o.vad },
  };
}
