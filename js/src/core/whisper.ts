// Thin, allocation-aware wrapper over the kotoba-whisper-bilingual ONNX sessions.
//
// transformers.js handles download, caching, device/dtype selection and the
// tokenizer; the decode loop is ours because AlignAtt needs to inspect each
// step's cross-attention and stop mid-sequence, which `generate()` cannot do.

import { AutoTokenizer, Tensor, WhisperForConditionalGeneration } from '@huggingface/transformers';
import { tensorF32, toHalf } from './fp16.ts';

// Minimal structural view of onnxruntime's InferenceSession / Tensor.
export interface OrtTensor {
  type: string;
  dims: readonly number[];
  data: unknown;
  location?: string;
  getData?(release?: boolean): Promise<unknown>;
  dispose(): void;
}
interface OrtSession {
  inputNames: readonly string[];
  outputNames: readonly string[];
  inputMetadata: readonly { name: string; type?: string; shape?: readonly (number | string)[] }[];
  run(feeds: Record<string, OrtTensor>): Promise<Record<string, OrtTensor>>;
}

export const TOK = {
  EOT: 50257,
  SOT: 50258,
  EN: 50259,
  JA: 50266,
  TRANSLATE: 50359,
  TRANSCRIBE: 50360,
  PREV: 50362,
  NO_SPEECH: 50363,
  NO_TIMESTAMPS: 50364,
  TIMESTAMP_BEGIN: 50365,
} as const;

export interface DecoderCache {
  kv: Record<string, OrtTensor>;
}

export interface DecodeOutput {
  /** Logits of the last input position. */
  logits: Float32Array;
  /** Logits of an extra requested row (e.g. the SOT position for no-speech). */
  extraRow?: Float32Array;
  /** `[heads][rows][1500]` cross-attention for the alignment heads. */
  xattn: Float32Array;
  rows: number;
  cache: DecoderCache;
}

function ort(t: Tensor): OrtTensor {
  return (t as unknown as { ort_tensor: OrtTensor }).ort_tensor;
}

async function cpuData(t: OrtTensor): Promise<unknown> {
  return t.location && t.location !== 'cpu' && t.getData ? t.getData(true) : t.data;
}

export interface LoadOptions {
  device: string;
  dtype: { encoder: string; decoder: string };
  heads: number[];
  progress?: (p: unknown) => void;
}

export class WhisperRunner {
  readonly vocab: number;
  readonly nHeads: number;
  private readonly heads: OrtTensor;
  private readonly kvNames: string[];
  private readonly kvMeta: Map<string, { type: string; shape: number[] }>;
  private readonly encInputType: string;
  /** Suppressed during free decoding: special tokens other than EOT, plus the generation config list. */
  private readonly suppress: Int32Array;

  readonly model: unknown;
  readonly tokenizer: Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>;
  private readonly enc: OrtSession;
  private readonly dec: OrtSession;

  private constructor(
    model: unknown,
    tokenizer: Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>,
    enc: OrtSession,
    dec: OrtSession,
    heads: number[],
    suppressTokens: number[],
  ) {
    this.model = model;
    this.tokenizer = tokenizer;
    this.enc = enc;
    this.dec = dec;
    if (!dec.inputNames.includes('xattn_heads')) {
      throw new Error('decoder has no xattn_heads input: run scripts/patch_xattn.py on the model first');
    }
    const cfg = (model as { config: { vocab_size: number } }).config;
    this.vocab = cfg.vocab_size;
    if (!heads.length) throw new Error('no alignment heads configured');
    this.nHeads = heads.length;
    this.heads = ort(new Tensor('int64', BigInt64Array.from(heads.map(BigInt)), [heads.length]));
    this.kvNames = dec.inputNames.filter((n) => n.startsWith('past_key_values.'));
    this.kvMeta = new Map();
    for (const m of dec.inputMetadata) {
      if (!m.name.startsWith('past_key_values.')) continue;
      const shape = (m.shape ?? []).map((d) => (typeof d === 'number' ? d : 0));
      shape[0] = 1;
      shape[1] = 20;
      shape[3] = 64;
      this.kvMeta.set(m.name, { type: m.type ?? 'float32', shape });
    }
    this.encInputType = enc.inputMetadata.find((m) => m.name === 'input_features')?.type ?? 'float32';
    const sup = new Set<number>(suppressTokens);
    for (let t = TOK.EOT + 1; t < this.vocab; t++) sup.add(t);
    this.suppress = Int32Array.from(sup);
  }

  static async load(path: string, o: LoadOptions): Promise<WhisperRunner> {
    const model = await WhisperForConditionalGeneration.from_pretrained(path, {
      device: o.device as never,
      dtype: { encoder_model: o.dtype.encoder, decoder_model_merged: o.dtype.decoder } as never,
      progress_callback: o.progress as never,
    });
    const tokenizer = await AutoTokenizer.from_pretrained(path);
    const sessions = (model as unknown as { sessions: Record<string, OrtSession> }).sessions;
    const gen = (model as unknown as { generation_config?: { suppress_tokens?: number[] } }).generation_config;
    return new WhisperRunner(model, tokenizer, sessions.model, sessions.decoder_model_merged, o.heads, gen?.suppress_tokens ?? []);
  }

  /** `mel` is the `128 x 3000` feature matrix; returns the encoder output tensor. */
  async encode(mel: Float32Array): Promise<OrtTensor> {
    const input =
      this.encInputType === 'float16'
        ? ort(new Tensor('float16', toHalf(mel) as never, [1, 128, 3000]))
        : ort(new Tensor('float32', mel, [1, 128, 3000]));
    const out = await this.enc.run({ input_features: input });
    input.dispose();
    const hs = out.last_hidden_state;
    for (const [k, v] of Object.entries(out)) if (k !== 'last_hidden_state') v.dispose();
    return hs;
  }

  /**
   * One decoder call. Without `cache`, `ids` is the whole prompt (no-past
   * branch); with it, `ids` are new tokens appended to the cached sequence.
   * The input cache tensors are released.
   */
  async decode(ids: number[], encoderOut: OrtTensor, cache?: DecoderCache, extraRow = -1): Promise<DecodeOutput> {
    const feeds: Record<string, OrtTensor> = {
      input_ids: ort(new Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length])),
      encoder_hidden_states: encoderOut,
      use_cache_branch: ort(new Tensor('bool', [!!cache] as never, [1])),
      xattn_heads: this.heads,
    };
    const empties: OrtTensor[] = [];
    for (const name of this.kvNames) {
      if (cache) {
        feeds[name] = cache.kv[name];
      } else {
        const m = this.kvMeta.get(name)!;
        const t = ort(new Tensor(m.type as never, (m.type === 'float16' ? new Uint16Array(0) : new Float32Array(0)) as never, m.shape));
        feeds[name] = t;
        empties.push(t);
      }
    }
    const out = await this.dec.run(feeds);
    feeds.input_ids.dispose();
    feeds.use_cache_branch.dispose();
    for (const t of empties) t.dispose();

    const kv: Record<string, OrtTensor> = {};
    for (const name of this.kvNames) {
      const present = name.replace('past_key_values', 'present');
      if (cache && name.includes('.encoder.')) {
        // The with-past branch does not recompute cross-attention K/V.
        kv[name] = cache.kv[name];
        out[present].dispose();
      } else {
        kv[name] = out[present];
        if (cache) cache.kv[name].dispose();
      }
    }

    const logitsT = out.logits;
    const rows = ids.length;
    const V = this.vocab;
    const ldata = await cpuData(logitsT);
    const logits = tensorF32(ldata, logitsT.type, (rows - 1) * V, rows * V).slice();
    const extra = extraRow >= 0 ? tensorF32(ldata, logitsT.type, extraRow * V, (extraRow + 1) * V).slice() : undefined;
    logitsT.dispose();

    const xT = out.cross_attentions;
    const xattn = tensorF32(await cpuData(xT), xT.type).slice();
    xT.dispose();
    return { logits, extraRow: extra, xattn, rows, cache: { kv } };
  }

  /** Greedy pick with special tokens suppressed (EOT allowed unless `noEot`). */
  argmax(logits: Float32Array, noEot = false): number {
    const l = logits;
    for (let i = 0; i < this.suppress.length; i++) l[this.suppress[i]] = -Infinity;
    if (noEot) l[TOK.EOT] = -Infinity;
    let best = 0;
    for (let i = 1; i < l.length; i++) if (l[i] > l[best]) best = i;
    return best;
  }

  static logProb(logits: Float32Array, id: number): number {
    let max = -Infinity;
    for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
    let sum = 0;
    for (let i = 0; i < logits.length; i++) if (logits[i] !== -Infinity) sum += Math.exp(logits[i] - max);
    return logits[id] - max - Math.log(sum);
  }

  releaseCache(c?: DecoderCache) {
    if (!c) return;
    for (const t of Object.values(c.kv)) t.dispose();
  }

  decodeText(ids: number[]): string {
    if (!ids.length) return '';
    return this.tokenizer.decode(ids, { skip_special_tokens: true });
  }

  encodeText(text: string): number[] {
    return this.tokenizer.encode(' ' + text.trim(), { add_special_tokens: false });
  }
}
