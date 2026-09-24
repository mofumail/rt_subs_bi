// Simultaneous JP->EN translation over a growing audio segment (AlignAtt).
//
// Each step re-encodes the segment, force-feeds the tokens already committed,
// then decodes greedily. A new token is committed only while its
// cross-attention points at audio we have fully heard; once it reaches the
// last `frameThreshold` frames we stop and wait for more speech. Committed
// text is never revised, so subtitles grow without flicker. The decoder's
// continuation past that point is returned separately as a tentative tail
// (the "prediction" of what is being said), which the UI can show dimmed.

import type { SubtitlerOptions } from './config.ts';
import { AlignAttPolicy, type AudioPipeline } from './wasm.ts';
import { type DecodeOutput, TOK, type OrtTensor, WhisperRunner } from './whisper.ts';
import { isHallucination, hasLoop } from './text.ts';

const SOT_SEQ = [TOK.SOT, TOK.EN, TOK.TRANSLATE, TOK.NO_TIMESTAMPS];
const ENC_FRAMES = 1500;
const SAMPLES_PER_FRAME = 320;

export interface StepResult {
  /** Newly committed text for the current line (appended to earlier text). */
  committed: string;
  tentative: string;
  /** The line is complete (segment end or forced cut). */
  final: boolean;
  /** Line start in absolute seconds of input audio. */
  start: number;
  end: number;
  /** Encoder / decoder time for this step, ms. */
  encMs: number;
  decMs: number;
  /** Absolute input-audio time (s) each committed token was aligned to. */
  tokenTimes: number[];
  dropped?: boolean;
}

export class StreamingTranslator {
  private readonly r: WhisperRunner;
  private readonly audio: AudioPipeline;
  private readonly o: SubtitlerOptions;
  private readonly policy: AlignAttPolicy;
  private readonly mel: Float32Array;
  private readonly staticIds: number[];

  private context: number[] = [];
  private committed: number[] = [];
  /** Segment-relative encoder frame each committed token was aligned to. */
  private frames: number[] = [];
  private lineText = '';
  private lineStart = 0;
  private lastStepSamples = 0;

  constructor(r: WhisperRunner, audio: AudioPipeline, o: SubtitlerOptions) {
    this.r = r;
    this.audio = audio;
    this.o = o;
    this.policy = new AlignAttPolicy(r.nHeads, ENC_FRAMES, 7);
    this.mel = new Float32Array(audio.mel_size());
    this.staticIds = o.staticPrompt ? r.encodeText(o.staticPrompt) : [];
  }

  get idle(): boolean {
    return this.committed.length === 0;
  }

  /** Audio added since the last step, in samples. */
  newAudio(): number {
    return this.audio.segment_len() - this.lastStepSamples;
  }

  begin() {
    this.committed = [];
    this.frames = [];
    this.lineText = '';
    this.lineStart = this.audio.segment_start() / 16000;
    this.lastStepSamples = 0;
  }

  private prompt(): { ids: number[]; sot: number } {
    const prev = [...this.staticIds, ...this.context.slice(-this.o.contextTokens)];
    const head = prev.length ? [TOK.PREV, ...prev] : [];
    return { ids: [...head, ...SOT_SEQ, ...this.committed], sot: head.length };
  }

  /**
   * One policy step. `final` decodes to end-of-text without the frontier
   * check (segment ended); otherwise emission stops at the frontier.
   */
  async step(final: boolean): Promise<StepResult> {
    const segLen = this.audio.segment_len();
    this.lastStepSamples = segLen;
    const content = this.audio.mel(this.mel);
    const t0 = performance.now();
    const enc = await this.r.encode(this.mel);
    const t1 = performance.now();

    const { ids, sot } = this.prompt();
    const maxTokens = Math.min(440 - ids.length, 12 + Math.ceil((segLen / 16000) * 9));
    this.policy.reset();
    let out: DecodeOutput = await this.r.decode(ids, enc, undefined, sot);
    this.policy.push(out.xattn, out.rows, sot);
    const noSpeech = Math.exp(WhisperRunner.logProb(out.extraRow!, TOK.NO_SPEECH));

    const fresh: number[] = [];
    const freshFrames: number[] = [];
    let tentative: number[] = [];
    let logprob = 0;
    let stoppedAtFrontier = false;
    const limit = Math.max(0, content - this.o.frameThreshold);
    // Until the line has text, trust Whisper's own no-speech estimate and hold.
    const hold = !final && this.committed.length === 0 && noSpeech > this.o.noSpeechThreshold;

    try {
      while (this.committed.length + fresh.length < maxTokens) {
        const next = this.r.argmax(out.logits);
        if (next === TOK.EOT) break;
        const frame = this.policy.attended_frame(content, ENC_FRAMES);
        if (hold || (!final && frame >= limit)) {
          stoppedAtFrontier = true;
          tentative = [next];
          break;
        }
        logprob += WhisperRunner.logProb(out.logits, next);
        fresh.push(next);
        freshFrames.push(frame);
        if (hasLoop([...this.committed, ...fresh])) {
          // Degenerate repetition: drop the loop and end the line here.
          fresh.length = Math.max(0, fresh.length - 8);
          freshFrames.length = fresh.length;
          break;
        }
        out = await this.decodeNext(next, enc, out);
      }

      // Speculative tail beyond the frontier (display only).
      if (stoppedAtFrontier && this.o.tentativeTokens > 0 && !hold) {
        out = await this.decodeNext(tentative[0], enc, out);
        while (tentative.length < this.o.tentativeTokens) {
          const next = this.r.argmax(out.logits);
          if (next === TOK.EOT) break;
          tentative.push(next);
          out = await this.decodeNext(next, enc, out);
        }
      } else {
        tentative = [];
      }
    } finally {
      this.r.releaseCache(out.cache);
      enc.dispose();
    }
    const t2 = performance.now();

    // Nothing but noise: Silero fired but Whisper hears no speech.
    if (final && this.committed.length === 0 && fresh.length === 0 && noSpeech > this.o.noSpeechThreshold) {
      return this.emit('', '', true, t1 - t0, t2 - t1, [], true);
    }

    const before = this.lineText;
    this.committed.push(...fresh);
    this.frames.push(...freshFrames);
    this.lineText = this.r.decodeText(this.committed);
    let tentText = '';
    if (tentative.length) {
      const full = this.r.decodeText([...this.committed, ...tentative]);
      tentText = full.startsWith(this.lineText) ? full.slice(this.lineText.length) : '';
    }
    const segStart = this.audio.segment_start();
    const tokenTimes = freshFrames.map((f) => (segStart + f * SAMPLES_PER_FRAME) / 16000);

    if (final) {
      const avgLogprob = this.committed.length ? logprob / Math.max(1, fresh.length) : 0;
      if (isHallucination(this.lineText, avgLogprob, noSpeech, segLen / 16000)) {
        return this.emit('', '', true, t1 - t0, t2 - t1, [], true);
      }
      return this.emit(this.lineText.slice(before.length), '', true, t1 - t0, t2 - t1, tokenTimes);
    }
    return this.emit(this.lineText.slice(before.length), tentText, false, t1 - t0, t2 - t1, tokenTimes);
  }

  private async decodeNext(id: number, enc: OrtTensor, prev: DecodeOutput): Promise<DecodeOutput> {
    const out = await this.r.decode([id], enc, prev.cache);
    this.policy.push(out.xattn, 1, 0);
    return out;
  }

  private emit(
    committed: string,
    tentative: string,
    final: boolean,
    encMs: number,
    decMs: number,
    tokenTimes: number[],
    dropped = false,
  ): StepResult {
    const end = (this.audio.segment_start() + this.audio.segment_len()) / 16000;
    const res: StepResult = { committed, tentative, final, start: this.lineStart, end, encMs, decMs, tokenTimes, dropped };
    if (final) {
      if (!dropped) this.context.push(...this.committed);
      this.context = this.context.slice(-this.o.contextTokens);
      this.committed = [];
      this.frames = [];
      this.lineText = '';
    }
    return res;
  }

  /**
   * The segment hit the length limit while speech continues. Close the
   * current line at the audio its committed tokens have consumed and keep the
   * rest as the start of the next line.
   */
  cut(): StepResult {
    const segLen = this.audio.segment_len();
    let cutSample: number;
    if (this.frames.length) {
      // Latest audio the committed text is aligned to (a couple of tokens of slack).
      const recent = this.frames.slice(-3);
      cutSample = Math.max(...recent) * SAMPLES_PER_FRAME;
    } else {
      cutSample = Math.floor(segLen / 2);
    }
    cutSample = Math.min(cutSample, segLen - 16000);
    cutSample = this.audio.quiet_point(Math.max(0, cutSample), 4800);
    const text = this.lineText;
    const res: StepResult = {
      committed: '',
      tentative: '',
      final: true,
      start: this.lineStart,
      end: (this.audio.segment_start() + cutSample) / 16000,
      encMs: 0,
      decMs: 0,
      tokenTimes: [],
      dropped: text.length === 0,
    };
    if (!res.dropped) res.committed = '';
    this.context.push(...this.committed);
    this.context = this.context.slice(-this.o.contextTokens);
    this.audio.trim_front(cutSample);
    this.begin();
    return res;
  }

  /** Forget the running line without emitting (e.g. VAD false alarm). */
  abort() {
    this.begin();
  }
}
