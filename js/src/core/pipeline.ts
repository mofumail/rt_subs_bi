// Orchestration shared by the Node CLI and the browser worker:
//   PCM in -> wasm (resample, filter, frames) -> Silero -> segmenter events
//          -> AlignAtt translator steps -> subtitle line events out.
//
// Everything runs on one async loop, so VAD work between steps naturally
// batches whatever audio arrived while the GPU was busy, and every policy
// step sees all audio available at that moment (compute-aware streaming).

import { type SubtitlerOptions, withDefaults } from './config.ts';
import { SileroVad } from './vad.ts';
import { AudioPipeline, EVENTS } from './wasm.ts';
import { StreamingTranslator, type StepResult } from './translator.ts';
import { WhisperRunner } from './whisper.ts';

export interface LineEvent {
  type: 'line';
  id: number;
  /** Committed text of the line so far (never revised). */
  text: string;
  /** Speculative continuation; replaced on the next update. */
  tentative: string;
  final: boolean;
  start: number;
  end: number;
}

export interface StatsEvent {
  type: 'stats';
  encMs: number;
  decMs: number;
  /** Wall time from hearing a word's audio to showing its translation (ms). */
  tokenLagMs: number[];
  /** Audio waiting to be processed when the step finished (ms). */
  backlogMs: number;
}

export type SubtitleEvent = LineEvent | StatsEvent | { type: 'status'; message: string };

export interface Models {
  whisper: WhisperRunner;
  vad: SileroVad;
}

export async function loadModels(o: SubtitlerOptions, progress?: (p: unknown) => void): Promise<Models> {
  const [whisper, vad] = await Promise.all([
    WhisperRunner.load(o.model, { device: o.device, dtype: o.dtype, heads: o.alignmentHeads, progress }),
    // Silero is tiny and latency-bound: always CPU (native in Node, wasm on the web).
    SileroVad.load(o.vadModel, typeof process !== 'undefined' && process.versions?.node ? 'cpu' : 'wasm'),
  ]);
  return { whisper, vad };
}

export class Subtitler {
  readonly o: SubtitlerOptions;
  private readonly audio: AudioPipeline;
  private readonly vad: SileroVad;
  private readonly tr: StreamingTranslator;
  private readonly onEvent: (e: SubtitleEvent) => void;

  private lineId = 0;
  private lineText = '';
  private ended = false;
  private stopped = false;
  private wake: (() => void) | null = null;
  private loopDone: Promise<void>;
  /** (absolute 16 kHz sample, wall ms) marks for latency accounting. */
  private arrivals: [number, number][] = [];
  private received = 0;
  private readonly inputRate: number;

  constructor(models: Models, inputRate: number, onEvent: (e: SubtitleEvent) => void, opts: Partial<SubtitlerOptions> = {}) {
    this.o = withDefaults(opts);
    const v = this.o.vad;
    this.inputRate = inputRate;
    this.audio = new AudioPipeline(inputRate, this.o.highpassHz, v.preRollMs, 128);
    this.audio.configure_vad(v.threshold, v.negThreshold, v.minSilenceMs, v.minSilenceFloorMs, v.softMaxMs, v.hardMaxMs, v.minSpeechMs);
    this.vad = models.vad;
    this.vad.reset();
    this.tr = new StreamingTranslator(models.whisper, this.audio, this.o);
    this.onEvent = onEvent;
    this.loopDone = this.loop().catch((e) => {
      this.onEvent({ type: 'status', message: `error: ${e?.stack ?? e}` });
      throw e;
    });
  }

  /** Mono PCM at the rate given to the constructor. */
  push(pcm: Float32Array) {
    if (this.ended) return;
    this.audio.push_input(pcm);
    this.received += (pcm.length * 16000) / this.inputRate;
    this.arrivals.push([this.received, performance.now()]);
    if (this.arrivals.length > 4096) this.arrivals.splice(0, 2048);
    this.wake?.();
  }

  /** End of input: finish the current line and resolve when done. */
  async end() {
    this.ended = true;
    this.wake?.();
    await this.loopDone;
  }

  stop() {
    this.stopped = true;
    this.ended = true;
    this.wake?.();
  }

  private arrivalWall(sample16k: number): number | undefined {
    for (const [s, t] of this.arrivals) if (s >= sample16k) return t;
    return undefined;
  }

  private publish(r: StepResult) {
    if (r.dropped) {
      // Hallucination / noise: retract anything already shown for this line.
      this.onEvent({ type: 'line', id: this.lineId, text: '', tentative: '', final: true, start: r.start, end: r.end });
    } else {
      this.lineText += r.committed;
      this.onEvent({
        type: 'line',
        id: this.lineId,
        text: this.lineText.trim(),
        tentative: r.tentative,
        final: r.final,
        start: r.start,
        end: r.end,
      });
    }
    if (r.encMs || r.decMs) {
      const now = performance.now();
      const lags = r.tokenTimes
        .map((t) => this.arrivalWall(t * 16000))
        .filter((w): w is number => w !== undefined)
        .map((w) => now - w);
      const backlog = ((this.audio.pending_samples() + 0) / 16000) * 1000;
      this.onEvent({ type: 'stats', encMs: r.encMs, decMs: r.decMs, tokenLagMs: lags, backlogMs: backlog });
    }
    if (r.final) {
      this.lineId++;
      this.lineText = '';
    }
  }

  private async drainVad() {
    while (this.audio.vad_frame_ready() && !this.stopped) {
      const prob = await this.vad.prob(this.audio.vad_input());
      const ev = this.audio.push_vad_prob(prob);
      if (ev & EVENTS.START) this.tr.begin();
      if (ev & EVENTS.END) {
        if (ev & EVENTS.DISCARD && this.tr.idle) {
          this.tr.abort();
        } else {
          this.publish(await this.tr.step(true));
        }
        this.audio.clear_segment();
      } else if (ev & EVENTS.OVERFLOW) {
        // Bring the committed text up to date, then cut behind it.
        this.publish(await this.tr.step(false));
        this.publish(this.tr.cut());
      }
    }
  }

  private async loop() {
    const minStep = (this.o.minStepMs * 16000) / 1000;
    while (!this.stopped) {
      await this.drainVad();
      if (this.stopped) break;
      if (this.audio.in_speech() && this.tr.newAudio() >= minStep) {
        this.publish(await this.tr.step(false));
        continue;
      }
      if (this.ended && !this.audio.vad_frame_ready()) {
        if (this.audio.in_speech()) {
          this.audio.force_end();
          this.publish(await this.tr.step(true));
          this.audio.clear_segment();
        }
        break;
      }
      await new Promise<void>((res) => {
        this.wake = () => {
          this.wake = null;
          res();
        };
      });
    }
  }
}
