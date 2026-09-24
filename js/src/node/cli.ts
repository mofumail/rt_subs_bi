#!/usr/bin/env node
// Live subtitles in the terminal for a YouTube URL (live or VOD) or a local file.
//
//   node src/node/cli.ts https://www.youtube.com/watch?v=... [--srt out.srt]
//   node src/node/cli.ts ../samples/clip.webm --from 30 --duration 120
//
// Audio: yt-dlp -> ffmpeg (-re: paced at 1x so latency numbers are real) ->
// 48 kHz f32 mono on stdout -> the same pipeline the browser extension runs.

import { spawn } from 'node:child_process';
import { existsSync, writeFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { withDefaults } from '../core/config.ts';
import { loadModels, Subtitler, type LineEvent, type SubtitleEvent } from '../core/pipeline.ts';
import { setupNode } from './env.ts';

const { values: a, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    device: { type: 'string', default: 'webgpu' },
    dtype: { type: 'string', default: 'fp16' },
    'encoder-dtype': { type: 'string' },
    model: { type: 'string', default: 'kotoba-whisper-bilingual-v1.0-xattn' },
    fast: { type: 'boolean', default: false },
    from: { type: 'string' },
    duration: { type: 'string' },
    srt: { type: 'string' },
    jsonl: { type: 'string' },
    prompt: { type: 'string', default: '' },
    'frame-threshold': { type: 'string' },
    tentative: { type: 'string' },
    quiet: { type: 'boolean', default: false },
    offline: { type: 'boolean', default: false },
    help: { type: 'boolean', short: 'h' },
  },
});
const src = positionals[0];
if (!src || a.help) {
  console.error('usage: cli.ts <youtube-url | audio file> [--srt out.srt] [--fast] [--from s] [--duration s] [--prompt "names, terms"]');
  process.exit(src ? 0 : 1);
}

const RATE = 48000;
await setupNode();
const o = withDefaults({
  model: a.model,
  vadModel: 'silero-vad',
  device: a.device as never,
  dtype: { encoder: a['encoder-dtype'] ?? a.dtype!, decoder: a.dtype! },
  alignmentHeads: [34, 25, 23, 29, 28, 39],
  staticPrompt: a.prompt,
  ...(a['frame-threshold'] ? { frameThreshold: +a['frame-threshold'] } : {}),
  ...(a.tentative ? { tentativeTokens: +a.tentative } : {}),
  // Baseline: translate whole utterances only (no simultaneous policy).
  ...(a.offline ? { minStepMs: 1e9, tentativeTokens: 0 } : {}),
});
const t0 = performance.now();
const models = await loadModels(o);
console.error(`models ready in ${((performance.now() - t0) / 1000).toFixed(1)} s (${o.device}, enc ${o.dtype.encoder}, dec ${o.dtype.decoder})`);

// --- audio source --------------------------------------------------------
const ffArgs = ['-hide_banner', '-loglevel', 'error'];
if (!a.fast) ffArgs.push('-re');
if (a.from) ffArgs.push('-ss', a.from);
const isFile = existsSync(src);
ffArgs.push('-i', isFile ? src : 'pipe:0');
if (a.duration) ffArgs.push('-t', a.duration);
ffArgs.push('-vn', '-ac', '1', '-ar', String(RATE), '-f', 'f32le', 'pipe:1');
const ff = spawn('ffmpeg', ffArgs, { stdio: [isFile ? 'ignore' : 'pipe', 'pipe', 'inherit'] });
if (!isFile) {
  const yt = spawn('yt-dlp', ['-q', '--no-warnings', '-f', 'bestaudio/best', '-o', '-', src], { stdio: ['ignore', 'pipe', 'inherit'] });
  yt.stdout.pipe(ff.stdin!);
  ff.on('close', () => yt.kill());
}

// --- output ---------------------------------------------------------------
const tty = process.stdout.isTTY && !a.quiet;
const finals: LineEvent[] = [];
const lags: number[] = [];
const enc: number[] = [];
const dec: number[] = [];
let live = '';
const fmt = (s: number) => {
  const ms = Math.max(0, Math.round(s * 1000));
  const p = (n: number, w = 2) => String(n).padStart(w, '0');
  return `${p(Math.floor(ms / 3600000))}:${p(Math.floor(ms / 60000) % 60)}:${p(Math.floor(ms / 1000) % 60)},${p(ms % 1000, 3)}`;
};
const jsonl: string[] = [];

function onEvent(e: SubtitleEvent) {
  if (a.jsonl) jsonl.push(JSON.stringify({ t: performance.now(), ...e }));
  if (e.type === 'status') return console.error(e.message);
  if (e.type === 'stats') {
    lags.push(...e.tokenLagMs);
    enc.push(e.encMs);
    dec.push(e.decMs);
    return;
  }
  if (e.final) {
    if (e.text) finals.push(e);
    if (tty) process.stdout.write('\r\x1b[2K');
    if (e.text) process.stdout.write(`[${fmt(e.start).slice(3, 8)}] ${e.text}\n`);
    live = '';
  } else if (tty) {
    live = `\r\x1b[2K\x1b[1m${e.text}\x1b[0m\x1b[2m${e.tentative}\x1b[0m`;
    const cols = process.stdout.columns || 120;
    const plain = e.text + e.tentative;
    process.stdout.write(plain.length < cols - 2 ? live : `\r\x1b[2K…\x1b[1m${e.text.slice(-(cols - 20))}\x1b[0m\x1b[2m${e.tentative.slice(0, 16)}\x1b[0m`);
  }
}

const sub = new Subtitler(models, RATE, onEvent, o);
let carry: Buffer = Buffer.alloc(0);
ff.stdout!.on('data', (chunk: Buffer) => {
  const buf = carry.length ? Buffer.concat([carry, chunk]) : chunk;
  const n = Math.floor(buf.length / 4);
  const pcm = new Float32Array(n);
  for (let i = 0; i < n; i++) pcm[i] = buf.readFloatLE(i * 4);
  carry = buf.subarray(n * 4);
  sub.push(pcm);
});
process.on('SIGINT', () => ff.kill('SIGINT'));
await new Promise((res) => ff.on('close', res));
await sub.end();

// --- summary --------------------------------------------------------------
const pct = (v: number[], p: number) => {
  if (!v.length) return NaN;
  const s = [...v].sort((x, y) => x - y);
  return s[Math.min(s.length - 1, Math.floor(p * s.length))];
};
const mean = (v: number[]) => v.reduce((x, y) => x + y, 0) / Math.max(1, v.length);
console.error(
  `\n${finals.length} lines | steps ${enc.length} | encoder ${mean(enc).toFixed(0)} ms, decoder ${mean(dec).toFixed(0)} ms per step` +
    `\ntoken lag (audio heard -> text shown): p50 ${(pct(lags, 0.5) / 1000).toFixed(2)} s, p90 ${(pct(lags, 0.9) / 1000).toFixed(2)} s, mean ${(mean(lags) / 1000).toFixed(2)} s (n=${lags.length})`,
);
if (a.srt) {
  writeFileSync(a.srt, finals.map((l, i) => `${i + 1}\n${fmt(l.start)} --> ${fmt(l.end)}\n${l.text}\n`).join('\n'));
  console.error(`wrote ${a.srt}`);
}
if (a.jsonl) writeFileSync(a.jsonl, jsonl.join('\n') + '\n');
process.exit(0);
