// Node-side setup: local model directory, wasm bytes, audio helpers.
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { env } from '@huggingface/transformers';
import { loadWasm } from '../core/wasm.ts';

export const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
export const MODELS = resolve(ROOT, 'models');

export async function setupNode() {
  env.localModelPath = MODELS + '/';
  env.allowLocalModels = true;
  env.allowRemoteModels = true;
  await loadWasm(readFileSync(resolve(ROOT, 'js/pkg/rt_subs/rt_subs_bg.wasm')));
}

/** 16-bit PCM WAV -> mono Float32 at the file's rate. */
export function readWav(path: string): { rate: number; pcm: Float32Array } {
  const b = readFileSync(path);
  let off = 12;
  let rate = 16000;
  let channels = 1;
  while (off < b.length) {
    const id = b.toString('ascii', off, off + 4);
    const size = b.readUInt32LE(off + 4);
    if (id === 'fmt ') {
      channels = b.readUInt16LE(off + 10);
      rate = b.readUInt32LE(off + 12);
    } else if (id === 'data') {
      const n = Math.min(size, b.length - off - 8) / 2 / channels;
      const pcm = new Float32Array(n);
      for (let i = 0; i < n; i++) pcm[i] = b.readInt16LE(off + 8 + i * 2 * channels) / 32768;
      return { rate, pcm };
    }
    off += 8 + size + (size & 1);
  }
  throw new Error('no data chunk');
}
