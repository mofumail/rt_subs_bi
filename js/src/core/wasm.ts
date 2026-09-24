// Loads the Rust DSP core. Callers pass the .wasm as bytes (Node) or a URL (web).
import initWasm, { initSync, AudioPipeline, AlignAttPolicy } from '../../pkg/rt_subs/rt_subs.js';

export { AudioPipeline, AlignAttPolicy };

let ready: Promise<void> | null = null;

export function loadWasm(source: BufferSource | URL | string): Promise<void> {
  if (ready) return ready;
  if (source instanceof URL || typeof source === 'string') {
    ready = initWasm({ module_or_path: source }).then(() => undefined);
  } else {
    initSync({ module: source });
    ready = Promise.resolve();
  }
  return ready;
}

export const EVENTS = { START: 1, END: 2, OVERFLOW: 4, DISCARD: 8 } as const;
