// Background page: owns the engine. The content script streams PCM here over a
// runtime port and gets subtitle events back.
//
// Engines:
//   browser - transformers.js + wasm core on WebGPU, right here in the page.
//   local   - WebSocket to `node src/node/server.ts` (same pipeline natively),
//             for browsers without WebGPU (Firefox on Linux, for now).

import { env } from '@huggingface/transformers';
import { withDefaults } from '../core/config.ts';
import { loadModels, type Models, Subtitler, type SubtitleEvent } from '../core/pipeline.ts';
import { loadWasm } from '../core/wasm.ts';
import { loadSettings, type Settings, type ToBackground } from './settings.ts';

const HEADS = [34, 25, 23, 29, 28, 39];

env.allowLocalModels = false;
const onnx = env.backends.onnx as { wasm?: { wasmPaths?: unknown } };
if (onnx.wasm) {
  // MV2/MV3 forbid remote code: ORT's runtime ships inside the extension.
  onnx.wasm.wasmPaths = {
    mjs: browser.runtime.getURL('ort/ort-wasm-simd-threaded.asyncify.mjs'),
    wasm: browser.runtime.getURL('ort/ort-wasm-simd-threaded.asyncify.wasm'),
  };
}

interface Session {
  push(pcm: Float32Array): void;
  stop(): void;
}

type Emit = (e: SubtitleEvent) => void;

async function webgpu(): Promise<{ f16: boolean } | null> {
  const gpu = (navigator as unknown as { gpu?: { requestAdapter(o?: object): Promise<{ features: Set<string> } | null> } }).gpu;
  if (!gpu) return null;
  try {
    const ad = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    return ad ? { f16: ad.features.has('shader-f16') } : null;
  } catch {
    return null;
  }
}

// --- in-browser engine -------------------------------------------------------
let modelsKey = '';
let models: Promise<Models> | null = null;

function getModels(s: Settings, f16: boolean, emit: Emit): Promise<Models> {
  const dtype = !f16 ? { encoder: 'q4', decoder: 'q4' } : s.quality === 'quality' ? { encoder: 'fp16', decoder: 'fp16' } : { encoder: 'q4f16', decoder: 'q4f16' };
  const key = JSON.stringify([s.modelBase, s.modelId, dtype]);
  if (models && key === modelsKey) return models;
  modelsKey = key;
  const onHf = /huggingface\.co/.test(s.modelBase);
  env.remoteHost = s.modelBase;
  env.remotePathTemplate = onHf ? '{model}/resolve/{revision}/' : '{model}/';
  const files = new Map<string, [number, number]>();
  let last = 0;
  const progress = (p: unknown) => {
    const q = p as { status: string; file?: string; loaded?: number; total?: number };
    if (q.status !== 'progress' || !q.file) return;
    files.set(q.file, [q.loaded ?? 0, q.total ?? 0]);
    const [l, t] = [...files.values()].reduce((a, b) => [a[0] + b[0], a[1] + b[1]], [0, 0]);
    const now = Date.now();
    if (now - last > 500 && t) {
      last = now;
      emit({ type: 'status', message: `downloading model ${((100 * l) / t).toFixed(0)}% of ${(t / 1e6).toFixed(0)} MB` });
    }
  };
  const o = withDefaults({
    model: s.modelId,
    vadModel: onHf ? 'onnx-community/silero-vad' : 'silero-vad',
    device: 'webgpu',
    dtype,
    alignmentHeads: HEADS,
  });
  models = (async () => {
    await loadWasm(browser.runtime.getURL('rt_subs_bg.wasm'));
    return loadModels(o, progress);
  })();
  models.catch(() => {
    models = null;
    modelsKey = '';
  });
  return models;
}

async function browserSession(rate: number, s: Settings, f16: boolean, emit: Emit): Promise<Session> {
  const m = await getModels(s, f16, emit);
  emit({ type: 'status', message: 'compiling shaders…' });
  const sub = new Subtitler(m, rate, emit, {
    alignmentHeads: HEADS,
    staticPrompt: s.prompt,
    frameThreshold: s.frameThreshold,
    tentativeTokens: s.showTentative ? 12 : 0,
  });
  return { push: (pcm) => sub.push(pcm), stop: () => sub.stop() };
}

// --- local engine ------------------------------------------------------------
function localSession(rate: number, s: Settings, emit: Emit): Promise<Session> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(s.localUrl);
    ws.binaryType = 'arraybuffer';
    let open = false;
    ws.onopen = () => {
      open = true;
      ws.send(
        JSON.stringify({
          type: 'hello',
          rate,
          options: { staticPrompt: s.prompt, frameThreshold: s.frameThreshold, tentativeTokens: s.showTentative ? 12 : 0 },
        }),
      );
      resolve({
        push: (pcm) => {
          // Drop audio rather than queue it if the engine falls behind.
          if (ws.readyState === WebSocket.OPEN && ws.bufferedAmount < 1 << 20) ws.send(pcm.buffer.slice(pcm.byteOffset, pcm.byteOffset + pcm.byteLength));
        },
        stop: () => ws.close(),
      });
    };
    ws.onmessage = (m) => emit(JSON.parse(m.data as string));
    ws.onerror = () => {
      if (!open) reject(new Error(`local engine not reachable at ${s.localUrl} (run: npm run engine)`));
    };
    ws.onclose = () => {
      if (open) emit({ type: 'status', message: 'local engine disconnected' });
    };
  });
}

async function startSession(rate: number, emit: Emit): Promise<Session> {
  const s = await loadSettings();
  if (s.engine !== 'local') {
    const gpu = await webgpu();
    if (gpu) {
      emit({ type: 'status', message: `loading model (WebGPU${gpu.f16 ? ', fp16' : ', no shader-f16 → q4'})…` });
      return browserSession(rate, s, gpu.f16, emit);
    }
    if (s.engine === 'browser') throw new Error('WebGPU is not available in this browser (Firefox: about:config → dom.webgpu.enabled)');
    emit({ type: 'status', message: 'no WebGPU here, using local engine…' });
  }
  return localSession(rate, s, emit);
}

// --- ports -------------------------------------------------------------------
let active: { port: browser.runtime.Port; session: Session | null; tabId?: number } | null = null;
const lastStats = new Map<number, { lag: number[]; enc: number }>();

function stopActive() {
  if (!active) return;
  active.session?.stop();
  active = null;
}

browser.runtime.onConnect.addListener((port) => {
  if (port.name !== 'rt-subs') return;
  const tabId = port.sender?.tab?.id;
  const emit: Emit = (e) => {
    try {
      port.postMessage(e);
    } catch {
      /* tab gone */
    }
    if (e.type === 'stats' && tabId !== undefined) {
      const st = lastStats.get(tabId) ?? { lag: [], enc: 0 };
      st.lag = [...st.lag, ...e.tokenLagMs].slice(-200);
      st.enc = e.encMs;
      lastStats.set(tabId, st);
    }
  };
  port.onMessage.addListener(async (raw) => {
    const msg = raw as ToBackground;
    if (msg.type === 'start') {
      if (active && active.port !== port) {
        active.port.postMessage({ type: 'status', message: 'stopped: subtitles started in another tab' });
        stopActive();
      }
      const me = { port, session: null as Session | null, tabId };
      active = me;
      try {
        const session = await startSession(msg.rate, emit);
        if (active !== me) return session.stop();
        me.session = session;
        emit({ type: 'status', message: 'listening' });
      } catch (err) {
        if (active === me) active = null;
        emit({ type: 'status', message: `error: ${(err as Error).message}` });
      }
    } else if (msg.type === 'pcm') {
      if (active?.port === port) active.session?.push(msg.data);
    } else if (msg.type === 'stop') {
      if (active?.port === port) stopActive();
    }
  });
  port.onDisconnect.addListener(() => {
    if (active?.port === port) stopActive();
  });
});

// Popup queries.
browser.runtime.onMessage.addListener((msg: { type: string; tabId?: number }) => {
  if (msg.type === 'state') {
    const st = msg.tabId !== undefined ? lastStats.get(msg.tabId) : undefined;
    const sorted = [...(st?.lag ?? [])].sort((a, b) => a - b);
    return Promise.resolve({
      running: active?.tabId === msg.tabId && !!active?.session,
      p50: sorted.length ? sorted[Math.floor(sorted.length / 2)] : null,
      enc: st?.enc ?? null,
    });
  }
  return undefined;
});
