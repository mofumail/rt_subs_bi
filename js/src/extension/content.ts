// YouTube page side: taps the <video>'s audio and draws the subtitle overlay.
//
// Firefox has no tabCapture, so audio is routed through Web Audio with
// createMediaElementSource: element -> destination (you still hear it) and
// -> a tap that streams mono PCM to the background engine. Tapping the actual
// playback path keeps subtitles in sync with what you hear, including
// pauses, seeks and live-stream latency.

import type { LineEvent, SubtitleEvent } from '../core/pipeline.ts';
import { loadSettings, type Settings, type ToBackground } from './settings.ts';

let ctx: AudioContext | null = null;
const sources = new WeakMap<HTMLMediaElement, MediaElementAudioSourceNode>();
let tap: ScriptProcessorNode | null = null;
let sink: GainNode | null = null;
let video: HTMLVideoElement | null = null;
let port: browser.runtime.Port | null = null;
let settings: Settings;
let watchdog: number | undefined;

// --- overlay -----------------------------------------------------------------
const STYLE = `
.rtsubs-root{position:absolute;left:0;right:0;bottom:11%;display:flex;flex-direction:column;align-items:center;gap:.25em;
  pointer-events:none;z-index:60;font-family:"YouTube Noto",Roboto,"Noto Sans",Arial,sans-serif}
.rtsubs-line{max-width:82%;padding:.1em .45em;border-radius:3px;background:rgba(8,8,8,.75);color:#fff;
  line-height:1.3;text-align:center;text-shadow:0 0 2px #000;transition:opacity .4s}
.rtsubs-prev{opacity:.7}
.rtsubs-tent{color:#b9b9b9;font-style:italic}
.rtsubs-status{position:absolute;top:10px;left:10px;z-index:61;pointer-events:none;font:12px/1.4 Roboto,Arial,sans-serif;
  color:#fff;background:rgba(0,0,0,.6);padding:2px 8px;border-radius:3px;transition:opacity .6s}
`;

const overlay = {
  root: null as HTMLDivElement | null,
  prev: null as HTMLDivElement | null,
  cur: null as HTMLDivElement | null,
  status: null as HTMLDivElement | null,
  prevTimer: 0,
  statusTimer: 0,
  lastId: -1,

  mount(player: HTMLElement) {
    if (this.root?.isConnected && this.root.parentElement === player) return;
    this.unmount();
    if (!document.getElementById('rtsubs-style')) {
      const st = document.createElement('style');
      st.id = 'rtsubs-style';
      st.textContent = STYLE;
      document.head.appendChild(st);
    }
    this.root = document.createElement('div');
    this.root.className = 'rtsubs-root';
    this.prev = document.createElement('div');
    this.prev.className = 'rtsubs-line rtsubs-prev';
    this.cur = document.createElement('div');
    this.cur.className = 'rtsubs-line';
    this.root.append(this.prev, this.cur);
    this.status = document.createElement('div');
    this.status.className = 'rtsubs-status';
    player.append(this.root, this.status);
    this.resize(player);
    this.prev.style.display = this.cur.style.display = 'none';
  },

  resize(player: HTMLElement) {
    if (!this.root) return;
    const h = player.clientHeight || 360;
    this.root.style.fontSize = `${Math.max(14, h * 0.042) * (settings?.fontScale ?? 1)}px`;
  },

  unmount() {
    this.root?.remove();
    this.status?.remove();
    this.root = this.prev = this.cur = this.status = null;
  },

  setStatus(msg: string, sticky = false) {
    if (!this.status) return;
    this.status.textContent = `JP→EN · ${msg}`;
    this.status.style.opacity = '1';
    clearTimeout(this.statusTimer);
    if (!sticky) this.statusTimer = window.setTimeout(() => this.status && (this.status.style.opacity = '0'), 4000);
  },

  /** Long lines: keep the tail, cut at a word boundary. */
  clip(s: string, max = 140) {
    if (s.length <= max) return s;
    const cut = s.slice(-max);
    const sp = cut.indexOf(' ');
    return '…' + (sp > 0 && sp < 20 ? cut.slice(sp + 1) : cut);
  },

  line(e: LineEvent) {
    if (!this.cur || !this.prev) return;
    if (e.final) {
      this.cur.style.display = 'none';
      if (e.text) {
        this.prev.textContent = this.clip(e.text);
        this.prev.style.display = '';
        this.prev.style.opacity = '1';
        clearTimeout(this.prevTimer);
        this.prevTimer = window.setTimeout(() => this.prev && (this.prev.style.opacity = '0'), 3500 + e.text.length * 40);
      }
      return;
    }
    const tent = settings.showTentative ? e.tentative : '';
    if (!e.text && !tent) return;
    const full = this.clip(e.text + tent);
    const committed = full.slice(0, Math.max(0, full.length - tent.length));
    this.cur.replaceChildren(document.createTextNode(committed));
    if (tent) {
      const span = document.createElement('span');
      span.className = 'rtsubs-tent';
      span.textContent = full.slice(committed.length);
      this.cur.append(span);
    }
    this.cur.style.display = '';
  },
};

// --- capture -----------------------------------------------------------------
function findVideo(): { video: HTMLVideoElement; player: HTMLElement } | null {
  const player = document.querySelector<HTMLElement>('#movie_player, .html5-video-player');
  const v = player?.querySelector<HTMLVideoElement>('video') ?? document.querySelector<HTMLVideoElement>('video');
  if (!v) return null;
  return { video: v, player: player ?? v.parentElement! };
}

function attach(v: HTMLVideoElement) {
  if (!ctx || !tap) return;
  let src = sources.get(v);
  if (!src) {
    // One-way door: from now on this element plays through our graph.
    src = ctx.createMediaElementSource(v);
    src.connect(ctx.destination);
    sources.set(v, src);
  }
  if (video && video !== v) sources.get(video)?.disconnect(tap);
  src.connect(tap);
  video = v;
}

async function start() {
  settings = await loadSettings();
  const found = findVideo();
  if (!found) throw new Error('no video on this page');
  overlay.mount(found.player);
  overlay.setStatus('starting…', true);

  ctx ??= new AudioContext();
  if (ctx.state !== 'running') await ctx.resume();
  // ScriptProcessor rather than AudioWorklet: worklet modules from an
  // extension URL are subject to the page's CSP; this runs anywhere, and its
  // output is muted so late callbacks cannot glitch playback.
  tap = ctx.createScriptProcessor(4096, 1, 1);
  sink = ctx.createGain();
  sink.gain.value = 0;
  tap.connect(sink).connect(ctx.destination);
  attach(found.video);

  port = browser.runtime.connect({ name: 'rt-subs' });
  port.onMessage.addListener((raw) => {
    const e = raw as SubtitleEvent;
    if (e.type === 'line') overlay.line(e);
    else if (e.type === 'status') {
      const sticky = !/^(listening|ready)/.test(e.message);
      overlay.setStatus(e.message, sticky);
      if (e.message.startsWith('error') || e.message.startsWith('stopped')) stop(false);
    }
  });
  port.onDisconnect.addListener(() => stop(false));
  tap.addEventListener('audioprocess', (ev) => {
    const data = new Float32Array((ev as AudioProcessingEvent).inputBuffer.getChannelData(0));
    port?.postMessage({ type: 'pcm', data } satisfies ToBackground);
  });
  port.postMessage({ type: 'start', rate: ctx.sampleRate } satisfies ToBackground);

  // YouTube is a SPA: follow the player if the <video> element is replaced.
  watchdog = window.setInterval(() => {
    const f = findVideo();
    if (!f) return;
    if (f.video !== video) attach(f.video);
    overlay.mount(f.player);
    overlay.resize(f.player);
  }, 1500);
}

function stop(notify = true) {
  if (notify) port?.postMessage({ type: 'stop' } satisfies ToBackground);
  port?.disconnect();
  port = null;
  clearInterval(watchdog);
  if (tap) {
    if (video) sources.get(video)?.disconnect(tap);
    tap.disconnect();
    tap = null;
  }
  sink?.disconnect();
  sink = null;
  // The element stays routed through ctx -> destination, so audio keeps playing.
  overlay.setStatus('off');
  setTimeout(() => !port && overlay.unmount(), 3000);
}

browser.runtime.onMessage.addListener((msg: { cmd: string }) => {
  if (msg.cmd === 'toggle') {
    if (port) {
      stop();
      return Promise.resolve({ running: false });
    }
    return start().then(
      () => ({ running: true }),
      (err: Error) => {
        overlay.setStatus(`error: ${err.message}`);
        stop(false);
        return { running: false, error: err.message };
      },
    );
  }
  if (msg.cmd === 'running') return Promise.resolve({ running: !!port });
  return undefined;
});

browser.storage.onChanged.addListener(async () => {
  settings = await loadSettings();
});
