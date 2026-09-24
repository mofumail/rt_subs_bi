export interface Settings {
  /** auto: WebGPU in the browser if available, else the local engine. */
  engine: 'auto' | 'browser' | 'local';
  localUrl: string;
  /** Where the in-browser engine downloads the xattn-patched model from. */
  modelBase: string;
  modelId: string;
  /** small: 4-bit encoder (~530 MB total). quality: fp16 encoder (~1.5 GB), lower lag. */
  quality: 'small' | 'quality';
  prompt: string;
  showTentative: boolean;
  fontScale: number;
  frameThreshold: number;
}

export const DEFAULT_SETTINGS: Settings = {
  engine: 'auto',
  localUrl: 'ws://127.0.0.1:8765',
  modelBase: 'http://127.0.0.1:8765/models/',
  modelId: 'kotoba-whisper-bilingual-v1.0-xattn',
  quality: 'small',
  prompt: '',
  showTentative: true,
  fontScale: 1,
  frameThreshold: 25,
};

export async function loadSettings(): Promise<Settings> {
  const s = (await browser.storage.local.get('settings')).settings as Partial<Settings> | undefined;
  return { ...DEFAULT_SETTINGS, ...s };
}

export async function saveSettings(s: Settings) {
  await browser.storage.local.set({ settings: s });
}

// content <-> background port messages
export type ToBackground = { type: 'start'; rate: number } | { type: 'pcm'; data: Float32Array } | { type: 'stop' };
