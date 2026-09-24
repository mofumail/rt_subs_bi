import { loadSettings, saveSettings, type Settings } from './settings.ts';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const toggle = $<HTMLButtonElement>('toggle');
const state = $<HTMLDivElement>('state');

async function tabId() {
  const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
  return tab?.id;
}

function render(running: boolean) {
  toggle.textContent = running ? 'Stop subtitles' : 'Start subtitles';
  toggle.classList.toggle('on', running);
}

async function refresh() {
  const id = await tabId();
  if (id === undefined) return;
  try {
    const r = (await browser.tabs.sendMessage(id, { cmd: 'running' })) as { running: boolean };
    render(r.running);
    const s = (await browser.runtime.sendMessage({ type: 'state', tabId: id })) as { p50: number | null; enc: number | null };
    state.textContent = r.running && s.p50 != null ? `median lag ${(s.p50 / 1000).toFixed(2)} s · encoder ${s.enc?.toFixed(0)} ms/step` : '';
  } catch {
    toggle.disabled = true;
    state.textContent = 'Open a YouTube video or stream to use subtitles.';
  }
}

toggle.addEventListener('click', async () => {
  const id = await tabId();
  if (id === undefined) return;
  toggle.disabled = true;
  const r = (await browser.tabs.sendMessage(id, { cmd: 'toggle' })) as { running: boolean; error?: string };
  toggle.disabled = false;
  render(r.running);
  state.textContent = r.error ?? (r.running ? 'Starting — first run downloads the model.' : '');
});

const fields = ['prompt', 'showTentative', 'engine', 'quality', 'frameThreshold', 'fontScale', 'localUrl', 'modelBase', 'modelId'] as const;

(async () => {
  const s = await loadSettings();
  for (const f of fields) {
    const el = $<HTMLInputElement>(f);
    if (el.type === 'checkbox') el.checked = s[f] as boolean;
    else el.value = String(s[f]);
    el.addEventListener('change', async () => {
      const cur = await loadSettings();
      const v = el.type === 'checkbox' ? el.checked : el.type === 'range' ? Number(el.value) : el.value;
      await saveSettings({ ...cur, [f]: v } as Settings);
    });
  }
  await refresh();
  setInterval(refresh, 2000);
})();
