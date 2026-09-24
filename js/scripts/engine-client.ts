// Exercises the local engine the way the extension does: WebSocket from an
// extension origin, hello, then 48 kHz Float32 PCM paced at 1x.
//   node scripts/engine-client.ts clip.webm [seconds]
import { spawn } from 'node:child_process';

const [file, secs = '40'] = process.argv.slice(2);
const ws = new WebSocket('ws://127.0.0.1:8765', { headers: { origin: 'moz-extension://test' } } as never);
ws.binaryType = 'arraybuffer';
await new Promise((res, rej) => {
  ws.onopen = res;
  ws.onerror = rej;
});
ws.send(JSON.stringify({ type: 'hello', rate: 48000, options: { staticPrompt: '' } }));
ws.onmessage = (m) => {
  const e = JSON.parse(m.data as string);
  if (e.type === 'status') console.log('status:', e.message);
  if (e.type === 'line' && e.final && e.text) console.log(`[${e.start.toFixed(1)}] ${e.text}`);
};
const ff = spawn('ffmpeg', ['-loglevel', 'error', '-re', '-i', file, '-t', secs, '-ac', '1', '-ar', '48000', '-f', 'f32le', 'pipe:1']);
ff.stdout.on('data', (b: Buffer) => ws.send(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)));
await new Promise((r) => ff.on('close', r));
await new Promise((r) => setTimeout(r, 3000));
ws.close();
process.exit(0);
