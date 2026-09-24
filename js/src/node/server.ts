#!/usr/bin/env node
// Local engine for the browser extension: the same pipeline as the CLI,
// driven over a localhost WebSocket. Used when the browser has no WebGPU
// (e.g. Firefox on Linux) or to keep the model out of the browser.
//
//   node src/node/server.ts [--port 8765] [--dtype fp16]
//
// Protocol: client sends {"type":"hello","rate":48000,"options":{...}} as
// text, then raw little-endian Float32 mono PCM as binary frames. The server
// answers with SubtitleEvent JSON text frames. Only extension origins may
// connect, so ordinary web pages cannot use the engine.

import { createHash } from 'node:crypto';
import { createReadStream, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { normalize, resolve } from 'node:path';
import type { Duplex } from 'node:stream';
import { parseArgs } from 'node:util';
import { withDefaults, type SubtitlerOptions } from '../core/config.ts';
import { loadModels, Subtitler, type SubtitleEvent } from '../core/pipeline.ts';
import { MODELS, setupNode } from './env.ts';

const { values: a } = parseArgs({
  options: {
    port: { type: 'string', default: '8765' },
    device: { type: 'string', default: 'webgpu' },
    dtype: { type: 'string', default: 'fp16' },
    'encoder-dtype': { type: 'string' },
    model: { type: 'string', default: 'kotoba-whisper-bilingual-v1.0-xattn' },
  },
});

await setupNode();
const base = withDefaults({
  model: a.model,
  vadModel: 'silero-vad',
  device: a.device as never,
  dtype: { encoder: a['encoder-dtype'] ?? a.dtype!, decoder: a.dtype! },
  alignmentHeads: [34, 25, 23, 29, 28, 39],
});
const models = await loadModels(base);
console.error(`models ready (${base.device}, enc ${base.dtype.encoder}, dec ${base.dtype.decoder})`);

// --- minimal RFC 6455 ------------------------------------------------------
const GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';

function frame(opcode: number, payload: Buffer): Buffer {
  const n = payload.length;
  const head = n < 126 ? Buffer.alloc(2) : n < 65536 ? Buffer.alloc(4) : Buffer.alloc(10);
  head[0] = 0x80 | opcode;
  if (n < 126) head[1] = n;
  else if (n < 65536) {
    head[1] = 126;
    head.writeUInt16BE(n, 2);
  } else {
    head[1] = 127;
    head.writeBigUInt64BE(BigInt(n), 2);
  }
  return Buffer.concat([head, payload]);
}

type Handler = (opcode: number, data: Buffer) => void;

function reader(sock: Duplex, onMessage: Handler) {
  let buf: Buffer = Buffer.alloc(0);
  let fragments: Buffer[] = [];
  let fragOpcode = 0;
  sock.on('data', (chunk: Buffer) => {
    buf = buf.length ? Buffer.concat([buf, chunk]) : chunk;
    for (;;) {
      if (buf.length < 2) return;
      const fin = buf[0] & 0x80;
      const opcode = buf[0] & 0x0f;
      const masked = buf[1] & 0x80;
      let len = buf[1] & 0x7f;
      let off = 2;
      if (len === 126) {
        if (buf.length < 4) return;
        len = buf.readUInt16BE(2);
        off = 4;
      } else if (len === 127) {
        if (buf.length < 10) return;
        len = Number(buf.readBigUInt64BE(2));
        off = 10;
      }
      const maskOff = off;
      if (masked) off += 4;
      if (buf.length < off + len) return;
      const data = Buffer.from(buf.subarray(off, off + len));
      if (masked) for (let i = 0; i < len; i++) data[i] ^= buf[maskOff + (i & 3)];
      buf = buf.subarray(off + len);
      if (opcode === 0) {
        fragments.push(data);
        if (fin) {
          onMessage(fragOpcode, Buffer.concat(fragments));
          fragments = [];
        }
      } else if (!fin) {
        fragOpcode = opcode;
        fragments = [data];
      } else {
        onMessage(opcode, data);
      }
    }
  });
}

// Also serves ../models so the in-browser engine can load the patched model
// without uploading it anywhere (extension origins only).
const server = createServer((req, res) => {
  const origin = req.headers.origin ?? '';
  const cors = /^(moz|chrome)-extension:\/\//.test(origin) ? { 'access-control-allow-origin': origin } : {};
  const url = new URL(req.url ?? '/', 'http://x');
  if (url.pathname.startsWith('/models/')) {
    const rel = normalize(decodeURIComponent(url.pathname.slice('/models/'.length)));
    const file = resolve(MODELS, rel);
    if (!file.startsWith(MODELS + '/') || rel.split('/').some((p) => p.startsWith('.'))) {
      res.writeHead(403, cors).end();
      return;
    }
    let size: number;
    try {
      const st = statSync(file);
      if (!st.isFile()) throw new Error();
      size = st.size;
    } catch {
      res.writeHead(404, cors).end();
      return;
    }
    res.writeHead(200, { ...cors, 'content-length': size, 'content-type': file.endsWith('.json') ? 'application/json' : 'application/octet-stream' });
    if (req.method === 'HEAD') res.end();
    else createReadStream(file).pipe(res);
    return;
  }
  res.writeHead(200, { 'content-type': 'text/plain' });
  res.end('rt-subs engine\n');
});

let busy = false;

server.on('upgrade', (req, sock: Duplex) => {
  const origin = req.headers.origin ?? '';
  const key = req.headers['sec-websocket-key'];
  if (!key || !/^(moz|chrome)-extension:\/\//.test(origin)) {
    sock.end('HTTP/1.1 403 Forbidden\r\n\r\n');
    return;
  }
  if (busy) {
    // One stream at a time: the GPU is the bottleneck anyway.
    sock.end('HTTP/1.1 409 Conflict\r\n\r\n');
    return;
  }
  busy = true;
  const accept = createHash('sha1').update(key + GUID).digest('base64');
  sock.write(`HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ${accept}\r\n\r\n`);

  let sub: Subtitler | null = null;
  const send = (e: SubtitleEvent) => {
    if (!sock.destroyed) sock.write(frame(1, Buffer.from(JSON.stringify(e))));
  };
  const close = () => {
    sub?.stop();
    sub = null;
    busy = false;
  };
  sock.on('close', close);
  sock.on('error', close);

  reader(sock, (opcode, data) => {
    if (opcode === 8) {
      sock.end(frame(8, Buffer.alloc(0)));
      return;
    }
    if (opcode === 9) {
      sock.write(frame(10, data));
      return;
    }
    if (opcode === 1) {
      const msg = JSON.parse(data.toString('utf8'));
      if (msg.type === 'hello') {
        sub?.stop();
        const o: Partial<SubtitlerOptions> = { ...base, ...pick(msg.options ?? {}) };
        sub = new Subtitler(models, msg.rate, send, o);
        send({ type: 'status', message: 'ready' });
        console.error(`stream started @ ${msg.rate} Hz from ${origin}`);
      }
      return;
    }
    if (opcode === 2 && sub) {
      const aligned = new Float32Array(data.length >> 2);
      for (let i = 0; i < aligned.length; i++) aligned[i] = data.readFloatLE(i * 4);
      sub.push(aligned);
    }
  });
});

/** Client-tunable options only (never model paths or devices). */
function pick(o: Record<string, unknown>): Partial<SubtitlerOptions> {
  const out: Partial<SubtitlerOptions> = {};
  if (typeof o.staticPrompt === 'string') out.staticPrompt = o.staticPrompt.slice(0, 400);
  if (typeof o.frameThreshold === 'number') out.frameThreshold = Math.max(4, Math.min(100, o.frameThreshold));
  if (typeof o.tentativeTokens === 'number') out.tentativeTokens = Math.max(0, Math.min(32, o.tentativeTokens));
  return out;
}

server.listen(+a.port!, '127.0.0.1', () => console.error(`listening on ws://127.0.0.1:${a.port}`));
