// Bundles the Firefox extension into dist/firefox (load it from about:debugging,
// or `npx web-ext run -s dist/firefox`).
import { build } from 'esbuild';
import { cpSync, mkdirSync, rmSync } from 'node:fs';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const out = resolve(root, 'dist/firefox');
rmSync(out, { recursive: true, force: true });
mkdirSync(resolve(out, 'ort'), { recursive: true });

const common = { bundle: true, target: 'firefox128', platform: 'browser', logLevel: 'warning', legalComments: 'none', minify: !process.env.DEV } as const;
await build({ ...common, entryPoints: [resolve(root, 'src/extension/background.ts')], outfile: resolve(out, 'background.js'), format: 'esm' });
await build({ ...common, entryPoints: [resolve(root, 'src/extension/content.ts')], outfile: resolve(out, 'content.js'), format: 'iife' });
await build({ ...common, entryPoints: [resolve(root, 'src/extension/popup.ts')], outfile: resolve(out, 'popup.js'), format: 'iife' });

for (const f of ['manifest.json', 'background.html', 'popup.html', 'icon.svg']) cpSync(resolve(root, 'src/extension', f), resolve(out, f));
cpSync(resolve(root, 'pkg/rt_subs/rt_subs_bg.wasm'), resolve(out, 'rt_subs_bg.wasm'));
const ort = resolve(root, 'node_modules/onnxruntime-web/dist');
for (const f of ['ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.wasm']) cpSync(resolve(ort, f), resolve(out, 'ort', f));
console.log(`built ${out}`);
