// Float16 <-> Float32 without relying on Float16Array (not in every runtime).

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

export function toHalf(src: Float32Array): Uint16Array {
  const out = new Uint16Array(src.length);
  for (let i = 0; i < src.length; i++) {
    f32[0] = src[i];
    const x = u32[0];
    const sign = (x >>> 16) & 0x8000;
    let exp = ((x >>> 23) & 0xff) - 127 + 15;
    let mant = x & 0x7fffff;
    if (exp <= 0) {
      if (exp < -10) {
        out[i] = sign;
        continue;
      }
      mant |= 0x800000;
      const shift = 14 - exp;
      out[i] = sign | ((mant + (1 << (shift - 1))) >>> shift);
      continue;
    }
    if (exp >= 31) {
      out[i] = sign | 0x7c00;
      continue;
    }
    // Round to nearest.
    mant += 0x1000;
    if (mant & 0x800000) {
      mant = 0;
      exp += 1;
      if (exp >= 31) {
        out[i] = sign | 0x7c00;
        continue;
      }
    }
    out[i] = sign | (exp << 10) | (mant >>> 13);
  }
  return out;
}

const TABLE = (() => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >>> 10) & 0x1f;
    const m = h & 0x3ff;
    t[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return t;
})();

export function fromHalf(src: Uint16Array, start = 0, end = src.length): Float32Array {
  const out = new Float32Array(end - start);
  for (let i = start; i < end; i++) out[i - start] = TABLE[src[i]];
  return out;
}

/** Data of an ORT float/float16 tensor as Float32, optionally a sub-range. */
export function tensorF32(data: unknown, type: string, start = 0, end?: number): Float32Array {
  if (type === 'float32') {
    const d = data as Float32Array;
    return d.subarray(start, end ?? d.length);
  }
  if (type === 'float16') {
    // ORT hands out Uint16Array (or Float16Array where supported).
    const d = data as ArrayBufferView & { length: number };
    const u = d instanceof Uint16Array ? d : new Uint16Array(d.buffer, d.byteOffset, d.length);
    return fromHalf(u, start, end ?? u.length);
  }
  throw new Error(`unsupported tensor type ${type}`);
}
