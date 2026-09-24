//! Streaming rational-ratio polyphase resampler (Kaiser-windowed sinc).
//!
//! Browsers hand us 44.1/48 kHz audio; Whisper and Silero want 16 kHz.
//! The ratio is reduced to L/M and every one of the L phases gets its own
//! pre-computed filter, so there is no interpolation error and no drift.

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Zeroth-order modified Bessel function of the first kind (series expansion).
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let q = x * x / 4.0;
    for k in 1..64 {
        term *= q / (k as f64 * k as f64);
        sum += term;
        if term < 1e-12 * sum {
            break;
        }
    }
    sum
}

pub struct Resampler {
    l: u64,
    m: u64,
    /// Taps on each side of the centre, in input samples.
    half: usize,
    /// `l` phases x `2 * half` taps.
    table: Vec<f32>,
    /// Input samples; `buf[0]` has absolute index `buf_start`.
    buf: Vec<f32>,
    buf_start: i64,
    next_out: u64,
    passthrough: bool,
}

impl Resampler {
    /// `zero_crossings` controls quality (16 is transparent for speech).
    pub fn new(in_rate: u32, out_rate: u32, zero_crossings: usize) -> Self {
        let g = gcd(in_rate as u64, out_rate as u64);
        let l = out_rate as u64 / g;
        let m = in_rate as u64 / g;
        if l == m {
            return Self {
                l: 1,
                m: 1,
                half: 0,
                table: Vec::new(),
                buf: Vec::new(),
                buf_start: 0,
                next_out: 0,
                passthrough: true,
            };
        }
        // Cutoff relative to the input Nyquist; leave a little transition band.
        let fc = (l as f64 / m as f64).min(1.0) * 0.94;
        let half = (zero_crossings as f64 / fc).ceil() as usize;
        let beta = 8.6;
        let i0_beta = bessel_i0(beta);
        let taps = 2 * half;
        let mut table = vec![0f32; l as usize * taps];
        for phase in 0..l as usize {
            let frac = phase as f64 / l as f64;
            let row = &mut table[phase * taps..(phase + 1) * taps];
            let mut sum = 0.0;
            let mut tmp = vec![0f64; taps];
            for (j, t) in tmp.iter_mut().enumerate() {
                // Distance (in input samples) between tap j and the output instant.
                let d = j as f64 - half as f64 + 1.0 - frac;
                let x = fc * d;
                let sinc = if x.abs() < 1e-12 { 1.0 } else { (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x) };
                let r = d / half as f64;
                let w = if r.abs() >= 1.0 { 0.0 } else { bessel_i0(beta * (1.0 - r * r).sqrt()) / i0_beta };
                *t = sinc * w;
                sum += *t;
            }
            for (o, t) in row.iter_mut().zip(tmp) {
                *o = (t / sum) as f32;
            }
        }
        Self {
            l,
            m,
            half,
            table,
            // Left zero padding so the very first outputs have full support.
            buf: vec![0.0; half],
            buf_start: -(half as i64),
            next_out: 0,
            passthrough: false,
        }
    }

    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        if self.passthrough {
            out.extend_from_slice(input);
            return;
        }
        self.buf.extend_from_slice(input);
        let taps = 2 * self.half;
        let avail_end = self.buf_start + self.buf.len() as i64; // exclusive
        loop {
            let t = self.next_out * self.m;
            let base = (t / self.l) as i64;
            let phase = (t % self.l) as usize;
            let first = base - self.half as i64 + 1;
            if first + taps as i64 > avail_end {
                break;
            }
            let off = (first - self.buf_start) as usize;
            let x = &self.buf[off..off + taps];
            let h = &self.table[phase * taps..(phase + 1) * taps];
            let mut acc = 0f32;
            for (a, b) in x.iter().zip(h) {
                acc += a * b;
            }
            out.push(acc);
            self.next_out += 1;
        }
        // Drop input that no future output can reference.
        let t = self.next_out * self.m;
        let first_needed = (t / self.l) as i64 - self.half as i64 + 1;
        let drop = (first_needed - self.buf_start).max(0) as usize;
        if drop > 0 {
            let drop = drop.min(self.buf.len());
            self.buf.drain(..drop);
            self.buf_start += drop as i64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(rate: u32, hz: f32, n: usize) -> Vec<f32> {
        (0..n).map(|i| (2.0 * std::f32::consts::PI * hz * i as f32 / rate as f32).sin()).collect()
    }

    fn run(in_rate: u32, input: &[f32], chunk: usize) -> Vec<f32> {
        let mut r = Resampler::new(in_rate, 16000, 16);
        let mut out = Vec::new();
        for c in input.chunks(chunk) {
            r.process(c, &mut out);
        }
        out
    }

    #[test]
    fn output_length_tracks_ratio() {
        for rate in [48000, 44100, 22050, 16000] {
            let input = tone(rate, 440.0, rate as usize * 2);
            let out = run(rate, &input, 1234);
            let expected = 32000.0;
            assert!((out.len() as f32 - expected).abs() < 200.0, "{rate}: {}", out.len());
        }
    }

    #[test]
    fn chunking_is_invisible() {
        let input = tone(44100, 1000.0, 44100);
        let a = run(44100, &input, 128);
        let b = run(44100, &input, 7777);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn passband_preserved_and_alias_rejected() {
        // 1 kHz passes with unit gain, 12 kHz (above the new Nyquist) is removed.
        let pass = run(48000, &tone(48000, 1000.0, 48000), 480);
        let stop = run(48000, &tone(48000, 12000.0, 48000), 480);
        let rms = |v: &[f32]| (v[2000..14000].iter().map(|x| x * x).sum::<f32>() / 12000.0).sqrt();
        assert!((rms(&pass) - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.01, "{}", rms(&pass));
        assert!(rms(&stop) < 1e-3, "{}", rms(&stop));
    }
}
