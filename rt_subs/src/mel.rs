//! Incremental Whisper log-mel spectrogram (large-v3 layout: 128 bins).
//!
//! Bit-for-bit the same recipe as `WhisperFeatureExtractor`: audio zero-padded
//! to 30 s, centred STFT (n_fft 400, hop 160, periodic Hann, reflect padding),
//! power spectrum, Slaney mel filters, log10, clamp to `max - 8`, `(x + 4) / 4`.
//!
//! The streaming buffer is re-encoded on every policy step, so frames whose
//! window lies fully inside the audio received so far are cached and only the
//! few tail frames touching the growing edge are recomputed.

use realfft::{RealFftPlanner, RealToComplex};
use std::sync::Arc;

pub const SAMPLE_RATE: usize = 16000;
pub const N_FFT: usize = 400;
pub const HOP: usize = 160;
pub const N_FRAMES: usize = 3000;
pub const N_SAMPLES: usize = N_FRAMES * HOP; // 30 s
const N_BINS: usize = N_FFT / 2 + 1;

fn hz_to_mel(f: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if f >= min_log_hz { min_log_mel + (f / min_log_hz).ln() / logstep } else { f / f_sp }
}

fn mel_to_hz(m: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if m >= min_log_mel { min_log_hz * (logstep * (m - min_log_mel)).exp() } else { f_sp * m }
}

/// Slaney-normalised triangular filters, `n_mels x N_BINS`.
pub fn mel_filters(n_mels: usize) -> Vec<f32> {
    let fmax = SAMPLE_RATE as f64 / 2.0;
    let mmin = hz_to_mel(0.0);
    let mmax = hz_to_mel(fmax);
    let pts: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(mmin + (mmax - mmin) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let mut w = vec![0f32; n_mels * N_BINS];
    for m in 0..n_mels {
        let (lo, ce, hi) = (pts[m], pts[m + 1], pts[m + 2]);
        let enorm = 2.0 / (hi - lo);
        for k in 0..N_BINS {
            let f = k as f64 * SAMPLE_RATE as f64 / N_FFT as f64;
            let lower = (f - lo) / (ce - lo);
            let upper = (hi - f) / (hi - ce);
            w[m * N_BINS + k] = (lower.min(upper).max(0.0) * enorm) as f32;
        }
    }
    w
}

pub struct MelCache {
    n_mels: usize,
    filters: Vec<f32>,
    /// Sparse view of `filters`: first non-zero bin and the weights from there.
    spans: Vec<(usize, Vec<f32>)>,
    window: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
    scratch_in: Vec<f32>,
    scratch_out: Vec<realfft::num_complex::Complex<f32>>,
    power: Vec<f32>,
    /// log10 mel frames (`n_mels` each) that will never change again.
    stable: Vec<f32>,
    stable_frames: usize,
    dirty_head: bool,
}

impl MelCache {
    pub fn new(n_mels: usize) -> Self {
        let filters = mel_filters(n_mels);
        let spans = (0..n_mels)
            .map(|m| {
                let row = &filters[m * N_BINS..(m + 1) * N_BINS];
                let first = row.iter().position(|&x| x > 0.0).unwrap_or(0);
                let last = row.iter().rposition(|&x| x > 0.0).unwrap_or(0);
                (first, row[first..=last.max(first)].to_vec())
            })
            .collect();
        // torch.hann_window(400) is periodic.
        let window = (0..N_FFT)
            .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos()) as f32)
            .collect();
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(N_FFT);
        let scratch_out = fft.make_output_vec();
        Self {
            n_mels,
            filters,
            spans,
            window,
            fft,
            scratch_in: vec![0.0; N_FFT],
            scratch_out,
            power: vec![0.0; N_BINS],
            stable: Vec::new(),
            stable_frames: 0,
            dirty_head: false,
        }
    }

    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    #[allow(dead_code)]
    pub fn filters(&self) -> &[f32] {
        &self.filters
    }

    pub fn reset(&mut self) {
        self.stable.clear();
        self.stable_frames = 0;
        self.dirty_head = false;
    }

    /// The first `hop_frames` frames of audio were dropped from the front of
    /// the buffer (the caller trims in multiples of `HOP`). Everything shifts;
    /// frames 0 and 1 reflect-pad across the new start and are rebuilt lazily.
    pub fn shift(&mut self, hop_frames: usize) {
        let drop = hop_frames.min(self.stable_frames);
        self.stable.drain(..drop * self.n_mels);
        self.stable_frames -= drop;
        self.dirty_head = true;
    }

    fn frame_log_mel(&mut self, audio: &[f32], frame: usize, out: &mut [f32]) {
        let len = audio.len() as isize;
        let start = frame as isize * HOP as isize - (N_FFT / 2) as isize;
        for i in 0..N_FFT {
            let mut j = start + i as isize;
            // numpy "reflect" at the left edge; zeros past the right edge
            // (the audio is zero-padded to 30 s before the STFT).
            if j < 0 {
                j = -j;
            }
            self.scratch_in[i] = if j < len { audio[j as usize] * self.window[i] } else { 0.0 };
        }
        self.fft.process(&mut self.scratch_in, &mut self.scratch_out).expect("fft");
        for (p, c) in self.power.iter_mut().zip(&self.scratch_out) {
            *p = c.re * c.re + c.im * c.im;
        }
        for (m, (first, w)) in self.spans.iter().enumerate() {
            let mut acc = 0f32;
            for (a, b) in w.iter().zip(&self.power[*first..]) {
                acc += a * b;
            }
            out[m] = acc.max(1e-10).log10();
        }
    }

    /// Writes the normalised `n_mels x 3000` feature matrix for `audio`
    /// (at most 30 s) into `out`, row-major by mel bin.
    pub fn compute(&mut self, audio: &[f32], out: &mut [f32]) {
        assert!(audio.len() <= N_SAMPLES, "segment longer than 30 s");
        assert_eq!(out.len(), self.n_mels * N_FRAMES);
        let n_mels = self.n_mels;
        // A frame is final once its right window edge is inside the audio.
        let final_frames = if audio.len() >= N_FFT / 2 {
            ((audio.len() - N_FFT / 2) / HOP + 1).min(N_FRAMES)
        } else {
            0
        };
        let mut frame = vec![0f32; n_mels];
        if self.dirty_head {
            for f in 0..self.stable_frames.min(2) {
                self.frame_log_mel(audio, f, &mut frame);
                self.stable[f * n_mels..(f + 1) * n_mels].copy_from_slice(&frame);
            }
            self.dirty_head = false;
        }
        while self.stable_frames < final_frames {
            let f = self.stable_frames;
            self.frame_log_mel(audio, f, &mut frame);
            self.stable.extend_from_slice(&frame);
            self.stable_frames += 1;
        }
        // Tail frames touch the (virtual) zero padding; frames entirely in the
        // padding are exactly log10(1e-10) = -10.
        let touched = (audio.len() + N_FFT / 2).div_ceil(HOP).min(N_FRAMES);
        let mut tail = Vec::with_capacity(touched.saturating_sub(self.stable_frames) * n_mels);
        for f in self.stable_frames..touched {
            self.frame_log_mel(audio, f, &mut frame);
            tail.extend_from_slice(&frame);
        }
        let mut max = -10f32;
        for &v in self.stable.iter().chain(&tail) {
            if v > max {
                max = v;
            }
        }
        let floor = max - 8.0;
        let norm = |v: f32| (v.max(floor) + 4.0) / 4.0;
        let pad_value = norm(-10.0);
        let computed = self.stable_frames + tail.len() / n_mels;
        for m in 0..n_mels {
            let row = &mut out[m * N_FRAMES..(m + 1) * N_FRAMES];
            for (f, o) in row.iter_mut().enumerate().take(computed) {
                let v = if f < self.stable_frames {
                    self.stable[f * n_mels + m]
                } else {
                    tail[(f - self.stable_frames) * n_mels + m]
                };
                *o = norm(v);
            }
            for o in row[computed..].iter_mut() {
                *o = pad_value;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_shape_and_norm() {
        let f = mel_filters(128);
        assert_eq!(f.len(), 128 * N_BINS);
        // Every filter has support and Slaney normalisation keeps them small.
        for m in 0..128 {
            let row = &f[m * N_BINS..(m + 1) * N_BINS];
            assert!(row.iter().any(|&x| x > 0.0), "empty filter {m}");
            assert!(row.iter().all(|&x| x < 0.05));
        }
    }

    #[test]
    fn incremental_matches_one_shot() {
        let audio: Vec<f32> = (0..16000 * 3)
            .map(|i| ((i as f32 * 0.013).sin() + (i as f32 * 0.0021).cos()) * 0.3)
            .collect();
        let mut one = MelCache::new(128);
        let mut full = vec![0f32; 128 * N_FRAMES];
        one.compute(&audio, &mut full);

        let mut inc = MelCache::new(128);
        let mut out = vec![0f32; 128 * N_FRAMES];
        for end in (1234..audio.len()).step_by(3217) {
            inc.compute(&audio[..end], &mut out);
        }
        inc.compute(&audio, &mut out);
        for (a, b) in full.iter().zip(&out) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn shift_matches_recompute() {
        let audio: Vec<f32> = (0..16000 * 4).map(|i| (i as f32 * 0.0171).sin() * 0.2 + (i as f32 * 0.3).sin() * 0.05).collect();
        let mut inc = MelCache::new(128);
        let mut out = vec![0f32; 128 * N_FRAMES];
        inc.compute(&audio, &mut out);
        let cut = 160 * 57;
        inc.shift(57);
        inc.compute(&audio[cut..], &mut out);
        let mut fresh = MelCache::new(128);
        let mut want = vec![0f32; 128 * N_FRAMES];
        fresh.compute(&audio[cut..], &mut want);
        for (a, b) in want.iter().zip(&out) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn silence_is_flat() {
        let mut c = MelCache::new(128);
        let mut out = vec![0f32; 128 * N_FRAMES];
        c.compute(&[0.0; 16000], &mut out);
        assert!(out.iter().all(|&v| (v - (-1.5)).abs() < 1e-6));
    }
}
