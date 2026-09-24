//! Wasm signal-processing core for real-time JP->EN subtitles.
//!
//! JS owns the models (Silero VAD, Whisper via transformers.js); this crate
//! owns everything numeric around them:
//!
//! ```text
//! PCM @ 44.1/48 kHz ─► resample 16 kHz ─► high-pass ─► 512-sample VAD frames ─┐
//!                                                                               │ Silero prob
//!             segment buffer ◄── segmenter (hysteresis, adaptive silence) ◄─────┘
//!                   │
//!                   └─► incremental log-mel (128 x 3000) ─► Whisper encoder
//!
//! decoder cross-attention ─► AlignAtt (z-score, median, head mean) ─► emit / wait
//! ```

pub mod alignatt;
pub mod filter;
pub mod mel;
pub mod resample;
pub mod segmenter;

use std::collections::VecDeque;
use wasm_bindgen::prelude::*;

use mel::{HOP, MelCache, N_FRAMES, N_SAMPLES};
use segmenter::{FRAME, Segmenter, SegmenterConfig, events};

/// Silero v5 expects 64 samples of left context in front of each 512 frame.
const VAD_CONTEXT: usize = 64;

#[wasm_bindgen]
pub struct AudioPipeline {
    resampler: resample::Resampler,
    hpf: filter::HighPass,
    pending: VecDeque<f32>,
    vad_context: [f32; VAD_CONTEXT],
    history: VecDeque<f32>,
    pre_roll: usize,
    segmenter: Segmenter,
    segment: Vec<f32>,
    seg_start: u64,
    processed: u64,
    mel: MelCache,
    scratch: Vec<f32>,
}

#[wasm_bindgen]
impl AudioPipeline {
    /// `highpass_hz <= 0` disables the filter.
    #[wasm_bindgen(constructor)]
    pub fn new(input_rate: u32, highpass_hz: f32, pre_roll_ms: u32, n_mels: usize) -> AudioPipeline {
        AudioPipeline {
            resampler: resample::Resampler::new(input_rate, 16000, 16),
            hpf: filter::HighPass::new(16000.0, highpass_hz),
            pending: VecDeque::new(),
            vad_context: [0.0; VAD_CONTEXT],
            history: VecDeque::new(),
            pre_roll: pre_roll_ms as usize * 16,
            segmenter: Segmenter::new(SegmenterConfig::default()),
            segment: Vec::new(),
            seg_start: 0,
            processed: 0,
            mel: MelCache::new(n_mels),
            scratch: Vec::new(),
        }
    }

    /// Times in milliseconds; converted to 32 ms frames.
    #[allow(clippy::too_many_arguments)]
    pub fn configure_vad(
        &mut self,
        threshold: f32,
        neg_threshold: f32,
        min_silence_ms: u32,
        min_silence_floor_ms: u32,
        soft_max_ms: u32,
        hard_max_ms: u32,
        min_speech_ms: u32,
    ) {
        let f = |ms: u32| ((ms as usize * 16) / FRAME).max(1);
        // Leave room for the overflow cut to land before the 30 s window ends.
        let hard = f(hard_max_ms).min((N_SAMPLES - 2 * 16000) / FRAME);
        self.segmenter.cfg = SegmenterConfig {
            threshold,
            neg_threshold,
            min_silence_frames: f(min_silence_ms),
            min_silence_floor_frames: f(min_silence_floor_ms),
            soft_max_frames: f(soft_max_ms).min(hard),
            hard_max_frames: hard,
            min_speech_frames: f(min_speech_ms),
        };
    }

    /// Raw mono PCM at the input rate.
    pub fn push_input(&mut self, pcm: &[f32]) {
        self.scratch.clear();
        self.resampler.process(pcm, &mut self.scratch);
        self.hpf.process(&mut self.scratch);
        self.pending.extend(self.scratch.iter());
    }

    /// Resampled audio that has not been run through the VAD yet.
    pub fn pending_samples(&self) -> usize {
        self.pending.len()
    }

    pub fn vad_frame_ready(&self) -> bool {
        self.pending.len() >= FRAME
    }

    /// The next Silero input: 64 context samples followed by 512 new ones.
    /// Does not consume; call `push_vad_prob` with the result.
    pub fn vad_input(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(VAD_CONTEXT + FRAME);
        v.extend_from_slice(&self.vad_context);
        v.extend(self.pending.iter().take(FRAME));
        v
    }

    /// Consumes one frame and returns segmenter event flags
    /// (1 start, 2 end, 4 overflow, 8 discard).
    pub fn push_vad_prob(&mut self, prob: f32) -> u32 {
        let frame: Vec<f32> = self.pending.drain(..FRAME).collect();
        self.vad_context.copy_from_slice(&frame[FRAME - VAD_CONTEXT..]);
        let was_active = self.segmenter.active();
        let ev = self.segmenter.push(prob);
        if ev & events::START != 0 {
            self.clear_segment();
            let pre = self.pre_roll.min(self.history.len());
            self.segment.extend(self.history.iter().skip(self.history.len() - pre));
            self.seg_start = self.processed - self.segment.len() as u64;
            self.mel.reset();
        }
        if was_active || ev & events::START != 0 {
            self.segment.extend_from_slice(&frame);
            // Safety net if the caller never trims: stay inside Whisper's window.
            if self.segment.len() > N_SAMPLES {
                let over = self.segment.len() - N_SAMPLES;
                self.trim_front(over);
            }
        }
        self.history.extend(frame.iter());
        let excess = self.history.len().saturating_sub(self.pre_roll.max(FRAME));
        self.history.drain(..excess);
        self.processed += FRAME as u64;
        ev
    }

    pub fn in_speech(&self) -> bool {
        self.segmenter.active()
    }

    /// Samples in the current segment (active or finished-but-not-cleared).
    pub fn segment_len(&self) -> usize {
        self.segment.len()
    }

    /// Absolute position of the segment start, in 16 kHz samples.
    pub fn segment_start(&self) -> f64 {
        self.seg_start as f64
    }

    /// Absolute 16 kHz samples consumed by the VAD so far.
    pub fn processed(&self) -> f64 {
        self.processed as f64
    }

    pub fn segment_audio(&self) -> Vec<f32> {
        self.segment.clone()
    }

    /// Normalised log-mel for the segment into `out` (`n_mels * 3000`).
    /// Returns the number of encoder frames (20 ms) that contain audio.
    pub fn mel(&mut self, out: &mut [f32]) -> usize {
        self.mel.compute(&self.segment, out);
        self.segment.len().div_ceil(2 * HOP)
    }

    pub fn mel_size(&self) -> usize {
        self.mel.n_mels() * N_FRAMES
    }

    /// Drops audio from the front of the segment (rounded down to a mel hop).
    /// Returns how many samples were removed.
    pub fn trim_front(&mut self, samples: usize) -> usize {
        let n = (samples.min(self.segment.len()) / HOP) * HOP;
        if n == 0 {
            return 0;
        }
        self.segment.drain(..n);
        self.seg_start += n as u64;
        self.mel.shift(n / HOP);
        self.segmenter.trimmed(n / FRAME);
        n
    }

    /// The quietest 20 ms position within `radius` samples of `center`
    /// (segment-relative), for placing forced cuts between syllables.
    pub fn quiet_point(&self, center: usize, radius: usize) -> usize {
        let win = 2 * HOP;
        let lo = center.saturating_sub(radius) / HOP * HOP;
        let hi = (center + radius).min(self.segment.len().saturating_sub(win));
        let mut best = (f32::INFINITY, center.min(self.segment.len()) / HOP * HOP);
        let mut p = lo;
        while p <= hi {
            let e: f32 = self.segment[p..p + win].iter().map(|x| x * x).sum();
            if e < best.0 {
                best = (e, p);
            }
            p += HOP;
        }
        best.1
    }

    pub fn clear_segment(&mut self) {
        self.segment.clear();
        self.mel.reset();
    }

    /// Ends the active segment without waiting for silence.
    pub fn force_end(&mut self) {
        self.segmenter.force_end();
    }
}

#[wasm_bindgen]
pub struct AlignAttPolicy {
    inner: alignatt::AlignAtt,
}

#[wasm_bindgen]
impl AlignAttPolicy {
    #[wasm_bindgen(constructor)]
    pub fn new(heads: usize, frames: usize, median_width: usize) -> AlignAttPolicy {
        AlignAttPolicy { inner: alignatt::AlignAtt::new(heads, frames, median_width) }
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// `attn` is `[heads][rows][frames]`; the first `skip` rows are ignored.
    pub fn push(&mut self, attn: &[f32], rows: usize, skip: usize) {
        self.inner.push(attn, rows, skip);
    }

    pub fn attended_frame(&mut self, content_frames: usize, lookahead: usize) -> usize {
        self.inner.attended_frame(content_frames, lookahead)
    }

    pub fn rows(&self) -> usize {
        self.inner.rows()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_segments_and_trims() {
        let mut p = AudioPipeline::new(48000, 80.0, 300, 128);
        p.configure_vad(0.5, 0.35, 500, 100, 10000, 24000, 250);
        p.push_input(&vec![0.1f32; 48000 * 3]);
        let mut n = 0;
        let mut started = false;
        while p.vad_frame_ready() {
            assert_eq!(p.vad_input().len(), 576);
            // Speech for the middle second.
            let prob = if (31..62).contains(&n) { 0.9 } else { 0.0 };
            let ev = p.push_vad_prob(prob);
            if ev & events::START != 0 {
                started = true;
                // Pre-roll pulled in ~300 ms of history.
                assert!(p.segment_len() >= 4000 && p.segment_len() <= 5400, "{}", p.segment_len());
            }
            if ev & events::END != 0 {
                break;
            }
            n += 1;
        }
        assert!(started);
        let before = p.segment_len();
        let mut mel = vec![0f32; p.mel_size()];
        let frames = p.mel(&mut mel);
        assert_eq!(frames, before.div_ceil(320));
        let cut = p.trim_front(1000);
        assert_eq!(cut, 960);
        assert_eq!(p.segment_len(), before - 960);
        let frames = p.mel(&mut mel);
        assert_eq!(frames, (before - 960).div_ceil(320));
    }
}
