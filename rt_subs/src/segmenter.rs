//! Utterance segmentation from Silero VAD probabilities (one per 32 ms frame).
//!
//! Silero is only a gate: music-only stretches never reach Whisper (which is
//! where "Thank you for watching" hallucinations come from) and utterance
//! boundaries give AlignAtt clean places to finalise and reset.
//!
//! VTubers often talk for a long time without a real pause, so the silence
//! needed to close a segment shrinks as the segment grows: 500 ms at first,
//! down to ~100 ms near the Whisper window limit. Past the hard limit the
//! caller cuts using attention alignment instead.

pub const FRAME: usize = 512; // samples at 16 kHz

#[derive(Clone, Copy, Debug)]
pub struct SegmenterConfig {
    pub threshold: f32,
    pub neg_threshold: f32,
    pub min_silence_frames: usize,
    pub min_silence_floor_frames: usize,
    /// Segment length (frames) at which the silence requirement starts shrinking.
    pub soft_max_frames: usize,
    /// Segment length (frames) at which the caller must cut.
    pub hard_max_frames: usize,
    /// Segments with fewer speech frames than this are flagged as noise.
    pub min_speech_frames: usize,
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            neg_threshold: 0.35,
            min_silence_frames: 16,      // 512 ms
            min_silence_floor_frames: 3, // 96 ms
            soft_max_frames: 312,        // 10 s
            hard_max_frames: 750,        // 24 s
            min_speech_frames: 8,        // 256 ms
        }
    }
}

pub mod events {
    pub const START: u32 = 1;
    pub const END: u32 = 2;
    pub const OVERFLOW: u32 = 4;
    pub const DISCARD: u32 = 8;
}

pub struct Segmenter {
    pub cfg: SegmenterConfig,
    active: bool,
    len_frames: usize,
    speech_frames: usize,
    silence_start: Option<usize>,
}

impl Segmenter {
    pub fn new(cfg: SegmenterConfig) -> Self {
        Self { cfg, active: false, len_frames: 0, speech_frames: 0, silence_start: None }
    }

    pub fn active(&self) -> bool {
        self.active
    }

    fn required_silence(&self) -> usize {
        let c = &self.cfg;
        if self.len_frames <= c.soft_max_frames {
            return c.min_silence_frames;
        }
        let span = (c.hard_max_frames - c.soft_max_frames).max(1) as f32;
        let t = ((self.len_frames - c.soft_max_frames) as f32 / span).min(1.0);
        let v = c.min_silence_frames as f32 + (c.min_silence_floor_frames as f32 - c.min_silence_frames as f32) * t;
        v.round() as usize
    }

    /// Feed one frame's probability; returns `events` bit flags. On START the
    /// current frame is the first of the segment; on END it is the last.
    pub fn push(&mut self, prob: f32) -> u32 {
        let c = self.cfg;
        let mut ev = 0;
        if !self.active {
            if prob >= c.threshold {
                self.active = true;
                self.len_frames = 1;
                self.speech_frames = 1;
                self.silence_start = None;
                ev |= events::START;
            }
            return ev;
        }
        self.len_frames += 1;
        if prob >= c.threshold {
            self.speech_frames += 1;
            self.silence_start = None;
        } else if prob < c.neg_threshold && self.silence_start.is_none() {
            self.silence_start = Some(self.len_frames - 1);
        }
        if let Some(s) = self.silence_start
            && self.len_frames - s >= self.required_silence() {
                ev |= events::END;
                if self.speech_frames < c.min_speech_frames {
                    ev |= events::DISCARD;
                }
                self.active = false;
                return ev;
            }
        if self.len_frames >= c.hard_max_frames {
            ev |= events::OVERFLOW;
        }
        ev
    }

    /// The caller dropped `frames` from the front of the active segment.
    pub fn trimmed(&mut self, frames: usize) {
        self.len_frames = self.len_frames.saturating_sub(frames);
        self.speech_frames = self.speech_frames.saturating_sub(frames);
        if let Some(s) = self.silence_start {
            self.silence_start = if s >= frames { Some(s - frames) } else { None };
        }
    }

    pub fn force_end(&mut self) {
        self.active = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_and_end() {
        let mut s = Segmenter::new(SegmenterConfig::default());
        assert_eq!(s.push(0.1), 0);
        assert_eq!(s.push(0.9), events::START);
        for _ in 0..20 {
            assert_eq!(s.push(0.9), 0);
        }
        // Mid-range probability neither starts nor resets the silence timer.
        let mut ended = None;
        for i in 0..40 {
            let ev = s.push(if i == 0 { 0.1 } else { 0.4 });
            if ev & events::END != 0 {
                ended = Some(i);
                break;
            }
        }
        assert_eq!(ended, Some(15));
        assert!(!s.active());
    }

    #[test]
    fn short_blip_is_discarded() {
        let mut s = Segmenter::new(SegmenterConfig::default());
        s.push(0.9);
        let mut last = 0;
        for _ in 0..16 {
            last = s.push(0.0);
        }
        assert_eq!(last, events::END | events::DISCARD);
    }

    #[test]
    fn silence_requirement_shrinks() {
        let mut s = Segmenter::new(SegmenterConfig::default());
        s.push(0.9);
        for _ in 0..700 {
            assert_eq!(s.push(0.9) & events::END, 0);
        }
        let mut n = 0;
        while s.push(0.0) & events::END == 0 {
            n += 1;
        }
        assert!(n < 6, "{n}");
    }

    #[test]
    fn overflow_flag() {
        let mut s = Segmenter::new(SegmenterConfig::default());
        s.push(0.9);
        let mut seen = false;
        for _ in 0..800 {
            if s.push(0.9) & events::OVERFLOW != 0 {
                seen = true;
                break;
            }
        }
        assert!(seen);
        s.trimmed(400);
        assert_eq!(s.push(0.9) & events::OVERFLOW, 0);
    }
}
