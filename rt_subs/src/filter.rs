//! Second-order Butterworth high-pass (RBJ biquad).
//!
//! Removes DC and the bass/kick part of stream BGM below the voice band. It
//! does not separate music from speech; Whisper tolerates the rest well, and
//! heavier denoising tends to hurt ASR more than it helps.

pub struct HighPass {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
    enabled: bool,
}

impl HighPass {
    pub fn new(sample_rate: f32, cutoff_hz: f32) -> Self {
        let enabled = cutoff_hz > 0.0;
        let w0 = 2.0 * std::f32::consts::PI * cutoff_hz.max(1.0) / sample_rate;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * std::f32::consts::FRAC_1_SQRT_2);
        let a0 = 1.0 + alpha;
        Self {
            b0: (1.0 + cos) / 2.0 / a0,
            b1: -(1.0 + cos) / a0,
            b2: (1.0 + cos) / 2.0 / a0,
            a1: -2.0 * cos / a0,
            a2: (1.0 - alpha) / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
            enabled,
        }
    }

    pub fn process(&mut self, data: &mut [f32]) {
        if !self.enabled {
            return;
        }
        for s in data.iter_mut() {
            let x = *s;
            let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2;
            self.x2 = self.x1;
            self.x1 = x;
            self.y2 = self.y1;
            self.y1 = y;
            *s = y;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gain(hz: f32) -> f32 {
        let mut f = HighPass::new(16000.0, 80.0);
        let mut v: Vec<f32> = (0..32000).map(|i| (2.0 * std::f32::consts::PI * hz * i as f32 / 16000.0).sin()).collect();
        f.process(&mut v);
        let rms = (v[16000..].iter().map(|x| x * x).sum::<f32>() / 16000.0).sqrt();
        rms / std::f32::consts::FRAC_1_SQRT_2
    }

    #[test]
    fn response() {
        assert!(gain(20.0) < 0.1);
        assert!((gain(80.0) - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.03);
        assert!((gain(1000.0) - 1.0).abs() < 0.01);
    }
}
