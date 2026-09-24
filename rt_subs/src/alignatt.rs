//! AlignAtt simultaneous policy (Papi et al. 2023; SimulStreaming, Macháček
//! et al. 2025) on Whisper cross-attention.
//!
//! On every decoder step we look at where the selected cross-attention heads
//! point for the token that is about to be emitted. If that frame is within
//! `frame_threshold` of the end of the received audio, the model is "reading
//! the frontier": the token depends on speech we have not fully heard, so we
//! stop and wait for more audio. Everything emitted before that point is final.
//!
//! Attention rows are z-scored per head and frame across the tokens of the
//! current step, median-filtered along time, and averaged across heads (the
//! same recipe Whisper uses for word timestamps), maintained incrementally so
//! each new token costs O(heads x frames).

pub struct AlignAtt {
    heads: usize,
    frames: usize,
    median_width: usize,
    sum: Vec<f64>,
    sumsq: Vec<f64>,
    rows: usize,
    last: Vec<f32>,
    scratch: Vec<f32>,
    avg: Vec<f32>,
}

impl AlignAtt {
    pub fn new(heads: usize, frames: usize, median_width: usize) -> Self {
        Self {
            heads,
            frames,
            median_width: median_width.max(1) | 1,
            sum: vec![0.0; heads * frames],
            sumsq: vec![0.0; heads * frames],
            rows: 0,
            last: vec![0.0; heads * frames],
            scratch: vec![0.0; frames],
            avg: vec![0.0; frames],
        }
    }

    pub fn reset(&mut self) {
        self.sum.fill(0.0);
        self.sumsq.fill(0.0);
        self.rows = 0;
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    /// `attn` is `[heads][n_rows][frames]` (the decoder output layout after
    /// gathering the alignment heads); rows `skip..n_rows` are accumulated.
    pub fn push(&mut self, attn: &[f32], n_rows: usize, skip: usize) {
        let (h, f) = (self.heads, self.frames);
        assert_eq!(attn.len(), h * n_rows * f, "attention shape mismatch");
        if skip >= n_rows {
            return;
        }
        for head in 0..h {
            let sum = &mut self.sum[head * f..(head + 1) * f];
            let sumsq = &mut self.sumsq[head * f..(head + 1) * f];
            for r in skip..n_rows {
                let row = &attn[(head * n_rows + r) * f..(head * n_rows + r + 1) * f];
                for i in 0..f {
                    let v = row[i] as f64;
                    sum[i] += v;
                    sumsq[i] += v * v;
                }
            }
            let last = &attn[(head * n_rows + n_rows - 1) * f..(head * n_rows + n_rows) * f];
            self.last[head * f..(head + 1) * f].copy_from_slice(last);
        }
        self.rows += n_rows - skip;
    }

    fn median_filter(src: &[f32], dst: &mut [f32], width: usize) {
        let half = width / 2;
        let n = src.len();
        let mut win = Vec::with_capacity(width);
        for i in 0..n {
            win.clear();
            for k in 0..width {
                // Reflect padding, like scipy / whisper.timing.
                let j = i as isize + k as isize - half as isize;
                let j = if j < 0 { -j } else if j >= n as isize { 2 * n as isize - 2 - j } else { j };
                win.push(src[j.clamp(0, n as isize - 1) as usize]);
            }
            win.sort_unstable_by(|a, b| a.total_cmp(b));
            dst[i] = win[half];
        }
    }

    /// Head-averaged alignment score of the most recent row over `0..limit` frames.
    fn score_last(&mut self, limit: usize) -> &[f32] {
        let (h, f) = (self.heads, self.frames);
        let limit = limit.min(f);
        self.avg[..limit].fill(0.0);
        let n = self.rows.max(1) as f64;
        let mut z = vec![0f32; limit];
        for head in 0..h {
            for i in 0..limit {
                let v = self.last[head * f + i] as f64;
                z[i] = if self.rows >= 2 {
                    let mean = self.sum[head * f + i] / n;
                    let var = (self.sumsq[head * f + i] / n - mean * mean).max(0.0);
                    ((v - mean) / (var.sqrt() + 1e-6)) as f32
                } else {
                    v as f32
                };
            }
            Self::median_filter(&z, &mut self.scratch[..limit], self.median_width.min(limit.max(1)) | 1);
            for i in 0..limit {
                self.avg[i] += self.scratch[i] / h as f32;
            }
        }
        &self.avg[..limit]
    }

    /// Most-attended encoder frame of the latest token. The search covers
    /// `content_frames + lookahead` so attention drifting into the zero
    /// padding (the model imagining speech it has not heard) is visible.
    pub fn attended_frame(&mut self, content_frames: usize, lookahead: usize) -> usize {
        if self.rows == 0 {
            return 0;
        }
        let limit = (content_frames + lookahead).min(self.frames).max(1);
        let s = self.score_last(limit);
        let mut best = 0;
        for (i, &v) in s.iter().enumerate() {
            if v > s[best] {
                best = i;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peaked(heads: usize, frames: usize, rows: &[usize]) -> Vec<f32> {
        let mut v = vec![0.001f32; heads * rows.len() * frames];
        for h in 0..heads {
            for (r, &p) in rows.iter().enumerate() {
                for d in 0..6usize {
                    let i = (p + d).min(frames - 1);
                    v[(h * rows.len() + r) * frames + i] = 0.3;
                }
            }
        }
        v
    }

    #[test]
    fn tracks_diagonal() {
        let mut a = AlignAtt::new(4, 200, 7);
        let rows = [5, 20, 40, 60, 80];
        a.push(&peaked(4, 200, &rows), rows.len(), 0);
        let f = a.attended_frame(150, 20);
        assert!((80..86).contains(&f), "{f}");
        a.push(&peaked(4, 200, &[120]), 1, 0);
        let f = a.attended_frame(150, 20);
        assert!((120..126).contains(&f), "{f}");
    }

    #[test]
    fn skip_rows_are_ignored() {
        let mut a = AlignAtt::new(2, 100, 7);
        a.push(&peaked(2, 100, &[90, 10, 30]), 3, 1);
        assert_eq!(a.rows(), 2);
        let f = a.attended_frame(100, 0);
        assert!((30..36).contains(&f), "{f}");
    }
}
