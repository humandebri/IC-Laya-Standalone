//! Symmetric per-output-row W8A8. Activations are quantized per token;
//! accumulation is i32, while scales, bias, norms and attention stay F32.
use candle_core::{Device, Tensor};
use std::sync::Arc;

#[derive(Clone)]
pub struct Int8Matrix {
    rows: usize,
    cols: usize,
    values: Arc<[i8]>,
    scales: Arc<[f32]>,
}
impl Int8Matrix {
    pub fn from_bytes(bytes: &[u8], rows: usize, cols: usize) -> candle_core::Result<Self> {
        // 16384 * 128 * 128 fits i32, including hostile -128 values.
        if rows == 0 || cols == 0 || cols > 16384 {
            candle_core::bail!("invalid int8 dimensions")
        }
        let count = rows
            .checked_mul(cols)
            .ok_or_else(|| candle_core::Error::Msg("int8 size overflow".into()))?;
        let length = rows
            .checked_mul(4)
            .and_then(|n| count.checked_add(n))
            .ok_or_else(|| candle_core::Error::Msg("int8 size overflow".into()))?;
        if bytes.len() != length {
            candle_core::bail!("invalid int8 payload length")
        }
        let scales: Vec<f32> = bytes[count..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        if scales.iter().any(|s| !s.is_finite() || *s <= 0.) {
            candle_core::bail!("invalid int8 scale")
        }
        Ok(Self {
            rows,
            cols,
            values: bytes[..count].iter().map(|b| *b as i8).collect(),
            scales: scales.into(),
        })
    }
    pub fn dims(&self) -> [usize; 2] {
        [self.rows, self.cols]
    }
    pub fn gather(&self, ids: &[u32]) -> candle_core::Result<Tensor> {
        let mut out = Vec::with_capacity(ids.len() * self.cols);
        for &id in ids {
            let row = id as usize;
            if row >= self.rows {
                candle_core::bail!("int8 embedding index")
            }
            out.extend(
                self.values[row * self.cols..(row + 1) * self.cols]
                    .iter()
                    .map(|v| *v as f32 * self.scales[row]),
            );
        }
        Tensor::from_vec(out, (ids.len(), self.cols), &Device::Cpu)
    }
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (tokens, cols) = x.dims2()?;
        if cols != self.cols {
            candle_core::bail!("int8 linear shape")
        }
        let shape = [tokens, self.rows, cols];
        let input =
            crate::profile::measure("int8.input", shape, || x.flatten_all()?.to_vec1::<f32>())?;
        let (activation, scales) = crate::profile::measure("int8.quantize", shape, || {
            let mut activation = vec![0i8; tokens * cols];
            let mut scales = Vec::with_capacity(tokens);
            for (t, row) in input.chunks_exact(cols).enumerate() {
                if row.iter().any(|v| !v.is_finite()) {
                    candle_core::bail!("nonfinite int8 activation")
                }
                let peak = row.iter().fold(0f32, |a, b| a.max(b.abs()));
                let scale = if peak == 0. {
                    1.
                } else {
                    (peak / 127.).max(f32::MIN_POSITIVE)
                };
                for (q, v) in activation[t * cols..(t + 1) * cols].iter_mut().zip(row) {
                    *q = (v / scale).round().clamp(-127., 127.) as i8;
                }
                scales.push(scale);
            }
            Ok::<_, candle_core::Error>((activation, scales))
        })?;
        let out = crate::profile::measure("int8.matmul", shape, || {
            let mut out = vec![0f32; tokens * self.rows];
            let full_t = tokens / 4 * 4;
            let full_r = self.rows / 4 * 4;
            for t in (0..full_t).step_by(4) {
                let a = &activation[t * cols..(t + 4) * cols];
                for r in (0..full_r).step_by(4) {
                    let sums = dot4x4(a, &self.values[r * cols..(r + 4) * cols], cols);
                    for i in 0..4 {
                        for j in 0..4 {
                            out[(t + i) * self.rows + r + j] =
                                sums[i * 4 + j] as f32 * scales[t + i] * self.scales[r + j];
                        }
                    }
                }
                for r in full_r..self.rows {
                    for i in 0..4 {
                        out[(t + i) * self.rows + r] = dot(
                            &a[i * cols..(i + 1) * cols],
                            &self.values[r * cols..(r + 1) * cols],
                        ) as f32
                            * scales[t + i]
                            * self.scales[r];
                    }
                }
            }
            for t in full_t..tokens {
                for r in 0..self.rows {
                    out[t * self.rows + r] = dot(
                        &activation[t * cols..(t + 1) * cols],
                        &self.values[r * cols..(r + 1) * cols],
                    ) as f32
                        * scales[t]
                        * self.scales[r];
                }
            }
            out
        });
        crate::profile::measure("int8.tensor", shape, || {
            Tensor::from_vec(out, (tokens, self.rows), x.device())
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn dot(a: &[i8], b: &[i8]) -> i32 {
    a.iter().zip(b).map(|(&x, &y)| x as i32 * y as i32).sum()
}

// SIMD is enabled only for this kernel, avoiding Candle's global simd128 cfg.
#[cfg(target_arch = "wasm32")]
#[allow(unsafe_code)]
fn dot(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::wasm32::*;
    #[target_feature(enable = "simd128")]
    unsafe fn kernel(a: &[i8], b: &[i8]) -> i32 {
        let mut sum = i32x4_splat(0);
        let mut ac = a.chunks_exact(16);
        let mut bc = b.chunks_exact(16);
        for (x, y) in ac.by_ref().zip(bc.by_ref()) {
            // Each chunk contains exactly 16 initialized bytes; unaligned loads are allowed.
            let x = unsafe { v128_load(x.as_ptr().cast()) };
            let y = unsafe { v128_load(y.as_ptr().cast()) };
            sum = i32x4_add(
                sum,
                i32x4_dot_i16x8(i16x8_extend_low_i8x16(x), i16x8_extend_low_i8x16(y)),
            );
            sum = i32x4_add(
                sum,
                i32x4_dot_i16x8(i16x8_extend_high_i8x16(x), i16x8_extend_high_i8x16(y)),
            );
        }
        let tail: i32 = ac
            .remainder()
            .iter()
            .zip(bc.remainder())
            .map(|(&x, &y)| x as i32 * y as i32)
            .sum();
        i32x4_extract_lane::<0>(sum)
            + i32x4_extract_lane::<1>(sum)
            + i32x4_extract_lane::<2>(sum)
            + i32x4_extract_lane::<3>(sum)
            + tail
    }
    // IC's Wasm runtime supports simd128. Equal length and <=16384 are enforced by forward.
    unsafe { kernel(a, b) }
}

// A 4-token x 4-output tile reuses each widened input/weight across four dots.
// Integer accumulation order may change, but sums are exact: K<=16384 bounds
// the worst absolute sum by 16384*128*128, safely below i32::MAX.
#[cfg(not(target_arch = "wasm32"))]
fn dot4x4(a: &[i8], b: &[i8], cols: usize) -> [i32; 16] {
    let mut result = [0; 16];
    for i in 0..4 {
        for j in 0..4 {
            result[i * 4 + j] = dot(&a[i * cols..(i + 1) * cols], &b[j * cols..(j + 1) * cols]);
        }
    }
    result
}
#[cfg(target_arch = "wasm32")]
#[allow(unsafe_code)]
fn dot4x4(a: &[i8], b: &[i8], cols: usize) -> [i32; 16] {
    use std::arch::wasm32::*;
    #[target_feature(enable = "simd128")]
    unsafe fn kernel(a: &[i8], b: &[i8], cols: usize) -> [i32; 16] {
        let mut s00 = i32x4_splat(0);
        let mut s01 = i32x4_splat(0);
        let mut s02 = i32x4_splat(0);
        let mut s03 = i32x4_splat(0);
        let mut s10 = i32x4_splat(0);
        let mut s11 = i32x4_splat(0);
        let mut s12 = i32x4_splat(0);
        let mut s13 = i32x4_splat(0);
        let mut s20 = i32x4_splat(0);
        let mut s21 = i32x4_splat(0);
        let mut s22 = i32x4_splat(0);
        let mut s23 = i32x4_splat(0);
        let mut s30 = i32x4_splat(0);
        let mut s31 = i32x4_splat(0);
        let mut s32 = i32x4_splat(0);
        let mut s33 = i32x4_splat(0);
        let bp0 = b.as_ptr();
        let bp1 = unsafe { b.as_ptr().add(cols) };
        let bp2 = unsafe { b.as_ptr().add(2 * cols) };
        let bp3 = unsafe { b.as_ptr().add(3 * cols) };
        let ap0 = a.as_ptr();
        let ap1 = unsafe { a.as_ptr().add(cols) };
        let ap2 = unsafe { a.as_ptr().add(2 * cols) };
        let ap3 = unsafe { a.as_ptr().add(3 * cols) };
        let end = cols / 16 * 16;
        let mut k = 0;
        while k < end {
            let w0 = unsafe { v128_load(bp0.add(k).cast()) };
            let lo0 = i16x8_extend_low_i8x16(w0);
            let hi0 = i16x8_extend_high_i8x16(w0);
            let w1 = unsafe { v128_load(bp1.add(k).cast()) };
            let lo1 = i16x8_extend_low_i8x16(w1);
            let hi1 = i16x8_extend_high_i8x16(w1);
            let w2 = unsafe { v128_load(bp2.add(k).cast()) };
            let lo2 = i16x8_extend_low_i8x16(w2);
            let hi2 = i16x8_extend_high_i8x16(w2);
            let w3 = unsafe { v128_load(bp3.add(k).cast()) };
            let lo3 = i16x8_extend_low_i8x16(w3);
            let hi3 = i16x8_extend_high_i8x16(w3);
            let x = unsafe { v128_load(ap0.add(k).cast()) };
            let xl = i16x8_extend_low_i8x16(x);
            let xh = i16x8_extend_high_i8x16(x);
            s00 = i32x4_add(
                s00,
                i32x4_add(i32x4_dot_i16x8(xl, lo0), i32x4_dot_i16x8(xh, hi0)),
            );
            s01 = i32x4_add(
                s01,
                i32x4_add(i32x4_dot_i16x8(xl, lo1), i32x4_dot_i16x8(xh, hi1)),
            );
            s02 = i32x4_add(
                s02,
                i32x4_add(i32x4_dot_i16x8(xl, lo2), i32x4_dot_i16x8(xh, hi2)),
            );
            s03 = i32x4_add(
                s03,
                i32x4_add(i32x4_dot_i16x8(xl, lo3), i32x4_dot_i16x8(xh, hi3)),
            );
            let x = unsafe { v128_load(ap1.add(k).cast()) };
            let xl = i16x8_extend_low_i8x16(x);
            let xh = i16x8_extend_high_i8x16(x);
            s10 = i32x4_add(
                s10,
                i32x4_add(i32x4_dot_i16x8(xl, lo0), i32x4_dot_i16x8(xh, hi0)),
            );
            s11 = i32x4_add(
                s11,
                i32x4_add(i32x4_dot_i16x8(xl, lo1), i32x4_dot_i16x8(xh, hi1)),
            );
            s12 = i32x4_add(
                s12,
                i32x4_add(i32x4_dot_i16x8(xl, lo2), i32x4_dot_i16x8(xh, hi2)),
            );
            s13 = i32x4_add(
                s13,
                i32x4_add(i32x4_dot_i16x8(xl, lo3), i32x4_dot_i16x8(xh, hi3)),
            );
            let x = unsafe { v128_load(ap2.add(k).cast()) };
            let xl = i16x8_extend_low_i8x16(x);
            let xh = i16x8_extend_high_i8x16(x);
            s20 = i32x4_add(
                s20,
                i32x4_add(i32x4_dot_i16x8(xl, lo0), i32x4_dot_i16x8(xh, hi0)),
            );
            s21 = i32x4_add(
                s21,
                i32x4_add(i32x4_dot_i16x8(xl, lo1), i32x4_dot_i16x8(xh, hi1)),
            );
            s22 = i32x4_add(
                s22,
                i32x4_add(i32x4_dot_i16x8(xl, lo2), i32x4_dot_i16x8(xh, hi2)),
            );
            s23 = i32x4_add(
                s23,
                i32x4_add(i32x4_dot_i16x8(xl, lo3), i32x4_dot_i16x8(xh, hi3)),
            );
            let x = unsafe { v128_load(ap3.add(k).cast()) };
            let xl = i16x8_extend_low_i8x16(x);
            let xh = i16x8_extend_high_i8x16(x);
            s30 = i32x4_add(
                s30,
                i32x4_add(i32x4_dot_i16x8(xl, lo0), i32x4_dot_i16x8(xh, hi0)),
            );
            s31 = i32x4_add(
                s31,
                i32x4_add(i32x4_dot_i16x8(xl, lo1), i32x4_dot_i16x8(xh, hi1)),
            );
            s32 = i32x4_add(
                s32,
                i32x4_add(i32x4_dot_i16x8(xl, lo2), i32x4_dot_i16x8(xh, hi2)),
            );
            s33 = i32x4_add(
                s33,
                i32x4_add(i32x4_dot_i16x8(xl, lo3), i32x4_dot_i16x8(xh, hi3)),
            );
            k = k.wrapping_add(16);
        }
        let mut result = [0; 16];
        result[0] = i32x4_extract_lane::<0>(s00)
            .wrapping_add(i32x4_extract_lane::<1>(s00))
            .wrapping_add(i32x4_extract_lane::<2>(s00))
            .wrapping_add(i32x4_extract_lane::<3>(s00));
        result[1] = i32x4_extract_lane::<0>(s01)
            .wrapping_add(i32x4_extract_lane::<1>(s01))
            .wrapping_add(i32x4_extract_lane::<2>(s01))
            .wrapping_add(i32x4_extract_lane::<3>(s01));
        result[2] = i32x4_extract_lane::<0>(s02)
            .wrapping_add(i32x4_extract_lane::<1>(s02))
            .wrapping_add(i32x4_extract_lane::<2>(s02))
            .wrapping_add(i32x4_extract_lane::<3>(s02));
        result[3] = i32x4_extract_lane::<0>(s03)
            .wrapping_add(i32x4_extract_lane::<1>(s03))
            .wrapping_add(i32x4_extract_lane::<2>(s03))
            .wrapping_add(i32x4_extract_lane::<3>(s03));
        result[4] = i32x4_extract_lane::<0>(s10)
            .wrapping_add(i32x4_extract_lane::<1>(s10))
            .wrapping_add(i32x4_extract_lane::<2>(s10))
            .wrapping_add(i32x4_extract_lane::<3>(s10));
        result[5] = i32x4_extract_lane::<0>(s11)
            .wrapping_add(i32x4_extract_lane::<1>(s11))
            .wrapping_add(i32x4_extract_lane::<2>(s11))
            .wrapping_add(i32x4_extract_lane::<3>(s11));
        result[6] = i32x4_extract_lane::<0>(s12)
            .wrapping_add(i32x4_extract_lane::<1>(s12))
            .wrapping_add(i32x4_extract_lane::<2>(s12))
            .wrapping_add(i32x4_extract_lane::<3>(s12));
        result[7] = i32x4_extract_lane::<0>(s13)
            .wrapping_add(i32x4_extract_lane::<1>(s13))
            .wrapping_add(i32x4_extract_lane::<2>(s13))
            .wrapping_add(i32x4_extract_lane::<3>(s13));
        result[8] = i32x4_extract_lane::<0>(s20)
            .wrapping_add(i32x4_extract_lane::<1>(s20))
            .wrapping_add(i32x4_extract_lane::<2>(s20))
            .wrapping_add(i32x4_extract_lane::<3>(s20));
        result[9] = i32x4_extract_lane::<0>(s21)
            .wrapping_add(i32x4_extract_lane::<1>(s21))
            .wrapping_add(i32x4_extract_lane::<2>(s21))
            .wrapping_add(i32x4_extract_lane::<3>(s21));
        result[10] = i32x4_extract_lane::<0>(s22)
            .wrapping_add(i32x4_extract_lane::<1>(s22))
            .wrapping_add(i32x4_extract_lane::<2>(s22))
            .wrapping_add(i32x4_extract_lane::<3>(s22));
        result[11] = i32x4_extract_lane::<0>(s23)
            .wrapping_add(i32x4_extract_lane::<1>(s23))
            .wrapping_add(i32x4_extract_lane::<2>(s23))
            .wrapping_add(i32x4_extract_lane::<3>(s23));
        result[12] = i32x4_extract_lane::<0>(s30)
            .wrapping_add(i32x4_extract_lane::<1>(s30))
            .wrapping_add(i32x4_extract_lane::<2>(s30))
            .wrapping_add(i32x4_extract_lane::<3>(s30));
        result[13] = i32x4_extract_lane::<0>(s31)
            .wrapping_add(i32x4_extract_lane::<1>(s31))
            .wrapping_add(i32x4_extract_lane::<2>(s31))
            .wrapping_add(i32x4_extract_lane::<3>(s31));
        result[14] = i32x4_extract_lane::<0>(s32)
            .wrapping_add(i32x4_extract_lane::<1>(s32))
            .wrapping_add(i32x4_extract_lane::<2>(s32))
            .wrapping_add(i32x4_extract_lane::<3>(s32));
        result[15] = i32x4_extract_lane::<0>(s33)
            .wrapping_add(i32x4_extract_lane::<1>(s33))
            .wrapping_add(i32x4_extract_lane::<2>(s33))
            .wrapping_add(i32x4_extract_lane::<3>(s33));
        for k in end..cols {
            for i in 0..4 {
                for j in 0..4 {
                    result[i * 4 + j] += a[i * cols + k] as i32 * b[j * cols + k] as i32;
                }
            }
        }
        result
    }
    // Caller passes four complete, contiguous rows, each of length cols<=16384.
    // k<floor(cols/16)*16 guarantees every unaligned v128 load is in bounds.
    unsafe { kernel(a, b, cols) }
}
