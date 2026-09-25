//! Bounded, CPU-only attention products for the canister's short sequences.
#![allow(unsafe_code)]

use candle_core::{Result, Tensor};

#[cfg(target_arch = "wasm32")]
fn compatible(q: &Tensor, k: &Tensor) -> bool {
    if let (Ok((qh, _, qd)), Ok((kh, _, kd))) = (q.dims3(), k.dims3()) {
        qh == kh && qd == kd && qd % 4 == 0
    } else {
        false
    }
}

pub fn qk(q: &Tensor, k: &Tensor) -> Result<Tensor> {
    #[cfg(target_arch = "wasm32")]
    if compatible(q, k) {
        let (heads, rows, width) = q.dims3()?;
        let (_, cols, _) = k.dims3()?;
        let qv = q.flatten_all()?.to_vec1::<f32>()?;
        let kv = k.flatten_all()?.to_vec1::<f32>()?;
        let mut result = vec![0f32; heads * rows * cols];
        unsafe { qk_simd(&qv, &kv, &mut result, heads, rows, cols, width) };
        return Tensor::from_vec(result, (heads, rows, cols), q.device());
    }
    q.matmul(&k.transpose(1, 2)?.contiguous()?)
}

pub fn av(probs: &Tensor, v: &Tensor) -> Result<Tensor> {
    #[cfg(target_arch = "wasm32")]
    {
        let (heads, rows, tokens) = probs.dims3()?;
        let (vh, vt, width) = v.dims3()?;
        if heads == vh && tokens == vt && width % 4 == 0 {
            let pv = probs.flatten_all()?.to_vec1::<f32>()?;
            let vv = v.flatten_all()?.to_vec1::<f32>()?;
            let mut result = vec![0f32; heads * rows * width];
            unsafe { av_simd(&pv, &vv, &mut result, heads, rows, tokens, width) };
            return Tensor::from_vec(result, (heads, rows, width), probs.device());
        }
    }
    probs.matmul(v)
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn qk_simd(q: &[f32], k: &[f32], out: &mut [f32], heads: usize, rows: usize, cols: usize, width: usize) {
    use std::arch::wasm32::*;
    for head in 0..heads {
        for row in 0..rows {
            let qp = unsafe { q.as_ptr().add((head * rows + row) * width) };
            for col in 0..cols {
                let kp = unsafe { k.as_ptr().add((head * cols + col) * width) };
                let mut sum = f32x4_splat(0.0);
                for lane in (0..width).step_by(4) {
                    let a = unsafe { v128_load(qp.add(lane).cast()) };
                    let b = unsafe { v128_load(kp.add(lane).cast()) };
                    sum = f32x4_add(sum, f32x4_mul(a, b));
                }
                out[(head * rows + row) * cols + col] =
                    f32x4_extract_lane::<0>(sum) + f32x4_extract_lane::<1>(sum)
                    + f32x4_extract_lane::<2>(sum) + f32x4_extract_lane::<3>(sum);
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn av_simd(probs: &[f32], v: &[f32], out: &mut [f32], heads: usize, rows: usize, tokens: usize, width: usize) {
    use std::arch::wasm32::*;
    for head in 0..heads {
        for row in 0..rows {
            let op = unsafe { out.as_mut_ptr().add((head * rows + row) * width) };
            for col in (0..width).step_by(4) {
                let mut sum = f32x4_splat(0.0);
                for token in 0..tokens {
                    let weight = f32x4_splat(probs[(head * rows + row) * tokens + token]);
                    let value = unsafe { v128_load(v.as_ptr().add((head * tokens + token) * width + col).cast()) };
                    sum = f32x4_add(sum, f32x4_mul(weight, value));
                }
                unsafe { v128_store(op.add(col).cast(), sum) };
            }
        }
    }
}
