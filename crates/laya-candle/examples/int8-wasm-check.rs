//! Runs the actual target-specific Int8Matrix kernel against independent i64 sums.
use candle_core::{Device, Tensor};
use laya_candle::int8::Int8Matrix;

#[no_mangle]
pub extern "C" fn check() -> u32 {
    let mut cases = 0;
    for cols in [1, 15, 16, 17, 31, 32, 33, 1024, 2624, 16384] {
        for (tokens, rows) in [(3, 3), (4, 4), (5, 7), (8, 9)] {
            for pattern in 0..4 {
                let weights: Vec<i8> = (0..rows * cols).map(|i| match pattern {
                    0 => ((i * 37 + 11) % 256) as u8 as i8,
                    1 => -128,
                    2 => 127,
                    _ => 0,
                }).collect();
                let input: Vec<f32> = (0..tokens * cols).map(|i| match pattern {
                    0 => ((i * 19 % 255) as i32 - 127) as f32,
                    1 => -127.,
                    2 => 127.,
                    _ => 0.,
                }).collect();
                let mut bytes: Vec<u8> = weights.iter().map(|v| *v as u8).collect();
                for _ in 0..rows { bytes.extend(0.25f32.to_le_bytes()); }
                let matrix = Int8Matrix::from_bytes(&bytes, rows, cols).unwrap();
                let tensor = Tensor::from_vec(input.clone(), (tokens, cols), &Device::Cpu).unwrap();
                let actual = matrix.forward(&tensor).unwrap().to_vec2::<f32>().unwrap();
                for t in 0..tokens {
                    let row = &input[t * cols..(t + 1) * cols];
                    let peak = row.iter().fold(0f32, |a, b| a.max(b.abs()));
                    let scale = if peak == 0. { 1. } else { peak / 127. };
                    for r in 0..rows {
                        let sum: i64 = row.iter().zip(&weights[r * cols..(r + 1) * cols])
                            .map(|(a, b)| (a / scale).round().clamp(-127., 127.) as i64 * *b as i64).sum();
                        assert_eq!(actual[t][r], sum as f32 * scale * 0.25,
                            "T={tokens} O={rows} K={cols} pattern={pattern}");
                    }
                }
                cases += 1;
            }
        }
    }
    cases
}
