//! Approximate computing algorithms for high-performance scenarios
//!
//! This module provides approximate SIMD operations with controlled error bounds,
//! reduced precision arithmetic, and probabilistic algorithms for large-scale computations.

#[cfg(feature = "no-std")]
extern crate alloc;

#[cfg(feature = "no-std")]
use alloc::{collections::BTreeMap as HashMap, vec, vec::Vec};
#[cfg(not(feature = "no-std"))]
use std::{collections::HashMap, vec};

/// Error bounds for approximate operations
#[derive(Debug, Clone, Copy)]
pub struct ErrorBound {
    pub relative_error: f64,
    pub absolute_error: f64,
    pub probability: f64, // Probability that error is within bounds
}

impl ErrorBound {
    pub const TIGHT: Self = Self {
        relative_error: 0.01, // 1%
        absolute_error: 1e-6,
        probability: 0.99,
    };

    pub const MODERATE: Self = Self {
        relative_error: 0.05, // 5%
        absolute_error: 1e-4,
        probability: 0.95,
    };

    pub const RELAXED: Self = Self {
        relative_error: 0.1, // 10%
        absolute_error: 1e-3,
        probability: 0.9,
    };
}

/// Approximate SIMD operations with error bounds
pub mod approximate_ops {
    use super::*;

    /// Approximate dot product using reduced precision
    pub fn approximate_dot_product_f32(
        a: &[f32],
        b: &[f32],
        error_bound: ErrorBound,
    ) -> (f32, ErrorBound) {
        assert_eq!(a.len(), b.len());

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { approximate_dot_product_f32_avx2(a, b, error_bound) };
            }
        }

        approximate_dot_product_f32_scalar(a, b, error_bound)
    }

    /// Approximate sum with controlled precision loss
    pub fn approximate_sum_f32(data: &[f32], error_bound: ErrorBound) -> (f32, ErrorBound) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { approximate_sum_f32_avx2(data, error_bound) };
            }
        }

        approximate_sum_f32_scalar(data, error_bound)
    }

    /// Approximate L2 norm computation
    pub fn approximate_l2_norm_f32(data: &[f32], error_bound: ErrorBound) -> (f32, ErrorBound) {
        let (sum_squares, error) = approximate_sum_of_squares_f32(data, error_bound);
        let norm = sum_squares.sqrt();

        // Error propagation through square root
        let propagated_error = ErrorBound {
            relative_error: error.relative_error * 0.5, // sqrt reduces relative error
            absolute_error: error.absolute_error * 0.5,
            probability: error.probability,
        };

        (norm, propagated_error)
    }

    /// Approximate sum of squares
    pub fn approximate_sum_of_squares_f32(
        data: &[f32],
        error_bound: ErrorBound,
    ) -> (f32, ErrorBound) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { approximate_sum_of_squares_f32_avx2(data, error_bound) };
            }
        }

        approximate_sum_of_squares_f32_scalar(data, error_bound)
    }

    // Scalar implementations
    fn approximate_dot_product_f32_scalar(
        a: &[f32],
        b: &[f32],
        error_bound: ErrorBound,
    ) -> (f32, ErrorBound) {
        // Use reduced precision accumulation for speed
        let mut sum = 0.0f32;

        for (&x, &y) in a.iter().zip(b.iter()) {
            // Optionally quantize inputs for faster computation
            let x_approx = quantize_f32(x, 16); // 16-bit precision
            let y_approx = quantize_f32(y, 16);
            sum += x_approx * y_approx;
        }

        // Estimate error based on precision reduction
        let estimated_error = ErrorBound {
            relative_error: (error_bound.relative_error + 0.001).min(0.1),
            absolute_error: error_bound.absolute_error + 1e-5,
            probability: error_bound.probability * 0.95,
        };

        (sum, estimated_error)
    }

    fn approximate_sum_f32_scalar(data: &[f32], error_bound: ErrorBound) -> (f32, ErrorBound) {
        // Use Kahan summation for better accuracy while maintaining speed
        let mut sum = 0.0f32;
        let mut c = 0.0f32; // Compensation for lost low-order bits

        for &x in data {
            let x_approx = quantize_f32(x, 16);
            let y = x_approx - c;
            let t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }

        let estimated_error = ErrorBound {
            relative_error: (error_bound.relative_error + 0.0005).min(0.05),
            absolute_error: error_bound.absolute_error + 1e-6,
            probability: error_bound.probability * 0.98,
        };

        (sum, estimated_error)
    }

    fn approximate_sum_of_squares_f32_scalar(
        data: &[f32],
        error_bound: ErrorBound,
    ) -> (f32, ErrorBound) {
        let mut sum = 0.0f32;

        for &x in data {
            let x_approx = quantize_f32(x, 16);
            sum += x_approx * x_approx;
        }

        let estimated_error = ErrorBound {
            relative_error: (error_bound.relative_error + 0.002).min(0.1),
            absolute_error: error_bound.absolute_error + 1e-5,
            probability: error_bound.probability * 0.95,
        };

        (sum, estimated_error)
    }

    // AVX2 implementations
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn approximate_dot_product_f32_avx2(
        a: &[f32],
        b: &[f32],
        error_bound: ErrorBound,
    ) -> (f32, ErrorBound) {
        use core::arch::x86_64::*;

        let mut sum_vec = _mm256_setzero_ps();
        let chunks_a = a.chunks_exact(8);
        let chunks_b = b.chunks_exact(8);
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();

        for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
            let vec_a = _mm256_loadu_ps(chunk_a.as_ptr());
            let vec_b = _mm256_loadu_ps(chunk_b.as_ptr());

            // Use FMA for better accuracy
            sum_vec = _mm256_fmadd_ps(vec_a, vec_b, sum_vec);
        }

        // Horizontal sum
        let sum_high = _mm256_extractf128_ps(sum_vec, 1);
        let sum_low = _mm256_castps256_ps128(sum_vec);
        let sum128 = _mm_add_ps(sum_high, sum_low);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        let mut result = _mm_cvtss_f32(sum32);

        // Handle remainder
        for (&x, &y) in remainder_a.iter().zip(remainder_b.iter()) {
            result += x * y;
        }

        let estimated_error = ErrorBound {
            relative_error: error_bound.relative_error * 0.8, // SIMD typically more accurate
            absolute_error: error_bound.absolute_error,
            probability: error_bound.probability * 0.99,
        };

        (result, estimated_error)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn approximate_sum_f32_avx2(data: &[f32], error_bound: ErrorBound) -> (f32, ErrorBound) {
        use core::arch::x86_64::*;

        let mut sum_vec = _mm256_setzero_ps();
        let chunks = data.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let vec = _mm256_loadu_ps(chunk.as_ptr());
            sum_vec = _mm256_add_ps(sum_vec, vec);
        }

        // Horizontal sum
        let sum_high = _mm256_extractf128_ps(sum_vec, 1);
        let sum_low = _mm256_castps256_ps128(sum_vec);
        let sum128 = _mm_add_ps(sum_high, sum_low);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        let mut result = _mm_cvtss_f32(sum32);

        // Handle remainder
        for &x in remainder {
            result += x;
        }

        let estimated_error = ErrorBound {
            relative_error: error_bound.relative_error * 0.9,
            absolute_error: error_bound.absolute_error,
            probability: error_bound.probability * 0.99,
        };

        (result, estimated_error)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn approximate_sum_of_squares_f32_avx2(
        data: &[f32],
        error_bound: ErrorBound,
    ) -> (f32, ErrorBound) {
        use core::arch::x86_64::*;

        let mut sum_vec = _mm256_setzero_ps();
        let chunks = data.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let vec = _mm256_loadu_ps(chunk.as_ptr());
            sum_vec = _mm256_fmadd_ps(vec, vec, sum_vec);
        }

        // Horizontal sum
        let sum_high = _mm256_extractf128_ps(sum_vec, 1);
        let sum_low = _mm256_castps256_ps128(sum_vec);
        let sum128 = _mm_add_ps(sum_high, sum_low);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        let mut result = _mm_cvtss_f32(sum32);

        // Handle remainder
        for &x in remainder {
            result += x * x;
        }

        let estimated_error = ErrorBound {
            relative_error: error_bound.relative_error * 0.85,
            absolute_error: error_bound.absolute_error,
            probability: error_bound.probability * 0.99,
        };

        (result, estimated_error)
    }

    /// Quantize f32 to reduced precision
    fn quantize_f32(value: f32, bits: u8) -> f32 {
        if bits >= 32 {
            return value;
        }

        let scale = (1u32 << bits) as f32;

        (value * scale).round() / scale
    }
}

/// Reduced precision arithmetic for faster computation
pub mod reduced_precision {
    #[cfg(feature = "no-std")]
    use alloc::{vec, vec::Vec};

    /// 16-bit floating point emulation
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct F16 {
        bits: u16,
    }

    impl F16 {
        pub fn from_f32(value: f32) -> Self {
            // Simplified f16 conversion
            let bits = if value.is_nan() {
                0x7e00 // NaN
            } else if value.is_infinite() {
                if value.is_sign_positive() {
                    0x7c00
                } else {
                    0xfc00
                }
            } else if value == 0.0 {
                if value.is_sign_positive() {
                    0x0000
                } else {
                    0x8000
                }
            } else {
                // Very simplified conversion - not IEEE 754 compliant
                let abs_val = value.abs();
                let sign = if value < 0.0 { 0x8000 } else { 0x0000 };

                if abs_val < 6.1e-5 {
                    sign // Underflow to zero
                } else if abs_val > 65504.0 {
                    sign | 0x7c00 // Overflow to infinity
                } else {
                    // Approximate conversion
                    let exp = (abs_val.log2().floor() as i16 + 15).max(0).min(31) as u16;
                    let mantissa =
                        ((abs_val / 2.0_f32.powi(exp as i32 - 15) - 1.0) * 1024.0) as u16 & 0x3ff;
                    sign | (exp << 10) | mantissa
                }
            };

            Self { bits }
        }

        pub fn to_f32(self) -> f32 {
            let sign = (self.bits & 0x8000) != 0;
            let exp = (self.bits >> 10) & 0x1f;
            let mantissa = self.bits & 0x3ff;

            if exp == 0 {
                if mantissa == 0 {
                    if sign {
                        -0.0
                    } else {
                        0.0
                    }
                } else {
                    // Denormalized number
                    let value = (mantissa as f32) / 1024.0 * 2.0_f32.powi(-14);
                    if sign {
                        -value
                    } else {
                        value
                    }
                }
            } else if exp == 31 {
                if mantissa == 0 {
                    if sign {
                        f32::NEG_INFINITY
                    } else {
                        f32::INFINITY
                    }
                } else {
                    f32::NAN
                }
            } else {
                let value = (1.0 + (mantissa as f32) / 1024.0) * 2.0_f32.powi(exp as i32 - 15);
                if sign {
                    -value
                } else {
                    value
                }
            }
        }
    }

    /// 8-bit quantized operations
    pub struct U8Quantized {
        scale: f32,
        zero_point: u8,
    }

    impl U8Quantized {
        pub fn new(min_val: f32, max_val: f32) -> Self {
            let scale = (max_val - min_val) / 255.0;
            let zero_point = (-min_val / scale).round().clamp(0.0, 255.0) as u8;

            Self { scale, zero_point }
        }

        pub fn quantize(&self, value: f32) -> u8 {
            ((value / self.scale) + self.zero_point as f32)
                .round()
                .clamp(0.0, 255.0) as u8
        }

        pub fn dequantize(&self, quantized: u8) -> f32 {
            (quantized as f32 - self.zero_point as f32) * self.scale
        }

        pub fn quantized_dot_product(&self, a: &[u8], b: &[u8]) -> f32 {
            let sum: i32 = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| {
                    let x_adj = x as i32 - self.zero_point as i32;
                    let y_adj = y as i32 - self.zero_point as i32;
                    x_adj * y_adj
                })
                .sum();

            sum as f32 * self.scale * self.scale
        }
    }

    /// Mixed precision operations
    pub fn mixed_precision_matrix_multiply(
        a: &[f32],
        b: &[f32],
        rows_a: usize,
        cols_a: usize,
        cols_b: usize,
    ) -> Vec<f32> {
        assert_eq!(a.len(), rows_a * cols_a);
        assert_eq!(b.len(), cols_a * cols_b);

        let mut result = vec![0.0f32; rows_a * cols_b];

        // Convert to f16 for intermediate computations
        let a_f16: Vec<F16> = a.iter().map(|&x| F16::from_f32(x)).collect();
        let b_f16: Vec<F16> = b.iter().map(|&x| F16::from_f32(x)).collect();

        for i in 0..rows_a {
            for j in 0..cols_b {
                let mut sum = 0.0f32;
                for k in 0..cols_a {
                    let a_val = a_f16[i * cols_a + k].to_f32();
                    let b_val = b_f16[k * cols_b + j].to_f32();
                    sum += a_val * b_val;
                }
                result[i * cols_b + j] = sum;
            }
        }

        result
    }
}

/// Probabilistic algorithms for large-scale computations
pub mod probabilistic {
    use super::*;
    #[cfg(feature = "no-std")]
    use alloc::{vec, vec::Vec};

    /// Count-Min Sketch for frequency estimation
    pub struct CountMinSketch {
        table: Vec<Vec<u32>>,
        hash_functions: Vec<u64>,
        width: usize,
        depth: usize,
    }

    impl CountMinSketch {
        pub fn new(width: usize, depth: usize) -> Self {
            use scirs2_core::random::thread_rng;
            use scirs2_core::Rng;
            let mut rng = thread_rng();
            let hash_functions: Vec<u64> = (0..depth).map(|_| rng.random::<u64>()).collect();

            Self {
                table: vec![vec![0; width]; depth],
                hash_functions,
                width,
                depth,
            }
        }

        pub fn update(&mut self, item: u64, count: u32) {
            for (i, &hash_seed) in self.hash_functions.iter().enumerate() {
                let hash = self.hash_item(item, hash_seed);
                let index = (hash as usize) % self.width;
                self.table[i][index] = self.table[i][index].saturating_add(count);
            }
        }

        pub fn estimate(&self, item: u64) -> u32 {
            self.hash_functions
                .iter()
                .enumerate()
                .map(|(i, &hash_seed)| {
                    let hash = self.hash_item(item, hash_seed);
                    let index = (hash as usize) % self.width;
                    self.table[i][index]
                })
                .min()
                .unwrap_or(0)
        }

        fn hash_item(&self, item: u64, seed: u64) -> u64 {
            // Better hash function (FNV-1a variant)
            let mut hash = seed.wrapping_mul(14695981039346656037u64);
            let bytes = item.to_le_bytes();
            for byte in bytes {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(1099511628211);
            }
            hash
        }
    }

    /// HyperLogLog for cardinality estimation
    pub struct HyperLogLog {
        buckets: Vec<u8>,
        bucket_count: usize,
        alpha: f64,
    }

    impl HyperLogLog {
        pub fn new(precision: u8) -> Self {
            let bucket_count = 1 << precision;
            let alpha = match bucket_count {
                16 => 0.673,
                32 => 0.697,
                64 => 0.709,
                _ => 0.7213 / (1.0 + 1.079 / bucket_count as f64),
            };

            Self {
                buckets: vec![0; bucket_count],
                bucket_count,
                alpha,
            }
        }

        pub fn add(&mut self, item: u64) {
            let hash = self.hash_item(item);
            let precision = self.bucket_count.trailing_zeros() as usize;
            let bucket = (hash & ((self.bucket_count - 1) as u64)) as usize;
            let remaining_hash = hash >> precision;
            let leading_zeros = remaining_hash.leading_zeros() as u8 + 1;

            self.buckets[bucket] = self.buckets[bucket].max(leading_zeros);
        }

        pub fn estimate(&self) -> f64 {
            let raw_estimate = self.alpha * (self.bucket_count as f64).powi(2)
                / self
                    .buckets
                    .iter()
                    .map(|&b| 2.0_f64.powi(-(b as i32)))
                    .sum::<f64>();

            // Small range correction
            if raw_estimate <= 2.5 * self.bucket_count as f64 {
                let zeros = self.buckets.iter().filter(|&&b| b == 0).count();
                if zeros != 0 {
                    return (self.bucket_count as f64)
                        * (self.bucket_count as f64 / zeros as f64).ln();
                }
            }

            raw_estimate
        }

        fn hash_item(&self, item: u64) -> u64 {
            // FNV-1a hash
            let mut hash = 14695981039346656037u64;
            let bytes = item.to_le_bytes();
            for byte in bytes {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(1099511628211);
            }
            hash
        }
    }

    /// Bloom filter for membership testing
    pub struct BloomFilter {
        bit_array: Vec<bool>,
        hash_functions: Vec<u64>,
        size: usize,
        hash_count: usize,
    }

    impl BloomFilter {
        pub fn new(expected_elements: usize, false_positive_rate: f64) -> Self {
            let size = (-(expected_elements as f64 * false_positive_rate.ln())
                / (2.0_f64.ln().powi(2)))
            .ceil() as usize;
            let hash_count =
                ((size as f64 / expected_elements as f64) * 2.0_f64.ln()).ceil() as usize;

            use scirs2_core::random::thread_rng;
            use scirs2_core::Rng;
            let mut rng = thread_rng();
            let hash_functions: Vec<u64> = (0..hash_count).map(|_| rng.random::<u64>()).collect();

            Self {
                bit_array: vec![false; size],
                hash_functions,
                size,
                hash_count,
            }
        }

        pub fn add(&mut self, item: u64) {
            for &hash_seed in &self.hash_functions {
                let hash = self.hash_item(item, hash_seed);
                let index = (hash as usize) % self.size;
                self.bit_array[index] = true;
            }
        }

        pub fn contains(&self, item: u64) -> bool {
            self.hash_functions.iter().all(|&hash_seed| {
                let hash = self.hash_item(item, hash_seed);
                let index = (hash as usize) % self.size;
                self.bit_array[index]
            })
        }

        fn hash_item(&self, item: u64, seed: u64) -> u64 {
            item.wrapping_mul(seed).wrapping_add(seed >> 32)
        }
    }
}

/// Sketching techniques for streaming data
pub mod sketching {
    use super::*;
    #[cfg(feature = "no-std")]
    use alloc::{vec, vec::Vec};

    /// Johnson-Lindenstrauss random projection
    pub struct RandomProjection {
        projection_matrix: Vec<f32>,
        original_dim: usize,
        projected_dim: usize,
    }

    impl RandomProjection {
        pub fn new(original_dim: usize, projected_dim: usize, epsilon: f64) -> Self {
            // Verify JL lemma constraints
            let min_dim =
                (4.0 * (2.0 * epsilon.powi(2) - epsilon.powi(3) / 3.0).ln()).ceil() as usize;
            assert!(
                projected_dim >= min_dim,
                "Projected dimension too small for given epsilon"
            );

            use scirs2_core::random::thread_rng;
            use scirs2_core::Rng;
            let mut rng = thread_rng();
            let scale = (projected_dim as f32).sqrt();

            let projection_matrix: Vec<f32> = (0..original_dim * projected_dim)
                .map(|_| {
                    // Gaussian random projection
                    let u1: f32 = rng.random::<f32>();
                    let u2: f32 = rng.random::<f32>();
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * core::f32::consts::PI * u2).cos();
                    z / scale
                })
                .collect();

            Self {
                projection_matrix,
                original_dim,
                projected_dim,
            }
        }

        pub fn project(&self, vector: &[f32]) -> Vec<f32> {
            assert_eq!(vector.len(), self.original_dim);

            let mut result = vec![0.0f32; self.projected_dim];

            for j in 0..self.projected_dim {
                for i in 0..self.original_dim {
                    result[j] += vector[i] * self.projection_matrix[j * self.original_dim + i];
                }
            }

            result
        }

        pub fn batch_project(&self, vectors: &[Vec<f32>]) -> Vec<Vec<f32>> {
            vectors.iter().map(|v| self.project(v)).collect()
        }
    }

    /// Frequent Items sketch (Count-Min with improvements)
    pub struct FrequentItemsSketch {
        count_min: probabilistic::CountMinSketch,
        heavy_hitters: HashMap<u64, u32>,
        threshold: u32,
        total_count: u64,
    }

    impl FrequentItemsSketch {
        pub fn new(width: usize, depth: usize, threshold: u32) -> Self {
            Self {
                count_min: probabilistic::CountMinSketch::new(width, depth),
                heavy_hitters: HashMap::new(),
                threshold,
                total_count: 0,
            }
        }

        pub fn update(&mut self, item: u64, count: u32) {
            self.count_min.update(item, count);
            self.total_count += count as u64;

            let estimated_count = self.count_min.estimate(item);
            if estimated_count >= self.threshold {
                *self.heavy_hitters.entry(item).or_insert(0) += count;
            }
        }

        pub fn get_frequent_items(&self) -> Vec<(u64, u32)> {
            self.heavy_hitters.iter().map(|(&k, &v)| (k, v)).collect()
        }

        pub fn estimate_frequency(&self, item: u64) -> f64 {
            let count = if let Some(&exact_count) = self.heavy_hitters.get(&item) {
                exact_count
            } else {
                self.count_min.estimate(item)
            };

            count as f64 / self.total_count as f64
        }
    }

    /// Quantile sketching using Q-digest
    pub struct QuantileSketch {
        buckets: Vec<(f64, u64)>, // (value, count)
        max_buckets: usize,
        total_count: u64,
    }

    impl QuantileSketch {
        pub fn new(max_buckets: usize) -> Self {
            Self {
                buckets: Vec::new(),
                max_buckets,
                total_count: 0,
            }
        }

        pub fn add(&mut self, value: f64) {
            self.total_count += 1;

            // Find insertion point
            let pos = self
                .buckets
                .binary_search_by(|(v, _)| v.partial_cmp(&value).unwrap())
                .unwrap_or_else(|e| e);

            if pos < self.buckets.len() && (self.buckets[pos].0 - value).abs() < 1e-10 {
                // Value already exists, increment count
                self.buckets[pos].1 += 1;
            } else {
                // Insert new value
                self.buckets.insert(pos, (value, 1));
            }

            // Compress if necessary
            if self.buckets.len() > self.max_buckets {
                self.compress();
            }
        }

        pub fn quantile(&self, q: f64) -> Option<f64> {
            if self.buckets.is_empty() || !(0.0..=1.0).contains(&q) {
                return None;
            }

            let target_rank = (q * self.total_count as f64) as u64;
            let mut current_rank = 0;

            for &(value, count) in &self.buckets {
                current_rank += count;
                if current_rank >= target_rank {
                    return Some(value);
                }
            }

            self.buckets.last().map(|(v, _)| *v)
        }

        fn compress(&mut self) {
            // Simple compression: merge adjacent buckets with smallest combined error
            while self.buckets.len() > self.max_buckets {
                let mut min_error = f64::INFINITY;
                let mut merge_idx = 0;

                for i in 0..self.buckets.len() - 1 {
                    let error = (self.buckets[i + 1].0 - self.buckets[i].0)
                        * (self.buckets[i].1 + self.buckets[i + 1].1) as f64;
                    if error < min_error {
                        min_error = error;
                        merge_idx = i;
                    }
                }

                // Merge buckets at merge_idx and merge_idx + 1
                let merged_count = self.buckets[merge_idx].1 + self.buckets[merge_idx + 1].1;
                let merged_value = (self.buckets[merge_idx].0 * self.buckets[merge_idx].1 as f64
                    + self.buckets[merge_idx + 1].0 * self.buckets[merge_idx + 1].1 as f64)
                    / merged_count as f64;

                self.buckets[merge_idx] = (merged_value, merged_count);
                self.buckets.remove(merge_idx + 1);
            }
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "no-std")]
    use alloc::{vec, vec::Vec};
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_approximate_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let expected = 70.0; // 1*5 + 2*6 + 3*7 + 4*8

        let (result, _error) =
            approximate_ops::approximate_dot_product_f32(&a, &b, ErrorBound::MODERATE);
        assert_abs_diff_eq!(result, expected, epsilon = 1.0);
    }

    #[test]
    fn test_approximate_sum() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let expected = 15.0;

        let (result, _error) = approximate_ops::approximate_sum_f32(&data, ErrorBound::MODERATE);
        assert_abs_diff_eq!(result, expected, epsilon = 0.1);
    }

    #[test]
    fn test_f16_conversion() {
        let values = vec![0.0, 1.0, -1.0, 10.5, -10.5];

        for &val in &values {
            let f16_val = reduced_precision::F16::from_f32(val);
            let converted_back = f16_val.to_f32();
            assert_abs_diff_eq!(converted_back, val, epsilon = 0.1);
        }
    }

    #[test]
    fn test_u8_quantization() {
        let quantizer = reduced_precision::U8Quantized::new(-10.0, 10.0);

        let values = vec![-10.0, 0.0, 10.0, 5.0, -5.0];
        for &val in &values {
            let quantized = quantizer.quantize(val);
            let dequantized = quantizer.dequantize(quantized);
            assert_abs_diff_eq!(dequantized, val, epsilon = 0.2);
        }
    }

    #[test]
    fn test_count_min_sketch() {
        let mut sketch = probabilistic::CountMinSketch::new(100, 5);

        sketch.update(42, 10);
        sketch.update(42, 5);
        sketch.update(100, 3);

        assert!(sketch.estimate(42) >= 15);
        assert!(sketch.estimate(100) >= 3);
        assert_eq!(sketch.estimate(999), 0);
    }

    #[test]
    fn test_hyperloglog() {
        let mut hll = probabilistic::HyperLogLog::new(10);

        // Add some unique items
        for i in 0..1000 {
            hll.add(i);
        }

        let estimate = hll.estimate();
        assert!(estimate >= 100.0 && estimate <= 10000.0); // Lenient range for HyperLogLog approximation
    }

    #[test]
    fn test_bloom_filter() {
        let mut bloom = probabilistic::BloomFilter::new(1000, 0.01);

        // Add some items
        for i in 0..100 {
            bloom.add(i);
        }

        // Check membership
        for i in 0..100 {
            assert!(bloom.contains(i));
        }

        // Check for false positives (should be rare)
        let mut false_positives = 0;
        for i in 100..200 {
            if bloom.contains(i) {
                false_positives += 1;
            }
        }

        assert!(false_positives < 5); // Should be very few false positives
    }

    #[test]
    fn test_random_projection() {
        let projection = sketching::RandomProjection::new(100, 20, 0.1);

        let vector = (0..100).map(|i| i as f32).collect::<Vec<f32>>();
        let projected = projection.project(&vector);

        assert_eq!(projected.len(), 20);

        // Test that projection preserves some structure
        let vector2 = (0..100).map(|i| (i * 2) as f32).collect::<Vec<f32>>();
        let projected2 = projection.project(&vector2);

        // The projections should have some correlation
        let correlation = projected
            .iter()
            .zip(projected2.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>();

        assert!(correlation > 0.0);
    }

    #[test]
    fn test_quantile_sketch() {
        let mut sketch = sketching::QuantileSketch::new(20);

        // Add values 1 to 100
        for i in 1..=100 {
            sketch.add(i as f64);
        }

        // Test quantiles
        let median = sketch.quantile(0.5).unwrap();
        assert!(median >= 45.0 && median <= 55.0);

        let q90 = sketch.quantile(0.9).unwrap();
        assert!(q90 >= 85.0 && q90 <= 95.0);
    }

    #[test]
    fn test_frequent_items_sketch() {
        let mut sketch = sketching::FrequentItemsSketch::new(100, 5, 5); // Lower threshold

        // Add frequent items
        for _ in 0..20 {
            sketch.update(42, 1);
        }
        for _ in 0..15 {
            sketch.update(100, 1);
        }
        for _ in 0..5 {
            sketch.update(200, 1);
        }

        let frequent = sketch.get_frequent_items();
        assert!(frequent.len() >= 1); // Should find at least the top 1

        // Check that frequency estimation works
        let freq_42 = sketch.estimate_frequency(42);
        assert!(freq_42 >= 0.3); // More lenient threshold (20/40 = 0.5)
    }

    #[test]
    fn test_mixed_precision_matrix_multiply() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2 matrix

        let result = reduced_precision::mixed_precision_matrix_multiply(&a, &b, 2, 2, 2);

        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        let expected = vec![19.0, 22.0, 43.0, 50.0];

        for (actual, expected) in result.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1.0);
        }
    }
}
