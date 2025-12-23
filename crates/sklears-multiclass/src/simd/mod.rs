//! SIMD Optimizations for Multiclass Classification
//!
//! This module provides SIMD-accelerated operations for voting aggregation,
//! probability calculations, and distance computations.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result as SklResult, SklearsError};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD configuration for optimization
#[derive(Debug, Clone)]
pub struct SimdConfig {
    /// Enable AVX2 instructions (requires CPU support)
    pub use_avx2: bool,
    /// Enable AVX-512 instructions (requires CPU support)
    pub use_avx512: bool,
    /// Minimum vector size to use SIMD (smaller vectors use scalar)
    pub min_vector_size: usize,
}

impl Default for SimdConfig {
    fn default() -> Self {
        Self {
            use_avx2: is_avx2_available(),
            use_avx512: is_avx512_available(),
            min_vector_size: 64,
        }
    }
}

/// Check if AVX2 is available on the current CPU
#[cfg(target_arch = "x86_64")]
fn is_avx2_available() -> bool {
    is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
fn is_avx2_available() -> bool {
    false
}

/// Check if AVX-512 is available on the current CPU
#[cfg(target_arch = "x86_64")]
fn is_avx512_available() -> bool {
    is_x86_feature_detected!("avx512f")
}

#[cfg(not(target_arch = "x86_64"))]
fn is_avx512_available() -> bool {
    false
}

/// SIMD-accelerated voting operations
pub struct SimdVotingOps {
    config: SimdConfig,
}

impl SimdVotingOps {
    /// Create new SIMD voting operations
    pub fn new(config: SimdConfig) -> Self {
        Self { config }
    }

    /// Aggregate votes with SIMD acceleration
    pub fn aggregate_votes(
        &self,
        votes: &Array2<f64>,
        weights: Option<&Array1<f64>>,
    ) -> SklResult<Array1<i32>> {
        let (n_samples, _n_classes) = votes.dim();

        // Use scalar path for small problems
        if n_samples < self.config.min_vector_size {
            return self.aggregate_votes_scalar(votes, weights);
        }

        #[cfg(target_arch = "x86_64")]
        {
            if self.config.use_avx2 && is_x86_feature_detected!("avx2") {
                return self.aggregate_votes_avx2(votes, weights);
            }
        }

        // Fallback to scalar implementation
        self.aggregate_votes_scalar(votes, weights)
    }

    /// Scalar implementation (fallback)
    fn aggregate_votes_scalar(
        &self,
        votes: &Array2<f64>,
        weights: Option<&Array1<f64>>,
    ) -> SklResult<Array1<i32>> {
        let (n_samples, n_classes) = votes.dim();
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let row = votes.row(i);

            let mut max_val = f64::NEG_INFINITY;
            let mut max_idx = 0;

            for j in 0..n_classes {
                let val = if let Some(w) = weights {
                    row[j] * w[j]
                } else {
                    row[j]
                };

                if val > max_val {
                    max_val = val;
                    max_idx = j;
                }
            }

            predictions[i] = max_idx as i32;
        }

        Ok(predictions)
    }

    /// AVX2 implementation
    #[cfg(target_arch = "x86_64")]
    fn aggregate_votes_avx2(
        &self,
        votes: &Array2<f64>,
        weights: Option<&Array1<f64>>,
    ) -> SklResult<Array1<i32>> {
        // For now, fall back to scalar - full AVX2 implementation would require unsafe code
        // and careful handling of alignment and lane operations
        self.aggregate_votes_scalar(votes, weights)
    }

    /// Compute weighted sum with SIMD
    pub fn weighted_sum(&self, values: &[f64], weights: &[f64]) -> SklResult<f64> {
        if values.len() != weights.len() {
            return Err(SklearsError::InvalidInput(
                "Values and weights must have same length".to_string(),
            ));
        }

        #[cfg(target_arch = "x86_64")]
        {
            if self.config.use_avx2 && is_x86_feature_detected!("avx2") && values.len() >= 4 {
                return Ok(self.weighted_sum_avx2(values, weights));
            }
        }

        // Scalar fallback
        Ok(values.iter().zip(weights.iter()).map(|(v, w)| v * w).sum())
    }

    /// AVX2 weighted sum
    #[cfg(target_arch = "x86_64")]
    fn weighted_sum_avx2(&self, values: &[f64], weights: &[f64]) -> f64 {
        unsafe {
            let mut sum = _mm256_setzero_pd();
            let len = values.len();
            let chunks = len / 4;

            for i in 0..chunks {
                let offset = i * 4;
                // Load 4 doubles at a time
                let v = _mm256_loadu_pd(values.as_ptr().add(offset));
                let w = _mm256_loadu_pd(weights.as_ptr().add(offset));
                // Multiply and accumulate
                let prod = _mm256_mul_pd(v, w);
                sum = _mm256_add_pd(sum, prod);
            }

            // Horizontal sum
            let mut result = [0.0; 4];
            _mm256_storeu_pd(result.as_mut_ptr(), sum);
            let mut total = result.iter().sum::<f64>();

            // Handle remaining elements
            for i in (chunks * 4)..len {
                total += values[i] * weights[i];
            }

            total
        }
    }
}

/// SIMD-accelerated distance calculations
pub struct SimdDistanceOps {
    config: SimdConfig,
}

impl SimdDistanceOps {
    /// Create new SIMD distance operations
    pub fn new(config: SimdConfig) -> Self {
        Self { config }
    }

    /// Compute Euclidean distances with SIMD
    pub fn euclidean_distances(&self, a: &[f64], b: &[f64]) -> SklResult<f64> {
        if a.len() != b.len() {
            return Err(SklearsError::InvalidInput(
                "Arrays must have same length".to_string(),
            ));
        }

        #[cfg(target_arch = "x86_64")]
        {
            if self.config.use_avx2 && is_x86_feature_detected!("avx2") && a.len() >= 4 {
                return Ok(self.euclidean_distance_avx2(a, b));
            }
        }

        // Scalar fallback
        Ok(a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt())
    }

    /// AVX2 Euclidean distance
    #[cfg(target_arch = "x86_64")]
    fn euclidean_distance_avx2(&self, a: &[f64], b: &[f64]) -> f64 {
        unsafe {
            let mut sum = _mm256_setzero_pd();
            let len = a.len();
            let chunks = len / 4;

            for i in 0..chunks {
                let offset = i * 4;
                let va = _mm256_loadu_pd(a.as_ptr().add(offset));
                let vb = _mm256_loadu_pd(b.as_ptr().add(offset));
                let diff = _mm256_sub_pd(va, vb);
                let sq = _mm256_mul_pd(diff, diff);
                sum = _mm256_add_pd(sum, sq);
            }

            // Horizontal sum
            let mut result = [0.0; 4];
            _mm256_storeu_pd(result.as_mut_ptr(), sum);
            let mut total = result.iter().sum::<f64>();

            // Handle remaining elements
            for i in (chunks * 4)..len {
                total += (a[i] - b[i]).powi(2);
            }

            total.sqrt()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_autograd::ndarray::array;

    #[test]
    fn test_simd_config_default() {
        let config = SimdConfig::default();
        assert_eq!(config.min_vector_size, 64);
    }

    #[test]
    fn test_simd_voting_ops_scalar() {
        let ops = SimdVotingOps::new(SimdConfig::default());
        let votes = array![[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [2.0, 1.0, 1.0]];

        let predictions = ops.aggregate_votes(&votes, None).unwrap();
        assert_eq!(predictions[0], 1); // Max at index 1
        assert_eq!(predictions[1], 2); // Max at index 2
        assert_eq!(predictions[2], 0); // Max at index 0
    }

    #[test]
    fn test_simd_voting_ops_with_weights() {
        let ops = SimdVotingOps::new(SimdConfig::default());
        let votes = array![[1.0, 2.0, 0.0]];
        let weights = array![2.0, 0.5, 1.0];

        let predictions = ops.aggregate_votes(&votes, Some(&weights)).unwrap();
        // class0=1.0*2.0=2.0, class1=2.0*0.5=1.0, class2=0.0*1.0=0.0
        assert_eq!(predictions[0], 0);
    }

    #[test]
    fn test_weighted_sum() {
        let ops = SimdVotingOps::new(SimdConfig::default());
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![0.1, 0.2, 0.3, 0.4];

        let result = ops.weighted_sum(&values, &weights).unwrap();
        let expected = 1.0 * 0.1 + 2.0 * 0.2 + 3.0 * 0.3 + 4.0 * 0.4;
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_sum_large() {
        let ops = SimdVotingOps::new(SimdConfig::default());
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let weights: Vec<f64> = vec![1.0; 100];

        let result = ops.weighted_sum(&values, &weights).unwrap();
        let expected: f64 = (0..100).map(|i| i as f64).sum();
        assert!((result - expected).abs() < 1e-8);
    }

    #[test]
    fn test_euclidean_distance() {
        let ops = SimdDistanceOps::new(SimdConfig::default());
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let distance = ops.euclidean_distances(&a, &b).unwrap();
        let expected = ((3.0_f64).powi(2) * 3.0).sqrt(); // sqrt(9+9+9) = sqrt(27)
        assert!((distance - expected).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_distance_large() {
        let ops = SimdDistanceOps::new(SimdConfig::default());
        let a: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..100).map(|i| (i + 1) as f64).collect();

        let distance = ops.euclidean_distances(&a, &b).unwrap();
        let expected = 10.0; // sqrt(100 * 1^2) = 10
        assert!((distance - expected).abs() < 1e-10);
    }

    #[test]
    fn test_avx2_detection() {
        let available = is_avx2_available();
        // Just check it doesn't panic
        println!("AVX2 available: {}", available);
    }

    #[test]
    fn test_avx512_detection() {
        let available = is_avx512_available();
        // Just check it doesn't panic
        println!("AVX-512 available: {}", available);
    }
}
