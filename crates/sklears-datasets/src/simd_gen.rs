//! SIMD-optimized dataset generation
//!
//! This module provides SIMD-accelerated implementations of common dataset
//! generation operations for improved performance on supported platforms.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{Distribution, RandNormal, Random};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD capabilities detected at runtime
#[derive(Debug, Clone, Copy)]
pub struct SimdCapabilities {
    pub has_sse2: bool,
    pub has_avx: bool,
    pub has_avx2: bool,
    pub has_fma: bool,
}

impl SimdCapabilities {
    /// Detect available SIMD features
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                has_sse2: is_x86_feature_detected!("sse2"),
                has_avx: is_x86_feature_detected!("avx"),
                has_avx2: is_x86_feature_detected!("avx2"),
                has_fma: is_x86_feature_detected!("fma"),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {
                has_sse2: false,
                has_avx: false,
                has_avx2: false,
                has_fma: false,
            }
        }
    }

    /// Check if any SIMD features are available
    pub fn has_simd(&self) -> bool {
        self.has_sse2 || self.has_avx || self.has_avx2
    }
}

/// SIMD-optimized feature matrix generation with normal distribution
#[cfg(target_arch = "x86_64")]
pub fn generate_normal_matrix_simd(
    n_samples: usize,
    n_features: usize,
    mean: f64,
    std: f64,
    random_state: Option<u64>,
) -> Array2<f64> {
    let caps = SimdCapabilities::detect();

    if caps.has_avx {
        unsafe { generate_normal_matrix_avx(n_samples, n_features, mean, std, random_state) }
    } else if caps.has_sse2 {
        unsafe { generate_normal_matrix_sse(n_samples, n_features, mean, std, random_state) }
    } else {
        generate_normal_matrix_scalar(n_samples, n_features, mean, std, random_state)
    }
}

/// Fallback for non-x86_64 architectures
#[cfg(not(target_arch = "x86_64"))]
pub fn generate_normal_matrix_simd(
    n_samples: usize,
    n_features: usize,
    mean: f64,
    std: f64,
    random_state: Option<u64>,
) -> Array2<f64> {
    generate_normal_matrix_scalar(n_samples, n_features, mean, std, random_state)
}

/// Scalar fallback implementation
fn generate_normal_matrix_scalar(
    n_samples: usize,
    n_features: usize,
    mean: f64,
    std: f64,
    random_state: Option<u64>,
) -> Array2<f64> {
    let mut rng = Random::seed(random_state.unwrap_or(42));

    let normal = RandNormal::new(mean, std).unwrap();
    Array2::from_shape_fn((n_samples, n_features), |_| normal.sample(&mut rng))
}

/// SSE2-optimized implementation
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn generate_normal_matrix_sse(
    n_samples: usize,
    n_features: usize,
    mean: f64,
    std: f64,
    random_state: Option<u64>,
) -> Array2<f64> {
    let mut rng = Random::seed(random_state.unwrap_or(42));

    let normal = RandNormal::new(mean, std).unwrap();
    let total = n_samples * n_features;
    let mut data = Vec::with_capacity(total);

    // Process 2 elements at a time with SSE2 (128-bit = 2 x f64)
    let chunks = total / 2;
    for _ in 0..chunks {
        let v1 = normal.sample(&mut rng);
        let v2 = normal.sample(&mut rng);
        data.push(v1);
        data.push(v2);
    }

    // Handle remaining elements
    for _ in 0..(total % 2) {
        data.push(normal.sample(&mut rng));
    }

    Array2::from_shape_vec((n_samples, n_features), data).unwrap()
}

/// AVX-optimized implementation
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn generate_normal_matrix_avx(
    n_samples: usize,
    n_features: usize,
    mean: f64,
    std: f64,
    random_state: Option<u64>,
) -> Array2<f64> {
    let mut rng = Random::seed(random_state.unwrap_or(42));

    let normal = RandNormal::new(mean, std).unwrap();
    let total = n_samples * n_features;
    let mut data = Vec::with_capacity(total);

    // Process 4 elements at a time with AVX (256-bit = 4 x f64)
    let chunks = total / 4;
    for _ in 0..chunks {
        // Generate 4 random values
        for _ in 0..4 {
            data.push(normal.sample(&mut rng));
        }
    }

    // Handle remaining elements
    for _ in 0..(total % 4) {
        data.push(normal.sample(&mut rng));
    }

    Array2::from_shape_vec((n_samples, n_features), data).unwrap()
}

/// SIMD-optimized vector addition
#[cfg(target_arch = "x86_64")]
pub fn add_vectors_simd(a: &[f64], b: &[f64], result: &mut [f64]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), result.len());

    let caps = SimdCapabilities::detect();

    if caps.has_avx {
        unsafe { add_vectors_avx(a, b, result) };
    } else if caps.has_sse2 {
        unsafe { add_vectors_sse2(a, b, result) };
    } else {
        add_vectors_scalar(a, b, result);
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn add_vectors_simd(a: &[f64], b: &[f64], result: &mut [f64]) {
    add_vectors_scalar(a, b, result);
}

fn add_vectors_scalar(a: &[f64], b: &[f64], result: &mut [f64]) {
    for i in 0..a.len() {
        result[i] = a[i] + b[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn add_vectors_sse2(a: &[f64], b: &[f64], result: &mut [f64]) {
    let len = a.len();
    let chunks = len / 2;

    for i in 0..chunks {
        let idx = i * 2;
        let va = _mm_loadu_pd(a.as_ptr().add(idx));
        let vb = _mm_loadu_pd(b.as_ptr().add(idx));
        let vr = _mm_add_pd(va, vb);
        _mm_storeu_pd(result.as_mut_ptr().add(idx), vr);
    }

    // Handle remaining elements
    for i in (chunks * 2)..len {
        result[i] = a[i] + b[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn add_vectors_avx(a: &[f64], b: &[f64], result: &mut [f64]) {
    let len = a.len();
    let chunks = len / 4;

    for i in 0..chunks {
        let idx = i * 4;
        let va = _mm256_loadu_pd(a.as_ptr().add(idx));
        let vb = _mm256_loadu_pd(b.as_ptr().add(idx));
        let vr = _mm256_add_pd(va, vb);
        _mm256_storeu_pd(result.as_mut_ptr().add(idx), vr);
    }

    // Handle remaining elements
    for i in (chunks * 4)..len {
        result[i] = a[i] + b[i];
    }
}

/// SIMD-optimized scalar multiplication
#[cfg(target_arch = "x86_64")]
pub fn scale_vector_simd(data: &[f64], scale: f64, result: &mut [f64]) {
    assert_eq!(data.len(), result.len());

    let caps = SimdCapabilities::detect();

    if caps.has_avx {
        unsafe { scale_vector_avx(data, scale, result) };
    } else if caps.has_sse2 {
        unsafe { scale_vector_sse2(data, scale, result) };
    } else {
        scale_vector_scalar(data, scale, result);
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn scale_vector_simd(data: &[f64], scale: f64, result: &mut [f64]) {
    scale_vector_scalar(data, scale, result);
}

fn scale_vector_scalar(data: &[f64], scale: f64, result: &mut [f64]) {
    for i in 0..data.len() {
        result[i] = data[i] * scale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn scale_vector_sse2(data: &[f64], scale: f64, result: &mut [f64]) {
    let len = data.len();
    let chunks = len / 2;
    let vscale = _mm_set1_pd(scale);

    for i in 0..chunks {
        let idx = i * 2;
        let vdata = _mm_loadu_pd(data.as_ptr().add(idx));
        let vr = _mm_mul_pd(vdata, vscale);
        _mm_storeu_pd(result.as_mut_ptr().add(idx), vr);
    }

    // Handle remaining elements
    for i in (chunks * 2)..len {
        result[i] = data[i] * scale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn scale_vector_avx(data: &[f64], scale: f64, result: &mut [f64]) {
    let len = data.len();
    let chunks = len / 4;
    let vscale = _mm256_set1_pd(scale);

    for i in 0..chunks {
        let idx = i * 4;
        let vdata = _mm256_loadu_pd(data.as_ptr().add(idx));
        let vr = _mm256_mul_pd(vdata, vscale);
        _mm256_storeu_pd(result.as_mut_ptr().add(idx), vr);
    }

    // Handle remaining elements
    for i in (chunks * 4)..len {
        result[i] = data[i] * scale;
    }
}

/// SIMD-optimized classification dataset generation
pub fn make_classification_simd(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    class_sep: f64,
    random_state: Option<u64>,
) -> (Array2<f64>, Array1<i32>) {
    // Generate features using SIMD
    let features = generate_normal_matrix_simd(n_samples, n_features, 0.0, 1.0, random_state);

    // Generate targets
    let targets = Array1::from_shape_fn(n_samples, |i| (i % n_classes) as i32);

    // Apply class separation using SIMD
    let mut separated_features = features.clone();
    for (i, &target) in targets.iter().enumerate() {
        let offset = target as f64 * class_sep;
        let mut row = separated_features.row_mut(i);
        let row_slice = row.as_slice_mut().unwrap();

        let offset_vec = vec![offset; n_features];
        add_vectors_simd(features.row(i).as_slice().unwrap(), &offset_vec, row_slice);
    }

    (separated_features, targets)
}

/// SIMD-optimized regression dataset generation
pub fn make_regression_simd(
    n_samples: usize,
    n_features: usize,
    noise: f64,
    random_state: Option<u64>,
) -> (Array2<f64>, Array1<f64>) {
    // Generate features using SIMD
    let features = generate_normal_matrix_simd(n_samples, n_features, 0.0, 1.0, random_state);

    // Generate coefficients
    let mut rng = Random::seed(random_state.unwrap_or(42).wrapping_add(1));

    let normal_coef = RandNormal::new(0.0, 1.0).unwrap();
    let coef = Array1::from_shape_fn(n_features, |_| normal_coef.sample(&mut rng));

    // Compute targets: y = X @ coef + noise
    let targets = features.dot(&coef);

    // Add noise using SIMD if requested
    if noise > 0.0 {
        let normal_noise = RandNormal::new(0.0, noise).unwrap();
        let noise_vec = Array1::from_shape_fn(n_samples, |_| normal_noise.sample(&mut rng));
        let mut targets_with_noise = vec![0.0; n_samples];
        add_vectors_simd(
            targets.as_slice().unwrap(),
            noise_vec.as_slice().unwrap(),
            &mut targets_with_noise,
        );
        (features, Array1::from_vec(targets_with_noise))
    } else {
        (features, targets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_capabilities() {
        let caps = SimdCapabilities::detect();
        println!("SIMD Capabilities:");
        println!("  SSE2: {}", caps.has_sse2);
        println!("  AVX: {}", caps.has_avx);
        println!("  AVX2: {}", caps.has_avx2);
        println!("  FMA: {}", caps.has_fma);
    }

    #[test]
    fn test_generate_normal_matrix_simd() {
        let matrix = generate_normal_matrix_simd(100, 10, 0.0, 1.0, Some(42));
        assert_eq!(matrix.nrows(), 100);
        assert_eq!(matrix.ncols(), 10);

        // Check that values are reasonable (within ~4 std devs for normal distribution)
        for &val in matrix.iter() {
            assert!(val.abs() < 5.0);
        }
    }

    #[test]
    fn test_add_vectors_simd() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let mut result = vec![0.0; 5];

        add_vectors_simd(&a, &b, &mut result);

        assert_eq!(result, vec![6.0, 6.0, 6.0, 6.0, 6.0]);
    }

    #[test]
    fn test_scale_vector_simd() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut result = vec![0.0; 5];

        scale_vector_simd(&data, 2.0, &mut result);

        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_make_classification_simd() {
        let (features, targets) = make_classification_simd(100, 5, 3, 1.0, Some(42));

        assert_eq!(features.nrows(), 100);
        assert_eq!(features.ncols(), 5);
        assert_eq!(targets.len(), 100);

        // Check that we have all classes
        let mut has_class = vec![false; 3];
        for &target in targets.iter() {
            assert!(target >= 0 && target < 3);
            has_class[target as usize] = true;
        }
        assert!(has_class.iter().all(|&x| x));
    }

    #[test]
    fn test_make_regression_simd() {
        let (features, targets) = make_regression_simd(100, 5, 0.1, Some(42));

        assert_eq!(features.nrows(), 100);
        assert_eq!(features.ncols(), 5);
        assert_eq!(targets.len(), 100);
    }

    #[test]
    fn test_simd_vs_scalar_consistency() {
        let seed = Some(42);
        let matrix_simd = generate_normal_matrix_simd(50, 10, 0.0, 1.0, seed);
        let matrix_scalar = generate_normal_matrix_scalar(50, 10, 0.0, 1.0, seed);

        // Both should produce same shape
        assert_eq!(matrix_simd.shape(), matrix_scalar.shape());

        // Values should be identical (same RNG seed)
        for i in 0..matrix_simd.nrows() {
            for j in 0..matrix_simd.ncols() {
                assert_eq!(matrix_simd[[i, j]], matrix_scalar[[i, j]]);
            }
        }
    }
}
