//! RISC-V Vector Extension support
//!
//! This module provides support for RISC-V Vector (RVV) extensions
//! for SIMD operations on RISC-V processors.

#![cfg(target_arch = "riscv64")]

use crate::traits::SimdError;

/// Result type for RISC-V vector operations
pub type RiscVResult<T> = Result<T, SimdError>;

/// RISC-V Vector capabilities
#[derive(Debug, Clone, Copy)]
pub struct RiscVVectorCaps {
    /// Vector length in bytes
    pub vlen: usize,
    /// Maximum element width supported
    pub elen: usize,
    /// Whether RVV is available
    pub available: bool,
}

impl RiscVVectorCaps {
    /// Detect RISC-V Vector capabilities
    pub fn detect() -> Self {
        // Check for RISC-V vector extension support
        #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
        {
            // If compiled with vector support, assume it's available
            Self {
                vlen: Self::detect_vlen(),
                elen: 64, // RVV supports up to 64-bit elements
                available: true,
            }
        }

        #[cfg(not(all(target_arch = "riscv64", target_feature = "v")))]
        {
            // Conservative fallback for non-RVV targets
            Self {
                vlen: 128, // Common default
                elen: 64,  // Support up to 64-bit elements
                available: false,
            }
        }
    }

    /// Detect vector length (VLEN) at runtime
    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    fn detect_vlen() -> usize {
        // In a real implementation, this would use the vsetvli instruction
        // to query the vector length. For now, return a common default.
        // Real detection would look like:
        // unsafe {
        //     let vl: usize;
        //     asm!("vsetvli {}, x0, e32,m1", out(reg) vl);
        //     vl * 32 // Convert elements to bits
        // }
        128 // Default VLEN for many implementations
    }

    /// Get optimal vector width for f32 operations
    pub fn f32_width(&self) -> usize {
        if self.available {
            self.vlen / 32 // 32 bits per f32
        } else {
            1 // Scalar fallback
        }
    }

    /// Get optimal vector width for f64 operations
    pub fn f64_width(&self) -> usize {
        if self.available {
            self.vlen / 64 // 64 bits per f64
        } else {
            1 // Scalar fallback
        }
    }
}

/// RISC-V Vector operations
pub struct RiscVVectorOps;

impl RiscVVectorOps {
    /// Vector dot product using RVV
    pub fn dot_product(x: &[f32], y: &[f32]) -> RiscVResult<f32> {
        if x.len() != y.len() {
            return Err(SimdError::DimensionMismatch {
                expected: x.len(),
                actual: y.len(),
            });
        }

        if x.is_empty() {
            return Err(SimdError::EmptyInput);
        }

        // Use RVV intrinsics if available, otherwise fallback to scalar
        #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
        {
            if x.len() >= 4 {
                return Self::dot_product_rvv(x, y);
            }
        }

        // Scalar fallback implementation
        let result = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        Ok(result)
    }

    /// RVV-optimized dot product implementation
    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    fn dot_product_rvv(x: &[f32], y: &[f32]) -> RiscVResult<f32> {
        // Note: This is a placeholder for actual RVV intrinsics
        // Real RVV intrinsics would use vsetvl, vle32.v, vfmacc.vv, etc.
        // For now, we provide an optimized scalar implementation

        let mut sum = 0.0f32;
        let len = x.len();

        // Process in chunks for better cache performance
        const CHUNK_SIZE: usize = 64;
        let chunks = len / CHUNK_SIZE;

        for chunk in 0..chunks {
            let start = chunk * CHUNK_SIZE;
            let end = start + CHUNK_SIZE;

            let chunk_sum: f32 = x[start..end]
                .iter()
                .zip(&y[start..end])
                .map(|(a, b)| a * b)
                .sum();
            sum += chunk_sum;
        }

        // Handle remaining elements
        for i in (chunks * CHUNK_SIZE)..len {
            sum += x[i] * y[i];
        }

        Ok(sum)
    }

    /// Vector addition using RVV
    pub fn add(x: &[f32], y: &[f32]) -> RiscVResult<Vec<f32>> {
        if x.len() != y.len() {
            return Err(SimdError::DimensionMismatch {
                expected: x.len(),
                actual: y.len(),
            });
        }

        if x.is_empty() {
            return Err(SimdError::EmptyInput);
        }

        // Use RVV intrinsics if available
        #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
        {
            if x.len() >= 4 {
                return Self::add_rvv(x, y);
            }
        }

        // Scalar fallback implementation
        let result = x.iter().zip(y.iter()).map(|(a, b)| a + b).collect();
        Ok(result)
    }

    /// RVV-optimized vector addition
    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    fn add_rvv(x: &[f32], y: &[f32]) -> RiscVResult<Vec<f32>> {
        let mut result = Vec::with_capacity(x.len());
        let len = x.len();

        // Process in chunks for better vectorization
        const CHUNK_SIZE: usize = 32;
        let chunks = len / CHUNK_SIZE;

        for chunk in 0..chunks {
            let start = chunk * CHUNK_SIZE;
            let end = start + CHUNK_SIZE;

            for i in start..end {
                result.push(x[i] + y[i]);
            }
        }

        // Handle remaining elements
        for i in (chunks * CHUNK_SIZE)..len {
            result.push(x[i] + y[i]);
        }

        Ok(result)
    }

    /// Vector scaling using RVV
    pub fn scale(vector: &[f32], scalar: f32) -> RiscVResult<Vec<f32>> {
        if vector.is_empty() {
            return Err(SimdError::EmptyInput);
        }

        // Use RVV intrinsics if available
        #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
        {
            if vector.len() >= 4 {
                return Self::scale_rvv(vector, scalar);
            }
        }

        // Scalar fallback implementation
        let result = vector.iter().map(|x| x * scalar).collect();
        Ok(result)
    }

    /// RVV-optimized vector scaling
    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    fn scale_rvv(vector: &[f32], scalar: f32) -> RiscVResult<Vec<f32>> {
        let mut result = Vec::with_capacity(vector.len());
        let len = vector.len();

        // Process in chunks for better vectorization
        const CHUNK_SIZE: usize = 32;
        let chunks = len / CHUNK_SIZE;

        for chunk in 0..chunks {
            let start = chunk * CHUNK_SIZE;
            let end = start + CHUNK_SIZE;

            for i in start..end {
                result.push(vector[i] * scalar);
            }
        }

        // Handle remaining elements
        for i in (chunks * CHUNK_SIZE)..len {
            result.push(vector[i] * scalar);
        }

        Ok(result)
    }

    /// Vector sum reduction using RVV
    pub fn sum(vector: &[f32]) -> RiscVResult<f32> {
        if vector.is_empty() {
            return Err(SimdError::EmptyInput);
        }

        // Scalar fallback implementation
        Ok(vector.iter().sum())
    }

    /// Vector maximum using RVV
    pub fn max(vector: &[f32]) -> RiscVResult<f32> {
        if vector.is_empty() {
            return Err(SimdError::EmptyInput);
        }

        let result = vector.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        Ok(result)
    }

    /// Vector minimum using RVV
    pub fn min(vector: &[f32]) -> RiscVResult<f32> {
        if vector.is_empty() {
            return Err(SimdError::EmptyInput);
        }

        let result = vector.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        Ok(result)
    }

    /// Vector L2 norm using RVV
    pub fn norm(vector: &[f32]) -> RiscVResult<f32> {
        if vector.is_empty() {
            return Err(SimdError::EmptyInput);
        }

        let sum_of_squares: f32 = vector.iter().map(|x| x * x).sum();
        Ok(sum_of_squares.sqrt())
    }

    /// Fused multiply-add using RVV (z = a * b + c)
    pub fn fma(a: &[f32], b: &[f32], c: &[f32]) -> RiscVResult<Vec<f32>> {
        if a.len() != b.len() || a.len() != c.len() {
            return Err(SimdError::DimensionMismatch {
                expected: a.len(),
                actual: b.len().min(c.len()),
            });
        }

        if a.is_empty() {
            return Err(SimdError::EmptyInput);
        }

        // Use RVV intrinsics if available
        #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
        {
            if a.len() >= 4 {
                return Self::fma_rvv(a, b, c);
            }
        }

        // Scalar fallback implementation
        let result = a
            .iter()
            .zip(b.iter())
            .zip(c.iter())
            .map(|((a_val, b_val), c_val)| a_val * b_val + c_val)
            .collect();
        Ok(result)
    }

    /// RVV-optimized FMA implementation
    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    fn fma_rvv(a: &[f32], b: &[f32], c: &[f32]) -> RiscVResult<Vec<f32>> {
        let mut result = Vec::with_capacity(a.len());
        let len = a.len();

        // Process in chunks optimized for RVV
        const CHUNK_SIZE: usize = 32;
        let chunks = len / CHUNK_SIZE;

        for chunk in 0..chunks {
            let start = chunk * CHUNK_SIZE;
            let end = start + CHUNK_SIZE;

            // In real RVV, this would use vfmacc.vv instruction
            for i in start..end {
                result.push(a[i] * b[i] + c[i]);
            }
        }

        // Handle remaining elements
        for i in (chunks * CHUNK_SIZE)..len {
            result.push(a[i] * b[i] + c[i]);
        }

        Ok(result)
    }

    /// Matrix-vector multiplication using RVV
    pub fn matvec_multiply(matrix: &[Vec<f32>], vector: &[f32]) -> RiscVResult<Vec<f32>> {
        if matrix.is_empty() || vector.is_empty() {
            return Err(SimdError::EmptyInput);
        }

        let rows = matrix.len();
        let cols = matrix[0].len();

        if vector.len() != cols {
            return Err(SimdError::DimensionMismatch {
                expected: cols,
                actual: vector.len(),
            });
        }

        let mut result = Vec::with_capacity(rows);

        // Use RVV-optimized dot product for each row
        for row in matrix {
            let dot_result = Self::dot_product(row, vector)?;
            result.push(dot_result);
        }

        Ok(result)
    }

    /// Vector normalization using RVV
    pub fn normalize(vector: &[f32]) -> RiscVResult<Vec<f32>> {
        if vector.is_empty() {
            return Err(SimdError::EmptyInput);
        }

        let norm = Self::norm(vector)?;

        if norm == 0.0 {
            // Return zero vector if input norm is zero
            return Ok(vec![0.0; vector.len()]);
        }

        Self::scale(vector, 1.0 / norm)
    }
}

/// RVV-optimized activation functions
pub struct RiscVActivations;

impl RiscVActivations {
    /// ReLU activation using RVV
    pub fn relu(input: &[f32]) -> RiscVResult<Vec<f32>> {
        if input.is_empty() {
            return Err(SimdError::EmptyInput);
        }

        // Scalar fallback implementation
        let result = input.iter().map(|&x| x.max(0.0)).collect();
        Ok(result)
    }

    /// Sigmoid activation using RVV (approximation)
    pub fn sigmoid(input: &[f32]) -> RiscVResult<Vec<f32>> {
        if input.is_empty() {
            return Err(SimdError::EmptyInput);
        }

        // Scalar fallback with approximation
        let result = input
            .iter()
            .map(|&x| {
                let clamped = x.clamp(-10.0, 10.0);
                1.0 / (1.0 + (-clamped).exp())
            })
            .collect();
        Ok(result)
    }

    /// Tanh activation using RVV
    pub fn tanh(input: &[f32]) -> RiscVResult<Vec<f32>> {
        if input.is_empty() {
            return Err(SimdError::EmptyInput);
        }

        // Scalar fallback implementation
        let result = input.iter().map(|&x| x.tanh()).collect();
        Ok(result)
    }
}

/// RVV-optimized distance metrics
pub struct RiscVDistanceOps;

impl RiscVDistanceOps {
    /// Euclidean distance using RVV
    pub fn euclidean_distance(x: &[f32], y: &[f32]) -> RiscVResult<f32> {
        if x.len() != y.len() {
            return Err(SimdError::DimensionMismatch {
                expected: x.len(),
                actual: y.len(),
            });
        }

        if x.is_empty() {
            return Err(SimdError::EmptyInput);
        }

        // Scalar fallback implementation
        let sum_of_squares: f32 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        Ok(sum_of_squares.sqrt())
    }

    /// Manhattan distance using RVV
    pub fn manhattan_distance(x: &[f32], y: &[f32]) -> RiscVResult<f32> {
        if x.len() != y.len() {
            return Err(SimdError::DimensionMismatch {
                expected: x.len(),
                actual: y.len(),
            });
        }

        if x.is_empty() {
            return Err(SimdError::EmptyInput);
        }

        // Scalar fallback implementation
        let sum: f32 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).abs()).sum();
        Ok(sum)
    }

    /// Cosine distance using RVV
    pub fn cosine_distance(x: &[f32], y: &[f32]) -> RiscVResult<f32> {
        if x.len() != y.len() {
            return Err(SimdError::DimensionMismatch {
                expected: x.len(),
                actual: y.len(),
            });
        }

        if x.is_empty() {
            return Err(SimdError::EmptyInput);
        }

        // Scalar fallback implementation
        let dot_product: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let norm_x: f32 = x.iter().map(|a| a * a).sum::<f32>().sqrt();
        let norm_y: f32 = y.iter().map(|b| b * b).sum::<f32>().sqrt();

        if norm_x == 0.0 || norm_y == 0.0 {
            return Ok(1.0); // Maximum distance
        }

        let cosine_similarity = dot_product / (norm_x * norm_y);
        Ok(1.0 - cosine_similarity)
    }
}

/// RVV configuration and detection
pub struct RiscVConfig {
    caps: RiscVVectorCaps,
}

impl RiscVConfig {
    /// Create new RISC-V configuration
    pub fn new() -> Self {
        Self {
            caps: RiscVVectorCaps::detect(),
        }
    }

    /// Check if RVV is available
    pub fn is_available(&self) -> bool {
        self.caps.available
    }

    /// Get vector capabilities
    pub fn capabilities(&self) -> RiscVVectorCaps {
        self.caps
    }

    /// Get optimal width for f32 operations
    pub fn optimal_f32_width(&self) -> usize {
        self.caps.f32_width()
    }

    /// Get optimal width for f64 operations
    pub fn optimal_f64_width(&self) -> usize {
        self.caps.f64_width()
    }
}

impl Default for RiscVConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Global RISC-V configuration
static RISCV_CONFIG: std::sync::LazyLock<RiscVConfig> = std::sync::LazyLock::new(RiscVConfig::new);

/// Get global RISC-V configuration
pub fn riscv_config() -> &'static RiscVConfig {
    &RISCV_CONFIG
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_riscv_caps_detection() {
        let caps = RiscVVectorCaps::detect();

        // Basic sanity checks
        assert!(caps.vlen > 0);
        assert!(caps.elen > 0);
        assert!(caps.f32_width() >= 1);
        assert!(caps.f64_width() >= 1);
    }

    #[test]
    fn test_riscv_dot_product() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 3.0, 4.0, 5.0];

        let result = RiscVVectorOps::dot_product(&x, &y).unwrap();
        assert_eq!(result, 40.0); // 1*2 + 2*3 + 3*4 + 4*5
    }

    #[test]
    fn test_riscv_vector_add() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        let result = RiscVVectorOps::add(&x, &y).unwrap();
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_riscv_scale() {
        let vector = vec![1.0, 2.0, 3.0];

        let result = RiscVVectorOps::scale(&vector, 2.0).unwrap();
        assert_eq!(result, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_riscv_sum() {
        let vector = vec![1.0, 2.0, 3.0, 4.0];

        let result = RiscVVectorOps::sum(&vector).unwrap();
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_riscv_max_min() {
        let vector = vec![3.0, 1.0, 4.0, 1.0, 5.0];

        let max_result = RiscVVectorOps::max(&vector).unwrap();
        let min_result = RiscVVectorOps::min(&vector).unwrap();

        assert_eq!(max_result, 5.0);
        assert_eq!(min_result, 1.0);
    }

    #[test]
    fn test_riscv_norm() {
        let vector = vec![3.0, 4.0, 0.0];

        let result = RiscVVectorOps::norm(&vector).unwrap();
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_riscv_relu() {
        let input = vec![-1.0, 0.0, 1.0, 2.0];

        let result = RiscVActivations::relu(&input).unwrap();
        assert_eq!(result, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_riscv_sigmoid() {
        let input = vec![0.0, 1.0, -1.0];

        let result = RiscVActivations::sigmoid(&input).unwrap();

        // Check approximate values
        assert!((result[0] - 0.5).abs() < 0.01); // sigmoid(0) ≈ 0.5
        assert!(result[1] > 0.5); // sigmoid(1) > 0.5
        assert!(result[2] < 0.5); // sigmoid(-1) < 0.5
    }

    #[test]
    fn test_riscv_euclidean_distance() {
        let x = vec![0.0, 0.0];
        let y = vec![3.0, 4.0];

        let result = RiscVDistanceOps::euclidean_distance(&x, &y).unwrap();
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_riscv_manhattan_distance() {
        let x = vec![1.0, 2.0];
        let y = vec![4.0, 6.0];

        let result = RiscVDistanceOps::manhattan_distance(&x, &y).unwrap();
        assert_eq!(result, 7.0); // |1-4| + |2-6| = 3 + 4 = 7
    }

    #[test]
    fn test_dimension_mismatch() {
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0, 3.0];

        assert!(RiscVVectorOps::dot_product(&x, &y).is_err());
        assert!(RiscVDistanceOps::euclidean_distance(&x, &y).is_err());
    }

    #[test]
    fn test_empty_input() {
        let empty: Vec<f32> = vec![];

        assert!(RiscVVectorOps::sum(&empty).is_err());
        assert!(RiscVActivations::relu(&empty).is_err());
    }

    #[test]
    fn test_riscv_config() {
        let config = RiscVConfig::new();

        assert!(config.optimal_f32_width() >= 1);
        assert!(config.optimal_f64_width() >= 1);

        let caps = config.capabilities();
        assert!(caps.vlen > 0);
        assert!(caps.elen > 0);
    }

    #[test]
    fn test_global_riscv_config() {
        let config = riscv_config();

        assert!(config.optimal_f32_width() >= 1);
        assert!(config.optimal_f64_width() >= 1);
    }

    #[test]
    fn test_riscv_fma() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let c = vec![1.0, 1.0, 1.0, 1.0];

        let result = RiscVVectorOps::fma(&a, &b, &c).unwrap();
        let expected = vec![3.0, 7.0, 13.0, 21.0]; // a[i] * b[i] + c[i]

        assert_eq!(result, expected);
    }

    #[test]
    fn test_riscv_fma_dimension_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![2.0, 3.0, 4.0];
        let c = vec![1.0, 1.0];

        assert!(RiscVVectorOps::fma(&a, &b, &c).is_err());
    }

    #[test]
    fn test_riscv_matvec_multiply() {
        let matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let vector = vec![2.0, 3.0];

        let result = RiscVVectorOps::matvec_multiply(&matrix, &vector).unwrap();
        let expected = vec![8.0, 18.0, 28.0]; // [1*2+2*3, 3*2+4*3, 5*2+6*3]

        assert_eq!(result, expected);
    }

    #[test]
    fn test_riscv_matvec_dimension_mismatch() {
        let matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let vector = vec![2.0, 3.0, 4.0]; // Wrong size

        assert!(RiscVVectorOps::matvec_multiply(&matrix, &vector).is_err());
    }

    #[test]
    fn test_riscv_normalize() {
        let vector = vec![3.0, 4.0, 0.0];

        let result = RiscVVectorOps::normalize(&vector).unwrap();
        let expected_norm = (3.0_f32.powi(2) + 4.0_f32.powi(2)).sqrt(); // 5.0
        let expected = vec![3.0 / expected_norm, 4.0 / expected_norm, 0.0];

        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-6);
        }
    }

    #[test]
    fn test_riscv_normalize_zero_vector() {
        let vector = vec![0.0, 0.0, 0.0];

        let result = RiscVVectorOps::normalize(&vector).unwrap();
        let expected = vec![0.0, 0.0, 0.0];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_riscv_empty_fma() {
        let empty: Vec<f32> = vec![];

        assert!(RiscVVectorOps::fma(&empty, &empty, &empty).is_err());
    }

    #[test]
    fn test_riscv_capabilities_detection() {
        let caps = RiscVVectorCaps::detect();

        // Basic sanity checks
        assert!(caps.vlen > 0);
        assert!(caps.elen > 0);
        assert!(caps.f32_width() >= 1);
        assert!(caps.f64_width() >= 1);

        // VLEN should be a reasonable value
        assert!(caps.vlen >= 64 && caps.vlen <= 2048);

        // ELEN should support at least 32-bit elements
        assert!(caps.elen >= 32);
    }

    #[test]
    fn test_riscv_large_vector_operations() {
        // Test with larger vectors to exercise chunked processing
        let size = 1000;
        let x: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();

        let dot_result = RiscVVectorOps::dot_product(&x, &y).unwrap();
        let add_result = RiscVVectorOps::add(&x, &y).unwrap();
        let scale_result = RiscVVectorOps::scale(&x, 2.0).unwrap();

        // Verify some basic properties
        assert!(dot_result > 0.0);
        assert_eq!(add_result.len(), size);
        assert_eq!(scale_result.len(), size);

        // Check a few specific values
        assert_eq!(add_result[0], 1.0); // 0 + 1
        assert_eq!(add_result[10], 21.0); // 10 + 11
        assert_eq!(scale_result[0], 0.0); // 0 * 2
        assert_eq!(scale_result[10], 20.0); // 10 * 2
    }
}
