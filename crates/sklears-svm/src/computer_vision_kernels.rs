//! Computer Vision Kernels for SVM
//!
//! This module implements specialized kernels for computer vision tasks including:
//! - Histogram Intersection Kernel
//! - Spatial Pyramid Kernels
//! - HOG (Histogram of Oriented Gradients) Feature Kernels
//! - Local Binary Pattern (LBP) Kernels
//! - Chi-Square Kernels for histograms
//! - Earth Mover's Distance (EMD) Kernels

use scirs2_core::ndarray::Array2;
use thiserror::Error;

/// Errors for computer vision kernels
#[derive(Error, Debug)]
pub enum CVKernelError {
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("Invalid histogram: negative values not allowed")]
    InvalidHistogram,
    #[error("Empty histogram")]
    EmptyHistogram,
    #[error("Invalid kernel parameters: {message}")]
    InvalidParameters { message: String },
}

/// Computer Vision Kernel Types
#[derive(Debug, Clone, PartialEq)]
pub enum CVKernelType {
    /// Histogram Intersection Kernel
    HistogramIntersection,
    /// Chi-Square Kernel with gamma parameter
    ChiSquare { gamma: f64 },
    /// Spatial Pyramid Kernel with levels and weights
    SpatialPyramid { levels: usize, weights: Vec<f64> },
    /// HOG Feature Kernel
    HOG { bins: usize, cell_size: usize },
    /// Local Binary Pattern Kernel
    LBP { radius: f64, neighbors: usize },
    /// Earth Mover's Distance Kernel
    EMD { distance_matrix: Array2<f64> },
    /// Additive Chi-Square Kernel
    AdditiveChiSquare,
    /// Jensen-Shannon Kernel
    JensenShannon,
    /// Bhattacharyya Kernel
    Bhattacharyya,
    /// Hellinger Kernel
    Hellinger,
}

/// Computer Vision Kernel Function
#[derive(Debug, Clone)]
pub struct CVKernelFunction {
    pub kernel_type: CVKernelType,
}

impl CVKernelFunction {
    /// Create a new computer vision kernel function
    pub fn new(kernel_type: CVKernelType) -> Self {
        Self { kernel_type }
    }

    /// Compute kernel value between two feature vectors
    pub fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64, CVKernelError> {
        if x.len() != y.len() {
            return Err(CVKernelError::DimensionMismatch {
                expected: x.len(),
                actual: y.len(),
            });
        }

        match &self.kernel_type {
            CVKernelType::HistogramIntersection => {
                self.validate_histogram(x)?;
                self.validate_histogram(y)?;
                Ok(self.histogram_intersection(x, y))
            }
            CVKernelType::ChiSquare { gamma } => {
                self.validate_histogram(x)?;
                self.validate_histogram(y)?;
                Ok(self.chi_square_kernel(x, y, *gamma))
            }
            CVKernelType::SpatialPyramid { levels, weights } => {
                self.spatial_pyramid_kernel(x, y, *levels, weights)
            }
            CVKernelType::HOG { bins, cell_size } => self.hog_kernel(x, y, *bins, *cell_size),
            CVKernelType::LBP { radius, neighbors } => self.lbp_kernel(x, y, *radius, *neighbors),
            CVKernelType::EMD { distance_matrix } => self.emd_kernel(x, y, distance_matrix),
            CVKernelType::AdditiveChiSquare => {
                self.validate_histogram(x)?;
                self.validate_histogram(y)?;
                Ok(self.additive_chi_square_kernel(x, y))
            }
            CVKernelType::JensenShannon => {
                self.validate_histogram(x)?;
                self.validate_histogram(y)?;
                Ok(self.jensen_shannon_kernel(x, y))
            }
            CVKernelType::Bhattacharyya => {
                self.validate_histogram(x)?;
                self.validate_histogram(y)?;
                Ok(self.bhattacharyya_kernel(x, y))
            }
            CVKernelType::Hellinger => {
                self.validate_histogram(x)?;
                self.validate_histogram(y)?;
                Ok(self.hellinger_kernel(x, y))
            }
        }
    }

    /// Compute kernel matrix for datasets
    pub fn compute_matrix(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
    ) -> Result<Array2<f64>, CVKernelError> {
        let (n_x, n_features_x) = x.dim();
        let (n_y, n_features_y) = y.dim();

        if n_features_x != n_features_y {
            return Err(CVKernelError::DimensionMismatch {
                expected: n_features_x,
                actual: n_features_y,
            });
        }

        let mut kernel_matrix = Array2::zeros((n_x, n_y));

        for i in 0..n_x {
            for j in 0..n_y {
                let x_row = x.row(i).to_vec();
                let y_row = y.row(j).to_vec();
                kernel_matrix[[i, j]] = self.compute(&x_row, &y_row)?;
            }
        }

        Ok(kernel_matrix)
    }

    /// Validate histogram (non-negative values)
    fn validate_histogram(&self, hist: &[f64]) -> Result<(), CVKernelError> {
        if hist.is_empty() {
            return Err(CVKernelError::EmptyHistogram);
        }

        for &value in hist {
            if value < 0.0 {
                return Err(CVKernelError::InvalidHistogram);
            }
        }

        Ok(())
    }

    /// Histogram Intersection Kernel
    fn histogram_intersection(&self, x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y.iter()).map(|(a, b)| a.min(*b)).sum()
    }

    /// Chi-Square Kernel
    fn chi_square_kernel(&self, x: &[f64], y: &[f64], gamma: f64) -> f64 {
        let chi_square_distance = x
            .iter()
            .zip(y.iter())
            .map(|(a, b)| {
                if a + b > 0.0 {
                    (a - b).powi(2) / (a + b)
                } else {
                    0.0
                }
            })
            .sum::<f64>();

        (-gamma * chi_square_distance).exp()
    }

    /// Spatial Pyramid Kernel
    fn spatial_pyramid_kernel(
        &self,
        x: &[f64],
        y: &[f64],
        levels: usize,
        weights: &[f64],
    ) -> Result<f64, CVKernelError> {
        if weights.len() != levels + 1 {
            return Err(CVKernelError::InvalidParameters {
                message: format!("Expected {} weights for {} levels", levels + 1, levels),
            });
        }

        let total_cells: usize = (0..=levels).map(|lvl| 1usize << (2 * lvl)).sum();

        if x.len() != y.len() {
            return Err(CVKernelError::DimensionMismatch {
                expected: x.len(),
                actual: y.len(),
            });
        }

        if x.len() % total_cells != 0 {
            return Err(CVKernelError::InvalidParameters {
                message: format!(
                    "Feature vector length {} is not divisible by spatial pyramid cell count {}",
                    x.len(),
                    total_cells
                ),
            });
        }

        let cell_hist_size = x.len() / total_cells;

        let mut kernel_value = 0.0;
        let mut offset = 0;

        for level in 0..=levels {
            let grid_size = 1 << (2 * level); // 4^level cells
            let level_span = grid_size * cell_hist_size;
            debug_assert!(offset + level_span <= x.len());

            let mut level_intersection = 0.0;
            for cell in 0..grid_size {
                let start = offset + cell * cell_hist_size;
                let end = start + cell_hist_size;

                let x_cell = &x[start..end];
                let y_cell = &y[start..end];

                level_intersection += self.histogram_intersection(x_cell, y_cell);
            }

            kernel_value += weights[level] * level_intersection;
            offset += level_span;
        }

        Ok(kernel_value)
    }

    /// HOG Feature Kernel
    fn hog_kernel(
        &self,
        x: &[f64],
        y: &[f64],
        bins: usize,
        cell_size: usize,
    ) -> Result<f64, CVKernelError> {
        if x.len() % (bins * cell_size) != 0 {
            return Err(CVKernelError::InvalidParameters {
                message: "Feature vector length must be divisible by bins * cell_size".to_string(),
            });
        }

        let num_cells = x.len() / (bins * cell_size);
        let mut total_intersection = 0.0;

        for cell in 0..num_cells {
            let start = cell * bins * cell_size;
            let end = start + bins * cell_size;

            let x_cell = &x[start..end];
            let y_cell = &y[start..end];

            total_intersection += self.histogram_intersection(x_cell, y_cell);
        }

        Ok(total_intersection / num_cells as f64)
    }

    /// Local Binary Pattern Kernel
    fn lbp_kernel(
        &self,
        x: &[f64],
        y: &[f64],
        radius: f64,
        neighbors: usize,
    ) -> Result<f64, CVKernelError> {
        // LBP features are typically histograms of local binary patterns
        // We use histogram intersection as the base kernel
        let expected_bins = 2_usize.pow(neighbors as u32);

        if x.len() != expected_bins || y.len() != expected_bins {
            return Err(CVKernelError::InvalidParameters {
                message: format!(
                    "Expected {} bins for {} neighbors",
                    expected_bins, neighbors
                ),
            });
        }

        // Normalize histograms
        let x_sum: f64 = x.iter().sum();
        let y_sum: f64 = y.iter().sum();

        if x_sum == 0.0 || y_sum == 0.0 {
            return Ok(0.0);
        }

        let x_normalized: Vec<f64> = x.iter().map(|v| v / x_sum).collect();
        let y_normalized: Vec<f64> = y.iter().map(|v| v / y_sum).collect();

        Ok(self.histogram_intersection(&x_normalized, &y_normalized))
    }

    /// Earth Mover's Distance Kernel
    fn emd_kernel(
        &self,
        x: &[f64],
        y: &[f64],
        distance_matrix: &Array2<f64>,
    ) -> Result<f64, CVKernelError> {
        if distance_matrix.nrows() != x.len() || distance_matrix.ncols() != y.len() {
            return Err(CVKernelError::InvalidParameters {
                message: "Distance matrix dimensions don't match feature vectors".to_string(),
            });
        }

        // Simplified EMD calculation (optimal transport)
        // For full EMD, we would need a linear programming solver
        let mut emd_distance = 0.0;
        let x_sum: f64 = x.iter().sum();
        let y_sum: f64 = y.iter().sum();

        if x_sum == 0.0 || y_sum == 0.0 {
            return Ok(0.0);
        }

        // Normalize to make them probability distributions
        let x_normalized: Vec<f64> = x.iter().map(|v| v / x_sum).collect();
        let y_normalized: Vec<f64> = y.iter().map(|v| v / y_sum).collect();

        // Approximate EMD using minimum cost flow
        for i in 0..x.len() {
            for j in 0..y.len() {
                let flow = x_normalized[i].min(y_normalized[j]);
                emd_distance += flow * distance_matrix[[i, j]];
            }
        }

        // Convert distance to kernel value
        Ok((-emd_distance).exp())
    }

    /// Additive Chi-Square Kernel
    fn additive_chi_square_kernel(&self, x: &[f64], y: &[f64]) -> f64 {
        x.iter()
            .zip(y.iter())
            .map(|(a, b)| {
                if a + b > 0.0 {
                    2.0 * a * b / (a + b)
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Jensen-Shannon Kernel
    fn jensen_shannon_kernel(&self, x: &[f64], y: &[f64]) -> f64 {
        let x_sum: f64 = x.iter().sum();
        let y_sum: f64 = y.iter().sum();

        if x_sum == 0.0 || y_sum == 0.0 {
            return 0.0;
        }

        let x_normalized: Vec<f64> = x.iter().map(|v| v / x_sum).collect();
        let y_normalized: Vec<f64> = y.iter().map(|v| v / y_sum).collect();

        let mut js_divergence = 0.0;

        for i in 0..x.len() {
            let p = x_normalized[i];
            let q = y_normalized[i];
            let m = (p + q) / 2.0;

            if p > 0.0 && m > 0.0 {
                js_divergence += p * (p / m).ln();
            }
            if q > 0.0 && m > 0.0 {
                js_divergence += q * (q / m).ln();
            }
        }

        js_divergence /= 2.0;

        // Convert divergence to kernel value
        (-js_divergence).exp()
    }

    /// Bhattacharyya Kernel
    fn bhattacharyya_kernel(&self, x: &[f64], y: &[f64]) -> f64 {
        let x_sum: f64 = x.iter().sum();
        let y_sum: f64 = y.iter().sum();

        if x_sum == 0.0 || y_sum == 0.0 {
            return 0.0;
        }

        let x_normalized: Vec<f64> = x.iter().map(|v| v / x_sum).collect();
        let y_normalized: Vec<f64> = y.iter().map(|v| v / y_sum).collect();

        x_normalized
            .iter()
            .zip(y_normalized.iter())
            .map(|(a, b)| (a * b).sqrt())
            .sum()
    }

    /// Hellinger Kernel
    fn hellinger_kernel(&self, x: &[f64], y: &[f64]) -> f64 {
        let bhattacharyya = self.bhattacharyya_kernel(x, y);
        bhattacharyya.sqrt()
    }
}

/// Utilities for computer vision kernels
pub mod cv_utils {
    use super::*;

    /// Create spatial pyramid weights (decreasing with level)
    pub fn create_pyramid_weights(levels: usize) -> Vec<f64> {
        let mut weights = Vec::with_capacity(levels + 1);

        for level in 0..=levels {
            if level == 0 {
                weights.push(1.0);
            } else {
                weights.push(0.5 / (1 << (level - 1)) as f64);
            }
        }

        weights
    }

    /// Create distance matrix for Earth Mover's Distance
    pub fn create_distance_matrix(size: usize, distance_type: &str) -> Array2<f64> {
        let mut matrix = Array2::zeros((size, size));

        match distance_type {
            "euclidean" => {
                for i in 0..size {
                    for j in 0..size {
                        matrix[[i, j]] = ((i as f64 - j as f64).powi(2)).sqrt();
                    }
                }
            }
            "manhattan" => {
                for i in 0..size {
                    for j in 0..size {
                        matrix[[i, j]] = (i as f64 - j as f64).abs();
                    }
                }
            }
            "grid" => {
                // For 2D grid distances
                let grid_size = (size as f64).sqrt() as usize;
                for i in 0..size {
                    for j in 0..size {
                        let i_x = i % grid_size;
                        let i_y = i / grid_size;
                        let j_x = j % grid_size;
                        let j_y = j / grid_size;

                        matrix[[i, j]] = ((i_x as f64 - j_x as f64).powi(2)
                            + (i_y as f64 - j_y as f64).powi(2))
                        .sqrt();
                    }
                }
            }
            _ => {
                // Default to identity matrix
                for i in 0..size {
                    matrix[[i, i]] = 1.0;
                }
            }
        }

        matrix
    }

    /// Normalize histogram to probability distribution
    pub fn normalize_histogram(hist: &[f64]) -> Vec<f64> {
        let sum: f64 = hist.iter().sum();
        if sum == 0.0 {
            return vec![0.0; hist.len()];
        }
        hist.iter().map(|v| v / sum).collect()
    }

    /// Compute histogram intersection efficiently
    pub fn fast_histogram_intersection(x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y.iter()).map(|(a, b)| a.min(*b)).sum()
    }

    /// Compute chi-square distance between histograms
    pub fn chi_square_distance(x: &[f64], y: &[f64]) -> f64 {
        x.iter()
            .zip(y.iter())
            .map(|(a, b)| {
                if a + b > 0.0 {
                    (a - b).powi(2) / (a + b)
                } else {
                    0.0
                }
            })
            .sum()
    }
}

/// Specialized kernels for different computer vision tasks
pub mod specialized_cv_kernels {
    use super::*;

    /// SIFT Descriptor Kernel
    pub struct SIFTKernel {
        pub sigma: f64,
        pub use_normalization: bool,
    }

    impl SIFTKernel {
        pub fn new(sigma: f64, use_normalization: bool) -> Self {
            Self {
                sigma,
                use_normalization,
            }
        }

        pub fn compute(&self, x: &[f64], y: &[f64]) -> f64 {
            if x.len() != y.len() || x.len() != 128 {
                return 0.0; // SIFT descriptors are 128-dimensional
            }

            let mut x_norm = x.to_vec();
            let mut y_norm = y.to_vec();

            if self.use_normalization {
                let x_magnitude = x.iter().map(|v| v * v).sum::<f64>().sqrt();
                let y_magnitude = y.iter().map(|v| v * v).sum::<f64>().sqrt();

                if x_magnitude > 0.0 {
                    x_norm.iter_mut().for_each(|v| *v /= x_magnitude);
                }
                if y_magnitude > 0.0 {
                    y_norm.iter_mut().for_each(|v| *v /= y_magnitude);
                }
            }

            // RBF kernel on normalized descriptors
            let squared_distance: f64 = x_norm
                .iter()
                .zip(y_norm.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();

            (-squared_distance / (2.0 * self.sigma.powi(2))).exp()
        }
    }

    /// Color Histogram Kernel
    pub struct ColorHistogramKernel {
        pub bins_per_channel: usize,
        pub num_channels: usize,
        pub kernel_type: CVKernelType,
    }

    impl ColorHistogramKernel {
        pub fn new(
            bins_per_channel: usize,
            num_channels: usize,
            kernel_type: CVKernelType,
        ) -> Self {
            Self {
                bins_per_channel,
                num_channels,
                kernel_type,
            }
        }

        pub fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64, CVKernelError> {
            let expected_size = self.bins_per_channel * self.num_channels;
            if x.len() != expected_size || y.len() != expected_size {
                return Err(CVKernelError::DimensionMismatch {
                    expected: expected_size,
                    actual: x.len(),
                });
            }

            let cv_kernel = CVKernelFunction::new(self.kernel_type.clone());
            cv_kernel.compute(x, y)
        }
    }

    /// Texture Kernel using Local Binary Patterns
    pub struct TextureKernel {
        pub radius: f64,
        pub neighbors: usize,
        pub uniform_patterns_only: bool,
    }

    impl TextureKernel {
        pub fn new(radius: f64, neighbors: usize, uniform_patterns_only: bool) -> Self {
            Self {
                radius,
                neighbors,
                uniform_patterns_only,
            }
        }

        pub fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64, CVKernelError> {
            let expected_bins = if self.uniform_patterns_only {
                self.neighbors + 2 // Uniform patterns + 1 for non-uniform
            } else {
                2_usize.pow(self.neighbors as u32)
            };

            if x.len() != expected_bins || y.len() != expected_bins {
                return Err(CVKernelError::DimensionMismatch {
                    expected: expected_bins,
                    actual: x.len(),
                });
            }

            let cv_kernel = CVKernelFunction::new(CVKernelType::LBP {
                radius: self.radius,
                neighbors: self.neighbors,
            });
            cv_kernel.compute(x, y)
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_histogram_intersection_kernel() {
        let kernel = CVKernelFunction::new(CVKernelType::HistogramIntersection);

        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 1.0, 4.0, 3.0];

        let result = kernel.compute(&x, &y).unwrap();
        assert_abs_diff_eq!(result, 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_chi_square_kernel() {
        let kernel = CVKernelFunction::new(CVKernelType::ChiSquare { gamma: 1.0 });

        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];

        let result = kernel.compute(&x, &y).unwrap();
        assert_abs_diff_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bhattacharyya_kernel() {
        let kernel = CVKernelFunction::new(CVKernelType::Bhattacharyya);

        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];

        let result = kernel.compute(&x, &y).unwrap();
        assert_abs_diff_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_additive_chi_square_kernel() {
        let kernel = CVKernelFunction::new(CVKernelType::AdditiveChiSquare);

        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 1.0, 4.0, 3.0];

        let result = kernel.compute(&x, &y).unwrap();
        assert!(result > 0.0);
    }

    #[test]
    fn test_spatial_pyramid_kernel() {
        let weights = cv_utils::create_pyramid_weights(2);
        let kernel = CVKernelFunction::new(CVKernelType::SpatialPyramid { levels: 2, weights });

        // Create feature vectors that represent spatial pyramid
        let x = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0,
        ];
        let y = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0,
        ];

        let result = kernel.compute(&x, &y).unwrap();
        assert!(result > 0.0);
    }

    #[test]
    fn test_kernel_matrix_computation() {
        let kernel = CVKernelFunction::new(CVKernelType::HistogramIntersection);

        let X_var = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let Y_var = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let result = kernel.compute_matrix(&X_var, &Y_var).unwrap();
        assert_eq!(result.dim(), (2, 2));
    }

    #[test]
    fn test_cv_utils() {
        let weights = cv_utils::create_pyramid_weights(2);
        assert_eq!(weights.len(), 3);
        assert_eq!(weights[0], 1.0);

        let distance_matrix = cv_utils::create_distance_matrix(3, "euclidean");
        assert_eq!(distance_matrix.dim(), (3, 3));

        let hist = vec![1.0, 2.0, 3.0, 4.0];
        let normalized = cv_utils::normalize_histogram(&hist);
        let sum: f64 = normalized.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_specialized_kernels() {
        let sift_kernel = specialized_cv_kernels::SIFTKernel::new(1.0, true);
        let x = vec![1.0; 128];
        let y = vec![1.0; 128];
        let result = sift_kernel.compute(&x, &y);
        assert_abs_diff_eq!(result, 1.0, epsilon = 1e-10);

        let color_kernel = specialized_cv_kernels::ColorHistogramKernel::new(
            8,
            3,
            CVKernelType::HistogramIntersection,
        );
        let x = vec![1.0; 24];
        let y = vec![1.0; 24];
        let result = color_kernel.compute(&x, &y).unwrap();
        assert_eq!(result, 24.0);
    }

    #[test]
    fn test_error_handling() {
        let kernel = CVKernelFunction::new(CVKernelType::HistogramIntersection);

        // Test dimension mismatch
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];

        let result = kernel.compute(&x, &y);
        assert!(result.is_err());

        // Test invalid histogram (negative values)
        let x = vec![-1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];

        let result = kernel.compute(&x, &y);
        assert!(result.is_err());
    }
}
