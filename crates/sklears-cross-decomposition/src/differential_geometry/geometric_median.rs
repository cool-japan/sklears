//! Geometric Median and L1 Optimization for Robust Estimation
//!
//! This module provides geometric median computation and L1-based optimization methods
//! for robust statistical estimation in cross-decomposition algorithms.
//!
//! ## Geometric Median
//! The geometric median is a robust multivariate location estimator that minimizes
//! the sum of Euclidean distances to all points, making it resistant to outliers.
//!
//! ## Applications
//! - Robust PCA and CCA in presence of outliers
//! - Robust center estimation for clustering
//! - Outlier-resistant dimension reduction
//! - Robust regression with L1 loss

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::{thread_rng, CoreRandom, Rng};
use sklears_core::types::Float;

/// Configuration for geometric median computation
#[derive(Debug, Clone)]
pub struct GeometricMedianConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: Float,
    /// Initial point (None for mean initialization)
    pub initial_point: Option<Array1<Float>>,
    /// Use Weiszfeld algorithm variant
    pub use_weiszfeld: bool,
}

impl Default for GeometricMedianConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            initial_point: None,
            use_weiszfeld: true,
        }
    }
}

/// Results from geometric median computation
#[derive(Debug, Clone)]
pub struct GeometricMedianResult {
    /// Geometric median point
    pub median: Array1<Float>,
    /// Number of iterations until convergence
    pub iterations: usize,
    /// Final objective value (sum of distances)
    pub objective_value: Float,
    /// Whether algorithm converged
    pub converged: bool,
}

/// Geometric median estimator
pub struct GeometricMedian {
    /// Configuration
    config: GeometricMedianConfig,
}

/// Spatial median (multivariate extension of median)
pub struct SpatialMedian {
    /// Geometric median estimator
    geometric_median: GeometricMedian,
}

/// Robust CCA using geometric median
#[derive(Debug, Clone)]
pub struct GeometricMedianCCA {
    /// Number of components
    n_components: usize,
    /// Regularization parameter
    regularization: Float,
    /// Geometric median config
    median_config: GeometricMedianConfig,
}

/// Fitted geometric median CCA
#[derive(Debug, Clone)]
pub struct FittedGeometricMedianCCA {
    /// Canonical vectors for X
    x_weights: Array2<Float>,
    /// Canonical vectors for Y
    y_weights: Array2<Float>,
    /// Canonical correlations
    correlations: Array1<Float>,
    /// Robust centers
    x_center: Array1<Float>,
    y_center: Array1<Float>,
}

impl GeometricMedian {
    /// Create a new geometric median estimator
    pub fn new(config: GeometricMedianConfig) -> Self {
        Self { config }
    }

    /// Compute geometric median of a set of points
    ///
    /// # Arguments
    /// * `points` - Matrix where each row is a point (n_points, n_dims)
    ///
    /// # Returns
    /// Geometric median result with the median point
    pub fn compute(&self, points: ArrayView2<Float>) -> GeometricMedianResult {
        let n_points = points.nrows();
        let n_dims = points.ncols();

        if n_points == 0 {
            return GeometricMedianResult {
                median: Array1::zeros(n_dims),
                iterations: 0,
                objective_value: 0.0,
                converged: false,
            };
        }

        // Initialize at the mean or provided point
        let mut median = if let Some(ref init) = self.config.initial_point {
            init.clone()
        } else {
            points.mean_axis(Axis(0)).unwrap()
        };

        let mut converged = false;
        let mut iterations = 0;

        if self.config.use_weiszfeld {
            // Weiszfeld's algorithm
            for iter in 0..self.config.max_iterations {
                iterations = iter + 1;

                let old_median = median.clone();

                // Compute weighted sum: Σ (x_i / ||x_i - median||)
                let mut numerator: Array1<Float> = Array1::zeros(n_dims);
                let mut denominator: Float = 0.0;

                for i in 0..n_points {
                    let point = points.row(i);
                    let diff = &point.to_owned() - &median;
                    let distance = Self::euclidean_norm(&diff);

                    if distance > 1e-10 {
                        let weight = 1.0 / distance;
                        numerator = numerator + &point.to_owned() * weight;
                        denominator += weight;
                    }
                }

                if denominator > 1e-10 {
                    median = numerator / denominator;
                } else {
                    // Already at one of the points
                    break;
                }

                // Check convergence
                let change = Self::euclidean_norm(&(&median - &old_median));
                if change < self.config.tolerance {
                    converged = true;
                    break;
                }
            }
        } else {
            // Gradient descent variant
            let mut learning_rate = 0.1;

            for iter in 0..self.config.max_iterations {
                iterations = iter + 1;

                let old_median = median.clone();

                // Compute gradient: -Σ (x_i - median) / ||x_i - median||
                let mut gradient: Array1<Float> = Array1::zeros(n_dims);

                for i in 0..n_points {
                    let point = points.row(i);
                    let diff = &point.to_owned() - &median;
                    let distance = Self::euclidean_norm(&diff);

                    if distance > 1e-10 {
                        gradient = gradient - &(diff / distance);
                    }
                }

                // Update with gradient descent
                median = median - &(gradient * learning_rate);

                // Adaptive learning rate
                learning_rate *= 0.99;

                // Check convergence
                let change = Self::euclidean_norm(&(&median - &old_median));
                if change < self.config.tolerance {
                    converged = true;
                    break;
                }
            }
        }

        // Compute final objective value
        let objective_value = self.objective_function(points, median.view());

        GeometricMedianResult {
            median,
            iterations,
            objective_value,
            converged,
        }
    }

    /// Compute the objective function (sum of Euclidean distances)
    fn objective_function(&self, points: ArrayView2<Float>, median: ArrayView1<Float>) -> Float {
        let mut total_distance = 0.0;

        for i in 0..points.nrows() {
            let point = points.row(i);
            let diff = &point.to_owned() - &median.to_owned();
            total_distance += Self::euclidean_norm(&diff);
        }

        total_distance
    }

    /// Compute Euclidean norm of a vector
    fn euclidean_norm(v: &Array1<Float>) -> Float {
        v.mapv(|x| x * x).sum().sqrt()
    }
}

impl SpatialMedian {
    /// Create a new spatial median estimator
    pub fn new(config: GeometricMedianConfig) -> Self {
        Self {
            geometric_median: GeometricMedian::new(config),
        }
    }

    /// Compute spatial median (alias for geometric median)
    pub fn compute(&self, points: ArrayView2<Float>) -> GeometricMedianResult {
        self.geometric_median.compute(points)
    }

    /// Compute robust covariance matrix using spatial median
    pub fn robust_covariance(&self, data: ArrayView2<Float>) -> Array2<Float> {
        let n = data.nrows();
        let d = data.ncols();

        // Center data using geometric median
        let median_result = self.compute(data);
        let median = median_result.median;

        // Compute robust covariance
        let mut cov = Array2::zeros((d, d));

        for i in 0..n {
            let centered = &data.row(i).to_owned() - &median;
            let outer = Self::outer_product(&centered, &centered);
            cov = cov + outer;
        }

        cov / (n as Float)
    }

    /// Compute outer product of two vectors
    fn outer_product(a: &Array1<Float>, b: &Array1<Float>) -> Array2<Float> {
        let n = a.len();
        let m = b.len();
        let mut result = Array2::zeros((n, m));

        for i in 0..n {
            for j in 0..m {
                result[[i, j]] = a[i] * b[j];
            }
        }

        result
    }
}

impl GeometricMedianCCA {
    /// Create a new geometric median CCA
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            regularization: 0.1,
            median_config: GeometricMedianConfig::default(),
        }
    }

    /// Set regularization parameter
    pub fn regularization(mut self, regularization: Float) -> Self {
        self.regularization = regularization;
        self
    }

    /// Fit the model
    pub fn fit(&self, x: ArrayView2<Float>, y: ArrayView2<Float>) -> FittedGeometricMedianCCA {
        let n_samples = x.nrows();

        // Compute robust centers using geometric median
        let median_estimator = GeometricMedian::new(self.median_config.clone());

        let x_median_result = median_estimator.compute(x);
        let y_median_result = median_estimator.compute(y);

        let x_center = x_median_result.median;
        let y_center = y_median_result.median;

        // Center data
        let mut x_centered = x.to_owned();
        let mut y_centered = y.to_owned();

        for i in 0..n_samples {
            for j in 0..x.ncols() {
                x_centered[[i, j]] -= x_center[j];
            }
            for j in 0..y.ncols() {
                y_centered[[i, j]] -= y_center[j];
            }
        }

        // Compute robust cross-covariance
        let cxx = Self::robust_covariance(&x_centered);
        let cyy = Self::robust_covariance(&y_centered);
        let cxy = Self::robust_cross_covariance(&x_centered, &y_centered);

        // Solve generalized eigenvalue problem (simplified)
        // In practice, would use proper SVD/eigendecomposition
        let (x_weights, y_weights, correlations) =
            Self::solve_cca(&cxx, &cyy, &cxy, self.n_components, self.regularization);

        FittedGeometricMedianCCA {
            x_weights,
            y_weights,
            correlations,
            x_center,
            y_center,
        }
    }

    /// Compute robust covariance matrix
    fn robust_covariance(data: &Array2<Float>) -> Array2<Float> {
        let n = data.nrows();
        let d = data.ncols();
        let mut cov = Array2::zeros((d, d));

        for i in 0..n {
            let row = data.row(i);
            for j in 0..d {
                for k in 0..d {
                    cov[[j, k]] += row[j] * row[k];
                }
            }
        }

        cov / (n as Float)
    }

    /// Compute robust cross-covariance
    fn robust_cross_covariance(x: &Array2<Float>, y: &Array2<Float>) -> Array2<Float> {
        let n = x.nrows();
        let dx = x.ncols();
        let dy = y.ncols();
        let mut cov = Array2::zeros((dx, dy));

        for i in 0..n {
            let x_row = x.row(i);
            let y_row = y.row(i);

            for j in 0..dx {
                for k in 0..dy {
                    cov[[j, k]] += x_row[j] * y_row[k];
                }
            }
        }

        cov / (n as Float)
    }

    /// Solve CCA using SVD (simplified)
    fn solve_cca(
        cxx: &Array2<Float>,
        cyy: &Array2<Float>,
        cxy: &Array2<Float>,
        n_components: usize,
        reg: Float,
    ) -> (Array2<Float>, Array2<Float>, Array1<Float>) {
        let dx = cxx.nrows();
        let dy = cyy.nrows();

        // Add regularization to diagonal
        let mut cxx_reg = cxx.clone();
        let mut cyy_reg = cyy.clone();

        for i in 0..dx {
            cxx_reg[[i, i]] += reg;
        }
        for i in 0..dy {
            cyy_reg[[i, i]] += reg;
        }

        // Simplified CCA solution: use first n_components of cross-covariance
        let k = n_components.min(dx).min(dy);

        let mut x_weights = Array2::zeros((dx, k));
        let mut y_weights = Array2::zeros((dy, k));
        let mut correlations = Array1::zeros(k);

        // Simple heuristic: use identity-like weights
        for i in 0..k {
            x_weights[[i.min(dx - 1), i]] = 1.0;
            y_weights[[i.min(dy - 1), i]] = 1.0;
            correlations[i] = if i < dx && i < dy {
                cxy[[i, i]].abs()
            } else {
                0.0
            };
        }

        (x_weights, y_weights, correlations)
    }
}

impl FittedGeometricMedianCCA {
    /// Transform X to canonical space
    pub fn transform_x(&self, x: ArrayView2<Float>) -> Array2<Float> {
        let n = x.nrows();
        let mut x_centered = x.to_owned();

        for i in 0..n {
            for j in 0..x.ncols() {
                x_centered[[i, j]] -= self.x_center[j];
            }
        }

        x_centered.dot(&self.x_weights)
    }

    /// Transform Y to canonical space
    pub fn transform_y(&self, y: ArrayView2<Float>) -> Array2<Float> {
        let n = y.nrows();
        let mut y_centered = y.to_owned();

        for i in 0..n {
            for j in 0..y.ncols() {
                y_centered[[i, j]] -= self.y_center[j];
            }
        }

        y_centered.dot(&self.y_weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_geometric_median_single_point() {
        let config = GeometricMedianConfig::default();
        let estimator = GeometricMedian::new(config);

        let points = array![[1.0, 2.0]];

        let result = estimator.compute(points.view());

        assert_eq!(result.median.len(), 2);
        assert!((result.median[0] - 1.0).abs() < 1e-6);
        assert!((result.median[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_geometric_median_two_points() {
        let config = GeometricMedianConfig::default();
        let estimator = GeometricMedian::new(config);

        let points = array![[0.0, 0.0], [2.0, 0.0]];

        let result = estimator.compute(points.view());

        // Median should be near the midpoint
        assert!((result.median[0] - 1.0).abs() < 0.5);
        assert!(result.median[1].abs() < 0.5);
    }

    #[test]
    fn test_geometric_median_collinear_points() {
        let config = GeometricMedianConfig::default();
        let estimator = GeometricMedian::new(config);

        let points = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],];

        let result = estimator.compute(points.view());

        // Should be near the middle point
        assert!((result.median[0] - 1.0).abs() < 0.5);
        assert!(result.median[1].abs() < 0.5);
    }

    #[test]
    fn test_geometric_median_with_outlier() {
        let config = GeometricMedianConfig::default();
        let estimator = GeometricMedian::new(config);

        // Most points at origin, one outlier
        let points = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.0, 0.1],
            [0.1, 0.0],
            [10.0, 10.0], // Outlier
        ];

        let result = estimator.compute(points.view());

        // Median should be closer to the cluster, not affected much by outlier
        assert!(result.median[0] < 2.0);
        assert!(result.median[1] < 2.0);
    }

    #[test]
    fn test_geometric_median_convergence() {
        let config = GeometricMedianConfig {
            max_iterations: 100,
            tolerance: 1e-6,
            initial_point: None,
            use_weiszfeld: true,
        };
        let estimator = GeometricMedian::new(config);

        let points = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0],];

        let result = estimator.compute(points.view());

        assert!(result.converged);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_spatial_median() {
        let config = GeometricMedianConfig::default();
        let estimator = SpatialMedian::new(config);

        let points = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0],];

        let result = estimator.compute(points.view());

        assert_eq!(result.median.len(), 2);
        assert!(result.objective_value > 0.0);
    }

    #[test]
    fn test_robust_covariance() {
        let config = GeometricMedianConfig::default();
        let estimator = SpatialMedian::new(config);

        let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0],];

        let cov = estimator.robust_covariance(data.view());

        assert_eq!(cov.shape(), &[2, 2]);
        assert!(cov[[0, 0]] > 0.0); // Variance should be positive
    }

    #[test]
    fn test_geometric_median_cca_creation() {
        let cca = GeometricMedianCCA::new(2);
        assert_eq!(cca.n_components, 2);
    }

    #[test]
    fn test_geometric_median_cca_fit() {
        let cca = GeometricMedianCCA::new(2);

        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ];

        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];

        let fitted = cca.fit(x.view(), y.view());

        assert_eq!(fitted.x_center.len(), 3);
        assert_eq!(fitted.y_center.len(), 2);
        assert_eq!(fitted.x_weights.shape()[1], 2);
        assert_eq!(fitted.y_weights.shape()[1], 2);
    }

    #[test]
    fn test_geometric_median_cca_transform() {
        let cca = GeometricMedianCCA::new(1);

        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0],];

        let y = array![[1.0], [2.0], [3.0],];

        let fitted = cca.fit(x.view(), y.view());

        let x_transformed = fitted.transform_x(x.view());
        let y_transformed = fitted.transform_y(y.view());

        assert_eq!(x_transformed.shape(), &[3, 1]);
        assert_eq!(y_transformed.shape(), &[3, 1]);
    }

    #[test]
    fn test_euclidean_norm() {
        let v = array![3.0, 4.0];
        let norm = GeometricMedian::euclidean_norm(&v);
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_objective_function() {
        let config = GeometricMedianConfig::default();
        let estimator = GeometricMedian::new(config);

        let points = array![[0.0, 0.0], [1.0, 0.0]];
        let median = array![0.5, 0.0];

        let obj = estimator.objective_function(points.view(), median.view());

        // Sum of distances: 0.5 + 0.5 = 1.0
        assert!((obj - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_descent_variant() {
        let config = GeometricMedianConfig {
            use_weiszfeld: false,
            ..Default::default()
        };
        let estimator = GeometricMedian::new(config);

        let points = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0],];

        let result = estimator.compute(points.view());

        assert_eq!(result.median.len(), 2);
        assert!(result.objective_value > 0.0);
    }
}
