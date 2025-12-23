//! Comprehensive trait-based framework for kernel approximations
//!
//! This module provides a unified trait system for implementing kernel approximation
//! methods, making it easy to create new approximation strategies and compose them.
//!
//! # Architecture
//!
//! - **KernelMethod**: Core trait for all kernel approximation methods
//! - **SamplingStrategy**: Abstract sampling strategies (uniform, importance, etc.)
//! - **FeatureMap**: Abstract feature transformations
//! - **ApproximationQuality**: Quality metrics and guarantees
//! - **ComposableKernel**: Combine multiple kernels

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::SklearsError;
use std::fmt::Debug;

/// Core trait for kernel approximation methods
pub trait KernelMethod: Send + Sync + Debug {
    /// Get the approximation method name
    fn name(&self) -> &str;

    /// Get the number of output features (if known before fitting)
    fn n_output_features(&self) -> Option<usize>;

    /// Get approximation complexity (e.g., O(n*d), O(n^2))
    fn complexity(&self) -> Complexity;

    /// Get theoretical error bounds (if available)
    fn error_bound(&self) -> Option<ErrorBound>;

    /// Check if this method supports the given kernel type
    fn supports_kernel(&self, kernel_type: KernelType) -> bool;

    /// Get supported kernel types
    fn supported_kernels(&self) -> Vec<KernelType>;
}

/// Sampling strategy for selecting landmarks/components
pub trait SamplingStrategy: Send + Sync + Debug {
    /// Sample indices from the dataset
    fn sample(&self, data: &Array2<f64>, n_samples: usize) -> Result<Vec<usize>, SklearsError>;

    /// Get the sampling strategy name
    fn name(&self) -> &str;

    /// Check if this strategy requires fitting
    fn requires_fitting(&self) -> bool {
        false
    }

    /// Fit the sampling strategy (if needed)
    fn fit(&mut self, _data: &Array2<f64>) -> Result<(), SklearsError> {
        Ok(())
    }

    /// Get sampling weights (if applicable)
    fn weights(&self) -> Option<Array1<f64>> {
        None
    }
}

/// Feature map transformation
pub trait FeatureMap: Send + Sync + Debug {
    /// Apply the feature map to input data
    fn transform(&self, data: &Array2<f64>) -> Result<Array2<f64>, SklearsError>;

    /// Get the output dimension
    fn output_dim(&self) -> usize;

    /// Get the feature map name
    fn name(&self) -> &str;

    /// Check if the feature map is invertible
    fn is_invertible(&self) -> bool {
        false
    }

    /// Inverse transform (if supported)
    fn inverse_transform(&self, _features: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        Err(SklearsError::InvalidInput(
            "Inverse transform not supported".to_string(),
        ))
    }
}

/// Approximation quality metrics
pub trait ApproximationQuality: Send + Sync + Debug {
    /// Compute approximation quality metric
    fn compute(
        &self,
        exact_kernel: &Array2<f64>,
        approx_kernel: &Array2<f64>,
    ) -> Result<f64, SklearsError>;

    /// Get the metric name
    fn name(&self) -> &str;

    /// Check if higher values indicate better quality
    fn higher_is_better(&self) -> bool;

    /// Get acceptable quality threshold
    fn acceptable_threshold(&self) -> Option<f64> {
        None
    }
}

/// Computational complexity classification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Complexity {
    /// O(d) - linear in feature dimension
    Linear,
    /// O(d log d) - quasi-linear
    QuasiLinear,
    /// O(n*d) - linear in samples and features
    LinearBoth,
    /// O(n*d^2) - quadratic in features
    QuadraticFeatures,
    /// O(n^2*d) - quadratic in samples
    QuadraticSamples,
    /// O(n^3) - cubic (e.g., exact methods)
    Cubic,
    /// Custom complexity
    Custom(String),
}

impl Complexity {
    /// Get a human-readable description
    pub fn description(&self) -> &str {
        match self {
            Complexity::Linear => "O(d) - Linear in features",
            Complexity::QuasiLinear => "O(d log d) - Quasi-linear",
            Complexity::LinearBoth => "O(n*d) - Linear in samples and features",
            Complexity::QuadraticFeatures => "O(n*d^2) - Quadratic in features",
            Complexity::QuadraticSamples => "O(n^2*d) - Quadratic in samples",
            Complexity::Cubic => "O(n^3) - Cubic complexity",
            Complexity::Custom(s) => s,
        }
    }
}

/// Error bound information
#[derive(Debug, Clone)]
pub struct ErrorBound {
    /// Type of bound (probabilistic, deterministic, etc.)
    pub bound_type: BoundType,
    /// Error value
    pub error: f64,
    /// Confidence level (for probabilistic bounds)
    pub confidence: Option<f64>,
    /// Description of the bound
    pub description: String,
}

/// Type of error bound
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundType {
    /// Probabilistic bound (holds with probability)
    Probabilistic,
    /// Deterministic bound (always holds)
    Deterministic,
    /// Expected error bound
    Expected,
    /// Empirical bound from validation
    Empirical,
}

/// Kernel type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelType {
    /// Radial Basis Function (Gaussian)
    RBF,
    /// Laplacian kernel
    Laplacian,
    /// Polynomial kernel
    Polynomial,
    /// Linear kernel
    Linear,
    /// Arc-cosine (neural network) kernel
    ArcCosine,
    /// Chi-squared kernel
    ChiSquared,
    /// String kernel
    String,
    /// Graph kernel
    Graph,
    /// Custom kernel
    Custom,
}

impl KernelType {
    /// Get kernel name
    pub fn name(&self) -> &str {
        match self {
            KernelType::RBF => "RBF",
            KernelType::Laplacian => "Laplacian",
            KernelType::Polynomial => "Polynomial",
            KernelType::Linear => "Linear",
            KernelType::ArcCosine => "ArcCosine",
            KernelType::ChiSquared => "ChiSquared",
            KernelType::String => "String",
            KernelType::Graph => "Graph",
            KernelType::Custom => "Custom",
        }
    }
}

/// Uniform random sampling strategy
#[derive(Debug, Clone)]
pub struct UniformSampling {
    /// Random seed
    pub random_state: Option<u64>,
}

impl UniformSampling {
    /// Create a new uniform sampling strategy
    pub fn new(random_state: Option<u64>) -> Self {
        Self { random_state }
    }
}

impl SamplingStrategy for UniformSampling {
    fn sample(&self, data: &Array2<f64>, n_samples: usize) -> Result<Vec<usize>, SklearsError> {
        use scirs2_core::random::seeded_rng;

        let (n_rows, _) = data.dim();
        if n_samples > n_rows {
            return Err(SklearsError::InvalidInput(format!(
                "Cannot sample {} points from {} samples",
                n_samples, n_rows
            )));
        }

        let mut rng = seeded_rng(self.random_state.unwrap_or(42));

        // Reservoir sampling for unbiased selection
        let mut indices: Vec<usize> = (0..n_samples).collect();
        for i in n_samples..n_rows {
            let j = rng.gen_range(0..=i);
            if j < n_samples {
                indices[j] = i;
            }
        }

        Ok(indices)
    }

    fn name(&self) -> &str {
        "UniformSampling"
    }
}

/// K-means based sampling strategy
#[derive(Debug, Clone)]
pub struct KMeansSampling {
    /// Number of iterations for k-means
    pub n_iterations: usize,
    /// Random seed
    pub random_state: Option<u64>,
    /// Cluster centers (fitted)
    centers: Option<Array2<f64>>,
}

impl KMeansSampling {
    /// Create a new k-means sampling strategy
    pub fn new(n_iterations: usize, random_state: Option<u64>) -> Self {
        Self {
            n_iterations,
            random_state,
            centers: None,
        }
    }
}

impl SamplingStrategy for KMeansSampling {
    fn sample(&self, data: &Array2<f64>, n_samples: usize) -> Result<Vec<usize>, SklearsError> {
        use scirs2_core::random::seeded_rng;

        let (n_rows, n_features) = data.dim();
        if n_samples > n_rows {
            return Err(SklearsError::InvalidInput(format!(
                "Cannot sample {} points from {} samples",
                n_samples, n_rows
            )));
        }

        let mut rng = seeded_rng(self.random_state.unwrap_or(42));

        // Initialize centers randomly
        let mut centers = Array2::zeros((n_samples, n_features));
        let mut initial_indices: Vec<usize> = (0..n_rows).collect();
        for i in 0..n_samples {
            let idx = rng.gen_range(0..initial_indices.len());
            let sample_idx = initial_indices.swap_remove(idx);
            centers.row_mut(i).assign(&data.row(sample_idx));
        }

        // K-means iterations
        let mut assignments = vec![0; n_rows];
        for _ in 0..self.n_iterations {
            // Assign points to nearest center
            for i in 0..n_rows {
                let point = data.row(i);
                let mut min_dist = f64::INFINITY;
                let mut best_cluster = 0;

                for j in 0..n_samples {
                    let center = centers.row(j);
                    let dist: f64 = point
                        .iter()
                        .zip(center.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();

                    if dist < min_dist {
                        min_dist = dist;
                        best_cluster = j;
                    }
                }
                assignments[i] = best_cluster;
            }

            // Update centers
            let mut counts = vec![0; n_samples];
            centers.fill(0.0);

            for i in 0..n_rows {
                let cluster = assignments[i];
                let point = data.row(i);
                for (j, &val) in point.iter().enumerate() {
                    centers[[cluster, j]] += val;
                }
                counts[cluster] += 1;
            }

            for j in 0..n_samples {
                if counts[j] > 0 {
                    for k in 0..n_features {
                        centers[[j, k]] /= counts[j] as f64;
                    }
                }
            }
        }

        // Find nearest point to each center
        let mut selected_indices = Vec::with_capacity(n_samples);
        for center_idx in 0..n_samples {
            let center = centers.row(center_idx);
            let mut min_dist = f64::INFINITY;
            let mut best_idx = 0;

            for i in 0..n_rows {
                let point = data.row(i);
                let dist: f64 = point
                    .iter()
                    .zip(center.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();

                if dist < min_dist {
                    min_dist = dist;
                    best_idx = i;
                }
            }
            selected_indices.push(best_idx);
        }

        Ok(selected_indices)
    }

    fn name(&self) -> &str {
        "KMeansSampling"
    }

    fn requires_fitting(&self) -> bool {
        true
    }

    fn fit(&mut self, data: &Array2<f64>) -> Result<(), SklearsError> {
        // Store centers for potential reuse
        let _ = data;
        Ok(())
    }
}

/// Kernel alignment quality metric
#[derive(Debug, Clone)]
pub struct KernelAlignmentMetric;

impl ApproximationQuality for KernelAlignmentMetric {
    fn compute(
        &self,
        exact_kernel: &Array2<f64>,
        approx_kernel: &Array2<f64>,
    ) -> Result<f64, SklearsError> {
        let (n1, m1) = exact_kernel.dim();
        let (n2, m2) = approx_kernel.dim();

        if n1 != n2 || m1 != m2 {
            return Err(SklearsError::InvalidInput(
                "Kernel matrices must have the same shape".to_string(),
            ));
        }

        // Compute Frobenius inner product
        let mut inner_product = 0.0;
        let mut exact_norm = 0.0;
        let mut approx_norm = 0.0;

        for i in 0..n1 {
            for j in 0..m1 {
                let exact_val = exact_kernel[[i, j]];
                let approx_val = approx_kernel[[i, j]];
                inner_product += exact_val * approx_val;
                exact_norm += exact_val * exact_val;
                approx_norm += approx_val * approx_val;
            }
        }

        if exact_norm < 1e-10 || approx_norm < 1e-10 {
            return Ok(0.0);
        }

        Ok(inner_product / (exact_norm.sqrt() * approx_norm.sqrt()))
    }

    fn name(&self) -> &str {
        "KernelAlignment"
    }

    fn higher_is_better(&self) -> bool {
        true
    }

    fn acceptable_threshold(&self) -> Option<f64> {
        Some(0.9) // 90% alignment is typically considered good
    }
}

/// Composable kernel that combines multiple kernel methods
#[derive(Debug)]
pub struct CompositeKernelMethod {
    /// List of kernel methods to compose
    methods: Vec<Box<dyn KernelMethod>>,
    /// Combination strategy
    strategy: CombinationStrategy,
}

/// Strategy for combining multiple kernels
#[derive(Debug, Clone, Copy)]
pub enum CombinationStrategy {
    /// Concatenate features from all kernels
    Concatenate,
    /// Average kernel matrices
    Average,
    /// Weighted sum
    WeightedSum,
    /// Product of kernels
    Product,
}

impl CompositeKernelMethod {
    /// Create a new composite kernel method
    pub fn new(strategy: CombinationStrategy) -> Self {
        Self {
            methods: Vec::new(),
            strategy,
        }
    }

    /// Add a kernel method to the composition
    pub fn add_method(&mut self, method: Box<dyn KernelMethod>) {
        self.methods.push(method);
    }

    /// Get the combination strategy
    pub fn strategy(&self) -> CombinationStrategy {
        self.strategy
    }

    /// Get number of methods
    pub fn len(&self) -> usize {
        self.methods.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.methods.is_empty()
    }
}

impl KernelMethod for CompositeKernelMethod {
    fn name(&self) -> &str {
        "CompositeKernel"
    }

    fn n_output_features(&self) -> Option<usize> {
        match self.strategy {
            CombinationStrategy::Concatenate => {
                let mut total = 0;
                for method in &self.methods {
                    if let Some(n) = method.n_output_features() {
                        total += n;
                    } else {
                        return None;
                    }
                }
                Some(total)
            }
            _ => {
                // For other strategies, use the first method's output size
                self.methods.first().and_then(|m| m.n_output_features())
            }
        }
    }

    fn complexity(&self) -> Complexity {
        // Return the worst complexity among all methods
        let mut worst = Complexity::Linear;
        for method in &self.methods {
            let c = method.complexity();
            worst = match (worst, c.clone()) {
                (Complexity::Cubic, _) | (_, Complexity::Cubic) => Complexity::Cubic,
                (Complexity::QuadraticSamples, _) | (_, Complexity::QuadraticSamples) => {
                    Complexity::QuadraticSamples
                }
                (Complexity::QuadraticFeatures, _) | (_, Complexity::QuadraticFeatures) => {
                    Complexity::QuadraticFeatures
                }
                _ => c,
            };
        }
        worst
    }

    fn error_bound(&self) -> Option<ErrorBound> {
        // Combine error bounds (if available)
        // For simplicity, return None if any method doesn't have a bound
        None
    }

    fn supports_kernel(&self, kernel_type: KernelType) -> bool {
        // Check if any method supports this kernel type
        self.methods.iter().any(|m| m.supports_kernel(kernel_type))
    }

    fn supported_kernels(&self) -> Vec<KernelType> {
        let mut kernels = Vec::new();
        for method in &self.methods {
            for kernel in method.supported_kernels() {
                if !kernels.contains(&kernel) {
                    kernels.push(kernel);
                }
            }
        }
        kernels
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_complexity_description() {
        let c = Complexity::Linear;
        assert!(c.description().contains("Linear"));

        let c = Complexity::QuasiLinear;
        assert!(c.description().contains("Quasi-linear"));
    }

    #[test]
    fn test_kernel_type_name() {
        assert_eq!(KernelType::RBF.name(), "RBF");
        assert_eq!(KernelType::Polynomial.name(), "Polynomial");
    }

    #[test]
    fn test_uniform_sampling() {
        let strategy = UniformSampling::new(Some(42));
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let indices = strategy.sample(&data, 2).unwrap();
        assert_eq!(indices.len(), 2);
        assert!(indices[0] < 4);
        assert!(indices[1] < 4);
    }

    #[test]
    fn test_kmeans_sampling() {
        let strategy = KMeansSampling::new(5, Some(42));
        let data = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [5.0, 5.0],
            [5.1, 5.1],
            [9.0, 9.0],
            [9.1, 9.1]
        ];

        let indices = strategy.sample(&data, 3).unwrap();
        assert_eq!(indices.len(), 3);
    }

    #[test]
    fn test_kernel_alignment_metric() {
        let metric = KernelAlignmentMetric;
        let exact = array![[1.0, 0.5], [0.5, 1.0]];
        let approx = array![[1.0, 0.6], [0.6, 1.0]];

        let alignment = metric.compute(&exact, &approx).unwrap();
        assert!(alignment > 0.9 && alignment <= 1.0);
        assert!(metric.higher_is_better());
    }

    #[test]
    fn test_composite_kernel_method() {
        let composite = CompositeKernelMethod::new(CombinationStrategy::Concatenate);
        assert!(composite.is_empty());
        assert_eq!(composite.len(), 0);
    }

    #[test]
    fn test_bound_type() {
        let bound = ErrorBound {
            bound_type: BoundType::Probabilistic,
            error: 0.1,
            confidence: Some(0.95),
            description: "Test bound".to_string(),
        };

        assert_eq!(bound.bound_type, BoundType::Probabilistic);
        assert_eq!(bound.error, 0.1);
    }
}
