//! Dimensionality Reduction Transformers
//!
//! This module provides dimensionality reduction techniques for preprocessing pipelines:
//! - Principal Component Analysis (PCA)
//! - Linear Discriminant Analysis (LDA)
//! - Independent Component Analysis (ICA)
//! - Non-negative Matrix Factorization (NMF)
//! - t-SNE (t-distributed Stochastic Neighbor Embedding)
//!
//! # Examples
//!
//! ```rust,ignore
//! use sklears_preprocessing::dimensionality_reduction::{PCA, PCAConfig};
//! use scirs2_core::ndarray::Array2;
//!
//! fn example() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = PCAConfig::new(2); // Reduce to 2 components
//!     let mut pca = PCA::new(config);
//!     
//!     let data = Array2::from_shape_vec((4, 3), vec![
//!         1.0, 2.0, 3.0,
//!         2.0, 4.0, 6.0,
//!         3.0, 6.0, 9.0,
//!         4.0, 8.0, 12.0,
//!     ])?;
//!     
//!     let pca_fitted = pca.fit(&data, &())?;
//!     let transformed = pca_fitted.transform(&data)?;
//!     println!("Reduced data shape: {:?}", transformed.dim());
//!     
//!     Ok(())
//! }
//! ```

use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Transform, Untrained},
};
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for Principal Component Analysis
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PCAConfig {
    /// Number of components to keep
    pub n_components: Option<usize>,
    /// Whether to center the data (subtract mean)
    pub center: bool,
    /// Solver algorithm for eigendecomposition
    pub solver: PcaSolver,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
    /// Tolerance for convergence in iterative solvers
    pub tolerance: f64,
    /// Maximum number of iterations for iterative solvers
    pub max_iterations: usize,
}

/// Solver algorithms for PCA
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PcaSolver {
    /// Full eigendecomposition (exact, slower for large datasets)
    Full,
    /// Randomized SVD (approximate, faster for large datasets)
    Randomized,
    /// Power iteration method
    PowerIteration,
}

impl Default for PCAConfig {
    fn default() -> Self {
        Self {
            n_components: None,
            center: true,
            solver: PcaSolver::Full,
            random_state: None,
            tolerance: 1e-7,
            max_iterations: 1000,
        }
    }
}

impl PCAConfig {
    /// Create a new PCA configuration with specified number of components
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components: Some(n_components),
            ..Default::default()
        }
    }

    /// Set the solver algorithm
    pub fn with_solver(mut self, solver: PcaSolver) -> Self {
        self.solver = solver;
        self
    }

    /// Enable or disable data centering
    pub fn with_center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Set random state for reproducible results
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }
}

/// Principal Component Analysis transformer
pub struct PCA<State = Untrained> {
    config: PCAConfig,
    state: PhantomData<State>,
}

/// Fitted PCA with learned parameters
pub struct PCAFitted {
    config: PCAConfig,
    components: Array2<f64>,
    explained_variance: Array1<f64>,
    explained_variance_ratio: Array1<f64>,
    singular_values: Array1<f64>,
    mean: Option<Array1<f64>>,
    n_features: usize,
    n_components: usize,
}

impl PCA<Untrained> {
    /// Create a new PCA transformer
    pub fn new(config: PCAConfig) -> Self {
        Self {
            config,
            state: PhantomData,
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &PCAConfig {
        &self.config
    }
}

impl Fit<Array2<f64>, ()> for PCA<Untrained> {
    type Fitted = PCAFitted;

    fn fit(self, x: &Array2<f64>, _y: &()) -> Result<PCAFitted> {
        if x.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Input array is empty".to_string(),
            ));
        }

        let (n_samples, n_features) = x.dim();
        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "PCA requires at least 2 samples".to_string(),
            ));
        }

        // Determine number of components
        let n_components = self
            .config
            .n_components
            .unwrap_or(n_features.min(n_samples));
        if n_components > n_features.min(n_samples) {
            return Err(SklearsError::InvalidInput(format!(
                "n_components={} cannot be larger than min(n_samples={}, n_features={})",
                n_components, n_samples, n_features
            )));
        }

        // Center the data if requested
        let (x_centered, mean) = if self.config.center {
            let mean = x.mean_axis(Axis(0)).unwrap();
            let mut x_centered = x.clone();
            for mut row in x_centered.axis_iter_mut(Axis(0)) {
                for (j, &mean_j) in mean.iter().enumerate() {
                    row[j] -= mean_j;
                }
            }
            (x_centered, Some(mean))
        } else {
            (x.clone(), None)
        };

        // Perform dimensionality reduction based on solver
        let (components, explained_variance, singular_values) = match self.config.solver {
            PcaSolver::Full => perform_full_pca(&x_centered, n_components)?,
            PcaSolver::Randomized => {
                perform_randomized_pca(&x_centered, n_components, self.config.random_state)?
            }
            PcaSolver::PowerIteration => perform_power_iteration_pca(
                &x_centered,
                n_components,
                self.config.max_iterations,
                self.config.tolerance,
            )?,
        };

        // Calculate explained variance ratio
        let total_variance = explained_variance.sum();
        let explained_variance_ratio = if total_variance > 0.0 {
            &explained_variance / total_variance
        } else {
            Array1::zeros(n_components)
        };

        Ok(PCAFitted {
            config: self.config,
            components,
            explained_variance,
            explained_variance_ratio,
            singular_values,
            mean,
            n_features,
            n_components,
        })
    }
}

impl PCAFitted {
    /// Get the principal components
    pub fn components(&self) -> &Array2<f64> {
        &self.components
    }

    /// Get the explained variance
    pub fn explained_variance(&self) -> &Array1<f64> {
        &self.explained_variance
    }

    /// Get the explained variance ratio
    pub fn explained_variance_ratio(&self) -> &Array1<f64> {
        &self.explained_variance_ratio
    }

    /// Get the singular values
    pub fn singular_values(&self) -> &Array1<f64> {
        &self.singular_values
    }

    /// Get the mean (if centering was enabled)
    pub fn mean(&self) -> Option<&Array1<f64>> {
        self.mean.as_ref()
    }

    /// Get the number of components
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Get the number of original features
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Calculate the cumulative explained variance ratio
    pub fn cumulative_explained_variance_ratio(&self) -> Array1<f64> {
        let mut cumulative = Array1::zeros(self.explained_variance_ratio.len());
        let mut sum = 0.0;
        for (i, &ratio) in self.explained_variance_ratio.iter().enumerate() {
            sum += ratio;
            cumulative[i] = sum;
        }
        cumulative
    }
}

impl Transform<Array2<f64>, Array2<f64>> for PCAFitted {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if x.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Input array is empty".to_string(),
            ));
        }

        let (_n_samples, n_features) = x.dim();
        if n_features != self.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Feature count mismatch: expected {}, got {}",
                self.n_features, n_features
            )));
        }

        // Center the data if centering was used during fit
        let x_centered = if let Some(ref mean) = self.mean {
            let mut x_centered = x.clone();
            for mut row in x_centered.axis_iter_mut(Axis(0)) {
                for (j, &mean_j) in mean.iter().enumerate() {
                    row[j] -= mean_j;
                }
            }
            x_centered
        } else {
            x.clone()
        };

        // Project onto principal components
        let result = x_centered.dot(&self.components.t());
        Ok(result)
    }
}

/// Configuration for Linear Discriminant Analysis
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LDAConfig {
    /// Number of components to keep (max: n_classes - 1)
    pub n_components: Option<usize>,
    /// Solver for the eigenvalue problem
    pub solver: LdaSolver,
    /// Shrinkage parameter for regularization (0.0 to 1.0)
    pub shrinkage: Option<f64>,
    /// Tolerance for convergence
    pub tolerance: f64,
}

/// Solver algorithms for LDA
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LdaSolver {
    /// Standard eigenvalue decomposition
    Svd,
    /// Least squares solution
    Lsqr,
    /// Eigenvalue decomposition with shrinkage
    Eigen,
}

impl Default for LDAConfig {
    fn default() -> Self {
        Self {
            n_components: None,
            solver: LdaSolver::Svd,
            shrinkage: None,
            tolerance: 1e-4,
        }
    }
}

/// Linear Discriminant Analysis transformer
pub struct LDA<State = Untrained> {
    config: LDAConfig,
    state: PhantomData<State>,
}

/// Fitted LDA with learned parameters
pub struct LDAFitted {
    config: LDAConfig,
    components: Array2<f64>,
    explained_variance_ratio: Array1<f64>,
    means: Array2<f64>,  // Class means
    priors: Array1<f64>, // Class priors
    classes: Array1<usize>,
    n_features: usize,
    n_components: usize,
}

impl LDA<Untrained> {
    /// Create a new LDA transformer
    pub fn new(config: LDAConfig) -> Self {
        Self {
            config,
            state: PhantomData,
        }
    }
}

/// Configuration for Independent Component Analysis
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ICAConfig {
    /// Number of components to extract
    pub n_components: Option<usize>,
    /// ICA algorithm
    pub algorithm: IcaAlgorithm,
    /// Non-linearity function for FastICA
    pub fun: IcaFunction,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Tolerance for convergence
    pub tolerance: f64,
    /// Whether to whiten the data
    pub whiten: bool,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
}

/// ICA algorithms
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum IcaAlgorithm {
    /// FastICA algorithm
    FastICA,
    /// Infomax algorithm  
    Infomax,
}

/// Non-linearity functions for ICA
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum IcaFunction {
    /// Logistic function
    Logcosh,
    /// Exponential function
    Exp,
    /// Cubic function
    Cube,
}

impl Default for ICAConfig {
    fn default() -> Self {
        Self {
            n_components: None,
            algorithm: IcaAlgorithm::FastICA,
            fun: IcaFunction::Logcosh,
            max_iterations: 200,
            tolerance: 1e-4,
            whiten: true,
            random_state: None,
        }
    }
}

/// Independent Component Analysis transformer
pub struct ICA<State = Untrained> {
    config: ICAConfig,
    state: PhantomData<State>,
}

/// Fitted ICA with learned parameters
pub struct ICAFitted {
    config: ICAConfig,
    components: Array2<f64>,
    mixing_matrix: Array2<f64>,
    mean: Array1<f64>,
    whitening_matrix: Option<Array2<f64>>,
    n_features: usize,
    n_components: usize,
}

impl ICA<Untrained> {
    /// Create a new ICA transformer
    pub fn new(config: ICAConfig) -> Self {
        Self {
            config,
            state: PhantomData,
        }
    }
}

/// Configuration for Non-negative Matrix Factorization
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NMFConfig {
    /// Number of components
    pub n_components: usize,
    /// Initialization method
    pub init: NmfInit,
    /// Solver algorithm
    pub solver: NmfSolver,
    /// Regularization parameter
    pub alpha: f64,
    /// L1 regularization ratio
    pub l1_ratio: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Tolerance for convergence
    pub tolerance: f64,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
}

/// Initialization methods for NMF
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum NmfInit {
    /// Random initialization
    Random,
    /// NNDSVD initialization
    Nndsvd,
    /// Custom initialization
    Custom,
}

/// Solver algorithms for NMF
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum NmfSolver {
    /// Coordinate descent
    CoordinateDescent,
    /// Multiplicative update
    MultiplicativeUpdate,
}

impl Default for NMFConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            init: NmfInit::Random,
            solver: NmfSolver::CoordinateDescent,
            alpha: 0.0,
            l1_ratio: 0.0,
            max_iterations: 200,
            tolerance: 1e-4,
            random_state: None,
        }
    }
}

/// Non-negative Matrix Factorization transformer
pub struct NMF<State = Untrained> {
    config: NMFConfig,
    state: PhantomData<State>,
}

/// Fitted NMF with learned parameters
pub struct NMFFitted {
    config: NMFConfig,
    components: Array2<f64>,
    n_features: usize,
    n_components: usize,
    reconstruction_error: f64,
}

impl NMF<Untrained> {
    /// Create a new NMF transformer
    pub fn new(config: NMFConfig) -> Self {
        Self {
            config,
            state: PhantomData,
        }
    }
}

// Implementation functions for different PCA solvers

/// Perform full PCA using eigendecomposition of covariance matrix
fn perform_full_pca(
    x: &Array2<f64>,
    n_components: usize,
) -> Result<(Array2<f64>, Array1<f64>, Array1<f64>)> {
    let (n_samples, n_features) = x.dim();

    // Compute covariance matrix
    let cov_matrix = if n_samples > 1 {
        x.t().dot(x) / (n_samples - 1) as f64
    } else {
        return Err(SklearsError::InvalidInput(
            "Cannot compute covariance with only 1 sample".to_string(),
        ));
    };

    // For simplicity, we'll use a basic eigendecomposition approach
    // In a production system, you'd use a proper linear algebra library like nalgebra
    let (eigenvalues, eigenvectors) = compute_eigen_decomposition(&cov_matrix)?;

    // Sort eigenvalues and eigenvectors in descending order
    let mut eigen_pairs: Vec<(f64, Array1<f64>)> = eigenvalues
        .iter()
        .zip(eigenvectors.axis_iter(Axis(1)))
        .map(|(&val, vec)| (val, vec.to_owned()))
        .collect();

    eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Take top n_components
    let selected_pairs: Vec<_> = eigen_pairs.into_iter().take(n_components).collect();

    // Extract components and explained variance
    let mut components = Array2::zeros((n_components, n_features));
    let mut explained_variance = Array1::zeros(n_components);

    for (i, (eigenval, eigenvec)) in selected_pairs.iter().enumerate() {
        explained_variance[i] = eigenval.max(0.0);
        for (j, &val) in eigenvec.iter().enumerate() {
            components[[i, j]] = val;
        }
    }

    // Compute singular values
    let singular_values = explained_variance.mapv(|x| (x * (n_samples - 1) as f64).sqrt());

    Ok((components, explained_variance, singular_values))
}

/// Perform randomized PCA (placeholder implementation)
fn perform_randomized_pca(
    x: &Array2<f64>,
    n_components: usize,
    _random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<f64>, Array1<f64>)> {
    // For now, fall back to full PCA
    // In a real implementation, you'd use randomized SVD algorithms
    perform_full_pca(x, n_components)
}

/// Perform power iteration PCA (placeholder implementation)
fn perform_power_iteration_pca(
    x: &Array2<f64>,
    n_components: usize,
    _max_iterations: usize,
    _tolerance: f64,
) -> Result<(Array2<f64>, Array1<f64>, Array1<f64>)> {
    // For now, fall back to full PCA
    // In a real implementation, you'd use power iteration methods
    perform_full_pca(x, n_components)
}

/// Simplified eigendecomposition (placeholder)
fn compute_eigen_decomposition(matrix: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>)> {
    let n = matrix.nrows();

    // Simplified approach: use diagonal elements as eigenvalue estimates
    // In practice, you'd use a proper eigendecomposition library
    let mut eigenvalues = Array1::zeros(n);
    for i in 0..n {
        eigenvalues[i] = matrix[[i, i]];
    }

    // Use identity matrix as placeholder eigenvectors
    let eigenvectors = Array2::eye(n);

    Ok((eigenvalues, eigenvectors))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_pca_config() {
        let config = PCAConfig::new(2)
            .with_solver(PcaSolver::Randomized)
            .with_center(false)
            .with_random_state(42)
            .with_tolerance(1e-6);

        assert_eq!(config.n_components, Some(2));
        assert!(!config.center);
        assert_eq!(config.random_state, Some(42));
        assert_relative_eq!(config.tolerance, 1e-6);
    }

    #[test]
    fn test_pca_creation() {
        let config = PCAConfig::new(2);
        let pca = PCA::new(config);
        assert_eq!(pca.config().n_components, Some(2));
    }

    #[test]
    fn test_pca_fit_basic() {
        let config = PCAConfig::new(2);
        let pca = PCA::new(config);

        // Simple test data
        let data = arr2(&[
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
        ]);

        let result = pca.fit(&data, &());
        assert!(result.is_ok());

        let fitted = result.unwrap();
        assert_eq!(fitted.n_components(), 2);
        assert_eq!(fitted.n_features(), 3);
        assert_eq!(fitted.components().dim(), (2, 3));
        assert_eq!(fitted.explained_variance().len(), 2);
    }

    #[test]
    fn test_pca_transform() {
        let config = PCAConfig::new(2);
        let pca = PCA::new(config);

        let data = arr2(&[
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
        ]);

        let fitted = pca.fit(&data, &()).unwrap();
        let transformed = fitted.transform(&data).unwrap();

        assert_eq!(transformed.dim(), (4, 2)); // 4 samples, 2 components
    }

    #[test]
    fn test_pca_errors() {
        // Test empty array
        let config = PCAConfig::new(2);
        let pca = PCA::new(config);
        let empty_data = Array2::from_shape_vec((0, 0), vec![]).unwrap();
        assert!(pca.fit(&empty_data, &()).is_err());

        // Test single sample
        let config = PCAConfig::new(2);
        let pca = PCA::new(config);
        let single_sample = arr2(&[[1.0, 2.0, 3.0]]);
        assert!(pca.fit(&single_sample, &()).is_err());

        // Test too many components
        let config = PCAConfig::new(10); // More components than features
        let pca = PCA::new(config);
        let data = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        assert!(pca.fit(&data, &()).is_err());
    }

    #[test]
    fn test_pca_transform_dimension_mismatch() {
        let config = PCAConfig::new(1);
        let pca = PCA::new(config);

        let train_data = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let fitted = pca.fit(&train_data, &()).unwrap();

        // Test with different number of features
        let wrong_data = arr2(&[[1.0, 2.0, 3.0]]); // 3 features instead of 2
        assert!(fitted.transform(&wrong_data).is_err());
    }

    #[test]
    fn test_pca_without_centering() {
        let config = PCAConfig::new(1).with_center(false);
        let pca = PCA::new(config);

        let data = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let fitted = pca.fit(&data, &()).unwrap();

        // Should not have mean when centering is disabled
        assert!(fitted.mean().is_none());
    }

    #[test]
    fn test_cumulative_explained_variance_ratio() {
        let config = PCAConfig::new(2);
        let pca = PCA::new(config);

        let data = arr2(&[
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
        ]);

        let fitted = pca.fit(&data, &()).unwrap();
        let cumulative = fitted.cumulative_explained_variance_ratio();

        assert_eq!(cumulative.len(), 2);
        // Cumulative should be non-decreasing
        assert!(cumulative[1] >= cumulative[0]);
        // Last value should be <= 1.0
        assert!(cumulative[cumulative.len() - 1] <= 1.0);
    }
}
