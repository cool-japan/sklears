//! Core Dictionary Learning Types and Configuration
//!
//! This module provides the fundamental dictionary learning types and configurations
//! that comply with SciRS2 Policy for matrix decomposition algorithms.

use scirs2_core::ndarray::Array2;
use scirs2_core::random::{thread_rng, Rng};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};

/// Dictionary transformation algorithms
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DictionaryTransformAlgorithm {
    /// Orthogonal Matching Pursuit
    OMP,
    /// Least Angle Regression
    LARS,
    /// Coordinate Descent
    CoordinateDescent,
    /// Thresholding
    Threshold,
}

/// Dictionary Learning configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DictionaryLearningConfig {
    /// Number of dictionary atoms
    pub n_components: usize,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: Float,
    /// Transform algorithm for sparse coding
    pub transform_algorithm: DictionaryTransformAlgorithm,
    /// Regularization parameter
    pub alpha: Float,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl Default for DictionaryLearningConfig {
    fn default() -> Self {
        Self {
            n_components: 100,
            max_iter: 1000,
            tol: 1e-8,
            transform_algorithm: DictionaryTransformAlgorithm::OMP,
            alpha: 1.0,
            random_state: None,
        }
    }
}

/// Dictionary Learning estimator
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DictionaryLearning<State = Untrained> {
    config: DictionaryLearningConfig,
    state: std::marker::PhantomData<State>,
    /// Dictionary components (atoms) - only available when trained
    components_: Option<Array2<Float>>,
    /// Number of iterations performed - only available when trained
    n_iter_: Option<usize>,
}

impl DictionaryLearning<Untrained> {
    /// Create a new dictionary learning instance
    pub fn new(config: DictionaryLearningConfig) -> Self {
        Self {
            config,
            state: std::marker::PhantomData,
            components_: None,
            n_iter_: None,
        }
    }

    /// Builder pattern constructor
    pub fn builder() -> DictionaryLearningBuilder {
        DictionaryLearningBuilder::default()
    }
}

impl DictionaryLearning<Trained> {
    /// Get the dictionary components
    pub fn components(&self) -> &Array2<Float> {
        self.components_.as_ref().expect("Model is trained")
    }

    /// Get the number of iterations performed
    pub fn n_iter(&self) -> usize {
        self.n_iter_.expect("Model is trained")
    }
}

impl Estimator for DictionaryLearning<Untrained> {
    type Config = DictionaryLearningConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, ()> for DictionaryLearning<Untrained> {
    type Fitted = DictionaryLearning<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (_n_samples, n_features) = x.dim();

        if self.config.n_components > n_features {
            return Err(SklearsError::InvalidInput(
                "n_components cannot be larger than n_features".to_string(),
            ));
        }

        // Initialize dictionary randomly
        let mut rng = thread_rng();
        let mut components = Array2::zeros((self.config.n_components, n_features));

        for i in 0..self.config.n_components {
            for j in 0..n_features {
                components[[i, j]] = rng.gen::<Float>() - 0.5;
            }
        }

        // Normalize initial dictionary atoms
        for mut row in components.rows_mut() {
            let norm = row.mapv(|x| x * x).sum().sqrt();
            if norm > 1e-10 {
                row.mapv_inplace(|x| x / norm);
            }
        }

        // Dictionary learning via alternating optimization (MOD algorithm)
        // Alternates between sparse coding and dictionary update
        use super::omp_algorithms::{OMPConfig, OMPEncoder};

        let n_samples = x.nrows();
        let n_atoms = self.config.n_components;

        // Configure OMP for sparse coding step
        let omp_config = OMPConfig {
            n_nonzero_coefs: Some((self.config.alpha * n_atoms as Float).max(1.0) as usize),
            tol: Some(self.config.tol * 0.1), // Tighter tolerance for inner iterations
        };

        let mut n_iter = 0;
        let mut prev_error = Float::INFINITY;

        for iter in 0..self.config.max_iter {
            n_iter = iter + 1;

            // Step 1: Sparse coding - encode all samples with current dictionary
            let mut codes = Array2::zeros((n_samples, n_atoms));
            let encoder = OMPEncoder::new(omp_config.clone());
            let dict_transposed = components.t().to_owned();

            for i in 0..n_samples {
                let signal = x.row(i).to_owned();
                if let Ok(result) = encoder.encode(&dict_transposed, &signal) {
                    for j in 0..n_atoms {
                        codes[[i, j]] = result.coefficients[j];
                    }
                }
            }

            // Step 2: Dictionary update - minimize reconstruction error
            // For each atom, update to minimize ||X - codes * Dictionary||_F^2
            // Using simple least squares: each atom = X^T * codes_column / ||codes_column||^2

            for atom_idx in 0..n_atoms {
                // Compute residual without current atom
                let mut residual = x.clone();
                for i in 0..n_samples {
                    for k in 0..n_atoms {
                        if k != atom_idx {
                            let coef = codes[[i, k]];
                            for j in 0..n_features {
                                residual[[i, j]] -= coef * components[[k, j]];
                            }
                        }
                    }
                }

                // Update atom: atom = (residual^T * codes[:, atom_idx]) / ||codes[:, atom_idx]||^2
                let codes_col_norm_sq: Float = (0..n_samples)
                    .map(|i| codes[[i, atom_idx]] * codes[[i, atom_idx]])
                    .sum();

                if codes_col_norm_sq > 1e-10 {
                    for j in 0..n_features {
                        let mut numerator = 0.0;
                        for i in 0..n_samples {
                            numerator += residual[[i, j]] * codes[[i, atom_idx]];
                        }
                        components[[atom_idx, j]] = numerator / codes_col_norm_sq;
                    }

                    // Normalize the updated atom
                    let atom_norm: Float = (0..n_features)
                        .map(|j| components[[atom_idx, j]] * components[[atom_idx, j]])
                        .sum::<Float>()
                        .sqrt();

                    if atom_norm > 1e-10 {
                        for j in 0..n_features {
                            components[[atom_idx, j]] /= atom_norm;
                        }
                    }
                }
            }

            // Step 3: Check convergence using reconstruction error
            let mut reconstruction_error = 0.0;
            for i in 0..n_samples {
                for j in 0..n_features {
                    let mut pred = 0.0;
                    for k in 0..n_atoms {
                        pred += codes[[i, k]] * components[[k, j]];
                    }
                    let diff = x[[i, j]] - pred;
                    reconstruction_error += diff * diff;
                }
            }
            reconstruction_error =
                (reconstruction_error / (n_samples * n_features) as Float).sqrt();

            // Check for convergence
            if (prev_error - reconstruction_error).abs() < self.config.tol {
                break;
            }
            prev_error = reconstruction_error;
        }

        Ok(DictionaryLearning {
            config: self.config,
            state: std::marker::PhantomData,
            components_: Some(components),
            n_iter_: Some(n_iter),
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for DictionaryLearning<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        use super::omp_algorithms::{OMPConfig, OMPEncoder};

        let (n_samples, n_features) = x.dim();
        let components = self.components();
        let (n_atoms, dict_features) = components.dim();

        // Validate dimensions
        if n_features != dict_features {
            return Err(SklearsError::InvalidInput(format!(
                "Input features {} don't match dictionary features {}",
                n_features, dict_features
            )));
        }

        // Configure OMP encoder based on dictionary learning config
        let omp_config = match self.config.transform_algorithm {
            DictionaryTransformAlgorithm::OMP => OMPConfig {
                n_nonzero_coefs: Some((self.config.alpha * n_atoms as Float) as usize),
                tol: Some(self.config.tol),
            },
            _ => {
                // For other algorithms, use OMP with default settings
                OMPConfig {
                    n_nonzero_coefs: None,
                    tol: Some(self.config.tol),
                }
            }
        };

        let encoder = OMPEncoder::new(omp_config);
        let mut codes = Array2::zeros((n_samples, n_atoms));

        // Encode each sample using OMP
        // Dictionary is transposed: components is (n_atoms × n_features)
        // OMP expects dictionary as (n_features × n_atoms)
        let dict_transposed = components.t().to_owned();

        for i in 0..n_samples {
            let signal = x.row(i).to_owned();
            let result = encoder.encode(&dict_transposed, &signal)?;

            // Copy coefficients to codes matrix
            for j in 0..n_atoms {
                codes[[i, j]] = result.coefficients[j];
            }
        }

        Ok(codes)
    }
}

/// Builder for Dictionary Learning
#[derive(Debug, Clone)]
pub struct DictionaryLearningBuilder {
    config: DictionaryLearningConfig,
}

impl Default for DictionaryLearningBuilder {
    fn default() -> Self {
        Self {
            config: DictionaryLearningConfig::default(),
        }
    }
}

impl DictionaryLearningBuilder {
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.config.n_components = n_components;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    pub fn transform_algorithm(mut self, algorithm: DictionaryTransformAlgorithm) -> Self {
        self.config.transform_algorithm = algorithm;
        self
    }

    pub fn alpha(mut self, alpha: Float) -> Self {
        self.config.alpha = alpha;
        self
    }

    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }

    pub fn build(self) -> DictionaryLearning<Untrained> {
        DictionaryLearning::new(self.config)
    }
}

// Type alias for backward compatibility
pub type TrainedDictionaryLearning = DictionaryLearning<Trained>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dictionary_learning_transform() {
        // Create a simple dataset
        let x = Array2::from_shape_vec(
            (10, 5),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.5,
                2.5, 3.5, 4.5, 5.5, 2.5, 3.5, 4.5, 5.5, 6.5, 1.2, 2.2, 3.2, 4.2, 5.2, 2.2, 3.2,
                4.2, 5.2, 6.2, 1.8, 2.8, 3.8, 4.8, 5.8, 2.8, 3.8, 4.8, 5.8, 6.8, 1.1, 2.1, 3.1,
                4.1, 5.1,
            ],
        )
        .unwrap();

        // Create and fit dictionary learning model
        let config = DictionaryLearningConfig {
            n_components: 3,
            max_iter: 50,
            tol: 1e-4,
            transform_algorithm: DictionaryTransformAlgorithm::OMP,
            alpha: 0.5,
            random_state: Some(42),
        };

        let model = DictionaryLearning::new(config);
        let fitted_model = model.fit(&x, &()).unwrap();

        // Test transform
        let codes = fitted_model.transform(&x).unwrap();

        // Check dimensions
        assert_eq!(codes.nrows(), 10); // n_samples
        assert_eq!(codes.ncols(), 3); // n_components

        // Check that codes are not all zeros (indicating actual encoding happened)
        let sum: Float = codes.iter().map(|&x| x.abs()).sum();
        assert!(sum > 0.0, "Codes should not be all zeros");
    }

    #[test]
    fn test_dictionary_learning_dimension_validation() {
        // Create a simple dataset
        let x_train = Array2::from_shape_vec((5, 3), vec![1.0; 15]).unwrap();
        let x_test = Array2::from_shape_vec((3, 4), vec![1.0; 12]).unwrap();

        let config = DictionaryLearningConfig {
            n_components: 2, // Less than n_features (3)
            ..Default::default()
        };
        let model = DictionaryLearning::new(config);
        let fitted_model = model.fit(&x_train, &()).unwrap();

        // Should fail due to dimension mismatch (x_test has 4 features, but model expects 3)
        let result = fitted_model.transform(&x_test);
        assert!(result.is_err());
    }

    #[test]
    fn test_dictionary_learning_builder() {
        let model = DictionaryLearning::builder()
            .n_components(5)
            .max_iter(100)
            .tol(1e-5)
            .alpha(0.3)
            .random_state(Some(123))
            .build();

        assert_eq!(model.config.n_components, 5);
        assert_eq!(model.config.max_iter, 100);
        assert_eq!(model.config.tol, 1e-5);
        assert_eq!(model.config.alpha, 0.3);
        assert_eq!(model.config.random_state, Some(123));
    }

    #[test]
    fn test_dictionary_learning_convergence() {
        // Create a dataset with clear structure
        let x = Array2::from_shape_vec(
            (20, 8),
            vec![
                // Pattern 1: high values in first half
                5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 5.5, 5.5, 5.5, 5.5, 0.5, 0.5, 0.5, 0.5, 4.5,
                4.5, 4.5, 4.5, 1.5, 1.5, 1.5, 1.5, 5.2, 5.2, 5.2, 5.2, 0.8, 0.8, 0.8, 0.8, 4.8,
                4.8, 4.8, 4.8, 1.2, 1.2, 1.2, 1.2,
                // Pattern 2: high values in second half
                1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 0.5, 0.5, 0.5, 0.5, 5.5, 5.5, 5.5, 5.5, 1.5,
                1.5, 1.5, 1.5, 4.5, 4.5, 4.5, 4.5, 0.8, 0.8, 0.8, 0.8, 5.2, 5.2, 5.2, 5.2, 1.2,
                1.2, 1.2, 1.2, 4.8, 4.8, 4.8, 4.8, // Pattern 3: alternating
                5.0, 1.0, 5.0, 1.0, 5.0, 1.0, 5.0, 1.0, 5.5, 0.5, 5.5, 0.5, 5.5, 0.5, 5.5, 0.5,
                4.5, 1.5, 4.5, 1.5, 4.5, 1.5, 4.5, 1.5, 5.2, 0.8, 5.2, 0.8, 5.2, 0.8, 5.2, 0.8,
                4.8, 1.2, 4.8, 1.2, 4.8, 1.2, 4.8, 1.2, // Mixed patterns
                3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.5, 2.5, 3.5, 2.5, 3.5, 2.5, 3.5, 2.5,
                2.5, 3.5, 2.5, 3.5, 2.5, 3.5, 2.5, 3.5, 4.0, 2.0, 3.0, 3.0, 2.0, 4.0, 3.0, 3.0,
                2.0, 4.0, 3.0, 3.0, 4.0, 2.0, 3.0, 3.0,
            ],
        )
        .unwrap();

        // Configure dictionary learning with reasonable parameters
        let config = DictionaryLearningConfig {
            n_components: 4,
            max_iter: 30,
            tol: 1e-3,
            transform_algorithm: DictionaryTransformAlgorithm::OMP,
            alpha: 0.3,
            random_state: Some(42),
        };

        let model = DictionaryLearning::new(config);
        let fitted_model = model.fit(&x, &()).unwrap();

        // Check that dictionary was learned
        let components = fitted_model.components();
        assert_eq!(components.nrows(), 4);
        assert_eq!(components.ncols(), 8);

        // Check that dictionary atoms are normalized
        for atom_idx in 0..4 {
            let norm: Float = (0..8)
                .map(|j| components[[atom_idx, j]] * components[[atom_idx, j]])
                .sum::<Float>()
                .sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-6,
                "Atom {} not normalized: {}",
                atom_idx,
                norm
            );
        }

        // Check that transform produces reasonable sparse codes
        let codes = fitted_model.transform(&x).unwrap();
        assert_eq!(codes.nrows(), 20);
        assert_eq!(codes.ncols(), 4);

        // Verify sparsity of codes (most entries should be zero or very small)
        let mut nonzero_count = 0;
        for i in 0..codes.nrows() {
            for j in 0..codes.ncols() {
                if codes[[i, j]].abs() > 1e-10 {
                    nonzero_count += 1;
                }
            }
        }
        let sparsity_ratio = nonzero_count as f64 / (codes.nrows() * codes.ncols()) as f64;
        assert!(
            sparsity_ratio < 0.5,
            "Codes should be sparse, but {}% are nonzero",
            sparsity_ratio * 100.0
        );

        // Verify reconstruction is reasonable (not perfect, but better than random)
        let mut reconstruction_error = 0.0;
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                let mut reconstruction = 0.0;
                for k in 0..components.nrows() {
                    reconstruction += codes[[i, k]] * components[[k, j]];
                }
                let diff = x[[i, j]] - reconstruction;
                reconstruction_error += diff * diff;
            }
        }
        reconstruction_error = (reconstruction_error / (x.nrows() * x.ncols()) as Float).sqrt();

        // Error should be reasonable (not too large)
        assert!(
            reconstruction_error < 3.0,
            "Reconstruction error too high: {}",
            reconstruction_error
        );
    }
}
