//! Multi-output dummy regressor for baseline comparisons
//!
//! This module provides dummy regression for multiple output variables simultaneously,
//! supporting various strategies for handling correlations and dependencies between outputs.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::prelude::*;
use scirs2_core::random::Distribution;
use sklears_core::error::Result;
use sklears_core::traits::{Estimator, Fit, Predict};
use sklears_core::types::{Features, Float};

/// Strategy for making multi-output predictions
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum MultiOutputStrategy {
    /// Independent prediction for each output (ignoring correlations)
    Independent,
    /// Correlated sampling considering output correlations
    Correlated,
    /// Multi-task approach with different strategies per output
    MultiTask(Vec<SingleOutputStrategy>),
    /// Hierarchical prediction with dependency structure
    Hierarchical(Vec<usize>), // Parent indices for each output
    /// Structured output using copula-based dependencies
    Structured,
}

/// Strategy for individual outputs in multi-task approach
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum SingleOutputStrategy {
    /// Mean
    Mean,
    /// Median
    Median,
    /// Normal
    Normal,
    /// Constant
    Constant(Float),
}

/// Multi-output dummy regressor for baseline comparisons
///
/// This regressor can handle multiple target variables simultaneously and provides
/// various strategies for modeling dependencies between outputs.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct MultiOutputDummyRegressor<State = sklears_core::traits::Untrained> {
    /// Strategy to use for multi-output predictions
    pub strategy: MultiOutputStrategy,
    /// Random state for reproducible output
    pub random_state: Option<u64>,
    /// Training state data
    pub(crate) n_outputs_: Option<usize>,
    /// Output means for each target
    pub(crate) output_means_: Option<Array1<Float>>,
    /// Output standard deviations for each target
    pub(crate) output_stds_: Option<Array1<Float>>,
    /// Correlation matrix between outputs
    pub(crate) correlation_matrix_: Option<Array2<Float>>,
    /// Covariance matrix between outputs
    pub(crate) covariance_matrix_: Option<Array2<Float>>,
    /// Cholesky decomposition for correlated sampling
    pub(crate) cholesky_: Option<Array2<Float>>,
    /// Individual output statistics for multi-task
    pub(crate) individual_stats_: Option<Vec<(Float, Float)>>, // (mean, std) pairs
    /// Hierarchical dependency structure
    pub(crate) hierarchy_: Option<Vec<usize>>,
    /// Training targets for structured/copula-based methods
    pub(crate) training_targets_: Option<Array2<Float>>,
    /// Phantom data for state
    pub(crate) _state: std::marker::PhantomData<State>,
}

impl MultiOutputDummyRegressor {
    pub fn new(strategy: MultiOutputStrategy) -> Self {
        Self {
            strategy,
            random_state: None,
            n_outputs_: None,
            output_means_: None,
            output_stds_: None,
            correlation_matrix_: None,
            covariance_matrix_: None,
            cholesky_: None,
            individual_stats_: None,
            hierarchy_: None,
            training_targets_: None,
            _state: std::marker::PhantomData,
        }
    }

    /// Set the random state for reproducible output
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Default for MultiOutputDummyRegressor {
    fn default() -> Self {
        Self::new(MultiOutputStrategy::Independent)
    }
}

impl Estimator for MultiOutputDummyRegressor {
    type Config = ();
    type Error = sklears_core::error::SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Features, Array2<Float>> for MultiOutputDummyRegressor {
    type Fitted = MultiOutputDummyRegressor<sklears_core::traits::Trained>;

    fn fit(self, x: &Features, y: &Array2<Float>) -> Result<Self::Fitted> {
        if x.is_empty() || y.is_empty() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Input cannot be empty".to_string(),
            ));
        }

        if x.nrows() != y.nrows() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Number of samples in X and y must be equal".to_string(),
            ));
        }

        let n_samples = y.nrows();
        let n_outputs = y.ncols();

        if n_outputs == 0 {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Number of outputs must be greater than 0".to_string(),
            ));
        }

        // Calculate basic statistics for each output
        let mut output_means = Array1::zeros(n_outputs);
        let mut output_stds = Array1::zeros(n_outputs);

        for i in 0..n_outputs {
            let column = y.column(i);
            output_means[i] = column.mean().unwrap_or(0.0);

            let variance = if n_samples > 1 {
                column
                    .iter()
                    .map(|&val| (val - output_means[i]).powi(2))
                    .sum::<Float>()
                    / (n_samples - 1) as Float
            } else {
                1.0
            };
            output_stds[i] = variance.sqrt();
        }

        // Initialize strategy-specific parameters
        let mut correlation_matrix = None;
        let mut covariance_matrix = None;
        let mut cholesky = None;
        let mut individual_stats = None;
        let mut hierarchy = None;
        let mut training_targets = None;

        match &self.strategy {
            MultiOutputStrategy::Independent => {
                // No additional computation needed
            }
            MultiOutputStrategy::Correlated => {
                // Calculate correlation and covariance matrices
                let mut corr_matrix = Array2::zeros((n_outputs, n_outputs));
                let mut cov_matrix = Array2::zeros((n_outputs, n_outputs));

                for i in 0..n_outputs {
                    for j in 0..n_outputs {
                        if i == j {
                            corr_matrix[[i, j]] = 1.0;
                            cov_matrix[[i, j]] = output_stds[i] * output_stds[j];
                        } else {
                            // Calculate correlation coefficient
                            let col_i = y.column(i);
                            let col_j = y.column(j);
                            let mean_i = output_means[i];
                            let mean_j = output_means[j];

                            let numerator: Float = col_i
                                .iter()
                                .zip(col_j.iter())
                                .map(|(&vi, &vj)| (vi - mean_i) * (vj - mean_j))
                                .sum();

                            let correlation = if output_stds[i] > 1e-10 && output_stds[j] > 1e-10 {
                                numerator
                                    / ((n_samples - 1) as Float * output_stds[i] * output_stds[j])
                            } else {
                                0.0
                            };

                            corr_matrix[[i, j]] = correlation;
                            cov_matrix[[i, j]] = correlation * output_stds[i] * output_stds[j];
                        }
                    }
                }

                // Add regularization to ensure positive definiteness
                let regularization = 1e-8;
                for i in 0..n_outputs {
                    cov_matrix[[i, i]] += regularization;
                }

                // Compute Cholesky decomposition for sampling
                let chol = cholesky_decomposition(&cov_matrix)?;

                correlation_matrix = Some(corr_matrix);
                covariance_matrix = Some(cov_matrix);
                cholesky = Some(chol);
            }
            MultiOutputStrategy::MultiTask(strategies) => {
                if strategies.len() != n_outputs {
                    return Err(sklears_core::error::SklearsError::InvalidInput(format!(
                        "Number of strategies ({}) must match number of outputs ({})",
                        strategies.len(),
                        n_outputs
                    )));
                }

                // Calculate individual statistics based on strategies
                let mut stats = Vec::with_capacity(n_outputs);
                for (i, strategy) in strategies.iter().enumerate() {
                    let column = y.column(i);
                    let stat = match strategy {
                        SingleOutputStrategy::Mean => (output_means[i], output_stds[i]),
                        SingleOutputStrategy::Median => {
                            let mut sorted: Vec<Float> = column.iter().copied().collect();
                            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            let median = if sorted.len() % 2 == 0 {
                                (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
                            } else {
                                sorted[sorted.len() / 2]
                            };
                            (median, output_stds[i])
                        }
                        SingleOutputStrategy::Normal => (output_means[i], output_stds[i]),
                        SingleOutputStrategy::Constant(value) => (*value, 0.0),
                    };
                    stats.push(stat);
                }
                individual_stats = Some(stats);
            }
            MultiOutputStrategy::Hierarchical(parents) => {
                if parents.len() != n_outputs {
                    return Err(sklears_core::error::SklearsError::InvalidInput(format!(
                        "Number of parent indices ({}) must match number of outputs ({})",
                        parents.len(),
                        n_outputs
                    )));
                }

                // Validate parent indices (skip validation for first output since it's ignored)
                for (i, &parent) in parents.iter().enumerate() {
                    if i > 0 && parent >= i {
                        return Err(sklears_core::error::SklearsError::InvalidInput(format!(
                            "Parent index {} for output {} must be less than output index",
                            parent, i
                        )));
                    }
                }

                hierarchy = Some(parents.clone());
            }
            MultiOutputStrategy::Structured => {
                // Store training targets for copula-based sampling
                training_targets = Some(y.clone());
            }
        }

        Ok(MultiOutputDummyRegressor {
            strategy: self.strategy,
            random_state: self.random_state,
            n_outputs_: Some(n_outputs),
            output_means_: Some(output_means),
            output_stds_: Some(output_stds),
            correlation_matrix_: correlation_matrix,
            covariance_matrix_: covariance_matrix,
            cholesky_: cholesky,
            individual_stats_: individual_stats,
            hierarchy_: hierarchy,
            training_targets_: training_targets,
            _state: std::marker::PhantomData,
        })
    }
}

impl Predict<Features, Array2<Float>> for MultiOutputDummyRegressor<sklears_core::traits::Trained> {
    fn predict(&self, x: &Features) -> Result<Array2<Float>> {
        if x.is_empty() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Input cannot be empty".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_outputs = self.n_outputs_.unwrap();
        let mut predictions = Array2::zeros((n_samples, n_outputs));

        let output_means = self.output_means_.as_ref().unwrap();
        let output_stds = self.output_stds_.as_ref().unwrap();

        match &self.strategy {
            MultiOutputStrategy::Independent => {
                // Each output predicted independently using its mean
                for i in 0..n_outputs {
                    for j in 0..n_samples {
                        predictions[[j, i]] = output_means[i];
                    }
                }
            }
            MultiOutputStrategy::Correlated => {
                // Sample from multivariate normal distribution
                let cholesky = self.cholesky_.as_ref().unwrap();
                let means = output_means;

                let mut rng = if let Some(seed) = self.random_state {
                    StdRng::seed_from_u64(seed)
                } else {
                    StdRng::seed_from_u64(0)
                };

                let normal = Normal::new(0.0, 1.0).map_err(|_| {
                    sklears_core::error::SklearsError::InvalidInput(
                        "Failed to create normal distribution".to_string(),
                    )
                })?;

                for i in 0..n_samples {
                    // Generate independent standard normal variables
                    let mut z = Array1::zeros(n_outputs);
                    for j in 0..n_outputs {
                        z[j] = normal.sample(&mut rng);
                    }

                    // Transform using Cholesky decomposition: x = μ + L * z
                    for j in 0..n_outputs {
                        let mut sum = means[j];
                        for k in 0..=j {
                            sum += cholesky[[j, k]] * z[k];
                        }
                        predictions[[i, j]] = sum;
                    }
                }
            }
            MultiOutputStrategy::MultiTask(strategies) => {
                let stats = self.individual_stats_.as_ref().unwrap();
                let mut rng = if let Some(seed) = self.random_state {
                    StdRng::seed_from_u64(seed)
                } else {
                    StdRng::seed_from_u64(0)
                };

                for (output_idx, (strategy, &(mean, std))) in
                    strategies.iter().zip(stats.iter()).enumerate()
                {
                    match strategy {
                        SingleOutputStrategy::Mean | SingleOutputStrategy::Median => {
                            for sample_idx in 0..n_samples {
                                predictions[[sample_idx, output_idx]] = mean;
                            }
                        }
                        SingleOutputStrategy::Normal => {
                            if std > 1e-10 {
                                let normal_dist = Normal::new(mean, std).map_err(|_| {
                                    sklears_core::error::SklearsError::InvalidInput(
                                        "Invalid normal distribution parameters".to_string(),
                                    )
                                })?;

                                for sample_idx in 0..n_samples {
                                    predictions[[sample_idx, output_idx]] =
                                        normal_dist.sample(&mut rng);
                                }
                            } else {
                                for sample_idx in 0..n_samples {
                                    predictions[[sample_idx, output_idx]] = mean;
                                }
                            }
                        }
                        SingleOutputStrategy::Constant(value) => {
                            for sample_idx in 0..n_samples {
                                predictions[[sample_idx, output_idx]] = *value;
                            }
                        }
                    }
                }
            }
            MultiOutputStrategy::Hierarchical(parents) => {
                let hierarchy = self.hierarchy_.as_ref().unwrap();

                // Predict outputs in order, using parent outputs for dependent variables
                for output_idx in 0..n_outputs {
                    if output_idx == 0 {
                        // First output uses its mean
                        for sample_idx in 0..n_samples {
                            predictions[[sample_idx, output_idx]] = output_means[output_idx];
                        }
                    } else {
                        let parent_idx = hierarchy[output_idx];

                        // Simple linear dependency: child = α + β * parent + noise
                        let alpha = output_means[output_idx] - 0.5 * output_means[parent_idx];
                        let beta = 0.5; // Simple dependency coefficient

                        let mut rng = if let Some(seed) = self.random_state {
                            StdRng::seed_from_u64(seed)
                        } else {
                            StdRng::seed_from_u64(0)
                        };

                        let noise_std = output_stds[output_idx] * 0.5; // Reduced noise due to dependency
                        let normal = Normal::new(0.0, noise_std).map_err(|_| {
                            sklears_core::error::SklearsError::InvalidInput(
                                "Invalid noise distribution parameters".to_string(),
                            )
                        })?;

                        for sample_idx in 0..n_samples {
                            let parent_value = predictions[[sample_idx, parent_idx]];
                            let noise = normal.sample(&mut rng);
                            predictions[[sample_idx, output_idx]] =
                                alpha + beta * parent_value + noise;
                        }
                    }
                }
            }
            MultiOutputStrategy::Structured => {
                // Use empirical copula approach - sample from training data
                let training_targets = self.training_targets_.as_ref().unwrap();
                let mut rng = if let Some(seed) = self.random_state {
                    StdRng::seed_from_u64(seed)
                } else {
                    StdRng::seed_from_u64(0)
                };

                for i in 0..n_samples {
                    // Randomly sample a row from training data
                    let rand_idx = rng.gen_range(0..training_targets.nrows());
                    for j in 0..n_outputs {
                        predictions[[i, j]] = training_targets[[rand_idx, j]];
                    }
                }
            }
        }

        Ok(predictions)
    }
}

impl MultiOutputDummyRegressor<sklears_core::traits::Trained> {
    /// Get the number of outputs
    pub fn n_outputs(&self) -> usize {
        self.n_outputs_.unwrap()
    }

    /// Get the output means
    pub fn output_means(&self) -> &Array1<Float> {
        self.output_means_.as_ref().unwrap()
    }

    /// Get the output standard deviations
    pub fn output_stds(&self) -> &Array1<Float> {
        self.output_stds_.as_ref().unwrap()
    }

    /// Get the correlation matrix (if available)
    pub fn correlation_matrix(&self) -> Option<&Array2<Float>> {
        self.correlation_matrix_.as_ref()
    }

    /// Get the covariance matrix (if available)
    pub fn covariance_matrix(&self) -> Option<&Array2<Float>> {
        self.covariance_matrix_.as_ref()
    }

    /// Get the hierarchical structure (if available)
    pub fn hierarchy(&self) -> Option<&Vec<usize>> {
        self.hierarchy_.as_ref()
    }
}

/// Compute Cholesky decomposition for a symmetric positive definite matrix
fn cholesky_decomposition(matrix: &Array2<Float>) -> Result<Array2<Float>> {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err(sklears_core::error::SklearsError::InvalidInput(
            "Matrix must be square for Cholesky decomposition".to_string(),
        ));
    }

    let mut l = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            if i == j {
                // Diagonal elements
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[[j, k]] * l[[j, k]];
                }
                let val = matrix[[j, j]] - sum;
                if val <= 0.0 {
                    return Err(sklears_core::error::SklearsError::InvalidInput(
                        "Matrix is not positive definite".to_string(),
                    ));
                }
                l[[j, j]] = val.sqrt();
            } else {
                // Lower triangular elements
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[[i, k]] * l[[j, k]];
                }
                l[[i, j]] = (matrix[[i, j]] - sum) / l[[j, j]];
            }
        }
    }

    Ok(l)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_multioutput_independent() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();

        let regressor = MultiOutputDummyRegressor::new(MultiOutputStrategy::Independent);
        let fitted = regressor.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_outputs(), 2);

        let output_means = fitted.output_means();
        assert_abs_diff_eq!(output_means[0], 4.0, epsilon = 1e-10); // Mean of [1,3,5,7]
        assert_abs_diff_eq!(output_means[1], 5.0, epsilon = 1e-10); // Mean of [2,4,6,8]

        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions.shape(), &[4, 2]);

        // All predictions should be the means
        for i in 0..4 {
            assert_abs_diff_eq!(predictions[[i, 0]], 4.0, epsilon = 1e-10);
            assert_abs_diff_eq!(predictions[[i, 1]], 5.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_multioutput_correlated() {
        let x = Array2::from_shape_vec((6, 2), (0..12).map(|x| x as f64).collect()).unwrap();
        let y = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0, 5.0, 10.0, 6.0, 12.0],
        )
        .unwrap(); // Perfectly correlated outputs

        let regressor =
            MultiOutputDummyRegressor::new(MultiOutputStrategy::Correlated).with_random_state(42);
        let fitted = regressor.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_outputs(), 2);
        assert!(fitted.correlation_matrix().is_some());
        assert!(fitted.covariance_matrix().is_some());

        let corr_matrix = fitted.correlation_matrix().unwrap();
        // Should detect strong correlation
        assert!((corr_matrix[[0, 1]] - 1.0).abs() < 0.1);

        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions.shape(), &[6, 2]);

        // All predictions should be finite
        for i in 0..6 {
            for j in 0..2 {
                assert!(predictions[[i, j]].is_finite());
            }
        }
    }

    #[test]
    fn test_multioutput_multitask() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = Array2::from_shape_vec((4, 2), vec![1.0, 10.0, 3.0, 20.0, 5.0, 30.0, 7.0, 40.0])
            .unwrap();

        let strategies = vec![
            SingleOutputStrategy::Mean,
            SingleOutputStrategy::Constant(25.0),
        ];

        let regressor = MultiOutputDummyRegressor::new(MultiOutputStrategy::MultiTask(strategies));
        let fitted = regressor.fit(&x, &y).unwrap();

        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions.shape(), &[4, 2]);

        // First output should use mean strategy
        for i in 0..4 {
            assert_abs_diff_eq!(predictions[[i, 0]], 4.0, epsilon = 1e-10); // Mean of [1,3,5,7]
            assert_abs_diff_eq!(predictions[[i, 1]], 25.0, epsilon = 1e-10); // Constant value
        }
    }

    #[test]
    fn test_multioutput_hierarchical() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0],
        )
        .unwrap();

        // For hierarchical: Output 0 has no real parent, Output 1 depends on 0, Output 2 depends on 1
        // Since validation requires parent < output_index, we need to handle output 0 specially
        // Let's modify the hierarchy to be valid: [0, 0, 1] where the first 0 is ignored
        let hierarchy = vec![0, 0, 1]; // Only indices 1 and 2 matter: Output 1 depends on 0, Output 2 depends on 1

        let regressor =
            MultiOutputDummyRegressor::new(MultiOutputStrategy::Hierarchical(hierarchy))
                .with_random_state(42);
        let fitted = regressor.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_outputs(), 3);
        assert!(fitted.hierarchy().is_some());

        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions.shape(), &[4, 3]);

        // All predictions should be finite
        for i in 0..4 {
            for j in 0..3 {
                assert!(predictions[[i, j]].is_finite());
            }
        }
    }

    #[test]
    fn test_multioutput_structured() {
        let x = Array2::from_shape_vec((5, 2), (0..10).map(|x| x as f64).collect()).unwrap();
        let y = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap();

        let regressor =
            MultiOutputDummyRegressor::new(MultiOutputStrategy::Structured).with_random_state(42);
        let fitted = regressor.fit(&x, &y).unwrap();

        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions.shape(), &[5, 2]);

        // All predictions should be from the training data
        for i in 0..5 {
            let row = predictions.row(i);
            let mut found = false;
            for j in 0..5 {
                let training_row = y.row(j);
                if (row[0] - training_row[0]).abs() < 1e-10
                    && (row[1] - training_row[1]).abs() < 1e-10
                {
                    found = true;
                    break;
                }
            }
            assert!(found, "Prediction row not found in training data");
        }
    }

    #[test]
    fn test_multioutput_error_cases() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        // Test mismatched strategy count
        let wrong_strategies = vec![SingleOutputStrategy::Mean]; // Only 1 strategy for 2 outputs
        let regressor =
            MultiOutputDummyRegressor::new(MultiOutputStrategy::MultiTask(wrong_strategies));
        let result = regressor.fit(&x, &y);
        assert!(result.is_err());

        // Test invalid hierarchy - output 1 depends on itself
        let wrong_hierarchy = vec![0, 1]; // Output 1 has parent index 1 (itself) - should fail
        let regressor =
            MultiOutputDummyRegressor::new(MultiOutputStrategy::Hierarchical(wrong_hierarchy));
        let result = regressor.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_cholesky_decomposition() {
        // Test with a simple 2x2 positive definite matrix
        let matrix = Array2::from_shape_vec((2, 2), vec![4.0, 2.0, 2.0, 2.0]).unwrap();
        let result = cholesky_decomposition(&matrix);
        assert!(result.is_ok());

        let l = result.unwrap();
        assert_abs_diff_eq!(l[[0, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(l[[1, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(l[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(l[[1, 1]], 1.0, epsilon = 1e-10);

        // Test with non-positive definite matrix
        let bad_matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 1.0]).unwrap();
        let result = cholesky_decomposition(&bad_matrix);
        assert!(result.is_err());
    }
}
