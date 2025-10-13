//! RANSAC (RANdom SAmple Consensus) Regressor
//!
//! RANSAC is an iterative method for robustly fitting a regression model in the
//! presence of outliers. The algorithm selects random subsets of the original
//! data and fits the model to these subsets.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::SliceRandomExt;
use scirs2_core::random::{rngs::StdRng, SeedableRng};
use std::marker::PhantomData;

use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Score, Trained, Untrained},
    types::Float,
};

use crate::linear_regression::LinearRegression;

/// Configuration for RANSACRegressor
#[derive(Debug, Clone)]
pub struct RANSACRegressorConfig {
    /// Minimum number of samples chosen randomly from original data
    pub min_samples: Option<usize>,
    /// Maximum distance for a sample to be classified as an inlier
    pub residual_threshold: Option<Float>,
    /// Maximum number of iterations for random sample selection
    pub max_trials: usize,
    /// Number of randomly chosen samples for each trial
    pub max_skips: usize,
    /// Stop iteration if at least this number of inliers are found
    pub stop_n_inliers: Option<usize>,
    /// Stop iteration if score is greater equal to this threshold
    pub stop_score: Option<Float>,
    /// Stop iteration if at least this fraction of inliers is found
    pub stop_probability: Option<Float>,
    /// Method for scoring predictions
    pub loss: RANSACLoss,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

/// Loss function for RANSAC
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RANSACLoss {
    Absolute,
    Squared,
}

impl Default for RANSACRegressorConfig {
    fn default() -> Self {
        Self {
            min_samples: None,
            residual_threshold: None,
            max_trials: 100,
            max_skips: 1_000_000,
            stop_n_inliers: None,
            stop_score: None,
            stop_probability: Some(0.99),
            loss: RANSACLoss::Absolute,
            random_state: None,
        }
    }
}

/// RANSAC Regressor
pub struct RANSACRegressor<State = Untrained> {
    config: RANSACRegressorConfig,
    state: PhantomData<State>,
    estimator_: Option<LinearRegression<Trained>>,
    n_skips_no_inliers_: Option<usize>,
    n_skips_invalid_data_: Option<usize>,
    n_skips_invalid_model_: Option<usize>,
    n_trials_: Option<usize>,
    inlier_mask_: Option<Array1<bool>>,
    n_features_in_: Option<usize>,
}

impl RANSACRegressor<Untrained> {
    /// Create a new RANSACRegressor with default configuration
    pub fn new() -> Self {
        Self {
            config: RANSACRegressorConfig::default(),
            state: PhantomData,
            estimator_: None,
            n_skips_no_inliers_: None,
            n_skips_invalid_data_: None,
            n_skips_invalid_model_: None,
            n_trials_: None,
            inlier_mask_: None,
            n_features_in_: None,
        }
    }

    /// Set the residual threshold
    pub fn residual_threshold(mut self, threshold: Float) -> Self {
        self.config.residual_threshold = Some(threshold);
        self
    }

    /// Set the minimum number of samples
    pub fn min_samples(mut self, min_samples: usize) -> Self {
        self.config.min_samples = Some(min_samples);
        self
    }

    /// Set the maximum number of trials
    pub fn max_trials(mut self, max_trials: usize) -> Self {
        self.config.max_trials = max_trials;
        self
    }

    /// Set the loss function
    pub fn loss(mut self, loss: RANSACLoss) -> Self {
        self.config.loss = loss;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }
}

impl Default for RANSACRegressor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for RANSACRegressor<Untrained> {
    type Float = Float;
    type Config = RANSACRegressorConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for RANSACRegressor<Trained> {
    type Float = Float;
    type Config = RANSACRegressorConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

/// Calculate residuals based on loss function
fn calculate_residuals(
    y_true: &Array1<Float>,
    y_pred: &Array1<Float>,
    loss: RANSACLoss,
) -> Array1<Float> {
    match loss {
        RANSACLoss::Absolute => (y_true - y_pred).mapv(|x| x.abs()),
        RANSACLoss::Squared => (y_true - y_pred).mapv(|x| x * x),
    }
}

/// Select random subset of samples
fn random_subset(n_samples: usize, min_samples: usize, rng: &mut StdRng) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(rng);
    indices.truncate(min_samples);
    indices
}

impl Fit<Array2<Float>, Array1<Float>> for RANSACRegressor<Untrained> {
    type Fitted = RANSACRegressor<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        // Determine min_samples
        let min_samples = self.config.min_samples.unwrap_or_else(|| {
            n_features + 1 // At least n_features + 1 for a unique solution
        });

        if min_samples > n_samples {
            return Err(SklearsError::InvalidInput(format!(
                "min_samples ({}) must be <= n_samples ({})",
                min_samples, n_samples
            )));
        }

        // Determine residual threshold
        let residual_threshold = match self.config.residual_threshold {
            Some(threshold) => threshold,
            None => {
                // Use MAD (Median Absolute Deviation) of residuals from initial fit
                let base_model = LinearRegression::new().fit(x, y)?;
                let predictions = base_model.predict(x)?;
                let residuals = (y - &predictions).mapv(|x| x.abs());
                let mut sorted_residuals = residuals.to_vec();
                sorted_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = sorted_residuals[sorted_residuals.len() / 2];
                median * 1.4826 // MAD to std deviation conversion factor
            }
        };

        // Initialize random number generator
        let mut rng = match self.config.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(42), // Use fixed seed for deterministic behavior
        };

        // RANSAC main loop
        let mut best_estimator: Option<LinearRegression<Trained>> = None;
        let mut best_inlier_mask = Array1::<bool>::from_elem(n_samples, false);
        let mut best_n_inliers = 0;
        let mut best_score = Float::NEG_INFINITY;

        let mut n_trials = 0;
        let mut n_skips_no_inliers = 0;
        let n_skips_invalid_data = 0;
        let n_skips_invalid_model = 0;

        for trial in 0..self.config.max_trials {
            n_trials = trial + 1;

            // Select random subset
            let subset_indices = random_subset(n_samples, min_samples, &mut rng);

            // Extract subset data
            let x_subset = x.select(Axis(0), &subset_indices);
            let y_subset = y.select(Axis(0), &subset_indices);

            // Fit model on subset
            let base_model = LinearRegression::new().fit(&x_subset, &y_subset)?;

            // Predict on all data
            let y_pred = base_model.predict(x)?;
            let residuals = calculate_residuals(y, &y_pred, self.config.loss);

            // Determine inliers
            let inlier_mask = residuals.mapv(|r| r <= residual_threshold);
            let n_inliers = inlier_mask.iter().filter(|&&x| x).count();

            if n_inliers == 0 {
                n_skips_no_inliers += 1;
                if n_skips_no_inliers > self.config.max_skips {
                    break;
                }
                continue;
            }

            // Refit on all inliers
            let inlier_indices: Vec<usize> = inlier_mask
                .iter()
                .enumerate()
                .filter_map(|(i, &is_inlier)| if is_inlier { Some(i) } else { None })
                .collect();

            let x_inliers = x.select(Axis(0), &inlier_indices);
            let y_inliers = y.select(Axis(0), &inlier_indices);

            let inlier_model = LinearRegression::new().fit(&x_inliers, &y_inliers)?;
            let score = inlier_model.score(&x_inliers, &y_inliers)?;

            // Check if this is the best model so far
            if score > best_score || (score == best_score && n_inliers > best_n_inliers) {
                best_estimator = Some(inlier_model);
                best_inlier_mask = inlier_mask;
                best_n_inliers = n_inliers;
                best_score = score;
            }

            // Check stopping criteria
            if let Some(stop_n_inliers) = self.config.stop_n_inliers {
                if best_n_inliers >= stop_n_inliers {
                    break;
                }
            }

            if let Some(stop_score) = self.config.stop_score {
                if best_score >= stop_score {
                    break;
                }
            }

            if let Some(stop_probability) = self.config.stop_probability {
                let inlier_ratio = best_n_inliers as Float / n_samples as Float;
                let probability =
                    1.0 - (1.0 - inlier_ratio.powi(min_samples as i32)).powi(trial as i32 + 1);
                if probability >= stop_probability {
                    break;
                }
            }
        }

        if let Some(estimator) = best_estimator {
            Ok(RANSACRegressor {
                config: self.config,
                state: PhantomData,
                estimator_: Some(estimator),
                n_skips_no_inliers_: Some(n_skips_no_inliers),
                n_skips_invalid_data_: Some(n_skips_invalid_data),
                n_skips_invalid_model_: Some(n_skips_invalid_model),
                n_trials_: Some(n_trials),
                inlier_mask_: Some(best_inlier_mask),
                n_features_in_: Some(n_features),
            })
        } else {
            Err(SklearsError::InvalidInput(
                "RANSAC could not find a valid consensus set".to_string(),
            ))
        }
    }
}

impl Predict<Array2<Float>, Array1<Float>> for RANSACRegressor<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        if x.ncols() != self.n_features_in_.unwrap() {
            return Err(SklearsError::InvalidInput(format!(
                "X has {} features, but RANSACRegressor is expecting {} features",
                x.ncols(),
                self.n_features_in_.unwrap()
            )));
        }

        self.estimator_.as_ref().unwrap().predict(x)
    }
}

impl Score<Array2<Float>, Array1<Float>> for RANSACRegressor<Trained> {
    type Float = Float;

    fn score(&self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Float> {
        self.estimator_.as_ref().unwrap().score(x, y)
    }
}

impl RANSACRegressor<Trained> {
    /// Get the final estimator
    pub fn estimator(&self) -> &LinearRegression<Trained> {
        self.estimator_.as_ref().unwrap()
    }

    /// Get the boolean mask of inliers
    pub fn inlier_mask(&self) -> &Array1<bool> {
        self.inlier_mask_.as_ref().unwrap()
    }

    /// Get the number of trials run
    pub fn n_trials(&self) -> usize {
        self.n_trials_.unwrap()
    }

    /// Get the number of skips due to no inliers
    pub fn n_skips_no_inliers(&self) -> usize {
        self.n_skips_no_inliers_.unwrap()
    }

    /// Get the number of skips due to invalid data
    pub fn n_skips_invalid_data(&self) -> usize {
        self.n_skips_invalid_data_.unwrap()
    }

    /// Get the number of skips due to invalid model
    pub fn n_skips_invalid_model(&self) -> usize {
        self.n_skips_invalid_model_.unwrap()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_ransac_regressor_basic() {
        // Create data with outliers
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [100.0], // outlier
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 200.0]; // last is outlier

        let model = RANSACRegressor::new().random_state(42).fit(&x, &y).unwrap();

        // Check that some outliers are detected
        let inlier_mask = model.inlier_mask();
        let n_inliers = inlier_mask.iter().filter(|&&x| x).count();
        // Should detect at least some inliers (but not all due to outlier)
        assert!(
            n_inliers >= 4,
            "Expected at least 4 inliers, got {}",
            n_inliers
        );

        // Predictions should follow the pattern 2*x
        let x_test = array![[2.5], [3.5]];
        let predictions = model.predict(&x_test).unwrap();
        assert_abs_diff_eq!(predictions[0], 5.0, epsilon = 0.5);
        assert_abs_diff_eq!(predictions[1], 7.0, epsilon = 0.5);
    }

    #[test]
    fn test_ransac_regressor_min_samples() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let model = RANSACRegressor::new()
            .min_samples(3)
            .random_state(42)
            .fit(&x, &y)
            .unwrap();

        assert!(model.estimator_.is_some());
    }

    #[test]
    fn test_ransac_regressor_threshold() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0],];
        let y = array![1.0, 2.1, 2.9, 4.0, 10.0]; // Last point is outlier

        let model = RANSACRegressor::new()
            .residual_threshold(0.5)
            .random_state(42)
            .fit(&x, &y)
            .unwrap();

        let inlier_mask = model.inlier_mask();
        let n_inliers = inlier_mask.iter().filter(|&&x| x).count();
        // With residual threshold 0.5, should find at least some inliers
        assert!(n_inliers >= 2 && n_inliers <= 4);
    }

    #[test]
    fn test_ransac_regressor_squared_loss() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![1.0, 2.0, 3.0, 10.0]; // Last is outlier

        let model = RANSACRegressor::new()
            .loss(RANSACLoss::Squared)
            .residual_threshold(1.0) // 1.0 squared error threshold
            .random_state(42)
            .fit(&x, &y)
            .unwrap();

        let inlier_mask = model.inlier_mask();
        assert!(inlier_mask
            .slice(scirs2_core::ndarray::s![0..3])
            .iter()
            .all(|&x| x));
    }

    #[test]
    fn test_ransac_regressor_multivariate() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 5.0],
            [4.0, 7.0],
            [5.0, 8.0],
            [100.0, 50.0], // outlier
        ];
        let y = array![5.0, 8.0, 13.0, 18.0, 21.0, 300.0]; // last is outlier

        let model = RANSACRegressor::new().random_state(42).fit(&x, &y).unwrap();

        let inlier_mask = model.inlier_mask();
        assert_eq!(inlier_mask.iter().filter(|&&x| x).count(), 5); // 5 inliers
    }
}
