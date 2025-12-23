//! Theil-Sen Estimator: robust median-based regression
//!
//! The Theil-Sen estimator uses a generalization of the median in multiple dimensions.
//! It estimates the slope as the median of all slopes between pairs of points.
//! This makes it robust to outliers.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::SliceRandomExt;
use scirs2_core::random::{rngs::StdRng, SeedableRng};
use std::marker::PhantomData;

use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Score, Trained, Untrained},
    types::Float,
};

/// Configuration for TheilSenRegressor
#[derive(Debug, Clone)]
pub struct TheilSenRegressorConfig {
    /// Whether to fit the intercept
    pub fit_intercept: bool,
    /// Whether to copy X and y
    pub copy_x: bool,
    /// Maximum number of subpopulations for subsampling
    pub max_subpopulation: Option<usize>,
    /// Number of samples to use for calculating the parameters
    pub n_subsamples: Option<usize>,
    /// Maximum number of iterations for the calculation of spatial median
    pub max_iter: usize,
    /// Tolerance for the calculation of spatial median
    pub tol: Float,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Number of CPUs to use (not implemented, for API compatibility)
    pub n_jobs: Option<usize>,
}

impl Default for TheilSenRegressorConfig {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            copy_x: true,
            max_subpopulation: Some(10_000),
            n_subsamples: None,
            max_iter: 300,
            tol: 1e-3,
            random_state: None,
            n_jobs: None,
        }
    }
}

/// Theil-Sen Estimator
pub struct TheilSenRegressor<State = Untrained> {
    config: TheilSenRegressorConfig,
    state: PhantomData<State>,
    coef_: Option<Array1<Float>>,
    intercept_: Option<Float>,
    breakdown_: Option<Float>,
    n_subpopulation_: Option<usize>,
    n_features_in_: Option<usize>,
}

impl TheilSenRegressor<Untrained> {
    /// Create a new TheilSenRegressor with default configuration
    pub fn new() -> Self {
        Self {
            config: TheilSenRegressorConfig::default(),
            state: PhantomData,
            coef_: None,
            intercept_: None,
            breakdown_: None,
            n_subpopulation_: None,
            n_features_in_: None,
        }
    }

    /// Set whether to fit the intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Set the maximum number of subpopulations
    pub fn max_subpopulation(mut self, max_subpopulation: usize) -> Self {
        self.config.max_subpopulation = Some(max_subpopulation);
        self
    }

    /// Set the number of subsamples
    pub fn n_subsamples(mut self, n_subsamples: usize) -> Self {
        self.config.n_subsamples = Some(n_subsamples);
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }
}

impl Default for TheilSenRegressor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for TheilSenRegressor<Untrained> {
    type Float = Float;
    type Config = TheilSenRegressorConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for TheilSenRegressor<Trained> {
    type Float = Float;
    type Config = TheilSenRegressorConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

/// Calculate the spatial median (geometric median) of points
fn spatial_median(points: &Array2<Float>, max_iter: usize, tol: Float) -> Array1<Float> {
    let n_points = points.nrows();
    let n_features = points.ncols();

    if n_points == 0 {
        return Array1::zeros(n_features);
    }

    // Initialize with arithmetic mean
    let mut median = points.mean_axis(Axis(0)).unwrap();

    for _ in 0..max_iter {
        let old_median = median.clone();

        // Calculate distances from current median to all points
        let mut weights = Vec::with_capacity(n_points);
        let mut total_weight = 0.0;

        for i in 0..n_points {
            let diff = &points.row(i) - &median;
            let dist = diff.mapv(|x| x * x).sum().sqrt();

            if dist > Float::EPSILON {
                let weight = 1.0 / dist;
                weights.push(weight);
                total_weight += weight;
            } else {
                weights.push(0.0);
            }
        }

        // Update median
        median = Array1::zeros(n_features);
        for (i, &weight) in weights.iter().enumerate().take(n_points) {
            if weight > 0.0 {
                median = median + &points.row(i) * weight / total_weight;
            }
        }

        // Check convergence
        let change = (&median - &old_median).mapv(|x| x * x).sum().sqrt();
        if change < tol {
            break;
        }
    }

    median
}

/// Get combinations of indices for subsampling
fn get_combinations(
    n: usize,
    k: usize,
    max_combinations: usize,
    rng: &mut StdRng,
) -> Vec<Vec<usize>> {
    // Calculate total combinations
    let total_combinations = if k > n {
        0
    } else {
        // Approximate n choose k
        let mut result = 1usize;
        for i in 0..k {
            result = result.saturating_mul(n - i).saturating_div(i + 1);
            if result > max_combinations {
                break;
            }
        }
        result
    };

    let n_combinations = total_combinations.min(max_combinations);
    let mut combinations = Vec::with_capacity(n_combinations);

    // If we can enumerate all combinations efficiently
    if n_combinations == total_combinations && n_combinations < 10000 {
        // Generate all combinations
        let mut indices: Vec<usize> = (0..k).collect();
        combinations.push(indices.clone());

        while combinations.len() < n_combinations {
            // Find the rightmost element that can be incremented
            let mut i = k - 1;
            while i > 0 && indices[i] == n - k + i {
                i -= 1;
            }

            if indices[i] < n - k + i {
                indices[i] += 1;
                for j in i + 1..k {
                    indices[j] = indices[j - 1] + 1;
                }
                combinations.push(indices.clone());
            } else {
                break;
            }
        }
    } else {
        // Random sampling for large cases
        let mut seen = std::collections::HashSet::new();

        while combinations.len() < n_combinations {
            let mut indices: Vec<usize> = (0..n).collect();
            indices.shuffle(rng);
            indices.truncate(k);
            indices.sort_unstable();

            if seen.insert(indices.clone()) {
                combinations.push(indices);
            }
        }
    }

    combinations
}

impl Fit<Array2<Float>, Array1<Float>> for TheilSenRegressor<Untrained> {
    type Fitted = TheilSenRegressor<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        // Initialize random number generator
        let mut rng = match self.config.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(42), // Use fixed seed for deterministic behavior
        };

        // Determine n_subsamples
        let n_subsamples = self
            .config
            .n_subsamples
            .unwrap_or_else(|| n_features.max(1) + 1);

        if n_subsamples > n_samples {
            return Err(SklearsError::InvalidInput(format!(
                "n_subsamples ({}) must be <= n_samples ({})",
                n_subsamples, n_samples
            )));
        }

        // Get max subpopulation
        let max_subpopulation = self.config.max_subpopulation.unwrap_or(10_000);

        // Generate combinations of sample indices
        let combinations = get_combinations(n_samples, n_subsamples, max_subpopulation, &mut rng);
        let n_subpopulation = combinations.len();

        // Calculate breakdown point
        let breakdown_ = if n_features > 0 {
            (n_samples as Float - n_features as Float + 1.0) / (2.0 * n_samples as Float)
        } else {
            0.5
        };

        // Store slopes for each combination
        let mut slopes = Vec::with_capacity(n_subpopulation);

        for indices in combinations {
            // Extract subsamples
            let x_subsample = x.select(Axis(0), &indices);
            let y_subsample = y.select(Axis(0), &indices);

            // Fit model on subsample
            if n_features == 1 {
                // Special case for 1D: calculate all pairwise slopes
                let mut pairwise_slopes = Vec::new();

                for i in 0..indices.len() {
                    for j in i + 1..indices.len() {
                        let dx = x_subsample[[j, 0]] - x_subsample[[i, 0]];
                        if dx.abs() > Float::EPSILON {
                            let dy = y_subsample[j] - y_subsample[i];
                            pairwise_slopes.push(dy / dx);
                        }
                    }
                }

                if !pairwise_slopes.is_empty() {
                    pairwise_slopes.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let median_slope = pairwise_slopes[pairwise_slopes.len() / 2];
                    slopes.push(Array1::from_elem(1, median_slope));
                }
            } else {
                // Multi-dimensional case: use least squares on the subsample
                match solve_least_squares(&x_subsample, &y_subsample) {
                    Ok(coef) => slopes.push(coef),
                    Err(_) => continue, // Skip singular subsamples
                }
            }
        }

        if slopes.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Could not find valid subsamples".to_string(),
            ));
        }

        // Calculate spatial median of slopes
        let slopes_array = Array2::from_shape_vec(
            (slopes.len(), n_features),
            slopes.into_iter().flatten().collect(),
        )
        .unwrap();

        let coef = spatial_median(&slopes_array, self.config.max_iter, self.config.tol);

        // Calculate intercept if needed
        let intercept = if self.config.fit_intercept {
            // Median of residuals
            let predictions = x.dot(&coef);
            let residuals = y - &predictions;
            let mut sorted_residuals = residuals.to_vec();
            sorted_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted_residuals[sorted_residuals.len() / 2]
        } else {
            0.0
        };

        Ok(TheilSenRegressor {
            config: self.config,
            state: PhantomData,
            coef_: Some(coef),
            intercept_: Some(intercept),
            breakdown_: Some(breakdown_),
            n_subpopulation_: Some(n_subpopulation),
            n_features_in_: Some(n_features),
        })
    }
}

/// Solve least squares problem
fn solve_least_squares(x: &Array2<Float>, y: &Array1<Float>) -> Result<Array1<Float>> {
    let xt = x.t();
    let xtx = xt.dot(x);
    let xty = xt.dot(y);

    scirs2_linalg::solve(&xtx.view(), &xty.view(), None)
        .map_err(|e| SklearsError::NumericalError(format!("Failed to solve least squares: {}", e)))
}

impl Predict<Array2<Float>, Array1<Float>> for TheilSenRegressor<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let coef = self.coef_.as_ref().unwrap();
        let intercept = self.intercept_.unwrap();

        if x.ncols() != self.n_features_in_.unwrap() {
            return Err(SklearsError::InvalidInput(format!(
                "X has {} features, but TheilSenRegressor is expecting {} features",
                x.ncols(),
                self.n_features_in_.unwrap()
            )));
        }

        Ok(x.dot(coef) + intercept)
    }
}

impl Score<Array2<Float>, Array1<Float>> for TheilSenRegressor<Trained> {
    type Float = Float;

    fn score(&self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Float> {
        let predictions = self.predict(x)?;
        let residuals = y - &predictions;
        let ss_res = residuals.mapv(|r| r * r).sum();
        let y_mean = y.mean().unwrap();
        let ss_tot = y.mapv(|yi| (yi - y_mean).powi(2)).sum();

        Ok(1.0 - ss_res / ss_tot)
    }
}

impl TheilSenRegressor<Trained> {
    /// Get the coefficients
    pub fn coef(&self) -> &Array1<Float> {
        self.coef_.as_ref().unwrap()
    }

    /// Get the intercept
    pub fn intercept(&self) -> Option<Float> {
        Some(self.intercept_.unwrap())
    }

    /// Get the breakdown point
    pub fn breakdown(&self) -> Float {
        self.breakdown_.unwrap()
    }

    /// Get the number of subpopulations used
    pub fn n_subpopulation(&self) -> usize {
        self.n_subpopulation_.unwrap()
    }

    /// Get the number of features seen during fit
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_.unwrap()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_theil_sen_basic() {
        // Simple linear relationship
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0],];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let model = TheilSenRegressor::new()
            .random_state(42)
            .fit(&x, &y)
            .unwrap();

        // Should find slope of 2 and intercept of 0
        assert_abs_diff_eq!(model.coef()[0], 2.0, epsilon = 0.1);
        assert_abs_diff_eq!(model.intercept().unwrap(), 0.0, epsilon = 0.1);
    }

    #[test]
    fn test_theil_sen_with_outliers() {
        // Data with outliers
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [100.0], // outlier
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 200.0]; // last is outlier

        let model = TheilSenRegressor::new()
            .random_state(42)
            .fit(&x, &y)
            .unwrap();

        // Should be robust to outliers and find approximately slope of 2
        assert!(model.coef()[0] > 1.5 && model.coef()[0] < 2.5);
    }

    #[test]
    fn test_theil_sen_no_intercept() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![2.0, 4.0, 6.0];

        let model = TheilSenRegressor::new()
            .fit_intercept(false)
            .fit(&x, &y)
            .unwrap();

        assert_eq!(model.intercept().unwrap(), 0.0);
    }

    #[test]
    fn test_theil_sen_multivariate() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0],];
        let y = array![5.0, 8.0, 11.0, 14.0, 17.0]; // y = 1*x1 + 2*x2

        let model = TheilSenRegressor::new()
            .random_state(42)
            .n_subsamples(3)
            .fit(&x, &y)
            .unwrap();

        // Should find coefficients close to [1, 2]
        assert_abs_diff_eq!(model.coef()[0], 1.0, epsilon = 0.5);
        assert_abs_diff_eq!(model.coef()[1], 2.0, epsilon = 0.5);
    }

    #[test]
    fn test_spatial_median() {
        let points = array![
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [100.0, 100.0], // outlier
        ];

        let median = spatial_median(&points, 100, 1e-6);

        // Spatial median should be less affected by the outlier than the mean
        assert!(median[0] < 20.0 && median[1] < 20.0);
    }
}
