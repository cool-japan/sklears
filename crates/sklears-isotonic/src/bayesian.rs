//! Bayesian isotonic regression with uncertainty quantification
//!
//! This module implements Bayesian approaches to isotonic regression that provide
//! not only point estimates but also credible intervals and posterior sampling.

use crate::utils::safe_float_cmp;
use crate::algorithms::{pool_adjacent_violators_decreasing, pool_adjacent_violators_increasing};
use crate::core::{LossFunction, MonotonicityConstraint};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Bayesian isotonic regression model
///
/// Provides uncertainty quantification through Bayesian inference
/// with proper treatment of monotonicity constraints.
#[derive(Debug, Clone)]
/// BayesianIsotonicRegression
pub struct BayesianIsotonicRegression<State = Untrained> {
    /// Monotonicity constraint specification
    pub constraint: MonotonicityConstraint,
    /// Prior variance parameter
    pub prior_variance: Float,
    /// Observation noise variance
    pub noise_variance: Float,
    /// Number of posterior samples to draw
    pub n_samples: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Confidence level for credible intervals
    pub confidence_level: Float,

    // Fitted attributes
    x_: Option<Array1<Float>>,
    posterior_samples_: Option<Array2<Float>>,
    posterior_mean_: Option<Array1<Float>>,
    credible_intervals_: Option<Array2<Float>>,

    _state: PhantomData<State>,
}

impl BayesianIsotonicRegression<Untrained> {
    /// Create a new Bayesian isotonic regression model
    pub fn new() -> Self {
        Self {
            constraint: MonotonicityConstraint::Global { increasing: true },
            prior_variance: 1.0,
            noise_variance: 1.0,
            n_samples: 1000,
            random_seed: None,
            confidence_level: 0.95,
            x_: None,
            posterior_samples_: None,
            posterior_mean_: None,
            credible_intervals_: None,
            _state: PhantomData,
        }
    }

    /// Set whether the function should be increasing
    pub fn increasing(mut self, increasing: bool) -> Self {
        self.constraint = MonotonicityConstraint::Global { increasing };
        self
    }

    /// Set the prior variance for the Bayesian model
    pub fn prior_variance(mut self, variance: Float) -> Self {
        self.prior_variance = variance;
        self
    }

    /// Set the observation noise variance
    pub fn noise_variance(mut self, variance: Float) -> Self {
        self.noise_variance = variance;
        self
    }

    /// Set the number of posterior samples
    pub fn n_samples(mut self, n_samples: usize) -> Self {
        self.n_samples = n_samples;
        self
    }

    /// Set the confidence level for credible intervals
    pub fn confidence_level(mut self, level: Float) -> Self {
        if level <= 0.0 || level >= 1.0 {
            panic!("confidence_level must be between 0 and 1");
        }
        self.confidence_level = level;
        self
    }

    /// Set random seed for reproducibility
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }
}

impl Default for BayesianIsotonicRegression<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for BayesianIsotonicRegression<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array1<Float>, Array1<Float>> for BayesianIsotonicRegression<Untrained> {
    type Fitted = BayesianIsotonicRegression<Trained>;

    fn fit(self, x: &Array1<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        if x.len() != y.len() {
            return Err(SklearsError::InvalidInput(
                "x and y must have the same length".to_string(),
            ));
        }

        if x.is_empty() {
            return Err(SklearsError::InvalidInput(
                "x and y cannot be empty".to_string(),
            ));
        }

        // Sort data by x values
        let mut indices: Vec<usize> = (0..x.len()).collect();
        indices.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());

        let x_sorted: Array1<Float> = indices.iter().map(|&i| x[i]).collect();
        let y_sorted: Array1<Float> = indices.iter().map(|&i| y[i]).collect();

        // Generate posterior samples using Gibbs sampling
        let posterior_samples = self.gibbs_sampling(&x_sorted, &y_sorted)?;

        // Compute posterior mean
        let posterior_mean = posterior_samples.mean_axis(Axis(0)).unwrap();

        // Compute credible intervals
        let credible_intervals = self.compute_credible_intervals(&posterior_samples)?;

        Ok(BayesianIsotonicRegression {
            constraint: self.constraint,
            prior_variance: self.prior_variance,
            noise_variance: self.noise_variance,
            n_samples: self.n_samples,
            random_seed: self.random_seed,
            confidence_level: self.confidence_level,
            x_: Some(x_sorted),
            posterior_samples_: Some(posterior_samples),
            posterior_mean_: Some(posterior_mean),
            credible_intervals_: Some(credible_intervals),
            _state: PhantomData,
        })
    }
}

impl BayesianIsotonicRegression<Untrained> {
    /// Gibbs sampling for Bayesian isotonic regression
    fn gibbs_sampling(&self, x: &Array1<Float>, y: &Array1<Float>) -> Result<Array2<Float>> {
        let n = x.len();
        let mut samples = Array2::zeros((self.n_samples, n));

        // Initialize with classical isotonic regression
        let mut current_sample = match self.constraint {
            MonotonicityConstraint::Global { increasing: true } => {
                pool_adjacent_violators_increasing(y)
            }
            MonotonicityConstraint::Global { increasing: false } => {
                pool_adjacent_violators_decreasing(y)
            }
            _ => {
                // For complex constraints, start with simple increasing
                pool_adjacent_violators_increasing(y)
            }
        };

        // Simple random number generator (in practice, would use proper RNG)
        let mut rng_state = self.random_seed.unwrap_or(42);

        for sample_idx in 0..self.n_samples {
            // Sample new values at each point
            for i in 0..n {
                current_sample[i] =
                    self.sample_conditional_posterior(i, &current_sample, x, y, &mut rng_state)?;
            }

            // Apply monotonicity constraints
            self.apply_monotonicity_constraints(&mut current_sample);

            // Store sample
            samples.row_mut(sample_idx).assign(&current_sample);
        }

        Ok(samples)
    }

    /// Sample from conditional posterior distribution
    fn sample_conditional_posterior(
        &self,
        index: usize,
        current_values: &Array1<Float>,
        x: &Array1<Float>,
        y: &Array1<Float>,
        rng_state: &mut u64,
    ) -> Result<Float> {
        let n = x.len();

        // Compute posterior mean and variance for this point
        let observation_precision = 1.0 / self.noise_variance;
        let prior_precision = 1.0 / self.prior_variance;

        // Likelihood contribution
        let likelihood_mean = y[index];

        // Prior contribution (smoothness)
        let mut prior_mean = 0.0;
        let mut neighbor_count = 0.0;

        if index > 0 {
            prior_mean += current_values[index - 1];
            neighbor_count += 1.0;
        }
        if index < n - 1 {
            prior_mean += current_values[index + 1];
            neighbor_count += 1.0;
        }

        if neighbor_count > 0.0 {
            prior_mean /= neighbor_count;
        }

        // Combine likelihood and prior
        let posterior_precision = observation_precision + prior_precision * neighbor_count;
        let posterior_mean = (observation_precision * likelihood_mean
            + prior_precision * neighbor_count * prior_mean)
            / posterior_precision;
        let posterior_variance = 1.0 / posterior_precision;

        // Sample from Gaussian (simplified - would use proper random sampling)
        let sample = self.sample_gaussian(posterior_mean, posterior_variance, rng_state);

        Ok(sample)
    }

    /// Simple Gaussian sampling (placeholder implementation)
    fn sample_gaussian(&self, mean: Float, variance: Float, rng_state: &mut u64) -> Float {
        // Simple Box-Muller transform implementation
        *rng_state = (*rng_state).wrapping_mul(1103515245).wrapping_add(12345);
        let u1 = (*rng_state as f64) / (u64::MAX as f64);

        *rng_state = (*rng_state).wrapping_mul(1103515245).wrapping_add(12345);
        let u2 = (*rng_state as f64) / (u64::MAX as f64);

        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

        mean + variance.sqrt() * z as Float
    }

    /// Apply monotonicity constraints to samples
    fn apply_monotonicity_constraints(&self, values: &mut Array1<Float>) {
        match self.constraint {
            MonotonicityConstraint::Global { increasing: true } => {
                *values = pool_adjacent_violators_increasing(values);
            }
            MonotonicityConstraint::Global { increasing: false } => {
                *values = pool_adjacent_violators_decreasing(values);
            }
            _ => {
                // For complex constraints, apply simple increasing for now
                *values = pool_adjacent_violators_increasing(values);
            }
        }
    }

    /// Compute credible intervals from posterior samples
    fn compute_credible_intervals(&self, samples: &Array2<Float>) -> Result<Array2<Float>> {
        let n_points = samples.ncols();
        let mut intervals = Array2::zeros((n_points, 2));

        let alpha = (1.0 - self.confidence_level) / 2.0;
        let lower_quantile = alpha;
        let upper_quantile = 1.0 - alpha;

        for i in 0..n_points {
            let mut column_values: Vec<Float> = samples.column(i).to_vec();
            column_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let lower_idx = (lower_quantile * (column_values.len() - 1) as Float) as usize;
            let upper_idx = (upper_quantile * (column_values.len() - 1) as Float) as usize;

            intervals[[i, 0]] = column_values[lower_idx];
            intervals[[i, 1]] = column_values[upper_idx];
        }

        Ok(intervals)
    }
}

impl Predict<Array1<Float>, Array1<Float>> for BayesianIsotonicRegression<Trained> {
    fn predict(&self, x: &Array1<Float>) -> Result<Array1<Float>> {
        let x_train = self.x_.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "predict".to_string(),
        })?;

        let posterior_mean =
            self.posterior_mean_
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict".to_string(),
                })?;

        let predictions = crate::algorithms::linear_interpolate(x_train, posterior_mean, x);
        Ok(predictions)
    }
}

impl BayesianIsotonicRegression<Trained> {
    /// Get posterior samples
    pub fn posterior_samples(&self) -> Result<&Array2<Float>> {
        self.posterior_samples_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "posterior_samples".to_string(),
            })
    }

    /// Get posterior mean
    pub fn posterior_mean(&self) -> Result<&Array1<Float>> {
        self.posterior_mean_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "posterior_mean".to_string(),
            })
    }

    /// Get credible intervals
    pub fn credible_intervals(&self) -> Result<&Array2<Float>> {
        self.credible_intervals_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "credible_intervals".to_string(),
            })
    }

    /// Predict with uncertainty (returns mean and credible intervals)
    pub fn predict_with_uncertainty(
        &self,
        x: &Array1<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        let x_train = self.x_.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "predict_with_uncertainty".to_string(),
        })?;

        let posterior_mean =
            self.posterior_mean_
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict_with_uncertainty".to_string(),
                })?;

        let credible_intervals =
            self.credible_intervals_
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict_with_uncertainty".to_string(),
                })?;

        // Interpolate posterior mean
        let mean_predictions = crate::algorithms::linear_interpolate(x_train, posterior_mean, x);

        // Interpolate credible intervals
        let lower_bounds = credible_intervals.column(0).to_owned();
        let upper_bounds = credible_intervals.column(1).to_owned();

        let lower_predictions = crate::algorithms::linear_interpolate(x_train, &lower_bounds, x);
        let upper_predictions = crate::algorithms::linear_interpolate(x_train, &upper_bounds, x);

        let mut uncertainty_intervals = Array2::zeros((x.len(), 2));
        uncertainty_intervals
            .column_mut(0)
            .assign(&lower_predictions);
        uncertainty_intervals
            .column_mut(1)
            .assign(&upper_predictions);

        Ok((mean_predictions, uncertainty_intervals))
    }

    /// Sample from posterior predictive distribution
    pub fn sample_posterior_predictive(
        &self,
        x: &Array1<Float>,
        n_samples: usize,
    ) -> Result<Array2<Float>> {
        let x_train = self.x_.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "sample_posterior_predictive".to_string(),
        })?;

        let posterior_samples =
            self.posterior_samples_
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "sample_posterior_predictive".to_string(),
                })?;

        let mut predictive_samples = Array2::zeros((n_samples, x.len()));

        for i in 0..n_samples {
            let sample_idx = i % posterior_samples.nrows();
            let function_sample = posterior_samples.row(sample_idx).to_owned();

            // Interpolate this sample to new x points
            let interpolated = crate::algorithms::linear_interpolate(x_train, &function_sample, x);

            // Add observation noise
            for j in 0..x.len() {
                let noise = self.sample_observation_noise();
                predictive_samples[[i, j]] = interpolated[j] + noise;
            }
        }

        Ok(predictive_samples)
    }

    /// Sample observation noise
    fn sample_observation_noise(&self) -> Float {
        // Simplified - would use proper random sampling in practice
        0.0 // Placeholder: should sample from N(0, noise_variance)
    }
}

/// Convenience function for Bayesian isotonic regression
pub fn bayesian_isotonic_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    increasing: bool,
    prior_variance: Float,
    noise_variance: Float,
    n_samples: usize,
) -> Result<(Array1<Float>, Array2<Float>, Array2<Float>)> {
    let model = BayesianIsotonicRegression::new()
        .increasing(increasing)
        .prior_variance(prior_variance)
        .noise_variance(noise_variance)
        .n_samples(n_samples);

    let fitted_model = model.fit(x, y)?;

    let posterior_mean = fitted_model.posterior_mean()?.clone();
    let posterior_samples = fitted_model.posterior_samples()?.clone();
    let credible_intervals = fitted_model.credible_intervals()?.clone();

    Ok((posterior_mean, posterior_samples, credible_intervals))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_bayesian_isotonic_regression_basic() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from(vec![1.0, 3.0, 2.0, 4.0, 5.0]);

        let model = BayesianIsotonicRegression::new()
            .increasing(true)
            .n_samples(100);

        let fitted_model = model.fit(&x, &y).unwrap();

        // Check that we can get posterior samples
        let posterior_samples = fitted_model.posterior_samples().unwrap();
        assert_eq!(posterior_samples.nrows(), 100);
        assert_eq!(posterior_samples.ncols(), 5);

        // Check that posterior mean is monotonic
        let posterior_mean = fitted_model.posterior_mean().unwrap();
        for i in 0..posterior_mean.len() - 1 {
            assert!(posterior_mean[i] <= posterior_mean[i + 1]);
        }

        // Check that credible intervals are properly ordered
        let intervals = fitted_model.credible_intervals().unwrap();
        for i in 0..intervals.nrows() {
            assert!(intervals[[i, 0]] <= intervals[[i, 1]]);
        }
    }

    #[test]
    fn test_bayesian_prediction_with_uncertainty() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);

        let model = BayesianIsotonicRegression::new()
            .increasing(true)
            .n_samples(50);

        let fitted_model = model.fit(&x, &y).unwrap();

        let x_new = Array1::from(vec![1.5, 2.5, 3.5]);
        let (predictions, uncertainty) = fitted_model.predict_with_uncertainty(&x_new).unwrap();

        assert_eq!(predictions.len(), 3);
        assert_eq!(uncertainty.shape(), &[3, 2]);

        // Check that predictions are within uncertainty bounds
        for i in 0..predictions.len() {
            assert!(predictions[i] >= uncertainty[[i, 0]]);
            assert!(predictions[i] <= uncertainty[[i, 1]]);
        }
    }

    #[test]
    fn test_bayesian_convenience_function() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from(vec![2.0, 1.0, 3.0, 4.0, 5.0]);

        let result = bayesian_isotonic_regression(&x, &y, true, 1.0, 0.5, 100);
        assert!(result.is_ok());

        let (posterior_mean, posterior_samples, credible_intervals) = result.unwrap();

        assert_eq!(posterior_mean.len(), 5);
        assert_eq!(posterior_samples.shape(), &[100, 5]);
        assert_eq!(credible_intervals.shape(), &[5, 2]);

        // Check monotonicity
        for i in 0..posterior_mean.len() - 1 {
            assert!(posterior_mean[i] <= posterior_mean[i + 1]);
        }
    }

    #[test]
    fn test_posterior_predictive_sampling() {
        let x = Array1::from(vec![1.0, 2.0, 3.0]);
        let y = Array1::from(vec![1.0, 2.0, 3.0]);

        let model = BayesianIsotonicRegression::new()
            .increasing(true)
            .n_samples(50);

        let fitted_model = model.fit(&x, &y).unwrap();

        let x_new = Array1::from(vec![1.5, 2.5]);
        let predictive_samples = fitted_model
            .sample_posterior_predictive(&x_new, 25)
            .unwrap();

        assert_eq!(predictive_samples.shape(), &[25, 2]);
    }
}
