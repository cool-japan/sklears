//! Approximate algorithms for large-scale isotonic regression
//!
//! This module provides approximate algorithms for isotonic regression that can
//! handle very large datasets efficiently by trading off accuracy for speed.

use crate::algorithms::{pool_adjacent_violators_decreasing, pool_adjacent_violators_increasing};
use crate::core::{LossFunction, MonotonicityConstraint};
use scirs2_core::ndarray::Array1;
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Approximate isotonic regression for large-scale problems
///
/// Uses sampling and binning strategies to efficiently handle large datasets.
#[derive(Debug, Clone)]
/// ApproximateIsotonicRegression
pub struct ApproximateIsotonicRegression<State = Untrained> {
    /// Monotonicity constraint specification
    pub constraint: MonotonicityConstraint,
    /// Number of bins for approximation
    pub n_bins: usize,
    /// Sample size for approximation (if None, uses all data)
    pub sample_size: Option<usize>,
    /// Random seed for sampling
    pub random_seed: Option<u64>,
    /// Approximation method
    pub method: ApproximationMethod,
    /// Tolerance for approximation error
    pub tolerance: Float,

    // Fitted attributes
    x_bins_: Option<Array1<Float>>,
    y_fitted_: Option<Array1<Float>>,
    bin_edges_: Option<Array1<Float>>,
    approximation_error_: Option<Float>,

    _state: PhantomData<State>,
}

/// Approximation methods for large-scale isotonic regression
#[derive(Debug, Clone, Copy, PartialEq)]
/// ApproximationMethod
pub enum ApproximationMethod {
    /// Uniform binning across the range
    UniformBinning,
    /// Quantile-based binning for better coverage
    QuantileBinning,
    /// Random sampling with PAVA
    RandomSampling,
    /// Hierarchical binning
    HierarchicalBinning,
    /// Streaming approximation
    StreamingApproximation,
}

impl ApproximateIsotonicRegression<Untrained> {
    /// Create a new approximate isotonic regression model
    pub fn new() -> Self {
        Self {
            constraint: MonotonicityConstraint::Global { increasing: true },
            n_bins: 100,
            sample_size: None,
            random_seed: None,
            method: ApproximationMethod::QuantileBinning,
            tolerance: 1e-6,
            x_bins_: None,
            y_fitted_: None,
            bin_edges_: None,
            approximation_error_: None,
            _state: PhantomData,
        }
    }

    /// Set whether the function should be increasing
    pub fn increasing(mut self, increasing: bool) -> Self {
        self.constraint = MonotonicityConstraint::Global { increasing };
        self
    }

    /// Set the number of bins for approximation
    pub fn n_bins(mut self, n_bins: usize) -> Self {
        self.n_bins = n_bins;
        self
    }

    /// Set the sample size for approximation
    pub fn sample_size(mut self, size: usize) -> Self {
        self.sample_size = Some(size);
        self
    }

    /// Set the approximation method
    pub fn method(mut self, method: ApproximationMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the tolerance for approximation error
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set random seed for reproducibility
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }
}

impl Default for ApproximateIsotonicRegression<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for ApproximateIsotonicRegression<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array1<Float>, Array1<Float>> for ApproximateIsotonicRegression<Untrained> {
    type Fitted = ApproximateIsotonicRegression<Trained>;

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

        // Apply the chosen approximation method
        let (x_bins, y_fitted, bin_edges) = match self.method {
            ApproximationMethod::UniformBinning => self.uniform_binning(&x_sorted, &y_sorted)?,
            ApproximationMethod::QuantileBinning => self.quantile_binning(&x_sorted, &y_sorted)?,
            ApproximationMethod::RandomSampling => self.random_sampling(&x_sorted, &y_sorted)?,
            ApproximationMethod::HierarchicalBinning => {
                self.hierarchical_binning(&x_sorted, &y_sorted)?
            }
            ApproximationMethod::StreamingApproximation => {
                self.streaming_approximation(&x_sorted, &y_sorted)?
            }
        };

        // Compute approximation error on training data
        let y_pred = crate::algorithms::linear_interpolate(&x_bins, &y_fitted, &x_sorted);
        let approximation_error = {
            let squared_errors: Float = y_sorted
                .iter()
                .zip(y_pred.iter())
                .map(|(y_true, y_pred)| (y_true - y_pred).powi(2))
                .sum();
            (squared_errors / y_sorted.len() as Float).sqrt() // RMSE
        };

        Ok(ApproximateIsotonicRegression {
            constraint: self.constraint,
            n_bins: self.n_bins,
            sample_size: self.sample_size,
            random_seed: self.random_seed,
            method: self.method,
            tolerance: self.tolerance,
            x_bins_: Some(x_bins),
            y_fitted_: Some(y_fitted),
            bin_edges_: Some(bin_edges),
            approximation_error_: Some(approximation_error),
            _state: PhantomData,
        })
    }
}

impl ApproximateIsotonicRegression<Untrained> {
    /// Uniform binning approximation
    fn uniform_binning(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<(Array1<Float>, Array1<Float>, Array1<Float>)> {
        let x_min = x.iter().fold(Float::INFINITY, |acc, &val| acc.min(val));
        let x_max = x.iter().fold(Float::NEG_INFINITY, |acc, &val| acc.max(val));

        let bin_width = (x_max - x_min) / (self.n_bins as Float);
        let mut bin_edges = Array1::zeros(self.n_bins + 1);

        for i in 0..=self.n_bins {
            bin_edges[i] = x_min + (i as Float) * bin_width;
        }

        // Assign data points to bins
        let mut bin_sums = vec![0.0; self.n_bins];
        let mut bin_counts = vec![0; self.n_bins];

        for (xi, yi) in x.iter().zip(y.iter()) {
            let bin_idx = ((xi - x_min) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(self.n_bins - 1);

            bin_sums[bin_idx] += yi;
            bin_counts[bin_idx] += 1;
        }

        // Compute bin averages
        let mut bin_means = Array1::zeros(self.n_bins);
        let mut x_bins = Array1::zeros(self.n_bins);

        for i in 0..self.n_bins {
            x_bins[i] = bin_edges[i] + bin_width / 2.0;
            if bin_counts[i] > 0 {
                bin_means[i] = bin_sums[i] / (bin_counts[i] as Float);
            }
        }

        // Apply isotonic regression to bin means
        let y_fitted = self.apply_isotonic_constraint(&bin_means);

        Ok((x_bins, y_fitted, bin_edges))
    }

    /// Quantile-based binning approximation
    fn quantile_binning(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<(Array1<Float>, Array1<Float>, Array1<Float>)> {
        let mut x_sorted = x.to_vec();
        x_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut bin_edges = Array1::zeros(self.n_bins + 1);
        bin_edges[0] = x_sorted[0];
        bin_edges[self.n_bins] = x_sorted[x_sorted.len() - 1];

        // Create quantile-based bin edges
        for i in 1..self.n_bins {
            let quantile = (i as Float) / (self.n_bins as Float);
            let idx = (quantile * (x_sorted.len() - 1) as Float) as usize;
            bin_edges[i] = x_sorted[idx];
        }

        // Assign data points to bins
        let mut bin_sums = vec![0.0; self.n_bins];
        let mut bin_counts = vec![0; self.n_bins];
        let mut bin_x_sums = vec![0.0; self.n_bins];

        for (xi, yi) in x.iter().zip(y.iter()) {
            let mut bin_idx = 0;
            for j in 1..=self.n_bins {
                if *xi <= bin_edges[j] {
                    bin_idx = j - 1;
                    break;
                }
            }

            bin_sums[bin_idx] += yi;
            bin_x_sums[bin_idx] += xi;
            bin_counts[bin_idx] += 1;
        }

        // Compute bin centers and means
        let mut bin_means = Array1::zeros(self.n_bins);
        let mut x_bins = Array1::zeros(self.n_bins);

        for i in 0..self.n_bins {
            if bin_counts[i] > 0 {
                x_bins[i] = bin_x_sums[i] / (bin_counts[i] as Float);
                bin_means[i] = bin_sums[i] / (bin_counts[i] as Float);
            } else {
                x_bins[i] = (bin_edges[i] + bin_edges[i + 1]) / 2.0;
                bin_means[i] = 0.0;
            }
        }

        // Apply isotonic regression to bin means
        let y_fitted = self.apply_isotonic_constraint(&bin_means);

        Ok((x_bins, y_fitted, bin_edges))
    }

    /// Random sampling approximation
    fn random_sampling(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<(Array1<Float>, Array1<Float>, Array1<Float>)> {
        let sample_size = self.sample_size.unwrap_or(self.n_bins * 10).min(x.len());

        // Simple random sampling (in practice would use proper RNG)
        let mut rng_state = self.random_seed.unwrap_or(42);
        let mut sampled_indices = Vec::new();

        for _ in 0..sample_size {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let idx = (rng_state as usize) % x.len();
            sampled_indices.push(idx);
        }

        // Create sampled arrays
        let x_sampled: Array1<Float> = sampled_indices.iter().map(|&i| x[i]).collect();
        let y_sampled: Array1<Float> = sampled_indices.iter().map(|&i| y[i]).collect();

        // Apply full isotonic regression on sample
        let y_fitted = self.apply_isotonic_constraint(&y_sampled);

        // Create bin edges from sampled data
        let mut x_sorted = x_sampled.to_vec();
        x_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut bin_edges = Array1::zeros(self.n_bins + 1);

        for i in 0..=self.n_bins {
            let quantile = (i as Float) / (self.n_bins as Float);
            let idx = (quantile * (x_sorted.len() - 1) as Float) as usize;
            bin_edges[i] = x_sorted[idx.min(x_sorted.len() - 1)];
        }

        Ok((x_sampled, y_fitted, bin_edges))
    }

    /// Hierarchical binning approximation
    fn hierarchical_binning(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<(Array1<Float>, Array1<Float>, Array1<Float>)> {
        // Start with coarse binning and refine
        let mut current_bins = 10;
        let mut best_x_bins = Array1::zeros(0);
        let mut best_y_fitted = Array1::zeros(0);
        let mut best_bin_edges = Array1::zeros(0);
        let mut best_error = Float::INFINITY;

        while current_bins <= self.n_bins {
            // Create temporary model with current number of bins
            let temp_model = ApproximateIsotonicRegression::new()
                .n_bins(current_bins)
                .method(ApproximationMethod::QuantileBinning);

            let (x_bins, y_fitted, bin_edges) = temp_model.quantile_binning(x, y)?;

            // Compute approximation error
            let error = self.compute_approximation_error(x, y, &x_bins, &y_fitted);

            if error < self.tolerance || error < best_error {
                best_x_bins = x_bins;
                best_y_fitted = y_fitted;
                best_bin_edges = bin_edges;
                best_error = error;

                if error < self.tolerance {
                    break;
                }
            }

            current_bins = (current_bins as Float * 1.5) as usize;
        }

        Ok((best_x_bins, best_y_fitted, best_bin_edges))
    }

    /// Streaming approximation for online scenarios
    fn streaming_approximation(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<(Array1<Float>, Array1<Float>, Array1<Float>)> {
        let chunk_size = (x.len() / self.n_bins).max(1);
        let mut x_bins = Vec::new();
        let mut y_bins = Vec::new();

        // Process data in chunks
        for chunk_start in (0..x.len()).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(x.len());

            let x_chunk = x.slice(s![chunk_start..chunk_end]).to_owned();
            let y_chunk = y.slice(s![chunk_start..chunk_end]).to_owned();

            // Compute chunk average
            let x_mean = x_chunk.mean().unwrap();
            let y_mean = y_chunk.mean().unwrap();

            x_bins.push(x_mean);
            y_bins.push(y_mean);
        }

        let x_bins_array = Array1::from(x_bins);
        let y_bins_array = Array1::from(y_bins);

        // Apply isotonic regression to chunk means
        let y_fitted = self.apply_isotonic_constraint(&y_bins_array);

        // Create bin edges
        let mut bin_edges = Array1::zeros(x_bins_array.len() + 1);
        bin_edges[0] = x[0];
        for i in 1..x_bins_array.len() {
            bin_edges[i] = (x_bins_array[i - 1] + x_bins_array[i]) / 2.0;
        }
        bin_edges[x_bins_array.len()] = x[x.len() - 1];

        Ok((x_bins_array, y_fitted, bin_edges))
    }

    /// Apply isotonic constraint based on the configured constraint
    fn apply_isotonic_constraint(&self, y: &Array1<Float>) -> Array1<Float> {
        match self.constraint {
            MonotonicityConstraint::Global { increasing: true } => {
                pool_adjacent_violators_increasing(y)
            }
            MonotonicityConstraint::Global { increasing: false } => {
                pool_adjacent_violators_decreasing(y)
            }
            _ => {
                // For complex constraints, use simple increasing for now
                pool_adjacent_violators_increasing(y)
            }
        }
    }

    /// Compute approximation error
    fn compute_approximation_error(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
        x_bins: &Array1<Float>,
        y_fitted: &Array1<Float>,
    ) -> Float {
        let mut total_error = 0.0;

        for (xi, yi) in x.iter().zip(y.iter()) {
            let interpolated =
                crate::algorithms::linear_interpolate(x_bins, y_fitted, &Array1::from(vec![*xi]))
                    [0];

            total_error += (yi - interpolated).powi(2);
        }

        total_error / (x.len() as Float)
    }
}

impl Predict<Array1<Float>, Array1<Float>> for ApproximateIsotonicRegression<Trained> {
    fn predict(&self, x: &Array1<Float>) -> Result<Array1<Float>> {
        let x_bins = self
            .x_bins_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let y_fitted = self
            .y_fitted_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let predictions = crate::algorithms::linear_interpolate(x_bins, y_fitted, x);
        Ok(predictions)
    }
}

impl ApproximateIsotonicRegression<Trained> {
    /// Get the approximation error (RMSE) on training data
    ///
    /// Returns the root mean squared error between the training labels
    /// and the approximated predictions. This measures the quality of
    /// the approximation.
    pub fn approximation_error(&self) -> Float {
        self.approximation_error_.unwrap_or(0.0)
    }

    /// Get the bin edges used for approximation
    pub fn bin_edges(&self) -> Result<&Array1<Float>> {
        self.bin_edges_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "bin_edges".to_string(),
            })
    }

    /// Get the fitted bin centers
    pub fn bin_centers(&self) -> Result<&Array1<Float>> {
        self.x_bins_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "bin_centers".to_string(),
            })
    }

    /// Get the fitted bin values
    pub fn bin_values(&self) -> Result<&Array1<Float>> {
        self.y_fitted_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "bin_values".to_string(),
            })
    }
}

/// Convenience function for approximate isotonic regression
pub fn approximate_isotonic_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    method: ApproximationMethod,
    n_bins: usize,
    increasing: bool,
) -> Result<(Array1<Float>, Array1<Float>)> {
    let model = ApproximateIsotonicRegression::new()
        .increasing(increasing)
        .n_bins(n_bins)
        .method(method);

    let fitted_model = model.fit(x, y)?;

    let x_bins = fitted_model.x_bins_.unwrap();
    let y_fitted = fitted_model.y_fitted_.unwrap();

    Ok((x_bins, y_fitted))
}

use scirs2_core::ndarray::s;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_approximate_isotonic_uniform_binning() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let y = Array1::from(vec![1.0, 3.0, 2.0, 4.0, 5.0, 4.0, 6.0, 7.0, 8.0, 9.0]);

        let model = ApproximateIsotonicRegression::new()
            .increasing(true)
            .n_bins(5)
            .method(ApproximationMethod::UniformBinning);

        let fitted_model = model.fit(&x, &y).unwrap();

        // Check that bin values are monotonic
        let bin_values = fitted_model.bin_values().unwrap();
        if bin_values.len() > 1 {
            for i in 0..bin_values.len() - 1 {
                assert!(bin_values[i] <= bin_values[i + 1]);
            }
        }

        // Test prediction
        let x_test = Array1::from(vec![2.5, 5.5, 8.5]);
        let predictions = fitted_model.predict(&x_test).unwrap();
        assert_eq!(predictions.len(), 3);
    }

    #[test]
    fn test_approximate_isotonic_quantile_binning() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let y = Array1::from(vec![2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0]);

        let model = ApproximateIsotonicRegression::new()
            .increasing(true)
            .n_bins(4)
            .method(ApproximationMethod::QuantileBinning);

        let fitted_model = model.fit(&x, &y).unwrap();

        // Check that bin values are monotonic
        let bin_values = fitted_model.bin_values().unwrap();
        if bin_values.len() > 1 {
            for i in 0..bin_values.len() - 1 {
                assert!(bin_values[i] <= bin_values[i + 1]);
            }
        }

        // Test prediction
        let predictions = fitted_model.predict(&x).unwrap();
        assert_eq!(predictions.len(), x.len());
    }

    #[test]
    fn test_approximate_isotonic_random_sampling() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = Array1::from(vec![1.0, 3.0, 2.0, 4.0, 3.0, 5.0]);

        let model = ApproximateIsotonicRegression::new()
            .increasing(true)
            .sample_size(4)
            .method(ApproximationMethod::RandomSampling)
            .random_seed(123);

        let fitted_model = model.fit(&x, &y).unwrap();

        // Test that we can get predictions
        let predictions = fitted_model.predict(&x).unwrap();
        assert_eq!(predictions.len(), x.len());
    }

    #[test]
    fn test_approximate_isotonic_hierarchical() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let y = Array1::from(vec![1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0]);

        let model = ApproximateIsotonicRegression::new()
            .increasing(true)
            .n_bins(6)
            .method(ApproximationMethod::HierarchicalBinning)
            .tolerance(0.1);

        let fitted_model = model.fit(&x, &y).unwrap();

        let bin_values = fitted_model.bin_values().unwrap();
        if bin_values.len() > 1 {
            for i in 0..bin_values.len() - 1 {
                assert!(bin_values[i] <= bin_values[i + 1]);
            }
        }
    }

    #[test]
    fn test_approximate_isotonic_streaming() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let y = Array1::from(vec![1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0]);

        let model = ApproximateIsotonicRegression::new()
            .increasing(true)
            .n_bins(5)
            .method(ApproximationMethod::StreamingApproximation);

        let fitted_model = model.fit(&x, &y).unwrap();

        let bin_values = fitted_model.bin_values().unwrap();
        if bin_values.len() > 1 {
            for i in 0..bin_values.len() - 1 {
                assert!(bin_values[i] <= bin_values[i + 1]);
            }
        }
    }

    #[test]
    fn test_approximate_isotonic_convenience_function() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from(vec![2.0, 1.0, 3.0, 2.0, 4.0]);

        let result =
            approximate_isotonic_regression(&x, &y, ApproximationMethod::QuantileBinning, 3, true);
        assert!(result.is_ok());

        let (x_bins, y_fitted) = result.unwrap();
        assert_eq!(x_bins.len(), 3);
        assert_eq!(y_fitted.len(), 3);

        // Check monotonicity
        for i in 0..y_fitted.len() - 1 {
            assert!(y_fitted[i] <= y_fitted[i + 1]);
        }
    }

    #[test]
    fn test_decreasing_constraint() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from(vec![5.0, 4.0, 6.0, 3.0, 2.0]);

        let model = ApproximateIsotonicRegression::new()
            .increasing(false)
            .n_bins(3)
            .method(ApproximationMethod::QuantileBinning);

        let fitted_model = model.fit(&x, &y).unwrap();

        let bin_values = fitted_model.bin_values().unwrap();
        if bin_values.len() > 1 {
            for i in 0..bin_values.len() - 1 {
                assert!(bin_values[i] >= bin_values[i + 1]);
            }
        }
    }
}
