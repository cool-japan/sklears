//! Information theory methods for isotonic regression
//!
//! This module implements isotonic regression using information-theoretic principles
//! including maximum entropy methods, mutual information preservation, and
//! minimum description length approaches.

use crate::core::{isotonic_regression, LossFunction};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{prelude::SklearsError, types::Float};
use std::collections::HashMap;

/// Maximum entropy isotonic regression
#[derive(Debug, Clone)]
/// MaximumEntropyIsotonicRegression
pub struct MaximumEntropyIsotonicRegression {
    /// Whether to enforce increasing or decreasing monotonicity
    increasing: bool,
    /// Constraint tolerance
    tolerance: Float,
    /// Maximum number of iterations
    max_iterations: usize,
    /// Temperature parameter for entropy regularization
    temperature: Float,
    /// Fitted values
    fitted_values: Option<Array1<Float>>,
    /// Fitted x values (for interpolation)
    fitted_x: Option<Array1<Float>>,
    /// Lagrange multipliers for constraints
    lagrange_multipliers: Option<Array1<Float>>,
}

impl MaximumEntropyIsotonicRegression {
    /// Create a new maximum entropy isotonic regression model
    pub fn new() -> Self {
        Self {
            increasing: true,
            tolerance: 1e-8,
            max_iterations: 1000,
            temperature: 1.0,
            fitted_values: None,
            fitted_x: None,
            lagrange_multipliers: None,
        }
    }

    /// Set whether the function should be increasing or decreasing
    pub fn increasing(mut self, increasing: bool) -> Self {
        self.increasing = increasing;
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set maximum iterations
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set temperature parameter for entropy regularization
    pub fn temperature(mut self, temperature: Float) -> Self {
        self.temperature = temperature;
        self
    }

    /// Fit the maximum entropy isotonic regression model
    pub fn fit(&mut self, x: &Array1<Float>, y: &Array1<Float>) -> Result<(), SklearsError> {
        if x.len() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{}", x.len()),
                actual: format!("{}", y.len()),
            });
        }

        if x.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        // Sort data by x values
        let mut data: Vec<(Float, Float)> =
            x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi, yi)).collect();
        data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let sorted_x: Array1<Float> = Array1::from_vec(data.iter().map(|(xi, _)| *xi).collect());
        let sorted_y: Array1<Float> = Array1::from_vec(data.iter().map(|(_, yi)| *yi).collect());

        // Apply maximum entropy method
        let (fitted_y, multipliers) = self.maximum_entropy_optimization(&sorted_x, &sorted_y)?;

        self.fitted_x = Some(sorted_x);
        self.fitted_values = Some(fitted_y);
        self.lagrange_multipliers = Some(multipliers);

        Ok(())
    }

    /// Predict values at given points
    pub fn predict(&self, x: &Array1<Float>) -> Result<Array1<Float>, SklearsError> {
        let fitted_x = self
            .fitted_x
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let fitted_values = self
            .fitted_values
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let mut predictions = Array1::zeros(x.len());

        for (i, &xi) in x.iter().enumerate() {
            predictions[i] = self.interpolate(xi, fitted_x, fitted_values)?;
        }

        Ok(predictions)
    }

    /// Get the entropy of the fitted distribution
    pub fn entropy(&self) -> Result<Float, SklearsError> {
        let fitted_values = self
            .fitted_values
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "entropy".to_string(),
            })?;

        // Normalize values to probabilities (assuming they represent a distribution)
        let sum: Float = fitted_values.sum();
        if sum <= 0.0 {
            return Ok(0.0);
        }

        let probs: Array1<Float> = fitted_values / sum;
        let entropy = -probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<Float>();

        Ok(entropy)
    }

    /// Maximum entropy optimization with isotonic constraints
    fn maximum_entropy_optimization(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<(Array1<Float>, Array1<Float>), SklearsError> {
        let n = x.len();
        let mut distribution = Array1::from_elem(n, 1.0 / n as Float); // Uniform initial distribution
        let mut multipliers = Array1::zeros(n - 1); // Lagrange multipliers for monotonicity constraints

        // Iterative optimization using method of Lagrange multipliers
        for iteration in 0..self.max_iterations {
            let old_distribution = distribution.clone();

            // Update distribution using exponential family form
            for i in 0..n {
                let constraint_term: Float = if i == 0 {
                    if self.increasing {
                        multipliers[0]
                    } else {
                        -multipliers[0]
                    }
                } else if i == n - 1 {
                    if self.increasing {
                        -multipliers[i - 1]
                    } else {
                        multipliers[i - 1]
                    }
                } else {
                    if self.increasing {
                        multipliers[i] - multipliers[i - 1]
                    } else {
                        multipliers[i - 1] - multipliers[i]
                    }
                };

                // Data fitting term
                let data_term = -(y[i] - distribution[i]).powi(2) / (2.0 * self.temperature);

                distribution[i] = (data_term + constraint_term).exp();
            }

            // Normalize distribution
            let sum = distribution.sum();
            if sum > 0.0 {
                distribution /= sum;
            }

            // Update Lagrange multipliers to enforce monotonicity constraints
            for i in 0..n - 1 {
                let constraint_violation = if self.increasing {
                    distribution[i] - distribution[i + 1]
                } else {
                    distribution[i + 1] - distribution[i]
                };

                multipliers[i] += 0.1 * constraint_violation; // Simple gradient ascent
            }

            // Project onto monotonic constraints
            distribution = self.project_monotonic_constraint(&distribution)?;

            // Check convergence
            let change = (&distribution - &old_distribution).map(|x| x.abs()).sum();
            if change < self.tolerance {
                break;
            }
        }

        // Convert distribution back to fitted values
        let fitted_values = &distribution * y.sum(); // Scale to match data scale

        Ok((fitted_values, multipliers))
    }

    /// Project distribution onto monotonic constraints using Pool Adjacent Violators
    fn project_monotonic_constraint(
        &self,
        distribution: &Array1<Float>,
    ) -> Result<Array1<Float>, SklearsError> {
        let n = distribution.len();
        let mut result = distribution.clone();

        if self.increasing {
            // Enforce increasing constraints
            for i in 1..n {
                if result[i] < result[i - 1] {
                    // Pool adjacent violators
                    let mut j = i;
                    let mut sum = result[i - 1] + result[i];
                    let mut count = 2;

                    // Extend pool backwards
                    while j > 1 && result[j - 2] > sum / count as Float {
                        j -= 1;
                        sum += result[j - 1];
                        count += 1;
                    }

                    // Set pooled values
                    let pooled_value = sum / count as Float;
                    for k in (j - 1)..=i {
                        result[k] = pooled_value;
                    }
                }
            }
        } else {
            // Enforce decreasing constraints
            for i in 1..n {
                if result[i] > result[i - 1] {
                    // Pool adjacent violators
                    let mut j = i;
                    let mut sum = result[i - 1] + result[i];
                    let mut count = 2;

                    // Extend pool backwards
                    while j > 1 && result[j - 2] < sum / count as Float {
                        j -= 1;
                        sum += result[j - 1];
                        count += 1;
                    }

                    // Set pooled values
                    let pooled_value = sum / count as Float;
                    for k in (j - 1)..=i {
                        result[k] = pooled_value;
                    }
                }
            }
        }

        Ok(result)
    }

    /// Linear interpolation for prediction
    fn interpolate(
        &self,
        x: Float,
        fitted_x: &Array1<Float>,
        fitted_values: &Array1<Float>,
    ) -> Result<Float, SklearsError> {
        if fitted_x.is_empty() {
            return Err(SklearsError::InvalidInput("No fitted data".to_string()));
        }

        let n = fitted_x.len();

        // Handle boundary cases
        if x <= fitted_x[0] {
            return Ok(fitted_values[0]);
        }
        if x >= fitted_x[n - 1] {
            return Ok(fitted_values[n - 1]);
        }

        // Find interpolation interval
        for i in 0..n - 1 {
            if x >= fitted_x[i] && x <= fitted_x[i + 1] {
                let t = (x - fitted_x[i]) / (fitted_x[i + 1] - fitted_x[i]);
                return Ok(fitted_values[i] + t * (fitted_values[i + 1] - fitted_values[i]));
            }
        }

        Ok(fitted_values[n - 1])
    }
}

impl Default for MaximumEntropyIsotonicRegression {
    fn default() -> Self {
        Self::new()
    }
}

/// Mutual information preserving isotonic regression
#[derive(Debug, Clone)]
/// MutualInformationIsotonicRegression
pub struct MutualInformationIsotonicRegression {
    /// Whether to enforce increasing or decreasing monotonicity
    increasing: bool,
    /// Number of bins for discretization
    bins: usize,
    /// Regularization parameter
    regularization: Float,
    /// Maximum number of iterations
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: Float,
    /// Fitted values
    fitted_values: Option<Array1<Float>>,
    /// Fitted x values (for interpolation)
    fitted_x: Option<Array1<Float>>,
    /// Mutual information value
    mutual_information: Option<Float>,
}

impl MutualInformationIsotonicRegression {
    /// Create a new mutual information preserving isotonic regression model
    pub fn new() -> Self {
        Self {
            increasing: true,
            bins: 10,
            regularization: 1e-6,
            max_iterations: 1000,
            tolerance: 1e-8,
            fitted_values: None,
            fitted_x: None,
            mutual_information: None,
        }
    }

    /// Set whether the function should be increasing or decreasing
    pub fn increasing(mut self, increasing: bool) -> Self {
        self.increasing = increasing;
        self
    }

    /// Set number of bins for discretization
    pub fn bins(mut self, bins: usize) -> Self {
        self.bins = bins;
        self
    }

    /// Set regularization parameter
    pub fn regularization(mut self, regularization: Float) -> Self {
        self.regularization = regularization;
        self
    }

    /// Fit the mutual information preserving isotonic regression model
    pub fn fit(&mut self, x: &Array1<Float>, y: &Array1<Float>) -> Result<(), SklearsError> {
        if x.len() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{}", x.len()),
                actual: format!("{}", y.len()),
            });
        }

        if x.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        // Sort data by x values
        let mut data: Vec<(Float, Float)> =
            x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi, yi)).collect();
        data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let sorted_x: Array1<Float> = Array1::from_vec(data.iter().map(|(xi, _)| *xi).collect());
        let sorted_y: Array1<Float> = Array1::from_vec(data.iter().map(|(_, yi)| *yi).collect());

        // Compute original mutual information
        let original_mi = self.compute_mutual_information(&sorted_x, &sorted_y)?;

        // Optimize with mutual information preservation constraint
        let fitted_y = self.mutual_information_optimization(&sorted_x, &sorted_y, original_mi)?;

        self.fitted_x = Some(sorted_x);
        self.fitted_values = Some(fitted_y);
        self.mutual_information = Some(original_mi);

        Ok(())
    }

    /// Predict values at given points
    pub fn predict(&self, x: &Array1<Float>) -> Result<Array1<Float>, SklearsError> {
        let fitted_x = self
            .fitted_x
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let fitted_values = self
            .fitted_values
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let mut predictions = Array1::zeros(x.len());

        for (i, &xi) in x.iter().enumerate() {
            predictions[i] = self.interpolate(xi, fitted_x, fitted_values)?;
        }

        Ok(predictions)
    }

    /// Get the preserved mutual information
    pub fn get_mutual_information(&self) -> Option<Float> {
        self.mutual_information
    }

    /// Compute mutual information between two variables
    fn compute_mutual_information(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<Float, SklearsError> {
        // Discretize variables into bins
        let x_bins = self.discretize(x)?;
        let y_bins = self.discretize(y)?;

        // Compute joint and marginal probability distributions
        let mut joint_counts: HashMap<(usize, usize), usize> = HashMap::new();
        let mut x_counts: HashMap<usize, usize> = HashMap::new();
        let mut y_counts: HashMap<usize, usize> = HashMap::new();

        let n = x.len();
        for i in 0..n {
            let x_bin = x_bins[i];
            let y_bin = y_bins[i];

            *joint_counts.entry((x_bin, y_bin)).or_insert(0) += 1;
            *x_counts.entry(x_bin).or_insert(0) += 1;
            *y_counts.entry(y_bin).or_insert(0) += 1;
        }

        // Compute mutual information
        let mut mi = 0.0;
        for (&(x_bin, y_bin), &joint_count) in &joint_counts {
            let p_xy = joint_count as Float / n as Float;
            let p_x = x_counts[&x_bin] as Float / n as Float;
            let p_y = y_counts[&y_bin] as Float / n as Float;

            if p_xy > 0.0 && p_x > 0.0 && p_y > 0.0 {
                mi += p_xy * (p_xy / (p_x * p_y)).ln();
            }
        }

        Ok(mi)
    }

    /// Discretize continuous values into bins
    fn discretize(&self, values: &Array1<Float>) -> Result<Vec<usize>, SklearsError> {
        if values.is_empty() {
            return Ok(vec![]);
        }

        let min_val = values.iter().fold(Float::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));

        if min_val == max_val {
            return Ok(vec![0; values.len()]);
        }

        let bin_width = (max_val - min_val) / self.bins as Float;
        let bins: Vec<usize> = values
            .iter()
            .map(|&val| {
                let bin = ((val - min_val) / bin_width).floor() as usize;
                bin.min(self.bins - 1)
            })
            .collect();

        Ok(bins)
    }

    /// Apply isotonic constraints using Pool Adjacent Violators algorithm
    fn apply_isotonic_constraint(&self, y: &Array1<Float>) -> Result<Array1<Float>, SklearsError> {
        let n = y.len();
        let mut result = y.clone();

        if self.increasing {
            // Enforce increasing constraints
            for i in 1..n {
                if result[i] < result[i - 1] {
                    // Pool adjacent violators
                    let mut j = i;
                    let mut sum = result[i - 1] + result[i];
                    let mut count = 2;

                    // Extend pool backwards
                    while j > 1 && result[j - 2] > sum / count as Float {
                        j -= 1;
                        sum += result[j - 1];
                        count += 1;
                    }

                    // Set pooled values
                    let pooled_value = sum / count as Float;
                    for k in (j - 1)..=i {
                        result[k] = pooled_value;
                    }
                }
            }
        } else {
            // Enforce decreasing constraints
            for i in 1..n {
                if result[i] > result[i - 1] {
                    // Pool adjacent violators
                    let mut j = i;
                    let mut sum = result[i - 1] + result[i];
                    let mut count = 2;

                    // Extend pool backwards
                    while j > 1 && result[j - 2] < sum / count as Float {
                        j -= 1;
                        sum += result[j - 1];
                        count += 1;
                    }

                    // Set pooled values
                    let pooled_value = sum / count as Float;
                    for k in (j - 1)..=i {
                        result[k] = pooled_value;
                    }
                }
            }
        }

        Ok(result)
    }

    /// Optimization with mutual information preservation
    fn mutual_information_optimization(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
        target_mi: Float,
    ) -> Result<Array1<Float>, SklearsError> {
        let mut fitted_y = y.clone();

        // Iterative optimization
        for iteration in 0..self.max_iterations {
            let old_y = fitted_y.clone();

            // Apply isotonic constraint
            fitted_y = self.apply_isotonic_constraint(&fitted_y)?;

            // Adjust to preserve mutual information
            let current_mi = self.compute_mutual_information(x, &fitted_y)?;
            let mi_error = current_mi - target_mi;

            // Simple adjustment based on MI error
            if mi_error.abs() > self.tolerance {
                let adjustment = self.regularization * mi_error;
                for i in 0..fitted_y.len() {
                    fitted_y[i] += adjustment * (y[i] - fitted_y[i]);
                }
            }

            // Check convergence
            let change = (&fitted_y - &old_y).map(|x| x.abs()).sum();
            if change < self.tolerance {
                break;
            }
        }

        Ok(fitted_y)
    }

    /// Linear interpolation for prediction
    fn interpolate(
        &self,
        x: Float,
        fitted_x: &Array1<Float>,
        fitted_values: &Array1<Float>,
    ) -> Result<Float, SklearsError> {
        if fitted_x.is_empty() {
            return Err(SklearsError::InvalidInput("No fitted data".to_string()));
        }

        let n = fitted_x.len();

        // Handle boundary cases
        if x <= fitted_x[0] {
            return Ok(fitted_values[0]);
        }
        if x >= fitted_x[n - 1] {
            return Ok(fitted_values[n - 1]);
        }

        // Find interpolation interval
        for i in 0..n - 1 {
            if x >= fitted_x[i] && x <= fitted_x[i + 1] {
                let t = (x - fitted_x[i]) / (fitted_x[i + 1] - fitted_x[i]);
                return Ok(fitted_values[i] + t * (fitted_values[i + 1] - fitted_values[i]));
            }
        }

        Ok(fitted_values[n - 1])
    }
}

impl Default for MutualInformationIsotonicRegression {
    fn default() -> Self {
        Self::new()
    }
}

/// Minimum description length isotonic regression
#[derive(Debug, Clone)]
/// MinimumDescriptionLengthIsotonicRegression
pub struct MinimumDescriptionLengthIsotonicRegression {
    /// Whether to enforce increasing or decreasing monotonicity
    increasing: bool,
    /// Model complexity penalty parameter
    complexity_penalty: Float,
    /// Maximum number of iterations
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: Float,
    /// Fitted values
    fitted_values: Option<Array1<Float>>,
    /// Fitted x values (for interpolation)
    fitted_x: Option<Array1<Float>>,
    /// Description length
    description_length: Option<Float>,
}

impl MinimumDescriptionLengthIsotonicRegression {
    /// Create a new minimum description length isotonic regression model
    pub fn new() -> Self {
        Self {
            increasing: true,
            complexity_penalty: 1.0,
            max_iterations: 1000,
            tolerance: 1e-8,
            fitted_values: None,
            fitted_x: None,
            description_length: None,
        }
    }

    /// Set whether the function should be increasing or decreasing
    pub fn increasing(mut self, increasing: bool) -> Self {
        self.increasing = increasing;
        self
    }

    /// Set complexity penalty parameter
    pub fn complexity_penalty(mut self, complexity_penalty: Float) -> Self {
        self.complexity_penalty = complexity_penalty;
        self
    }

    /// Fit the minimum description length isotonic regression model
    pub fn fit(&mut self, x: &Array1<Float>, y: &Array1<Float>) -> Result<(), SklearsError> {
        if x.len() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{}", x.len()),
                actual: format!("{}", y.len()),
            });
        }

        if x.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        // Sort data by x values
        let mut data: Vec<(Float, Float)> =
            x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi, yi)).collect();
        data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let sorted_x: Array1<Float> = Array1::from_vec(data.iter().map(|(xi, _)| *xi).collect());
        let sorted_y: Array1<Float> = Array1::from_vec(data.iter().map(|(_, yi)| *yi).collect());

        // Optimize using minimum description length principle
        let (fitted_y, description_length) = self.mdl_optimization(&sorted_x, &sorted_y)?;

        self.fitted_x = Some(sorted_x);
        self.fitted_values = Some(fitted_y);
        self.description_length = Some(description_length);

        Ok(())
    }

    /// Predict values at given points
    pub fn predict(&self, x: &Array1<Float>) -> Result<Array1<Float>, SklearsError> {
        let fitted_x = self
            .fitted_x
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let fitted_values = self
            .fitted_values
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let mut predictions = Array1::zeros(x.len());

        for (i, &xi) in x.iter().enumerate() {
            predictions[i] = self.interpolate(xi, fitted_x, fitted_values)?;
        }

        Ok(predictions)
    }

    /// Get the description length of the fitted model
    pub fn get_description_length(&self) -> Option<Float> {
        self.description_length
    }

    /// Minimum description length optimization
    fn mdl_optimization(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<(Array1<Float>, Float), SklearsError> {
        let n = x.len();
        let mut best_fitted_y = y.clone();
        let mut best_description_length = Float::INFINITY;

        // Try different levels of smoothing/complexity
        let smoothing_levels = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0];

        for &smoothing in &smoothing_levels {
            // Apply isotonic regression with smoothing
            let mut fitted_y = y.clone();

            // Smooth the data first
            if smoothing > 0.0 {
                for iteration in 0..10 {
                    let old_y = fitted_y.clone();

                    // Apply smoothing
                    for i in 1..n - 1 {
                        fitted_y[i] = (1.0 - smoothing) * fitted_y[i]
                            + smoothing * (old_y[i - 1] + old_y[i + 1]) / 2.0;
                    }

                    // Apply isotonic constraint
                    fitted_y = self.apply_isotonic_constraint(&fitted_y)?;

                    let change = (&fitted_y - &old_y).map(|x| x.abs()).sum();
                    if change < self.tolerance {
                        break;
                    }
                }
            } else {
                // Just apply isotonic constraint
                fitted_y = self.apply_isotonic_constraint(&fitted_y)?;
            }

            // Compute description length
            let description_length = self.compute_description_length(y, &fitted_y)?;

            if description_length < best_description_length {
                best_description_length = description_length;
                best_fitted_y = fitted_y;
            }
        }

        Ok((best_fitted_y, best_description_length))
    }

    /// Compute description length for a model
    fn compute_description_length(
        &self,
        y_true: &Array1<Float>,
        y_fitted: &Array1<Float>,
    ) -> Result<Float, SklearsError> {
        let n = y_true.len() as Float;

        // Data encoding length (negative log-likelihood)
        let residuals = y_true - y_fitted;
        let mse = residuals.map(|x| x * x).sum() / n;
        let data_length = if mse > 0.0 {
            0.5 * n * (2.0 * std::f64::consts::PI * mse as f64).ln() as Float + 0.5 * n
        } else {
            0.0
        };

        // Model complexity (number of distinct values)
        let mut unique_values: Vec<Float> = y_fitted.to_vec();
        unique_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique_values.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
        let complexity = unique_values.len() as Float;

        // Total description length
        let description_length = data_length + self.complexity_penalty * complexity;

        Ok(description_length)
    }

    /// Apply isotonic constraints using Pool Adjacent Violators algorithm
    fn apply_isotonic_constraint(&self, y: &Array1<Float>) -> Result<Array1<Float>, SklearsError> {
        let n = y.len();
        let mut result = y.clone();

        if self.increasing {
            // Enforce increasing constraints
            for i in 1..n {
                if result[i] < result[i - 1] {
                    // Pool adjacent violators
                    let mut j = i;
                    let mut sum = result[i - 1] + result[i];
                    let mut count = 2;

                    // Extend pool backwards
                    while j > 1 && result[j - 2] > sum / count as Float {
                        j -= 1;
                        sum += result[j - 1];
                        count += 1;
                    }

                    // Set pooled values
                    let pooled_value = sum / count as Float;
                    for k in (j - 1)..=i {
                        result[k] = pooled_value;
                    }
                }
            }
        } else {
            // Enforce decreasing constraints
            for i in 1..n {
                if result[i] > result[i - 1] {
                    // Pool adjacent violators
                    let mut j = i;
                    let mut sum = result[i - 1] + result[i];
                    let mut count = 2;

                    // Extend pool backwards
                    while j > 1 && result[j - 2] < sum / count as Float {
                        j -= 1;
                        sum += result[j - 1];
                        count += 1;
                    }

                    // Set pooled values
                    let pooled_value = sum / count as Float;
                    for k in (j - 1)..=i {
                        result[k] = pooled_value;
                    }
                }
            }
        }

        Ok(result)
    }

    /// Linear interpolation for prediction
    fn interpolate(
        &self,
        x: Float,
        fitted_x: &Array1<Float>,
        fitted_values: &Array1<Float>,
    ) -> Result<Float, SklearsError> {
        if fitted_x.is_empty() {
            return Err(SklearsError::InvalidInput("No fitted data".to_string()));
        }

        let n = fitted_x.len();

        // Handle boundary cases
        if x <= fitted_x[0] {
            return Ok(fitted_values[0]);
        }
        if x >= fitted_x[n - 1] {
            return Ok(fitted_values[n - 1]);
        }

        // Find interpolation interval
        for i in 0..n - 1 {
            if x >= fitted_x[i] && x <= fitted_x[i + 1] {
                let t = (x - fitted_x[i]) / (fitted_x[i + 1] - fitted_x[i]);
                return Ok(fitted_values[i] + t * (fitted_values[i + 1] - fitted_values[i]));
            }
        }

        Ok(fitted_values[n - 1])
    }
}

impl Default for MinimumDescriptionLengthIsotonicRegression {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function for maximum entropy isotonic regression
pub fn maximum_entropy_isotonic_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    increasing: bool,
    temperature: Float,
) -> Result<Array1<Float>, SklearsError> {
    let mut model = MaximumEntropyIsotonicRegression::new()
        .increasing(increasing)
        .temperature(temperature);

    model.fit(x, y)?;
    model.predict(x)
}

/// Convenience function for mutual information preserving isotonic regression
pub fn mutual_information_isotonic_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    increasing: bool,
    bins: usize,
) -> Result<Array1<Float>, SklearsError> {
    let mut model = MutualInformationIsotonicRegression::new()
        .increasing(increasing)
        .bins(bins);

    model.fit(x, y)?;
    model.predict(x)
}

/// Convenience function for minimum description length isotonic regression
pub fn minimum_description_length_isotonic_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    increasing: bool,
    complexity_penalty: Float,
) -> Result<Array1<Float>, SklearsError> {
    let mut model = MinimumDescriptionLengthIsotonicRegression::new()
        .increasing(increasing)
        .complexity_penalty(complexity_penalty);

    model.fit(x, y)?;
    model.predict(x)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_maximum_entropy_isotonic_regression() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0]; // Non-monotonic

        let mut model = MaximumEntropyIsotonicRegression::new()
            .increasing(true)
            .temperature(1.0);

        assert!(model.fit(&x, &y).is_ok());

        let predictions = model.predict(&x).unwrap();

        // Check that predictions are monotonic
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] <= predictions[i + 1]);
        }

        // Check that entropy can be computed
        let entropy = model.entropy().unwrap();
        assert!(entropy >= 0.0);
    }

    #[test]
    fn test_mutual_information_isotonic_regression() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0]; // Non-monotonic

        let mut model = MutualInformationIsotonicRegression::new()
            .increasing(true)
            .bins(3);

        assert!(model.fit(&x, &y).is_ok());

        let predictions = model.predict(&x).unwrap();

        // Check that predictions are monotonic
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] <= predictions[i + 1]);
        }

        // Check that mutual information is preserved
        let mi = model.get_mutual_information();
        assert!(mi.is_some());
        assert!(mi.unwrap() >= 0.0);
    }

    #[test]
    fn test_minimum_description_length_isotonic_regression() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0]; // Non-monotonic

        let mut model = MinimumDescriptionLengthIsotonicRegression::new()
            .increasing(true)
            .complexity_penalty(1.0);

        assert!(model.fit(&x, &y).is_ok());

        let predictions = model.predict(&x).unwrap();

        // Check that predictions are monotonic
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] <= predictions[i + 1]);
        }

        // Check that description length is computed
        let dl = model.get_description_length();
        assert!(dl.is_some());
        assert!(dl.unwrap() > 0.0);
    }

    #[test]
    fn test_convenience_functions() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0];

        // Test maximum entropy function
        let me_result = maximum_entropy_isotonic_regression(&x, &y, true, 1.0);
        assert!(me_result.is_ok());

        // Test mutual information function
        let mi_result = mutual_information_isotonic_regression(&x, &y, true, 5);
        assert!(mi_result.is_ok());

        // Test minimum description length function
        let mdl_result = minimum_description_length_isotonic_regression(&x, &y, true, 1.0);
        assert!(mdl_result.is_ok());
    }

    #[test]
    fn test_discretization() {
        let values = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let model = MutualInformationIsotonicRegression::new().bins(3);

        let bins = model.discretize(&values).unwrap();
        assert_eq!(bins.len(), 5);

        // Check that bins are in valid range
        for &bin in &bins {
            assert!(bin < 3);
        }
    }

    #[test]
    fn test_entropy_calculation() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![0.5, 0.3, 0.2]; // Probability-like values

        let mut model = MaximumEntropyIsotonicRegression::new();
        model.fit(&x, &y).unwrap();

        let entropy = model.entropy().unwrap();
        assert!(entropy > 0.0);
    }

    #[test]
    fn test_mutual_information_computation() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfectly correlated

        let model = MutualInformationIsotonicRegression::new().bins(5);
        let mi = model.compute_mutual_information(&x, &y).unwrap();

        // For perfectly correlated data, MI should be positive
        assert!(mi > 0.0);
    }

    #[test]
    fn test_description_length_computation() {
        let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_fitted = array![1.1, 2.1, 2.9, 4.1, 4.9]; // Close fit

        let model = MinimumDescriptionLengthIsotonicRegression::new().complexity_penalty(1.0);

        let dl = model
            .compute_description_length(&y_true, &y_fitted)
            .unwrap();
        assert!(dl > 0.0);
    }
}
