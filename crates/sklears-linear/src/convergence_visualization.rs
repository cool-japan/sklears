//! Convergence visualization utilities for linear models
//!
//! This module provides tools for visualizing and analyzing convergence
//! behavior of optimization algorithms used in linear models.

use sklears_core::{error::Result, types::Float};
use std::collections::HashMap;

/// Types of convergence metrics to track
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvergenceMetric {
    /// Objective function value
    Objective,
    /// Gradient norm
    GradientNorm,
    /// Parameter change norm
    ParameterChange,
    /// Residual norm
    ResidualNorm,
    /// Dual gap (for constrained problems)
    DualGap,
    /// Validation error
    ValidationError,
}

/// Configuration for convergence tracking
#[derive(Debug, Clone)]
pub struct ConvergenceConfig {
    /// Metrics to track during optimization
    pub track_metrics: Vec<ConvergenceMetric>,
    /// How often to record metrics (every N iterations)
    pub record_frequency: usize,
    /// Whether to compute validation metrics
    pub track_validation: bool,
    /// Maximum number of data points to store
    pub max_history_size: usize,
    /// Whether to smooth metrics using moving average
    pub smooth_metrics: bool,
    /// Window size for smoothing
    pub smoothing_window: usize,
}

impl Default for ConvergenceConfig {
    fn default() -> Self {
        Self {
            track_metrics: vec![
                ConvergenceMetric::Objective,
                ConvergenceMetric::GradientNorm,
                ConvergenceMetric::ParameterChange,
            ],
            record_frequency: 1,
            track_validation: false,
            max_history_size: 10000,
            smooth_metrics: false,
            smoothing_window: 10,
        }
    }
}

/// Convergence data for a single metric
#[derive(Debug, Clone)]
pub struct MetricHistory {
    /// Iteration numbers
    pub iterations: Vec<usize>,
    /// Metric values
    pub values: Vec<Float>,
    /// Smoothed values (if smoothing is enabled)
    pub smoothed_values: Option<Vec<Float>>,
}

impl MetricHistory {
    /// Create new metric history
    pub fn new() -> Self {
        Self {
            iterations: Vec::new(),
            values: Vec::new(),
            smoothed_values: None,
        }
    }

    /// Add a new data point
    pub fn add_point(&mut self, iteration: usize, value: Float, max_size: usize) {
        self.iterations.push(iteration);
        self.values.push(value);

        // Limit history size
        if self.iterations.len() > max_size {
            let remove_count = self.iterations.len() - max_size;
            self.iterations.drain(0..remove_count);
            self.values.drain(0..remove_count);

            if let Some(ref mut smoothed) = self.smoothed_values {
                smoothed.drain(0..remove_count);
            }
        }
    }

    /// Apply smoothing to the values
    pub fn apply_smoothing(&mut self, window_size: usize) {
        if self.values.len() < window_size {
            return;
        }

        let mut smoothed = Vec::with_capacity(self.values.len());

        for i in 0..self.values.len() {
            let start = if i >= window_size {
                i - window_size + 1
            } else {
                0
            };
            let end = i + 1;

            let sum: Float = self.values[start..end].iter().sum();
            let avg = sum / (end - start) as Float;
            smoothed.push(avg);
        }

        self.smoothed_values = Some(smoothed);
    }

    /// Get the latest value
    pub fn latest_value(&self) -> Option<Float> {
        self.values.last().copied()
    }

    /// Get the latest smoothed value
    pub fn latest_smoothed_value(&self) -> Option<Float> {
        self.smoothed_values.as_ref()?.last().copied()
    }

    /// Compute convergence rate (linear approximation)
    pub fn convergence_rate(&self) -> Option<Float> {
        if self.values.len() < 2 {
            return None;
        }

        let n = self.values.len();
        let x_vals: Vec<Float> = (0..n).map(|i| i as Float).collect();
        let log_vals: Vec<Float> = self
            .values
            .iter()
            .filter_map(|&v| if v > 0.0 { Some(v.ln()) } else { None })
            .collect();

        if log_vals.len() < 2 {
            return None;
        }

        // Simple linear regression on log values
        let n_f = log_vals.len() as Float;
        let x_mean = x_vals.iter().take(log_vals.len()).sum::<Float>() / n_f;
        let y_mean = log_vals.iter().sum::<Float>() / n_f;

        let numerator: Float = x_vals
            .iter()
            .take(log_vals.len())
            .zip(log_vals.iter())
            .map(|(&x, &y)| (x - x_mean) * (y - y_mean))
            .sum();

        let denominator: Float = x_vals
            .iter()
            .take(log_vals.len())
            .map(|&x| (x - x_mean).powi(2))
            .sum();

        if denominator.abs() < 1e-10 {
            None
        } else {
            Some(numerator / denominator)
        }
    }
}

/// Convergence tracker for optimization algorithms
#[derive(Debug, Clone)]
pub struct ConvergenceTracker {
    config: ConvergenceConfig,
    metrics: HashMap<ConvergenceMetric, MetricHistory>,
    current_iteration: usize,
}

impl ConvergenceTracker {
    /// Create a new convergence tracker
    pub fn new(config: ConvergenceConfig) -> Self {
        let mut metrics = HashMap::new();

        for &metric in &config.track_metrics {
            metrics.insert(metric, MetricHistory::new());
        }

        Self {
            config,
            metrics,
            current_iteration: 0,
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(ConvergenceConfig::default())
    }

    /// Record metrics for current iteration
    pub fn record_metrics(
        &mut self,
        metric_values: HashMap<ConvergenceMetric, Float>,
    ) -> Result<()> {
        if self.current_iteration % self.config.record_frequency != 0 {
            self.current_iteration += 1;
            return Ok(());
        }

        for (&metric, &value) in &metric_values {
            if let Some(history) = self.metrics.get_mut(&metric) {
                history.add_point(self.current_iteration, value, self.config.max_history_size);

                // Apply smoothing if configured
                if self.config.smooth_metrics
                    && history.values.len() >= self.config.smoothing_window
                {
                    history.apply_smoothing(self.config.smoothing_window);
                }
            }
        }

        self.current_iteration += 1;
        Ok(())
    }

    /// Get metric history
    pub fn get_metric_history(&self, metric: ConvergenceMetric) -> Option<&MetricHistory> {
        self.metrics.get(&metric)
    }

    /// Get current iteration
    pub fn current_iteration(&self) -> usize {
        self.current_iteration
    }

    /// Check if algorithm has converged based on multiple criteria
    pub fn check_convergence(
        &self,
        convergence_criteria: &ConvergenceCriteria,
    ) -> Result<ConvergenceStatus> {
        let mut status = ConvergenceStatus {
            converged: false,
            reason: None,
            iterations_taken: self.current_iteration,
            final_metrics: HashMap::new(),
        };

        // Collect final metric values
        for (&metric, history) in &self.metrics {
            if let Some(value) = history.latest_value() {
                status.final_metrics.insert(metric, value);
            }
        }

        // Check each convergence criterion
        for criterion in &convergence_criteria.criteria {
            if self.check_single_criterion(criterion)? {
                status.converged = true;
                status.reason = Some(format!("{:?}", criterion));
                break;
            }
        }

        Ok(status)
    }

    /// Check a single convergence criterion
    fn check_single_criterion(&self, criterion: &ConvergenceCriterion) -> Result<bool> {
        match criterion {
            ConvergenceCriterion::AbsoluteThreshold { metric, threshold } => {
                if let Some(history) = self.metrics.get(metric) {
                    if let Some(value) = history.latest_value() {
                        return Ok(value.abs() < *threshold);
                    }
                }
                Ok(false)
            }
            ConvergenceCriterion::RelativeChange {
                metric,
                threshold,
                window,
            } => {
                if let Some(history) = self.metrics.get(metric) {
                    let n = history.values.len();
                    if n >= *window {
                        let recent_values = &history.values[n - window..];
                        let max_val = recent_values
                            .iter()
                            .fold(Float::NEG_INFINITY, |a, &b| a.max(b));
                        let min_val = recent_values.iter().fold(Float::INFINITY, |a, &b| a.min(b));

                        if max_val > 0.0 {
                            let relative_change = (max_val - min_val) / max_val;
                            return Ok(relative_change < *threshold);
                        }
                    }
                }
                Ok(false)
            }
            ConvergenceCriterion::GradientNorm { threshold } => {
                if let Some(history) = self.metrics.get(&ConvergenceMetric::GradientNorm) {
                    if let Some(value) = history.latest_value() {
                        return Ok(value < *threshold);
                    }
                }
                Ok(false)
            }
            ConvergenceCriterion::ParameterStability { threshold, window } => {
                if let Some(history) = self.metrics.get(&ConvergenceMetric::ParameterChange) {
                    let n = history.values.len();
                    if n >= *window {
                        let recent_values = &history.values[n - window..];
                        let max_change = recent_values.iter().fold(0.0_f64, |a, &b| a.max(b));
                        return Ok(max_change < *threshold);
                    }
                }
                Ok(false)
            }
        }
    }

    /// Generate convergence report
    pub fn generate_report(&self) -> ConvergenceReport {
        let mut summary = HashMap::new();

        for (&metric, history) in &self.metrics {
            let metric_summary = MetricSummary {
                name: format!("{:?}", metric),
                initial_value: history.values.first().copied(),
                final_value: history.values.last().copied(),
                min_value: history
                    .values
                    .iter()
                    .fold(Float::INFINITY, |a, &b| a.min(b)),
                max_value: history
                    .values
                    .iter()
                    .fold(Float::NEG_INFINITY, |a, &b| a.max(b)),
                convergence_rate: history.convergence_rate(),
                n_iterations: history.values.len(),
            };
            summary.insert(metric, metric_summary);
        }

        ConvergenceReport {
            total_iterations: self.current_iteration,
            metric_summaries: summary,
            configuration: self.config.clone(),
        }
    }

    /// Export data for external plotting (e.g., Python matplotlib)
    pub fn export_for_plotting(&self) -> Result<HashMap<String, PlotData>> {
        let mut plot_data = HashMap::new();

        for (&metric, history) in &self.metrics {
            let data = PlotData {
                x_values: history.iterations.iter().map(|&i| i as Float).collect(),
                y_values: history.values.clone(),
                smoothed_y_values: history.smoothed_values.clone(),
                metric_name: format!("{:?}", metric),
            };
            plot_data.insert(format!("{:?}", metric), data);
        }

        Ok(plot_data)
    }
}

/// Convergence criteria for determining when optimization should stop
#[derive(Debug, Clone)]
pub struct ConvergenceCriteria {
    pub criteria: Vec<ConvergenceCriterion>,
}

impl ConvergenceCriteria {
    /// Create new convergence criteria
    pub fn new() -> Self {
        Self {
            criteria: Vec::new(),
        }
    }

    /// Add an absolute threshold criterion
    pub fn add_absolute_threshold(mut self, metric: ConvergenceMetric, threshold: Float) -> Self {
        self.criteria
            .push(ConvergenceCriterion::AbsoluteThreshold { metric, threshold });
        self
    }

    /// Add a relative change criterion
    pub fn add_relative_change(
        mut self,
        metric: ConvergenceMetric,
        threshold: Float,
        window: usize,
    ) -> Self {
        self.criteria.push(ConvergenceCriterion::RelativeChange {
            metric,
            threshold,
            window,
        });
        self
    }

    /// Add a gradient norm criterion
    pub fn add_gradient_norm(mut self, threshold: Float) -> Self {
        self.criteria
            .push(ConvergenceCriterion::GradientNorm { threshold });
        self
    }

    /// Add a parameter stability criterion
    pub fn add_parameter_stability(mut self, threshold: Float, window: usize) -> Self {
        self.criteria
            .push(ConvergenceCriterion::ParameterStability { threshold, window });
        self
    }
}

/// Individual convergence criterion
#[derive(Debug, Clone)]
pub enum ConvergenceCriterion {
    /// Metric value below absolute threshold
    AbsoluteThreshold {
        metric: ConvergenceMetric,
        threshold: Float,
    },
    /// Relative change in metric over window below threshold
    RelativeChange {
        metric: ConvergenceMetric,
        threshold: Float,
        window: usize,
    },
    /// Gradient norm below threshold
    GradientNorm { threshold: Float },
    /// Parameter change stability over window
    ParameterStability { threshold: Float, window: usize },
}

/// Result of convergence check
#[derive(Debug, Clone)]
pub struct ConvergenceStatus {
    pub converged: bool,
    pub reason: Option<String>,
    pub iterations_taken: usize,
    pub final_metrics: HashMap<ConvergenceMetric, Float>,
}

/// Summary statistics for a single metric
#[derive(Debug, Clone)]
pub struct MetricSummary {
    pub name: String,
    pub initial_value: Option<Float>,
    pub final_value: Option<Float>,
    pub min_value: Float,
    pub max_value: Float,
    pub convergence_rate: Option<Float>,
    pub n_iterations: usize,
}

/// Complete convergence report
#[derive(Debug, Clone)]
pub struct ConvergenceReport {
    pub total_iterations: usize,
    pub metric_summaries: HashMap<ConvergenceMetric, MetricSummary>,
    pub configuration: ConvergenceConfig,
}

/// Data structure for external plotting
#[derive(Debug, Clone)]
pub struct PlotData {
    pub x_values: Vec<Float>,
    pub y_values: Vec<Float>,
    pub smoothed_y_values: Option<Vec<Float>>,
    pub metric_name: String,
}

/// Utility functions for convergence analysis
pub struct ConvergenceAnalysis;

impl ConvergenceAnalysis {
    /// Detect stagnation in convergence
    pub fn detect_stagnation(
        history: &MetricHistory,
        window_size: usize,
        threshold: Float,
    ) -> bool {
        if history.values.len() < window_size {
            return false;
        }

        let n = history.values.len();
        let recent_values = &history.values[n - window_size..];
        let max_val = recent_values
            .iter()
            .fold(Float::NEG_INFINITY, |a, &b| a.max(b));
        let min_val = recent_values.iter().fold(Float::INFINITY, |a, &b| a.min(b));

        if max_val > 0.0 {
            (max_val - min_val) / max_val < threshold
        } else {
            (max_val - min_val).abs() < threshold
        }
    }

    /// Estimate remaining iterations to convergence
    pub fn estimate_remaining_iterations(
        history: &MetricHistory,
        target_value: Float,
    ) -> Option<usize> {
        let rate = history.convergence_rate()?;
        let current_value = history.latest_value()?;

        if rate >= 0.0 || current_value <= target_value {
            return None; // Not converging or already converged
        }

        let iterations_needed = ((target_value / current_value).ln() / rate).ceil() as usize;
        Some(iterations_needed)
    }

    /// Compare convergence between different runs
    pub fn compare_convergence(
        histories: &[&MetricHistory],
        metric: ConvergenceMetric,
    ) -> ComparisonResult {
        if histories.is_empty() {
            return ComparisonResult {
                best_run: None,
                fastest_convergence: None,
                final_values: Vec::new(),
                convergence_rates: Vec::new(),
            };
        }

        let mut final_values = Vec::new();
        let mut convergence_rates = Vec::new();
        let mut best_run = 0;
        let mut best_final_value = Float::INFINITY;

        for (i, history) in histories.iter().enumerate() {
            let final_value = history.latest_value().unwrap_or(Float::INFINITY);
            let rate = history.convergence_rate();

            final_values.push(final_value);
            convergence_rates.push(rate);

            if final_value < best_final_value {
                best_final_value = final_value;
                best_run = i;
            }
        }

        let fastest_convergence = convergence_rates
            .iter()
            .enumerate()
            .filter_map(|(i, &rate)| rate.map(|r| (i, r)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i);

        ComparisonResult {
            best_run: Some(best_run),
            fastest_convergence,
            final_values,
            convergence_rates,
        }
    }
}

/// Result of convergence comparison
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub best_run: Option<usize>,
    pub fastest_convergence: Option<usize>,
    pub final_values: Vec<Float>,
    pub convergence_rates: Vec<Option<Float>>,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_metric_history() {
        let mut history = MetricHistory::new();

        history.add_point(0, 1.0, 1000);
        history.add_point(1, 0.5, 1000);
        history.add_point(2, 0.25, 1000);

        assert_eq!(history.latest_value(), Some(0.25));
        assert_eq!(history.values.len(), 3);

        // Test smoothing
        history.apply_smoothing(2);
        assert!(history.smoothed_values.is_some());
    }

    #[test]
    fn test_convergence_tracker() {
        let config = ConvergenceConfig {
            track_metrics: vec![ConvergenceMetric::Objective],
            record_frequency: 1,
            ..Default::default()
        };

        let mut tracker = ConvergenceTracker::new(config);

        // Record some decreasing objective values
        for i in 0..10 {
            let mut metrics = HashMap::new();
            metrics.insert(ConvergenceMetric::Objective, 1.0 / (i + 1) as Float);
            tracker.record_metrics(metrics).unwrap();
        }

        let history = tracker
            .get_metric_history(ConvergenceMetric::Objective)
            .unwrap();
        assert_eq!(history.values.len(), 10);
        assert!(history.values[0] > history.values[9]); // Should be decreasing
    }

    #[test]
    fn test_convergence_criteria() {
        let criteria = ConvergenceCriteria::new()
            .add_absolute_threshold(ConvergenceMetric::Objective, 0.01)
            .add_gradient_norm(1e-6);

        assert_eq!(criteria.criteria.len(), 2);
    }

    #[test]
    fn test_convergence_rate_calculation() {
        let mut history = MetricHistory::new();

        // Add exponentially decreasing values
        for i in 0..10 {
            let value = (0.5_f64).powi(i as i32) as Float;
            history.add_point(i, value, 1000);
        }

        let rate = history.convergence_rate();
        assert!(rate.is_some());

        // Rate should be negative (decreasing)
        assert!(rate.unwrap() < 0.0);
    }

    #[test]
    fn test_stagnation_detection() {
        let mut history = MetricHistory::new();

        // Add values that stagnate
        for i in 0..10 {
            let value = if i < 5 { 1.0 / (i + 1) as Float } else { 0.2 };
            history.add_point(i, value, 1000);
        }

        let stagnant = ConvergenceAnalysis::detect_stagnation(&history, 5, 0.01);
        assert!(stagnant);
    }

    #[test]
    fn test_export_for_plotting() {
        let config = ConvergenceConfig {
            track_metrics: vec![
                ConvergenceMetric::Objective,
                ConvergenceMetric::GradientNorm,
            ],
            ..Default::default()
        };

        let mut tracker = ConvergenceTracker::new(config);

        // Record some data
        for i in 0..5 {
            let mut metrics = HashMap::new();
            metrics.insert(ConvergenceMetric::Objective, 1.0 / (i + 1) as Float);
            metrics.insert(ConvergenceMetric::GradientNorm, 0.1 / (i + 1) as Float);
            tracker.record_metrics(metrics).unwrap();
        }

        let plot_data = tracker.export_for_plotting().unwrap();
        assert_eq!(plot_data.len(), 2);
        assert!(plot_data.contains_key("Objective"));
        assert!(plot_data.contains_key("GradientNorm"));
    }
}
