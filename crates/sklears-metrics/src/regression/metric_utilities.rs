//! Metric utilities and aggregation for regression metrics
//!
//! This module provides utilities for aggregating metrics, bootstrapping,
//! cross-validation, and confidence intervals.
//! Implements SciRS2 Policy for array operations and numerical computations.

use crate::{MetricsError, MetricsResult};

/// Metric aggregator for combining multiple metrics
pub struct MetricAggregator {
    metrics: Vec<f64>,
}

impl Default for MetricAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricAggregator {
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
        }
    }

    pub fn add_metric(&mut self, value: f64) {
        self.metrics.push(value);
    }

    pub fn mean(&self) -> MetricsResult<f64> {
        if self.metrics.is_empty() {
            return Err(MetricsError::EmptyInput);
        }
        Ok(self.metrics.iter().sum::<f64>() / self.metrics.len() as f64)
    }

    pub fn std(&self) -> MetricsResult<f64> {
        if self.metrics.len() < 2 {
            return Err(MetricsError::InvalidInput(
                "Need at least 2 values".to_string(),
            ));
        }
        let mean = self.mean()?;
        let variance = self.metrics.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / (self.metrics.len() - 1) as f64;
        Ok(variance.sqrt())
    }
}

/// Weighted metrics calculator
pub struct WeightedMetrics {
    values: Vec<f64>,
    weights: Vec<f64>,
}

impl WeightedMetrics {
    pub fn new(values: Vec<f64>, weights: Vec<f64>) -> MetricsResult<Self> {
        if values.len() != weights.len() {
            return Err(MetricsError::ShapeMismatch {
                expected: vec![values.len()],
                actual: vec![weights.len()],
            });
        }

        if values.is_empty() {
            return Err(MetricsError::EmptyInput);
        }

        Ok(Self { values, weights })
    }

    pub fn weighted_mean(&self) -> MetricsResult<f64> {
        let weighted_sum: f64 = self
            .values
            .iter()
            .zip(self.weights.iter())
            .map(|(v, w)| v * w)
            .sum();

        let weight_sum: f64 = self.weights.iter().sum();

        if weight_sum < f64::EPSILON {
            return Err(MetricsError::DivisionByZero);
        }

        Ok(weighted_sum / weight_sum)
    }
}

/// Bootstrap metrics for uncertainty estimation
pub struct BootstrapMetrics {
    samples: Vec<f64>,
    _n_bootstrap: usize,
}

impl BootstrapMetrics {
    pub fn new(samples: Vec<f64>, n_bootstrap: usize) -> Self {
        Self {
            samples,
            _n_bootstrap: n_bootstrap,
        }
    }

    pub fn bootstrap_confidence_interval(&self, confidence: f64) -> MetricsResult<(f64, f64)> {
        if !(0.0..1.0).contains(&confidence) {
            return Err(MetricsError::InvalidParameter(
                "Confidence must be between 0 and 1".to_string(),
            ));
        }

        if self.samples.is_empty() {
            return Err(MetricsError::EmptyInput);
        }

        // Placeholder implementation - would need proper bootstrap resampling
        let mut sorted_samples = self.samples.clone();
        sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_samples.len();
        let alpha = 1.0 - confidence;
        let lower_idx = ((alpha / 2.0) * n as f64) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * n as f64) as usize;

        let lower_bound = sorted_samples[lower_idx.min(n - 1)];
        let upper_bound = sorted_samples[upper_idx.min(n - 1)];

        Ok((lower_bound, upper_bound))
    }
}

/// Cross-validation metrics aggregator
pub struct CrossValidationMetrics {
    fold_scores: Vec<f64>,
}

impl Default for CrossValidationMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl CrossValidationMetrics {
    pub fn new() -> Self {
        Self {
            fold_scores: Vec::new(),
        }
    }

    pub fn add_fold_score(&mut self, score: f64) {
        self.fold_scores.push(score);
    }

    pub fn mean_score(&self) -> MetricsResult<f64> {
        if self.fold_scores.is_empty() {
            return Err(MetricsError::EmptyInput);
        }
        Ok(self.fold_scores.iter().sum::<f64>() / self.fold_scores.len() as f64)
    }

    pub fn std_score(&self) -> MetricsResult<f64> {
        if self.fold_scores.len() < 2 {
            return Err(MetricsError::InvalidInput(
                "Need at least 2 folds".to_string(),
            ));
        }
        let mean = self.mean_score()?;
        let variance = self
            .fold_scores
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / (self.fold_scores.len() - 1) as f64;
        Ok(variance.sqrt())
    }

    pub fn confidence_interval(&self, _confidence: f64) -> MetricsResult<(f64, f64)> {
        let mean = self.mean_score()?;
        let std = self.std_score()?;
        let n = self.fold_scores.len() as f64;

        // Using t-distribution approximation (simplified)
        let t_critical = 2.0; // Approximation for 95% confidence
        let margin = t_critical * std / n.sqrt();

        Ok((mean - margin, mean + margin))
    }
}

/// Metric confidence intervals calculator
pub struct MetricConfidenceIntervals;

impl MetricConfidenceIntervals {
    /// Calculate confidence interval for a metric using normal approximation
    pub fn normal_ci(
        metric_value: f64,
        standard_error: f64,
        confidence: f64,
    ) -> MetricsResult<(f64, f64)> {
        if !(0.0..1.0).contains(&confidence) {
            return Err(MetricsError::InvalidParameter(
                "Confidence must be between 0 and 1".to_string(),
            ));
        }

        // Using normal distribution critical values (simplified)
        let z_critical = match confidence {
            c if c >= 0.99 => 2.576,
            c if c >= 0.95 => 1.96,
            c if c >= 0.90 => 1.645,
            _ => 1.96, // Default to 95%
        };

        let margin = z_critical * standard_error;
        Ok((metric_value - margin, metric_value + margin))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_aggregator() {
        let mut aggregator = MetricAggregator::new();
        aggregator.add_metric(1.0);
        aggregator.add_metric(2.0);
        aggregator.add_metric(3.0);

        let mean = aggregator.mean().unwrap();
        assert!((mean - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_weighted_metrics() {
        let values = vec![1.0, 2.0, 3.0];
        let weights = vec![1.0, 2.0, 3.0];
        let weighted = WeightedMetrics::new(values, weights).unwrap();

        let weighted_mean = weighted.weighted_mean().unwrap();
        assert!(weighted_mean > 2.0); // Should be weighted towards larger values
    }

    #[test]
    fn test_cross_validation_metrics() {
        let mut cv = CrossValidationMetrics::new();
        cv.add_fold_score(0.8);
        cv.add_fold_score(0.9);
        cv.add_fold_score(0.85);

        let mean = cv.mean_score().unwrap();
        assert!((mean - 0.85).abs() < f64::EPSILON);
    }
}
