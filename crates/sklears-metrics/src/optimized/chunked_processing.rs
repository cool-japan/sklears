//! Chunked Processing for Memory-Efficient Metric Computation
//!
//! This module provides memory-efficient implementations of metrics using chunked processing.
//! By processing data in small chunks rather than loading everything into memory at once,
//! these implementations can handle very large datasets with limited memory usage.
//!
//! ## Key Features
//!
//! - **Memory Efficiency**: Processes data in configurable chunks to reduce memory pressure
//! - **Large Dataset Support**: Can handle datasets larger than available memory
//! - **Configurable Chunk Size**: Allows tuning chunk size for optimal performance
//! - **Multiple Metrics**: Supports MAE, MSE, RMSE, R², and cosine similarity
//! - **Consistent API**: Same interface as regular metric functions

use super::OptimizedConfig;
use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::{s, Array1};
use scirs2_core::numeric::{Float as FloatTrait, FromPrimitive};

/// Chunked processing for memory-efficient metric computation
pub struct ChunkedMetricProcessor<F> {
    config: OptimizedConfig,
    chunk_size: usize,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: FloatTrait + FromPrimitive + Send + Sync> ChunkedMetricProcessor<F> {
    pub fn new(config: OptimizedConfig) -> Self {
        let chunk_size = config.chunk_size;
        Self {
            config,
            chunk_size,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a new processor with custom chunk size
    pub fn with_chunk_size(mut config: OptimizedConfig, chunk_size: usize) -> Self {
        config.chunk_size = chunk_size;
        Self::new(config)
    }

    /// Compute MAE using chunked processing
    pub fn chunked_mean_absolute_error(
        &self,
        y_true: &Array1<F>,
        y_pred: &Array1<F>,
    ) -> MetricsResult<F> {
        if y_true.len() != y_pred.len() {
            return Err(MetricsError::ShapeMismatch {
                expected: vec![y_true.len()],
                actual: vec![y_pred.len()],
            });
        }

        if y_true.is_empty() {
            return Err(MetricsError::EmptyInput);
        }

        let len = y_true.len();
        let mut total_sum = F::zero();

        // Process in chunks to reduce memory pressure
        for chunk_start in (0..len).step_by(self.chunk_size) {
            let chunk_end = (chunk_start + self.chunk_size).min(len);
            let true_chunk = y_true.slice(s![chunk_start..chunk_end]);
            let pred_chunk = y_pred.slice(s![chunk_start..chunk_end]);

            let chunk_sum = true_chunk
                .iter()
                .zip(pred_chunk.iter())
                .map(|(t, p)| (*t - *p).abs())
                .fold(F::zero(), |acc, x| acc + x);

            total_sum = total_sum + chunk_sum;
        }

        Ok(total_sum / F::from(len).unwrap())
    }

    /// Compute MSE using chunked processing
    pub fn chunked_mean_squared_error(
        &self,
        y_true: &Array1<F>,
        y_pred: &Array1<F>,
    ) -> MetricsResult<F> {
        if y_true.len() != y_pred.len() {
            return Err(MetricsError::ShapeMismatch {
                expected: vec![y_true.len()],
                actual: vec![y_pred.len()],
            });
        }

        if y_true.is_empty() {
            return Err(MetricsError::EmptyInput);
        }

        let len = y_true.len();
        let mut total_sum = F::zero();

        for chunk_start in (0..len).step_by(self.chunk_size) {
            let chunk_end = (chunk_start + self.chunk_size).min(len);
            let true_chunk = y_true.slice(s![chunk_start..chunk_end]);
            let pred_chunk = y_pred.slice(s![chunk_start..chunk_end]);

            let chunk_sum = true_chunk
                .iter()
                .zip(pred_chunk.iter())
                .map(|(t, p)| {
                    let diff = *t - *p;
                    diff * diff
                })
                .fold(F::zero(), |acc, x| acc + x);

            total_sum = total_sum + chunk_sum;
        }

        Ok(total_sum / F::from(len).unwrap())
    }

    /// Compute RMSE using chunked processing
    pub fn chunked_root_mean_squared_error(
        &self,
        y_true: &Array1<F>,
        y_pred: &Array1<F>,
    ) -> MetricsResult<F> {
        Ok(self.chunked_mean_squared_error(y_true, y_pred)?.sqrt())
    }

    /// Compute R² score using chunked processing
    pub fn chunked_r2_score(&self, y_true: &Array1<F>, y_pred: &Array1<F>) -> MetricsResult<F> {
        if y_true.len() != y_pred.len() {
            return Err(MetricsError::ShapeMismatch {
                expected: vec![y_true.len()],
                actual: vec![y_pred.len()],
            });
        }

        if y_true.is_empty() {
            return Err(MetricsError::EmptyInput);
        }

        let len = y_true.len();
        let mut sum_squared_residuals = F::zero();
        let mut sum_true = F::zero();
        let mut sum_true_squared = F::zero();

        // Process chunks to compute sums
        for chunk_start in (0..len).step_by(self.chunk_size) {
            let chunk_end = (chunk_start + self.chunk_size).min(len);
            let true_chunk = y_true.slice(s![chunk_start..chunk_end]);
            let pred_chunk = y_pred.slice(s![chunk_start..chunk_end]);

            for (t, p) in true_chunk.iter().zip(pred_chunk.iter()) {
                let diff = *t - *p;
                sum_squared_residuals = sum_squared_residuals + diff * diff;
                sum_true = sum_true + *t;
                sum_true_squared = sum_true_squared + *t * *t;
            }
        }

        // Calculate R² score
        let n = F::from(len).unwrap();
        let mean_true = sum_true / n;
        let total_sum_squares = sum_true_squared - n * mean_true * mean_true;

        if total_sum_squares == F::zero() {
            return Ok(F::zero()); // Perfect prediction when all true values are the same
        }

        Ok(F::one() - (sum_squared_residuals / total_sum_squares))
    }

    /// Compute cosine similarity using chunked processing
    pub fn chunked_cosine_similarity(&self, a: &Array1<F>, b: &Array1<F>) -> MetricsResult<F> {
        if a.len() != b.len() {
            return Err(MetricsError::ShapeMismatch {
                expected: vec![a.len()],
                actual: vec![b.len()],
            });
        }

        if a.is_empty() {
            return Err(MetricsError::EmptyInput);
        }

        let len = a.len();
        let mut dot_product = F::zero();
        let mut norm_a_squared = F::zero();
        let mut norm_b_squared = F::zero();

        // Process chunks to compute dot product and norms
        for chunk_start in (0..len).step_by(self.chunk_size) {
            let chunk_end = (chunk_start + self.chunk_size).min(len);
            let a_chunk = a.slice(s![chunk_start..chunk_end]);
            let b_chunk = b.slice(s![chunk_start..chunk_end]);

            for (a_val, b_val) in a_chunk.iter().zip(b_chunk.iter()) {
                dot_product = dot_product + *a_val * *b_val;
                norm_a_squared = norm_a_squared + *a_val * *a_val;
                norm_b_squared = norm_b_squared + *b_val * *b_val;
            }
        }

        let norm_a = norm_a_squared.sqrt();
        let norm_b = norm_b_squared.sqrt();

        if norm_a == F::zero() || norm_b == F::zero() {
            return Err(MetricsError::DivisionByZero);
        }

        Ok(dot_product / (norm_a * norm_b))
    }

    /// Compute Pearson correlation coefficient using chunked processing
    pub fn chunked_correlation(&self, x: &Array1<F>, y: &Array1<F>) -> MetricsResult<F> {
        if x.len() != y.len() {
            return Err(MetricsError::ShapeMismatch {
                expected: vec![x.len()],
                actual: vec![y.len()],
            });
        }

        if x.is_empty() {
            return Err(MetricsError::EmptyInput);
        }

        let len = x.len();
        let n = F::from(len).unwrap();
        let mut sum_x = F::zero();
        let mut sum_y = F::zero();
        let mut sum_x_squared = F::zero();
        let mut sum_y_squared = F::zero();
        let mut sum_xy = F::zero();

        // First pass: compute sums
        for chunk_start in (0..len).step_by(self.chunk_size) {
            let chunk_end = (chunk_start + self.chunk_size).min(len);
            let x_chunk = x.slice(s![chunk_start..chunk_end]);
            let y_chunk = y.slice(s![chunk_start..chunk_end]);

            for (x_val, y_val) in x_chunk.iter().zip(y_chunk.iter()) {
                sum_x = sum_x + *x_val;
                sum_y = sum_y + *y_val;
                sum_x_squared = sum_x_squared + *x_val * *x_val;
                sum_y_squared = sum_y_squared + *y_val * *y_val;
                sum_xy = sum_xy + *x_val * *y_val;
            }
        }

        // Calculate correlation coefficient
        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator_x = n * sum_x_squared - sum_x * sum_x;
        let denominator_y = n * sum_y_squared - sum_y * sum_y;

        if denominator_x == F::zero() || denominator_y == F::zero() {
            return Err(MetricsError::DivisionByZero);
        }

        let denominator = (denominator_x * denominator_y).sqrt();
        if denominator == F::zero() {
            return Err(MetricsError::DivisionByZero);
        }

        Ok(numerator / denominator)
    }

    /// Compute multiple regression metrics at once using chunked processing
    pub fn chunked_regression_metrics(
        &self,
        y_true: &Array1<F>,
        y_pred: &Array1<F>,
    ) -> MetricsResult<ChunkedRegressionMetrics<F>> {
        if y_true.len() != y_pred.len() {
            return Err(MetricsError::ShapeMismatch {
                expected: vec![y_true.len()],
                actual: vec![y_pred.len()],
            });
        }

        if y_true.is_empty() {
            return Err(MetricsError::EmptyInput);
        }

        let len = y_true.len();
        let mut sum_absolute_error = F::zero();
        let mut sum_squared_error = F::zero();
        let mut sum_true = F::zero();
        let mut sum_true_squared = F::zero();

        // Single pass through chunks to compute all required sums
        for chunk_start in (0..len).step_by(self.chunk_size) {
            let chunk_end = (chunk_start + self.chunk_size).min(len);
            let true_chunk = y_true.slice(s![chunk_start..chunk_end]);
            let pred_chunk = y_pred.slice(s![chunk_start..chunk_end]);

            for (t, p) in true_chunk.iter().zip(pred_chunk.iter()) {
                let diff = *t - *p;
                sum_absolute_error = sum_absolute_error + diff.abs();
                sum_squared_error = sum_squared_error + diff * diff;
                sum_true = sum_true + *t;
                sum_true_squared = sum_true_squared + *t * *t;
            }
        }

        let n = F::from(len).unwrap();
        let mae = sum_absolute_error / n;
        let mse = sum_squared_error / n;
        let rmse = mse.sqrt();

        // Calculate R²
        let mean_true = sum_true / n;
        let total_sum_squares = sum_true_squared - n * mean_true * mean_true;
        let r2 = if total_sum_squares == F::zero() {
            F::zero()
        } else {
            F::one() - (sum_squared_error / total_sum_squares)
        };

        Ok(ChunkedRegressionMetrics {
            mae,
            mse,
            rmse,
            r2,
            n_samples: len,
        })
    }

    /// Get the current chunk size
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Get the configuration
    pub fn config(&self) -> &OptimizedConfig {
        &self.config
    }
}

/// Result struct for chunked regression metrics computation
#[derive(Debug, Clone)]
pub struct ChunkedRegressionMetrics<F> {
    pub mae: F,
    pub mse: F,
    pub rmse: F,
    pub r2: F,
    pub n_samples: usize,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_chunked_processing_basic() {
        let processor = ChunkedMetricProcessor::new(OptimizedConfig {
            chunk_size: 2,
            ..OptimizedConfig::default()
        });

        let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = array![1.1, 1.9, 3.1, 3.9, 5.1];

        let mae = processor
            .chunked_mean_absolute_error(&y_true, &y_pred)
            .unwrap();
        let mse = processor
            .chunked_mean_squared_error(&y_true, &y_pred)
            .unwrap();

        let expected_mae = crate::regression::mean_absolute_error(&y_true, &y_pred).unwrap();
        let expected_mse = crate::regression::mean_squared_error(&y_true, &y_pred).unwrap();

        assert_relative_eq!(mae, expected_mae, epsilon = 1e-10);
        assert_relative_eq!(mse, expected_mse, epsilon = 1e-10);
    }

    #[test]
    fn test_chunked_r2_score() {
        let processor = ChunkedMetricProcessor::new(OptimizedConfig {
            chunk_size: 3,
            ..OptimizedConfig::default()
        });

        let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y_pred = array![1.1, 2.1, 2.9, 4.1, 4.9, 6.1];

        let r2 = processor.chunked_r2_score(&y_true, &y_pred).unwrap();
        let expected_r2 = crate::regression::r2_score(&y_true, &y_pred).unwrap();

        assert_relative_eq!(r2, expected_r2, epsilon = 1e-10);
    }

    #[test]
    fn test_chunked_cosine_similarity() {
        let processor = ChunkedMetricProcessor::new(OptimizedConfig {
            chunk_size: 2,
            ..OptimizedConfig::default()
        });

        let a = array![1.0, 2.0, 3.0, 4.0];
        let b = array![2.0, 4.0, 6.0, 8.0];

        let similarity = processor.chunked_cosine_similarity(&a, &b).unwrap();

        // Manual calculation: a·b = 60, |a| = √30, |b| = √120, cos = 60/(√30*√120) = 1.0
        assert_relative_eq!(similarity, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_chunked_correlation() {
        let processor = ChunkedMetricProcessor::new(OptimizedConfig {
            chunk_size: 2,
            ..OptimizedConfig::default()
        });

        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect correlation

        let correlation = processor.chunked_correlation(&x, &y).unwrap();
        assert_relative_eq!(correlation, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_chunked_all_metrics() {
        let processor = ChunkedMetricProcessor::new(OptimizedConfig {
            chunk_size: 3,
            ..OptimizedConfig::default()
        });

        let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y_pred = array![1.1, 2.1, 2.9, 4.1, 4.9, 6.1];

        let metrics = processor
            .chunked_regression_metrics(&y_true, &y_pred)
            .unwrap();

        assert_eq!(metrics.n_samples, 6);
        assert!(metrics.mae > 0.0);
        assert!(metrics.mse > 0.0);
        assert!(metrics.rmse > 0.0);
        assert!(metrics.r2 > 0.5); // Should be high correlation
    }

    #[test]
    fn test_custom_chunk_size() {
        let processor =
            ChunkedMetricProcessor::<f64>::with_chunk_size(OptimizedConfig::default(), 10);
        assert_eq!(processor.chunk_size(), 10);
    }
}
