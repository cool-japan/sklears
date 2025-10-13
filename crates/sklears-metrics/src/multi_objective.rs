//! Multi-Objective Evaluation Framework
//!
//! This module provides comprehensive multi-objective evaluation capabilities
//! for comparing machine learning models across multiple performance metrics.
//!
//! # Features
//!
//! - Pareto frontier analysis for finding non-dominated solutions
//! - Multi-criteria decision analysis (MCDA) methods
//! - Trade-off analysis between competing metrics
//! - Utility function optimization frameworks
//! - TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
//! - Model ranking and selection utilities
//!
//! # Examples
//!
//! ```rust
//! use sklears_metrics::multi_objective::*;
//! use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
//!
//! // Define performance metrics for multiple models
//! let metrics = Array2::from_shape_vec((3, 2), vec![
//!     0.95, 0.1,   // Model 1: high accuracy, low inference time
//!     0.90, 0.05,  // Model 2: good accuracy, very low inference time  
//!     0.85, 0.02   // Model 3: decent accuracy, minimal inference time
//! ]).unwrap();
//!
//! // Find Pareto frontier (maximize both metrics)
//! let pareto_indices = pareto_frontier(&metrics, &[true, true]).unwrap();
//! println!("Pareto optimal models: {:?}", pareto_indices);
//!
//! // Perform TOPSIS ranking
//! let weights = Array1::from_vec(vec![0.7, 0.3]); // Prioritize accuracy
//! let ranking = topsis_ranking(&metrics, &weights, &[true, true]).unwrap();
//! println!("TOPSIS ranking: {:?}", ranking);
//! ```

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Represents a multi-objective evaluation result
#[derive(Debug, Clone)]
pub struct MultiObjectiveResult {
    /// Model performance metrics
    pub metrics: Array2<f64>,
    /// Model names/identifiers
    pub model_names: Vec<String>,
    /// Metric names
    pub metric_names: Vec<String>,
    /// Pareto frontier indices
    pub pareto_indices: Vec<usize>,
    /// Ranking scores
    pub ranking_scores: Array1<f64>,
    /// Trade-off analysis results
    pub trade_offs: HashMap<String, f64>,
}

/// Configuration for multi-objective evaluation
#[derive(Debug, Clone)]
pub struct MultiObjectiveConfig {
    /// Weights for each metric (sum should be 1.0)
    pub weights: Array1<f64>,
    /// Whether each metric should be maximized (true) or minimized (false)
    pub maximize: Vec<bool>,
    /// Normalization method for metrics
    pub normalization: NormalizationMethod,
    /// Distance metric for similarity calculations
    pub distance_metric: DistanceMetric,
}

/// Normalization methods for multi-objective evaluation
#[derive(Debug, Clone, Copy)]
pub enum NormalizationMethod {
    /// Min-max normalization to [0, 1]
    MinMax,
    /// Z-score normalization (mean=0, std=1)
    ZScore,
    /// Vector normalization (unit vector)
    Vector,
    /// No normalization
    None,
}

/// Distance metrics for multi-objective evaluation
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Manhattan distance  
    Manhattan,
    /// Chebyshev distance
    Chebyshev,
    /// Cosine distance
    Cosine,
}

/// Find Pareto frontier (non-dominated solutions)
///
/// # Arguments
///
/// * `metrics` - Matrix of performance metrics (models x metrics)
/// * `maximize` - Whether each metric should be maximized
///
/// # Returns
///
/// Vector of indices representing Pareto optimal solutions
pub fn pareto_frontier(metrics: &Array2<f64>, maximize: &[bool]) -> MetricsResult<Vec<usize>> {
    if metrics.nrows() == 0 || metrics.ncols() == 0 {
        return Err(MetricsError::EmptyInput);
    }

    if maximize.len() != metrics.ncols() {
        return Err(MetricsError::InvalidParameter(
            "maximize array length must match number of metrics".to_string(),
        ));
    }

    let mut pareto_indices = Vec::new();
    let n_models = metrics.nrows();

    for i in 0..n_models {
        let mut is_dominated = false;

        for j in 0..n_models {
            if i == j {
                continue;
            }

            if dominates(metrics, j, i, maximize) {
                is_dominated = true;
                break;
            }
        }

        if !is_dominated {
            pareto_indices.push(i);
        }
    }

    Ok(pareto_indices)
}

/// Check if model i dominates model j
fn dominates(metrics: &Array2<f64>, i: usize, j: usize, maximize: &[bool]) -> bool {
    let mut strictly_better = false;

    for k in 0..metrics.ncols() {
        let val_i = metrics[[i, k]];
        let val_j = metrics[[j, k]];

        if maximize[k] {
            if val_i < val_j {
                return false; // i is worse than j in this metric
            }
            if val_i > val_j {
                strictly_better = true;
            }
        } else {
            if val_i > val_j {
                return false; // i is worse than j in this metric
            }
            if val_i < val_j {
                strictly_better = true;
            }
        }
    }

    strictly_better
}

/// TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
///
/// # Arguments
///
/// * `metrics` - Matrix of performance metrics (models x metrics)
/// * `weights` - Weights for each metric
/// * `maximize` - Whether each metric should be maximized
///
/// # Returns
///
/// Vector of TOPSIS scores (higher is better)
pub fn topsis_ranking(
    metrics: &Array2<f64>,
    weights: &Array1<f64>,
    maximize: &[bool],
) -> MetricsResult<Array1<f64>> {
    if metrics.nrows() == 0 || metrics.ncols() == 0 {
        return Err(MetricsError::EmptyInput);
    }

    if weights.len() != metrics.ncols() {
        return Err(MetricsError::InvalidParameter(
            "weights length must match number of metrics".to_string(),
        ));
    }

    if maximize.len() != metrics.ncols() {
        return Err(MetricsError::InvalidParameter(
            "maximize array length must match number of metrics".to_string(),
        ));
    }

    // Step 1: Normalize the decision matrix
    let normalized = normalize_matrix(metrics, &NormalizationMethod::Vector)?;

    // Step 2: Calculate weighted normalized decision matrix
    let weighted: Array2<f64> = normalized * weights;

    // Step 3: Determine ideal and negative-ideal solutions
    let (ideal_solution, negative_ideal_solution) = find_ideal_solutions(&weighted, maximize)?;

    // Step 4: Calculate separation measures
    let mut scores = Array1::zeros(metrics.nrows());

    for i in 0..metrics.nrows() {
        let alternative = weighted.row(i);

        let dist_to_ideal = euclidean_distance(&alternative, &ideal_solution);
        let dist_to_negative_ideal = euclidean_distance(&alternative, &negative_ideal_solution);

        // Step 5: Calculate relative closeness to ideal solution
        if dist_to_ideal + dist_to_negative_ideal > 0.0 {
            scores[i] = dist_to_negative_ideal / (dist_to_ideal + dist_to_negative_ideal);
        } else {
            scores[i] = 0.0;
        }
    }

    Ok(scores)
}

/// Normalize a matrix using the specified method
fn normalize_matrix(
    matrix: &Array2<f64>,
    method: &NormalizationMethod,
) -> MetricsResult<Array2<f64>> {
    match method {
        NormalizationMethod::MinMax => {
            let mut normalized = matrix.clone();

            for j in 0..matrix.ncols() {
                let col = matrix.column(j);
                let min_val = col.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max_val = col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                if (max_val - min_val).abs() > f64::EPSILON {
                    for i in 0..matrix.nrows() {
                        normalized[[i, j]] = (matrix[[i, j]] - min_val) / (max_val - min_val);
                    }
                }
            }

            Ok(normalized)
        }
        NormalizationMethod::ZScore => {
            let mut normalized = matrix.clone();

            for j in 0..matrix.ncols() {
                let col = matrix.column(j);
                let mean = col.mean().unwrap_or(0.0);
                let std = col.std(0.0);

                if std > f64::EPSILON {
                    for i in 0..matrix.nrows() {
                        normalized[[i, j]] = (matrix[[i, j]] - mean) / std;
                    }
                }
            }

            Ok(normalized)
        }
        NormalizationMethod::Vector => {
            let mut normalized = matrix.clone();

            for j in 0..matrix.ncols() {
                let col = matrix.column(j);
                let norm = col.iter().map(|x| x * x).sum::<f64>().sqrt();

                if norm > f64::EPSILON {
                    for i in 0..matrix.nrows() {
                        normalized[[i, j]] = matrix[[i, j]] / norm;
                    }
                }
            }

            Ok(normalized)
        }
        NormalizationMethod::None => Ok(matrix.clone()),
    }
}

/// Find ideal and negative-ideal solutions for TOPSIS
fn find_ideal_solutions(
    weighted_matrix: &Array2<f64>,
    maximize: &[bool],
) -> MetricsResult<(Array1<f64>, Array1<f64>)> {
    let n_metrics = weighted_matrix.ncols();
    let mut ideal_solution = Array1::zeros(n_metrics);
    let mut negative_ideal_solution = Array1::zeros(n_metrics);

    for j in 0..n_metrics {
        let col = weighted_matrix.column(j);
        let min_val = col.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if maximize[j] {
            ideal_solution[j] = max_val;
            negative_ideal_solution[j] = min_val;
        } else {
            ideal_solution[j] = min_val;
            negative_ideal_solution[j] = max_val;
        }
    }

    Ok((ideal_solution, negative_ideal_solution))
}

/// Calculate Euclidean distance between two vectors
fn euclidean_distance(a: &scirs2_core::ndarray::ArrayView1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Multi-criteria decision analysis using weighted sum
///
/// # Arguments
///
/// * `metrics` - Matrix of performance metrics (models x metrics)
/// * `weights` - Weights for each metric
/// * `maximize` - Whether each metric should be maximized
///
/// # Returns
///
/// Vector of weighted scores (higher is better)
pub fn weighted_sum_ranking(
    metrics: &Array2<f64>,
    weights: &Array1<f64>,
    maximize: &[bool],
) -> MetricsResult<Array1<f64>> {
    if metrics.nrows() == 0 || metrics.ncols() == 0 {
        return Err(MetricsError::EmptyInput);
    }

    if weights.len() != metrics.ncols() {
        return Err(MetricsError::InvalidParameter(
            "weights length must match number of metrics".to_string(),
        ));
    }

    if maximize.len() != metrics.ncols() {
        return Err(MetricsError::InvalidParameter(
            "maximize array length must match number of metrics".to_string(),
        ));
    }

    // Normalize metrics to [0, 1] range
    let normalized = normalize_matrix(metrics, &NormalizationMethod::MinMax)?;

    let mut scores = Array1::zeros(metrics.nrows());

    for i in 0..metrics.nrows() {
        let mut score = 0.0;

        for j in 0..metrics.ncols() {
            let metric_value = if maximize[j] {
                normalized[[i, j]]
            } else {
                1.0 - normalized[[i, j]]
            };

            score += weights[j] * metric_value;
        }

        scores[i] = score;
    }

    Ok(scores)
}

/// Calculate trade-off ratios between metrics
///
/// # Arguments
///
/// * `metrics` - Matrix of performance metrics (models x metrics)
/// * `metric_names` - Names of metrics for reporting
///
/// # Returns
///
/// HashMap of trade-off ratios between metric pairs
pub fn trade_off_analysis(
    metrics: &Array2<f64>,
    metric_names: &[String],
) -> MetricsResult<HashMap<String, f64>> {
    if metrics.nrows() == 0 || metrics.ncols() == 0 {
        return Err(MetricsError::EmptyInput);
    }

    if metric_names.len() != metrics.ncols() {
        return Err(MetricsError::InvalidParameter(
            "metric_names length must match number of metrics".to_string(),
        ));
    }

    let mut trade_offs = HashMap::new();

    for i in 0..metrics.ncols() {
        for j in (i + 1)..metrics.ncols() {
            let col_i = metrics.column(i);
            let col_j = metrics.column(j);

            // Calculate correlation coefficient as trade-off measure
            let correlation = pearson_correlation(&col_i, &col_j)?;

            let key = format!("{}_vs_{}", metric_names[i], metric_names[j]);
            trade_offs.insert(key, correlation);
        }
    }

    Ok(trade_offs)
}

/// Calculate Pearson correlation coefficient
fn pearson_correlation(
    x: &scirs2_core::ndarray::ArrayView1<f64>,
    y: &scirs2_core::ndarray::ArrayView1<f64>,
) -> MetricsResult<f64> {
    if x.len() != y.len() {
        return Err(MetricsError::InvalidParameter(
            "arrays must have the same length".to_string(),
        ));
    }

    let n = x.len() as f64;
    if n < 2.0 {
        return Err(MetricsError::InvalidParameter(
            "need at least 2 data points".to_string(),
        ));
    }

    let mean_x = x.mean().unwrap_or(0.0);
    let mean_y = y.mean().unwrap_or(0.0);

    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;

    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        sum_xy += dx * dy;
        sum_x2 += dx * dx;
        sum_y2 += dy * dy;
    }

    let denominator = (sum_x2 * sum_y2).sqrt();

    if denominator < f64::EPSILON {
        Ok(0.0)
    } else {
        Ok(sum_xy / denominator)
    }
}

/// Comprehensive multi-objective evaluation
///
/// # Arguments
///
/// * `metrics` - Matrix of performance metrics (models x metrics)
/// * `model_names` - Names of models
/// * `metric_names` - Names of metrics
/// * `config` - Configuration for evaluation
///
/// # Returns
///
/// Complete multi-objective evaluation result
pub fn multi_objective_evaluation(
    metrics: &Array2<f64>,
    model_names: &[String],
    metric_names: &[String],
    config: &MultiObjectiveConfig,
) -> MetricsResult<MultiObjectiveResult> {
    if metrics.nrows() == 0 || metrics.ncols() == 0 {
        return Err(MetricsError::EmptyInput);
    }

    if model_names.len() != metrics.nrows() {
        return Err(MetricsError::InvalidParameter(
            "model_names length must match number of models".to_string(),
        ));
    }

    if metric_names.len() != metrics.ncols() {
        return Err(MetricsError::InvalidParameter(
            "metric_names length must match number of metrics".to_string(),
        ));
    }

    // Find Pareto frontier
    let pareto_indices = pareto_frontier(metrics, &config.maximize)?;

    // Calculate ranking scores using TOPSIS
    let ranking_scores = topsis_ranking(metrics, &config.weights, &config.maximize)?;

    // Perform trade-off analysis
    let trade_offs = trade_off_analysis(metrics, metric_names)?;

    Ok(MultiObjectiveResult {
        metrics: metrics.clone(),
        model_names: model_names.to_vec(),
        metric_names: metric_names.to_vec(),
        pareto_indices,
        ranking_scores,
        trade_offs,
    })
}

/// Utility function optimization using genetic algorithm approach
///
/// # Arguments
///
/// * `metrics` - Matrix of performance metrics (models x metrics)
/// * `utility_func` - Custom utility function
/// * `maximize` - Whether each metric should be maximized
///
/// # Returns
///
/// Vector of utility scores for each model
pub fn utility_optimization<F>(
    metrics: &Array2<f64>,
    utility_func: F,
    maximize: &[bool],
) -> MetricsResult<Array1<f64>>
where
    F: Fn(&scirs2_core::ndarray::ArrayView1<f64>) -> f64,
{
    if metrics.nrows() == 0 || metrics.ncols() == 0 {
        return Err(MetricsError::EmptyInput);
    }

    if maximize.len() != metrics.ncols() {
        return Err(MetricsError::InvalidParameter(
            "maximize array length must match number of metrics".to_string(),
        ));
    }

    // Normalize metrics
    let normalized = normalize_matrix(metrics, &NormalizationMethod::MinMax)?;

    let mut utility_scores = Array1::zeros(metrics.nrows());

    for i in 0..metrics.nrows() {
        let mut model_metrics = normalized.row(i).to_owned();

        // Convert minimization metrics to maximization
        for j in 0..metrics.ncols() {
            if !maximize[j] {
                model_metrics[j] = 1.0 - model_metrics[j];
            }
        }

        utility_scores[i] = utility_func(&model_metrics.view());
    }

    Ok(utility_scores)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_pareto_frontier() {
        let metrics = Array2::from_shape_vec(
            (4, 2),
            vec![
                0.9, 0.8, // Model 0: high acc, high speed - clearly best
                0.8, 0.7, // Model 1: good acc, good speed - dominated by 0
                0.7, 0.6, // Model 2: ok acc, ok speed - dominated by 0 and 1
                0.6, 0.9, // Model 3: low acc, very high speed - Pareto optimal
            ],
        )
        .unwrap();

        let maximize = vec![true, true]; // Maximize both accuracy and speed
        let pareto = pareto_frontier(&metrics, &maximize).unwrap();

        // Models 0 and 3 should be Pareto optimal
        assert_eq!(pareto, vec![0, 3]);
    }

    #[test]
    fn test_topsis_ranking() {
        let metrics = Array2::from_shape_vec(
            (3, 2),
            vec![
                0.9, 0.8, // Model 0: high accuracy, high speed - clearly best
                0.5, 0.5, // Model 1: medium accuracy, medium speed - middle
                0.2, 0.3, // Model 2: low accuracy, low speed - clearly worst
            ],
        )
        .unwrap();

        let weights = Array1::from_vec(vec![0.7, 0.3]);
        let maximize = vec![true, true];

        let scores = topsis_ranking(&metrics, &weights, &maximize).unwrap();

        // Model 0 should have highest score, Model 2 should have lowest
        assert!(scores[0] > scores[1]);
        assert!(scores[1] > scores[2]);
    }

    #[test]
    fn test_weighted_sum_ranking() {
        let metrics = Array2::from_shape_vec(
            (3, 2),
            vec![
                0.9, 0.1, // Model 0
                0.8, 0.2, // Model 1
                0.7, 0.3, // Model 2
            ],
        )
        .unwrap();

        let weights = Array1::from_vec(vec![0.7, 0.3]);
        let maximize = vec![true, true];

        let scores = weighted_sum_ranking(&metrics, &weights, &maximize).unwrap();

        // Model 0 should have highest score
        assert!(scores[0] > scores[1]);
        assert!(scores[1] > scores[2]);
    }

    #[test]
    fn test_trade_off_analysis() {
        let metrics = Array2::from_shape_vec(
            (3, 2),
            vec![
                0.9, 0.1, // High accuracy, low speed
                0.8, 0.2, // Medium accuracy, medium speed
                0.7, 0.3, // Low accuracy, high speed
            ],
        )
        .unwrap();

        let metric_names = vec!["accuracy".to_string(), "speed".to_string()];

        let trade_offs = trade_off_analysis(&metrics, &metric_names).unwrap();

        // Should have one trade-off entry
        assert_eq!(trade_offs.len(), 1);
        assert!(trade_offs.contains_key("accuracy_vs_speed"));

        // Should show negative correlation (trade-off)
        let correlation = trade_offs["accuracy_vs_speed"];
        assert!(correlation < 0.0);
    }

    #[test]
    fn test_normalization_methods() {
        let metrics =
            Array2::from_shape_vec((3, 2), vec![10.0, 1.0, 20.0, 2.0, 30.0, 3.0]).unwrap();

        // Test MinMax normalization
        let normalized = normalize_matrix(&metrics, &NormalizationMethod::MinMax).unwrap();
        assert_abs_diff_eq!(normalized[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(normalized[[2, 0]], 1.0, epsilon = 1e-10);

        // Test Vector normalization
        let normalized = normalize_matrix(&metrics, &NormalizationMethod::Vector).unwrap();
        let col0_norm = normalized
            .column(0)
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        assert_abs_diff_eq!(col0_norm, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_utility_optimization() {
        let metrics = Array2::from_shape_vec((3, 2), vec![0.9, 0.1, 0.8, 0.2, 0.7, 0.3]).unwrap();

        let maximize = vec![true, true];

        // Simple utility function: weighted sum
        let utility_func =
            |metrics: &scirs2_core::ndarray::ArrayView1<f64>| 0.7 * metrics[0] + 0.3 * metrics[1];

        let scores = utility_optimization(&metrics, utility_func, &maximize).unwrap();

        // First model should have highest utility
        assert!(scores[0] > scores[1]);
        assert!(scores[1] > scores[2]);
    }

    #[test]
    fn test_multi_objective_evaluation() {
        let metrics = Array2::from_shape_vec((3, 2), vec![0.9, 0.1, 0.8, 0.2, 0.7, 0.3]).unwrap();

        let model_names = vec![
            "Model1".to_string(),
            "Model2".to_string(),
            "Model3".to_string(),
        ];
        let metric_names = vec!["accuracy".to_string(), "speed".to_string()];

        let config = MultiObjectiveConfig {
            weights: Array1::from_vec(vec![0.7, 0.3]),
            maximize: vec![true, true],
            normalization: NormalizationMethod::MinMax,
            distance_metric: DistanceMetric::Euclidean,
        };

        let result =
            multi_objective_evaluation(&metrics, &model_names, &metric_names, &config).unwrap();

        assert_eq!(result.model_names.len(), 3);
        assert_eq!(result.metric_names.len(), 2);
        assert!(!result.pareto_indices.is_empty());
        assert_eq!(result.ranking_scores.len(), 3);
        assert!(!result.trade_offs.is_empty());
    }
}
