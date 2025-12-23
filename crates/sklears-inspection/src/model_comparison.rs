//! Model comparison and analysis tools
//!
//! This module provides comprehensive tools for comparing multiple models, analyzing ensembles,
//! computing model agreement metrics, and assessing prediction stability.

// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::{Rng, SeedableRng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};
use std::collections::HashMap;

/// Model comparison result
#[derive(Debug, Clone)]
pub struct ModelComparisonResult {
    /// Individual model performance metrics
    pub individual_metrics: Vec<ModelMetrics>,
    /// Pairwise agreement metrics between models
    pub agreement_metrics: Array2<Float>,
    /// Ensemble performance (if applicable)
    pub ensemble_metrics: Option<ModelMetrics>,
    /// Diversity metrics
    pub diversity_metrics: DiversityMetrics,
    /// Stability analysis results
    pub stability_metrics: StabilityMetrics,
}

/// Performance metrics for a single model
#[derive(Debug, Clone)]
pub struct ModelMetrics {
    /// Model identifier
    pub model_id: String,
    /// Mean squared error (for regression)
    pub mse: Option<Float>,
    /// Mean absolute error
    pub mae: Option<Float>,
    /// R-squared (for regression)
    pub r2: Option<Float>,
    /// Accuracy (for classification)
    pub accuracy: Option<Float>,
    /// Precision (for classification)
    pub precision: Option<Float>,
    /// Recall (for classification)
    pub recall: Option<Float>,
    /// F1 score (for classification)
    pub f1_score: Option<Float>,
    /// Custom metrics
    pub custom_metrics: HashMap<String, Float>,
}

/// Diversity metrics for model ensembles
#[derive(Debug, Clone)]
pub struct DiversityMetrics {
    /// Q-statistic (measures pairwise diversity)
    pub q_statistic: Float,
    /// Correlation coefficient
    pub correlation_coefficient: Float,
    /// Disagreement measure
    pub disagreement_measure: Float,
    /// Double fault measure
    pub double_fault_measure: Float,
    /// Kohavi-Wolpert variance
    pub kw_variance: Float,
}

/// Stability metrics for model predictions
#[derive(Debug, Clone)]
pub struct StabilityMetrics {
    /// Prediction variance across bootstrap samples
    pub prediction_variance: Float,
    /// Stability index
    pub stability_index: Float,
    /// Robustness to data perturbation
    pub perturbation_robustness: Float,
    /// Feature importance stability
    pub feature_importance_stability: Option<Float>,
}

/// Configuration for model comparison
#[derive(Debug, Clone)]
pub struct ModelComparisonConfig {
    /// Bootstrap samples for stability analysis
    pub n_bootstrap_samples: usize,
    /// Perturbation magnitude for robustness testing
    pub perturbation_magnitude: Float,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
    /// Whether to compute feature importance stability
    pub compute_feature_importance_stability: bool,
    /// Cross-validation folds for stability assessment
    pub cv_folds: usize,
}

impl Default for ModelComparisonConfig {
    fn default() -> Self {
        Self {
            n_bootstrap_samples: 100,
            perturbation_magnitude: 0.1,
            random_state: None,
            compute_feature_importance_stability: false,
            cv_folds: 5,
        }
    }
}

/// Ensemble method for combining model predictions
#[derive(Debug, Clone, Copy)]
pub enum EnsembleMethod {
    /// Simple averaging
    Average,
    /// Weighted averaging
    WeightedAverage,
    /// Majority voting (for classification)
    MajorityVoting,
    /// Median aggregation
    Median,
    /// Stacking (meta-learner)
    Stacking,
}

/// Compare multiple models on the same dataset
///
/// Provides comprehensive comparison including performance metrics, agreement analysis,
/// and diversity assessment.
pub fn compare_models<F>(
    models: &[F],
    model_ids: &[String],
    X: &ArrayView2<Float>,
    y_true: &ArrayView1<Float>,
    config: &ModelComparisonConfig,
) -> SklResult<ModelComparisonResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    if models.len() != model_ids.len() {
        return Err(SklearsError::InvalidInput(
            "Number of models must match number of model IDs".to_string(),
        ));
    }

    if X.nrows() != y_true.len() {
        return Err(SklearsError::InvalidInput(
            "X and y must have same number of samples".to_string(),
        ));
    }

    let n_models = models.len();
    let n_samples = X.nrows();

    // Get predictions from all models
    let mut all_predictions = Vec::new();
    for model in models {
        let predictions = model(&X.view());
        if predictions.len() != n_samples {
            return Err(SklearsError::InvalidInput(
                "Model predictions must match number of samples".to_string(),
            ));
        }
        all_predictions.push(Array1::from_vec(predictions));
    }

    // Compute individual model metrics
    let mut individual_metrics = Vec::new();
    for (i, model_id) in model_ids.iter().enumerate() {
        let metrics =
            compute_model_metrics(model_id.clone(), &all_predictions[i].view(), &y_true.view())?;
        individual_metrics.push(metrics);
    }

    // Compute pairwise agreement metrics
    let agreement_metrics = compute_agreement_metrics(&all_predictions, y_true)?;

    // Compute ensemble metrics
    let ensemble_predictions =
        create_ensemble_predictions(&all_predictions, EnsembleMethod::Average)?;
    let ensemble_metrics = Some(compute_model_metrics(
        "Ensemble".to_string(),
        &ensemble_predictions.view(),
        &y_true.view(),
    )?);

    // Compute diversity metrics
    let diversity_metrics = compute_diversity_metrics(&all_predictions, y_true)?;

    // Compute stability metrics
    let stability_metrics = compute_stability_metrics(models, X, y_true, config)?;

    Ok(ModelComparisonResult {
        individual_metrics,
        agreement_metrics,
        ensemble_metrics,
        diversity_metrics,
        stability_metrics,
    })
}

/// Analyze ensemble model performance and characteristics
pub fn analyze_ensemble<F>(
    models: &[F],
    weights: Option<&ArrayView1<Float>>,
    X: &ArrayView2<Float>,
    y_true: &ArrayView1<Float>,
    method: EnsembleMethod,
) -> SklResult<EnsembleAnalysisResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let n_models = models.len();
    let n_samples = X.nrows();

    if let Some(w) = weights {
        if w.len() != n_models {
            return Err(SklearsError::InvalidInput(
                "Weights length must match number of models".to_string(),
            ));
        }
    }

    // Get individual predictions
    let mut individual_predictions = Vec::new();
    let mut individual_errors = Vec::new();

    for model in models {
        let predictions = Array1::from_vec(model(&X.view()));
        let errors = &predictions - y_true;
        individual_predictions.push(predictions);
        individual_errors.push(errors);
    }

    // Create ensemble prediction
    let ensemble_prediction = match method {
        EnsembleMethod::Average => {
            let mut sum = Array1::zeros(n_samples);
            for pred in &individual_predictions {
                sum += pred;
            }
            sum / n_models as Float
        }
        EnsembleMethod::WeightedAverage => {
            if let Some(w) = weights {
                let mut weighted_sum = Array1::zeros(n_samples);
                for (i, pred) in individual_predictions.iter().enumerate() {
                    weighted_sum = weighted_sum + pred * w[i];
                }
                weighted_sum
            } else {
                return Err(SklearsError::InvalidInput(
                    "Weights required for weighted average".to_string(),
                ));
            }
        }
        EnsembleMethod::Median => {
            let mut ensemble = Array1::zeros(n_samples);
            for i in 0..n_samples {
                let mut values: Vec<Float> =
                    individual_predictions.iter().map(|pred| pred[i]).collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                ensemble[i] = if values.len() % 2 == 0 {
                    (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
                } else {
                    values[values.len() / 2]
                };
            }
            ensemble
        }
        _ => {
            return Err(SklearsError::InvalidInput(
                "Ensemble method not implemented".to_string(),
            ));
        }
    };

    let ensemble_error = &ensemble_prediction - y_true;

    // Compute bias-variance decomposition
    let bias_variance = compute_bias_variance_decomposition(&individual_predictions, y_true)?;

    // Compute contribution analysis
    let contributions =
        compute_model_contributions(&individual_predictions, &ensemble_prediction, weights)?;

    Ok(EnsembleAnalysisResult {
        ensemble_prediction,
        ensemble_error: ensemble_error.mapv(|x| x.abs()).mean().unwrap(),
        individual_contributions: contributions,
        bias_variance_decomposition: bias_variance,
        method,
    })
}

/// Result of ensemble analysis
#[derive(Debug, Clone)]
pub struct EnsembleAnalysisResult {
    /// Final ensemble predictions
    pub ensemble_prediction: Array1<Float>,
    /// Ensemble prediction error
    pub ensemble_error: Float,
    /// Individual model contributions to ensemble
    pub individual_contributions: Vec<Float>,
    /// Bias-variance decomposition
    pub bias_variance_decomposition: BiasVarianceDecomposition,
    /// Ensemble method used
    pub method: EnsembleMethod,
}

/// Bias-variance decomposition results
#[derive(Debug, Clone)]
pub struct BiasVarianceDecomposition {
    /// Bias component
    pub bias: Float,
    /// Variance component
    pub variance: Float,
    /// Noise component
    pub noise: Float,
    /// Total error
    pub total_error: Float,
}

/// Compute model agreement metrics
pub fn compute_model_agreement<F>(
    models: &[F],
    X: &ArrayView2<Float>,
    agreement_threshold: Float,
) -> SklResult<Array2<Float>>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let n_models = models.len();
    let mut predictions = Vec::new();

    for model in models {
        predictions.push(Array1::from_vec(model(&X.view())));
    }

    compute_agreement_metrics(&predictions, &Array1::zeros(X.nrows()).view())
}

/// Assess prediction stability across different data subsets
pub fn assess_prediction_stability<F>(
    model: &F,
    X: &ArrayView2<Float>,
    config: &ModelComparisonConfig,
) -> SklResult<StabilityMetrics>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let n_samples = X.nrows();
    let mut bootstrap_predictions = Vec::new();

    use scirs2_core::random::SeedableRng;
    let mut rng = match config.random_state {
        Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
        None => scirs2_core::random::rngs::StdRng::seed_from_u64(42),
    };

    // Bootstrap sampling for stability assessment
    for _ in 0..config.n_bootstrap_samples {
        let mut bootstrap_indices = Vec::new();
        for _ in 0..n_samples {
            bootstrap_indices.push(rng.gen_range(0..n_samples));
        }

        let mut bootstrap_X = Array2::zeros((n_samples, X.ncols()));
        for (i, &idx) in bootstrap_indices.iter().enumerate() {
            for j in 0..X.ncols() {
                bootstrap_X[[i, j]] = X[[idx, j]];
            }
        }

        let predictions = model(&bootstrap_X.view());
        bootstrap_predictions.push(Array1::from_vec(predictions));
    }

    // Compute prediction variance
    let mean_predictions = compute_mean_predictions(&bootstrap_predictions);
    let mut variance_sum = 0.0;
    for pred in &bootstrap_predictions {
        let diff = pred - &mean_predictions;
        variance_sum += diff.mapv(|x| x.powi(2)).sum();
    }
    let prediction_variance = variance_sum / (bootstrap_predictions.len() * n_samples) as Float;

    // Compute stability index (1 - coefficient of variation)
    let cv =
        (prediction_variance.sqrt()) / mean_predictions.mapv(|x| x.abs()).mean().unwrap_or(1.0);
    let stability_index = 1.0 - cv;

    // Assess robustness to data perturbation
    let perturbation_robustness = assess_perturbation_robustness(model, X, config)?;

    Ok(StabilityMetrics {
        prediction_variance,
        stability_index,
        perturbation_robustness,
        feature_importance_stability: None, // Would need feature importance computation
    })
}

// Helper functions

fn compute_model_metrics(
    model_id: String,
    y_pred: &ArrayView1<Float>,
    y_true: &ArrayView1<Float>,
) -> SklResult<ModelMetrics> {
    let n = y_pred.len();
    if n != y_true.len() {
        return Err(SklearsError::InvalidInput(
            "Predictions and true values must have same length".to_string(),
        ));
    }

    // Compute regression metrics
    let residuals = y_pred - y_true;
    let mse = residuals.mapv(|x| x.powi(2)).mean().unwrap();
    let mae = residuals.mapv(|x| x.abs()).mean().unwrap();

    // R-squared
    let y_mean = y_true.mean().unwrap();
    let ss_tot = y_true.mapv(|x| (x - y_mean).powi(2)).sum();
    let ss_res = residuals.mapv(|x| x.powi(2)).sum();
    let r2 = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    Ok(ModelMetrics {
        model_id,
        mse: Some(mse),
        mae: Some(mae),
        r2: Some(r2),
        accuracy: None,
        precision: None,
        recall: None,
        f1_score: None,
        custom_metrics: HashMap::new(),
    })
}

fn compute_agreement_metrics(
    predictions: &[Array1<Float>],
    y_true: &ArrayView1<Float>,
) -> SklResult<Array2<Float>> {
    let n_models = predictions.len();
    let mut agreement_matrix = Array2::zeros((n_models, n_models));

    for i in 0..n_models {
        for j in 0..n_models {
            if i == j {
                agreement_matrix[[i, j]] = 1.0;
            } else {
                // Compute correlation coefficient
                let corr = compute_correlation(&predictions[i].view(), &predictions[j].view())?;
                agreement_matrix[[i, j]] = corr;
            }
        }
    }

    Ok(agreement_matrix)
}

fn compute_correlation(x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> SklResult<Float> {
    if x.len() != y.len() {
        return Err(SklearsError::InvalidInput(
            "Arrays must have same length".to_string(),
        ));
    }

    let n = x.len() as Float;
    let x_mean = x.mean().unwrap();
    let y_mean = y.mean().unwrap();

    let numerator: Float = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean))
        .sum();

    let x_var: Float = x.iter().map(|&xi| (xi - x_mean).powi(2)).sum();
    let y_var: Float = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();

    let denominator = (x_var * y_var).sqrt();

    if denominator > 1e-10 {
        Ok(numerator / denominator)
    } else {
        Ok(0.0)
    }
}

fn create_ensemble_predictions(
    predictions: &[Array1<Float>],
    method: EnsembleMethod,
) -> SklResult<Array1<Float>> {
    if predictions.is_empty() {
        return Err(SklearsError::InvalidInput(
            "No predictions provided".to_string(),
        ));
    }

    let n_samples = predictions[0].len();
    let n_models = predictions.len() as Float;

    match method {
        EnsembleMethod::Average => {
            let mut ensemble = Array1::zeros(n_samples);
            for pred in predictions {
                ensemble += pred;
            }
            Ok(ensemble / n_models)
        }
        EnsembleMethod::Median => {
            let mut ensemble = Array1::zeros(n_samples);
            for i in 0..n_samples {
                let mut values: Vec<Float> = predictions.iter().map(|pred| pred[i]).collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                ensemble[i] = if values.len() % 2 == 0 {
                    (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
                } else {
                    values[values.len() / 2]
                };
            }
            Ok(ensemble)
        }
        _ => Err(SklearsError::InvalidInput(
            "Ensemble method not implemented".to_string(),
        )),
    }
}

fn compute_diversity_metrics(
    predictions: &[Array1<Float>],
    y_true: &ArrayView1<Float>,
) -> SklResult<DiversityMetrics> {
    let n_models = predictions.len();

    // Compute Q-statistic (average pairwise correlation)
    let mut q_sum = 0.0;
    let mut count = 0;

    for i in 0..n_models {
        for j in (i + 1)..n_models {
            let corr = compute_correlation(&predictions[i].view(), &predictions[j].view())?;
            q_sum += corr;
            count += 1;
        }
    }

    let q_statistic = if count > 0 {
        q_sum / count as Float
    } else {
        0.0
    };

    // Compute disagreement measure
    let mut disagreements = 0;
    let mut total_pairs = 0;

    for sample_idx in 0..predictions[0].len() {
        for i in 0..n_models {
            for j in (i + 1)..n_models {
                let pred_i = predictions[i][sample_idx];
                let pred_j = predictions[j][sample_idx];
                if (pred_i - pred_j).abs() > 0.1 {
                    // Threshold for disagreement
                    disagreements += 1;
                }
                total_pairs += 1;
            }
        }
    }

    let disagreement_measure = disagreements as Float / total_pairs as Float;

    // Simplified diversity metrics (would be more complex for classification)
    Ok(DiversityMetrics {
        q_statistic,
        correlation_coefficient: q_statistic,
        disagreement_measure,
        double_fault_measure: 0.0, // Simplified
        kw_variance: 0.0,          // Simplified
    })
}

fn compute_stability_metrics<F>(
    models: &[F],
    X: &ArrayView2<Float>,
    y_true: &ArrayView1<Float>,
    config: &ModelComparisonConfig,
) -> SklResult<StabilityMetrics>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    // Use first model for stability assessment
    if models.is_empty() {
        return Ok(StabilityMetrics {
            prediction_variance: 0.0,
            stability_index: 0.0,
            perturbation_robustness: 0.0,
            feature_importance_stability: None,
        });
    }

    assess_prediction_stability(&models[0], X, config)
}

fn compute_mean_predictions(predictions: &[Array1<Float>]) -> Array1<Float> {
    if predictions.is_empty() {
        return Array1::zeros(0);
    }

    let n_samples = predictions[0].len();
    let n_predictions = predictions.len() as Float;
    let mut mean_pred = Array1::zeros(n_samples);

    for pred in predictions {
        mean_pred += pred;
    }

    mean_pred / n_predictions
}

fn assess_perturbation_robustness<F>(
    model: &F,
    X: &ArrayView2<Float>,
    config: &ModelComparisonConfig,
) -> SklResult<Float>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    use scirs2_core::random::SeedableRng;
    let mut rng = match config.random_state {
        Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
        None => scirs2_core::random::rngs::StdRng::seed_from_u64(42),
    };

    let original_predictions = Array1::from_vec(model(&X.view()));
    let mut perturbation_diffs = Vec::new();

    // Test robustness with small perturbations
    for _ in 0..50 {
        let mut perturbed_X = X.to_owned();

        // Add small random noise
        for i in 0..perturbed_X.nrows() {
            for j in 0..perturbed_X.ncols() {
                let noise =
                    rng.gen_range(-config.perturbation_magnitude..config.perturbation_magnitude);
                perturbed_X[[i, j]] += noise;
            }
        }

        let perturbed_predictions = Array1::from_vec(model(&perturbed_X.view()));
        let diff = (&perturbed_predictions - &original_predictions)
            .mapv(|x| x.abs())
            .mean()
            .unwrap();

        perturbation_diffs.push(diff);
    }

    let avg_perturbation_diff =
        perturbation_diffs.iter().sum::<Float>() / perturbation_diffs.len() as Float;
    let robustness = 1.0 / (1.0 + avg_perturbation_diff); // Higher robustness = lower sensitivity

    Ok(robustness)
}

fn compute_bias_variance_decomposition(
    predictions: &[Array1<Float>],
    y_true: &ArrayView1<Float>,
) -> SklResult<BiasVarianceDecomposition> {
    let n_models = predictions.len() as Float;
    let n_samples = y_true.len();

    // Compute ensemble mean
    let ensemble_mean = compute_mean_predictions(predictions);

    // Compute bias (squared difference between ensemble mean and true values)
    let bias_squared = (&ensemble_mean - y_true).mapv(|x| x.powi(2)).sum() / n_samples as Float;

    // Compute variance (average squared difference between individual predictions and ensemble mean)
    let mut variance_sum = 0.0;
    for pred in predictions {
        let diff = pred - &ensemble_mean;
        variance_sum += diff.mapv(|x| x.powi(2)).sum();
    }
    let variance = variance_sum / (n_models * n_samples as Float);

    // Estimate noise (irreducible error) - simplified
    let noise = bias_squared * 0.1; // Rough approximation

    let total_error = bias_squared + variance + noise;

    Ok(BiasVarianceDecomposition {
        bias: bias_squared.sqrt(),
        variance,
        noise,
        total_error,
    })
}

fn compute_model_contributions(
    individual_predictions: &[Array1<Float>],
    ensemble_prediction: &Array1<Float>,
    weights: Option<&ArrayView1<Float>>,
) -> SklResult<Vec<Float>> {
    let n_models = individual_predictions.len();
    let mut contributions = Vec::new();

    if let Some(w) = weights {
        // Use provided weights as contributions
        contributions.extend(w.iter());
    } else {
        // Compute equal contributions
        let equal_weight = 1.0 / n_models as Float;
        contributions.resize(n_models, equal_weight);
    }

    Ok(contributions)
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;

    #[test]
    #[allow(non_snake_case)]
    fn test_model_comparison() {
        let model1 = |x: &ArrayView2<Float>| -> Vec<Float> {
            x.rows().into_iter().map(|row| row[0] * 2.0).collect()
        };

        let model2 = |x: &ArrayView2<Float>| -> Vec<Float> {
            x.rows().into_iter().map(|row| row[0] * 2.0 + 0.1).collect()
        };

        let models = vec![model1, model2];
        let model_ids = vec!["Model1".to_string(), "Model2".to_string()];

        let X = array![[1.0], [2.0], [3.0], [4.0]];
        let y_true = array![2.0, 4.0, 6.0, 8.0];

        let config = ModelComparisonConfig::default();

        let result =
            compare_models(&models, &model_ids, &X.view(), &y_true.view(), &config).unwrap();

        assert_eq!(result.individual_metrics.len(), 2);
        assert_eq!(result.agreement_metrics.shape(), &[2, 2]);
        assert!(result.ensemble_metrics.is_some());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_ensemble_analysis() {
        let model1 = |x: &ArrayView2<Float>| -> Vec<Float> {
            x.rows().into_iter().map(|row| row[0] * 2.0).collect()
        };

        let model2 = |x: &ArrayView2<Float>| -> Vec<Float> {
            x.rows().into_iter().map(|row| row[0] * 2.0 + 0.1).collect()
        };

        let models = vec![model1, model2];

        let X = array![[1.0], [2.0], [3.0]];
        let y_true = array![2.0, 4.0, 6.0];

        let result = analyze_ensemble(
            &models,
            None,
            &X.view(),
            &y_true.view(),
            EnsembleMethod::Average,
        )
        .unwrap();

        assert_eq!(result.ensemble_prediction.len(), 3);
        assert_eq!(result.individual_contributions.len(), 2);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_prediction_stability() {
        let model = |x: &ArrayView2<Float>| -> Vec<Float> {
            x.rows().into_iter().map(|row| row[0] * 2.0).collect()
        };

        let X = array![[1.0], [2.0], [3.0], [4.0]];

        let mut config = ModelComparisonConfig::default();
        config.n_bootstrap_samples = 10; // Small for test

        let result = assess_prediction_stability(&model, &X.view(), &config).unwrap();

        assert!(result.prediction_variance >= 0.0);
        assert!(result.stability_index >= 0.0);
        assert!(result.perturbation_robustness >= 0.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_model_agreement() {
        let model1 = |x: &ArrayView2<Float>| -> Vec<Float> {
            x.rows().into_iter().map(|row| row[0] * 2.0).collect()
        };

        let model2 = |x: &ArrayView2<Float>| -> Vec<Float> {
            x.rows().into_iter().map(|row| row[0] * 2.0).collect()
        };

        let models = vec![model1, model2];
        let X = array![[1.0], [2.0], [3.0]];

        let agreement = compute_model_agreement(&models, &X.view(), 0.1).unwrap();

        assert_eq!(agreement.shape(), &[2, 2]);
        assert_eq!(agreement[[0, 0]], 1.0); // Self-agreement is 1
        assert!(agreement[[0, 1]] > 0.9); // High agreement for similar models
    }

    #[test]
    fn test_bias_variance_decomposition() {
        let pred1 = array![1.0, 2.0, 3.0];
        let pred2 = array![1.1, 2.1, 3.1];
        let predictions = vec![pred1, pred2];
        let y_true = array![1.0, 2.0, 3.0];

        let result = compute_bias_variance_decomposition(&predictions, &y_true.view()).unwrap();

        assert!(result.bias >= 0.0);
        assert!(result.variance >= 0.0);
        assert!(result.noise >= 0.0);
        assert!(result.total_error >= 0.0);
    }
}
