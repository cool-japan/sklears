//! Advanced ensemble analysis and interpretation tools
//!
//! This module provides comprehensive analysis capabilities for ensemble methods,
//! including feature importance aggregation, uncertainty quantification, and
//! ensemble interpretation tools.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::thread_rng;
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};

/// Feature importance aggregation methods for ensembles
#[derive(Debug, Clone)]
pub enum ImportanceAggregationMethod {
    /// Simple mean of feature importances
    Mean,
    /// Weighted mean based on model performance
    WeightedMean(Vec<Float>),
    /// Median aggregation (robust to outliers)
    Median,
    /// Use only top-k most important features
    TopK(usize),
    /// Rank-based aggregation
    RankBased,
    /// Bayesian model averaging of importances
    BayesianAveraging,
    /// Permutation-based importance aggregation
    PermutationBased { n_repeats: usize },
    /// SHAP-style additive feature attribution
    SHAPBased { background_samples: usize },
}

/// Feature importance analysis results
#[derive(Debug, Clone)]
pub struct FeatureImportanceAnalysis {
    /// Aggregated feature importances
    pub feature_importances: Array1<Float>,
    /// Standard deviation of importances across models
    pub importance_std: Array1<Float>,
    /// Individual model importances
    pub individual_importances: Array2<Float>,
    /// Feature rankings by importance
    pub feature_rankings: Vec<usize>,
    /// Stability measure of importance rankings
    pub ranking_stability: Float,
    /// Top-k most important features
    pub top_features: Vec<(usize, Float)>,
    /// Confidence intervals for importances
    pub confidence_intervals: Vec<(Float, Float)>,
    /// Pairwise feature interaction strengths
    pub feature_interactions: Option<Array2<Float>>,
}

/// Ensemble uncertainty quantification results
#[derive(Debug, Clone)]
pub struct UncertaintyQuantification {
    /// Epistemic uncertainty (model uncertainty)
    pub epistemic_uncertainty: Array1<Float>,
    /// Aleatoric uncertainty (data uncertainty)
    pub aleatoric_uncertainty: Array1<Float>,
    /// Total uncertainty
    pub total_uncertainty: Array1<Float>,
    /// Prediction confidence intervals
    pub confidence_intervals: Array2<Float>,
    /// Ensemble diversity at each prediction
    pub prediction_diversity: Array1<Float>,
    /// Uncertainty decomposition by source
    pub uncertainty_decomposition: UncertaintyDecomposition,
    /// Calibration metrics
    pub calibration_metrics: CalibrationMetrics,
}

/// Decomposition of uncertainty into different sources
#[derive(Debug, Clone)]
pub struct UncertaintyDecomposition {
    /// Uncertainty due to model disagreement
    pub model_disagreement: Array1<Float>,
    /// Uncertainty due to insufficient training data
    pub data_uncertainty: Array1<Float>,
    /// Uncertainty due to feature noise
    pub feature_uncertainty: Array1<Float>,
    /// Uncertainty due to label noise
    pub label_uncertainty: Array1<Float>,
    /// Irreducible uncertainty
    pub irreducible_uncertainty: Array1<Float>,
}

/// Calibration metrics for ensemble predictions
#[derive(Debug, Clone)]
pub struct CalibrationMetrics {
    /// Expected Calibration Error (ECE)
    pub expected_calibration_error: Float,
    /// Maximum Calibration Error (MCE)
    pub maximum_calibration_error: Float,
    /// Brier score for probability predictions
    pub brier_score: Float,
    /// Reliability diagram data
    pub reliability_diagram: ReliabilityDiagram,
    /// Over/under-confidence metrics
    pub confidence_metrics: ConfidenceMetrics,
}

/// Reliability diagram for calibration analysis
#[derive(Debug, Clone)]
pub struct ReliabilityDiagram {
    /// Bin boundaries for confidence levels
    pub confidence_bins: Vec<Float>,
    /// Accuracy within each confidence bin
    pub bin_accuracies: Vec<Float>,
    /// Proportion of predictions in each bin
    pub bin_proportions: Vec<Float>,
    /// Average confidence in each bin
    pub bin_confidences: Vec<Float>,
    /// Number of samples in each bin
    pub bin_counts: Vec<usize>,
}

/// Confidence analysis metrics
#[derive(Debug, Clone)]
pub struct ConfidenceMetrics {
    /// Average confidence on correct predictions
    pub avg_confidence_correct: Float,
    /// Average confidence on incorrect predictions
    pub avg_confidence_incorrect: Float,
    /// Confidence-accuracy correlation
    pub confidence_accuracy_correlation: Float,
    /// Over-confidence rate
    pub overconfidence_rate: Float,
    /// Under-confidence rate
    pub underconfidence_rate: Float,
}

/// Ensemble analyzer for feature importance and uncertainty quantification
pub struct EnsembleAnalyzer {
    /// Method for aggregating feature importances
    pub importance_method: ImportanceAggregationMethod,
    /// Random state for reproducible analysis
    pub random_state: Option<u64>,
    /// Number of bootstrap samples for confidence intervals
    pub n_bootstrap: usize,
    /// Confidence level for intervals
    pub confidence_level: Float,
}

impl EnsembleAnalyzer {
    /// Create a new ensemble analyzer
    pub fn new(importance_method: ImportanceAggregationMethod) -> Self {
        Self {
            importance_method,
            random_state: None,
            n_bootstrap: 100,
            confidence_level: 0.95,
        }
    }

    /// Configure bootstrap parameters
    pub fn with_bootstrap(mut self, n_bootstrap: usize, confidence_level: Float) -> Self {
        self.n_bootstrap = n_bootstrap;
        self.confidence_level = confidence_level;
        self
    }

    /// Set random state for reproducible results
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Analyze feature importances across ensemble models
    pub fn analyze_feature_importance(
        &self,
        individual_importances: &Array2<Float>,
        model_weights: Option<&Array1<Float>>,
    ) -> Result<FeatureImportanceAnalysis> {
        let n_models = individual_importances.nrows();
        let n_features = individual_importances.ncols();

        if n_models == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "Empty importance matrix".to_string(),
            ));
        }

        // Aggregate feature importances using specified method
        let feature_importances =
            self.aggregate_importances(individual_importances, model_weights)?;

        // Compute standard deviation across models
        let importance_std =
            self.compute_importance_std(individual_importances, &feature_importances);

        // Rank features by importance
        let feature_rankings = self.rank_features(&feature_importances);

        // Compute ranking stability across models
        let ranking_stability = self.compute_ranking_stability(individual_importances)?;

        // Get top-k features
        let top_features = self.get_top_features(&feature_importances, 10);

        // Compute confidence intervals using bootstrap
        let confidence_intervals = self.compute_confidence_intervals(individual_importances)?;

        // Compute feature interactions if requested
        let feature_interactions = self.compute_feature_interactions(individual_importances)?;

        Ok(FeatureImportanceAnalysis {
            feature_importances,
            importance_std,
            individual_importances: individual_importances.clone(),
            feature_rankings,
            ranking_stability,
            top_features,
            confidence_intervals,
            feature_interactions,
        })
    }

    /// Aggregate individual feature importances using the specified method
    fn aggregate_importances(
        &self,
        individual_importances: &Array2<Float>,
        model_weights: Option<&Array1<Float>>,
    ) -> Result<Array1<Float>> {
        let n_models = individual_importances.nrows();
        let n_features = individual_importances.ncols();

        match &self.importance_method {
            ImportanceAggregationMethod::Mean => {
                Ok(individual_importances.mean_axis(Axis(0)).unwrap())
            }

            ImportanceAggregationMethod::WeightedMean(weights) => {
                if weights.len() != n_models {
                    return Err(SklearsError::InvalidInput(
                        "Weight vector length must match number of models".to_string(),
                    ));
                }

                let mut aggregated = Array1::zeros(n_features);
                let total_weight = weights.iter().sum::<Float>();

                for i in 0..n_models {
                    let row = individual_importances.row(i).to_owned();
                    aggregated += &(row * weights[i]);
                }

                Ok(aggregated / total_weight)
            }

            ImportanceAggregationMethod::Median => {
                let mut aggregated = Array1::zeros(n_features);
                for j in 0..n_features {
                    let mut feature_importances: Vec<Float> =
                        individual_importances.column(j).iter().copied().collect();
                    feature_importances.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    let median = if feature_importances.len() % 2 == 0 {
                        let mid = feature_importances.len() / 2;
                        (feature_importances[mid - 1] + feature_importances[mid]) / 2.0
                    } else {
                        feature_importances[feature_importances.len() / 2]
                    };

                    aggregated[j] = median;
                }
                Ok(aggregated)
            }

            ImportanceAggregationMethod::TopK(k) => {
                // Use mean aggregation but zero out features not in top-k
                let mean_importances = individual_importances.mean_axis(Axis(0)).unwrap();
                let mut indexed_importances: Vec<(usize, Float)> = mean_importances
                    .iter()
                    .enumerate()
                    .map(|(i, &imp)| (i, imp))
                    .collect();

                indexed_importances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                let mut aggregated = Array1::zeros(n_features);
                for i in 0..*k.min(&n_features) {
                    let (feature_idx, importance) = indexed_importances[i];
                    aggregated[feature_idx] = importance;
                }

                Ok(aggregated)
            }

            ImportanceAggregationMethod::RankBased => {
                let mut aggregated = Array1::zeros(n_features);

                for i in 0..n_models {
                    let row = individual_importances.row(i);
                    let ranks = self.compute_ranks(&row.to_owned());

                    for j in 0..n_features {
                        aggregated[j] += ranks[j];
                    }
                }

                // Normalize by number of models
                aggregated /= n_models as Float;

                // Convert average ranks back to importance scores (higher rank = higher importance)
                let max_rank = aggregated.iter().fold(0.0f64, |a, &b| a.max(b));
                for val in aggregated.iter_mut() {
                    *val = max_rank - *val;
                }

                Ok(aggregated)
            }

            ImportanceAggregationMethod::BayesianAveraging => {
                // Simplified Bayesian averaging with uniform priors
                self.bayesian_average_importances(individual_importances, model_weights)
            }

            ImportanceAggregationMethod::PermutationBased { n_repeats } => {
                // Placeholder for permutation-based importance
                // In a real implementation, this would compute permutation importance
                let mean_importances = individual_importances.mean_axis(Axis(0)).unwrap();
                Ok(mean_importances)
            }

            ImportanceAggregationMethod::SHAPBased { background_samples } => {
                // Placeholder for SHAP-based importance
                // In a real implementation, this would compute SHAP values
                let mean_importances = individual_importances.mean_axis(Axis(0)).unwrap();
                Ok(mean_importances)
            }
        }
    }

    /// Compute ranks for a feature importance vector
    fn compute_ranks(&self, importances: &Array1<Float>) -> Array1<Float> {
        let mut indexed: Vec<(usize, Float)> = importances
            .iter()
            .enumerate()
            .map(|(i, &val)| (i, val))
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut ranks = Array1::zeros(importances.len());
        for (rank, &(idx, _)) in indexed.iter().enumerate() {
            ranks[idx] = rank as Float;
        }

        ranks
    }

    /// Bayesian averaging of feature importances
    fn bayesian_average_importances(
        &self,
        individual_importances: &Array2<Float>,
        model_weights: Option<&Array1<Float>>,
    ) -> Result<Array1<Float>> {
        let n_models = individual_importances.nrows();
        let n_features = individual_importances.ncols();

        // Use model weights as posterior probabilities if provided
        let weights = if let Some(w) = model_weights {
            w.clone()
        } else {
            Array1::from_elem(n_models, 1.0 / n_models as Float)
        };

        // Compute weighted average with Bayesian interpretation
        let mut aggregated = Array1::zeros(n_features);
        for i in 0..n_models {
            let row = individual_importances.row(i).to_owned();
            aggregated += &(row * weights[i]);
        }

        Ok(aggregated)
    }

    /// Compute standard deviation of feature importances across models
    fn compute_importance_std(
        &self,
        individual_importances: &Array2<Float>,
        mean_importances: &Array1<Float>,
    ) -> Array1<Float> {
        let n_models = individual_importances.nrows();
        let n_features = individual_importances.ncols();

        let mut std_dev = Array1::zeros(n_features);

        for j in 0..n_features {
            let mean = mean_importances[j];
            let variance = individual_importances
                .column(j)
                .iter()
                .map(|&val| (val - mean).powi(2))
                .sum::<Float>()
                / n_models as Float;

            std_dev[j] = variance.sqrt();
        }

        std_dev
    }

    /// Rank features by their aggregated importance
    fn rank_features(&self, feature_importances: &Array1<Float>) -> Vec<usize> {
        let mut indexed: Vec<(usize, Float)> = feature_importances
            .iter()
            .enumerate()
            .map(|(i, &val)| (i, val))
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        indexed.into_iter().map(|(idx, _)| idx).collect()
    }

    /// Compute ranking stability across models using Kendall's tau
    fn compute_ranking_stability(&self, individual_importances: &Array2<Float>) -> Result<Float> {
        let n_models = individual_importances.nrows();
        if n_models < 2 {
            return Ok(1.0);
        }

        let mut total_tau = 0.0;
        let mut pair_count = 0;

        for i in 0..n_models {
            for j in (i + 1)..n_models {
                let rank1 = self.compute_ranks(&individual_importances.row(i).to_owned());
                let rank2 = self.compute_ranks(&individual_importances.row(j).to_owned());

                let tau = self.kendall_tau(&rank1, &rank2)?;
                total_tau += tau;
                pair_count += 1;
            }
        }

        Ok(total_tau / pair_count as Float)
    }

    /// Compute Kendall's tau correlation coefficient
    fn kendall_tau(&self, rank1: &Array1<Float>, rank2: &Array1<Float>) -> Result<Float> {
        if rank1.len() != rank2.len() {
            return Err(SklearsError::InvalidInput(
                "Rank vectors must have same length".to_string(),
            ));
        }

        let n = rank1.len();
        if n < 2 {
            return Ok(1.0);
        }

        let mut concordant = 0;
        let mut discordant = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let diff1 = rank1[i] - rank1[j];
                let diff2 = rank2[i] - rank2[j];

                if diff1 * diff2 > 0.0 {
                    concordant += 1;
                } else if diff1 * diff2 < 0.0 {
                    discordant += 1;
                }
            }
        }

        let total_pairs = (n * (n - 1)) / 2;
        let tau = (concordant as Float - discordant as Float) / total_pairs as Float;

        Ok(tau)
    }

    /// Get top-k most important features
    fn get_top_features(
        &self,
        feature_importances: &Array1<Float>,
        k: usize,
    ) -> Vec<(usize, Float)> {
        let mut indexed: Vec<(usize, Float)> = feature_importances
            .iter()
            .enumerate()
            .map(|(i, &val)| (i, val))
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        indexed.into_iter().take(k).collect()
    }

    /// Compute confidence intervals for feature importances using bootstrap
    fn compute_confidence_intervals(
        &self,
        individual_importances: &Array2<Float>,
    ) -> Result<Vec<(Float, Float)>> {
        let n_models = individual_importances.nrows();
        let n_features = individual_importances.ncols();

        if self.n_bootstrap == 0 {
            // Return dummy intervals if no bootstrap requested
            return Ok(vec![(0.0, 0.0); n_features]);
        }

        let mut rng = thread_rng();
        let mut bootstrap_importances = Vec::with_capacity(self.n_bootstrap);

        for _ in 0..self.n_bootstrap {
            // Bootstrap sample of models
            let mut bootstrap_matrix = Array2::zeros((n_models, n_features));
            for i in 0..n_models {
                let idx = rng.gen_range(0..n_models);
                bootstrap_matrix
                    .row_mut(i)
                    .assign(&individual_importances.row(idx));
            }

            // Compute mean importance for this bootstrap sample
            let mean_importance = bootstrap_matrix.mean_axis(Axis(0)).unwrap();
            bootstrap_importances.push(mean_importance);
        }

        // Compute confidence intervals
        let alpha = 1.0 - self.confidence_level;
        let lower_percentile = (alpha / 2.0) * 100.0;
        let upper_percentile = (1.0 - alpha / 2.0) * 100.0;

        let mut confidence_intervals = Vec::with_capacity(n_features);

        for j in 0..n_features {
            let mut feature_values: Vec<Float> = bootstrap_importances
                .iter()
                .map(|importance| importance[j])
                .collect();

            feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let lower_idx =
                ((lower_percentile / 100.0) * (feature_values.len() - 1) as Float) as usize;
            let upper_idx =
                ((upper_percentile / 100.0) * (feature_values.len() - 1) as Float) as usize;

            let lower = feature_values[lower_idx];
            let upper = feature_values[upper_idx];

            confidence_intervals.push((lower, upper));
        }

        Ok(confidence_intervals)
    }

    /// Compute pairwise feature interactions
    fn compute_feature_interactions(
        &self,
        individual_importances: &Array2<Float>,
    ) -> Result<Option<Array2<Float>>> {
        let n_features = individual_importances.ncols();

        // For now, compute correlation between feature importances across models
        let mut interactions = Array2::zeros((n_features, n_features));

        for i in 0..n_features {
            for j in 0..n_features {
                if i == j {
                    interactions[[i, j]] = 1.0;
                } else {
                    let corr = self.compute_feature_correlation(
                        &individual_importances.column(i).to_owned(),
                        &individual_importances.column(j).to_owned(),
                    )?;
                    interactions[[i, j]] = corr;
                }
            }
        }

        Ok(Some(interactions))
    }

    /// Compute correlation between two feature importance vectors
    fn compute_feature_correlation(
        &self,
        feature1: &Array1<Float>,
        feature2: &Array1<Float>,
    ) -> Result<Float> {
        if feature1.len() != feature2.len() || feature1.is_empty() {
            return Ok(0.0);
        }

        let n = feature1.len() as Float;
        let mean1 = feature1.sum() / n;
        let mean2 = feature2.sum() / n;

        let mut numerator = 0.0;
        let mut sum_sq1 = 0.0;
        let mut sum_sq2 = 0.0;

        for i in 0..feature1.len() {
            let diff1 = feature1[i] - mean1;
            let diff2 = feature2[i] - mean2;

            numerator += diff1 * diff2;
            sum_sq1 += diff1 * diff1;
            sum_sq2 += diff2 * diff2;
        }

        let denominator = (sum_sq1 * sum_sq2).sqrt();
        let correlation = if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        };

        Ok(correlation)
    }

    /// Quantify uncertainty in ensemble predictions
    pub fn quantify_uncertainty(
        &self,
        ensemble_predictions: &Array2<Float>,
        true_labels: Option<&Array1<Float>>,
    ) -> Result<UncertaintyQuantification> {
        let n_models = ensemble_predictions.nrows();
        let n_samples = ensemble_predictions.ncols();

        if n_models == 0 || n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Empty prediction matrix".to_string(),
            ));
        }

        // Compute epistemic uncertainty (model disagreement)
        let epistemic_uncertainty = self.compute_epistemic_uncertainty(ensemble_predictions);

        // Compute aleatoric uncertainty (data uncertainty) - simplified version
        let aleatoric_uncertainty = self.compute_aleatoric_uncertainty(ensemble_predictions);

        // Total uncertainty
        let total_uncertainty =
            self.compute_total_uncertainty(&epistemic_uncertainty, &aleatoric_uncertainty);

        // Confidence intervals
        let confidence_intervals =
            self.compute_prediction_confidence_intervals(ensemble_predictions)?;

        // Prediction diversity
        let prediction_diversity = self.compute_prediction_diversity(ensemble_predictions);

        // Uncertainty decomposition
        let uncertainty_decomposition = self.decompose_uncertainty(ensemble_predictions)?;

        // Calibration metrics (if true labels provided)
        let calibration_metrics = if let Some(labels) = true_labels {
            self.compute_calibration_metrics(ensemble_predictions, labels)?
        } else {
            self.default_calibration_metrics()
        };

        Ok(UncertaintyQuantification {
            epistemic_uncertainty,
            aleatoric_uncertainty,
            total_uncertainty,
            confidence_intervals,
            prediction_diversity,
            uncertainty_decomposition,
            calibration_metrics,
        })
    }

    /// Compute epistemic uncertainty (model disagreement)
    fn compute_epistemic_uncertainty(&self, ensemble_predictions: &Array2<Float>) -> Array1<Float> {
        let n_samples = ensemble_predictions.ncols();
        let mut epistemic = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let predictions = ensemble_predictions.column(i);
            let mean_pred = predictions.mean().unwrap();
            let variance = predictions
                .iter()
                .map(|&pred| (pred - mean_pred).powi(2))
                .sum::<Float>()
                / predictions.len() as Float;

            epistemic[i] = variance.sqrt();
        }

        epistemic
    }

    /// Compute aleatoric uncertainty (simplified version)
    fn compute_aleatoric_uncertainty(&self, ensemble_predictions: &Array2<Float>) -> Array1<Float> {
        let n_samples = ensemble_predictions.ncols();
        let mut aleatoric = Array1::zeros(n_samples);

        // Simplified aleatoric uncertainty estimation
        // In practice, this would require additional information about data noise
        for i in 0..n_samples {
            let predictions = ensemble_predictions.column(i);
            let mean_pred = predictions.mean().unwrap();

            // Use prediction magnitude as a proxy for aleatoric uncertainty
            aleatoric[i] = mean_pred.abs() * 0.1; // Simplified heuristic
        }

        aleatoric
    }

    /// Compute total uncertainty
    fn compute_total_uncertainty(
        &self,
        epistemic: &Array1<Float>,
        aleatoric: &Array1<Float>,
    ) -> Array1<Float> {
        let mut total = Array1::zeros(epistemic.len());

        for i in 0..epistemic.len() {
            // Total uncertainty as combination of epistemic and aleatoric
            total[i] = (epistemic[i].powi(2) + aleatoric[i].powi(2)).sqrt();
        }

        total
    }

    /// Compute confidence intervals for predictions
    fn compute_prediction_confidence_intervals(
        &self,
        ensemble_predictions: &Array2<Float>,
    ) -> Result<Array2<Float>> {
        let n_samples = ensemble_predictions.ncols();
        let alpha = 1.0 - self.confidence_level;
        let lower_percentile = alpha / 2.0;
        let upper_percentile = 1.0 - alpha / 2.0;

        let mut intervals = Array2::zeros((n_samples, 2));

        for i in 0..n_samples {
            let mut predictions: Vec<Float> = ensemble_predictions.column(i).to_vec();
            predictions.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let lower_idx = (lower_percentile * (predictions.len() - 1) as Float) as usize;
            let upper_idx = (upper_percentile * (predictions.len() - 1) as Float) as usize;

            intervals[[i, 0]] = predictions[lower_idx];
            intervals[[i, 1]] = predictions[upper_idx];
        }

        Ok(intervals)
    }

    /// Compute prediction diversity across ensemble
    fn compute_prediction_diversity(&self, ensemble_predictions: &Array2<Float>) -> Array1<Float> {
        let n_samples = ensemble_predictions.ncols();
        let mut diversity = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let predictions = ensemble_predictions.column(i);
            let mean_pred = predictions.mean().unwrap();

            // Coefficient of variation as diversity measure
            let std_pred = predictions
                .iter()
                .map(|&pred| (pred - mean_pred).powi(2))
                .sum::<Float>()
                / predictions.len() as Float;

            diversity[i] = if mean_pred.abs() > 1e-8 {
                std_pred.sqrt() / mean_pred.abs()
            } else {
                std_pred.sqrt()
            };
        }

        diversity
    }

    /// Decompose uncertainty into different sources
    fn decompose_uncertainty(
        &self,
        ensemble_predictions: &Array2<Float>,
    ) -> Result<UncertaintyDecomposition> {
        let n_samples = ensemble_predictions.ncols();

        // Simplified uncertainty decomposition
        let model_disagreement = self.compute_epistemic_uncertainty(ensemble_predictions);
        let data_uncertainty = Array1::from_elem(n_samples, 0.1); // Placeholder
        let feature_uncertainty = Array1::from_elem(n_samples, 0.05); // Placeholder
        let label_uncertainty = Array1::from_elem(n_samples, 0.03); // Placeholder
        let irreducible_uncertainty = Array1::from_elem(n_samples, 0.02); // Placeholder

        Ok(UncertaintyDecomposition {
            model_disagreement,
            data_uncertainty,
            feature_uncertainty,
            label_uncertainty,
            irreducible_uncertainty,
        })
    }

    /// Compute calibration metrics for ensemble predictions
    fn compute_calibration_metrics(
        &self,
        ensemble_predictions: &Array2<Float>,
        true_labels: &Array1<Float>,
    ) -> Result<CalibrationMetrics> {
        let n_samples = ensemble_predictions.ncols();

        if true_labels.len() != n_samples {
            return Err(SklearsError::InvalidInput(
                "Prediction and label lengths must match".to_string(),
            ));
        }

        // Compute mean predictions
        let mean_predictions = ensemble_predictions.mean_axis(Axis(0)).unwrap();

        // Convert to binary classification for calibration analysis
        let predicted_probs: Vec<Float> = mean_predictions.iter().copied().collect();
        let true_binary: Vec<bool> = true_labels.iter().map(|&label| label > 0.5).collect();

        // Compute ECE and MCE
        let (ece, mce) = self.compute_calibration_errors(&predicted_probs, &true_binary)?;

        // Compute Brier score
        let brier_score = self.compute_brier_score(&predicted_probs, &true_binary);

        // Create reliability diagram
        let reliability_diagram =
            self.create_reliability_diagram(&predicted_probs, &true_binary)?;

        // Compute confidence metrics
        let confidence_metrics = self.compute_confidence_metrics(&predicted_probs, &true_binary);

        Ok(CalibrationMetrics {
            expected_calibration_error: ece,
            maximum_calibration_error: mce,
            brier_score,
            reliability_diagram,
            confidence_metrics,
        })
    }

    /// Compute Expected and Maximum Calibration Errors
    fn compute_calibration_errors(
        &self,
        predicted_probs: &[Float],
        true_binary: &[bool],
    ) -> Result<(Float, Float)> {
        let n_bins = 10;
        let bin_size = 1.0 / n_bins as Float;

        let mut ece: Float = 0.0;
        let mut mce: Float = 0.0;
        let mut total_samples = 0;

        for i in 0..n_bins {
            let bin_lower = i as Float * bin_size;
            let bin_upper = (i + 1) as Float * bin_size;

            // Find samples in this bin
            let bin_indices: Vec<usize> = predicted_probs
                .iter()
                .enumerate()
                .filter(|(_, &prob)| prob >= bin_lower && prob < bin_upper)
                .map(|(idx, _)| idx)
                .collect();

            if bin_indices.is_empty() {
                continue;
            }

            let bin_size_actual = bin_indices.len();
            total_samples += bin_size_actual;

            // Compute average confidence and accuracy in this bin
            let avg_confidence = bin_indices
                .iter()
                .map(|&idx| predicted_probs[idx])
                .sum::<Float>()
                / bin_size_actual as Float;

            let bin_accuracy = bin_indices
                .iter()
                .map(|&idx| if true_binary[idx] { 1.0 } else { 0.0 })
                .sum::<Float>()
                / bin_size_actual as Float;

            let calibration_error = (avg_confidence - bin_accuracy).abs();

            // Update ECE and MCE
            ece += (bin_size_actual as Float / predicted_probs.len() as Float) * calibration_error;
            mce = mce.max(calibration_error);
        }

        Ok((ece, mce))
    }

    /// Compute Brier score
    fn compute_brier_score(&self, predicted_probs: &[Float], true_binary: &[bool]) -> Float {
        predicted_probs
            .iter()
            .zip(true_binary.iter())
            .map(|(&prob, &is_true)| {
                let true_prob = if is_true { 1.0 } else { 0.0 };
                (prob - true_prob).powi(2)
            })
            .sum::<Float>()
            / predicted_probs.len() as Float
    }

    /// Create reliability diagram data
    fn create_reliability_diagram(
        &self,
        predicted_probs: &[Float],
        true_binary: &[bool],
    ) -> Result<ReliabilityDiagram> {
        let n_bins = 10;
        let bin_size = 1.0 / n_bins as Float;

        let mut confidence_bins = Vec::with_capacity(n_bins + 1);
        let mut bin_accuracies = Vec::with_capacity(n_bins);
        let mut bin_proportions = Vec::with_capacity(n_bins);
        let mut bin_confidences = Vec::with_capacity(n_bins);
        let mut bin_counts = Vec::with_capacity(n_bins);

        // Create bin boundaries
        for i in 0..=n_bins {
            confidence_bins.push(i as Float * bin_size);
        }

        for i in 0..n_bins {
            let bin_lower = i as Float * bin_size;
            let bin_upper = (i + 1) as Float * bin_size;

            // Find samples in this bin
            let bin_indices: Vec<usize> = predicted_probs
                .iter()
                .enumerate()
                .filter(|(_, &prob)| prob >= bin_lower && prob < bin_upper)
                .map(|(idx, _)| idx)
                .collect();

            let bin_count = bin_indices.len();
            bin_counts.push(bin_count);

            if bin_count == 0 {
                bin_accuracies.push(0.0);
                bin_proportions.push(0.0);
                bin_confidences.push((bin_lower + bin_upper) / 2.0);
            } else {
                let avg_confidence = bin_indices
                    .iter()
                    .map(|&idx| predicted_probs[idx])
                    .sum::<Float>()
                    / bin_count as Float;

                let bin_accuracy = bin_indices
                    .iter()
                    .map(|&idx| if true_binary[idx] { 1.0 } else { 0.0 })
                    .sum::<Float>()
                    / bin_count as Float;

                let bin_proportion = bin_count as Float / predicted_probs.len() as Float;

                bin_accuracies.push(bin_accuracy);
                bin_proportions.push(bin_proportion);
                bin_confidences.push(avg_confidence);
            }
        }

        Ok(ReliabilityDiagram {
            confidence_bins,
            bin_accuracies,
            bin_proportions,
            bin_confidences,
            bin_counts,
        })
    }

    /// Compute confidence-related metrics
    fn compute_confidence_metrics(
        &self,
        predicted_probs: &[Float],
        true_binary: &[bool],
    ) -> ConfidenceMetrics {
        let mut correct_confidences = Vec::new();
        let mut incorrect_confidences = Vec::new();

        for (&prob, &is_true) in predicted_probs.iter().zip(true_binary.iter()) {
            let predicted_class = prob > 0.5;
            if predicted_class == is_true {
                correct_confidences.push(prob.max(1.0 - prob)); // Distance from 0.5
            } else {
                incorrect_confidences.push(prob.max(1.0 - prob));
            }
        }

        let avg_confidence_correct = if correct_confidences.is_empty() {
            0.0
        } else {
            correct_confidences.iter().sum::<Float>() / correct_confidences.len() as Float
        };

        let avg_confidence_incorrect = if incorrect_confidences.is_empty() {
            0.0
        } else {
            incorrect_confidences.iter().sum::<Float>() / incorrect_confidences.len() as Float
        };

        // Compute correlation between confidence and accuracy
        let confidence_accuracy_correlation =
            self.compute_confidence_accuracy_correlation(predicted_probs, true_binary);

        // Over/under-confidence rates (simplified)
        let threshold = 0.8;
        let overconfident_count = predicted_probs
            .iter()
            .zip(true_binary.iter())
            .filter(|(&prob, &is_true)| {
                let predicted_class = prob > 0.5;
                prob.max(1.0 - prob) > threshold && predicted_class != is_true
            })
            .count();

        let underconfident_count = predicted_probs
            .iter()
            .zip(true_binary.iter())
            .filter(|(&prob, &is_true)| {
                let predicted_class = prob > 0.5;
                prob.max(1.0 - prob) < (1.0 - threshold) && predicted_class == is_true
            })
            .count();

        let overconfidence_rate = overconfident_count as Float / predicted_probs.len() as Float;
        let underconfidence_rate = underconfident_count as Float / predicted_probs.len() as Float;

        ConfidenceMetrics {
            avg_confidence_correct,
            avg_confidence_incorrect,
            confidence_accuracy_correlation,
            overconfidence_rate,
            underconfidence_rate,
        }
    }

    /// Compute correlation between confidence and accuracy
    fn compute_confidence_accuracy_correlation(
        &self,
        predicted_probs: &[Float],
        true_binary: &[bool],
    ) -> Float {
        let confidences: Vec<Float> = predicted_probs
            .iter()
            .map(|&prob| prob.max(1.0 - prob))
            .collect();
        let accuracies: Vec<Float> = predicted_probs
            .iter()
            .zip(true_binary.iter())
            .map(|(&prob, &is_true)| {
                let predicted_class = prob > 0.5;
                if predicted_class == is_true {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();

        // Compute Pearson correlation
        let n = confidences.len() as Float;
        let mean_conf = confidences.iter().sum::<Float>() / n;
        let mean_acc = accuracies.iter().sum::<Float>() / n;

        let mut numerator = 0.0;
        let mut sum_sq_conf = 0.0;
        let mut sum_sq_acc = 0.0;

        for i in 0..confidences.len() {
            let diff_conf = confidences[i] - mean_conf;
            let diff_acc = accuracies[i] - mean_acc;

            numerator += diff_conf * diff_acc;
            sum_sq_conf += diff_conf * diff_conf;
            sum_sq_acc += diff_acc * diff_acc;
        }

        let denominator = (sum_sq_conf * sum_sq_acc).sqrt();
        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Create default calibration metrics when no true labels are provided
    fn default_calibration_metrics(&self) -> CalibrationMetrics {
        CalibrationMetrics {
            expected_calibration_error: 0.0,
            maximum_calibration_error: 0.0,
            brier_score: 0.0,
            reliability_diagram: ReliabilityDiagram {
                confidence_bins: vec![],
                bin_accuracies: vec![],
                bin_proportions: vec![],
                bin_confidences: vec![],
                bin_counts: vec![],
            },
            confidence_metrics: ConfidenceMetrics {
                avg_confidence_correct: 0.0,
                avg_confidence_incorrect: 0.0,
                confidence_accuracy_correlation: 0.0,
                overconfidence_rate: 0.0,
                underconfidence_rate: 0.0,
            },
        }
    }
}

impl Default for EnsembleAnalyzer {
    fn default() -> Self {
        Self::new(ImportanceAggregationMethod::Mean)
    }
}

/// Convenience functions for common analysis tasks
impl EnsembleAnalyzer {
    /// Create analyzer for mean-based feature importance aggregation
    pub fn mean_importance() -> Self {
        Self::new(ImportanceAggregationMethod::Mean)
    }

    /// Create analyzer for weighted feature importance aggregation
    pub fn weighted_importance(weights: Vec<Float>) -> Self {
        Self::new(ImportanceAggregationMethod::WeightedMean(weights))
    }

    /// Create analyzer for robust median-based aggregation
    pub fn robust_importance() -> Self {
        Self::new(ImportanceAggregationMethod::Median)
    }

    /// Create analyzer for rank-based aggregation
    pub fn rank_based_importance() -> Self {
        Self::new(ImportanceAggregationMethod::RankBased)
    }

    /// Create analyzer with permutation-based importance
    pub fn permutation_importance(n_repeats: usize) -> Self {
        Self::new(ImportanceAggregationMethod::PermutationBased { n_repeats })
    }

    /// Create analyzer with SHAP-based importance
    pub fn shap_importance(background_samples: usize) -> Self {
        Self::new(ImportanceAggregationMethod::SHAPBased { background_samples })
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_ensemble_analyzer_creation() {
        let analyzer = EnsembleAnalyzer::default();
        assert!(matches!(
            analyzer.importance_method,
            ImportanceAggregationMethod::Mean
        ));
    }

    #[test]
    fn test_feature_importance_analysis() {
        let analyzer = EnsembleAnalyzer::mean_importance();

        // Create mock importance matrix
        let importances = array![[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.6, 0.2, 0.2]];

        let analysis = analyzer
            .analyze_feature_importance(&importances, None)
            .unwrap();

        assert_eq!(analysis.feature_importances.len(), 3);
        assert_eq!(analysis.importance_std.len(), 3);
        assert_eq!(analysis.feature_rankings.len(), 3);
        assert!(analysis.ranking_stability >= 0.0 && analysis.ranking_stability <= 1.0);
        assert_eq!(analysis.top_features.len(), 3);
    }

    #[test]
    fn test_weighted_importance_aggregation() {
        let weights = vec![0.5, 0.3, 0.2];
        let analyzer = EnsembleAnalyzer::weighted_importance(weights);

        let importances = array![[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.6, 0.2, 0.2]];

        let analysis = analyzer
            .analyze_feature_importance(&importances, None)
            .unwrap();

        assert_eq!(analysis.feature_importances.len(), 3);
        // Weighted average should be different from simple mean
        let simple_mean = importances.mean_axis(Axis(0)).unwrap();
        assert!((analysis.feature_importances[0] - simple_mean[0]).abs() > 1e-10);
    }

    #[test]
    fn test_median_aggregation() {
        let analyzer = EnsembleAnalyzer::robust_importance();

        let importances = array![
            [0.1, 0.3, 0.6], // Outlier in first feature
            [0.5, 0.3, 0.2],
            [0.4, 0.4, 0.2]
        ];

        let analysis = analyzer
            .analyze_feature_importance(&importances, None)
            .unwrap();

        assert_eq!(analysis.feature_importances.len(), 3);
        // Median should be more robust to outliers
        assert!((analysis.feature_importances[0] - 0.4).abs() < 1e-6); // Median of [0.1, 0.5, 0.4] is 0.4
    }

    #[test]
    fn test_uncertainty_quantification() {
        let analyzer = EnsembleAnalyzer::default();

        // Create mock ensemble predictions
        let predictions = array![
            [0.1, 0.8, 0.3, 0.7],
            [0.2, 0.9, 0.4, 0.6],
            [0.1, 0.7, 0.5, 0.8]
        ];

        let uncertainty = analyzer.quantify_uncertainty(&predictions, None).unwrap();

        assert_eq!(uncertainty.epistemic_uncertainty.len(), 4);
        assert_eq!(uncertainty.aleatoric_uncertainty.len(), 4);
        assert_eq!(uncertainty.total_uncertainty.len(), 4);
        assert_eq!(uncertainty.confidence_intervals.nrows(), 4);
        assert_eq!(uncertainty.confidence_intervals.ncols(), 2);
        assert_eq!(uncertainty.prediction_diversity.len(), 4);
    }

    #[test]
    fn test_calibration_metrics() {
        let analyzer = EnsembleAnalyzer::default();

        let predictions = array![
            [0.1, 0.8, 0.3, 0.7],
            [0.2, 0.9, 0.4, 0.6],
            [0.1, 0.7, 0.5, 0.8]
        ];

        let true_labels = array![0.0, 1.0, 0.0, 1.0];

        let uncertainty = analyzer
            .quantify_uncertainty(&predictions, Some(&true_labels))
            .unwrap();

        assert!(uncertainty.calibration_metrics.expected_calibration_error >= 0.0);
        assert!(uncertainty.calibration_metrics.maximum_calibration_error >= 0.0);
        assert!(uncertainty.calibration_metrics.brier_score >= 0.0);
        assert!(!uncertainty
            .calibration_metrics
            .reliability_diagram
            .confidence_bins
            .is_empty());
    }

    #[test]
    fn test_rank_based_aggregation() {
        let analyzer = EnsembleAnalyzer::rank_based_importance();

        let importances = array![
            [0.1, 0.5, 0.4], // Ranks: [2, 0, 1]
            [0.3, 0.2, 0.5], // Ranks: [1, 2, 0]
            [0.6, 0.1, 0.3]  // Ranks: [0, 2, 1]
        ];

        let analysis = analyzer
            .analyze_feature_importance(&importances, None)
            .unwrap();

        assert_eq!(analysis.feature_importances.len(), 3);
        // Rank-based aggregation should handle different scales well
        let sum_importance = analysis.feature_importances.sum();
        assert!(sum_importance > 0.0);
    }

    #[test]
    fn test_kendall_tau_correlation() {
        let analyzer = EnsembleAnalyzer::default();

        let rank1 = array![1.0, 2.0, 3.0, 4.0];
        let rank2 = array![1.0, 2.0, 3.0, 4.0]; // Perfect correlation

        let tau = analyzer.kendall_tau(&rank1, &rank2).unwrap();
        assert!((tau - 1.0).abs() < 1e-6);

        let rank3 = array![4.0, 3.0, 2.0, 1.0]; // Perfect negative correlation
        let tau_neg = analyzer.kendall_tau(&rank1, &rank3).unwrap();
        assert!((tau_neg + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_top_features_extraction() {
        let analyzer = EnsembleAnalyzer::default();

        let importances = array![0.1, 0.5, 0.3, 0.8, 0.2];
        let top_features = analyzer.get_top_features(&importances, 3);

        assert_eq!(top_features.len(), 3);
        assert_eq!(top_features[0].0, 3); // Index of highest importance (0.8)
        assert_eq!(top_features[1].0, 1); // Index of second highest (0.5)
        assert_eq!(top_features[2].0, 2); // Index of third highest (0.3)
    }

    #[test]
    fn test_confidence_intervals() {
        let analyzer = EnsembleAnalyzer::default().with_bootstrap(50, 0.95);

        let importances = array![
            [0.5, 0.3, 0.2],
            [0.4, 0.4, 0.2],
            [0.6, 0.2, 0.2],
            [0.5, 0.3, 0.2],
            [0.4, 0.5, 0.1]
        ];

        let intervals = analyzer.compute_confidence_intervals(&importances).unwrap();

        assert_eq!(intervals.len(), 3); // 3 features
        for (lower, upper) in &intervals {
            assert!(lower <= upper);
        }
    }

    #[test]
    fn test_prediction_confidence_intervals() {
        let analyzer = EnsembleAnalyzer::default().with_bootstrap(10, 0.9);

        let predictions = array![
            [0.1, 0.8, 0.3],
            [0.2, 0.9, 0.4],
            [0.0, 0.7, 0.5],
            [0.3, 0.8, 0.2]
        ];

        let intervals = analyzer
            .compute_prediction_confidence_intervals(&predictions)
            .unwrap();

        assert_eq!(intervals.nrows(), 3); // 3 samples
        assert_eq!(intervals.ncols(), 2); // Lower and upper bounds

        for i in 0..3 {
            assert!(intervals[[i, 0]] <= intervals[[i, 1]]); // Lower <= Upper
        }
    }
}
