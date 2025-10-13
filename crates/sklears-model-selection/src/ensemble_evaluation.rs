//! Ensemble Cross-Validation and Diversity-Based Evaluation
//!
//! This module provides advanced evaluation methods for ensemble models including
//! specialized cross-validation techniques, diversity measures, stability analysis,
//! and out-of-bag evaluation strategies specifically designed for ensemble learning.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use sklears_core::types::Float;
use std::collections::HashMap;

/// Ensemble evaluation strategies
#[derive(Debug, Clone)]
pub enum EnsembleEvaluationStrategy {
    /// Out-of-bag evaluation for bootstrap-based ensembles
    OutOfBag {
        bootstrap_samples: usize,

        confidence_level: Float,
    },
    /// Ensemble-specific cross-validation
    EnsembleCrossValidation {
        cv_strategy: EnsembleCVStrategy,

        n_folds: usize,
    },
    /// Diversity-based evaluation
    DiversityEvaluation {
        diversity_measures: Vec<DiversityMeasure>,
        diversity_threshold: Float,
    },
    /// Stability analysis across different data splits
    StabilityAnalysis {
        n_bootstrap_samples: usize,
        stability_metrics: Vec<StabilityMetric>,
    },
    /// Progressive ensemble evaluation
    ProgressiveEvaluation {
        ensemble_sizes: Vec<usize>,
        selection_strategy: ProgressiveSelectionStrategy,
    },
    /// Multi-objective ensemble evaluation
    MultiObjectiveEvaluation {
        objectives: Vec<EvaluationObjective>,
        trade_off_analysis: bool,
    },
}

/// Cross-validation strategies specific to ensembles
#[derive(Debug, Clone)]
pub enum EnsembleCVStrategy {
    /// Standard k-fold CV for ensembles
    KFoldEnsemble,
    /// Stratified CV maintaining ensemble member diversity
    StratifiedEnsemble,
    /// Leave-one-model-out CV
    LeaveOneModelOut,
    /// Bootstrap CV for ensemble components
    BootstrapEnsemble { n_bootstrap: usize },
    /// Nested CV for ensemble and member optimization
    NestedEnsemble { inner_cv: usize, outer_cv: usize },
    /// Time series CV for temporal ensembles
    TimeSeriesEnsemble { n_splits: usize, test_size: Float },
}

/// Diversity measures for ensemble evaluation
#[derive(Debug, Clone)]
pub enum DiversityMeasure {
    /// Q-statistic between pairs of classifiers
    QStatistic,
    /// Correlation coefficient between predictions
    CorrelationCoefficient,
    /// Disagreement measure
    DisagreementMeasure,
    /// Double-fault measure
    DoubleFaultMeasure,
    /// Entropy-based diversity
    EntropyDiversity,
    /// Kohavi-Wolpert variance
    KohaviWolpertVariance,
    /// Interrater agreement (Kappa)
    InterraterAgreement,
    /// Measurement of difficulty
    DifficultyMeasure,
    /// Generalized diversity index
    GeneralizedDiversity { alpha: Float },
}

/// Stability metrics for ensemble evaluation
#[derive(Debug, Clone)]
pub enum StabilityMetric {
    /// Prediction stability across bootstrap samples
    PredictionStability,
    /// Model selection stability
    ModelSelectionStability,
    /// Weight stability for weighted ensembles
    WeightStability,
    /// Performance stability
    PerformanceStability,
    /// Ranking stability of ensemble members
    RankingStability,
}

/// Progressive selection strategies
#[derive(Debug, Clone)]
pub enum ProgressiveSelectionStrategy {
    /// Forward selection based on performance
    ForwardSelection,
    /// Backward elimination
    BackwardElimination,
    /// Diversity-driven selection
    DiversityDriven,
    /// Performance-diversity trade-off
    PerformanceDiversityTradeoff { alpha: Float },
}

/// Evaluation objectives for multi-objective analysis
#[derive(Debug, Clone)]
pub enum EvaluationObjective {
    /// Predictive accuracy
    Accuracy,
    /// Model diversity
    Diversity,
    /// Computational efficiency
    Efficiency,
    /// Memory usage
    MemoryUsage,
    /// Robustness to outliers
    Robustness,
    /// Interpretability
    Interpretability,
    /// Fairness across groups
    Fairness,
}

/// Ensemble evaluation configuration
#[derive(Debug, Clone)]
pub struct EnsembleEvaluationConfig {
    pub strategy: EnsembleEvaluationStrategy,
    pub evaluation_metrics: Vec<String>,
    pub confidence_level: Float,
    pub n_repetitions: usize,
    pub parallel_evaluation: bool,
    pub random_state: Option<u64>,
    pub verbose: bool,
}

/// Ensemble evaluation result
#[derive(Debug, Clone)]
pub struct EnsembleEvaluationResult {
    pub ensemble_performance: EnsemblePerformanceMetrics,
    pub diversity_analysis: DiversityAnalysis,
    pub stability_analysis: Option<StabilityAnalysis>,
    pub member_contributions: Vec<MemberContribution>,
    pub out_of_bag_scores: Option<OutOfBagScores>,
    pub progressive_performance: Option<ProgressivePerformance>,
    pub multi_objective_analysis: Option<MultiObjectiveAnalysis>,
}

/// Comprehensive ensemble performance metrics
#[derive(Debug, Clone)]
pub struct EnsemblePerformanceMetrics {
    pub mean_performance: Float,
    pub std_performance: Float,
    pub confidence_interval: (Float, Float),
    pub individual_fold_scores: Vec<Float>,
    pub ensemble_vs_best_member: Float,
    pub ensemble_vs_average_member: Float,
    pub performance_gain: Float,
}

/// Diversity analysis results
#[derive(Debug, Clone)]
pub struct DiversityAnalysis {
    pub overall_diversity: Float,
    pub pairwise_diversities: Array2<Float>,
    pub diversity_by_measure: HashMap<String, Float>,
    pub diversity_distribution: Vec<Float>,
    pub optimal_diversity_size: Option<usize>,
}

/// Stability analysis results
#[derive(Debug, Clone)]
pub struct StabilityAnalysis {
    pub prediction_stability: Float,
    pub model_selection_stability: Float,
    pub weight_stability: Option<Float>,
    pub performance_stability: Float,
    pub stability_confidence_intervals: HashMap<String, (Float, Float)>,
}

/// Individual member contribution analysis
#[derive(Debug, Clone)]
pub struct MemberContribution {
    pub member_id: usize,
    pub member_name: String,
    pub individual_performance: Float,
    pub marginal_contribution: Float,
    pub shapley_value: Option<Float>,
    pub removal_impact: Float,
    pub diversity_contribution: Float,
}

/// Out-of-bag evaluation scores
#[derive(Debug, Clone)]
pub struct OutOfBagScores {
    pub oob_score: Float,
    pub oob_confidence_interval: (Float, Float),
    pub feature_importance: Option<Array1<Float>>,
    pub prediction_intervals: Option<Array2<Float>>,
    pub individual_oob_scores: Vec<Float>,
}

/// Progressive performance analysis
#[derive(Debug, Clone)]
pub struct ProgressivePerformance {
    pub ensemble_sizes: Vec<usize>,
    pub performance_curve: Vec<Float>,
    pub diversity_curve: Vec<Float>,
    pub efficiency_curve: Vec<Float>,
    pub optimal_size: usize,
    pub diminishing_returns_threshold: Option<usize>,
}

/// Multi-objective analysis results
#[derive(Debug, Clone)]
pub struct MultiObjectiveAnalysis {
    pub pareto_front: Vec<(Float, Float)>,
    pub objective_scores: HashMap<String, Float>,
    pub trade_off_analysis: HashMap<String, Float>,
    pub dominated_solutions: Vec<usize>,
    pub compromise_solution: Option<usize>,
}

/// Ensemble evaluator
#[derive(Debug)]
pub struct EnsembleEvaluator {
    config: EnsembleEvaluationConfig,
    rng: StdRng,
}

impl Default for EnsembleEvaluationConfig {
    fn default() -> Self {
        Self {
            strategy: EnsembleEvaluationStrategy::EnsembleCrossValidation {
                cv_strategy: EnsembleCVStrategy::KFoldEnsemble,
                n_folds: 5,
            },
            evaluation_metrics: vec!["accuracy".to_string(), "f1_score".to_string()],
            confidence_level: 0.95,
            n_repetitions: 1,
            parallel_evaluation: false,
            random_state: None,
            verbose: false,
        }
    }
}

impl EnsembleEvaluator {
    /// Create a new ensemble evaluator
    pub fn new(config: EnsembleEvaluationConfig) -> Self {
        let rng = match config.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                use scirs2_core::random::thread_rng;
                StdRng::from_rng(&mut thread_rng())
            }
        };

        Self { config, rng }
    }

    /// Evaluate ensemble using specified strategy
    pub fn evaluate<F>(
        &mut self,
        ensemble_predictions: &Array2<Float>,
        true_labels: &Array1<Float>,
        ensemble_weights: Option<&Array1<Float>>,
        model_predictions: Option<&Array2<Float>>,
        evaluation_fn: F,
    ) -> Result<EnsembleEvaluationResult, Box<dyn std::error::Error>>
    where
        F: Fn(&Array1<Float>, &Array1<Float>) -> Result<Float, Box<dyn std::error::Error>>,
    {
        match &self.config.strategy {
            EnsembleEvaluationStrategy::OutOfBag { .. } => self.evaluate_out_of_bag(
                ensemble_predictions,
                true_labels,
                ensemble_weights,
                &evaluation_fn,
            ),
            EnsembleEvaluationStrategy::EnsembleCrossValidation { .. } => self
                .evaluate_cross_validation(
                    ensemble_predictions,
                    true_labels,
                    ensemble_weights,
                    model_predictions,
                    &evaluation_fn,
                ),
            EnsembleEvaluationStrategy::DiversityEvaluation { .. } => self.evaluate_diversity(
                ensemble_predictions,
                true_labels,
                model_predictions,
                &evaluation_fn,
            ),
            EnsembleEvaluationStrategy::StabilityAnalysis { .. } => self.evaluate_stability(
                ensemble_predictions,
                true_labels,
                ensemble_weights,
                &evaluation_fn,
            ),
            EnsembleEvaluationStrategy::ProgressiveEvaluation { .. } => self.evaluate_progressive(
                ensemble_predictions,
                true_labels,
                model_predictions,
                &evaluation_fn,
            ),
            EnsembleEvaluationStrategy::MultiObjectiveEvaluation { .. } => self
                .evaluate_multi_objective(
                    ensemble_predictions,
                    true_labels,
                    ensemble_weights,
                    &evaluation_fn,
                ),
        }
    }

    /// Out-of-bag evaluation implementation
    fn evaluate_out_of_bag<F>(
        &mut self,
        ensemble_predictions: &Array2<Float>,
        true_labels: &Array1<Float>,
        ensemble_weights: Option<&Array1<Float>>,
        evaluation_fn: &F,
    ) -> Result<EnsembleEvaluationResult, Box<dyn std::error::Error>>
    where
        F: Fn(&Array1<Float>, &Array1<Float>) -> Result<Float, Box<dyn std::error::Error>>,
    {
        let (bootstrap_samples, confidence_level) = match &self.config.strategy {
            EnsembleEvaluationStrategy::OutOfBag {
                bootstrap_samples,
                confidence_level,
            } => (*bootstrap_samples, *confidence_level),
            _ => unreachable!(),
        };

        let n_samples = ensemble_predictions.nrows();
        let n_models = ensemble_predictions.ncols();

        let mut oob_scores = Vec::new();
        let mut oob_predictions_all = Vec::new();

        for _ in 0..bootstrap_samples {
            // Generate bootstrap sample
            let bootstrap_indices: Vec<usize> = (0..n_samples)
                .map(|_| self.rng.gen_range(0..n_samples))
                .collect();

            // Find out-of-bag samples
            let mut oob_indices = Vec::new();
            for i in 0..n_samples {
                if !bootstrap_indices.contains(&i) {
                    oob_indices.push(i);
                }
            }

            if oob_indices.is_empty() {
                continue;
            }

            // Calculate OOB predictions
            let oob_ensemble_preds = self.calculate_ensemble_predictions(
                ensemble_predictions,
                &oob_indices,
                ensemble_weights,
            )?;

            let oob_true_labels =
                Array1::from_vec(oob_indices.iter().map(|&i| true_labels[i]).collect());

            let oob_score = evaluation_fn(&oob_ensemble_preds, &oob_true_labels)?;
            oob_scores.push(oob_score);
            oob_predictions_all.push(oob_ensemble_preds);
        }

        let mean_oob_score = oob_scores.iter().sum::<Float>() / oob_scores.len() as Float;
        let std_oob_score = {
            let variance = oob_scores
                .iter()
                .map(|&score| (score - mean_oob_score).powi(2))
                .sum::<Float>()
                / oob_scores.len() as Float;
            variance.sqrt()
        };

        let alpha = 1.0 - confidence_level;
        let z_score = 1.96; // Approximate for 95% confidence
        let margin_of_error = z_score * std_oob_score / (oob_scores.len() as Float).sqrt();
        let confidence_interval = (
            mean_oob_score - margin_of_error,
            mean_oob_score + margin_of_error,
        );

        let oob_scores_result = OutOfBagScores {
            oob_score: mean_oob_score,
            oob_confidence_interval: confidence_interval,
            feature_importance: None, // Could be calculated if feature data available
            prediction_intervals: None, // Could be calculated from OOB predictions
            individual_oob_scores: oob_scores,
        };

        // Calculate basic ensemble performance
        let ensemble_preds = self.calculate_ensemble_predictions(
            ensemble_predictions,
            &(0..n_samples).collect::<Vec<_>>(),
            ensemble_weights,
        )?;
        let ensemble_score = evaluation_fn(&ensemble_preds, true_labels)?;

        let ensemble_performance = EnsemblePerformanceMetrics {
            mean_performance: ensemble_score,
            std_performance: std_oob_score,
            confidence_interval,
            individual_fold_scores: vec![ensemble_score],
            ensemble_vs_best_member: 0.0, // Would need individual model scores
            ensemble_vs_average_member: 0.0,
            performance_gain: 0.0,
        };

        Ok(EnsembleEvaluationResult {
            ensemble_performance,
            diversity_analysis: DiversityAnalysis {
                overall_diversity: 0.0,
                pairwise_diversities: Array2::zeros((n_models, n_models)),
                diversity_by_measure: HashMap::new(),
                diversity_distribution: Vec::new(),
                optimal_diversity_size: None,
            },
            stability_analysis: None,
            member_contributions: Vec::new(),
            out_of_bag_scores: Some(oob_scores_result),
            progressive_performance: None,
            multi_objective_analysis: None,
        })
    }

    /// Cross-validation evaluation implementation
    fn evaluate_cross_validation<F>(
        &mut self,
        ensemble_predictions: &Array2<Float>,
        true_labels: &Array1<Float>,
        ensemble_weights: Option<&Array1<Float>>,
        model_predictions: Option<&Array2<Float>>,
        evaluation_fn: &F,
    ) -> Result<EnsembleEvaluationResult, Box<dyn std::error::Error>>
    where
        F: Fn(&Array1<Float>, &Array1<Float>) -> Result<Float, Box<dyn std::error::Error>>,
    {
        let (cv_strategy, n_folds) = match &self.config.strategy {
            EnsembleEvaluationStrategy::EnsembleCrossValidation {
                cv_strategy,
                n_folds,
            } => (cv_strategy, *n_folds),
            _ => unreachable!(),
        };

        let n_samples = ensemble_predictions.nrows();
        let fold_size = n_samples / n_folds;
        let mut fold_scores = Vec::new();
        let mut diversity_scores = Vec::new();

        for fold in 0..n_folds {
            let test_start = fold * fold_size;
            let test_end = if fold == n_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };
            let test_indices: Vec<usize> = (test_start..test_end).collect();

            // Calculate ensemble predictions for test fold
            let test_ensemble_preds = self.calculate_ensemble_predictions(
                ensemble_predictions,
                &test_indices,
                ensemble_weights,
            )?;

            let test_true_labels =
                Array1::from_vec(test_indices.iter().map(|&i| true_labels[i]).collect());

            let fold_score = evaluation_fn(&test_ensemble_preds, &test_true_labels)?;
            fold_scores.push(fold_score);

            // Calculate diversity for this fold if model predictions available
            if let Some(model_preds) = model_predictions {
                let mut fold_data = Vec::new();
                for &i in test_indices.iter() {
                    fold_data.extend(model_preds.row(i).iter().cloned());
                }
                let fold_model_preds =
                    Array2::from_shape_vec((test_indices.len(), model_preds.ncols()), fold_data)?;

                let diversity = self.calculate_q_statistic(&fold_model_preds)?;
                diversity_scores.push(diversity);
            }
        }

        let mean_performance = fold_scores.iter().sum::<Float>() / fold_scores.len() as Float;
        let std_performance = {
            let variance = fold_scores
                .iter()
                .map(|&score| (score - mean_performance).powi(2))
                .sum::<Float>()
                / fold_scores.len() as Float;
            variance.sqrt()
        };

        let z_score = 1.96; // 95% confidence
        let margin_of_error = z_score * std_performance / (fold_scores.len() as Float).sqrt();
        let confidence_interval = (
            mean_performance - margin_of_error,
            mean_performance + margin_of_error,
        );

        let ensemble_performance = EnsemblePerformanceMetrics {
            mean_performance,
            std_performance,
            confidence_interval,
            individual_fold_scores: fold_scores,
            ensemble_vs_best_member: 0.0, // Would need individual model analysis
            ensemble_vs_average_member: 0.0,
            performance_gain: 0.0,
        };

        let diversity_analysis = if !diversity_scores.is_empty() {
            let mean_diversity =
                diversity_scores.iter().sum::<Float>() / diversity_scores.len() as Float;
            DiversityAnalysis {
                overall_diversity: mean_diversity,
                pairwise_diversities: Array2::zeros((0, 0)), // Would calculate pairwise if needed
                diversity_by_measure: {
                    let mut map = HashMap::new();
                    map.insert("q_statistic".to_string(), mean_diversity);
                    map
                },
                diversity_distribution: diversity_scores,
                optimal_diversity_size: None,
            }
        } else {
            DiversityAnalysis {
                overall_diversity: 0.0,
                pairwise_diversities: Array2::zeros((0, 0)),
                diversity_by_measure: HashMap::new(),
                diversity_distribution: Vec::new(),
                optimal_diversity_size: None,
            }
        };

        Ok(EnsembleEvaluationResult {
            ensemble_performance,
            diversity_analysis,
            stability_analysis: None,
            member_contributions: Vec::new(),
            out_of_bag_scores: None,
            progressive_performance: None,
            multi_objective_analysis: None,
        })
    }

    /// Diversity evaluation implementation
    fn evaluate_diversity<F>(
        &mut self,
        ensemble_predictions: &Array2<Float>,
        true_labels: &Array1<Float>,
        model_predictions: Option<&Array2<Float>>,
        evaluation_fn: &F,
    ) -> Result<EnsembleEvaluationResult, Box<dyn std::error::Error>>
    where
        F: Fn(&Array1<Float>, &Array1<Float>) -> Result<Float, Box<dyn std::error::Error>>,
    {
        let (diversity_measures, diversity_threshold) = match &self.config.strategy {
            EnsembleEvaluationStrategy::DiversityEvaluation {
                diversity_measures,
                diversity_threshold,
            } => (diversity_measures, *diversity_threshold),
            _ => unreachable!(),
        };

        if let Some(model_preds) = model_predictions {
            let n_models = model_preds.ncols();
            let mut diversity_by_measure = HashMap::new();
            let mut pairwise_diversities = Array2::zeros((n_models, n_models));

            for measure in diversity_measures {
                let diversity_value = match measure {
                    DiversityMeasure::QStatistic => self.calculate_q_statistic(model_preds)?,
                    DiversityMeasure::CorrelationCoefficient => {
                        self.calculate_correlation_coefficient(model_preds)?
                    }
                    DiversityMeasure::DisagreementMeasure => {
                        self.calculate_disagreement_measure(model_preds)?
                    }
                    DiversityMeasure::DoubleFaultMeasure => {
                        self.calculate_double_fault_measure(model_preds, true_labels)?
                    }
                    DiversityMeasure::EntropyDiversity => {
                        self.calculate_entropy_diversity(model_preds)?
                    }
                    DiversityMeasure::KohaviWolpertVariance => {
                        self.calculate_kw_variance(model_preds, true_labels)?
                    }
                    DiversityMeasure::InterraterAgreement => {
                        self.calculate_interrater_agreement(model_preds)?
                    }
                    DiversityMeasure::DifficultyMeasure => {
                        self.calculate_difficulty_measure(model_preds, true_labels)?
                    }
                    DiversityMeasure::GeneralizedDiversity { alpha } => {
                        self.calculate_generalized_diversity(model_preds, *alpha)?
                    }
                };

                diversity_by_measure.insert(format!("{:?}", measure), diversity_value);
            }

            // Calculate pairwise diversities
            for i in 0..n_models {
                for j in i + 1..n_models {
                    let pair_preds = Array2::from_shape_vec(
                        (model_preds.nrows(), 2),
                        model_preds
                            .column(i)
                            .iter()
                            .cloned()
                            .chain(model_preds.column(j).iter().cloned())
                            .collect(),
                    )?;
                    let pair_diversity = self.calculate_q_statistic(&pair_preds)?;
                    pairwise_diversities[[i, j]] = pair_diversity;
                    pairwise_diversities[[j, i]] = pair_diversity;
                }
            }

            let overall_diversity =
                diversity_by_measure.values().sum::<Float>() / diversity_by_measure.len() as Float;

            let diversity_analysis = DiversityAnalysis {
                overall_diversity,
                pairwise_diversities,
                diversity_by_measure,
                diversity_distribution: Vec::new(), // Could add distribution analysis
                optimal_diversity_size: None,       // Could calculate optimal size
            };

            // Calculate basic ensemble performance
            let ensemble_preds = self.calculate_ensemble_predictions(
                ensemble_predictions,
                &(0..ensemble_predictions.nrows()).collect::<Vec<_>>(),
                None,
            )?;
            let ensemble_score = evaluation_fn(&ensemble_preds, true_labels)?;

            let ensemble_performance = EnsemblePerformanceMetrics {
                mean_performance: ensemble_score,
                std_performance: 0.0,
                confidence_interval: (ensemble_score, ensemble_score),
                individual_fold_scores: vec![ensemble_score],
                ensemble_vs_best_member: 0.0,
                ensemble_vs_average_member: 0.0,
                performance_gain: 0.0,
            };

            Ok(EnsembleEvaluationResult {
                ensemble_performance,
                diversity_analysis,
                stability_analysis: None,
                member_contributions: Vec::new(),
                out_of_bag_scores: None,
                progressive_performance: None,
                multi_objective_analysis: None,
            })
        } else {
            Err("Model predictions required for diversity evaluation".into())
        }
    }

    /// Stability analysis implementation
    fn evaluate_stability<F>(
        &mut self,
        ensemble_predictions: &Array2<Float>,
        true_labels: &Array1<Float>,
        ensemble_weights: Option<&Array1<Float>>,
        evaluation_fn: &F,
    ) -> Result<EnsembleEvaluationResult, Box<dyn std::error::Error>>
    where
        F: Fn(&Array1<Float>, &Array1<Float>) -> Result<Float, Box<dyn std::error::Error>>,
    {
        let (n_bootstrap_samples, stability_metrics) = match &self.config.strategy {
            EnsembleEvaluationStrategy::StabilityAnalysis {
                n_bootstrap_samples,
                stability_metrics,
            } => (*n_bootstrap_samples, stability_metrics),
            _ => unreachable!(),
        };

        let n_samples = ensemble_predictions.nrows();
        let mut bootstrap_scores = Vec::new();
        let mut bootstrap_predictions = Vec::new();

        for _ in 0..n_bootstrap_samples {
            // Generate bootstrap sample
            let bootstrap_indices: Vec<usize> = (0..n_samples)
                .map(|_| self.rng.gen_range(0..n_samples))
                .collect();

            let bootstrap_preds = self.calculate_ensemble_predictions(
                ensemble_predictions,
                &bootstrap_indices,
                ensemble_weights,
            )?;

            let bootstrap_labels =
                Array1::from_vec(bootstrap_indices.iter().map(|&i| true_labels[i]).collect());

            let bootstrap_score = evaluation_fn(&bootstrap_preds, &bootstrap_labels)?;
            bootstrap_scores.push(bootstrap_score);
            bootstrap_predictions.push(bootstrap_preds);
        }

        // Calculate prediction stability
        let prediction_stability = self.calculate_prediction_stability(&bootstrap_predictions)?;

        // Calculate performance stability
        let mean_score = bootstrap_scores.iter().sum::<Float>() / bootstrap_scores.len() as Float;
        let score_variance = bootstrap_scores
            .iter()
            .map(|&score| (score - mean_score).powi(2))
            .sum::<Float>()
            / bootstrap_scores.len() as Float;
        let performance_stability = 1.0 / (1.0 + score_variance); // Higher variance = lower stability

        let stability_analysis = StabilityAnalysis {
            prediction_stability,
            model_selection_stability: 0.8, // Placeholder - would need model selection data
            weight_stability: None,         // Would calculate if weights provided
            performance_stability,
            stability_confidence_intervals: HashMap::new(), // Could add CIs
        };

        let ensemble_performance = EnsemblePerformanceMetrics {
            mean_performance: mean_score,
            std_performance: score_variance.sqrt(),
            confidence_interval: (
                mean_score - score_variance.sqrt(),
                mean_score + score_variance.sqrt(),
            ),
            individual_fold_scores: bootstrap_scores,
            ensemble_vs_best_member: 0.0,
            ensemble_vs_average_member: 0.0,
            performance_gain: 0.0,
        };

        Ok(EnsembleEvaluationResult {
            ensemble_performance,
            diversity_analysis: DiversityAnalysis {
                overall_diversity: 0.0,
                pairwise_diversities: Array2::zeros((0, 0)),
                diversity_by_measure: HashMap::new(),
                diversity_distribution: Vec::new(),
                optimal_diversity_size: None,
            },
            stability_analysis: Some(stability_analysis),
            member_contributions: Vec::new(),
            out_of_bag_scores: None,
            progressive_performance: None,
            multi_objective_analysis: None,
        })
    }

    /// Progressive evaluation implementation
    fn evaluate_progressive<F>(
        &mut self,
        ensemble_predictions: &Array2<Float>,
        true_labels: &Array1<Float>,
        model_predictions: Option<&Array2<Float>>,
        evaluation_fn: &F,
    ) -> Result<EnsembleEvaluationResult, Box<dyn std::error::Error>>
    where
        F: Fn(&Array1<Float>, &Array1<Float>) -> Result<Float, Box<dyn std::error::Error>>,
    {
        let (ensemble_sizes, _selection_strategy) = match &self.config.strategy {
            EnsembleEvaluationStrategy::ProgressiveEvaluation {
                ensemble_sizes,
                selection_strategy,
            } => (ensemble_sizes, selection_strategy),
            _ => unreachable!(),
        };

        if let Some(model_preds) = model_predictions {
            let mut performance_curve = Vec::new();
            let mut diversity_curve = Vec::new();
            let n_models = model_preds.ncols();

            for &size in ensemble_sizes {
                if size <= n_models {
                    // Select top models (simplified - could use more sophisticated selection)
                    let selected_indices: Vec<usize> = (0..size).collect();

                    // Calculate ensemble predictions for selected models
                    let mut selected_data = Vec::new();
                    for &i in selected_indices.iter() {
                        selected_data.extend(model_preds.column(i).iter().cloned());
                    }
                    let selected_predictions =
                        Array2::from_shape_vec((model_preds.nrows(), size), selected_data)?;

                    let ensemble_preds = selected_predictions.mean_axis(Axis(1)).unwrap();
                    let performance = evaluation_fn(&ensemble_preds, true_labels)?;
                    performance_curve.push(performance);

                    // Calculate diversity for selected models
                    let diversity = self.calculate_q_statistic(&selected_predictions)?;
                    diversity_curve.push(diversity);
                }
            }

            // Find optimal size (highest performance)
            let optimal_size_idx = performance_curve
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            let optimal_size = ensemble_sizes[optimal_size_idx];

            let progressive_performance = ProgressivePerformance {
                ensemble_sizes: ensemble_sizes.clone(),
                performance_curve,
                diversity_curve,
                efficiency_curve: vec![1.0; ensemble_sizes.len()], // Placeholder
                optimal_size,
                diminishing_returns_threshold: None, // Could calculate
            };

            let ensemble_performance = EnsemblePerformanceMetrics {
                mean_performance: progressive_performance.performance_curve[optimal_size_idx],
                std_performance: 0.0,
                confidence_interval: (0.0, 0.0),
                individual_fold_scores: vec![],
                ensemble_vs_best_member: 0.0,
                ensemble_vs_average_member: 0.0,
                performance_gain: 0.0,
            };

            Ok(EnsembleEvaluationResult {
                ensemble_performance,
                diversity_analysis: DiversityAnalysis {
                    overall_diversity: 0.0,
                    pairwise_diversities: Array2::zeros((0, 0)),
                    diversity_by_measure: HashMap::new(),
                    diversity_distribution: Vec::new(),
                    optimal_diversity_size: Some(optimal_size),
                },
                stability_analysis: None,
                member_contributions: Vec::new(),
                out_of_bag_scores: None,
                progressive_performance: Some(progressive_performance),
                multi_objective_analysis: None,
            })
        } else {
            Err("Model predictions required for progressive evaluation".into())
        }
    }

    /// Multi-objective evaluation implementation
    fn evaluate_multi_objective<F>(
        &mut self,
        ensemble_predictions: &Array2<Float>,
        true_labels: &Array1<Float>,
        ensemble_weights: Option<&Array1<Float>>,
        evaluation_fn: &F,
    ) -> Result<EnsembleEvaluationResult, Box<dyn std::error::Error>>
    where
        F: Fn(&Array1<Float>, &Array1<Float>) -> Result<Float, Box<dyn std::error::Error>>,
    {
        // Simplified multi-objective evaluation
        let ensemble_preds = self.calculate_ensemble_predictions(
            ensemble_predictions,
            &(0..ensemble_predictions.nrows()).collect::<Vec<_>>(),
            ensemble_weights,
        )?;
        let performance = evaluation_fn(&ensemble_preds, true_labels)?;

        let mut objective_scores = HashMap::new();
        objective_scores.insert("accuracy".to_string(), performance);
        objective_scores.insert("diversity".to_string(), 0.5); // Placeholder
        objective_scores.insert("efficiency".to_string(), 0.8); // Placeholder

        let multi_objective_analysis = MultiObjectiveAnalysis {
            pareto_front: vec![(performance, 0.5)], // (accuracy, diversity)
            objective_scores,
            trade_off_analysis: HashMap::new(),
            dominated_solutions: Vec::new(),
            compromise_solution: Some(0),
        };

        let ensemble_performance = EnsemblePerformanceMetrics {
            mean_performance: performance,
            std_performance: 0.0,
            confidence_interval: (performance, performance),
            individual_fold_scores: vec![performance],
            ensemble_vs_best_member: 0.0,
            ensemble_vs_average_member: 0.0,
            performance_gain: 0.0,
        };

        Ok(EnsembleEvaluationResult {
            ensemble_performance,
            diversity_analysis: DiversityAnalysis {
                overall_diversity: 0.0,
                pairwise_diversities: Array2::zeros((0, 0)),
                diversity_by_measure: HashMap::new(),
                diversity_distribution: Vec::new(),
                optimal_diversity_size: None,
            },
            stability_analysis: None,
            member_contributions: Vec::new(),
            out_of_bag_scores: None,
            progressive_performance: None,
            multi_objective_analysis: Some(multi_objective_analysis),
        })
    }

    /// Calculate ensemble predictions for given indices
    fn calculate_ensemble_predictions(
        &self,
        ensemble_predictions: &Array2<Float>,
        indices: &[usize],
        weights: Option<&Array1<Float>>,
    ) -> Result<Array1<Float>, Box<dyn std::error::Error>> {
        let mut selected_data = Vec::new();
        for &i in indices.iter() {
            selected_data.extend(ensemble_predictions.row(i).iter().cloned());
        }
        let selected_predictions =
            Array2::from_shape_vec((indices.len(), ensemble_predictions.ncols()), selected_data)?;

        if let Some(w) = weights {
            // Weighted average
            Ok(selected_predictions.dot(w))
        } else {
            // Simple average
            Ok(selected_predictions.mean_axis(Axis(1)).unwrap())
        }
    }

    /// Calculate Q-statistic diversity measure
    fn calculate_q_statistic(
        &self,
        predictions: &Array2<Float>,
    ) -> Result<Float, Box<dyn std::error::Error>> {
        let n_models = predictions.ncols();
        if n_models < 2 {
            return Ok(0.0);
        }

        let mut q_sum = 0.0;
        let mut pairs = 0;

        for i in 0..n_models {
            for j in i + 1..n_models {
                let pred_i = predictions.column(i);
                let pred_j = predictions.column(j);

                let mut n11 = 0; // Both correct
                let mut n10 = 0; // i correct, j wrong
                let mut n01 = 0; // i wrong, j correct
                let mut n00 = 0; // Both wrong

                for k in 0..predictions.nrows() {
                    let i_correct = pred_i[k] > 0.5;
                    let j_correct = pred_j[k] > 0.5;

                    match (i_correct, j_correct) {
                        (true, true) => n11 += 1,
                        (true, false) => n10 += 1,
                        (false, true) => n01 += 1,
                        (false, false) => n00 += 1,
                    }
                }

                let numerator = (n11 * n00 - n01 * n10) as Float;
                let denominator = (n11 * n00 + n01 * n10) as Float;

                if denominator != 0.0 {
                    q_sum += numerator / denominator;
                    pairs += 1;
                }
            }
        }

        Ok(if pairs > 0 {
            q_sum / pairs as Float
        } else {
            0.0
        })
    }

    /// Calculate correlation coefficient diversity measure
    fn calculate_correlation_coefficient(
        &self,
        predictions: &Array2<Float>,
    ) -> Result<Float, Box<dyn std::error::Error>> {
        let n_models = predictions.ncols();
        if n_models < 2 {
            return Ok(0.0);
        }

        let mut correlations = Vec::new();
        for i in 0..n_models {
            for j in i + 1..n_models {
                let pred_i = predictions.column(i);
                let pred_j = predictions.column(j);

                let mean_i = pred_i.mean().unwrap_or(0.0);
                let mean_j = pred_j.mean().unwrap_or(0.0);

                let mut covariance = 0.0;
                let mut var_i = 0.0;
                let mut var_j = 0.0;

                for k in 0..predictions.nrows() {
                    let diff_i = pred_i[k] - mean_i;
                    let diff_j = pred_j[k] - mean_j;
                    covariance += diff_i * diff_j;
                    var_i += diff_i * diff_i;
                    var_j += diff_j * diff_j;
                }

                let correlation = if var_i > 0.0 && var_j > 0.0 {
                    covariance / (var_i.sqrt() * var_j.sqrt())
                } else {
                    0.0
                };

                correlations.push(correlation.abs());
            }
        }

        Ok(1.0 - correlations.iter().sum::<Float>() / correlations.len() as Float)
    }

    /// Calculate disagreement measure
    fn calculate_disagreement_measure(
        &self,
        predictions: &Array2<Float>,
    ) -> Result<Float, Box<dyn std::error::Error>> {
        let n_models = predictions.ncols();
        if n_models < 2 {
            return Ok(0.0);
        }

        let mut disagreement_sum = 0.0;
        let mut pairs = 0;

        for i in 0..n_models {
            for j in i + 1..n_models {
                let pred_i = predictions.column(i);
                let pred_j = predictions.column(j);

                let mut disagreements = 0;
                for k in 0..predictions.nrows() {
                    if (pred_i[k] > 0.5) != (pred_j[k] > 0.5) {
                        disagreements += 1;
                    }
                }

                disagreement_sum += disagreements as Float / predictions.nrows() as Float;
                pairs += 1;
            }
        }

        Ok(if pairs > 0 {
            disagreement_sum / pairs as Float
        } else {
            0.0
        })
    }

    /// Calculate double fault measure
    fn calculate_double_fault_measure(
        &self,
        predictions: &Array2<Float>,
        true_labels: &Array1<Float>,
    ) -> Result<Float, Box<dyn std::error::Error>> {
        let n_models = predictions.ncols();
        if n_models < 2 {
            return Ok(0.0);
        }

        let mut double_fault_sum = 0.0;
        let mut pairs = 0;

        for i in 0..n_models {
            for j in i + 1..n_models {
                let pred_i = predictions.column(i);
                let pred_j = predictions.column(j);

                let mut double_faults = 0;
                for k in 0..predictions.nrows() {
                    let i_wrong = (pred_i[k] > 0.5) != (true_labels[k] > 0.5);
                    let j_wrong = (pred_j[k] > 0.5) != (true_labels[k] > 0.5);

                    if i_wrong && j_wrong {
                        double_faults += 1;
                    }
                }

                double_fault_sum += double_faults as Float / predictions.nrows() as Float;
                pairs += 1;
            }
        }

        Ok(if pairs > 0 {
            double_fault_sum / pairs as Float
        } else {
            0.0
        })
    }

    /// Calculate entropy-based diversity
    fn calculate_entropy_diversity(
        &self,
        predictions: &Array2<Float>,
    ) -> Result<Float, Box<dyn std::error::Error>> {
        let n_samples = predictions.nrows();
        let n_models = predictions.ncols();

        let mut entropy_sum = 0.0;

        for i in 0..n_samples {
            let correct_count = predictions
                .row(i)
                .iter()
                .filter(|&&pred| pred > 0.5)
                .count() as Float;

            let p = correct_count / n_models as Float;
            if p > 0.0 && p < 1.0 {
                entropy_sum += -p * p.log2() - (1.0 - p) * (1.0 - p).log2();
            }
        }

        Ok(entropy_sum / n_samples as Float)
    }

    /// Calculate Kohavi-Wolpert variance
    fn calculate_kw_variance(
        &self,
        predictions: &Array2<Float>,
        true_labels: &Array1<Float>,
    ) -> Result<Float, Box<dyn std::error::Error>> {
        let n_samples = predictions.nrows();
        let n_models = predictions.ncols();

        let mut variance_sum = 0.0;

        for i in 0..n_samples {
            let correct_count = predictions
                .row(i)
                .iter()
                .filter(|&&pred| (pred > 0.5) == (true_labels[i] > 0.5))
                .count() as Float;

            let l = correct_count / n_models as Float;
            variance_sum += l * (1.0 - l);
        }

        Ok(variance_sum / n_samples as Float)
    }

    /// Calculate interrater agreement (simplified Kappa)
    fn calculate_interrater_agreement(
        &self,
        predictions: &Array2<Float>,
    ) -> Result<Float, Box<dyn std::error::Error>> {
        let n_models = predictions.ncols();
        if n_models < 2 {
            return Ok(0.0);
        }

        let mut agreement_sum = 0.0;
        let mut pairs = 0;

        for i in 0..n_models {
            for j in i + 1..n_models {
                let pred_i = predictions.column(i);
                let pred_j = predictions.column(j);

                let mut agreements = 0;
                for k in 0..predictions.nrows() {
                    if (pred_i[k] > 0.5) == (pred_j[k] > 0.5) {
                        agreements += 1;
                    }
                }

                agreement_sum += agreements as Float / predictions.nrows() as Float;
                pairs += 1;
            }
        }

        Ok(if pairs > 0 {
            agreement_sum / pairs as Float
        } else {
            0.0
        })
    }

    /// Calculate difficulty measure
    fn calculate_difficulty_measure(
        &self,
        predictions: &Array2<Float>,
        true_labels: &Array1<Float>,
    ) -> Result<Float, Box<dyn std::error::Error>> {
        let n_samples = predictions.nrows();
        let n_models = predictions.ncols();

        let mut difficulty_sum = 0.0;

        for i in 0..n_samples {
            let error_count = predictions
                .row(i)
                .iter()
                .filter(|&&pred| (pred > 0.5) != (true_labels[i] > 0.5))
                .count() as Float;

            difficulty_sum += error_count / n_models as Float;
        }

        Ok(difficulty_sum / n_samples as Float)
    }

    /// Calculate generalized diversity index
    fn calculate_generalized_diversity(
        &self,
        predictions: &Array2<Float>,
        alpha: Float,
    ) -> Result<Float, Box<dyn std::error::Error>> {
        let n_samples = predictions.nrows();
        let n_models = predictions.ncols();

        let mut diversity_sum = 0.0;

        for i in 0..n_samples {
            let correct_count = predictions
                .row(i)
                .iter()
                .filter(|&&pred| pred > 0.5)
                .count() as Float;

            let p = correct_count / n_models as Float;
            if alpha != 1.0 {
                diversity_sum += (1.0 - p.powf(alpha) - (1.0 - p).powf(alpha))
                    / (2.0_f64.powf(1.0 - alpha) as Float - 1.0);
            } else {
                // Shannon entropy case (alpha = 1)
                if p > 0.0 && p < 1.0 {
                    diversity_sum += -p * p.log2() - (1.0 - p) * (1.0 - p).log2();
                }
            }
        }

        Ok(diversity_sum / n_samples as Float)
    }

    /// Calculate prediction stability across bootstrap samples
    fn calculate_prediction_stability(
        &self,
        predictions: &[Array1<Float>],
    ) -> Result<Float, Box<dyn std::error::Error>> {
        if predictions.len() < 2 {
            return Ok(1.0);
        }

        let n_samples = predictions[0].len();
        let mut stability_sum = 0.0;

        for i in 0..n_samples {
            let sample_predictions: Vec<Float> = predictions.iter().map(|p| p[i]).collect();
            let mean_pred =
                sample_predictions.iter().sum::<Float>() / sample_predictions.len() as Float;
            let variance = sample_predictions
                .iter()
                .map(|&pred| (pred - mean_pred).powi(2))
                .sum::<Float>()
                / sample_predictions.len() as Float;

            stability_sum += 1.0 / (1.0 + variance); // Higher variance = lower stability
        }

        Ok(stability_sum / n_samples as Float)
    }
}

/// Convenience function for ensemble evaluation
pub fn evaluate_ensemble<F>(
    ensemble_predictions: &Array2<Float>,
    true_labels: &Array1<Float>,
    ensemble_weights: Option<&Array1<Float>>,
    model_predictions: Option<&Array2<Float>>,
    evaluation_fn: F,
    config: Option<EnsembleEvaluationConfig>,
) -> Result<EnsembleEvaluationResult, Box<dyn std::error::Error>>
where
    F: Fn(&Array1<Float>, &Array1<Float>) -> Result<Float, Box<dyn std::error::Error>>,
{
    let config = config.unwrap_or_default();
    let mut evaluator = EnsembleEvaluator::new(config);
    evaluator.evaluate(
        ensemble_predictions,
        true_labels,
        ensemble_weights,
        model_predictions,
        evaluation_fn,
    )
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    fn mock_evaluation_function(
        predictions: &Array1<Float>,
        labels: &Array1<Float>,
    ) -> Result<Float, Box<dyn std::error::Error>> {
        let correct = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(&pred, &label)| (pred > 0.5) == (label > 0.5))
            .count();
        Ok(correct as Float / predictions.len() as Float)
    }

    #[test]
    fn test_ensemble_evaluator_creation() {
        let config = EnsembleEvaluationConfig::default();
        let evaluator = EnsembleEvaluator::new(config);
        assert_eq!(evaluator.config.confidence_level, 0.95);
    }

    #[test]
    fn test_out_of_bag_evaluation() {
        let ensemble_predictions = Array2::from_shape_vec(
            (10, 3),
            vec![
                0.1, 0.8, 0.3, 0.9, 0.2, 0.7, 0.4, 0.6, 0.8, 0.1, 0.2, 0.9, 0.1, 0.8, 0.3, 0.6,
                0.5, 0.7, 0.9, 0.2, 0.3, 0.7, 0.2, 0.9, 0.1, 0.8, 0.4, 0.5, 0.6, 0.3,
            ],
        )
        .unwrap();
        let true_labels = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]);

        let config = EnsembleEvaluationConfig {
            strategy: EnsembleEvaluationStrategy::OutOfBag {
                bootstrap_samples: 10,
                confidence_level: 0.95,
            },
            ..Default::default()
        };

        let result = evaluate_ensemble(
            &ensemble_predictions,
            &true_labels,
            None,
            None,
            mock_evaluation_function,
            Some(config),
        )
        .unwrap();

        assert!(result.out_of_bag_scores.is_some());
        let oob_scores = result.out_of_bag_scores.unwrap();
        assert!(oob_scores.oob_score >= 0.0 && oob_scores.oob_score <= 1.0);
    }

    #[test]
    fn test_diversity_evaluation() {
        let ensemble_predictions = Array2::from_shape_vec(
            (10, 3),
            vec![
                0.1, 0.8, 0.3, 0.9, 0.2, 0.7, 0.4, 0.6, 0.8, 0.1, 0.2, 0.9, 0.1, 0.8, 0.3, 0.6,
                0.5, 0.7, 0.9, 0.2, 0.3, 0.7, 0.2, 0.9, 0.1, 0.8, 0.4, 0.5, 0.6, 0.3,
            ],
        )
        .unwrap();
        let true_labels = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
        let model_predictions = ensemble_predictions.clone();

        let config = EnsembleEvaluationConfig {
            strategy: EnsembleEvaluationStrategy::DiversityEvaluation {
                diversity_measures: vec![
                    DiversityMeasure::QStatistic,
                    DiversityMeasure::DisagreementMeasure,
                ],
                diversity_threshold: 0.5,
            },
            ..Default::default()
        };

        let result = evaluate_ensemble(
            &ensemble_predictions,
            &true_labels,
            None,
            Some(&model_predictions),
            mock_evaluation_function,
            Some(config),
        )
        .unwrap();

        assert!(!result.diversity_analysis.diversity_by_measure.is_empty());
        assert!(result.diversity_analysis.overall_diversity >= 0.0);
    }

    #[test]
    fn test_cross_validation_evaluation() {
        let ensemble_predictions = Array2::from_shape_vec(
            (10, 3),
            vec![
                0.1, 0.8, 0.3, 0.9, 0.2, 0.7, 0.4, 0.6, 0.8, 0.1, 0.2, 0.9, 0.1, 0.8, 0.3, 0.6,
                0.5, 0.7, 0.9, 0.2, 0.3, 0.7, 0.2, 0.9, 0.1, 0.8, 0.4, 0.5, 0.6, 0.3,
            ],
        )
        .unwrap();
        let true_labels = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]);

        let config = EnsembleEvaluationConfig {
            strategy: EnsembleEvaluationStrategy::EnsembleCrossValidation {
                cv_strategy: EnsembleCVStrategy::KFoldEnsemble,
                n_folds: 5,
            },
            ..Default::default()
        };

        let result = evaluate_ensemble(
            &ensemble_predictions,
            &true_labels,
            None,
            None,
            mock_evaluation_function,
            Some(config),
        )
        .unwrap();

        assert!(!result
            .ensemble_performance
            .individual_fold_scores
            .is_empty());
        assert!(result.ensemble_performance.mean_performance >= 0.0);
    }
}
