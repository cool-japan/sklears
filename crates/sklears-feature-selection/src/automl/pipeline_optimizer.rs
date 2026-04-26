//! Pipeline Optimization Module for AutoML Feature Selection
//!
//! Creates optimal feature selection pipelines by evaluating and combining methods.
//! All implementations follow the SciRS2 policy using scirs2-core for numerical computations.

use scirs2_core::ndarray::{ArrayView1, ArrayView2};

use super::automl_core::{AutoMLError, AutoMLMethod, DataCharacteristics};
use super::hyperparameter_optimizer::{MethodConfig, OptimizedMethod, TrainedMethod};
use sklears_core::error::Result as SklResult;
use std::collections::HashMap;

type Result<T> = SklResult<T>;

/// Pipeline optimizer for creating optimal feature selection pipelines
#[derive(Debug, Clone)]
pub struct PipelineOptimizer;

impl PipelineOptimizer {
    /// new
    pub fn new() -> Self {
        Self
    }

    /// create_optimal_pipeline
    pub fn create_optimal_pipeline(
        &self,
        methods: &[OptimizedMethod],
        performances: &[MethodPerformance],
        target_n_features: Option<usize>,
    ) -> Result<OptimalPipeline> {
        if methods.is_empty() || performances.is_empty() {
            return Err(AutoMLError::OptimizationFailed.into());
        }

        // Find best performing method
        let best_idx = performances
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.score
                    .partial_cmp(&b.score)
                    .expect("operation should succeed")
            })
            .map(|(idx, _)| idx)
            .ok_or(AutoMLError::OptimizationFailed)?;

        let best_method = methods[best_idx].clone();
        let best_performance = performances[best_idx].clone();

        // Create pipeline configuration
        let pipeline_config = PipelineConfigResult {
            method: best_method.method_type.clone(),
            hyperparameters: best_method.config.clone(),
            expected_performance: best_performance.score,
            feature_stability: best_performance.feature_stability,
            computational_cost: best_method.estimated_cost,
        };

        Ok(OptimalPipeline {
            method: best_method,
            performance: best_performance,
            config: pipeline_config,
            target_n_features,
        })
    }
}

impl Default for PipelineOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for the automated pipeline
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// validation_strategy
    pub validation_strategy: ValidationStrategy,
    /// max_optimization_iterations
    pub max_optimization_iterations: usize,
    /// target_performance
    pub target_performance: Option<f64>,
    /// prefer_interpretability
    pub prefer_interpretability: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            validation_strategy: ValidationStrategy::CrossValidation { folds: 5 },
            max_optimization_iterations: 20,
            target_performance: None,
            prefer_interpretability: false,
        }
    }
}

/// Validation strategy for evaluating methods
#[derive(Debug, Clone)]
pub enum ValidationStrategy {
    /// CrossValidation
    CrossValidation {
        /// folds
        folds: usize,
    },
    /// HoldOut
    HoldOut {
        /// test_size
        test_size: f64,
    },
    /// TimeSeriesSplit
    TimeSeriesSplit {
        /// n_splits
        n_splits: usize,
    },
}

/// Performance metrics for a method
#[derive(Debug, Clone)]
pub struct MethodPerformance {
    /// method_name
    pub method_name: String,
    /// score
    pub score: f64,
    /// score_std
    pub score_std: f64,
    /// scores
    pub scores: Vec<f64>,
    /// feature_stability
    pub feature_stability: f64,
    /// computational_cost
    pub computational_cost: f64,
    /// method_config
    pub method_config: MethodConfig,
}

/// Optimal pipeline result
#[derive(Debug, Clone)]
pub struct OptimalPipeline {
    /// method
    pub method: OptimizedMethod,
    /// performance
    pub performance: MethodPerformance,
    /// config
    pub config: PipelineConfigResult,
    /// target_n_features
    pub target_n_features: Option<usize>,
}

impl OptimalPipeline {
    /// fit
    pub fn fit(self, X: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<TrainedOptimalPipeline> {
        let trained_method = self.method.fit(X, y)?;

        Ok(TrainedOptimalPipeline {
            trained_method,
            performance: self.performance,
            config: self.config,
        })
    }

    /// get_config
    pub fn get_config(&self) -> PipelineConfigResult {
        self.config.clone()
    }
}

/// Trained optimal pipeline
#[derive(Debug, Clone)]
pub struct TrainedOptimalPipeline {
    /// trained_method
    pub trained_method: TrainedMethod,
    /// performance
    pub performance: MethodPerformance,
    /// config
    pub config: PipelineConfigResult,
}

impl TrainedOptimalPipeline {
    /// transform_indices
    pub fn transform_indices(&self) -> Result<Vec<usize>> {
        self.trained_method.transform_indices()
    }

    /// get_feature_importances
    pub fn get_feature_importances(&self) -> Result<Vec<f64>> {
        Ok(self.trained_method.feature_importances.clone())
    }

    /// method_info
    pub fn method_info(&self) -> MethodInfo {
        MethodInfo {
            name: format!("{:?}", self.trained_method.method_type),
            parameters: self.build_parameter_map(),
        }
    }

    fn build_parameter_map(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        match &self.trained_method.config {
            MethodConfig::Univariate { k } => {
                params.insert("k".to_string(), k.to_string());
            }
            MethodConfig::Correlation { threshold } => {
                params.insert("threshold".to_string(), threshold.to_string());
            }
            MethodConfig::Tree {
                n_estimators,
                max_depth,
            } => {
                params.insert("n_estimators".to_string(), n_estimators.to_string());
                params.insert("max_depth".to_string(), max_depth.to_string());
            }
            MethodConfig::Lasso { alpha } => {
                params.insert("alpha".to_string(), alpha.to_string());
            }
            MethodConfig::Wrapper { cv_folds, scoring } => {
                params.insert("cv_folds".to_string(), cv_folds.to_string());
                params.insert("scoring".to_string(), scoring.clone());
            }
            MethodConfig::Ensemble {
                n_methods,
                aggregation,
            } => {
                params.insert("n_methods".to_string(), n_methods.to_string());
                params.insert("aggregation".to_string(), aggregation.clone());
            }
            MethodConfig::Hybrid {
                stage1_method,
                stage2_method,
                stage1_features,
            } => {
                params.insert("stage1_method".to_string(), stage1_method.clone());
                params.insert("stage2_method".to_string(), stage2_method.clone());
                params.insert("stage1_features".to_string(), stage1_features.to_string());
            }
            MethodConfig::NeuralArchitectureSearch {
                max_epochs,
                population_size,
                mutation_rate,
                early_stopping_patience,
            } => {
                params.insert("max_epochs".to_string(), max_epochs.to_string());
                params.insert("population_size".to_string(), population_size.to_string());
                params.insert("mutation_rate".to_string(), mutation_rate.to_string());
                params.insert(
                    "early_stopping_patience".to_string(),
                    early_stopping_patience.to_string(),
                );
            }
            MethodConfig::TransferLearning {
                source_domain,
                adaptation_method,
                fine_tuning_epochs,
                transfer_ratio,
            } => {
                params.insert("source_domain".to_string(), source_domain.clone());
                params.insert("adaptation_method".to_string(), adaptation_method.clone());
                params.insert(
                    "fine_tuning_epochs".to_string(),
                    fine_tuning_epochs.to_string(),
                );
                params.insert("transfer_ratio".to_string(), transfer_ratio.to_string());
            }
            MethodConfig::MetaLearningEnsemble {
                base_methods,
                meta_learner,
                adaptation_strategy,
                ensemble_size,
            } => {
                params.insert("base_methods".to_string(), base_methods.join(","));
                params.insert("meta_learner".to_string(), meta_learner.clone());
                params.insert(
                    "adaptation_strategy".to_string(),
                    adaptation_strategy.clone(),
                );
                params.insert("ensemble_size".to_string(), ensemble_size.to_string());
            }
        }
        params
    }
}

/// Method information
#[derive(Debug, Clone)]
pub struct MethodInfo {
    /// name
    pub name: String,
    /// parameters
    pub parameters: HashMap<String, String>,
}

/// Pipeline configuration result
#[derive(Debug, Clone)]
pub struct PipelineConfigResult {
    /// method
    pub method: AutoMLMethod,
    /// hyperparameters
    pub hyperparameters: MethodConfig,
    /// expected_performance
    pub expected_performance: f64,
    /// feature_stability
    pub feature_stability: f64,
    /// computational_cost
    pub computational_cost: f64,
}

/// Complete AutoML results
#[derive(Debug, Clone)]
pub struct AutoMLResults {
    /// selected_features
    pub selected_features: Vec<usize>,
    /// feature_importances
    pub feature_importances: Vec<f64>,
    /// best_method
    pub best_method: MethodInfo,
    /// data_characteristics
    pub data_characteristics: DataCharacteristics,
    /// method_performances
    pub method_performances: Vec<MethodPerformance>,
    /// pipeline_config
    pub pipeline_config: PipelineConfigResult,
    /// validation_scores
    pub validation_scores: Vec<f64>,
    /// recommendation
    pub recommendation: String,
}

impl AutoMLResults {
    /// Generate comprehensive report
    pub fn report(&self) -> String {
        let mut report = String::new();

        report.push_str(
            "╔═══════════════════════════════════════════════════════════════════════════╗\n",
        );
        report.push_str(
            "║                    AutoML Feature Selection Report                        ║\n",
        );
        report.push_str(
            "╚═══════════════════════════════════════════════════════════════════════════╝\n\n",
        );

        // Data summary
        report.push_str("=== Data Summary ===\n");
        report.push_str(&format!(
            "Original features: {}\n",
            self.data_characteristics.n_features
        ));
        report.push_str(&format!(
            "Selected features: {}\n",
            self.selected_features.len()
        ));
        report.push_str(&format!(
            "Reduction ratio: {:.2}%\n",
            (1.0 - self.selected_features.len() as f64
                / self.data_characteristics.n_features as f64)
                * 100.0
        ));

        // Method summary
        report.push_str("\n=== Best Method ===\n");
        report.push_str(&format!("Method: {}\n", self.best_method.name));
        report.push_str(&format!(
            "Expected performance: {:.3}\n",
            self.pipeline_config.expected_performance
        ));
        report.push_str(&format!(
            "Feature stability: {:.3}\n",
            self.pipeline_config.feature_stability
        ));

        // Recommendations
        report.push_str("\n=== Recommendations ===\n");
        report.push_str(&self.recommendation);

        report
    }

    /// Generate a comprehensive summary of the AutoML process
    pub fn generate_summary(&self) -> AutoMLSummary {
        AutoMLSummary {
            selected_feature_count: self.selected_features.len(),
            best_method_name: self.best_method.name.clone(),
            best_score: self.validation_scores.iter().fold(0.0f64, |a, &b| a.max(b)),
            feature_reduction_ratio: self.selected_features.len() as f64
                / self.data_characteristics.n_features as f64,
            recommendation_summary: self.recommendation.clone(),
        }
    }
}

/// Summary of AutoML results
#[derive(Debug, Clone)]
pub struct AutoMLSummary {
    /// selected_feature_count
    pub selected_feature_count: usize,
    /// best_method_name
    pub best_method_name: String,
    /// best_score
    pub best_score: f64,
    /// feature_reduction_ratio
    pub feature_reduction_ratio: f64,
    /// recommendation_summary
    pub recommendation_summary: String,
}
