//! Ensemble Learning Framework
//!
//! This module provides a comprehensive ensemble learning framework with multiple
//! composition strategies, voting systems, and advanced fusion techniques for
//! building robust machine learning models.
//!
//! The ensemble system is organized into specialized modules:
//! - **Voting Systems**: Democratic ensemble decision making (VotingClassifier, VotingRegressor)
//! - **Dynamic Selection**: Adaptive ensemble member selection based on competence
//! - **Model Fusion**: Advanced fusion strategies for combining model outputs
//! - **Hierarchical Composition**: Multi-level ensemble architectures
//! - **Ensemble Types**: Common types, traits, and enumerations
//! - **Ensemble Utils**: Utility functions and helper methods
//!
//! # Architecture Overview
//!
//! The ensemble framework supports multiple composition paradigms:
//!
//! ## 1. Democratic Voting
//! - **Hard Voting**: Majority vote from discrete predictions
//! - **Soft Voting**: Weighted average of prediction probabilities
//! - **Weighted Voting**: Member-specific importance weights
//!
//! ## 2. Dynamic Selection
//! - **Competence-Based Selection**: Choose best performers per region
//! - **Adaptive Weighting**: Dynamic weight adjustment based on performance
//! - **Context-Aware Selection**: Selection based on input characteristics
//!
//! ## 3. Model Fusion
//! - **Linear Fusion**: Weighted linear combinations
//! - **Nonlinear Fusion**: Neural network-based fusion
//! - **Gating Networks**: Learned selection mechanisms
//!
//! ## 4. Hierarchical Composition
//! - **Tree-Based Hierarchies**: Decision tree ensemble organization
//! - **Multi-Level Ensembles**: Ensembles of ensembles
//! - **Cascaded Systems**: Sequential ensemble processing
//!
//! # Quick Start Examples
//!
//! ```rust,ignore
//! use sklears_compose::ensemble::*;
//! use scirs2_core::ndarray::array;
//!
//! // Basic voting classifier
//! let voting_clf = VotingClassifier::builder()
//!     .estimator("svm", Box::new(svm_classifier))
//!     .estimator("rf", Box::new(random_forest))
//!     .estimator("nb", Box::new(naive_bayes))
//!     .voting("soft")
//!     .weights(vec![0.3, 0.5, 0.2])
//!     .build();
//!
//! // Dynamic ensemble selector
//! let dynamic_ensemble = DynamicEnsembleSelector::builder()
//!     .estimator("model1", Box::new(model1))
//!     .estimator("model2", Box::new(model2))
//!     .selection_strategy(SelectionStrategy::BestPerformance)
//!     .competence_estimation(CompetenceEstimation::Accuracy)
//!     .build();
//!
//! // Model fusion with gating network
//! let fusion_model = ModelFusion::builder()
//!     .base_model("cnn", Box::new(cnn_model))
//!     .base_model("rnn", Box::new(rnn_model))
//!     .fusion_strategy(FusionStrategy::GatingNetwork)
//!     .build();
//!
//! // Hierarchical composition
//! let hierarchical = HierarchicalComposition::builder()
//!     .strategy(HierarchicalStrategy::TreeBased)
//!     .add_level_models(vec![model_a, model_b, model_c])
//!     .add_level_models(vec![meta_model])
//!     .build();
//! ```
//!
//! # Performance Considerations
//!
//! - **Memory Efficiency**: Streaming prediction for large ensembles
//! - **Computational Optimization**: Parallel prediction execution
//! - **Model Compression**: Ensemble distillation and pruning
//! - **Adaptive Complexity**: Runtime complexity adjustment
//!
//! # Theory and Background
//!
//! Ensemble methods leverage the wisdom of crowds principle, where combining
//! multiple weak learners often produces a stronger predictor than any individual
//! component. Key theoretical foundations include:
//!
//! - **Bias-Variance Decomposition**: Ensemble methods primarily reduce variance
//! - **Diversity-Accuracy Trade-off**: Balance between member diversity and individual accuracy
//! - **Combining Rules**: Mathematical frameworks for prediction aggregation
//! - **Dynamic Selection Theory**: Optimal member selection strategies

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::Result as SklResult,
    prelude::{Predict, SklearsError, Transform},
    traits::{Estimator, Fit, Trained, Untrained},
    types::{Float, FloatBounds},
};
use std::collections::HashMap;

use crate::PipelinePredictor;

// Import specialized ensemble modules
mod voting_systems;
mod dynamic_selection;
mod model_fusion;
mod hierarchical_composition;
mod ensemble_types;
mod ensemble_utils;

// Re-export all public components for unified API access
pub use voting_systems::{
    VotingClassifier, VotingRegressor,
    VotingClassifierTrained, VotingRegressorTrained,
    VotingClassifierBuilder, VotingRegressorBuilder,
};

pub use dynamic_selection::{
    DynamicEnsembleSelector, DynamicEnsembleSelectorTrained,
    DynamicEnsembleSelectorBuilder, SelectionStrategy, CompetenceEstimation,
};

pub use model_fusion::{
    ModelFusion, ModelFusionTrained, ModelFusionBuilder,
    FusionStrategy,
};

pub use hierarchical_composition::{
    HierarchicalComposition, HierarchicalCompositionTrained,
    HierarchicalCompositionBuilder, HierarchicalStrategy,
    HierarchicalNode,
};

pub use ensemble_types::{
    EnsembleMethod, CombiningRule, WeightingScheme,
    PredictionMode, EnsembleMetrics, ValidationStrategy,
};

pub use ensemble_utils::{
    EnsembleValidator, PerformanceAnalyzer, ModelSelector,
    cross_validate_ensemble, optimize_ensemble_weights,
    analyze_ensemble_diversity, generate_ensemble_report,
};

/// Ensemble configuration for global settings
#[derive(Debug, Clone)]
pub struct EnsembleConfig {
    /// Default number of parallel jobs
    pub default_n_jobs: Option<i32>,
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    /// Memory limit for ensemble operations (MB)
    pub memory_limit_mb: Option<usize>,
    /// Cache predictions for performance
    pub cache_predictions: bool,
    /// Validation strategy for ensemble evaluation
    pub validation_strategy: ValidationStrategy,
    /// Enable automatic hyperparameter optimization
    pub auto_optimize: bool,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            default_n_jobs: None,
            enable_monitoring: false,
            memory_limit_mb: None,
            cache_predictions: true,
            validation_strategy: ValidationStrategy::CrossValidation { folds: 5 },
            auto_optimize: false,
        }
    }
}

/// Global ensemble configuration management
static mut ENSEMBLE_CONFIG: EnsembleConfig = EnsembleConfig {
    default_n_jobs: None,
    enable_monitoring: false,
    memory_limit_mb: None,
    cache_predictions: true,
    validation_strategy: ValidationStrategy::HoldOut { test_size: 0.2 },
    auto_optimize: false,
};

/// Set global ensemble configuration
pub fn set_ensemble_config(config: EnsembleConfig) {
    unsafe {
        ENSEMBLE_CONFIG = config;
    }
}

/// Get current ensemble configuration
pub fn get_ensemble_config() -> EnsembleConfig {
    unsafe { ENSEMBLE_CONFIG.clone() }
}

/// Factory functions for creating ensemble methods
pub mod factory {
    use super::*;

    /// Create a simple voting classifier with default settings
    pub fn simple_voting_classifier(
        estimators: Vec<(&str, Box<dyn PipelinePredictor>)>,
        voting: &str,
    ) -> VotingClassifier<Untrained> {
        let mut builder = VotingClassifier::builder().voting(voting);
        for (name, estimator) in estimators {
            builder = builder.estimator(name, estimator);
        }
        builder.build()
    }

    /// Create a simple voting regressor with default settings
    pub fn simple_voting_regressor(
        estimators: Vec<(&str, Box<dyn PipelinePredictor>)>,
    ) -> VotingRegressor<Untrained> {
        let mut builder = VotingRegressor::builder();
        for (name, estimator) in estimators {
            builder = builder.estimator(name, estimator);
        }
        builder.build()
    }

    /// Create an optimized ensemble using automatic model selection
    pub fn auto_ensemble_classifier(
        candidate_models: Vec<Box<dyn PipelinePredictor>>,
        max_ensemble_size: usize,
    ) -> SklResult<DynamicEnsembleSelector<Untrained>> {
        let mut builder = DynamicEnsembleSelector::builder()
            .selection_strategy(SelectionStrategy::GreedySelection)
            .competence_estimation(CompetenceEstimation::Accuracy);

        for (i, model) in candidate_models.into_iter().enumerate().take(max_ensemble_size) {
            builder = builder.estimator(&format!("model_{}", i), model);
        }

        Ok(builder.build())
    }

    /// Create a hierarchical ensemble with automatic architecture
    pub fn auto_hierarchical_ensemble(
        base_models: Vec<Box<dyn PipelinePredictor>>,
        meta_model: Box<dyn PipelinePredictor>,
        strategy: HierarchicalStrategy,
    ) -> HierarchicalComposition<Untrained> {
        HierarchicalComposition::builder()
            .strategy(strategy)
            .add_level_models(base_models)
            .add_level_models(vec![meta_model])
            .build()
    }
}

/// Ensemble evaluation and benchmarking utilities
pub mod evaluation {
    use super::*;

    /// Comprehensive ensemble evaluation metrics
    #[derive(Debug, Clone)]
    pub struct EnsembleEvaluationReport {
        /// Individual member performance
        pub member_performances: HashMap<String, f64>,
        /// Ensemble performance
        pub ensemble_performance: f64,
        /// Performance gain over best individual member
        pub performance_gain: f64,
        /// Diversity metrics
        pub diversity_metrics: DiversityMetrics,
        /// Computational complexity
        pub complexity_metrics: ComplexityMetrics,
    }

    /// Diversity measurement metrics
    #[derive(Debug, Clone)]
    pub struct DiversityMetrics {
        /// Pairwise disagreement between members
        pub disagreement_measure: f64,
        /// Q-statistic for diversity
        pub q_statistic: f64,
        /// Correlation coefficient
        pub correlation_coefficient: f64,
        /// Entropy-based diversity
        pub entropy_diversity: f64,
    }

    /// Computational complexity metrics
    #[derive(Debug, Clone)]
    pub struct ComplexityMetrics {
        /// Training time (seconds)
        pub training_time_s: f64,
        /// Prediction time (milliseconds)
        pub prediction_time_ms: f64,
        /// Memory usage (MB)
        pub memory_usage_mb: f64,
        /// Model size (parameters)
        pub model_size: usize,
    }

    /// Evaluate ensemble performance comprehensively
    pub fn evaluate_ensemble_comprehensive(
        ensemble: &dyn EnsemblePredictor,
        x_test: &ArrayView2<'_, Float>,
        y_test: &ArrayView1<'_, Float>,
    ) -> SklResult<EnsembleEvaluationReport> {
        // This would be implemented with actual evaluation logic
        Ok(EnsembleEvaluationReport {
            member_performances: HashMap::new(),
            ensemble_performance: 0.0,
            performance_gain: 0.0,
            diversity_metrics: DiversityMetrics {
                disagreement_measure: 0.0,
                q_statistic: 0.0,
                correlation_coefficient: 0.0,
                entropy_diversity: 0.0,
            },
            complexity_metrics: ComplexityMetrics {
                training_time_s: 0.0,
                prediction_time_ms: 0.0,
                memory_usage_mb: 0.0,
                model_size: 0,
            },
        })
    }

    /// Compare multiple ensemble methods
    pub fn compare_ensemble_methods(
        methods: Vec<Box<dyn EnsemblePredictor>>,
        x_test: &ArrayView2<'_, Float>,
        y_test: &ArrayView1<'_, Float>,
    ) -> SklResult<Vec<EnsembleEvaluationReport>> {
        let mut reports = Vec::new();
        for method in methods.iter() {
            let report = evaluate_ensemble_comprehensive(method.as_ref(), x_test, y_test)?;
            reports.push(report);
        }
        Ok(reports)
    }
}

/// Common trait for all ensemble predictors
pub trait EnsemblePredictor: Predict<ArrayView2<'_, Float>, Array1<Float>> {
    /// Get ensemble member names
    fn member_names(&self) -> Vec<String>;

    /// Get ensemble size
    fn ensemble_size(&self) -> usize;

    /// Get individual member predictions
    fn member_predictions(&self, x: &ArrayView2<'_, Float>) -> SklResult<Vec<Array1<Float>>>;

    /// Get prediction confidence/uncertainty
    fn prediction_confidence(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<Float>>;

    /// Analyze ensemble diversity
    fn analyze_diversity(&self, x: &ArrayView2<'_, Float>) -> SklResult<evaluation::DiversityMetrics>;
}

/// Ensemble optimization strategies
pub mod optimization {
    use super::*;

    /// Optimization objectives for ensemble tuning
    #[derive(Debug, Clone)]
    pub enum OptimizationObjective {
        /// Maximize prediction accuracy
        Accuracy,
        /// Maximize AUC-ROC
        AucRoc,
        /// Minimize prediction time
        Speed,
        /// Minimize memory usage
        Memory,
        /// Balance accuracy and speed
        AccuracySpeed { accuracy_weight: f64, speed_weight: f64 },
        /// Custom objective function
        Custom { objective_fn: String },
    }

    /// Hyperparameter optimization for ensembles
    pub struct EnsembleOptimizer {
        objective: OptimizationObjective,
        max_iterations: usize,
        tolerance: f64,
    }

    impl EnsembleOptimizer {
        /// Create new ensemble optimizer
        pub fn new(objective: OptimizationObjective) -> Self {
            Self {
                objective,
                max_iterations: 100,
                tolerance: 1e-6,
            }
        }

        /// Optimize ensemble weights
        pub fn optimize_weights(
            &self,
            ensemble: &mut dyn EnsemblePredictor,
            x_val: &ArrayView2<'_, Float>,
            y_val: &ArrayView1<'_, Float>,
        ) -> SklResult<Vec<f64>> {
            // Implementation would use optimization algorithms
            // like gradient descent, genetic algorithms, etc.
            Ok(vec![1.0; ensemble.ensemble_size()])
        }

        /// Optimize ensemble architecture
        pub fn optimize_architecture(
            &self,
            candidate_models: Vec<Box<dyn PipelinePredictor>>,
            x_val: &ArrayView2<'_, Float>,
            y_val: &ArrayView1<'_, Float>,
        ) -> SklResult<Vec<usize>> {
            // Implementation would use architecture search
            Ok(vec![0, 1, 2]) // Return indices of selected models
        }
    }
}

/// Advanced ensemble techniques
pub mod advanced {
    use super::*;

    /// Simple distilled model for ensemble compression
    #[derive(Debug, Clone)]
    pub struct DistilledModel {
        /// Linear weights learned from ensemble
        weights: Array2<Float>,
        /// Bias terms
        bias: Array1<Float>,
        /// Input dimension
        n_features: usize,
        /// Output dimension
        n_outputs: usize,
    }

    impl DistilledModel {
        /// Create new distilled model
        pub fn new(n_features: usize, n_outputs: usize) -> Self {
            Self {
                weights: Array2::zeros((n_features, n_outputs)),
                bias: Array1::zeros(n_outputs),
                n_features,
                n_outputs,
            }
        }

        /// Train via simple gradient descent on soft targets
        fn fit_from_ensemble(
            &mut self,
            x: &ArrayView2<'_, Float>,
            soft_targets: &Array2<Float>,
            learning_rate: Float,
            n_epochs: usize,
        ) {
            for _epoch in 0..n_epochs {
                // Forward pass: prediction = x @ weights + bias
                let predictions = x.dot(&self.weights) + &self.bias;

                // Compute gradient of MSE loss
                let error = &predictions - soft_targets;

                // Gradient descent update
                // dL/dW = X^T @ error
                let grad_weights = x.t().dot(&error) / (x.nrows() as Float);
                // dL/db = sum(error, axis=0)
                let grad_bias = error.sum_axis(Axis(0)) / (x.nrows() as Float);

                // Update parameters
                self.weights = &self.weights - &(grad_weights * learning_rate);
                self.bias = &self.bias - &(grad_bias * learning_rate);
            }
        }
    }

    impl PipelinePredictor for DistilledModel {
        fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<Float>> {
            // Linear prediction: y = x @ weights + bias
            // For regression/single output, return first column or mean
            let predictions = x.dot(&self.weights) + &self.bias;

            if self.n_outputs == 1 {
                Ok(predictions.column(0).to_owned())
            } else {
                // For multi-output, return mean or first output
                Ok(predictions.column(0).to_owned())
            }
        }
    }

    pub struct EnsembleDistiller {
        temperature: f64,
        student_architecture: String,
        distillation_weight: f64,
    }

    impl EnsembleDistiller {
        /// Create new ensemble distiller
        pub fn new() -> Self {
            Self {
                temperature: 3.0,
                student_architecture: "mlp".to_string(),
                distillation_weight: 0.7,
            }
        }

        /// Distill ensemble into single model using knowledge distillation
        ///
        /// This implements a simplified knowledge distillation algorithm:
        /// 1. Generate soft targets from the ensemble at elevated temperature
        /// 2. Train a simpler student model to match these soft targets
        /// 3. Optionally blend with hard targets from ground truth
        ///
        /// # Arguments
        /// * `ensemble` - The ensemble model to distill
        /// * `x_train` - Training features
        /// * `y_train` - Training labels (ground truth)
        ///
        /// # Returns
        /// A distilled single model that approximates the ensemble
        pub fn distill(
            &self,
            ensemble: &dyn EnsemblePredictor,
            x_train: &ArrayView2<'_, Float>,
            y_train: &ArrayView1<'_, Float>,
        ) -> SklResult<Box<dyn PipelinePredictor>> {
            let n_features = x_train.ncols();
            let n_samples = x_train.nrows();

            // Step 1: Generate soft targets from ensemble
            // Get ensemble predictions (these are the teacher's knowledge)
            let ensemble_predictions = ensemble.predict(x_train)?;

            // Convert to 2D array for easier manipulation
            let soft_targets = if ensemble_predictions.ndim() == 1 {
                // Single output: reshape to column vector
                Array2::from_shape_vec(
                    (n_samples, 1),
                    ensemble_predictions.to_vec()
                ).map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))?
            } else {
                // Already 2D
                ensemble_predictions.into_dimensionality::<scirs2_core::ndarray::Ix2>()
                    .map_err(|e| SklearsError::InvalidInput(format!("Dimensionality error: {}", e)))?
            };

            let n_outputs = soft_targets.ncols();

            // Step 2: Create and train student model
            let mut student = DistilledModel::new(n_features, n_outputs);

            // Blend soft targets with hard targets using distillation weight
            let mut blended_targets = soft_targets.clone();
            for i in 0..n_samples {
                for j in 0..n_outputs {
                    let soft = soft_targets[[i, j]];
                    let hard = y_train[i];
                    blended_targets[[i, j]] = self.distillation_weight * soft
                        + (1.0 - self.distillation_weight) * hard;
                }
            }

            // Step 3: Train student model via gradient descent
            let learning_rate = 0.01;
            let n_epochs = 100;
            student.fit_from_ensemble(
                x_train,
                &blended_targets,
                learning_rate,
                n_epochs,
            );

            // Return the distilled model
            Ok(Box::new(student))
        }
    }

    /// Online ensemble learning for streaming data
    pub struct OnlineEnsemble {
        window_size: usize,
        adaptation_rate: f64,
        drift_detection: bool,
    }

    impl OnlineEnsemble {
        /// Create new online ensemble
        pub fn new(window_size: usize) -> Self {
            Self {
                window_size,
                adaptation_rate: 0.01,
                drift_detection: true,
            }
        }

        /// Update ensemble with new data
        pub fn partial_fit(
            &mut self,
            x: &ArrayView2<'_, Float>,
            y: &ArrayView1<'_, Float>,
        ) -> SklResult<()> {
            // Implementation would update ensemble incrementally
            Ok(())
        }

        /// Detect concept drift
        pub fn detect_drift(&self, x: &ArrayView2<'_, Float>) -> bool {
            // Implementation would use drift detection algorithms
            false
        }
    }
}

/// Legacy compatibility layer
pub mod legacy {
    //! Legacy API compatibility for smooth migration

    use super::*;

    #[deprecated(note = "Use VotingClassifier from voting_systems module")]
    pub type LegacyVotingClassifier = VotingClassifier<Untrained>;

    #[deprecated(note = "Use VotingRegressor from voting_systems module")]
    pub type LegacyVotingRegressor = VotingRegressor<Untrained>;
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensemble_config() {
        let config = EnsembleConfig::default();
        assert!(!config.enable_monitoring);
        assert!(config.cache_predictions);

        set_ensemble_config(EnsembleConfig {
            enable_monitoring: true,
            ..config
        });

        let retrieved_config = get_ensemble_config();
        assert!(retrieved_config.enable_monitoring);
    }

    #[test]
    fn test_factory_functions() {
        // Test would create mock predictors and verify factory functions work
        // This is a placeholder for actual test implementation
    }
}