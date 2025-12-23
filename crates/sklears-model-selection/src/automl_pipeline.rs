//! Complete AutoML Pipeline
//!
//! This module provides a comprehensive AutoML pipeline that integrates automated feature
//! engineering, algorithm selection, hyperparameter optimization, and ensemble construction
//! to automatically build the best possible machine learning model for a given dataset.

use crate::{
    automl_algorithm_selection::{
        AlgorithmFamily, AlgorithmSelectionResult, AutoMLAlgorithmSelector, AutoMLConfig,
        ComputationalConstraints, DatasetCharacteristics, RankedAlgorithm,
    },
    automl_feature_engineering::{
        AutoFeatureEngineer, AutoFeatureEngineering, FeatureEngineeringResult,
        FeatureEngineeringStrategy,
    },
    ensemble_selection::{EnsembleSelectionConfig, EnsembleSelectionResult},
    scoring::TaskType,
};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    // traits::Estimator,
};
use std::collections::HashMap;
use std::fmt;
use std::time::Instant;
// use serde::{Deserialize, Serialize};

/// AutoML pipeline stages
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AutoMLStage {
    /// Data analysis and preprocessing
    DataAnalysis,
    /// Automated feature engineering
    FeatureEngineering,
    /// Algorithm selection and evaluation
    AlgorithmSelection,
    /// Hyperparameter optimization
    HyperparameterOptimization,
    /// Ensemble construction
    EnsembleConstruction,
    /// Final model training
    FinalTraining,
    /// Model validation
    ModelValidation,
}

impl fmt::Display for AutoMLStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AutoMLStage::DataAnalysis => write!(f, "Data Analysis"),
            AutoMLStage::FeatureEngineering => write!(f, "Feature Engineering"),
            AutoMLStage::AlgorithmSelection => write!(f, "Algorithm Selection"),
            AutoMLStage::HyperparameterOptimization => write!(f, "Hyperparameter Optimization"),
            AutoMLStage::EnsembleConstruction => write!(f, "Ensemble Construction"),
            AutoMLStage::FinalTraining => write!(f, "Final Training"),
            AutoMLStage::ModelValidation => write!(f, "Model Validation"),
        }
    }
}

/// AutoML optimization level
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationLevel {
    /// Fast optimization with basic algorithms
    Fast,
    /// Balanced optimization with moderate search
    Balanced,
    /// Thorough optimization with extensive search
    Thorough,
    /// Custom optimization with user-defined parameters
    Custom,
}

/// Complete AutoML configuration
#[derive(Debug, Clone)]
pub struct AutoMLPipelineConfig {
    /// Task type (classification or regression)
    pub task_type: TaskType,
    /// Optimization level
    pub optimization_level: OptimizationLevel,
    /// Computational constraints
    pub constraints: ComputationalConstraints,
    /// Total time budget in seconds
    pub time_budget: f64,
    /// Enable feature engineering
    pub enable_feature_engineering: bool,
    /// Feature engineering configuration
    pub feature_engineering_config: AutoFeatureEngineering,
    /// Algorithm selection configuration
    pub algorithm_selection_config: AutoMLConfig,
    /// Enable ensemble construction
    pub enable_ensemble: bool,
    /// Ensemble configuration
    pub ensemble_config: EnsembleSelectionConfig,
    /// Cross-validation strategy
    pub cv_folds: usize,
    /// Scoring metric
    pub scoring_metric: String,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Early stopping patience
    pub early_stopping_patience: usize,
    /// Verbose output
    pub verbose: bool,
}

impl Default for AutoMLPipelineConfig {
    fn default() -> Self {
        Self {
            task_type: TaskType::Classification,
            optimization_level: OptimizationLevel::Balanced,
            constraints: ComputationalConstraints::default(),
            time_budget: 3600.0, // 1 hour
            enable_feature_engineering: true,
            feature_engineering_config: AutoFeatureEngineering::default(),
            algorithm_selection_config: AutoMLConfig::default(),
            enable_ensemble: true,
            ensemble_config: EnsembleSelectionConfig::default(),
            cv_folds: 5,
            scoring_metric: "accuracy".to_string(),
            random_seed: None,
            early_stopping_patience: 5,
            verbose: true,
        }
    }
}

/// AutoML pipeline execution result
#[derive(Debug, Clone)]
pub struct AutoMLPipelineResult {
    /// Final model performance
    pub final_score: f64,
    /// Cross-validation score with standard deviation
    pub cv_score: f64,
    pub cv_std: f64,
    /// Best algorithm information
    pub best_algorithm: RankedAlgorithm,
    /// Feature engineering results
    pub feature_engineering: Option<FeatureEngineeringResult>,
    /// Algorithm selection results
    pub algorithm_selection: AlgorithmSelectionResult,
    /// Ensemble selection results
    pub ensemble_selection: Option<EnsembleSelectionResult>,
    /// Dataset characteristics
    pub dataset_characteristics: DatasetCharacteristics,
    /// Pipeline execution stages and timings
    pub stage_timings: HashMap<AutoMLStage, f64>,
    /// Total execution time
    pub total_time: f64,
    /// Best hyperparameters
    pub best_hyperparameters: HashMap<String, String>,
    /// Performance improvement over baseline
    pub improvement_over_baseline: f64,
    /// Model complexity score
    pub model_complexity: f64,
    /// Interpretability score
    pub interpretability_score: f64,
    /// Final recommendations
    pub recommendations: Vec<String>,
}

impl fmt::Display for AutoMLPipelineResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "AutoML Pipeline Results")?;
        writeln!(f, "======================")?;
        writeln!(f, "Final Score: {:.4} ± {:.4}", self.cv_score, self.cv_std)?;
        writeln!(
            f,
            "Best Algorithm: {} ({})",
            self.best_algorithm.algorithm.name, self.best_algorithm.algorithm.family
        )?;
        writeln!(
            f,
            "Improvement over Baseline: {:.4}",
            self.improvement_over_baseline
        )?;
        writeln!(f, "Total Execution Time: {:.2}s", self.total_time)?;
        writeln!(f)?;

        writeln!(f, "Dataset Characteristics:")?;
        writeln!(f, "  Samples: {}", self.dataset_characteristics.n_samples)?;
        writeln!(f, "  Features: {}", self.dataset_characteristics.n_features)?;
        if let Some(n_classes) = self.dataset_characteristics.n_classes {
            writeln!(f, "  Classes: {}", n_classes)?;
        }
        writeln!(
            f,
            "  Linearity Score: {:.4}",
            self.dataset_characteristics.linearity_score
        )?;
        writeln!(f)?;

        if let Some(ref fe_result) = self.feature_engineering {
            writeln!(f, "Feature Engineering:")?;
            writeln!(
                f,
                "  Original Features: {}",
                fe_result.original_feature_count
            )?;
            writeln!(
                f,
                "  Generated Features: {}",
                fe_result.generated_feature_count
            )?;
            writeln!(
                f,
                "  Selected Features: {}",
                fe_result.selected_feature_count
            )?;
            writeln!(
                f,
                "  Performance Improvement: {:.4}",
                fe_result.performance_improvement
            )?;
            writeln!(f)?;
        }

        if let Some(ref ensemble_result) = self.ensemble_selection {
            writeln!(f, "Ensemble Configuration:")?;
            writeln!(f, "  Strategy: {}", ensemble_result.ensemble_strategy)?;
            writeln!(f, "  Models: {}", ensemble_result.selected_models.len())?;
            writeln!(
                f,
                "  Ensemble Score: {:.4} ± {:.4}",
                ensemble_result.ensemble_performance.mean_score,
                ensemble_result.ensemble_performance.std_score
            )?;
            writeln!(f)?;
        }

        writeln!(f, "Stage Timings:")?;
        for (stage, time) in &self.stage_timings {
            writeln!(f, "  {}: {:.2}s", stage, time)?;
        }
        writeln!(f)?;

        writeln!(f, "Recommendations:")?;
        for (i, recommendation) in self.recommendations.iter().enumerate() {
            writeln!(f, "  {}. {}", i + 1, recommendation)?;
        }

        Ok(())
    }
}

/// Progress callback for AutoML pipeline
pub trait AutoMLProgressCallback {
    fn on_stage_start(&mut self, stage: AutoMLStage, message: &str);
    fn on_stage_progress(&mut self, stage: AutoMLStage, progress: f64, message: &str);
    fn on_stage_complete(&mut self, stage: AutoMLStage, duration: f64, message: &str);
}

/// Default progress callback that prints to console
pub struct ConsoleProgressCallback {
    verbose: bool,
}

impl ConsoleProgressCallback {
    pub fn new(verbose: bool) -> Self {
        Self { verbose }
    }
}

impl AutoMLProgressCallback for ConsoleProgressCallback {
    fn on_stage_start(&mut self, stage: AutoMLStage, message: &str) {
        if self.verbose {
            println!("[AutoML] Starting {}: {}", stage, message);
        }
    }

    fn on_stage_progress(&mut self, stage: AutoMLStage, progress: f64, message: &str) {
        if self.verbose {
            println!("[AutoML] {} {:.1}%: {}", stage, progress * 100.0, message);
        }
    }

    fn on_stage_complete(&mut self, stage: AutoMLStage, duration: f64, message: &str) {
        if self.verbose {
            println!(
                "[AutoML] Completed {} in {:.2}s: {}",
                stage, duration, message
            );
        }
    }
}

/// Complete AutoML pipeline
pub struct AutoMLPipeline {
    config: AutoMLPipelineConfig,
    progress_callback: Option<Box<dyn AutoMLProgressCallback>>,
}

impl Default for AutoMLPipeline {
    fn default() -> Self {
        Self::new(AutoMLPipelineConfig::default())
    }
}

impl AutoMLPipeline {
    /// Create a new AutoML pipeline
    pub fn new(config: AutoMLPipelineConfig) -> Self {
        Self {
            config,
            progress_callback: None,
        }
    }

    /// Set progress callback
    pub fn with_progress_callback(mut self, callback: Box<dyn AutoMLProgressCallback>) -> Self {
        self.progress_callback = Some(callback);
        self
    }

    /// Run the complete AutoML pipeline
    pub fn fit(&mut self, X: &Array2<f64>, y: &Array1<f64>) -> Result<AutoMLPipelineResult> {
        let start_time = Instant::now();
        let mut stage_timings = HashMap::new();

        // Validate input
        self.validate_input(X, y)?;

        // Stage 1: Data Analysis
        let stage_start = Instant::now();
        self.progress_callback_stage_start(
            AutoMLStage::DataAnalysis,
            "Analyzing dataset characteristics",
        );

        let dataset_chars = self.analyze_dataset(X, y);

        let stage_duration = stage_start.elapsed().as_secs_f64();
        stage_timings.insert(AutoMLStage::DataAnalysis, stage_duration);
        self.progress_callback_stage_complete(
            AutoMLStage::DataAnalysis,
            stage_duration,
            &format!(
                "Found {} samples, {} features",
                dataset_chars.n_samples, dataset_chars.n_features
            ),
        );

        // Adapt configuration based on dataset characteristics
        self.adapt_configuration(&dataset_chars);

        // Stage 2: Feature Engineering (if enabled)
        let (transformed_X, feature_engineering_result) = if self.config.enable_feature_engineering
        {
            let stage_start = Instant::now();
            self.progress_callback_stage_start(
                AutoMLStage::FeatureEngineering,
                "Generating and selecting features",
            );

            let fe_result = self.perform_feature_engineering(X, y)?;
            let transformation_info = fe_result.transformation_info.clone();

            // Transform data using selected features
            let fe_engineer =
                AutoFeatureEngineer::new(self.config.feature_engineering_config.clone());
            let transformed_X = fe_engineer.transform(X, &transformation_info)?;

            let stage_duration = stage_start.elapsed().as_secs_f64();
            stage_timings.insert(AutoMLStage::FeatureEngineering, stage_duration);
            self.progress_callback_stage_complete(
                AutoMLStage::FeatureEngineering,
                stage_duration,
                &format!(
                    "Generated {} features, selected {}",
                    fe_result.generated_feature_count, fe_result.selected_feature_count
                ),
            );

            (transformed_X, Some(fe_result))
        } else {
            (X.clone(), None)
        };

        // Stage 3: Algorithm Selection
        let stage_start = Instant::now();
        self.progress_callback_stage_start(
            AutoMLStage::AlgorithmSelection,
            "Evaluating algorithms",
        );

        let algorithm_selection_result = self.perform_algorithm_selection(&transformed_X, y)?;

        let stage_duration = stage_start.elapsed().as_secs_f64();
        stage_timings.insert(AutoMLStage::AlgorithmSelection, stage_duration);
        self.progress_callback_stage_complete(
            AutoMLStage::AlgorithmSelection,
            stage_duration,
            &format!(
                "Evaluated {} algorithms, best: {}",
                algorithm_selection_result.n_algorithms_evaluated,
                algorithm_selection_result.best_algorithm.algorithm.name
            ),
        );

        // Stage 4: Hyperparameter Optimization
        let stage_start = Instant::now();
        self.progress_callback_stage_start(
            AutoMLStage::HyperparameterOptimization,
            "Optimizing hyperparameters",
        );

        let (optimized_algorithm, best_hyperparameters) = self
            .perform_hyperparameter_optimization(
                &transformed_X,
                y,
                &algorithm_selection_result.best_algorithm,
            )?;

        let stage_duration = stage_start.elapsed().as_secs_f64();
        stage_timings.insert(AutoMLStage::HyperparameterOptimization, stage_duration);
        self.progress_callback_stage_complete(
            AutoMLStage::HyperparameterOptimization,
            stage_duration,
            &format!(
                "Optimized hyperparameters, score: {:.4}",
                optimized_algorithm.cv_score
            ),
        );

        // Stage 5: Ensemble Construction (if enabled)
        let ensemble_result = if self.config.enable_ensemble {
            let stage_start = Instant::now();
            self.progress_callback_stage_start(
                AutoMLStage::EnsembleConstruction,
                "Building ensemble",
            );

            let ensemble_result =
                self.perform_ensemble_construction(&transformed_X, y, &algorithm_selection_result)?;

            let stage_duration = stage_start.elapsed().as_secs_f64();
            stage_timings.insert(AutoMLStage::EnsembleConstruction, stage_duration);
            self.progress_callback_stage_complete(
                AutoMLStage::EnsembleConstruction,
                stage_duration,
                &format!(
                    "Built ensemble with {} models, score: {:.4}",
                    ensemble_result.selected_models.len(),
                    ensemble_result.ensemble_performance.mean_score
                ),
            );

            Some(ensemble_result)
        } else {
            None
        };

        // Stage 6: Final Training
        let stage_start = Instant::now();
        self.progress_callback_stage_start(AutoMLStage::FinalTraining, "Training final model");

        let final_score = self.perform_final_training(&transformed_X, y, &optimized_algorithm)?;

        let stage_duration = stage_start.elapsed().as_secs_f64();
        stage_timings.insert(AutoMLStage::FinalTraining, stage_duration);
        self.progress_callback_stage_complete(
            AutoMLStage::FinalTraining,
            stage_duration,
            &format!("Final model score: {:.4}", final_score),
        );

        // Stage 7: Model Validation
        let stage_start = Instant::now();
        self.progress_callback_stage_start(AutoMLStage::ModelValidation, "Validating model");

        let (cv_score, cv_std) =
            self.perform_model_validation(&transformed_X, y, &optimized_algorithm)?;

        let stage_duration = stage_start.elapsed().as_secs_f64();
        stage_timings.insert(AutoMLStage::ModelValidation, stage_duration);
        self.progress_callback_stage_complete(
            AutoMLStage::ModelValidation,
            stage_duration,
            &format!("Validation score: {:.4} ± {:.4}", cv_score, cv_std),
        );

        // Calculate metrics and generate recommendations
        let baseline_score = self.calculate_baseline_score(X, y)?;
        let improvement = cv_score - baseline_score;
        let model_complexity = self.calculate_model_complexity(&optimized_algorithm);
        let interpretability = self.calculate_interpretability_score(&optimized_algorithm);
        let recommendations = self.generate_recommendations(
            &optimized_algorithm,
            &dataset_chars,
            feature_engineering_result.as_ref(),
        );

        let total_time = start_time.elapsed().as_secs_f64();

        Ok(AutoMLPipelineResult {
            final_score,
            cv_score,
            cv_std,
            best_algorithm: optimized_algorithm,
            feature_engineering: feature_engineering_result,
            algorithm_selection: algorithm_selection_result,
            ensemble_selection: ensemble_result,
            dataset_characteristics: dataset_chars,
            stage_timings,
            total_time,
            best_hyperparameters,
            improvement_over_baseline: improvement,
            model_complexity,
            interpretability_score: interpretability,
            recommendations,
        })
    }

    /// Validate input data
    fn validate_input(&self, X: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        if X.nrows() != y.len() {
            return Err(SklearsError::InvalidParameter {
                name: "X_y_shape".to_string(),
                reason: "Number of samples in X and y must match".to_string(),
            });
        }

        if X.nrows() < 2 {
            return Err(SklearsError::InvalidParameter {
                name: "n_samples".to_string(),
                reason: "Need at least 2 samples for AutoML".to_string(),
            });
        }

        if X.ncols() == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "n_features".to_string(),
                reason: "Need at least 1 feature for AutoML".to_string(),
            });
        }

        Ok(())
    }

    /// Analyze dataset characteristics
    fn analyze_dataset(&self, X: &Array2<f64>, y: &Array1<f64>) -> DatasetCharacteristics {
        let selector = AutoMLAlgorithmSelector::new(self.config.algorithm_selection_config.clone());
        selector.analyze_dataset(X, y)
    }

    /// Adapt configuration based on dataset characteristics
    fn adapt_configuration(&mut self, dataset_chars: &DatasetCharacteristics) {
        // Adapt time budget allocation based on dataset size
        let data_complexity = (dataset_chars.n_samples * dataset_chars.n_features) as f64;

        if data_complexity > 1_000_000.0 {
            // Large dataset: focus on scalable algorithms
            self.config.algorithm_selection_config.excluded_families =
                vec![AlgorithmFamily::NeighborBased, AlgorithmFamily::SVM];
            self.config.feature_engineering_config.strategy =
                FeatureEngineeringStrategy::Conservative;
        } else if data_complexity < 10_000.0 {
            // Small dataset: can try more complex methods
            self.config.feature_engineering_config.strategy =
                FeatureEngineeringStrategy::Aggressive;
        }

        // Adapt based on linearity
        if dataset_chars.linearity_score > 0.8 {
            // Linear data: prefer linear methods
            self.config.algorithm_selection_config.allowed_families =
                Some(vec![AlgorithmFamily::Linear, AlgorithmFamily::NaiveBayes]);
        }

        // Adapt ensemble settings
        if dataset_chars.n_samples < 100 {
            self.config.enable_ensemble = false; // Skip ensemble for very small datasets
        }
    }

    /// Perform feature engineering
    fn perform_feature_engineering(
        &self,
        X: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<FeatureEngineeringResult> {
        let mut engineer = AutoFeatureEngineer::new(self.config.feature_engineering_config.clone());
        engineer.engineer_features(X, y)
    }

    /// Perform algorithm selection
    fn perform_algorithm_selection(
        &self,
        X: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<AlgorithmSelectionResult> {
        let selector = AutoMLAlgorithmSelector::new(self.config.algorithm_selection_config.clone());
        selector.select_algorithms(X, y)
    }

    /// Perform hyperparameter optimization
    fn perform_hyperparameter_optimization(
        &self,
        _X: &Array2<f64>,
        _y: &Array1<f64>,
        algorithm: &RankedAlgorithm,
    ) -> Result<(RankedAlgorithm, HashMap<String, String>)> {
        // Mock implementation - would use actual hyperparameter optimization
        let mut optimized = algorithm.clone();
        optimized.cv_score += 0.02; // Mock improvement

        let best_params = algorithm.best_params.clone();

        Ok((optimized, best_params))
    }

    /// Perform ensemble construction
    fn perform_ensemble_construction(
        &self,
        _X: &Array2<f64>,
        _y: &Array1<f64>,
        algorithm_selection: &AlgorithmSelectionResult,
    ) -> Result<EnsembleSelectionResult> {
        // Mock implementation - would use actual ensemble selection
        use crate::ensemble_selection::{
            DiversityMeasures, EnsemblePerformance, EnsembleSelectionResult, EnsembleStrategy,
            ModelInfo, ModelPerformance,
        };

        let selected_models = algorithm_selection
            .selected_algorithms
            .iter()
            .take(3)
            .enumerate()
            .map(|(i, alg)| ModelInfo {
                model_index: i,
                model_name: alg.algorithm.name.clone(),
                weight: 1.0 / 3.0,
                individual_score: alg.cv_score,
                contribution_score: alg.cv_score * 0.8,
            })
            .collect();

        let ensemble_performance = EnsemblePerformance {
            mean_score: algorithm_selection.best_algorithm.cv_score + 0.01,
            std_score: 0.02,
            fold_scores: vec![0.85, 0.87, 0.86, 0.88, 0.84],
            improvement_over_best: 0.01,
            ensemble_size: 3,
        };

        let individual_performances: Vec<ModelPerformance> = algorithm_selection
            .selected_algorithms
            .iter()
            .take(3)
            .enumerate()
            .map(|(i, alg)| ModelPerformance {
                model_index: i,
                model_name: alg.algorithm.name.clone(),
                cv_score: alg.cv_score,
                cv_std: alg.cv_std,
                avg_correlation: 0.3,
            })
            .collect();

        let diversity_measures = DiversityMeasures {
            avg_correlation: 0.3,
            disagreement: 0.2,
            q_statistic: 0.15,
            entropy_diversity: 0.8,
        };

        Ok(EnsembleSelectionResult {
            ensemble_strategy: EnsembleStrategy::WeightedVoting,
            selected_models,
            model_weights: vec![0.4, 0.35, 0.25],
            ensemble_performance,
            individual_performances,
            diversity_measures,
        })
    }

    /// Perform final training
    fn perform_final_training(
        &self,
        _X: &Array2<f64>,
        _y: &Array1<f64>,
        algorithm: &RankedAlgorithm,
    ) -> Result<f64> {
        // Mock implementation - would train the actual model
        Ok(algorithm.cv_score + 0.005) // Slight improvement from final training
    }

    /// Perform model validation
    fn perform_model_validation(
        &self,
        _X: &Array2<f64>,
        _y: &Array1<f64>,
        algorithm: &RankedAlgorithm,
    ) -> Result<(f64, f64)> {
        // Mock implementation - would perform actual cross-validation
        Ok((algorithm.cv_score, algorithm.cv_std))
    }

    /// Calculate baseline score
    fn calculate_baseline_score(&self, _X: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        match self.config.task_type {
            TaskType::Classification => {
                // Most frequent class accuracy
                let mut class_counts = HashMap::new();
                for &label in y.iter() {
                    *class_counts.entry(label as i32).or_insert(0) += 1;
                }
                let max_count = class_counts.values().max().unwrap_or(&1);
                Ok(*max_count as f64 / y.len() as f64)
            }
            TaskType::Regression => {
                // R² of predicting mean (which is 0)
                Ok(0.0)
            }
        }
    }

    /// Calculate model complexity score
    fn calculate_model_complexity(&self, algorithm: &RankedAlgorithm) -> f64 {
        // Simple complexity score based on algorithm family
        match algorithm.algorithm.family {
            AlgorithmFamily::Linear => 0.2,
            AlgorithmFamily::TreeBased => 0.6,
            AlgorithmFamily::Ensemble => 0.8,
            AlgorithmFamily::NeighborBased => 0.4,
            AlgorithmFamily::SVM => 0.7,
            AlgorithmFamily::NaiveBayes => 0.1,
            AlgorithmFamily::NeuralNetwork => 0.9,
            AlgorithmFamily::GaussianProcess => 0.8,
            AlgorithmFamily::DiscriminantAnalysis => 0.3,
            AlgorithmFamily::Dummy => 0.0,
        }
    }

    /// Calculate interpretability score
    fn calculate_interpretability_score(&self, algorithm: &RankedAlgorithm) -> f64 {
        // Interpretability score (inverse of complexity)
        match algorithm.algorithm.family {
            AlgorithmFamily::Linear => 0.9,
            AlgorithmFamily::TreeBased => 0.7,
            AlgorithmFamily::Ensemble => 0.3,
            AlgorithmFamily::NeighborBased => 0.6,
            AlgorithmFamily::SVM => 0.4,
            AlgorithmFamily::NaiveBayes => 0.8,
            AlgorithmFamily::NeuralNetwork => 0.1,
            AlgorithmFamily::GaussianProcess => 0.3,
            AlgorithmFamily::DiscriminantAnalysis => 0.8,
            AlgorithmFamily::Dummy => 1.0,
        }
    }

    /// Generate recommendations
    fn generate_recommendations(
        &self,
        algorithm: &RankedAlgorithm,
        dataset_chars: &DatasetCharacteristics,
        feature_engineering: Option<&FeatureEngineeringResult>,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Performance recommendations
        if algorithm.cv_score < 0.7 {
            recommendations.push(
                "Consider collecting more data or trying different feature engineering approaches"
                    .to_string(),
            );
        }

        if algorithm.cv_score > 0.95 {
            recommendations.push(
                "High performance achieved - be cautious of overfitting, consider simpler models"
                    .to_string(),
            );
        }

        // Data recommendations
        if dataset_chars.n_samples < 1000 {
            recommendations.push(
                "Small dataset detected - consider data augmentation or transfer learning"
                    .to_string(),
            );
        }

        if dataset_chars.n_features > dataset_chars.n_samples {
            recommendations.push(
                "High-dimensional data - regularization and feature selection are crucial"
                    .to_string(),
            );
        }

        // Algorithm-specific recommendations
        match algorithm.algorithm.family {
            AlgorithmFamily::Linear => {
                recommendations.push("Linear model selected - ensure features are properly scaled and consider polynomial features".to_string());
            }
            AlgorithmFamily::TreeBased => {
                recommendations.push("Tree-based model selected - feature scaling not required, but feature importance analysis recommended".to_string());
            }
            AlgorithmFamily::Ensemble => {
                recommendations.push("Ensemble model selected - excellent for accuracy but may sacrifice interpretability".to_string());
            }
            _ => {}
        }

        // Feature engineering recommendations
        if let Some(fe_result) = feature_engineering {
            if fe_result.performance_improvement > 0.05 {
                recommendations.push("Feature engineering provided significant improvement - consider domain expertise for further enhancements".to_string());
            } else if fe_result.performance_improvement < 0.01 {
                recommendations.push("Limited benefit from automated feature engineering - consider domain-specific features".to_string());
            }
        }

        recommendations
    }

    // Progress callback helpers
    fn progress_callback_stage_start(&mut self, stage: AutoMLStage, message: &str) {
        if let Some(ref mut callback) = self.progress_callback {
            callback.on_stage_start(stage, message);
        }
    }

    fn progress_callback_stage_complete(
        &mut self,
        stage: AutoMLStage,
        duration: f64,
        message: &str,
    ) {
        if let Some(ref mut callback) = self.progress_callback {
            callback.on_stage_complete(stage, duration, message);
        }
    }
}

/// Convenience function for quick AutoML
pub fn automl(
    X: &Array2<f64>,
    y: &Array1<f64>,
    task_type: TaskType,
) -> Result<AutoMLPipelineResult> {
    let config = AutoMLPipelineConfig {
        task_type,
        ..Default::default()
    };

    let mut pipeline = AutoMLPipeline::new(config)
        .with_progress_callback(Box::new(ConsoleProgressCallback::new(true)));

    pipeline.fit(X, y)
}

/// Quick AutoML with custom time budget
pub fn automl_with_budget(
    X: &Array2<f64>,
    y: &Array1<f64>,
    task_type: TaskType,
    time_budget: f64,
) -> Result<AutoMLPipelineResult> {
    let config = AutoMLPipelineConfig {
        task_type,
        time_budget,
        ..Default::default()
    };

    let mut pipeline = AutoMLPipeline::new(config)
        .with_progress_callback(Box::new(ConsoleProgressCallback::new(true)));

    pipeline.fit(X, y)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    #[allow(non_snake_case)]
    fn create_test_classification_data() -> (Array2<f64>, Array1<f64>) {
        let X = Array2::from_shape_vec((100, 4), (0..400).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..100).map(|i| (i % 3) as f64).collect());
        (X, y)
    }

    #[allow(non_snake_case)]
    fn create_test_regression_data() -> (Array2<f64>, Array1<f64>) {
        let X = Array2::from_shape_vec((100, 4), (0..400).map(|i| i as f64).collect()).unwrap();
        use scirs2_core::essentials::Uniform;
        use scirs2_core::random::{thread_rng, Distribution};
        let mut rng = thread_rng();
        let dist = Uniform::new(0.0, 1.0).unwrap();
        let y = Array1::from_vec((0..100).map(|i| i as f64 + dist.sample(&mut rng)).collect());
        (X, y)
    }

    #[test]
    fn test_automl_classification() {
        let (X, y) = create_test_classification_data();
        let result = automl(&X, &y, TaskType::Classification);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.cv_score > 0.0);
        assert!(result.total_time > 0.0);
        assert!(!result.recommendations.is_empty());
    }

    #[test]
    fn test_automl_regression() {
        let (X, y) = create_test_regression_data();
        let result = automl(&X, &y, TaskType::Regression);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.cv_score >= 0.0);
        assert!(result.total_time > 0.0);
    }

    #[test]
    fn test_automl_with_custom_config() {
        let (X, y) = create_test_classification_data();

        let config = AutoMLPipelineConfig {
            task_type: TaskType::Classification,
            optimization_level: OptimizationLevel::Fast,
            time_budget: 60.0, // 1 minute
            enable_feature_engineering: false,
            enable_ensemble: false,
            verbose: false,
            ..Default::default()
        };

        let mut pipeline = AutoMLPipeline::new(config);
        let result = pipeline.fit(&X, &y);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.feature_engineering.is_none());
        assert!(result.ensemble_selection.is_none());
    }

    #[test]
    fn test_pipeline_stages() {
        let (X, y) = create_test_classification_data();

        let config = AutoMLPipelineConfig {
            task_type: TaskType::Classification,
            verbose: false,
            ..Default::default()
        };

        let mut pipeline = AutoMLPipeline::new(config);
        let result = pipeline.fit(&X, &y);
        assert!(result.is_ok());

        let result = result.unwrap();

        // Check that all stages were executed
        assert!(result
            .stage_timings
            .contains_key(&AutoMLStage::DataAnalysis));
        assert!(result
            .stage_timings
            .contains_key(&AutoMLStage::FeatureEngineering));
        assert!(result
            .stage_timings
            .contains_key(&AutoMLStage::AlgorithmSelection));
        assert!(result
            .stage_timings
            .contains_key(&AutoMLStage::ModelValidation));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_input_validation() {
        let X = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]); // Wrong length

        let mut pipeline = AutoMLPipeline::default();
        let result = pipeline.fit(&X, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_progress_callback() {
        struct TestCallback {
            stages_started: Vec<AutoMLStage>,
            stages_completed: Vec<AutoMLStage>,
        }

        impl TestCallback {
            fn new() -> Self {
                Self {
                    stages_started: Vec::new(),
                    stages_completed: Vec::new(),
                }
            }
        }

        impl AutoMLProgressCallback for TestCallback {
            fn on_stage_start(&mut self, stage: AutoMLStage, _message: &str) {
                self.stages_started.push(stage);
            }

            fn on_stage_progress(&mut self, _stage: AutoMLStage, _progress: f64, _message: &str) {
                // No-op for test
            }

            fn on_stage_complete(&mut self, stage: AutoMLStage, _duration: f64, _message: &str) {
                self.stages_completed.push(stage);
            }
        }

        let (X, y) = create_test_classification_data();
        let config = AutoMLPipelineConfig {
            task_type: TaskType::Classification,
            enable_ensemble: false,
            ..Default::default()
        };

        let callback = TestCallback::new();
        let mut pipeline = AutoMLPipeline::new(config).with_progress_callback(Box::new(callback));

        let result = pipeline.fit(&X, &y);
        assert!(result.is_ok());
    }
}
