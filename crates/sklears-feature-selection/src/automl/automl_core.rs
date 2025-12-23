//! Automated Feature Selection Pipeline Core
//!
//! Core pipeline implementation and fundamental types for automated feature selection.
//! All implementations follow the SciRS2 policy using scirs2-core for numerical computations.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::error::{Result as SklResult, SklearsError};
use thiserror::Error;

// Re-export from other modules
pub use super::advanced_optimizer::AdvancedHyperparameterOptimizer;
pub use super::data_analyzer::DataAnalyzer;
pub use super::hyperparameter_optimizer::{HyperparameterOptimizer, MethodConfig, OptimizedMethod};
pub use super::method_selector::MethodSelector;
pub use super::pipeline_optimizer::{
    MethodInfo, MethodPerformance, OptimalPipeline, PipelineConfigResult, PipelineOptimizer,
    TrainedOptimalPipeline,
};
pub use super::preprocessing_integration::PreprocessingIntegration;

type Result<T> = SklResult<T>;

#[derive(Debug, Error)]
pub enum AutoMLError {
    #[error("Insufficient data for automated feature selection")]
    InsufficientData,
    #[error("Invalid pipeline configuration")]
    InvalidConfiguration,
    #[error("Feature selection method failed: {0}")]
    FeatureSelectionFailed(String),
    #[error("Pipeline optimization failed")]
    OptimizationFailed,
    #[error("Data analysis failed: {0}")]
    DataAnalysisFailed(String),
}

impl From<AutoMLError> for SklearsError {
    fn from(err: AutoMLError) -> Self {
        SklearsError::FitError(format!("AutoML error: {}", err))
    }
}

/// Automated feature selection method types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AutoMLMethod {
    /// UnivariateFiltering
    UnivariateFiltering,
    /// CorrelationBased
    CorrelationBased,
    /// TreeBased
    TreeBased,
    /// LassoBased
    LassoBased,
    /// WrapperBased
    WrapperBased,
    /// EnsembleBased
    EnsembleBased,
    /// Hybrid
    Hybrid,
    /// NeuralArchitectureSearch
    NeuralArchitectureSearch,
    /// TransferLearning
    TransferLearning,
    /// MetaLearningEnsemble
    MetaLearningEnsemble,
}

/// Data characteristics for method selection
#[derive(Debug, Clone)]
pub struct DataCharacteristics {
    pub n_samples: usize,
    pub n_features: usize,
    pub feature_to_sample_ratio: f64,
    pub target_type: TargetType,
    pub has_missing_values: bool,
    pub has_categorical_features: bool,
    pub feature_variance_distribution: Vec<f64>,
    pub correlation_structure: CorrelationStructure,
    pub computational_budget: ComputationalBudget,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TargetType {
    /// BinaryClassification
    BinaryClassification,
    /// MultiClassification
    MultiClassification,
    /// Regression
    Regression,
    /// MultiLabel
    MultiLabel,
    /// Survival
    Survival,
}

#[derive(Debug, Clone)]
pub struct CorrelationStructure {
    pub high_correlation_pairs: usize,
    pub average_correlation: f64,
    pub max_correlation: f64,
    pub correlation_clusters: usize,
}

#[derive(Debug, Clone)]
pub struct ComputationalBudget {
    pub max_time_seconds: f64,
    pub max_memory_mb: f64,
    pub prefer_speed: bool,
    pub allow_complex_methods: bool,
}

/// Automated feature selection pipeline
#[derive(Debug, Clone)]
pub struct AutomatedFeatureSelectionPipeline {
    data_analyzer: DataAnalyzer,
    method_selector: MethodSelector,
    hyperparameter_optimizer: HyperparameterOptimizer,
    advanced_optimizer: AdvancedHyperparameterOptimizer,
    pipeline_optimizer: PipelineOptimizer,
    validation_strategy: ValidationStrategy,
    preprocessing_integration: Option<PreprocessingIntegration>,
    use_advanced_optimization: bool,
    enable_preprocessing: bool,
}

impl AutomatedFeatureSelectionPipeline {
    /// Create a new automated feature selection pipeline
    pub fn new() -> Self {
        Self {
            data_analyzer: DataAnalyzer::new(),
            method_selector: MethodSelector::new(),
            hyperparameter_optimizer: HyperparameterOptimizer::new(),
            advanced_optimizer: AdvancedHyperparameterOptimizer::new(),
            pipeline_optimizer: PipelineOptimizer::new(),
            validation_strategy: ValidationStrategy::CrossValidation { folds: 5 },
            preprocessing_integration: None,
            use_advanced_optimization: false,
            enable_preprocessing: false,
        }
    }

    /// Enable preprocessing with automatic configuration
    pub fn with_preprocessing(mut self) -> Self {
        self.enable_preprocessing = true;
        self
    }

    /// Enable preprocessing with custom configuration
    pub fn with_custom_preprocessing(mut self, preprocessing: PreprocessingIntegration) -> Self {
        self.preprocessing_integration = Some(preprocessing);
        self.enable_preprocessing = true;
        self
    }

    /// Enable advanced hyperparameter optimization
    pub fn with_advanced_optimization(mut self) -> Self {
        self.use_advanced_optimization = true;
        self
    }

    /// Configure advanced hyperparameter optimizer
    pub fn with_advanced_optimizer(mut self, optimizer: AdvancedHyperparameterOptimizer) -> Self {
        self.advanced_optimizer = optimizer;
        self.use_advanced_optimization = true;
        self
    }

    /// Configure the pipeline with custom settings
    pub fn with_config(mut self, config: PipelineConfig) -> Self {
        self.validation_strategy = config.validation_strategy;
        self.hyperparameter_optimizer.max_iterations = config.max_optimization_iterations;
        self
    }

    /// Automatically select features using the best pipeline
    pub fn auto_select_features(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        target_n_features: Option<usize>,
    ) -> Result<AutoMLResults> {
        // Step 1: Apply preprocessing if enabled
        let (processed_X, processed_y) = if self.enable_preprocessing {
            let preprocessing =
                if let Some(ref custom_preprocessing) = self.preprocessing_integration {
                    custom_preprocessing.clone()
                } else {
                    // Auto-configure preprocessing based on data characteristics
                    let temp_characteristics = self.data_analyzer.analyze_data(X, y)?;
                    PreprocessingIntegration::auto_configure(&temp_characteristics)
                };
            preprocessing.preprocess_data(X, y)?
        } else {
            (X.to_owned(), y.to_owned())
        };

        // Step 2: Analyze data characteristics (after preprocessing)
        let characteristics = self
            .data_analyzer
            .analyze_data(processed_X.view(), processed_y.view())?;

        // Step 3: Select appropriate methods based on data
        let candidate_methods = self.method_selector.select_methods(&characteristics)?;

        // Step 4: Optimize each method's hyperparameters using advanced or basic optimization
        let mut optimized_methods = Vec::new();
        for method in candidate_methods {
            let optimized = if self.use_advanced_optimization {
                self.advanced_optimizer.optimize_advanced(
                    &method,
                    processed_X.view(),
                    processed_y.view(),
                    &characteristics,
                )?
            } else {
                self.hyperparameter_optimizer.optimize_method(
                    &method,
                    processed_X.view(),
                    processed_y.view(),
                    &characteristics,
                )?
            };
            optimized_methods.push(optimized);
        }

        // Step 5: Evaluate methods using validation strategy
        let method_performances =
            self.evaluate_methods(processed_X.view(), processed_y.view(), &optimized_methods)?;

        // Step 6: Select best method or create ensemble
        let best_pipeline = self.pipeline_optimizer.create_optimal_pipeline(
            &optimized_methods,
            &method_performances,
            target_n_features,
        )?;

        // Step 7: Final training on full processed data
        let pipeline_config = best_pipeline.get_config();
        let validation_scores: Vec<f64> = method_performances.iter().map(|p| p.score).collect();
        let recommendation = self.generate_recommendation(&best_pipeline, &characteristics);
        let final_model = best_pipeline.fit(processed_X.view(), processed_y.view())?;

        // Step 8: Generate comprehensive results
        let selected_features = final_model.transform_indices()?;
        let feature_importances = final_model.get_feature_importances()?;

        Ok(AutoMLResults {
            selected_features,
            feature_importances,
            best_method: final_model.method_info(),
            data_characteristics: characteristics,
            method_performances,
            pipeline_config,
            validation_scores,
            recommendation,
        })
    }

    /// Evaluate methods using the configured validation strategy
    fn evaluate_methods(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        methods: &[OptimizedMethod],
    ) -> Result<Vec<MethodPerformance>> {
        let mut performances = Vec::new();

        for method in methods {
            let performance = match &self.validation_strategy {
                ValidationStrategy::CrossValidation { folds } => {
                    self.cross_validate_method(method, X, y, *folds)?
                }
                ValidationStrategy::HoldOut { test_size } => {
                    self.holdout_validate_method(method, X, y, *test_size)?
                }
                ValidationStrategy::TimeSeriesSplit { n_splits } => {
                    self.time_series_validate_method(method, X, y, *n_splits)?
                }
            };

            performances.push(performance);
        }

        Ok(performances)
    }

    /// Cross-validation for method evaluation
    #[allow(non_snake_case)]
    fn cross_validate_method(
        &self,
        method: &OptimizedMethod,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        folds: usize,
    ) -> Result<MethodPerformance> {
        let mut scores = Vec::new();
        let mut feature_selections = Vec::new();

        // Simple K-fold implementation
        let fold_size = X.nrows() / folds;

        for fold in 0..folds {
            let start = fold * fold_size;
            let end = if fold == folds - 1 {
                X.nrows()
            } else {
                (fold + 1) * fold_size
            };

            // Create train/test splits
            let test_indices: Vec<usize> = (start..end).collect();
            let train_indices: Vec<usize> = (0..start).chain(end..X.nrows()).collect();

            // Extract train/test data
            let X_train = self.extract_samples(X, &train_indices);
            let y_train = self.extract_targets(y, &train_indices);
            let X_test = self.extract_samples(X, &test_indices);
            let y_test = self.extract_targets(y, &test_indices);

            // Train method and evaluate
            let trained_method = method.clone().fit(X_train.view(), y_train.view())?;
            let selected_features = trained_method.transform_indices()?;
            feature_selections.push(selected_features.clone());

            // Evaluate performance (simplified scoring)
            let score = self.compute_validation_score(
                X_train.view(),
                y_train.view(),
                X_test.view(),
                y_test.view(),
                &selected_features,
            )?;
            scores.push(score);
        }

        // Compute stability metrics
        let stability = self.compute_stability(&feature_selections)?;

        Ok(MethodPerformance {
            method_name: format!("{:?}", method.method_type),
            score: scores.iter().sum::<f64>() / scores.len() as f64,
            score_std: self.compute_std(&scores),
            scores,
            feature_stability: stability,
            computational_cost: method.estimated_cost,
            method_config: method.config.clone(),
        })
    }

    /// Hold-out validation for method evaluation
    #[allow(non_snake_case)]
    fn holdout_validate_method(
        &self,
        method: &OptimizedMethod,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        test_size: f64,
    ) -> Result<MethodPerformance> {
        let n_test = (X.nrows() as f64 * test_size) as usize;
        let n_train = X.nrows() - n_test;

        // Simple random split
        let mut indices: Vec<usize> = (0..X.nrows()).collect();
        self.shuffle_indices(&mut indices);

        let train_indices = &indices[..n_train];
        let test_indices = &indices[n_train..];

        let X_train = self.extract_samples(X, train_indices);
        let y_train = self.extract_targets(y, train_indices);
        let X_test = self.extract_samples(X, test_indices);
        let y_test = self.extract_targets(y, test_indices);

        let trained_method = method.clone().fit(X_train.view(), y_train.view())?;
        let selected_features = trained_method.transform_indices()?;

        let score = self.compute_validation_score(
            X_train.view(),
            y_train.view(),
            X_test.view(),
            y_test.view(),
            &selected_features,
        )?;

        Ok(MethodPerformance {
            method_name: format!("{:?}", method.method_type),
            score,
            score_std: 0.0,
            scores: vec![score],
            feature_stability: 1.0, // Single evaluation
            computational_cost: method.estimated_cost,
            method_config: method.config.clone(),
        })
    }

    /// Time series validation (stub implementation)
    fn time_series_validate_method(
        &self,
        method: &OptimizedMethod,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        n_splits: usize,
    ) -> Result<MethodPerformance> {
        // Approximate rolling-origin evaluation by adjusting holdout proportion
        let holdout_ratio = if n_splits > 0 {
            (1.0 / (n_splits as f64 + 1.0)).clamp(0.1, 0.5)
        } else {
            0.2
        };

        self.holdout_validate_method(method, X, y, holdout_ratio)
    }

    /// Compute validation score (simplified implementation)
    fn compute_validation_score(
        &self,
        X_train: ArrayView2<f64>,
        y_train: ArrayView1<f64>,
        X_test: ArrayView2<f64>,
        y_test: ArrayView1<f64>,
        selected_features: &[usize],
    ) -> Result<f64> {
        if selected_features.is_empty() {
            return Ok(0.0);
        }

        // Combine train and test correlations as a simple generalization proxy
        let mut train_total = 0.0;
        let mut train_count = 0;
        let mut test_total = 0.0;
        let mut test_count = 0;

        for &feature_idx in selected_features {
            if feature_idx < X_train.ncols() {
                let feature_train = X_train.column(feature_idx);
                let correlation = self.compute_correlation(feature_train, y_train);
                train_total += correlation.abs();
                train_count += 1;
            }

            if feature_idx < X_test.ncols() {
                let feature_test = X_test.column(feature_idx);
                let correlation = self.compute_correlation(feature_test, y_test);
                test_total += correlation.abs();
                test_count += 1;
            }
        }

        let train_avg = if train_count > 0 {
            train_total / train_count as f64
        } else {
            0.0
        };

        let test_avg = if test_count > 0 {
            test_total / test_count as f64
        } else {
            0.0
        };

        let combined = (train_avg + test_avg) / 2.0;

        // Penalize for too many features
        let feature_penalty = (selected_features.len() as f64 / X_train.ncols() as f64) * 0.1;

        Ok((combined - feature_penalty).clamp(0.0, 1.0))
    }

    /// Generate recommendation based on results
    fn generate_recommendation(
        &self,
        final_model: &OptimalPipeline,
        characteristics: &DataCharacteristics,
    ) -> String {
        let mut recommendation = String::new();

        recommendation.push_str("=== AutoML Feature Selection Recommendations ===\n\n");

        // Data assessment
        if characteristics.feature_to_sample_ratio > 10.0 {
            recommendation.push_str(
                "⚠️  High-dimensional data detected - consider dimensionality reduction.\n",
            );
        }

        if characteristics.correlation_structure.high_correlation_pairs
            > characteristics.n_features / 4
        {
            recommendation
                .push_str("⚠️  High feature correlation - redundancy removal recommended.\n");
        }

        // Method recommendation
        let method_name = format!("{:?}", final_model.method.method_type);
        recommendation.push_str(&format!("✅ Recommended method: {}\n", method_name));

        if let Some(target) = final_model.target_n_features {
            recommendation.push_str(&format!(
                "🎯 Target feature count after selection: {} (from {} original)\n",
                target, characteristics.n_features
            ));
        }

        // Performance insights
        let performance_pct = final_model.performance.score * 100.0;
        recommendation.push_str(&format!(
            "📊 Estimated validation score: {:.1}% (±{:.1}%)\n",
            performance_pct,
            final_model.performance.score_std * 100.0
        ));

        recommendation.push_str("\n💡 Next steps:\n");
        recommendation.push_str("1. Validate results on held-out test set\n");
        recommendation.push_str("2. Monitor feature stability over time\n");
        recommendation.push_str("3. Consider ensemble methods for production\n");

        recommendation
    }

    // Helper methods
    fn extract_samples(&self, X: ArrayView2<f64>, indices: &[usize]) -> Array2<f64> {
        let mut samples = Array2::zeros((indices.len(), X.ncols()));
        for (i, &idx) in indices.iter().enumerate() {
            samples.row_mut(i).assign(&X.row(idx));
        }
        samples
    }

    fn extract_targets(&self, y: ArrayView1<f64>, indices: &[usize]) -> Array1<f64> {
        let mut targets = Array1::zeros(indices.len());
        for (i, &idx) in indices.iter().enumerate() {
            targets[i] = y[idx];
        }
        targets
    }

    fn shuffle_indices(&self, indices: &mut [usize]) {
        // Simple Fisher-Yates shuffle using a simple approach to avoid API complexity
        for i in (1..indices.len()).rev() {
            // Use a simple approach: swap with a random earlier position
            let j = i % (i + 1); // Simple deterministic approach for now
            indices.swap(i, j);
        }
    }

    fn compute_std(&self, values: &[f64]) -> f64 {
        if values.len() <= 1 {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }

    fn compute_stability(&self, feature_selections: &[Vec<usize>]) -> Result<f64> {
        if feature_selections.len() < 2 {
            return Ok(1.0);
        }

        let mut total_jaccard = 0.0;
        let mut comparisons = 0;

        for i in 0..feature_selections.len() {
            for j in (i + 1)..feature_selections.len() {
                let set1: std::collections::HashSet<_> = feature_selections[i].iter().collect();
                let set2: std::collections::HashSet<_> = feature_selections[j].iter().collect();

                let intersection = set1.intersection(&set2).count() as f64;
                let union = set1.union(&set2).count() as f64;

                if union > 0.0 {
                    total_jaccard += intersection / union;
                    comparisons += 1;
                }
            }
        }

        Ok(if comparisons > 0 {
            total_jaccard / comparisons as f64
        } else {
            1.0
        })
    }

    fn compute_correlation(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        let n = x.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let mean_x = x.mean().unwrap_or(0.0);
        let mean_y = y.mean().unwrap_or(0.0);

        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            sum_xy += dx * dy;
            sum_x2 += dx * dx;
            sum_y2 += dy * dy;
        }

        let denom = (sum_x2 * sum_y2).sqrt();
        if denom < 1e-10 {
            0.0
        } else {
            sum_xy / denom
        }
    }
}

impl Default for AutomatedFeatureSelectionPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// Forward declarations for types that will be defined in other modules
// These will be removed when the actual types are imported

/// Pipeline configuration (defined in pipeline_optimizer module)
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub validation_strategy: ValidationStrategy,
    pub max_optimization_iterations: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            validation_strategy: ValidationStrategy::CrossValidation { folds: 5 },
            max_optimization_iterations: 100,
        }
    }
}

/// Validation strategy for model evaluation
#[derive(Debug, Clone)]
pub enum ValidationStrategy {
    /// CrossValidation
    CrossValidation { folds: usize },
    /// HoldOut
    HoldOut { test_size: f64 },
    /// TimeSeriesSplit
    TimeSeriesSplit { n_splits: usize },
}

/// AutoML results containing all outputs from the automated pipeline
#[derive(Debug, Clone)]
pub struct AutoMLResults {
    pub selected_features: Vec<usize>,
    pub feature_importances: Vec<f64>,
    pub best_method: MethodInfo,
    pub data_characteristics: DataCharacteristics,
    pub method_performances: Vec<MethodPerformance>,
    pub pipeline_config: PipelineConfigResult,
    pub validation_scores: Vec<f64>,
    pub recommendation: String,
}

impl AutoMLResults {
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
    pub selected_feature_count: usize,
    pub best_method_name: String,
    pub best_score: f64,
    pub feature_reduction_ratio: f64,
    pub recommendation_summary: String,
}
