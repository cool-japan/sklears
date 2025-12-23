//! AutoML Feature Selection Module
//!
//! Comprehensive automated feature selection framework with multiple specialized modules.
//! All implementations follow the SciRS2 policy using scirs2-core for numerical computations.

// Module declarations
pub mod advanced_optimizer;
pub mod automl_core;
pub mod benchmark_framework;
pub mod data_analyzer;
pub mod hyperparameter_optimizer;
pub mod method_selector;
pub mod pipeline_optimizer;
pub mod preprocessing_integration;

// Re-export core types and functionality
pub use automl_core::{
    AutoMLError, AutoMLMethod, AutoMLResults, AutoMLSummary, AutomatedFeatureSelectionPipeline,
    ComputationalBudget, CorrelationStructure, DataCharacteristics, TargetType,
};

pub use data_analyzer::DataAnalyzer;

pub use method_selector::MethodSelector;

pub use hyperparameter_optimizer::{
    HyperparameterOptimizer, MethodConfig, OptimizedMethod, TrainedMethod,
};

pub use pipeline_optimizer::{
    MethodInfo, MethodPerformance, OptimalPipeline, PipelineConfig, PipelineConfigResult,
    PipelineOptimizer, TrainedOptimalPipeline, ValidationStrategy,
};

pub use preprocessing_integration::{
    DimensionalityReduction, FeatureEngineering, MissingValueStrategy, OutlierHandling,
    PreprocessingIntegration, ScalerType,
};

pub use advanced_optimizer::{
    AdvancedHyperparameterOptimizer, EarlyStoppingConfig, OptimizationStrategy,
};

pub use benchmark_framework::{
    AutoMLBenchmark, BenchmarkDataset, BenchmarkMetric, BenchmarkResults, DatasetType,
    DetailedBenchmarkResults, DifficultyLevel, ErrorAnalysis, ImprovementRatios, MethodComparison,
    OptimizationDetails, PerformanceMetrics,
};

// Factory pattern for easy construction
use scirs2_core::ndarray::{ArrayView1, ArrayView2};
use sklears_core::error::Result as SklResult;

type Result<T> = SklResult<T>;

/// Comprehensive AutoML Factory for creating and managing all components
#[derive(Debug, Clone)]
pub struct AutoMLFactory {
    config: AutoMLFactoryConfig,
}

/// Configuration for the AutoML factory
#[derive(Debug, Clone)]
pub struct AutoMLFactoryConfig {
    pub enable_advanced_optimization: bool,
    pub enable_preprocessing: bool,
    pub enable_benchmarking: bool,
    pub parallel_workers: usize,
    pub time_budget_seconds: u64,
}

impl Default for AutoMLFactoryConfig {
    fn default() -> Self {
        Self {
            enable_advanced_optimization: false,
            enable_preprocessing: true,
            enable_benchmarking: false,
            parallel_workers: 1,
            time_budget_seconds: 300,
        }
    }
}

impl AutoMLFactory {
    /// Create a new AutoML factory with default configuration
    pub fn new() -> Self {
        Self {
            config: AutoMLFactoryConfig::default(),
        }
    }

    /// Create factory with custom configuration
    pub fn with_config(config: AutoMLFactoryConfig) -> Self {
        Self { config }
    }

    /// Enable advanced hyperparameter optimization
    pub fn with_advanced_optimization(mut self) -> Self {
        self.config.enable_advanced_optimization = true;
        self
    }

    /// Enable preprocessing integration
    pub fn with_preprocessing(mut self) -> Self {
        self.config.enable_preprocessing = true;
        self
    }

    /// Enable benchmarking capabilities
    pub fn with_benchmarking(mut self) -> Self {
        self.config.enable_benchmarking = true;
        self
    }

    /// Set time budget for optimization
    pub fn with_time_budget(mut self, seconds: u64) -> Self {
        self.config.time_budget_seconds = seconds;
        self
    }

    /// Set number of parallel workers
    pub fn with_parallel_workers(mut self, workers: usize) -> Self {
        self.config.parallel_workers = workers;
        self
    }

    /// Create a basic AutoML pipeline
    pub fn create_basic_pipeline(&self) -> AutomatedFeatureSelectionPipeline {
        let mut pipeline = AutomatedFeatureSelectionPipeline::new();

        if self.config.enable_preprocessing {
            pipeline = pipeline.with_preprocessing();
        }

        if self.config.enable_advanced_optimization {
            let advanced_optimizer = AdvancedHyperparameterOptimizer::new()
                .with_time_budget(std::time::Duration::from_secs(
                    self.config.time_budget_seconds,
                ))
                .with_parallel_workers(self.config.parallel_workers);
            pipeline = pipeline.with_advanced_optimizer(advanced_optimizer);
        }

        pipeline
    }

    /// Create an advanced AutoML pipeline with full configuration
    pub fn create_advanced_pipeline(&self) -> AutomatedFeatureSelectionPipeline {
        let preprocessing = PreprocessingIntegration::new()
            .with_scaler(ScalerType::StandardScaler)
            .with_missing_value_strategy(MissingValueStrategy::KNN { k: 5 })
            .with_outlier_handling(OutlierHandling::IQR { multiplier: 1.5 })
            .with_feature_engineering(FeatureEngineering::Polynomial { degree: 2 });

        let advanced_optimizer = AdvancedHyperparameterOptimizer::new()
            .with_strategy(OptimizationStrategy::BayesianOptimization)
            .with_time_budget(std::time::Duration::from_secs(
                self.config.time_budget_seconds,
            ))
            .with_parallel_workers(self.config.parallel_workers)
            .with_early_stopping(EarlyStoppingConfig {
                patience: 10,
                min_improvement: 0.001,
                restore_best: true,
            });

        AutomatedFeatureSelectionPipeline::new()
            .with_custom_preprocessing(preprocessing)
            .with_advanced_optimizer(advanced_optimizer)
    }

    /// Create a speed-optimized pipeline for large datasets
    pub fn create_speed_optimized_pipeline(&self) -> AutomatedFeatureSelectionPipeline {
        let preprocessing = PreprocessingIntegration::new()
            .with_scaler(ScalerType::MinMaxScaler)
            .with_missing_value_strategy(MissingValueStrategy::Mean);

        AutomatedFeatureSelectionPipeline::new().with_custom_preprocessing(preprocessing)
    }

    /// Create a comprehensive benchmark suite
    pub fn create_benchmark_suite(&self) -> Result<AutoMLBenchmark> {
        if !self.config.enable_benchmarking {
            return Err(AutoMLError::InvalidConfiguration.into());
        }

        let mut benchmark = AutoMLBenchmark::new()
            .with_methods(vec![
                AutoMLMethod::UnivariateFiltering,
                AutoMLMethod::CorrelationBased,
                AutoMLMethod::TreeBased,
                AutoMLMethod::LassoBased,
                AutoMLMethod::WrapperBased,
                AutoMLMethod::EnsembleBased,
                AutoMLMethod::Hybrid,
                AutoMLMethod::NeuralArchitectureSearch,
                AutoMLMethod::TransferLearning,
                AutoMLMethod::MetaLearningEnsemble,
            ])
            .with_metrics(vec![
                BenchmarkMetric::Accuracy,
                BenchmarkMetric::F1Score,
                BenchmarkMetric::FeatureReduction,
                BenchmarkMetric::ComputationalTime,
                BenchmarkMetric::FeatureStability,
            ]);

        // Generate synthetic datasets for comprehensive evaluation
        benchmark.generate_synthetic_datasets(10)?;

        Ok(benchmark)
    }

    /// Run quick AutoML feature selection
    pub fn quick_feature_selection(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        target_features: Option<usize>,
    ) -> Result<AutoMLResults> {
        let pipeline = self.create_basic_pipeline();
        pipeline.auto_select_features(X, y, target_features)
    }

    /// Run comprehensive AutoML feature selection with all optimizations
    pub fn comprehensive_feature_selection(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        target_features: Option<usize>,
    ) -> Result<AutoMLResults> {
        let pipeline = self.create_advanced_pipeline();
        pipeline.auto_select_features(X, y, target_features)
    }

    /// Analyze data characteristics
    pub fn analyze_data_characteristics(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> Result<DataCharacteristics> {
        let analyzer = DataAnalyzer::new();
        analyzer.analyze_data(X, y)
    }

    /// Get method recommendations based on data characteristics
    pub fn recommend_methods(
        &self,
        characteristics: &DataCharacteristics,
    ) -> Result<Vec<AutoMLMethod>> {
        let selector = MethodSelector::new();
        selector.select_methods(characteristics)
    }

    /// Create custom preprocessing configuration based on data
    pub fn auto_configure_preprocessing(
        &self,
        characteristics: &DataCharacteristics,
    ) -> PreprocessingIntegration {
        PreprocessingIntegration::auto_configure(characteristics)
    }

    /// Run benchmarking evaluation
    pub fn run_benchmark_evaluation(&self) -> Result<BenchmarkResults> {
        let benchmark = self.create_benchmark_suite()?;
        benchmark.run_benchmark()
    }

    /// Generate a comprehensive report of AutoML capabilities
    pub fn generate_capability_report(&self) -> String {
        let mut report = String::new();

        report.push_str(
            "╔══════════════════════════════════════════════════════════════════════════════╗\n",
        );
        report.push_str(
            "║                          AutoML Factory Capabilities                         ║\n",
        );
        report.push_str(
            "╚══════════════════════════════════════════════════════════════════════════════╝\n\n",
        );

        // Configuration summary
        report.push_str("=== Configuration ===\n");
        report.push_str(&format!(
            "Advanced Optimization: {}\n",
            self.config.enable_advanced_optimization
        ));
        report.push_str(&format!(
            "Preprocessing: {}\n",
            self.config.enable_preprocessing
        ));
        report.push_str(&format!(
            "Benchmarking: {}\n",
            self.config.enable_benchmarking
        ));
        report.push_str(&format!(
            "Parallel Workers: {}\n",
            self.config.parallel_workers
        ));
        report.push_str(&format!(
            "Time Budget: {} seconds\n",
            self.config.time_budget_seconds
        ));

        // Available methods
        report.push_str("\n=== Available Methods ===\n");
        let methods = vec![
            "• Univariate Filtering - Fast statistical feature selection",
            "• Correlation-Based - Remove redundant features",
            "• Tree-Based - Feature importance from tree models",
            "• Lasso-Based - L1 regularization feature selection",
            "• Wrapper-Based - Model-based selection with CV",
            "• Ensemble-Based - Combine multiple selection methods",
            "• Hybrid - Multi-stage selection pipeline",
            "• Neural Architecture Search - Deep learning optimization",
            "• Transfer Learning - Leverage pre-trained models",
            "• Meta-Learning Ensemble - Adaptive method combination",
        ];
        for method in methods {
            report.push_str(&format!("{}\n", method));
        }

        // Available optimizations
        if self.config.enable_advanced_optimization {
            report.push_str("\n=== Optimization Strategies ===\n");
            let strategies = vec![
                "• Bayesian Optimization - Gaussian process guided search",
                "• Genetic Algorithm - Evolutionary optimization",
                "• Random Search - Efficient random exploration",
                "• Grid Search - Exhaustive parameter exploration",
                "• Particle Swarm Optimization - Swarm intelligence",
                "• Simulated Annealing - Temperature-based optimization",
                "• HyperBand - Multi-fidelity optimization",
            ];
            for strategy in strategies {
                report.push_str(&format!("{}\n", strategy));
            }
        }

        // Preprocessing capabilities
        if self.config.enable_preprocessing {
            report.push_str("\n=== Preprocessing Features ===\n");
            let preprocessing = vec![
                "• Scaling: Standard, MinMax, Robust, Quantile",
                "• Missing Values: Mean, Median, KNN, Interpolation",
                "• Outlier Handling: IQR, Z-Score, Isolation Forest",
                "• Feature Engineering: Polynomial, Interaction terms",
                "• Dimensionality Reduction: PCA, ICA, SVD",
            ];
            for feature in preprocessing {
                report.push_str(&format!("{}\n", feature));
            }
        }

        // Benchmarking capabilities
        if self.config.enable_benchmarking {
            report.push_str("\n=== Benchmarking Features ===\n");
            let benchmarking = vec![
                "• Synthetic Dataset Generation",
                "• Multi-metric Evaluation (Accuracy, F1, Time, Stability)",
                "• Statistical Significance Testing",
                "• Performance Comparison and Ranking",
                "• Error Analysis and Diagnostics",
                "• Improvement Ratio Calculations",
            ];
            for feature in benchmarking {
                report.push_str(&format!("{}\n", feature));
            }
        }

        report.push_str("\n💡 Use AutoMLFactory::quick_feature_selection() for fast results\n");
        report
            .push_str("💡 Use AutoMLFactory::comprehensive_feature_selection() for best quality\n");

        report
    }
}

impl Default for AutoMLFactory {
    fn default() -> Self {
        Self::new()
    }
}

// Convenience functions for quick access
/// Quick feature selection with default settings
pub fn quick_automl(
    X: ArrayView2<f64>,
    y: ArrayView1<f64>,
    target_features: Option<usize>,
) -> Result<AutoMLResults> {
    let factory = AutoMLFactory::new();
    factory.quick_feature_selection(X, y, target_features)
}

/// Comprehensive feature selection with all optimizations
pub fn comprehensive_automl(
    X: ArrayView2<f64>,
    y: ArrayView1<f64>,
    target_features: Option<usize>,
) -> Result<AutoMLResults> {
    let factory = AutoMLFactory::new()
        .with_advanced_optimization()
        .with_preprocessing()
        .with_time_budget(600); // 10 minutes for comprehensive analysis
    factory.comprehensive_feature_selection(X, y, target_features)
}

/// Analyze dataset and get method recommendations
pub fn analyze_and_recommend(
    X: ArrayView2<f64>,
    y: ArrayView1<f64>,
) -> Result<(DataCharacteristics, Vec<AutoMLMethod>)> {
    let factory = AutoMLFactory::new();
    let characteristics = factory.analyze_data_characteristics(X, y)?;
    let methods = factory.recommend_methods(&characteristics)?;
    Ok((characteristics, methods))
}
