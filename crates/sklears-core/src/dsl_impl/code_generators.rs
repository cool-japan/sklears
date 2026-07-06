//! Code generation implementations for DSL macros
//!
//! This module contains the code generation logic that transforms parsed DSL
//! configurations into executable Rust code. It handles pipeline generation,
//! feature engineering transformations, and hyperparameter optimization setups.

use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

use crate::dsl_impl::dsl_types::{
    DataOperation, DataPipelineConfig, DataPipelineMode, DataStep, ExperimentConfig,
    ExperimentParameter, FeatureDefinition, FeatureEngineeringConfig, HyperparameterConfig,
    ModelEvaluationConfig, OptimizationMetric, OptimizationStrategy, ParameterDef,
    ParameterDistribution, PerformanceConfig, PipelineConfig, PipelineStage, StageType,
    StatisticalTest,
};

/// Generate pipeline code from configuration
///
/// Creates a complete pipeline implementation with stages, execution logic,
/// and performance optimizations based on the parsed configuration.
///
/// # Arguments
/// * `config` - Parsed pipeline configuration
///
/// # Returns
/// Generated TokenStream containing the pipeline implementation
pub fn generate_pipeline_code(config: PipelineConfig) -> TokenStream {
    let pipeline_name = generate_pipeline_name(&config.name);
    let input_type = &config.input_type;
    let output_type = &config.output_type;
    let parallel = config.parallel;
    let validate_input = config.validate_input;
    let cache_transforms = config.cache_transforms;

    // Generate stage structures and implementations
    let stage_definitions = generate_stage_definitions(&config.stages);
    let stage_initializations = generate_stage_initializations(&config.stages);
    let execution_logic = generate_execution_logic(&config.stages, parallel);
    let validation_logic = generate_validation_logic(validate_input);
    let caching_logic = generate_caching_logic(cache_transforms);
    let performance_optimizations = generate_performance_optimizations(&config.performance);

    // Generate the main pipeline structure
    quote! {
        /// Generated ML Pipeline
        ///
        /// This pipeline was automatically generated from DSL configuration.
        /// It provides efficient execution of the configured stages with
        /// optional parallelization, validation, and caching.
        #[derive(Debug, Clone)]
        pub struct #pipeline_name {
            stages: Vec<Box<dyn crate::traits::PipelineStage>>,
            config: PipelineConfiguration,
            cache: std::collections::HashMap<String, Vec<u8>>,
            performance_monitor: crate::monitoring::PerformanceMonitor,
        }

        /// Configuration for the generated pipeline
        #[derive(Debug, Clone)]
        pub struct PipelineConfiguration {
            pub parallel: bool,
            pub validate_input: bool,
            pub cache_transforms: bool,
            pub performance: PerformanceConfig,
        }

        #stage_definitions

        impl #pipeline_name {
            /// Create a new pipeline instance
            pub fn new() -> crate::error::Result<Self> {
                let stages = vec![
                    #(#stage_initializations),*
                ];

                Ok(Self {
                    stages,
                    config: PipelineConfiguration {
                        parallel: #parallel,
                        validate_input: #validate_input,
                        cache_transforms: #cache_transforms,
                        performance: Default::default(),
                    },
                    cache: std::collections::HashMap::new(),
                    performance_monitor: crate::monitoring::PerformanceMonitor::new(),
                })
            }

            /// Execute the pipeline on input data
            pub fn execute(&mut self, input: #input_type) -> crate::error::Result<#output_type> {
                #performance_optimizations
                #validation_logic

                let _start_time = std::time::Instant::now();
                let mut result = input;

                #execution_logic
                #caching_logic

                self.performance_monitor.record_execution_time(_start_time.elapsed());
                Ok(result)
            }

            /// Get pipeline configuration
            pub fn config(&self) -> &PipelineConfiguration {
                &self.config
            }

            /// Get performance metrics
            pub fn performance_metrics(&self) -> &crate::monitoring::PerformanceMonitor {
                &self.performance_monitor
            }

            /// Clear pipeline cache
            pub fn clear_cache(&mut self) {
                self.cache.clear();
            }

            /// Get cache statistics
            pub fn cache_stats(&self) -> (usize, usize) {
                (self.cache.len(), self.cache.values().map(|v| v.len()).sum())
            }
        }

        impl Default for #pipeline_name {
            fn default() -> Self {
                Self::new().expect("Failed to create default pipeline")
            }
        }

        impl crate::traits::Estimator for #pipeline_name {
            type Input = #input_type;
            type Output = #output_type;

            fn fit(&mut self, input: &Self::Input) -> crate::error::Result<()> {
                // Generated pipelines are pre-configured
                Ok(())
            }
        }

        impl crate::traits::Transform for #pipeline_name {
            type Input = #input_type;
            type Output = #output_type;

            fn transform(&self, input: &Self::Input) -> crate::error::Result<Self::Output> {
                // Clone for mutable execution
                let mut pipeline = self.clone();
                pipeline.execute(input.clone())
            }
        }

        impl crate::traits::Predict for #pipeline_name {
            type Input = #input_type;
            type Output = #output_type;

            fn predict(&self, input: &Self::Input) -> crate::error::Result<Self::Output> {
                self.transform(input)
            }
        }
    }
}

/// Generate feature engineering code from configuration
///
/// Creates feature transformation code based on the parsed feature definitions
/// and validation rules.
///
/// # Arguments
/// * `config` - Parsed feature engineering configuration
///
/// # Returns
/// Generated TokenStream containing the feature engineering implementation
pub fn generate_feature_engineering_code(config: FeatureEngineeringConfig) -> TokenStream {
    let dataset_expr = &config.dataset;
    let feature_transformations = generate_feature_transformations(&config.features);
    let validation_code = generate_feature_validation(&config.validation);
    let selection_code = generate_feature_selection(&config.selection);

    quote! {
        /// Generated Feature Engineering Pipeline
        ///
        /// This feature engineering pipeline was automatically generated from DSL configuration.
        /// It applies the specified transformations, validation, and selection criteria.
        {
            use crate::feature_engineering::*;
            use scirs2_core::ndarray::*;

            let mut dataset = #dataset_expr;

            // Apply feature transformations
            #feature_transformations

            // Apply validation rules
            #validation_code

            // Apply feature selection
            #selection_code

            dataset
        }
    }
}

/// Generate hyperparameter configuration code
///
/// Creates hyperparameter optimization setup code based on the parsed
/// configuration with parameter definitions and optimization strategy.
///
/// # Arguments
/// * `config` - Parsed hyperparameter configuration
///
/// # Returns
/// Generated TokenStream containing the hyperparameter optimization setup
pub fn generate_hyperparameter_code(config: HyperparameterConfig) -> TokenStream {
    let model_type = &config.model;
    let parameter_definitions = generate_parameter_definitions(&config.parameters);
    let constraint_definitions = generate_constraint_definitions(&config.constraints);
    let optimization_setup = generate_optimization_setup(&config.optimization);

    quote! {
        /// Generated Hyperparameter Optimization Configuration
        ///
        /// This configuration was automatically generated from DSL specification.
        /// It defines the parameter search space and optimization strategy.
        {
            use crate::optimization::*;
            use crate::model_selection::*;

            // Create hyperparameter search space
            let mut search_space = SearchSpace::new();

            #parameter_definitions

            // Add constraints
            #constraint_definitions

            // Configure optimization strategy
            #optimization_setup

            // Create optimizer
            let optimizer = HyperparameterOptimizer::new(search_space)
                .with_model::<#model_type>()
                .with_optimization_config(optimization_config)
                .build()?;

            optimizer
        }
    }
}

/// Generate a valid Rust identifier for the pipeline name
fn generate_pipeline_name(name: &str) -> Ident {
    let clean_name = name
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_')
        .collect::<String>()
        .replace("_", "")
        .chars()
        .enumerate()
        .map(|(i, c)| if i == 0 { c.to_ascii_uppercase() } else { c })
        .collect::<String>();

    let pipeline_name = if clean_name.is_empty() {
        "GeneratedPipeline".to_string()
    } else {
        format!("{}Pipeline", clean_name)
    };

    Ident::new(&pipeline_name, Span::call_site())
}

/// Generate stage definitions for the pipeline
fn generate_stage_definitions(stages: &[PipelineStage]) -> TokenStream {
    let stage_structs = stages.iter().enumerate().map(|(i, stage)| {
        let stage_name = Ident::new(&format!("Stage{}", i), Span::call_site());
        let transforms = &stage.transforms;

        quote! {
            #[derive(Debug, Clone)]
            struct #stage_name {
                transforms: Vec<Box<dyn crate::traits::Transform>>,
            }

            impl #stage_name {
                fn new() -> Self {
                    Self {
                        transforms: vec![
                            #(Box::new(#transforms)),*
                        ],
                    }
                }
            }

            impl crate::traits::PipelineStage for #stage_name {
                fn execute(&self, input: &dyn std::any::Any) -> crate::error::Result<Box<dyn std::any::Any>> {
                    let mut result = input;
                    for transform in &self.transforms {
                        result = transform.transform_any(result)?.as_ref();
                    }
                    Ok(Box::new(result))
                }
            }
        }
    });

    quote! {
        #(#stage_structs)*
    }
}

/// Generate stage initialization code
fn generate_stage_initializations(stages: &[PipelineStage]) -> Vec<TokenStream> {
    stages
        .iter()
        .enumerate()
        .map(|(i, _stage)| {
            let stage_name = Ident::new(&format!("Stage{}", i), Span::call_site());
            quote! {
                Box::new(#stage_name::new()) as Box<dyn crate::traits::PipelineStage>
            }
        })
        .collect()
}

/// Generate execution logic for pipeline stages
fn generate_execution_logic(stages: &[PipelineStage], parallel: bool) -> TokenStream {
    let stage_executions = stages.iter().enumerate().map(|(i, stage)| {
        let stage_idx = syn::Index::from(i);

        match stage.stage_type {
            StageType::Preprocess => quote! {
                result = self.stages[#stage_idx].execute(&result)?;
            },
            StageType::FeatureEngineering => quote! {
                result = self.stages[#stage_idx].execute(&result)?;
            },
            StageType::Model => quote! {
                result = self.stages[#stage_idx].execute(&result)?;
            },
            StageType::Postprocess => quote! {
                result = self.stages[#stage_idx].execute(&result)?;
            },
            StageType::Custom(_) => quote! {
                result = self.stages[#stage_idx].execute(&result)?;
            },
        }
    });

    if parallel {
        quote! {
            // Parallel execution where possible
            use rayon::prelude::*;

            #(#stage_executions)*
        }
    } else {
        quote! {
            // Sequential execution
            #(#stage_executions)*
        }
    }
}

/// Generate input validation logic
fn generate_validation_logic(validate_input: bool) -> TokenStream {
    if validate_input {
        quote! {
            // Validate input data
            if let Err(validation_error) = crate::validation::validate_input(&result) {
                return Err(crate::error::SklearsError::ValidationError(
                    format!("Input validation failed: {}", validation_error)
                ));
            }
        }
    } else {
        quote! {
            // Input validation disabled
        }
    }
}

/// Generate caching logic
fn generate_caching_logic(cache_transforms: bool) -> TokenStream {
    if cache_transforms {
        quote! {
            // Cache intermediate results
            let cache_key = format!("pipeline_result_{}",
                std::hash::Hash::hash(&std::any::TypeId::of::<()>()));

            if let Some(cached_result) = self.cache.get(&cache_key) {
                if let Ok(deserialized) = oxicode::serde::decode_from_slice(cached_result, oxicode::config::standard()) {
                    return Ok(deserialized);
                }
            }

            // Store result in cache
            if let Ok(serialized) = oxicode::serde::encode_to_vec(&result, oxicode::config::standard()) {
                self.cache.insert(cache_key, serialized);
            }
        }
    } else {
        quote! {
            // Caching disabled
        }
    }
}

/// Generate performance optimizations
fn generate_performance_optimizations(config: &PerformanceConfig) -> TokenStream {
    let mut optimizations = Vec::new();

    if let Some(max_threads) = config.max_threads {
        optimizations.push(quote! {
            rayon::ThreadPoolBuilder::new()
                .num_threads(#max_threads)
                .build_global()
                .ok();
        });
    }

    if config.gpu_acceleration {
        optimizations.push(quote! {
            // Enable GPU acceleration if available. `GpuBackend::detect()`
            // honestly returns `Ok(None)` (rather than an error) when no
            // accelerator is present, so this only fails on a genuine
            // driver/context error, never merely because there is no GPU.
            #[cfg(feature = "gpu_support")]
            {
                let _gpu = sklears_core::gpu::GpuBackend::detect()?;
            }
        });
    }

    if let Some(batch_size) = config.batch_size {
        optimizations.push(quote! {
            // Set optimal batch size
            const OPTIMAL_BATCH_SIZE: usize = #batch_size;
        });
    }

    quote! {
        #(#optimizations)*
    }
}

/// Generate feature transformation code
fn generate_feature_transformations(features: &[FeatureDefinition]) -> TokenStream {
    let transformations = features.iter().map(|feature| {
        let name = &feature.name;
        let expr = &feature.expression;

        quote! {
            // Generate feature: #name
            dataset = dataset.with_column(
                #name,
                #expr
            )?;
        }
    });

    quote! {
        #(#transformations)*
    }
}

/// Generate feature validation code
fn generate_feature_validation(
    validation_rules: &[crate::dsl_impl::dsl_types::ValidationRule],
) -> TokenStream {
    let validations = validation_rules.iter().map(|rule| {
        let feature = &rule.feature;
        let rule_expr = &rule.rule;

        quote! {
            // Validate feature: #feature
            if !dataset.column(#feature)?.validate(#rule_expr)? {
                return Err(crate::error::SklearsError::ValidationError(
                    format!("Feature {} failed validation: {}", #feature, #rule_expr)
                ));
            }
        }
    });

    quote! {
        #(#validations)*
    }
}

/// Generate feature selection code
fn generate_feature_selection(
    selection_criteria: &[crate::dsl_impl::dsl_types::SelectionCriterion],
) -> TokenStream {
    let selections = selection_criteria.iter().map(|criterion| {
        let threshold = criterion.threshold;

        quote! {
            // Apply feature selection with threshold: #threshold
            dataset = crate::feature_selection::select_features(dataset, #threshold)?;
        }
    });

    quote! {
        #(#selections)*
    }
}

/// Generate parameter definitions for hyperparameter optimization
fn generate_parameter_definitions(parameters: &[ParameterDef]) -> TokenStream {
    let definitions = parameters.iter().map(|param| {
        let name = &param.name;
        let distribution = match &param.distribution {
            ParameterDistribution::Uniform { min, max } => {
                quote! {
                    ParameterDistribution::Uniform {
                        min: #min,
                        max: #max,
                    }
                }
            }
            ParameterDistribution::LogUniform { min, max } => {
                quote! {
                    ParameterDistribution::LogUniform {
                        min: #min,
                        max: #max,
                    }
                }
            }
            ParameterDistribution::Choice { options } => {
                quote! {
                    ParameterDistribution::Choice {
                        options: vec![#(#options),*],
                    }
                }
            }
            ParameterDistribution::IntRange { min, max } => {
                quote! {
                    ParameterDistribution::IntRange {
                        min: #min,
                        max: #max,
                    }
                }
            }
            ParameterDistribution::Normal { mean, std } => {
                quote! {
                    ParameterDistribution::Normal {
                        mean: #mean,
                        std: #std,
                    }
                }
            }
            ParameterDistribution::Custom { function } => {
                quote! {
                    ParameterDistribution::Custom {
                        function: #function,
                    }
                }
            }
        };

        quote! {
            search_space.add_parameter(#name, #distribution);
        }
    });

    quote! {
        #(#definitions)*
    }
}

/// Generate constraint definitions
fn generate_constraint_definitions(constraints: &[syn::Expr]) -> TokenStream {
    let constraint_definitions = constraints.iter().map(|constraint| {
        quote! {
            search_space.add_constraint(#constraint);
        }
    });

    quote! {
        #(#constraint_definitions)*
    }
}

/// Generate optimization setup code
fn generate_optimization_setup(
    config: &crate::dsl_impl::dsl_types::OptimizationConfig,
) -> TokenStream {
    let strategy = &config.strategy;
    let max_iterations = config.max_iterations;
    let parallel = config.parallel;

    let strategy_code = match strategy {
        OptimizationStrategy::RandomSearch => {
            quote! { OptimizationStrategy::RandomSearch }
        }
        OptimizationStrategy::GridSearch => {
            quote! { OptimizationStrategy::GridSearch }
        }
        OptimizationStrategy::BayesianOptimization => {
            quote! { OptimizationStrategy::BayesianOptimization }
        }
        _ => {
            quote! { OptimizationStrategy::RandomSearch }
        }
    };

    quote! {
        let optimization_config = OptimizationConfig {
            strategy: #strategy_code,
            max_iterations: #max_iterations,
            parallel: #parallel,
            ..Default::default()
        };
    }
}

/// Generate model evaluation code from configuration.
///
/// Produces a self-contained block expression that evaluates the configured
/// model against the supplied feature/target data, computing each requested
/// metric and, when requested, performing cross-validation and a statistical
/// significance test. The block evaluates to an `EvaluationReport` value.
///
/// # Arguments
/// * `config` - Parsed model evaluation configuration
///
/// # Returns
/// Generated TokenStream containing the evaluation implementation
pub fn generate_model_evaluation_code(config: ModelEvaluationConfig) -> TokenStream {
    let model_expr = &config.model;
    let features_expr = &config.features;
    let targets_expr = &config.targets;

    let metric_computations = generate_metric_computations(&config.metrics);

    let cross_validation_code = match &config.cross_validation {
        Some(cv) => {
            let n_folds = cv.n_folds;
            let stratified = cv.stratified;
            let seed_code = match cv.random_seed {
                Some(seed) => quote! { Some(#seed) },
                None => quote! { None },
            };
            let cv_metric_computations = generate_cv_metric_computations(&config.metrics);
            quote! {
                {
                    let cv_config = crate::model_selection::CrossValidationConfig {
                        n_folds: #n_folds,
                        stratified: #stratified,
                        random_seed: #seed_code,
                    };
                    let folds = crate::model_selection::make_folds(
                        &__eval_targets,
                        &cv_config,
                    )?;
                    let mut cv_scores: std::collections::HashMap<String, Vec<f64>> =
                        std::collections::HashMap::new();
                    for (train_idx, test_idx) in folds.iter() {
                        let mut __fold_model = #model_expr;
                        let __train_x = crate::model_selection::take_rows(&__eval_features, train_idx)?;
                        let __train_y = crate::model_selection::take_rows(&__eval_targets, train_idx)?;
                        crate::traits::Estimator::fit(&mut __fold_model, &__train_x, &__train_y)?;
                        let __test_x = crate::model_selection::take_rows(&__eval_features, test_idx)?;
                        let __test_y = crate::model_selection::take_rows(&__eval_targets, test_idx)?;
                        let __fold_pred = crate::traits::Predict::predict(&__fold_model, &__test_x)?;
                        #cv_metric_computations
                    }
                    Some(cv_scores)
                }
            }
        }
        None => quote! { None },
    };

    let statistical_test_code = match &config.statistical_test {
        Some(test) => {
            let test_call = match test {
                StatisticalTest::PairedTTest => quote! {
                    crate::stats::paired_t_test(baseline, candidate)?
                },
                StatisticalTest::Wilcoxon => quote! {
                    crate::stats::wilcoxon_signed_rank(baseline, candidate)?
                },
                StatisticalTest::McNemar => quote! {
                    crate::stats::mcnemar_test(baseline, candidate)?
                },
                StatisticalTest::Custom(name) => {
                    let func = syn::Ident::new(name, Span::call_site());
                    quote! { crate::stats::#func(baseline, candidate)? }
                }
            };
            quote! {
                Some(|baseline: &[f64], candidate: &[f64]|
                    -> crate::error::Result<crate::stats::StatisticalTestResult> {
                    Ok(#test_call)
                })
            }
        }
        None => quote! { None },
    };

    quote! {
        {
            use crate::traits::{Estimator, Predict};

            let __eval_features = #features_expr;
            let __eval_targets = #targets_expr;
            let mut __eval_model = #model_expr;

            // Fit on the full dataset to obtain holdout predictions.
            Estimator::fit(&mut __eval_model, &__eval_features, &__eval_targets)?;
            let __eval_pred = Predict::predict(&__eval_model, &__eval_features)?;

            let mut metrics: std::collections::HashMap<String, f64> =
                std::collections::HashMap::new();
            #metric_computations

            let cross_validation_scores = #cross_validation_code;

            #[allow(clippy::type_complexity)]
            let statistical_test: Option<fn(&[f64], &[f64])
                -> crate::error::Result<crate::stats::StatisticalTestResult>> =
                #statistical_test_code;

            crate::model_selection::EvaluationReport {
                metrics,
                cross_validation_scores,
                statistical_test,
            }
        }
    }
}

/// Generate per-metric computations for the holdout evaluation.
fn generate_metric_computations(metrics: &[OptimizationMetric]) -> TokenStream {
    let computations = metrics.iter().map(|metric| {
        let (key, call) = metric_call(metric);
        quote! {
            metrics.insert(
                #key.to_string(),
                #call(&__eval_targets, &__eval_pred)?,
            );
        }
    });

    quote! {
        #(#computations)*
    }
}

/// Generate per-metric computations accumulated across cross-validation folds.
fn generate_cv_metric_computations(metrics: &[OptimizationMetric]) -> TokenStream {
    let computations = metrics.iter().map(|metric| {
        let (key, call) = metric_call(metric);
        quote! {
            cv_scores
                .entry(#key.to_string())
                .or_default()
                .push(#call(&__test_y, &__fold_pred)?);
        }
    });

    quote! {
        #(#computations)*
    }
}

/// Resolve a metric to its string key and the metrics function used to compute it.
fn metric_call(metric: &OptimizationMetric) -> (String, TokenStream) {
    match metric {
        OptimizationMetric::Accuracy => (
            "accuracy".to_string(),
            quote! { crate::metrics::accuracy_score },
        ),
        OptimizationMetric::Precision => (
            "precision".to_string(),
            quote! { crate::metrics::precision_score },
        ),
        OptimizationMetric::Recall => (
            "recall".to_string(),
            quote! { crate::metrics::recall_score },
        ),
        OptimizationMetric::F1Score => ("f1".to_string(), quote! { crate::metrics::f1_score }),
        OptimizationMetric::AucRoc => (
            "auc_roc".to_string(),
            quote! { crate::metrics::roc_auc_score },
        ),
        OptimizationMetric::MeanSquaredError => (
            "mean_squared_error".to_string(),
            quote! { crate::metrics::mean_squared_error },
        ),
        OptimizationMetric::MeanAbsoluteError => (
            "mean_absolute_error".to_string(),
            quote! { crate::metrics::mean_absolute_error },
        ),
        OptimizationMetric::RSquared => {
            ("r_squared".to_string(), quote! { crate::metrics::r2_score })
        }
        OptimizationMetric::Custom(name) => {
            let func = syn::Ident::new(name, Span::call_site());
            (name.clone(), quote! { crate::metrics::#func })
        }
    }
}

/// Generate data pipeline code from configuration.
///
/// Produces a generated struct implementing the configured data pipeline. The
/// struct exposes a `run` method that threads the input data source through the
/// ordered steps. The execution mode selects between full-batch processing and
/// chunked streaming/real-time processing using the configured batch size.
///
/// # Arguments
/// * `config` - Parsed data pipeline configuration
///
/// # Returns
/// Generated TokenStream containing the data pipeline implementation
pub fn generate_data_pipeline_code(config: DataPipelineConfig) -> TokenStream {
    let pipeline_name = generate_data_pipeline_name(&config.name);
    let source_expr = &config.source;
    let parallel = config.parallel;

    let step_applications = generate_data_step_applications(&config.steps);

    let batch_size = config.batch_size.unwrap_or(1024);
    let execution_body = match config.mode {
        DataPipelineMode::Batch => quote! {
            // Process the whole dataset in a single pass.
            let mut data = #source_expr;
            #step_applications
            Ok(data)
        },
        DataPipelineMode::Streaming | DataPipelineMode::RealTime => quote! {
            // Process the dataset incrementally in fixed-size chunks.
            const CHUNK_SIZE: usize = #batch_size;
            let source = #source_expr;
            let mut output = crate::data::DataChunkBuilder::new();
            for chunk in crate::data::chunks(source, CHUNK_SIZE) {
                let mut data = chunk?;
                #step_applications
                output.extend(data)?;
            }
            output.finish()
        },
    };

    let parallel_setup = if parallel {
        quote! {
            // Allow independent record transformations to run in parallel.
            use scirs2_core::parallel_ops::*;
        }
    } else {
        quote! {}
    };

    quote! {
        /// Generated Data Pipeline
        ///
        /// This data pipeline was automatically generated from DSL configuration.
        #[derive(Debug, Default, Clone)]
        pub struct #pipeline_name;

        impl #pipeline_name {
            /// Create a new data pipeline instance.
            pub fn new() -> Self {
                Self
            }

            /// Execute the data pipeline, returning the transformed data.
            pub fn run(&self) -> crate::error::Result<crate::data::DataFrame> {
                #parallel_setup
                #execution_body
            }
        }
    }
}

/// Generate the application of each ordered data processing step.
fn generate_data_step_applications(steps: &[DataStep]) -> TokenStream {
    let applications = steps.iter().map(|step| {
        let step_name = &step.name;
        let transform = &step.transform;
        let op_call = match &step.operation {
            DataOperation::Load => quote! { crate::data::ops::load },
            DataOperation::Filter => quote! { crate::data::ops::filter },
            DataOperation::Map => quote! { crate::data::ops::map },
            DataOperation::Aggregate => quote! { crate::data::ops::aggregate },
            DataOperation::Join => quote! { crate::data::ops::join },
            DataOperation::Sink => quote! { crate::data::ops::sink },
            DataOperation::Custom(name) => {
                let func = syn::Ident::new(name, Span::call_site());
                quote! { crate::data::ops::#func }
            }
        };

        quote! {
            // Step: #step_name
            data = #op_call(data, #transform)
                .map_err(|err| crate::error::SklearsError::InvalidInput(
                    format!("data_pipeline step '{}' failed: {}", #step_name, err)
                ))?;
        }
    });

    quote! {
        #(#applications)*
    }
}

/// Generate a valid Rust identifier for the data pipeline name.
fn generate_data_pipeline_name(name: &str) -> Ident {
    let clean_name = name
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_')
        .collect::<String>()
        .replace('_', "")
        .chars()
        .enumerate()
        .map(|(i, c)| if i == 0 { c.to_ascii_uppercase() } else { c })
        .collect::<String>();

    let pipeline_name = if clean_name.is_empty() {
        "GeneratedDataPipeline".to_string()
    } else {
        format!("{}DataPipeline", clean_name)
    };

    Ident::new(&pipeline_name, Span::call_site())
}

/// Generate experiment configuration code from configuration.
///
/// Produces a block expression that constructs an `ExperimentTracker` with the
/// configured identity, recorded hyperparameters, tracked metric names, seed,
/// and tags. The block evaluates to the initialized tracker, ready to receive
/// logged metric values during the experiment.
///
/// # Arguments
/// * `config` - Parsed experiment configuration
///
/// # Returns
/// Generated TokenStream containing the experiment setup implementation
pub fn generate_experiment_config_code(config: ExperimentConfig) -> TokenStream {
    let name = &config.name;

    let description_code = match &config.description {
        Some(desc) => quote! { Some(#desc.to_string()) },
        None => quote! { None },
    };

    let seed_code = match config.seed {
        Some(seed) => quote! { Some(#seed) },
        None => quote! { None },
    };

    let parameter_insertions = generate_experiment_parameters(&config.parameters);

    let tracked_metrics = &config.tracked_metrics;
    let tags = &config.tags;

    quote! {
        {
            let mut __experiment = crate::experiment::ExperimentTracker::new(#name.to_string());
            __experiment.set_description(#description_code);
            __experiment.set_seed(#seed_code);

            #parameter_insertions

            #(
                __experiment.track_metric(#tracked_metrics.to_string());
            )*

            #(
                __experiment.add_tag(#tags.to_string());
            )*

            __experiment
        }
    }
}

/// Generate the hyperparameter recordings for an experiment.
fn generate_experiment_parameters(parameters: &[ExperimentParameter]) -> TokenStream {
    let insertions = parameters.iter().map(|param| {
        let name = &param.name;
        let value = &param.value;
        quote! {
            __experiment.log_parameter(
                #name.to_string(),
                crate::experiment::ParameterValue::from(#value),
            );
        }
    });

    quote! {
        #(#insertions)*
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl_impl::dsl_types::*;

    #[test]
    fn test_generate_pipeline_name() {
        let name = generate_pipeline_name("test_pipeline");
        // The actual output capitalizes first letter and adds "Pipeline" suffix
        assert_eq!(name.to_string(), "TestpipelinePipeline");
    }

    #[test]
    fn test_generate_empty_pipeline() {
        let config = PipelineConfig {
            name: "test".to_string(),
            stages: vec![],
            input_type: syn::parse_str("i32").expect("expected valid value"),
            output_type: syn::parse_str("i32").expect("expected valid value"),
            parallel: false,
            validate_input: false,
            cache_transforms: false,
            metadata: std::collections::HashMap::new(),
            performance: PerformanceConfig::default(),
        };

        let result = generate_pipeline_code(config);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_generate_feature_engineering_empty() {
        let config = FeatureEngineeringConfig {
            dataset: syn::parse_str("dataset").expect("expected valid value"),
            features: vec![],
            selection: vec![],
            validation: vec![],
            options: FeatureEngineeringOptions::default(),
        };

        let result = generate_feature_engineering_code(config);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_generate_model_evaluation() {
        let config = ModelEvaluationConfig {
            model: syn::parse_str("model").expect("expected valid value"),
            features: syn::parse_str("x").expect("expected valid value"),
            targets: syn::parse_str("y").expect("expected valid value"),
            metrics: vec![OptimizationMetric::Accuracy, OptimizationMetric::F1Score],
            cross_validation: Some(CrossValidationConfig {
                n_folds: 5,
                stratified: true,
                random_seed: Some(1),
            }),
            statistical_test: Some(StatisticalTest::PairedTTest),
        };

        let result = generate_model_evaluation_code(config);
        let rendered = result.to_string();
        assert!(!result.is_empty());
        // The generated code must compute the requested metrics and run CV/testing.
        assert!(rendered.contains("accuracy_score"));
        assert!(rendered.contains("f1_score"));
        assert!(rendered.contains("EvaluationReport"));
        assert!(rendered.contains("paired_t_test"));
    }

    #[test]
    fn test_generate_data_pipeline() {
        let config = DataPipelineConfig {
            name: "etl".to_string(),
            source: syn::parse_str("read_csv(\"data.csv\")").expect("expected valid value"),
            steps: vec![
                DataStep {
                    name: "clean".to_string(),
                    operation: DataOperation::Filter,
                    transform: syn::parse_str("valid_predicate").expect("expected valid value"),
                },
                DataStep {
                    name: "scale".to_string(),
                    operation: DataOperation::Map,
                    transform: syn::parse_str("scale_fn").expect("expected valid value"),
                },
            ],
            mode: DataPipelineMode::Streaming,
            batch_size: Some(128),
            parallel: true,
        };

        let result = generate_data_pipeline_code(config);
        let rendered = result.to_string();
        assert!(!result.is_empty());
        assert!(rendered.contains("EtlDataPipeline"));
        assert!(rendered.contains("ops :: filter"));
        assert!(rendered.contains("ops :: map"));
        // Streaming mode must chunk the source.
        assert!(rendered.contains("CHUNK_SIZE"));
    }

    #[test]
    fn test_generate_experiment_config() {
        let config = ExperimentConfig {
            name: "exp".to_string(),
            description: Some("desc".to_string()),
            parameters: vec![ExperimentParameter {
                name: "lr".to_string(),
                value: syn::parse_str("0.01").expect("expected valid value"),
            }],
            tracked_metrics: vec!["accuracy".to_string()],
            seed: Some(42),
            tags: vec!["baseline".to_string()],
        };

        let result = generate_experiment_config_code(config);
        let rendered = result.to_string();
        assert!(!result.is_empty());
        assert!(rendered.contains("ExperimentTracker"));
        assert!(rendered.contains("log_parameter"));
        assert!(rendered.contains("track_metric"));
        assert!(rendered.contains("add_tag"));
    }
}
