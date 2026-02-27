//! Code generation implementations for DSL macros
//!
//! This module contains the code generation logic that transforms parsed DSL
//! configurations into executable Rust code. It handles pipeline generation,
//! feature engineering transformations, and hyperparameter optimization setups.

use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

use crate::dsl_impl::dsl_types::{
    FeatureDefinition, FeatureEngineeringConfig, HyperparameterConfig, OptimizationStrategy,
    ParameterDef, ParameterDistribution, PerformanceConfig, PipelineConfig, PipelineStage,
    StageType,
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
            // Enable GPU acceleration if available
            #[cfg(feature = "gpu")]
            {
                crate::gpu::initialize_gpu_context()?;
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
}
