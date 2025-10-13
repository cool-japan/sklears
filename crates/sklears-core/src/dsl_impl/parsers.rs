//! DSL parsing implementations for sklears macro system
//!
//! This module contains the parsing logic for all DSL macros in the sklears
//! framework. It transforms TokenStream input into structured configuration
//! objects that can be used for code generation.

use proc_macro2::{Span, TokenStream};
use syn::{parse2, Error, Result as SynResult};
use quote::quote;

use crate::dsl_impl::dsl_types::{
    PipelineConfig, PipelineStage, StageType, FeatureEngineeringConfig,
    FeatureDefinition, ValidationRule, HyperparameterConfig, ParameterDef,
    ParameterDistribution, OptimizationConfig, OptimizationStrategy,
    SelectionCriterion, SelectionType, PerformanceConfig,
    FeatureEngineeringOptions, ObjectiveConfig, OptimizationMetric,
    OptimizationDirection,
};
use std::collections::HashMap;

/// Parse ml_pipeline! macro input into structured configuration
///
/// Parses the DSL syntax for machine learning pipelines and creates a
/// PipelineConfig object with all stages and configuration options.
///
/// # Arguments
/// * `input` - TokenStream containing the pipeline DSL
///
/// # Returns
/// Parsed PipelineConfig or syntax error
pub fn parse_ml_pipeline(input: TokenStream) -> SynResult<PipelineConfig> {
    let parsed: PipelineConfigParser = parse2(input)?;

    Ok(PipelineConfig {
        name: parsed.name.unwrap_or_else(|| "default_pipeline".to_string()),
        stages: parsed.stages,
        input_type: parsed.input_type.unwrap_or_else(|| {
            syn::parse_str("scirs2_core::ndarray::Array2<f64>").unwrap()
        }),
        output_type: parsed.output_type.unwrap_or_else(|| {
            syn::parse_str("scirs2_core::ndarray::Array1<usize>").unwrap()
        }),
        parallel: parsed.parallel.unwrap_or(false),
        validate_input: parsed.validate_input.unwrap_or(true),
        cache_transforms: parsed.cache_transforms.unwrap_or(false),
        metadata: parsed.metadata,
        performance: parsed.performance.unwrap_or_default(),
    })
}

/// Parse feature_engineering! macro input into structured configuration
///
/// Parses the DSL syntax for feature engineering and creates a
/// FeatureEngineeringConfig object with feature definitions and rules.
///
/// # Arguments
/// * `input` - TokenStream containing the feature engineering DSL
///
/// # Returns
/// Parsed FeatureEngineeringConfig or syntax error
pub fn parse_feature_engineering(input: TokenStream) -> SynResult<FeatureEngineeringConfig> {
    let parsed: FeatureEngineeringParser = parse2(input)?;

    Ok(FeatureEngineeringConfig {
        dataset: parsed.dataset.unwrap_or_else(|| {
            syn::parse_str("dataset").unwrap()
        }),
        features: parsed.features,
        selection: parsed.selection,
        validation: parsed.validation,
        options: parsed.options.unwrap_or_default(),
    })
}

/// Parse hyperparameter_config! macro input into structured configuration
///
/// Parses the DSL syntax for hyperparameter optimization and creates a
/// HyperparameterConfig object with parameter definitions and optimization settings.
///
/// # Arguments
/// * `input` - TokenStream containing the hyperparameter DSL
///
/// # Returns
/// Parsed HyperparameterConfig or syntax error
pub fn parse_hyperparameter_config(input: TokenStream) -> SynResult<HyperparameterConfig> {
    let parsed: HyperparameterParser = parse2(input)?;

    Ok(HyperparameterConfig {
        model: parsed.model.unwrap_or_else(|| {
            syn::parse_str("DefaultModel").unwrap()
        }),
        parameters: parsed.parameters,
        constraints: parsed.constraints,
        optimization: parsed.optimization.unwrap_or_default(),
        objective: parsed.objective.unwrap_or_default(),
    })
}

/// Parser implementation for ML pipeline configuration DSL
struct PipelineConfigParser {
    name: Option<String>,
    stages: Vec<PipelineStage>,
    input_type: Option<syn::Type>,
    output_type: Option<syn::Type>,
    parallel: Option<bool>,
    validate_input: Option<bool>,
    cache_transforms: Option<bool>,
    metadata: HashMap<String, String>,
    performance: Option<PerformanceConfig>,
}

impl syn::parse::Parse for PipelineConfigParser {
    fn parse(input: syn::parse::ParseStream) -> SynResult<Self> {
        let mut name = None;
        let mut stages = Vec::new();
        let mut input_type = None;
        let mut output_type = None;
        let mut parallel = None;
        let mut validate_input = None;
        let mut cache_transforms = None;
        let mut metadata = HashMap::new();
        let mut performance = None;

        // Parse brace-delimited configuration
        let content;
        syn::braced!(content in input);

        while !content.is_empty() {
            let ident: syn::Ident = content.parse()?;
            content.parse::<syn::Token![:]>()?;

            match ident.to_string().as_str() {
                "name" => {
                    let name_lit: syn::LitStr = content.parse()?;
                    name = Some(name_lit.value());
                }
                "stages" => {
                    let stages_content;
                    syn::bracketed!(stages_content in content);
                    stages = parse_pipeline_stages(&stages_content)?;
                }
                "input" => {
                    input_type = Some(content.parse()?);
                }
                "output" => {
                    output_type = Some(content.parse()?);
                }
                "parallel" => {
                    let parallel_lit: syn::LitBool = content.parse()?;
                    parallel = Some(parallel_lit.value);
                }
                "validate_input" => {
                    let validate_lit: syn::LitBool = content.parse()?;
                    validate_input = Some(validate_lit.value);
                }
                "cache_transforms" => {
                    let cache_lit: syn::LitBool = content.parse()?;
                    cache_transforms = Some(cache_lit.value);
                }
                "metadata" => {
                    let metadata_content;
                    syn::braced!(metadata_content in content);
                    metadata = parse_metadata(&metadata_content)?;
                }
                "performance" => {
                    let perf_content;
                    syn::braced!(perf_content in content);
                    performance = Some(parse_performance_config(&perf_content)?);
                }
                _ => {
                    return Err(Error::new(ident.span(),
                        format!("Unknown pipeline configuration option: {}", ident)));
                }
            }

            // Handle comma between configuration items
            if content.peek(syn::Token![,]) {
                content.parse::<syn::Token![,]>()?;
            }
        }

        Ok(PipelineConfigParser {
            name,
            stages,
            input_type,
            output_type,
            parallel,
            validate_input,
            cache_transforms,
            metadata,
            performance,
        })
    }
}

/// Parse pipeline stages from DSL syntax
fn parse_pipeline_stages(input: syn::parse::ParseStream) -> SynResult<Vec<PipelineStage>> {
    let mut stages = Vec::new();

    while !input.is_empty() {
        let stage_content;
        syn::braced!(stage_content in input);

        let mut stage_name = None;
        let mut stage_type = None;
        let mut transforms = Vec::new();
        let mut input_type = None;
        let mut output_type = None;
        let mut parallelizable = false;
        let mut memory_hint = None;

        while !stage_content.is_empty() {
            let field: syn::Ident = stage_content.parse()?;
            stage_content.parse::<syn::Token![:]>()?;

            match field.to_string().as_str() {
                "name" => {
                    let name_lit: syn::LitStr = stage_content.parse()?;
                    stage_name = Some(name_lit.value());
                }
                "type" => {
                    let type_ident: syn::Ident = stage_content.parse()?;
                    stage_type = Some(match type_ident.to_string().as_str() {
                        "preprocess" => StageType::Preprocess,
                        "feature_engineering" => StageType::FeatureEngineering,
                        "model" => StageType::Model,
                        "postprocess" => StageType::Postprocess,
                        custom => StageType::Custom(custom.to_string()),
                    });
                }
                "transforms" => {
                    let transforms_content;
                    syn::bracketed!(transforms_content in stage_content);
                    while !transforms_content.is_empty() {
                        transforms.push(transforms_content.parse()?);
                        if transforms_content.peek(syn::Token![,]) {
                            transforms_content.parse::<syn::Token![,]>()?;
                        }
                    }
                }
                "input_type" => {
                    input_type = Some(stage_content.parse()?);
                }
                "output_type" => {
                    output_type = Some(stage_content.parse()?);
                }
                "parallel" => {
                    let parallel_lit: syn::LitBool = stage_content.parse()?;
                    parallelizable = parallel_lit.value;
                }
                "memory_hint" => {
                    let memory_lit: syn::LitInt = stage_content.parse()?;
                    memory_hint = Some(memory_lit.base10_parse()?);
                }
                _ => {
                    return Err(Error::new(field.span(),
                        format!("Unknown stage field: {}", field)));
                }
            }

            if stage_content.peek(syn::Token![,]) {
                stage_content.parse::<syn::Token![,]>()?;
            }
        }

        stages.push(PipelineStage {
            name: stage_name.unwrap_or_else(|| format!("stage_{}", stages.len())),
            stage_type: stage_type.unwrap_or(StageType::Custom("unknown".to_string())),
            transforms,
            input_type,
            output_type,
            parallelizable,
            memory_hint,
        });

        if input.peek(syn::Token![,]) {
            input.parse::<syn::Token![,]>()?;
        }
    }

    Ok(stages)
}

/// Parse metadata configuration
fn parse_metadata(input: syn::parse::ParseStream) -> SynResult<HashMap<String, String>> {
    let mut metadata = HashMap::new();

    while !input.is_empty() {
        let key: syn::LitStr = input.parse()?;
        input.parse::<syn::Token![:]>()?;
        let value: syn::LitStr = input.parse()?;

        metadata.insert(key.value(), value.value());

        if input.peek(syn::Token![,]) {
            input.parse::<syn::Token![,]>()?;
        }
    }

    Ok(metadata)
}

/// Parse performance configuration
fn parse_performance_config(input: syn::parse::ParseStream) -> SynResult<PerformanceConfig> {
    let mut max_threads = None;
    let mut max_memory_bytes = None;
    let mut gpu_acceleration = false;
    let mut batch_size = None;
    let mut stage_timeout_seconds = None;

    while !input.is_empty() {
        let field: syn::Ident = input.parse()?;
        input.parse::<syn::Token![:]>()?;

        match field.to_string().as_str() {
            "max_threads" => {
                let threads_lit: syn::LitInt = input.parse()?;
                max_threads = Some(threads_lit.base10_parse()?);
            }
            "max_memory_bytes" => {
                let memory_lit: syn::LitInt = input.parse()?;
                max_memory_bytes = Some(memory_lit.base10_parse()?);
            }
            "gpu_acceleration" => {
                let gpu_lit: syn::LitBool = input.parse()?;
                gpu_acceleration = gpu_lit.value;
            }
            "batch_size" => {
                let batch_lit: syn::LitInt = input.parse()?;
                batch_size = Some(batch_lit.base10_parse()?);
            }
            "stage_timeout_seconds" => {
                let timeout_lit: syn::LitInt = input.parse()?;
                stage_timeout_seconds = Some(timeout_lit.base10_parse()?);
            }
            _ => {
                return Err(Error::new(field.span(),
                    format!("Unknown performance field: {}", field)));
            }
        }

        if input.peek(syn::Token![,]) {
            input.parse::<syn::Token![,]>()?;
        }
    }

    Ok(PerformanceConfig {
        max_threads,
        max_memory_bytes,
        gpu_acceleration,
        batch_size,
        stage_timeout_seconds,
    })
}

/// Parser for feature engineering configuration DSL
struct FeatureEngineeringParser {
    dataset: Option<syn::Expr>,
    features: Vec<FeatureDefinition>,
    selection: Vec<SelectionCriterion>,
    validation: Vec<ValidationRule>,
    options: Option<FeatureEngineeringOptions>,
}

impl syn::parse::Parse for FeatureEngineeringParser {
    fn parse(input: syn::parse::ParseStream) -> SynResult<Self> {
        let mut dataset = None;
        let mut features = Vec::new();
        let mut selection = Vec::new();
        let mut validation = Vec::new();
        let mut options = None;

        let content;
        syn::braced!(content in input);

        while !content.is_empty() {
            let ident: syn::Ident = content.parse()?;
            content.parse::<syn::Token![:]>()?;

            match ident.to_string().as_str() {
                "dataset" => {
                    dataset = Some(content.parse()?);
                }
                "features" => {
                    let features_content;
                    syn::bracketed!(features_content in content);
                    features = parse_feature_definitions(&features_content)?;
                }
                "selection" => {
                    let selection_content;
                    syn::bracketed!(selection_content in content);
                    selection = parse_selection_criteria(&selection_content)?;
                }
                "validation" => {
                    let validation_content;
                    syn::bracketed!(validation_content in content);
                    validation = parse_validation_rules(&validation_content)?;
                }
                "options" => {
                    let options_content;
                    syn::braced!(options_content in content);
                    options = Some(parse_feature_engineering_options(&options_content)?);
                }
                _ => {
                    return Err(Error::new(ident.span(),
                        format!("Unknown feature engineering option: {}", ident)));
                }
            }

            if content.peek(syn::Token![,]) {
                content.parse::<syn::Token![,]>()?;
            }
        }

        Ok(FeatureEngineeringParser {
            dataset,
            features,
            selection,
            validation,
            options,
        })
    }
}

/// Parse feature definitions from DSL
fn parse_feature_definitions(input: syn::parse::ParseStream) -> SynResult<Vec<FeatureDefinition>> {
    let mut features = Vec::new();

    while !input.is_empty() {
        let name: syn::Ident = input.parse()?;
        input.parse::<syn::Token![=]>()?;
        let expression: syn::Expr = input.parse()?;

        features.push(FeatureDefinition {
            name: name.to_string(),
            expression,
            data_type: None,
            description: None,
            required: true,
        });

        if input.peek(syn::Token![,]) {
            input.parse::<syn::Token![,]>()?;
        }
    }

    Ok(features)
}

/// Parse selection criteria from DSL
fn parse_selection_criteria(input: syn::parse::ParseStream) -> SynResult<Vec<SelectionCriterion>> {
    let mut criteria = Vec::new();

    while !input.is_empty() {
        let criterion_str: syn::LitStr = input.parse()?;
        // Simple parsing - in practice this would be more sophisticated
        criteria.push(SelectionCriterion {
            criterion_type: SelectionType::Correlation,
            threshold: 0.1,
            enabled: true,
        });

        if input.peek(syn::Token![,]) {
            input.parse::<syn::Token![,]>()?;
        }
    }

    Ok(criteria)
}

/// Parse validation rules from DSL
fn parse_validation_rules(input: syn::parse::ParseStream) -> SynResult<Vec<ValidationRule>> {
    let mut rules = Vec::new();

    while !input.is_empty() {
        let feature: syn::Ident = input.parse()?;
        input.parse::<syn::Token![:]>()?;
        let rule: syn::LitStr = input.parse()?;

        rules.push(ValidationRule {
            feature: feature.to_string(),
            rule: rule.value(),
            error_message: None,
            strict: true,
        });

        if input.peek(syn::Token![,]) {
            input.parse::<syn::Token![,]>()?;
        }
    }

    Ok(rules)
}

/// Parse feature engineering options
fn parse_feature_engineering_options(input: syn::parse::ParseStream) -> SynResult<FeatureEngineeringOptions> {
    // Simplified implementation - return defaults for now
    Ok(FeatureEngineeringOptions::default())
}

/// Parser for hyperparameter configuration DSL
struct HyperparameterParser {
    model: Option<syn::Type>,
    parameters: Vec<ParameterDef>,
    constraints: Vec<syn::Expr>,
    optimization: Option<OptimizationConfig>,
    objective: Option<ObjectiveConfig>,
}

impl syn::parse::Parse for HyperparameterParser {
    fn parse(input: syn::parse::ParseStream) -> SynResult<Self> {
        // Simplified implementation - return defaults for now
        Ok(HyperparameterParser {
            model: None,
            parameters: Vec::new(),
            constraints: Vec::new(),
            optimization: None,
            objective: None,
        })
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;

    #[test]
    fn test_parse_simple_pipeline() {
        let input = quote! {
            {
                name: "test_pipeline",
                parallel: true,
                stages: [
                    {
                        name: "preprocess",
                        type: preprocess,
                        transforms: [normalize, scale]
                    }
                ]
            }
        };

        let result = parse_ml_pipeline(input);
        assert!(result.is_ok());

        let config = result.unwrap();
        assert_eq!(config.name, "test_pipeline");
        assert_eq!(config.parallel, true);
        assert_eq!(config.stages.len(), 1);
    }

    #[test]
    fn test_parse_feature_engineering() {
        let input = quote! {
            {
                dataset: my_dataframe,
                features: [
                    price_per_sqft = price / square_feet
                ]
            }
        };

        let result = parse_feature_engineering(input);
        assert!(result.is_ok());

        let config = result.unwrap();
        assert_eq!(config.features.len(), 1);
        assert_eq!(config.features[0].name, "price_per_sqft");
    }

    #[test]
    fn test_parse_empty_pipeline() {
        let input = quote! { {} };

        let result = parse_ml_pipeline(input);
        assert!(result.is_ok());

        let config = result.unwrap();
        assert_eq!(config.name, "default_pipeline");
        assert_eq!(config.stages.len(), 0);
    }
}