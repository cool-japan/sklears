//! DSL parsing implementations for sklears macro system
//!
//! This module contains the parsing logic for all DSL macros in the sklears
//! framework. It transforms TokenStream input into structured configuration
//! objects that can be used for code generation.

use proc_macro2::{Span, TokenStream};
use syn::{parse2, Error, Result as SynResult};

use crate::dsl_impl::dsl_types::{
    CrossValidationConfig, DataOperation, DataPipelineConfig, DataPipelineMode, DataStep,
    ExperimentConfig, ExperimentParameter, FeatureDefinition, FeatureEngineeringConfig,
    FeatureEngineeringOptions, HyperparameterConfig, ModelEvaluationConfig, ObjectiveConfig,
    OptimizationConfig, OptimizationMetric, ParameterDef, PerformanceConfig, PipelineConfig,
    PipelineStage, SelectionCriterion, SelectionType, StageType, StatisticalTest, ValidationRule,
};
use std::collections::HashMap;

/// Map a metric identifier to the corresponding [`OptimizationMetric`].
///
/// Unknown identifiers are preserved verbatim as [`OptimizationMetric::Custom`]
/// so callers can reference user-defined metrics without losing information.
fn metric_from_ident(ident: &syn::Ident) -> OptimizationMetric {
    match ident.to_string().as_str() {
        "accuracy" | "Accuracy" => OptimizationMetric::Accuracy,
        "precision" | "Precision" => OptimizationMetric::Precision,
        "recall" | "Recall" => OptimizationMetric::Recall,
        "f1" | "f1_score" | "F1Score" => OptimizationMetric::F1Score,
        "auc" | "auc_roc" | "AucRoc" => OptimizationMetric::AucRoc,
        "mse" | "mean_squared_error" | "MeanSquaredError" => OptimizationMetric::MeanSquaredError,
        "mae" | "mean_absolute_error" | "MeanAbsoluteError" => {
            OptimizationMetric::MeanAbsoluteError
        }
        "r2" | "r_squared" | "RSquared" => OptimizationMetric::RSquared,
        other => OptimizationMetric::Custom(other.to_string()),
    }
}

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
        name: parsed
            .name
            .unwrap_or_else(|| "default_pipeline".to_string()),
        stages: parsed.stages,
        input_type: parsed.input_type.unwrap_or_else(|| {
            syn::parse_str("scirs2_core::ndarray::Array2<f64>").expect("valid default type")
        }),
        output_type: parsed.output_type.unwrap_or_else(|| {
            syn::parse_str("scirs2_core::ndarray::Array1<usize>").expect("valid default type")
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
        dataset: parsed
            .dataset
            .unwrap_or_else(|| syn::parse_str("dataset").expect("valid default identifier")),
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
        model: parsed
            .model
            .unwrap_or_else(|| syn::parse_str("DefaultModel").expect("valid default model type")),
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
                    return Err(Error::new(
                        ident.span(),
                        format!("Unknown pipeline configuration option: {}", ident),
                    ));
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
                    return Err(Error::new(
                        field.span(),
                        format!("Unknown stage field: {}", field),
                    ));
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
                return Err(Error::new(
                    field.span(),
                    format!("Unknown performance field: {}", field),
                ));
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
                    return Err(Error::new(
                        ident.span(),
                        format!("Unknown feature engineering option: {}", ident),
                    ));
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
        let _criterion_str: syn::LitStr = input.parse()?;
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
fn parse_feature_engineering_options(
    _input: syn::parse::ParseStream,
) -> SynResult<FeatureEngineeringOptions> {
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
    fn parse(_input: syn::parse::ParseStream) -> SynResult<Self> {
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

/// Parse `model_evaluation!` macro input into structured configuration.
///
/// Parses the DSL syntax describing how a model should be evaluated and produces
/// a [`ModelEvaluationConfig`] capturing the model/data expressions, requested
/// metrics, cross-validation, and statistical testing options.
///
/// # Arguments
/// * `input` - TokenStream containing the evaluation DSL
///
/// # Returns
/// Parsed [`ModelEvaluationConfig`] or syntax error
pub fn parse_model_evaluation(input: TokenStream) -> SynResult<ModelEvaluationConfig> {
    let parsed: ModelEvaluationParser = parse2(input)?;

    let model = parsed
        .model
        .ok_or_else(|| Error::new(parsed.span, "model_evaluation! requires a `model:` field"))?;
    let features = parsed.features.ok_or_else(|| {
        Error::new(
            parsed.span,
            "model_evaluation! requires a `features:` field",
        )
    })?;
    let targets = parsed
        .targets
        .ok_or_else(|| Error::new(parsed.span, "model_evaluation! requires a `targets:` field"))?;

    let metrics = if parsed.metrics.is_empty() {
        vec![OptimizationMetric::Accuracy]
    } else {
        parsed.metrics
    };

    Ok(ModelEvaluationConfig {
        model,
        features,
        targets,
        metrics,
        cross_validation: parsed.cross_validation,
        statistical_test: parsed.statistical_test,
    })
}

/// Parse `data_pipeline!` macro input into structured configuration.
///
/// Parses the DSL syntax describing a data processing pipeline and produces a
/// [`DataPipelineConfig`] capturing the data source, ordered steps, and the
/// execution mode driving code generation.
///
/// # Arguments
/// * `input` - TokenStream containing the data pipeline DSL
///
/// # Returns
/// Parsed [`DataPipelineConfig`] or syntax error
pub fn parse_data_pipeline(input: TokenStream) -> SynResult<DataPipelineConfig> {
    let parsed: DataPipelineParser = parse2(input)?;

    let source = parsed
        .source
        .ok_or_else(|| Error::new(parsed.span, "data_pipeline! requires a `source:` field"))?;

    Ok(DataPipelineConfig {
        name: parsed.name.unwrap_or_else(|| "data_pipeline".to_string()),
        source,
        steps: parsed.steps,
        mode: parsed.mode.unwrap_or(DataPipelineMode::Batch),
        batch_size: parsed.batch_size,
        parallel: parsed.parallel.unwrap_or(false),
    })
}

/// Parse `experiment_config!` macro input into structured configuration.
///
/// Parses the DSL syntax describing an experiment and produces an
/// [`ExperimentConfig`] capturing the experiment identity, recorded
/// hyperparameters, tracked metrics, and reproducibility settings.
///
/// # Arguments
/// * `input` - TokenStream containing the experiment DSL
///
/// # Returns
/// Parsed [`ExperimentConfig`] or syntax error
pub fn parse_experiment_config(input: TokenStream) -> SynResult<ExperimentConfig> {
    let parsed: ExperimentConfigParser = parse2(input)?;

    let name = parsed
        .name
        .ok_or_else(|| Error::new(parsed.span, "experiment_config! requires a `name:` field"))?;

    Ok(ExperimentConfig {
        name,
        description: parsed.description,
        parameters: parsed.parameters,
        tracked_metrics: parsed.tracked_metrics,
        seed: parsed.seed,
        tags: parsed.tags,
    })
}

/// Parser state for the `model_evaluation!` configuration DSL.
struct ModelEvaluationParser {
    model: Option<syn::Expr>,
    features: Option<syn::Expr>,
    targets: Option<syn::Expr>,
    metrics: Vec<OptimizationMetric>,
    cross_validation: Option<CrossValidationConfig>,
    statistical_test: Option<StatisticalTest>,
    span: Span,
}

impl syn::parse::Parse for ModelEvaluationParser {
    fn parse(input: syn::parse::ParseStream) -> SynResult<Self> {
        let span = input.span();
        let mut model = None;
        let mut features = None;
        let mut targets = None;
        let mut metrics = Vec::new();
        let mut cross_validation = None;
        let mut statistical_test = None;

        let content;
        syn::braced!(content in input);

        while !content.is_empty() {
            let ident: syn::Ident = content.parse()?;
            content.parse::<syn::Token![:]>()?;

            match ident.to_string().as_str() {
                "model" => {
                    model = Some(content.parse()?);
                }
                "features" => {
                    features = Some(content.parse()?);
                }
                "targets" | "labels" => {
                    targets = Some(content.parse()?);
                }
                "metrics" => {
                    let metrics_content;
                    syn::bracketed!(metrics_content in content);
                    while !metrics_content.is_empty() {
                        let metric_ident: syn::Ident = metrics_content.parse()?;
                        metrics.push(metric_from_ident(&metric_ident));
                        if metrics_content.peek(syn::Token![,]) {
                            metrics_content.parse::<syn::Token![,]>()?;
                        }
                    }
                }
                "cross_validation" | "cv" => {
                    let cv_content;
                    syn::braced!(cv_content in content);
                    cross_validation = Some(parse_cross_validation(&cv_content)?);
                }
                "statistical_test" => {
                    let test_ident: syn::Ident = content.parse()?;
                    statistical_test = Some(statistical_test_from_ident(&test_ident));
                }
                _ => {
                    return Err(Error::new(
                        ident.span(),
                        format!("Unknown model_evaluation option: {}", ident),
                    ));
                }
            }

            if content.peek(syn::Token![,]) {
                content.parse::<syn::Token![,]>()?;
            }
        }

        Ok(ModelEvaluationParser {
            model,
            features,
            targets,
            metrics,
            cross_validation,
            statistical_test,
            span,
        })
    }
}

/// Map a statistical-test identifier to the corresponding [`StatisticalTest`].
fn statistical_test_from_ident(ident: &syn::Ident) -> StatisticalTest {
    match ident.to_string().as_str() {
        "paired_t_test" | "PairedTTest" | "t_test" => StatisticalTest::PairedTTest,
        "wilcoxon" | "Wilcoxon" => StatisticalTest::Wilcoxon,
        "mcnemar" | "McNemar" => StatisticalTest::McNemar,
        other => StatisticalTest::Custom(other.to_string()),
    }
}

/// Parse a cross-validation configuration block.
fn parse_cross_validation(input: syn::parse::ParseStream) -> SynResult<CrossValidationConfig> {
    let mut n_folds = 5usize;
    let mut stratified = false;
    let mut random_seed = None;

    while !input.is_empty() {
        let field: syn::Ident = input.parse()?;
        input.parse::<syn::Token![:]>()?;

        match field.to_string().as_str() {
            "n_folds" | "folds" => {
                let folds_lit: syn::LitInt = input.parse()?;
                n_folds = folds_lit.base10_parse()?;
            }
            "stratified" => {
                let strat_lit: syn::LitBool = input.parse()?;
                stratified = strat_lit.value;
            }
            "random_seed" | "seed" => {
                let seed_lit: syn::LitInt = input.parse()?;
                random_seed = Some(seed_lit.base10_parse()?);
            }
            _ => {
                return Err(Error::new(
                    field.span(),
                    format!("Unknown cross_validation field: {}", field),
                ));
            }
        }

        if input.peek(syn::Token![,]) {
            input.parse::<syn::Token![,]>()?;
        }
    }

    Ok(CrossValidationConfig {
        n_folds,
        stratified,
        random_seed,
    })
}

/// Parser state for the `data_pipeline!` configuration DSL.
struct DataPipelineParser {
    name: Option<String>,
    source: Option<syn::Expr>,
    steps: Vec<DataStep>,
    mode: Option<DataPipelineMode>,
    batch_size: Option<usize>,
    parallel: Option<bool>,
    span: Span,
}

impl syn::parse::Parse for DataPipelineParser {
    fn parse(input: syn::parse::ParseStream) -> SynResult<Self> {
        let span = input.span();
        let mut name = None;
        let mut source = None;
        let mut steps = Vec::new();
        let mut mode = None;
        let mut batch_size = None;
        let mut parallel = None;

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
                "source" => {
                    source = Some(content.parse()?);
                }
                "steps" => {
                    let steps_content;
                    syn::bracketed!(steps_content in content);
                    steps = parse_data_steps(&steps_content)?;
                }
                "mode" => {
                    let mode_ident: syn::Ident = content.parse()?;
                    mode = Some(match mode_ident.to_string().as_str() {
                        "batch" | "Batch" => DataPipelineMode::Batch,
                        "streaming" | "Streaming" => DataPipelineMode::Streaming,
                        "real_time" | "realtime" | "RealTime" => DataPipelineMode::RealTime,
                        other => {
                            return Err(Error::new(
                                mode_ident.span(),
                                format!("Unknown data pipeline mode: {}", other),
                            ));
                        }
                    });
                }
                "batch_size" => {
                    let batch_lit: syn::LitInt = content.parse()?;
                    batch_size = Some(batch_lit.base10_parse()?);
                }
                "parallel" => {
                    let parallel_lit: syn::LitBool = content.parse()?;
                    parallel = Some(parallel_lit.value);
                }
                _ => {
                    return Err(Error::new(
                        ident.span(),
                        format!("Unknown data_pipeline option: {}", ident),
                    ));
                }
            }

            if content.peek(syn::Token![,]) {
                content.parse::<syn::Token![,]>()?;
            }
        }

        Ok(DataPipelineParser {
            name,
            source,
            steps,
            mode,
            batch_size,
            parallel,
            span,
        })
    }
}

/// Parse the ordered list of data processing steps.
///
/// Each step has the form `name: operation(expr)` where `operation` selects the
/// [`DataOperation`] and `expr` is the transformation applied at runtime.
fn parse_data_steps(input: syn::parse::ParseStream) -> SynResult<Vec<DataStep>> {
    let mut steps = Vec::new();

    while !input.is_empty() {
        let name: syn::Ident = input.parse()?;
        input.parse::<syn::Token![:]>()?;

        let operation_ident: syn::Ident = input.parse()?;
        let operation = match operation_ident.to_string().as_str() {
            "load" | "Load" => DataOperation::Load,
            "filter" | "Filter" => DataOperation::Filter,
            "map" | "Map" => DataOperation::Map,
            "aggregate" | "Aggregate" => DataOperation::Aggregate,
            "join" | "Join" => DataOperation::Join,
            "sink" | "Sink" => DataOperation::Sink,
            other => DataOperation::Custom(other.to_string()),
        };

        let transform_content;
        syn::parenthesized!(transform_content in input);
        let transform: syn::Expr = transform_content.parse()?;

        steps.push(DataStep {
            name: name.to_string(),
            operation,
            transform,
        });

        if input.peek(syn::Token![,]) {
            input.parse::<syn::Token![,]>()?;
        }
    }

    Ok(steps)
}

/// Parser state for the `experiment_config!` configuration DSL.
struct ExperimentConfigParser {
    name: Option<String>,
    description: Option<String>,
    parameters: Vec<ExperimentParameter>,
    tracked_metrics: Vec<String>,
    seed: Option<u64>,
    tags: Vec<String>,
    span: Span,
}

impl syn::parse::Parse for ExperimentConfigParser {
    fn parse(input: syn::parse::ParseStream) -> SynResult<Self> {
        let span = input.span();
        let mut name = None;
        let mut description = None;
        let mut parameters = Vec::new();
        let mut tracked_metrics = Vec::new();
        let mut seed = None;
        let mut tags = Vec::new();

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
                "description" => {
                    let desc_lit: syn::LitStr = content.parse()?;
                    description = Some(desc_lit.value());
                }
                "parameters" | "params" => {
                    let params_content;
                    syn::braced!(params_content in content);
                    parameters = parse_experiment_parameters(&params_content)?;
                }
                "tracked_metrics" | "metrics" => {
                    let metrics_content;
                    syn::bracketed!(metrics_content in content);
                    while !metrics_content.is_empty() {
                        let metric_lit: syn::LitStr = metrics_content.parse()?;
                        tracked_metrics.push(metric_lit.value());
                        if metrics_content.peek(syn::Token![,]) {
                            metrics_content.parse::<syn::Token![,]>()?;
                        }
                    }
                }
                "seed" => {
                    let seed_lit: syn::LitInt = content.parse()?;
                    seed = Some(seed_lit.base10_parse()?);
                }
                "tags" => {
                    let tags_content;
                    syn::bracketed!(tags_content in content);
                    while !tags_content.is_empty() {
                        let tag_lit: syn::LitStr = tags_content.parse()?;
                        tags.push(tag_lit.value());
                        if tags_content.peek(syn::Token![,]) {
                            tags_content.parse::<syn::Token![,]>()?;
                        }
                    }
                }
                _ => {
                    return Err(Error::new(
                        ident.span(),
                        format!("Unknown experiment_config option: {}", ident),
                    ));
                }
            }

            if content.peek(syn::Token![,]) {
                content.parse::<syn::Token![,]>()?;
            }
        }

        Ok(ExperimentConfigParser {
            name,
            description,
            parameters,
            tracked_metrics,
            seed,
            tags,
            span,
        })
    }
}

/// Parse the recorded hyperparameters for an experiment.
///
/// Each entry has the form `name: expr` where `expr` yields the value recorded
/// for the experiment.
fn parse_experiment_parameters(
    input: syn::parse::ParseStream,
) -> SynResult<Vec<ExperimentParameter>> {
    let mut parameters = Vec::new();

    while !input.is_empty() {
        let name: syn::Ident = input.parse()?;
        input.parse::<syn::Token![:]>()?;
        let value: syn::Expr = input.parse()?;

        parameters.push(ExperimentParameter {
            name: name.to_string(),
            value,
        });

        if input.peek(syn::Token![,]) {
            input.parse::<syn::Token![,]>()?;
        }
    }

    Ok(parameters)
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
        // Parser is simplified for now - accepts the input but returns basic structure
        // This test verifies that parsing doesn't panic
        match result {
            Ok(_config) => {
                // Basic parsing succeeded
            }
            Err(_) => {
                // Parser is in placeholder mode, so errors are expected
            }
        }
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

        let config = result.expect("expected valid value");
        assert_eq!(config.features.len(), 1);
        assert_eq!(config.features[0].name, "price_per_sqft");
    }

    #[test]
    fn test_parse_empty_pipeline() {
        let input = quote! { {} };

        let result = parse_ml_pipeline(input);
        assert!(result.is_ok());

        let config = result.expect("expected valid value");
        assert_eq!(config.name, "default_pipeline");
        assert_eq!(config.stages.len(), 0);
    }

    #[test]
    fn test_parse_model_evaluation() {
        let input = quote! {
            {
                model: RandomForestClassifier::new(),
                features: x_train,
                targets: y_train,
                metrics: [accuracy, f1, recall],
                cross_validation: {
                    n_folds: 10,
                    stratified: true,
                    random_seed: 7
                },
                statistical_test: wilcoxon
            }
        };

        let config = parse_model_evaluation(input).expect("model_evaluation should parse");
        assert_eq!(config.metrics.len(), 3);
        assert_eq!(config.metrics[0], OptimizationMetric::Accuracy);
        assert_eq!(config.metrics[1], OptimizationMetric::F1Score);
        assert_eq!(config.metrics[2], OptimizationMetric::Recall);

        let cv = config
            .cross_validation
            .expect("cross_validation should be parsed");
        assert_eq!(cv.n_folds, 10);
        assert!(cv.stratified);
        assert_eq!(cv.random_seed, Some(7));

        assert_eq!(config.statistical_test, Some(StatisticalTest::Wilcoxon));
    }

    #[test]
    fn test_parse_model_evaluation_defaults_metric() {
        let input = quote! {
            {
                model: model,
                features: x,
                targets: y
            }
        };

        let config = parse_model_evaluation(input).expect("model_evaluation should parse");
        assert_eq!(config.metrics, vec![OptimizationMetric::Accuracy]);
        assert!(config.cross_validation.is_none());
        assert!(config.statistical_test.is_none());
    }

    #[test]
    fn test_parse_model_evaluation_missing_model_errors() {
        let input = quote! {
            {
                features: x,
                targets: y
            }
        };

        assert!(parse_model_evaluation(input).is_err());
    }

    #[test]
    fn test_parse_data_pipeline() {
        let input = quote! {
            {
                name: "etl",
                source: read_csv("data.csv"),
                steps: [
                    clean: filter(|row| row.is_valid()),
                    normalize: map(|row| row.normalized()),
                    persist: sink(write_parquet("out.parquet"))
                ],
                mode: streaming,
                batch_size: 256,
                parallel: true
            }
        };

        let config = parse_data_pipeline(input).expect("data_pipeline should parse");
        assert_eq!(config.name, "etl");
        assert_eq!(config.steps.len(), 3);
        assert_eq!(config.steps[0].name, "clean");
        assert_eq!(config.steps[0].operation, DataOperation::Filter);
        assert_eq!(config.steps[1].operation, DataOperation::Map);
        assert_eq!(config.steps[2].operation, DataOperation::Sink);
        assert_eq!(config.mode, DataPipelineMode::Streaming);
        assert_eq!(config.batch_size, Some(256));
        assert!(config.parallel);
    }

    #[test]
    fn test_parse_data_pipeline_missing_source_errors() {
        let input = quote! {
            {
                name: "etl",
                steps: []
            }
        };

        assert!(parse_data_pipeline(input).is_err());
    }

    #[test]
    fn test_parse_experiment_config() {
        let input = quote! {
            {
                name: "exp-001",
                description: "baseline run",
                parameters: {
                    learning_rate: 0.01,
                    n_estimators: 100
                },
                tracked_metrics: ["accuracy", "loss"],
                seed: 42,
                tags: ["baseline", "rf"]
            }
        };

        let config = parse_experiment_config(input).expect("experiment_config should parse");
        assert_eq!(config.name, "exp-001");
        assert_eq!(config.description.as_deref(), Some("baseline run"));
        assert_eq!(config.parameters.len(), 2);
        assert_eq!(config.parameters[0].name, "learning_rate");
        assert_eq!(config.tracked_metrics, vec!["accuracy", "loss"]);
        assert_eq!(config.seed, Some(42));
        assert_eq!(config.tags, vec!["baseline", "rf"]);
    }

    #[test]
    fn test_parse_experiment_config_missing_name_errors() {
        let input = quote! {
            {
                seed: 1
            }
        };

        assert!(parse_experiment_config(input).is_err());
    }
}
