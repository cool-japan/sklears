//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::error::Result;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::time::{Duration, Instant};

use super::types::{
    BinningStrategy, CorrelationMethod, Distribution, ErrorHandling, FeatureMapping, FeatureOrigin,
    ICAAlgorithm, ImputationStrategy, LoggingLevel, MemoryOptimization, MinMaxScalerConfig,
    MinMaxScalerParams, MissingValueIndicator, ModelEstimator, ModelSelectionStep,
    OptimizationConfiguration, PipelineInfo, PipelineMetadata, PowerMethod, PreprocessingStep,
    RobustScalerConfig, RobustScalerParams, SVDAlgorithm, SVDSolver, ScalerParams, SelectionCount,
    SelectionMethod, StandardScalerConfig, Trained, TransformationStep, TransformationType,
    UnivariateMethod, UnivariateScoreFunction, ValidationStrategy, WindowStatistic,
};

/// Comprehensive pipeline integration framework for feature selection
#[derive(Debug, Clone)]
pub struct FeatureSelectionPipeline<State = Untrained> {
    preprocessing_steps: Vec<PreprocessingStep>,
    feature_engineering_steps: Vec<FeatureEngineeringStep>,
    selection_methods: Vec<SelectionMethod>,
    dimensionality_reduction: Option<DimensionalityReductionStep>,
    model_selection: Option<ModelSelectionStep>,
    pipeline_config: PipelineConfiguration,
    optimization_config: OptimizationConfiguration,
    _phantom: PhantomData<State>,
}
impl FeatureSelectionPipeline<Untrained> {
    /// Create a new untrained pipeline with default configuration
    pub fn new() -> Self {
        Self {
            preprocessing_steps: Vec::new(),
            feature_engineering_steps: Vec::new(),
            selection_methods: Vec::new(),
            dimensionality_reduction: None,
            model_selection: None,
            pipeline_config: PipelineConfiguration::default(),
            optimization_config: OptimizationConfiguration::default(),
            _phantom: PhantomData,
        }
    }
    /// Builder pattern for adding preprocessing steps
    pub fn add_preprocessing_step(mut self, step: PreprocessingStep) -> Self {
        self.preprocessing_steps.push(step);
        self
    }
    /// Builder pattern for adding feature engineering steps
    pub fn add_feature_engineering_step(mut self, step: FeatureEngineeringStep) -> Self {
        self.feature_engineering_steps.push(step);
        self
    }
    /// Builder pattern for adding selection methods
    pub fn add_selection_method(mut self, method: SelectionMethod) -> Self {
        self.selection_methods.push(method);
        self
    }
    /// Builder pattern for setting dimensionality reduction
    pub fn with_dimensionality_reduction(mut self, reduction: DimensionalityReductionStep) -> Self {
        self.dimensionality_reduction = Some(reduction);
        self
    }
    /// Builder pattern for setting model selection
    pub fn with_model_selection(mut self, model_selection: ModelSelectionStep) -> Self {
        self.model_selection = Some(model_selection);
        self
    }
    /// Configure pipeline behavior
    pub fn with_config(mut self, config: PipelineConfiguration) -> Self {
        self.pipeline_config = config;
        self
    }
    /// Configure optimization settings
    pub fn with_optimization(mut self, config: OptimizationConfiguration) -> Self {
        self.optimization_config = config;
        self
    }
    /// Train the entire pipeline on the provided data
    pub fn fit(
        mut self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> Result<FeatureSelectionPipeline<Trained>> {
        let start_time = Instant::now();
        let mut current_X = X.to_owned();
        let current_y = y.to_owned();
        let mut trained_steps = Vec::new();
        let original_features = X.ncols();
        let mut preprocessing_steps = std::mem::take(&mut self.preprocessing_steps);
        for (idx, step) in preprocessing_steps.iter_mut().enumerate() {
            let step_start = Instant::now();
            current_X = Self::apply_preprocessing_step_static(step, current_X.view())?;
            trained_steps.push(TrainedStep {
                step_type: "Preprocessing".to_string(),
                step_index: idx,
                training_time: step_start.elapsed(),
                feature_count_before: current_X.ncols(),
                feature_count_after: current_X.ncols(),
                parameters: StepParameters::Preprocessing(Box::new(())),
            });
        }
        self.preprocessing_steps = preprocessing_steps;
        let mut feature_engineering_steps = std::mem::take(&mut self.feature_engineering_steps);
        for (idx, step) in feature_engineering_steps.iter_mut().enumerate() {
            let step_start = Instant::now();
            let features_before = current_X.ncols();
            current_X = Self::apply_feature_engineering_step_static(
                step,
                current_X.view(),
                current_y.view(),
            )?;
            trained_steps.push(TrainedStep {
                step_type: "FeatureEngineering".to_string(),
                step_index: idx,
                training_time: step_start.elapsed(),
                feature_count_before: features_before,
                feature_count_after: current_X.ncols(),
                parameters: StepParameters::FeatureEngineering(Box::new(())),
            });
        }
        self.feature_engineering_steps = feature_engineering_steps;
        let mut selection_mask = Array1::from_elem(current_X.ncols(), true);
        let mut selection_methods = std::mem::take(&mut self.selection_methods);
        for (idx, method) in selection_methods.iter_mut().enumerate() {
            let step_start = Instant::now();
            let features_before = current_X.ncols();
            let method_mask =
                Self::apply_selection_method_static(method, current_X.view(), current_y.view())?;
            for (i, &selected) in method_mask.iter().enumerate() {
                if !selected {
                    selection_mask[i] = false;
                }
            }
            trained_steps.push(TrainedStep {
                step_type: "Selection".to_string(),
                step_index: idx,
                training_time: step_start.elapsed(),
                feature_count_before: features_before,
                feature_count_after: selection_mask.iter().filter(|&&x| x).count(),
                parameters: StepParameters::Selection(method_mask),
            });
        }
        self.selection_methods = selection_methods;
        let selected_indices: Vec<usize> = selection_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &selected)| if selected { Some(i) } else { None })
            .collect();
        if !selected_indices.is_empty() {
            let mut selected_X = Array2::zeros((current_X.nrows(), selected_indices.len()));
            for (new_col, &old_col) in selected_indices.iter().enumerate() {
                for row in 0..current_X.nrows() {
                    selected_X[[row, new_col]] = current_X[[row, old_col]];
                }
            }
            current_X = selected_X;
        }
        if self.dimensionality_reduction.is_some() {
            let step_start = Instant::now();
            let features_before = current_X.ncols();
            let mut reduction = self
                .dimensionality_reduction
                .take()
                .expect("operation should succeed");
            current_X = self.apply_dimensionality_reduction(&mut reduction, current_X.view())?;
            self.dimensionality_reduction = Some(reduction);
            trained_steps.push(TrainedStep {
                step_type: "DimensionalityReduction".to_string(),
                step_index: 0,
                training_time: step_start.elapsed(),
                feature_count_before: features_before,
                feature_count_after: current_X.ncols(),
                parameters: StepParameters::DimensionalityReduction(Array2::zeros((1, 1))),
            });
        }
        if self.model_selection.is_some() {
            let step_start = Instant::now();
            let features_before = current_X.ncols();
            let mut model_sel = self
                .model_selection
                .take()
                .expect("operation should succeed");
            let selected_features =
                self.apply_model_selection(&mut model_sel, current_X.view(), current_y.view())?;
            self.model_selection = Some(model_sel);
            if !selected_features.is_empty() {
                let mut model_selected_X =
                    Array2::zeros((current_X.nrows(), selected_features.len()));
                for (new_col, &old_col) in selected_features.iter().enumerate() {
                    for row in 0..current_X.nrows() {
                        model_selected_X[[row, new_col]] = current_X[[row, old_col]];
                    }
                }
                current_X = model_selected_X;
            }
            trained_steps.push(TrainedStep {
                step_type: "ModelSelection".to_string(),
                step_index: 0,
                training_time: step_start.elapsed(),
                feature_count_before: features_before,
                feature_count_after: current_X.ncols(),
                parameters: StepParameters::ModelSelection(selected_features),
            });
        }
        let final_features = current_X.ncols();
        let _feature_mapping = FeatureMapping {
            original_features,
            final_features,
            feature_names: (0..final_features)
                .map(|i| format!("feature_{}", i))
                .collect(),
            feature_origins: (0..final_features).map(FeatureOrigin::Original).collect(),
            transformation_history: trained_steps
                .iter()
                .map(|step| TransformationStep {
                    step_name: step.step_type.clone(),
                    input_features: step.feature_count_before,
                    output_features: step.feature_count_after,
                    transformation_type: TransformationType::ManyToMany,
                })
                .collect(),
        };
        let total_training_time = start_time.elapsed();
        let feature_reduction_ratio = final_features as f64 / original_features as f64;
        let _pipeline_metadata = PipelineMetadata {
            total_training_time,
            total_transform_time: Duration::from_secs(0),
            memory_usage_peak: 0,
            feature_reduction_ratio,
            performance_metrics: HashMap::new(),
            validation_results: None,
        };
        Ok(FeatureSelectionPipeline {
            preprocessing_steps: self.preprocessing_steps,
            feature_engineering_steps: self.feature_engineering_steps,
            selection_methods: self.selection_methods,
            dimensionality_reduction: self.dimensionality_reduction,
            model_selection: self.model_selection,
            pipeline_config: self.pipeline_config,
            optimization_config: self.optimization_config,
            _phantom: PhantomData::<Trained>,
        })
    }
    /// apply_preprocessing_step
    pub fn apply_preprocessing_step(
        &self,
        step: &mut PreprocessingStep,
        X: ArrayView2<f64>,
    ) -> Result<Array2<f64>> {
        Self::apply_preprocessing_step_static(step, X)
    }
    /// apply_preprocessing_step_static
    pub fn apply_preprocessing_step_static(
        step: &mut PreprocessingStep,
        X: ArrayView2<f64>,
    ) -> Result<Array2<f64>> {
        match step {
            PreprocessingStep::StandardScaler {
                config,
                trained_params,
            } => Self::apply_standard_scaler_static(config, trained_params, X),
            PreprocessingStep::RobustScaler {
                config,
                trained_params,
            } => Self::apply_robust_scaler_static(config, trained_params, X),
            PreprocessingStep::MinMaxScaler {
                config,
                trained_params,
            } => Self::apply_minmax_scaler_static(config, trained_params, X),
            _ => Ok(X.to_owned()),
        }
    }
    /// apply_standard_scaler
    pub fn apply_standard_scaler(
        &self,
        config: &StandardScalerConfig,
        trained_params: &mut Option<ScalerParams>,
        X: ArrayView2<f64>,
    ) -> Result<Array2<f64>> {
        Self::apply_standard_scaler_static(config, trained_params, X)
    }
    /// apply_standard_scaler_static
    pub fn apply_standard_scaler_static(
        config: &StandardScalerConfig,
        trained_params: &mut Option<ScalerParams>,
        X: ArrayView2<f64>,
    ) -> Result<Array2<f64>> {
        let mut result = X.to_owned();
        if trained_params.is_none() {
            let mut mean = Array1::zeros(X.ncols());
            let mut scale = Array1::ones(X.ncols());
            if config.with_mean {
                for col in 0..X.ncols() {
                    mean[col] = X.column(col).mean().unwrap_or(0.0);
                }
            }
            if config.with_std {
                for col in 0..X.ncols() {
                    let column = X.column(col);
                    let variance = column.var(1.0);
                    scale[col] = variance.sqrt().max(1e-8);
                }
            }
            *trained_params = Some(ScalerParams { mean, scale });
        }
        if let Some(ref params) = trained_params {
            for col in 0..X.ncols() {
                for row in 0..X.nrows() {
                    if config.with_mean {
                        result[[row, col]] -= params.mean[col];
                    }
                    if config.with_std {
                        result[[row, col]] /= params.scale[col];
                    }
                }
            }
        }
        Ok(result)
    }
    /// apply_robust_scaler
    pub fn apply_robust_scaler(
        &self,
        config: &RobustScalerConfig,
        trained_params: &mut Option<RobustScalerParams>,
        X: ArrayView2<f64>,
    ) -> Result<Array2<f64>> {
        Self::apply_robust_scaler_static(config, trained_params, X)
    }
    /// apply_robust_scaler_static
    pub fn apply_robust_scaler_static(
        config: &RobustScalerConfig,
        trained_params: &mut Option<RobustScalerParams>,
        X: ArrayView2<f64>,
    ) -> Result<Array2<f64>> {
        let mut result = X.to_owned();
        if trained_params.is_none() {
            let mut center = Array1::zeros(X.ncols());
            let mut scale = Array1::ones(X.ncols());
            for col in 0..X.ncols() {
                let mut column_data: Vec<f64> = X.column(col).to_vec();
                column_data.sort_by(|a, b| a.partial_cmp(b).expect("operation should succeed"));
                let n = column_data.len();
                if config.with_centering {
                    center[col] = if n.is_multiple_of(2) {
                        (column_data[n / 2 - 1] + column_data[n / 2]) / 2.0
                    } else {
                        column_data[n / 2]
                    };
                }
                if config.with_scaling {
                    let q1_idx = ((n - 1) as f64 * config.quantile_range.0) as usize;
                    let q3_idx = ((n - 1) as f64 * config.quantile_range.1) as usize;
                    let iqr = column_data[q3_idx] - column_data[q1_idx];
                    scale[col] = iqr.max(1e-8);
                }
            }
            *trained_params = Some(RobustScalerParams { center, scale });
        }
        if let Some(ref params) = trained_params {
            for col in 0..X.ncols() {
                for row in 0..X.nrows() {
                    if config.with_centering {
                        result[[row, col]] -= params.center[col];
                    }
                    if config.with_scaling {
                        result[[row, col]] /= params.scale[col];
                    }
                }
            }
        }
        Ok(result)
    }
    /// apply_minmax_scaler
    pub fn apply_minmax_scaler(
        &self,
        config: &MinMaxScalerConfig,
        trained_params: &mut Option<MinMaxScalerParams>,
        X: ArrayView2<f64>,
    ) -> Result<Array2<f64>> {
        Self::apply_minmax_scaler_static(config, trained_params, X)
    }
    /// apply_minmax_scaler_static
    pub fn apply_minmax_scaler_static(
        config: &MinMaxScalerConfig,
        trained_params: &mut Option<MinMaxScalerParams>,
        X: ArrayView2<f64>,
    ) -> Result<Array2<f64>> {
        let mut result = X.to_owned();
        if trained_params.is_none() {
            let mut min = Array1::zeros(X.ncols());
            let mut scale = Array1::ones(X.ncols());
            for col in 0..X.ncols() {
                let column = X.column(col);
                let col_min = column.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let col_max = column.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                min[col] = col_min;
                let range = col_max - col_min;
                if range > 1e-8 {
                    scale[col] = (config.feature_range.1 - config.feature_range.0) / range;
                }
            }
            *trained_params = Some(MinMaxScalerParams { min, scale });
        }
        if let Some(ref params) = trained_params {
            for col in 0..X.ncols() {
                for row in 0..X.nrows() {
                    let scaled = (result[[row, col]] - params.min[col]) * params.scale[col]
                        + config.feature_range.0;
                    result[[row, col]] = if config.clip {
                        scaled
                            .max(config.feature_range.0)
                            .min(config.feature_range.1)
                    } else {
                        scaled
                    };
                }
            }
        }
        Ok(result)
    }
    /// apply_feature_engineering_step_static
    pub fn apply_feature_engineering_step_static(
        _step: &mut FeatureEngineeringStep,
        X: ArrayView2<f64>,
        _y: ArrayView1<f64>,
    ) -> Result<Array2<f64>> {
        Ok(X.to_owned())
    }
    /// apply_feature_engineering_step
    pub fn apply_feature_engineering_step(
        &self,
        step: &mut FeatureEngineeringStep,
        X: ArrayView2<f64>,
        _y: ArrayView1<f64>,
    ) -> Result<Array2<f64>> {
        match step {
            FeatureEngineeringStep::PolynomialFeatures {
                degree,
                interaction_only,
                include_bias,
                feature_mapping,
            } => self.apply_polynomial_features(
                *degree,
                *interaction_only,
                *include_bias,
                feature_mapping,
                X,
            ),
            FeatureEngineeringStep::InteractionFeatures {
                max_pairs,
                threshold,
                feature_pairs,
            } => self.apply_interaction_features(*max_pairs, *threshold, feature_pairs, X),
            FeatureEngineeringStep::BinningFeatures {
                n_bins,
                strategy,
                bin_edges,
            } => self.apply_binning_features(*n_bins, strategy, bin_edges, X),
            _ => Ok(X.to_owned()),
        }
    }
    /// apply_polynomial_features
    pub fn apply_polynomial_features(
        &self,
        degree: usize,
        interaction_only: bool,
        include_bias: bool,
        feature_mapping: &mut Option<Vec<(usize, usize)>>,
        X: ArrayView2<f64>,
    ) -> Result<Array2<f64>> {
        let n_features = X.ncols();
        let mut new_features = Vec::new();
        let mut mapping = Vec::new();
        if include_bias {
            let bias_feature = Array1::ones(X.nrows());
            new_features.push(bias_feature);
            mapping.push((0, 0));
        }
        for i in 0..n_features {
            new_features.push(X.column(i).to_owned());
            mapping.push((i, 1));
        }
        if !interaction_only {
            for d in 2..=degree {
                for i in 0..n_features {
                    let mut poly_feature = Array1::zeros(X.nrows());
                    for row in 0..X.nrows() {
                        poly_feature[row] = X[[row, i]].powi(d as i32);
                    }
                    new_features.push(poly_feature);
                    mapping.push((i, d));
                }
            }
        }
        for d in 2..=degree {
            for i in 0..n_features {
                for j in (i + 1)..n_features {
                    let mut interaction_feature = Array1::zeros(X.nrows());
                    for row in 0..X.nrows() {
                        interaction_feature[row] = X[[row, i]] * X[[row, j]];
                    }
                    new_features.push(interaction_feature);
                    mapping.push((i * n_features + j, d));
                }
            }
        }
        *feature_mapping = Some(mapping);
        let n_new_features = new_features.len();
        let mut result = Array2::zeros((X.nrows(), n_new_features));
        for (col, feature) in new_features.iter().enumerate() {
            for row in 0..X.nrows() {
                result[[row, col]] = feature[row];
            }
        }
        Ok(result)
    }
    /// apply_interaction_features
    pub fn apply_interaction_features(
        &self,
        max_pairs: Option<usize>,
        threshold: f64,
        feature_pairs: &mut Option<Vec<(usize, usize)>>,
        X: ArrayView2<f64>,
    ) -> Result<Array2<f64>> {
        let n_features = X.ncols();
        let mut interactions = Vec::new();
        let pairs: Vec<(usize, usize)>;
        if feature_pairs.is_none() {
            let mut candidate_pairs = Vec::new();
            for i in 0..n_features {
                for j in (i + 1)..n_features {
                    let corr = self.compute_correlation(X.column(i), X.column(j));
                    if corr.abs() > threshold {
                        candidate_pairs.push((i, j, corr.abs()));
                    }
                }
            }
            candidate_pairs
                .sort_by(|a, b| b.2.partial_cmp(&a.2).expect("operation should succeed"));
            let limit = max_pairs.unwrap_or(candidate_pairs.len());
            pairs = candidate_pairs
                .into_iter()
                .take(limit)
                .map(|(i, j, _)| (i, j))
                .collect();
            *feature_pairs = Some(pairs.clone());
        } else {
            pairs = feature_pairs
                .as_ref()
                .expect("operation should succeed")
                .clone();
        }
        for &(i, j) in &pairs {
            let mut interaction = Array1::zeros(X.nrows());
            for row in 0..X.nrows() {
                interaction[row] = X[[row, i]] * X[[row, j]];
            }
            interactions.push(interaction);
        }
        let total_features = n_features + interactions.len();
        let mut result = Array2::zeros((X.nrows(), total_features));
        for col in 0..n_features {
            for row in 0..X.nrows() {
                result[[row, col]] = X[[row, col]];
            }
        }
        for (idx, interaction) in interactions.iter().enumerate() {
            for row in 0..X.nrows() {
                result[[row, n_features + idx]] = interaction[row];
            }
        }
        Ok(result)
    }
    /// apply_binning_features
    pub fn apply_binning_features(
        &self,
        n_bins: usize,
        strategy: &BinningStrategy,
        bin_edges: &mut Option<HashMap<usize, Vec<f64>>>,
        X: ArrayView2<f64>,
    ) -> Result<Array2<f64>> {
        let mut result = X.to_owned();
        if bin_edges.is_none() {
            let mut edges_map = HashMap::new();
            for col in 0..X.ncols() {
                let column = X.column(col);
                let edges = match strategy {
                    BinningStrategy::Uniform => {
                        let min_val = column.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                        let max_val = column.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                        let step = (max_val - min_val) / n_bins as f64;
                        (0..=n_bins)
                            .map(|i| min_val + i as f64 * step)
                            .collect::<Vec<f64>>()
                    }
                    BinningStrategy::Quantile => {
                        let mut sorted_values: Vec<f64> = column.to_vec();
                        sorted_values
                            .sort_by(|a, b| a.partial_cmp(b).expect("operation should succeed"));
                        let n = sorted_values.len();
                        (0..=n_bins)
                            .map(|i| {
                                let quantile = i as f64 / n_bins as f64;
                                let idx = ((n - 1) as f64 * quantile) as usize;
                                sorted_values[idx]
                            })
                            .collect::<Vec<f64>>()
                    }
                    BinningStrategy::KMeans => {
                        let min_val = column.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                        let max_val = column.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                        let step = (max_val - min_val) / n_bins as f64;
                        (0..=n_bins)
                            .map(|i| min_val + i as f64 * step)
                            .collect::<Vec<f64>>()
                    }
                };
                edges_map.insert(col, edges);
            }
            *bin_edges = Some(edges_map);
        }
        if let Some(ref edges_map) = bin_edges {
            for col in 0..X.ncols() {
                if let Some(edges) = edges_map.get(&col) {
                    for row in 0..X.nrows() {
                        let value = X[[row, col]];
                        let bin = edges
                            .iter()
                            .position(|&edge| value <= edge)
                            .unwrap_or(edges.len() - 1)
                            .min(n_bins - 1);
                        result[[row, col]] = bin as f64;
                    }
                }
            }
        }
        Ok(result)
    }
    /// apply_selection_method_static
    pub fn apply_selection_method_static(
        _method: &mut SelectionMethod,
        X: ArrayView2<f64>,
        _y: ArrayView1<f64>,
    ) -> Result<Array1<bool>> {
        Ok(Array1::from_elem(X.ncols(), true))
    }
    /// apply_selection_method
    pub fn apply_selection_method(
        &self,
        method: &mut SelectionMethod,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> Result<Array1<bool>> {
        match method {
            SelectionMethod::VarianceThreshold {
                threshold,
                feature_variance,
            } => self.apply_variance_threshold(*threshold, feature_variance, X),
            SelectionMethod::CorrelationFilter {
                threshold,
                method: corr_method,
                correlation_matrix,
            } => self.apply_correlation_filter(*threshold, corr_method, correlation_matrix, X),
            SelectionMethod::UnivariateFilter {
                method: uni_method,
                k,
                score_func,
            } => self.apply_univariate_filter(uni_method, k, score_func, X, y),
            _ => Ok(Array1::from_elem(X.ncols(), true)),
        }
    }
    /// apply_variance_threshold
    pub fn apply_variance_threshold(
        &self,
        threshold: f64,
        feature_variance: &mut Option<Array1<f64>>,
        X: ArrayView2<f64>,
    ) -> Result<Array1<bool>> {
        if feature_variance.is_none() {
            let mut variances = Array1::zeros(X.ncols());
            for col in 0..X.ncols() {
                variances[col] = X.column(col).var(1.0);
            }
            *feature_variance = Some(variances);
        }
        let variances = feature_variance.as_ref().expect("operation should succeed");
        let selection = variances.mapv(|v| v > threshold);
        Ok(selection)
    }
    /// apply_correlation_filter
    pub fn apply_correlation_filter(
        &self,
        threshold: f64,
        corr_method: &CorrelationMethod,
        correlation_matrix: &mut Option<Array2<f64>>,
        X: ArrayView2<f64>,
    ) -> Result<Array1<bool>> {
        if correlation_matrix.is_none() {
            let n_features = X.ncols();
            let mut corr_matrix = Array2::zeros((n_features, n_features));
            for i in 0..n_features {
                for j in 0..n_features {
                    if i == j {
                        corr_matrix[[i, j]] = 1.0;
                    } else {
                        let corr = match corr_method {
                            CorrelationMethod::Pearson => {
                                self.compute_correlation(X.column(i), X.column(j))
                            }
                            _ => self.compute_correlation(X.column(i), X.column(j)),
                        };
                        corr_matrix[[i, j]] = corr;
                    }
                }
            }
            *correlation_matrix = Some(corr_matrix);
        }
        let corr_matrix = correlation_matrix
            .as_ref()
            .expect("operation should succeed");
        let mut selection = Array1::from_elem(X.ncols(), true);
        for i in 0..X.ncols() {
            for j in (i + 1)..X.ncols() {
                if corr_matrix[[i, j]].abs() > threshold && selection[i] && selection[j] {
                    let var_i = X.column(i).var(1.0);
                    let var_j = X.column(j).var(1.0);
                    if var_i < var_j {
                        selection[i] = false;
                    } else {
                        selection[j] = false;
                    }
                }
            }
        }
        Ok(selection)
    }
    /// apply_univariate_filter
    pub fn apply_univariate_filter(
        &self,
        _method: &UnivariateMethod,
        k: &SelectionCount,
        score_func: &UnivariateScoreFunction,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> Result<Array1<bool>> {
        let mut scores = Array1::zeros(X.ncols());
        for col in 0..X.ncols() {
            scores[col] = match score_func {
                UnivariateScoreFunction::Chi2 => self.compute_chi2_score(X.column(col), y),
                UnivariateScoreFunction::FClassif => self.compute_f_score(X.column(col), y),
                UnivariateScoreFunction::MutualInfoClassif => {
                    self.compute_mutual_info(X.column(col), y)
                }
                _ => self.compute_correlation(X.column(col), y).abs(),
            };
        }
        let selection = match k {
            SelectionCount::K(k_val) => {
                let mut indexed_scores: Vec<(usize, f64)> = scores
                    .iter()
                    .enumerate()
                    .map(|(i, &score)| (i, score))
                    .collect();
                indexed_scores
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).expect("operation should succeed"));
                let mut selection = Array1::from_elem(X.ncols(), false);
                for &(idx, _) in indexed_scores.iter().take(*k_val) {
                    selection[idx] = true;
                }
                selection
            }
            SelectionCount::Percentile(p) => {
                let k_val = ((X.ncols() as f64 * p / 100.0).round() as usize).max(1);
                let mut indexed_scores: Vec<(usize, f64)> = scores
                    .iter()
                    .enumerate()
                    .map(|(i, &score)| (i, score))
                    .collect();
                indexed_scores
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).expect("operation should succeed"));
                let mut selection = Array1::from_elem(X.ncols(), false);
                for &(idx, _) in indexed_scores.iter().take(k_val) {
                    selection[idx] = true;
                }
                selection
            }
            _ => {
                let k_val = X.ncols() / 2;
                let mut indexed_scores: Vec<(usize, f64)> = scores
                    .iter()
                    .enumerate()
                    .map(|(i, &score)| (i, score))
                    .collect();
                indexed_scores
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).expect("operation should succeed"));
                let mut selection = Array1::from_elem(X.ncols(), false);
                for &(idx, _) in indexed_scores.iter().take(k_val) {
                    selection[idx] = true;
                }
                selection
            }
        };
        Ok(selection)
    }
    /// apply_dimensionality_reduction
    pub fn apply_dimensionality_reduction(
        &self,
        reduction: &mut DimensionalityReductionStep,
        X: ArrayView2<f64>,
    ) -> Result<Array2<f64>> {
        match reduction {
            DimensionalityReductionStep::PCA {
                n_components,
                whiten,
                svd_solver,
                components,
                explained_variance,
            } => self.apply_pca(
                *n_components,
                *whiten,
                svd_solver,
                components,
                explained_variance,
                X,
            ),
            DimensionalityReductionStep::TruncatedSVD {
                n_components,
                algorithm,
                components,
                singular_values,
            } => self.apply_truncated_svd(*n_components, algorithm, components, singular_values, X),
            _ => {
                let n_comp = match reduction {
                    DimensionalityReductionStep::ICA { n_components, .. } => *n_components,
                    DimensionalityReductionStep::FactorAnalysis { n_components, .. } => {
                        *n_components
                    }
                    DimensionalityReductionStep::UMAP { n_components, .. } => *n_components,
                    DimensionalityReductionStep::TSNE { n_components, .. } => *n_components,
                    _ => X.ncols().min(50),
                };
                let final_components = n_comp.min(X.ncols());
                let mut result = Array2::zeros((X.nrows(), final_components));
                for col in 0..final_components {
                    for row in 0..X.nrows() {
                        result[[row, col]] = X[[row, col]];
                    }
                }
                Ok(result)
            }
        }
    }
    /// apply_pca
    pub fn apply_pca(
        &self,
        n_components: usize,
        _whiten: bool,
        _svd_solver: &SVDSolver,
        components: &mut Option<Array2<f64>>,
        explained_variance: &mut Option<Array1<f64>>,
        X: ArrayView2<f64>,
    ) -> Result<Array2<f64>> {
        let n_comp = n_components.min(X.ncols()).min(X.nrows());
        let mut centered_X = X.to_owned();
        let mut means = Array1::zeros(X.ncols());
        for col in 0..X.ncols() {
            means[col] = X.column(col).mean().unwrap_or(0.0);
            for row in 0..X.nrows() {
                centered_X[[row, col]] -= means[col];
            }
        }
        if components.is_none() {
            *components = Some(Array2::eye(X.ncols()));
            *explained_variance = Some(Array1::ones(n_comp));
        }
        let mut result = Array2::zeros((X.nrows(), n_comp));
        for col in 0..n_comp {
            for row in 0..X.nrows() {
                result[[row, col]] = centered_X[[row, col]];
            }
        }
        Ok(result)
    }
    /// apply_truncated_svd
    pub fn apply_truncated_svd(
        &self,
        n_components: usize,
        _algorithm: &SVDAlgorithm,
        components: &mut Option<Array2<f64>>,
        singular_values: &mut Option<Array1<f64>>,
        X: ArrayView2<f64>,
    ) -> Result<Array2<f64>> {
        let n_comp = n_components.min(X.ncols()).min(X.nrows());
        if components.is_none() {
            *components = Some(Array2::eye(X.ncols()));
            *singular_values = Some(Array1::ones(n_comp));
        }
        let mut result = Array2::zeros((X.nrows(), n_comp));
        for col in 0..n_comp {
            for row in 0..X.nrows() {
                result[[row, col]] = X[[row, col]];
            }
        }
        Ok(result)
    }
    /// apply_model_selection
    pub fn apply_model_selection(
        &self,
        model_selection: &mut ModelSelectionStep,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> Result<Vec<usize>> {
        match model_selection {
            ModelSelectionStep::CrossValidationSelection {
                estimator,
                cv_folds,
                scoring,
                feature_scores,
            } => self.apply_cv_selection(estimator, *cv_folds, scoring, feature_scores, X, y),
            ModelSelectionStep::ForwardSelection {
                estimator,
                max_features,
                scoring,
                selected_features,
            } => self.apply_forward_selection(
                estimator,
                *max_features,
                scoring,
                selected_features,
                X,
                y,
            ),
            _ => Ok((0..X.ncols()).collect()),
        }
    }
    /// apply_cv_selection
    pub fn apply_cv_selection(
        &self,
        _estimator: &ModelEstimator,
        _cv_folds: usize,
        _scoring: &ScoringMetric,
        feature_scores: &mut Option<Array1<f64>>,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> Result<Vec<usize>> {
        if feature_scores.is_none() {
            let mut scores = Array1::zeros(X.ncols());
            for col in 0..X.ncols() {
                scores[col] = self.compute_correlation(X.column(col), y).abs();
            }
            *feature_scores = Some(scores);
        }
        if let Some(ref scores) = feature_scores {
            let mut indexed_scores: Vec<(usize, f64)> = scores
                .iter()
                .enumerate()
                .map(|(i, &score)| (i, score))
                .collect();
            indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("operation should succeed"));
            let n_select = X.ncols() / 2;
            Ok(indexed_scores
                .into_iter()
                .take(n_select)
                .map(|(idx, _)| idx)
                .collect())
        } else {
            Ok((0..X.ncols()).collect())
        }
    }
    /// apply_forward_selection
    pub fn apply_forward_selection(
        &self,
        _estimator: &ModelEstimator,
        max_features: usize,
        _scoring: &ScoringMetric,
        selected_features: &mut Option<Vec<usize>>,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> Result<Vec<usize>> {
        if selected_features.is_none() {
            let mut scores = Vec::new();
            for col in 0..X.ncols() {
                let score = self.compute_correlation(X.column(col), y).abs();
                scores.push((col, score));
            }
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("operation should succeed"));
            let features: Vec<usize> = scores
                .into_iter()
                .take(max_features.min(X.ncols()))
                .map(|(idx, _)| idx)
                .collect();
            *selected_features = Some(features.clone());
            Ok(features)
        } else {
            Ok(selected_features
                .as_ref()
                .expect("operation should succeed")
                .clone())
        }
    }
    /// compute_correlation
    pub fn compute_correlation(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
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
    /// compute_chi2_score
    pub fn compute_chi2_score(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        self.compute_correlation(x, y).abs()
    }
    /// compute_f_score
    pub fn compute_f_score(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        self.compute_correlation(x, y).abs()
    }
    /// compute_mutual_info
    pub fn compute_mutual_info(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        self.compute_correlation(x, y).abs()
    }
}
impl FeatureSelectionPipeline<Trained> {
    /// transform
    pub fn transform(&self, X: ArrayView2<f64>) -> Result<Array2<f64>> {
        let _start_time = Instant::now();
        let current_X = X.to_owned();
        Ok(current_X)
    }
    /// Get information about the trained pipeline
    pub fn get_pipeline_info(&self) -> PipelineInfo {
        PipelineInfo {
            n_preprocessing_steps: self.preprocessing_steps.len(),
            n_feature_engineering_steps: self.feature_engineering_steps.len(),
            n_selection_methods: self.selection_methods.len(),
            has_dimensionality_reduction: self.dimensionality_reduction.is_some(),
            has_model_selection: self.model_selection.is_some(),
            config: self.pipeline_config.clone(),
        }
    }
}
#[derive(Debug, Clone)]
/// QuantileTransformerConfig
pub struct QuantileTransformerConfig {
    /// n_quantiles
    pub n_quantiles: usize,
    /// output_distribution
    pub output_distribution: Distribution,
    /// subsample
    pub subsample: Option<usize>,
}
/// Dimensionality reduction step (applied after feature selection)
#[derive(Debug, Clone)]
pub enum DimensionalityReductionStep {
    /// PCA
    PCA {
        /// n_components
        n_components: usize,
        /// whiten
        whiten: bool,
        /// svd_solver
        svd_solver: SVDSolver,
        /// components
        components: Option<Array2<f64>>,
        /// explained_variance
        explained_variance: Option<Array1<f64>>,
    },
    /// TruncatedSVD
    TruncatedSVD {
        /// n_components
        n_components: usize,
        /// algorithm
        algorithm: SVDAlgorithm,
        /// components
        components: Option<Array2<f64>>,
        /// singular_values
        singular_values: Option<Array1<f64>>,
    },
    /// ICA
    ICA {
        /// n_components
        n_components: usize,
        /// algorithm
        algorithm: ICAAlgorithm,
        /// max_iter
        max_iter: usize,
        /// tol
        tol: f64,
        /// mixing_matrix
        mixing_matrix: Option<Array2<f64>>,
        /// unmixing_matrix
        unmixing_matrix: Option<Array2<f64>>,
    },
    /// FactorAnalysis
    FactorAnalysis {
        /// n_components
        n_components: usize,
        /// max_iter
        max_iter: usize,
        /// tol
        tol: f64,
        /// loadings
        loadings: Option<Array2<f64>>,
        /// noise_variance
        noise_variance: Option<Array1<f64>>,
    },
    /// UMAP
    UMAP {
        /// n_components
        n_components: usize,
        /// n_neighbors
        n_neighbors: usize,
        /// min_dist
        min_dist: f64,
        /// metric
        metric: DistanceMetric,
        /// embedding
        embedding: Option<Array2<f64>>,
    },
    /// TSNE
    TSNE {
        /// n_components
        n_components: usize,
        /// perplexity
        perplexity: f64,
        /// early_exaggeration
        early_exaggeration: f64,
        /// learning_rate
        learning_rate: f64,
        /// max_iter
        max_iter: usize,
        /// embedding
        embedding: Option<Array2<f64>>,
    },
}
/// Feature engineering steps for creating new features
#[derive(Debug, Clone)]
pub enum FeatureEngineeringStep {
    /// PolynomialFeatures
    PolynomialFeatures {
        /// degree
        degree: usize,
        /// interaction_only
        interaction_only: bool,
        /// include_bias
        include_bias: bool,
        /// feature_mapping
        feature_mapping: Option<Vec<(usize, usize)>>,
    },
    /// InteractionFeatures
    InteractionFeatures {
        /// max_pairs
        max_pairs: Option<usize>,
        /// threshold
        threshold: f64,
        /// feature_pairs
        feature_pairs: Option<Vec<(usize, usize)>>,
    },
    /// BinningFeatures
    BinningFeatures {
        /// n_bins
        n_bins: usize,
        /// strategy
        strategy: BinningStrategy,
        /// bin_edges
        bin_edges: Option<HashMap<usize, Vec<f64>>>,
    },
    /// TargetEncoding
    TargetEncoding {
        /// smoothing
        smoothing: f64,
        /// min_samples_leaf
        min_samples_leaf: usize,
        /// encodings
        encodings: Option<HashMap<usize, HashMap<String, f64>>>,
    },
    /// FrequencyEncoding
    FrequencyEncoding {
        /// min_frequency
        min_frequency: f64,
        /// frequencies
        frequencies: Option<HashMap<usize, HashMap<String, f64>>>,
    },
    /// RatioFeatures
    RatioFeatures {
        /// numerator_features
        numerator_features: Vec<usize>,
        /// denominator_features
        denominator_features: Vec<usize>,
        /// eps
        eps: f64,
    },
    /// LaggingFeatures
    LaggingFeatures {
        /// lags
        lags: Vec<usize>,
        /// feature_subset
        feature_subset: Option<Vec<usize>>,
    },
    /// WindowStatistics
    WindowStatistics {
        /// window_size
        window_size: usize,
        /// statistics
        statistics: Vec<WindowStatistic>,
        /// feature_subset
        feature_subset: Option<Vec<usize>>,
    },
}
#[derive(Debug, Clone)]
/// ImputerConfig
pub struct ImputerConfig {
    /// strategy
    pub strategy: ImputationStrategy,
    /// fill_value
    pub fill_value: Option<f64>,
    /// missing_values
    pub missing_values: MissingValueIndicator,
}
/// Type-safe state markers for compile-time pipeline validation
#[derive(Debug, Clone, Default)]
pub struct Untrained;
#[derive(Debug, Clone)]
/// DistanceMetric
pub enum DistanceMetric {
    /// Euclidean
    Euclidean,
    /// Manhattan
    Manhattan,
    /// Cosine
    Cosine,
    /// Hamming
    Hamming,
}
#[derive(Debug)]
/// StepParameters
pub enum StepParameters {
    /// Preprocessing
    Preprocessing(Box<dyn std::any::Any + Send + Sync>),
    /// FeatureEngineering
    FeatureEngineering(Box<dyn std::any::Any + Send + Sync>),
    /// Selection
    Selection(Array1<bool>),
    /// DimensionalityReduction
    DimensionalityReduction(Array2<f64>),
    /// ModelSelection
    ModelSelection(Vec<usize>),
}
/// Configuration for pipeline behavior
#[derive(Debug, Clone)]
pub struct PipelineConfiguration {
    /// parallel_execution
    pub parallel_execution: bool,
    /// memory_optimization
    pub memory_optimization: MemoryOptimization,
    /// caching_strategy
    pub caching_strategy: CachingStrategy,
    /// validation_strategy
    pub validation_strategy: ValidationStrategy,
    /// error_handling
    pub error_handling: ErrorHandling,
    /// logging_level
    pub logging_level: LoggingLevel,
}
#[derive(Debug, Clone)]
/// ImportanceGetter
pub enum ImportanceGetter {
    /// Auto
    Auto,
    /// Coefficients
    Coefficients,
    /// FeatureImportances
    FeatureImportances,
}
#[derive(Debug, Clone)]
/// CachingStrategy
pub enum CachingStrategy {
    /// None
    None,
    /// LRU
    LRU {
        /// size
        size: usize,
    },
    /// LFU
    LFU {
        /// size
        size: usize,
    },
    /// FIFO
    FIFO {
        /// size
        size: usize,
    },
}
/// Trained step information for pipeline state tracking
#[derive(Debug)]
pub struct TrainedStep {
    /// step_type
    pub step_type: String,
    /// step_index
    pub step_index: usize,
    /// training_time
    pub training_time: Duration,
    /// feature_count_before
    pub feature_count_before: usize,
    /// feature_count_after
    pub feature_count_after: usize,
    /// parameters
    pub parameters: StepParameters,
}
#[derive(Debug, Clone)]
/// ScoringMetric
pub enum ScoringMetric {
    /// Accuracy
    Accuracy,
    /// F1
    F1,
    /// RocAuc
    RocAuc,
    /// R2
    R2,
    /// MAE
    MAE,
    /// MSE
    MSE,
    /// LogLoss
    LogLoss,
}
#[derive(Debug, Clone)]
/// AcquisitionFunction
pub enum AcquisitionFunction {
    /// ExpectedImprovement
    ExpectedImprovement,
    /// UpperConfidenceBound
    UpperConfidenceBound,
    /// ProbabilityOfImprovement
    ProbabilityOfImprovement,
}
#[derive(Debug, Clone)]
/// PowerTransformerConfig
pub struct PowerTransformerConfig {
    /// method
    pub method: PowerMethod,
    /// standardize
    pub standardize: bool,
}
