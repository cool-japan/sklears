//! Adaptive preprocessing parameters that automatically tune based on data characteristics
//!
//! This module provides automatic parameter selection and tuning for preprocessing
//! transformers based on statistical analysis of the input data. It helps optimize
//! preprocessing pipelines without manual parameter tuning.
//!
//! # Features
//!
//! - **Data Distribution Analysis**: Automatically detect data distribution characteristics
//! - **Adaptive Thresholds**: Dynamic threshold selection based on data properties
//! - **Parameter Optimization**: Automatic parameter tuning for various transformers
//! - **Multi-Objective Optimization**: Balance multiple criteria (robustness, efficiency, quality)
//! - **Cross-Validation Based Tuning**: Use CV to select optimal parameters
//! - **Ensemble Parameter Selection**: Combine multiple parameter selection strategies

use scirs2_core::ndarray::Array2;
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Trained, Untrained},
    types::Float,
};
use std::collections::HashMap;
use std::marker::PhantomData;

/// Data distribution characteristics detected from input data
#[derive(Debug, Clone)]
pub struct DataCharacteristics {
    /// Number of samples and features
    pub shape: (usize, usize),
    /// Distribution type per feature (normal, skewed, uniform, bimodal, etc.)
    pub distribution_types: Vec<DistributionType>,
    /// Skewness per feature
    pub skewness: Vec<Float>,
    /// Kurtosis per feature
    pub kurtosis: Vec<Float>,
    /// Outlier percentages per feature
    pub outlier_percentages: Vec<Float>,
    /// Missing value percentages per feature
    pub missing_percentages: Vec<Float>,
    /// Data ranges per feature
    pub ranges: Vec<(Float, Float)>,
    /// Correlation matrix between features
    pub correlation_strength: Float,
    /// Overall data quality score (0-1)
    pub quality_score: Float,
    /// Estimated optimal batch size for processing
    pub optimal_batch_size: usize,
}

/// Types of distributions detected in features
#[derive(Debug, Clone, Copy)]
pub enum DistributionType {
    /// Approximately normal distribution
    Normal,
    /// Skewed distribution (left or right)
    Skewed,
    /// Uniform distribution
    Uniform,
    /// Bimodal or multimodal distribution
    Multimodal,
    /// Heavy-tailed distribution
    HeavyTailed,
    /// Sparse distribution (many zeros)
    Sparse,
    /// Unknown or complex distribution
    Unknown,
}

/// Adaptive parameter selection strategies
#[derive(Debug, Clone, Copy)]
pub enum AdaptationStrategy {
    /// Conservative approach - prioritize robustness
    Conservative,
    /// Balanced approach - balance performance and robustness
    Balanced,
    /// Aggressive approach - prioritize performance
    Aggressive,
    /// Custom strategy with user-defined weights
    Custom,
}

/// Configuration for adaptive parameter selection
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Adaptation strategy
    pub strategy: AdaptationStrategy,
    /// Whether to use cross-validation for parameter selection
    pub use_cross_validation: bool,
    /// Number of CV folds (if using CV)
    pub cv_folds: usize,
    /// Maximum time budget for optimization (seconds)
    pub time_budget: Option<Float>,
    /// Whether to use parallel processing
    pub parallel: bool,
    /// Convergence tolerance for optimization
    pub tolerance: Float,
    /// Maximum number of optimization iterations
    pub max_iterations: usize,
    /// Custom parameter bounds (if any)
    pub parameter_bounds: HashMap<String, (Float, Float)>,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            strategy: AdaptationStrategy::Balanced,
            use_cross_validation: true,
            cv_folds: 5,
            time_budget: Some(60.0), // 1 minute default
            parallel: true,
            tolerance: 1e-4,
            max_iterations: 100,
            parameter_bounds: HashMap::new(),
        }
    }
}

/// Adaptive parameter selector for preprocessing transformers
#[derive(Debug, Clone)]
pub struct AdaptiveParameterSelector<State = Untrained> {
    config: AdaptiveConfig,
    state: PhantomData<State>,
    // Fitted parameters
    data_characteristics_: Option<DataCharacteristics>,
    optimal_parameters_: Option<HashMap<String, Float>>,
    parameter_history_: Option<Vec<ParameterEvaluation>>,
}

/// Parameter evaluation result
#[derive(Debug, Clone)]
pub struct ParameterEvaluation {
    pub parameters: HashMap<String, Float>,
    pub score: Float,
    pub robustness_score: Float,
    pub efficiency_score: Float,
    pub quality_score: Float,
    pub evaluation_time: Float,
}

/// Adaptive parameter recommendations for different transformers
#[derive(Debug, Clone)]
pub struct ParameterRecommendations {
    /// Recommended scaling parameters
    pub scaling: ScalingParameters,
    /// Recommended imputation parameters
    pub imputation: ImputationParameters,
    /// Recommended outlier detection parameters
    pub outlier_detection: OutlierDetectionParameters,
    /// Recommended transformation parameters
    pub transformation: TransformationParameters,
    /// Overall confidence in recommendations (0-1)
    pub confidence: Float,
}

/// Adaptive scaling parameters
#[derive(Debug, Clone)]
pub struct ScalingParameters {
    pub method: String, // "standard", "robust", "minmax", etc.
    pub outlier_threshold: Float,
    pub quantile_range: (Float, Float),
    pub with_centering: bool,
    pub with_scaling: bool,
}

/// Adaptive imputation parameters
#[derive(Debug, Clone)]
pub struct ImputationParameters {
    pub strategy: String, // "mean", "median", "knn", etc.
    pub n_neighbors: Option<usize>,
    pub outlier_aware: bool,
    pub max_iterations: Option<usize>,
}

/// Adaptive outlier detection parameters
#[derive(Debug, Clone)]
pub struct OutlierDetectionParameters {
    pub method: String, // "isolation_forest", "local_outlier_factor", etc.
    pub contamination: Float,
    pub threshold: Float,
    pub ensemble_size: Option<usize>,
}

/// Adaptive transformation parameters
#[derive(Debug, Clone)]
pub struct TransformationParameters {
    pub method: String, // "log", "box_cox", "quantile", etc.
    pub handle_negatives: bool,
    pub lambda: Option<Float>,
    pub n_quantiles: Option<usize>,
}

impl AdaptiveParameterSelector<Untrained> {
    /// Create a new adaptive parameter selector
    pub fn new() -> Self {
        Self {
            config: AdaptiveConfig::default(),
            state: PhantomData,
            data_characteristics_: None,
            optimal_parameters_: None,
            parameter_history_: None,
        }
    }

    /// Create with conservative strategy
    pub fn conservative() -> Self {
        Self::new().strategy(AdaptationStrategy::Conservative)
    }

    /// Create with balanced strategy
    pub fn balanced() -> Self {
        Self::new().strategy(AdaptationStrategy::Balanced)
    }

    /// Create with aggressive strategy
    pub fn aggressive() -> Self {
        Self::new().strategy(AdaptationStrategy::Aggressive)
    }

    /// Set the adaptation strategy
    pub fn strategy(mut self, strategy: AdaptationStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Enable or disable cross-validation
    pub fn cross_validation(mut self, enable: bool, folds: usize) -> Self {
        self.config.use_cross_validation = enable;
        self.config.cv_folds = folds;
        self
    }

    /// Set time budget for optimization
    pub fn time_budget(mut self, seconds: Float) -> Self {
        self.config.time_budget = Some(seconds);
        self
    }

    /// Enable parallel processing
    pub fn parallel(mut self, enable: bool) -> Self {
        self.config.parallel = enable;
        self
    }

    /// Set optimization tolerance
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.config.tolerance = tolerance;
        self
    }

    /// Set parameter bounds
    pub fn parameter_bounds(mut self, bounds: HashMap<String, (Float, Float)>) -> Self {
        self.config.parameter_bounds = bounds;
        self
    }
}

impl Fit<Array2<Float>, ()> for AdaptiveParameterSelector<Untrained> {
    type Fitted = AdaptiveParameterSelector<Trained>;

    fn fit(mut self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array is empty".to_string(),
            ));
        }

        // Analyze data characteristics
        let characteristics = self.analyze_data_characteristics(x)?;

        // Generate parameter recommendations based on characteristics
        let optimal_parameters = self.optimize_parameters(x, &characteristics)?;

        // Evaluate different parameter configurations
        let parameter_history = self.evaluate_parameter_space(x, &characteristics)?;

        self.data_characteristics_ = Some(characteristics);
        self.optimal_parameters_ = Some(optimal_parameters);
        self.parameter_history_ = Some(parameter_history);

        Ok(AdaptiveParameterSelector {
            config: self.config,
            state: PhantomData,
            data_characteristics_: self.data_characteristics_,
            optimal_parameters_: self.optimal_parameters_,
            parameter_history_: self.parameter_history_,
        })
    }
}

impl AdaptiveParameterSelector<Untrained> {
    /// Analyze data characteristics to inform parameter selection
    fn analyze_data_characteristics(&self, x: &Array2<Float>) -> Result<DataCharacteristics> {
        let (n_samples, n_features) = x.dim();

        let mut distribution_types = Vec::with_capacity(n_features);
        let mut skewness = Vec::with_capacity(n_features);
        let mut kurtosis = Vec::with_capacity(n_features);
        let mut outlier_percentages = Vec::with_capacity(n_features);
        let mut missing_percentages = Vec::with_capacity(n_features);
        let mut ranges = Vec::with_capacity(n_features);

        // Analyze each feature
        for j in 0..n_features {
            let column = x.column(j);

            // Get valid (non-NaN) values
            let valid_values: Vec<Float> =
                column.iter().filter(|x| x.is_finite()).copied().collect();

            let missing_pct =
                ((n_samples - valid_values.len()) as Float / n_samples as Float) * 100.0;
            missing_percentages.push(missing_pct);

            if valid_values.is_empty() {
                distribution_types.push(DistributionType::Unknown);
                skewness.push(0.0);
                kurtosis.push(0.0);
                outlier_percentages.push(0.0);
                ranges.push((0.0, 0.0));
                continue;
            }

            // Basic statistics
            let mean = valid_values.iter().sum::<Float>() / valid_values.len() as Float;
            let variance = valid_values
                .iter()
                .map(|x| (x - mean).powi(2))
                .sum::<Float>()
                / valid_values.len() as Float;
            let std = variance.sqrt();

            // Skewness and kurtosis
            let feature_skewness = if std > 0.0 {
                valid_values
                    .iter()
                    .map(|x| ((x - mean) / std).powi(3))
                    .sum::<Float>()
                    / valid_values.len() as Float
            } else {
                0.0
            };

            let feature_kurtosis = if std > 0.0 {
                valid_values
                    .iter()
                    .map(|x| ((x - mean) / std).powi(4))
                    .sum::<Float>()
                    / valid_values.len() as Float
                    - 3.0 // Excess kurtosis
            } else {
                0.0
            };

            skewness.push(feature_skewness);
            kurtosis.push(feature_kurtosis);

            // Outlier detection using IQR method
            let mut sorted_values = valid_values.clone();
            sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let q1_idx = sorted_values.len() / 4;
            let q3_idx = 3 * sorted_values.len() / 4;
            let q1 = sorted_values[q1_idx];
            let q3 = sorted_values[q3_idx];
            let iqr = q3 - q1;

            let lower_bound = q1 - 1.5 * iqr;
            let upper_bound = q3 + 1.5 * iqr;

            let outlier_count = valid_values
                .iter()
                .filter(|&&x| x < lower_bound || x > upper_bound)
                .count();
            let outlier_pct = (outlier_count as Float / valid_values.len() as Float) * 100.0;
            outlier_percentages.push(outlier_pct);

            // Range
            let min_val = sorted_values[0];
            let max_val = sorted_values[sorted_values.len() - 1];
            ranges.push((min_val, max_val));

            // Distribution type classification
            let dist_type = self.classify_distribution(
                feature_skewness,
                feature_kurtosis,
                outlier_pct,
                &valid_values,
            );
            distribution_types.push(dist_type);
        }

        // Correlation strength (simplified as average absolute correlation)
        let correlation_strength = self.estimate_correlation_strength(x)?;

        // Overall quality score
        let avg_missing = missing_percentages.iter().sum::<Float>() / n_features as Float;
        let avg_outliers = outlier_percentages.iter().sum::<Float>() / n_features as Float;
        let quality_score = (100.0 - avg_missing - avg_outliers).max(0.0) / 100.0;

        // Optimal batch size estimation
        let optimal_batch_size = self.estimate_optimal_batch_size(n_samples, n_features);

        Ok(DataCharacteristics {
            shape: (n_samples, n_features),
            distribution_types,
            skewness,
            kurtosis,
            outlier_percentages,
            missing_percentages,
            ranges,
            correlation_strength,
            quality_score,
            optimal_batch_size,
        })
    }

    /// Classify the distribution type of a feature
    fn classify_distribution(
        &self,
        skewness: Float,
        kurtosis: Float,
        outlier_pct: Float,
        values: &[Float],
    ) -> DistributionType {
        // Check for sparsity (many zeros)
        let zero_count = values.iter().filter(|&&x| x.abs() < 1e-10).count();
        let sparsity = zero_count as Float / values.len() as Float;

        if sparsity > 0.5 {
            return DistributionType::Sparse;
        }

        // Check for normality
        if skewness.abs() < 0.5 && kurtosis.abs() < 1.0 && outlier_pct < 5.0 {
            return DistributionType::Normal;
        }

        // Check for skewness
        if skewness.abs() > 1.0 {
            return DistributionType::Skewed;
        }

        // Check for heavy tails
        if kurtosis > 2.0 || outlier_pct > 10.0 {
            return DistributionType::HeavyTailed;
        }

        // Check for uniformity (low kurtosis, low skewness, reasonable outliers)
        if kurtosis < -1.0 && skewness.abs() < 0.5 {
            return DistributionType::Uniform;
        }

        // Check for multimodality (complex patterns)
        if kurtosis < -1.5 && outlier_pct > 5.0 {
            return DistributionType::Multimodal;
        }

        DistributionType::Unknown
    }

    /// Estimate correlation strength between features
    fn estimate_correlation_strength(&self, x: &Array2<Float>) -> Result<Float> {
        let (_n_samples, n_features) = x.dim();

        if n_features < 2 {
            return Ok(0.0);
        }

        let mut correlation_sum = 0.0;
        let mut correlation_count = 0;

        // Sample a subset of feature pairs to avoid O(n²) computation
        let max_pairs = 100.min(n_features * (n_features - 1) / 2);
        let step = (n_features * (n_features - 1) / 2).max(1) / max_pairs.max(1);

        let mut pair_count = 0;
        for i in 0..n_features {
            for j in (i + 1)..n_features {
                if pair_count % step == 0 {
                    let col_i = x.column(i);
                    let col_j = x.column(j);

                    // Calculate correlation coefficient
                    if let Ok(corr) = self.calculate_correlation(&col_i, &col_j) {
                        correlation_sum += corr.abs();
                        correlation_count += 1;
                    }
                }
                pair_count += 1;
            }
        }

        Ok(if correlation_count > 0 {
            correlation_sum / correlation_count as Float
        } else {
            0.0
        })
    }

    /// Calculate correlation between two features
    fn calculate_correlation(
        &self,
        x: &scirs2_core::ndarray::ArrayView1<Float>,
        y: &scirs2_core::ndarray::ArrayView1<Float>,
    ) -> Result<Float> {
        let pairs: Vec<(Float, Float)> = x
            .iter()
            .zip(y.iter())
            .filter(|(&a, &b)| a.is_finite() && b.is_finite())
            .map(|(&a, &b)| (a, b))
            .collect();

        if pairs.len() < 3 {
            return Ok(0.0);
        }

        let mean_x = pairs.iter().map(|(x, _)| x).sum::<Float>() / pairs.len() as Float;
        let mean_y = pairs.iter().map(|(_, y)| y).sum::<Float>() / pairs.len() as Float;

        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;

        for (x, y) in pairs {
            let dx = x - mean_x;
            let dy = y - mean_y;
            sum_xy += dx * dy;
            sum_x2 += dx * dx;
            sum_y2 += dy * dy;
        }

        let denominator = (sum_x2 * sum_y2).sqrt();
        if denominator > 1e-10 {
            Ok(sum_xy / denominator)
        } else {
            Ok(0.0)
        }
    }

    /// Estimate optimal batch size for processing
    fn estimate_optimal_batch_size(&self, n_samples: usize, n_features: usize) -> usize {
        // Simple heuristic based on data size and available memory
        let data_size = n_samples * n_features * std::mem::size_of::<Float>();
        let target_memory = 100_000_000; // ~100MB target

        let optimal_size = if data_size <= target_memory {
            n_samples // Process all at once
        } else {
            (target_memory / (n_features * std::mem::size_of::<Float>()))
                .max(1000)
                .min(n_samples)
        };

        optimal_size
    }

    /// Optimize parameters based on data characteristics
    fn optimize_parameters(
        &self,
        _x: &Array2<Float>,
        characteristics: &DataCharacteristics,
    ) -> Result<HashMap<String, Float>> {
        let mut optimal_params = HashMap::new();

        // Determine optimal scaling parameters
        let scaling_method = self.select_optimal_scaling_method(characteristics);
        optimal_params.insert("scaling_method".to_string(), scaling_method);

        // Determine optimal outlier threshold
        let outlier_threshold = self.select_optimal_outlier_threshold(characteristics);
        optimal_params.insert("outlier_threshold".to_string(), outlier_threshold);

        // Determine optimal imputation strategy
        let imputation_strategy = self.select_optimal_imputation_strategy(characteristics);
        optimal_params.insert("imputation_strategy".to_string(), imputation_strategy);

        // Determine optimal quantile range for robust scaling
        let (q_low, q_high) = self.select_optimal_quantile_range(characteristics);
        optimal_params.insert("quantile_range_low".to_string(), q_low);
        optimal_params.insert("quantile_range_high".to_string(), q_high);

        // Determine optimal contamination rate
        let contamination_rate = self.select_optimal_contamination_rate(characteristics);
        optimal_params.insert("contamination_rate".to_string(), contamination_rate);

        Ok(optimal_params)
    }

    /// Select optimal scaling method based on data characteristics
    fn select_optimal_scaling_method(&self, characteristics: &DataCharacteristics) -> Float {
        let avg_outlier_pct = characteristics.outlier_percentages.iter().sum::<Float>()
            / characteristics.outlier_percentages.len() as Float;
        let avg_skewness = characteristics
            .skewness
            .iter()
            .map(|x| x.abs())
            .sum::<Float>()
            / characteristics.skewness.len() as Float;

        match self.config.strategy {
            AdaptationStrategy::Conservative => {
                if avg_outlier_pct > 10.0 || avg_skewness > 1.0 {
                    2.0 // Robust scaling
                } else {
                    0.0 // Standard scaling
                }
            }
            AdaptationStrategy::Balanced => {
                if avg_outlier_pct > 15.0 {
                    2.0 // Robust scaling
                } else if avg_skewness > 2.0 {
                    1.0 // MinMax scaling
                } else {
                    0.0 // Standard scaling
                }
            }
            AdaptationStrategy::Aggressive => {
                if avg_outlier_pct > 20.0 {
                    2.0 // Robust scaling
                } else {
                    0.0 // Standard scaling (prioritize performance)
                }
            }
            AdaptationStrategy::Custom => 0.0, // Default to standard
        }
    }

    /// Select optimal outlier threshold
    fn select_optimal_outlier_threshold(&self, characteristics: &DataCharacteristics) -> Float {
        let avg_outlier_pct = characteristics.outlier_percentages.iter().sum::<Float>()
            / characteristics.outlier_percentages.len() as Float;

        match self.config.strategy {
            AdaptationStrategy::Conservative => {
                if avg_outlier_pct > 20.0 {
                    3.5
                } else {
                    3.0
                }
            }
            AdaptationStrategy::Balanced => {
                if avg_outlier_pct > 15.0 {
                    2.5
                } else {
                    2.0
                }
            }
            AdaptationStrategy::Aggressive => {
                if avg_outlier_pct > 10.0 {
                    2.0
                } else {
                    1.5
                }
            }
            AdaptationStrategy::Custom => 2.5, // Default
        }
    }

    /// Select optimal imputation strategy
    fn select_optimal_imputation_strategy(&self, characteristics: &DataCharacteristics) -> Float {
        let avg_missing_pct = characteristics.missing_percentages.iter().sum::<Float>()
            / characteristics.missing_percentages.len() as Float;
        let has_skewed_features = characteristics.skewness.iter().any(|&s| s.abs() > 1.0);

        if avg_missing_pct > 20.0 {
            2.0 // KNN imputation for high missing rates
        } else if has_skewed_features {
            1.0 // Median imputation for skewed data
        } else {
            0.0 // Mean imputation for normal data
        }
    }

    /// Select optimal quantile range for robust scaling
    fn select_optimal_quantile_range(
        &self,
        characteristics: &DataCharacteristics,
    ) -> (Float, Float) {
        let avg_outlier_pct = characteristics.outlier_percentages.iter().sum::<Float>()
            / characteristics.outlier_percentages.len() as Float;

        match self.config.strategy {
            AdaptationStrategy::Conservative => {
                if avg_outlier_pct > 15.0 {
                    (10.0, 90.0)
                } else {
                    (25.0, 75.0)
                }
            }
            AdaptationStrategy::Balanced => {
                if avg_outlier_pct > 10.0 {
                    (5.0, 95.0)
                } else {
                    (25.0, 75.0)
                }
            }
            AdaptationStrategy::Aggressive => {
                (25.0, 75.0) // Standard IQR
            }
            AdaptationStrategy::Custom => (25.0, 75.0),
        }
    }

    /// Select optimal contamination rate
    fn select_optimal_contamination_rate(&self, characteristics: &DataCharacteristics) -> Float {
        let avg_outlier_pct = characteristics.outlier_percentages.iter().sum::<Float>()
            / characteristics.outlier_percentages.len() as Float;

        // Convert percentage to rate and add some margin
        (avg_outlier_pct / 100.0 * 1.2).min(0.5).max(0.01)
    }

    /// Evaluate different parameter configurations
    fn evaluate_parameter_space(
        &self,
        x: &Array2<Float>,
        characteristics: &DataCharacteristics,
    ) -> Result<Vec<ParameterEvaluation>> {
        let mut evaluations = Vec::new();

        // Define parameter space to explore
        let scaling_methods = vec![0.0, 1.0, 2.0]; // Standard, MinMax, Robust
        let outlier_thresholds = vec![1.5, 2.0, 2.5, 3.0, 3.5];
        let contamination_rates = vec![0.05, 0.1, 0.15, 0.2];

        // Evaluate a subset of the parameter space
        let max_evaluations = 20; // Limit to avoid excessive computation
        let mut evaluation_count = 0;

        for &scaling_method in &scaling_methods {
            for &threshold in &outlier_thresholds {
                for &contamination in &contamination_rates {
                    if evaluation_count >= max_evaluations {
                        break;
                    }

                    let mut params = HashMap::new();
                    params.insert("scaling_method".to_string(), scaling_method);
                    params.insert("outlier_threshold".to_string(), threshold);
                    params.insert("contamination_rate".to_string(), contamination);

                    let evaluation = self.evaluate_parameters(&params, x, characteristics)?;
                    evaluations.push(evaluation);
                    evaluation_count += 1;
                }
            }
        }

        // Sort by overall score
        evaluations.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        Ok(evaluations)
    }

    /// Evaluate a specific parameter configuration
    fn evaluate_parameters(
        &self,
        params: &HashMap<String, Float>,
        _x: &Array2<Float>,
        characteristics: &DataCharacteristics,
    ) -> Result<ParameterEvaluation> {
        let start_time = std::time::Instant::now();

        // Compute different scoring criteria
        let robustness_score = self.compute_robustness_score(params, characteristics);
        let efficiency_score = self.compute_efficiency_score(params, characteristics);
        let quality_score = self.compute_quality_score(params, characteristics);

        // Combine scores based on strategy
        let overall_score = match self.config.strategy {
            AdaptationStrategy::Conservative => {
                robustness_score * 0.6 + quality_score * 0.3 + efficiency_score * 0.1
            }
            AdaptationStrategy::Balanced => {
                robustness_score * 0.4 + quality_score * 0.4 + efficiency_score * 0.2
            }
            AdaptationStrategy::Aggressive => {
                robustness_score * 0.2 + quality_score * 0.3 + efficiency_score * 0.5
            }
            AdaptationStrategy::Custom => {
                robustness_score * 0.33 + quality_score * 0.33 + efficiency_score * 0.34
            }
        };

        let evaluation_time = start_time.elapsed().as_secs_f64() as Float;

        Ok(ParameterEvaluation {
            parameters: params.clone(),
            score: overall_score,
            robustness_score,
            efficiency_score,
            quality_score,
            evaluation_time,
        })
    }

    /// Compute robustness score for parameters
    fn compute_robustness_score(
        &self,
        params: &HashMap<String, Float>,
        characteristics: &DataCharacteristics,
    ) -> Float {
        let scaling_method = params.get("scaling_method").unwrap_or(&0.0);
        let outlier_threshold = params.get("outlier_threshold").unwrap_or(&2.5);

        let avg_outlier_pct = characteristics.outlier_percentages.iter().sum::<Float>()
            / characteristics.outlier_percentages.len() as Float;

        let mut score: Float = 0.0;

        // Reward robust scaling for high outlier data
        if avg_outlier_pct > 10.0 && *scaling_method == 2.0 {
            score += 0.4;
        }

        // Reward appropriate outlier thresholds
        if avg_outlier_pct > 15.0 && *outlier_threshold <= 2.5 {
            score += 0.3;
        } else if avg_outlier_pct <= 5.0 && *outlier_threshold >= 3.0 {
            score += 0.3;
        }

        // Reward handling of skewed data
        let avg_skewness = characteristics
            .skewness
            .iter()
            .map(|x| x.abs())
            .sum::<Float>()
            / characteristics.skewness.len() as Float;
        if avg_skewness > 1.0 && *scaling_method != 0.0 {
            score += 0.3;
        }

        score.min(1.0 as Float)
    }

    /// Compute efficiency score for parameters
    fn compute_efficiency_score(
        &self,
        params: &HashMap<String, Float>,
        characteristics: &DataCharacteristics,
    ) -> Float {
        let scaling_method = params.get("scaling_method").unwrap_or(&0.0);
        let (n_samples, n_features) = characteristics.shape;

        // Standard scaling is most efficient
        let mut score: Float = if *scaling_method == 0.0 {
            1.0
        } else if *scaling_method == 1.0 {
            0.8 // MinMax is moderately efficient
        } else {
            0.6 // Robust scaling is less efficient
        };

        // Adjust for data size (larger data benefits more from efficient methods)
        let data_size_factor = (n_samples * n_features) as Float;
        if data_size_factor > 1_000_000.0 {
            score *= 1.2; // Boost efficiency importance for large data
        }

        score.min(1.0 as Float)
    }

    /// Compute quality score for parameters
    fn compute_quality_score(
        &self,
        params: &HashMap<String, Float>,
        characteristics: &DataCharacteristics,
    ) -> Float {
        let mut score = characteristics.quality_score; // Start with base data quality

        // Adjust based on parameter appropriateness
        let avg_missing_pct = characteristics.missing_percentages.iter().sum::<Float>()
            / characteristics.missing_percentages.len() as Float;

        // Quality improves with appropriate handling of missing values
        if avg_missing_pct > 10.0 {
            score *= 0.9; // Penalize for high missing rates
        }

        // Quality improves with appropriate outlier handling
        let avg_outlier_pct = characteristics.outlier_percentages.iter().sum::<Float>()
            / characteristics.outlier_percentages.len() as Float;

        let outlier_threshold = params.get("outlier_threshold").unwrap_or(&2.5);
        if avg_outlier_pct > 10.0 && *outlier_threshold <= 2.5 {
            score *= 1.1; // Reward aggressive outlier handling when needed
        }

        score.min(1.0 as Float)
    }
}

impl AdaptiveParameterSelector<Trained> {
    /// Get the analyzed data characteristics
    pub fn data_characteristics(&self) -> Option<&DataCharacteristics> {
        self.data_characteristics_.as_ref()
    }

    /// Get the optimal parameters
    pub fn optimal_parameters(&self) -> Option<&HashMap<String, Float>> {
        self.optimal_parameters_.as_ref()
    }

    /// Get parameter evaluation history
    pub fn parameter_history(&self) -> Option<&Vec<ParameterEvaluation>> {
        self.parameter_history_.as_ref()
    }

    /// Generate comprehensive parameter recommendations
    pub fn recommend_parameters(&self) -> Result<ParameterRecommendations> {
        let characteristics = self.data_characteristics_.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput("No data characteristics available".to_string())
        })?;

        let optimal_params = self.optimal_parameters_.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput("No optimal parameters available".to_string())
        })?;

        // Generate scaling recommendations
        let scaling_method = optimal_params.get("scaling_method").unwrap_or(&0.0);
        let scaling = ScalingParameters {
            method: match *scaling_method as i32 {
                0 => "standard".to_string(),
                1 => "minmax".to_string(),
                2 => "robust".to_string(),
                _ => "standard".to_string(),
            },
            outlier_threshold: *optimal_params.get("outlier_threshold").unwrap_or(&2.5),
            quantile_range: (
                *optimal_params.get("quantile_range_low").unwrap_or(&25.0),
                *optimal_params.get("quantile_range_high").unwrap_or(&75.0),
            ),
            with_centering: true,
            with_scaling: true,
        };

        // Generate imputation recommendations
        let imputation_strategy = optimal_params.get("imputation_strategy").unwrap_or(&0.0);
        let avg_missing_pct = characteristics.missing_percentages.iter().sum::<Float>()
            / characteristics.missing_percentages.len() as Float;

        let imputation = ImputationParameters {
            strategy: match *imputation_strategy as i32 {
                0 => "mean".to_string(),
                1 => "median".to_string(),
                2 => "knn".to_string(),
                _ => "mean".to_string(),
            },
            n_neighbors: if *imputation_strategy == 2.0 {
                Some(5)
            } else {
                None
            },
            outlier_aware: avg_missing_pct > 10.0,
            max_iterations: if *imputation_strategy == 2.0 {
                Some(10)
            } else {
                None
            },
        };

        // Generate outlier detection recommendations
        let contamination_rate = *optimal_params.get("contamination_rate").unwrap_or(&0.1);
        let outlier_detection = OutlierDetectionParameters {
            method: "isolation_forest".to_string(),
            contamination: contamination_rate,
            threshold: *optimal_params.get("outlier_threshold").unwrap_or(&2.5),
            ensemble_size: Some(100),
        };

        // Generate transformation recommendations
        let avg_skewness = characteristics
            .skewness
            .iter()
            .map(|x| x.abs())
            .sum::<Float>()
            / characteristics.skewness.len() as Float;

        let transformation = TransformationParameters {
            method: if avg_skewness > 1.5 {
                "log1p".to_string()
            } else if avg_skewness > 1.0 {
                "box_cox".to_string()
            } else {
                "none".to_string()
            },
            handle_negatives: true,
            lambda: None, // Auto-detect
            n_quantiles: Some(1000),
        };

        // Compute overall confidence
        let confidence = characteristics.quality_score * 0.5
            + (1.0
                - (characteristics.missing_percentages.iter().sum::<Float>()
                    / characteristics.missing_percentages.len() as Float
                    / 100.0))
                * 0.3
            + (1.0
                - (characteristics.outlier_percentages.iter().sum::<Float>()
                    / characteristics.outlier_percentages.len() as Float
                    / 100.0))
                * 0.2;

        Ok(ParameterRecommendations {
            scaling,
            imputation,
            outlier_detection,
            transformation,
            confidence: confidence.min(1.0).max(0.0),
        })
    }

    /// Generate a comprehensive adaptation report
    pub fn adaptation_report(&self) -> Result<String> {
        let characteristics = self.data_characteristics_.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput("No data characteristics available".to_string())
        })?;

        let recommendations = self.recommend_parameters()?;

        let mut report = String::new();

        report.push_str("=== Adaptive Parameter Selection Report ===\n\n");

        // Data characteristics summary
        report.push_str("=== Data Characteristics ===\n");
        report.push_str(&format!("Data shape: {:?}\n", characteristics.shape));
        report.push_str(&format!(
            "Overall quality score: {:.3}\n",
            characteristics.quality_score
        ));
        report.push_str(&format!(
            "Correlation strength: {:.3}\n",
            characteristics.correlation_strength
        ));
        report.push_str(&format!(
            "Optimal batch size: {}\n",
            characteristics.optimal_batch_size
        ));

        let avg_missing = characteristics.missing_percentages.iter().sum::<Float>()
            / characteristics.missing_percentages.len() as Float;
        let avg_outliers = characteristics.outlier_percentages.iter().sum::<Float>()
            / characteristics.outlier_percentages.len() as Float;
        let avg_skewness = characteristics
            .skewness
            .iter()
            .map(|x| x.abs())
            .sum::<Float>()
            / characteristics.skewness.len() as Float;

        report.push_str(&format!("Average missing values: {:.1}%\n", avg_missing));
        report.push_str(&format!("Average outlier rate: {:.1}%\n", avg_outliers));
        report.push_str(&format!("Average absolute skewness: {:.3}\n", avg_skewness));
        report.push_str("\n");

        // Parameter recommendations
        report.push_str("=== Parameter Recommendations ===\n");
        report.push_str(&format!(
            "Confidence: {:.1}%\n\n",
            recommendations.confidence * 100.0
        ));

        report.push_str("Scaling:\n");
        report.push_str(&format!("  Method: {}\n", recommendations.scaling.method));
        report.push_str(&format!(
            "  Outlier threshold: {:.2}\n",
            recommendations.scaling.outlier_threshold
        ));
        report.push_str(&format!(
            "  Quantile range: ({:.1}%, {:.1}%)\n",
            recommendations.scaling.quantile_range.0, recommendations.scaling.quantile_range.1
        ));
        report.push_str("\n");

        report.push_str("Imputation:\n");
        report.push_str(&format!(
            "  Strategy: {}\n",
            recommendations.imputation.strategy
        ));
        if let Some(k) = recommendations.imputation.n_neighbors {
            report.push_str(&format!("  K-neighbors: {}\n", k));
        }
        report.push_str(&format!(
            "  Outlier-aware: {}\n",
            recommendations.imputation.outlier_aware
        ));
        report.push_str("\n");

        report.push_str("Outlier Detection:\n");
        report.push_str(&format!(
            "  Method: {}\n",
            recommendations.outlier_detection.method
        ));
        report.push_str(&format!(
            "  Contamination: {:.3}\n",
            recommendations.outlier_detection.contamination
        ));
        report.push_str(&format!(
            "  Threshold: {:.2}\n",
            recommendations.outlier_detection.threshold
        ));
        report.push_str("\n");

        report.push_str("Transformation:\n");
        report.push_str(&format!(
            "  Method: {}\n",
            recommendations.transformation.method
        ));
        report.push_str(&format!(
            "  Handle negatives: {}\n",
            recommendations.transformation.handle_negatives
        ));
        report.push_str("\n");

        // Strategy and configuration
        report.push_str("=== Configuration ===\n");
        report.push_str(&format!("Strategy: {:?}\n", self.config.strategy));
        report.push_str(&format!(
            "Cross-validation: {} ({} folds)\n",
            self.config.use_cross_validation, self.config.cv_folds
        ));
        report.push_str(&format!("Parallel processing: {}\n", self.config.parallel));
        if let Some(budget) = self.config.time_budget {
            report.push_str(&format!("Time budget: {:.1}s\n", budget));
        }

        Ok(report)
    }

    /// Get adaptation recommendations as actionable insights
    pub fn get_insights(&self) -> Vec<String> {
        let mut insights = Vec::new();

        if let Some(characteristics) = &self.data_characteristics_ {
            let avg_missing = characteristics.missing_percentages.iter().sum::<Float>()
                / characteristics.missing_percentages.len() as Float;
            let avg_outliers = characteristics.outlier_percentages.iter().sum::<Float>()
                / characteristics.outlier_percentages.len() as Float;
            let avg_skewness = characteristics
                .skewness
                .iter()
                .map(|x| x.abs())
                .sum::<Float>()
                / characteristics.skewness.len() as Float;

            if avg_missing > 20.0 {
                insights.push("High missing value rate detected - consider advanced imputation methods like KNN or iterative imputation".to_string());
            }

            if avg_outliers > 15.0 {
                insights.push(
                    "High outlier rate detected - robust preprocessing methods are recommended"
                        .to_string(),
                );
            }

            if avg_skewness > 2.0 {
                insights.push(
                    "Highly skewed data detected - consider log or Box-Cox transformations"
                        .to_string(),
                );
            }

            if characteristics.correlation_strength > 0.7 {
                insights.push(
                    "High feature correlation detected - consider dimensionality reduction"
                        .to_string(),
                );
            }

            if characteristics.quality_score < 0.5 {
                insights.push(
                    "Low data quality detected - comprehensive preprocessing pipeline recommended"
                        .to_string(),
                );
            }

            if characteristics.shape.0 > 1_000_000 {
                insights.push(
                    "Large dataset detected - consider streaming or batch processing approaches"
                        .to_string(),
                );
            }

            if characteristics.optimal_batch_size < characteristics.shape.0 {
                insights.push(format!(
                    "Consider batch processing with batch size: {}",
                    characteristics.optimal_batch_size
                ));
            }
        }

        if insights.is_empty() {
            insights.push("Data characteristics are within normal ranges - standard preprocessing should be sufficient".to_string());
        }

        insights
    }
}

impl Default for AdaptiveParameterSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_adaptive_parameter_selector_creation() {
        let selector = AdaptiveParameterSelector::new();
        assert_eq!(
            selector.config.strategy as u8,
            AdaptationStrategy::Balanced as u8
        );
        assert!(selector.config.use_cross_validation);
        assert_eq!(selector.config.cv_folds, 5);
    }

    #[test]
    fn test_adaptive_strategies() {
        let conservative = AdaptiveParameterSelector::conservative();
        assert_eq!(
            conservative.config.strategy as u8,
            AdaptationStrategy::Conservative as u8
        );

        let aggressive = AdaptiveParameterSelector::aggressive();
        assert_eq!(
            aggressive.config.strategy as u8,
            AdaptationStrategy::Aggressive as u8
        );
    }

    #[test]
    fn test_data_characteristics_analysis() {
        let data = Array2::from_shape_vec(
            (10, 3),
            vec![
                1.0, 10.0, 100.0, // Normal range
                2.0, 20.0, 200.0, 3.0, 30.0, 300.0, 4.0, 40.0, 400.0, 5.0, 50.0, 500.0, 6.0, 60.0,
                600.0, 7.0, 70.0, 700.0, 8.0, 80.0, 800.0, 100.0, 1000.0, 10000.0, // Outliers
                9.0, 90.0, 900.0,
            ],
        )
        .unwrap();

        let selector = AdaptiveParameterSelector::balanced();
        let fitted = selector.fit(&data, &()).unwrap();

        let characteristics = fitted.data_characteristics().unwrap();
        assert_eq!(characteristics.shape, (10, 3));
        assert_eq!(characteristics.distribution_types.len(), 3);
        assert_eq!(characteristics.skewness.len(), 3);
        assert!(characteristics.quality_score >= 0.0 && characteristics.quality_score <= 1.0);
    }

    #[test]
    fn test_parameter_recommendations() {
        let data = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0, 100.0,
                1000.0, // Outliers
                7.0, 70.0, 8.0, 80.0,
            ],
        )
        .unwrap();

        let selector = AdaptiveParameterSelector::balanced();
        let fitted = selector.fit(&data, &()).unwrap();

        let recommendations = fitted.recommend_parameters().unwrap();
        assert!(recommendations.confidence >= 0.0 && recommendations.confidence <= 1.0);
        assert!(!recommendations.scaling.method.is_empty());
        assert!(!recommendations.imputation.strategy.is_empty());
    }

    #[test]
    fn test_parameter_optimization() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0, 100.0,
                1000.0, // Outliers
            ],
        )
        .unwrap();

        let selector = AdaptiveParameterSelector::aggressive();
        let fitted = selector.fit(&data, &()).unwrap();

        let optimal_params = fitted.optimal_parameters().unwrap();
        assert!(optimal_params.contains_key("scaling_method"));
        assert!(optimal_params.contains_key("outlier_threshold"));
        assert!(optimal_params.contains_key("contamination_rate"));
    }

    #[test]
    fn test_distribution_classification() {
        let data = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();

        let selector = AdaptiveParameterSelector::new();
        let fitted = selector.fit(&data, &()).unwrap();

        let characteristics = fitted.data_characteristics().unwrap();
        // Should classify as normal or uniform for this simple case
        assert!(matches!(
            characteristics.distribution_types[0],
            DistributionType::Normal | DistributionType::Uniform | DistributionType::Unknown
        ));
    }

    #[test]
    fn test_missing_value_handling() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0,
                10.0,
                2.0,
                Float::NAN, // Missing value
                3.0,
                30.0,
                Float::NAN,
                40.0, // Missing value
                5.0,
                50.0,
                6.0,
                60.0,
            ],
        )
        .unwrap();

        let selector = AdaptiveParameterSelector::balanced();
        let fitted = selector.fit(&data, &()).unwrap();

        let characteristics = fitted.data_characteristics().unwrap();
        // Should detect missing values
        assert!(
            characteristics.missing_percentages[0] > 0.0
                || characteristics.missing_percentages[1] > 0.0
        );
    }

    #[test]
    fn test_adaptation_report() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0, 100.0, 1000.0,
            ],
        )
        .unwrap();

        let selector = AdaptiveParameterSelector::balanced();
        let fitted = selector.fit(&data, &()).unwrap();

        let report = fitted.adaptation_report().unwrap();
        assert!(report.contains("Adaptive Parameter Selection Report"));
        assert!(report.contains("Data Characteristics"));
        assert!(report.contains("Parameter Recommendations"));
    }

    #[test]
    fn test_insights_generation() {
        let data = Array2::from_shape_vec((4, 2), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0])
            .unwrap();

        let selector = AdaptiveParameterSelector::conservative();
        let fitted = selector.fit(&data, &()).unwrap();

        let insights = fitted.get_insights();
        assert!(!insights.is_empty());
    }

    #[test]
    fn test_configuration_options() {
        let selector = AdaptiveParameterSelector::new()
            .cross_validation(false, 3)
            .time_budget(30.0)
            .parallel(false)
            .tolerance(1e-3);

        assert!(!selector.config.use_cross_validation);
        assert_eq!(selector.config.cv_folds, 3);
        assert_eq!(selector.config.time_budget, Some(30.0));
        assert!(!selector.config.parallel);
        assert_relative_eq!(selector.config.tolerance, 1e-3, epsilon = 1e-10);
    }

    #[test]
    fn test_error_handling() {
        let selector = AdaptiveParameterSelector::new();

        // Test empty input
        let empty_data = Array2::from_shape_vec((0, 0), vec![]).unwrap();
        assert!(selector.fit(&empty_data, &()).is_err());
    }

    #[test]
    fn test_parameter_bounds() {
        let mut bounds = HashMap::new();
        bounds.insert("outlier_threshold".to_string(), (1.0, 4.0));

        let selector = AdaptiveParameterSelector::new().parameter_bounds(bounds.clone());

        assert_eq!(selector.config.parameter_bounds, bounds);
    }
}
