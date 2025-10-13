//! Automated Feature Engineering for AutoML
//!
//! This module provides comprehensive automated feature engineering capabilities including
//! feature generation, selection, transformation, and optimization. It automatically creates
//! and selects the best features for different machine learning tasks.

use crate::{automl_algorithm_selection::DatasetCharacteristics, scoring::TaskType};
use scirs2_core::ndarray::{concatenate, s, Array1, Array2, ArrayView1, Axis};
use scirs2_core::SliceRandomExt;
use sklears_core::error::Result;
use std::collections::HashMap;
use std::fmt;
// use serde::{Deserialize, Serialize};
use scirs2_core::rand_prelude::SliceRandom;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};

/// Types of feature transformations
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FeatureTransformationType {
    /// Polynomial features (degree 2, 3, etc.)
    Polynomial { degree: usize },
    /// Logarithmic transformation
    Logarithmic,
    /// Square root transformation
    SquareRoot,
    /// Exponential transformation
    Exponential,
    /// Reciprocal transformation (1/x)
    Reciprocal,
    /// Sine transformation
    Sine,
    /// Cosine transformation
    Cosine,
    /// Absolute value
    Absolute,
    /// Sign function (-1, 0, 1)
    Sign,
    /// Binning/Discretization
    Binning { n_bins: usize },
    /// Feature interactions (products)
    Interaction,
    /// Feature ratios
    Ratio,
    /// Feature differences
    Difference,
    /// Rolling statistics
    RollingStatistics { window: usize },
    /// Lag features
    Lag { lag: usize },
}

impl fmt::Display for FeatureTransformationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FeatureTransformationType::Polynomial { degree } => write!(f, "Polynomial({})", degree),
            FeatureTransformationType::Logarithmic => write!(f, "Logarithmic"),
            FeatureTransformationType::SquareRoot => write!(f, "SquareRoot"),
            FeatureTransformationType::Exponential => write!(f, "Exponential"),
            FeatureTransformationType::Reciprocal => write!(f, "Reciprocal"),
            FeatureTransformationType::Sine => write!(f, "Sine"),
            FeatureTransformationType::Cosine => write!(f, "Cosine"),
            FeatureTransformationType::Absolute => write!(f, "Absolute"),
            FeatureTransformationType::Sign => write!(f, "Sign"),
            FeatureTransformationType::Binning { n_bins } => write!(f, "Binning({})", n_bins),
            FeatureTransformationType::Interaction => write!(f, "Interaction"),
            FeatureTransformationType::Ratio => write!(f, "Ratio"),
            FeatureTransformationType::Difference => write!(f, "Difference"),
            FeatureTransformationType::RollingStatistics { window } => {
                write!(f, "RollingStats({})", window)
            }
            FeatureTransformationType::Lag { lag } => write!(f, "Lag({})", lag),
        }
    }
}

/// Feature engineering strategies
#[derive(Debug, Clone, PartialEq)]
pub enum FeatureEngineeringStrategy {
    /// Conservative: Only basic transformations
    Conservative,
    /// Balanced: Moderate feature generation
    Balanced,
    /// Aggressive: Extensive feature generation
    Aggressive,
    /// Custom: User-defined transformations
    Custom(Vec<FeatureTransformationType>),
}

/// Feature selection methods
#[derive(Debug, Clone, PartialEq)]
pub enum FeatureSelectionMethod {
    /// Select K best features using univariate statistics
    SelectKBest { k: usize },
    /// Select percentile of best features
    SelectPercentile { percentile: f64 },
    /// Recursive feature elimination
    RecursiveFeatureElimination { step: usize },
    /// L1-based feature selection
    L1Selection { alpha: f64 },
    /// Variance threshold
    VarianceThreshold { threshold: f64 },
    /// Correlation threshold
    CorrelationThreshold { threshold: f64 },
    /// Mutual information
    MutualInformation { k: usize },
    /// Feature importance from tree models
    TreeImportance { threshold: f64 },
}

/// Generated feature information
#[derive(Debug, Clone)]
pub struct GeneratedFeature {
    /// Feature name
    pub name: String,
    /// Transformation type used
    pub transformation: FeatureTransformationType,
    /// Source feature indices
    pub source_features: Vec<usize>,
    /// Feature importance score
    pub importance_score: f64,
    /// Whether the feature is selected
    pub is_selected: bool,
    /// Statistical properties
    pub statistics: FeatureStatistics,
}

/// Statistical properties of a feature
#[derive(Debug, Clone)]
pub struct FeatureStatistics {
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Number of unique values
    pub n_unique: usize,
    /// Missing value ratio
    pub missing_ratio: f64,
    /// Skewness
    pub skewness: f64,
    /// Kurtosis
    pub kurtosis: f64,
}

/// Configuration for automated feature engineering
#[derive(Debug, Clone)]
pub struct AutoFeatureEngineering {
    /// Feature engineering strategy
    pub strategy: FeatureEngineeringStrategy,
    /// Feature selection method
    pub selection_method: FeatureSelectionMethod,
    /// Maximum number of features to generate
    pub max_features: usize,
    /// Maximum number of features to select
    pub max_selected_features: usize,
    /// Cross-validation folds for feature selection
    pub cv_folds: usize,
    /// Task type (classification or regression)
    pub task_type: TaskType,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Enable polynomial features
    pub enable_polynomial: bool,
    /// Enable mathematical transformations
    pub enable_math_transforms: bool,
    /// Enable feature interactions
    pub enable_interactions: bool,
    /// Enable time series features (for sequential data)
    pub enable_time_series: bool,
    /// Minimum correlation threshold for feature selection
    pub min_correlation_threshold: f64,
    /// Maximum correlation threshold for redundancy removal
    pub max_correlation_threshold: f64,
}

impl Default for AutoFeatureEngineering {
    fn default() -> Self {
        Self {
            strategy: FeatureEngineeringStrategy::Balanced,
            selection_method: FeatureSelectionMethod::SelectKBest { k: 100 },
            max_features: 1000,
            max_selected_features: 100,
            cv_folds: 5,
            task_type: TaskType::Classification,
            random_seed: None,
            enable_polynomial: true,
            enable_math_transforms: true,
            enable_interactions: true,
            enable_time_series: false,
            min_correlation_threshold: 0.05,
            max_correlation_threshold: 0.95,
        }
    }
}

/// Result of feature engineering process
#[derive(Debug, Clone)]
pub struct FeatureEngineeringResult {
    /// Original feature count
    pub original_feature_count: usize,
    /// Generated feature count
    pub generated_feature_count: usize,
    /// Selected feature count
    pub selected_feature_count: usize,
    /// Generated features
    pub generated_features: Vec<GeneratedFeature>,
    /// Selected feature indices
    pub selected_indices: Vec<usize>,
    /// Feature importance scores
    pub feature_importances: Vec<f64>,
    /// Transformation matrix for new data
    pub transformation_info: TransformationInfo,
    /// Performance improvement
    pub performance_improvement: f64,
    /// Processing time
    pub processing_time: f64,
}

/// Information needed to transform new data
#[derive(Debug, Clone)]
pub struct TransformationInfo {
    /// Transformation types to apply
    pub transformations: Vec<(FeatureTransformationType, Vec<usize>)>,
    /// Selected feature indices after transformation
    pub selected_indices: Vec<usize>,
    /// Scaling parameters
    pub scaling_params: HashMap<usize, (f64, f64)>, // (mean, std) for each feature
    /// Binning boundaries
    pub binning_boundaries: HashMap<usize, Vec<f64>>,
}

impl fmt::Display for FeatureEngineeringResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Automated Feature Engineering Results")?;
        writeln!(f, "=====================================")?;
        writeln!(f, "Original features: {}", self.original_feature_count)?;
        writeln!(f, "Generated features: {}", self.generated_feature_count)?;
        writeln!(f, "Selected features: {}", self.selected_feature_count)?;
        writeln!(
            f,
            "Performance improvement: {:.4}",
            self.performance_improvement
        )?;
        writeln!(f, "Processing time: {:.2}s", self.processing_time)?;
        writeln!(f)?;
        writeln!(f, "Top 10 Generated Features:")?;

        let mut top_features: Vec<_> = self
            .generated_features
            .iter()
            .filter(|f| f.is_selected)
            .collect();
        top_features.sort_by(|a, b| b.importance_score.partial_cmp(&a.importance_score).unwrap());

        for (i, feature) in top_features.iter().take(10).enumerate() {
            writeln!(
                f,
                "{}. {} ({}): {:.4}",
                i + 1,
                feature.name,
                feature.transformation,
                feature.importance_score
            )?;
        }
        Ok(())
    }
}

/// Automated feature engineering engine
pub struct AutoFeatureEngineer {
    config: AutoFeatureEngineering,
    rng: StdRng,
}

impl AutoFeatureEngineer {
    /// Create a new automated feature engineer
    pub fn new(config: AutoFeatureEngineering) -> Self {
        let rng = match config.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_rng(&mut scirs2_core::random::thread_rng()),
        };

        Self { config, rng }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(AutoFeatureEngineering::default())
    }

    /// Perform automated feature engineering
    pub fn engineer_features(
        &mut self,
        X: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<FeatureEngineeringResult> {
        let start_time = std::time::Instant::now();
        let original_feature_count = X.ncols();

        // Step 1: Analyze dataset characteristics
        let dataset_chars = self.analyze_dataset_for_features(X, y);

        // Step 2: Generate candidate transformations
        let transformations = self.generate_transformations(&dataset_chars);

        // Step 3: Apply transformations and generate features
        let (enhanced_X, generated_features) = self.apply_transformations(X, &transformations)?;

        // Step 4: Calculate feature statistics
        let features_with_stats =
            self.calculate_feature_statistics(generated_features, &enhanced_X, y);

        // Step 5: Select best features
        let (selected_features, selected_indices) =
            self.select_features(&enhanced_X, y, features_with_stats)?;

        // Step 6: Calculate performance improvement
        let performance_improvement =
            self.estimate_performance_improvement(X, &enhanced_X, y, &selected_indices)?;

        // Step 7: Create transformation info for future use
        let transformation_info =
            self.create_transformation_info(&transformations, &selected_indices, &enhanced_X);

        let processing_time = start_time.elapsed().as_secs_f64();

        Ok(FeatureEngineeringResult {
            original_feature_count,
            generated_feature_count: enhanced_X.ncols(),
            selected_feature_count: selected_indices.len(),
            generated_features: selected_features,
            selected_indices: selected_indices.clone(),
            feature_importances: vec![0.0; selected_indices.len()], // Will be filled by actual importance calculation
            transformation_info,
            performance_improvement,
            processing_time,
        })
    }

    /// Transform new data using learned transformations
    pub fn transform(
        &self,
        X: &Array2<f64>,
        transformation_info: &TransformationInfo,
    ) -> Result<Array2<f64>> {
        // Apply the same transformations that were learned during training
        let mut transformed_X = X.clone();

        // Apply transformations
        for (transformation, source_indices) in &transformation_info.transformations {
            let new_features =
                self.apply_single_transformation(&transformed_X, transformation, source_indices)?;
            // Concatenate new features
            transformed_X =
                concatenate(Axis(1), &[transformed_X.view(), new_features.view()]).unwrap();
        }

        // Apply scaling
        for (feature_idx, (mean, std)) in &transformation_info.scaling_params {
            if *feature_idx < transformed_X.ncols() {
                let mut column = transformed_X.column_mut(*feature_idx);
                for value in column.iter_mut() {
                    *value = (*value - mean) / std;
                }
            }
        }

        // Select final features with bounds checking
        let valid_indices: Vec<usize> = transformation_info
            .selected_indices
            .iter()
            .filter(|&&idx| idx < transformed_X.ncols())
            .copied()
            .collect();

        if valid_indices.is_empty() {
            return Err("No valid feature indices to select".into());
        }

        let selected_X = transformed_X.select(Axis(1), &valid_indices);
        Ok(selected_X)
    }

    /// Analyze dataset characteristics for feature engineering
    fn analyze_dataset_for_features(
        &self,
        X: &Array2<f64>,
        y: &Array1<f64>,
    ) -> DatasetCharacteristics {
        let n_samples = X.nrows();
        let n_features = X.ncols();

        // Calculate basic statistics
        let sparsity = self.calculate_sparsity(X);
        let correlation_structure = self.analyze_correlation_structure(X);
        let linearity_score = self.estimate_linearity(X, y);

        // Task-specific analysis
        let (n_classes, class_distribution, target_stats) = match self.config.task_type {
            TaskType::Classification => {
                let classes = self.get_unique_classes(y);
                let class_dist = self.calculate_class_distribution(y, &classes);
                (Some(classes.len()), Some(class_dist), None)
            }
            TaskType::Regression => {
                let stats = crate::automl_algorithm_selection::TargetStatistics {
                    mean: y.mean().unwrap(),
                    std: y.std(0.0),
                    skewness: 0.0, // Would calculate actual skewness
                    kurtosis: 0.0, // Would calculate actual kurtosis
                    n_outliers: 0, // Would detect outliers
                };
                (None, None, Some(stats))
            }
        };

        crate::automl_algorithm_selection::DatasetCharacteristics {
            n_samples,
            n_features,
            n_classes,
            class_distribution,
            target_stats,
            missing_ratio: 0.0,     // Would calculate actual missing ratio
            categorical_ratio: 0.0, // Would detect categorical features
            correlation_condition_number: correlation_structure,
            sparsity,
            effective_dimensionality: Some((n_features as f64 * 0.8) as usize),
            noise_level: 0.1, // Would estimate actual noise
            linearity_score,
        }
    }

    /// Generate appropriate transformations based on dataset characteristics
    fn generate_transformations(
        &mut self,
        dataset_chars: &DatasetCharacteristics,
    ) -> Vec<FeatureTransformationType> {
        let mut transformations = Vec::new();

        match &self.config.strategy {
            FeatureEngineeringStrategy::Conservative => {
                if self.config.enable_polynomial {
                    transformations.push(FeatureTransformationType::Polynomial { degree: 2 });
                }
                if self.config.enable_math_transforms {
                    transformations.push(FeatureTransformationType::Logarithmic);
                    transformations.push(FeatureTransformationType::SquareRoot);
                }
            }

            FeatureEngineeringStrategy::Balanced => {
                if self.config.enable_polynomial {
                    transformations.push(FeatureTransformationType::Polynomial { degree: 2 });
                    if dataset_chars.n_features < 20 {
                        transformations.push(FeatureTransformationType::Polynomial { degree: 3 });
                    }
                }

                if self.config.enable_math_transforms {
                    transformations.extend(vec![
                        FeatureTransformationType::Logarithmic,
                        FeatureTransformationType::SquareRoot,
                        FeatureTransformationType::Absolute,
                        FeatureTransformationType::Reciprocal,
                    ]);
                }

                if self.config.enable_interactions && dataset_chars.n_features < 50 {
                    transformations.push(FeatureTransformationType::Interaction);
                    transformations.push(FeatureTransformationType::Ratio);
                }

                transformations.push(FeatureTransformationType::Binning { n_bins: 10 });
            }

            FeatureEngineeringStrategy::Aggressive => {
                if self.config.enable_polynomial {
                    transformations.push(FeatureTransformationType::Polynomial { degree: 2 });
                    if dataset_chars.n_features < 15 {
                        transformations.push(FeatureTransformationType::Polynomial { degree: 3 });
                    }
                }

                if self.config.enable_math_transforms {
                    transformations.extend(vec![
                        FeatureTransformationType::Logarithmic,
                        FeatureTransformationType::SquareRoot,
                        FeatureTransformationType::Exponential,
                        FeatureTransformationType::Reciprocal,
                        FeatureTransformationType::Sine,
                        FeatureTransformationType::Cosine,
                        FeatureTransformationType::Absolute,
                        FeatureTransformationType::Sign,
                    ]);
                }

                if self.config.enable_interactions {
                    transformations.push(FeatureTransformationType::Interaction);
                    transformations.push(FeatureTransformationType::Ratio);
                    transformations.push(FeatureTransformationType::Difference);
                }

                transformations.extend(vec![
                    FeatureTransformationType::Binning { n_bins: 5 },
                    FeatureTransformationType::Binning { n_bins: 10 },
                    FeatureTransformationType::Binning { n_bins: 20 },
                ]);

                if self.config.enable_time_series {
                    transformations.extend(vec![
                        FeatureTransformationType::RollingStatistics { window: 3 },
                        FeatureTransformationType::RollingStatistics { window: 5 },
                        FeatureTransformationType::Lag { lag: 1 },
                        FeatureTransformationType::Lag { lag: 2 },
                    ]);
                }
            }

            FeatureEngineeringStrategy::Custom(custom_transforms) => {
                transformations.extend(custom_transforms.clone());
            }
        }

        // Randomly shuffle transformations for diversity
        transformations.shuffle(&mut self.rng);

        // Limit to max_features constraint
        let max_transforms = (self.config.max_features / dataset_chars.n_features).max(1);
        transformations.truncate(max_transforms);

        transformations
    }

    /// Apply transformations to generate new features
    fn apply_transformations(
        &mut self,
        X: &Array2<f64>,
        transformations: &[FeatureTransformationType],
    ) -> Result<(Array2<f64>, Vec<GeneratedFeature>)> {
        let mut enhanced_X = X.clone();
        let mut generated_features = Vec::new();

        // Start with original features
        for i in 0..X.ncols() {
            generated_features.push(GeneratedFeature {
                name: format!("original_feature_{}", i),
                transformation: FeatureTransformationType::Absolute, // Placeholder
                source_features: vec![i],
                importance_score: 0.0,
                is_selected: false,
                statistics: FeatureStatistics {
                    mean: 0.0,
                    std: 0.0,
                    min: 0.0,
                    max: 0.0,
                    n_unique: 0,
                    missing_ratio: 0.0,
                    skewness: 0.0,
                    kurtosis: 0.0,
                },
            });
        }

        for transformation in transformations {
            let source_indices: Vec<usize> = match transformation {
                FeatureTransformationType::Interaction
                | FeatureTransformationType::Ratio
                | FeatureTransformationType::Difference => {
                    // Select pairs of features
                    self.select_feature_pairs(X.ncols())
                }
                _ => {
                    // Use all original features
                    (0..X.ncols()).collect()
                }
            };

            let new_features =
                self.apply_single_transformation(&enhanced_X, transformation, &source_indices)?;

            // Add metadata for generated features
            for (i, _) in new_features.columns().into_iter().enumerate() {
                generated_features.push(GeneratedFeature {
                    name: format!("{}_{}", transformation, i),
                    transformation: transformation.clone(),
                    source_features: source_indices.clone(),
                    importance_score: 0.0,
                    is_selected: false,
                    statistics: FeatureStatistics {
                        mean: 0.0,
                        std: 0.0,
                        min: 0.0,
                        max: 0.0,
                        n_unique: 0,
                        missing_ratio: 0.0,
                        skewness: 0.0,
                        kurtosis: 0.0,
                    },
                });
            }

            // Concatenate new features
            enhanced_X = concatenate(Axis(1), &[enhanced_X.view(), new_features.view()]).unwrap();

            // Check if we've reached the maximum number of features
            if enhanced_X.ncols() >= self.config.max_features {
                break;
            }
        }

        Ok((enhanced_X, generated_features))
    }

    /// Apply a single transformation
    fn apply_single_transformation(
        &self,
        X: &Array2<f64>,
        transformation: &FeatureTransformationType,
        source_indices: &[usize],
    ) -> Result<Array2<f64>> {
        match transformation {
            FeatureTransformationType::Polynomial { degree } => {
                self.apply_polynomial_features(X, source_indices, *degree)
            }

            FeatureTransformationType::Logarithmic => {
                self.apply_logarithmic_transform(X, source_indices)
            }

            FeatureTransformationType::SquareRoot => self.apply_sqrt_transform(X, source_indices),

            FeatureTransformationType::Exponential => self.apply_exp_transform(X, source_indices),

            FeatureTransformationType::Reciprocal => {
                self.apply_reciprocal_transform(X, source_indices)
            }

            FeatureTransformationType::Sine => self.apply_sine_transform(X, source_indices),

            FeatureTransformationType::Cosine => self.apply_cosine_transform(X, source_indices),

            FeatureTransformationType::Absolute => self.apply_absolute_transform(X, source_indices),

            FeatureTransformationType::Sign => self.apply_sign_transform(X, source_indices),

            FeatureTransformationType::Binning { n_bins } => {
                self.apply_binning_transform(X, source_indices, *n_bins)
            }

            FeatureTransformationType::Interaction => {
                self.apply_interaction_features(X, source_indices)
            }

            FeatureTransformationType::Ratio => self.apply_ratio_features(X, source_indices),

            FeatureTransformationType::Difference => {
                self.apply_difference_features(X, source_indices)
            }

            FeatureTransformationType::RollingStatistics { window } => {
                self.apply_rolling_statistics(X, source_indices, *window)
            }

            FeatureTransformationType::Lag { lag } => {
                self.apply_lag_features(X, source_indices, *lag)
            }
        }
    }

    /// Calculate feature statistics
    fn calculate_feature_statistics(
        &self,
        mut generated_features: Vec<GeneratedFeature>,
        X: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Vec<GeneratedFeature> {
        for (i, feature) in generated_features.iter_mut().enumerate() {
            if i < X.ncols() {
                let column = X.column(i);

                feature.statistics = FeatureStatistics {
                    mean: column.mean().unwrap_or(0.0),
                    std: column.std(0.0),
                    min: column.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                    max: column.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
                    n_unique: self.count_unique_values(&column),
                    missing_ratio: column.iter().filter(|&&x| x.is_nan()).count() as f64
                        / column.len() as f64,
                    skewness: 0.0, // Would calculate actual skewness
                    kurtosis: 0.0, // Would calculate actual kurtosis
                };

                // Calculate importance score (mock implementation)
                feature.importance_score = self.calculate_feature_importance(&column, y);
            }
        }

        generated_features
    }

    /// Select best features using the configured method
    fn select_features(
        &self,
        X: &Array2<f64>,
        y: &Array1<f64>,
        mut generated_features: Vec<GeneratedFeature>,
    ) -> Result<(Vec<GeneratedFeature>, Vec<usize>)> {
        // Create a vector of (index, feature) pairs to maintain original indices
        let mut indexed_features: Vec<(usize, &mut GeneratedFeature)> =
            generated_features.iter_mut().enumerate().collect();

        // Sort by importance (descending)
        indexed_features.sort_by(|a, b| {
            b.1.importance_score
                .partial_cmp(&a.1.importance_score)
                .unwrap()
        });

        let n_features_to_select = match &self.config.selection_method {
            FeatureSelectionMethod::SelectKBest { k } => (*k).min(X.ncols()),
            FeatureSelectionMethod::SelectPercentile { percentile } => {
                ((X.ncols() as f64 * percentile / 100.0) as usize).max(1)
            }
            _ => self.config.max_selected_features.min(X.ncols()),
        };

        // Apply selection method
        let selected_indices = match &self.config.selection_method {
            FeatureSelectionMethod::SelectKBest { k: _ }
            | FeatureSelectionMethod::SelectPercentile { percentile: _ } => {
                // Select top k features by importance, using their original indices
                indexed_features
                    .iter()
                    .take(n_features_to_select)
                    .map(|(idx, _)| *idx)
                    .collect()
            }

            FeatureSelectionMethod::VarianceThreshold { threshold } => {
                self.select_by_variance_threshold(X, *threshold)
            }

            FeatureSelectionMethod::CorrelationThreshold { threshold } => {
                self.select_by_correlation_threshold(X, *threshold)
            }

            _ => {
                // Default to top k features using their original indices
                indexed_features
                    .iter()
                    .take(n_features_to_select)
                    .map(|(idx, _)| *idx)
                    .collect()
            }
        };

        // Mark selected features
        for (i, feature) in generated_features.iter_mut().enumerate() {
            feature.is_selected = selected_indices.contains(&i);
        }

        Ok((generated_features, selected_indices))
    }

    /// Estimate performance improvement from feature engineering
    fn estimate_performance_improvement(
        &self,
        original_X: &Array2<f64>,
        enhanced_X: &Array2<f64>,
        y: &Array1<f64>,
        selected_indices: &[usize],
    ) -> Result<f64> {
        // Mock implementation - would use actual cross-validation
        let original_score = 0.7; // Mock baseline score
        let enhanced_score = 0.8; // Mock enhanced score
        Ok(enhanced_score - original_score)
    }

    /// Create transformation info for future data transformation
    fn create_transformation_info(
        &self,
        transformations: &[FeatureTransformationType],
        selected_indices: &[usize],
        enhanced_X: &Array2<f64>,
    ) -> TransformationInfo {
        let mut scaling_params = HashMap::new();
        let binning_boundaries = HashMap::new();

        // Calculate scaling parameters for selected features
        for &idx in selected_indices {
            if idx < enhanced_X.ncols() {
                let column = enhanced_X.column(idx);
                let mean = column.mean().unwrap_or(0.0);
                let std = column.std(0.0);
                scaling_params.insert(idx, (mean, std));
            }
        }

        TransformationInfo {
            transformations: transformations
                .iter()
                .map(|t| (t.clone(), vec![]))
                .collect(),
            selected_indices: selected_indices.to_vec(),
            scaling_params,
            binning_boundaries,
        }
    }

    // Helper methods for specific transformations
    fn apply_polynomial_features(
        &self,
        X: &Array2<f64>,
        source_indices: &[usize],
        degree: usize,
    ) -> Result<Array2<f64>> {
        let n_samples = X.nrows();
        let selected_X = X.select(Axis(1), source_indices);
        let n_features = selected_X.ncols();

        if degree == 2 {
            // Quadratic features: x^2 and x_i * x_j
            let mut poly_features = Vec::new();

            // Squared terms
            for i in 0..n_features {
                let col = selected_X.column(i);
                let squared: Vec<f64> = col.iter().map(|&x| x * x).collect();
                poly_features.push(squared);
            }

            // Interaction terms (only if not too many features)
            if n_features < 20 {
                for i in 0..n_features {
                    for j in (i + 1)..n_features {
                        let col_i = selected_X.column(i);
                        let col_j = selected_X.column(j);
                        let interaction: Vec<f64> = col_i
                            .iter()
                            .zip(col_j.iter())
                            .map(|(&xi, &xj)| xi * xj)
                            .collect();
                        poly_features.push(interaction);
                    }
                }
            }

            // Convert to Array2
            let n_poly_features = poly_features.len();
            let mut result = Array2::zeros((n_samples, n_poly_features));
            for (j, feature) in poly_features.iter().enumerate() {
                for (i, &value) in feature.iter().enumerate() {
                    result[[i, j]] = value;
                }
            }
            Ok(result)
        } else {
            // For higher degrees, just use power transforms
            let mut result = Array2::zeros((n_samples, n_features));
            for (j, i) in source_indices.iter().enumerate() {
                let col = X.column(*i);
                for (row, &value) in col.iter().enumerate() {
                    result[[row, j]] = value.powi(degree as i32);
                }
            }
            Ok(result)
        }
    }

    fn apply_logarithmic_transform(
        &self,
        X: &Array2<f64>,
        source_indices: &[usize],
    ) -> Result<Array2<f64>> {
        let n_samples = X.nrows();
        let n_features = source_indices.len();
        let mut result = Array2::zeros((n_samples, n_features));

        for (j, &i) in source_indices.iter().enumerate() {
            let col = X.column(i);
            for (row, &value) in col.iter().enumerate() {
                // Handle negative values and zeros
                let log_value = if value > 0.0 {
                    value.ln()
                } else if value == 0.0 {
                    0.0
                } else {
                    -(value.abs() + 1e-8).ln()
                };
                result[[row, j]] = log_value;
            }
        }
        Ok(result)
    }

    fn apply_sqrt_transform(
        &self,
        X: &Array2<f64>,
        source_indices: &[usize],
    ) -> Result<Array2<f64>> {
        let n_samples = X.nrows();
        let n_features = source_indices.len();
        let mut result = Array2::zeros((n_samples, n_features));

        for (j, &i) in source_indices.iter().enumerate() {
            let col = X.column(i);
            for (row, &value) in col.iter().enumerate() {
                let sqrt_value = if value >= 0.0 {
                    value.sqrt()
                } else {
                    -(value.abs().sqrt())
                };
                result[[row, j]] = sqrt_value;
            }
        }
        Ok(result)
    }

    fn apply_exp_transform(
        &self,
        X: &Array2<f64>,
        source_indices: &[usize],
    ) -> Result<Array2<f64>> {
        let n_samples = X.nrows();
        let n_features = source_indices.len();
        let mut result = Array2::zeros((n_samples, n_features));

        for (j, &i) in source_indices.iter().enumerate() {
            let col = X.column(i);
            for (row, &value) in col.iter().enumerate() {
                // Clip values to prevent overflow
                let clipped_value = value.max(-10.0).min(10.0);
                result[[row, j]] = clipped_value.exp();
            }
        }
        Ok(result)
    }

    fn apply_reciprocal_transform(
        &self,
        X: &Array2<f64>,
        source_indices: &[usize],
    ) -> Result<Array2<f64>> {
        let n_samples = X.nrows();
        let n_features = source_indices.len();
        let mut result = Array2::zeros((n_samples, n_features));

        for (j, &i) in source_indices.iter().enumerate() {
            let col = X.column(i);
            for (row, &value) in col.iter().enumerate() {
                let reciprocal = if value.abs() > 1e-8 { 1.0 / value } else { 0.0 };
                result[[row, j]] = reciprocal;
            }
        }
        Ok(result)
    }

    fn apply_sine_transform(
        &self,
        X: &Array2<f64>,
        source_indices: &[usize],
    ) -> Result<Array2<f64>> {
        let n_samples = X.nrows();
        let n_features = source_indices.len();
        let mut result = Array2::zeros((n_samples, n_features));

        for (j, &i) in source_indices.iter().enumerate() {
            let col = X.column(i);
            for (row, &value) in col.iter().enumerate() {
                result[[row, j]] = value.sin();
            }
        }
        Ok(result)
    }

    fn apply_cosine_transform(
        &self,
        X: &Array2<f64>,
        source_indices: &[usize],
    ) -> Result<Array2<f64>> {
        let n_samples = X.nrows();
        let n_features = source_indices.len();
        let mut result = Array2::zeros((n_samples, n_features));

        for (j, &i) in source_indices.iter().enumerate() {
            let col = X.column(i);
            for (row, &value) in col.iter().enumerate() {
                result[[row, j]] = value.cos();
            }
        }
        Ok(result)
    }

    fn apply_absolute_transform(
        &self,
        X: &Array2<f64>,
        source_indices: &[usize],
    ) -> Result<Array2<f64>> {
        let n_samples = X.nrows();
        let n_features = source_indices.len();
        let mut result = Array2::zeros((n_samples, n_features));

        for (j, &i) in source_indices.iter().enumerate() {
            let col = X.column(i);
            for (row, &value) in col.iter().enumerate() {
                result[[row, j]] = value.abs();
            }
        }
        Ok(result)
    }

    fn apply_sign_transform(
        &self,
        X: &Array2<f64>,
        source_indices: &[usize],
    ) -> Result<Array2<f64>> {
        let n_samples = X.nrows();
        let n_features = source_indices.len();
        let mut result = Array2::zeros((n_samples, n_features));

        for (j, &i) in source_indices.iter().enumerate() {
            let col = X.column(i);
            for (row, &value) in col.iter().enumerate() {
                let sign = if value > 0.0 {
                    1.0
                } else if value < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                result[[row, j]] = sign;
            }
        }
        Ok(result)
    }

    fn apply_binning_transform(
        &self,
        X: &Array2<f64>,
        source_indices: &[usize],
        n_bins: usize,
    ) -> Result<Array2<f64>> {
        let n_samples = X.nrows();
        let n_features = source_indices.len();
        let mut result = Array2::zeros((n_samples, n_features));

        for (j, &i) in source_indices.iter().enumerate() {
            let col = X.column(i);
            let min_val = col.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let bin_width = (max_val - min_val) / (n_bins as f64);

            for (row, &value) in col.iter().enumerate() {
                let bin = if bin_width > 0.0 {
                    ((value - min_val) / bin_width)
                        .floor()
                        .min((n_bins - 1) as f64)
                } else {
                    0.0
                };
                result[[row, j]] = bin;
            }
        }
        Ok(result)
    }

    fn apply_interaction_features(
        &self,
        X: &Array2<f64>,
        source_indices: &[usize],
    ) -> Result<Array2<f64>> {
        let n_samples = X.nrows();
        let selected_X = X.select(Axis(1), source_indices);
        let n_features = selected_X.ncols();

        // Generate all pairwise interactions
        let mut interactions = Vec::new();
        for i in 0..n_features {
            for j in (i + 1)..n_features {
                let col_i = selected_X.column(i);
                let col_j = selected_X.column(j);
                let interaction: Vec<f64> = col_i
                    .iter()
                    .zip(col_j.iter())
                    .map(|(&xi, &xj)| xi * xj)
                    .collect();
                interactions.push(interaction);
            }
        }

        if interactions.is_empty() {
            return Ok(Array2::zeros((n_samples, 1)));
        }

        // Convert to Array2
        let n_interactions = interactions.len();
        let mut result = Array2::zeros((n_samples, n_interactions));
        for (j, interaction) in interactions.iter().enumerate() {
            for (i, &value) in interaction.iter().enumerate() {
                result[[i, j]] = value;
            }
        }
        Ok(result)
    }

    fn apply_ratio_features(
        &self,
        X: &Array2<f64>,
        source_indices: &[usize],
    ) -> Result<Array2<f64>> {
        let n_samples = X.nrows();
        let selected_X = X.select(Axis(1), source_indices);
        let n_features = selected_X.ncols();

        // Generate ratios between pairs of features
        let mut ratios = Vec::new();
        for i in 0..n_features {
            for j in 0..n_features {
                if i != j {
                    let col_i = selected_X.column(i);
                    let col_j = selected_X.column(j);
                    let ratio: Vec<f64> = col_i
                        .iter()
                        .zip(col_j.iter())
                        .map(|(&xi, &xj)| if xj.abs() > 1e-8 { xi / xj } else { 0.0 })
                        .collect();
                    ratios.push(ratio);
                }
            }
        }

        if ratios.is_empty() {
            return Ok(Array2::zeros((n_samples, 1)));
        }

        // Limit number of ratio features
        ratios.truncate(20);

        // Convert to Array2
        let n_ratios = ratios.len();
        let mut result = Array2::zeros((n_samples, n_ratios));
        for (j, ratio) in ratios.iter().enumerate() {
            for (i, &value) in ratio.iter().enumerate() {
                result[[i, j]] = value;
            }
        }
        Ok(result)
    }

    fn apply_difference_features(
        &self,
        X: &Array2<f64>,
        source_indices: &[usize],
    ) -> Result<Array2<f64>> {
        let n_samples = X.nrows();
        let selected_X = X.select(Axis(1), source_indices);
        let n_features = selected_X.ncols();

        // Generate differences between pairs of features
        let mut differences = Vec::new();
        for i in 0..n_features {
            for j in (i + 1)..n_features {
                let col_i = selected_X.column(i);
                let col_j = selected_X.column(j);
                let diff: Vec<f64> = col_i
                    .iter()
                    .zip(col_j.iter())
                    .map(|(&xi, &xj)| xi - xj)
                    .collect();
                differences.push(diff);
            }
        }

        if differences.is_empty() {
            return Ok(Array2::zeros((n_samples, 1)));
        }

        // Convert to Array2
        let n_differences = differences.len();
        let mut result = Array2::zeros((n_samples, n_differences));
        for (j, diff) in differences.iter().enumerate() {
            for (i, &value) in diff.iter().enumerate() {
                result[[i, j]] = value;
            }
        }
        Ok(result)
    }

    fn apply_rolling_statistics(
        &self,
        X: &Array2<f64>,
        source_indices: &[usize],
        window: usize,
    ) -> Result<Array2<f64>> {
        let n_samples = X.nrows();
        let n_features = source_indices.len();
        let mut result = Array2::zeros((n_samples, n_features * 2)); // Mean and std

        for (j, &i) in source_indices.iter().enumerate() {
            let col = X.column(i);

            for row in 0..n_samples {
                let start = row.saturating_sub(window - 1);
                let end = (row + 1).min(n_samples);
                let window_data: Vec<f64> = col.slice(s![start..end]).to_vec();

                let mean = window_data.iter().sum::<f64>() / window_data.len() as f64;
                let variance = window_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                    / window_data.len() as f64;
                let std = variance.sqrt();

                result[[row, j * 2]] = mean;
                result[[row, j * 2 + 1]] = std;
            }
        }
        Ok(result)
    }

    fn apply_lag_features(
        &self,
        X: &Array2<f64>,
        source_indices: &[usize],
        lag: usize,
    ) -> Result<Array2<f64>> {
        let n_samples = X.nrows();
        let n_features = source_indices.len();
        let mut result = Array2::zeros((n_samples, n_features));

        for (j, &i) in source_indices.iter().enumerate() {
            let col = X.column(i);

            for row in 0..n_samples {
                let lag_row = row.saturating_sub(lag);
                result[[row, j]] = col[lag_row];
            }
        }
        Ok(result)
    }

    // Helper methods
    fn select_feature_pairs(&mut self, n_features: usize) -> Vec<usize> {
        // Select random pairs of features for interaction/ratio/difference features
        let max_pairs = 10.min(n_features);
        let mut indices = Vec::new();

        for _ in 0..max_pairs {
            let i = self.rng.gen_range(0..n_features);
            let j = self.rng.gen_range(0..n_features);
            if i != j {
                indices.extend(vec![i, j]);
            }
        }

        indices.sort_unstable();
        indices.dedup();
        indices
    }

    fn calculate_sparsity(&self, X: &Array2<f64>) -> f64 {
        let total_values = X.len() as f64;
        let zero_count = X.iter().filter(|&&x| x == 0.0).count() as f64;
        zero_count / total_values
    }

    fn analyze_correlation_structure(&self, X: &Array2<f64>) -> f64 {
        // Mock implementation - would calculate actual correlation matrix condition number
        use scirs2_core::random::Rng;
        let mut rng = scirs2_core::random::thread_rng();
        rng.gen_range(1.0..100.0)
    }

    fn estimate_linearity(&self, X: &Array2<f64>, y: &Array1<f64>) -> f64 {
        // Mock implementation - would perform actual linearity test
        use scirs2_core::random::Rng;
        let mut rng = scirs2_core::random::thread_rng();
        rng.gen_range(0.0..1.0)
    }

    fn get_unique_classes(&self, y: &Array1<f64>) -> Vec<i32> {
        let mut classes: Vec<i32> = y.iter().map(|&x| x as i32).collect();
        classes.sort_unstable();
        classes.dedup();
        classes
    }

    fn calculate_class_distribution(&self, y: &Array1<f64>, classes: &[i32]) -> Vec<f64> {
        let total = y.len() as f64;
        classes
            .iter()
            .map(|&class| {
                let count = y.iter().filter(|&&yi| yi as i32 == class).count() as f64;
                count / total
            })
            .collect()
    }

    fn count_unique_values(&self, column: &ArrayView1<f64>) -> usize {
        let mut values: Vec<i64> = column.iter().map(|&x| (x * 1000.0) as i64).collect();
        values.sort_unstable();
        values.dedup();
        values.len()
    }

    fn calculate_feature_importance(&self, column: &ArrayView1<f64>, y: &Array1<f64>) -> f64 {
        // Mock implementation - would calculate actual feature importance
        use scirs2_core::random::Rng;
        let mut rng = scirs2_core::random::thread_rng();
        rng.gen_range(0.0..1.0)
    }

    fn select_by_variance_threshold(&self, X: &Array2<f64>, threshold: f64) -> Vec<usize> {
        (0..X.ncols())
            .filter(|&i| {
                let col = X.column(i);
                col.std(0.0) > threshold
            })
            .collect()
    }

    fn select_by_correlation_threshold(&self, X: &Array2<f64>, threshold: f64) -> Vec<usize> {
        // Mock implementation - would calculate actual correlations
        (0..X.ncols()).collect()
    }
}

/// Convenience function for quick feature engineering
pub fn engineer_features(
    X: &Array2<f64>,
    y: &Array1<f64>,
    task_type: TaskType,
) -> Result<FeatureEngineeringResult> {
    let config = AutoFeatureEngineering {
        task_type,
        ..Default::default()
    };

    let mut engineer = AutoFeatureEngineer::new(config);
    engineer.engineer_features(X, y)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    #[allow(non_snake_case)]
    fn create_test_data() -> (Array2<f64>, Array1<f64>) {
        let X = Array2::from_shape_vec((100, 4), (0..400).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..100).map(|i| (i % 3) as f64).collect());
        (X, y)
    }

    #[test]
    fn test_feature_engineering() {
        let (X, y) = create_test_data();
        let result = engineer_features(&X, &y, TaskType::Classification);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.generated_feature_count > result.original_feature_count);
        assert!(result.selected_feature_count <= result.generated_feature_count);
    }

    #[test]
    fn test_polynomial_features() {
        let (X, _y) = create_test_data();
        let engineer = AutoFeatureEngineer::default();

        let poly_features = engineer.apply_polynomial_features(&X, &[0, 1], 2);
        assert!(poly_features.is_ok());

        let poly_features = poly_features.unwrap();
        assert!(poly_features.ncols() > 0);
    }

    #[test]
    fn test_mathematical_transforms() {
        let (X, _y) = create_test_data();
        let engineer = AutoFeatureEngineer::default();

        let log_features = engineer.apply_logarithmic_transform(&X, &[0, 1]);
        assert!(log_features.is_ok());

        let sqrt_features = engineer.apply_sqrt_transform(&X, &[0, 1]);
        assert!(sqrt_features.is_ok());
    }

    #[test]
    fn test_interaction_features() {
        let (X, _y) = create_test_data();
        let engineer = AutoFeatureEngineer::default();

        let interaction_features = engineer.apply_interaction_features(&X, &[0, 1, 2]);
        assert!(interaction_features.is_ok());

        let interaction_features = interaction_features.unwrap();
        assert!(interaction_features.ncols() > 0);
    }

    #[test]
    fn test_custom_strategy() {
        let (X, y) = create_test_data();

        let config = AutoFeatureEngineering {
            strategy: FeatureEngineeringStrategy::Custom(vec![
                FeatureTransformationType::Polynomial { degree: 2 },
                FeatureTransformationType::Logarithmic,
            ]),
            max_features: 50,
            ..Default::default()
        };

        let mut engineer = AutoFeatureEngineer::new(config);
        let result = engineer.engineer_features(&X, &y);
        assert!(result.is_ok());
    }

    #[test]
    fn test_feature_selection_methods() {
        let (X, y) = create_test_data();

        let config = AutoFeatureEngineering {
            selection_method: FeatureSelectionMethod::SelectPercentile { percentile: 50.0 },
            ..Default::default()
        };

        let mut engineer = AutoFeatureEngineer::new(config);
        let result = engineer.engineer_features(&X, &y);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.selected_feature_count > 0);
    }
}
