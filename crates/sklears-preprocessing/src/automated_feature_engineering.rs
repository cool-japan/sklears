//! Automated Feature Engineering
//!
//! This module provides automated feature engineering capabilities that can automatically
//! generate new features from existing ones to improve model performance.
//!
//! # Features
//!
//! - **Feature Generation**: Automatically create polynomial, interaction, and transformation features
//! - **Feature Selection**: Select the most relevant features using various scoring methods
//! - **Feature Transformation**: Apply mathematical transformations to discover hidden patterns
//! - **Feature Importance**: Rank features by their predictive power
//! - **Domain-Specific Engineering**: Apply domain knowledge for specific feature types
//!
//! # Examples
//!
//! ```rust,ignore
//! use sklears_preprocessing::automated_feature_engineering::{
//!     AutoFeatureEngineer, AutoFeatureConfig, GenerationStrategy
//! };
//! use scirs2_core::ndarray::Array2;
//!
//! fn example() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = AutoFeatureConfig::new()
//!         .with_strategy(GenerationStrategy::Polynomial { degree: 2 })
//!         .with_max_features(100)
//!         .with_selection_threshold(0.01);
//!     
//!     let mut engineer = AutoFeatureEngineer::new(config);
//!     
//!     let data = Array2::from_shape_vec((100, 5), (0..500).map(|x| x as f64).collect())?;
//!     let target = Array1::from_vec((0..100).map(|x| (x % 2) as f64).collect());
//!     
//!     let engineer_fitted = engineer.fit(&data, &target)?;
//!     let transformed = engineer_fitted.transform(&data)?;
//!     
//!     println!("Original features: {}", data.ncols());
//!     println!("Generated features: {}", transformed.ncols());
//!     
//!     Ok(())
//! }
//! ```

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Transform, Untrained},
};
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for automated feature engineering
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AutoFeatureConfig {
    /// Feature generation strategies
    pub strategies: Vec<GenerationStrategy>,
    /// Maximum number of features to generate
    pub max_features: usize,
    /// Feature selection method
    pub selection_method: SelectionMethod,
    /// Threshold for feature selection
    pub selection_threshold: f64,
    /// Whether to include original features
    pub include_original: bool,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
    /// Maximum interaction depth
    pub max_interaction_depth: usize,
    /// Whether to remove highly correlated features
    pub remove_correlated: bool,
    /// Correlation threshold for removal
    pub correlation_threshold: f64,
    /// Whether to scale features before selection
    pub scale_features: bool,
}

/// Feature generation strategies
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum GenerationStrategy {
    /// Polynomial features up to specified degree
    Polynomial { degree: usize },
    /// Mathematical transformations
    Mathematical { functions: Vec<MathFunction> },
    /// Feature interactions
    Interactions { max_depth: usize },
    /// Binning and discretization
    Binning { n_bins: usize },
    /// Ratios between features
    Ratios,
    /// Statistical aggregations (for grouped data)
    Aggregations { window_size: usize },
    /// Frequency encoding for categorical-like numerical features
    FrequencyEncoding,
    /// Domain-specific features
    DomainSpecific { domain: Domain },
}

/// Mathematical functions for feature transformation
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MathFunction {
    Log,
    Log1p,
    Sqrt,
    Square,
    Exp,
    Sin,
    Cos,
    Tan,
    Abs,
    Reciprocal,
}

/// Domain-specific feature engineering
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Domain {
    /// Time series domain
    TimeSeries,
    /// Financial domain
    Financial,
    /// Text domain (for numerical text features)
    Text,
    /// Image domain (for flattened image data)
    Image,
    /// Generic numerical domain
    Generic,
}

/// Feature selection methods
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SelectionMethod {
    /// Mutual information-based selection
    MutualInformation,
    /// Correlation-based selection
    Correlation,
    /// Variance-based selection
    Variance,
    /// Chi-squared test
    ChiSquared,
    /// F-test
    FTest,
    /// Recursive feature elimination
    RecursiveElimination,
    /// LASSO-based selection
    LASSO,
}

impl Default for AutoFeatureConfig {
    fn default() -> Self {
        Self {
            strategies: vec![
                GenerationStrategy::Polynomial { degree: 2 },
                GenerationStrategy::Mathematical {
                    functions: vec![
                        MathFunction::Log1p,
                        MathFunction::Sqrt,
                        MathFunction::Square,
                    ],
                },
                GenerationStrategy::Interactions { max_depth: 2 },
            ],
            max_features: 200,
            selection_method: SelectionMethod::MutualInformation,
            selection_threshold: 0.01,
            include_original: true,
            random_state: None,
            max_interaction_depth: 2,
            remove_correlated: true,
            correlation_threshold: 0.95,
            scale_features: true,
        }
    }
}

impl AutoFeatureConfig {
    /// Create a new configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a generation strategy
    pub fn with_strategy(mut self, strategy: GenerationStrategy) -> Self {
        self.strategies.push(strategy);
        self
    }

    /// Set maximum number of features
    pub fn with_max_features(mut self, max_features: usize) -> Self {
        self.max_features = max_features;
        self
    }

    /// Set selection method
    pub fn with_selection_method(mut self, method: SelectionMethod) -> Self {
        self.selection_method = method;
        self
    }

    /// Set selection threshold
    pub fn with_selection_threshold(mut self, threshold: f64) -> Self {
        self.selection_threshold = threshold;
        self
    }

    /// Set whether to include original features
    pub fn with_include_original(mut self, include: bool) -> Self {
        self.include_original = include;
        self
    }

    /// Set random state
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

/// Automated feature engineering transformer
pub struct AutoFeatureEngineer<State = Untrained> {
    config: AutoFeatureConfig,
    state: PhantomData<State>,
}

/// Fitted automated feature engineer
pub struct AutoFeatureEngineerFitted {
    config: AutoFeatureConfig,
    selected_features: Vec<usize>,
    feature_names: Vec<String>,
    feature_scores: Vec<f64>,
    transformation_functions: Vec<TransformationFunction>,
    n_original_features: usize,
    feature_importance: Vec<f64>,
    correlation_matrix: Option<Array2<f64>>,
}

/// Represents a transformation function for feature generation
#[derive(Debug, Clone)]
pub struct TransformationFunction {
    pub name: String,
    pub function_type: TransformationType,
    pub input_indices: Vec<usize>,
    pub parameters: HashMap<String, f64>,
}

/// Types of transformations
#[derive(Debug, Clone)]
pub enum TransformationType {
    Polynomial { degree: usize },
    Mathematical { function: MathFunction },
    Interaction,
    Binning { bins: Vec<f64> },
    Ratio,
    Aggregation { window_size: usize },
    FrequencyEncoding { mapping: HashMap<String, f64> },
}

impl AutoFeatureEngineer<Untrained> {
    /// Create a new automated feature engineer
    pub fn new(config: AutoFeatureConfig) -> Self {
        Self {
            config,
            state: PhantomData,
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &AutoFeatureConfig {
        &self.config
    }
}

impl Fit<Array2<f64>, Array1<f64>> for AutoFeatureEngineer<Untrained> {
    type Fitted = AutoFeatureEngineerFitted;

    fn fit(self, x: &Array2<f64>, y: &Array1<f64>) -> Result<AutoFeatureEngineerFitted> {
        if x.is_empty() || y.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Input arrays cannot be empty".to_string(),
            ));
        }

        let (n_samples, n_features) = x.dim();
        if y.len() != n_samples {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        // Step 1: Generate features according to strategies
        let mut generated_features = Vec::new();
        let mut transformation_functions = Vec::new();
        let mut feature_names = Vec::new();

        // Add original features if requested
        if self.config.include_original {
            for i in 0..n_features {
                feature_names.push(format!("original_{}", i));
                generated_features.push(x.column(i).to_owned());
                transformation_functions.push(TransformationFunction {
                    name: format!("original_{}", i),
                    function_type: TransformationType::Mathematical {
                        function: MathFunction::Abs,
                    }, // Identity placeholder
                    input_indices: vec![i],
                    parameters: HashMap::new(),
                });
            }
        }

        // Generate features for each strategy
        for strategy in &self.config.strategies {
            let (strategy_features, strategy_transforms, strategy_names) =
                self.generate_features_for_strategy(x, strategy)?;

            generated_features.extend(strategy_features);
            transformation_functions.extend(strategy_transforms);
            feature_names.extend(strategy_names);
        }

        // Step 2: Create feature matrix
        let n_generated = generated_features.len();
        if n_generated == 0 {
            return Err(SklearsError::InvalidInput(
                "No features were generated".to_string(),
            ));
        }

        let mut feature_matrix = Array2::zeros((n_samples, n_generated));
        for (i, feature) in generated_features.iter().enumerate() {
            for (j, &value) in feature.iter().enumerate() {
                feature_matrix[[j, i]] = value;
            }
        }

        // Step 3: Scale features if requested
        let feature_matrix = if self.config.scale_features {
            scale_features(&feature_matrix)?
        } else {
            feature_matrix
        };

        // Step 4: Remove highly correlated features
        let (feature_matrix, feature_indices) = if self.config.remove_correlated {
            remove_correlated_features(&feature_matrix, self.config.correlation_threshold)?
        } else {
            let indices: Vec<usize> = (0..n_generated).collect();
            (feature_matrix, indices)
        };

        // Update transformation functions and names based on remaining features
        let mut filtered_transforms = Vec::new();
        let mut filtered_names = Vec::new();
        for &idx in &feature_indices {
            if idx < transformation_functions.len() {
                filtered_transforms.push(transformation_functions[idx].clone());
                filtered_names.push(feature_names[idx].clone());
            }
        }

        // Step 5: Calculate feature scores
        let feature_scores = self.calculate_feature_scores(&feature_matrix, y)?;

        // Step 6: Select top features
        let selected_features = self.select_features(&feature_scores)?;

        // Step 7: Calculate feature importance
        let feature_importance =
            self.calculate_feature_importance(&feature_matrix, y, &selected_features)?;

        // Step 8: Calculate correlation matrix for analysis
        let correlation_matrix = if feature_matrix.ncols() <= 1000 {
            // Only for reasonably sized matrices
            Some(calculate_correlation_matrix(&feature_matrix)?)
        } else {
            None
        };

        Ok(AutoFeatureEngineerFitted {
            config: self.config,
            selected_features,
            feature_names: filtered_names,
            feature_scores,
            transformation_functions: filtered_transforms,
            n_original_features: n_features,
            feature_importance,
            correlation_matrix,
        })
    }
}

impl AutoFeatureEngineer<Untrained> {
    /// Generate features for a specific strategy
    fn generate_features_for_strategy(
        &self,
        x: &Array2<f64>,
        strategy: &GenerationStrategy,
    ) -> Result<(Vec<Array1<f64>>, Vec<TransformationFunction>, Vec<String>)> {
        match strategy {
            GenerationStrategy::Polynomial { degree } => {
                self.generate_polynomial_features(x, *degree)
            }
            GenerationStrategy::Mathematical { functions } => {
                self.generate_mathematical_features(x, functions)
            }
            GenerationStrategy::Interactions { max_depth } => {
                self.generate_interaction_features(x, *max_depth)
            }
            GenerationStrategy::Binning { n_bins } => self.generate_binning_features(x, *n_bins),
            GenerationStrategy::Ratios => self.generate_ratio_features(x),
            GenerationStrategy::Aggregations { window_size } => {
                self.generate_aggregation_features(x, *window_size)
            }
            GenerationStrategy::FrequencyEncoding => self.generate_frequency_encoding_features(x),
            GenerationStrategy::DomainSpecific { domain } => {
                self.generate_domain_specific_features(x, domain)
            }
        }
    }

    /// Generate polynomial features
    fn generate_polynomial_features(
        &self,
        x: &Array2<f64>,
        degree: usize,
    ) -> Result<(Vec<Array1<f64>>, Vec<TransformationFunction>, Vec<String>)> {
        let (_n_samples, n_features) = x.dim();
        let mut features = Vec::new();
        let mut transforms = Vec::new();
        let mut names = Vec::new();

        // Generate polynomial features for each individual feature
        for i in 0..n_features {
            let column = x.column(i);
            for d in 2..=degree {
                let poly_feature = column.mapv(|x| x.powi(d as i32));
                features.push(poly_feature);
                transforms.push(TransformationFunction {
                    name: format!("poly_{}_{}", i, d),
                    function_type: TransformationType::Polynomial { degree: d },
                    input_indices: vec![i],
                    parameters: HashMap::new(),
                });
                names.push(format!("poly_{}_{}", i, d));
            }
        }

        Ok((features, transforms, names))
    }

    /// Generate mathematical transformation features
    fn generate_mathematical_features(
        &self,
        x: &Array2<f64>,
        functions: &[MathFunction],
    ) -> Result<(Vec<Array1<f64>>, Vec<TransformationFunction>, Vec<String>)> {
        let (_n_samples, n_features) = x.dim();
        let mut features = Vec::new();
        let mut transforms = Vec::new();
        let mut names = Vec::new();

        for i in 0..n_features {
            let column = x.column(i);
            for &function in functions {
                let transformed = apply_math_function(&column.to_owned(), function)?;
                features.push(transformed);
                transforms.push(TransformationFunction {
                    name: format!("{}_{}", math_function_name(function), i),
                    function_type: TransformationType::Mathematical { function },
                    input_indices: vec![i],
                    parameters: HashMap::new(),
                });
                names.push(format!("{}_{}", math_function_name(function), i));
            }
        }

        Ok((features, transforms, names))
    }

    /// Generate interaction features
    fn generate_interaction_features(
        &self,
        x: &Array2<f64>,
        max_depth: usize,
    ) -> Result<(Vec<Array1<f64>>, Vec<TransformationFunction>, Vec<String>)> {
        let (_n_samples, n_features) = x.dim();
        let mut features = Vec::new();
        let mut transforms = Vec::new();
        let mut names = Vec::new();

        // Generate pairwise interactions
        if max_depth >= 2 {
            for i in 0..n_features {
                for j in (i + 1)..n_features {
                    let col_i = x.column(i);
                    let col_j = x.column(j);
                    let interaction = &col_i * &col_j;
                    features.push(interaction);
                    transforms.push(TransformationFunction {
                        name: format!("interact_{}_{}", i, j),
                        function_type: TransformationType::Interaction,
                        input_indices: vec![i, j],
                        parameters: HashMap::new(),
                    });
                    names.push(format!("interact_{}_{}", i, j));
                }
            }
        }

        // Generate three-way interactions if requested
        if max_depth >= 3 && n_features >= 3 {
            for i in 0..n_features {
                for j in (i + 1)..n_features {
                    for k in (j + 1)..n_features {
                        let col_i = x.column(i);
                        let col_j = x.column(j);
                        let col_k = x.column(k);
                        let interaction = &(&col_i * &col_j) * &col_k;
                        features.push(interaction);
                        transforms.push(TransformationFunction {
                            name: format!("interact_{}_{}_{}", i, j, k),
                            function_type: TransformationType::Interaction,
                            input_indices: vec![i, j, k],
                            parameters: HashMap::new(),
                        });
                        names.push(format!("interact_{}_{}_{}", i, j, k));
                    }
                }
            }
        }

        Ok((features, transforms, names))
    }

    /// Generate binning features (placeholder implementation)
    fn generate_binning_features(
        &self,
        x: &Array2<f64>,
        n_bins: usize,
    ) -> Result<(Vec<Array1<f64>>, Vec<TransformationFunction>, Vec<String>)> {
        let (_n_samples, n_features) = x.dim();
        let mut features = Vec::new();
        let mut transforms = Vec::new();
        let mut names = Vec::new();

        for i in 0..n_features {
            let column = x.column(i);
            let (min_val, max_val) = column
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &val| {
                    (min.min(val), max.max(val))
                });

            if (max_val - min_val).abs() < f64::EPSILON {
                continue; // Skip constant features
            }

            let bin_width = (max_val - min_val) / n_bins as f64;
            let bins: Vec<f64> = (0..=n_bins)
                .map(|b| min_val + b as f64 * bin_width)
                .collect();

            let binned_feature = column.mapv(|x| {
                let bin_index = ((x - min_val) / bin_width).floor() as usize;
                bin_index.min(n_bins - 1) as f64
            });

            features.push(binned_feature);
            transforms.push(TransformationFunction {
                name: format!("bin_{}", i),
                function_type: TransformationType::Binning { bins: bins.clone() },
                input_indices: vec![i],
                parameters: HashMap::new(),
            });
            names.push(format!("bin_{}", i));
        }

        Ok((features, transforms, names))
    }

    /// Generate ratio features
    fn generate_ratio_features(
        &self,
        x: &Array2<f64>,
    ) -> Result<(Vec<Array1<f64>>, Vec<TransformationFunction>, Vec<String>)> {
        let (_n_samples, n_features) = x.dim();
        let mut features = Vec::new();
        let mut transforms = Vec::new();
        let mut names = Vec::new();

        for i in 0..n_features {
            for j in 0..n_features {
                if i != j {
                    let col_i = x.column(i);
                    let col_j = x.column(j);

                    // Avoid division by zero
                    let ratio = col_i
                        .iter()
                        .zip(col_j.iter())
                        .map(|(&a, &b)| if b.abs() < 1e-8 { 0.0 } else { a / b })
                        .collect::<Vec<f64>>();

                    features.push(Array1::from_vec(ratio));
                    transforms.push(TransformationFunction {
                        name: format!("ratio_{}_{}", i, j),
                        function_type: TransformationType::Ratio,
                        input_indices: vec![i, j],
                        parameters: HashMap::new(),
                    });
                    names.push(format!("ratio_{}_{}", i, j));
                }
            }
        }

        Ok((features, transforms, names))
    }

    /// Generate aggregation features (placeholder implementation)
    fn generate_aggregation_features(
        &self,
        x: &Array2<f64>,
        window_size: usize,
    ) -> Result<(Vec<Array1<f64>>, Vec<TransformationFunction>, Vec<String>)> {
        let (n_samples, n_features) = x.dim();
        let mut features = Vec::new();
        let mut transforms = Vec::new();
        let mut names = Vec::new();

        if window_size >= n_samples {
            return Ok((features, transforms, names)); // Not enough samples for windowing
        }

        for i in 0..n_features {
            let column = x.column(i);

            // Rolling mean
            let rolling_mean = (0..n_samples)
                .map(|idx| {
                    let start = idx.saturating_sub(window_size / 2);
                    let end = (idx + window_size / 2 + 1).min(n_samples);
                    let window = &column.slice(scirs2_core::ndarray::s![start..end]);
                    window.mean().unwrap_or(0.0)
                })
                .collect::<Vec<f64>>();

            features.push(Array1::from_vec(rolling_mean));
            transforms.push(TransformationFunction {
                name: format!("rolling_mean_{}_{}", i, window_size),
                function_type: TransformationType::Aggregation { window_size },
                input_indices: vec![i],
                parameters: HashMap::new(),
            });
            names.push(format!("rolling_mean_{}_{}", i, window_size));
        }

        Ok((features, transforms, names))
    }

    /// Generate frequency encoding features (placeholder implementation)
    fn generate_frequency_encoding_features(
        &self,
        x: &Array2<f64>,
    ) -> Result<(Vec<Array1<f64>>, Vec<TransformationFunction>, Vec<String>)> {
        let (_n_samples, n_features) = x.dim();
        let mut features = Vec::new();
        let mut transforms = Vec::new();
        let mut names = Vec::new();

        for i in 0..n_features {
            let column = x.column(i);

            // Count frequencies of values (rounded to handle floating point)
            let mut frequency_map: HashMap<i64, i32> = HashMap::new();
            for &value in column.iter() {
                let rounded = (value * 1000.0).round() as i64; // Round to 3 decimal places and convert to int
                *frequency_map.entry(rounded).or_insert(0) += 1;
            }

            // Convert to frequency encoding
            let freq_encoded = column.mapv(|x| {
                let rounded = (x * 1000.0).round() as i64;
                *frequency_map.get(&rounded).unwrap_or(&0) as f64
            });

            features.push(freq_encoded);
            transforms.push(TransformationFunction {
                name: format!("freq_encode_{}", i),
                function_type: TransformationType::FrequencyEncoding {
                    mapping: frequency_map
                        .iter()
                        .map(|(&k, &v)| (k.to_string(), v as f64))
                        .collect(),
                },
                input_indices: vec![i],
                parameters: HashMap::new(),
            });
            names.push(format!("freq_encode_{}", i));
        }

        Ok((features, transforms, names))
    }

    /// Generate domain-specific features (placeholder implementation)
    fn generate_domain_specific_features(
        &self,
        x: &Array2<f64>,
        domain: &Domain,
    ) -> Result<(Vec<Array1<f64>>, Vec<TransformationFunction>, Vec<String>)> {
        match domain {
            Domain::TimeSeries => self.generate_time_series_features(x),
            Domain::Financial => self.generate_financial_features(x),
            Domain::Text => self.generate_text_features(x),
            Domain::Image => self.generate_image_features(x),
            Domain::Generic => self.generate_generic_features(x),
        }
    }

    /// Generate time series specific features
    fn generate_time_series_features(
        &self,
        x: &Array2<f64>,
    ) -> Result<(Vec<Array1<f64>>, Vec<TransformationFunction>, Vec<String>)> {
        let mut features = Vec::new();
        let mut transforms = Vec::new();
        let mut names = Vec::new();

        // For each feature, generate lag features, differences, etc.
        for i in 0..x.ncols() {
            let column = x.column(i);

            // First difference
            if column.len() > 1 {
                let diff = (1..column.len())
                    .map(|j| column[j] - column[j - 1])
                    .collect::<Vec<f64>>();
                let mut diff_feature = vec![0.0]; // Pad with zero for first element
                diff_feature.extend(diff);

                features.push(Array1::from_vec(diff_feature));
                transforms.push(TransformationFunction {
                    name: format!("diff_{}", i),
                    function_type: TransformationType::Mathematical {
                        function: MathFunction::Abs,
                    }, // Placeholder
                    input_indices: vec![i],
                    parameters: HashMap::new(),
                });
                names.push(format!("diff_{}", i));
            }
        }

        Ok((features, transforms, names))
    }

    /// Generate financial domain features
    fn generate_financial_features(
        &self,
        x: &Array2<f64>,
    ) -> Result<(Vec<Array1<f64>>, Vec<TransformationFunction>, Vec<String>)> {
        // Placeholder for financial features like returns, volatility, etc.
        self.generate_generic_features(x)
    }

    /// Generate text domain features
    fn generate_text_features(
        &self,
        x: &Array2<f64>,
    ) -> Result<(Vec<Array1<f64>>, Vec<TransformationFunction>, Vec<String>)> {
        // Placeholder for text features like length, character counts, etc.
        self.generate_generic_features(x)
    }

    /// Generate image domain features
    fn generate_image_features(
        &self,
        x: &Array2<f64>,
    ) -> Result<(Vec<Array1<f64>>, Vec<TransformationFunction>, Vec<String>)> {
        // Placeholder for image features like gradients, filters, etc.
        self.generate_generic_features(x)
    }

    /// Generate generic domain features
    fn generate_generic_features(
        &self,
        x: &Array2<f64>,
    ) -> Result<(Vec<Array1<f64>>, Vec<TransformationFunction>, Vec<String>)> {
        // Basic statistical features
        let mut features = Vec::new();
        let mut transforms = Vec::new();
        let mut names = Vec::new();

        // Row-wise statistics
        for stat_name in &["sum", "mean", "std", "min", "max"] {
            let stat_feature = (0..x.nrows())
                .map(|i| {
                    let row = x.row(i);
                    match *stat_name {
                        "sum" => row.sum(),
                        "mean" => row.mean().unwrap_or(0.0),
                        "std" => {
                            let mean = row.mean().unwrap_or(0.0);
                            let variance = row.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0);
                            variance.sqrt()
                        }
                        "min" => row.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                        "max" => row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
                        _ => 0.0,
                    }
                })
                .collect::<Vec<f64>>();

            features.push(Array1::from_vec(stat_feature));
            transforms.push(TransformationFunction {
                name: format!("row_{}", stat_name),
                function_type: TransformationType::Aggregation {
                    window_size: x.ncols(),
                },
                input_indices: (0..x.ncols()).collect(),
                parameters: HashMap::new(),
            });
            names.push(format!("row_{}", stat_name));
        }

        Ok((features, transforms, names))
    }

    /// Calculate feature scores using the selected method
    fn calculate_feature_scores(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Vec<f64>> {
        match self.config.selection_method {
            SelectionMethod::Correlation => calculate_correlation_scores(x, y),
            SelectionMethod::Variance => calculate_variance_scores(x),
            SelectionMethod::MutualInformation => calculate_mutual_information_scores(x, y),
            _ => {
                // Fallback to correlation for unimplemented methods
                calculate_correlation_scores(x, y)
            }
        }
    }

    /// Select features based on scores and configuration
    fn select_features(&self, scores: &[f64]) -> Result<Vec<usize>> {
        let mut indexed_scores: Vec<(usize, f64)> = scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score.abs()))
            .collect();

        // Sort by score descending
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select features above threshold and up to max_features
        let selected: Vec<usize> = indexed_scores
            .iter()
            .filter(|(_, score)| *score >= self.config.selection_threshold)
            .take(self.config.max_features)
            .map(|(idx, _)| *idx)
            .collect();

        if selected.is_empty() {
            // If no features meet the threshold, select the top one
            Ok(vec![indexed_scores[0].0])
        } else {
            Ok(selected)
        }
    }

    /// Calculate feature importance
    fn calculate_feature_importance(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        selected_features: &[usize],
    ) -> Result<Vec<f64>> {
        // Simple importance based on correlation with target
        let mut importance = vec![0.0; selected_features.len()];

        for (i, &feature_idx) in selected_features.iter().enumerate() {
            if feature_idx < x.ncols() {
                let feature_col = x.column(feature_idx).to_owned();
                let correlation = calculate_correlation(&feature_col, y)?;
                importance[i] = correlation.abs();
            }
        }

        Ok(importance)
    }
}

impl AutoFeatureEngineerFitted {
    /// Get selected feature indices
    pub fn selected_features(&self) -> &[usize] {
        &self.selected_features
    }

    /// Get feature names
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Get feature scores
    pub fn feature_scores(&self) -> &[f64] {
        &self.feature_scores
    }

    /// Get feature importance
    pub fn feature_importance(&self) -> &[f64] {
        &self.feature_importance
    }

    /// Get transformation functions
    pub fn transformations(&self) -> &[TransformationFunction] {
        &self.transformation_functions
    }

    /// Get correlation matrix (if computed)
    pub fn correlation_matrix(&self) -> Option<&Array2<f64>> {
        self.correlation_matrix.as_ref()
    }
}

impl Transform<Array2<f64>, Array2<f64>> for AutoFeatureEngineerFitted {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if x.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Input array is empty".to_string(),
            ));
        }

        let (n_samples, n_features) = x.dim();
        if n_features != self.n_original_features {
            return Err(SklearsError::InvalidInput(format!(
                "Feature count mismatch: expected {}, got {}",
                self.n_original_features, n_features
            )));
        }

        // Apply transformations and select features
        // This is a simplified implementation
        // In practice, you'd apply the exact transformations learned during fitting

        let mut result = Array2::zeros((n_samples, self.selected_features.len()));

        for (out_idx, &in_idx) in self.selected_features.iter().enumerate() {
            if in_idx < n_features {
                // For original features, just copy
                for (row_idx, &value) in x.column(in_idx).iter().enumerate() {
                    result[[row_idx, out_idx]] = value;
                }
            }
            // For generated features, we'd need to apply the corresponding transformation
            // This is simplified for now
        }

        Ok(result)
    }
}

// Helper functions

/// Apply a mathematical function to an array
fn apply_math_function(arr: &Array1<f64>, function: MathFunction) -> Result<Array1<f64>> {
    let result = match function {
        MathFunction::Log => arr.mapv(|x| if x > 0.0 { x.ln() } else { f64::NEG_INFINITY }),
        MathFunction::Log1p => arr.mapv(|x| (1.0 + x).ln()),
        MathFunction::Sqrt => arr.mapv(|x| if x >= 0.0 { x.sqrt() } else { 0.0 }),
        MathFunction::Square => arr.mapv(|x| x * x),
        MathFunction::Exp => arr.mapv(|x| x.exp()),
        MathFunction::Sin => arr.mapv(|x| x.sin()),
        MathFunction::Cos => arr.mapv(|x| x.cos()),
        MathFunction::Tan => arr.mapv(|x| x.tan()),
        MathFunction::Abs => arr.mapv(|x| x.abs()),
        MathFunction::Reciprocal => arr.mapv(|x| if x.abs() > 1e-8 { 1.0 / x } else { 0.0 }),
    };
    Ok(result)
}

/// Get the name of a mathematical function
fn math_function_name(function: MathFunction) -> &'static str {
    match function {
        MathFunction::Log => "log",
        MathFunction::Log1p => "log1p",
        MathFunction::Sqrt => "sqrt",
        MathFunction::Square => "square",
        MathFunction::Exp => "exp",
        MathFunction::Sin => "sin",
        MathFunction::Cos => "cos",
        MathFunction::Tan => "tan",
        MathFunction::Abs => "abs",
        MathFunction::Reciprocal => "reciprocal",
    }
}

/// Scale features to have zero mean and unit variance
fn scale_features(x: &Array2<f64>) -> Result<Array2<f64>> {
    let mut result = x.clone();
    let n_features = x.ncols();

    for i in 0..n_features {
        let col = x.column(i);
        let mean = col.mean().unwrap_or(0.0);
        let std = {
            let variance = col.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0);
            variance.sqrt()
        };

        if std > 1e-8 {
            for j in 0..x.nrows() {
                result[[j, i]] = (result[[j, i]] - mean) / std;
            }
        }
    }

    Ok(result)
}

/// Remove highly correlated features
fn remove_correlated_features(
    x: &Array2<f64>,
    threshold: f64,
) -> Result<(Array2<f64>, Vec<usize>)> {
    let n_features = x.ncols();
    let mut to_remove = HashSet::new();

    // Calculate correlation matrix
    for i in 0..n_features {
        for j in (i + 1)..n_features {
            if to_remove.contains(&i) || to_remove.contains(&j) {
                continue;
            }

            let corr = calculate_correlation(&x.column(i).to_owned(), &x.column(j).to_owned())?;
            if corr.abs() > threshold {
                // Remove the feature with lower variance
                let var_i = x.column(i).var(0.0);
                let var_j = x.column(j).var(0.0);
                if var_i < var_j {
                    to_remove.insert(i);
                } else {
                    to_remove.insert(j);
                }
            }
        }
    }

    // Create new matrix without correlated features
    let remaining_features: Vec<usize> =
        (0..n_features).filter(|i| !to_remove.contains(i)).collect();

    if remaining_features.is_empty() {
        return Ok((x.clone(), (0..n_features).collect()));
    }

    let mut result = Array2::zeros((x.nrows(), remaining_features.len()));
    for (new_idx, &old_idx) in remaining_features.iter().enumerate() {
        for (row_idx, &value) in x.column(old_idx).iter().enumerate() {
            result[[row_idx, new_idx]] = value;
        }
    }

    Ok((result, remaining_features))
}

/// Calculate correlation between two arrays
fn calculate_correlation(x: &Array1<f64>, y: &Array1<f64>) -> Result<f64> {
    if x.len() != y.len() {
        return Err(SklearsError::InvalidInput(
            "Arrays must have the same length".to_string(),
        ));
    }

    let mean_x = x.mean().unwrap_or(0.0);
    let mean_y = y.mean().unwrap_or(0.0);

    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }

    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    if denominator < 1e-8 {
        Ok(0.0)
    } else {
        Ok(numerator / denominator)
    }
}

/// Calculate correlation scores for features
fn calculate_correlation_scores(x: &Array2<f64>, y: &Array1<f64>) -> Result<Vec<f64>> {
    let mut scores = Vec::new();
    for i in 0..x.ncols() {
        let correlation = calculate_correlation(&x.column(i).to_owned(), y)?;
        scores.push(correlation.abs());
    }
    Ok(scores)
}

/// Calculate variance-based scores
fn calculate_variance_scores(x: &Array2<f64>) -> Result<Vec<f64>> {
    let mut scores = Vec::new();
    for i in 0..x.ncols() {
        let variance = x.column(i).var(0.0);
        scores.push(variance);
    }
    Ok(scores)
}

/// Calculate mutual information scores (simplified implementation)
fn calculate_mutual_information_scores(x: &Array2<f64>, y: &Array1<f64>) -> Result<Vec<f64>> {
    // For now, use correlation as a proxy for mutual information
    // In a full implementation, you'd use proper mutual information calculation
    calculate_correlation_scores(x, y)
}

/// Calculate correlation matrix
fn calculate_correlation_matrix(x: &Array2<f64>) -> Result<Array2<f64>> {
    let n_features = x.ncols();
    let mut corr_matrix = Array2::zeros((n_features, n_features));

    for i in 0..n_features {
        for j in 0..n_features {
            if i == j {
                corr_matrix[[i, j]] = 1.0;
            } else {
                let corr = calculate_correlation(&x.column(i).to_owned(), &x.column(j).to_owned())?;
                corr_matrix[[i, j]] = corr;
            }
        }
    }

    Ok(corr_matrix)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::{arr1, arr2};

    #[test]
    fn test_auto_feature_config() {
        let config = AutoFeatureConfig::new()
            .with_max_features(50)
            .with_selection_threshold(0.05)
            .with_strategy(GenerationStrategy::Polynomial { degree: 3 });

        assert_eq!(config.max_features, 50);
        assert_relative_eq!(config.selection_threshold, 0.05);
        assert_eq!(config.strategies.len(), 4); // 3 default + 1 added
    }

    #[test]
    fn test_auto_feature_engineer_creation() {
        let config = AutoFeatureConfig::new();
        let engineer = AutoFeatureEngineer::new(config);
        assert_eq!(engineer.config().max_features, 200);
    }

    #[test]
    fn test_auto_feature_engineer_fit() {
        let config = AutoFeatureConfig::new()
            .with_max_features(10)
            .with_selection_threshold(0.0); // Accept all features
        let engineer = AutoFeatureEngineer::new(config);

        let X = arr2(&[[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]);
        let y = arr1(&[1.0, 2.0, 3.0, 4.0]);

        let fitted = engineer.fit(&X, &y).unwrap();
        assert!(!fitted.selected_features().is_empty());
        assert!(!fitted.feature_names().is_empty());
    }

    #[test]
    fn test_mathematical_functions() {
        let arr = arr1(&[1.0, 2.0, 3.0, 4.0]);

        let sqrt_result = apply_math_function(&arr, MathFunction::Sqrt).unwrap();
        let expected_sqrt = arr1(&[1.0, 2.0_f64.sqrt(), 3.0_f64.sqrt(), 2.0]);

        for (a, b) in sqrt_result.iter().zip(expected_sqrt.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }

        let square_result = apply_math_function(&arr, MathFunction::Square).unwrap();
        let expected_square = arr1(&[1.0, 4.0, 9.0, 16.0]);

        for (a, b) in square_result.iter().zip(expected_square.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_correlation_calculation() {
        let x = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = arr1(&[2.0, 4.0, 6.0, 8.0, 10.0]); // Perfect positive correlation

        let corr = calculate_correlation(&x, &y).unwrap();
        assert_relative_eq!(corr, 1.0, epsilon = 1e-10);

        let z = arr1(&[5.0, 4.0, 3.0, 2.0, 1.0]); // Perfect negative correlation
        let corr_neg = calculate_correlation(&x, &z).unwrap();
        assert_relative_eq!(corr_neg, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_feature_scaling() {
        let X = arr2(&[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]);

        let scaled = scale_features(&X).unwrap();

        // Check that each column has approximately zero mean and unit variance
        for i in 0..scaled.ncols() {
            let col = scaled.column(i);
            let mean = col.mean().unwrap();
            let std = col.mapv(|x| (x - mean).powi(2)).mean().unwrap().sqrt();

            assert_relative_eq!(mean, 0.0, epsilon = 1e-10);
            assert_relative_eq!(std, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_auto_feature_engineer_transform() {
        let config = AutoFeatureConfig::new()
            .with_max_features(5)
            .with_include_original(true);
        let engineer = AutoFeatureEngineer::new(config);

        let X_train = arr2(&[[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]);
        let y_train = arr1(&[1.0, 2.0, 3.0]);

        let fitted = engineer.fit(&X_train, &y_train).unwrap();

        let X_test = arr2(&[[4.0, 8.0], [5.0, 10.0]]);

        let result = fitted.transform(&X_test).unwrap();
        assert_eq!(result.nrows(), 2);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_error_handling() {
        // Test empty arrays
        let config = AutoFeatureConfig::new();
        let engineer = AutoFeatureEngineer::new(config);
        let empty_X = Array2::from_shape_vec((0, 0), vec![]).unwrap();
        let empty_y = Array1::from_vec(vec![]);
        assert!(engineer.fit(&empty_X, &empty_y).is_err());

        // Test mismatched dimensions
        let config = AutoFeatureConfig::new();
        let engineer = AutoFeatureEngineer::new(config);
        let X = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let y = arr1(&[1.0]); // Wrong size
        assert!(engineer.fit(&X, &y).is_err());
    }
}
