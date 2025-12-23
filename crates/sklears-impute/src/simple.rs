//! Simple imputation methods
//!
//! This module provides basic imputation strategies including mean, median,
//! mode, constant, and time series imputation methods.

use crate::core::{ImputationError, ImputationResult, Imputer};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::random::Random;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};
use std::collections::HashMap;

/// Simple Imputer
///
/// Imputation transformer for completing missing values using simple strategies.
/// The imputer replaces missing values using the mean, median, most frequent value,
/// a constant value, or time series imputation along each column.
///
/// # Parameters
///
/// * `missing_values` - The placeholder for missing values (NaN by default)
/// * `strategy` - Imputation strategy ('mean', 'median', 'most_frequent', 'constant', 'forward_fill', 'backward_fill', 'random_sampling')
/// * `fill_value` - Fill value to use when strategy is 'constant'
/// * `copy` - Whether to make a copy of the input data
///
/// # Examples
///
/// ```
/// use sklears_impute::SimpleImputer;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [f64::NAN, 3.0], [7.0, 6.0]];
///
/// let imputer = SimpleImputer::new()
///     .strategy("mean".to_string());
/// let fitted = imputer.fit(&X.view(), &()).unwrap();
/// let X_imputed = fitted.transform(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SimpleImputer<S = Untrained> {
    state: S,
    missing_values: f64,
    strategy: String,
    fill_value: Option<f64>,
    copy: bool,
}

/// Trained state for SimpleImputer
#[derive(Debug, Clone)]
pub struct SimpleImputerTrained {
    statistics: Array1<f64>,
    valid_values: Vec<Vec<f64>>,
}

impl SimpleImputer<Untrained> {
    /// Create a new SimpleImputer instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            missing_values: f64::NAN,
            strategy: "mean".to_string(),
            fill_value: None,
            copy: true,
        }
    }

    /// Set the missing values placeholder
    pub fn missing_values(mut self, missing_values: f64) -> Self {
        self.missing_values = missing_values;
        self
    }

    /// Set the imputation strategy
    pub fn strategy(mut self, strategy: String) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the fill value for constant strategy
    pub fn fill_value(mut self, fill_value: Option<f64>) -> Self {
        self.fill_value = fill_value;
        self
    }

    /// Set whether to copy the input data
    pub fn copy(mut self, copy: bool) -> Self {
        self.copy = copy;
        self
    }

    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

impl Default for SimpleImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for SimpleImputer<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for SimpleImputer<Untrained> {
    type Fitted = SimpleImputer<SimpleImputerTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.mapv(|x| x);
        let (_, n_features) = X.dim();
        let mut statistics = Vec::new();
        let mut all_valid_values = Vec::new();

        for feature_idx in 0..n_features {
            let column = X.column(feature_idx);
            let valid_values: Vec<f64> = column
                .iter()
                .filter(|&&x| !self.is_missing(x))
                .cloned()
                .collect();

            if valid_values.is_empty() {
                return Err(SklearsError::InvalidInput(format!(
                    "All values are missing in feature {feature_idx}"
                )));
            }

            let statistic = match self.strategy.as_str() {
                "mean" => {
                    let sum: f64 = valid_values.iter().sum();
                    sum / valid_values.len() as f64
                }
                "median" => {
                    let mut sorted_values = valid_values.clone();
                    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let len = sorted_values.len();
                    if len % 2 == 0 {
                        (sorted_values[len / 2 - 1] + sorted_values[len / 2]) / 2.0
                    } else {
                        sorted_values[len / 2]
                    }
                }
                "most_frequent" => {
                    let mut counts = HashMap::new();
                    for &value in &valid_values {
                        *counts.entry(value.to_bits()).or_insert(0) += 1;
                    }
                    let most_frequent_bits = counts
                        .into_iter()
                        .max_by_key(|&(_, count)| count)
                        .unwrap()
                        .0;
                    f64::from_bits(most_frequent_bits)
                }
                "constant" => self.fill_value.unwrap_or(0.0),
                "forward_fill" | "backward_fill" => {
                    // For time series strategies, we'll store the mean as fallback
                    // The actual forward/backward fill will be done in transform
                    let sum: f64 = valid_values.iter().sum();
                    sum / valid_values.len() as f64
                }
                "random_sampling" => {
                    // For random sampling, we'll store the mean as fallback
                    // The actual random sampling will be done in transform
                    let sum: f64 = valid_values.iter().sum();
                    sum / valid_values.len() as f64
                }
                _ => {
                    return Err(SklearsError::InvalidInput(format!(
                        "Unknown strategy: {}",
                        self.strategy
                    )));
                }
            };

            statistics.push(statistic);
            all_valid_values.push(valid_values.clone());
        }

        Ok(SimpleImputer {
            state: SimpleImputerTrained {
                statistics: Array1::from(statistics),
                valid_values: all_valid_values,
            },
            missing_values: self.missing_values,
            strategy: self.strategy,
            fill_value: self.fill_value,
            copy: self.copy,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>> for SimpleImputer<SimpleImputerTrained> {
    #[allow(non_snake_case)]
    fn transform(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let X = X.mapv(|x| x);
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.statistics.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features {} does not match training features {}",
                n_features,
                self.state.statistics.len()
            )));
        }

        let mut X_imputed = if self.copy { X.clone() } else { X };

        match self.strategy.as_str() {
            "forward_fill" => {
                for feature_idx in 0..n_features {
                    let mut last_valid = None;
                    for sample_idx in 0..n_samples {
                        let value = X_imputed[[sample_idx, feature_idx]];
                        if self.is_missing(value) {
                            if let Some(fill_value) = last_valid {
                                X_imputed[[sample_idx, feature_idx]] = fill_value;
                            } else {
                                // No previous valid value, use mean as fallback
                                X_imputed[[sample_idx, feature_idx]] =
                                    self.state.statistics[feature_idx];
                            }
                        } else {
                            last_valid = Some(value);
                        }
                    }
                }
            }
            "backward_fill" => {
                for feature_idx in 0..n_features {
                    let mut next_valid = None;
                    for sample_idx in (0..n_samples).rev() {
                        let value = X_imputed[[sample_idx, feature_idx]];
                        if self.is_missing(value) {
                            if let Some(fill_value) = next_valid {
                                X_imputed[[sample_idx, feature_idx]] = fill_value;
                            } else {
                                // No next valid value, use mean as fallback
                                X_imputed[[sample_idx, feature_idx]] =
                                    self.state.statistics[feature_idx];
                            }
                        } else {
                            next_valid = Some(value);
                        }
                    }
                }
            }
            "random_sampling" => {
                let mut rng = Random::default();
                for feature_idx in 0..n_features {
                    let valid_values = &self.state.valid_values[feature_idx];
                    if !valid_values.is_empty() {
                        for sample_idx in 0..n_samples {
                            if self.is_missing(X_imputed[[sample_idx, feature_idx]]) {
                                let random_idx = rng.gen_range(0..valid_values.len());
                                let random_value = &valid_values[random_idx];
                                X_imputed[[sample_idx, feature_idx]] = *random_value;
                            }
                        }
                    }
                }
            }
            _ => {
                // Standard strategies: mean, median, most_frequent, constant
                for feature_idx in 0..n_features {
                    let fill_value = self.state.statistics[feature_idx];
                    for sample_idx in 0..n_samples {
                        if self.is_missing(X_imputed[[sample_idx, feature_idx]]) {
                            X_imputed[[sample_idx, feature_idx]] = fill_value;
                        }
                    }
                }
            }
        }

        Ok(X_imputed.mapv(|x| x as Float))
    }
}

impl SimpleImputer<SimpleImputerTrained> {
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

/// Missing Indicator
///
/// Binary indicator for missing values.
///
/// # Parameters
///
/// * `missing_values` - The placeholder for missing values (NaN by default)
/// * `features` - Which features to generate indicators for ('missing-only' or 'all')
/// * `sparse` - Whether to return sparse indicators
/// * `error_on_new` - Whether to raise an error when a new feature is completely missing during transform
///
/// # Examples
///
/// ```
/// use sklears_impute::MissingIndicator;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [f64::NAN, 3.0], [7.0, 6.0]];
///
/// let indicator = MissingIndicator::new();
/// let fitted = indicator.fit(&X.view(), &()).unwrap();
/// let indicators = fitted.transform(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MissingIndicator<S = Untrained> {
    state: S,
    missing_values: f64,
    features: String,
    sparse: bool,
    error_on_new: bool,
}

/// Trained state for MissingIndicator
#[derive(Debug, Clone)]
pub struct MissingIndicatorTrained {
    features_: Vec<usize>,
    n_features_in_: usize,
}

impl MissingIndicator<Untrained> {
    /// Create a new MissingIndicator instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            missing_values: f64::NAN,
            features: "missing-only".to_string(),
            sparse: false,
            error_on_new: true,
        }
    }

    /// Set the missing values placeholder
    pub fn missing_values(mut self, missing_values: f64) -> Self {
        self.missing_values = missing_values;
        self
    }

    /// Set which features to generate indicators for
    pub fn features(mut self, features: String) -> Self {
        self.features = features;
        self
    }

    /// Set whether to return sparse indicators
    pub fn sparse(mut self, sparse: bool) -> Self {
        self.sparse = sparse;
        self
    }

    /// Set whether to raise an error on new missing features
    pub fn error_on_new(mut self, error_on_new: bool) -> Self {
        self.error_on_new = error_on_new;
        self
    }

    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

impl Default for MissingIndicator<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MissingIndicator<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for MissingIndicator<Untrained> {
    type Fitted = MissingIndicator<MissingIndicatorTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.mapv(|x| x);
        let (_, n_features) = X.dim();

        let features_ = match self.features.as_str() {
            "missing-only" => {
                // Only include features that have missing values
                let mut selected_features = Vec::new();
                for feature_idx in 0..n_features {
                    let column = X.column(feature_idx);
                    if column.iter().any(|&x| self.is_missing(x)) {
                        selected_features.push(feature_idx);
                    }
                }
                selected_features
            }
            "all" => (0..n_features).collect(),
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown features option: {}",
                    self.features
                )));
            }
        };

        Ok(MissingIndicator {
            state: MissingIndicatorTrained {
                features_,
                n_features_in_: n_features,
            },
            missing_values: self.missing_values,
            features: self.features,
            sparse: self.sparse,
            error_on_new: self.error_on_new,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>> for MissingIndicator<MissingIndicatorTrained> {
    #[allow(non_snake_case)]
    fn transform(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let X = X.mapv(|x| x);
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features_in_ {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features {} does not match training features {}",
                n_features, self.state.n_features_in_
            )));
        }

        if self.error_on_new {
            // Check for new missing features
            for feature_idx in 0..n_features {
                if !self.state.features_.contains(&feature_idx) {
                    let column = X.column(feature_idx);
                    if column.iter().any(|&x| self.is_missing(x)) {
                        return Err(SklearsError::InvalidInput(format!(
                            "Feature {} has missing values but was not seen during fit",
                            feature_idx
                        )));
                    }
                }
            }
        }

        let n_indicator_features = self.state.features_.len();
        let mut indicators = Array2::<f64>::zeros((n_samples, n_indicator_features));

        for (indicator_idx, &feature_idx) in self.state.features_.iter().enumerate() {
            let column = X.column(feature_idx);
            for (sample_idx, &value) in column.iter().enumerate() {
                if self.is_missing(value) {
                    indicators[[sample_idx, indicator_idx]] = 1.0;
                }
            }
        }

        Ok(indicators.mapv(|x| x as Float))
    }
}

impl MissingIndicator<MissingIndicatorTrained> {
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

// Implement the Imputer trait for SimpleImputer
impl Imputer for SimpleImputer<Untrained> {
    #[allow(non_snake_case)]
    fn fit_transform(
        &self,
        X: &scirs2_core::ndarray::ArrayView2<f64>,
    ) -> ImputationResult<scirs2_core::ndarray::Array2<f64>> {
        // Convert from f64 array to Float array for sklears-core compatibility
        let X_float = X.mapv(|x| x as Float);
        let X_view = X_float.view();

        // Use the sklears-core fit and transform pattern
        let fitted = self.clone().fit(&X_view, &()).map_err(|e| {
            ImputationError::ProcessingError(format!("Failed to fit imputer: {}", e))
        })?;

        let result = fitted.transform(&X_view).map_err(|e| {
            ImputationError::ProcessingError(format!("Failed to transform data: {}", e))
        })?;

        // Convert back to f64 for the imputation interface
        Ok(result.mapv(|x| x))
    }
}
