//! Preprocessing Integration Module for AutoML Feature Selection
//!
//! Provides automated data preprocessing including scaling, missing value handling, and feature engineering.
//! All implementations follow the SciRS2 policy using scirs2-core for numerical computations.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use super::automl_core::DataCharacteristics;
use sklears_core::error::Result as SklResult;

type Result<T> = SklResult<T>;

/// Preprocessing integration for automated data preparation
#[derive(Debug, Clone)]
pub struct PreprocessingIntegration {
    scaler_type: ScalerType,
    missing_value_strategy: MissingValueStrategy,
    outlier_handling: OutlierHandling,
    feature_engineering: FeatureEngineering,
    dimensionality_reduction: Option<DimensionalityReduction>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ScalerType {
    /// StandardScaler
    StandardScaler,
    /// MinMaxScaler
    MinMaxScaler,
    /// RobustScaler
    RobustScaler,
    /// QuantileUniform
    QuantileUniform,
    /// QuantileNormal
    QuantileNormal,

    None,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MissingValueStrategy {
    /// Mean
    Mean,
    /// Median
    Median,
    /// Mode
    Mode,
    /// Forward
    Forward,
    /// Backward
    Backward,
    /// Interpolation
    Interpolation,
    /// Remove
    Remove,
    /// KNN
    KNN { k: usize },
}

#[derive(Debug, Clone, PartialEq)]
pub enum OutlierHandling {
    /// IQR
    IQR {
        multiplier: f64,
    },
    /// ZScore
    ZScore {
        threshold: f64,
    },
    /// Isolation
    Isolation,
    /// LocalOutlierFactor
    LocalOutlierFactor {
        k: usize,
    },

    None,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FeatureEngineering {
    /// Polynomial
    Polynomial {
        degree: usize,
    },
    /// Interaction
    Interaction,
    /// TargetEncoding
    TargetEncoding,
    /// FrequencyEncoding
    FrequencyEncoding,
    /// BinDiscretization
    BinDiscretization {
        bins: usize,
    },

    None,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DimensionalityReduction {
    /// PCA
    PCA { n_components: usize },
    /// ICA
    ICA { n_components: usize },
    /// TruncatedSVD
    TruncatedSVD { n_components: usize },
    /// FactorAnalysis
    FactorAnalysis { n_components: usize },
}

impl PreprocessingIntegration {
    pub fn new() -> Self {
        Self {
            scaler_type: ScalerType::StandardScaler,
            missing_value_strategy: MissingValueStrategy::Mean,
            outlier_handling: OutlierHandling::None,
            feature_engineering: FeatureEngineering::None,
            dimensionality_reduction: None,
        }
    }

    pub fn with_scaler(mut self, scaler_type: ScalerType) -> Self {
        self.scaler_type = scaler_type;
        self
    }

    pub fn with_missing_value_strategy(mut self, strategy: MissingValueStrategy) -> Self {
        self.missing_value_strategy = strategy;
        self
    }

    pub fn with_outlier_handling(mut self, handling: OutlierHandling) -> Self {
        self.outlier_handling = handling;
        self
    }

    pub fn with_feature_engineering(mut self, engineering: FeatureEngineering) -> Self {
        self.feature_engineering = engineering;
        self
    }

    pub fn with_dimensionality_reduction(mut self, reduction: DimensionalityReduction) -> Self {
        self.dimensionality_reduction = Some(reduction);
        self
    }

    /// Apply preprocessing to the data
    pub fn preprocess_data(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        let mut processed_X = X.to_owned();
        let processed_y = y.to_owned();

        // Step 1: Handle missing values
        processed_X = self.handle_missing_values(processed_X)?;

        // Step 2: Handle outliers
        processed_X = self.handle_outliers(processed_X)?;

        // Step 3: Scale features
        processed_X = self.scale_features(processed_X)?;

        // Step 4: Feature engineering
        processed_X = self.apply_feature_engineering(processed_X)?;

        // Step 5: Dimensionality reduction (if specified)
        if let Some(ref reduction) = self.dimensionality_reduction {
            processed_X = self.apply_dimensionality_reduction(processed_X, reduction)?;
        }

        Ok((processed_X, processed_y))
    }

    /// Auto-configure preprocessing based on data characteristics
    pub fn auto_configure(characteristics: &DataCharacteristics) -> Self {
        let mut config = Self::new();

        // Choose scaler based on data properties
        config.scaler_type = if characteristics
            .feature_variance_distribution
            .iter()
            .any(|&v| v > 1000.0)
        {
            ScalerType::RobustScaler
        } else {
            ScalerType::StandardScaler
        };

        // Choose missing value strategy
        config.missing_value_strategy = if characteristics.has_missing_values {
            if characteristics.n_samples > 1000 {
                MissingValueStrategy::KNN { k: 5 }
            } else {
                MissingValueStrategy::Mean
            }
        } else {
            MissingValueStrategy::Mean // No missing values, strategy doesn't matter
        };

        // Configure outlier handling for high-dimensional data
        config.outlier_handling = if characteristics.n_features > 100 {
            OutlierHandling::IQR { multiplier: 1.5 }
        } else {
            OutlierHandling::None
        };

        // Feature engineering for small datasets
        config.feature_engineering =
            if characteristics.n_features < 50 && characteristics.n_samples > 200 {
                FeatureEngineering::Polynomial { degree: 2 }
            } else {
                FeatureEngineering::None
            };

        // Dimensionality reduction for high-dimensional data
        config.dimensionality_reduction = if characteristics.feature_to_sample_ratio > 2.0 {
            Some(DimensionalityReduction::PCA {
                n_components: (characteristics.n_samples / 2).min(100),
            })
        } else {
            None
        };

        config
    }

    fn handle_missing_values(&self, mut X: Array2<f64>) -> Result<Array2<f64>> {
        match &self.missing_value_strategy {
            MissingValueStrategy::Mean => {
                for col in 0..X.ncols() {
                    let mut column = X.column_mut(col);
                    let valid_values: Vec<f64> =
                        column.iter().filter(|&&x| !x.is_nan()).cloned().collect();
                    if !valid_values.is_empty() {
                        let mean = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
                        for val in column.iter_mut() {
                            if val.is_nan() {
                                *val = mean;
                            }
                        }
                    }
                }
            }
            MissingValueStrategy::Median => {
                for col in 0..X.ncols() {
                    let mut column = X.column_mut(col);
                    let mut valid_values: Vec<f64> =
                        column.iter().filter(|&&x| !x.is_nan()).cloned().collect();
                    if !valid_values.is_empty() {
                        valid_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let median = if valid_values.len() % 2 == 0 {
                            (valid_values[valid_values.len() / 2 - 1]
                                + valid_values[valid_values.len() / 2])
                                / 2.0
                        } else {
                            valid_values[valid_values.len() / 2]
                        };
                        for val in column.iter_mut() {
                            if val.is_nan() {
                                *val = median;
                            }
                        }
                    }
                }
            }
            // Simplified implementations for other strategies
            _ => {
                // For other strategies, use mean as fallback
                return self.handle_missing_values_fallback(X);
            }
        }
        Ok(X)
    }

    fn handle_missing_values_fallback(&self, mut X: Array2<f64>) -> Result<Array2<f64>> {
        for col in 0..X.ncols() {
            let mut column = X.column_mut(col);
            let valid_values: Vec<f64> = column.iter().filter(|&&x| !x.is_nan()).cloned().collect();
            if !valid_values.is_empty() {
                let mean = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
                for val in column.iter_mut() {
                    if val.is_nan() {
                        *val = mean;
                    }
                }
            }
        }
        Ok(X)
    }

    fn handle_outliers(&self, mut X: Array2<f64>) -> Result<Array2<f64>> {
        match &self.outlier_handling {
            OutlierHandling::IQR { multiplier } => {
                for col in 0..X.ncols() {
                    let column = X.column(col);
                    let mut values: Vec<f64> = column.to_vec();
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    let q1_idx = values.len() / 4;
                    let q3_idx = 3 * values.len() / 4;
                    let q1 = values[q1_idx];
                    let q3 = values[q3_idx];
                    let iqr = q3 - q1;

                    let lower_bound = q1 - multiplier * iqr;
                    let upper_bound = q3 + multiplier * iqr;

                    // Cap outliers
                    for val in X.column_mut(col).iter_mut() {
                        if *val < lower_bound {
                            *val = lower_bound;
                        } else if *val > upper_bound {
                            *val = upper_bound;
                        }
                    }
                }
            }
            OutlierHandling::ZScore { threshold } => {
                for col in 0..X.ncols() {
                    let column = X.column(col);
                    let mean = column.mean().unwrap_or(0.0);
                    let std = column.std(1.0);

                    for val in X.column_mut(col).iter_mut() {
                        let z_score = (*val - mean) / std;
                        if z_score.abs() > *threshold {
                            *val = mean; // Replace outliers with mean
                        }
                    }
                }
            }
            _ => {
                // No outlier handling
            }
        }
        Ok(X)
    }

    fn scale_features(&self, mut X: Array2<f64>) -> Result<Array2<f64>> {
        match &self.scaler_type {
            ScalerType::StandardScaler => {
                for col in 0..X.ncols() {
                    let column = X.column(col);
                    let mean = column.mean().unwrap_or(0.0);
                    let std = column.std(1.0);

                    if std > 1e-10 {
                        for val in X.column_mut(col).iter_mut() {
                            *val = (*val - mean) / std;
                        }
                    }
                }
            }
            ScalerType::MinMaxScaler => {
                for col in 0..X.ncols() {
                    let column = X.column(col);
                    let min_val = column.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    let max_val = column.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let range = max_val - min_val;

                    if range > 1e-10 {
                        for val in X.column_mut(col).iter_mut() {
                            *val = (*val - min_val) / range;
                        }
                    }
                }
            }
            ScalerType::RobustScaler => {
                for col in 0..X.ncols() {
                    let mut values: Vec<f64> = X.column(col).to_vec();
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    let median = if values.len() % 2 == 0 {
                        (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
                    } else {
                        values[values.len() / 2]
                    };

                    let q1 = values[values.len() / 4];
                    let q3 = values[3 * values.len() / 4];
                    let iqr = q3 - q1;

                    if iqr > 1e-10 {
                        for val in X.column_mut(col).iter_mut() {
                            *val = (*val - median) / iqr;
                        }
                    }
                }
            }
            _ => {
                // No scaling
            }
        }
        Ok(X)
    }

    fn apply_feature_engineering(&self, X: Array2<f64>) -> Result<Array2<f64>> {
        match &self.feature_engineering {
            FeatureEngineering::Polynomial { degree: 2 } => {
                // Simple quadratic features (x^2 for each feature)
                let mut new_X = Array2::zeros((X.nrows(), X.ncols() * 2));

                // Original features
                for i in 0..X.nrows() {
                    for j in 0..X.ncols() {
                        new_X[[i, j]] = X[[i, j]];
                    }
                }

                // Squared features
                for i in 0..X.nrows() {
                    for j in 0..X.ncols() {
                        new_X[[i, X.ncols() + j]] = X[[i, j]] * X[[i, j]];
                    }
                }

                Ok(new_X)
            }
            _ => Ok(X),
        }
    }

    fn apply_dimensionality_reduction(
        &self,
        X: Array2<f64>,
        reduction: &DimensionalityReduction,
    ) -> Result<Array2<f64>> {
        match reduction {
            DimensionalityReduction::PCA { n_components } => {
                // Simplified PCA implementation - just select first n_components features
                let n_comp = (*n_components).min(X.ncols());
                let mut reduced_X = Array2::zeros((X.nrows(), n_comp));

                for i in 0..X.nrows() {
                    for j in 0..n_comp {
                        reduced_X[[i, j]] = X[[i, j]];
                    }
                }

                Ok(reduced_X)
            }
            _ => Ok(X), // Other reduction methods not implemented
        }
    }
}

impl Default for PreprocessingIntegration {
    fn default() -> Self {
        Self::new()
    }
}
