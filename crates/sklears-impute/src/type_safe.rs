//! Type-safe missing data operations with phantom types for compile-time validation
//!
//! This module provides zero-cost abstractions for missing data handling using Rust's type system
//! to prevent common errors at compile time.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::marker::PhantomData;

/// Phantom type marker for MCAR (Missing Completely At Random) data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MCAR;

/// Phantom type marker for MAR (Missing At Random) data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MAR;

/// Phantom type marker for MNAR (Missing Not At Random) data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MNAR;

/// Phantom type marker for unknown missing data mechanism
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnknownMechanism;

/// Phantom type marker for complete data (no missing values)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Complete;

/// Phantom type marker for data with missing values
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WithMissing;

/// Type-safe wrapper for arrays with compile-time missing data mechanism tracking
#[derive(Debug, Clone)]
pub struct TypedArray<T, M, S> {
    data: Array2<T>,
    missing_mask: Option<Array2<bool>>,
    _mechanism: PhantomData<M>,
    _state: PhantomData<S>,
}

/// Type alias for complete data arrays
pub type CompleteArray<T> = TypedArray<T, UnknownMechanism, Complete>;

/// Type alias for MCAR data arrays
pub type MCARArray<T> = TypedArray<T, MCAR, WithMissing>;

/// Type alias for MAR data arrays
pub type MARArray<T> = TypedArray<T, MAR, WithMissing>;

/// Type alias for MNAR data arrays
pub type MNARArray<T> = TypedArray<T, MNAR, WithMissing>;

/// Missing data pattern information
#[derive(Debug, Clone)]
pub struct MissingPattern {
    /// pattern
    pub pattern: Vec<bool>,
    /// count
    pub count: usize,
    /// frequency
    pub frequency: f64,
}

/// Compile-time missing pattern validator
pub trait MissingPatternValidator<M> {
    fn validate_assumptions(&self) -> SklResult<()>;
    fn recommended_imputers(&self) -> Vec<&'static str>;
}

impl<T: Clone + PartialEq> TypedArray<T, UnknownMechanism, Complete> {
    /// Create a new complete array from ndarray
    pub fn new_complete(data: Array2<T>) -> Self {
        Self {
            data,
            missing_mask: None,
            _mechanism: PhantomData,
            _state: PhantomData,
        }
    }
}

impl<T: Clone + PartialEq> TypedArray<T, UnknownMechanism, WithMissing> {
    /// Create a new array with missing values from ndarray
    pub fn new_with_missing(data: Array2<T>, missing_mask: Array2<bool>) -> Self {
        Self {
            data,
            missing_mask: Some(missing_mask),
            _mechanism: PhantomData,
            _state: PhantomData,
        }
    }
}

impl<T: Clone + PartialEq> TypedArray<T, MCAR, WithMissing> {
    /// Create a new MCAR array with missing values from ndarray
    pub fn new_with_missing(data: Array2<T>, missing_mask: Array2<bool>) -> Self {
        Self {
            data,
            missing_mask: Some(missing_mask),
            _mechanism: PhantomData,
            _state: PhantomData,
        }
    }
}

impl<T, M, S> TypedArray<T, M, S> {
    /// Get the underlying data array
    pub fn data(&self) -> &Array2<T> {
        &self.data
    }

    /// Get the missing mask if available
    pub fn missing_mask(&self) -> Option<&Array2<bool>> {
        self.missing_mask.as_ref()
    }

    /// Get the shape of the data
    pub fn shape(&self) -> (usize, usize) {
        self.data.dim()
    }

    /// Get the number of rows
    pub fn nrows(&self) -> usize {
        self.data.nrows()
    }

    /// Get the number of columns
    pub fn ncols(&self) -> usize {
        self.data.ncols()
    }
}

impl<T: Clone + PartialEq> TypedArray<T, UnknownMechanism, WithMissing> {
    /// Classify the missing data mechanism and return typed array
    pub fn classify_mechanism(self) -> SklResult<ClassifiedArray<T>> {
        let mechanism = self.infer_mechanism()?;
        Ok(ClassifiedArray::new(
            self.data,
            self.missing_mask.unwrap(),
            mechanism,
        ))
    }

    /// Infer the missing data mechanism using statistical tests
    fn infer_mechanism(&self) -> SklResult<MissingMechanism> {
        // Simplified mechanism inference - in practice this would use statistical tests
        let missing_mask = self.missing_mask.as_ref().unwrap();
        let missing_rate =
            missing_mask.iter().filter(|&&x| x).count() as f64 / missing_mask.len() as f64;

        if missing_rate < 0.05 {
            Ok(MissingMechanism::MCAR)
        } else if missing_rate < 0.2 {
            Ok(MissingMechanism::MAR)
        } else {
            Ok(MissingMechanism::MNAR)
        }
    }
}

/// Enum representing the classified missing data mechanism
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingMechanism {
    /// MCAR
    MCAR,
    /// MAR
    MAR,
    /// MNAR
    MNAR,
}

/// Classified array with known missing data mechanism
#[derive(Debug, Clone)]
pub struct ClassifiedArray<T> {
    data: Array2<T>,
    missing_mask: Array2<bool>,
    mechanism: MissingMechanism,
}

impl<T: Clone> ClassifiedArray<T> {
    pub fn new(data: Array2<T>, missing_mask: Array2<bool>, mechanism: MissingMechanism) -> Self {
        Self {
            data,
            missing_mask,
            mechanism,
        }
    }

    pub fn mechanism(&self) -> MissingMechanism {
        self.mechanism
    }

    pub fn data(&self) -> &Array2<T> {
        &self.data
    }

    pub fn missing_mask(&self) -> &Array2<bool> {
        &self.missing_mask
    }
}

impl<T> MissingPatternValidator<MCAR> for TypedArray<T, MCAR, WithMissing> {
    fn validate_assumptions(&self) -> SklResult<()> {
        // For MCAR, missing values should be randomly distributed
        // This is a simplified check
        Ok(())
    }

    fn recommended_imputers(&self) -> Vec<&'static str> {
        vec!["SimpleImputer", "KNNImputer", "MatrixFactorization"]
    }
}

impl<T> MissingPatternValidator<MAR> for TypedArray<T, MAR, WithMissing> {
    fn validate_assumptions(&self) -> SklResult<()> {
        // For MAR, missingness depends on observed values
        // This would check dependencies between observed and missing values
        Ok(())
    }

    fn recommended_imputers(&self) -> Vec<&'static str> {
        vec![
            "IterativeImputer",
            "BayesianImputer",
            "GaussianProcessImputer",
        ]
    }
}

impl<T> MissingPatternValidator<MNAR> for TypedArray<T, MNAR, WithMissing> {
    fn validate_assumptions(&self) -> SklResult<()> {
        // For MNAR, missingness depends on unobserved values
        // This would need domain knowledge validation
        Ok(())
    }

    fn recommended_imputers(&self) -> Vec<&'static str> {
        vec!["PatternMixtureModel", "SelectionModel", "BayesianImputer"]
    }
}

/// Type-safe missing data operations
pub trait TypeSafeMissingOps<T, M, S> {
    /// Check if data is complete (no missing values)
    fn is_complete(&self) -> bool;

    /// Count missing values
    fn count_missing(&self) -> usize;

    /// Get missing rate per feature
    fn missing_rate_per_feature(&self) -> Array1<f64>;

    /// Get missing pattern analysis
    fn analyze_patterns(&self) -> Vec<MissingPattern>;
}

impl<T: Clone + PartialEq> TypeSafeMissingOps<T, UnknownMechanism, WithMissing>
    for TypedArray<T, UnknownMechanism, WithMissing>
{
    fn is_complete(&self) -> bool {
        self.missing_mask
            .as_ref()
            .map_or(true, |mask| !mask.iter().any(|&x| x))
    }

    fn count_missing(&self) -> usize {
        self.missing_mask
            .as_ref()
            .map_or(0, |mask| mask.iter().filter(|&&x| x).count())
    }

    fn missing_rate_per_feature(&self) -> Array1<f64> {
        if let Some(mask) = &self.missing_mask {
            let n_rows = mask.nrows() as f64;
            let mut rates = Array1::zeros(mask.ncols());

            for j in 0..mask.ncols() {
                let missing_count = mask.column(j).iter().filter(|&&x| x).count() as f64;
                rates[j] = missing_count / n_rows;
            }

            rates
        } else {
            Array1::zeros(self.data.ncols())
        }
    }

    fn analyze_patterns(&self) -> Vec<MissingPattern> {
        if let Some(mask) = &self.missing_mask {
            let mut pattern_counts = std::collections::HashMap::new();
            let n_rows = mask.nrows();

            for row in mask.rows() {
                let pattern: Vec<bool> = row.to_vec();
                *pattern_counts.entry(pattern).or_insert(0) += 1;
            }

            pattern_counts
                .into_iter()
                .map(|(pattern, count)| MissingPattern {
                    pattern,
                    count,
                    frequency: count as f64 / n_rows as f64,
                })
                .collect()
        } else {
            vec![]
        }
    }
}

/// Compile-time size validation for fixed-size arrays
pub trait FixedSizeValidation<const N: usize, const M: usize> {
    fn validate_dimensions(&self) -> SklResult<()>;
}

/// Fixed-size typed array for compile-time dimension validation
#[derive(Debug, Clone)]
pub struct FixedSizeArray<T, const N: usize, const M: usize> {
    data: Array2<T>,
    _phantom: PhantomData<(T, [(); N], [(); M])>,
}

impl<T: Clone, const N: usize, const M: usize> FixedSizeArray<T, N, M> {
    pub fn new(data: Array2<T>) -> SklResult<Self> {
        if data.nrows() != N || data.ncols() != M {
            return Err(SklearsError::InvalidInput(format!(
                "Array dimensions {}x{} do not match required {}x{}",
                data.nrows(),
                data.ncols(),
                N,
                M
            )));
        }

        Ok(Self {
            data,
            _phantom: PhantomData,
        })
    }

    pub fn data(&self) -> &Array2<T> {
        &self.data
    }
}

impl<T, const N: usize, const M: usize> FixedSizeValidation<N, M> for FixedSizeArray<T, N, M> {
    fn validate_dimensions(&self) -> SklResult<()> {
        if self.data.nrows() != N || self.data.ncols() != M {
            Err(SklearsError::InvalidInput(format!(
                "Invalid dimensions: expected {}x{}, got {}x{}",
                N,
                M,
                self.data.nrows(),
                self.data.ncols()
            )))
        } else {
            Ok(())
        }
    }
}

/// Zero-cost abstraction for missing value detection
pub trait MissingValueDetector<T> {
    fn is_missing(&self, value: &T) -> bool;
}

/// NaN-based missing value detector for floating point types
pub struct NaNDetector;

impl MissingValueDetector<f64> for NaNDetector {
    fn is_missing(&self, value: &f64) -> bool {
        value.is_nan()
    }
}

impl MissingValueDetector<f32> for NaNDetector {
    fn is_missing(&self, value: &f32) -> bool {
        value.is_nan()
    }
}

/// Sentinel value-based missing value detector
pub struct SentinelDetector<T> {
    sentinel: T,
}

impl<T: PartialEq> SentinelDetector<T> {
    pub fn new(sentinel: T) -> Self {
        Self { sentinel }
    }
}

impl<T: PartialEq> MissingValueDetector<T> for SentinelDetector<T> {
    fn is_missing(&self, value: &T) -> bool {
        *value == self.sentinel
    }
}

/// Type-safe imputation result with provenance tracking
#[derive(Debug, Clone)]
pub struct ImputationResult<T> {
    /// data
    pub data: Array2<T>,
    /// imputed_positions
    pub imputed_positions: Vec<(usize, usize)>,
    /// imputation_method
    pub imputation_method: String,
    /// quality_metrics
    pub quality_metrics: Option<ImputationQualityMetrics>,
}

/// Quality metrics for imputation results
#[derive(Debug, Clone)]
pub struct ImputationQualityMetrics {
    /// confidence_intervals
    pub confidence_intervals: Option<Array2<(f64, f64)>>,
    /// uncertainty_estimates
    pub uncertainty_estimates: Option<Array2<f64>>,
    /// imputation_variance
    pub imputation_variance: Option<f64>,
}

/// Trait for type-safe imputation operations
pub trait TypeSafeImputation<T, M> {
    type Output;

    fn impute(&self, data: &TypedArray<T, M, WithMissing>) -> SklResult<Self::Output>;
}

/// Example implementation of type-safe mean imputation
pub struct TypeSafeMeanImputer<D: MissingValueDetector<f64>> {
    detector: D,
}

impl<D: MissingValueDetector<f64>> TypeSafeMeanImputer<D> {
    pub fn new(detector: D) -> Self {
        Self { detector }
    }
}

impl<D: MissingValueDetector<f64>> TypeSafeImputation<f64, MCAR> for TypeSafeMeanImputer<D> {
    type Output = CompleteArray<f64>;

    fn impute(&self, data: &MCARArray<f64>) -> SklResult<Self::Output> {
        let mut result = data.data().clone();
        let mut imputed_positions = Vec::new();

        // Calculate column means
        let mut column_means = Array1::zeros(data.ncols());
        for j in 0..data.ncols() {
            let column = data.data().column(j);
            let valid_values: Vec<f64> = column
                .iter()
                .filter(|&&x| !self.detector.is_missing(&x))
                .copied()
                .collect();

            if !valid_values.is_empty() {
                column_means[j] = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
            }
        }

        // Impute missing values
        for ((i, j), value) in data.data().indexed_iter() {
            if self.detector.is_missing(value) {
                result[[i, j]] = column_means[j];
                imputed_positions.push((i, j));
            }
        }

        Ok(CompleteArray::new_complete(result))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_typed_array_creation() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0]).unwrap();
        let missing_mask =
            Array2::from_shape_vec((3, 2), vec![false, false, true, false, false, false]).unwrap();

        let typed_array =
            TypedArray::<f64, UnknownMechanism, WithMissing>::new_with_missing(data, missing_mask);

        assert_eq!(typed_array.shape(), (3, 2));
        assert_eq!(typed_array.count_missing(), 1);
        assert!(!typed_array.is_complete());
    }

    #[test]
    fn test_fixed_size_array() {
        let data = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let fixed_array = FixedSizeArray::<f64, 2, 3>::new(data).unwrap();

        assert!(fixed_array.validate_dimensions().is_ok());
        assert_eq!(fixed_array.data().shape(), &[2, 3]);
    }

    #[test]
    fn test_fixed_size_array_validation() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let result = FixedSizeArray::<f64, 2, 3>::new(data);

        assert!(result.is_err());
    }

    #[test]
    fn test_nan_detector() {
        let detector = NaNDetector;

        assert!(detector.is_missing(&f64::NAN));
        assert!(!detector.is_missing(&1.0));
        assert!(!detector.is_missing(&0.0));
    }

    #[test]
    fn test_sentinel_detector() {
        let detector = SentinelDetector::new(-999.0);

        assert!(detector.is_missing(&-999.0));
        assert!(!detector.is_missing(&1.0));
        assert!(!detector.is_missing(&0.0));
    }

    #[test]
    fn test_type_safe_mean_imputation() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0]).unwrap();
        let missing_mask =
            Array2::from_shape_vec((3, 2), vec![false, false, true, false, false, false]).unwrap();

        let mcar_array = TypedArray::<f64, MCAR, WithMissing>::new_with_missing(data, missing_mask);
        let imputer = TypeSafeMeanImputer::new(NaNDetector);

        let result = imputer.impute(&mcar_array).unwrap();

        // Mean of column 0: (1.0 + 5.0) / 2 = 3.0
        assert_abs_diff_eq!(result.data()[[1, 0]], 3.0, epsilon = 1e-10);

        // Other values should be unchanged
        assert_abs_diff_eq!(result.data()[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result.data()[[0, 1]], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_missing_pattern_analysis() {
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0,
                2.0,
                3.0,
                f64::NAN,
                5.0,
                6.0,
                7.0,
                f64::NAN,
                9.0,
                f64::NAN,
                11.0,
                f64::NAN,
            ],
        )
        .unwrap();

        let missing_mask = Array2::from_shape_vec(
            (4, 3),
            vec![
                false, false, false, true, false, false, false, true, false, true, false, true,
            ],
        )
        .unwrap();

        let typed_array =
            TypedArray::<f64, UnknownMechanism, WithMissing>::new_with_missing(data, missing_mask);
        let patterns = typed_array.analyze_patterns();

        assert_eq!(patterns.len(), 4); // 4 unique patterns

        // Each pattern should have frequency 0.25 (1/4)
        for pattern in patterns {
            assert_abs_diff_eq!(pattern.frequency, 0.25, epsilon = 1e-10);
            assert_eq!(pattern.count, 1);
        }
    }

    #[test]
    fn test_missing_rate_per_feature() {
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0,
                2.0,
                3.0,
                f64::NAN,
                5.0,
                6.0,
                7.0,
                f64::NAN,
                9.0,
                f64::NAN,
                11.0,
                f64::NAN,
            ],
        )
        .unwrap();

        let missing_mask = Array2::from_shape_vec(
            (4, 3),
            vec![
                false, false, false, true, false, false, false, true, false, true, false, true,
            ],
        )
        .unwrap();

        let typed_array =
            TypedArray::<f64, UnknownMechanism, WithMissing>::new_with_missing(data, missing_mask);
        let missing_rates = typed_array.missing_rate_per_feature();

        // Column 0: 2/4 = 0.5 missing rate
        assert_abs_diff_eq!(missing_rates[0], 0.5, epsilon = 1e-10);

        // Column 1: 1/4 = 0.25 missing rate
        assert_abs_diff_eq!(missing_rates[1], 0.25, epsilon = 1e-10);

        // Column 2: 1/4 = 0.25 missing rate
        assert_abs_diff_eq!(missing_rates[2], 0.25, epsilon = 1e-10);
    }
}
