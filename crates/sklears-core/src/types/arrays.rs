/// Array type aliases and common array operations for machine learning
///
/// This module provides convenient type aliases for ndarray types commonly
/// used in machine learning operations.
// SciRS2 Policy: Using scirs2_core::ndarray for unified access (COMPLIANT)
use scirs2_core::ndarray::{
    Array1 as NdArray1, Array2 as NdArray2, ArrayView1 as NdArrayView1, ArrayView2 as NdArrayView2,
    ArrayViewMut1 as NdArrayViewMut1, ArrayViewMut2 as NdArrayViewMut2,
};
use scirs2_core::numeric::{FromPrimitive, One, Zero};
use std::borrow::Cow;

use super::traits::{FloatBounds, IntBounds};

/// 1-dimensional array type alias
pub type Array1<T> = NdArray1<T>;

/// 2-dimensional array type alias  
pub type Array2<T> = NdArray2<T>;

/// 1-dimensional array view type alias
pub type ArrayView1<'a, T> = NdArrayView1<'a, T>;

/// 2-dimensional array view type alias
pub type ArrayView2<'a, T> = NdArrayView2<'a, T>;

/// 1-dimensional mutable array view type alias
pub type ArrayViewMut1<'a, T> = NdArrayViewMut1<'a, T>;

/// 2-dimensional mutable array view type alias
pub type ArrayViewMut2<'a, T> = NdArrayViewMut2<'a, T>;

/// Default floating point type for the library
pub type Float = f64;

/// Default integer type for the library
pub type Int = i32;

/// Domain-specific type aliases for machine learning
/// Feature matrix type (n_samples x n_features)
pub type Features<T = Float> = Array2<T>;

/// Target vector type (n_samples,)
pub type Target<T = Float> = Array1<T>;

/// Sample weights type (n_samples,)
pub type SampleWeight<T = Float> = Array1<T>;

/// Predictions vector type (n_samples,)
pub type Predictions<T = Float> = Array1<T>;

/// Probability matrix type (n_samples x n_classes)
pub type Probabilities<T = Float> = Array2<T>;

/// Labels vector type (n_samples,)
pub type Labels<T = Int> = Array1<T>;

/// Distance matrix type (n_samples x n_samples)
pub type Distances<T = Float> = Array2<T>;

/// Similarity matrix type (n_samples x n_samples)
pub type Similarities<T = Float> = Array2<T>;

/// Copy-on-write (Cow) variants for efficient memory usage
/// Copy-on-write features matrix
pub type CowFeatures<'a, T = Float> = Cow<'a, Array2<T>>;

/// Copy-on-write target vector
pub type CowTarget<'a, T = Float> = Cow<'a, Array1<T>>;

/// Copy-on-write predictions vector
pub type CowPredictions<'a, T = Float> = Cow<'a, Array1<T>>;

/// Copy-on-write probabilities matrix
pub type CowProbabilities<'a, T = Float> = Cow<'a, Array2<T>>;

/// Copy-on-write sample weights vector
pub type CowSampleWeight<'a, T = Float> = Cow<'a, Array1<T>>;

/// Copy-on-write labels vector
pub type CowLabels<'a, T = Int> = Cow<'a, Array1<T>>;

/// Array shape utilities
pub mod shape {
    use super::*;
    use crate::error::{Result, SklearsError};

    /// Check if two arrays have compatible shapes for element-wise operations
    pub fn check_compatible_shapes<T, U>(a: &Array2<T>, b: &Array2<U>) -> Result<()> {
        if a.dim() != b.dim() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{:?}", a.dim()),
                actual: format!("{:?}", b.dim()),
            });
        }
        Ok(())
    }

    /// Check if X and y have compatible number of samples
    pub fn check_n_samples<T, U>(x: &Array2<T>, y: &Array1<U>) -> Result<()> {
        let n_samples_x = x.nrows();
        let n_samples_y = y.len();

        if n_samples_x != n_samples_y {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("X.shape[0] == y.shape[0] ({n_samples_x})"),
                actual: format!("X.shape[0]={n_samples_x}, y.shape[0]={n_samples_y}"),
            });
        }
        Ok(())
    }

    /// Check if arrays have compatible shapes for matrix multiplication
    pub fn check_matmul_compatible<T, U>(a: &Array2<T>, b: &Array2<U>) -> Result<()> {
        if a.ncols() != b.nrows() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("A.shape[1] == B.shape[0] ({})", a.ncols()),
                actual: format!("A.shape[1]={}, B.shape[0]={}", a.ncols(), b.nrows()),
            });
        }
        Ok(())
    }

    /// Get the shape of a 2D array as (rows, cols)
    pub fn get_shape<T>(array: &Array2<T>) -> (usize, usize) {
        array.dim()
    }

    /// Calculate the total number of elements in an array
    pub fn total_elements<T>(array: &Array2<T>) -> usize {
        let (rows, cols) = array.dim();
        rows * cols
    }

    /// Check if an array is square
    pub fn is_square<T>(array: &Array2<T>) -> bool {
        let (rows, cols) = array.dim();
        rows == cols
    }

    /// Check if arrays have the same shape
    pub fn same_shape<T, U>(a: &Array2<T>, b: &Array2<U>) -> bool {
        a.dim() == b.dim()
    }
}

/// Array creation utilities
pub mod creation {
    use super::*;
    use crate::error::{Result, SklearsError};

    /// Create a zero-filled array with the given shape
    pub fn zeros<T>(shape: (usize, usize)) -> Array2<T>
    where
        T: Clone + Zero,
    {
        Array2::zeros(shape)
    }

    /// Create a one-filled array with the given shape
    pub fn ones<T>(shape: (usize, usize)) -> Array2<T>
    where
        T: Clone + One,
    {
        Array2::ones(shape)
    }

    /// Create an identity matrix of the given size
    pub fn eye<T>(n: usize) -> Array2<T>
    where
        T: Clone + Zero + One,
    {
        Array2::eye(n)
    }

    /// Create an array filled with a specific value
    pub fn full<T>(shape: (usize, usize), value: T) -> Array2<T>
    where
        T: Clone,
    {
        Array2::from_elem(shape, value)
    }

    /// Create a random array with values between 0 and 1
    #[cfg(feature = "std")]
    pub fn random<T>(shape: (usize, usize)) -> Array2<T>
    where
        T: FloatBounds + FromPrimitive,
    {
        // SciRS2 Policy: Using scirs2_core::random (COMPLIANT)
        use scirs2_core::random::{thread_rng, Rng};
        let mut rng = thread_rng();
        Array2::from_shape_fn(shape, |_| {
            let random_f64: f64 = rng.random();
            T::from_f64(random_f64).unwrap_or_else(|| T::zero())
        })
    }

    /// Create a linearly spaced 1D array
    pub fn linspace<T>(start: T, stop: T, num: usize) -> Result<Array1<T>>
    where
        T: FloatBounds + FromPrimitive,
    {
        if num == 0 {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be positive".to_string(),
            ));
        }

        if num == 1 {
            return Ok(Array1::from_vec(vec![start]));
        }

        let step = (stop - start) / T::from_usize(num - 1).unwrap();
        let values: Vec<T> = (0..num)
            .map(|i| start + step * T::from_usize(i).unwrap())
            .collect();

        Ok(Array1::from_vec(values))
    }

    /// Create an array from nested vectors
    pub fn from_nested_vec<T>(data: Vec<Vec<T>>) -> Result<Array2<T>>
    where
        T: Clone,
    {
        if data.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Cannot create array from empty data".to_string(),
            ));
        }

        let rows = data.len();
        let cols = data[0].len();

        // Check that all rows have the same length
        for (i, row) in data.iter().enumerate() {
            if row.len() != cols {
                return Err(SklearsError::ShapeMismatch {
                    expected: format!("All rows should have length {cols}"),
                    actual: format!("Row {i} has length {}", row.len()),
                });
            }
        }

        let flat_data: Vec<T> = data.into_iter().flatten().collect();
        Array2::from_shape_vec((rows, cols), flat_data)
            .map_err(|e| SklearsError::Other(format!("Array creation failed: {e}")))
    }
}

/// Array validation utilities
pub mod validation {
    use super::*;
    use crate::error::{Result, SklearsError};

    /// Check if array contains only finite values
    pub fn check_finite<T>(array: &Array2<T>) -> Result<()>
    where
        T: FloatBounds,
    {
        for value in array.iter() {
            if !value.is_finite() {
                return Err(SklearsError::InvalidData {
                    reason: "Array contains non-finite values (NaN or infinity)".to_string(),
                });
            }
        }
        Ok(())
    }

    /// Check if array is empty
    pub fn check_not_empty<T>(array: &Array2<T>) -> Result<()> {
        if array.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Array cannot be empty".to_string(),
            ));
        }
        Ok(())
    }

    /// Check if array has minimum required dimensions
    pub fn check_min_samples<T>(array: &Array2<T>, min_samples: usize) -> Result<()> {
        if array.nrows() < min_samples {
            return Err(SklearsError::InvalidInput(format!(
                "Array has {} samples, but at least {min_samples} are required",
                array.nrows()
            )));
        }
        Ok(())
    }

    /// Check if array has minimum required features
    pub fn check_min_features<T>(array: &Array2<T>, min_features: usize) -> Result<()> {
        if array.ncols() < min_features {
            return Err(SklearsError::InvalidInput(format!(
                "Array has {} features, but at least {min_features} are required",
                array.ncols()
            )));
        }
        Ok(())
    }

    /// Check if target array has valid values for classification
    pub fn check_classification_targets<T>(targets: &Array1<T>) -> Result<()>
    where
        T: IntBounds + std::fmt::Display,
    {
        for &target in targets.iter() {
            if target < T::zero() {
                return Err(SklearsError::InvalidData {
                    reason: format!("Classification targets must be non-negative, found {target}"),
                });
            }
        }
        Ok(())
    }

    /// Check if probabilities sum to 1 for each sample
    pub fn check_probabilities<T>(probs: &Array2<T>, tolerance: T) -> Result<()>
    where
        T: FloatBounds,
    {
        // SciRS2 Policy: Using scirs2_core::ndarray::Axis for unified access (COMPLIANT)
        for (i, row) in probs.axis_iter(scirs2_core::ndarray::Axis(0)).enumerate() {
            let sum: T = row.sum();
            if (sum - T::one()).abs() > tolerance {
                return Err(SklearsError::InvalidData {
                    reason: format!(
                        "Probabilities for sample {i} sum to {sum}, expected 1.0 ± {tolerance}"
                    ),
                });
            }
        }
        Ok(())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_array_creation() {
        let zeros = creation::zeros::<f64>((3, 4));
        assert_eq!(zeros.dim(), (3, 4));
        assert_eq!(zeros[[0, 0]], 0.0);

        let ones = creation::ones::<f64>((2, 3));
        assert_eq!(ones.dim(), (2, 3));
        assert_eq!(ones[[1, 2]], 1.0);

        let eye = creation::eye::<f64>(3);
        assert_eq!(eye.dim(), (3, 3));
        assert_eq!(eye[[0, 0]], 1.0);
        assert_eq!(eye[[0, 1]], 0.0);
    }

    #[test]
    fn test_linspace() {
        let result = creation::linspace(0.0, 10.0, 11).unwrap();
        assert_eq!(result.len(), 11);
        assert_abs_diff_eq!(result[0], 0.0);
        assert_abs_diff_eq!(result[10], 10.0);
        assert_abs_diff_eq!(result[5], 5.0);
    }

    #[test]
    fn test_from_nested_vec() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let array = creation::from_nested_vec(data).unwrap();
        assert_eq!(array.dim(), (2, 3));
        assert_eq!(array[[0, 0]], 1.0);
        assert_eq!(array[[1, 2]], 6.0);
    }

    #[test]
    fn test_shape_utilities() {
        let a = Array2::<f64>::zeros((3, 4));
        let b = Array2::<f64>::ones((3, 4));
        let c = Array2::<f64>::ones((4, 3));

        assert!(shape::same_shape(&a, &b));
        assert!(!shape::same_shape(&a, &c));
        assert_eq!(shape::get_shape(&a), (3, 4));
        assert_eq!(shape::total_elements(&a), 12);
        assert!(!shape::is_square(&a));

        let square = Array2::<f64>::eye(3);
        assert!(shape::is_square(&square));
    }

    #[test]
    fn test_validation() {
        let finite_array = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(validation::check_finite(&finite_array).is_ok());

        let infinite_array =
            Array2::from_shape_vec((2, 2), vec![1.0, f64::INFINITY, 3.0, 4.0]).unwrap();
        assert!(validation::check_finite(&infinite_array).is_err());

        assert!(validation::check_not_empty(&finite_array).is_ok());
        assert!(validation::check_min_samples(&finite_array, 2).is_ok());
        assert!(validation::check_min_samples(&finite_array, 3).is_err());
        assert!(validation::check_min_features(&finite_array, 2).is_ok());
        assert!(validation::check_min_features(&finite_array, 3).is_err());
    }

    #[test]
    fn test_classification_targets() {
        let valid_targets = Array1::from_vec(vec![0, 1, 2, 1, 0]);
        assert!(validation::check_classification_targets(&valid_targets).is_ok());

        let invalid_targets = Array1::from_vec(vec![0, 1, -1, 1, 0]);
        assert!(validation::check_classification_targets(&invalid_targets).is_err());
    }

    #[test]
    fn test_probability_validation() {
        let valid_probs =
            Array2::from_shape_vec((2, 3), vec![0.3, 0.4, 0.3, 0.5, 0.2, 0.3]).unwrap();
        assert!(validation::check_probabilities(&valid_probs, 1e-6).is_ok());

        let invalid_probs =
            Array2::from_shape_vec((2, 3), vec![0.3, 0.4, 0.4, 0.5, 0.2, 0.3]).unwrap();
        assert!(validation::check_probabilities(&invalid_probs, 1e-6).is_err());
    }
}
