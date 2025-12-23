//! Type safety utilities for compile-time validation and zero-cost abstractions
//!
//! This module provides phantom types, zero-cost wrappers, and compile-time
//! validation utilities to ensure type safety in machine learning operations.

use crate::{UtilsError, UtilsResult};
use scirs2_core::ndarray::{Array1, Array2};
use std::marker::PhantomData;

// ===== PHANTOM TYPES FOR STATE VALIDATION =====

/// Phantom type for untrained state
pub struct Untrained;

/// Phantom type for trained state
pub struct Trained;

/// Phantom type for validated data
pub struct Validated;

/// Phantom type for unvalidated data
pub struct Unvalidated;

/// State-based wrapper for ML models
#[derive(Debug, Clone)]
pub struct ModelState<T, State> {
    pub inner: T,
    _state: PhantomData<State>,
}

impl<T> ModelState<T, Untrained> {
    /// Create a new untrained model
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            _state: PhantomData,
        }
    }

    /// Transition to trained state (only available for untrained models)
    pub fn train(self) -> ModelState<T, Trained> {
        ModelState {
            inner: self.inner,
            _state: PhantomData,
        }
    }
}

impl<T> ModelState<T, Trained> {
    /// Predict (only available for trained models)
    pub fn predict<F, Input, Output>(&self, predict_fn: F, input: Input) -> Output
    where
        F: Fn(&T, Input) -> Output,
    {
        predict_fn(&self.inner, input)
    }

    /// Reset to untrained state
    pub fn reset(self) -> ModelState<T, Untrained> {
        ModelState {
            inner: self.inner,
            _state: PhantomData,
        }
    }
}

// ===== VALIDATED DATA TYPES =====

/// Data wrapper with validation state
#[derive(Debug, Clone)]
pub struct DataState<T, State> {
    pub data: T,
    _state: PhantomData<State>,
}

impl<T> DataState<T, Unvalidated> {
    /// Create new unvalidated data
    pub fn new(data: T) -> Self {
        Self {
            data,
            _state: PhantomData,
        }
    }

    /// Validate data and transition to validated state
    pub fn validate<F>(self, validator: F) -> UtilsResult<DataState<T, Validated>>
    where
        F: FnOnce(&T) -> UtilsResult<()>,
    {
        validator(&self.data)?;
        Ok(DataState {
            data: self.data,
            _state: PhantomData,
        })
    }
}

impl<T> DataState<T, Validated> {
    /// Access validated data (only available after validation)
    pub fn as_validated(&self) -> &T {
        &self.data
    }

    /// Transform validated data while preserving validation state
    pub fn map<U, F>(self, transform: F) -> DataState<U, Validated>
    where
        F: FnOnce(T) -> U,
    {
        DataState {
            data: transform(self.data),
            _state: PhantomData,
        }
    }
}

// ===== DIMENSIONAL TYPE SAFETY =====

/// Phantom types for dimensions
pub struct D1;
pub struct D2;
pub struct D3;

/// Dimensionally-typed array wrapper
#[derive(Debug, Clone)]
pub struct TypedArray<T, D> {
    data: T,
    _dimension: PhantomData<D>,
}

impl<T> TypedArray<Array1<T>, D1> {
    /// Create a 1D typed array
    pub fn new_1d(array: Array1<T>) -> Self {
        Self {
            data: array,
            _dimension: PhantomData,
        }
    }

    /// Get the underlying 1D array
    pub fn as_array1(&self) -> &Array1<T> {
        &self.data
    }

    /// Convert to 2D array (single row)
    pub fn to_2d(self) -> TypedArray<Array2<T>, D2>
    where
        T: Clone,
    {
        let shape = (1, self.data.len());
        let data = Array2::from_shape_vec(shape, self.data.to_vec()).unwrap();
        TypedArray {
            data,
            _dimension: PhantomData,
        }
    }
}

impl<T> TypedArray<Array2<T>, D2> {
    /// Create a 2D typed array
    pub fn new_2d(array: Array2<T>) -> Self {
        Self {
            data: array,
            _dimension: PhantomData,
        }
    }

    /// Get the underlying 2D array
    pub fn as_array2(&self) -> &Array2<T> {
        &self.data
    }

    /// Get shape information
    pub fn shape(&self) -> (usize, usize) {
        let shape = self.data.shape();
        (shape[0], shape[1])
    }

    /// Flatten to 1D array
    pub fn flatten(self) -> TypedArray<Array1<T>, D1>
    where
        T: Clone,
    {
        let (vec, offset) = self.data.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0), "Array offset must be zero for conversion");
        let data = Array1::from_vec(vec);
        TypedArray {
            data,
            _dimension: PhantomData,
        }
    }
}

// ===== UNITS AND MEASUREMENTS =====

/// Unit types for type-safe measurements
pub trait Unit: 'static {
    const NAME: &'static str;
}

pub struct Meters;
pub struct Seconds;
pub struct Kilograms;
pub struct Pixels;

impl Unit for Meters {
    const NAME: &'static str = "meters";
}

impl Unit for Seconds {
    const NAME: &'static str = "seconds";
}

impl Unit for Kilograms {
    const NAME: &'static str = "kilograms";
}

impl Unit for Pixels {
    const NAME: &'static str = "pixels";
}

/// Type-safe measurement with units
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Measurement<T, U: Unit> {
    value: T,
    _unit: PhantomData<U>,
}

impl<T, U: Unit> Measurement<T, U> {
    /// Create a new measurement
    pub fn new(value: T) -> Self {
        Self {
            value,
            _unit: PhantomData,
        }
    }

    /// Get the value
    pub fn value(&self) -> &T {
        &self.value
    }

    /// Convert to different unit (unsafe, requires manual verification)
    ///
    /// # Safety
    ///
    /// The caller must ensure that the conversion between units is mathematically valid
    /// and that the value makes sense in the target unit system.
    pub unsafe fn convert_unit<V: Unit>(self) -> Measurement<T, V> {
        Measurement {
            value: self.value,
            _unit: PhantomData,
        }
    }
}

impl<T, U: Unit> std::ops::Add for Measurement<T, U>
where
    T: std::ops::Add<Output = T>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self {
            value: self.value + other.value,
            _unit: PhantomData,
        }
    }
}

impl<T, U: Unit> std::ops::Sub for Measurement<T, U>
where
    T: std::ops::Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self {
            value: self.value - other.value,
            _unit: PhantomData,
        }
    }
}

// ===== COMPILE-TIME VALIDATION =====

/// Trait for compile-time shape validation
pub trait ShapeValidation {
    type Shape;
    fn validate_shape(shape: Self::Shape) -> bool;
}

/// Shape constraint: exactly N elements
pub struct ExactSize<const N: usize>;

impl<const N: usize> ShapeValidation for ExactSize<N> {
    type Shape = usize;

    fn validate_shape(shape: Self::Shape) -> bool {
        shape == N
    }
}

/// Shape constraint: minimum N elements
pub struct MinSize<const N: usize>;

impl<const N: usize> ShapeValidation for MinSize<N> {
    type Shape = usize;

    fn validate_shape(shape: Self::Shape) -> bool {
        shape >= N
    }
}

/// Shape-validated array
pub struct ValidatedArray<T, V: ShapeValidation> {
    data: Array1<T>,
    _validator: PhantomData<V>,
}

impl<T, V: ShapeValidation<Shape = usize>> ValidatedArray<T, V> {
    /// Create a validated array (compile-time check)
    pub fn new(data: Array1<T>) -> Option<Self> {
        if V::validate_shape(data.len()) {
            Some(Self {
                data,
                _validator: PhantomData,
            })
        } else {
            None
        }
    }

    /// Access the validated data
    pub fn data(&self) -> &Array1<T> {
        &self.data
    }
}

// ===== ZERO-COST ABSTRACTIONS =====

/// Zero-cost wrapper for normalized values [0, 1]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Normalized<T>(T);

impl<T> Normalized<T> {
    /// Create a normalized value (unsafe - assumes value is in [0, 1])
    ///
    /// # Safety
    ///
    /// The caller must ensure that the value is within the range [0, 1].
    /// Using values outside this range will result in undefined behavior.
    pub unsafe fn new_unchecked(value: T) -> Self {
        Self(value)
    }

    /// Get the inner value
    pub fn get(self) -> T {
        self.0
    }
}

impl Normalized<f64> {
    /// Create a normalized value with validation
    pub fn new(value: f64) -> UtilsResult<Self> {
        if (0.0..=1.0).contains(&value) {
            Ok(Self(value))
        } else {
            Err(UtilsError::InvalidParameter(format!(
                "Value {value} is not in range [0, 1]"
            )))
        }
    }

    /// Clamp value to [0, 1] range
    pub fn clamp(value: f64) -> Self {
        Self(value.clamp(0.0, 1.0))
    }
}

/// Zero-cost wrapper for positive values
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Positive<T>(T);

impl<T> Positive<T> {
    /// Get the inner value
    pub fn get(self) -> T {
        self.0
    }
}

impl Positive<f64> {
    /// Create a positive value with validation
    pub fn new(value: f64) -> UtilsResult<Self> {
        if value > 0.0 {
            Ok(Self(value))
        } else {
            Err(UtilsError::InvalidParameter(format!(
                "Value {value} is not positive"
            )))
        }
    }
}

/// Zero-cost wrapper for non-negative values
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct NonNegative<T>(T);

impl<T> NonNegative<T> {
    /// Get the inner value
    pub fn get(self) -> T {
        self.0
    }
}

impl NonNegative<f64> {
    /// Create a non-negative value with validation
    pub fn new(value: f64) -> UtilsResult<Self> {
        if value >= 0.0 {
            Ok(Self(value))
        } else {
            Err(UtilsError::InvalidParameter(format!(
                "Value {value} is negative"
            )))
        }
    }
}

// ===== COMPILE-TIME ASSERTIONS =====

/// Compile-time assertion macro
#[macro_export]
macro_rules! const_assert {
    ($condition:expr) => {
        const _: () = if !$condition {
            panic!("Compile-time assertion failed");
        } else {
            ()
        };
    };
}

/// Compile-time shape assertion
#[macro_export]
macro_rules! assert_shape {
    ($array:expr, $expected:expr) => {
        if $array.shape() != $expected {
            return Err(UtilsError::ShapeMismatch {
                expected: $expected.to_vec(),
                actual: $array.shape().to_vec(),
            });
        }
    };
}

// ===== TYPE-LEVEL COMPUTATION =====

/// Type-level arithmetic for compile-time computation
pub trait TypeNum {
    const VALUE: usize;
}

pub struct Zero;
pub struct One;
pub struct Two;
pub struct Three;

impl TypeNum for Zero {
    const VALUE: usize = 0;
}
impl TypeNum for One {
    const VALUE: usize = 1;
}
impl TypeNum for Two {
    const VALUE: usize = 2;
}
impl TypeNum for Three {
    const VALUE: usize = 3;
}

/// Add two type-level numbers
pub trait Add<Rhs> {
    type Output: TypeNum;
}

impl Add<Zero> for Zero {
    type Output = Zero;
}
impl Add<One> for Zero {
    type Output = One;
}
impl Add<Two> for Zero {
    type Output = Two;
}
impl Add<Zero> for One {
    type Output = One;
}
impl Add<One> for One {
    type Output = Two;
}
impl Add<Two> for One {
    type Output = Three;
}

/// Compile-time validated matrix multiplication
pub struct MatrixMul<L: TypeNum, M: TypeNum, N: TypeNum> {
    _phantom: PhantomData<(L, M, N)>,
}

impl<L: TypeNum, M: TypeNum, N: TypeNum> MatrixMul<L, M, N> {
    /// Validate matrix multiplication at compile time
    pub fn multiply(left: &Array2<f64>, right: &Array2<f64>) -> UtilsResult<Array2<f64>> {
        // Runtime validation (would be compile-time in a full implementation)
        let left_shape = left.shape();
        let right_shape = right.shape();

        if left_shape[1] != right_shape[0] {
            return Err(UtilsError::ShapeMismatch {
                expected: vec![left_shape[0], right_shape[1]],
                actual: vec![left_shape[0], left_shape[1], right_shape[0], right_shape[1]],
            });
        }

        Ok(left.dot(right))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_state_transitions() {
        #[derive(Debug, Clone)]
        struct MockModel {
            value: i32,
        }

        let model = MockModel { value: 42 };
        let untrained = ModelState::new(model);

        // Can only train untrained models
        let trained = untrained.train();

        // Can only predict with trained models
        let result = trained.predict(|model, input: i32| model.value + input, 10);
        assert_eq!(result, 52);

        // Can reset trained model to untrained
        let _reset = trained.reset();
    }

    #[test]
    fn test_data_validation() {
        let data = vec![1, 2, 3, 4, 5];
        let unvalidated = DataState::new(data);

        // Validate that all elements are positive
        let validated = unvalidated
            .validate(|data| {
                if data.iter().all(|&x| x > 0) {
                    Ok(())
                } else {
                    Err(UtilsError::InvalidParameter(
                        "Negative values found".to_string(),
                    ))
                }
            })
            .unwrap();

        // Can access validated data
        let validated_data = validated.as_validated();
        assert_eq!(validated_data.len(), 5);

        // Transform while preserving validation
        let transformed = validated.map(|data| data.len());
        assert_eq!(*transformed.as_validated(), 5);
    }

    #[test]
    fn test_typed_arrays() {
        let array1d = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let typed1d = TypedArray::new_1d(array1d);

        // Convert to 2D
        let typed2d = typed1d.to_2d();
        assert_eq!(typed2d.shape(), (1, 3));

        // Flatten back to 1D
        let flattened = typed2d.flatten();
        assert_eq!(flattened.as_array1().len(), 3);
    }

    #[test]
    fn test_measurements() {
        let distance1 = Measurement::<f64, Meters>::new(10.0);
        let distance2 = Measurement::<f64, Meters>::new(5.0);

        let total_distance = distance1 + distance2;
        assert_eq!(*total_distance.value(), 15.0);

        let _time = Measurement::<f64, Seconds>::new(2.0);
        // This would not compile: distance1 + time (different units)
    }

    #[test]
    fn test_normalized_values() {
        // Valid normalized value
        let norm1 = Normalized::new(0.5).unwrap();
        assert_eq!(norm1.get(), 0.5);

        // Invalid normalized value
        assert!(Normalized::new(1.5).is_err());

        // Clamped value
        let norm2 = Normalized::clamp(1.5);
        assert_eq!(norm2.get(), 1.0);
    }

    #[test]
    fn test_positive_values() {
        let pos = Positive::new(5.0).unwrap();
        assert_eq!(pos.get(), 5.0);

        assert!(Positive::new(-1.0).is_err());
        assert!(Positive::new(0.0).is_err());
    }

    #[test]
    fn test_validated_arrays() {
        let data = Array1::from_vec(vec![1, 2, 3]);

        // Should succeed for ExactSize<3>
        let validated = ValidatedArray::<i32, ExactSize<3>>::new(data.clone());
        assert!(validated.is_some());

        // Should fail for ExactSize<5>
        let validated = ValidatedArray::<i32, ExactSize<5>>::new(data.clone());
        assert!(validated.is_none());

        // Should succeed for MinSize<2>
        let validated = ValidatedArray::<i32, MinSize<2>>::new(data);
        assert!(validated.is_some());
    }

    #[test]
    fn test_matrix_multiplication_validation() {
        let left = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let right = Array2::from_shape_vec((3, 2), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();

        let result = MatrixMul::<Two, Three, Two>::multiply(&left, &right).unwrap();
        assert_eq!(result.shape(), &[2, 2]);

        // Should fail with incompatible shapes
        let wrong_right = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(MatrixMul::<Two, Three, Two>::multiply(&left, &wrong_right).is_err());
    }
}
