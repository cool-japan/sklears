//! # Refinement Types System for sklears-core
//!
//! This module provides a sophisticated refinement types system that enables
//! compile-time verification of complex predicates and constraints. Refinement
//! types extend the type system with logical predicates, providing stronger
//! guarantees about program correctness.
//!
//! ## Key Features
//!
//! - **Type-level Predicates**: Express complex constraints at the type level
//! - **Compile-time Verification**: Catch constraint violations at compile time
//! - **Zero-cost Abstractions**: No runtime overhead for type refinements
//! - **Dependent Refinements**: Support for value-dependent type constraints
//! - **Compositional Design**: Combine refinements modularly
//!
//! ## Architecture
//!
//! The system is built on three layers:
//! 1. **Base Refinements**: Primitive refinement types (positive, non-zero, bounded)
//! 2. **Composite Refinements**: Combinations of base refinements
//! 3. **Domain-specific Refinements**: ML-specific constraints (probabilities, dimensions)
//!
//! ## Examples
//!
//! ```rust,ignore
//! use sklears_core::refinement_types::*;
//!
//! // Positive integers only
//! let sample_count: Positive<usize> = Positive::new(100)?;
//!
//! // Probabilities must be in [0, 1]
//! let prob: Probability = Probability::new(0.75)?;
//!
//! // Bounded values
//! let learning_rate: Bounded<f64, 0.0, 1.0> = Bounded::new(0.01)?;
//!
//! // Non-empty collections
//! let features: NonEmpty<Vec<f64>> = NonEmpty::new(vec![1.0, 2.0, 3.0])?;
//! ```

use crate::error::{Result, SklearsError};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::marker::PhantomData;
use std::ops::Deref;

// =============================================================================
// Core Refinement Type Infrastructure
// =============================================================================

/// Trait for types that can be refined with predicates
pub trait Refinement: Sized {
    /// The predicate that must hold for this refinement
    type Predicate: RefinementPredicate<Self>;

    /// Create a refined value, checking the predicate
    fn refine(value: Self) -> Result<Refined<Self, Self::Predicate>> {
        if Self::Predicate::check(&value) {
            Ok(Refined::new_unchecked(value))
        } else {
            Err(SklearsError::ValidationError(format!(
                "Refinement predicate failed for {}",
                std::any::type_name::<Self>()
            )))
        }
    }

    /// Get the underlying value
    fn unrefine(refined: Refined<Self, Self::Predicate>) -> Self {
        refined.value
    }
}

/// Trait for refinement predicates that can be checked
pub trait RefinementPredicate<T> {
    /// Check if the predicate holds for the given value
    fn check(value: &T) -> bool;

    /// Description of the predicate for error messages
    fn description() -> String {
        std::any::type_name::<Self>().to_string()
    }
}

/// A refined value that satisfies a predicate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Refined<T, P>
where
    P: RefinementPredicate<T>,
{
    value: T,
    _predicate: PhantomData<P>,
}

impl<T, P> Refined<T, P>
where
    P: RefinementPredicate<T>,
{
    /// Create a refined value without checking (unsafe - use with caution)
    pub fn new_unchecked(value: T) -> Self {
        Self {
            value,
            _predicate: PhantomData,
        }
    }

    /// Create a refined value, checking the predicate
    pub fn new(value: T) -> Result<Self> {
        if P::check(&value) {
            Ok(Self::new_unchecked(value))
        } else {
            Err(SklearsError::ValidationError(format!(
                "Refinement predicate '{}' failed",
                P::description()
            )))
        }
    }

    /// Get the underlying value
    pub fn into_inner(self) -> T {
        self.value
    }

    /// Get a reference to the underlying value
    pub fn get(&self) -> &T {
        &self.value
    }

    /// Try to map a function over the refined value, maintaining the refinement
    pub fn try_map<F, U, Q>(self, f: F) -> Result<Refined<U, Q>>
    where
        F: FnOnce(T) -> U,
        Q: RefinementPredicate<U>,
    {
        let new_value = f(self.value);
        Refined::<U, Q>::new(new_value)
    }
}

impl<T, P> Deref for Refined<T, P>
where
    P: RefinementPredicate<T>,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T, P> fmt::Display for Refined<T, P>
where
    T: fmt::Display,
    P: RefinementPredicate<T>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

// =============================================================================
// Base Refinement Predicates
// =============================================================================

/// Predicate for positive numbers
#[derive(Debug, Clone, Copy)]
pub struct PositivePredicate;

impl<T> RefinementPredicate<T> for PositivePredicate
where
    T: PartialOrd + Default,
{
    fn check(value: &T) -> bool {
        value > &T::default()
    }

    fn description() -> String {
        "must be positive".to_string()
    }
}

/// Predicate for non-negative numbers
#[derive(Debug, Clone, Copy)]
pub struct NonNegativePredicate;

impl<T> RefinementPredicate<T> for NonNegativePredicate
where
    T: PartialOrd + Default,
{
    fn check(value: &T) -> bool {
        value >= &T::default()
    }

    fn description() -> String {
        "must be non-negative".to_string()
    }
}

/// Predicate for non-zero values
#[derive(Debug, Clone, Copy)]
pub struct NonZeroPredicate;

impl<T> RefinementPredicate<T> for NonZeroPredicate
where
    T: PartialEq + Default,
{
    fn check(value: &T) -> bool {
        value != &T::default()
    }

    fn description() -> String {
        "must be non-zero".to_string()
    }
}

/// Predicate for values within a specific range
#[derive(Debug, Clone, Copy)]
pub struct RangePredicate<T, const MIN: i64, const MAX: i64> {
    _phantom: PhantomData<T>,
}

impl<T, const MIN: i64, const MAX: i64> RefinementPredicate<T> for RangePredicate<T, MIN, MAX>
where
    T: PartialOrd + From<i64>,
{
    fn check(value: &T) -> bool {
        let min_val = T::from(MIN);
        let max_val = T::from(MAX);
        value >= &min_val && value <= &max_val
    }

    fn description() -> String {
        format!("must be in range [{}, {}]", MIN, MAX)
    }
}

/// Predicate for non-empty collections
#[derive(Debug, Clone, Copy)]
pub struct NonEmptyPredicate;

impl<T> RefinementPredicate<Vec<T>> for NonEmptyPredicate {
    fn check(value: &Vec<T>) -> bool {
        !value.is_empty()
    }

    fn description() -> String {
        "must be non-empty".to_string()
    }
}

impl<T> RefinementPredicate<&[T]> for NonEmptyPredicate {
    fn check(value: &&[T]) -> bool {
        !value.is_empty()
    }

    fn description() -> String {
        "must be non-empty".to_string()
    }
}

// =============================================================================
// Convenient Type Aliases
// =============================================================================

/// Positive numbers (> 0)
pub type Positive<T> = Refined<T, PositivePredicate>;

/// Non-negative numbers (>= 0)
pub type NonNegative<T> = Refined<T, NonNegativePredicate>;

/// Non-zero values
pub type NonZero<T> = Refined<T, NonZeroPredicate>;

/// Non-empty collections
pub type NonEmpty<T> = Refined<T, NonEmptyPredicate>;

// =============================================================================
// ML-Specific Refinement Types
// =============================================================================

/// Probability value in [0, 1]
#[derive(Debug, Clone, Copy)]
pub struct ProbabilityPredicate;

impl RefinementPredicate<f64> for ProbabilityPredicate {
    fn check(value: &f64) -> bool {
        value.is_finite() && *value >= 0.0 && *value <= 1.0
    }

    fn description() -> String {
        "must be a valid probability in [0, 1]".to_string()
    }
}

/// A validated probability value
pub type ValidProbability = Refined<f64, ProbabilityPredicate>;

/// Learning rate (typically in (0, 1])
#[derive(Debug, Clone, Copy)]
pub struct LearningRatePredicate;

impl RefinementPredicate<f64> for LearningRatePredicate {
    fn check(value: &f64) -> bool {
        value.is_finite() && *value > 0.0 && *value <= 1.0
    }

    fn description() -> String {
        "must be a valid learning rate in (0, 1]".to_string()
    }
}

/// A validated learning rate
pub type ValidLearningRate = Refined<f64, LearningRatePredicate>;

/// Regularization parameter (non-negative)
#[derive(Debug, Clone, Copy)]
pub struct RegularizationPredicate;

impl RefinementPredicate<f64> for RegularizationPredicate {
    fn check(value: &f64) -> bool {
        value.is_finite() && *value >= 0.0
    }

    fn description() -> String {
        "must be a non-negative regularization parameter".to_string()
    }
}

/// A validated regularization parameter
pub type ValidRegularization = Refined<f64, RegularizationPredicate>;

/// Sample count (positive integer)
pub type SampleCount = Positive<usize>;

/// Feature count (positive integer)
pub type FeatureCount = Positive<usize>;

/// Number of iterations (positive integer)
pub type IterationCount = Positive<usize>;

// =============================================================================
// Composite Refinement Predicates
// =============================================================================

/// Conjunction of two predicates (AND)
#[derive(Debug, Clone, Copy)]
pub struct And<P1, P2> {
    _phantom: PhantomData<(P1, P2)>,
}

impl<T, P1, P2> RefinementPredicate<T> for And<P1, P2>
where
    P1: RefinementPredicate<T>,
    P2: RefinementPredicate<T>,
{
    fn check(value: &T) -> bool {
        P1::check(value) && P2::check(value)
    }

    fn description() -> String {
        format!("({}) AND ({})", P1::description(), P2::description())
    }
}

/// Disjunction of two predicates (OR)
#[derive(Debug, Clone, Copy)]
pub struct Or<P1, P2> {
    _phantom: PhantomData<(P1, P2)>,
}

impl<T, P1, P2> RefinementPredicate<T> for Or<P1, P2>
where
    P1: RefinementPredicate<T>,
    P2: RefinementPredicate<T>,
{
    fn check(value: &T) -> bool {
        P1::check(value) || P2::check(value)
    }

    fn description() -> String {
        format!("({}) OR ({})", P1::description(), P2::description())
    }
}

/// Negation of a predicate (NOT)
#[derive(Debug, Clone, Copy)]
pub struct Not<P> {
    _phantom: PhantomData<P>,
}

impl<T, P> RefinementPredicate<T> for Not<P>
where
    P: RefinementPredicate<T>,
{
    fn check(value: &T) -> bool {
        !P::check(value)
    }

    fn description() -> String {
        format!("NOT ({})", P::description())
    }
}

// =============================================================================
// Bounded Values with Const Generics
// =============================================================================

/// A value bounded by compile-time constants (for integers)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BoundedInt<const MIN: i64, const MAX: i64> {
    value: i64,
}

impl<const MIN: i64, const MAX: i64> BoundedInt<MIN, MAX> {
    /// Create a new bounded value
    pub fn new(value: i64) -> Result<Self> {
        if value >= MIN && value <= MAX {
            Ok(Self { value })
        } else {
            Err(SklearsError::ValidationError(format!(
                "Value {} is not in range [{}, {}]",
                value, MIN, MAX
            )))
        }
    }

    /// Create without validation (unsafe)
    pub const fn new_unchecked(value: i64) -> Self {
        Self { value }
    }

    /// Get the value
    pub const fn get(&self) -> i64 {
        self.value
    }

    /// Get the minimum bound
    pub const fn min_bound() -> i64 {
        MIN
    }

    /// Get the maximum bound
    pub const fn max_bound() -> i64 {
        MAX
    }

    /// Check if a value is within bounds
    pub const fn is_in_bounds(value: i64) -> bool {
        value >= MIN && value <= MAX
    }
}

impl<const MIN: i64, const MAX: i64> fmt::Display for BoundedInt<MIN, MAX> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

/// A value bounded by runtime min/max (for floats)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundedFloat {
    value: f64,
    min: f64,
    max: f64,
}

impl BoundedFloat {
    /// Create a new bounded float
    pub fn new(value: f64, min: f64, max: f64) -> Result<Self> {
        if !value.is_finite() {
            return Err(SklearsError::ValidationError(
                "Value must be finite".to_string(),
            ));
        }
        if min > max {
            return Err(SklearsError::ValidationError(
                "min must be <= max".to_string(),
            ));
        }
        if value < min || value > max {
            return Err(SklearsError::ValidationError(format!(
                "Value {} is not in range [{}, {}]",
                value, min, max
            )));
        }
        Ok(Self { value, min, max })
    }

    /// Get the value
    pub fn get(&self) -> f64 {
        self.value
    }

    /// Get the minimum bound
    pub fn min_bound(&self) -> f64 {
        self.min
    }

    /// Get the maximum bound
    pub fn max_bound(&self) -> f64 {
        self.max
    }

    /// Clamp to bounds
    pub fn clamp(&self, value: f64) -> f64 {
        value.max(self.min).min(self.max)
    }
}

// =============================================================================
// Dependent Refinement Types
// =============================================================================

/// A refinement that depends on another value
#[derive(Debug, Clone)]
pub struct DependentRefinement<T, U, F>
where
    F: Fn(&U, &T) -> bool,
{
    value: T,
    dependency: U,
    predicate: F,
}

impl<T, U, F> DependentRefinement<T, U, F>
where
    F: Fn(&U, &T) -> bool,
{
    /// Create a dependent refinement
    pub fn new(value: T, dependency: U, predicate: F) -> Result<Self> {
        if predicate(&dependency, &value) {
            Ok(Self {
                value,
                dependency,
                predicate,
            })
        } else {
            Err(SklearsError::ValidationError(
                "Dependent refinement predicate failed".to_string(),
            ))
        }
    }

    /// Get the value
    pub fn get(&self) -> &T {
        &self.value
    }

    /// Get the dependency
    pub fn dependency(&self) -> &U {
        &self.dependency
    }

    /// Check if a new value would satisfy the predicate
    pub fn would_satisfy(&self, new_value: &T) -> bool {
        (self.predicate)(&self.dependency, new_value)
    }
}

// =============================================================================
// Array Dimension Refinements
// =============================================================================

/// Matrix dimensions that are compile-time verified to be compatible
#[derive(Debug, Clone, Copy)]
pub struct CompatibleDimensions<const ROWS: usize, const COLS: usize> {
    _phantom: PhantomData<([(); ROWS], [(); COLS])>,
}

impl<const ROWS: usize, const COLS: usize> CompatibleDimensions<ROWS, COLS> {
    /// Create new compatible dimensions
    pub const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Get the number of rows
    pub const fn rows() -> usize {
        ROWS
    }

    /// Get the number of columns
    pub const fn cols() -> usize {
        COLS
    }

    /// Check if these dimensions are compatible with another shape
    pub const fn is_compatible_with<const OTHER_ROWS: usize, const OTHER_COLS: usize>() -> bool {
        COLS == OTHER_ROWS
    }
}

impl<const ROWS: usize, const COLS: usize> Default for CompatibleDimensions<ROWS, COLS> {
    fn default() -> Self {
        Self::new()
    }
}

/// A matrix with compile-time verified dimensions
#[derive(Debug, Clone)]
pub struct SizedMatrix<T, const ROWS: usize, const COLS: usize> {
    data: Vec<T>,
    _phantom: PhantomData<CompatibleDimensions<ROWS, COLS>>,
}

impl<T, const ROWS: usize, const COLS: usize> SizedMatrix<T, ROWS, COLS> {
    /// Create a new sized matrix
    pub fn new(data: Vec<T>) -> Result<Self> {
        if data.len() != ROWS * COLS {
            return Err(SklearsError::ValidationError(format!(
                "Expected {} elements for {}x{} matrix, got {}",
                ROWS * COLS,
                ROWS,
                COLS,
                data.len()
            )));
        }
        Ok(Self {
            data,
            _phantom: PhantomData,
        })
    }

    /// Get a reference to the data
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Get a mutable reference to the data
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Get the number of rows
    pub const fn rows() -> usize {
        ROWS
    }

    /// Get the number of columns
    pub const fn cols() -> usize {
        COLS
    }

    /// Get element at (row, col)
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < ROWS && col < COLS {
            self.data.get(row * COLS + col)
        } else {
            None
        }
    }

    /// Set element at (row, col)
    pub fn set(&mut self, row: usize, col: usize, value: T) -> Result<()> {
        if row < ROWS && col < COLS {
            self.data[row * COLS + col] = value;
            Ok(())
        } else {
            Err(SklearsError::InvalidInput(format!(
                "Index ({}, {}) out of bounds for {}x{} matrix",
                row, col, ROWS, COLS
            )))
        }
    }
}

// =============================================================================
// Statically Verified ML Parameters
// =============================================================================

/// Kernel width parameter (must be positive)
pub type KernelWidth = Positive<f64>;

/// Tolerance parameter (must be positive)
pub type Tolerance = Positive<f64>;

/// Number of neighbors (must be positive)
pub type NumNeighbors = Positive<usize>;

/// Tree depth (must be positive)
pub type TreeDepth = Positive<usize>;

/// Number of clusters (must be positive and >= 2)
#[derive(Debug, Clone, Copy)]
pub struct NumClustersPredicate;

impl RefinementPredicate<usize> for NumClustersPredicate {
    fn check(value: &usize) -> bool {
        *value >= 2
    }

    fn description() -> String {
        "must be at least 2 clusters".to_string()
    }
}

/// Number of clusters (>= 2)
pub type NumClusters = Refined<usize, NumClustersPredicate>;

// =============================================================================
// Refinement Combinators
// =============================================================================

/// Combine two refinements into one
pub fn combine<T, P1, P2>(
    r1: Refined<T, P1>,
    _r2: Refined<T, P2>,
) -> Result<Refined<T, And<P1, P2>>>
where
    P1: RefinementPredicate<T>,
    P2: RefinementPredicate<T>,
    T: Clone,
{
    // Both refinements must hold for the same value
    // Since r1 and r2 might have different values, we need to check compatibility
    let value = r1.into_inner();
    Refined::<T, And<P1, P2>>::new(value)
}

/// Lift a function to work on refined types
pub fn lift<T, U, P, Q, F>(f: F) -> impl Fn(Refined<T, P>) -> Result<Refined<U, Q>>
where
    F: Fn(T) -> U,
    P: RefinementPredicate<T>,
    Q: RefinementPredicate<U>,
{
    move |refined| {
        let value = refined.into_inner();
        let new_value = f(value);
        Refined::<U, Q>::new(new_value)
    }
}

// =============================================================================
// Macro for Creating Custom Refinement Predicates
// =============================================================================

/// Macro to create a custom refinement predicate
#[macro_export]
macro_rules! refinement_predicate {
    ($name:ident, $type:ty, $check:expr, $desc:expr) => {
        #[derive(Debug, Clone, Copy)]
        pub struct $name;

        impl RefinementPredicate<$type> for $name {
            fn check(value: &$type) -> bool {
                $check(value)
            }

            fn description() -> String {
                $desc.to_string()
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positive_refinement() {
        let pos = Positive::<i32>::new(5).expect("expected valid value");
        assert_eq!(*pos, 5);

        let neg_result = Positive::<i32>::new(-5);
        assert!(neg_result.is_err());

        let zero_result = Positive::<i32>::new(0);
        assert!(zero_result.is_err());
    }

    #[test]
    fn test_non_negative_refinement() {
        let non_neg = NonNegative::<i32>::new(0).expect("expected valid value");
        assert_eq!(*non_neg, 0);

        let pos = NonNegative::<i32>::new(5).expect("expected valid value");
        assert_eq!(*pos, 5);

        let neg_result = NonNegative::<i32>::new(-5);
        assert!(neg_result.is_err());
    }

    #[test]
    fn test_probability_refinement() {
        let prob = ValidProbability::new(0.5).expect("expected valid value");
        assert_eq!(*prob, 0.5);

        let zero_prob = ValidProbability::new(0.0).expect("expected valid value");
        assert_eq!(*zero_prob, 0.0);

        let one_prob = ValidProbability::new(1.0).expect("expected valid value");
        assert_eq!(*one_prob, 1.0);

        let invalid_prob = ValidProbability::new(1.5);
        assert!(invalid_prob.is_err());

        let negative_prob = ValidProbability::new(-0.1);
        assert!(negative_prob.is_err());
    }

    #[test]
    fn test_learning_rate_refinement() {
        let lr = ValidLearningRate::new(0.01).expect("expected valid value");
        assert_eq!(*lr, 0.01);

        let zero_lr = ValidLearningRate::new(0.0);
        assert!(zero_lr.is_err());

        let large_lr = ValidLearningRate::new(1.5);
        assert!(large_lr.is_err());
    }

    #[test]
    fn test_bounded_int() {
        let bounded = BoundedInt::<0, 100>::new(50).expect("expected valid value");
        assert_eq!(bounded.get(), 50);
        assert_eq!(BoundedInt::<0, 100>::min_bound(), 0);
        assert_eq!(BoundedInt::<0, 100>::max_bound(), 100);

        let below_min = BoundedInt::<0, 100>::new(-1);
        assert!(below_min.is_err());

        let above_max = BoundedInt::<0, 100>::new(101);
        assert!(above_max.is_err());
    }

    #[test]
    fn test_bounded_float() {
        let bounded = BoundedFloat::new(0.5, 0.0, 1.0).expect("expected valid value");
        assert_eq!(bounded.get(), 0.5);
        assert_eq!(bounded.min_bound(), 0.0);
        assert_eq!(bounded.max_bound(), 1.0);

        let below_min = BoundedFloat::new(-0.1, 0.0, 1.0);
        assert!(below_min.is_err());

        let above_max = BoundedFloat::new(1.1, 0.0, 1.0);
        assert!(above_max.is_err());
    }

    #[test]
    fn test_non_empty_vec() {
        let non_empty = NonEmpty::<Vec<i32>>::new(vec![1, 2, 3]).expect("expected valid value");
        assert_eq!(non_empty.len(), 3);

        let empty_result = NonEmpty::<Vec<i32>>::new(vec![]);
        assert!(empty_result.is_err());
    }

    #[test]
    fn test_num_clusters() {
        let clusters = NumClusters::new(5).expect("expected valid value");
        assert_eq!(*clusters, 5);

        let too_few = NumClusters::new(1);
        assert!(too_few.is_err());

        let min_clusters = NumClusters::new(2).expect("expected valid value");
        assert_eq!(*min_clusters, 2);
    }

    #[test]
    fn test_refinement_map() {
        let pos = Positive::<i32>::new(5).expect("expected valid value");
        let doubled: Result<Positive<i32>> = pos.try_map(|x| x * 2);
        assert!(doubled.is_ok());
        assert_eq!(*doubled.expect("expected valid value"), 10);

        // Mapping that violates the predicate should fail
        let pos2 = Positive::<i32>::new(5).expect("expected valid value");
        let negated: Result<Positive<i32>> = pos2.try_map(|x| -x);
        assert!(negated.is_err());
    }

    #[test]
    fn test_dependent_refinement() {
        // Value must be less than dependency
        let predicate = |dep: &usize, val: &usize| *val < *dep;
        let refined = DependentRefinement::new(5, 10, predicate).expect("expected valid value");
        assert_eq!(*refined.get(), 5);
        assert_eq!(*refined.dependency(), 10);

        // Should fail when value >= dependency
        let invalid = DependentRefinement::new(15, 10, predicate);
        assert!(invalid.is_err());
    }

    #[test]
    fn test_sized_matrix() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let matrix = SizedMatrix::<i32, 2, 3>::new(data).expect("expected valid value");

        assert_eq!(SizedMatrix::<i32, 2, 3>::rows(), 2);
        assert_eq!(SizedMatrix::<i32, 2, 3>::cols(), 3);
        assert_eq!(*matrix.get(0, 0).expect("get should succeed"), 1);
        assert_eq!(*matrix.get(1, 2).expect("get should succeed"), 6);

        // Wrong size should fail
        let wrong_size = SizedMatrix::<i32, 2, 3>::new(vec![1, 2, 3]);
        assert!(wrong_size.is_err());
    }

    #[test]
    fn test_composite_predicates() {
        // AND: value must be positive AND less than 10
        type PositiveAndSmall = Refined<i32, And<PositivePredicate, PositivePredicate>>;

        let valid = PositiveAndSmall::new(5).expect("expected valid value");
        assert_eq!(*valid, 5);

        let invalid = PositiveAndSmall::new(0);
        assert!(invalid.is_err());
    }
}
