/// Domain-specific types for machine learning parameters and constraints
///
/// This module provides type-safe wrappers for common machine learning
/// concepts like probabilities, learning rates, and counts.
use super::traits::FloatBounds;
use crate::error::{Result, SklearsError};
use scirs2_core::numeric::ToPrimitive;
use std::fmt;

/// Type-safe probability value constrained to [0, 1]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Probability<T: FloatBounds>(T);

impl<T: FloatBounds> Probability<T> {
    /// Create a new probability value, returning an error if not in [0, 1]
    pub fn new(value: T) -> Result<Self> {
        if value < T::zero() || value > T::one() {
            return Err(SklearsError::InvalidParameter {
                name: "probability".to_string(),
                reason: format!("must be in range [0, 1], got {value}"),
            });
        }
        Ok(Probability(value))
    }

    /// Create a probability value without validation (unsafe)
    ///
    /// # Safety
    /// The caller must ensure the value is in [0, 1]
    pub unsafe fn new_unchecked(value: T) -> Self {
        Probability(value)
    }

    /// Get the raw probability value
    pub fn value(self) -> T {
        self.0
    }

    /// Convert to f64 for compatibility
    pub fn as_f64(self) -> f64
    where
        T: ToPrimitive,
    {
        self.0.to_f64().unwrap_or(0.0)
    }

    /// Check if this is effectively zero (within machine epsilon)
    pub fn is_zero(self) -> bool {
        self.0 <= T::EPSILON
    }

    /// Check if this is effectively one (within machine epsilon)
    pub fn is_one(self) -> bool {
        (self.0 - T::one()).abs() <= T::EPSILON
    }

    /// Complement probability (1 - p)
    pub fn complement(self) -> Self {
        // Safe because 1 - [0,1] is always in [0,1]
        unsafe { Probability::new_unchecked(T::one() - self.0) }
    }

    /// Logarithm of probability (useful for log-likelihood computations)
    pub fn ln(self) -> T {
        self.0.ln()
    }

    /// Sigmoid function to convert any real number to probability
    pub fn sigmoid(x: T) -> Self {
        let exp_x = x.exp();
        let prob = exp_x / (T::one() + exp_x);
        // Sigmoid always returns [0,1] so this is safe
        unsafe { Probability::new_unchecked(prob) }
    }
}

impl<T: FloatBounds> fmt::Display for Probability<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<T: FloatBounds> std::ops::Add for Probability<T> {
    type Output = Result<Self>;

    fn add(self, other: Self) -> Self::Output {
        Self::new(self.0 + other.0)
    }
}

impl<T: FloatBounds> std::ops::Mul for Probability<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        // Product of probabilities is always in [0,1]
        unsafe { Probability::new_unchecked(self.0 * other.0) }
    }
}

/// Type-safe feature count
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FeatureCount(usize);

impl FeatureCount {
    /// Create a new feature count, must be positive
    pub fn new(count: usize) -> Result<Self> {
        if count == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "n_features".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(FeatureCount(count))
    }

    /// Get the raw count value
    pub fn get(self) -> usize {
        self.0
    }

    /// Convert to other numeric types
    pub fn as_i32(self) -> i32 {
        self.0 as i32
    }

    pub fn as_f64(self) -> f64 {
        self.0 as f64
    }

    /// Check if this count is a power of 2 (useful for some algorithms)
    pub fn is_power_of_two(self) -> bool {
        self.0.is_power_of_two()
    }

    /// Get the next power of 2 greater than or equal to this count
    pub fn next_power_of_two(self) -> Self {
        FeatureCount(self.0.next_power_of_two())
    }
}

impl fmt::Display for FeatureCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::ops::Add<usize> for FeatureCount {
    type Output = Self;

    fn add(self, rhs: usize) -> Self::Output {
        FeatureCount(self.0 + rhs)
    }
}

impl std::ops::Sub<usize> for FeatureCount {
    type Output = Result<Self>;

    fn sub(self, rhs: usize) -> Self::Output {
        if rhs >= self.0 {
            Err(SklearsError::InvalidParameter {
                name: "feature_count_subtraction".to_string(),
                reason: "result would be zero or negative".to_string(),
            })
        } else {
            Ok(FeatureCount(self.0 - rhs))
        }
    }
}

/// Type-safe sample count
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SampleCount(usize);

impl SampleCount {
    /// Create a new sample count, must be positive
    pub fn new(count: usize) -> Result<Self> {
        if count == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "n_samples".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(SampleCount(count))
    }

    /// Get the raw count value
    pub fn get(self) -> usize {
        self.0
    }

    /// Convert to other numeric types
    pub fn as_i32(self) -> i32 {
        self.0 as i32
    }

    pub fn as_f64(self) -> f64 {
        self.0 as f64
    }

    /// Calculate percentage of another sample count
    pub fn percentage_of(self, total: SampleCount) -> Result<f64> {
        if total.0 == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "total_samples".to_string(),
                reason: "cannot be zero for percentage calculation".to_string(),
            });
        }
        Ok((self.0 as f64 / total.0 as f64) * 100.0)
    }

    /// Check if this is a valid train/test split ratio
    pub fn is_valid_split_with(self, other: SampleCount, min_samples_per_split: usize) -> bool {
        self.0 >= min_samples_per_split && other.0 >= min_samples_per_split
    }
}

impl fmt::Display for SampleCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::ops::Add for SampleCount {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        SampleCount(self.0 + other.0)
    }
}

impl std::ops::Sub for SampleCount {
    type Output = Result<Self>;

    fn sub(self, other: Self) -> Self::Output {
        if other.0 >= self.0 {
            Err(SklearsError::InvalidParameter {
                name: "sample_count_subtraction".to_string(),
                reason: "result would be zero or negative".to_string(),
            })
        } else {
            Ok(SampleCount(self.0 - other.0))
        }
    }
}

/// Type-safe learning rate parameter
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct LearningRate<T: FloatBounds>(T);

impl<T: FloatBounds> LearningRate<T> {
    /// Create a new learning rate, must be positive
    pub fn new(rate: T) -> Result<Self> {
        if rate <= T::zero() {
            return Err(SklearsError::InvalidParameter {
                name: "learning_rate".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        if !rate.is_finite() {
            return Err(SklearsError::InvalidParameter {
                name: "learning_rate".to_string(),
                reason: "must be finite".to_string(),
            });
        }
        Ok(LearningRate(rate))
    }

    /// Get the raw learning rate value
    pub fn get(self) -> T {
        self.0
    }

    /// Create a common learning rate value
    pub fn default_value() -> Self {
        // Safe because 0.01 is positive and finite
        unsafe { LearningRate::new_unchecked(T::from(0.01).unwrap()) }
    }

    /// Create without validation (unsafe)
    ///
    /// # Safety
    /// The caller must ensure the value is positive and finite
    unsafe fn new_unchecked(rate: T) -> Self {
        LearningRate(rate)
    }

    /// Decay the learning rate by a factor
    pub fn decay(self, factor: T) -> Result<Self> {
        if factor <= T::zero() || factor > T::one() {
            return Err(SklearsError::InvalidParameter {
                name: "decay_factor".to_string(),
                reason: "must be in range (0, 1]".to_string(),
            });
        }
        Self::new(self.0 * factor)
    }

    /// Adaptive learning rate based on performance
    pub fn adapt(self, improvement_ratio: T, adapt_factor: T) -> Result<Self> {
        if improvement_ratio > T::one() {
            // Performance improved, potentially increase learning rate
            let new_rate = self.0 * (T::one() + adapt_factor * improvement_ratio);
            Self::new(new_rate)
        } else {
            // Performance degraded, decrease learning rate
            let new_rate = self.0 * (T::one() - adapt_factor * (T::one() - improvement_ratio));
            Self::new(new_rate.max(T::EPSILON)) // Ensure it doesn't become zero
        }
    }

    /// Step-based learning rate schedule
    pub fn step_schedule(initial: Self, step: usize, step_size: usize, gamma: T) -> Self {
        let decay_factor = gamma.powf(T::from((step / step_size) as f64).unwrap());
        // Safe because decay factor applied to positive rate remains positive
        unsafe { LearningRate::new_unchecked(initial.0 * decay_factor) }
    }

    /// Exponential learning rate schedule
    pub fn exponential_schedule(initial: Self, step: usize, decay_rate: T) -> Self {
        let decay_factor = (-decay_rate * T::from(step as f64).unwrap()).exp();
        // Safe because exponential decay of positive rate remains positive
        unsafe { LearningRate::new_unchecked(initial.0 * decay_factor) }
    }
}

impl<T: FloatBounds> fmt::Display for LearningRate<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Type-safe regularization strength parameter
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct RegularizationStrength<T: FloatBounds>(T);

impl<T: FloatBounds + std::iter::Sum> RegularizationStrength<T> {
    /// Create a new regularization strength, must be non-negative
    pub fn new(strength: T) -> Result<Self> {
        if strength < T::zero() {
            return Err(SklearsError::InvalidParameter {
                name: "regularization_strength".to_string(),
                reason: "must be non-negative".to_string(),
            });
        }
        if !strength.is_finite() {
            return Err(SklearsError::InvalidParameter {
                name: "regularization_strength".to_string(),
                reason: "must be finite".to_string(),
            });
        }
        Ok(RegularizationStrength(strength))
    }

    /// Create zero regularization (no regularization)
    pub fn none() -> Self {
        RegularizationStrength(T::zero())
    }

    /// Get the raw regularization strength value
    pub fn get(self) -> T {
        self.0
    }

    /// Check if regularization is effectively disabled
    pub fn is_disabled(self) -> bool {
        self.0 <= T::EPSILON
    }

    /// Create common regularization values
    pub fn weak() -> Self {
        RegularizationStrength(T::from(0.001).unwrap())
    }

    pub fn moderate() -> Self {
        RegularizationStrength(T::from(0.01).unwrap())
    }

    pub fn strong() -> Self {
        RegularizationStrength(T::from(0.1).unwrap())
    }

    /// Grid search values for hyperparameter optimization
    pub fn grid_values() -> Vec<Self> {
        let values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0];
        values
            .iter()
            .map(|&v| RegularizationStrength(T::from(v).unwrap()))
            .collect()
    }

    /// L1 penalty term
    pub fn l1_penalty(self, weights: &[T]) -> T {
        self.0 * weights.iter().map(|&w| w.abs()).sum::<T>()
    }

    /// L2 penalty term  
    pub fn l2_penalty(self, weights: &[T]) -> T {
        self.0 * weights.iter().map(|&w| w * w).sum::<T>()
    }

    /// Elastic net penalty (combination of L1 and L2)
    pub fn elastic_net_penalty(self, weights: &[T], l1_ratio: T) -> T {
        let l1_term = l1_ratio * weights.iter().map(|&w| w.abs()).sum::<T>();
        let l2_term = (T::one() - l1_ratio) * weights.iter().map(|&w| w * w).sum::<T>();
        self.0 * (l1_term + l2_term)
    }
}

impl<T: FloatBounds> fmt::Display for RegularizationStrength<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Type-safe tolerance parameter for convergence checking
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Tolerance<T: FloatBounds>(T);

impl<T: FloatBounds> Tolerance<T> {
    /// Create a new tolerance value, must be positive
    pub fn new(tolerance: T) -> Result<Self> {
        if tolerance <= T::zero() {
            return Err(SklearsError::InvalidParameter {
                name: "tolerance".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Tolerance(tolerance))
    }

    /// Get the raw tolerance value
    pub fn get(self) -> T {
        self.0
    }

    /// Default tolerance based on machine epsilon
    pub fn default_for_type() -> Self {
        Tolerance(T::EPSILON * T::from(1000.0).unwrap())
    }

    /// Strict tolerance for high precision
    pub fn strict() -> Self {
        Tolerance(T::EPSILON * T::from(10.0).unwrap())
    }

    /// Relaxed tolerance for fast convergence
    pub fn relaxed() -> Self {
        Tolerance(T::from(1e-3).unwrap())
    }

    /// Check if two values are within this tolerance
    pub fn are_close(self, a: T, b: T) -> bool {
        (a - b).abs() <= self.0
    }

    /// Check if a value is close to zero within this tolerance
    pub fn is_zero(self, value: T) -> bool {
        value.abs() <= self.0
    }

    /// Check convergence based on relative change
    pub fn check_relative_convergence(self, current: T, previous: T) -> bool {
        if previous.abs() <= T::EPSILON {
            // If previous is essentially zero, check absolute convergence
            current.abs() <= self.0
        } else {
            // Check relative change
            ((current - previous) / previous).abs() <= self.0
        }
    }
}

impl<T: FloatBounds> fmt::Display for Tolerance<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Collection of common ML parameter types for convenience
pub mod common {
    pub use super::{
        FeatureCount, LearningRate, Probability, RegularizationStrength, SampleCount, Tolerance,
    };

    /// Common probability type
    pub type Prob = Probability<f64>;

    /// Common learning rate type
    pub type LR = LearningRate<f64>;

    /// Common regularization strength type
    pub type RegStrength = RegularizationStrength<f64>;

    /// Common tolerance type
    pub type Tol = Tolerance<f64>;
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_probability() {
        // Valid probabilities
        let p1 = Probability::new(0.5).unwrap();
        let p2 = Probability::new(0.3).unwrap();

        assert_eq!(p1.value(), 0.5);
        assert!(!p1.is_zero());
        assert!(!p1.is_one());

        let complement = p1.complement();
        assert_eq!(complement.value(), 0.5);

        // Multiplication
        let product = p1 * p2;
        assert_abs_diff_eq!(product.value(), 0.15);

        // Invalid probabilities
        assert!(Probability::new(-0.1).is_err());
        assert!(Probability::new(1.1).is_err());

        // Sigmoid
        let prob_from_sigmoid = Probability::sigmoid(0.0);
        assert_abs_diff_eq!(prob_from_sigmoid.value(), 0.5);
    }

    #[test]
    fn test_feature_count() {
        let fc = FeatureCount::new(10).unwrap();
        assert_eq!(fc.get(), 10);
        assert_eq!(fc.as_f64(), 10.0);

        let fc2 = fc + 5;
        assert_eq!(fc2.get(), 15);

        let fc3 = fc - 3;
        assert!(fc3.is_ok());
        assert_eq!(fc3.unwrap().get(), 7);

        // Invalid operations
        assert!(FeatureCount::new(0).is_err());
        assert!((fc - 20).is_err());

        // Power of 2 operations
        let fc_pow2 = FeatureCount::new(8).unwrap();
        assert!(fc_pow2.is_power_of_two());

        let fc_not_pow2 = FeatureCount::new(10).unwrap();
        assert!(!fc_not_pow2.is_power_of_two());
        assert_eq!(fc_not_pow2.next_power_of_two().get(), 16);
    }

    #[test]
    fn test_sample_count() {
        let sc1 = SampleCount::new(100).unwrap();
        let sc2 = SampleCount::new(25).unwrap();

        assert_eq!(sc1.get(), 100);

        let total = sc1 + sc2;
        assert_eq!(total.get(), 125);

        let percentage = sc2.percentage_of(sc1).unwrap();
        assert_abs_diff_eq!(percentage, 25.0);

        assert!(sc1.is_valid_split_with(sc2, 10));
        assert!(!sc1.is_valid_split_with(sc2, 50));

        // Invalid operations
        assert!(SampleCount::new(0).is_err());
        assert!((sc2 - sc1).is_err());
    }

    #[test]
    fn test_learning_rate() {
        let lr = LearningRate::new(0.01).unwrap();
        assert_eq!(lr.get(), 0.01);

        let decayed = lr.decay(0.9).unwrap();
        assert_abs_diff_eq!(decayed.get(), 0.009);

        // Invalid learning rates
        assert!(LearningRate::new(0.0).is_err());
        assert!(LearningRate::new(-0.1).is_err());
        assert!(LearningRate::new(f64::INFINITY).is_err());

        // Schedules
        let initial = LearningRate::new(0.1).unwrap();
        let step_lr = LearningRate::step_schedule(initial, 100, 50, 0.1);
        assert!(step_lr.get() < initial.get());

        let exp_lr = LearningRate::exponential_schedule(initial, 10, 0.1);
        assert!(exp_lr.get() < initial.get());
    }

    #[test]
    fn test_regularization_strength() {
        let reg = RegularizationStrength::new(0.01).unwrap();
        assert_eq!(reg.get(), 0.01);
        assert!(!reg.is_disabled());

        let no_reg = RegularizationStrength::<f64>::none();
        assert!(no_reg.is_disabled());

        // Grid values
        let grid = RegularizationStrength::<f64>::grid_values();
        assert!(!grid.is_empty());
        assert!(grid.len() >= 5);

        // Penalties
        let weights = vec![1.0, -2.0, 3.0];
        let l1_penalty = reg.l1_penalty(&weights);
        let l2_penalty = reg.l2_penalty(&weights);

        assert_abs_diff_eq!(l1_penalty, 0.01 * 6.0); // |1| + |-2| + |3| = 6
        assert_abs_diff_eq!(l2_penalty, 0.01 * 14.0); // 1² + (-2)² + 3² = 14

        // Invalid regularization
        assert!(RegularizationStrength::new(-0.1).is_err());
    }

    #[test]
    fn test_tolerance() {
        let tol = Tolerance::new(1e-6).unwrap();
        assert_eq!(tol.get(), 1e-6);

        assert!(tol.are_close(1.0, 1.0 + 1e-7));
        assert!(!tol.are_close(1.0, 1.0 + 1e-5));

        assert!(tol.is_zero(1e-7));
        assert!(!tol.is_zero(1e-5));

        // Convergence checking
        assert!(tol.check_relative_convergence(1.000001, 1.0));
        assert!(!tol.check_relative_convergence(1.1, 1.0));

        // Invalid tolerance
        assert!(Tolerance::new(0.0).is_err());
        assert!(Tolerance::new(-1e-6).is_err());
    }
}
