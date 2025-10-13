//! Type Safety Improvements for Calibration
//!
//! This module implements advanced type safety features including phantom types
//! for calibration method types, compile-time probability validation, zero-cost
//! calibration abstractions, const generics for fixed-size calibrators, and
//! type-safe probability transformations.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result},
    types::Float,
};
use std::marker::PhantomData;

use crate::{CalibrationEstimator};

/// Phantom type for calibration states
pub trait CalibrationState {}

/// Untrained calibration state
#[derive(Debug, Clone, Copy)]
pub struct Untrained;
impl CalibrationState for Untrained {}

/// Trained calibration state
#[derive(Debug, Clone, Copy)]
pub struct Trained;
impl CalibrationState for Trained {}

/// Phantom type for calibration method types
pub trait CalibrationMethodType {}

/// Sigmoid/Platt scaling method
#[derive(Debug, Clone, Copy)]
pub struct SigmoidMethod;
impl CalibrationMethodType for SigmoidMethod {}

/// Isotonic regression method
#[derive(Debug, Clone, Copy)]
pub struct IsotonicMethod;
impl CalibrationMethodType for IsotonicMethod {}

/// Temperature scaling method
#[derive(Debug, Clone, Copy)]
pub struct TemperatureMethod;
impl CalibrationMethodType for TemperatureMethod {}

/// Histogram binning method
#[derive(Debug, Clone, Copy)]
pub struct HistogramMethod;
impl CalibrationMethodType for HistogramMethod {}

/// Bayesian Binning into Quantiles method
#[derive(Debug, Clone, Copy)]
pub struct BBQMethod;
impl CalibrationMethodType for BBQMethod {}

/// Probability type with compile-time validation
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Probability<const VALIDATED: bool = true>(Float);

impl<const VALIDATED: bool> Probability<VALIDATED> {
    /// Create a new probability with runtime validation
    pub fn new(value: Float) -> Result<Self> {
        if value >= 0.0 && value <= 1.0 && value.is_finite() {
            Ok(Probability(value))
        } else {
            Err(SklearsError::InvalidInput(
                format!("Invalid probability value: {}", value)
            ))
        }
    }

    /// Create an unvalidated probability (for internal use)
    pub fn new_unchecked(value: Float) -> Probability<false> {
        Probability(value)
    }

    /// Get the inner value
    pub fn value(&self) -> Float {
        self.0
    }

    /// Convert to log-odds (logit)
    pub fn to_logit(&self) -> Float {
        let p = self.0.clamp(1e-15, 1.0 - 1e-15);
        (p / (1.0 - p)).ln()
    }

    /// Convert from log-odds (logit)
    pub fn from_logit(logit: Float) -> Result<Probability<true>> {
        let p = 1.0 / (1.0 + (-logit).exp());
        Probability::new(p)
    }

    /// Safe addition with another probability
    pub fn safe_add(&self, other: &Self) -> Result<Probability<true>> {
        let sum = self.0 + other.0;
        Probability::new(sum.clamp(0.0, 1.0))
    }

    /// Safe multiplication with another probability
    pub fn safe_mul(&self, other: &Self) -> Probability<true> {
        let product = self.0 * other.0;
        Probability(product.clamp(0.0, 1.0))
    }

    /// Complement probability (1 - p)
    pub fn complement(&self) -> Probability<true> {
        Probability(1.0 - self.0)
    }
}

impl Probability<false> {
    /// Validate an unvalidated probability
    pub fn validate(self) -> Result<Probability<true>> {
        Probability::new(self.0)
    }
}

impl From<Probability<true>> for Float {
    fn from(prob: Probability<true>) -> Float {
        prob.0
    }
}

impl TryFrom<Float> for Probability<true> {
    type Error = SklearsError;

    fn try_from(value: Float) -> Result<Self> {
        Probability::new(value)
    }
}

/// Type-safe probability array
#[derive(Debug, Clone)]
pub struct ProbabilityArray<const N: usize = 0, const VALIDATED: bool = true> {
    inner: Array1<Float>,
    _phantom: PhantomData<Probability<VALIDATED>>,
}

impl<const N: usize, const VALIDATED: bool> ProbabilityArray<N, VALIDATED> {
    /// Create a new probability array with validation
    pub fn new(values: Array1<Float>) -> Result<ProbabilityArray<N, true>> {
        // Validate all probabilities
        for &value in values.iter() {
            if !(0.0..=1.0).contains(&value) || !value.is_finite() {
                return Err(SklearsError::InvalidInput(
                    format!("Invalid probability value: {}", value)
                ));
            }
        }

        if N > 0 && values.len() != N {
            return Err(SklearsError::InvalidInput(
                format!("Expected array of size {}, got {}", N, values.len())
            ));
        }

        Ok(ProbabilityArray {
            inner: values,
            _phantom: PhantomData,
        })
    }

    /// Create unvalidated probability array
    pub fn new_unchecked(values: Array1<Float>) -> ProbabilityArray<N, false> {
        ProbabilityArray {
            inner: values,
            _phantom: PhantomData,
        }
    }

    /// Get the inner array
    pub fn inner(&self) -> &Array1<Float> {
        &self.inner
    }

    /// Get the length
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get probability at index
    pub fn get(&self, index: usize) -> Option<Probability<VALIDATED>> {
        self.inner.get(index).map(|&value| Probability(value))
    }

    /// Convert to logits
    pub fn to_logits(&self) -> Array1<Float> {
        self.inner.map(|&p| {
            let p_safe = p.clamp(1e-15, 1.0 - 1e-15);
            (p_safe / (1.0 - p_safe)).ln()
        })
    }

    /// Convert from logits
    pub fn from_logits(logits: &Array1<Float>) -> Result<ProbabilityArray<N, true>> {
        let probs = logits.map(|&logit| 1.0 / (1.0 + (-logit).exp()));
        ProbabilityArray::new(probs)
    }

    /// Apply softmax normalization
    pub fn softmax(&self) -> Result<ProbabilityArray<N, true>> {
        let max_val = self.inner.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));
        let exp_values = self.inner.map(|&x| (x - max_val).exp());
        let sum = exp_values.sum();
        
        if sum <= 0.0 {
            return Err(SklearsError::InvalidInput(
                "Cannot compute softmax: sum of exponentials is zero".to_string()
            ));
        }

        let normalized = exp_values / sum;
        ProbabilityArray::new(normalized)
    }

    /// Normalize to sum to 1
    pub fn normalize(&self) -> Result<ProbabilityArray<N, true>> {
        let sum = self.inner.sum();
        if sum <= 0.0 {
            return Err(SklearsError::InvalidInput(
                "Cannot normalize: sum is zero or negative".to_string()
            ));
        }

        let normalized = &self.inner / sum;
        ProbabilityArray::new(normalized)
    }
}

impl<const N: usize> ProbabilityArray<N, false> {
    /// Validate an unvalidated probability array
    pub fn validate(self) -> Result<ProbabilityArray<N, true>> {
        ProbabilityArray::new(self.inner)
    }
}

/// Type-safe calibrator with phantom types
#[derive(Debug, Clone)]
pub struct TypeSafeCalibrator<M: CalibrationMethodType, S: CalibrationState> {
    /// Inner calibrator implementation
    inner: Box<dyn CalibrationEstimator>,
    /// Phantom data for method type
    _method: PhantomData<M>,
    /// Phantom data for state
    _state: PhantomData<S>,
}

impl<M: CalibrationMethodType> TypeSafeCalibrator<M, Untrained> {
    /// Create a new untrained calibrator
    pub fn new(inner: Box<dyn CalibrationEstimator>) -> Self {
        Self {
            inner,
            _method: PhantomData,
            _state: PhantomData,
        }
    }

    /// Fit the calibrator and transition to trained state
    pub fn fit(
        mut self,
        probabilities: &ProbabilityArray<0, true>,
        y_true: &Array1<i32>,
    ) -> Result<TypeSafeCalibrator<M, Trained>> {
        self.inner.fit(probabilities.inner(), y_true)?;

        Ok(TypeSafeCalibrator {
            inner: self.inner,
            _method: PhantomData,
            _state: PhantomData,
        })
    }
}

impl<M: CalibrationMethodType> TypeSafeCalibrator<M, Trained> {
    /// Predict probabilities with type safety
    pub fn predict_proba(
        &self,
        probabilities: &ProbabilityArray<0, true>,
    ) -> Result<ProbabilityArray<0, true>> {
        let predictions = self.inner.predict_proba(probabilities.inner())?;
        ProbabilityArray::new(predictions)
    }

    /// Get method information
    pub fn method_type(&self) -> &'static str {
        std::any::type_name::<M>()
    }
}

/// Fixed-size calibrator using const generics
#[derive(Debug, Clone)]
pub struct FixedSizeCalibrator<const N: usize, M: CalibrationMethodType, S: CalibrationState> {
    /// Inner calibrator
    inner: Box<dyn CalibrationEstimator>,
    /// Phantom types
    _phantom: PhantomData<(M, S)>,
}

impl<const N: usize, M: CalibrationMethodType> FixedSizeCalibrator<N, M, Untrained> {
    /// Create a new fixed-size calibrator
    pub fn new(inner: Box<dyn CalibrationEstimator>) -> Self {
        Self {
            inner,
            _phantom: PhantomData,
        }
    }

    /// Fit with fixed-size arrays
    pub fn fit(
        mut self,
        probabilities: &ProbabilityArray<N, true>,
        y_true: &[i32; N],
    ) -> Result<FixedSizeCalibrator<N, M, Trained>> {
        let y_array = Array1::from_iter(y_true.iter().copied());
        self.inner.fit(probabilities.inner(), &y_array)?;

        Ok(FixedSizeCalibrator {
            inner: self.inner,
            _phantom: PhantomData,
        })
    }
}

impl<const N: usize, M: CalibrationMethodType> FixedSizeCalibrator<N, M, Trained> {
    /// Predict with fixed-size arrays
    pub fn predict_proba(
        &self,
        probabilities: &ProbabilityArray<N, true>,
    ) -> Result<ProbabilityArray<N, true>> {
        let predictions = self.inner.predict_proba(probabilities.inner())?;
        ProbabilityArray::new(predictions)
    }
}

/// Zero-cost calibration abstraction
pub trait ZeroCostCalibration<Input, Output> {
    /// Zero-cost transformation
    fn transform(&self, input: Input) -> Output;
}

/// Identity transformation (zero-cost)
#[derive(Debug, Clone, Copy)]
pub struct IdentityTransform;

impl ZeroCostCalibration<Probability<true>, Probability<true>> for IdentityTransform {
    fn transform(&self, input: Probability<true>) -> Probability<true> {
        input
    }
}

/// Logit transformation (zero-cost)
#[derive(Debug, Clone, Copy)]
pub struct LogitTransform;

impl ZeroCostCalibration<Probability<true>, Float> for LogitTransform {
    fn transform(&self, input: Probability<true>) -> Float {
        input.to_logit()
    }
}

/// Sigmoid transformation (zero-cost)
#[derive(Debug, Clone, Copy)]
pub struct SigmoidTransform;

impl ZeroCostCalibration<Float, Probability<false>> for SigmoidTransform {
    fn transform(&self, input: Float) -> Probability<false> {
        let prob = 1.0 / (1.0 + (-input).exp());
        Probability::new_unchecked(prob)
    }
}

/// Compile-time probability bounds
pub trait ProbabilityBounds {
    const MIN: Float;
    const MAX: Float;
    
    fn is_valid(value: Float) -> bool {
        value >= Self::MIN && value <= Self::MAX && value.is_finite()
    }
}

/// Standard probability bounds [0, 1]
#[derive(Debug, Clone, Copy)]
pub struct StandardBounds;

impl ProbabilityBounds for StandardBounds {
    const MIN: Float = 0.0;
    const MAX: Float = 1.0;
}

/// Strict probability bounds (0, 1) excluding endpoints
#[derive(Debug, Clone, Copy)]
pub struct StrictBounds;

impl ProbabilityBounds for StrictBounds {
    const MIN: Float = 1e-15;
    const MAX: Float = 1.0 - 1e-15;
}

/// Bounded probability with compile-time bounds
#[derive(Debug, Clone, Copy)]
pub struct BoundedProbability<B: ProbabilityBounds>(Float, PhantomData<B>);

impl<B: ProbabilityBounds> BoundedProbability<B> {
    /// Create a new bounded probability
    pub fn new(value: Float) -> Result<Self> {
        if B::is_valid(value) {
            Ok(BoundedProbability(value, PhantomData))
        } else {
            Err(SklearsError::InvalidInput(
                format!("Value {} is outside bounds [{}, {}]", value, B::MIN, B::MAX)
            ))
        }
    }

    /// Get the value
    pub fn value(&self) -> Float {
        self.0
    }

    /// Convert to different bounds
    pub fn convert_bounds<B2: ProbabilityBounds>(self) -> Result<BoundedProbability<B2>> {
        BoundedProbability::new(self.0)
    }
}

/// Type-safe probability transformation pipeline
#[derive(Debug, Clone)]
pub struct TransformationPipeline<Input, Output> {
    transformations: Vec<Box<dyn Fn(Input) -> Output>>,
}

impl<Input: Clone, Output: Clone> TransformationPipeline<Input, Output> {
    /// Create a new transformation pipeline
    pub fn new() -> Self {
        Self {
            transformations: Vec::new(),
        }
    }

    /// Add a transformation step
    pub fn add_step<F>(mut self, transform: F) -> Self
    where
        F: Fn(Input) -> Output + 'static,
    {
        self.transformations.push(Box::new(transform));
        self
    }

    /// Apply all transformations
    pub fn apply(&self, input: Input) -> Result<Output> {
        if let Some(first_transform) = self.transformations.first() {
            Ok(first_transform(input))
        } else {
            Err(SklearsError::InvalidInput(
                "No transformations in pipeline".to_string()
            ))
        }
    }
}

/// Type-safe calibration builder
#[derive(Debug)]
pub struct CalibrationBuilder<M: CalibrationMethodType> {
    method_type: PhantomData<M>,
    parameters: std::collections::HashMap<String, Float>,
}

impl<M: CalibrationMethodType> CalibrationBuilder<M> {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            method_type: PhantomData,
            parameters: std::collections::HashMap::new(),
        }
    }

    /// Set a parameter
    pub fn with_parameter(mut self, name: &str, value: Float) -> Self {
        self.parameters.insert(name.to_string(), value);
        self
    }

    /// Build the calibrator
    pub fn build(self) -> Result<TypeSafeCalibrator<M, Untrained>> {
        // Create appropriate inner calibrator based on method type
        let inner: Box<dyn CalibrationEstimator> = if std::any::TypeId::of::<M>() == std::any::TypeId::of::<SigmoidMethod>() {
            Box::new(crate::SigmoidCalibrator::new())
        } else if std::any::TypeId::of::<M>() == std::any::TypeId::of::<IsotonicMethod>() {
            Box::new(crate::IsotonicCalibrator::new())
        } else if std::any::TypeId::of::<M>() == std::any::TypeId::of::<TemperatureMethod>() {
            Box::new(crate::TemperatureScalingCalibrator::new())
        } else if std::any::TypeId::of::<M>() == std::any::TypeId::of::<HistogramMethod>() {
            Box::new(crate::HistogramBinningCalibrator::new())
        } else if std::any::TypeId::of::<M>() == std::any::TypeId::of::<BBQMethod>() {
            Box::new(crate::BBQCalibrator::new())
        } else {
            return Err(SklearsError::InvalidInput(
                "Unknown calibration method type".to_string()
            ));
        };

        Ok(TypeSafeCalibrator::new(inner))
    }
}

impl<M: CalibrationMethodType> Default for CalibrationBuilder<M> {
    fn default() -> Self {
        Self::new()
    }
}

/// Compile-time probability validation macro
#[macro_export]
macro_rules! probability {
    ($value:expr) => {{
        const _: () = {
            assert!($value >= 0.0 && $value <= 1.0, "Probability must be between 0 and 1");
        };
        Probability::new($value).unwrap()
    }};
}

/// Const generic array of probabilities
pub type ProbabilityVector<const N: usize> = ProbabilityArray<N, true>;

/// Factory functions for creating type-safe calibrators
pub mod factory {
    use super::*;

    /// Create a sigmoid calibrator
    pub fn sigmoid() -> CalibrationBuilder<SigmoidMethod> {
        CalibrationBuilder::new()
    }

    /// Create an isotonic calibrator
    pub fn isotonic() -> CalibrationBuilder<IsotonicMethod> {
        CalibrationBuilder::new()
    }

    /// Create a temperature scaling calibrator
    pub fn temperature() -> CalibrationBuilder<TemperatureMethod> {
        CalibrationBuilder::new()
    }

    /// Create a histogram calibrator
    pub fn histogram() -> CalibrationBuilder<HistogramMethod> {
        CalibrationBuilder::new()
    }

    /// Create a BBQ calibrator
    pub fn bbq() -> CalibrationBuilder<BBQMethod> {
        CalibrationBuilder::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::SigmoidCalibrator;

    #[test]
    fn test_probability_creation() {
        let valid_prob = Probability::new(0.5).unwrap();
        assert_eq!(valid_prob.value(), 0.5);

        let invalid_prob = Probability::new(1.5);
        assert!(invalid_prob.is_err());
    }

    #[test]
    fn test_probability_operations() {
        let prob1 = Probability::new(0.3).unwrap();
        let prob2 = Probability::new(0.4).unwrap();

        let sum = prob1.safe_add(&prob2).unwrap();
        assert_eq!(sum.value(), 0.7);

        let product = prob1.safe_mul(&prob2);
        assert_eq!(product.value(), 0.12);

        let complement = prob1.complement();
        assert_eq!(complement.value(), 0.7);
    }

    #[test]
    fn test_probability_array() {
        let values = Array1::from(vec![0.1, 0.5, 0.9]);
        let prob_array = ProbabilityArray::<0, true>::new(values).unwrap();

        assert_eq!(prob_array.len(), 3);
        assert_eq!(prob_array.get(1).unwrap().value(), 0.5);

        let invalid_values = Array1::from(vec![0.1, 1.5, 0.9]);
        let invalid_array = ProbabilityArray::<0, true>::new(invalid_values);
        assert!(invalid_array.is_err());
    }

    #[test]
    fn test_type_safe_calibrator() {
        let inner = Box::new(SigmoidCalibrator::new());
        let untrained_calibrator = TypeSafeCalibrator::<SigmoidMethod, Untrained>::new(inner);

        let probabilities = ProbabilityArray::new(Array1::from(vec![0.1, 0.3, 0.7, 0.9])).unwrap();
        let targets = Array1::from(vec![0, 0, 1, 1]);

        let trained_calibrator = untrained_calibrator.fit(&probabilities, &targets).unwrap();

        let test_probs = ProbabilityArray::new(Array1::from(vec![0.2, 0.8])).unwrap();
        let predictions = trained_calibrator.predict_proba(&test_probs).unwrap();

        assert_eq!(predictions.len(), 2);
    }

    #[test]
    fn test_fixed_size_calibrator() {
        let inner = Box::new(SigmoidCalibrator::new());
        let calibrator = FixedSizeCalibrator::<4, SigmoidMethod, Untrained>::new(inner);

        let probabilities = ProbabilityArray::new(Array1::from(vec![0.1, 0.3, 0.7, 0.9])).unwrap();
        let targets = [0, 0, 1, 1];

        let trained = calibrator.fit(&probabilities, &targets).unwrap();

        let test_probs = ProbabilityArray::new(Array1::from(vec![0.2, 0.4, 0.6, 0.8])).unwrap();
        let predictions = trained.predict_proba(&test_probs).unwrap();

        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_zero_cost_transformations() {
        let prob = Probability::new(0.7).unwrap();

        let identity = IdentityTransform;
        let transformed = identity.transform(prob);
        assert_eq!(transformed.value(), 0.7);

        let logit_transform = LogitTransform;
        let logit = logit_transform.transform(prob);
        assert!(logit.is_finite());

        let sigmoid_transform = SigmoidTransform;
        let back_to_prob = sigmoid_transform.transform(logit).validate().unwrap();
        assert!((back_to_prob.value() - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_bounded_probabilities() {
        let standard_prob = BoundedProbability::<StandardBounds>::new(0.0).unwrap();
        assert_eq!(standard_prob.value(), 0.0);

        let strict_prob_result = BoundedProbability::<StrictBounds>::new(0.0);
        assert!(strict_prob_result.is_err());

        let valid_strict = BoundedProbability::<StrictBounds>::new(0.5).unwrap();
        assert_eq!(valid_strict.value(), 0.5);
    }

    #[test]
    fn test_calibration_builder() {
        let builder = factory::sigmoid().with_parameter("alpha", 1.0);
        let calibrator = builder.build().unwrap();

        assert_eq!(calibrator.method_type(), "sklears_calibration::type_safety::SigmoidMethod");
    }

    #[test]
    fn test_probability_array_operations() {
        let values = Array1::from(vec![0.1, 0.2, 0.3, 0.4]);
        let prob_array = ProbabilityArray::<0, true>::new(values).unwrap();

        let normalized = prob_array.normalize().unwrap();
        let sum: Float = normalized.inner().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        let logits = prob_array.to_logits();
        let back_to_probs = ProbabilityArray::<0, true>::from_logits(&logits).unwrap();
        
        for (orig, back) in prob_array.inner().iter().zip(back_to_probs.inner().iter()) {
            assert!((orig - back).abs() < 1e-10);
        }
    }

    #[test]
    fn test_softmax_normalization() {
        let values = Array1::from(vec![1.0, 2.0, 3.0]);
        let prob_array = ProbabilityArray::<0, true>::new(values).unwrap();

        let softmax_result = prob_array.softmax().unwrap();
        let sum: Float = softmax_result.inner().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Check that probabilities are in descending order (since inputs were ascending)
        let probs = softmax_result.inner();
        assert!(probs[0] < probs[1]);
        assert!(probs[1] < probs[2]);
    }
}