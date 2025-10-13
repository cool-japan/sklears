//! Type-safe probability computations with phantom types
//!
//! This module provides zero-cost abstractions for probabilistic computations
//! using phantom types and const generics for compile-time type safety.

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use sklears_core::error::SklearsError;
use std::marker::PhantomData;

/// Distribution type markers
pub mod distribution_types {
    /// Marker for Gaussian distribution
    #[derive(Debug, Clone, Copy)]
    pub struct Gaussian;

    /// Marker for Multinomial distribution
    #[derive(Debug, Clone, Copy)]
    pub struct Multinomial;

    /// Marker for Bernoulli distribution
    #[derive(Debug, Clone, Copy)]
    pub struct Bernoulli;

    /// Marker for Poisson distribution
    #[derive(Debug, Clone, Copy)]
    pub struct Poisson;

    /// Marker for Beta distribution
    #[derive(Debug, Clone, Copy)]
    pub struct Beta;

    /// Marker for Gamma distribution
    #[derive(Debug, Clone, Copy)]
    pub struct Gamma;

    /// Marker for Categorical distribution
    #[derive(Debug, Clone, Copy)]
    pub struct Categorical;

    /// Marker for Exponential distribution
    #[derive(Debug, Clone, Copy)]
    pub struct Exponential;
}

/// Feature type markers
pub mod feature_types {
    /// Marker for continuous features
    #[derive(Debug, Clone, Copy)]
    pub struct Continuous;

    /// Marker for discrete features
    #[derive(Debug, Clone, Copy)]
    pub struct Discrete;

    /// Marker for binary features
    #[derive(Debug, Clone, Copy)]
    pub struct Binary;

    /// Marker for count features
    #[derive(Debug, Clone, Copy)]
    pub struct Count;
}

/// Probability wrapper with compile-time type safety
#[derive(Debug, Clone)]
pub struct TypedProbability<T, Distribution, Feature> {
    value: T,
    _distribution: PhantomData<Distribution>,
    _feature: PhantomData<Feature>,
}

impl<T, Distribution, Feature> TypedProbability<T, Distribution, Feature> {
    /// Create a new typed probability (private constructor)
    fn new(value: T) -> Self {
        Self {
            value,
            _distribution: PhantomData,
            _feature: PhantomData,
        }
    }

    /// Get the underlying value
    pub fn value(&self) -> &T {
        &self.value
    }

    /// Extract the underlying value
    pub fn into_value(self) -> T {
        self.value
    }
}

/// Type-safe probability operations
pub trait TypedProbabilityOps<T: Float> {
    /// Create a typed probability with validation
    fn typed_prob<Distribution, Feature>(
        self,
    ) -> Result<TypedProbability<T, Distribution, Feature>, SklearsError>;

    /// Create a typed log probability with validation
    fn typed_log_prob<Distribution, Feature>(
        self,
    ) -> Result<TypedProbability<T, Distribution, Feature>, SklearsError>;
}

impl<T: Float> TypedProbabilityOps<T> for T {
    fn typed_prob<Distribution, Feature>(
        self,
    ) -> Result<TypedProbability<T, Distribution, Feature>, SklearsError> {
        if self < T::zero() || self > T::one() {
            Err(SklearsError::InvalidOperation(
                "Probability must be between 0 and 1".to_string(),
            ))
        } else {
            Ok(TypedProbability::new(self))
        }
    }

    fn typed_log_prob<Distribution, Feature>(
        self,
    ) -> Result<TypedProbability<T, Distribution, Feature>, SklearsError> {
        if self > T::zero() {
            Err(SklearsError::InvalidOperation(
                "Log probability must be negative or zero".to_string(),
            ))
        } else {
            Ok(TypedProbability::new(self))
        }
    }
}

/// Fixed-size model configuration using const generics
#[derive(Debug, Clone)]
pub struct FixedSizeModel<const N_FEATURES: usize, const N_CLASSES: usize, Distribution> {
    /// Feature parameters
    pub feature_params: [[f64; N_FEATURES]; N_CLASSES],
    /// Class priors
    pub class_priors: [f64; N_CLASSES],
    /// Distribution marker
    _distribution: PhantomData<Distribution>,
}

impl<const N_FEATURES: usize, const N_CLASSES: usize, Distribution>
    FixedSizeModel<N_FEATURES, N_CLASSES, Distribution>
{
    /// Create a new fixed-size model
    pub fn new(
        feature_params: [[f64; N_FEATURES]; N_CLASSES],
        class_priors: [f64; N_CLASSES],
    ) -> Self {
        Self {
            feature_params,
            class_priors,
            _distribution: PhantomData,
        }
    }

    /// Get feature parameters for a specific class
    pub fn feature_params_for_class(&self, class_idx: usize) -> Option<&[f64; N_FEATURES]> {
        self.feature_params.get(class_idx)
    }

    /// Get class prior for a specific class
    pub fn class_prior(&self, class_idx: usize) -> Option<f64> {
        self.class_priors.get(class_idx).copied()
    }
}

/// Zero-cost probabilistic abstractions
pub trait ProbabilisticModel<T: Float + std::ops::Add<Output = T> + Copy> {
    type Distribution;
    type Features;

    /// Compute log likelihood with type safety
    fn log_likelihood(
        &self,
        features: &Array1<T>,
    ) -> TypedProbability<T, Self::Distribution, Self::Features>;

    /// Compute log prior with type safety
    fn log_prior(
        &self,
        class_idx: usize,
    ) -> TypedProbability<T, Self::Distribution, Self::Features>;

    /// Compute log posterior with type safety
    fn log_posterior(
        &self,
        features: &Array1<T>,
        class_idx: usize,
    ) -> TypedProbability<T, Self::Distribution, Self::Features> {
        let log_likelihood = self.log_likelihood(features);
        let log_prior = self.log_prior(class_idx);
        TypedProbability::new(*log_likelihood.value() + *log_prior.value())
    }
}

/// Compile-time feature type validation
pub trait ValidateFeatureType<Feature> {
    /// Validate that the feature type is compatible with the distribution
    fn validate_feature_compatibility() -> Result<(), SklearsError>;
}

/// Gaussian distribution can handle continuous features
impl ValidateFeatureType<feature_types::Continuous> for distribution_types::Gaussian {
    fn validate_feature_compatibility() -> Result<(), SklearsError> {
        Ok(())
    }
}

/// Multinomial distribution can handle count features
impl ValidateFeatureType<feature_types::Count> for distribution_types::Multinomial {
    fn validate_feature_compatibility() -> Result<(), SklearsError> {
        Ok(())
    }
}

/// Bernoulli distribution can handle binary features
impl ValidateFeatureType<feature_types::Binary> for distribution_types::Bernoulli {
    fn validate_feature_compatibility() -> Result<(), SklearsError> {
        Ok(())
    }
}

/// Poisson distribution can handle count features
impl ValidateFeatureType<feature_types::Count> for distribution_types::Poisson {
    fn validate_feature_compatibility() -> Result<(), SklearsError> {
        Ok(())
    }
}

/// Beta distribution can handle continuous features (0-1 range)
impl ValidateFeatureType<feature_types::Continuous> for distribution_types::Beta {
    fn validate_feature_compatibility() -> Result<(), SklearsError> {
        Ok(())
    }
}

/// Gamma distribution can handle continuous features (positive values)
impl ValidateFeatureType<feature_types::Continuous> for distribution_types::Gamma {
    fn validate_feature_compatibility() -> Result<(), SklearsError> {
        Ok(())
    }
}

/// Categorical distribution can handle discrete features
impl ValidateFeatureType<feature_types::Discrete> for distribution_types::Categorical {
    fn validate_feature_compatibility() -> Result<(), SklearsError> {
        Ok(())
    }
}

/// Exponential distribution can handle continuous features (positive values)
impl ValidateFeatureType<feature_types::Continuous> for distribution_types::Exponential {
    fn validate_feature_compatibility() -> Result<(), SklearsError> {
        Ok(())
    }
}

/// Type-safe probability computation utilities
pub mod utils {
    use super::*;

    /// Compute log-sum-exp with type safety
    pub fn log_sum_exp<T: Float, Distribution, Feature>(
        values: &[TypedProbability<T, Distribution, Feature>],
    ) -> TypedProbability<T, Distribution, Feature> {
        if values.is_empty() {
            return TypedProbability::new(T::neg_infinity());
        }

        let max_val = values
            .iter()
            .map(|v| *v.value())
            .fold(T::neg_infinity(), |a, b| a.max(b));

        if max_val == T::neg_infinity() {
            return TypedProbability::new(T::neg_infinity());
        }

        let sum = values
            .iter()
            .map(|v| (*v.value() - max_val).exp())
            .fold(T::zero(), |a, b| a + b);

        TypedProbability::new(max_val + sum.ln())
    }

    /// Normalize log probabilities with type safety
    pub fn normalize_log_probs<T: Float, Distribution, Feature>(
        log_probs: &[TypedProbability<T, Distribution, Feature>],
    ) -> Vec<TypedProbability<T, Distribution, Feature>> {
        let log_sum = log_sum_exp(log_probs);
        log_probs
            .iter()
            .map(|prob| TypedProbability::new(*prob.value() - *log_sum.value()))
            .collect()
    }

    /// Convert log probabilities to probabilities with type safety
    pub fn log_to_prob<T: Float, Distribution, Feature>(
        log_prob: &TypedProbability<T, Distribution, Feature>,
    ) -> TypedProbability<T, Distribution, Feature> {
        TypedProbability::new(log_prob.value().exp())
    }

    /// Convert probabilities to log probabilities with type safety
    pub fn prob_to_log<T: Float, Distribution, Feature>(
        prob: &TypedProbability<T, Distribution, Feature>,
    ) -> TypedProbability<T, Distribution, Feature> {
        TypedProbability::new(prob.value().ln())
    }
}

/// Const generic utilities for fixed-size models
pub mod const_utils {
    use super::*;

    /// Fixed-size Gaussian model
    pub type FixedGaussianModel<const N_FEATURES: usize, const N_CLASSES: usize> =
        FixedSizeModel<N_FEATURES, N_CLASSES, distribution_types::Gaussian>;

    /// Fixed-size Multinomial model
    pub type FixedMultinomialModel<const N_FEATURES: usize, const N_CLASSES: usize> =
        FixedSizeModel<N_FEATURES, N_CLASSES, distribution_types::Multinomial>;

    /// Fixed-size Bernoulli model
    pub type FixedBernoulliModel<const N_FEATURES: usize, const N_CLASSES: usize> =
        FixedSizeModel<N_FEATURES, N_CLASSES, distribution_types::Bernoulli>;

    /// Common fixed-size models
    pub mod common {
        use super::*;

        /// Small binary classification model (10 features, 2 classes)
        pub type SmallBinaryClassifier<Distribution> = FixedSizeModel<10, 2, Distribution>;

        /// Medium multi-class model (100 features, 5 classes)
        pub type MediumMultiClassifier<Distribution> = FixedSizeModel<100, 5, Distribution>;

        /// Large multi-class model (1000 features, 10 classes)
        pub type LargeMultiClassifier<Distribution> = FixedSizeModel<1000, 10, Distribution>;
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typed_probability_creation() {
        // Valid probability
        let prob = 0.5f64.typed_prob::<distribution_types::Gaussian, feature_types::Continuous>();
        assert!(prob.is_ok());
        assert_eq!(*prob.unwrap().value(), 0.5);

        // Invalid probability (too large)
        let prob = 1.5f64.typed_prob::<distribution_types::Gaussian, feature_types::Continuous>();
        assert!(prob.is_err());

        // Invalid probability (negative)
        let prob =
            (-0.1f64).typed_prob::<distribution_types::Gaussian, feature_types::Continuous>();
        assert!(prob.is_err());
    }

    #[test]
    fn test_typed_log_probability_creation() {
        // Valid log probability
        let log_prob =
            (-0.5f64).typed_log_prob::<distribution_types::Gaussian, feature_types::Continuous>();
        assert!(log_prob.is_ok());
        assert_eq!(*log_prob.unwrap().value(), -0.5);

        // Invalid log probability (positive)
        let log_prob =
            0.5f64.typed_log_prob::<distribution_types::Gaussian, feature_types::Continuous>();
        assert!(log_prob.is_err());
    }

    #[test]
    fn test_feature_type_validation() {
        // Valid combinations
        assert!(<distribution_types::Gaussian as ValidateFeatureType<feature_types::Continuous>>::validate_feature_compatibility().is_ok());
        assert!(<distribution_types::Multinomial as ValidateFeatureType<feature_types::Count>>::validate_feature_compatibility().is_ok());
        assert!(<distribution_types::Bernoulli as ValidateFeatureType<feature_types::Binary>>::validate_feature_compatibility().is_ok());
    }

    #[test]
    fn test_fixed_size_model() {
        let feature_params = [[1.0, 2.0], [3.0, 4.0]];
        let class_priors = [0.3, 0.7];

        let model =
            FixedSizeModel::<2, 2, distribution_types::Gaussian>::new(feature_params, class_priors);

        assert_eq!(model.feature_params_for_class(0), Some(&[1.0, 2.0]));
        assert_eq!(model.feature_params_for_class(1), Some(&[3.0, 4.0]));
        assert_eq!(model.class_prior(0), Some(0.3));
        assert_eq!(model.class_prior(1), Some(0.7));
    }

    #[test]
    fn test_log_sum_exp() {
        let values = vec![
            TypedProbability::<f64, distribution_types::Gaussian, feature_types::Continuous>::new(
                -1.0,
            ),
            TypedProbability::<f64, distribution_types::Gaussian, feature_types::Continuous>::new(
                -2.0,
            ),
            TypedProbability::<f64, distribution_types::Gaussian, feature_types::Continuous>::new(
                -3.0,
            ),
        ];

        let result = utils::log_sum_exp(&values);
        let expected = (-1.0f64 + ((-2.0f64 + 1.0).exp() + (-3.0f64 + 1.0).exp() + 1.0).ln());
        assert!((result.value() - expected).abs() < 1e-10);
    }
}
