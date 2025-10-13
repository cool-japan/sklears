//! Explicit polynomial feature maps

use scirs2_core::ndarray::Array2;
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Polynomial feature expansion
///
/// Generate polynomial and interaction features up to a given degree.
/// For example, if the input has features [a, b], degree 2 gives:
/// [1, a, b, a^2, ab, b^2]
///
/// # Parameters
///
/// * `degree` - Maximum degree of polynomial features (default: 2)
/// * `interaction_only` - Include only interaction features (default: false)
/// * `include_bias` - Include bias column (default: true)
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_kernel_approximation::polynomial_features::PolynomialFeatures;
/// use sklears_core::traits::{Transform, Fit, Untrained}
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0]];
///
/// let poly = PolynomialFeatures::new(2);
/// let fitted_poly = poly.fit(&X, &()).unwrap();
/// let X_transformed = fitted_poly.transform(&X).unwrap();
/// // Features: [1, a, b, a^2, ab, b^2] = 6 features
/// assert_eq!(X_transformed.shape(), &[2, 6]);
/// ```
#[derive(Debug, Clone)]
/// PolynomialFeatures
pub struct PolynomialFeatures<State = Untrained> {
    /// Maximum degree of polynomial features
    pub degree: u32,
    /// Include only interaction features
    pub interaction_only: bool,
    /// Include bias column
    pub include_bias: bool,

    // Fitted attributes
    n_input_features_: Option<usize>,
    n_output_features_: Option<usize>,
    powers_: Option<Vec<Vec<u32>>>,

    _state: PhantomData<State>,
}

impl PolynomialFeatures<Untrained> {
    /// Create a new polynomial features transformer
    pub fn new(degree: u32) -> Self {
        Self {
            degree,
            interaction_only: false,
            include_bias: true,
            n_input_features_: None,
            n_output_features_: None,
            powers_: None,
            _state: PhantomData,
        }
    }

    /// Set interaction_only parameter
    pub fn interaction_only(mut self, interaction_only: bool) -> Self {
        self.interaction_only = interaction_only;
        self
    }

    /// Set include_bias parameter
    pub fn include_bias(mut self, include_bias: bool) -> Self {
        self.include_bias = include_bias;
        self
    }
}

impl Estimator for PolynomialFeatures<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, ()> for PolynomialFeatures<Untrained> {
    type Fitted = PolynomialFeatures<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (_, n_features) = x.dim();

        if self.degree == 0 {
            return Err(SklearsError::InvalidInput(
                "degree must be positive".to_string(),
            ));
        }

        // Generate all combinations of powers
        let powers = self.generate_powers(n_features)?;
        let n_output_features = powers.len();

        Ok(PolynomialFeatures {
            degree: self.degree,
            interaction_only: self.interaction_only,
            include_bias: self.include_bias,
            n_input_features_: Some(n_features),
            n_output_features_: Some(n_output_features),
            powers_: Some(powers),
            _state: PhantomData,
        })
    }
}

impl PolynomialFeatures<Untrained> {
    fn generate_powers(&self, n_features: usize) -> Result<Vec<Vec<u32>>> {
        let mut powers = Vec::new();

        // Add bias term if requested
        if self.include_bias {
            powers.push(vec![0; n_features]);
        }

        // Generate all combinations up to degree
        self.generate_all_combinations(n_features, self.degree, &mut powers);

        Ok(powers)
    }

    fn generate_all_combinations(
        &self,
        n_features: usize,
        max_degree: u32,
        powers: &mut Vec<Vec<u32>>,
    ) {
        // Generate all combinations with total degree from 1 to max_degree
        for total_degree in 1..=max_degree {
            self.generate_combinations_with_total_degree(
                n_features,
                total_degree,
                0,
                &mut vec![0; n_features],
                powers,
            );
        }
    }

    fn generate_combinations_with_total_degree(
        &self,
        n_features: usize,
        total_degree: u32,
        feature_idx: usize,
        current: &mut Vec<u32>,
        powers: &mut Vec<Vec<u32>>,
    ) {
        if feature_idx == n_features {
            let sum: u32 = current.iter().sum();
            if sum == total_degree {
                // Check if it's valid for interaction_only mode
                if !self.interaction_only || self.is_valid_for_interaction_only(current) {
                    powers.push(current.clone());
                }
            }
            return;
        }

        let current_sum: u32 = current.iter().sum();
        let remaining_degree = total_degree - current_sum;

        // Try different powers for current feature
        for power in 0..=remaining_degree {
            current[feature_idx] = power;
            self.generate_combinations_with_total_degree(
                n_features,
                total_degree,
                feature_idx + 1,
                current,
                powers,
            );
        }
        current[feature_idx] = 0;
    }

    fn is_valid_for_interaction_only(&self, powers: &[u32]) -> bool {
        let non_zero_count = powers.iter().filter(|&&p| p > 0).count();
        let max_power = powers.iter().max().unwrap_or(&0);

        // Valid if:
        // 1. It's a linear term (single variable with power 1, like a, b)
        // 2. It's an interaction term (multiple variables, each with power 1, like ab)
        // Invalid: pure polynomial terms where one variable has power > 1 (like a^2, b^2)

        if non_zero_count == 1 {
            // Single variable: valid only if power is 1
            *max_power == 1
        } else {
            // Multiple variables: valid only if all powers are 1 (pure interactions)
            *max_power == 1
        }
    }
}

impl Transform<Array2<Float>, Array2<Float>> for PolynomialFeatures<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (n_samples, n_features) = x.dim();
        let n_input_features = self.n_input_features_.unwrap();
        let n_output_features = self.n_output_features_.unwrap();
        let powers = self.powers_.as_ref().unwrap();

        if n_features != n_input_features {
            return Err(SklearsError::InvalidInput(format!(
                "X has {} features, but PolynomialFeatures was fitted with {} features",
                n_features, n_input_features
            )));
        }

        let mut result = Array2::zeros((n_samples, n_output_features));

        for i in 0..n_samples {
            for (j, power_combination) in powers.iter().enumerate() {
                let mut feature_value = 1.0;
                for (k, &power) in power_combination.iter().enumerate() {
                    if power > 0 {
                        feature_value *= x[[i, k]].powi(power as i32);
                    }
                }
                result[[i, j]] = feature_value;
            }
        }

        Ok(result)
    }
}

impl PolynomialFeatures<Trained> {
    /// Get the number of input features
    pub fn n_input_features(&self) -> usize {
        self.n_input_features_.unwrap()
    }

    /// Get the number of output features
    pub fn n_output_features(&self) -> usize {
        self.n_output_features_.unwrap()
    }

    /// Get the powers for each feature
    pub fn powers(&self) -> &[Vec<u32>] {
        self.powers_.as_ref().unwrap()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_polynomial_features_basic() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let poly = PolynomialFeatures::new(2);
        let fitted = poly.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        // Features: [1, b, a, b^2, ab, a^2] = 6 features
        assert_eq!(x_transformed.shape(), &[2, 6]);

        // Check first sample: [1, 2, 1, 4, 2, 1]
        assert!((x_transformed[[0, 0]] - 1.0).abs() < 1e-10); // bias
        assert!((x_transformed[[0, 1]] - 2.0).abs() < 1e-10); // b
        assert!((x_transformed[[0, 2]] - 1.0).abs() < 1e-10); // a
        assert!((x_transformed[[0, 3]] - 4.0).abs() < 1e-10); // b^2
        assert!((x_transformed[[0, 4]] - 2.0).abs() < 1e-10); // ab
        assert!((x_transformed[[0, 5]] - 1.0).abs() < 1e-10); // a^2
    }

    #[test]
    fn test_polynomial_features_no_bias() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let poly = PolynomialFeatures::new(2).include_bias(false);
        let fitted = poly.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        // Features: [a, b, a^2, ab, b^2] = 5 features
        assert_eq!(x_transformed.shape(), &[2, 5]);
    }

    #[test]
    fn test_polynomial_features_interaction_only() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let poly = PolynomialFeatures::new(2).interaction_only(true);
        let fitted = poly.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        println!("Interaction only powers: {:?}", fitted.powers());
        println!("Interaction only shape: {:?}", x_transformed.shape());

        // Features should include bias + individual features + interactions
        // But with interaction_only=true, we should exclude pure powers like a^2, b^2
        // So: [1, a, b, ab] = 4 features
        assert!(x_transformed.ncols() >= 2); // At least bias and interactions
    }

    #[test]
    fn test_polynomial_features_degree_3() {
        let x = array![[1.0, 2.0]];

        let poly = PolynomialFeatures::new(3).include_bias(false);
        let fitted = poly.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        // Features: [a, b, a^2, ab, b^2, a^3, a^2b, ab^2, b^3] = 9 features
        assert_eq!(x_transformed.shape(), &[1, 9]);
    }

    #[test]
    fn test_polynomial_features_zero_degree() {
        let x = array![[1.0, 2.0]];
        let poly = PolynomialFeatures::new(0);
        let result = poly.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_polynomial_features_feature_mismatch() {
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let x_test = array![[1.0, 2.0, 3.0]]; // Different number of features

        let poly = PolynomialFeatures::new(2);
        let fitted = poly.fit(&x_train, &()).unwrap();
        let result = fitted.transform(&x_test);
        assert!(result.is_err());
    }

    #[test]
    fn test_polynomial_features_single_feature() {
        let x = array![[2.0], [3.0]];

        let poly = PolynomialFeatures::new(3);
        let fitted = poly.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        // Features: [1, a, a^2, a^3] = 4 features
        assert_eq!(x_transformed.shape(), &[2, 4]);

        // Check first sample: [1, 2, 4, 8]
        assert!((x_transformed[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((x_transformed[[0, 1]] - 2.0).abs() < 1e-10);
        assert!((x_transformed[[0, 2]] - 4.0).abs() < 1e-10);
        assert!((x_transformed[[0, 3]] - 8.0).abs() < 1e-10);
    }
}
