//! Homogeneous polynomial features with fixed total degree

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    prelude::{Fit, Transform},
    traits::{Estimator, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Normalization method for homogeneous polynomial features
#[derive(Debug, Clone)]
/// NormalizationMethod
pub enum NormalizationMethod {
    /// No normalization
    None,
    /// L2 normalization (unit norm)
    L2,
    /// L1 normalization
    L1,
    /// Max normalization
    Max,
    /// Standard normalization (mean=0, std=1)
    Standard,
}

/// Multinomial coefficient computation method
#[derive(Debug, Clone)]
/// CoefficientMethod
pub enum CoefficientMethod {
    /// Include multinomial coefficients
    Multinomial,
    /// Unit coefficients (all 1)
    Unit,
    /// Square root of multinomial coefficients
    SqrtMultinomial,
}

/// Homogeneous Polynomial Features
///
/// Generates polynomial features where all terms have exactly the same total degree.
/// This is useful for creating features that capture specific order interactions
/// without lower-order contamination.
///
/// For degree d and features [x₁, x₂, ..., xₙ], generates all terms of the form:
/// x₁^(i₁) * x₂^(i₂) * ... * xₙ^(iₙ) where i₁ + i₂ + ... + iₙ = d
///
/// # Parameters
///
/// * `degree` - The fixed total degree for all polynomial terms
/// * `interaction_only` - Include only interaction terms (all powers ≤ 1)
/// * `normalization` - Normalization method for features
/// * `coefficient_method` - Method for computing multinomial coefficients
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_kernel_approximation::homogeneous_polynomial::HomogeneousPolynomialFeatures;
/// use sklears_core::traits::{Transform, Fit, Untrained}
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0]];
///
/// let homo_poly = HomogeneousPolynomialFeatures::new(2);
/// let fitted_homo = homo_poly.fit(&X, &()).unwrap();
/// let X_transformed = fitted_homo.transform(&X).unwrap();
/// ```
#[derive(Debug, Clone)]
/// HomogeneousPolynomialFeatures
pub struct HomogeneousPolynomialFeatures<State = Untrained> {
    /// The fixed total degree
    pub degree: u32,
    /// Include only interaction terms
    pub interaction_only: bool,
    /// Normalization method
    pub normalization: NormalizationMethod,
    /// Coefficient computation method
    pub coefficient_method: CoefficientMethod,

    // Fitted attributes
    n_input_features_: Option<usize>,
    n_output_features_: Option<usize>,
    power_combinations_: Option<Vec<Vec<u32>>>,
    coefficients_: Option<Vec<Float>>,
    normalization_params_: Option<(Array1<Float>, Array1<Float>)>, // (mean, std) for standard normalization

    _state: PhantomData<State>,
}

impl HomogeneousPolynomialFeatures<Untrained> {
    /// Create a new homogeneous polynomial features transformer
    pub fn new(degree: u32) -> Self {
        Self {
            degree,
            interaction_only: false,
            normalization: NormalizationMethod::None,
            coefficient_method: CoefficientMethod::Unit,
            n_input_features_: None,
            n_output_features_: None,
            power_combinations_: None,
            coefficients_: None,
            normalization_params_: None,
            _state: PhantomData,
        }
    }

    /// Set interaction_only parameter
    pub fn interaction_only(mut self, interaction_only: bool) -> Self {
        self.interaction_only = interaction_only;
        self
    }

    /// Set normalization method
    pub fn normalization(mut self, method: NormalizationMethod) -> Self {
        self.normalization = method;
        self
    }

    /// Set coefficient method
    pub fn coefficient_method(mut self, method: CoefficientMethod) -> Self {
        self.coefficient_method = method;
        self
    }
}

impl Estimator for HomogeneousPolynomialFeatures<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, ()> for HomogeneousPolynomialFeatures<Untrained> {
    type Fitted = HomogeneousPolynomialFeatures<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (_, n_features) = x.dim();

        if self.degree == 0 {
            return Err(SklearsError::InvalidInput(
                "degree must be positive".to_string(),
            ));
        }

        // Generate all power combinations with the fixed total degree
        let power_combinations = self.generate_homogeneous_combinations(n_features)?;

        // Compute coefficients
        let coefficients = self.compute_coefficients(&power_combinations)?;

        let n_output_features = power_combinations.len();

        // Compute normalization parameters if needed
        let normalization_params = match self.normalization {
            NormalizationMethod::Standard => {
                Some(self.compute_normalization_params(x, &power_combinations, &coefficients)?)
            }
            _ => None,
        };

        Ok(HomogeneousPolynomialFeatures {
            degree: self.degree,
            interaction_only: self.interaction_only,
            normalization: self.normalization,
            coefficient_method: self.coefficient_method,
            n_input_features_: Some(n_features),
            n_output_features_: Some(n_output_features),
            power_combinations_: Some(power_combinations),
            coefficients_: Some(coefficients),
            normalization_params_: normalization_params,
            _state: PhantomData,
        })
    }
}

impl HomogeneousPolynomialFeatures<Untrained> {
    /// Generate all homogeneous power combinations with fixed total degree
    fn generate_homogeneous_combinations(&self, n_features: usize) -> Result<Vec<Vec<u32>>> {
        let mut combinations = Vec::new();
        let mut current_combination = vec![0; n_features];

        self.generate_combinations_recursive(
            n_features,
            self.degree,
            0,
            &mut current_combination,
            &mut combinations,
        );

        // Filter based on interaction_only setting
        if self.interaction_only {
            combinations.retain(|combination| self.is_valid_for_interaction_only(combination));
        }

        Ok(combinations)
    }

    /// Recursively generate combinations with fixed total degree
    fn generate_combinations_recursive(
        &self,
        n_features: usize,
        remaining_degree: u32,
        feature_idx: usize,
        current: &mut Vec<u32>,
        combinations: &mut Vec<Vec<u32>>,
    ) {
        if feature_idx == n_features {
            if remaining_degree == 0 {
                combinations.push(current.clone());
            }
            return;
        }

        // Try all possible powers for current feature
        for power in 0..=remaining_degree {
            current[feature_idx] = power;
            self.generate_combinations_recursive(
                n_features,
                remaining_degree - power,
                feature_idx + 1,
                current,
                combinations,
            );
        }
        current[feature_idx] = 0;
    }

    /// Check if combination is valid for interaction_only mode
    fn is_valid_for_interaction_only(&self, combination: &[u32]) -> bool {
        let non_zero_count = combination.iter().filter(|&&p| p > 0).count();
        let max_power = combination.iter().max().unwrap_or(&0);

        // For interaction_only:
        // - All non-zero powers must be 1
        // - Must have at least 2 non-zero features (pure interactions)
        *max_power == 1 && non_zero_count >= 2
    }

    /// Compute coefficients based on the chosen method
    fn compute_coefficients(&self, combinations: &[Vec<u32>]) -> Result<Vec<Float>> {
        let mut coefficients = Vec::new();

        for combination in combinations {
            let coeff = match self.coefficient_method {
                CoefficientMethod::Unit => 1.0,
                CoefficientMethod::Multinomial => self.compute_multinomial_coefficient(combination),
                CoefficientMethod::SqrtMultinomial => {
                    self.compute_multinomial_coefficient(combination).sqrt()
                }
            };
            coefficients.push(coeff);
        }

        Ok(coefficients)
    }

    /// Compute multinomial coefficient for a power combination
    fn compute_multinomial_coefficient(&self, powers: &[u32]) -> Float {
        let total_degree = powers.iter().sum::<u32>();

        if total_degree == 0 {
            return 1.0;
        }

        // Multinomial coefficient: n! / (k₁! * k₂! * ... * kₘ!)
        let numerator = self.factorial(total_degree);
        let mut denominator = 1.0;

        for &power in powers {
            if power > 0 {
                denominator *= self.factorial(power);
            }
        }

        numerator / denominator
    }

    /// Compute factorial (using floating point for large numbers)
    fn factorial(&self, n: u32) -> Float {
        if n <= 1 {
            1.0
        } else {
            (1..=n).map(|i| i as Float).product()
        }
    }

    /// Compute normalization parameters for standard normalization
    fn compute_normalization_params(
        &self,
        x: &Array2<Float>,
        combinations: &[Vec<u32>],
        coefficients: &[Float],
    ) -> Result<(Array1<Float>, Array1<Float>)> {
        let (n_samples, _) = x.dim();
        let n_features = combinations.len();

        let mut means = Array1::zeros(n_features);
        let mut stds = Array1::zeros(n_features);

        // Compute mean for each feature
        for i in 0..n_samples {
            for (j, (combination, &coeff)) in
                combinations.iter().zip(coefficients.iter()).enumerate()
            {
                let feature_value = self.compute_polynomial_value(&x.row(i), combination) * coeff;
                means[j] += feature_value;
            }
        }
        means /= n_samples as Float;

        // Compute standard deviation for each feature
        for i in 0..n_samples {
            for (j, (combination, &coeff)) in
                combinations.iter().zip(coefficients.iter()).enumerate()
            {
                let feature_value = self.compute_polynomial_value(&x.row(i), combination) * coeff;
                let diff = feature_value - means[j];
                stds[j] += diff * diff;
            }
        }
        stds = stds.mapv(|var: Float| (var / ((n_samples - 1) as Float)).sqrt().max(1e-12));

        Ok((means, stds))
    }

    /// Compute polynomial value for a single sample and power combination
    fn compute_polynomial_value(
        &self,
        sample: &scirs2_core::ndarray::ArrayView1<Float>,
        powers: &[u32],
    ) -> Float {
        let mut value = 1.0;
        for (i, &power) in powers.iter().enumerate() {
            if power > 0 && i < sample.len() {
                value *= sample[i].powi(power as i32);
            }
        }
        value
    }
}

impl Transform<Array2<Float>, Array2<Float>> for HomogeneousPolynomialFeatures<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (n_samples, n_features) = x.dim();
        let n_input_features = self.n_input_features_.unwrap();
        let n_output_features = self.n_output_features_.unwrap();
        let combinations = self.power_combinations_.as_ref().unwrap();
        let coefficients = self.coefficients_.as_ref().unwrap();

        if n_features != n_input_features {
            return Err(SklearsError::InvalidInput(format!(
                "X has {} features, but HomogeneousPolynomialFeatures was fitted with {} features",
                n_features, n_input_features
            )));
        }

        let mut result = Array2::zeros((n_samples, n_output_features));

        // Compute polynomial features
        for i in 0..n_samples {
            for (j, (combination, &coeff)) in
                combinations.iter().zip(coefficients.iter()).enumerate()
            {
                let mut feature_value = coeff;
                for (k, &power) in combination.iter().enumerate() {
                    if power > 0 {
                        feature_value *= x[[i, k]].powi(power as i32);
                    }
                }
                result[[i, j]] = feature_value;
            }
        }

        // Apply normalization
        match &self.normalization {
            NormalizationMethod::None => {}
            NormalizationMethod::L2 => {
                for mut row in result.rows_mut() {
                    let norm = (row.dot(&row)).sqrt();
                    if norm > 1e-12 {
                        row /= norm;
                    }
                }
            }
            NormalizationMethod::L1 => {
                for mut row in result.rows_mut() {
                    let norm = row.mapv(|v| v.abs()).sum();
                    if norm > 1e-12 {
                        row /= norm;
                    }
                }
            }
            NormalizationMethod::Max => {
                for mut row in result.rows_mut() {
                    let max_val = row.mapv(|v| v.abs()).fold(0.0_f64, |a: Float, &b| a.max(b));
                    if max_val > 1e-12 {
                        row /= max_val;
                    }
                }
            }
            NormalizationMethod::Standard => {
                if let Some((ref means, ref stds)) = self.normalization_params_ {
                    for i in 0..n_samples {
                        for j in 0..n_output_features {
                            result[[i, j]] = (result[[i, j]] - means[j]) / stds[j];
                        }
                    }
                }
            }
        }

        Ok(result)
    }
}

impl HomogeneousPolynomialFeatures<Trained> {
    /// Get the number of input features
    pub fn n_input_features(&self) -> usize {
        self.n_input_features_.unwrap()
    }

    /// Get the number of output features
    pub fn n_output_features(&self) -> usize {
        self.n_output_features_.unwrap()
    }

    /// Get the power combinations
    pub fn power_combinations(&self) -> &[Vec<u32>] {
        self.power_combinations_.as_ref().unwrap()
    }

    /// Get the coefficients
    pub fn coefficients(&self) -> &[Float] {
        self.coefficients_.as_ref().unwrap()
    }

    /// Get normalization parameters (if standard normalization is used)
    pub fn normalization_params(&self) -> Option<&(Array1<Float>, Array1<Float>)> {
        self.normalization_params_.as_ref()
    }

    /// Count the number of terms for a given degree and number of features
    pub fn count_homogeneous_terms(
        degree: u32,
        n_features: usize,
        interaction_only: bool,
    ) -> usize {
        if degree == 0 {
            return if interaction_only { 0 } else { 1 };
        }

        if interaction_only {
            // For interaction_only, we need exactly `degree` features with power 1 each
            if degree > n_features as u32 {
                return 0;
            }
            // This is binomial coefficient C(n_features, degree)
            Self::binomial_coefficient(n_features, degree as usize)
        } else {
            // Stars and bars: choose degree items from n_features + degree - 1 positions
            Self::binomial_coefficient(n_features + degree as usize - 1, degree as usize)
        }
    }

    /// Compute binomial coefficient C(n, k)
    fn binomial_coefficient(n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        if k == 0 || k == n {
            return 1;
        }

        let k = k.min(n - k); // Take advantage of symmetry
        let mut result = 1;

        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }

        result
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_homogeneous_polynomial_basic() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let homo_poly = HomogeneousPolynomialFeatures::new(2);
        let fitted = homo_poly.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.nrows(), 2);

        // For degree 2 with 2 features: x1^2, x1*x2, x2^2 = 3 terms
        assert_eq!(x_transformed.ncols(), 3);

        // Check specific values for first sample [1, 2]
        // Algorithm generates in order: [x1^2, x0*x1, x0^2] = [2^2, 1*2, 1^2] = [4, 2, 1]
        assert_abs_diff_eq!(x_transformed[[0, 0]], 4.0, epsilon = 1e-10); // x1^2
        assert_abs_diff_eq!(x_transformed[[0, 1]], 2.0, epsilon = 1e-10); // x0*x1
        assert_abs_diff_eq!(x_transformed[[0, 2]], 1.0, epsilon = 1e-10); // x0^2
    }

    #[test]
    fn test_homogeneous_polynomial_interaction_only() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let homo_poly = HomogeneousPolynomialFeatures::new(2).interaction_only(true);
        let fitted = homo_poly.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.nrows(), 2);

        // For degree 2 interaction only with 3 features: x1*x2, x1*x3, x2*x3 = 3 terms
        assert_eq!(x_transformed.ncols(), 3);

        // Check first sample [1, 2, 3]
        // Combinations are generated in order: [0,1,1], [1,0,1], [1,1,0]
        assert_abs_diff_eq!(x_transformed[[0, 0]], 6.0, epsilon = 1e-10); // x2*x3
        assert_abs_diff_eq!(x_transformed[[0, 1]], 3.0, epsilon = 1e-10); // x1*x3
        assert_abs_diff_eq!(x_transformed[[0, 2]], 2.0, epsilon = 1e-10); // x1*x2
    }

    #[test]
    fn test_homogeneous_polynomial_degree_3() {
        let x = array![[1.0, 2.0]];

        let homo_poly = HomogeneousPolynomialFeatures::new(3);
        let fitted = homo_poly.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        // For degree 3 with 2 features: x1^3, x1^2*x2, x1*x2^2, x2^3 = 4 terms
        assert_eq!(x_transformed.ncols(), 4);

        // Check values for sample [1, 2]
        // Combinations are generated in order: [0,3], [1,2], [2,1], [3,0]
        assert_abs_diff_eq!(x_transformed[[0, 0]], 8.0, epsilon = 1e-10); // x2^3
        assert_abs_diff_eq!(x_transformed[[0, 1]], 4.0, epsilon = 1e-10); // x1*x2^2
        assert_abs_diff_eq!(x_transformed[[0, 2]], 2.0, epsilon = 1e-10); // x1^2*x2
        assert_abs_diff_eq!(x_transformed[[0, 3]], 1.0, epsilon = 1e-10); // x1^3
    }

    #[test]
    fn test_homogeneous_polynomial_multinomial_coefficients() {
        let x = array![[1.0, 1.0]];

        let homo_poly = HomogeneousPolynomialFeatures::new(2)
            .coefficient_method(CoefficientMethod::Multinomial);
        let fitted = homo_poly.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        // Multinomial coefficients for degree 2:
        // x1^2: 2!/(2!*0!) = 1
        // x1*x2: 2!/(1!*1!) = 2
        // x2^2: 2!/(0!*2!) = 1
        assert_abs_diff_eq!(x_transformed[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x_transformed[[0, 1]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x_transformed[[0, 2]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_homogeneous_polynomial_l2_normalization() {
        let x = array![[3.0, 4.0]]; // This will give [9, 12, 16] before normalization

        let homo_poly =
            HomogeneousPolynomialFeatures::new(2).normalization(NormalizationMethod::L2);
        let fitted = homo_poly.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        // Check that the row has unit L2 norm
        let row_norm = (x_transformed.row(0).dot(&x_transformed.row(0))).sqrt();
        assert_abs_diff_eq!(row_norm, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_homogeneous_polynomial_l1_normalization() {
        let x = array![[2.0, 2.0]]; // This will give [4, 4, 4] before normalization

        let homo_poly =
            HomogeneousPolynomialFeatures::new(2).normalization(NormalizationMethod::L1);
        let fitted = homo_poly.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        // Check that the row has unit L1 norm
        let row_norm = x_transformed.row(0).mapv(|v| v.abs()).sum();
        assert_abs_diff_eq!(row_norm, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_homogeneous_polynomial_standard_normalization() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let homo_poly =
            HomogeneousPolynomialFeatures::new(2).normalization(NormalizationMethod::Standard);
        let fitted = homo_poly.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        // Check that each column has approximately zero mean and unit variance
        for j in 0..x_transformed.ncols() {
            let column = x_transformed.column(j);
            let mean = column.sum() / column.len() as Float;
            let variance = column.mapv(|v| (v - mean).powi(2)).sum() / (column.len() - 1) as Float;

            assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-10);
            assert_abs_diff_eq!(variance, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_homogeneous_polynomial_count_terms() {
        // Test term counting for various configurations
        assert_eq!(
            HomogeneousPolynomialFeatures::<Trained>::count_homogeneous_terms(2, 2, false),
            3
        );
        assert_eq!(
            HomogeneousPolynomialFeatures::<Trained>::count_homogeneous_terms(2, 3, false),
            6
        );
        assert_eq!(
            HomogeneousPolynomialFeatures::<Trained>::count_homogeneous_terms(3, 2, false),
            4
        );

        // Interaction only
        assert_eq!(
            HomogeneousPolynomialFeatures::<Trained>::count_homogeneous_terms(2, 3, true),
            3
        );
        assert_eq!(
            HomogeneousPolynomialFeatures::<Trained>::count_homogeneous_terms(3, 4, true),
            4
        );
    }

    #[test]
    fn test_homogeneous_polynomial_zero_degree() {
        let x = array![[1.0, 2.0]];
        let homo_poly = HomogeneousPolynomialFeatures::new(0);
        let result = homo_poly.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_homogeneous_polynomial_feature_mismatch() {
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let x_test = array![[1.0, 2.0, 3.0]]; // Different number of features

        let homo_poly = HomogeneousPolynomialFeatures::new(2);
        let fitted = homo_poly.fit(&x_train, &()).unwrap();
        let result = fitted.transform(&x_test);
        assert!(result.is_err());
    }

    #[test]
    fn test_homogeneous_polynomial_single_feature() {
        let x = array![[2.0], [3.0]];

        let homo_poly = HomogeneousPolynomialFeatures::new(3);
        let fitted = homo_poly.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        // For degree 3 with 1 feature: only x1^3
        assert_eq!(x_transformed.shape(), &[2, 1]);
        assert_abs_diff_eq!(x_transformed[[0, 0]], 8.0, epsilon = 1e-10); // 2^3
        assert_abs_diff_eq!(x_transformed[[1, 0]], 27.0, epsilon = 1e-10); // 3^3
    }

    #[test]
    fn test_homogeneous_polynomial_degree_1() {
        let x = array![[1.0, 2.0, 3.0]];

        let homo_poly = HomogeneousPolynomialFeatures::new(1);
        let fitted = homo_poly.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        // For degree 1: features in order [0,0,1], [0,1,0], [1,0,0]
        assert_eq!(x_transformed.shape(), &[1, 3]);
        assert_abs_diff_eq!(x_transformed[[0, 0]], 3.0, epsilon = 1e-10); // x3
        assert_abs_diff_eq!(x_transformed[[0, 1]], 2.0, epsilon = 1e-10); // x2
        assert_abs_diff_eq!(x_transformed[[0, 2]], 1.0, epsilon = 1e-10); // x1
    }

    #[test]
    fn test_homogeneous_polynomial_interaction_high_degree() {
        let x = array![[1.0, 2.0]];

        // Degree 3 interaction only with 2 features is impossible
        let homo_poly = HomogeneousPolynomialFeatures::new(3).interaction_only(true);
        let fitted = homo_poly.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        // Should have 0 features since we can't have 3 different features interacting
        assert_eq!(x_transformed.ncols(), 0);
    }
}
