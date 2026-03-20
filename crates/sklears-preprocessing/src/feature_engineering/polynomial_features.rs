//! Polynomial feature generation
//!
//! This module provides polynomial and interaction feature generation capabilities.

use scirs2_core::ndarray::Array2;
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Configuration for PolynomialFeatures
#[derive(Debug, Clone)]
pub struct PolynomialFeaturesConfig {
    /// The degree of the polynomial features
    pub degree: usize,
    /// Whether to include interaction terms
    pub interaction_only: bool,
    /// Whether to include bias (constant) term
    pub include_bias: bool,
    /// Feature name order for output features
    pub order: FeatureOrder,
    /// Maximum depth of interactions (None for no limit)
    pub interaction_depth: Option<usize>,
    /// Maximum number of features to include (for feature selection)
    pub max_features: Option<usize>,
    /// Whether to use regularized feature selection during expansion
    pub feature_selection: bool,
    /// Regularization parameter for feature selection (higher = more selective)
    pub alpha: Float,
}

impl Default for PolynomialFeaturesConfig {
    fn default() -> Self {
        Self {
            degree: 2,
            interaction_only: false,
            include_bias: true,
            order: FeatureOrder::C,
            interaction_depth: None,
            max_features: None,
            feature_selection: false,
            alpha: 1.0,
        }
    }
}

/// Feature ordering for polynomial features
#[derive(Debug, Clone, Copy, Default)]
pub enum FeatureOrder {
    /// C-style ordering (default)
    #[default]
    C,
    /// Fortran-style ordering
    F,
}

/// PolynomialFeatures generates polynomial and interaction features
///
/// Generate a new feature matrix consisting of all polynomial combinations
/// of the features with degree less than or equal to the specified degree.
/// For example, if an input sample is two dimensional and of the form [a, b],
/// the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].
#[derive(Debug, Clone)]
pub struct PolynomialFeatures<State = Untrained> {
    config: PolynomialFeaturesConfig,
    state: PhantomData<State>,
    // Fitted parameters
    n_features_in_: Option<usize>,
    n_output_features_: Option<usize>,
    powers_: Option<Array2<usize>>,
}

impl PolynomialFeatures<Untrained> {
    /// Create a new PolynomialFeatures
    pub fn new() -> Self {
        Self {
            config: PolynomialFeaturesConfig::default(),
            state: PhantomData,
            n_features_in_: None,
            n_output_features_: None,
            powers_: None,
        }
    }

    /// Create a new PolynomialFeatures with custom configuration
    pub fn with_config(config: PolynomialFeaturesConfig) -> Self {
        Self {
            config,
            state: PhantomData,
            n_features_in_: None,
            n_output_features_: None,
            powers_: None,
        }
    }

    /// Set the degree of polynomial features
    pub fn degree(mut self, degree: usize) -> Self {
        self.config.degree = degree;
        self
    }

    /// Set whether to include only interaction terms
    pub fn interaction_only(mut self, interaction_only: bool) -> Self {
        self.config.interaction_only = interaction_only;
        self
    }

    /// Set whether to include bias term
    pub fn include_bias(mut self, include_bias: bool) -> Self {
        self.config.include_bias = include_bias;
        self
    }

    /// Set maximum depth of interactions
    pub fn interaction_depth(mut self, depth: Option<usize>) -> Self {
        self.config.interaction_depth = depth;
        self
    }

    /// Set maximum number of features
    pub fn max_features(mut self, max_features: Option<usize>) -> Self {
        self.config.max_features = max_features;
        self
    }

    /// Enable feature selection
    pub fn with_feature_selection(mut self, alpha: Float) -> Self {
        self.config.feature_selection = true;
        self.config.alpha = alpha;
        self
    }

    /// Get all possible power combinations for given features and degree
    fn get_powers(&self, n_features: usize) -> Array2<usize> {
        let mut powers = Vec::new();

        // Add bias term if requested
        if self.config.include_bias {
            powers.push(vec![0; n_features]);
        }

        // Generate all combinations
        if self.config.interaction_only {
            // Only interaction terms (no pure powers)
            self.generate_interaction_powers(n_features, &mut powers);
        } else {
            // All polynomial terms up to degree
            self.generate_all_powers(n_features, &mut powers);
        }

        // Remove duplicates that may have been generated
        powers.sort();
        powers.dedup();

        // Apply proper polynomial feature ordering
        self.sort_powers(&mut powers, n_features);

        // Apply feature selection if enabled
        if self.config.feature_selection {
            powers = self.select_features(powers);
        }

        // Apply max_features limit
        if let Some(max_features) = self.config.max_features {
            powers.truncate(max_features);
        }

        let n_output = powers.len();
        let mut powers_array = Array2::zeros((n_output, n_features));

        for (i, power_vec) in powers.iter().enumerate() {
            for (j, &power) in power_vec.iter().enumerate() {
                powers_array[[i, j]] = power;
            }
        }

        powers_array
    }

    /// Generate all polynomial powers up to degree
    fn generate_all_powers(&self, n_features: usize, powers: &mut Vec<Vec<usize>>) {
        fn generate_recursive(
            current: Vec<usize>,
            n_features: usize,
            max_degree: usize,
            powers: &mut Vec<Vec<usize>>,
            interaction_depth: Option<usize>,
        ) {
            let current_degree: usize = current.iter().sum();
            if current_degree > max_degree {
                return;
            }

            // Check interaction depth constraint
            if let Some(max_depth) = interaction_depth {
                let non_zero_count = current.iter().filter(|&&x| x > 0).count();
                if non_zero_count > max_depth {
                    return;
                }
            }

            if current_degree > 0 {
                powers.push(current.clone());
            }

            for i in 0..n_features {
                let mut next = current.clone();
                next[i] += 1;
                if next.iter().sum::<usize>() <= max_degree {
                    generate_recursive(next, n_features, max_degree, powers, interaction_depth);
                }
            }
        }

        let initial = vec![0; n_features];
        generate_recursive(
            initial,
            n_features,
            self.config.degree,
            powers,
            self.config.interaction_depth,
        );
    }

    /// Generate only interaction powers (no pure powers)
    fn generate_interaction_powers(&self, n_features: usize, powers: &mut Vec<Vec<usize>>) {
        fn generate_interactions(
            current: Vec<usize>,
            start_idx: usize,
            n_features: usize,
            max_degree: usize,
            powers: &mut Vec<Vec<usize>>,
            interaction_depth: Option<usize>,
        ) {
            let current_degree: usize = current.iter().sum();

            if current_degree > max_degree {
                return;
            }

            // Check interaction depth constraint
            if let Some(max_depth) = interaction_depth {
                let non_zero_count = current.iter().filter(|&&x| x > 0).count();
                if non_zero_count > max_depth {
                    return;
                }
            }

            // Must have at least 2 features involved for interactions
            let non_zero_count = current.iter().filter(|&&x| x > 0).count();
            if current_degree > 0 && non_zero_count >= 2 {
                powers.push(current.clone());
            }

            for i in start_idx..n_features {
                let mut next = current.clone();
                next[i] += 1;
                if next.iter().sum::<usize>() <= max_degree {
                    generate_interactions(
                        next,
                        i,
                        n_features,
                        max_degree,
                        powers,
                        interaction_depth,
                    );
                }
            }
        }

        let initial = vec![0; n_features];
        generate_interactions(
            initial,
            0,
            n_features,
            self.config.degree,
            powers,
            self.config.interaction_depth,
        );
    }

    /// Select features based on regularization (placeholder implementation)
    fn select_features(&self, mut powers: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        // Simple selection based on alpha parameter
        // Higher alpha means more selective (fewer features)
        if self.config.alpha > 1.0 {
            let keep_ratio = 1.0 / self.config.alpha;
            let keep_count = ((powers.len() as f64) * keep_ratio).ceil() as usize;
            powers.truncate(keep_count);
        }
        powers
    }

    /// Sort powers according to polynomial feature ordering convention
    fn sort_powers(&self, powers: &mut Vec<Vec<usize>>, _n_features: usize) {
        // Sort by degree first, then by reverse lexicographic order within each degree
        // This matches scikit-learn's default ordering
        powers.sort_by(|a, b| {
            let degree_a: usize = a.iter().sum();
            let degree_b: usize = b.iter().sum();

            // First sort by degree
            degree_a.cmp(&degree_b).then_with(|| {
                // Within same degree, sort by reverse lexicographic order
                // For scikit-learn compatibility: [1,0] comes before [0,1]
                b.cmp(a)
            })
        });
    }
}

/// Static methods for PolynomialFeatures
impl<State> PolynomialFeatures<State> {
    /// Calculate number of output features for given input features and degree
    pub fn get_n_output_features(
        n_features: usize,
        degree: usize,
        include_bias: bool,
        interaction_only: bool,
    ) -> usize {
        use std::collections::HashMap;

        fn binomial_coefficient(n: usize, k: usize) -> usize {
            if k > n {
                return 0;
            }
            if k == 0 || k == n {
                return 1;
            }

            let mut memo = HashMap::new();
            fn binomial_memo(
                n: usize,
                k: usize,
                memo: &mut HashMap<(usize, usize), usize>,
            ) -> usize {
                if let Some(&result) = memo.get(&(n, k)) {
                    return result;
                }

                let result = if k > n {
                    0
                } else if k == 0 || k == n {
                    1
                } else {
                    binomial_memo(n - 1, k - 1, memo) + binomial_memo(n - 1, k, memo)
                };

                memo.insert((n, k), result);
                result
            }

            binomial_memo(n, k, &mut memo)
        }

        let mut count = 0;

        if include_bias {
            count += 1;
        }

        if interaction_only {
            // Only interaction terms (degree >= 2)
            for d in 2..=degree {
                count += binomial_coefficient(n_features, d);
            }
        } else {
            // All polynomial terms
            for d in 1..=degree {
                count += binomial_coefficient(n_features + d - 1, d);
            }
        }

        count
    }
}

impl PolynomialFeatures<Trained> {
    /// Get the number of output features
    pub fn n_output_features(&self) -> usize {
        self.n_output_features_.unwrap_or(0)
    }

    /// Get the power matrix
    pub fn powers(&self) -> Option<&Array2<usize>> {
        self.powers_.as_ref()
    }

    /// Get feature names (if input feature names are provided)
    pub fn get_feature_names(&self, input_features: Option<&[String]>) -> Vec<String> {
        let powers = match &self.powers_ {
            Some(p) => p,
            None => return Vec::new(),
        };

        let n_features = self.n_features_in_.unwrap_or(0);
        let feature_names = match input_features {
            Some(names) => names.to_vec(),
            None => (0..n_features).map(|i| format!("x{}", i)).collect(),
        };

        let mut output_names = Vec::new();

        for i in 0..powers.nrows() {
            let power_row = powers.row(i);
            let mut name_parts = Vec::new();

            for (j, &power) in power_row.iter().enumerate() {
                if power == 0 {
                    continue;
                } else if power == 1 {
                    name_parts.push(feature_names[j].clone());
                } else {
                    name_parts.push(format!("{}^{}", feature_names[j], power));
                }
            }

            let feature_name = if name_parts.is_empty() {
                "1".to_string() // Bias term
            } else {
                name_parts.join(" ")
            };

            output_names.push(feature_name);
        }

        output_names
    }
}

impl Default for PolynomialFeatures<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<Array2<Float>, ()> for PolynomialFeatures<Untrained> {
    type Fitted = PolynomialFeatures<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let n_features = x.ncols();

        if n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "Input must have at least one feature".to_string(),
            ));
        }

        if self.config.degree == 0 {
            return Err(SklearsError::InvalidInput(
                "Degree must be at least 1".to_string(),
            ));
        }

        let powers = self.get_powers(n_features);
        let n_output_features = powers.nrows();

        Ok(PolynomialFeatures::<Trained> {
            config: self.config,
            state: PhantomData,
            n_features_in_: Some(n_features),
            n_output_features_: Some(n_output_features),
            powers_: Some(powers),
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for PolynomialFeatures<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let powers = self
            .powers_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform".to_string(),
            })?;

        let n_samples = x.nrows();
        let n_features_in = x.ncols();
        let n_output_features = powers.nrows();

        if n_features_in != self.n_features_in_.unwrap_or(0) {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features_in_.unwrap_or(0),
                n_features_in
            )));
        }

        let mut output = Array2::zeros((n_samples, n_output_features));

        // FIXME: SIMD implementation disabled for compilation
        // Use CPU fallback for now
        self.transform_cpu(x, powers, &mut output)?;

        Ok(output)
    }
}

impl PolynomialFeatures<Trained> {
    /// CPU implementation of polynomial feature transformation
    fn transform_cpu(
        &self,
        x: &Array2<Float>,
        powers: &Array2<usize>,
        output: &mut Array2<Float>,
    ) -> Result<()> {
        let n_samples = x.nrows();
        let n_output_features = powers.nrows();

        for sample_idx in 0..n_samples {
            let sample = x.row(sample_idx);

            for feature_idx in 0..n_output_features {
                let power_row = powers.row(feature_idx);
                let mut feature_value = 1.0;

                for (input_feature_idx, &power) in power_row.iter().enumerate() {
                    if power > 0 {
                        let base_value = sample[input_feature_idx];
                        feature_value *= base_value.powi(power as i32);
                    }
                }

                output[[sample_idx, feature_idx]] = feature_value;
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
    use scirs2_core::ndarray::array;

    #[test]
    fn test_polynomial_features_basic() -> Result<()> {
        let x = array![[1.0, 2.0], [2.0, 3.0]];
        let poly = PolynomialFeatures::new().degree(2);

        let fitted = poly.fit(&x, &())?;
        let transformed = fitted.transform(&x)?;

        // Expected features: [1, x0, x1, x0^2, x0*x1, x1^2]
        assert_eq!(transformed.ncols(), 6);

        // Check first sample: [1, 1, 2, 1, 2, 4]
        assert_abs_diff_eq!(transformed[[0, 0]], 1.0, epsilon = 1e-10); // bias
        assert_abs_diff_eq!(transformed[[0, 1]], 1.0, epsilon = 1e-10); // x0
        assert_abs_diff_eq!(transformed[[0, 2]], 2.0, epsilon = 1e-10); // x1
        assert_abs_diff_eq!(transformed[[0, 3]], 1.0, epsilon = 1e-10); // x0^2
        assert_abs_diff_eq!(transformed[[0, 4]], 2.0, epsilon = 1e-10); // x0*x1
        assert_abs_diff_eq!(transformed[[0, 5]], 4.0, epsilon = 1e-10); // x1^2

        Ok(())
    }

    #[test]
    fn test_polynomial_features_interaction_only() -> Result<()> {
        let x = array![[1.0, 2.0], [2.0, 3.0]];
        let poly = PolynomialFeatures::new().degree(2).interaction_only(true);

        let fitted = poly.fit(&x, &())?;
        let transformed = fitted.transform(&x)?;

        // Expected features: [1, x0*x1] (only bias and interaction)
        assert_eq!(transformed.ncols(), 2);

        Ok(())
    }

    #[test]
    fn test_polynomial_features_no_bias() -> Result<()> {
        let x = array![[1.0, 2.0], [2.0, 3.0]];
        let poly = PolynomialFeatures::new().degree(2).include_bias(false);

        let fitted = poly.fit(&x, &())?;
        let transformed = fitted.transform(&x)?;

        // Expected features: [x0, x1, x0^2, x0*x1, x1^2] (no bias)
        assert_eq!(transformed.ncols(), 5);

        Ok(())
    }

    #[test]
    fn test_feature_names() -> Result<()> {
        let x = array![[1.0, 2.0]];
        let poly = PolynomialFeatures::new().degree(2);

        let fitted = poly.fit(&x, &())?;
        let feature_names = fitted.get_feature_names(Some(&["A".to_string(), "B".to_string()]));

        assert_eq!(feature_names.len(), 6);
        assert_eq!(feature_names[0], "1"); // bias
        assert_eq!(feature_names[1], "A"); // A
        assert_eq!(feature_names[2], "B"); // B

        Ok(())
    }

    #[test]
    fn test_n_output_features() {
        // Degree 2, 2 features, with bias
        assert_eq!(
            PolynomialFeatures::<()>::get_n_output_features(2, 2, true, false),
            6
        );

        // Degree 2, 2 features, no bias
        assert_eq!(
            PolynomialFeatures::<()>::get_n_output_features(2, 2, false, false),
            5
        );

        // Degree 2, 2 features, interaction only, with bias
        assert_eq!(
            PolynomialFeatures::<()>::get_n_output_features(2, 2, true, true),
            2
        );
    }
}
