//! Polynomial feature generation for linear models
//!
//! This module provides functionality to generate polynomial and interaction features
//! from input data, enabling linear models to capture non-linear relationships.
//! It supports various polynomial degrees and interaction configurations.

use sklears_core::error::SklearsError;
use std::collections::HashMap;

/// Configuration for polynomial feature generation
#[derive(Debug, Clone)]
pub struct PolynomialConfig {
    /// Maximum degree of polynomial features to generate
    pub degree: usize,
    /// Whether to include interaction terms
    pub include_interactions: bool,
    /// Whether to include bias/intercept term (degree 0)
    pub include_bias: bool,
    /// Maximum number of features to interact (to control complexity)
    pub max_interaction_features: Option<usize>,
    /// Specific feature indices to exclude from polynomial expansion
    pub exclude_features: Vec<usize>,
    /// Whether to include only specific degrees
    pub only_degrees: Option<Vec<usize>>,
    /// Memory limit for feature generation (in number of features)
    pub max_output_features: Option<usize>,
}

impl Default for PolynomialConfig {
    fn default() -> Self {
        Self {
            degree: 2,
            include_interactions: true,
            include_bias: true,
            max_interaction_features: None,
            exclude_features: vec![],
            only_degrees: None,
            max_output_features: Some(10000), // Limit to 10k features by default
        }
    }
}

/// Information about generated polynomial features
#[derive(Debug, Clone)]
pub struct FeatureInfo {
    /// Original feature indices that compose this polynomial feature
    pub feature_indices: Vec<usize>,
    /// Powers of each original feature in this polynomial term
    pub powers: Vec<usize>,
    /// Total degree of this polynomial term
    pub degree: usize,
    /// Human-readable description of the feature
    pub description: String,
}

impl FeatureInfo {
    fn new(feature_indices: Vec<usize>, powers: Vec<usize>) -> Self {
        let degree = powers.iter().sum();
        let description = if feature_indices.is_empty() {
            "bias".to_string()
        } else {
            let terms: Vec<String> = feature_indices
                .iter()
                .zip(powers.iter())
                .filter(|(_, &power)| power > 0)
                .map(|(&idx, &power)| {
                    if power == 1 {
                        format!("x{}", idx)
                    } else {
                        format!("x{}^{}", idx, power)
                    }
                })
                .collect();
            terms.join(" * ")
        };

        Self {
            feature_indices,
            powers,
            degree,
            description,
        }
    }
}

/// Polynomial feature generator
#[derive(Debug, Clone)]
pub struct PolynomialFeatures {
    /// Configuration
    pub config: PolynomialConfig,
    /// Information about generated features
    pub feature_info: Vec<FeatureInfo>,
    /// Number of input features
    pub n_input_features: Option<usize>,
    /// Number of output features
    pub n_output_features: Option<usize>,
    /// Whether the transformer has been fitted
    pub fitted: bool,
    /// Mapping from feature combinations to output indices
    feature_map: HashMap<Vec<usize>, usize>,
}

impl PolynomialFeatures {
    /// Create a new polynomial feature generator
    pub fn new(config: PolynomialConfig) -> Self {
        Self {
            config,
            feature_info: vec![],
            n_input_features: None,
            n_output_features: None,
            fitted: false,
            feature_map: HashMap::new(),
        }
    }

    /// Create a simple polynomial feature generator with specified degree
    pub fn with_degree(degree: usize) -> Self {
        Self::new(PolynomialConfig {
            degree,
            ..Default::default()
        })
    }

    /// Create a polynomial feature generator with interactions only
    pub fn interactions_only(degree: usize) -> Self {
        Self::new(PolynomialConfig {
            degree,
            include_interactions: true,
            include_bias: false,
            only_degrees: Some((2..=degree).collect()),
            ..Default::default()
        })
    }

    /// Fit the polynomial feature generator (determine output features)
    pub fn fit(&mut self, x: &[Vec<f64>]) -> Result<(), SklearsError> {
        if x.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        let n_features = x[0].len();
        self.n_input_features = Some(n_features);

        // Validate exclude_features
        for &idx in &self.config.exclude_features {
            if idx >= n_features {
                return Err(SklearsError::InvalidInput(format!(
                    "Exclude feature index {} is out of bounds for {} features",
                    idx, n_features
                )));
            }
        }

        // Generate all polynomial feature combinations
        let mut feature_info = vec![];
        let mut feature_map = HashMap::new();
        let mut output_idx = 0;

        // Add bias term if requested
        if self.config.include_bias {
            let info = FeatureInfo::new(vec![], vec![]);
            feature_map.insert(vec![], output_idx);
            feature_info.push(info);
            output_idx += 1;
        }

        // Generate polynomial features for each degree
        let degrees_to_generate = if let Some(ref only_degrees) = self.config.only_degrees {
            only_degrees.clone()
        } else {
            (1..=self.config.degree).collect()
        };

        for degree in degrees_to_generate {
            let combinations = self.generate_combinations(n_features, degree)?;

            for combination in combinations {
                if let Some(max_features) = self.config.max_output_features {
                    if output_idx >= max_features {
                        log::warn!("Reached maximum output features limit: {}", max_features);
                        break;
                    }
                }

                let (feature_indices, powers) =
                    self.combination_to_indices_and_powers(&combination, n_features);
                let info = FeatureInfo::new(feature_indices.clone(), powers);

                feature_map.insert(combination, output_idx);
                feature_info.push(info);
                output_idx += 1;
            }
        }

        self.feature_info = feature_info;
        self.n_output_features = Some(output_idx);
        self.feature_map = feature_map;
        self.fitted = true;

        log::info!(
            "Generated {} polynomial features from {} input features",
            output_idx,
            n_features
        );

        Ok(())
    }

    /// Transform input data to polynomial features
    pub fn transform(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, SklearsError> {
        if !self.fitted {
            return Err(SklearsError::NotFitted {
                operation: "transform".to_string(),
            });
        }

        if x.is_empty() {
            return Ok(vec![]);
        }

        let n_input_features = self.n_input_features.unwrap();
        let n_output_features = self.n_output_features.unwrap();

        // Validate input dimensions
        for (i, row) in x.iter().enumerate() {
            if row.len() != n_input_features {
                return Err(SklearsError::InvalidInput(format!(
                    "Input row {} has {} features, expected {}",
                    i,
                    row.len(),
                    n_input_features
                )));
            }
        }

        let mut result = vec![vec![0.0; n_output_features]; x.len()];

        // Transform each sample
        for (sample_idx, input_row) in x.iter().enumerate() {
            let output_row = &mut result[sample_idx];

            // Generate polynomial features for this sample
            for (feature_idx, feature_info) in self.feature_info.iter().enumerate() {
                let value = if feature_info.feature_indices.is_empty() {
                    // Bias term
                    1.0
                } else {
                    // Compute polynomial term
                    let mut product = 1.0;
                    for (&orig_idx, &power) in feature_info
                        .feature_indices
                        .iter()
                        .zip(feature_info.powers.iter())
                    {
                        if power > 0 {
                            product *= input_row[orig_idx].powi(power as i32);
                        }
                    }
                    product
                };

                output_row[feature_idx] = value;
            }
        }

        Ok(result)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, SklearsError> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Get names/descriptions of all generated features
    pub fn get_feature_names(&self) -> Vec<String> {
        self.feature_info
            .iter()
            .map(|info| info.description.clone())
            .collect()
    }

    /// Get information about a specific output feature
    pub fn get_feature_info(&self, feature_idx: usize) -> Option<&FeatureInfo> {
        self.feature_info.get(feature_idx)
    }

    /// Get the number of output features that would be generated
    pub fn get_n_output_features(&self) -> Option<usize> {
        self.n_output_features
    }

    /// Estimate the number of output features before fitting
    pub fn estimate_n_output_features(&self, n_input_features: usize) -> usize {
        let mut count = 0;

        // Count bias term
        if self.config.include_bias {
            count += 1;
        }

        // Count polynomial features for each degree
        let degrees_to_generate = if let Some(ref only_degrees) = self.config.only_degrees {
            only_degrees.clone()
        } else {
            (1..=self.config.degree).collect()
        };

        for degree in degrees_to_generate {
            count += self.count_combinations(n_input_features, degree);
        }

        if let Some(max_features) = self.config.max_output_features {
            count.min(max_features)
        } else {
            count
        }
    }

    /// Check if the feature generation is computationally feasible
    pub fn check_feasibility(&self, n_input_features: usize) -> Result<(), SklearsError> {
        let estimated_features = self.estimate_n_output_features(n_input_features);

        if estimated_features > 100000 {
            return Err(SklearsError::InvalidInput(format!(
                "Polynomial feature generation would create {} features, which may be too many. \
                 Consider reducing the degree or setting max_output_features",
                estimated_features
            )));
        }

        if self.config.degree > 10 {
            return Err(SklearsError::InvalidInput(
                "Polynomial degree > 10 is not recommended due to numerical instability"
                    .to_string(),
            ));
        }

        Ok(())
    }

    fn generate_combinations(
        &self,
        n_features: usize,
        degree: usize,
    ) -> Result<Vec<Vec<usize>>, SklearsError> {
        let mut combinations = vec![];

        if degree == 0 {
            return Ok(combinations);
        }

        // Generate all combinations with replacement of the specified degree
        let available_features: Vec<usize> = (0..n_features)
            .filter(|&i| !self.config.exclude_features.contains(&i))
            .collect();

        self.generate_combinations_recursive(
            &available_features,
            degree,
            &mut vec![],
            &mut combinations,
        );

        Ok(combinations)
    }

    fn generate_combinations_recursive(
        &self,
        available_features: &[usize],
        remaining_degree: usize,
        current_combination: &mut Vec<usize>,
        all_combinations: &mut Vec<Vec<usize>>,
    ) {
        if remaining_degree == 0 {
            // Check interaction constraints
            if self.should_include_combination(current_combination) {
                let mut sorted_combination = current_combination.clone();
                sorted_combination.sort_unstable();
                all_combinations.push(sorted_combination);
            }
            return;
        }

        let start_idx = current_combination
            .last()
            .map(|&last| {
                available_features
                    .iter()
                    .position(|&x| x == last)
                    .unwrap_or(0)
            })
            .unwrap_or(0);

        for (i, &feature) in available_features.iter().enumerate().skip(start_idx) {
            current_combination.push(feature);
            self.generate_combinations_recursive(
                available_features,
                remaining_degree - 1,
                current_combination,
                all_combinations,
            );
            current_combination.pop();
        }
    }

    fn should_include_combination(&self, combination: &[usize]) -> bool {
        // Check if interactions are allowed
        if !self.config.include_interactions {
            let unique_features: std::collections::HashSet<_> = combination.iter().collect();
            if unique_features.len() > 1 {
                return false;
            }
        }

        // Check max interaction features constraint
        if let Some(max_interact) = self.config.max_interaction_features {
            let unique_features: std::collections::HashSet<_> = combination.iter().collect();
            if unique_features.len() > max_interact {
                return false;
            }
        }

        true
    }

    fn combination_to_indices_and_powers(
        &self,
        combination: &[usize],
        n_features: usize,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut powers = vec![0; n_features];
        let mut unique_indices = vec![];

        for &feature_idx in combination {
            powers[feature_idx] += 1;
        }

        for (idx, &power) in powers.iter().enumerate() {
            if power > 0 {
                unique_indices.push(idx);
            }
        }

        (unique_indices, powers)
    }

    fn count_combinations(&self, n_features: usize, degree: usize) -> usize {
        if degree == 0 {
            return 0;
        }

        let available_features = n_features - self.config.exclude_features.len();

        if !self.config.include_interactions {
            // Only pure powers, not interactions
            available_features
        } else {
            // Combinations with replacement: C(n + k - 1, k)
            self.binomial_coefficient(available_features + degree - 1, degree)
        }
    }

    fn binomial_coefficient(&self, n: usize, k: usize) -> usize {
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

/// Builder for polynomial features configuration
pub struct PolynomialFeaturesBuilder {
    config: PolynomialConfig,
}

impl PolynomialFeaturesBuilder {
    pub fn new() -> Self {
        Self {
            config: PolynomialConfig::default(),
        }
    }

    pub fn degree(mut self, degree: usize) -> Self {
        self.config.degree = degree;
        self
    }

    pub fn include_interactions(mut self, include: bool) -> Self {
        self.config.include_interactions = include;
        self
    }

    pub fn include_bias(mut self, include: bool) -> Self {
        self.config.include_bias = include;
        self
    }

    pub fn max_interaction_features(mut self, max: Option<usize>) -> Self {
        self.config.max_interaction_features = max;
        self
    }

    pub fn exclude_features(mut self, features: Vec<usize>) -> Self {
        self.config.exclude_features = features;
        self
    }

    pub fn only_degrees(mut self, degrees: Option<Vec<usize>>) -> Self {
        self.config.only_degrees = degrees;
        self
    }

    pub fn max_output_features(mut self, max: Option<usize>) -> Self {
        self.config.max_output_features = max;
        self
    }

    pub fn build(self) -> PolynomialFeatures {
        PolynomialFeatures::new(self.config)
    }
}

impl Default for PolynomialFeaturesBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Utilities for polynomial feature analysis
pub struct PolynomialUtils;

impl PolynomialUtils {
    /// Calculate memory usage estimate for polynomial features
    pub fn estimate_memory_usage(
        n_samples: usize,
        n_input_features: usize,
        degree: usize,
        include_interactions: bool,
    ) -> usize {
        let mut n_output_features = 1; // bias

        for d in 1..=degree {
            if include_interactions {
                // Combinations with replacement
                n_output_features += Self::binomial_coefficient(n_input_features + d - 1, d);
            } else {
                // Only pure powers
                n_output_features += n_input_features;
            }
        }

        n_samples * n_output_features * std::mem::size_of::<f64>()
    }

    /// Check if polynomial degree is reasonable for given input size
    pub fn validate_polynomial_config(
        n_input_features: usize,
        degree: usize,
        include_interactions: bool,
    ) -> Result<(), SklearsError> {
        let memory_estimate =
            Self::estimate_memory_usage(1000, n_input_features, degree, include_interactions);
        let memory_gb = memory_estimate as f64 / (1024.0 * 1024.0 * 1024.0);

        if memory_gb > 1.0 {
            return Err(SklearsError::InvalidInput(format!(
                "Polynomial features with degree {} would use approximately {:.2} GB for 1000 samples",
                degree, memory_gb
            )));
        }

        if degree > 6 && include_interactions {
            return Err(SklearsError::InvalidInput(
                "High degree polynomials with interactions can cause numerical instability"
                    .to_string(),
            ));
        }

        Ok(())
    }

    fn binomial_coefficient(n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        if k == 0 || k == n {
            return 1;
        }

        let k = k.min(n - k);
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
    use approx::assert_relative_eq;

    #[test]
    fn test_polynomial_degree_2() {
        let data = vec![vec![2.0, 3.0], vec![4.0, 5.0]];

        let mut poly = PolynomialFeatures::with_degree(2);
        let transformed = poly.fit_transform(&data).unwrap();

        // Features should be: [1, x0, x1, x0^2, x0*x1, x1^2]
        assert_eq!(transformed[0].len(), 6);

        // Check first sample: [1, 2, 3, 4, 6, 9]
        assert_relative_eq!(transformed[0][0], 1.0, epsilon = 1e-10); // bias
        assert_relative_eq!(transformed[0][1], 2.0, epsilon = 1e-10); // x0
        assert_relative_eq!(transformed[0][2], 3.0, epsilon = 1e-10); // x1
        assert_relative_eq!(transformed[0][3], 4.0, epsilon = 1e-10); // x0^2
        assert_relative_eq!(transformed[0][4], 6.0, epsilon = 1e-10); // x0*x1
        assert_relative_eq!(transformed[0][5], 9.0, epsilon = 1e-10); // x1^2

        // Check second sample: [1, 4, 5, 16, 20, 25]
        assert_relative_eq!(transformed[1][0], 1.0, epsilon = 1e-10); // bias
        assert_relative_eq!(transformed[1][1], 4.0, epsilon = 1e-10); // x0
        assert_relative_eq!(transformed[1][2], 5.0, epsilon = 1e-10); // x1
        assert_relative_eq!(transformed[1][3], 16.0, epsilon = 1e-10); // x0^2
        assert_relative_eq!(transformed[1][4], 20.0, epsilon = 1e-10); // x0*x1
        assert_relative_eq!(transformed[1][5], 25.0, epsilon = 1e-10); // x1^2
    }

    #[test]
    fn test_interactions_only() {
        let data = vec![vec![2.0, 3.0]];

        let mut poly = PolynomialFeatures::interactions_only(2);
        let transformed = poly.fit_transform(&data).unwrap();

        // Should only have interaction term x0*x1, no bias or linear terms
        assert_eq!(transformed[0].len(), 1);
        assert_relative_eq!(transformed[0][0], 6.0, epsilon = 1e-10); // x0*x1
    }

    #[test]
    fn test_no_interactions() {
        let data = vec![vec![2.0, 3.0]];

        let config = PolynomialConfig {
            degree: 2,
            include_interactions: false,
            include_bias: true,
            ..Default::default()
        };
        let mut poly = PolynomialFeatures::new(config);
        let transformed = poly.fit_transform(&data).unwrap();

        // Features should be: [1, x0, x1, x0^2, x1^2] (no x0*x1)
        assert_eq!(transformed[0].len(), 5);
        assert_relative_eq!(transformed[0][0], 1.0, epsilon = 1e-10); // bias
        assert_relative_eq!(transformed[0][1], 2.0, epsilon = 1e-10); // x0
        assert_relative_eq!(transformed[0][2], 3.0, epsilon = 1e-10); // x1
        assert_relative_eq!(transformed[0][3], 4.0, epsilon = 1e-10); // x0^2
        assert_relative_eq!(transformed[0][4], 9.0, epsilon = 1e-10); // x1^2
    }

    #[test]
    fn test_exclude_features() {
        let data = vec![vec![2.0, 3.0, 4.0]];

        let config = PolynomialConfig {
            degree: 2,
            exclude_features: vec![1], // Exclude second feature
            ..Default::default()
        };
        let mut poly = PolynomialFeatures::new(config);
        let transformed = poly.fit_transform(&data).unwrap();

        // Should have features: [1, x0, x2, x0^2, x0*x2, x2^2]
        // Feature x1 and its combinations should be excluded
        assert_eq!(transformed[0].len(), 6);

        let feature_names = poly.get_feature_names();
        assert!(!feature_names.iter().any(|name| name.contains("x1")));
    }

    #[test]
    fn test_feature_names() {
        let data = vec![vec![1.0, 2.0]];

        let mut poly = PolynomialFeatures::with_degree(2);
        poly.fit_transform(&data).unwrap();

        let names = poly.get_feature_names();
        assert_eq!(names.len(), 6);
        assert_eq!(names[0], "bias");
        assert_eq!(names[1], "x0");
        assert_eq!(names[2], "x1");
        assert_eq!(names[3], "x0^2");
        assert_eq!(names[4], "x0 * x1");
        assert_eq!(names[5], "x1^2");
    }

    #[test]
    fn test_builder_pattern() {
        let poly = PolynomialFeaturesBuilder::new()
            .degree(3)
            .include_bias(false)
            .max_interaction_features(Some(2))
            .exclude_features(vec![0])
            .build();

        assert_eq!(poly.config.degree, 3);
        assert!(!poly.config.include_bias);
        assert_eq!(poly.config.max_interaction_features, Some(2));
        assert_eq!(poly.config.exclude_features, vec![0]);
    }

    #[test]
    fn test_feature_info() {
        let data = vec![vec![1.0, 2.0]];

        let mut poly = PolynomialFeatures::with_degree(2);
        poly.fit_transform(&data).unwrap();

        let info = poly.get_feature_info(4).unwrap(); // x0*x1 term
        assert_eq!(info.feature_indices, vec![0, 1]);
        assert_eq!(info.powers, vec![1, 1, 0]); // Note: powers vector has length = n_input_features
        assert_eq!(info.degree, 2);
        assert_eq!(info.description, "x0 * x1");
    }

    #[test]
    fn test_memory_estimation() {
        let memory = PolynomialUtils::estimate_memory_usage(1000, 10, 2, true);
        assert!(memory > 0);

        let validation = PolynomialUtils::validate_polynomial_config(5, 2, true);
        assert!(validation.is_ok());
    }

    #[test]
    fn test_feasibility_check() {
        let config = PolynomialConfig {
            degree: 15, // Very high degree
            ..Default::default()
        };
        let poly = PolynomialFeatures::new(config);

        let result = poly.check_feasibility(10);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_data() {
        let data: Vec<Vec<f64>> = vec![];
        let mut poly = PolynomialFeatures::with_degree(2);

        let result = poly.fit(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_feature() {
        let data = vec![vec![3.0], vec![4.0]];

        let mut poly = PolynomialFeatures::with_degree(3);
        let transformed = poly.fit_transform(&data).unwrap();

        // Features should be: [1, x0, x0^2, x0^3]
        assert_eq!(transformed[0].len(), 4);
        assert_relative_eq!(transformed[0][0], 1.0, epsilon = 1e-10); // bias
        assert_relative_eq!(transformed[0][1], 3.0, epsilon = 1e-10); // x0
        assert_relative_eq!(transformed[0][2], 9.0, epsilon = 1e-10); // x0^2
        assert_relative_eq!(transformed[0][3], 27.0, epsilon = 1e-10); // x0^3
    }
}
