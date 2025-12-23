//! Extensible feature generation framework
//!
//! This module provides a flexible system for creating and composing feature
//! generation methods, making it easy to extend with new techniques.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::seeded_rng;
use sklears_core::error::SklearsError;

/// Feature generator trait
pub trait FeatureGenerator: Send + Sync {
    /// Generate features from input data
    fn generate(&self, data: &Array2<f64>) -> Result<Array2<f64>, SklearsError>;

    /// Get the output dimension
    fn output_dim(&self) -> usize;

    /// Get generator name
    fn name(&self) -> &str;

    /// Check if generator is stateful (needs fitting)
    fn is_stateful(&self) -> bool {
        false
    }

    /// Fit the generator if stateful
    fn fit_generator(&mut self, _data: &Array2<f64>) -> Result<(), SklearsError> {
        Ok(())
    }
}

/// Random Fourier feature generator
#[derive(Debug, Clone)]
pub struct RandomFourierGenerator {
    /// Number of components
    pub n_components: usize,
    /// Gamma parameter
    pub gamma: f64,
    /// Random weights (fitted)
    weights: Option<Array2<f64>>,
    /// Random offset (fitted)
    offset: Option<Array1<f64>>,
    /// Random seed
    pub random_state: Option<u64>,
}

impl RandomFourierGenerator {
    /// Create a new Random Fourier generator
    pub fn new(n_components: usize, gamma: f64, random_state: Option<u64>) -> Self {
        Self {
            n_components,
            gamma,
            weights: None,
            offset: None,
            random_state,
        }
    }

    /// Initialize weights and offsets
    fn initialize(&mut self, n_features: usize) -> Result<(), SklearsError> {
        use scirs2_core::random::StandardNormal;

        let mut rng = seeded_rng(self.random_state.unwrap_or(42));

        // Sample weights from N(0, gamma)
        let mut weights = Array2::zeros((n_features, self.n_components));
        for elem in weights.iter_mut() {
            *elem = rng.sample::<f64, _>(StandardNormal) * self.gamma.sqrt();
        }

        // Sample offset from Uniform(0, 2π)
        let mut offset = Array1::zeros(self.n_components);
        for elem in offset.iter_mut() {
            *elem = rng.gen_range(0.0..(2.0 * std::f64::consts::PI));
        }

        self.weights = Some(weights);
        self.offset = Some(offset);

        Ok(())
    }
}

impl FeatureGenerator for RandomFourierGenerator {
    fn generate(&self, data: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "RandomFourierGenerator must be fitted before generating features"
                    .to_string(),
            })?;

        let offset = self.offset.as_ref().unwrap();

        // Compute X @ W
        let projection = data.dot(weights);

        // Apply cos(X @ W + b)
        let scale = (2.0 / self.n_components as f64).sqrt();
        let features = projection.mapv(|x| x + offset[0]).mapv(|x| scale * x.cos());

        Ok(features)
    }

    fn output_dim(&self) -> usize {
        self.n_components
    }

    fn name(&self) -> &str {
        "RandomFourierFeatures"
    }

    fn is_stateful(&self) -> bool {
        true
    }

    fn fit_generator(&mut self, data: &Array2<f64>) -> Result<(), SklearsError> {
        let (_, n_features) = data.dim();
        self.initialize(n_features)
    }
}

/// Polynomial feature generator
#[derive(Debug, Clone)]
pub struct PolynomialGenerator {
    /// Polynomial degree
    pub degree: usize,
    /// Include bias term
    pub include_bias: bool,
    /// Interaction only (no powers)
    pub interaction_only: bool,
}

impl PolynomialGenerator {
    /// Create a new polynomial generator
    pub fn new(degree: usize, include_bias: bool, interaction_only: bool) -> Self {
        Self {
            degree,
            include_bias,
            interaction_only,
        }
    }

    /// Calculate number of output features
    fn calculate_n_output_features(&self, n_input_features: usize) -> usize {
        if self.interaction_only {
            // Combinations with repetition
            let mut count = if self.include_bias { 1 } else { 0 };
            count += n_input_features; // degree 1

            for d in 2..=self.degree {
                // C(n + d - 1, d)
                let mut comb = 1;
                for i in 0..d {
                    comb = comb * (n_input_features + d - 1 - i) / (i + 1);
                }
                count += comb;
            }
            count
        } else {
            // All monomials up to degree
            let mut count = if self.include_bias { 1 } else { 0 };
            for d in 1..=self.degree {
                // Number of monomials of degree d in n variables
                let mut monomials = 1;
                for i in 0..d {
                    monomials = monomials * (n_input_features + d - 1 - i) / (i + 1);
                }
                count += monomials;
            }
            count
        }
    }
}

impl FeatureGenerator for PolynomialGenerator {
    fn generate(&self, data: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        let (n_samples, n_features) = data.dim();
        let n_output = self.calculate_n_output_features(n_features);

        let mut features = Array2::zeros((n_samples, n_output));
        let mut col_idx = 0;

        // Bias term
        if self.include_bias {
            for i in 0..n_samples {
                features[[i, col_idx]] = 1.0;
            }
            col_idx += 1;
        }

        // Degree 1 (original features)
        for j in 0..n_features {
            for i in 0..n_samples {
                features[[i, col_idx]] = data[[i, j]];
            }
            col_idx += 1;
        }

        // Higher degrees
        if self.degree > 1 {
            // Generate all combinations
            for d in 2..=self.degree {
                if col_idx >= n_output {
                    break;
                }
                if self.interaction_only {
                    // Only interactions, no powers
                    self.generate_interactions(data, &mut features, &mut col_idx, d, n_output);
                } else {
                    // All terms including powers
                    self.generate_all_terms(data, &mut features, &mut col_idx, d, n_output);
                }
            }
        }

        Ok(features)
    }

    fn output_dim(&self) -> usize {
        // We need to know input dimension to compute this
        // Return 0 as placeholder
        0
    }

    fn name(&self) -> &str {
        "PolynomialFeatures"
    }
}

impl PolynomialGenerator {
    fn generate_interactions(
        &self,
        data: &Array2<f64>,
        features: &mut Array2<f64>,
        col_idx: &mut usize,
        degree: usize,
        max_cols: usize,
    ) {
        let (n_samples, n_features) = data.dim();
        let mut indices = vec![0; degree];

        loop {
            if *col_idx >= max_cols {
                return;
            }

            // Check if this is a valid interaction (all different)
            let mut is_valid = true;
            for i in 0..degree - 1 {
                if indices[i] == indices[i + 1] {
                    is_valid = false;
                    break;
                }
            }

            if is_valid {
                // Compute product
                for sample in 0..n_samples {
                    let mut product = 1.0;
                    for &idx in &indices {
                        product *= data[[sample, idx]];
                    }
                    features[[sample, *col_idx]] = product;
                }
                *col_idx += 1;
            }

            // Next combination
            let mut pos = degree - 1;
            loop {
                indices[pos] += 1;
                if indices[pos] < n_features {
                    break;
                }
                if pos == 0 {
                    return;
                }
                indices[pos] = indices[pos - 1];
                pos -= 1;
            }
            for i in pos + 1..degree {
                indices[i] = indices[pos];
            }
        }
    }

    fn generate_all_terms(
        &self,
        data: &Array2<f64>,
        features: &mut Array2<f64>,
        col_idx: &mut usize,
        degree: usize,
        max_cols: usize,
    ) {
        let (n_samples, n_features) = data.dim();
        let mut indices = vec![0; degree];

        loop {
            if *col_idx >= max_cols {
                return;
            }

            // Compute product
            for sample in 0..n_samples {
                let mut product = 1.0;
                for &idx in &indices {
                    product *= data[[sample, idx]];
                }
                features[[sample, *col_idx]] = product;
            }
            *col_idx += 1;

            // Next combination with repetition
            let mut pos = degree - 1;
            loop {
                indices[pos] += 1;
                if indices[pos] < n_features {
                    break;
                }
                if pos == 0 {
                    return;
                }
                indices[pos] = 0;
                pos -= 1;
            }
        }
    }
}

/// Composable feature generator
pub struct CompositeGenerator {
    generators: Vec<Box<dyn FeatureGenerator>>,
}

impl std::fmt::Debug for CompositeGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompositeGenerator")
            .field("n_generators", &self.generators.len())
            .finish()
    }
}

impl CompositeGenerator {
    /// Create a new composite generator
    pub fn new() -> Self {
        Self {
            generators: Vec::new(),
        }
    }

    /// Add a generator to the composition
    pub fn add_generator(&mut self, generator: Box<dyn FeatureGenerator>) {
        self.generators.push(generator);
    }

    /// Get number of generators
    pub fn len(&self) -> usize {
        self.generators.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.generators.is_empty()
    }
}

impl Default for CompositeGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureGenerator for CompositeGenerator {
    fn generate(&self, data: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        if self.generators.is_empty() {
            return Ok(data.clone());
        }

        let mut all_features = Vec::new();

        for generator in &self.generators {
            let features = generator.generate(data)?;
            all_features.push(features);
        }

        // Concatenate all features
        let (n_samples, _) = data.dim();
        let total_features: usize = all_features.iter().map(|f| f.ncols()).sum();

        let mut result = Array2::zeros((n_samples, total_features));
        let mut col_offset = 0;

        for feature_matrix in all_features {
            let n_cols = feature_matrix.ncols();
            for i in 0..n_samples {
                for j in 0..n_cols {
                    result[[i, col_offset + j]] = feature_matrix[[i, j]];
                }
            }
            col_offset += n_cols;
        }

        Ok(result)
    }

    fn output_dim(&self) -> usize {
        self.generators.iter().map(|g| g.output_dim()).sum()
    }

    fn name(&self) -> &str {
        "CompositeFeatureGenerator"
    }

    fn is_stateful(&self) -> bool {
        self.generators.iter().any(|g| g.is_stateful())
    }

    fn fit_generator(&mut self, data: &Array2<f64>) -> Result<(), SklearsError> {
        for generator in &mut self.generators {
            if generator.is_stateful() {
                generator.fit_generator(data)?;
            }
        }
        Ok(())
    }
}

/// Feature generator builder for fluent API
pub struct FeatureGeneratorBuilder {
    composite: CompositeGenerator,
}

impl FeatureGeneratorBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            composite: CompositeGenerator::new(),
        }
    }

    /// Add Random Fourier features
    pub fn with_random_fourier(
        mut self,
        n_components: usize,
        gamma: f64,
        random_state: Option<u64>,
    ) -> Self {
        self.composite
            .add_generator(Box::new(RandomFourierGenerator::new(
                n_components,
                gamma,
                random_state,
            )));
        self
    }

    /// Add polynomial features
    pub fn with_polynomial(mut self, degree: usize, include_bias: bool) -> Self {
        self.composite
            .add_generator(Box::new(PolynomialGenerator::new(
                degree,
                include_bias,
                false,
            )));
        self
    }

    /// Add a custom generator
    pub fn with_custom(mut self, generator: Box<dyn FeatureGenerator>) -> Self {
        self.composite.add_generator(generator);
        self
    }

    /// Build the composite generator
    pub fn build(self) -> CompositeGenerator {
        self.composite
    }
}

impl Default for FeatureGeneratorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_random_fourier_generator() {
        let mut generator = RandomFourierGenerator::new(50, 1.0, Some(42));
        let data = array![[1.0, 2.0], [3.0, 4.0]];

        generator.fit_generator(&data).unwrap();
        let features = generator.generate(&data).unwrap();

        assert_eq!(features.shape(), &[2, 50]);
        assert_eq!(generator.output_dim(), 50);
        assert!(generator.is_stateful());
    }

    #[test]
    fn test_polynomial_generator() {
        let generator = PolynomialGenerator::new(2, true, false);
        let data = array![[1.0, 2.0], [3.0, 4.0]];

        let features = generator.generate(&data).unwrap();
        assert!(features.ncols() >= 3); // bias + 2 features + interactions
    }

    #[test]
    fn test_composite_generator() {
        let mut composite = CompositeGenerator::new();
        assert!(composite.is_empty());

        composite.add_generator(Box::new(RandomFourierGenerator::new(10, 1.0, Some(42))));
        assert_eq!(composite.len(), 1);
        assert!(!composite.is_empty());
    }

    #[test]
    fn test_feature_generator_builder() {
        let generator = FeatureGeneratorBuilder::new()
            .with_random_fourier(50, 1.0, Some(42))
            .with_polynomial(2, true)
            .build();

        assert_eq!(generator.len(), 2);
    }

    #[test]
    fn test_polynomial_interaction_only() {
        let generator = PolynomialGenerator::new(2, false, true);
        let data = array![[1.0, 2.0, 3.0]];

        let features = generator.generate(&data).unwrap();
        // Should have original 3 features + interactions
        assert!(features.ncols() >= 3);
    }
}
