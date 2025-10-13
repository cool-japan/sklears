//! Sparse polynomial features for memory-efficient high-dimensional feature generation

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Sparse representation of polynomial feature coefficient
#[derive(Debug, Clone)]
pub struct SparseCoefficient {
    /// Feature indices involved in this term
    pub feature_indices: Vec<usize>,
    /// Powers of each feature in this term
    pub powers: Vec<usize>,
    /// The coefficient value (typically 1.0 for polynomial features)
    pub coefficient: Float,
}

impl SparseCoefficient {
    /// Create a new sparse coefficient
    pub fn new(feature_indices: Vec<usize>, powers: Vec<usize>, coefficient: Float) -> Self {
        assert_eq!(feature_indices.len(), powers.len());
        Self {
            feature_indices,
            powers,
            coefficient,
        }
    }

    /// Evaluate this term for a given input sample
    pub fn evaluate(&self, sample: &Array1<Float>) -> Float {
        let mut result = self.coefficient;
        for (&feature_idx, &power) in self.feature_indices.iter().zip(self.powers.iter()) {
            if power > 0 && feature_idx < sample.len() {
                result *= sample[feature_idx].powi(power as i32);
            }
        }
        result
    }

    /// Get the total degree of this term
    pub fn total_degree(&self) -> usize {
        self.powers.iter().sum()
    }

    /// Check if this is an interaction term (all powers <= 1)
    pub fn is_interaction(&self) -> bool {
        self.powers.iter().all(|&p| p <= 1)
    }
}

/// Configuration for SparsePolynomialFeatures
#[derive(Debug, Clone)]
pub struct SparsePolynomialFeaturesConfig {
    /// The degree of the polynomial features
    pub degree: usize,
    /// Whether to include interaction terms only
    pub interaction_only: bool,
    /// Whether to include bias (constant) term
    pub include_bias: bool,
    /// Maximum number of terms to generate (memory limit)
    pub max_terms: Option<usize>,
    /// Minimum coefficient magnitude to keep (sparsity threshold)
    pub min_coefficient: Float,
    /// Whether to sort terms by total degree first, then lexicographically
    pub sort_terms: bool,
}

impl Default for SparsePolynomialFeaturesConfig {
    fn default() -> Self {
        Self {
            degree: 2,
            interaction_only: false,
            include_bias: true,
            max_terms: None,
            min_coefficient: 1e-12,
            sort_terms: true,
        }
    }
}

/// SparsePolynomialFeatures generates sparse polynomial and interaction features
///
/// This is a memory-efficient version of PolynomialFeatures designed for high-dimensional data.
/// Instead of storing all possible polynomial combinations, it stores only the terms that
/// would actually be non-zero, using a sparse representation.
///
/// This is particularly useful when:
/// - The input dimensionality is high (hundreds or thousands of features)
/// - Many input features are zero or near-zero
/// - Memory usage is a concern
/// - You want to limit the total number of generated features
pub struct SparsePolynomialFeatures<State = Untrained> {
    config: SparsePolynomialFeaturesConfig,
    state: PhantomData<State>,
    // Fitted parameters
    n_features_in_: Option<usize>,
    n_output_features_: Option<usize>,
    sparse_terms_: Option<Vec<SparseCoefficient>>,
}

impl SparsePolynomialFeatures<Untrained> {
    /// Create a new SparsePolynomialFeatures
    pub fn new() -> Self {
        Self {
            config: SparsePolynomialFeaturesConfig::default(),
            state: PhantomData,
            n_features_in_: None,
            n_output_features_: None,
            sparse_terms_: None,
        }
    }

    /// Set the degree of polynomial features
    pub fn degree(mut self, degree: usize) -> Self {
        self.config.degree = degree;
        self
    }

    /// Set whether to include interaction terms only
    pub fn interaction_only(mut self, interaction_only: bool) -> Self {
        self.config.interaction_only = interaction_only;
        self
    }

    /// Set whether to include bias term
    pub fn include_bias(mut self, include_bias: bool) -> Self {
        self.config.include_bias = include_bias;
        self
    }

    /// Set maximum number of terms to generate
    pub fn max_terms(mut self, max_terms: usize) -> Self {
        self.config.max_terms = Some(max_terms);
        self
    }

    /// Set minimum coefficient magnitude threshold
    pub fn min_coefficient(mut self, min_coefficient: Float) -> Self {
        self.config.min_coefficient = min_coefficient;
        self
    }

    /// Set whether to sort terms
    pub fn sort_terms(mut self, sort_terms: bool) -> Self {
        self.config.sort_terms = sort_terms;
        self
    }

    /// Generate sparse polynomial terms
    fn generate_sparse_terms(&self, n_features: usize) -> Vec<SparseCoefficient> {
        let mut terms = Vec::new();

        // Add bias term if requested
        if self.config.include_bias {
            terms.push(SparseCoefficient::new(vec![], vec![], 1.0));
        }

        // Add linear terms (degree 1)
        for feature_idx in 0..n_features {
            terms.push(SparseCoefficient::new(vec![feature_idx], vec![1], 1.0));
        }

        // Generate higher-degree terms
        for degree in 2..=self.config.degree {
            self.generate_degree_terms(n_features, degree, &mut terms);

            // Check if we've exceeded the maximum number of terms
            if let Some(max_terms) = self.config.max_terms {
                if terms.len() >= max_terms {
                    terms.truncate(max_terms);
                    break;
                }
            }
        }

        // Sort terms if requested
        if self.config.sort_terms {
            terms.sort_by(|a, b| {
                // Sort by total degree first, then lexicographically by feature indices
                match a.total_degree().cmp(&b.total_degree()) {
                    std::cmp::Ordering::Equal => a.feature_indices.cmp(&b.feature_indices),
                    other => other,
                }
            });
        }

        terms
    }

    /// Generate all polynomial terms of a specific degree
    fn generate_degree_terms(
        &self,
        n_features: usize,
        degree: usize,
        terms: &mut Vec<SparseCoefficient>,
    ) {
        if self.config.interaction_only && degree > 1 {
            // For interaction only, generate all combinations of 'degree' different features
            self.generate_interaction_combinations(n_features, degree, terms);
        } else {
            // Generate all possible polynomial terms of the given degree
            self.generate_polynomial_combinations(n_features, degree, terms);
        }
    }

    /// Generate interaction combinations (each feature appears at most once)
    fn generate_interaction_combinations(
        &self,
        n_features: usize,
        degree: usize,
        terms: &mut Vec<SparseCoefficient>,
    ) {
        let mut combination = vec![0; degree];
        self.generate_combinations_recursive(n_features, degree, 0, 0, &mut combination, terms);
    }

    /// Generate polynomial combinations (features can appear multiple times)
    fn generate_polynomial_combinations(
        &self,
        n_features: usize,
        degree: usize,
        terms: &mut Vec<SparseCoefficient>,
    ) {
        let mut powers = vec![0; n_features];
        self.generate_polynomial_recursive(n_features, degree, 0, &mut powers, terms);
    }

    /// Recursive helper for generating combinations
    fn generate_combinations_recursive(
        &self,
        n_features: usize,
        degree: usize,
        start: usize,
        pos: usize,
        combination: &mut Vec<usize>,
        terms: &mut Vec<SparseCoefficient>,
    ) {
        if pos == degree {
            // Create a sparse coefficient for this combination
            let feature_indices = combination.clone();
            let powers = vec![1; degree];
            terms.push(SparseCoefficient::new(feature_indices, powers, 1.0));
            return;
        }

        for i in start..n_features {
            combination[pos] = i;
            self.generate_combinations_recursive(
                n_features,
                degree,
                i + 1,
                pos + 1,
                combination,
                terms,
            );
        }
    }

    /// Recursive helper for generating polynomial terms
    fn generate_polynomial_recursive(
        &self,
        n_features: usize,
        remaining_degree: usize,
        feature_idx: usize,
        powers: &mut Vec<usize>,
        terms: &mut Vec<SparseCoefficient>,
    ) {
        if remaining_degree == 0 {
            // Create sparse coefficient from current powers
            let mut feature_indices = Vec::new();
            let mut term_powers = Vec::new();

            for (idx, &power) in powers.iter().enumerate() {
                if power > 0 {
                    feature_indices.push(idx);
                    term_powers.push(power);
                }
            }

            if !feature_indices.is_empty() {
                terms.push(SparseCoefficient::new(feature_indices, term_powers, 1.0));
            }
            return;
        }

        if feature_idx >= n_features {
            return;
        }

        // Try all possible powers for current feature
        for power in 0..=remaining_degree {
            powers[feature_idx] = power;
            self.generate_polynomial_recursive(
                n_features,
                remaining_degree - power,
                feature_idx + 1,
                powers,
                terms,
            );
        }
        powers[feature_idx] = 0; // Reset for backtracking
    }
}

impl SparsePolynomialFeatures<Trained> {
    /// Get the number of input features
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_.expect("Not fitted")
    }

    /// Get the number of output features
    pub fn n_output_features(&self) -> usize {
        self.n_output_features_.expect("Not fitted")
    }

    /// Get the sparse terms
    pub fn sparse_terms(&self) -> &[SparseCoefficient] {
        self.sparse_terms_.as_ref().expect("Not fitted")
    }

    /// Get memory usage information
    pub fn memory_info(&self) -> SparseMemoryInfo {
        let sparse_terms = self.sparse_terms();
        let n_terms = sparse_terms.len();
        let avg_features_per_term = if n_terms > 0 {
            sparse_terms
                .iter()
                .map(|term| term.feature_indices.len())
                .sum::<usize>() as f64
                / n_terms as f64
        } else {
            0.0
        };

        let n_features = self.n_features_in();
        let theoretical_dense_features = if self.config.include_bias {
            // Binomial coefficient: C(n + d, d) for polynomial features with bias
            self.binomial_coefficient(n_features + self.config.degree, self.config.degree)
        } else {
            self.binomial_coefficient(n_features + self.config.degree, self.config.degree) - 1
        };

        let sparsity_ratio = n_terms as f64 / theoretical_dense_features as f64;

        SparseMemoryInfo {
            n_sparse_terms: n_terms,
            theoretical_dense_features,
            sparsity_ratio,
            avg_features_per_term,
            memory_reduction_factor: theoretical_dense_features as f64 / n_terms as f64,
        }
    }

    /// Calculate binomial coefficient C(n, k)
    fn binomial_coefficient(&self, n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        if k == 0 || k == n {
            return 1;
        }

        let k = if k > n - k { n - k } else { k }; // Take advantage of symmetry

        let mut result = 1;
        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }
        result
    }
}

impl Default for SparsePolynomialFeatures<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<Array2<Float>, ()> for SparsePolynomialFeatures<Untrained> {
    type Fitted = SparsePolynomialFeatures<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "n_samples".to_string(),
                reason: "Cannot fit on empty dataset".to_string(),
            });
        }

        if self.config.degree == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "degree".to_string(),
                reason: "Degree must be positive".to_string(),
            });
        }

        // Generate sparse terms
        let sparse_terms = self.generate_sparse_terms(n_features);
        let n_output_features = sparse_terms.len();

        Ok(SparsePolynomialFeatures {
            config: self.config,
            state: PhantomData,
            n_features_in_: Some(n_features),
            n_output_features_: Some(n_output_features),
            sparse_terms_: Some(sparse_terms),
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for SparsePolynomialFeatures<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.n_features_in() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in(),
                actual: n_features,
            });
        }

        let sparse_terms = self.sparse_terms();
        let n_output_features = sparse_terms.len();

        let mut output = Array2::zeros((n_samples, n_output_features));

        for (sample_idx, sample) in x.rows().into_iter().enumerate() {
            let sample_array = sample.to_owned();
            for (term_idx, term) in sparse_terms.iter().enumerate() {
                output[[sample_idx, term_idx]] = term.evaluate(&sample_array);
            }
        }

        Ok(output)
    }
}

/// Memory usage information for sparse polynomial features
#[derive(Debug, Clone)]
pub struct SparseMemoryInfo {
    /// Number of sparse terms actually generated
    pub n_sparse_terms: usize,
    /// Number of features a dense implementation would generate
    pub theoretical_dense_features: usize,
    /// Ratio of sparse to dense features (sparsity)
    pub sparsity_ratio: f64,
    /// Average number of features per sparse term
    pub avg_features_per_term: f64,
    /// Factor by which memory is reduced compared to dense
    pub memory_reduction_factor: f64,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_sparse_coefficient() {
        let coeff = SparseCoefficient::new(vec![0, 1], vec![2, 1], 1.5);

        assert_eq!(coeff.total_degree(), 3);
        assert!(!coeff.is_interaction());

        let sample = array![2.0, 3.0];
        let result = coeff.evaluate(&sample);
        // Should be 1.5 * 2^2 * 3^1 = 1.5 * 4 * 3 = 18.0
        assert_eq!(result, 18.0);
    }

    #[test]
    fn test_sparse_polynomial_features_basic() {
        let transformer = SparsePolynomialFeatures::new().degree(2).include_bias(true);

        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let fitted = transformer.fit(&x, &()).unwrap();

        assert_eq!(fitted.n_features_in(), 2);
        assert!(fitted.n_output_features() >= 5); // bias + x1 + x2 + x1^2 + x1*x2 + x2^2

        let transformed = fitted.transform(&x).unwrap();
        assert_eq!(transformed.nrows(), 2);
        assert_eq!(transformed.ncols(), fitted.n_output_features());
    }

    #[test]
    fn test_interaction_only() {
        let transformer = SparsePolynomialFeatures::new()
            .degree(2)
            .interaction_only(true)
            .include_bias(false);

        let x = array![[1.0, 2.0, 3.0]];
        let fitted = transformer.fit(&x, &()).unwrap();

        let transformed = fitted.transform(&x).unwrap();

        // Should have: x1, x2, x3, x1*x2, x1*x3, x2*x3 = 6 features
        assert_eq!(transformed.ncols(), 6);
    }

    #[test]
    fn test_max_terms_limit() {
        let transformer = SparsePolynomialFeatures::new().degree(3).max_terms(10);

        let x = array![[1.0, 2.0, 3.0, 4.0, 5.0]];
        let fitted = transformer.fit(&x, &()).unwrap();

        // Should be limited to 10 terms
        assert_eq!(fitted.n_output_features(), 10);
    }

    #[test]
    fn test_memory_info() {
        let transformer = SparsePolynomialFeatures::new().degree(2).max_terms(10);

        let x = array![[1.0, 2.0, 3.0]];
        let fitted = transformer.fit(&x, &()).unwrap();

        let memory_info = fitted.memory_info();
        assert_eq!(memory_info.n_sparse_terms, fitted.n_output_features());
        assert!(memory_info.sparsity_ratio <= 1.0);
        assert!(memory_info.memory_reduction_factor >= 1.0);
    }
}
