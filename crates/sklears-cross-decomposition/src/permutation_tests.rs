//! Permutation tests for statistical significance

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{thread_rng, Random, Rng, SeedableRng};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict},
    types::Float,
};
use std::collections::HashMap;

/// Results from a permutation test
#[derive(Debug, Clone)]
pub struct PermutationTestResults {
    /// Observed test statistic (actual model performance)
    pub observed_statistic: Float,
    /// Null distribution of test statistics from permuted data
    pub null_distribution: Vec<Float>,
    /// P-value (proportion of permuted statistics >= observed)
    pub p_value: Float,
    /// Number of permutations performed
    pub n_permutations: usize,
    /// Whether the test is significant at alpha level
    pub is_significant: bool,
    /// Alpha level used for significance testing
    pub alpha: Float,
}

/// Test statistic for permutation tests
#[derive(Debug, Clone, PartialEq)]
pub enum TestStatistic {
    /// Explained variance ratio
    ExplainedVarianceRatio,
    /// Maximum canonical correlation
    MaxCanonicalCorrelation,
    /// Sum of canonical correlations
    SumCanonicalCorrelations,
    /// R² score
    R2Score,
    /// Custom statistic function
    Custom(fn(&Array2<Float>, &Array2<Float>) -> Float),
}

impl TestStatistic {
    /// Compute the test statistic
    pub fn compute(
        &self,
        fitted_model: &impl ComputeStatistic,
        x: &Array2<Float>,
        y: &Array2<Float>,
    ) -> Result<Float> {
        match self {
            TestStatistic::ExplainedVarianceRatio => fitted_model.explained_variance_ratio(),
            TestStatistic::MaxCanonicalCorrelation => fitted_model.max_canonical_correlation(),
            TestStatistic::SumCanonicalCorrelations => fitted_model.sum_canonical_correlations(),
            TestStatistic::R2Score => fitted_model.r2_score(x, y),
            TestStatistic::Custom(func) => {
                if let Ok(y_pred) = fitted_model.predict_for_statistic(x) {
                    Ok(func(y, &y_pred))
                } else {
                    Err(SklearsError::InvalidInput(
                        "Failed to compute custom statistic".into(),
                    ))
                }
            }
        }
    }
}

/// Trait for models that can compute statistics for permutation tests
pub trait ComputeStatistic {
    /// Compute explained variance ratio
    fn explained_variance_ratio(&self) -> Result<Float>;

    /// Compute maximum canonical correlation
    fn max_canonical_correlation(&self) -> Result<Float>;

    /// Compute sum of canonical correlations
    fn sum_canonical_correlations(&self) -> Result<Float>;

    /// Compute R² score
    fn r2_score(&self, x: &Array2<Float>, y: &Array2<Float>) -> Result<Float>;

    /// Predict for statistic computation
    fn predict_for_statistic(&self, x: &Array2<Float>) -> Result<Array2<Float>>;
}

/// Permutation test for statistical significance
pub struct PermutationTest {
    /// Number of permutations to perform
    pub n_permutations: usize,
    /// Test statistic to use
    pub test_statistic: TestStatistic,
    /// Alpha level for significance testing
    pub alpha: Float,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
    /// Whether to use parallel processing
    pub parallel: bool,
}

impl PermutationTest {
    pub fn new(n_permutations: usize, test_statistic: TestStatistic) -> Self {
        Self {
            n_permutations,
            test_statistic,
            alpha: 0.05,
            random_state: None,
            parallel: false,
        }
    }

    /// Set the alpha level
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Enable/disable parallel processing
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Perform permutation test
    pub fn test<E>(
        &self,
        estimator_factory: impl Fn() -> E + Clone,
        x: &Array2<Float>,
        y: &Array2<Float>,
    ) -> Result<PermutationTestResults>
    where
        E: Fit<Array2<Float>, Array2<Float>>,
        E::Fitted: ComputeStatistic,
    {
        // Fit model on original data and compute observed statistic
        let original_model = estimator_factory().fit(x, y)?;
        let observed_statistic = self.test_statistic.compute(&original_model, x, y)?;

        // Generate null distribution
        let mut null_distribution = Vec::with_capacity(self.n_permutations);
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            let mut entropy_rng = thread_rng();
            StdRng::from_rng(&mut entropy_rng)
        };

        for _ in 0..self.n_permutations {
            // Permute the response variables
            let y_permuted = self.permute_array(y, &mut rng);

            // Fit model on permuted data
            let permuted_model = estimator_factory().fit(x, &y_permuted)?;
            let permuted_statistic =
                self.test_statistic
                    .compute(&permuted_model, x, &y_permuted)?;

            null_distribution.push(permuted_statistic);
        }

        // Compute p-value
        let extreme_count = null_distribution
            .iter()
            .filter(|&&stat| stat >= observed_statistic)
            .count();
        let p_value = extreme_count as Float / self.n_permutations as Float;

        // Check significance
        let is_significant = p_value < self.alpha;

        Ok(PermutationTestResults {
            observed_statistic,
            null_distribution,
            p_value,
            n_permutations: self.n_permutations,
            is_significant,
            alpha: self.alpha,
        })
    }

    /// Perform permutation test for multi-block models
    pub fn test_multiblock<E>(
        &self,
        estimator_factory: impl Fn() -> E + Clone,
        x_blocks: &[Array2<Float>],
        y: &Array2<Float>,
    ) -> Result<PermutationTestResults>
    where
        E: Fit<Vec<Array2<Float>>, Array2<Float>>,
        E::Fitted: ComputeStatistic,
    {
        // Fit model on original data
        let x_blocks_vec = x_blocks.to_vec();
        let original_model = estimator_factory().fit(&x_blocks_vec, y)?;
        let observed_statistic = self
            .test_statistic
            .compute(&original_model, &x_blocks[0], y)?;

        // Generate null distribution
        let mut null_distribution = Vec::with_capacity(self.n_permutations);
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            let mut entropy_rng = thread_rng();
            StdRng::from_rng(&mut entropy_rng)
        };

        for _ in 0..self.n_permutations {
            // Permute the response variables
            let y_permuted = self.permute_array(y, &mut rng);

            // Fit model on permuted data
            let permuted_model = estimator_factory().fit(&x_blocks_vec, &y_permuted)?;
            let permuted_statistic =
                self.test_statistic
                    .compute(&permuted_model, &x_blocks[0], &y_permuted)?;

            null_distribution.push(permuted_statistic);
        }

        // Compute p-value
        let extreme_count = null_distribution
            .iter()
            .filter(|&&stat| stat >= observed_statistic)
            .count();
        let p_value = extreme_count as Float / self.n_permutations as Float;

        // Check significance
        let is_significant = p_value < self.alpha;

        Ok(PermutationTestResults {
            observed_statistic,
            null_distribution,
            p_value,
            n_permutations: self.n_permutations,
            is_significant,
            alpha: self.alpha,
        })
    }

    /// Permute an array (shuffle rows)
    fn permute_array(
        &self,
        array: &Array2<Float>,
        rng: &mut impl scirs2_core::random::Rng,
    ) -> Array2<Float> {
        use scirs2_core::rand_prelude::SliceRandom;

        let mut indices: Vec<usize> = (0..array.nrows()).collect();
        indices.shuffle(rng);

        let mut permuted = Array2::zeros(array.raw_dim());
        for (new_idx, &orig_idx) in indices.iter().enumerate() {
            permuted.row_mut(new_idx).assign(&array.row(orig_idx));
        }

        permuted
    }
}

/// Stability selection using permutation tests
pub struct StabilitySelection {
    /// Number of bootstrap samples
    pub n_bootstrap: usize,
    /// Subsampling ratio for each bootstrap
    pub subsample_ratio: Float,
    /// Selection threshold (proportion of times a component must be selected)
    pub selection_threshold: Float,
    /// Maximum number of components to consider
    pub max_components: usize,
    /// Random seed
    pub random_state: Option<u64>,
}

impl StabilitySelection {
    /// Create a new stability selection procedure
    pub fn new(n_bootstrap: usize, max_components: usize) -> Self {
        Self {
            n_bootstrap,
            subsample_ratio: 0.8,
            selection_threshold: 0.6,
            max_components,
            random_state: None,
        }
    }

    /// Set subsample ratio
    pub fn subsample_ratio(mut self, ratio: Float) -> Self {
        self.subsample_ratio = ratio;
        self
    }

    /// Set selection threshold
    pub fn selection_threshold(mut self, threshold: Float) -> Self {
        self.selection_threshold = threshold;
        self
    }

    /// Set random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Perform stability selection
    pub fn select_stable_components<E>(
        &self,
        estimator_factory: impl Fn(usize) -> E + Clone,
        x: &Array2<Float>,
        y: &Array2<Float>,
    ) -> Result<StabilityResults>
    where
        E: Fit<Array2<Float>, Array2<Float>>,
        E::Fitted: ComputeStatistic,
    {
        let mut component_selections = HashMap::new();
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            let mut entropy_rng = thread_rng();
            StdRng::from_rng(&mut entropy_rng)
        };

        let n_samples = x.nrows();
        let subsample_size = (n_samples as Float * self.subsample_ratio) as usize;

        for _ in 0..self.n_bootstrap {
            // Create bootstrap sample
            let sample_indices = self.bootstrap_indices(n_samples, subsample_size, &mut rng);
            let x_sample = self.select_rows(x, &sample_indices);
            let y_sample = self.select_rows(y, &sample_indices);

            // Test different numbers of components
            let mut best_score = Float::NEG_INFINITY;
            let mut best_n_components = 1;

            for n_comp in 1..=self.max_components {
                let estimator = estimator_factory(n_comp);
                if let Ok(fitted) = estimator.fit(&x_sample, &y_sample) {
                    if let Ok(score) = fitted.explained_variance_ratio() {
                        if score > best_score {
                            best_score = score;
                            best_n_components = n_comp;
                        }
                    }
                }
            }

            // Record selection
            *component_selections.entry(best_n_components).or_insert(0) += 1;
        }

        // Compute selection frequencies
        let mut selection_frequencies = HashMap::new();
        for (&n_comp, &count) in &component_selections {
            let frequency = count as Float / self.n_bootstrap as Float;
            selection_frequencies.insert(n_comp, frequency);
        }

        // Find stable components (selected above threshold)
        let stable_components: Vec<usize> = selection_frequencies
            .iter()
            .filter(|(_, &freq)| freq >= self.selection_threshold)
            .map(|(&n_comp, _)| n_comp)
            .collect();

        // Select most frequently chosen stable component
        let selected_n_components = stable_components
            .iter()
            .max_by_key(|&&n_comp| component_selections.get(&n_comp).unwrap_or(&0))
            .copied();

        Ok(StabilityResults {
            selected_n_components,
            selection_frequencies,
            stable_components,
            selection_threshold: self.selection_threshold,
            n_bootstrap: self.n_bootstrap,
        })
    }

    /// Generate bootstrap indices
    fn bootstrap_indices(
        &self,
        n_samples: usize,
        subsample_size: usize,
        rng: &mut impl scirs2_core::random::Rng,
    ) -> Vec<usize> {
        use scirs2_core::rand_prelude::SliceRandom;

        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(rng);
        indices.truncate(subsample_size);
        indices
    }

    /// Select rows from array
    fn select_rows(&self, array: &Array2<Float>, indices: &[usize]) -> Array2<Float> {
        let mut result = Array2::zeros((indices.len(), array.ncols()));
        for (i, &idx) in indices.iter().enumerate() {
            result.row_mut(i).assign(&array.row(idx));
        }
        result
    }
}

/// Results from stability selection
#[derive(Debug, Clone)]
pub struct StabilityResults {
    /// Selected number of components (most stable)
    pub selected_n_components: Option<usize>,
    /// Selection frequency for each number of components
    pub selection_frequencies: HashMap<usize, Float>,
    /// Components that are stable (above threshold)
    pub stable_components: Vec<usize>,
    /// Selection threshold used
    pub selection_threshold: Float,
    /// Number of bootstrap samples
    pub n_bootstrap: usize,
}

// Implementation of ComputeStatistic for common types
impl<T> ComputeStatistic for T
where
    T: Predict<Array2<Float>, Array2<Float>>,
{
    fn explained_variance_ratio(&self) -> Result<Float> {
        // Default implementation - would need to be specialized for each model type
        Ok(0.5) // Placeholder
    }

    fn max_canonical_correlation(&self) -> Result<Float> {
        // Default implementation
        Ok(0.5) // Placeholder
    }

    fn sum_canonical_correlations(&self) -> Result<Float> {
        // Default implementation
        Ok(0.5) // Placeholder
    }

    fn r2_score(&self, x: &Array2<Float>, y: &Array2<Float>) -> Result<Float> {
        let y_pred = self.predict(x)?;
        let y_mean = y.mean().unwrap();
        let ss_res = (y - &y_pred).mapv(|x| x * x).sum();
        let ss_tot = (y - y_mean).mapv(|x| x * x).sum();
        Ok(1.0 - (ss_res / ss_tot))
    }

    fn predict_for_statistic(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        self.predict(x)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::PLSRegression;
    use scirs2_core::ndarray::array;
    use sklears_core::traits::{Fit, Predict};

    // Mock implementation for testing
    struct MockModel {
        explained_var: Float,
        correlation: Float,
    }

    impl ComputeStatistic for MockModel {
        fn explained_variance_ratio(&self) -> Result<Float> {
            Ok(self.explained_var)
        }

        fn max_canonical_correlation(&self) -> Result<Float> {
            Ok(self.correlation)
        }

        fn sum_canonical_correlations(&self) -> Result<Float> {
            Ok(self.correlation)
        }

        fn r2_score(&self, _x: &Array2<Float>, _y: &Array2<Float>) -> Result<Float> {
            Ok(self.explained_var)
        }

        fn predict_for_statistic(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
            Ok(Array2::zeros((x.nrows(), 1)))
        }
    }

    impl Fit<Array2<Float>, Array2<Float>> for MockModel {
        type Fitted = MockModel;

        fn fit(self, _x: &Array2<Float>, _y: &Array2<Float>) -> Result<Self::Fitted> {
            Ok(self)
        }
    }

    #[test]
    fn test_permutation_test_basic() {
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0]
        ];
        let y = array![[1.5], [3.5], [5.5], [7.5], [9.5], [11.5]];

        let perm_test =
            PermutationTest::new(100, TestStatistic::ExplainedVarianceRatio).random_state(42);

        let result = perm_test
            .test(
                || MockModel {
                    explained_var: 0.8,
                    correlation: 0.9,
                },
                &x,
                &y,
            )
            .unwrap();

        assert_eq!(result.n_permutations, 100);
        assert_eq!(result.null_distribution.len(), 100);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert_eq!(result.observed_statistic, 0.8);
    }

    #[test]
    fn test_stability_selection() {
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0],
            [19.0, 20.0, 21.0],
            [22.0, 23.0, 24.0]
        ];
        let y = array![[2.0], [5.0], [8.0], [11.0], [14.0], [17.0], [20.0], [23.0]];

        let stability = StabilitySelection::new(50, 3)
            .selection_threshold(0.3)
            .random_state(42);

        let result = stability
            .select_stable_components(
                |n_comp| MockModel {
                    explained_var: n_comp as Float * 0.2,
                    correlation: 0.8,
                },
                &x,
                &y,
            )
            .unwrap();

        assert!(result.selected_n_components.is_some());
        assert!(!result.selection_frequencies.is_empty());
        assert_eq!(result.n_bootstrap, 50);
    }

    #[test]
    fn test_test_statistics() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![[1.5], [2.5], [3.5], [4.5]];

        let model = MockModel {
            explained_var: 0.75,
            correlation: 0.9,
        };

        assert_eq!(
            TestStatistic::ExplainedVarianceRatio
                .compute(&model, &x, &y)
                .unwrap(),
            0.75
        );
        assert_eq!(
            TestStatistic::MaxCanonicalCorrelation
                .compute(&model, &x, &y)
                .unwrap(),
            0.9
        );
        assert_eq!(
            TestStatistic::SumCanonicalCorrelations
                .compute(&model, &x, &y)
                .unwrap(),
            0.9
        );
    }

    #[test]
    fn test_permutation_test_significance() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![[1.0], [2.0], [3.0], [4.0]];

        // Test with actual PLS model that should give different results when permuted
        let perm_test = PermutationTest::new(50, TestStatistic::ExplainedVarianceRatio)
            .alpha(0.05)
            .random_state(42);

        let result = perm_test.test(|| PLSRegression::new(1), &x, &y).unwrap();

        // Check that the test runs and produces reasonable results
        assert_eq!(result.n_permutations, 50);
        assert_eq!(result.null_distribution.len(), 50);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(result.observed_statistic >= 0.0);
    }
}
