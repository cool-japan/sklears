//! Ensemble-based feature selection methods
//!
//! This module provides feature selection methods that combine multiple approaches
//! or use ensemble techniques for more robust feature selection.

use crate::{FeatureImportance, IndexableTarget};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayBase, Axis};
use scirs2_core::rand_prelude::SliceRandom;
use scirs2_core::random::{rngs::StdRng, thread_rng, Rng, SeedableRng};
use sklears_core::{
    error::{validate, Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

use crate::base::SelectorMixin;

/// Extract features from array based on selected indices
pub fn extract_features(x: &Array2<Float>, features: &[usize]) -> SklResult<Array2<Float>> {
    let n_samples = x.nrows();
    let n_selected = features.len();
    let mut x_subset = Array2::zeros((n_samples, n_selected));

    for (new_idx, &old_idx) in features.iter().enumerate() {
        if old_idx >= x.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "Feature index {} out of bounds",
                old_idx
            )));
        }
        x_subset.column_mut(new_idx).assign(&x.column(old_idx));
    }

    Ok(x_subset)
}

/// Bootstrap-based Feature Selection
/// Uses bootstrap resampling to estimate feature stability and importance
#[derive(Debug, Clone)]
pub struct BootstrapSelector<E, State = Untrained> {
    estimator: E,
    n_bootstrap: usize,
    sample_ratio: f64,
    stability_threshold: f64,
    min_selection_frequency: f64,
    max_features: Option<usize>,
    random_state: Option<u64>,
    state: PhantomData<State>,
    // Trained state
    feature_frequencies_: Option<Array1<Float>>,
    stability_scores_: Option<Array1<Float>>,
    selected_features_: Option<Vec<usize>>,
    n_features_: Option<usize>,
    bootstrap_selections_: Option<Vec<Vec<usize>>>,
}

impl<E: Clone> BootstrapSelector<E, Untrained> {
    /// Create a new BootstrapSelector
    pub fn new(estimator: E) -> Self {
        Self {
            estimator,
            n_bootstrap: 100,
            sample_ratio: 0.8,
            stability_threshold: 0.7,
            min_selection_frequency: 0.5,
            max_features: None,
            random_state: None,
            state: PhantomData,
            feature_frequencies_: None,
            stability_scores_: None,
            selected_features_: None,
            n_features_: None,
            bootstrap_selections_: None,
        }
    }

    /// Set the number of bootstrap iterations
    pub fn n_bootstrap(mut self, n_bootstrap: usize) -> Self {
        if n_bootstrap < 1 {
            panic!("n_bootstrap must be at least 1");
        }
        self.n_bootstrap = n_bootstrap;
        self
    }

    /// Set the sample ratio for each bootstrap iteration
    pub fn sample_ratio(mut self, sample_ratio: f64) -> Self {
        if !(0.0..=1.0).contains(&sample_ratio) {
            panic!("sample_ratio must be between 0 and 1");
        }
        self.sample_ratio = sample_ratio;
        self
    }

    /// Set the stability threshold (features selected in at least this fraction of bootstrap iterations)
    pub fn stability_threshold(mut self, threshold: f64) -> Self {
        if !(0.0..=1.0).contains(&threshold) {
            panic!("stability_threshold must be between 0 and 1");
        }
        self.stability_threshold = threshold;
        self
    }

    /// Set the minimum selection frequency for a feature to be considered
    pub fn min_selection_frequency(mut self, frequency: f64) -> Self {
        if !(0.0..=1.0).contains(&frequency) {
            panic!("min_selection_frequency must be between 0 and 1");
        }
        self.min_selection_frequency = frequency;
        self
    }

    /// Set the maximum number of features to select
    pub fn max_features(mut self, max_features: Option<usize>) -> Self {
        self.max_features = max_features;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl<E> Estimator for BootstrapSelector<E, Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl<E, Y> Fit<Array2<Float>, Y> for BootstrapSelector<E, Untrained>
where
    E: Clone + Fit<Array2<Float>, Y> + Send + Sync,
    E::Fitted: FeatureImportance + Send + Sync,
    Y: Clone + IndexableTarget,
{
    type Fitted = BootstrapSelector<E, Trained>;

    fn fit(self, x: &Array2<Float>, y: &Y) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Bootstrap selection requires at least 2 samples".to_string(),
            ));
        }

        // Initialize random number generator
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_rng(&mut thread_rng())
        };

        let sample_size = (n_samples as f64 * self.sample_ratio).round() as usize;
        let mut feature_counts = Array1::zeros(n_features);
        let mut bootstrap_selections = Vec::with_capacity(self.n_bootstrap);

        // Bootstrap iterations
        for _ in 0..self.n_bootstrap {
            // Generate bootstrap sample indices
            let mut bootstrap_indices = Vec::with_capacity(sample_size);
            for _ in 0..sample_size {
                bootstrap_indices.push(rng.gen_range(0..n_samples));
            }

            // Extract bootstrap sample
            let x_bootstrap = x.select(Axis(0), &bootstrap_indices);
            let y_bootstrap = y.select(&bootstrap_indices);

            // Fit estimator on bootstrap sample
            match self.estimator.clone().fit(&x_bootstrap, &y_bootstrap) {
                Ok(fitted_estimator) => {
                    // Get feature importances
                    if let Ok(importances) = fitted_estimator.feature_importances() {
                        // Find features with above-threshold importance
                        let threshold = importances.mean().unwrap_or(0.0);
                        let selected_features: Vec<usize> = importances
                            .iter()
                            .enumerate()
                            .filter(|(_, &importance)| importance > threshold)
                            .map(|(idx, _)| idx)
                            .collect();

                        // Update feature counts
                        for &feature_idx in &selected_features {
                            feature_counts[feature_idx] += 1.0;
                        }

                        bootstrap_selections.push(selected_features);
                    }
                }
                Err(_) => {
                    // If fitting fails for this bootstrap sample, skip it
                    continue;
                }
            }
        }

        // Calculate selection frequencies
        let feature_frequencies = feature_counts / self.n_bootstrap as f64;

        // Calculate stability scores (Jaccard similarity between consecutive selections)
        let mut stability_scores = Array1::zeros(n_features);
        if bootstrap_selections.len() >= 2 {
            for feature_idx in 0..n_features {
                let mut jaccard_similarities = Vec::new();

                for i in 0..(bootstrap_selections.len() - 1) {
                    let set1_contains = bootstrap_selections[i].contains(&feature_idx);
                    let set2_contains = bootstrap_selections[i + 1].contains(&feature_idx);

                    // Simple stability measure: consistency across consecutive selections
                    let similarity = if set1_contains == set2_contains {
                        1.0
                    } else {
                        0.0
                    };
                    jaccard_similarities.push(similarity);
                }

                stability_scores[feature_idx] =
                    jaccard_similarities.iter().sum::<f64>() / jaccard_similarities.len() as f64;
            }
        }

        // Select features based on frequency and stability thresholds
        let mut selected_features: Vec<usize> = (0..n_features)
            .filter(|&i| {
                feature_frequencies[i] >= self.min_selection_frequency
                    && stability_scores[i] >= self.stability_threshold
            })
            .collect();

        // Apply max_features limit if specified
        if let Some(max_feat) = self.max_features {
            if selected_features.len() > max_feat {
                // Sort by combined score (frequency * stability) and take top max_feat
                selected_features.sort_by(|&a, &b| {
                    let score_a: f64 = feature_frequencies[a] * stability_scores[a];
                    let score_b: f64 = feature_frequencies[b] * stability_scores[b];
                    score_b.partial_cmp(&score_a).unwrap()
                });
                selected_features.truncate(max_feat);
                selected_features.sort(); // Restore original order
            }
        }

        if selected_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features selected by Bootstrap method. Try reducing thresholds.".to_string(),
            ));
        }

        Ok(BootstrapSelector {
            estimator: self.estimator,
            n_bootstrap: self.n_bootstrap,
            sample_ratio: self.sample_ratio,
            stability_threshold: self.stability_threshold,
            min_selection_frequency: self.min_selection_frequency,
            max_features: self.max_features,
            random_state: self.random_state,
            state: PhantomData,
            feature_frequencies_: Some(feature_frequencies),
            stability_scores_: Some(stability_scores),
            selected_features_: Some(selected_features),
            n_features_: Some(n_features),
            bootstrap_selections_: Some(bootstrap_selections),
        })
    }
}

impl<E> Transform<Array2<Float>> for BootstrapSelector<E, Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        validate::check_n_features(x, self.n_features_.unwrap())?;

        let selected_features = self.selected_features_.as_ref().unwrap();
        extract_features(x, selected_features)
    }
}

impl<E> SelectorMixin for BootstrapSelector<E, Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let n_features = self.n_features_.unwrap();
        let selected_features = self.selected_features_.as_ref().unwrap();
        let mut support = Array1::from_elem(n_features, false);

        for &idx in selected_features {
            support[idx] = true;
        }

        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected_features.iter().position(|&f| f == idx))
            .collect())
    }
}

impl<E> BootstrapSelector<E, Trained> {
    /// Get feature selection frequencies across bootstrap iterations
    pub fn feature_frequencies(&self) -> &Array1<Float> {
        self.feature_frequencies_.as_ref().unwrap()
    }

    /// Get feature stability scores
    pub fn stability_scores(&self) -> &Array1<Float> {
        self.stability_scores_.as_ref().unwrap()
    }

    /// Get selected features
    pub fn selected_features(&self) -> &[usize] {
        self.selected_features_.as_ref().unwrap()
    }

    /// Get all bootstrap selections for analysis
    pub fn bootstrap_selections(&self) -> &[Vec<usize>] {
        self.bootstrap_selections_.as_ref().unwrap()
    }

    /// Calculate the overall stability of the feature selection
    /// Returns the average Jaccard similarity across all bootstrap iterations
    pub fn selection_stability(&self) -> f64 {
        let selections = self.bootstrap_selections_.as_ref().unwrap();
        if selections.len() < 2 {
            return 1.0; // Perfect stability with only one selection
        }

        let mut similarities = Vec::new();
        for i in 0..(selections.len() - 1) {
            for j in (i + 1)..selections.len() {
                let set1: std::collections::HashSet<usize> =
                    selections[i].iter().cloned().collect();
                let set2: std::collections::HashSet<usize> =
                    selections[j].iter().cloned().collect();

                let intersection = set1.intersection(&set2).count();
                let union = set1.union(&set2).count();

                let jaccard = if union == 0 {
                    1.0
                } else {
                    intersection as f64 / union as f64
                };
                similarities.push(jaccard);
            }
        }

        similarities.iter().sum::<f64>() / similarities.len() as f64
    }
}

/// Boruta feature selection algorithm
/// Boruta is an all-relevant feature selection wrapper algorithm that finds all features
/// carrying information usable for prediction, rather than finding a minimal feature subset.
/// It works by creating shadow features (permuted versions) and comparing real feature
/// importance against these random features.
#[derive(Debug, Clone)]
pub struct BorutaSelector<E, State = Untrained> {
    estimator: E,
    n_estimators: usize,
    max_iter: usize,
    alpha: f64,
    two_step: bool,
    percentile: f64,
    random_state: Option<u64>,
    state: PhantomData<State>,
    // Trained state
    confirmed_features_: Option<Vec<usize>>,
    tentative_features_: Option<Vec<usize>>,
    rejected_features_: Option<Vec<usize>>,
    feature_importances_: Option<Array1<Float>>,
    shadow_importances_: Option<Array1<Float>>,
    n_features_: Option<usize>,
    n_iterations_: Option<usize>,
}

impl<E> BorutaSelector<E, Untrained> {
    /// Create a new Boruta selector
    pub fn new(estimator: E) -> Self {
        Self {
            estimator,
            n_estimators: 100,
            max_iter: 100,
            alpha: 0.05,
            two_step: true,
            percentile: 100.0,
            random_state: None,
            state: PhantomData,
            confirmed_features_: None,
            tentative_features_: None,
            rejected_features_: None,
            feature_importances_: None,
            shadow_importances_: None,
            n_features_: None,
            n_iterations_: None,
        }
    }

    /// Set the number of estimators for the ensemble
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    /// Set maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the significance level alpha for statistical tests
    pub fn alpha(mut self, alpha: f64) -> Self {
        if alpha <= 0.0 || alpha >= 1.0 {
            panic!("alpha must be between 0 and 1");
        }
        self.alpha = alpha;
        self
    }

    /// Enable/disable two-step procedure for rejecting features
    pub fn two_step(mut self, two_step: bool) -> Self {
        self.two_step = two_step;
        self
    }

    /// Set percentile of shadow feature importance used as threshold
    pub fn percentile(mut self, percentile: f64) -> Self {
        if percentile <= 0.0 || percentile > 100.0 {
            panic!("percentile must be between 0 and 100");
        }
        self.percentile = percentile;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl<E> Estimator for BorutaSelector<E, Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl<E> Fit<Array2<Float>, Array1<Float>> for BorutaSelector<E, Untrained>
where
    E: Clone + Fit<Array2<Float>, Array1<Float>> + FeatureImportance + Send + Sync,
    E::Fitted: FeatureImportance + Send + Sync,
{
    type Fitted = BorutaSelector<E, Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> SklResult<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        let (n_samples, n_features) = x.dim();
        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Boruta requires at least 2 samples".to_string(),
            ));
        }

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_rng(&mut thread_rng())
        };

        let mut confirmed_features = Vec::new();
        let mut tentative_features: Vec<usize> = (0..n_features).collect();
        let mut rejected_features = Vec::new();

        let mut feature_hits = vec![0; n_features];
        let mut shadow_hits = 0;
        let mut iteration = 0;

        while iteration < self.max_iter && !tentative_features.is_empty() {
            // Create augmented dataset with shadow features
            let (x_augmented, _feature_mapping) =
                self.create_shadow_features(x, &tentative_features, &mut rng)?;

            // Train ensemble and get feature importances
            let importances = self.get_ensemble_importances(&x_augmented, y)?;

            // Calculate threshold based on shadow feature importances
            let n_real_features = tentative_features.len();
            let shadow_importances = &importances.slice(s![n_real_features..]);
            let shadow_threshold = self.calculate_threshold(shadow_importances)?;

            // Update hit counts
            shadow_hits += 1;
            for (i, &feature_idx) in tentative_features.iter().enumerate() {
                if importances[i] > shadow_threshold {
                    feature_hits[feature_idx] += 1;
                }
            }

            // Statistical tests for feature confirmation/rejection
            let _critical_value = self.calculate_critical_value(iteration + 1)?;

            let mut to_confirm = Vec::new();
            let mut to_reject = Vec::new();

            for &feature_idx in &tentative_features {
                let hit_rate = feature_hits[feature_idx] as f64 / (iteration + 1) as f64;
                let shadow_hit_rate = shadow_hits as f64 / (iteration + 1) as f64;

                // Binomial test for confirmation
                if self.binomial_test(feature_hits[feature_idx], iteration + 1, shadow_hit_rate)
                    < self.alpha
                    && hit_rate > shadow_hit_rate
                {
                    to_confirm.push(feature_idx);
                }

                // Binomial test for rejection (if two-step enabled)
                if self.two_step
                    && self.binomial_test(
                        shadow_hits - feature_hits[feature_idx],
                        iteration + 1,
                        1.0 - shadow_hit_rate,
                    ) < self.alpha
                    && hit_rate < shadow_hit_rate
                {
                    to_reject.push(feature_idx);
                }
            }

            // Update feature lists
            for feature in to_confirm {
                if let Some(pos) = tentative_features.iter().position(|&x| x == feature) {
                    tentative_features.remove(pos);
                    confirmed_features.push(feature);
                }
            }

            for feature in to_reject {
                if let Some(pos) = tentative_features.iter().position(|&x| x == feature) {
                    tentative_features.remove(pos);
                    rejected_features.push(feature);
                }
            }

            iteration += 1;
        }

        // Final feature importances (from last iteration)
        let final_x_augmented = self
            .create_shadow_features(x, &(0..n_features).collect::<Vec<_>>(), &mut rng)?
            .0;
        let final_importances = self.get_ensemble_importances(&final_x_augmented, y)?;
        let feature_importances = final_importances.slice(s![..n_features]).to_owned();
        let shadow_importances = final_importances.slice(s![n_features..]).to_owned();

        // Remaining tentative features are treated as rejected if max_iter reached
        if iteration >= self.max_iter {
            rejected_features.extend(tentative_features.clone());
            tentative_features.clear();
        }

        // Sort the lists for consistency
        confirmed_features.sort_unstable();
        tentative_features.sort_unstable();
        rejected_features.sort_unstable();

        Ok(BorutaSelector {
            estimator: self.estimator,
            n_estimators: self.n_estimators,
            max_iter: self.max_iter,
            alpha: self.alpha,
            two_step: self.two_step,
            percentile: self.percentile,
            random_state: self.random_state,
            state: PhantomData,
            confirmed_features_: Some(confirmed_features),
            tentative_features_: Some(tentative_features),
            rejected_features_: Some(rejected_features),
            feature_importances_: Some(feature_importances),
            shadow_importances_: Some(shadow_importances),
            n_features_: Some(n_features),
            n_iterations_: Some(iteration),
        })
    }
}

impl<E> BorutaSelector<E, Untrained>
where
    E: Clone + Fit<Array2<Float>, Array1<Float>> + FeatureImportance,
    E::Fitted: FeatureImportance,
{
    /// Create shadow features by permuting columns
    fn create_shadow_features(
        &self,
        x: &Array2<Float>,
        feature_indices: &[usize],
        rng: &mut StdRng,
    ) -> SklResult<(Array2<Float>, Vec<usize>)> {
        let (n_samples, _) = x.dim();
        let n_features = feature_indices.len();

        // Create augmented matrix with original + shadow features
        let mut x_augmented = Array2::<Float>::zeros((n_samples, n_features * 2));

        // Copy original features
        for (new_idx, &orig_idx) in feature_indices.iter().enumerate() {
            x_augmented.column_mut(new_idx).assign(&x.column(orig_idx));
        }

        // Create shadow features by permuting
        for (new_idx, &orig_idx) in feature_indices.iter().enumerate() {
            let mut shadow_column: Vec<Float> = x.column(orig_idx).to_vec();
            shadow_column.shuffle(rng);

            let shadow_idx = n_features + new_idx;
            for (row, &value) in shadow_column.iter().enumerate() {
                x_augmented[[row, shadow_idx]] = value;
            }
        }

        let feature_mapping = feature_indices.to_vec();
        Ok((x_augmented, feature_mapping))
    }

    /// Get ensemble feature importances
    fn get_ensemble_importances(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> SklResult<Array1<Float>> {
        let n_features = x.ncols();
        let mut total_importances = Array1::<Float>::zeros(n_features);

        // Simple ensemble - train multiple estimators and average importances
        for _ in 0..self.n_estimators {
            let estimator = self.estimator.clone();
            let trained = estimator.fit(x, y)?;
            let importances = trained.feature_importances()?;
            total_importances = total_importances + importances;
        }

        // Average the importances
        total_importances /= self.n_estimators as Float;

        Ok(total_importances)
    }

    /// Calculate threshold from shadow feature importances
    fn calculate_threshold(
        &self,
        shadow_importances: &ArrayBase<
            scirs2_core::ndarray::ViewRepr<&Float>,
            scirs2_core::ndarray::Dim<[usize; 1]>,
        >,
    ) -> SklResult<Float> {
        if shadow_importances.is_empty() {
            return Ok(0.0);
        }

        let mut sorted_importances: Vec<Float> = shadow_importances.to_vec();
        sorted_importances.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let index = ((self.percentile / 100.0) * sorted_importances.len() as f64).ceil() as usize;
        let threshold_index = (index - 1).min(sorted_importances.len() - 1);

        Ok(sorted_importances[threshold_index])
    }

    /// Calculate critical value for statistical test
    fn calculate_critical_value(&self, n_trials: usize) -> SklResult<f64> {
        // Bonferroni correction for multiple testing
        let corrected_alpha = self.alpha / n_trials as f64;
        Ok(corrected_alpha)
    }

    /// Simple binomial test (approximated)
    fn binomial_test(&self, successes: usize, trials: usize, expected_prob: f64) -> f64 {
        if trials == 0 {
            return 1.0;
        }

        let observed_prob = successes as f64 / trials as f64;

        // Use normal approximation for large n
        if trials > 30 {
            let mean = expected_prob;
            let variance = expected_prob * (1.0 - expected_prob) / trials as f64;
            let std_dev = variance.sqrt();

            if std_dev > 1e-10 {
                let z_score = (observed_prob - mean) / std_dev;
                // Two-tailed test
                2.0 * (1.0 - self.standard_normal_cdf(z_score.abs()))
            } else {
                1.0
            }
        } else {
            // For small n, use exact binomial (simplified)
            if (observed_prob - expected_prob).abs() < 1e-10 {
                1.0
            } else {
                0.01 // Conservative estimate
            }
        }
    }

    /// Standard normal CDF approximation
    fn standard_normal_cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + self.erf(x / 2.0_f64.sqrt()))
    }

    /// Error function approximation
    fn erf(&self, x: f64) -> f64 {
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }
}

impl<E> Transform<Array2<f64>> for BorutaSelector<E, Trained> {
    fn transform(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let support = self.get_support()?;
        let selected_indices: Vec<usize> = support
            .iter()
            .enumerate()
            .filter_map(|(i, &selected)| if selected { Some(i) } else { None })
            .collect();

        if selected_indices.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features were selected".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_selected = selected_indices.len();
        let mut result = Array2::<f64>::zeros((n_samples, n_selected));

        for (new_idx, &orig_idx) in selected_indices.iter().enumerate() {
            result.column_mut(new_idx).assign(&x.column(orig_idx));
        }

        Ok(result)
    }
}

impl<E> SelectorMixin for BorutaSelector<E, Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let n_features = self.n_features_.unwrap();
        let confirmed = self.confirmed_features_.as_ref().unwrap();
        let tentative = self.tentative_features_.as_ref().unwrap();

        let mut support = Array1::<bool>::from_elem(n_features, false);

        // Include confirmed features
        for &idx in confirmed {
            support[idx] = true;
        }

        // Optionally include tentative features
        for &idx in tentative {
            support[idx] = true;
        }

        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let confirmed = self.confirmed_features_.as_ref().unwrap();
        let tentative = self.tentative_features_.as_ref().unwrap();

        let mut selected_set = std::collections::HashSet::new();
        for &idx in confirmed {
            selected_set.insert(idx);
        }
        for &idx in tentative {
            selected_set.insert(idx);
        }

        let transformed: Vec<usize> = indices
            .iter()
            .filter(|&&idx| selected_set.contains(&idx))
            .enumerate()
            .map(|(new_idx, _)| new_idx)
            .collect();

        Ok(transformed)
    }
}

impl<E> BorutaSelector<E, Trained> {
    /// Get confirmed features (definitively relevant)
    pub fn confirmed_features(&self) -> &[usize] {
        self.confirmed_features_.as_ref().unwrap()
    }

    /// Get tentative features (undecided)
    pub fn tentative_features(&self) -> &[usize] {
        self.tentative_features_.as_ref().unwrap()
    }

    /// Get rejected features (definitively irrelevant)
    pub fn rejected_features(&self) -> &[usize] {
        self.rejected_features_.as_ref().unwrap()
    }

    /// Get feature importances from the final iteration
    pub fn feature_importances(&self) -> &Array1<Float> {
        self.feature_importances_.as_ref().unwrap()
    }

    /// Get shadow feature importances from the final iteration
    pub fn shadow_importances(&self) -> &Array1<Float> {
        self.shadow_importances_.as_ref().unwrap()
    }

    /// Get number of iterations performed
    pub fn n_iterations(&self) -> usize {
        self.n_iterations_.unwrap()
    }

    /// Get all selected features (confirmed + tentative)
    pub fn selected_features(&self) -> Vec<usize> {
        let mut selected = self.confirmed_features_.as_ref().unwrap().clone();
        selected.extend_from_slice(self.tentative_features_.as_ref().unwrap());
        selected.sort_unstable();
        selected
    }

    /// Check if the algorithm converged (no tentative features remaining)
    pub fn converged(&self) -> bool {
        self.tentative_features_.as_ref().unwrap().is_empty()
    }
}

/// Method for aggregating rankings from multiple selectors
#[derive(Debug, Clone)]
pub enum AggregationMethod {
    /// Simple average of ranks
    Average,
    /// Weighted average of ranks
    WeightedAverage,
    /// Median of ranks
    Median,
    /// Borda count method
    BordaCount,
    /// Rank aggregation using sum of reciprocal ranks
    ReciprocalRankFusion,
    /// Majority voting (for feature selection rather than ranking)
    MajorityVoting,
}

/// Trait for selector functions that can be used in ensemble
pub trait SelectorFunction: Send + Sync + std::fmt::Debug {
    fn compute_scores(
        &self,
        x: &Array2<f64>,
        y_classif: Option<&Array1<i32>>,
        y_regression: Option<&Array1<f64>>,
    ) -> SklResult<Array1<f64>>;
    fn name(&self) -> &str;
    fn clone_box(&self) -> Box<dyn SelectorFunction>;
}

impl Clone for Box<dyn SelectorFunction> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Wrapper for univariate statistical tests
#[derive(Debug, Clone)]
pub struct UnivariateSelector {
    method: UnivariateMethod,
}

#[derive(Debug, Clone)]
pub enum UnivariateMethod {
    /// Chi2
    Chi2,
    /// FClassif
    FClassif,
    /// FRegression
    FRegression,
    /// MutualInfoClassif
    MutualInfoClassif,
    /// MutualInfoRegression
    MutualInfoRegression,
    /// VarianceThreshold
    VarianceThreshold(f64),
}

impl UnivariateSelector {
    pub fn new(method: UnivariateMethod) -> Self {
        Self { method }
    }
}

impl SelectorFunction for UnivariateSelector {
    fn compute_scores(
        &self,
        x: &Array2<f64>,
        y_classif: Option<&Array1<i32>>,
        y_regression: Option<&Array1<f64>>,
    ) -> SklResult<Array1<f64>> {
        use crate::statistical_tests::*;

        match &self.method {
            UnivariateMethod::Chi2 => {
                let y = y_classif.ok_or_else(|| {
                    SklearsError::InvalidInput("Chi2 requires classification target".to_string())
                })?;
                let y_f64 = y.mapv(|v| v as f64);
                let (scores, _) = chi2(&x.view(), &y_f64.view())
                    .map_err(|e| SklearsError::FitError(format!("Chi2 test failed: {:?}", e)))?;
                Ok(scores)
            }
            UnivariateMethod::FClassif => {
                let y = y_classif.ok_or_else(|| {
                    SklearsError::InvalidInput(
                        "F-classif requires classification target".to_string(),
                    )
                })?;
                let y_f64 = y.mapv(|v| v as f64);
                let (scores, _) = f_classif(&x.view(), &y_f64.view())
                    .map_err(|e| SklearsError::FitError(format!("F-test failed: {:?}", e)))?;
                Ok(scores)
            }
            UnivariateMethod::FRegression => {
                let y = y_regression.ok_or_else(|| {
                    SklearsError::InvalidInput(
                        "F-regression requires regression target".to_string(),
                    )
                })?;
                let (scores, _) = f_regression(&x.view(), &y.view())
                    .map_err(|e| SklearsError::FitError(format!("F-regression failed: {:?}", e)))?;
                Ok(scores)
            }
            UnivariateMethod::MutualInfoClassif => {
                let y = y_classif.ok_or_else(|| {
                    SklearsError::InvalidInput(
                        "Mutual info classif requires classification target".to_string(),
                    )
                })?;
                let y_f64 = y.mapv(|v| v as f64);
                mutual_info_classif(&x.view(), &y_f64.view()).map_err(|e| {
                    SklearsError::FitError(format!("Mutual info classif failed: {:?}", e))
                })
            }
            UnivariateMethod::MutualInfoRegression => {
                let y = y_regression.ok_or_else(|| {
                    SklearsError::InvalidInput(
                        "Mutual info regression requires regression target".to_string(),
                    )
                })?;
                mutual_info_regression(&x.view(), &y.view()).map_err(|e| {
                    SklearsError::FitError(format!("Mutual info regression failed: {:?}", e))
                })
            }
            UnivariateMethod::VarianceThreshold(_threshold) => {
                let variances = x
                    .axis_iter(Axis(1))
                    .map(|col| {
                        let mean = col.mean().unwrap_or(0.0);
                        col.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0)
                    })
                    .collect::<Vec<f64>>();
                Ok(Array1::from(variances))
            }
        }
    }

    fn name(&self) -> &str {
        match &self.method {
            UnivariateMethod::Chi2 => "chi2",
            UnivariateMethod::FClassif => "f_classif",
            UnivariateMethod::FRegression => "f_regression",
            UnivariateMethod::MutualInfoClassif => "mutual_info_classif",
            UnivariateMethod::MutualInfoRegression => "mutual_info_regression",
            UnivariateMethod::VarianceThreshold(_) => "variance_threshold",
        }
    }

    fn clone_box(&self) -> Box<dyn SelectorFunction> {
        Box::new(self.clone())
    }
}

/// Ensemble feature ranking that combines multiple feature selection methods
/// Uses various ranking aggregation methods to combine rankings from different selectors
#[derive(Debug, Clone)]
pub struct EnsembleFeatureRanking<State = Untrained> {
    selectors: Vec<Box<dyn SelectorFunction>>,
    aggregation_method: AggregationMethod,
    weights: Option<Vec<f64>>,
    state: PhantomData<State>,
    // Trained state
    rankings_: Option<Vec<Array1<f64>>>,
    final_ranking_: Option<Array1<f64>>,
    n_features_: Option<usize>,
}

impl Default for EnsembleFeatureRanking<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl EnsembleFeatureRanking<Untrained> {
    /// Create a new ensemble feature ranking
    pub fn new() -> Self {
        Self {
            selectors: Vec::new(),
            aggregation_method: AggregationMethod::Average,
            weights: None,
            state: PhantomData,
            rankings_: None,
            final_ranking_: None,
            n_features_: None,
        }
    }

    /// Add a selector to the ensemble
    pub fn add_selector(mut self, selector: Box<dyn SelectorFunction>) -> Self {
        self.selectors.push(selector);
        self
    }

    /// Set the aggregation method
    pub fn aggregation_method(mut self, method: AggregationMethod) -> Self {
        self.aggregation_method = method;
        self
    }

    /// Set weights for weighted aggregation
    pub fn weights(mut self, weights: Vec<f64>) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Convenience method to add common selectors
    pub fn add_univariate_selectors_classif(mut self) -> Self {
        self.selectors
            .push(Box::new(UnivariateSelector::new(UnivariateMethod::Chi2)));
        self.selectors.push(Box::new(UnivariateSelector::new(
            UnivariateMethod::FClassif,
        )));
        self.selectors.push(Box::new(UnivariateSelector::new(
            UnivariateMethod::MutualInfoClassif,
        )));
        self
    }

    /// Convenience method to add common selectors for regression
    pub fn add_univariate_selectors_regression(mut self) -> Self {
        self.selectors.push(Box::new(UnivariateSelector::new(
            UnivariateMethod::FRegression,
        )));
        self.selectors.push(Box::new(UnivariateSelector::new(
            UnivariateMethod::MutualInfoRegression,
        )));
        self.selectors.push(Box::new(UnivariateSelector::new(
            UnivariateMethod::VarianceThreshold(0.0),
        )));
        self
    }

    /// Add tree-based importance selector
    pub fn add_tree_selector(mut self) -> Self {
        self.selectors.push(Box::new(TreeImportanceSelector::new()));
        self
    }
}

/// Tree-based importance selector placeholder
/// This is a placeholder implementation - the actual implementation should use tree-based models
#[derive(Debug, Clone)]
pub struct TreeImportanceSelector {}

impl Default for TreeImportanceSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl TreeImportanceSelector {
    pub fn new() -> Self {
        Self {}
    }
}

impl SelectorFunction for TreeImportanceSelector {
    fn compute_scores(
        &self,
        x: &Array2<f64>,
        _y_classif: Option<&Array1<i32>>,
        _y_regression: Option<&Array1<f64>>,
    ) -> SklResult<Array1<f64>> {
        // Placeholder implementation - should use actual tree-based feature importance
        let n_features = x.ncols();
        Ok(Array1::ones(n_features))
    }

    fn name(&self) -> &str {
        "tree_importance"
    }

    fn clone_box(&self) -> Box<dyn SelectorFunction> {
        Box::new(self.clone())
    }
}

impl Fit<Array2<f64>, Array1<i32>> for EnsembleFeatureRanking<Untrained> {
    type Fitted = EnsembleFeatureRanking<Trained>;

    fn fit(self, x: &Array2<f64>, y: &Array1<i32>) -> SklResult<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        let n_features = x.ncols();
        let n_selectors = self.selectors.len();

        if n_selectors == 0 {
            return Err(SklearsError::InvalidInput(
                "No selectors added to ensemble".to_string(),
            ));
        }

        // Compute rankings from each selector
        let mut rankings = Vec::with_capacity(n_selectors);

        for selector in &self.selectors {
            let scores = selector.compute_scores(x, Some(y), None)?;
            rankings.push(scores);
        }

        // Aggregate rankings
        let final_ranking = self.aggregate_rankings(&rankings)?;

        Ok(EnsembleFeatureRanking {
            selectors: self.selectors,
            aggregation_method: self.aggregation_method,
            weights: self.weights,
            state: PhantomData,
            rankings_: Some(rankings),
            final_ranking_: Some(final_ranking),
            n_features_: Some(n_features),
        })
    }
}

impl Fit<Array2<f64>, Array1<f64>> for EnsembleFeatureRanking<Untrained> {
    type Fitted = EnsembleFeatureRanking<Trained>;

    fn fit(self, x: &Array2<f64>, y: &Array1<f64>) -> SklResult<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        let n_features = x.ncols();
        let n_selectors = self.selectors.len();

        if n_selectors == 0 {
            return Err(SklearsError::InvalidInput(
                "No selectors added to ensemble".to_string(),
            ));
        }

        // Compute rankings from each selector
        let mut rankings = Vec::with_capacity(n_selectors);

        for selector in &self.selectors {
            let scores = selector.compute_scores(x, None, Some(y))?;
            rankings.push(scores);
        }

        // Aggregate rankings
        let final_ranking = self.aggregate_rankings(&rankings)?;

        Ok(EnsembleFeatureRanking {
            selectors: self.selectors,
            aggregation_method: self.aggregation_method,
            weights: self.weights,
            state: PhantomData,
            rankings_: Some(rankings),
            final_ranking_: Some(final_ranking),
            n_features_: Some(n_features),
        })
    }
}

impl EnsembleFeatureRanking<Untrained> {
    /// Aggregate rankings using the specified method
    fn aggregate_rankings(&self, rankings: &[Array1<f64>]) -> SklResult<Array1<f64>> {
        let n_features = rankings[0].len();
        let n_selectors = rankings.len();

        match &self.aggregation_method {
            AggregationMethod::Average => {
                let mut aggregated = Array1::<f64>::zeros(n_features);
                for ranking in rankings {
                    aggregated += ranking;
                }
                aggregated /= n_selectors as f64;
                Ok(aggregated)
            }

            AggregationMethod::WeightedAverage => {
                let weights = self.weights.as_ref().ok_or_else(|| {
                    SklearsError::InvalidInput("Weights required for weighted average".to_string())
                })?;

                if weights.len() != n_selectors {
                    return Err(SklearsError::InvalidInput(
                        "Number of weights must match number of selectors".to_string(),
                    ));
                }

                let mut aggregated = Array1::<f64>::zeros(n_features);
                let weight_sum: f64 = weights.iter().sum();

                for (ranking, &weight) in rankings.iter().zip(weights.iter()) {
                    aggregated += &(ranking * weight);
                }
                aggregated /= weight_sum;
                Ok(aggregated)
            }

            AggregationMethod::Median => {
                let mut aggregated = Array1::<f64>::zeros(n_features);
                for i in 0..n_features {
                    let mut feature_scores: Vec<f64> = rankings.iter().map(|r| r[i]).collect();
                    feature_scores
                        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                    if n_selectors % 2 == 0 {
                        aggregated[i] = (feature_scores[n_selectors / 2 - 1]
                            + feature_scores[n_selectors / 2])
                            / 2.0;
                    } else {
                        aggregated[i] = feature_scores[n_selectors / 2];
                    }
                }
                Ok(aggregated)
            }

            AggregationMethod::BordaCount => {
                // Convert scores to ranks, then aggregate ranks
                let mut aggregated = Array1::<f64>::zeros(n_features);

                for ranking in rankings {
                    let ranks = self.scores_to_ranks(ranking);
                    aggregated += &ranks;
                }

                // Lower rank sum means better feature (invert for final score)
                let max_rank = aggregated.iter().fold(0.0f64, |acc, &x| acc.max(x));
                aggregated.mapv_inplace(|rank| max_rank - rank);
                Ok(aggregated)
            }

            AggregationMethod::ReciprocalRankFusion => {
                let mut aggregated = Array1::<f64>::zeros(n_features);

                for ranking in rankings {
                    let ranks = self.scores_to_ranks(ranking);
                    for (i, &rank) in ranks.iter().enumerate() {
                        aggregated[i] += 1.0 / (rank + 60.0); // RRF with k=60
                    }
                }

                Ok(aggregated)
            }

            AggregationMethod::MajorityVoting => {
                // For majority voting, we need to threshold each ranking
                // This is more suitable for feature selection than ranking
                let mut vote_count = Array1::<f64>::zeros(n_features);

                for ranking in rankings {
                    // Use median as threshold for each selector
                    let mut sorted_scores = ranking.to_vec();
                    sorted_scores
                        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let threshold = sorted_scores[sorted_scores.len() / 2];

                    for (i, &score) in ranking.iter().enumerate() {
                        if score >= threshold {
                            vote_count[i] += 1.0;
                        }
                    }
                }

                Ok(vote_count)
            }
        }
    }

    /// Convert scores to ranks (higher score = lower rank number)
    fn scores_to_ranks(&self, scores: &Array1<f64>) -> Array1<f64> {
        let mut indexed_scores: Vec<(usize, f64)> = scores
            .indexed_iter()
            .map(|(idx, &score)| (idx, score))
            .collect();

        // Sort by score descending (higher score gets lower rank)
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut ranks = Array1::<f64>::zeros(scores.len());
        for (rank, &(idx, _)) in indexed_scores.iter().enumerate() {
            ranks[idx] = rank as f64;
        }

        ranks
    }
}

impl EnsembleFeatureRanking<Trained> {
    /// Get the final aggregated ranking
    pub fn feature_importances(&self) -> &Array1<f64> {
        self.final_ranking_.as_ref().unwrap()
    }

    /// Get rankings from individual selectors
    pub fn individual_rankings(&self) -> &[Array1<f64>] {
        self.rankings_.as_ref().unwrap()
    }

    /// Select top k features based on ensemble ranking
    pub fn select_k_best(&self, k: usize) -> Vec<usize> {
        let ranking = self.final_ranking_.as_ref().unwrap();
        let mut indexed_scores: Vec<(usize, f64)> = ranking
            .indexed_iter()
            .map(|(idx, &score)| (idx, score))
            .collect();

        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed_scores
            .into_iter()
            .take(k)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Select features based on threshold
    pub fn select_by_threshold(&self, threshold: f64) -> Vec<usize> {
        let ranking = self.final_ranking_.as_ref().unwrap();
        ranking
            .indexed_iter()
            .filter_map(|(idx, &score)| if score >= threshold { Some(idx) } else { None })
            .collect()
    }
}

impl Transform<Array2<f64>> for EnsembleFeatureRanking<Trained> {
    fn transform(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_features = self.n_features_.unwrap();
        if x.ncols() != n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                n_features,
                x.ncols()
            )));
        }

        let selected = self.select_k_best(n_features / 2);
        let n_samples = x.nrows();
        let n_selected = selected.len();

        let mut x_new = Array2::<f64>::zeros((n_samples, n_selected));
        for (new_idx, &old_idx) in selected.iter().enumerate() {
            x_new.column_mut(new_idx).assign(&x.column(old_idx));
        }

        Ok(x_new)
    }
}

impl SelectorMixin for EnsembleFeatureRanking<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        // Default: select top 50% of features
        let n_features = self.n_features_.unwrap();
        let k = n_features / 2;
        let selected = self.select_k_best(k);

        let mut support = vec![false; n_features];
        for &idx in &selected {
            support[idx] = true;
        }
        Ok(Array1::from(support))
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected = self.select_k_best(self.n_features_.unwrap() / 2);
        Ok(indices
            .iter()
            .filter_map(|&idx| selected.iter().position(|&f| f == idx))
            .collect())
    }
}
