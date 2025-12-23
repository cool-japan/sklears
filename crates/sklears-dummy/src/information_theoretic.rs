//! Information-Theoretic baseline estimators
//!
//! This module provides baseline estimators based on information theory principles:
//! - Maximum entropy baselines
//! - Minimum description length (MDL) baselines  
//! - Mutual information baselines
//! - Entropy-based sampling methods
//! - Information gain baselines

use scirs2_core::ndarray::Array1;
use scirs2_core::random::{prelude::*, Rng};
// Note: SliceRandom not available, will implement manually where needed
use sklears_core::error::Result;
use sklears_core::types::{Features, Float, Int};
use std::collections::HashMap;

/// Information-theoretic strategy selection
#[derive(Debug, Clone, PartialEq)]
pub enum InformationTheoreticStrategy {
    /// Maximum entropy principle
    MaximumEntropy,
    /// Minimum description length
    MinimumDescriptionLength,
    /// Mutual information maximization
    MutualInformation,
    /// Entropy-based sampling
    EntropySampling,
    /// Information gain optimization
    InformationGain,
}

/// Maximum entropy estimator following the principle of maximum entropy
#[derive(Debug, Clone)]
pub struct MaximumEntropyEstimator {
    /// Constraints for maximum entropy optimization
    pub constraints: Option<Array1<Float>>,
    /// Lagrange multipliers
    pub lagrange_multipliers_: Option<Array1<Float>>,
    /// Estimated maximum entropy distribution
    pub max_entropy_distribution_: Option<Array1<Float>>,
    /// Entropy value of the distribution
    pub entropy_: Option<Float>,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl MaximumEntropyEstimator {
    /// Create new maximum entropy estimator
    pub fn new() -> Self {
        Self {
            constraints: None,
            lagrange_multipliers_: None,
            max_entropy_distribution_: None,
            entropy_: None,
            random_state: None,
        }
    }

    /// Set constraints for maximum entropy optimization
    pub fn with_constraints(mut self, constraints: Array1<Float>) -> Self {
        self.constraints = Some(constraints);
        self
    }

    /// Set random state
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Fit maximum entropy model
    pub fn fit_classification(&mut self, y: &Array1<Int>) -> Result<()> {
        let mut class_counts: HashMap<Int, usize> = HashMap::new();
        for &label in y.iter() {
            *class_counts.entry(label).or_insert(0) += 1;
        }

        let mut classes: Vec<Int> = class_counts.keys().copied().collect();
        classes.sort();
        let n_classes = classes.len();
        let n_samples = y.len() as Float;

        // If no constraints provided, use uniform distribution (max entropy)
        let distribution = if let Some(constraints) = self.constraints.clone() {
            if constraints.len() != n_classes {
                return Err(sklears_core::error::SklearsError::InvalidInput(
                    "Constraints length must match number of classes".to_string(),
                ));
            }

            // Optimize with constraints using method of Lagrange multipliers
            self.optimize_with_constraints(&constraints, &class_counts, &classes)?
        } else {
            // No constraints - uniform distribution maximizes entropy
            Array1::from_elem(n_classes, 1.0 / n_classes as Float)
        };

        // Compute entropy
        let entropy = self.compute_entropy(&distribution);

        self.max_entropy_distribution_ = Some(distribution);
        self.entropy_ = Some(entropy);
        Ok(())
    }

    /// Optimize maximum entropy distribution with constraints
    fn optimize_with_constraints(
        &mut self,
        constraints: &Array1<Float>,
        class_counts: &HashMap<Int, usize>,
        classes: &[Int],
    ) -> Result<Array1<Float>> {
        let n_classes = classes.len();

        // Use method of Lagrange multipliers
        // For simplicity, we'll use iterative scaling (not full optimization)
        let mut distribution = Array1::from_elem(n_classes, 1.0 / n_classes as Float);
        let max_iter = 100;
        let tolerance = 1e-6;

        for _iter in 0..max_iter {
            let old_dist = distribution.clone();

            // Update distribution to satisfy constraints
            for i in 0..n_classes {
                let observed_constraint = *class_counts.get(&classes[i]).unwrap() as Float;
                if constraints[i] > 0.0 {
                    // Scale to match constraint
                    distribution[i] *= constraints[i] / observed_constraint.max(1e-10);
                }
            }

            // Normalize to maintain probability constraint
            let sum = distribution.sum();
            distribution = distribution.mapv(|x| x / sum);

            // Check convergence
            let diff: Float = (&distribution - &old_dist).mapv(|x| x.abs()).sum();
            if diff < tolerance {
                break;
            }
        }

        Ok(distribution)
    }

    /// Compute entropy of a distribution
    fn compute_entropy(&self, distribution: &Array1<Float>) -> Float {
        let mut entropy = 0.0;
        for &p in distribution.iter() {
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }
        entropy
    }

    /// Get maximum entropy distribution
    pub fn max_entropy_distribution(&self) -> Option<&Array1<Float>> {
        self.max_entropy_distribution_.as_ref()
    }

    /// Get entropy value
    pub fn entropy(&self) -> Option<Float> {
        self.entropy_
    }
}

/// Minimum Description Length (MDL) estimator
#[derive(Debug, Clone)]
pub struct MDLEstimator {
    /// Model complexity penalty factor
    pub complexity_penalty: Float,
    /// Selected model complexity
    pub model_complexity_: Option<Float>,
    /// Data likelihood
    pub data_likelihood_: Option<Float>,
    /// MDL score (description length)
    pub mdl_score_: Option<Float>,
    /// Optimal distribution
    pub optimal_distribution_: Option<Array1<Float>>,
    /// Random state
    pub random_state: Option<u64>,
}

impl MDLEstimator {
    /// Create new MDL estimator
    pub fn new() -> Self {
        Self {
            complexity_penalty: 1.0,
            model_complexity_: None,
            data_likelihood_: None,
            mdl_score_: None,
            optimal_distribution_: None,
            random_state: None,
        }
    }

    /// Set complexity penalty factor
    pub fn with_complexity_penalty(mut self, penalty: Float) -> Self {
        self.complexity_penalty = penalty;
        self
    }

    /// Set random state
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Fit MDL model
    pub fn fit_classification(&mut self, y: &Array1<Int>) -> Result<()> {
        let mut class_counts: HashMap<Int, usize> = HashMap::new();
        for &label in y.iter() {
            *class_counts.entry(label).or_insert(0) += 1;
        }

        let mut classes: Vec<Int> = class_counts.keys().copied().collect();
        classes.sort();
        let n_classes = classes.len();
        let n_samples = y.len() as Float;

        // Empirical distribution
        let empirical_dist: Array1<Float> = classes
            .iter()
            .map(|&class| *class_counts.get(&class).unwrap() as Float / n_samples)
            .collect();

        // Compute data likelihood (negative log-likelihood)
        let mut data_likelihood = 0.0;
        for &prob in empirical_dist.iter() {
            if prob > 0.0 {
                data_likelihood -= prob * n_samples * prob.ln();
            }
        }

        // Compute model complexity (number of free parameters)
        let effective_params = (n_classes - 1) as Float; // Probability simplex has n-1 free parameters
        let model_complexity = self.complexity_penalty * effective_params * (n_samples.ln() / 2.0);

        // MDL score = data likelihood + model complexity
        let mdl_score = data_likelihood + model_complexity;

        // For baseline, use empirical distribution as optimal
        self.model_complexity_ = Some(model_complexity);
        self.data_likelihood_ = Some(data_likelihood);
        self.mdl_score_ = Some(mdl_score);
        self.optimal_distribution_ = Some(empirical_dist);
        Ok(())
    }

    /// Get model complexity
    pub fn model_complexity(&self) -> Option<Float> {
        self.model_complexity_
    }

    /// Get data likelihood
    pub fn data_likelihood(&self) -> Option<Float> {
        self.data_likelihood_
    }

    /// Get MDL score
    pub fn mdl_score(&self) -> Option<Float> {
        self.mdl_score_
    }

    /// Get optimal distribution
    pub fn optimal_distribution(&self) -> Option<&Array1<Float>> {
        self.optimal_distribution_.as_ref()
    }
}

/// Mutual Information baseline estimator
#[derive(Debug, Clone)]
pub struct MutualInformationEstimator {
    /// Number of bins for discretization
    pub n_bins: usize,
    /// Mutual information values per feature
    pub mutual_information_: Option<Array1<Float>>,
    /// Feature importance based on MI
    pub feature_importance_: Option<Array1<Float>>,
    /// Selected features based on MI threshold
    pub selected_features_: Option<Array1<usize>>,
    /// MI threshold for feature selection
    pub mi_threshold: Float,
    /// Random state
    pub random_state: Option<u64>,
}

impl MutualInformationEstimator {
    /// Create new mutual information estimator
    pub fn new() -> Self {
        Self {
            n_bins: 10,
            mutual_information_: None,
            feature_importance_: None,
            selected_features_: None,
            mi_threshold: 0.0,
            random_state: None,
        }
    }

    /// Set number of bins for discretization
    pub fn with_n_bins(mut self, n_bins: usize) -> Self {
        self.n_bins = n_bins;
        self
    }

    /// Set MI threshold for feature selection
    pub fn with_mi_threshold(mut self, threshold: Float) -> Self {
        self.mi_threshold = threshold;
        self
    }

    /// Set random state
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Fit mutual information model
    pub fn fit(&mut self, x: &Features, y: &Array1<Int>) -> Result<()> {
        if x.is_empty() || y.is_empty() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Input cannot be empty".to_string(),
            ));
        }

        if x.nrows() != y.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Number of samples in X and y must be equal".to_string(),
            ));
        }

        let n_features = x.ncols();
        let mut mi_values = Array1::<Float>::zeros(n_features);

        // Compute mutual information for each feature
        for feature_idx in 0..n_features {
            let feature_column = x.column(feature_idx);
            let mi = self.compute_mutual_information(&feature_column.to_owned(), y)?;
            mi_values[feature_idx] = mi;
        }

        // Normalize to get feature importance
        let max_mi = mi_values.iter().fold(0.0f64, |a, &b| a.max(b));
        let feature_importance = if max_mi > 0.0 {
            mi_values.mapv(|x| x / max_mi)
        } else {
            Array1::from_elem(n_features, 1.0 / n_features as Float)
        };

        // Select features above threshold
        let selected_features: Vec<usize> = mi_values
            .iter()
            .enumerate()
            .filter(|(_, &mi)| mi >= self.mi_threshold)
            .map(|(idx, _)| idx)
            .collect();

        self.mutual_information_ = Some(mi_values);
        self.feature_importance_ = Some(feature_importance);
        self.selected_features_ = Some(Array1::from_vec(selected_features));
        Ok(())
    }

    /// Compute mutual information between feature and target
    fn compute_mutual_information(
        &self,
        feature: &Array1<Float>,
        target: &Array1<Int>,
    ) -> Result<Float> {
        // Discretize continuous feature into bins
        let (feature_bins, target_bins) = self.discretize_feature_target(feature, target)?;

        // Compute joint and marginal distributions
        let mut joint_counts: HashMap<(usize, Int), usize> = HashMap::new();
        let mut feature_counts: HashMap<usize, usize> = HashMap::new();
        let mut target_counts: HashMap<Int, usize> = HashMap::new();

        for (&f_bin, &t_val) in feature_bins.iter().zip(target.iter()) {
            *joint_counts.entry((f_bin, t_val)).or_insert(0) += 1;
            *feature_counts.entry(f_bin).or_insert(0) += 1;
            *target_counts.entry(t_val).or_insert(0) += 1;
        }

        let n_samples = feature.len() as Float;
        let mut mi = 0.0;

        // Compute MI = sum p(x,y) * log(p(x,y) / (p(x) * p(y)))
        for ((f_bin, t_val), &joint_count) in joint_counts.iter() {
            let p_joint = joint_count as Float / n_samples;
            let p_feature = *feature_counts.get(f_bin).unwrap() as Float / n_samples;
            let p_target = *target_counts.get(t_val).unwrap() as Float / n_samples;

            if p_joint > 0.0 && p_feature > 0.0 && p_target > 0.0 {
                mi += p_joint * (p_joint / (p_feature * p_target)).ln();
            }
        }

        Ok(mi)
    }

    /// Discretize feature into bins and return bin indices
    fn discretize_feature_target(
        &self,
        feature: &Array1<Float>,
        _target: &Array1<Int>,
    ) -> Result<(Array1<usize>, Array1<Int>)> {
        let n_samples = feature.len();

        // Find min and max for binning
        let min_val = feature.iter().fold(Float::INFINITY, |a, &b| a.min(b));
        let max_val = feature.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));

        if min_val == max_val {
            // Constant feature - all samples in same bin
            return Ok((Array1::zeros(n_samples), Array1::from_vec(_target.to_vec())));
        }

        let bin_width = (max_val - min_val) / self.n_bins as Float;
        let mut feature_bins = Array1::<usize>::zeros(n_samples);

        for (i, &val) in feature.iter().enumerate() {
            let bin_idx = ((val - min_val) / bin_width) as usize;
            feature_bins[i] = bin_idx.min(self.n_bins - 1);
        }

        Ok((feature_bins, Array1::from_vec(_target.to_vec())))
    }

    /// Get mutual information values
    pub fn mutual_information(&self) -> Option<&Array1<Float>> {
        self.mutual_information_.as_ref()
    }

    /// Get feature importance
    pub fn feature_importance(&self) -> Option<&Array1<Float>> {
        self.feature_importance_.as_ref()
    }

    /// Get selected features
    pub fn selected_features(&self) -> Option<&Array1<usize>> {
        self.selected_features_.as_ref()
    }
}

/// Entropy-based sampling estimator
#[derive(Debug, Clone)]
pub struct EntropySamplingEstimator {
    /// Temperature parameter for entropy regularization
    pub temperature: Float,
    /// Entropy-regularized distribution
    pub entropy_distribution_: Option<Array1<Float>>,
    /// Entropy value
    pub entropy_: Option<Float>,
    /// Random state
    pub random_state: Option<u64>,
}

impl EntropySamplingEstimator {
    /// Create new entropy sampling estimator
    pub fn new() -> Self {
        Self {
            temperature: 1.0,
            entropy_distribution_: None,
            entropy_: None,
            random_state: None,
        }
    }

    /// Set temperature parameter
    pub fn with_temperature(mut self, temperature: Float) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set random state
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Fit entropy sampling model
    pub fn fit_classification(&mut self, y: &Array1<Int>) -> Result<()> {
        let mut class_counts: HashMap<Int, usize> = HashMap::new();
        for &label in y.iter() {
            *class_counts.entry(label).or_insert(0) += 1;
        }

        let mut classes: Vec<Int> = class_counts.keys().copied().collect();
        classes.sort();
        let n_classes = classes.len();
        let n_samples = y.len() as Float;

        // Empirical distribution
        let empirical_probs: Array1<Float> = classes
            .iter()
            .map(|&class| *class_counts.get(&class).unwrap() as Float / n_samples)
            .collect();

        // Apply temperature scaling for entropy regularization
        let entropy_probs = if self.temperature != 1.0 {
            // Temperature scaling: p_i^(1/T) / sum(p_j^(1/T))
            let scaled_probs: Array1<Float> = empirical_probs.mapv(|p| {
                if p > 0.0 {
                    p.powf(1.0 / self.temperature)
                } else {
                    0.0
                }
            });
            let sum = scaled_probs.sum();
            if sum > 0.0 {
                scaled_probs.mapv(|p| p / sum)
            } else {
                Array1::from_elem(n_classes, 1.0 / n_classes as Float)
            }
        } else {
            empirical_probs
        };

        // Compute entropy
        let entropy = self.compute_entropy(&entropy_probs);

        self.entropy_distribution_ = Some(entropy_probs);
        self.entropy_ = Some(entropy);
        Ok(())
    }

    /// Compute entropy of distribution
    fn compute_entropy(&self, distribution: &Array1<Float>) -> Float {
        let mut entropy = 0.0;
        for &p in distribution.iter() {
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }
        entropy
    }

    /// Sample from entropy-regularized distribution
    pub fn sample(&self, n_samples: usize) -> Result<Array1<Int>> {
        let distribution = self.entropy_distribution_.as_ref().ok_or_else(|| {
            sklears_core::error::SklearsError::InvalidInput("Model not fitted".to_string())
        })?;

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(0)
        };

        let cumulative: Vec<f64> = distribution
            .iter()
            .scan(0.0, |acc, &x| {
                *acc += x;
                Some(*acc)
            })
            .collect();

        let mut samples = Array1::<Int>::zeros(n_samples);
        for i in 0..n_samples {
            let rand_val: f64 = rng.gen();
            let idx = cumulative
                .iter()
                .position(|&x| x >= rand_val)
                .unwrap_or(cumulative.len() - 1);
            samples[i] = idx as Int;
        }

        Ok(samples)
    }

    /// Get entropy distribution
    pub fn entropy_distribution(&self) -> Option<&Array1<Float>> {
        self.entropy_distribution_.as_ref()
    }

    /// Get entropy value
    pub fn entropy(&self) -> Option<Float> {
        self.entropy_
    }
}

/// Information Gain estimator for feature selection and prediction
#[derive(Debug, Clone)]
pub struct InformationGainEstimator {
    /// Number of bins for discretization
    pub n_bins: usize,
    /// Information gain values per feature
    pub information_gain_: Option<Array1<Float>>,
    /// Feature ranking based on information gain
    pub feature_ranking_: Option<Array1<usize>>,
    /// Threshold for feature selection
    pub ig_threshold: Float,
    /// Random state
    pub random_state: Option<u64>,
}

impl InformationGainEstimator {
    /// Create new information gain estimator
    pub fn new() -> Self {
        Self {
            n_bins: 10,
            information_gain_: None,
            feature_ranking_: None,
            ig_threshold: 0.0,
            random_state: None,
        }
    }

    /// Set number of bins
    pub fn with_n_bins(mut self, n_bins: usize) -> Self {
        self.n_bins = n_bins;
        self
    }

    /// Set IG threshold
    pub fn with_ig_threshold(mut self, threshold: Float) -> Self {
        self.ig_threshold = threshold;
        self
    }

    /// Set random state
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Fit information gain model
    pub fn fit(&mut self, x: &Features, y: &Array1<Int>) -> Result<()> {
        if x.is_empty() || y.is_empty() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Input cannot be empty".to_string(),
            ));
        }

        let n_features = x.ncols();
        let mut ig_values = Array1::<Float>::zeros(n_features);

        // Compute target entropy (H(Y))
        let target_entropy = self.compute_target_entropy(y);

        // Compute information gain for each feature
        for feature_idx in 0..n_features {
            let feature_column = x.column(feature_idx);
            let conditional_entropy =
                self.compute_conditional_entropy(&feature_column.to_owned(), y)?;
            ig_values[feature_idx] = target_entropy - conditional_entropy;
        }

        // Create feature ranking
        let mut indexed_ig: Vec<(usize, Float)> = ig_values
            .iter()
            .enumerate()
            .map(|(i, &ig)| (i, ig))
            .collect();
        indexed_ig.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // Sort by IG descending
        let feature_ranking: Array1<usize> = indexed_ig.iter().map(|(idx, _)| *idx).collect();

        self.information_gain_ = Some(ig_values);
        self.feature_ranking_ = Some(feature_ranking);
        Ok(())
    }

    /// Compute target entropy H(Y)
    fn compute_target_entropy(&self, target: &Array1<Int>) -> Float {
        let mut target_counts: HashMap<Int, usize> = HashMap::new();
        for &label in target.iter() {
            *target_counts.entry(label).or_insert(0) += 1;
        }

        let n_samples = target.len() as Float;
        let mut entropy = 0.0;

        for &count in target_counts.values() {
            let p = count as Float / n_samples;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }

        entropy
    }

    /// Compute conditional entropy H(Y|X)
    fn compute_conditional_entropy(
        &self,
        feature: &Array1<Float>,
        target: &Array1<Int>,
    ) -> Result<Float> {
        // Discretize feature
        let feature_bins = self.discretize_feature(feature)?;

        // Compute conditional entropy
        let mut feature_bin_counts: HashMap<usize, usize> = HashMap::new();
        let mut conditional_counts: HashMap<(usize, Int), usize> = HashMap::new();

        for (&bin, &label) in feature_bins.iter().zip(target.iter()) {
            *feature_bin_counts.entry(bin).or_insert(0) += 1;
            *conditional_counts.entry((bin, label)).or_insert(0) += 1;
        }

        let n_samples = feature.len() as Float;
        let mut conditional_entropy = 0.0;

        for (&bin, &bin_count) in feature_bin_counts.iter() {
            let bin_prob = bin_count as Float / n_samples;
            let mut bin_entropy = 0.0;

            for &label in target.iter() {
                if let Some(&joint_count) = conditional_counts.get(&(bin, label)) {
                    let conditional_prob = joint_count as Float / bin_count as Float;
                    if conditional_prob > 0.0 {
                        bin_entropy -= conditional_prob * conditional_prob.ln();
                    }
                }
            }

            conditional_entropy += bin_prob * bin_entropy;
        }

        Ok(conditional_entropy)
    }

    /// Discretize feature into bins
    fn discretize_feature(&self, feature: &Array1<Float>) -> Result<Array1<usize>> {
        let n_samples = feature.len();
        let min_val = feature.iter().fold(Float::INFINITY, |a, &b| a.min(b));
        let max_val = feature.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));

        if min_val == max_val {
            return Ok(Array1::zeros(n_samples));
        }

        let bin_width = (max_val - min_val) / self.n_bins as Float;
        let mut bins = Array1::<usize>::zeros(n_samples);

        for (i, &val) in feature.iter().enumerate() {
            let bin_idx = ((val - min_val) / bin_width) as usize;
            bins[i] = bin_idx.min(self.n_bins - 1);
        }

        Ok(bins)
    }

    /// Get information gain values
    pub fn information_gain(&self) -> Option<&Array1<Float>> {
        self.information_gain_.as_ref()
    }

    /// Get feature ranking
    pub fn feature_ranking(&self) -> Option<&Array1<usize>> {
        self.feature_ranking_.as_ref()
    }
}

/// Default implementations
impl Default for MaximumEntropyEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for MDLEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for MutualInformationEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for EntropySamplingEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for InformationGainEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_maximum_entropy_basic() {
        let y = array![0, 0, 0, 1, 1, 2]; // 3 classes
        let mut estimator = MaximumEntropyEstimator::new().with_random_state(42);

        let result = estimator.fit_classification(&y);
        assert!(result.is_ok());

        let distribution = estimator.max_entropy_distribution().unwrap();
        assert_eq!(distribution.len(), 3);

        // Distribution should sum to 1
        let sum: Float = distribution.sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);

        let entropy = estimator.entropy().unwrap();
        assert!(entropy >= 0.0);
    }

    #[test]
    fn test_maximum_entropy_with_constraints() {
        let y = array![0, 0, 1, 1];
        let constraints = array![0.6, 0.4]; // Favor class 0

        let mut estimator = MaximumEntropyEstimator::new()
            .with_constraints(constraints)
            .with_random_state(42);

        let result = estimator.fit_classification(&y);
        assert!(result.is_ok());

        let distribution = estimator.max_entropy_distribution().unwrap();
        assert_eq!(distribution.len(), 2);

        // Should be close to constraints
        assert!(distribution[0] > distribution[1]);
    }

    #[test]
    fn test_mdl_estimator_basic() {
        let y = array![0, 0, 0, 1, 1, 2];
        let mut estimator = MDLEstimator::new()
            .with_complexity_penalty(0.5)
            .with_random_state(42);

        let result = estimator.fit_classification(&y);
        assert!(result.is_ok());

        assert!(estimator.model_complexity().is_some());
        assert!(estimator.data_likelihood().is_some());
        assert!(estimator.mdl_score().is_some());

        let distribution = estimator.optimal_distribution().unwrap();
        assert_eq!(distribution.len(), 3);

        let sum: Float = distribution.sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mutual_information_estimator() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let mut estimator = MutualInformationEstimator::new()
            .with_n_bins(3)
            .with_mi_threshold(0.1)
            .with_random_state(42);

        let result = estimator.fit(&x, &y);
        assert!(result.is_ok());

        let mi_values = estimator.mutual_information().unwrap();
        assert_eq!(mi_values.len(), 2);

        // All MI values should be non-negative
        for &mi in mi_values.iter() {
            assert!(mi >= 0.0);
        }

        let importance = estimator.feature_importance().unwrap();
        assert_eq!(importance.len(), 2);

        let selected = estimator.selected_features().unwrap();
        assert!(selected.len() <= 2);
    }

    #[test]
    fn test_entropy_sampling_estimator() {
        let y = array![0, 0, 0, 1, 1, 2];
        let mut estimator = EntropySamplingEstimator::new()
            .with_temperature(0.5)
            .with_random_state(42);

        let result = estimator.fit_classification(&y);
        assert!(result.is_ok());

        let distribution = estimator.entropy_distribution().unwrap();
        assert_eq!(distribution.len(), 3);

        let sum: Float = distribution.sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);

        let entropy = estimator.entropy().unwrap();
        assert!(entropy >= 0.0);

        // Test sampling
        let samples = estimator.sample(10).unwrap();
        assert_eq!(samples.len(), 10);

        // All samples should be valid class labels
        for &sample in samples.iter() {
            assert!(sample >= 0 && sample <= 2);
        }
    }

    #[test]
    fn test_information_gain_estimator() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let mut estimator = InformationGainEstimator::new()
            .with_n_bins(3)
            .with_ig_threshold(0.1)
            .with_random_state(42);

        let result = estimator.fit(&x, &y);
        assert!(result.is_ok());

        let ig_values = estimator.information_gain().unwrap();
        assert_eq!(ig_values.len(), 2);

        // Information gain should be non-negative
        for &ig in ig_values.iter() {
            assert!(ig >= 0.0);
        }

        let ranking = estimator.feature_ranking().unwrap();
        assert_eq!(ranking.len(), 2);

        // Ranking should contain valid feature indices
        for &rank in ranking.iter() {
            assert!(rank < 2);
        }
    }
}
