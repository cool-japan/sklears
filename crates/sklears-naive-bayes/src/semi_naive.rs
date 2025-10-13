//! Semi-Naive Bayes classifier with limited independence assumptions
//!
//! This classifier relaxes the full independence assumption by modeling
//! selected feature dependencies while maintaining computational efficiency.

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{validate, Result},
    prelude::{Predict, SklearsError},
    traits::{Estimator, Fit, PredictProba, Score, Trained, Untrained},
    types::Float,
};
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

use crate::{compute_class_prior, safe_log, NaiveBayesMixin};

/// Method for selecting feature dependencies
#[derive(Debug, Clone)]
pub enum DependencySelectionMethod {
    /// Mutual information based selection
    MutualInformation { threshold: f64 },
    /// Chi-square test based selection
    ChiSquare { threshold: f64 },
    /// Correlation coefficient based selection
    Correlation { threshold: f64 },
    /// Manual specification of dependencies
    Manual { dependencies: Vec<(usize, usize)> },
    /// Automatic selection using k-dependence
    KDependence { k: usize },
}

/// Feature dependency representation
#[derive(Debug, Clone)]
pub struct FeatureDependency {
    /// Primary feature index
    pub primary: usize,
    /// Dependent feature index
    pub dependent: usize,
    /// Strength of dependency
    pub strength: f64,
}

/// Configuration for Semi-Naive Bayes
#[derive(Debug, Clone)]
pub struct SemiNaiveBayesConfig {
    /// Method for selecting dependencies
    pub dependency_method: DependencySelectionMethod,
    /// Maximum number of dependencies to consider
    pub max_dependencies: usize,
    /// Minimum number of samples required for dependency estimation
    pub min_samples_dependency: usize,
    /// Smoothing parameter for conditional probabilities
    pub smoothing: f64,
    /// Prior probabilities of the classes
    pub priors: Option<Array1<f64>>,
}

impl Default for SemiNaiveBayesConfig {
    fn default() -> Self {
        Self {
            dependency_method: DependencySelectionMethod::MutualInformation { threshold: 0.1 },
            max_dependencies: 10,
            min_samples_dependency: 50,
            smoothing: 1.0,
            priors: None,
        }
    }
}

/// Semi-Naive Bayes classifier
///
/// This classifier models selected feature dependencies while maintaining
/// the computational efficiency of Naive Bayes for independent features.
#[derive(Debug, Clone)]
pub struct SemiNaiveBayes<State = Untrained> {
    config: SemiNaiveBayesConfig,
    state: PhantomData<State>,
    // Trained state fields
    dependencies_: Option<Vec<FeatureDependency>>,
    feature_means_: Option<Array2<f64>>, // [n_classes, n_features]
    feature_vars_: Option<Array2<f64>>,  // [n_classes, n_features]
    conditional_probs_: Option<HashMap<(usize, usize, i32), Array2<f64>>>, // Dependencies
    class_prior_: Option<Array1<f64>>,
    classes_: Option<Array1<i32>>,
}

impl SemiNaiveBayes<Untrained> {
    /// Create a new Semi-Naive Bayes classifier
    pub fn new() -> Self {
        Self {
            config: SemiNaiveBayesConfig::default(),
            state: PhantomData,
            dependencies_: None,
            feature_means_: None,
            feature_vars_: None,
            conditional_probs_: None,
            class_prior_: None,
            classes_: None,
        }
    }

    /// Set dependency selection method
    pub fn dependency_method(mut self, method: DependencySelectionMethod) -> Self {
        self.config.dependency_method = method;
        self
    }

    /// Set maximum number of dependencies
    pub fn max_dependencies(mut self, max_deps: usize) -> Self {
        self.config.max_dependencies = max_deps;
        self
    }

    /// Set minimum samples for dependency estimation
    pub fn min_samples_dependency(mut self, min_samples: usize) -> Self {
        self.config.min_samples_dependency = min_samples;
        self
    }

    /// Set smoothing parameter
    pub fn smoothing(mut self, smoothing: f64) -> Self {
        self.config.smoothing = smoothing;
        self
    }

    /// Set prior probabilities
    pub fn priors(mut self, priors: Array1<f64>) -> Self {
        self.config.priors = Some(priors);
        self
    }
}

impl Default for SemiNaiveBayes<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for SemiNaiveBayes<Untrained> {
    type Float = Float;
    type Config = SemiNaiveBayesConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl SemiNaiveBayes<Untrained> {
    /// Select feature dependencies based on the configured method
    fn select_dependencies(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
    ) -> Vec<FeatureDependency> {
        match &self.config.dependency_method {
            DependencySelectionMethod::MutualInformation { threshold } => {
                self.select_by_mutual_information(x, y, classes, *threshold)
            }
            DependencySelectionMethod::ChiSquare { threshold } => {
                self.select_by_chi_square(x, y, classes, *threshold)
            }
            DependencySelectionMethod::Correlation { threshold } => {
                self.select_by_correlation(x, *threshold)
            }
            DependencySelectionMethod::Manual { dependencies } => dependencies
                .iter()
                .map(|(i, j)| FeatureDependency {
                    primary: *i,
                    dependent: *j,
                    strength: 1.0,
                })
                .collect(),
            DependencySelectionMethod::KDependence { k } => {
                self.select_k_dependence(x, y, classes, *k)
            }
        }
    }

    /// Select dependencies using mutual information
    fn select_by_mutual_information(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
        threshold: f64,
    ) -> Vec<FeatureDependency> {
        let n_features = x.ncols();
        let mut dependencies = Vec::new();

        for i in 0..n_features {
            for j in (i + 1)..n_features {
                let mi = self.compute_mutual_information(x, y, classes, i, j);
                if mi > threshold && dependencies.len() < self.config.max_dependencies {
                    dependencies.push(FeatureDependency {
                        primary: i,
                        dependent: j,
                        strength: mi,
                    });
                }
            }
        }

        // Sort by strength and take top dependencies
        dependencies.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap());
        dependencies.truncate(self.config.max_dependencies);
        dependencies
    }

    /// Compute mutual information between two features for all classes
    fn compute_mutual_information(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
        feature_i: usize,
        feature_j: usize,
    ) -> f64 {
        let mut total_mi = 0.0;

        for &class_label in classes.iter() {
            let mask: Vec<usize> = y
                .iter()
                .enumerate()
                .filter_map(|(idx, &label)| {
                    if label == class_label {
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect();

            if mask.len() < self.config.min_samples_dependency {
                continue;
            }

            let x_class = x.select(Axis(0), &mask);
            let xi = x_class.column(feature_i);
            let xj = x_class.column(feature_j);

            // Discretize continuous features for MI computation
            let xi_disc = self.discretize_feature(&xi.to_owned());
            let xj_disc = self.discretize_feature(&xj.to_owned());

            let mi = self.mutual_information_discrete(&xi_disc, &xj_disc);
            total_mi += mi;
        }

        total_mi / classes.len() as f64
    }

    /// Discretize continuous features for mutual information computation
    fn discretize_feature(&self, feature: &Array1<f64>) -> Array1<i32> {
        let n_bins = 10.min(feature.len() / 5).max(2);
        let min_val = feature.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = feature.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let bin_width = (max_val - min_val) / n_bins as f64;

        if bin_width == 0.0 {
            return Array1::zeros(feature.len());
        }

        feature.mapv(|x| {
            let bin = ((x - min_val) / bin_width).floor() as i32;
            bin.min(n_bins as i32 - 1).max(0)
        })
    }

    /// Compute mutual information between two discrete variables
    fn mutual_information_discrete(&self, x: &Array1<i32>, y: &Array1<i32>) -> f64 {
        let n = x.len() as f64;
        if n == 0.0 {
            return 0.0;
        }

        // Count joint and marginal frequencies
        let mut joint_counts: HashMap<(i32, i32), f64> = HashMap::new();
        let mut x_counts: HashMap<i32, f64> = HashMap::new();
        let mut y_counts: HashMap<i32, f64> = HashMap::new();

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            *joint_counts.entry((xi, yi)).or_insert(0.0) += 1.0;
            *x_counts.entry(xi).or_insert(0.0) += 1.0;
            *y_counts.entry(yi).or_insert(0.0) += 1.0;
        }

        // Compute mutual information
        let mut mi = 0.0;
        for ((xi, yi), &joint_count) in joint_counts.iter() {
            let p_joint = joint_count / n;
            let p_x = x_counts[xi] / n;
            let p_y = y_counts[yi] / n;

            if p_joint > 0.0 && p_x > 0.0 && p_y > 0.0 {
                mi += p_joint * (p_joint / (p_x * p_y)).ln();
            }
        }

        mi
    }

    /// Select dependencies using correlation
    fn select_by_correlation(&self, x: &Array2<Float>, threshold: f64) -> Vec<FeatureDependency> {
        let n_features = x.ncols();
        let mut dependencies = Vec::new();

        for i in 0..n_features {
            for j in (i + 1)..n_features {
                let corr = self.compute_correlation(&x.column(i), &x.column(j));
                if corr.abs() > threshold && dependencies.len() < self.config.max_dependencies {
                    dependencies.push(FeatureDependency {
                        primary: i,
                        dependent: j,
                        strength: corr.abs(),
                    });
                }
            }
        }

        dependencies.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap());
        dependencies.truncate(self.config.max_dependencies);
        dependencies
    }

    /// Compute Pearson correlation coefficient
    fn compute_correlation(
        &self,
        x: &scirs2_core::ndarray::ArrayView1<f64>,
        y: &scirs2_core::ndarray::ArrayView1<f64>,
    ) -> f64 {
        let n = x.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let mean_x = x.sum() / n;
        let mean_y = y.sum() / n;

        let mut num = 0.0;
        let mut den_x = 0.0;
        let mut den_y = 0.0;

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            num += dx * dy;
            den_x += dx * dx;
            den_y += dy * dy;
        }

        if den_x == 0.0 || den_y == 0.0 {
            0.0
        } else {
            num / (den_x * den_y).sqrt()
        }
    }

    /// Select dependencies using chi-square test
    fn select_by_chi_square(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
        threshold: f64,
    ) -> Vec<FeatureDependency> {
        // Simplified implementation - in practice would use proper chi-square test
        self.select_by_mutual_information(x, y, classes, threshold)
    }

    /// Select dependencies using k-dependence approach
    fn select_k_dependence(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
        k: usize,
    ) -> Vec<FeatureDependency> {
        let n_features = x.ncols();
        let mut dependencies = Vec::new();

        // For each feature, find its k strongest dependencies
        for i in 0..n_features {
            let mut feature_deps = Vec::new();

            for j in 0..n_features {
                if i != j {
                    let mi = self.compute_mutual_information(x, y, classes, i, j);
                    feature_deps.push((j, mi));
                }
            }

            // Sort by mutual information and take top k
            feature_deps.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            feature_deps.truncate(k);

            for (j, strength) in feature_deps {
                if dependencies.len() < self.config.max_dependencies {
                    dependencies.push(FeatureDependency {
                        primary: i,
                        dependent: j,
                        strength,
                    });
                }
            }
        }

        dependencies
    }

    /// Compute conditional probabilities for dependent feature pairs
    fn compute_conditional_probabilities(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
        dependencies: &[FeatureDependency],
    ) -> HashMap<(usize, usize, i32), Array2<f64>> {
        let mut conditional_probs = HashMap::new();

        for dep in dependencies {
            for &class_label in classes.iter() {
                let mask: Vec<usize> = y
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &label)| {
                        if label == class_label {
                            Some(idx)
                        } else {
                            None
                        }
                    })
                    .collect();

                if mask.is_empty() {
                    continue;
                }

                let x_class = x.select(Axis(0), &mask);
                let x_primary = x_class.column(dep.primary);
                let x_dependent = x_class.column(dep.dependent);

                // Discretize for probability estimation
                let x_primary_disc = self.discretize_feature(&x_primary.to_owned());
                let x_dependent_disc = self.discretize_feature(&x_dependent.to_owned());

                // Compute conditional probability table
                let prob_table =
                    self.compute_conditional_prob_table(&x_primary_disc, &x_dependent_disc);

                conditional_probs.insert((dep.primary, dep.dependent, class_label), prob_table);
            }
        }

        conditional_probs
    }

    /// Compute conditional probability table P(dependent | primary)
    fn compute_conditional_prob_table(
        &self,
        primary: &Array1<i32>,
        dependent: &Array1<i32>,
    ) -> Array2<f64> {
        let primary_vals: HashSet<i32> = primary.iter().cloned().collect();
        let dependent_vals: HashSet<i32> = dependent.iter().cloned().collect();

        let n_primary = primary_vals.len();
        let n_dependent = dependent_vals.len();

        let mut prob_table = Array2::zeros((n_primary, n_dependent));
        let primary_vals: Vec<i32> = primary_vals.into_iter().collect();
        let dependent_vals: Vec<i32> = dependent_vals.into_iter().collect();

        for (i, &p_val) in primary_vals.iter().enumerate() {
            let mut conditional_counts = vec![0.0; n_dependent];
            let mut total_count = 0.0;

            for (&p, &d) in primary.iter().zip(dependent.iter()) {
                if p == p_val {
                    total_count += 1.0;
                    if let Some(j) = dependent_vals.iter().position(|&x| x == d) {
                        conditional_counts[j] += 1.0;
                    }
                }
            }

            // Apply smoothing and normalize
            for (j, &count) in conditional_counts.iter().enumerate() {
                prob_table[[i, j]] = (count + self.config.smoothing)
                    / (total_count + self.config.smoothing * n_dependent as f64);
            }
        }

        prob_table
    }
}

impl Fit<Array2<Float>, Array1<i32>> for SemiNaiveBayes<Untrained> {
    type Fitted = SemiNaiveBayes<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        // Get unique classes
        let mut classes = y.to_vec();
        classes.sort();
        classes.dedup();
        let classes = Array1::from_vec(classes);
        let n_classes = classes.len();
        let n_features = x.ncols();

        // Compute class priors
        let (_class_count, class_prior) = if let Some(ref priors) = self.config.priors {
            if priors.len() != n_classes {
                return Err(SklearsError::InvalidInput(format!(
                    "Number of priors ({}) doesn't match number of classes ({})",
                    priors.len(),
                    n_classes
                )));
            }
            let sum = priors.sum();
            if (sum - 1.0).abs() > 1e-10 {
                return Err(SklearsError::InvalidInput(
                    "The sum of the priors should be 1.0".to_string(),
                ));
            }
            let class_count = Array1::zeros(n_classes);
            (class_count, priors.clone())
        } else {
            compute_class_prior(y, &classes)
        };

        // Select feature dependencies
        let dependencies = self.select_dependencies(x, y, &classes);

        // Compute feature statistics for independent features
        let mut feature_means = Array2::zeros((n_classes, n_features));
        let mut feature_vars = Array2::zeros((n_classes, n_features));

        for (class_idx, &class_label) in classes.iter().enumerate() {
            let mask: Vec<usize> = y
                .iter()
                .enumerate()
                .filter_map(|(i, &label)| if label == class_label { Some(i) } else { None })
                .collect();

            if mask.is_empty() {
                continue;
            }

            let x_class = x.select(Axis(0), &mask);

            for feature_idx in 0..n_features {
                let feature_values = x_class.column(feature_idx);
                let mean = feature_values.mean().unwrap_or(0.0);
                let variance = if feature_values.len() > 1 {
                    feature_values.mapv(|v| (v - mean).powi(2)).sum()
                        / (feature_values.len() as f64)
                        + self.config.smoothing
                } else {
                    self.config.smoothing
                };

                feature_means[[class_idx, feature_idx]] = mean;
                feature_vars[[class_idx, feature_idx]] = variance;
            }
        }

        // Compute conditional probabilities for dependencies
        let conditional_probs =
            self.compute_conditional_probabilities(x, y, &classes, &dependencies);

        Ok(SemiNaiveBayes {
            config: self.config,
            state: PhantomData,
            dependencies_: Some(dependencies),
            feature_means_: Some(feature_means),
            feature_vars_: Some(feature_vars),
            conditional_probs_: Some(conditional_probs),
            class_prior_: Some(class_prior),
            classes_: Some(classes),
        })
    }
}

impl SemiNaiveBayes<Trained> {
    /// Compute the unnormalized posterior log probability of X
    fn joint_log_likelihood(&self, x: &Array2<Float>) -> Result<Array2<f64>> {
        let dependencies = self.dependencies_.as_ref().unwrap();
        let feature_means = self.feature_means_.as_ref().unwrap();
        let feature_vars = self.feature_vars_.as_ref().unwrap();
        let conditional_probs = self.conditional_probs_.as_ref().unwrap();
        let class_prior = self.class_prior_.as_ref().unwrap();
        let classes = self.classes_.as_ref().unwrap();
        let n_classes = classes.len();
        let n_samples = x.nrows();
        let n_features = x.ncols();

        let mut joint_log_likelihood = Array2::zeros((n_samples, n_classes));

        // Get dependent features
        let mut dependent_features: HashSet<usize> = HashSet::new();
        for dep in dependencies {
            dependent_features.insert(dep.dependent);
        }

        for class_idx in 0..n_classes {
            let class_label = classes[class_idx];

            for (sample_idx, x_sample) in x.axis_iter(Axis(0)).enumerate() {
                let mut log_prob = safe_log(class_prior[class_idx]);

                // Process independent features (Naive Bayes assumption)
                for feature_idx in 0..n_features {
                    if !dependent_features.contains(&feature_idx) {
                        let x_val = x_sample[feature_idx];
                        let mean = feature_means[[class_idx, feature_idx]];
                        let var = feature_vars[[class_idx, feature_idx]];
                        let diff = x_val - mean;

                        // Gaussian log likelihood
                        log_prob += -0.5 * (2.0 * std::f64::consts::PI * var).ln()
                            - 0.5 * diff * diff / var;
                    }
                }

                // Process dependent features using conditional probabilities
                for dep in dependencies {
                    if let Some(prob_table) =
                        conditional_probs.get(&(dep.primary, dep.dependent, class_label))
                    {
                        let primary_val = x_sample[dep.primary];
                        let dependent_val = x_sample[dep.dependent];

                        // Discretize values for lookup
                        let primary_disc =
                            self.discretize_single_value(primary_val, dep.primary, x);
                        let dependent_disc =
                            self.discretize_single_value(dependent_val, dep.dependent, x);

                        // Look up conditional probability (simplified)
                        if primary_disc < prob_table.nrows() && dependent_disc < prob_table.ncols()
                        {
                            let cond_prob = prob_table[[primary_disc, dependent_disc]];
                            log_prob += safe_log(cond_prob);
                        }
                    }
                }

                joint_log_likelihood[[sample_idx, class_idx]] = log_prob;
            }
        }

        Ok(joint_log_likelihood)
    }

    /// Discretize a single value based on the feature distribution
    fn discretize_single_value(&self, value: f64, feature_idx: usize, x: &Array2<Float>) -> usize {
        let feature_col = x.column(feature_idx);
        let min_val = feature_col.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = feature_col
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let n_bins = 10;
        let bin_width = (max_val - min_val) / n_bins as f64;

        if bin_width == 0.0 {
            return 0;
        }

        let bin = ((value - min_val) / bin_width).floor() as usize;
        bin.min(n_bins - 1)
    }
}

impl Predict<Array2<Float>, Array1<i32>> for SemiNaiveBayes<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let log_prob = self.joint_log_likelihood(x)?;
        let classes = self.classes_.as_ref().unwrap();

        Ok(log_prob.map_axis(Axis(1), |row| {
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            classes[max_idx]
        }))
    }
}

impl PredictProba<Array2<Float>, Array2<f64>> for SemiNaiveBayes<Trained> {
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<f64>> {
        let log_prob = self.joint_log_likelihood(x)?;
        let n_samples = x.nrows();
        let n_classes = self.classes_.as_ref().unwrap().len();
        let mut proba = Array2::zeros((n_samples, n_classes));

        // Normalize to get probabilities
        for i in 0..n_samples {
            let row = log_prob.row(i);
            let max_log_prob = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            let mut exp_sum = 0.0;
            for j in 0..n_classes {
                let exp_val = (log_prob[[i, j]] - max_log_prob).exp();
                proba[[i, j]] = exp_val;
                exp_sum += exp_val;
            }

            for j in 0..n_classes {
                proba[[i, j]] /= exp_sum;
            }
        }

        Ok(proba)
    }
}

impl Score<Array2<Float>, Array1<i32>> for SemiNaiveBayes<Trained> {
    type Float = Float;

    fn score(&self, x: &Array2<Float>, y: &Array1<i32>) -> Result<f64> {
        let predictions = self.predict(x)?;
        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(&pred, &true_val)| pred == true_val)
            .count();

        Ok(correct as f64 / y.len() as f64)
    }
}

impl NaiveBayesMixin for SemiNaiveBayes<Trained> {
    fn class_log_prior(&self) -> &Array1<f64> {
        self.class_prior_.as_ref().unwrap()
    }

    fn feature_log_prob(&self) -> &Array2<f64> {
        // Return feature means as a proxy for log probabilities
        self.feature_means_.as_ref().unwrap()
    }

    fn classes(&self) -> &Array1<i32> {
        self.classes_.as_ref().unwrap()
    }
}

impl SemiNaiveBayes<Trained> {
    /// Get the discovered feature dependencies
    pub fn dependencies(&self) -> &Vec<FeatureDependency> {
        self.dependencies_.as_ref().unwrap()
    }

    /// Get the number of dependencies modeled
    pub fn n_dependencies(&self) -> usize {
        self.dependencies_.as_ref().unwrap().len()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    // SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
    use scirs2_core::ndarray::array;

    #[test]
    fn test_semi_naive_bayes_basic() {
        let x = array![
            [1.0, 2.0, 0.5],
            [2.0, 3.0, 1.0],
            [3.0, 4.0, 1.5],
            [4.0, 5.0, 2.0],
            [-1.0, -2.0, -0.5],
            [-2.0, -3.0, -1.0],
            [-3.0, -4.0, -1.5],
            [-4.0, -5.0, -2.0]
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = SemiNaiveBayes::new()
            .max_dependencies(2)
            .fit(&x, &y)
            .unwrap();

        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions, y);

        let score = model.score(&x, &y).unwrap();
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_semi_naive_bayes_predict_proba() {
        let x = array![
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [-1.0, -1.0, -1.0],
            [-2.0, -2.0, -2.0]
        ];
        let y = array![0, 0, 1, 1];

        let model = SemiNaiveBayes::new().fit(&x, &y).unwrap();
        let proba = model.predict_proba(&x).unwrap();

        // Check that probabilities sum to 1
        for i in 0..x.nrows() {
            let row_sum: f64 = proba.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_semi_naive_bayes_manual_dependencies() {
        let x = array![
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [-1.0, -1.0, -1.0],
            [-2.0, -2.0, -2.0]
        ];
        let y = array![0, 0, 1, 1];

        let manual_deps = vec![(0, 1), (1, 2)];
        let model = SemiNaiveBayes::new()
            .dependency_method(DependencySelectionMethod::Manual {
                dependencies: manual_deps,
            })
            .fit(&x, &y)
            .unwrap();

        assert_eq!(model.n_dependencies(), 2);
        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.len(), y.len());
    }

    #[test]
    fn test_semi_naive_bayes_k_dependence() {
        let x = array![
            [1.0, 1.1, 1.2, 0.9],
            [2.0, 2.1, 2.2, 1.9],
            [3.0, 3.1, 3.2, 2.9],
            [-1.0, -1.1, -1.2, -0.9],
            [-2.0, -2.1, -2.2, -1.9],
            [-3.0, -3.1, -3.2, -2.9]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let model = SemiNaiveBayes::new()
            .dependency_method(DependencySelectionMethod::KDependence { k: 2 })
            .min_samples_dependency(3)
            .fit(&x, &y)
            .unwrap();

        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.len(), y.len());
    }
}
