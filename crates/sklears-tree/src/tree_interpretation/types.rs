//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{Random, rng};
use scirs2_core::random::distributions::Normal;
use sklears_core::error::{Result, SklearsError};
use std::collections::HashMap;

use std::collections::{HashMap};

/// Perturbation strategies for generating local samples
#[derive(Debug, Clone, Copy)]
pub enum PerturbationStrategy {
    /// Gaussian noise around the instance
    Gaussian,
    /// Random sampling from training data distribution
    RandomSampling,
    /// Feature-wise perturbation with different distributions
    FeatureWise,
}
/// LIME (Local Interpretable Model-agnostic Explanations) for tree models
///
/// LIME explains individual predictions by approximating the model locally
/// with an interpretable model (linear regression) around the instance being explained.
#[derive(Debug, Clone)]
pub struct LimeExplainer {
    /// Number of samples to generate for perturbation
    pub n_samples: usize,
    /// Standard deviation for Gaussian perturbation
    pub sigma: f64,
    /// Distance metric for sample weighting
    pub distance_metric: DistanceMetric,
    /// Feature perturbation strategy
    pub perturbation_strategy: PerturbationStrategy,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}
impl LimeExplainer {
    /// Create a new LIME explainer
    pub fn new(
        n_samples: usize,
        sigma: f64,
        distance_metric: DistanceMetric,
        perturbation_strategy: PerturbationStrategy,
    ) -> Self {
        Self {
            n_samples,
            sigma,
            distance_metric,
            perturbation_strategy,
            random_seed: None,
        }
    }
    /// Set random seed for reproducibility
    pub fn with_random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }
    /// Explain a single prediction using LIME
    pub fn explain_instance<F>(
        &self,
        instance: &Array1<f64>,
        predict_fn: F,
        feature_names: Option<Vec<String>>,
    ) -> Result<LimeExplanation>
    where
        F: Fn(&Array2<f64>) -> Result<Array1<f64>>,
    {
        let perturbed_samples = self.generate_perturbations(instance)?;
        let predictions = predict_fn(&perturbed_samples)?;
        let distances = self.calculate_distances(instance, &perturbed_samples)?;
        let weights = self.calculate_weights(&distances)?;
        let (feature_importances, local_score) = self
            .fit_local_model(&perturbed_samples, &predictions, &weights)?;
        let instance_2d = instance.clone().insert_axis(scirs2_core::ndarray::Axis(0));
        let original_prediction = predict_fn(&instance_2d)?[0];
        Ok(LimeExplanation {
            instance: instance.clone(),
            prediction: original_prediction,
            feature_importances,
            feature_names,
            local_score,
            n_samples_used: perturbed_samples.nrows(),
        })
    }
    /// Generate perturbed samples around the instance
    fn generate_perturbations(&self, instance: &Array1<f64>) -> Result<Array2<f64>> {
        let mut rng = thread_rng();
        let n_features = instance.len();
        let mut samples = Array2::zeros((self.n_samples, n_features));
        match self.perturbation_strategy {
            PerturbationStrategy::Gaussian => {
                for i in 0..self.n_samples {
                    for j in 0..n_features {
                        let normal = Normal::new(instance[j], self.sigma)
                            .map_err(|e| {
                                SklearsError::InvalidInput(
                                    format!("Failed to create normal distribution: {}", e),
                                )
                            })?;
                        samples[(i, j)] = normal.sample(&mut rng);
                    }
                }
            }
            PerturbationStrategy::RandomSampling => {
                for i in 0..self.n_samples {
                    for j in 0..n_features {
                        let noise = rng.random_range(-self.sigma..self.sigma);
                        samples[(i, j)] = instance[j] + noise;
                    }
                }
            }
            PerturbationStrategy::FeatureWise => {
                for i in 0..self.n_samples {
                    for j in 0..n_features {
                        if rng.gen_bool(0.8) {
                            let normal = Normal::new(instance[j], self.sigma)
                                .map_err(|e| {
                                    SklearsError::InvalidInput(
                                        format!("Failed to create normal distribution: {}", e),
                                    )
                                })?;
                            samples[(i, j)] = normal.sample(&mut rng);
                        } else {
                            let normal = Normal::new(instance[j], self.sigma * 2.0)
                                .map_err(|e| {
                                    SklearsError::InvalidInput(
                                        format!("Failed to create normal distribution: {}", e),
                                    )
                                })?;
                            samples[(i, j)] = normal.sample(&mut rng);
                        }
                    }
                }
            }
        }
        Ok(samples)
    }
    /// Calculate distances from instance to perturbed samples
    fn calculate_distances(
        &self,
        instance: &Array1<f64>,
        samples: &Array2<f64>,
    ) -> Result<Array1<f64>> {
        let n_samples = samples.nrows();
        let mut distances = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let sample = samples.row(i);
            distances[i] = match self.distance_metric {
                DistanceMetric::Euclidean => {
                    instance
                        .iter()
                        .zip(sample.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt()
                }
                DistanceMetric::Manhattan => {
                    instance
                        .iter()
                        .zip(sample.iter())
                        .map(|(a, b)| (a - b).abs())
                        .sum::<f64>()
                }
                DistanceMetric::Cosine => {
                    let dot_product: f64 = instance
                        .iter()
                        .zip(sample.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    let norm_a: f64 = instance
                        .iter()
                        .map(|x| x.powi(2))
                        .sum::<f64>()
                        .sqrt();
                    let norm_b: f64 = sample
                        .iter()
                        .map(|x| x.powi(2))
                        .sum::<f64>()
                        .sqrt();
                    if norm_a == 0.0 || norm_b == 0.0 {
                        1.0
                    } else {
                        1.0 - (dot_product / (norm_a * norm_b))
                    }
                }
            };
        }
        Ok(distances)
    }
    /// Calculate weights based on distances (closer samples get higher weights)
    fn calculate_weights(&self, distances: &Array1<f64>) -> Result<Array1<f64>> {
        let kernel_width = distances.iter().cloned().fold(0.0f64, f64::max) / 4.0;
        let weights = distances
            .mapv(|d| (-d.powi(2) / (2.0 * kernel_width.powi(2))).exp());
        Ok(weights)
    }
    /// Fit local linear model using weighted least squares
    fn fit_local_model(
        &self,
        samples: &Array2<f64>,
        predictions: &Array1<f64>,
        weights: &Array1<f64>,
    ) -> Result<(Array1<f64>, f64)> {
        let n_samples = samples.nrows();
        let n_features = samples.ncols();
        if n_samples == 0 || n_features == 0 {
            return Err(
                SklearsError::InvalidInput("Empty samples or features".to_string()),
            );
        }
        let mut X = Array2::ones((n_samples, n_features + 1));
        for i in 0..n_samples {
            for j in 0..n_features {
                X[(i, j + 1)] = samples[(i, j)] * weights[i];
            }
            X[(i, 0)] *= weights[i];
        }
        let y_weighted = predictions
            .iter()
            .zip(weights.iter())
            .map(|(pred, weight)| pred * weight)
            .collect::<Array1<f64>>();
        let xtx = X.t().dot(&X);
        let xty = X.t().dot(&y_weighted);
        let coefficients = self.solve_linear_system(&xtx, &xty)?;
        let feature_importances = coefficients
            .slice(scirs2_core::ndarray::s![1..])
            .to_owned();
        let y_pred = X.dot(&coefficients);
        let local_score = self.calculate_r_squared(&y_weighted, &y_pred);
        Ok((feature_importances, local_score))
    }
    /// Simple linear system solver for weighted least squares
    fn solve_linear_system(
        &self,
        A: &Array2<f64>,
        b: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let n = A.nrows();
        if n != A.ncols() || n != b.len() {
            return Err(
                SklearsError::InvalidInput("Matrix dimensions mismatch".to_string()),
            );
        }
        let mut augmented = Array2::zeros((n, n + 1));
        for i in 0..n {
            for j in 0..n {
                augmented[(i, j)] = A[(i, j)];
            }
            augmented[(i, n)] = b[i];
        }
        for i in 0..n {
            let mut max_row = i;
            for k in i + 1..n {
                if augmented[(k, i)].abs() > augmented[(max_row, i)].abs() {
                    max_row = k;
                }
            }
            if max_row != i {
                for j in 0..=n {
                    let temp = augmented[(i, j)];
                    augmented[(i, j)] = augmented[(max_row, j)];
                    augmented[(max_row, j)] = temp;
                }
            }
            if augmented[(i, i)].abs() < 1e-10 {
                return Err(
                    SklearsError::InvalidInput(
                        "Matrix is singular or near-singular".to_string(),
                    ),
                );
            }
            let pivot = augmented[(i, i)];
            for j in i..=n {
                augmented[(i, j)] /= pivot;
            }
            for k in i + 1..n {
                let factor = augmented[(k, i)];
                for j in i..=n {
                    augmented[(k, j)] -= factor * augmented[(i, j)];
                }
            }
        }
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            x[i] = augmented[(i, n)];
            for j in i + 1..n {
                x[i] -= augmented[(i, j)] * x[j];
            }
        }
        Ok(x)
    }
    /// Calculate R-squared score
    fn calculate_r_squared(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        let y_mean = y_true.mean().unwrap_or(0.0);
        let ss_tot: f64 = y_true.iter().map(|y| (y - y_mean).powi(2)).sum();
        let ss_res: f64 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(y_t, y_p)| (y_t - y_p).powi(2))
            .sum();
        if ss_tot == 0.0 { 1.0 } else { 1.0 - (ss_res / ss_tot) }
    }
}
/// Tree node representation for path extraction
#[derive(Debug, Clone)]
pub enum TreeNode {
    /// Internal node with split condition
    Internal {
        feature_idx: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
        n_samples: Option<usize>,
        impurity: Option<f64>,
    },
    /// Leaf node with prediction
    Leaf {
        prediction: f64,
        confidence: Option<f64>,
        n_samples: Option<usize>,
        impurity: Option<f64>,
        class_distribution: Option<HashMap<usize, usize>>,
    },
}
/// Predicate operators for anchors
#[derive(Debug, Clone, PartialEq)]
pub enum PredicateOperator {
    /// Less than or equal to (<=)
    LessEqualThan,
    /// Greater than (>)
    GreaterThan,
    /// Equal to (categorical)
    Equal,
    /// In range [min, max]
    InRange,
}
/// A single predicate in an anchor
#[derive(Debug, Clone, PartialEq)]
pub struct AnchorPredicate {
    /// Feature index
    pub feature_idx: usize,
    /// Feature name (if available)
    pub feature_name: Option<String>,
    /// Predicate operator
    pub operator: PredicateOperator,
    /// Threshold or category values
    pub value: PredicateValue,
}
impl AnchorPredicate {
    /// Create a human-readable string representation of the predicate
    pub fn to_string(&self, feature_names: &Option<Vec<String>>) -> String {
        let default_name = format!("feature_{}", self.feature_idx);
        let feature_name = feature_names
            .as_ref()
            .and_then(|names| names.get(self.feature_idx))
            .map(|s| s.as_str())
            .unwrap_or(&default_name);
        match (&self.operator, &self.value) {
            (PredicateOperator::LessEqualThan, PredicateValue::Threshold(threshold)) => {
                format!("{} <= {:.3}", feature_name, threshold)
            }
            (PredicateOperator::GreaterThan, PredicateValue::Threshold(threshold)) => {
                format!("{} > {:.3}", feature_name, threshold)
            }
            (PredicateOperator::InRange, PredicateValue::Range(min, max)) => {
                format!("{:.3} <= {} <= {:.3}", min, feature_name, max)
            }
            (PredicateOperator::Equal, PredicateValue::Category(category)) => {
                format!("{} = {}", feature_name, category)
            }
            _ => format!("{} <condition>", feature_name),
        }
    }
}
/// A complete decision path from root to leaf
#[derive(Debug, Clone)]
pub struct DecisionPath {
    /// Sequence of conditions leading to the prediction
    pub conditions: Vec<DecisionCondition>,
    /// Final prediction value
    pub prediction: f64,
    /// Confidence/probability of the prediction
    pub confidence: Option<f64>,
    /// Number of samples in the leaf node
    pub n_samples: Option<usize>,
    /// Impurity of the leaf node
    pub impurity: Option<f64>,
    /// Class distribution (for classification)
    pub class_distribution: Option<HashMap<usize, usize>>,
}
impl DecisionPath {
    /// Create a new decision path
    pub fn new(
        conditions: Vec<DecisionCondition>,
        prediction: f64,
        confidence: Option<f64>,
        n_samples: Option<usize>,
        impurity: Option<f64>,
        class_distribution: Option<HashMap<usize, usize>>,
    ) -> Self {
        Self {
            conditions,
            prediction,
            confidence,
            n_samples,
            impurity,
            class_distribution,
        }
    }
    /// Check if a sample satisfies all conditions in this path
    pub fn matches_sample(&self, sample: &Array1<f64>) -> bool {
        for condition in &self.conditions {
            if condition.feature_idx >= sample.len() {
                return false;
            }
            let feature_value = sample[condition.feature_idx];
            let satisfies = match condition.operator {
                ComparisonOperator::LessThanOrEqual => {
                    feature_value <= condition.threshold
                }
                ComparisonOperator::GreaterThan => feature_value > condition.threshold,
                ComparisonOperator::Equal => {
                    (feature_value - condition.threshold).abs() < f64::EPSILON
                }
                ComparisonOperator::NotEqual => {
                    (feature_value - condition.threshold).abs() >= f64::EPSILON
                }
                _ => false,
            };
            if !satisfies {
                return false;
            }
        }
        true
    }
    /// Get the depth of this path (number of conditions)
    pub fn depth(&self) -> usize {
        self.conditions.len()
    }
    /// Check if this is a pure leaf (all samples have same class)
    pub fn is_pure(&self) -> bool {
        if let Some(ref dist) = self.class_distribution {
            dist.len() <= 1
        } else {
            self.impurity.map_or(false, |imp| imp < f64::EPSILON)
        }
    }
}
/// Comparison operators for decision conditions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComparisonOperator {
    /// Less than or equal to
    LessThanOrEqual,
    /// Greater than
    GreaterThan,
    /// Equal to (for categorical features)
    Equal,
    /// Not equal to (for categorical features)
    NotEqual,
    /// In set (for categorical features)
    In,
    /// Not in set (for categorical features)
    NotIn,
}
/// Anchor explanations for tree models
///
/// Anchors are rule-based explanations that identify minimal sets of conditions
/// that sufficiently anchor a prediction. An anchor is a set of predicates such that
/// when they hold, the prediction remains the same with high probability.
#[derive(Debug, Clone)]
pub struct AnchorExplainer {
    /// Precision threshold for anchor acceptance
    pub precision_threshold: f64,
    /// Maximum number of features to consider in an anchor
    pub max_anchor_size: usize,
    /// Number of samples for coverage estimation
    pub coverage_samples: usize,
    /// Beam width for search
    pub beam_width: usize,
    /// Minimum coverage required for an anchor
    pub min_coverage: f64,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}
impl AnchorExplainer {
    /// Create a new anchor explainer
    pub fn new(
        precision_threshold: f64,
        max_anchor_size: usize,
        coverage_samples: usize,
        beam_width: usize,
        min_coverage: f64,
    ) -> Self {
        Self {
            precision_threshold,
            max_anchor_size,
            coverage_samples,
            beam_width,
            min_coverage,
            random_seed: None,
        }
    }
    /// Set random seed for reproducibility
    pub fn with_random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }
    /// Generate anchor explanation for an instance
    pub fn explain_instance<F>(
        &self,
        instance: &Array1<f64>,
        predict_fn: F,
        feature_names: Option<Vec<String>>,
    ) -> Result<AnchorExplanation>
    where
        F: Fn(&Array2<f64>) -> Result<Array1<f64>>,
    {
        let instance_2d = instance.clone().insert_axis(scirs2_core::ndarray::Axis(0));
        let original_prediction = predict_fn(&instance_2d)?[0];
        let candidate_predicates = self
            .generate_candidate_predicates(instance, &feature_names)?;
        let best_anchor = self
            .beam_search_anchor(
                instance,
                original_prediction,
                &candidate_predicates,
                &predict_fn,
            )?;
        let (precision, coverage, n_samples) = self
            .evaluate_anchor(
                &best_anchor.predicates,
                original_prediction,
                instance.len(),
                &predict_fn,
            )?;
        Ok(AnchorExplanation {
            instance: instance.clone(),
            prediction: original_prediction,
            anchor: best_anchor.predicates,
            precision,
            coverage,
            n_samples_evaluated: n_samples,
            feature_names,
        })
    }
    /// Generate candidate predicates based on the instance
    fn generate_candidate_predicates(
        &self,
        instance: &Array1<f64>,
        feature_names: &Option<Vec<String>>,
    ) -> Result<Vec<AnchorPredicate>> {
        let mut predicates = Vec::new();
        for (feature_idx, &value) in instance.iter().enumerate() {
            let feature_name = feature_names
                .as_ref()
                .and_then(|names| names.get(feature_idx))
                .cloned();
            predicates
                .push(AnchorPredicate {
                    feature_idx,
                    feature_name: feature_name.clone(),
                    operator: PredicateOperator::LessEqualThan,
                    value: PredicateValue::Threshold(value),
                });
            predicates
                .push(AnchorPredicate {
                    feature_idx,
                    feature_name: feature_name.clone(),
                    operator: PredicateOperator::GreaterThan,
                    value: PredicateValue::Threshold(value),
                });
            let range_width = value.abs() * 0.2;
            if range_width > 0.0 {
                predicates
                    .push(AnchorPredicate {
                        feature_idx,
                        feature_name: feature_name.clone(),
                        operator: PredicateOperator::InRange,
                        value: PredicateValue::Range(
                            value - range_width,
                            value + range_width,
                        ),
                    });
            }
        }
        Ok(predicates)
    }
    /// Beam search to find the best anchor
    fn beam_search_anchor<F>(
        &self,
        _instance: &Array1<f64>,
        target_prediction: f64,
        candidate_predicates: &[AnchorPredicate],
        predict_fn: F,
    ) -> Result<AnchorCandidate>
    where
        F: Fn(&Array2<f64>) -> Result<Array1<f64>>,
    {
        let mut beam: Vec<AnchorCandidate> = vec![
            AnchorCandidate { predicates : vec![], precision : 0.0, coverage : 1.0, score
            : 0.0, }
        ];
        for _depth in 1..=self.max_anchor_size {
            let mut new_candidates = Vec::new();
            for candidate in &beam {
                for predicate in candidate_predicates {
                    if candidate
                        .predicates
                        .iter()
                        .any(|p| p.feature_idx == predicate.feature_idx)
                    {
                        continue;
                    }
                    let mut new_predicates = candidate.predicates.clone();
                    new_predicates.push(predicate.clone());
                    let (precision, coverage, _) = self
                        .evaluate_anchor(
                            &new_predicates,
                            target_prediction,
                            _instance.len(),
                            &predict_fn,
                        )?;
                    if coverage < self.min_coverage {
                        continue;
                    }
                    let score = self.calculate_anchor_score(precision, coverage);
                    new_candidates
                        .push(AnchorCandidate {
                            predicates: new_predicates,
                            precision,
                            coverage,
                            score,
                        });
                }
            }
            if new_candidates.is_empty() {
                break;
            }
            new_candidates
                .sort_by(|a, b| {
                    b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
                });
            new_candidates.truncate(self.beam_width);
            for candidate in &new_candidates {
                if candidate.precision >= self.precision_threshold {
                    return Ok(candidate.clone());
                }
            }
            beam = new_candidates;
        }
        beam.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(
            beam
                .into_iter()
                .next()
                .unwrap_or(AnchorCandidate {
                    predicates: vec![],
                    precision: 0.0,
                    coverage: 1.0,
                    score: 0.0,
                }),
        )
    }
    /// Evaluate an anchor's precision and coverage
    fn evaluate_anchor<F>(
        &self,
        predicates: &[AnchorPredicate],
        target_prediction: f64,
        n_features: usize,
        predict_fn: F,
    ) -> Result<(f64, f64, usize)>
    where
        F: Fn(&Array2<f64>) -> Result<Array1<f64>>,
    {
        let mut rng = thread_rng();
        if predicates.is_empty() {
            return Ok((0.0, 1.0, 0));
        }
        let mut feature_bounds = vec![(- 5.0, 5.0); n_features];
        for predicate in predicates {
            let (min, max) = &mut feature_bounds[predicate.feature_idx];
            match &predicate.value {
                PredicateValue::Threshold(threshold) => {
                    match predicate.operator {
                        PredicateOperator::LessEqualThan => {
                            *max = (*max as f64).min(*threshold);
                        }
                        PredicateOperator::GreaterThan => {
                            *min = (*min as f64).max(*threshold);
                        }
                        _ => {}
                    }
                }
                PredicateValue::Range(range_min, range_max) => {
                    *min = (*min as f64).max(*range_min);
                    *max = (*max as f64).min(*range_max);
                }
                _ => {}
            }
        }
        let mut total_samples = 0;
        let mut covered_samples = 0;
        let mut correct_predictions = 0;
        for _ in 0..self.coverage_samples {
            let mut sample = Array1::zeros(n_features);
            for i in 0..n_features {
                sample[i] = rng.random_range(-5.0..5.0);
            }
            total_samples += 1;
            if self.sample_satisfies_anchor(&sample, predicates) {
                covered_samples += 1;
                let sample_2d = sample
                    .clone()
                    .insert_axis(scirs2_core::ndarray::Axis(0));
                let prediction = predict_fn(&sample_2d)?[0];
                if (prediction - target_prediction).abs() < 1e-6 {
                    correct_predictions += 1;
                }
            }
        }
        let coverage = covered_samples as f64 / total_samples as f64;
        let precision = if covered_samples > 0 {
            correct_predictions as f64 / covered_samples as f64
        } else {
            0.0
        };
        Ok((precision, coverage, total_samples))
    }
    /// Check if a sample satisfies all predicates in an anchor
    fn sample_satisfies_anchor(
        &self,
        sample: &Array1<f64>,
        predicates: &[AnchorPredicate],
    ) -> bool {
        for predicate in predicates {
            if predicate.feature_idx >= sample.len() {
                return false;
            }
            let feature_value = sample[predicate.feature_idx];
            let satisfies = match (&predicate.operator, &predicate.value) {
                (
                    PredicateOperator::LessEqualThan,
                    PredicateValue::Threshold(threshold),
                ) => feature_value <= *threshold,
                (
                    PredicateOperator::GreaterThan,
                    PredicateValue::Threshold(threshold),
                ) => feature_value > *threshold,
                (PredicateOperator::InRange, PredicateValue::Range(min, max)) => {
                    feature_value >= *min && feature_value <= *max
                }
                _ => true,
            };
            if !satisfies {
                return false;
            }
        }
        true
    }
    /// Calculate anchor score (balance precision and coverage)
    fn calculate_anchor_score(&self, precision: f64, coverage: f64) -> f64 {
        0.8 * precision + 0.2 * coverage
    }
}
/// An anchor explanation consisting of predicates and metrics
#[derive(Debug, Clone)]
pub struct AnchorExplanation {
    /// Instance being explained
    pub instance: Array1<f64>,
    /// Original prediction
    pub prediction: f64,
    /// Anchor predicates
    pub anchor: Vec<AnchorPredicate>,
    /// Precision (how often anchor preserves prediction)
    pub precision: f64,
    /// Coverage (how much of input space anchor covers)
    pub coverage: f64,
    /// Number of samples used for evaluation
    pub n_samples_evaluated: usize,
    /// Feature names (if available)
    pub feature_names: Option<Vec<String>>,
}
impl AnchorExplanation {
    /// Generate a human-readable explanation
    pub fn explain(&self) -> String {
        let mut explanation = format!(
            "Anchor Explanation for prediction: {:.4}\n", self.prediction
        );
        explanation.push_str(&format!("Precision: {:.4}\n", self.precision));
        explanation.push_str(&format!("Coverage: {:.4}\n", self.coverage));
        explanation
            .push_str(&format!("Based on {} samples\n\n", self.n_samples_evaluated));
        if self.anchor.is_empty() {
            explanation.push_str("No reliable anchor found.\n");
        } else {
            explanation.push_str("IF ");
            for (i, predicate) in self.anchor.iter().enumerate() {
                if i > 0 {
                    explanation.push_str(" AND ");
                }
                explanation.push_str(&predicate.to_string(&self.feature_names));
            }
            explanation
                .push_str(
                    &format!(
                        "\nTHEN prediction = {:.4} (with {:.1}% confidence)\n", self
                        .prediction, self.precision * 100.0
                    ),
                );
        }
        explanation
    }
    /// Get anchor size (number of predicates)
    pub fn anchor_size(&self) -> usize {
        self.anchor.len()
    }
    /// Check if the anchor is sufficient (meets precision threshold)
    pub fn is_sufficient(&self, threshold: f64) -> bool {
        self.precision >= threshold
    }
}
/// LIME explanation for a single instance
#[derive(Debug, Clone)]
pub struct LimeExplanation {
    /// Instance being explained
    pub instance: Array1<f64>,
    /// Original model prediction
    pub prediction: f64,
    /// Feature importance scores (positive = supporting prediction)
    pub feature_importances: Array1<f64>,
    /// Feature names (if available)
    pub feature_names: Option<Vec<String>>,
    /// Local model score (R-squared)
    pub local_score: f64,
    /// Number of samples used in explanation
    pub n_samples_used: usize,
}
impl LimeExplanation {
    /// Get the top k most important features
    pub fn top_features(&self, k: usize) -> Vec<(usize, f64, Option<String>)> {
        let mut features: Vec<(usize, f64)> = self
            .feature_importances
            .iter()
            .enumerate()
            .map(|(i, &importance)| (i, importance))
            .collect();
        features
            .sort_by(|a, b| {
                b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal)
            });
        features
            .into_iter()
            .take(k)
            .map(|(idx, importance)| {
                let name = self
                    .feature_names
                    .as_ref()
                    .and_then(|names| names.get(idx))
                    .cloned();
                (idx, importance, name)
            })
            .collect()
    }
    /// Generate a human-readable explanation
    pub fn explain(&self, max_features: Option<usize>) -> String {
        let k = max_features.unwrap_or(5).min(self.feature_importances.len());
        let top_features = self.top_features(k);
        let mut explanation = format!(
            "LIME Explanation for prediction: {:.4}\n", self.prediction
        );
        explanation
            .push_str(&format!("Local model score (R²): {:.4}\n", self.local_score));
        explanation
            .push_str(
                &format!("Based on {} perturbed samples\n\n", self.n_samples_used),
            );
        explanation.push_str("Top feature contributions:\n");
        for (i, (feature_idx, importance, feature_name)) in top_features
            .iter()
            .enumerate()
        {
            let default_name = format!("feature_{}", feature_idx);
            let feature_str = feature_name
                .as_ref()
                .map(|name| name.as_str())
                .unwrap_or(&default_name);
            let direction = if *importance > 0.0 { "supports" } else { "opposes" };
            explanation
                .push_str(
                    &format!(
                        "  {}. {} ({}): {:.4}\n", i + 1, feature_str, direction,
                        importance.abs()
                    ),
                );
        }
        explanation
    }
}
/// Candidate anchor during search
#[derive(Debug, Clone)]
struct AnchorCandidate {
    predicates: Vec<AnchorPredicate>,
    precision: f64,
    coverage: f64,
    score: f64,
}
/// A decision rule extracted from a tree
#[derive(Debug, Clone)]
pub struct DecisionRule {
    /// Conditions that must be satisfied
    pub antecedent: Vec<DecisionCondition>,
    /// Predicted class or value
    pub consequent: f64,
    /// Support (fraction of samples that satisfy the antecedent)
    pub support: f64,
    /// Confidence (fraction of antecedent samples that have the consequent)
    pub confidence: f64,
    /// Lift (confidence / prior probability of consequent)
    pub lift: Option<f64>,
}
impl DecisionRule {
    /// Create a new decision rule
    pub fn new(
        antecedent: Vec<DecisionCondition>,
        consequent: f64,
        support: f64,
        confidence: f64,
        lift: Option<f64>,
    ) -> Self {
        Self {
            antecedent,
            consequent,
            support,
            confidence,
            lift,
        }
    }
    /// Get the complexity of this rule (number of conditions)
    pub fn complexity(&self) -> usize {
        self.antecedent.len()
    }
    /// Check if this rule is significant based on support and confidence thresholds
    pub fn is_significant(&self, min_support: f64, min_confidence: f64) -> bool {
        self.support >= min_support && self.confidence >= min_confidence
    }
}
/// Values for anchor predicates
#[derive(Debug, Clone, PartialEq)]
pub enum PredicateValue {
    /// Single threshold value
    Threshold(f64),
    /// Range [min, max]
    Range(f64, f64),
    /// Categorical value
    Category(String),
}
/// Simplified tree structure for path extraction
#[derive(Debug, Clone)]
pub struct TreeStructure {
    /// Root node of the tree
    pub root: TreeNode,
}
impl TreeStructure {
    /// Create a simple two-level tree for testing
    pub fn create_simple_tree() -> Self {
        let left_leaf = TreeNode::Leaf {
            prediction: 0.0,
            confidence: Some(0.95),
            n_samples: Some(45),
            impurity: Some(0.1),
            class_distribution: Some(HashMap::from([(0, 43), (1, 2)])),
        };
        let right_leaf = TreeNode::Leaf {
            prediction: 1.0,
            confidence: Some(0.90),
            n_samples: Some(55),
            impurity: Some(0.2),
            class_distribution: Some(HashMap::from([(0, 5), (1, 50)])),
        };
        let root = TreeNode::Internal {
            feature_idx: 0,
            threshold: 0.5,
            left: Box::new(left_leaf),
            right: Box::new(right_leaf),
            n_samples: Some(100),
            impurity: Some(0.5),
        };
        Self { root }
    }
}
/// Tree path extractor for generating decision paths and rules
pub struct TreePathExtractor {
    /// Feature names for readable output
    pub feature_names: Option<Vec<String>>,
    /// Minimum support for rule extraction
    pub min_support: f64,
    /// Minimum confidence for rule extraction
    pub min_confidence: f64,
    /// Maximum depth for path extraction
    pub max_depth: Option<usize>,
}
impl TreePathExtractor {
    /// Create a new tree path extractor
    pub fn new(
        feature_names: Option<Vec<String>>,
        min_support: f64,
        min_confidence: f64,
        max_depth: Option<usize>,
    ) -> Self {
        Self {
            feature_names,
            min_support,
            min_confidence,
            max_depth,
        }
    }
    /// Extract decision paths for a sample
    pub fn extract_sample_path(
        &self,
        sample: &Array1<f64>,
        tree_structure: &TreeStructure,
    ) -> Result<DecisionPath> {
        let mut conditions = Vec::new();
        let mut current_node = &tree_structure.root;
        loop {
            match current_node {
                TreeNode::Internal { feature_idx, threshold, left, right, .. } => {
                    let feature_value = sample[*feature_idx];
                    let feature_name = self
                        .feature_names
                        .as_ref()
                        .and_then(|names| names.get(*feature_idx))
                        .cloned();
                    if feature_value <= *threshold {
                        conditions
                            .push(DecisionCondition {
                                feature_idx: *feature_idx,
                                feature_name,
                                threshold: *threshold,
                                operator: ComparisonOperator::LessThanOrEqual,
                            });
                        current_node = left;
                    } else {
                        conditions
                            .push(DecisionCondition {
                                feature_idx: *feature_idx,
                                feature_name,
                                threshold: *threshold,
                                operator: ComparisonOperator::GreaterThan,
                            });
                        current_node = right;
                    }
                }
                TreeNode::Leaf {
                    prediction,
                    confidence,
                    n_samples,
                    impurity,
                    class_distribution,
                    ..
                } => {
                    return Ok(
                        DecisionPath::new(
                            conditions,
                            *prediction,
                            *confidence,
                            *n_samples,
                            *impurity,
                            class_distribution.clone(),
                        ),
                    );
                }
            }
            if let Some(max_depth) = self.max_depth {
                if conditions.len() >= max_depth {
                    break;
                }
            }
        }
        Err(
            SklearsError::InvalidInput(
                "Failed to reach leaf node in tree traversal".to_string(),
            ),
        )
    }
    /// Extract all decision paths from a tree
    pub fn extract_all_paths(
        &self,
        tree_structure: &TreeStructure,
    ) -> Result<Vec<DecisionPath>> {
        let mut paths = Vec::new();
        let mut conditions = Vec::new();
        self.extract_paths_recursive(&tree_structure.root, &mut conditions, &mut paths)?;
        Ok(paths)
    }
    /// Recursively extract paths from tree nodes
    fn extract_paths_recursive(
        &self,
        node: &TreeNode,
        current_conditions: &mut Vec<DecisionCondition>,
        paths: &mut Vec<DecisionPath>,
    ) -> Result<()> {
        match node {
            TreeNode::Internal { feature_idx, threshold, left, right, .. } => {
                let feature_name = self
                    .feature_names
                    .as_ref()
                    .and_then(|names| names.get(*feature_idx))
                    .cloned();
                current_conditions
                    .push(DecisionCondition {
                        feature_idx: *feature_idx,
                        feature_name: feature_name.clone(),
                        threshold: *threshold,
                        operator: ComparisonOperator::LessThanOrEqual,
                    });
                if self
                    .max_depth
                    .map_or(true, |max_depth| current_conditions.len() < max_depth)
                {
                    self.extract_paths_recursive(left, current_conditions, paths)?;
                }
                current_conditions.pop();
                current_conditions
                    .push(DecisionCondition {
                        feature_idx: *feature_idx,
                        feature_name,
                        threshold: *threshold,
                        operator: ComparisonOperator::GreaterThan,
                    });
                if self
                    .max_depth
                    .map_or(true, |max_depth| current_conditions.len() < max_depth)
                {
                    self.extract_paths_recursive(right, current_conditions, paths)?;
                }
                current_conditions.pop();
            }
            TreeNode::Leaf {
                prediction,
                confidence,
                n_samples,
                impurity,
                class_distribution,
                ..
            } => {
                paths
                    .push(
                        DecisionPath::new(
                            current_conditions.clone(),
                            *prediction,
                            *confidence,
                            *n_samples,
                            *impurity,
                            class_distribution.clone(),
                        ),
                    );
            }
        }
        Ok(())
    }
    /// Extract decision rules from paths
    pub fn extract_rules_from_paths(
        &self,
        paths: &[DecisionPath],
        total_samples: usize,
    ) -> Vec<DecisionRule> {
        let mut rules = Vec::new();
        for path in paths {
            let n_samples = path.n_samples.unwrap_or(1);
            let support = n_samples as f64 / total_samples as f64;
            if support < self.min_support {
                continue;
            }
            let confidence = path.confidence.unwrap_or(1.0);
            if confidence < self.min_confidence {
                continue;
            }
            rules
                .push(
                    DecisionRule::new(
                        path.conditions.clone(),
                        path.prediction,
                        support,
                        confidence,
                        None,
                    ),
                );
        }
        rules
    }
    /// Generate human-readable summary of tree structure
    pub fn generate_tree_summary(&self, paths: &[DecisionPath]) -> String {
        let mut summary = String::new();
        summary.push_str(&format!("Tree Summary:\n"));
        summary.push_str(&format!("- Total paths: {}\n", paths.len()));
        let depths: Vec<usize> = paths.iter().map(|p| p.depth()).collect();
        let avg_depth = depths.iter().sum::<usize>() as f64 / depths.len() as f64;
        let max_depth = depths.iter().max().unwrap_or(&0);
        let min_depth = depths.iter().min().unwrap_or(&0);
        summary.push_str(&format!("- Average depth: {:.2}\n", avg_depth));
        summary.push_str(&format!("- Max depth: {}\n", max_depth));
        summary.push_str(&format!("- Min depth: {}\n", min_depth));
        let mut feature_usage: HashMap<usize, usize> = HashMap::new();
        for path in paths {
            for condition in &path.conditions {
                *feature_usage.entry(condition.feature_idx).or_insert(0) += 1;
            }
        }
        if !feature_usage.is_empty() {
            summary.push_str("\nFeature Usage:\n");
            let mut usage_vec: Vec<(usize, usize)> = feature_usage.into_iter().collect();
            usage_vec.sort_by(|a, b| b.1.cmp(&a.1));
            for (feature_idx, count) in usage_vec.iter().take(10) {
                let default_name = format!("feature_{}", feature_idx);
                let feature_name = self
                    .feature_names
                    .as_ref()
                    .and_then(|names| names.get(*feature_idx))
                    .map(|s| s.as_str())
                    .unwrap_or(&default_name);
                summary.push_str(&format!("- {}: {} paths\n", feature_name, count));
            }
        }
        summary
    }
}
/// Distance metrics for LIME sample weighting
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Manhattan distance
    Manhattan,
    /// Cosine distance
    Cosine,
}
/// A single condition in a decision path
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionCondition {
    /// Feature index
    pub feature_idx: usize,
    /// Feature name (if available)
    pub feature_name: Option<String>,
    /// Threshold value
    pub threshold: f64,
    /// Comparison operator
    pub operator: ComparisonOperator,
}
