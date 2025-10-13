//! Active Learning Methods for Semi-Supervised Learning
//!
//! This module provides active learning algorithms that can be integrated with
//! semi-supervised learning methods to intelligently select the most informative
//! samples for labeling. These methods help optimize the labeling budget by
//! focusing on the most uncertain or diverse samples.

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::error::{Result as SklResult, SklearsError};

/// Uncertainty Sampling for Active Learning
///
/// Uncertainty sampling selects samples for labeling based on the model's
/// uncertainty about their predictions. It supports multiple uncertainty
/// measures including entropy, margin, and least confident sampling.
///
/// # Parameters
///
/// * `strategy` - Uncertainty sampling strategy ("entropy", "margin", "least_confident")
/// * `n_samples` - Number of samples to select for labeling
/// * `temperature` - Temperature scaling for probability calibration
/// * `diversity_weight` - Weight for diversity-based selection
/// * `batch_size` - Size of batches for efficient computation
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_semi_supervised::UncertaintySampling;
///
///
/// let probas = array![
///     [0.9, 0.1],
///     [0.6, 0.4],
///     [0.5, 0.5],
///     [0.8, 0.2]
/// ];
///
/// let us = UncertaintySampling::new()
///     .strategy("entropy".to_string())
///     .n_samples(2);
/// let selected_indices = us.select_samples(&probas.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct UncertaintySampling {
    strategy: String,
    n_samples: usize,
    temperature: f64,
    diversity_weight: f64,
    batch_size: usize,
    random_tie_breaking: bool,
}

impl UncertaintySampling {
    /// Create a new UncertaintySampling instance
    pub fn new() -> Self {
        Self {
            strategy: "entropy".to_string(),
            n_samples: 10,
            temperature: 1.0,
            diversity_weight: 0.0,
            batch_size: 1000,
            random_tie_breaking: true,
        }
    }

    /// Set the uncertainty sampling strategy
    pub fn strategy(mut self, strategy: String) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the number of samples to select
    pub fn n_samples(mut self, n_samples: usize) -> Self {
        self.n_samples = n_samples;
        self
    }

    /// Set the temperature for probability calibration
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set the diversity weight
    pub fn diversity_weight(mut self, weight: f64) -> Self {
        self.diversity_weight = weight;
        self
    }

    /// Set the batch size for computation
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Enable/disable random tie breaking
    pub fn random_tie_breaking(mut self, random: bool) -> Self {
        self.random_tie_breaking = random;
        self
    }

    /// Select samples based on uncertainty
    pub fn select_samples(&self, probas: &ArrayView2<f64>) -> SklResult<Vec<usize>> {
        let n_samples = probas.nrows();

        // Validate strategy first
        match self.strategy.as_str() {
            "entropy" | "margin" | "least_confident" => {}
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown uncertainty strategy: {}",
                    self.strategy
                )))
            }
        }

        if self.n_samples >= n_samples {
            return Ok((0..n_samples).collect());
        }

        // Apply temperature scaling
        let calibrated_probas = self.apply_temperature_scaling(probas);

        // Compute uncertainty scores
        let uncertainty_scores = match self.strategy.as_str() {
            "entropy" => self.entropy_uncertainty(&calibrated_probas),
            "margin" => self.margin_uncertainty(&calibrated_probas),
            "least_confident" => self.least_confident_uncertainty(&calibrated_probas),
            _ => unreachable!(), // Already validated above
        }?;

        // Select top uncertain samples
        let mut indexed_scores: Vec<(usize, f64)> = uncertainty_scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();

        // Sort by uncertainty (descending)
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Apply diversity if weight > 0
        let selected_indices = if self.diversity_weight > 0.0 {
            self.diverse_selection(&indexed_scores, probas)?
        } else {
            indexed_scores
                .iter()
                .take(self.n_samples)
                .map(|(idx, _)| *idx)
                .collect()
        };

        Ok(selected_indices)
    }

    fn apply_temperature_scaling(&self, probas: &ArrayView2<f64>) -> Array2<f64> {
        if (self.temperature - 1.0).abs() < 1e-10 {
            return probas.to_owned();
        }

        let mut calibrated = Array2::zeros(probas.dim());
        for (i, row) in probas.axis_iter(Axis(0)).enumerate() {
            let scaled: Array1<f64> = row.mapv(|p| (p.ln() / self.temperature).exp());
            let sum = scaled.sum();
            if sum > 0.0 {
                calibrated.row_mut(i).assign(&(scaled / sum));
            } else {
                calibrated.row_mut(i).assign(&row);
            }
        }
        calibrated
    }

    fn entropy_uncertainty(&self, probas: &Array2<f64>) -> SklResult<Array1<f64>> {
        let mut entropies = Array1::zeros(probas.nrows());

        for (i, row) in probas.axis_iter(Axis(0)).enumerate() {
            let mut entropy = 0.0;
            for &p in row.iter() {
                if p > 1e-15 {
                    entropy -= p * p.ln();
                }
            }
            entropies[i] = entropy;
        }

        Ok(entropies)
    }

    fn margin_uncertainty(&self, probas: &Array2<f64>) -> SklResult<Array1<f64>> {
        let mut margins = Array1::zeros(probas.nrows());

        for (i, row) in probas.axis_iter(Axis(0)).enumerate() {
            let mut sorted_probs: Vec<f64> = row.iter().cloned().collect();
            sorted_probs.sort_by(|a, b| b.partial_cmp(a).unwrap());

            if sorted_probs.len() >= 2 {
                margins[i] = -(sorted_probs[0] - sorted_probs[1]); // Negative for ascending order
            } else {
                margins[i] = -sorted_probs[0];
            }
        }

        Ok(margins)
    }

    fn least_confident_uncertainty(&self, probas: &Array2<f64>) -> SklResult<Array1<f64>> {
        let mut uncertainties = Array1::zeros(probas.nrows());

        for (i, row) in probas.axis_iter(Axis(0)).enumerate() {
            let max_prob = row.iter().fold(0.0f64, |a, &b| a.max(b));
            uncertainties[i] = 1.0 - max_prob;
        }

        Ok(uncertainties)
    }

    fn diverse_selection(
        &self,
        indexed_scores: &[(usize, f64)],
        probas: &ArrayView2<f64>,
    ) -> SklResult<Vec<usize>> {
        let mut selected = Vec::new();
        let mut remaining: Vec<(usize, f64)> = indexed_scores.to_vec();

        // Select first sample (most uncertain)
        if let Some((first_idx, _)) = remaining.first().cloned() {
            selected.push(first_idx);
            remaining.retain(|(idx, _)| *idx != first_idx);
        }

        // Select remaining samples considering diversity
        while selected.len() < self.n_samples && !remaining.is_empty() {
            let mut best_idx = 0;
            let mut best_score = f64::NEG_INFINITY;

            for (candidate_idx, (sample_idx, uncertainty)) in remaining.iter().enumerate() {
                // Compute diversity score (distance from already selected samples)
                let mut min_distance = f64::INFINITY;
                for &selected_idx in &selected {
                    let distance =
                        self.compute_distance(probas.row(*sample_idx), probas.row(selected_idx));
                    min_distance = min_distance.min(distance);
                }

                // Combined score: uncertainty + diversity
                let combined_score = (1.0 - self.diversity_weight) * uncertainty
                    + self.diversity_weight * min_distance;

                if combined_score > best_score {
                    best_score = combined_score;
                    best_idx = candidate_idx;
                }
            }

            let (selected_sample_idx, _) = remaining.remove(best_idx);
            selected.push(selected_sample_idx);
        }

        Ok(selected)
    }

    fn compute_distance(&self, prob1: ArrayView1<f64>, prob2: ArrayView1<f64>) -> f64 {
        // Jensen-Shannon divergence
        let m = (&prob1 + &prob2) / 2.0;
        let kl1 = self.kl_divergence(&prob1, &m.view());
        let kl2 = self.kl_divergence(&prob2, &m.view());
        (kl1 + kl2) / 2.0
    }

    fn kl_divergence(&self, p: &ArrayView1<f64>, q: &ArrayView1<f64>) -> f64 {
        let mut kl = 0.0;
        for (pi, qi) in p.iter().zip(q.iter()) {
            if *pi > 1e-15 && *qi > 1e-15 {
                kl += pi * (pi / qi).ln();
            }
        }
        kl
    }
}

/// Query by Committee for Active Learning
///
/// Query by Committee selects samples where multiple models disagree the most.
/// It maintains an ensemble of models and selects samples with the highest
/// disagreement among committee members.
///
/// # Parameters
///
/// * `n_committee_members` - Number of models in the committee
/// * `disagreement_measure` - Measure of disagreement ("vote_entropy", "kl_divergence")
/// * `n_samples` - Number of samples to select
/// * `diversity_weight` - Weight for diversity in selection
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_semi_supervised::QueryByCommittee;
///
///
/// let committee_probas = vec![
///     array![[0.8, 0.2], [0.6, 0.4], [0.3, 0.7]],
///     array![[0.7, 0.3], [0.5, 0.5], [0.4, 0.6]],
///     array![[0.9, 0.1], [0.4, 0.6], [0.2, 0.8]],
/// ];
///
/// let qbc = QueryByCommittee::new()
///     .n_committee_members(3)
///     .disagreement_measure("vote_entropy".to_string())
///     .n_samples(2);
/// let selected = qbc.select_samples(&committee_probas).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct QueryByCommittee {
    n_committee_members: usize,
    disagreement_measure: String,
    n_samples: usize,
    diversity_weight: f64,
    normalize_disagreement: bool,
}

impl QueryByCommittee {
    /// Create a new QueryByCommittee instance
    pub fn new() -> Self {
        Self {
            n_committee_members: 3,
            disagreement_measure: "vote_entropy".to_string(),
            n_samples: 10,
            diversity_weight: 0.0,
            normalize_disagreement: true,
        }
    }

    /// Set the number of committee members
    pub fn n_committee_members(mut self, n_members: usize) -> Self {
        self.n_committee_members = n_members;
        self
    }

    /// Set the disagreement measure
    pub fn disagreement_measure(mut self, measure: String) -> Self {
        self.disagreement_measure = measure;
        self
    }

    /// Set the number of samples to select
    pub fn n_samples(mut self, n_samples: usize) -> Self {
        self.n_samples = n_samples;
        self
    }

    /// Set the diversity weight
    pub fn diversity_weight(mut self, weight: f64) -> Self {
        self.diversity_weight = weight;
        self
    }

    /// Enable/disable disagreement normalization
    pub fn normalize_disagreement(mut self, normalize: bool) -> Self {
        self.normalize_disagreement = normalize;
        self
    }

    /// Select samples based on committee disagreement
    pub fn select_samples(&self, committee_probas: &[Array2<f64>]) -> SklResult<Vec<usize>> {
        if committee_probas.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Empty committee provided".to_string(),
            ));
        }

        let n_samples = committee_probas[0].nrows();
        let n_classes = committee_probas[0].ncols();

        // Validate committee dimensions
        for (i, probas) in committee_probas.iter().enumerate() {
            if probas.dim() != (n_samples, n_classes) {
                return Err(SklearsError::InvalidInput(format!(
                    "Committee member {} has incompatible dimensions",
                    i
                )));
            }
        }

        if self.n_samples >= n_samples {
            return Ok((0..n_samples).collect());
        }

        // Compute disagreement scores
        let disagreement_scores = match self.disagreement_measure.as_str() {
            "vote_entropy" => self.vote_entropy_disagreement(committee_probas)?,
            "kl_divergence" => self.kl_divergence_disagreement(committee_probas)?,
            "variance" => self.variance_disagreement(committee_probas)?,
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown disagreement measure: {}",
                    self.disagreement_measure
                )))
            }
        };

        // Normalize disagreement scores if requested
        let normalized_scores = if self.normalize_disagreement {
            self.normalize_scores(&disagreement_scores)
        } else {
            disagreement_scores
        };

        // Select samples with highest disagreement
        let mut indexed_scores: Vec<(usize, f64)> = normalized_scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();

        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let selected_indices: Vec<usize> = indexed_scores
            .iter()
            .take(self.n_samples)
            .map(|(idx, _)| *idx)
            .collect();

        Ok(selected_indices)
    }

    fn vote_entropy_disagreement(
        &self,
        committee_probas: &[Array2<f64>],
    ) -> SklResult<Array1<f64>> {
        let n_samples = committee_probas[0].nrows();
        let n_classes = committee_probas[0].ncols();
        let mut disagreements = Array1::zeros(n_samples);

        for sample_idx in 0..n_samples {
            // Get predictions from all committee members
            let mut class_votes = Array1::zeros(n_classes);

            for committee_probas in committee_probas.iter() {
                let sample_probas = committee_probas.row(sample_idx);
                let predicted_class = sample_probas
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
                class_votes[predicted_class] += 1.0;
            }

            // Normalize vote counts to probabilities
            let total_votes: f64 = class_votes.sum();
            if total_votes > 0.0 {
                class_votes /= total_votes;
            }

            // Compute entropy of vote distribution
            let mut entropy = 0.0;
            for &vote_prob in class_votes.iter() {
                if vote_prob > 1e-15 {
                    entropy -= vote_prob * vote_prob.ln();
                }
            }
            disagreements[sample_idx] = entropy;
        }

        Ok(disagreements)
    }

    fn kl_divergence_disagreement(
        &self,
        committee_probas: &[Array2<f64>],
    ) -> SklResult<Array1<f64>> {
        let n_samples = committee_probas[0].nrows();
        let mut disagreements = Array1::zeros(n_samples);

        for sample_idx in 0..n_samples {
            let mut total_disagreement = 0.0;
            let mut pair_count = 0;

            // Compute pairwise KL divergences
            for i in 0..committee_probas.len() {
                for j in (i + 1)..committee_probas.len() {
                    let p1 = committee_probas[i].row(sample_idx);
                    let p2 = committee_probas[j].row(sample_idx);

                    let kl_div = self.symmetric_kl_divergence(&p1, &p2);
                    total_disagreement += kl_div;
                    pair_count += 1;
                }
            }

            if pair_count > 0 {
                disagreements[sample_idx] = total_disagreement / pair_count as f64;
            }
        }

        Ok(disagreements)
    }

    fn variance_disagreement(&self, committee_probas: &[Array2<f64>]) -> SklResult<Array1<f64>> {
        let n_samples = committee_probas[0].nrows();
        let n_classes = committee_probas[0].ncols();
        let mut disagreements = Array1::zeros(n_samples);

        for sample_idx in 0..n_samples {
            let mut total_variance = 0.0;

            for class_idx in 0..n_classes {
                // Collect probabilities for this class from all committee members
                let class_probs: Vec<f64> = committee_probas
                    .iter()
                    .map(|probas| probas[[sample_idx, class_idx]])
                    .collect();

                // Compute variance
                let mean = class_probs.iter().sum::<f64>() / class_probs.len() as f64;
                let variance = class_probs.iter().map(|&p| (p - mean).powi(2)).sum::<f64>()
                    / class_probs.len() as f64;

                total_variance += variance;
            }

            disagreements[sample_idx] = total_variance;
        }

        Ok(disagreements)
    }

    fn symmetric_kl_divergence(&self, p1: &ArrayView1<f64>, p2: &ArrayView1<f64>) -> f64 {
        let kl1 = self.kl_divergence(p1, p2);
        let kl2 = self.kl_divergence(p2, p1);
        (kl1 + kl2) / 2.0
    }

    fn kl_divergence(&self, p: &ArrayView1<f64>, q: &ArrayView1<f64>) -> f64 {
        let mut kl = 0.0;
        for (pi, qi) in p.iter().zip(q.iter()) {
            if *pi > 1e-15 && *qi > 1e-15 {
                kl += pi * (pi / qi).ln();
            }
        }
        kl
    }

    fn normalize_scores(&self, scores: &Array1<f64>) -> Array1<f64> {
        let min_score = scores.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_score = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if (max_score - min_score).abs() < 1e-15 {
            Array1::from_elem(scores.len(), 0.5)
        } else {
            scores.mapv(|x| (x - min_score) / (max_score - min_score))
        }
    }
}

impl Default for UncertaintySampling {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for QueryByCommittee {
    fn default() -> Self {
        Self::new()
    }
}

/// Expected Model Change for Active Learning
///
/// Expected Model Change selects samples that are expected to cause the largest
/// change in the model parameters when added to the training set. This strategy
/// looks ahead to predict which samples would be most informative for model updating.
///
/// # Parameters
///
/// * `n_samples` - Number of samples to select for labeling
/// * `approximation_method` - Method for approximating model change ("gradient_norm", "fisher_information", "parameter_variance")
/// * `learning_rate` - Learning rate for gradient approximation
/// * `epsilon` - Small value for numerical stability
/// * `normalize_scores` - Whether to normalize change scores
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_semi_supervised::ExpectedModelChange;
///
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
/// let gradients = array![[0.1, 0.2], [0.3, 0.1], [0.05, 0.4]];
///
/// let emc = ExpectedModelChange::new()
///     .approximation_method("gradient_norm".to_string())
///     .n_samples(2);
/// let selected = emc.select_samples(&X.view(), &gradients.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ExpectedModelChange {
    n_samples: usize,
    approximation_method: String,
    learning_rate: f64,
    epsilon: f64,
    normalize_scores: bool,
    diversity_weight: f64,
    batch_size: usize,
}

impl ExpectedModelChange {
    /// Create a new ExpectedModelChange instance
    pub fn new() -> Self {
        Self {
            n_samples: 10,
            approximation_method: "gradient_norm".to_string(),
            learning_rate: 0.01,
            epsilon: 1e-8,
            normalize_scores: true,
            diversity_weight: 0.0,
            batch_size: 1000,
        }
    }

    /// Set the number of samples to select
    pub fn n_samples(mut self, n_samples: usize) -> Self {
        self.n_samples = n_samples;
        self
    }

    /// Set the approximation method for model change
    pub fn approximation_method(mut self, method: String) -> Self {
        self.approximation_method = method;
        self
    }

    /// Set the learning rate for gradient approximation
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set epsilon for numerical stability
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set whether to normalize scores
    pub fn normalize_scores(mut self, normalize: bool) -> Self {
        self.normalize_scores = normalize;
        self
    }

    /// Set diversity weight for selection
    pub fn diversity_weight(mut self, weight: f64) -> Self {
        self.diversity_weight = weight;
        self
    }

    /// Set batch size for computation
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Select samples based on expected model change
    pub fn select_samples(
        &self,
        X: &ArrayView2<f64>,
        gradients: &ArrayView2<f64>,
    ) -> SklResult<Vec<usize>> {
        let n_samples = X.nrows();

        if gradients.nrows() != n_samples {
            return Err(SklearsError::InvalidInput(
                "Number of gradients must match number of samples".to_string(),
            ));
        }

        if self.n_samples >= n_samples {
            return Ok((0..n_samples).collect());
        }

        // Validate approximation method
        match self.approximation_method.as_str() {
            "gradient_norm" | "fisher_information" | "parameter_variance" => {}
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown approximation method: {}",
                    self.approximation_method
                )))
            }
        }

        // Compute expected model change scores
        let change_scores = match self.approximation_method.as_str() {
            "gradient_norm" => self.gradient_norm_scores(gradients)?,
            "fisher_information" => self.fisher_information_scores(X, gradients)?,
            "parameter_variance" => self.parameter_variance_scores(gradients)?,
            _ => unreachable!(),
        };

        // Normalize scores if requested
        let final_scores = if self.normalize_scores {
            self.normalize_change_scores(&change_scores)
        } else {
            change_scores
        };

        // Select samples with highest expected change
        let mut indexed_scores: Vec<(usize, f64)> = final_scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();

        // Sort by expected change (descending)
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Apply diversity if weight > 0
        let selected_indices = if self.diversity_weight > 0.0 {
            self.diverse_model_change_selection(&indexed_scores, X)?
        } else {
            indexed_scores
                .into_iter()
                .take(self.n_samples)
                .map(|(idx, _)| idx)
                .collect()
        };

        Ok(selected_indices)
    }

    fn gradient_norm_scores(&self, gradients: &ArrayView2<f64>) -> SklResult<Array1<f64>> {
        let n_samples = gradients.nrows();
        let mut scores = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let gradient = gradients.row(i);
            // L2 norm of gradient (expected parameter change)
            let norm = gradient.iter().map(|&x| x * x).sum::<f64>().sqrt();
            scores[i] = norm * self.learning_rate;
        }

        Ok(scores)
    }

    fn fisher_information_scores(
        &self,
        X: &ArrayView2<f64>,
        gradients: &ArrayView2<f64>,
    ) -> SklResult<Array1<f64>> {
        let n_samples = X.nrows();
        let mut scores = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let gradient = gradients.row(i);
            let features = X.row(i);

            // Approximate Fisher Information as outer product of gradients
            // weighted by feature magnitudes
            let feature_weight = features.iter().map(|&x| x * x).sum::<f64>().sqrt() + self.epsilon;
            let gradient_magnitude = gradient.iter().map(|&x| x * x).sum::<f64>();

            scores[i] = gradient_magnitude * feature_weight * self.learning_rate;
        }

        Ok(scores)
    }

    fn parameter_variance_scores(&self, gradients: &ArrayView2<f64>) -> SklResult<Array1<f64>> {
        let n_samples = gradients.nrows();
        let n_features = gradients.ncols();
        let mut scores = Array1::zeros(n_samples);

        // Compute mean gradient across samples
        let mean_gradient = gradients.mean_axis(Axis(0)).unwrap();

        for i in 0..n_samples {
            let gradient = gradients.row(i);

            // Compute variance from mean gradient
            let mut variance = 0.0;
            for j in 0..n_features {
                let diff = gradient[j] - mean_gradient[j];
                variance += diff * diff;
            }

            variance /= n_features as f64;
            scores[i] = variance * self.learning_rate;
        }

        Ok(scores)
    }

    fn normalize_change_scores(&self, scores: &Array1<f64>) -> Array1<f64> {
        let min_score = scores.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_score = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if (max_score - min_score).abs() < self.epsilon {
            Array1::from_elem(scores.len(), 0.5)
        } else {
            scores.mapv(|x| (x - min_score) / (max_score - min_score))
        }
    }

    fn diverse_model_change_selection(
        &self,
        indexed_scores: &[(usize, f64)],
        X: &ArrayView2<f64>,
    ) -> SklResult<Vec<usize>> {
        let mut selected = Vec::new();
        let mut remaining: Vec<(usize, f64)> = indexed_scores.to_vec();

        // Select first sample with highest change score
        if let Some((first_idx, _)) = remaining.first() {
            selected.push(*first_idx);
            remaining.remove(0);
        }

        // Select remaining samples balancing change and diversity
        while selected.len() < self.n_samples && !remaining.is_empty() {
            let mut best_score = f64::NEG_INFINITY;
            let mut best_idx = 0;

            for (candidate_idx, (sample_idx, change_score)) in remaining.iter().enumerate() {
                // Compute minimum distance to already selected samples
                let mut min_distance = f64::INFINITY;
                for &selected_idx in &selected {
                    let dist = self.euclidean_distance(X.row(*sample_idx), X.row(selected_idx));
                    min_distance = min_distance.min(dist);
                }

                // Combined score: model change + diversity
                let combined_score = (1.0 - self.diversity_weight) * change_score
                    + self.diversity_weight * min_distance;

                if combined_score > best_score {
                    best_score = combined_score;
                    best_idx = candidate_idx;
                }
            }

            let (selected_sample_idx, _) = remaining.remove(best_idx);
            selected.push(selected_sample_idx);
        }

        Ok(selected)
    }

    fn euclidean_distance(&self, x1: ArrayView1<f64>, x2: ArrayView1<f64>) -> f64 {
        x1.iter()
            .zip(x2.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

impl Default for ExpectedModelChange {
    fn default() -> Self {
        Self::new()
    }
}

/// Information Density for Active Learning
///
/// Information Density methods combine uncertainty measures with density-based
/// sample selection. These methods prefer samples that are both uncertain and
/// located in dense regions of the feature space, as such samples are more
/// representative and likely to improve model performance.
///
/// # Parameters
///
/// * `uncertainty_measure` - Uncertainty measure to use ("entropy", "margin", "least_confident")
/// * `density_measure` - Density measure to use ("knn_density", "gaussian_density", "cosine_similarity")
/// * `n_samples` - Number of samples to select for labeling
/// * `density_weight` - Weight for density component (0.0 = pure uncertainty, 1.0 = pure density)
/// * `bandwidth` - Bandwidth parameter for density estimation
/// * `k_neighbors` - Number of neighbors for k-NN density estimation
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_semi_supervised::InformationDensity;
///
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
/// let probas = array![[0.9, 0.1], [0.6, 0.4], [0.5, 0.5], [0.8, 0.2]];
///
/// let id = InformationDensity::new()
///     .uncertainty_measure("entropy".to_string())
///     .density_measure("knn_density".to_string())
///     .density_weight(0.5)
///     .n_samples(2);
/// let selected = id.select_samples(&X.view(), &probas.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct InformationDensity {
    uncertainty_measure: String,
    density_measure: String,
    n_samples: usize,
    density_weight: f64,
    bandwidth: f64,
    k_neighbors: usize,
    temperature: f64,
    normalize_scores: bool,
}

impl InformationDensity {
    /// Create a new InformationDensity instance
    pub fn new() -> Self {
        Self {
            uncertainty_measure: "entropy".to_string(),
            density_measure: "knn_density".to_string(),
            n_samples: 10,
            density_weight: 0.5,
            bandwidth: 1.0,
            k_neighbors: 5,
            temperature: 1.0,
            normalize_scores: true,
        }
    }

    /// Set the uncertainty measure
    pub fn uncertainty_measure(mut self, measure: String) -> Self {
        self.uncertainty_measure = measure;
        self
    }

    /// Set the density measure
    pub fn density_measure(mut self, measure: String) -> Self {
        self.density_measure = measure;
        self
    }

    /// Set the number of samples to select
    pub fn n_samples(mut self, n_samples: usize) -> Self {
        self.n_samples = n_samples;
        self
    }

    /// Set the density weight
    pub fn density_weight(mut self, weight: f64) -> Self {
        self.density_weight = weight;
        self
    }

    /// Set the bandwidth for density estimation
    pub fn bandwidth(mut self, bandwidth: f64) -> Self {
        self.bandwidth = bandwidth;
        self
    }

    /// Set the number of neighbors for k-NN density
    pub fn k_neighbors(mut self, k: usize) -> Self {
        self.k_neighbors = k;
        self
    }

    /// Set temperature for probability calibration
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set whether to normalize scores
    pub fn normalize_scores(mut self, normalize: bool) -> Self {
        self.normalize_scores = normalize;
        self
    }

    /// Select samples based on information density
    pub fn select_samples(
        &self,
        X: &ArrayView2<f64>,
        probas: &ArrayView2<f64>,
    ) -> SklResult<Vec<usize>> {
        let n_samples = X.nrows();

        if probas.nrows() != n_samples {
            return Err(SklearsError::InvalidInput(
                "Number of probabilities must match number of samples".to_string(),
            ));
        }

        if self.n_samples >= n_samples {
            return Ok((0..n_samples).collect());
        }

        // Validate measures
        match self.uncertainty_measure.as_str() {
            "entropy" | "margin" | "least_confident" => {}
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown uncertainty measure: {}",
                    self.uncertainty_measure
                )))
            }
        }

        match self.density_measure.as_str() {
            "knn_density" | "gaussian_density" | "cosine_similarity" => {}
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown density measure: {}",
                    self.density_measure
                )))
            }
        }

        // Apply temperature scaling to probabilities
        let calibrated_probas = self.apply_temperature_scaling(probas);

        // Compute uncertainty scores
        let uncertainty_scores = match self.uncertainty_measure.as_str() {
            "entropy" => self.entropy_uncertainty(&calibrated_probas)?,
            "margin" => self.margin_uncertainty(&calibrated_probas)?,
            "least_confident" => self.least_confident_uncertainty(&calibrated_probas)?,
            _ => unreachable!(),
        };

        // Compute density scores
        let density_scores = match self.density_measure.as_str() {
            "knn_density" => self.knn_density_scores(X)?,
            "gaussian_density" => self.gaussian_density_scores(X)?,
            "cosine_similarity" => self.cosine_similarity_scores(X)?,
            _ => unreachable!(),
        };

        // Normalize scores if requested
        let normalized_uncertainty = if self.normalize_scores {
            self.normalize_array(&uncertainty_scores)
        } else {
            uncertainty_scores
        };

        let normalized_density = if self.normalize_scores {
            self.normalize_array(&density_scores)
        } else {
            density_scores
        };

        // Combine uncertainty and density scores
        let mut combined_scores = Array1::zeros(n_samples);
        for i in 0..n_samples {
            combined_scores[i] = (1.0 - self.density_weight) * normalized_uncertainty[i]
                + self.density_weight * normalized_density[i];
        }

        // Select samples with highest combined scores
        let mut indexed_scores: Vec<(usize, f64)> = combined_scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();

        // Sort by combined score (descending)
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let selected_indices: Vec<usize> = indexed_scores
            .into_iter()
            .take(self.n_samples)
            .map(|(idx, _)| idx)
            .collect();

        Ok(selected_indices)
    }

    fn apply_temperature_scaling(&self, probas: &ArrayView2<f64>) -> Array2<f64> {
        if (self.temperature - 1.0).abs() < 1e-10 {
            return probas.to_owned();
        }

        let mut calibrated = Array2::zeros(probas.dim());
        for i in 0..probas.nrows() {
            let row = probas.row(i);
            // Apply temperature scaling: p_i = exp(logit_i / T) / sum(exp(logit_j / T))
            let max_prob = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let logits: Vec<f64> = row.iter().map(|&p| (p / max_prob).ln()).collect();

            let scaled_logits: Vec<f64> = logits.iter().map(|&l| l / self.temperature).collect();
            let max_logit = scaled_logits
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            let exp_sum: f64 = scaled_logits.iter().map(|&l| (l - max_logit).exp()).sum();

            for j in 0..probas.ncols() {
                calibrated[[i, j]] = (scaled_logits[j] - max_logit).exp() / exp_sum;
            }
        }

        calibrated
    }

    fn entropy_uncertainty(&self, probas: &Array2<f64>) -> SklResult<Array1<f64>> {
        let n_samples = probas.nrows();
        let mut entropies = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut entropy = 0.0;
            for &p in probas.row(i).iter() {
                if p > 1e-15 {
                    entropy -= p * p.ln();
                }
            }
            entropies[i] = entropy;
        }

        Ok(entropies)
    }

    fn margin_uncertainty(&self, probas: &Array2<f64>) -> SklResult<Array1<f64>> {
        let n_samples = probas.nrows();
        let mut margins = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let row = probas.row(i);
            let mut sorted_probs: Vec<f64> = row.iter().cloned().collect();
            sorted_probs.sort_by(|a, b| b.partial_cmp(a).unwrap());

            let margin = if sorted_probs.len() >= 2 {
                sorted_probs[0] - sorted_probs[1] // largest - second largest
            } else {
                sorted_probs[0]
            };
            margins[i] = -margin; // Negative so higher uncertainty = higher score
        }

        Ok(margins)
    }

    fn least_confident_uncertainty(&self, probas: &Array2<f64>) -> SklResult<Array1<f64>> {
        let n_samples = probas.nrows();
        let mut uncertainties = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let max_prob = probas
                .row(i)
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            uncertainties[i] = 1.0 - max_prob;
        }

        Ok(uncertainties)
    }

    fn knn_density_scores(&self, X: &ArrayView2<f64>) -> SklResult<Array1<f64>> {
        let n_samples = X.nrows();
        let mut density_scores = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let sample_i = X.row(i);
            let mut distances = Vec::new();

            // Compute distances to all other samples
            for j in 0..n_samples {
                if i != j {
                    let sample_j = X.row(j);
                    let distance = self.euclidean_distance(sample_i, sample_j);
                    distances.push(distance);
                }
            }

            // Sort distances and take k-th nearest neighbor distance
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let k_distance = if self.k_neighbors <= distances.len() {
                distances[self.k_neighbors - 1]
            } else {
                distances.last().cloned().unwrap_or(1.0)
            };

            // Higher density = smaller distance to k-th neighbor
            density_scores[i] = 1.0 / (k_distance + 1e-8);
        }

        Ok(density_scores)
    }

    fn gaussian_density_scores(&self, X: &ArrayView2<f64>) -> SklResult<Array1<f64>> {
        let n_samples = X.nrows();
        let mut density_scores = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let sample_i = X.row(i);
            let mut density = 0.0;

            // Compute Gaussian kernel density
            for j in 0..n_samples {
                if i != j {
                    let sample_j = X.row(j);
                    let distance_sq = self.euclidean_distance_squared(sample_i, sample_j);
                    density += (-distance_sq / (2.0 * self.bandwidth * self.bandwidth)).exp();
                }
            }

            density_scores[i] = density / (n_samples - 1) as f64;
        }

        Ok(density_scores)
    }

    fn cosine_similarity_scores(&self, X: &ArrayView2<f64>) -> SklResult<Array1<f64>> {
        let n_samples = X.nrows();
        let mut similarity_scores = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let sample_i = X.row(i);
            let mut total_similarity = 0.0;

            for j in 0..n_samples {
                if i != j {
                    let sample_j = X.row(j);
                    let similarity = self.cosine_similarity(sample_i, sample_j);
                    total_similarity += similarity;
                }
            }

            similarity_scores[i] = total_similarity / (n_samples - 1) as f64;
        }

        Ok(similarity_scores)
    }

    fn normalize_array(&self, array: &Array1<f64>) -> Array1<f64> {
        let min_val = array.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = array.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if (max_val - min_val).abs() < 1e-15 {
            Array1::from_elem(array.len(), 0.5)
        } else {
            array.mapv(|x| (x - min_val) / (max_val - min_val))
        }
    }

    fn euclidean_distance(&self, x1: ArrayView1<f64>, x2: ArrayView1<f64>) -> f64 {
        x1.iter()
            .zip(x2.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn euclidean_distance_squared(&self, x1: ArrayView1<f64>, x2: ArrayView1<f64>) -> f64 {
        x1.iter()
            .zip(x2.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
    }

    fn cosine_similarity(&self, x1: ArrayView1<f64>, x2: ArrayView1<f64>) -> f64 {
        let dot_product: f64 = x1.iter().zip(x2.iter()).map(|(&a, &b)| a * b).sum();
        let norm_x1 = x1.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let norm_x2 = x2.iter().map(|&x| x * x).sum::<f64>().sqrt();

        if norm_x1 < 1e-15 || norm_x2 < 1e-15 {
            0.0
        } else {
            dot_product / (norm_x1 * norm_x2)
        }
    }
}

impl Default for InformationDensity {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::array;

    #[test]
    fn test_uncertainty_sampling_entropy() {
        let probas = array![
            [0.9, 0.1], // Low entropy (certain)
            [0.6, 0.4], // Medium entropy
            [0.5, 0.5], // High entropy (uncertain)
            [0.8, 0.2], // Low entropy
        ];

        let us = UncertaintySampling::new()
            .strategy("entropy".to_string())
            .n_samples(2);
        let selected = us.select_samples(&probas.view()).unwrap();

        assert_eq!(selected.len(), 2);
        // Most uncertain sample should be selected first (index 2: [0.5, 0.5])
        assert!(selected.contains(&2));
    }

    #[test]
    fn test_uncertainty_sampling_margin() {
        let probas = array![
            [0.9, 0.1], // Large margin
            [0.6, 0.4], // Small margin
            [0.5, 0.5], // No margin (most uncertain)
            [0.8, 0.2], // Medium margin
        ];

        let us = UncertaintySampling::new()
            .strategy("margin".to_string())
            .n_samples(2);
        let selected = us.select_samples(&probas.view()).unwrap();

        assert_eq!(selected.len(), 2);
        // Sample with smallest margin should be selected (index 2)
        assert!(selected.contains(&2));
    }

    #[test]
    fn test_uncertainty_sampling_least_confident() {
        let probas = array![
            [0.9, 0.1], // High confidence
            [0.6, 0.4], // Medium confidence
            [0.5, 0.5], // Least confident
            [0.8, 0.2], // High confidence
        ];

        let us = UncertaintySampling::new()
            .strategy("least_confident".to_string())
            .n_samples(2);
        let selected = us.select_samples(&probas.view()).unwrap();

        assert_eq!(selected.len(), 2);
        // Least confident sample should be selected (index 2)
        assert!(selected.contains(&2));
    }

    #[test]
    fn test_uncertainty_sampling_temperature_scaling() {
        let us = UncertaintySampling::new().temperature(2.0);
        let probas = array![[0.8, 0.2], [0.6, 0.4]];

        let calibrated = us.apply_temperature_scaling(&probas.view());

        // Temperature scaling should make probabilities less extreme
        assert!(calibrated[[0, 0]] < 0.8);
        assert!(calibrated[[0, 1]] > 0.2);

        // Check that probabilities still sum to 1
        for i in 0..calibrated.nrows() {
            let sum: f64 = calibrated.row(i).sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_query_by_committee_vote_entropy() {
        let committee_probas = vec![
            array![[0.8, 0.2], [0.6, 0.4], [0.3, 0.7]], // Member 1
            array![[0.7, 0.3], [0.5, 0.5], [0.4, 0.6]], // Member 2
            array![[0.9, 0.1], [0.4, 0.6], [0.2, 0.8]], // Member 3
        ];

        let qbc = QueryByCommittee::new()
            .disagreement_measure("vote_entropy".to_string())
            .n_samples(2);
        let selected = qbc.select_samples(&committee_probas).unwrap();

        assert_eq!(selected.len(), 2);
        // Should select samples where committee members disagree most
    }

    #[test]
    fn test_query_by_committee_kl_divergence() {
        let committee_probas = vec![
            array![[0.8, 0.2], [0.6, 0.4]],
            array![[0.2, 0.8], [0.4, 0.6]], // Very different from first
        ];

        let qbc = QueryByCommittee::new()
            .disagreement_measure("kl_divergence".to_string())
            .n_samples(1);
        let selected = qbc.select_samples(&committee_probas).unwrap();

        assert_eq!(selected.len(), 1);
    }

    #[test]
    fn test_query_by_committee_variance() {
        let committee_probas = vec![
            array![[0.9, 0.1], [0.5, 0.5]],
            array![[0.8, 0.2], [0.6, 0.4]],
            array![[0.7, 0.3], [0.4, 0.6]],
        ];

        let qbc = QueryByCommittee::new()
            .disagreement_measure("variance".to_string())
            .n_samples(1);
        let selected = qbc.select_samples(&committee_probas).unwrap();

        assert_eq!(selected.len(), 1);
    }

    #[test]
    fn test_uncertainty_sampling_diversity() {
        let probas = array![
            [0.5, 0.5], // Uncertain
            [0.4, 0.6], // Uncertain and different
            [0.6, 0.4], // Uncertain and different
            [0.9, 0.1], // Certain
        ];

        let us = UncertaintySampling::new()
            .strategy("entropy".to_string())
            .diversity_weight(0.5)
            .n_samples(2);
        let selected = us.select_samples(&probas.view()).unwrap();

        assert_eq!(selected.len(), 2);
        // Should select diverse uncertain samples
    }

    #[test]
    fn test_uncertainty_sampling_edge_cases() {
        // Test with n_samples >= total samples
        let probas = array![[0.7, 0.3], [0.6, 0.4]];
        let us = UncertaintySampling::new().n_samples(5);
        let selected = us.select_samples(&probas.view()).unwrap();
        assert_eq!(selected.len(), 2);
        assert_eq!(selected, vec![0, 1]);

        // Test with invalid strategy
        let us_invalid = UncertaintySampling::new().strategy("invalid".to_string());
        let result = us_invalid.select_samples(&probas.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_query_by_committee_edge_cases() {
        // Test with empty committee
        let qbc = QueryByCommittee::new();
        let result = qbc.select_samples(&[]);
        assert!(result.is_err());

        // Test with mismatched dimensions
        let committee_probas = vec![
            array![[0.8, 0.2], [0.6, 0.4]],
            array![[0.7, 0.3]], // Different number of samples
        ];
        let result = qbc.select_samples(&committee_probas);
        assert!(result.is_err());
    }

    #[test]
    fn test_kl_divergence_computation() {
        let us = UncertaintySampling::new();
        let p1 = array![0.8, 0.2];
        let p2 = array![0.6, 0.4];

        let kl = us.kl_divergence(&p1.view(), &p2.view());
        assert!(kl > 0.0);

        // KL divergence with self should be 0
        let kl_self = us.kl_divergence(&p1.view(), &p1.view());
        assert!(kl_self.abs() < 1e-10);
    }

    #[test]
    fn test_score_normalization() {
        let qbc = QueryByCommittee::new();
        let scores = array![1.0, 5.0, 3.0, 2.0];

        let normalized = qbc.normalize_scores(&scores);

        // Check range [0, 1]
        for &score in normalized.iter() {
            assert!(score >= 0.0 && score <= 1.0);
        }

        // Check min and max
        let min_normalized = normalized.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_normalized = normalized.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        assert!((min_normalized - 0.0).abs() < 1e-10);
        assert!((max_normalized - 1.0).abs() < 1e-10);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_expected_model_change_gradient_norm() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let gradients = array![
            [0.1, 0.2],  // Small gradient
            [0.8, 0.6],  // Large gradient
            [0.05, 0.1], // Very small gradient
            [0.4, 0.3],  // Medium gradient
        ];

        let emc = ExpectedModelChange::new()
            .approximation_method("gradient_norm".to_string())
            .n_samples(2);
        let selected = emc.select_samples(&X.view(), &gradients.view()).unwrap();

        assert_eq!(selected.len(), 2);
        // Sample with largest gradient should be selected first (index 1)
        assert!(selected.contains(&1));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_expected_model_change_fisher_information() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let gradients = array![[0.1, 0.2], [0.3, 0.1], [0.05, 0.4], [0.2, 0.3],];

        let emc = ExpectedModelChange::new()
            .approximation_method("fisher_information".to_string())
            .n_samples(2);
        let selected = emc.select_samples(&X.view(), &gradients.view()).unwrap();

        assert_eq!(selected.len(), 2);
        // Check that valid indices are selected
        for &idx in &selected {
            assert!(idx < 4);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_expected_model_change_parameter_variance() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let gradients = array![
            [0.1, 0.2],
            [0.8, 0.1], // Different from mean
            [0.1, 0.2],
            [0.1, 0.9], // Very different from mean
        ];

        let emc = ExpectedModelChange::new()
            .approximation_method("parameter_variance".to_string())
            .n_samples(2);
        let selected = emc.select_samples(&X.view(), &gradients.view()).unwrap();

        assert_eq!(selected.len(), 2);
        // Samples with high variance should be selected
        assert!(selected.contains(&3)); // High variance in second component
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_expected_model_change_diversity() {
        let X = array![
            [1.0, 1.0],
            [1.1, 1.1], // Close to first sample
            [5.0, 5.0], // Far from first samples
            [5.1, 5.1], // Close to third sample
        ];
        let gradients = array![
            [0.5, 0.5], // High gradient
            [0.4, 0.4], // High gradient, close to first
            [0.3, 0.3], // Medium gradient, far from others
            [0.2, 0.2], // Low gradient
        ];

        let emc = ExpectedModelChange::new()
            .approximation_method("gradient_norm".to_string())
            .diversity_weight(0.5)
            .n_samples(2);
        let selected = emc.select_samples(&X.view(), &gradients.view()).unwrap();

        assert_eq!(selected.len(), 2);
        // Should balance gradient magnitude with diversity
        // First sample (highest gradient) should be selected
        assert!(selected.contains(&0));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_expected_model_change_edge_cases() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let gradients = array![[0.1, 0.2], [0.3, 0.1]];

        // Test with n_samples >= total samples
        let emc = ExpectedModelChange::new().n_samples(5);
        let selected = emc.select_samples(&X.view(), &gradients.view()).unwrap();
        assert_eq!(selected.len(), 2);
        assert_eq!(selected, vec![0, 1]);

        // Test with mismatched dimensions
        let bad_gradients = array![[0.1, 0.2]]; // Only one gradient for two samples
        let result = emc.select_samples(&X.view(), &bad_gradients.view());
        assert!(result.is_err());

        // Test with invalid approximation method
        let emc_invalid = ExpectedModelChange::new()
            .approximation_method("invalid".to_string())
            .n_samples(1); // Ensure validation happens
        let result = emc_invalid.select_samples(&X.view(), &gradients.view());
        assert!(result.is_err());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_expected_model_change_normalization() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let gradients = array![[1.0, 0.0], [10.0, 0.0], [5.0, 0.0]];

        let emc_normalized = ExpectedModelChange::new()
            .normalize_scores(true)
            .n_samples(2);
        let selected_norm = emc_normalized
            .select_samples(&X.view(), &gradients.view())
            .unwrap();

        let emc_unnormalized = ExpectedModelChange::new()
            .normalize_scores(false)
            .n_samples(2);
        let selected_unnorm = emc_unnormalized
            .select_samples(&X.view(), &gradients.view())
            .unwrap();

        // Both should select same samples (highest gradients)
        assert_eq!(selected_norm.len(), 2);
        assert_eq!(selected_unnorm.len(), 2);
        assert!(selected_norm.contains(&1)); // Highest gradient
        assert!(selected_unnorm.contains(&1)); // Highest gradient
    }

    #[test]
    fn test_euclidean_distance_computation() {
        let emc = ExpectedModelChange::new();
        let x1 = array![1.0, 2.0];
        let x2 = array![4.0, 6.0];

        let distance = emc.euclidean_distance(x1.view(), x2.view());
        let expected = ((4.0_f64 - 1.0).powi(2) + (6.0_f64 - 2.0).powi(2)).sqrt();
        assert!((distance - expected).abs() < 1e-10);

        // Distance to self should be 0
        let distance_self = emc.euclidean_distance(x1.view(), x1.view());
        assert!(distance_self.abs() < 1e-10);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_information_density_knn() {
        let X = array![
            [1.0, 1.0], // Clustered samples
            [1.1, 1.1],
            [1.2, 1.2],
            [10.0, 10.0], // Isolated sample
        ];
        let probas = array![
            [0.5, 0.5], // High uncertainty
            [0.6, 0.4], // Medium uncertainty
            [0.7, 0.3], // Lower uncertainty
            [0.9, 0.1], // Low uncertainty
        ];

        let id = InformationDensity::new()
            .uncertainty_measure("entropy".to_string())
            .density_measure("knn_density".to_string())
            .density_weight(0.5)
            .k_neighbors(2)
            .n_samples(2);
        let selected = id.select_samples(&X.view(), &probas.view()).unwrap();

        assert_eq!(selected.len(), 2);
        // Should prefer uncertain samples in dense regions
        // Clustered uncertain samples should be preferred over isolated uncertain ones
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_information_density_gaussian() {
        let X = array![[1.0, 1.0], [1.5, 1.5], [2.0, 2.0], [10.0, 10.0],];
        let probas = array![
            [0.5, 0.5], // High uncertainty
            [0.6, 0.4],
            [0.7, 0.3],
            [0.5, 0.5], // High uncertainty but isolated
        ];

        let id = InformationDensity::new()
            .uncertainty_measure("entropy".to_string())
            .density_measure("gaussian_density".to_string())
            .density_weight(0.7)
            .bandwidth(1.0)
            .n_samples(2);
        let selected = id.select_samples(&X.view(), &probas.view()).unwrap();

        assert_eq!(selected.len(), 2);
        // Should favor dense regions with high weight on density
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_information_density_cosine_similarity() {
        let X = array![
            [1.0, 0.0], // Orthogonal vectors
            [0.0, 1.0],
            [1.0, 1.0], // Similar to first two
            [0.5, 0.5],
        ];
        let probas = array![
            [0.5, 0.5], // High uncertainty
            [0.5, 0.5], // High uncertainty
            [0.6, 0.4],
            [0.7, 0.3],
        ];

        let id = InformationDensity::new()
            .uncertainty_measure("entropy".to_string())
            .density_measure("cosine_similarity".to_string())
            .density_weight(0.3)
            .n_samples(2);
        let selected = id.select_samples(&X.view(), &probas.view()).unwrap();

        assert_eq!(selected.len(), 2);
        // Should select based on both uncertainty and cosine similarity
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_information_density_margin_uncertainty() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let probas = array![
            [0.9, 0.1],   // Low margin (certain)
            [0.6, 0.4],   // High margin (uncertain)
            [0.55, 0.45], // Highest margin (most uncertain)
            [0.8, 0.2],   // Low margin
        ];

        let id = InformationDensity::new()
            .uncertainty_measure("margin".to_string())
            .density_measure("knn_density".to_string())
            .density_weight(0.0) // Pure uncertainty
            .n_samples(2);
        let selected = id.select_samples(&X.view(), &probas.view()).unwrap();

        assert_eq!(selected.len(), 2);
        // Should select samples with smallest margins (most uncertain)
        assert!(selected.contains(&2)); // Smallest margin
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_information_density_least_confident() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let probas = array![
            [0.9, 0.1],   // Very confident
            [0.6, 0.4],   // Less confident
            [0.55, 0.45], // Least confident
            [0.8, 0.2],   // Confident
        ];

        let id = InformationDensity::new()
            .uncertainty_measure("least_confident".to_string())
            .density_measure("knn_density".to_string())
            .density_weight(0.0) // Pure uncertainty
            .n_samples(2);
        let selected = id.select_samples(&X.view(), &probas.view()).unwrap();

        assert_eq!(selected.len(), 2);
        // Should select least confident samples
        assert!(selected.contains(&2)); // Least confident
        assert!(selected.contains(&1)); // Second least confident
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_information_density_temperature_scaling() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let probas = array![[0.8, 0.2], [0.6, 0.4], [0.7, 0.3],];

        // Test with different temperatures
        let id_low_temp = InformationDensity::new()
            .temperature(0.5) // Sharper probabilities
            .density_weight(0.0)
            .n_samples(2);
        let selected_low = id_low_temp
            .select_samples(&X.view(), &probas.view())
            .unwrap();

        let id_high_temp = InformationDensity::new()
            .temperature(2.0) // Smoother probabilities
            .density_weight(0.0)
            .n_samples(2);
        let selected_high = id_high_temp
            .select_samples(&X.view(), &probas.view())
            .unwrap();

        assert_eq!(selected_low.len(), 2);
        assert_eq!(selected_high.len(), 2);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_information_density_edge_cases() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let probas = array![[0.6, 0.4], [0.7, 0.3]];

        // Test with n_samples >= total samples
        let id = InformationDensity::new().n_samples(5);
        let selected = id.select_samples(&X.view(), &probas.view()).unwrap();
        assert_eq!(selected.len(), 2);
        assert_eq!(selected, vec![0, 1]);

        // Test with mismatched dimensions
        let bad_probas = array![[0.6, 0.4]]; // Only one probability for two samples
        let result = id.select_samples(&X.view(), &bad_probas.view());
        assert!(result.is_err());

        // Test with invalid uncertainty measure
        let id_invalid = InformationDensity::new()
            .uncertainty_measure("invalid".to_string())
            .n_samples(1); // Ensure validation happens
        let result = id_invalid.select_samples(&X.view(), &probas.view());
        assert!(result.is_err());

        // Test with invalid density measure
        let id_invalid = InformationDensity::new()
            .density_measure("invalid".to_string())
            .n_samples(1); // Ensure validation happens
        let result = id_invalid.select_samples(&X.view(), &probas.view());
        assert!(result.is_err());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_information_density_pure_modes() {
        let X = array![
            [1.0, 1.0], // Dense cluster
            [1.1, 1.1],
            [10.0, 10.0], // Isolated
            [11.0, 11.0], // Another cluster
        ];
        let probas = array![
            [0.5, 0.5], // High uncertainty
            [0.9, 0.1], // Low uncertainty
            [0.5, 0.5], // High uncertainty
            [0.8, 0.2], // Low uncertainty
        ];

        // Pure uncertainty (density_weight = 0)
        let id_uncertainty = InformationDensity::new().density_weight(0.0).n_samples(2);
        let selected_unc = id_uncertainty
            .select_samples(&X.view(), &probas.view())
            .unwrap();
        assert!(selected_unc.contains(&0)); // High uncertainty
        assert!(selected_unc.contains(&2)); // High uncertainty

        // Pure density (density_weight = 1)
        let id_density = InformationDensity::new()
            .density_weight(1.0)
            .k_neighbors(1)
            .n_samples(2);
        let selected_den = id_density
            .select_samples(&X.view(), &probas.view())
            .unwrap();
        // Should prefer samples in denser regions
        assert_eq!(selected_den.len(), 2);
    }

    #[test]
    fn test_cosine_similarity_computation() {
        let id = InformationDensity::new();

        // Test orthogonal vectors
        let x1 = array![1.0, 0.0];
        let x2 = array![0.0, 1.0];
        let sim = id.cosine_similarity(x1.view(), x2.view());
        assert!((sim - 0.0).abs() < 1e-10);

        // Test identical vectors
        let x3 = array![1.0, 1.0];
        let x4 = array![1.0, 1.0];
        let sim_identical = id.cosine_similarity(x3.view(), x4.view());
        assert!((sim_identical - 1.0).abs() < 1e-10);

        // Test opposite vectors
        let x5 = array![1.0, 0.0];
        let x6 = array![-1.0, 0.0];
        let sim_opposite = id.cosine_similarity(x5.view(), x6.view());
        assert!((sim_opposite - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_array_normalization() {
        let id = InformationDensity::new();
        let array = array![1.0, 5.0, 3.0, 2.0];

        let normalized = id.normalize_array(&array);

        // Check range [0, 1]
        for &value in normalized.iter() {
            assert!(value >= 0.0 && value <= 1.0);
        }

        // Check min and max
        let min_norm = normalized.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_norm = normalized.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        assert!((min_norm - 0.0).abs() < 1e-10);
        assert!((max_norm - 1.0).abs() < 1e-10);

        // Test with constant array
        let constant = array![5.0, 5.0, 5.0];
        let normalized_constant = id.normalize_array(&constant);
        for &value in normalized_constant.iter() {
            assert!((value - 0.5).abs() < 1e-10);
        }
    }
}
