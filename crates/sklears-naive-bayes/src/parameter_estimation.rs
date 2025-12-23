//! Parameter estimation methods for Naive Bayes classifiers

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;

/// Parameter estimation method enumeration
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ParameterEstimationMethod {
    /// Maximum Likelihood Estimation
    #[default]
    MaximumLikelihood,
    /// Maximum A Posteriori estimation with uniform priors
    MaximumAPosteriori,
    /// Bayesian parameter estimation with conjugate priors
    Bayesian,
    /// Empirical Bayes estimation
    EmpiricalBayes,
    /// Method of moments estimation
    MethodOfMoments,
}

/// Trait for parameter estimation
pub trait ParameterEstimator<F: Float> {
    type Parameters;
    type Error;

    /// Estimate parameters from data
    fn estimate(&self, data: &[F]) -> Result<Self::Parameters, Self::Error>;

    /// Get the estimation method used
    fn method(&self) -> ParameterEstimationMethod;
}

/// Gaussian distribution parameter estimator
#[derive(Debug, Clone)]
pub struct GaussianEstimator<F: Float> {
    method: ParameterEstimationMethod,
    prior_mean: Option<F>,
    prior_variance: Option<F>,
    prior_strength: F, // How strong the prior is (for MAP/Bayesian)
}

#[derive(Debug, Clone)]
pub struct GaussianParameters<F: Float> {
    pub mean: F,
    pub variance: F,
}

impl<F: Float> GaussianEstimator<F> {
    pub fn new(method: ParameterEstimationMethod) -> Self {
        Self {
            method,
            prior_mean: None,
            prior_variance: None,
            prior_strength: F::from(1.0).unwrap(),
        }
    }

    pub fn with_priors(mut self, prior_mean: F, prior_variance: F, strength: F) -> Self {
        self.prior_mean = Some(prior_mean);
        self.prior_variance = Some(prior_variance);
        self.prior_strength = strength;
        self
    }
}

impl<F: Float> ParameterEstimator<F> for GaussianEstimator<F> {
    type Parameters = GaussianParameters<F>;
    type Error = String;

    fn estimate(&self, data: &[F]) -> Result<Self::Parameters, Self::Error> {
        if data.is_empty() {
            return Err("Empty data provided".to_string());
        }

        match self.method {
            ParameterEstimationMethod::MaximumLikelihood => {
                let n = F::from(data.len()).unwrap();
                let mean = data.iter().fold(F::zero(), |acc, &x| acc + x) / n;
                let variance = if data.len() > 1 {
                    data.iter()
                        .map(|&x| (x - mean) * (x - mean))
                        .fold(F::zero(), |acc, x| acc + x)
                        / n
                } else {
                    F::from(1e-10).unwrap() // Small variance for single sample
                };

                Ok(GaussianParameters { mean, variance })
            }

            ParameterEstimationMethod::MaximumAPosteriori => {
                if let (Some(prior_mean), Some(prior_var)) = (self.prior_mean, self.prior_variance)
                {
                    let n = F::from(data.len()).unwrap();
                    let sample_mean = data.iter().fold(F::zero(), |acc, &x| acc + x) / n;

                    // MAP estimate with Normal-Inverse-Gamma prior
                    let precision = self.prior_strength;
                    let posterior_precision = precision + n;
                    let posterior_mean =
                        (precision * prior_mean + n * sample_mean) / posterior_precision;

                    // For variance, use a simple regularized estimate
                    let sample_var = if data.len() > 1 {
                        data.iter()
                            .map(|&x| (x - sample_mean) * (x - sample_mean))
                            .fold(F::zero(), |acc, x| acc + x)
                            / n
                    } else {
                        prior_var
                    };

                    let regularized_var =
                        (sample_var + prior_var * precision / n) / (F::one() + precision / n);

                    Ok(GaussianParameters {
                        mean: posterior_mean,
                        variance: regularized_var,
                    })
                } else {
                    // Fall back to MLE if no priors specified
                    self.estimate(data)
                }
            }

            ParameterEstimationMethod::Bayesian => {
                // For simplicity, use MAP estimate
                self.estimate(data)
            }

            ParameterEstimationMethod::EmpiricalBayes => {
                // Estimate hyperparameters from data, then use those as priors
                let n = F::from(data.len()).unwrap();
                let sample_mean = data.iter().fold(F::zero(), |acc, &x| acc + x) / n;
                let sample_var = if data.len() > 1 {
                    data.iter()
                        .map(|&x| (x - sample_mean) * (x - sample_mean))
                        .fold(F::zero(), |acc, x| acc + x)
                        / (n - F::one())
                } else {
                    F::from(1e-10).unwrap()
                };

                // Use empirical moments as priors and then do MAP
                let estimator =
                    GaussianEstimator::new(ParameterEstimationMethod::MaximumAPosteriori)
                        .with_priors(sample_mean, sample_var, F::from(0.1).unwrap());
                estimator.estimate(data)
            }

            ParameterEstimationMethod::MethodOfMoments => {
                // Same as MLE for Gaussian
                let n = F::from(data.len()).unwrap();
                let mean = data.iter().fold(F::zero(), |acc, &x| acc + x) / n;
                let variance = if data.len() > 1 {
                    data.iter()
                        .map(|&x| (x - mean) * (x - mean))
                        .fold(F::zero(), |acc, x| acc + x)
                        / (n - F::one()) // Unbiased estimate
                } else {
                    F::from(1e-10).unwrap()
                };

                Ok(GaussianParameters { mean, variance })
            }
        }
    }

    fn method(&self) -> ParameterEstimationMethod {
        self.method
    }
}

/// Multinomial distribution parameter estimator
#[derive(Debug, Clone)]
pub struct MultinomialEstimator<F: Float> {
    method: ParameterEstimationMethod,
    alpha: F, // Dirichlet prior parameter
}

#[derive(Debug, Clone)]
pub struct MultinomialParameters<F: Float> {
    pub probabilities: Vec<F>,
}

impl<F: Float> MultinomialEstimator<F> {
    pub fn new(method: ParameterEstimationMethod) -> Self {
        Self {
            method,
            alpha: F::one(), // Default uniform prior
        }
    }

    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }
}

impl<F: Float> ParameterEstimator<F> for MultinomialEstimator<F> {
    type Parameters = MultinomialParameters<F>;
    type Error = String;

    fn estimate(&self, data: &[F]) -> Result<Self::Parameters, Self::Error> {
        if data.is_empty() {
            return Err("Empty data provided".to_string());
        }

        match self.method {
            ParameterEstimationMethod::MaximumLikelihood => {
                let total = data.iter().fold(F::zero(), |acc, &x| acc + x);
                let probabilities = data.iter().map(|&x| x / total).collect();
                Ok(MultinomialParameters { probabilities })
            }

            ParameterEstimationMethod::MaximumAPosteriori => {
                // MAP with Dirichlet prior
                let total = data.iter().fold(F::zero(), |acc, &x| acc + x);
                let vocab_size = F::from(data.len()).unwrap();
                let smoothed_total = total + self.alpha * vocab_size;

                let probabilities = data
                    .iter()
                    .map(|&x| (x + self.alpha) / smoothed_total)
                    .collect();

                Ok(MultinomialParameters { probabilities })
            }

            _ => {
                // For other methods, use MAP as default
                let total = data.iter().fold(F::zero(), |acc, &x| acc + x);
                let vocab_size = F::from(data.len()).unwrap();
                let smoothed_total = total + self.alpha * vocab_size;

                let probabilities = data
                    .iter()
                    .map(|&x| (x + self.alpha) / smoothed_total)
                    .collect();

                Ok(MultinomialParameters { probabilities })
            }
        }
    }

    fn method(&self) -> ParameterEstimationMethod {
        self.method
    }
}

/// Cross-validation for hyperparameter selection
pub struct CrossValidationSelector {
    pub folds: usize,
    pub random_state: Option<u64>,
}

impl CrossValidationSelector {
    pub fn new(folds: usize) -> Self {
        Self {
            folds,
            random_state: None,
        }
    }

    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Select best hyperparameters using cross-validation
    /// This is a simplified version - in practice you'd want more sophisticated CV
    pub fn select_best_alpha<F: Float + std::fmt::Debug + Copy>(
        &self,
        data: &Array2<F>,
        labels: &Array1<i32>,
        alpha_candidates: &[F],
    ) -> Result<F, String> {
        if alpha_candidates.is_empty() {
            return Err("No alpha candidates provided".to_string());
        }

        let n_samples = data.nrows();
        let fold_size = n_samples / self.folds;

        let mut best_alpha = alpha_candidates[0];
        let mut best_score = F::neg_infinity();

        for &alpha in alpha_candidates {
            let mut scores = Vec::new();

            // Simple k-fold cross-validation
            for fold in 0..self.folds {
                let start_idx = fold * fold_size;
                let end_idx = if fold == self.folds - 1 {
                    n_samples
                } else {
                    (fold + 1) * fold_size
                };

                // Create train/test split
                let mut train_indices = Vec::new();
                let mut test_indices = Vec::new();

                for i in 0..n_samples {
                    if i >= start_idx && i < end_idx {
                        test_indices.push(i);
                    } else {
                        train_indices.push(i);
                    }
                }

                if train_indices.is_empty() || test_indices.is_empty() {
                    continue;
                }

                // Train a Multinomial NB model with this alpha and evaluate it
                let score =
                    self.evaluate_fold(data, labels, &train_indices, &test_indices, alpha)?;
                scores.push(score);
            }

            if !scores.is_empty() {
                let avg_score = scores.iter().fold(F::zero(), |acc, &x| acc + x)
                    / F::from(scores.len()).unwrap();
                if avg_score > best_score {
                    best_score = avg_score;
                    best_alpha = alpha;
                }
            }
        }

        Ok(best_alpha)
    }

    /// Evaluate a single fold by training and testing a Multinomial NB model
    fn evaluate_fold<F: Float + std::fmt::Debug + Copy>(
        &self,
        data: &Array2<F>,
        labels: &Array1<i32>,
        train_indices: &[usize],
        test_indices: &[usize],
        alpha: F,
    ) -> Result<F, String> {
        // Extract training data
        let mut train_data = Vec::new();
        let mut train_labels = Vec::new();
        for &idx in train_indices {
            train_data.push(data.row(idx).to_owned());
            train_labels.push(labels[idx]);
        }

        // Get unique classes from training labels
        let mut unique_classes: Vec<i32> = train_labels.clone();
        unique_classes.sort_unstable();
        unique_classes.dedup();

        if unique_classes.is_empty() {
            return Err("No classes found in training data".to_string());
        }

        // Compute class priors
        let mut class_counts: std::collections::HashMap<i32, usize> =
            std::collections::HashMap::new();
        for &label in &train_labels {
            *class_counts.entry(label).or_insert(0) += 1;
        }

        let total_samples = train_labels.len() as f64;
        let mut class_priors: std::collections::HashMap<i32, F> = std::collections::HashMap::new();
        for (&class, &count) in &class_counts {
            class_priors.insert(class, F::from(count as f64 / total_samples).unwrap());
        }

        // Compute feature log probabilities for each class with smoothing
        let n_features = data.ncols();
        let mut class_feature_log_probs: std::collections::HashMap<i32, Vec<F>> =
            std::collections::HashMap::new();

        for &class in &unique_classes {
            let mut feature_counts = vec![F::zero(); n_features];
            let mut total_count = F::zero();

            // Sum features for this class
            for (i, &label) in train_labels.iter().enumerate() {
                if label == class {
                    for (j, count) in feature_counts.iter_mut().enumerate().take(n_features) {
                        *count = *count + train_data[i][j];
                        total_count = total_count + train_data[i][j];
                    }
                }
            }

            // Apply Laplace smoothing and compute log probabilities
            let vocab_size = F::from(n_features).unwrap();
            let smoothed_total = total_count + alpha * vocab_size;

            let log_probs: Vec<F> = feature_counts
                .iter()
                .map(|&count| {
                    let smoothed_count = count + alpha;
                    (smoothed_count / smoothed_total).ln()
                })
                .collect();

            class_feature_log_probs.insert(class, log_probs);
        }

        // Evaluate on test set
        let mut correct = 0;
        for &idx in test_indices {
            let test_sample = data.row(idx);

            // Compute log probabilities for each class
            let mut class_scores: Vec<(i32, F)> = Vec::new();
            for &class in &unique_classes {
                let mut log_score = class_priors[&class].ln();
                let log_probs = &class_feature_log_probs[&class];

                for j in 0..n_features {
                    log_score = log_score + test_sample[j] * log_probs[j];
                }

                class_scores.push((class, log_score));
            }

            // Find class with maximum score
            let predicted_class = class_scores
                .iter()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(c, _)| *c)
                .ok_or("No predictions made")?;

            if predicted_class == labels[idx] {
                correct += 1;
            }
        }

        // Return accuracy as score
        let accuracy = F::from(correct as f64 / test_indices.len() as f64).unwrap();
        Ok(accuracy)
    }
}

/// Empirical Bayes estimation for hyperparameters
pub struct EmpiricalBayesEstimator {
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl EmpiricalBayesEstimator {
    pub fn new() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }

    /// Estimate Dirichlet hyperparameter alpha using moment matching
    pub fn estimate_dirichlet_alpha<F: Float>(
        &self,
        class_probabilities: &[Vec<F>],
    ) -> Result<F, String> {
        if class_probabilities.is_empty() {
            return Err("No probability vectors provided".to_string());
        }

        let k = class_probabilities[0].len();
        let n = class_probabilities.len();

        // Compute sample moments
        let mut mean_probs = vec![F::zero(); k];
        let mut mean_log_probs = vec![F::zero(); k];

        for probs in class_probabilities {
            if probs.len() != k {
                return Err("Inconsistent probability vector lengths".to_string());
            }
            for (i, &p) in probs.iter().enumerate() {
                mean_probs[i] = mean_probs[i] + p;
                mean_log_probs[i] = mean_log_probs[i] + p.ln();
            }
        }

        let n_f = F::from(n).unwrap();
        for i in 0..k {
            mean_probs[i] = mean_probs[i] / n_f;
            mean_log_probs[i] = mean_log_probs[i] / n_f;
        }

        // Use moment matching to estimate alpha
        // For Dirichlet: E[ln(X_i)] = ψ(α_i) - ψ(Σα_j)
        // Assuming symmetric Dirichlet: α_i = α for all i

        let mean_log_prob =
            mean_log_probs.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(k).unwrap();
        let mean_prob = mean_probs.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(k).unwrap();

        // Simple approximation: use the relationship between mean and variance
        let mut alpha = F::one();

        // Newton-Raphson iteration would go here for more precise estimation
        // For now, use a simple heuristic
        alpha = mean_prob / (F::one() - mean_prob + F::from(1e-10).unwrap());
        alpha = alpha
            .max(F::from(0.1).unwrap())
            .min(F::from(100.0).unwrap());

        Ok(alpha)
    }
}

impl Default for EmpiricalBayesEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_gaussian_mle() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let estimator = GaussianEstimator::new(ParameterEstimationMethod::MaximumLikelihood);
        let params = estimator.estimate(&data).unwrap();

        assert_abs_diff_eq!(params.mean, 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(params.variance, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gaussian_map() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let estimator = GaussianEstimator::new(ParameterEstimationMethod::MaximumAPosteriori)
            .with_priors(0.0, 1.0, 1.0);
        let params = estimator.estimate(&data).unwrap();

        // Should be pulled towards prior
        assert!(params.mean < 3.0);
        assert!(params.mean > 0.0);
    }

    #[test]
    fn test_multinomial_mle() {
        let data = vec![10.0, 20.0, 30.0];
        let estimator = MultinomialEstimator::new(ParameterEstimationMethod::MaximumLikelihood);
        let params = estimator.estimate(&data).unwrap();

        let expected = vec![10.0 / 60.0, 20.0 / 60.0, 30.0 / 60.0];
        for (i, &prob) in params.probabilities.iter().enumerate() {
            assert_abs_diff_eq!(prob, expected[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_multinomial_map() {
        let data = vec![10.0, 20.0, 30.0];
        let estimator = MultinomialEstimator::new(ParameterEstimationMethod::MaximumAPosteriori)
            .with_alpha(1.0);
        let params = estimator.estimate(&data).unwrap();

        // Should be smoothed
        let total = 60.0 + 3.0; // data sum + alpha * vocab_size
        let expected = vec![11.0 / total, 21.0 / total, 31.0 / total];
        for (i, &prob) in params.probabilities.iter().enumerate() {
            assert_abs_diff_eq!(prob, expected[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_cross_validation_selector() {
        // SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
        use scirs2_core::ndarray::Array2;

        let data = Array2::zeros((10, 2));
        let labels = Array1::from_vec(vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);
        let alphas = vec![0.1, 0.5, 1.0, 2.0];

        let selector = CrossValidationSelector::new(3);
        let best_alpha = selector.select_best_alpha(&data, &labels, &alphas).unwrap();

        // Should return one of the candidates
        assert!(alphas.contains(&best_alpha));
    }
}
