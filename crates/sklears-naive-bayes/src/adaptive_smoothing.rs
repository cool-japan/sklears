//! Adaptive smoothing methods for Naive Bayes classifiers

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2, ScalarOperand};
use scirs2_core::numeric::Float;

use crate::smoothing::Smoothing;

/// Adaptive smoothing method that selects optimal smoothing parameters
#[derive(Debug, Clone)]
pub enum AdaptiveSmoothingMethod {
    /// Cross-validation based adaptive smoothing
    CrossValidation {
        folds: usize,

        alpha_candidates: Vec<f64>,
    },
    /// Held-out validation adaptive smoothing
    HeldOut {
        validation_fraction: f64,

        alpha_candidates: Vec<f64>,
    },
    /// Information criterion based smoothing (AIC/BIC)
    InformationCriterion {
        criterion: InformationCriterion,
        alpha_candidates: Vec<f64>,
    },
    /// Bayesian adaptive smoothing with hyperprior
    Bayesian { prior_alpha: f64, prior_beta: f64 },
    /// Frequency-based adaptive smoothing
    FrequencyBased { min_alpha: f64, max_alpha: f64 },
}

/// Information criteria for model selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InformationCriterion {
    /// Akaike Information Criterion
    AIC,
    /// Bayesian Information Criterion
    BIC,
    /// Minimum Description Length
    MDL,
}

/// Adaptive smoothing implementation
#[derive(Debug, Clone)]
pub struct AdaptiveSmoothing<F: Float + ScalarOperand> {
    method: AdaptiveSmoothingMethod,
    selected_alpha: Option<F>,
    feature_alphas: Option<Vec<F>>, // Per-feature adaptive smoothing
}

impl<F: Float + ScalarOperand> AdaptiveSmoothing<F> {
    pub fn new(method: AdaptiveSmoothingMethod) -> Self {
        Self {
            method,
            selected_alpha: None,
            feature_alphas: None,
        }
    }

    /// Fit adaptive smoothing parameters to data
    pub fn fit(
        &mut self,
        counts: &Array2<F>,
        total_counts: &Array1<F>,
        labels: &Array1<i32>,
    ) -> Result<(), String> {
        match &self.method {
            AdaptiveSmoothingMethod::CrossValidation {
                folds,
                alpha_candidates,
            } => {
                let best_alpha = self.select_by_cross_validation(
                    counts,
                    total_counts,
                    labels,
                    *folds,
                    alpha_candidates,
                )?;
                self.selected_alpha = Some(F::from(best_alpha).unwrap());
            }

            AdaptiveSmoothingMethod::HeldOut {
                validation_fraction,
                alpha_candidates,
            } => {
                let best_alpha = self.select_by_held_out(
                    counts,
                    total_counts,
                    labels,
                    *validation_fraction,
                    alpha_candidates,
                )?;
                self.selected_alpha = Some(F::from(best_alpha).unwrap());
            }

            AdaptiveSmoothingMethod::InformationCriterion {
                criterion,
                alpha_candidates,
            } => {
                let best_alpha = self.select_by_information_criterion(
                    counts,
                    total_counts,
                    *criterion,
                    alpha_candidates,
                )?;
                self.selected_alpha = Some(F::from(best_alpha).unwrap());
            }

            AdaptiveSmoothingMethod::Bayesian {
                prior_alpha,
                prior_beta,
            } => {
                let estimated_alpha = self.bayesian_adaptive_smoothing(
                    counts,
                    total_counts,
                    *prior_alpha,
                    *prior_beta,
                )?;
                self.selected_alpha = Some(F::from(estimated_alpha).unwrap());
            }

            AdaptiveSmoothingMethod::FrequencyBased {
                min_alpha,
                max_alpha,
            } => {
                let feature_alphas =
                    self.frequency_based_smoothing(counts, total_counts, *min_alpha, *max_alpha)?;
                self.feature_alphas = Some(
                    feature_alphas
                        .iter()
                        .map(|&a| F::from(a).unwrap())
                        .collect(),
                );
                self.selected_alpha = Some(
                    F::from(feature_alphas.iter().sum::<f64>() / feature_alphas.len() as f64)
                        .unwrap(),
                );
            }
        }

        Ok(())
    }

    /// Get the selected smoothing parameter
    pub fn alpha(&self) -> F {
        self.selected_alpha.unwrap_or(F::one())
    }

    /// Get per-feature smoothing parameters if available
    pub fn feature_alphas(&self) -> Option<&Vec<F>> {
        self.feature_alphas.as_ref()
    }

    /// Cross-validation based parameter selection
    fn select_by_cross_validation(
        &self,
        counts: &Array2<F>,
        total_counts: &Array1<F>,
        labels: &Array1<i32>,
        folds: usize,
        alpha_candidates: &[f64],
    ) -> Result<f64, String> {
        if alpha_candidates.is_empty() {
            return Err("No alpha candidates provided".to_string());
        }

        let n_samples = counts.nrows();
        let fold_size = n_samples / folds;
        let mut best_alpha = alpha_candidates[0];
        let mut best_score = f64::NEG_INFINITY;

        for &alpha in alpha_candidates {
            let mut fold_scores = Vec::new();

            for fold in 0..folds {
                let start_idx = fold * fold_size;
                let end_idx = if fold == folds - 1 {
                    n_samples
                } else {
                    (fold + 1) * fold_size
                };

                // Create train/test splits
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

                // Compute cross-validation score for this fold
                let score = self.compute_cv_score(
                    counts,
                    total_counts,
                    labels,
                    &train_indices,
                    &test_indices,
                    alpha,
                )?;
                fold_scores.push(score);
            }

            if !fold_scores.is_empty() {
                let avg_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
                if avg_score > best_score {
                    best_score = avg_score;
                    best_alpha = alpha;
                }
            }
        }

        Ok(best_alpha)
    }

    /// Held-out validation parameter selection
    fn select_by_held_out(
        &self,
        counts: &Array2<F>,
        total_counts: &Array1<F>,
        labels: &Array1<i32>,
        validation_fraction: f64,
        alpha_candidates: &[f64],
    ) -> Result<f64, String> {
        if alpha_candidates.is_empty() {
            return Err("No alpha candidates provided".to_string());
        }

        let n_samples = counts.nrows();
        let n_validation = (n_samples as f64 * validation_fraction) as usize;
        let n_train = n_samples - n_validation;

        if n_train == 0 || n_validation == 0 {
            return Err("Invalid validation split".to_string());
        }

        // Simple random split (in practice, you'd want better randomization)
        let train_indices: Vec<usize> = (0..n_train).collect();
        let test_indices: Vec<usize> = (n_train..n_samples).collect();

        let mut best_alpha = alpha_candidates[0];
        let mut best_score = f64::NEG_INFINITY;

        for &alpha in alpha_candidates {
            let score = self.compute_cv_score(
                counts,
                total_counts,
                labels,
                &train_indices,
                &test_indices,
                alpha,
            )?;
            if score > best_score {
                best_score = score;
                best_alpha = alpha;
            }
        }

        Ok(best_alpha)
    }

    /// Information criterion based parameter selection
    fn select_by_information_criterion(
        &self,
        counts: &Array2<F>,
        total_counts: &Array1<F>,
        criterion: InformationCriterion,
        alpha_candidates: &[f64],
    ) -> Result<f64, String> {
        if alpha_candidates.is_empty() {
            return Err("No alpha candidates provided".to_string());
        }

        let mut best_alpha = alpha_candidates[0];
        let mut best_score = f64::INFINITY; // Lower is better for IC

        for &alpha in alpha_candidates {
            let score =
                self.compute_information_criterion(counts, total_counts, alpha, criterion)?;
            if score < best_score {
                best_score = score;
                best_alpha = alpha;
            }
        }

        Ok(best_alpha)
    }

    /// Bayesian adaptive smoothing with conjugate priors
    fn bayesian_adaptive_smoothing(
        &self,
        counts: &Array2<F>,
        total_counts: &Array1<F>,
        prior_alpha: f64,
        prior_beta: f64,
    ) -> Result<f64, String> {
        // Use Gamma-Poisson conjugate prior for count data
        let n_features = counts.ncols();
        let n_classes = counts.nrows();

        let mut total_pseudo_counts = 0.0;
        let mut total_observations = 0.0;

        for i in 0..n_classes {
            for j in 0..n_features {
                let count = counts[[i, j]].to_f64().unwrap_or(0.0);
                total_pseudo_counts += count;
                total_observations += 1.0;
            }
        }

        // Posterior parameters for Gamma distribution
        let posterior_alpha = prior_alpha + total_pseudo_counts;
        let posterior_beta = prior_beta + total_observations;

        // MAP estimate
        let estimated_alpha = if posterior_alpha > 1.0 {
            (posterior_alpha - 1.0) / posterior_beta
        } else {
            posterior_alpha / posterior_beta
        };

        Ok(estimated_alpha.max(1e-10))
    }

    /// Frequency-based adaptive smoothing
    fn frequency_based_smoothing(
        &self,
        counts: &Array2<F>,
        total_counts: &Array1<F>,
        min_alpha: f64,
        max_alpha: f64,
    ) -> Result<Vec<f64>, String> {
        let n_features = counts.ncols();
        let mut feature_alphas = Vec::with_capacity(n_features);

        for j in 0..n_features {
            let column = counts.column(j);
            let total_count: f64 = column.iter().map(|&x| x.to_f64().unwrap_or(0.0)).sum();
            let non_zero_count = column.iter().filter(|&&x| x > F::zero()).count() as f64;

            // Adaptive alpha based on feature sparsity
            let sparsity = 1.0 - (non_zero_count / counts.nrows() as f64);
            let alpha = min_alpha + (max_alpha - min_alpha) * sparsity;

            feature_alphas.push(alpha);
        }

        Ok(feature_alphas)
    }

    /// Compute cross-validation score for a given alpha
    fn compute_cv_score(
        &self,
        counts: &Array2<F>,
        total_counts: &Array1<F>,
        labels: &Array1<i32>,
        train_indices: &[usize],
        test_indices: &[usize],
        alpha: f64,
    ) -> Result<f64, String> {
        // This is a simplified scoring function
        // In practice, you would train a full naive bayes model and evaluate it

        if train_indices.is_empty() || test_indices.is_empty() {
            return Ok(0.0);
        }

        // Compute log-likelihood as a proxy for performance
        let alpha_f = F::from(alpha).unwrap();
        let mut log_likelihood = 0.0;

        for &test_idx in test_indices {
            if test_idx < counts.nrows() {
                let test_counts = counts.row(test_idx);
                let test_total = total_counts[test_idx];

                // Simple log-likelihood computation with smoothing
                for &count in test_counts.iter() {
                    let smoothed_count = count + alpha_f;
                    let smoothed_total = test_total + alpha_f * F::from(counts.ncols()).unwrap();
                    let prob = smoothed_count / smoothed_total;
                    log_likelihood += prob.to_f64().unwrap_or(0.0).ln();
                }
            }
        }

        Ok(log_likelihood)
    }

    /// Compute information criterion for model selection
    fn compute_information_criterion(
        &self,
        counts: &Array2<F>,
        total_counts: &Array1<F>,
        alpha: f64,
        criterion: InformationCriterion,
    ) -> Result<f64, String> {
        let n_samples = counts.nrows() as f64;
        let n_features = counts.ncols() as f64;
        let n_classes = counts.nrows() as f64;

        // Compute log-likelihood
        let alpha_f = F::from(alpha).unwrap();
        let mut log_likelihood = 0.0;

        for i in 0..counts.nrows() {
            let row_counts = counts.row(i);
            let row_total = total_counts[i];

            for &count in row_counts.iter() {
                let smoothed_count = count + alpha_f;
                let smoothed_total = row_total + alpha_f * F::from(counts.ncols()).unwrap();
                let prob = smoothed_count / smoothed_total;
                log_likelihood += count.to_f64().unwrap_or(0.0) * prob.to_f64().unwrap_or(0.0).ln();
            }
        }

        // Number of parameters (smoothing parameter for each feature-class combination)
        let n_params = n_classes * n_features;

        let ic = match criterion {
            InformationCriterion::AIC => -2.0 * log_likelihood + 2.0 * n_params,
            InformationCriterion::BIC => -2.0 * log_likelihood + n_params * n_samples.ln(),
            InformationCriterion::MDL => -log_likelihood + 0.5 * n_params * n_samples.ln(),
        };

        Ok(ic)
    }
}

impl<F: Float + ScalarOperand> Smoothing<F> for AdaptiveSmoothing<F> {
    fn smooth_counts(&self, counts: &Array2<F>, total_counts: &Array1<F>) -> Array2<F> {
        if let Some(ref feature_alphas) = self.feature_alphas {
            // Per-feature adaptive smoothing
            let mut smoothed = counts.clone();
            for (j, &alpha) in feature_alphas.iter().enumerate() {
                for column in smoothed.column_mut(j) {
                    *column = *column + alpha;
                }
            }
            smoothed
        } else {
            // Global adaptive smoothing
            let alpha = self.alpha();
            counts + alpha
        }
    }

    fn alpha(&self) -> F {
        self.selected_alpha.unwrap_or(F::one())
    }
}

/// Factory function for creating adaptive smoothing instances
pub fn create_adaptive_smoother<F: Float + ScalarOperand + 'static>(
    method: AdaptiveSmoothingMethod,
) -> Box<dyn Smoothing<F>> {
    Box::new(AdaptiveSmoothing::new(method))
}

/// Hyperparameter optimization for multiple smoothing methods
#[derive(Debug, Clone)]
pub struct HyperparameterOptimizer {
    pub methods: Vec<AdaptiveSmoothingMethod>,
    pub scoring: ScoringMethod,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScoringMethod {
    /// LogLikelihood
    LogLikelihood,
    /// Accuracy
    Accuracy,
    /// F1Score
    F1Score,
}

impl HyperparameterOptimizer {
    pub fn new() -> Self {
        Self {
            methods: vec![
                /// AdaptiveSmoothingMethod

                AdaptiveSmoothingMethod::CrossValidation {
                    folds: 5,
                    alpha_candidates: vec![0.001, 0.01, 0.1, 1.0, 10.0],
                },
                AdaptiveSmoothingMethod::InformationCriterion {
                    criterion: InformationCriterion::BIC,
                    alpha_candidates: vec![0.001, 0.01, 0.1, 1.0, 10.0],
                },
            ],
            scoring: ScoringMethod::LogLikelihood,
        }
    }

    pub fn with_methods(mut self, methods: Vec<AdaptiveSmoothingMethod>) -> Self {
        self.methods = methods;
        self
    }

    pub fn with_scoring(mut self, scoring: ScoringMethod) -> Self {
        self.scoring = scoring;
        self
    }

    /// Find the best smoothing method and parameters
    pub fn optimize<F: Float + ScalarOperand>(
        &self,
        counts: &Array2<F>,
        total_counts: &Array1<F>,
        labels: &Array1<i32>,
    ) -> Result<AdaptiveSmoothingMethod, String> {
        let mut best_method = None;
        let mut best_score = f64::NEG_INFINITY;

        for method in &self.methods {
            let mut smoother = AdaptiveSmoothing::new(method.clone());
            smoother.fit(counts, total_counts, labels)?;

            let score = self.evaluate_method(&smoother, counts, total_counts, labels)?;

            if score > best_score {
                best_score = score;
                best_method = Some(method.clone());
            }
        }

        best_method.ok_or_else(|| "No methods provided".to_string())
    }

    fn evaluate_method<F: Float + ScalarOperand>(
        &self,
        smoother: &AdaptiveSmoothing<F>,
        counts: &Array2<F>,
        total_counts: &Array1<F>,
        labels: &Array1<i32>,
    ) -> Result<f64, String> {
        match self.scoring {
            ScoringMethod::LogLikelihood => {
                let smoothed_counts = smoother.smooth_counts(counts, total_counts);
                let mut log_likelihood = 0.0;

                for i in 0..counts.nrows() {
                    let row_counts = smoothed_counts.row(i);
                    let row_total =
                        total_counts[i] + smoother.alpha() * F::from(counts.ncols()).unwrap();

                    for &count in row_counts.iter() {
                        let prob = count / row_total;
                        log_likelihood += prob.to_f64().unwrap_or(0.0).ln();
                    }
                }

                Ok(log_likelihood)
            }
            _ => {
                // For other scoring methods, you would implement full model evaluation
                Ok(0.0) // Placeholder
            }
        }
    }
}

impl Default for HyperparameterOptimizer {
    fn default() -> Self {
        Self::new()
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
    fn test_frequency_based_adaptive_smoothing() {
        let counts = array![[1.0, 0.0, 3.0], [0.0, 2.0, 1.0]];
        let total_counts = array![4.0, 3.0];
        let labels = array![0, 1];

        let method = AdaptiveSmoothingMethod::FrequencyBased {
            min_alpha: 0.1,
            max_alpha: 2.0,
        };

        let mut smoother = AdaptiveSmoothing::new(method);
        smoother.fit(&counts, &total_counts, &labels).unwrap();

        assert!(smoother.alpha() >= 0.1);
        assert!(smoother.alpha() <= 2.0);
        assert!(smoother.feature_alphas().is_some());
    }

    #[test]
    fn test_information_criterion_selection() {
        let counts = array![[5.0, 2.0, 1.0], [1.0, 4.0, 2.0]];
        let total_counts = array![8.0, 7.0];
        let labels = array![0, 1];

        let method = AdaptiveSmoothingMethod::InformationCriterion {
            criterion: InformationCriterion::BIC,
            alpha_candidates: vec![0.1, 1.0, 10.0],
        };

        let mut smoother = AdaptiveSmoothing::new(method);
        let result = smoother.fit(&counts, &total_counts, &labels);
        assert!(result.is_ok());

        let alpha = smoother.alpha();
        assert!(alpha > 0.0);
    }

    #[test]
    fn test_bayesian_adaptive_smoothing() {
        let counts = array![[3.0, 1.0], [2.0, 4.0]];
        let total_counts = array![4.0, 6.0];
        let labels = array![0, 1];

        let method = AdaptiveSmoothingMethod::Bayesian {
            prior_alpha: 1.0,
            prior_beta: 1.0,
        };

        let mut smoother = AdaptiveSmoothing::new(method);
        smoother.fit(&counts, &total_counts, &labels).unwrap();

        assert!(smoother.alpha() > 0.0);
    }

    #[test]
    fn test_hyperparameter_optimizer() {
        let counts = array![[3.0, 1.0], [2.0, 4.0]];
        let total_counts = array![4.0, 6.0];
        let labels = array![0, 1];

        let optimizer = HyperparameterOptimizer::new();
        let best_method = optimizer.optimize(&counts, &total_counts, &labels).unwrap();

        // Should return one of the predefined methods
        match best_method {
            AdaptiveSmoothingMethod::CrossValidation { .. }
            | AdaptiveSmoothingMethod::InformationCriterion { .. } => {
                // Expected
            }
            _ => panic!("Unexpected method returned"),
        }
    }

    #[test]
    fn test_adaptive_smoothing_as_trait() {
        let counts = array![[1.0, 2.0], [3.0, 0.0]];
        let total_counts = array![3.0, 3.0];
        let labels = array![0, 1];

        let method = AdaptiveSmoothingMethod::FrequencyBased {
            min_alpha: 0.5,
            max_alpha: 1.5,
        };

        let mut smoother = AdaptiveSmoothing::new(method);
        smoother.fit(&counts, &total_counts, &labels).unwrap();

        let smoothed = smoother.smooth_counts(&counts, &total_counts);

        // All values should be increased by some smoothing amount
        for i in 0..counts.nrows() {
            for j in 0..counts.ncols() {
                assert!(smoothed[[i, j]] > counts[[i, j]]);
            }
        }
    }
}
