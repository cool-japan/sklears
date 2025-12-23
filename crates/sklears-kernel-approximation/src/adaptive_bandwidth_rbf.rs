//! Adaptive bandwidth RBF kernel approximation methods
//!
//! This module implements RBF kernel approximation with automatic bandwidth selection
//! based on data characteristics. The bandwidth (gamma) parameter is optimized using
//! various strategies including cross-validation, maximum likelihood, and heuristic methods.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::rngs::StdRng as RealStdRng;
use scirs2_core::random::Rng;
use scirs2_core::random::{thread_rng, SeedableRng};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Bandwidth selection strategy for adaptive RBF
#[derive(Debug, Clone, Copy)]
/// BandwidthSelectionStrategy
pub enum BandwidthSelectionStrategy {
    /// Cross-validation to minimize approximation error
    CrossValidation,
    /// Maximum likelihood estimation
    MaximumLikelihood,
    /// Median heuristic based on pairwise distances
    MedianHeuristic,
    /// Scott's rule based on data dimensionality and sample size
    ScottRule,
    /// Silverman's rule of thumb
    SilvermanRule,
    /// Leave-one-out cross-validation
    LeaveOneOut,
    /// Grid search over a range of gamma values
    GridSearch,
}

/// Objective function for bandwidth optimization
#[derive(Debug, Clone, Copy)]
/// ObjectiveFunction
pub enum ObjectiveFunction {
    /// Kernel alignment
    KernelAlignment,
    /// Log-likelihood
    LogLikelihood,
    /// Cross-validation error
    CrossValidationError,
    /// Kernel matrix trace
    KernelTrace,
    /// Effective dimensionality
    EffectiveDimensionality,
}

/// Adaptive bandwidth RBF sampler with automatic gamma selection
///
/// This sampler automatically selects the optimal bandwidth parameter (gamma) for RBF
/// kernel approximation based on data characteristics. Multiple strategies are available
/// for bandwidth selection, from simple heuristics to sophisticated optimization methods.
///
/// # Mathematical Background
///
/// The RBF kernel with adaptive bandwidth is: K(x,y) = exp(-γ*||x-y||²)
/// where γ is automatically selected to optimize a given objective function.
///
/// Common bandwidth selection strategies:
/// - Median heuristic: γ = 1/(2*median²) where median is the median pairwise distance
/// - Scott's rule: γ = n^(-1/(d+4)) for n samples and d dimensions
/// - Cross-validation: γ = argmin CV_error(γ)
///
/// # Examples
///
/// ```ignore
/// use sklears_kernel_approximation::{AdaptiveBandwidthRBFSampler, BandwidthSelectionStrategy};
/// use sklears_core::traits::{Transform, Fit, Untrained}
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
///
/// let sampler = AdaptiveBandwidthRBFSampler::new(100)
///     .strategy(BandwidthSelectionStrategy::MedianHeuristic);
///
/// let fitted = sampler.fit(&x, &()).unwrap();
/// let features = fitted.transform(&x).unwrap();
/// let optimal_gamma = fitted.selected_gamma();
/// ```
#[derive(Debug, Clone)]
/// AdaptiveBandwidthRBFSampler
pub struct AdaptiveBandwidthRBFSampler<State = Untrained> {
    /// Number of random features
    pub n_components: usize,
    /// Bandwidth selection strategy
    pub strategy: BandwidthSelectionStrategy,
    /// Objective function for optimization
    pub objective_function: ObjectiveFunction,
    /// Search range for gamma (min, max)
    pub gamma_range: (Float, Float),
    /// Number of gamma candidates for grid search
    pub n_gamma_candidates: usize,
    /// Cross-validation folds
    pub cv_folds: usize,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
    /// Tolerance for optimization convergence
    pub tolerance: Float,
    /// Maximum iterations for optimization
    pub max_iterations: usize,

    // Fitted attributes
    selected_gamma_: Option<Float>,
    random_weights_: Option<Array2<Float>>,
    random_offset_: Option<Array1<Float>>,
    optimization_history_: Option<Vec<(Float, Float)>>, // (gamma, objective_value)

    // State marker
    _state: PhantomData<State>,
}

impl AdaptiveBandwidthRBFSampler<Untrained> {
    /// Create a new adaptive bandwidth RBF sampler
    ///
    /// # Arguments
    /// * `n_components` - Number of random features to generate
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            strategy: BandwidthSelectionStrategy::MedianHeuristic,
            objective_function: ObjectiveFunction::KernelAlignment,
            gamma_range: (1e-3, 1e3),
            n_gamma_candidates: 20,
            cv_folds: 5,
            random_state: None,
            tolerance: 1e-6,
            max_iterations: 100,
            selected_gamma_: None,
            random_weights_: None,
            random_offset_: None,
            optimization_history_: None,
            _state: PhantomData,
        }
    }

    /// Set the bandwidth selection strategy
    pub fn strategy(mut self, strategy: BandwidthSelectionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the objective function for bandwidth optimization
    pub fn objective_function(mut self, objective: ObjectiveFunction) -> Self {
        self.objective_function = objective;
        self
    }

    /// Set the search range for gamma values
    pub fn gamma_range(mut self, min: Float, max: Float) -> Self {
        self.gamma_range = (min, max);
        self
    }

    /// Set the number of gamma candidates for grid search
    pub fn n_gamma_candidates(mut self, n: usize) -> Self {
        self.n_gamma_candidates = n;
        self
    }

    /// Set the number of cross-validation folds
    pub fn cv_folds(mut self, folds: usize) -> Self {
        self.cv_folds = folds;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set optimization tolerance
    pub fn tolerance(mut self, tol: Float) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set maximum optimization iterations
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Select optimal gamma based on the chosen strategy
    fn select_gamma(&self, x: &Array2<Float>) -> Result<Float> {
        match self.strategy {
            BandwidthSelectionStrategy::MedianHeuristic => self.median_heuristic_gamma(x),
            BandwidthSelectionStrategy::ScottRule => self.scott_rule_gamma(x),
            BandwidthSelectionStrategy::SilvermanRule => self.silverman_rule_gamma(x),
            BandwidthSelectionStrategy::CrossValidation => self.cross_validation_gamma(x),
            BandwidthSelectionStrategy::MaximumLikelihood => self.maximum_likelihood_gamma(x),
            BandwidthSelectionStrategy::LeaveOneOut => self.leave_one_out_gamma(x),
            BandwidthSelectionStrategy::GridSearch => self.grid_search_gamma(x),
        }
    }

    /// Median heuristic: gamma = 1/(2 * median_distance²)
    fn median_heuristic_gamma(&self, x: &Array2<Float>) -> Result<Float> {
        let (n_samples, _) = x.dim();

        if n_samples < 2 {
            return Ok(1.0); // Default gamma for insufficient data
        }

        // Compute pairwise squared distances
        let n_pairs = if n_samples > 1000 {
            // Subsample for large datasets to avoid O(n²) complexity
            1000
        } else {
            n_samples * (n_samples - 1) / 2
        };

        let mut distances_sq = Vec::with_capacity(n_pairs);
        let step = if n_samples > 1000 { n_samples / 100 } else { 1 };

        for i in (0..n_samples).step_by(step) {
            for j in ((i + 1)..n_samples).step_by(step) {
                if distances_sq.len() >= n_pairs {
                    break;
                }
                let diff = &x.row(i) - &x.row(j);
                let dist_sq = diff.mapv(|v| v * v).sum();
                distances_sq.push(dist_sq);
            }
            if distances_sq.len() >= n_pairs {
                break;
            }
        }

        if distances_sq.is_empty() {
            return Ok(1.0);
        }

        // Find median
        distances_sq.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_dist_sq = distances_sq[distances_sq.len() / 2];

        // gamma = 1 / (2 * sigma²), where sigma² ≈ median_distance²
        Ok(if median_dist_sq > 0.0 {
            1.0 / (2.0 * median_dist_sq)
        } else {
            1.0
        })
    }

    /// Scott's rule: sigma = n^(-1/(d+4))
    fn scott_rule_gamma(&self, x: &Array2<Float>) -> Result<Float> {
        let (n_samples, n_features) = x.dim();
        let sigma = (n_samples as Float).powf(-1.0 / (n_features as Float + 4.0));
        Ok(1.0 / (2.0 * sigma * sigma))
    }

    /// Silverman's rule of thumb
    fn silverman_rule_gamma(&self, x: &Array2<Float>) -> Result<Float> {
        let (n_samples, n_features) = x.dim();

        // Compute standard deviations for each dimension
        let means = x.mean_axis(Axis(0)).unwrap();
        let mut stds = Array1::zeros(n_features);

        for j in 0..n_features {
            let var = x
                .column(j)
                .mapv(|v| {
                    let diff = v - means[j];
                    diff * diff
                })
                .mean()
                .unwrap();
            stds[j] = var.sqrt();
        }

        let avg_std = stds.mean().unwrap();
        let h = 1.06 * avg_std * (n_samples as Float).powf(-1.0 / 5.0);

        Ok(1.0 / (2.0 * h * h))
    }

    /// Cross-validation based gamma selection
    fn cross_validation_gamma(&self, x: &Array2<Float>) -> Result<Float> {
        let gamma_candidates = self.generate_gamma_candidates()?;
        let mut best_gamma = gamma_candidates[0];
        let mut best_score = Float::INFINITY;

        for &gamma in &gamma_candidates {
            let score = self.cross_validation_score(x, gamma)?;
            if score < best_score {
                best_score = score;
                best_gamma = gamma;
            }
        }

        Ok(best_gamma)
    }

    /// Maximum likelihood gamma selection
    fn maximum_likelihood_gamma(&self, x: &Array2<Float>) -> Result<Float> {
        let gamma_candidates = self.generate_gamma_candidates()?;
        let mut best_gamma = gamma_candidates[0];
        let mut best_likelihood = Float::NEG_INFINITY;

        for &gamma in &gamma_candidates {
            let likelihood = self.log_likelihood(x, gamma)?;
            if likelihood > best_likelihood {
                best_likelihood = likelihood;
                best_gamma = gamma;
            }
        }

        Ok(best_gamma)
    }

    /// Leave-one-out cross-validation gamma selection
    fn leave_one_out_gamma(&self, x: &Array2<Float>) -> Result<Float> {
        let gamma_candidates = self.generate_gamma_candidates()?;
        let mut best_gamma = gamma_candidates[0];
        let mut best_score = Float::INFINITY;

        for &gamma in &gamma_candidates {
            let score = self.leave_one_out_score(x, gamma)?;
            if score < best_score {
                best_score = score;
                best_gamma = gamma;
            }
        }

        Ok(best_gamma)
    }

    /// Grid search gamma selection
    fn grid_search_gamma(&self, x: &Array2<Float>) -> Result<Float> {
        let gamma_candidates = self.generate_gamma_candidates()?;
        let mut best_gamma = gamma_candidates[0];
        let mut best_score = match self.objective_function {
            ObjectiveFunction::LogLikelihood => Float::NEG_INFINITY,
            _ => Float::INFINITY,
        };

        for &gamma in &gamma_candidates {
            let score = self.evaluate_objective(x, gamma)?;
            let is_better = match self.objective_function {
                ObjectiveFunction::LogLikelihood => score > best_score,
                _ => score < best_score,
            };

            if is_better {
                best_score = score;
                best_gamma = gamma;
            }
        }

        Ok(best_gamma)
    }

    /// Generate candidates for gamma search
    fn generate_gamma_candidates(&self) -> Result<Vec<Float>> {
        let (gamma_min, gamma_max) = self.gamma_range;
        let log_min = gamma_min.ln();
        let log_max = gamma_max.ln();

        let mut candidates = Vec::with_capacity(self.n_gamma_candidates);
        for i in 0..self.n_gamma_candidates {
            let t = i as Float / (self.n_gamma_candidates - 1) as Float;
            let log_gamma = log_min + t * (log_max - log_min);
            candidates.push(log_gamma.exp());
        }

        Ok(candidates)
    }

    /// Cross-validation score for a given gamma
    fn cross_validation_score(&self, x: &Array2<Float>, gamma: Float) -> Result<Float> {
        let (n_samples, _) = x.dim();
        let fold_size = n_samples / self.cv_folds;
        let mut total_error = 0.0;

        for fold in 0..self.cv_folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == self.cv_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            // Create train/validation split
            let val_indices: Vec<usize> = (start_idx..end_idx).collect();
            let train_indices: Vec<usize> = (0..start_idx).chain(end_idx..n_samples).collect();

            if train_indices.is_empty() || val_indices.is_empty() {
                continue;
            }

            // Evaluate kernel approximation quality
            let error = self.kernel_approximation_error(x, gamma, &train_indices, &val_indices)?;
            total_error += error;
        }

        Ok(total_error / self.cv_folds as Float)
    }

    /// Log-likelihood for Gaussian process with RBF kernel
    fn log_likelihood(&self, x: &Array2<Float>, gamma: Float) -> Result<Float> {
        let (n_samples, _) = x.dim();

        // Build kernel matrix (simplified version for efficiency)
        let mut k_matrix = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in i..n_samples {
                let diff = &x.row(i) - &x.row(j);
                let dist_sq = diff.mapv(|v| v * v).sum();
                let k_val = (-gamma * dist_sq).exp();
                k_matrix[[i, j]] = k_val;
                if i != j {
                    k_matrix[[j, i]] = k_val;
                }
            }
        }

        // Add noise term for numerical stability
        for i in 0..n_samples {
            k_matrix[[i, i]] += 1e-6;
        }

        // Simplified log-likelihood (without full matrix decomposition for efficiency)
        let trace = k_matrix.diag().sum();
        let det_approx = trace; // Rough approximation

        Ok(-0.5 * det_approx.ln() - 0.5 * n_samples as Float)
    }

    /// Leave-one-out cross-validation score
    fn leave_one_out_score(&self, x: &Array2<Float>, gamma: Float) -> Result<Float> {
        let (n_samples, _) = x.dim();
        let mut total_error = 0.0;

        for i in 0..n_samples {
            let train_indices: Vec<usize> = (0..n_samples).filter(|&j| j != i).collect();
            let val_indices = vec![i];

            let error = self.kernel_approximation_error(x, gamma, &train_indices, &val_indices)?;
            total_error += error;
        }

        Ok(total_error / n_samples as Float)
    }

    /// Evaluate objective function
    fn evaluate_objective(&self, x: &Array2<Float>, gamma: Float) -> Result<Float> {
        match self.objective_function {
            ObjectiveFunction::KernelAlignment => self.kernel_alignment(x, gamma),
            ObjectiveFunction::LogLikelihood => self.log_likelihood(x, gamma),
            ObjectiveFunction::CrossValidationError => self.cross_validation_score(x, gamma),
            ObjectiveFunction::KernelTrace => self.kernel_trace(x, gamma),
            ObjectiveFunction::EffectiveDimensionality => self.effective_dimensionality(x, gamma),
        }
    }

    /// Kernel alignment objective
    fn kernel_alignment(&self, x: &Array2<Float>, gamma: Float) -> Result<Float> {
        let (n_samples, _) = x.dim();

        // Simplified kernel alignment computation
        let mut alignment = 0.0;
        let mut count = 0;

        for i in 0..n_samples.min(100) {
            // Limit for efficiency
            for j in (i + 1)..n_samples.min(100) {
                let diff = &x.row(i) - &x.row(j);
                let dist_sq = diff.mapv(|v| v * v).sum();
                let k_val = (-gamma * dist_sq).exp();
                alignment += k_val * k_val; // Self-alignment
                count += 1;
            }
        }

        Ok(if count > 0 {
            -alignment / count as Float
        } else {
            0.0
        })
    }

    /// Kernel trace objective
    fn kernel_trace(&self, x: &Array2<Float>, _gamma: Float) -> Result<Float> {
        let (n_samples, _) = x.dim();
        let trace = n_samples as Float; // All diagonal elements are 1.0 for RBF kernel
        Ok(-trace) // Negative because we typically minimize
    }

    /// Effective dimensionality objective
    fn effective_dimensionality(&self, x: &Array2<Float>, gamma: Float) -> Result<Float> {
        // Simplified effective dimensionality based on kernel scale
        let characteristic_length = (1.0 / gamma).sqrt();
        let (_, n_features) = x.dim();
        let eff_dim = (characteristic_length * n_features as Float).min(n_features as Float);
        Ok(-eff_dim) // Negative for minimization
    }

    /// Kernel approximation error for validation
    fn kernel_approximation_error(
        &self,
        x: &Array2<Float>,
        gamma: Float,
        train_indices: &[usize],
        val_indices: &[usize],
    ) -> Result<Float> {
        if train_indices.is_empty() || val_indices.is_empty() {
            return Ok(0.0);
        }

        // Simplified approximation quality metric
        let mut error = 0.0;
        let mut count = 0;

        for &i in val_indices {
            for &j in train_indices {
                let diff = &x.row(i) - &x.row(j);
                let dist_sq = diff.mapv(|v| v * v).sum();
                let true_kernel = (-gamma * dist_sq).exp();

                // Simulate RFF approximation error (simplified)
                let approx_error = (1.0 - true_kernel) * (1.0 - true_kernel);
                error += approx_error;
                count += 1;
            }
        }

        Ok(if count > 0 {
            error / count as Float
        } else {
            0.0
        })
    }
}

impl Fit<Array2<Float>, ()> for AdaptiveBandwidthRBFSampler<Untrained> {
    type Fitted = AdaptiveBandwidthRBFSampler<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array is empty".to_string(),
            ));
        }

        // Select optimal gamma
        let selected_gamma = self.select_gamma(x)?;

        let mut rng = match self.random_state {
            Some(seed) => RealStdRng::seed_from_u64(seed),
            None => RealStdRng::from_seed(thread_rng().gen()),
        };

        // Generate random weights ~ N(0, 2*gamma*I)
        let std_dev = (2.0 * selected_gamma).sqrt();
        let mut random_weights = Array2::zeros((self.n_components, n_features));
        for i in 0..self.n_components {
            for j in 0..n_features {
                // Use Box-Muller transformation for normal distribution
                let u1 = rng.gen::<Float>();
                let u2 = rng.gen::<Float>();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                random_weights[[i, j]] = z * std_dev;
            }
        }

        // Generate random offsets ~ Uniform[0, 2π]
        let mut random_offset = Array1::zeros(self.n_components);
        for i in 0..self.n_components {
            random_offset[i] = rng.gen::<Float>() * 2.0 * std::f64::consts::PI;
        }

        Ok(AdaptiveBandwidthRBFSampler {
            n_components: self.n_components,
            strategy: self.strategy,
            objective_function: self.objective_function,
            gamma_range: self.gamma_range,
            n_gamma_candidates: self.n_gamma_candidates,
            cv_folds: self.cv_folds,
            random_state: self.random_state,
            tolerance: self.tolerance,
            max_iterations: self.max_iterations,
            selected_gamma_: Some(selected_gamma),
            random_weights_: Some(random_weights),
            random_offset_: Some(random_offset),
            optimization_history_: None, // Could be populated during optimization
            _state: PhantomData,
        })
    }
}

impl AdaptiveBandwidthRBFSampler<Trained> {
    /// Get the selected gamma value
    pub fn selected_gamma(&self) -> Result<Float> {
        self.selected_gamma_.ok_or_else(|| SklearsError::NotFitted {
            operation: "selected_gamma".to_string(),
        })
    }

    /// Get the optimization history (if available)
    pub fn optimization_history(&self) -> Option<&Vec<(Float, Float)>> {
        self.optimization_history_.as_ref()
    }
}

impl Transform<Array2<Float>> for AdaptiveBandwidthRBFSampler<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let random_weights =
            self.random_weights_
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "transform".to_string(),
                })?;

        let random_offset =
            self.random_offset_
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "transform".to_string(),
                })?;

        let (_n_samples, n_features) = x.dim();

        if n_features != random_weights.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "Input has {} features, expected {}",
                n_features,
                random_weights.ncols()
            )));
        }

        // Compute X @ W.T + b
        let projection = x.dot(&random_weights.t()) + random_offset;

        // Apply cosine transformation and normalize
        let normalization = (2.0 / random_weights.nrows() as Float).sqrt();
        Ok(projection.mapv(|x| x.cos() * normalization))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_adaptive_bandwidth_rbf_sampler_basic() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let sampler = AdaptiveBandwidthRBFSampler::new(50)
            .strategy(BandwidthSelectionStrategy::MedianHeuristic)
            .random_state(42);

        let fitted = sampler.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[3, 50]);

        // Check that gamma was selected
        let gamma = fitted.selected_gamma().unwrap();
        assert!(gamma > 0.0);

        // Check that features are bounded (cosine function)
        for &val in features.iter() {
            assert!(val >= -2.0 && val <= 2.0);
        }
    }

    #[test]
    fn test_different_bandwidth_strategies() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let strategies = [
            BandwidthSelectionStrategy::MedianHeuristic,
            BandwidthSelectionStrategy::ScottRule,
            BandwidthSelectionStrategy::SilvermanRule,
            BandwidthSelectionStrategy::GridSearch,
        ];

        for strategy in &strategies {
            let sampler = AdaptiveBandwidthRBFSampler::new(20)
                .strategy(*strategy)
                .random_state(42);

            let fitted = sampler.fit(&x, &()).unwrap();
            let features = fitted.transform(&x).unwrap();
            let gamma = fitted.selected_gamma().unwrap();

            assert_eq!(features.shape(), &[4, 20]);
            assert!(gamma > 0.0);
        }
    }

    #[test]
    fn test_cross_validation_strategy() {
        let x = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [2.0, 2.0],
            [2.1, 2.1],
            [5.0, 5.0],
            [5.1, 5.1]
        ];

        let sampler = AdaptiveBandwidthRBFSampler::new(30)
            .strategy(BandwidthSelectionStrategy::CrossValidation)
            .cv_folds(3)
            .n_gamma_candidates(5)
            .random_state(42);

        let fitted = sampler.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();
        let gamma = fitted.selected_gamma().unwrap();

        assert_eq!(features.shape(), &[6, 30]);
        assert!(gamma > 0.0);
    }

    #[test]
    fn test_different_objective_functions() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let objectives = [
            ObjectiveFunction::KernelAlignment,
            ObjectiveFunction::LogLikelihood,
            ObjectiveFunction::KernelTrace,
            ObjectiveFunction::EffectiveDimensionality,
        ];

        for objective in &objectives {
            let sampler = AdaptiveBandwidthRBFSampler::new(25)
                .strategy(BandwidthSelectionStrategy::GridSearch)
                .objective_function(*objective)
                .n_gamma_candidates(5)
                .random_state(42);

            let fitted = sampler.fit(&x, &()).unwrap();
            let gamma = fitted.selected_gamma().unwrap();

            assert!(gamma > 0.0);
        }
    }

    #[test]
    fn test_median_heuristic() {
        let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

        let sampler = AdaptiveBandwidthRBFSampler::new(10);
        let gamma = sampler.median_heuristic_gamma(&x).unwrap();

        // With unit distances, median distance² ≈ 1, so gamma ≈ 0.5
        assert!(gamma > 0.1 && gamma < 2.0);
    }

    #[test]
    fn test_scott_rule() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let sampler = AdaptiveBandwidthRBFSampler::new(10);
        let gamma = sampler.scott_rule_gamma(&x).unwrap();

        assert!(gamma > 0.0);
    }

    #[test]
    fn test_silverman_rule() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let sampler = AdaptiveBandwidthRBFSampler::new(10);
        let gamma = sampler.silverman_rule_gamma(&x).unwrap();

        assert!(gamma > 0.0);
    }

    #[test]
    fn test_reproducibility() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let sampler1 = AdaptiveBandwidthRBFSampler::new(40)
            .strategy(BandwidthSelectionStrategy::MedianHeuristic)
            .random_state(123);

        let sampler2 = AdaptiveBandwidthRBFSampler::new(40)
            .strategy(BandwidthSelectionStrategy::MedianHeuristic)
            .random_state(123);

        let fitted1 = sampler1.fit(&x, &()).unwrap();
        let fitted2 = sampler2.fit(&x, &()).unwrap();

        let features1 = fitted1.transform(&x).unwrap();
        let features2 = fitted2.transform(&x).unwrap();

        let gamma1 = fitted1.selected_gamma().unwrap();
        let gamma2 = fitted2.selected_gamma().unwrap();

        assert_abs_diff_eq!(gamma1, gamma2, epsilon = 1e-10);

        for (f1, f2) in features1.iter().zip(features2.iter()) {
            assert_abs_diff_eq!(f1, f2, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_gamma_range() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let sampler = AdaptiveBandwidthRBFSampler::new(15)
            .strategy(BandwidthSelectionStrategy::GridSearch)
            .gamma_range(0.5, 2.0)
            .n_gamma_candidates(5)
            .random_state(42);

        let fitted = sampler.fit(&x, &()).unwrap();
        let gamma = fitted.selected_gamma().unwrap();

        // Selected gamma should be within the specified range
        assert!(gamma >= 0.5 && gamma <= 2.0);
    }

    #[test]
    fn test_error_handling() {
        // Empty input
        let empty = Array2::<Float>::zeros((0, 0));
        let sampler = AdaptiveBandwidthRBFSampler::new(10);
        assert!(sampler.clone().fit(&empty, &()).is_err());

        // Dimension mismatch in transform
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let x_test = array![[1.0, 2.0, 3.0]]; // Wrong number of features

        let fitted = sampler.fit(&x_train, &()).unwrap();
        assert!(fitted.transform(&x_test).is_err());
    }

    #[test]
    fn test_single_sample() {
        let x = array![[1.0, 2.0]];

        let sampler = AdaptiveBandwidthRBFSampler::new(10)
            .strategy(BandwidthSelectionStrategy::MedianHeuristic);

        let fitted = sampler.fit(&x, &()).unwrap();
        let gamma = fitted.selected_gamma().unwrap();

        // Should use default gamma for single sample
        assert!(gamma > 0.0);
    }

    #[test]
    fn test_large_dataset_efficiency() {
        // Test that median heuristic works efficiently on larger datasets
        let mut data = Vec::new();
        for i in 0..500 {
            data.push([i as Float, (i * 2) as Float]);
        }
        let x = Array2::from(data);

        let sampler = AdaptiveBandwidthRBFSampler::new(20)
            .strategy(BandwidthSelectionStrategy::MedianHeuristic);

        let fitted = sampler.fit(&x, &()).unwrap();
        let gamma = fitted.selected_gamma().unwrap();

        assert!(gamma > 0.0);
    }
}
