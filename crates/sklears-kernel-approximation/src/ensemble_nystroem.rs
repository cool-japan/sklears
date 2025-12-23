//! Ensemble Nyström method for improved kernel approximation
use crate::nystroem::{Kernel, Nystroem, SamplingStrategy};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::rngs::StdRng as RealStdRng;
use scirs2_core::random::Rng;
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

use scirs2_core::random::{thread_rng, SeedableRng};
/// Ensemble method for combining multiple Nyström approximations
#[derive(Debug, Clone)]
pub enum EnsembleMethod {
    /// Simple averaging of all approximations
    Average,
    /// Weighted average based on approximation quality (higher quality gets more weight)
    WeightedAverage,
    /// Concatenate all approximations
    Concatenate,
    /// Use the best approximation based on some quality metric
    BestApproximation,
}

/// Quality metric for evaluating Nyström approximations
#[derive(Debug, Clone)]
pub enum QualityMetric {
    /// Frobenius norm of the approximation error
    FrobeniusNorm,
    /// Trace of the approximation
    Trace,
    /// Spectral norm (largest eigenvalue)
    SpectralNorm,
    /// Nuclear norm (sum of eigenvalues)
    NuclearNorm,
}

/// Ensemble Nyström method for kernel approximation
///
/// Combines multiple Nyström approximations using different sampling strategies
/// and component sizes to achieve better approximation quality than a single
/// Nyström approximation.
///
/// # Parameters
///
/// * `kernel` - Kernel function to approximate
/// * `n_estimators` - Number of base Nyström estimators (default: 5)
/// * `n_components` - Number of samples per estimator (default: 100)
/// * `ensemble_method` - Method for combining estimators
/// * `sampling_strategies` - List of sampling strategies to use
/// * `random_state` - Random seed for reproducibility
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_kernel_approximation::ensemble_nystroem::{EnsembleNystroem, EnsembleMethod};
/// use sklears_kernel_approximation::nystroem::{Kernel, SamplingStrategy};
/// use sklears_core::traits::{Transform, Fit, Untrained}
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
///
/// let ensemble = EnsembleNystroem::new(Kernel::Rbf { gamma: 1.0 }, 3, 2)
///     .ensemble_method(EnsembleMethod::WeightedAverage);
/// let fitted_ensemble = ensemble.fit(&X, &()).unwrap();
/// let X_transformed = fitted_ensemble.transform(&X).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct EnsembleNystroem<State = Untrained> {
    /// Kernel function
    pub kernel: Kernel,
    /// Number of base estimators
    pub n_estimators: usize,
    /// Number of components per estimator
    pub n_components: usize,
    /// Method for ensemble combination
    pub ensemble_method: EnsembleMethod,
    /// Sampling strategies to use (if None, uses diverse set)
    pub sampling_strategies: Option<Vec<SamplingStrategy>>,
    /// Quality metric for evaluating approximations
    pub quality_metric: QualityMetric,
    /// Random seed
    pub random_state: Option<u64>,

    // Fitted attributes
    estimators_: Option<Vec<Nystroem<Trained>>>,
    weights_: Option<Vec<Float>>,
    n_features_out_: Option<usize>,

    _state: PhantomData<State>,
}

impl EnsembleNystroem<Untrained> {
    pub fn new(kernel: Kernel, n_estimators: usize, n_components: usize) -> Self {
        Self {
            kernel,
            n_estimators,
            n_components,
            ensemble_method: EnsembleMethod::WeightedAverage,
            sampling_strategies: None,
            quality_metric: QualityMetric::FrobeniusNorm,
            random_state: None,
            estimators_: None,
            weights_: None,
            n_features_out_: None,
            _state: PhantomData,
        }
    }

    /// Set the ensemble method
    pub fn ensemble_method(mut self, method: EnsembleMethod) -> Self {
        self.ensemble_method = method;
        self
    }

    /// Set custom sampling strategies
    pub fn sampling_strategies(mut self, strategies: Vec<SamplingStrategy>) -> Self {
        self.sampling_strategies = Some(strategies);
        self
    }

    /// Set the quality metric for evaluation
    pub fn quality_metric(mut self, metric: QualityMetric) -> Self {
        self.quality_metric = metric;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Generate diverse sampling strategies
    fn generate_sampling_strategies(&self) -> Vec<SamplingStrategy> {
        if let Some(ref strategies) = self.sampling_strategies {
            strategies.clone()
        } else {
            // Generate diverse set of strategies
            let mut strategies = Vec::new();
            let base_strategies = [
                SamplingStrategy::Random,
                SamplingStrategy::KMeans,
                SamplingStrategy::LeverageScore,
                SamplingStrategy::ColumnNorm,
            ];

            for i in 0..self.n_estimators {
                strategies.push(base_strategies[i % base_strategies.len()].clone());
            }
            strategies
        }
    }

    /// Compute quality score for a Nyström approximation
    fn compute_quality_score(
        &self,
        estimator: &Nystroem<Trained>,
        _x: &Array2<Float>,
    ) -> Result<Float> {
        match self.quality_metric {
            QualityMetric::FrobeniusNorm => {
                // Approximate quality based on component matrix properties
                let components = estimator.components();
                let norm = components.dot(&components.t()).mapv(|v| v * v).sum().sqrt();
                Ok(norm)
            }
            QualityMetric::Trace => {
                let components = estimator.components();
                let kernel_matrix = self.kernel.compute_kernel(components, components);
                Ok(kernel_matrix.diag().sum())
            }
            QualityMetric::SpectralNorm => {
                // Approximate spectral norm using power iteration
                let components = estimator.components();
                let kernel_matrix = self.kernel.compute_kernel(components, components);
                self.power_iteration_spectral_norm(&kernel_matrix)
            }
            QualityMetric::NuclearNorm => {
                let components = estimator.components();
                let kernel_matrix = self.kernel.compute_kernel(components, components);
                // Nuclear norm is sum of eigenvalues, approximate with trace
                Ok(kernel_matrix.diag().sum())
            }
        }
    }

    /// Approximate spectral norm using power iteration
    fn power_iteration_spectral_norm(&self, matrix: &Array2<Float>) -> Result<Float> {
        let n = matrix.nrows();
        if n == 0 {
            return Ok(0.0);
        }

        let mut v = Array1::ones(n) / (n as Float).sqrt();
        let max_iter = 100;
        let tolerance = 1e-6;

        for _ in 0..max_iter {
            let v_new = matrix.dot(&v);
            let norm = (v_new.dot(&v_new)).sqrt();

            if norm < tolerance {
                break;
            }

            let v_normalized = &v_new / norm;
            let diff = (&v_normalized - &v).dot(&(&v_normalized - &v)).sqrt();
            v = v_normalized;

            if diff < tolerance {
                break;
            }
        }

        let eigenvalue = v.dot(&matrix.dot(&v));
        Ok(eigenvalue.abs())
    }
}

impl Estimator for EnsembleNystroem<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, ()> for EnsembleNystroem<Untrained> {
    type Fitted = EnsembleNystroem<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        if self.n_estimators == 0 {
            return Err(SklearsError::InvalidInput(
                "n_estimators must be positive".to_string(),
            ));
        }

        if self.n_components == 0 {
            return Err(SklearsError::InvalidInput(
                "n_components must be positive".to_string(),
            ));
        }

        let mut rng = if let Some(seed) = self.random_state {
            RealStdRng::seed_from_u64(seed)
        } else {
            RealStdRng::from_seed(thread_rng().gen())
        };

        let sampling_strategies = self.generate_sampling_strategies();
        let mut estimators = Vec::new();
        let mut quality_scores = Vec::new();

        // Train base estimators
        for i in 0..self.n_estimators {
            let strategy = sampling_strategies[i % sampling_strategies.len()].clone();
            let seed = if self.random_state.is_some() {
                // Use deterministic seed sequence for reproducibility
                self.random_state.unwrap().wrapping_add(i as u64)
            } else {
                rng.gen::<u64>()
            };

            let nystroem = Nystroem::new(self.kernel.clone(), self.n_components)
                .sampling_strategy(strategy)
                .random_state(seed);

            let fitted_nystroem = nystroem.fit(x, &())?;

            // Compute quality score
            let quality = self.compute_quality_score(&fitted_nystroem, x)?;
            quality_scores.push(quality);
            estimators.push(fitted_nystroem);
        }

        // Compute weights based on ensemble method
        let weights = match self.ensemble_method {
            EnsembleMethod::Average => vec![1.0 / self.n_estimators as Float; self.n_estimators],
            EnsembleMethod::WeightedAverage => {
                let total_quality: Float = quality_scores.iter().sum();
                if total_quality > 0.0 {
                    quality_scores.iter().map(|&q| q / total_quality).collect()
                } else {
                    vec![1.0 / self.n_estimators as Float; self.n_estimators]
                }
            }
            EnsembleMethod::Concatenate => vec![1.0; self.n_estimators],
            EnsembleMethod::BestApproximation => {
                let best_idx = quality_scores
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                let mut weights = vec![0.0; self.n_estimators];
                weights[best_idx] = 1.0;
                weights
            }
        };

        // Determine output feature size
        let n_features_out = match self.ensemble_method {
            EnsembleMethod::Concatenate => self.n_estimators * self.n_components,
            _ => self.n_components,
        };

        Ok(EnsembleNystroem {
            kernel: self.kernel,
            n_estimators: self.n_estimators,
            n_components: self.n_components,
            ensemble_method: self.ensemble_method,
            sampling_strategies: self.sampling_strategies,
            quality_metric: self.quality_metric,
            random_state: self.random_state,
            estimators_: Some(estimators),
            weights_: Some(weights),
            n_features_out_: Some(n_features_out),
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for EnsembleNystroem<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let estimators = self.estimators_.as_ref().unwrap();
        let weights = self.weights_.as_ref().unwrap();
        let n_features_out = self.n_features_out_.unwrap();
        let (n_samples, _) = x.dim();

        match self.ensemble_method {
            EnsembleMethod::Average | EnsembleMethod::WeightedAverage => {
                let mut result = Array2::zeros((n_samples, self.n_components));

                for (estimator, &weight) in estimators.iter().zip(weights.iter()) {
                    if weight > 0.0 {
                        let transformed = estimator.transform(x)?;
                        result += &(transformed * weight);
                    }
                }

                Ok(result)
            }
            EnsembleMethod::Concatenate => {
                let mut result = Array2::zeros((n_samples, n_features_out));
                let mut col_offset = 0;

                for estimator in estimators.iter() {
                    let transformed = estimator.transform(x)?;
                    let n_cols = transformed.ncols();
                    result
                        .slice_mut(s![.., col_offset..col_offset + n_cols])
                        .assign(&transformed);
                    col_offset += n_cols;
                }

                Ok(result)
            }
            EnsembleMethod::BestApproximation => {
                let best_idx = weights
                    .iter()
                    .enumerate()
                    .find(|(_, &w)| w > 0.0)
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                estimators[best_idx].transform(x)
            }
        }
    }
}

impl EnsembleNystroem<Trained> {
    /// Get the base estimators
    pub fn estimators(&self) -> &[Nystroem<Trained>] {
        self.estimators_.as_ref().unwrap()
    }

    /// Get the estimator weights
    pub fn weights(&self) -> &[Float] {
        self.weights_.as_ref().unwrap()
    }

    /// Get the number of output features
    pub fn n_features_out(&self) -> usize {
        self.n_features_out_.unwrap()
    }

    /// Get quality scores for all estimators
    pub fn quality_scores(&self, x: &Array2<Float>) -> Result<Vec<Float>> {
        let estimators = self.estimators_.as_ref().unwrap();
        let mut scores = Vec::new();

        for estimator in estimators.iter() {
            let score = self.compute_quality_score_for_estimator(estimator, x)?;
            scores.push(score);
        }

        Ok(scores)
    }

    /// Compute quality score for a single estimator (helper method)
    fn compute_quality_score_for_estimator(
        &self,
        estimator: &Nystroem<Trained>,
        _x: &Array2<Float>,
    ) -> Result<Float> {
        match self.quality_metric {
            QualityMetric::FrobeniusNorm => {
                let components = estimator.components();
                let norm = components.dot(&components.t()).mapv(|v| v * v).sum().sqrt();
                Ok(norm)
            }
            QualityMetric::Trace => {
                let components = estimator.components();
                let kernel_matrix = self.kernel.compute_kernel(components, components);
                Ok(kernel_matrix.diag().sum())
            }
            QualityMetric::SpectralNorm => {
                let components = estimator.components();
                let kernel_matrix = self.kernel.compute_kernel(components, components);
                self.power_iteration_spectral_norm(&kernel_matrix)
            }
            QualityMetric::NuclearNorm => {
                let components = estimator.components();
                let kernel_matrix = self.kernel.compute_kernel(components, components);
                Ok(kernel_matrix.diag().sum())
            }
        }
    }

    /// Approximate spectral norm using power iteration
    fn power_iteration_spectral_norm(&self, matrix: &Array2<Float>) -> Result<Float> {
        let n = matrix.nrows();
        if n == 0 {
            return Ok(0.0);
        }

        let mut v = Array1::ones(n) / (n as Float).sqrt();
        let max_iter = 100;
        let tolerance = 1e-6;

        for _ in 0..max_iter {
            let v_new = matrix.dot(&v);
            let norm = (v_new.dot(&v_new)).sqrt();

            if norm < tolerance {
                break;
            }

            let v_normalized = &v_new / norm;
            let diff = (&v_normalized - &v).dot(&(&v_normalized - &v)).sqrt();
            v = v_normalized;

            if diff < tolerance {
                break;
            }
        }

        let eigenvalue = v.dot(&matrix.dot(&v));
        Ok(eigenvalue.abs())
    }
}

// Add ndarray slice import
use scirs2_core::ndarray::s;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_ensemble_nystroem_basic() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],];

        let ensemble = EnsembleNystroem::new(Kernel::Linear, 3, 2);
        let fitted = ensemble.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.nrows(), 4);
        assert_eq!(x_transformed.ncols(), 2); // n_components
    }

    #[test]
    fn test_ensemble_nystroem_average() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],];

        let ensemble = EnsembleNystroem::new(Kernel::Rbf { gamma: 0.1 }, 2, 3)
            .ensemble_method(EnsembleMethod::Average);
        let fitted = ensemble.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.shape(), &[3, 3]);
    }

    #[test]
    fn test_ensemble_nystroem_concatenate() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],];

        let ensemble = EnsembleNystroem::new(Kernel::Linear, 2, 3)
            .ensemble_method(EnsembleMethod::Concatenate);
        let fitted = ensemble.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.shape(), &[3, 6]); // 2 estimators * 3 components = 6
    }

    #[test]
    fn test_ensemble_nystroem_weighted_average() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],];

        let ensemble = EnsembleNystroem::new(Kernel::Rbf { gamma: 0.5 }, 3, 2)
            .ensemble_method(EnsembleMethod::WeightedAverage);
        let fitted = ensemble.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.shape(), &[4, 2]);

        // Check that weights sum to 1 (approximately)
        let weights = fitted.weights();
        let weight_sum: Float = weights.iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ensemble_nystroem_best_approximation() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],];

        let ensemble = EnsembleNystroem::new(Kernel::Linear, 3, 2)
            .ensemble_method(EnsembleMethod::BestApproximation);
        let fitted = ensemble.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.shape(), &[3, 2]);

        // Check that exactly one weight is 1.0 and others are 0.0
        let weights = fitted.weights();
        let active_weights: Vec<&Float> = weights.iter().filter(|&&w| w > 0.0).collect();
        assert_eq!(active_weights.len(), 1);
        assert!((active_weights[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ensemble_nystroem_custom_strategies() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],];

        let strategies = vec![SamplingStrategy::Random, SamplingStrategy::LeverageScore];

        let ensemble = EnsembleNystroem::new(Kernel::Linear, 2, 3).sampling_strategies(strategies);
        let fitted = ensemble.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.shape(), &[4, 3]);
        assert_eq!(fitted.estimators().len(), 2);
    }

    #[test]
    fn test_ensemble_nystroem_reproducibility() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],];

        let ensemble1 = EnsembleNystroem::new(Kernel::Linear, 2, 3).random_state(42);
        let fitted1 = ensemble1.fit(&x, &()).unwrap();
        let result1 = fitted1.transform(&x).unwrap();

        let ensemble2 = EnsembleNystroem::new(Kernel::Linear, 2, 3).random_state(42);
        let fitted2 = ensemble2.fit(&x, &()).unwrap();
        let result2 = fitted2.transform(&x).unwrap();

        // Results should be very similar with same random state (allowing for numerical precision)
        assert_eq!(result1.shape(), result2.shape());
        for (a, b) in result1.iter().zip(result2.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "Values differ too much: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_ensemble_nystroem_quality_metrics() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],];

        let ensemble = EnsembleNystroem::new(Kernel::Rbf { gamma: 0.1 }, 2, 2)
            .quality_metric(QualityMetric::Trace);
        let fitted = ensemble.fit(&x, &()).unwrap();
        let quality_scores = fitted.quality_scores(&x).unwrap();

        assert_eq!(quality_scores.len(), 2);
        for score in quality_scores.iter() {
            assert!(score.is_finite());
            assert!(*score >= 0.0);
        }
    }

    #[test]
    fn test_ensemble_nystroem_invalid_parameters() {
        let x = array![[1.0, 2.0]];

        // Zero estimators
        let ensemble = EnsembleNystroem::new(Kernel::Linear, 0, 2);
        assert!(ensemble.fit(&x, &()).is_err());

        // Zero components
        let ensemble = EnsembleNystroem::new(Kernel::Linear, 2, 0);
        assert!(ensemble.fit(&x, &()).is_err());
    }
}
