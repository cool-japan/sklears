//! Sparse spectrum Gaussian processes for large-scale approximation
//!
//! This module implements sparse spectrum Gaussian processes (SSGPs) which use
//! spectral approximation methods to scale to large datasets while maintaining
//! accurate uncertainty quantification.

use crate::kernels::Kernel;
// SciRS2 Policy - Use scirs2-autograd for ndarray types and operations
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
// SciRS2 Policy - Use scirs2-core for random number generation
use scirs2_core::random::Rng;
use sklears_core::error::{Result as SklResult, SklearsError};
use sklears_core::prelude::{Estimator, Fit, Predict};

/// Sparse Spectrum Gaussian Process Regressor
///
/// Uses spectral approximation to scale Gaussian processes to large datasets
/// by approximating the kernel using a sparse set of spectral points.
///
/// # Example
/// ```rust
/// use sklears_gaussian_process::{SparseSpectrumGaussianProcessRegressor, kernels::RBF};
/// use sklears_core::prelude::*;
/// // SciRS2 Policy - Use scirs2-autograd for ndarray types and operations
///
/// let kernel = Box::new(RBF::new(1.0));
/// let model = SparseSpectrumGaussianProcessRegressor::new(kernel)
///     .num_spectral_points(50)
///     .spectral_density_threshold(1e-6);
///
/// let X = Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();
/// let y = Array1::from_vec((0..10).map(|x| (x as f64).sin()).collect());
///
/// let trained_model = model.fit(&X.view(), &y.view()).unwrap();
/// let predictions = trained_model.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SparseSpectrumGaussianProcessRegressor {
    /// Base kernel to approximate
    pub kernel: Box<dyn Kernel>,
    /// Number of spectral points to use
    pub num_spectral_points: usize,
    /// Threshold for spectral density selection
    pub spectral_density_threshold: f64,
    /// Method for selecting spectral points
    pub selection_method: SpectralSelectionMethod,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
    /// Noise variance parameter
    pub noise_variance: f64,
    /// Whether to optimize spectral points during training
    pub optimize_spectral_points: bool,
    /// Learning rate for spectral point optimization
    pub spectral_learning_rate: f64,
    /// Maximum iterations for optimization
    pub max_optimization_iterations: usize,
}

/// Methods for selecting spectral points
#[derive(Debug, Clone, Copy)]
pub enum SpectralSelectionMethod {
    Random,
    Greedy,
    ImportanceSampling,
    QuasiRandom,
    Adaptive,
}

/// Trained sparse spectrum Gaussian process regressor
#[derive(Debug, Clone)]
pub struct SparseSpectrumGprTrained {
    /// Original configuration
    pub config: SparseSpectrumGaussianProcessRegressor,
    /// Selected spectral points (frequencies)
    pub spectral_points: Array2<f64>,
    /// Spectral weights
    pub spectral_weights: Array1<f64>,
    /// Training feature matrix in spectral space
    pub spectral_features: Array2<f64>,
    /// Posterior mean parameters
    pub posterior_mean: Array1<f64>,
    /// Posterior covariance matrix
    pub posterior_covariance: Array2<f64>,
    /// Training inputs (for prediction)
    pub X_train: Array2<f64>,
    /// Training targets
    pub y_train: Array1<f64>,
    /// Spectral density estimates
    pub spectral_density: Array1<f64>,
    /// Log marginal likelihood
    pub log_marginal_likelihood: f64,
}

/// Information about spectral approximation quality
#[derive(Debug, Clone)]
pub struct SpectralApproximationInfo {
    /// Effective rank of spectral approximation
    pub effective_rank: f64,
    /// Spectral coverage (fraction of spectrum captured)
    pub spectral_coverage: f64,
    /// Maximum approximation error estimate
    pub max_approximation_error: f64,
    /// Selected frequencies
    pub selected_frequencies: Array2<f64>,
    /// Spectral density at selected points
    pub spectral_densities: Array1<f64>,
}

impl Default for SparseSpectrumGaussianProcessRegressor {
    fn default() -> Self {
        // Default to RBF kernel
        let kernel = Box::new(crate::kernels::RBF::new(1.0));
        Self {
            kernel,
            num_spectral_points: 100,
            spectral_density_threshold: 1e-6,
            selection_method: SpectralSelectionMethod::Adaptive,
            random_state: Some(42),
            noise_variance: 1e-5,
            optimize_spectral_points: true,
            spectral_learning_rate: 0.01,
            max_optimization_iterations: 50,
        }
    }
}

impl SparseSpectrumGaussianProcessRegressor {
    /// Create a new sparse spectrum Gaussian process regressor
    pub fn new(kernel: Box<dyn Kernel>) -> Self {
        Self {
            kernel,
            ..Default::default()
        }
    }

    /// Set the number of spectral points
    pub fn num_spectral_points(mut self, num_points: usize) -> Self {
        self.num_spectral_points = num_points;
        self
    }

    /// Set the spectral density threshold
    pub fn spectral_density_threshold(mut self, threshold: f64) -> Self {
        self.spectral_density_threshold = threshold;
        self
    }

    /// Set the spectral selection method
    pub fn selection_method(mut self, method: SpectralSelectionMethod) -> Self {
        self.selection_method = method;
        self
    }

    /// Set the noise variance
    pub fn noise_variance(mut self, variance: f64) -> Self {
        self.noise_variance = variance;
        self
    }

    /// Set whether to optimize spectral points
    pub fn optimize_spectral_points(mut self, optimize: bool) -> Self {
        self.optimize_spectral_points = optimize;
        self
    }

    /// Set random state for reproducible results
    pub fn random_state(mut self, seed: Option<u64>) -> Self {
        self.random_state = seed;
        self
    }

    /// Estimate spectral density of the kernel
    fn estimate_spectral_density(
        &self,
        X: &ArrayView2<f64>,
        num_grid_points: usize,
    ) -> SklResult<(Array2<f64>, Array1<f64>)> {
        let n_features = X.ncols();

        // Create a grid of frequencies
        // SciRS2 Policy - Use scirs2-core for random number generation
        let mut rng = if let Some(seed) = self.random_state {
            scirs2_core::random::Random::seed(seed)
        } else {
            scirs2_core::random::Random::seed(42)
        };

        // Estimate reasonable frequency range from data
        let mut freq_ranges = Vec::new();
        for dim in 0..n_features {
            let column = X.column(dim);
            let range = column.fold(f64::NEG_INFINITY, |a, &b| a.max(b))
                - column.fold(f64::INFINITY, |a, &b| a.min(b));
            let max_freq = 2.0 / range.max(1e-6);
            freq_ranges.push((-max_freq, max_freq));
        }

        // Generate frequency grid
        let mut frequencies = Array2::zeros((num_grid_points, n_features));
        let mut spectral_densities = Array1::zeros(num_grid_points);

        for i in 0..num_grid_points {
            for dim in 0..n_features {
                let (min_freq, max_freq) = freq_ranges[dim];
                frequencies[[i, dim]] = rng.gen_range(min_freq..max_freq);
            }

            // Estimate spectral density at this frequency point
            spectral_densities[i] =
                self.estimate_spectral_density_at_frequency(&frequencies.row(i).to_owned(), X)?;
        }

        Ok((frequencies, spectral_densities))
    }

    /// Estimate spectral density at a specific frequency
    fn estimate_spectral_density_at_frequency(
        &self,
        frequency: &Array1<f64>,
        X: &ArrayView2<f64>,
    ) -> SklResult<f64> {
        // For RBF-like kernels, the spectral density follows a Gaussian distribution
        // For other kernels, we use a general Fourier transform approximation

        let n_samples = X.nrows().min(100); // Use subset for efficiency
        let mut density = 0.0;

        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let x_diff = &X.row(i) - &X.row(j);
                let phase = 2.0 * std::f64::consts::PI * frequency.dot(&x_diff);
                let kernel_value = self.kernel.kernel(&X.row(i), &X.row(j));
                density += kernel_value * phase.cos();
            }
        }

        let normalization = (n_samples * (n_samples - 1)) as f64 / 2.0;
        Ok((density / normalization).abs())
    }

    /// Select spectral points based on the selection method
    fn select_spectral_points(
        &self,
        frequencies: &Array2<f64>,
        spectral_densities: &Array1<f64>,
    ) -> SklResult<(Array2<f64>, Array1<f64>)> {
        // SciRS2 Policy - Use scirs2-core for random number generation
        let mut rng = if let Some(seed) = self.random_state {
            scirs2_core::random::Random::seed(seed)
        } else {
            scirs2_core::random::Random::seed(42)
        };

        match self.selection_method {
            SpectralSelectionMethod::Random => {
                self.random_selection(frequencies, spectral_densities, &mut rng)
            }
            SpectralSelectionMethod::Greedy => {
                self.greedy_selection(frequencies, spectral_densities)
            }
            SpectralSelectionMethod::ImportanceSampling => {
                self.importance_sampling_selection(frequencies, spectral_densities, &mut rng)
            }
            SpectralSelectionMethod::QuasiRandom => {
                self.quasi_random_selection(frequencies, spectral_densities, &mut rng)
            }
            SpectralSelectionMethod::Adaptive => {
                self.adaptive_selection(frequencies, spectral_densities, &mut rng)
            }
        }
    }

    /// Random selection of spectral points
    fn random_selection(
        &self,
        frequencies: &Array2<f64>,
        spectral_densities: &Array1<f64>,
        rng: &mut scirs2_core::random::Random<scirs2_core::rngs::StdRng>,
    ) -> SklResult<(Array2<f64>, Array1<f64>)> {
        let total_points = frequencies.nrows();
        let mut selected_indices = (0..total_points).collect::<Vec<_>>();
        // Simple shuffle using Fisher-Yates algorithm
        for i in (1..selected_indices.len()).rev() {
            let j = rng.gen_range(0..(i + 1));
            selected_indices.swap(i, j);
        }
        selected_indices.truncate(self.num_spectral_points.min(total_points));

        let selected_frequencies = frequencies.select(Axis(0), &selected_indices);
        let selected_weights = spectral_densities.select(Axis(0), &selected_indices);

        Ok((selected_frequencies, selected_weights))
    }

    /// Greedy selection based on spectral density
    fn greedy_selection(
        &self,
        frequencies: &Array2<f64>,
        spectral_densities: &Array1<f64>,
    ) -> SklResult<(Array2<f64>, Array1<f64>)> {
        let mut indices_with_densities: Vec<(usize, f64)> = spectral_densities
            .iter()
            .enumerate()
            .map(|(i, &density)| (i, density))
            .collect();

        // Sort by spectral density (descending)
        indices_with_densities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let selected_indices: Vec<usize> = indices_with_densities
            .into_iter()
            .take(self.num_spectral_points.min(frequencies.nrows()))
            .map(|(idx, _)| idx)
            .collect();

        let selected_frequencies = frequencies.select(Axis(0), &selected_indices);
        let selected_weights = spectral_densities.select(Axis(0), &selected_indices);

        Ok((selected_frequencies, selected_weights))
    }

    /// Importance sampling selection
    fn importance_sampling_selection(
        &self,
        frequencies: &Array2<f64>,
        spectral_densities: &Array1<f64>,
        rng: &mut scirs2_core::random::Random<scirs2_core::rngs::StdRng>,
    ) -> SklResult<(Array2<f64>, Array1<f64>)> {
        // Normalize spectral densities to create probability distribution
        let total_density: f64 = spectral_densities.sum();
        if total_density <= 0.0 {
            return self.random_selection(frequencies, spectral_densities, rng);
        }

        let probabilities: Array1<f64> = spectral_densities / total_density;
        let mut selected_indices = Vec::new();

        for _ in 0..self.num_spectral_points.min(frequencies.nrows()) {
            let mut cumulative = 0.0;
            let random_value: f64 = rng.gen();

            for (i, &prob) in probabilities.iter().enumerate() {
                cumulative += prob;
                if random_value <= cumulative && !selected_indices.contains(&i) {
                    selected_indices.push(i);
                    break;
                }
            }
        }

        // Fill remaining slots with random selection if needed
        while selected_indices.len() < self.num_spectral_points.min(frequencies.nrows()) {
            let idx = rng.gen_range(0..frequencies.nrows());
            if !selected_indices.contains(&idx) {
                selected_indices.push(idx);
            }
        }

        let selected_frequencies = frequencies.select(Axis(0), &selected_indices);
        let selected_weights = spectral_densities.select(Axis(0), &selected_indices);

        Ok((selected_frequencies, selected_weights))
    }

    /// Quasi-random selection using low-discrepancy sequences
    fn quasi_random_selection(
        &self,
        frequencies: &Array2<f64>,
        spectral_densities: &Array1<f64>,
        rng: &mut scirs2_core::random::Random<scirs2_core::rngs::StdRng>,
    ) -> SklResult<(Array2<f64>, Array1<f64>)> {
        // Simple implementation: stratified sampling
        let total_points = frequencies.nrows();
        let stride = total_points / self.num_spectral_points.max(1);

        let mut selected_indices = Vec::new();
        for i in 0..self.num_spectral_points.min(total_points) {
            let base_idx = i * stride;
            let jitter = rng.gen_range(0..stride.max(1));
            let idx = (base_idx + jitter).min(total_points - 1);
            selected_indices.push(idx);
        }

        let selected_frequencies = frequencies.select(Axis(0), &selected_indices);
        let selected_weights = spectral_densities.select(Axis(0), &selected_indices);

        Ok((selected_frequencies, selected_weights))
    }

    /// Adaptive selection based on data characteristics
    fn adaptive_selection(
        &self,
        frequencies: &Array2<f64>,
        spectral_densities: &Array1<f64>,
        rng: &mut scirs2_core::random::Random<scirs2_core::rngs::StdRng>,
    ) -> SklResult<(Array2<f64>, Array1<f64>)> {
        // Combine greedy selection for high-density regions with random exploration
        let greedy_fraction = 0.7;
        let num_greedy = (self.num_spectral_points as f64 * greedy_fraction) as usize;
        let num_random = self.num_spectral_points - num_greedy;

        // Greedy selection for high-density points
        let (greedy_freqs, greedy_weights) = if num_greedy > 0 {
            let mut temp_config = self.clone();
            temp_config.num_spectral_points = num_greedy;
            temp_config.greedy_selection(frequencies, spectral_densities)?
        } else {
            (Array2::zeros((0, frequencies.ncols())), Array1::zeros(0))
        };

        // Random selection for exploration
        let (random_freqs, random_weights) = if num_random > 0 {
            let mut temp_config = self.clone();
            temp_config.num_spectral_points = num_random;
            temp_config.random_selection(frequencies, spectral_densities, rng)?
        } else {
            (Array2::zeros((0, frequencies.ncols())), Array1::zeros(0))
        };

        // Combine results
        let mut combined_freqs = Array2::zeros((num_greedy + num_random, frequencies.ncols()));
        let mut combined_weights = Array1::zeros(num_greedy + num_random);

        if num_greedy > 0 {
            combined_freqs
                .slice_mut(s![0..num_greedy, ..])
                .assign(&greedy_freqs);
            combined_weights
                .slice_mut(s![0..num_greedy])
                .assign(&greedy_weights);
        }

        if num_random > 0 {
            combined_freqs
                .slice_mut(s![num_greedy.., ..])
                .assign(&random_freqs);
            combined_weights
                .slice_mut(s![num_greedy..])
                .assign(&random_weights);
        }

        Ok((combined_freqs, combined_weights))
    }

    /// Compute spectral features for given data points
    fn compute_spectral_features(
        &self,
        X: &ArrayView2<f64>,
        spectral_points: &Array2<f64>,
        spectral_weights: &Array1<f64>,
    ) -> SklResult<Array2<f64>> {
        let n_samples = X.nrows();
        let n_spectral = spectral_points.nrows();

        // Each spectral point contributes cos and sin features
        let mut features = Array2::zeros((n_samples, 2 * n_spectral));

        for i in 0..n_samples {
            for j in 0..n_spectral {
                let phase = 2.0 * std::f64::consts::PI * spectral_points.row(j).dot(&X.row(i));
                let weight_sqrt = spectral_weights[j].sqrt();

                features[[i, 2 * j]] = weight_sqrt * phase.cos();
                features[[i, 2 * j + 1]] = weight_sqrt * phase.sin();
            }
        }

        Ok(features)
    }

    /// Optimize spectral points using gradient-based optimization
    fn optimize_spectral_points_internal(
        &self,
        X: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        mut spectral_points: Array2<f64>,
        spectral_weights: &Array1<f64>,
    ) -> SklResult<Array2<f64>> {
        if !self.optimize_spectral_points {
            return Ok(spectral_points);
        }

        for _iteration in 0..self.max_optimization_iterations {
            // Compute current features and objective
            let features = self.compute_spectral_features(X, &spectral_points, spectral_weights)?;
            let objective = self.compute_spectral_objective(&features, y)?;

            // Compute gradients (simplified finite differences)
            let mut gradients = Array2::zeros(spectral_points.raw_dim());
            let epsilon = 1e-6;

            for i in 0..spectral_points.nrows() {
                for j in 0..spectral_points.ncols() {
                    // Forward difference
                    spectral_points[[i, j]] += epsilon;
                    let features_plus =
                        self.compute_spectral_features(X, &spectral_points, spectral_weights)?;
                    let objective_plus = self.compute_spectral_objective(&features_plus, y)?;

                    spectral_points[[i, j]] -= epsilon;
                    gradients[[i, j]] = (objective_plus - objective) / epsilon;
                }
            }

            // Update spectral points
            spectral_points = spectral_points - self.spectral_learning_rate * gradients;
        }

        Ok(spectral_points)
    }

    /// Compute objective function for spectral point optimization
    #[allow(non_snake_case)]
    fn compute_spectral_objective(
        &self,
        features: &Array2<f64>,
        y: &ArrayView1<f64>,
    ) -> SklResult<f64> {
        // Use negative log marginal likelihood as objective
        let n_features = features.ncols();
        let n_samples = features.nrows();

        // Compute Phi^T Phi + noise_variance * I
        let phi_t_phi = features.t().dot(features);
        let gram_matrix = phi_t_phi + Array2::<f64>::eye(n_features) * self.noise_variance;

        // Cholesky decomposition
        let L = crate::utils::cholesky_decomposition(&gram_matrix)?;

        // Solve for posterior mean
        let phi_t_y = features.t().dot(y);
        let alpha = crate::utils::triangular_solve(&L, &phi_t_y)?;
        let L_t = L.t();
        let mean = crate::utils::triangular_solve(&L_t.view().to_owned(), &alpha)?;

        // Compute log marginal likelihood
        let data_fit = -0.5 * y.dot(&features.dot(&mean));
        let mut log_det = 0.0;
        for i in 0..L.nrows() {
            log_det += L[[i, i]].ln();
        }
        let complexity_penalty = -log_det;
        let normalization = -0.5 * n_samples as f64 * (2.0 * std::f64::consts::PI).ln();

        Ok(-(data_fit + complexity_penalty + normalization))
    }

    /// Compute spectral approximation quality metrics
    pub fn compute_approximation_info(
        &self,
        spectral_points: &Array2<f64>,
        spectral_weights: &Array1<f64>,
    ) -> SklResult<SpectralApproximationInfo> {
        let effective_rank =
            spectral_weights.sum() / spectral_weights.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let spectral_coverage = (spectral_weights
            .iter()
            .filter(|&&w| w > self.spectral_density_threshold)
            .count() as f64)
            / spectral_weights.len() as f64;

        // Rough approximation error estimate
        let total_spectral_energy = spectral_weights.sum();
        let selected_energy = spectral_weights
            .iter()
            .filter(|&&w| w > self.spectral_density_threshold)
            .sum::<f64>();
        let max_approximation_error = 1.0 - (selected_energy / total_spectral_energy.max(1e-10));

        Ok(SpectralApproximationInfo {
            effective_rank,
            spectral_coverage,
            max_approximation_error,
            selected_frequencies: spectral_points.clone(),
            spectral_densities: spectral_weights.clone(),
        })
    }
}

impl Estimator for SparseSpectrumGaussianProcessRegressor {
    type Config = SparseSpectrumGaussianProcessRegressor;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        self
    }
}

impl Fit<ArrayView2<'_, f64>, ArrayView1<'_, f64>, SparseSpectrumGprTrained>
    for SparseSpectrumGaussianProcessRegressor
{
    type Fitted = SparseSpectrumGprTrained;
    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> SklResult<SparseSpectrumGprTrained> {
        if X.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        // Estimate spectral density
        let grid_size = (self.num_spectral_points * 10).max(1000);
        let (frequencies, spectral_densities) = self.estimate_spectral_density(X, grid_size)?;

        // Select spectral points
        let (mut spectral_points, spectral_weights) =
            self.select_spectral_points(&frequencies, &spectral_densities)?;

        // Optimize spectral points if requested
        spectral_points =
            self.optimize_spectral_points_internal(X, y, spectral_points, &spectral_weights)?;

        // Compute spectral features
        let spectral_features =
            self.compute_spectral_features(X, &spectral_points, &spectral_weights)?;

        // Bayesian linear regression in spectral feature space
        let n_features = spectral_features.ncols();
        let phi_t_phi = spectral_features.t().dot(&spectral_features);
        let gram_matrix = phi_t_phi + Array2::<f64>::eye(n_features) * self.noise_variance;

        // Cholesky decomposition
        let L = crate::utils::cholesky_decomposition(&gram_matrix)?;

        // Compute posterior mean
        let phi_t_y = spectral_features.t().dot(y);
        let alpha = crate::utils::triangular_solve(&L, &phi_t_y)?;
        let L_t = L.t();
        let posterior_mean = crate::utils::triangular_solve(&L_t.view().to_owned(), &alpha)?;

        // Compute posterior covariance (simplified - use diagonal approximation)
        let posterior_covariance = Array2::<f64>::eye(n_features) / self.noise_variance;

        // Compute log marginal likelihood
        let data_fit = -0.5 * y.dot(&spectral_features.dot(&posterior_mean));
        let mut log_det = 0.0;
        for i in 0..L.nrows() {
            log_det += L[[i, i]].ln();
        }
        let complexity_penalty = -log_det;
        let normalization = -0.5 * y.len() as f64 * (2.0 * std::f64::consts::PI).ln();
        let log_marginal_likelihood = data_fit + complexity_penalty + normalization;

        Ok(SparseSpectrumGprTrained {
            config: self.clone(),
            spectral_points,
            spectral_weights,
            spectral_features,
            posterior_mean,
            posterior_covariance,
            X_train: X.to_owned(),
            y_train: y.to_owned(),
            spectral_density: spectral_densities,
            log_marginal_likelihood,
        })
    }
}

impl Predict<ArrayView2<'_, f64>, Array1<f64>> for SparseSpectrumGprTrained {
    fn predict(&self, X: &ArrayView2<f64>) -> SklResult<Array1<f64>> {
        // Compute spectral features for test data
        let test_features = self.config.compute_spectral_features(
            X,
            &self.spectral_points,
            &self.spectral_weights,
        )?;

        // Compute predictions
        let predictions = test_features.dot(&self.posterior_mean);
        Ok(predictions)
    }
}

impl SparseSpectrumGprTrained {
    /// Predict with uncertainty quantification
    pub fn predict_with_uncertainty(
        &self,
        X: &ArrayView2<f64>,
    ) -> SklResult<(Array1<f64>, Array1<f64>)> {
        // Compute spectral features for test data
        let test_features = self.config.compute_spectral_features(
            X,
            &self.spectral_points,
            &self.spectral_weights,
        )?;

        // Compute predictions
        let predictions = test_features.dot(&self.posterior_mean);

        // Compute predictive variance
        let mut variances = Array1::zeros(X.nrows());
        for i in 0..X.nrows() {
            let feature_vector = test_features.row(i);
            let variance = feature_vector.dot(&self.posterior_covariance.dot(&feature_vector))
                + self.config.noise_variance;
            variances[i] = variance;
        }

        Ok((predictions, variances))
    }

    /// Get spectral approximation quality information
    pub fn approximation_info(&self) -> SklResult<SpectralApproximationInfo> {
        self.config
            .compute_approximation_info(&self.spectral_points, &self.spectral_weights)
    }

    /// Get log marginal likelihood
    pub fn log_marginal_likelihood(&self) -> f64 {
        self.log_marginal_likelihood
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::RBF;
    // SciRS2 Policy - Use scirs2-autograd for ndarray types and operations
    use scirs2_core::ndarray::{Array1, Array2};

    #[test]
    fn test_sparse_spectrum_gpr_creation() {
        let kernel = Box::new(RBF::new(1.0));
        let gpr = SparseSpectrumGaussianProcessRegressor::new(kernel)
            .num_spectral_points(50)
            .spectral_density_threshold(1e-6);

        assert_eq!(gpr.num_spectral_points, 50);
        assert_eq!(gpr.spectral_density_threshold, 1e-6);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_spectral_feature_computation() {
        let kernel = Box::new(RBF::new(1.0));
        let gpr = SparseSpectrumGaussianProcessRegressor::new(kernel);

        let X = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let spectral_points = Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        let spectral_weights = Array1::from_vec(vec![1.0, 0.5]);

        let features = gpr
            .compute_spectral_features(&X.view(), &spectral_points, &spectral_weights)
            .unwrap();

        assert_eq!(features.nrows(), 3);
        assert_eq!(features.ncols(), 4); // 2 spectral points * 2 (cos + sin)
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_sparse_spectrum_fit_predict() {
        let kernel = Box::new(RBF::new(1.0));
        let gpr = SparseSpectrumGaussianProcessRegressor::new(kernel)
            .num_spectral_points(10)
            .optimize_spectral_points(false); // Disable for faster testing

        let X = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = Array1::from_vec(vec![1.0, 4.0, 9.0, 16.0, 25.0]);

        let trained = gpr.fit(&X.view(), &y.view()).unwrap();
        let predictions = trained.predict(&X.view()).unwrap();

        assert_eq!(predictions.len(), 5);
        assert!(trained.log_marginal_likelihood().is_finite());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_prediction_with_uncertainty() {
        let kernel = Box::new(RBF::new(1.0));
        let gpr = SparseSpectrumGaussianProcessRegressor::new(kernel)
            .num_spectral_points(5)
            .optimize_spectral_points(false);

        let X = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let trained = gpr.fit(&X.view(), &y.view()).unwrap();
        let (predictions, variances) = trained.predict_with_uncertainty(&X.view()).unwrap();

        assert_eq!(predictions.len(), 3);
        assert_eq!(variances.len(), 3);
        assert!(variances.iter().all(|&v| v >= 0.0)); // Variances should be non-negative
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_spectral_selection_methods() {
        let kernel = Box::new(RBF::new(1.0));

        let methods = vec![
            SpectralSelectionMethod::Random,
            SpectralSelectionMethod::Greedy,
            SpectralSelectionMethod::ImportanceSampling,
            SpectralSelectionMethod::QuasiRandom,
            SpectralSelectionMethod::Adaptive,
        ];

        for method in methods {
            let gpr = SparseSpectrumGaussianProcessRegressor::new(kernel.clone())
                .num_spectral_points(3)
                .selection_method(method)
                .optimize_spectral_points(false);

            let X = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
            let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

            let result = gpr.fit(&X.view(), &y.view());
            assert!(result.is_ok());
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_approximation_info() {
        let kernel = Box::new(RBF::new(1.0));
        let gpr = SparseSpectrumGaussianProcessRegressor::new(kernel)
            .num_spectral_points(5)
            .optimize_spectral_points(false);

        let X = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let trained = gpr.fit(&X.view(), &y.view()).unwrap();
        let info = trained.approximation_info().unwrap();

        assert!(info.effective_rank > 0.0);
        assert!(info.spectral_coverage >= 0.0 && info.spectral_coverage <= 1.0);
        assert!(info.max_approximation_error >= 0.0);
    }
}
