//! Time series kernel approximations
//!
//! This module provides kernel approximation methods specifically designed for
//! time series data, including Dynamic Time Warping (DTW), autoregressive kernels,
//! spectral kernels, and other time-series specific kernel methods.

use rayon::prelude::*;
use scirs2_core::ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2};
use scirs2_core::random::essentials::Normal as RandNormal;
use scirs2_core::random::rngs::StdRng as RealStdRng;
use scirs2_core::random::Distribution;
use scirs2_core::random::{thread_rng, Rng, SeedableRng};
use sklears_core::error::Result;

/// Time series kernel type
#[derive(Clone, Debug, PartialEq)]
/// TimeSeriesKernelType
pub enum TimeSeriesKernelType {
    /// Dynamic Time Warping kernel
    DTW {
        /// Window size for DTW alignment
        window_size: Option<usize>,
        /// Penalty for insertions/deletions
        penalty: f64,
    },
    /// Autoregressive kernel
    Autoregressive {
        /// Order of the autoregressive model
        order: usize,
        /// Regularization parameter
        lambda: f64,
    },
    /// Spectral kernel using Fourier transform
    Spectral {
        /// Number of frequency components
        n_frequencies: usize,
        /// Whether to use magnitude only or include phase
        magnitude_only: bool,
    },
    /// Global Alignment Kernel (GAK)
    GlobalAlignment { sigma: f64, triangular: bool },
    /// Time Warp Edit Distance (TWED)
    TimeWarpEdit {
        /// Penalty for elastic transformation
        nu: f64,
        /// Penalty for time warping
        lambda: f64,
    },
    /// Subsequence Time Series Kernel
    Subsequence {
        /// Length of subsequences
        subsequence_length: usize,
        /// Step size for sliding window
        step_size: usize,
    },
    /// Shapelet-based kernel
    Shapelet {
        /// Number of shapelets to extract
        n_shapelets: usize,
        /// Minimum shapelet length
        min_length: usize,
        /// Maximum shapelet length
        max_length: usize,
    },
}

/// DTW distance computation configuration
#[derive(Clone, Debug)]
/// DTWConfig
pub struct DTWConfig {
    /// Window constraint for DTW alignment
    pub window_type: DTWWindowType,
    /// Distance metric for individual points
    pub distance_metric: DTWDistanceMetric,
    /// Step pattern for DTW
    pub step_pattern: DTWStepPattern,
    /// Whether to normalize by path length
    pub normalize: bool,
}

/// DTW window constraint types
#[derive(Clone, Debug, PartialEq)]
/// DTWWindowType
pub enum DTWWindowType {
    /// No window constraint
    None,
    /// Sakoe-Chiba band
    SakoeChiba { window_size: usize },
    /// Itakura parallelogram
    Itakura,
    /// Custom window function
    Custom { window_func: Vec<(usize, usize)> },
}

/// Distance metrics for DTW
#[derive(Clone, Debug, PartialEq)]
/// DTWDistanceMetric
pub enum DTWDistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Manhattan distance
    Manhattan,
    /// Cosine distance
    Cosine,
    /// Custom distance function
    Custom,
}

/// DTW step patterns
#[derive(Clone, Debug, PartialEq)]
/// DTWStepPattern
pub enum DTWStepPattern {
    /// Symmetric step pattern
    Symmetric,
    /// Asymmetric step pattern
    Asymmetric,
    /// Custom step pattern
    Custom { steps: Vec<(i32, i32, f64)> },
}

impl Default for DTWConfig {
    fn default() -> Self {
        Self {
            window_type: DTWWindowType::None,
            distance_metric: DTWDistanceMetric::Euclidean,
            step_pattern: DTWStepPattern::Symmetric,
            normalize: true,
        }
    }
}

/// Time series kernel configuration
#[derive(Clone, Debug)]
/// TimeSeriesKernelConfig
pub struct TimeSeriesKernelConfig {
    /// Type of time series kernel
    pub kernel_type: TimeSeriesKernelType,
    /// Number of random features for approximation
    pub n_components: usize,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// DTW-specific configuration
    pub dtw_config: Option<DTWConfig>,
    /// Whether to normalize time series
    pub normalize_series: bool,
    /// Number of parallel workers
    pub n_workers: usize,
}

impl Default for TimeSeriesKernelConfig {
    fn default() -> Self {
        Self {
            kernel_type: TimeSeriesKernelType::DTW {
                window_size: None,
                penalty: 0.0,
            },
            n_components: 100,
            random_state: None,
            dtw_config: Some(DTWConfig::default()),
            normalize_series: true,
            n_workers: num_cpus::get(),
        }
    }
}

/// Dynamic Time Warping kernel approximation
pub struct DTWKernelApproximation {
    config: TimeSeriesKernelConfig,
    reference_series: Option<Array2<f64>>,
    random_indices: Option<Vec<usize>>,
    dtw_distances: Option<Array2<f64>>,
    kernel_bandwidth: f64,
}

impl DTWKernelApproximation {
    /// Create a new DTW kernel approximation
    pub fn new(n_components: usize) -> Self {
        Self {
            config: TimeSeriesKernelConfig {
                n_components,
                kernel_type: TimeSeriesKernelType::DTW {
                    window_size: None,
                    penalty: 0.0,
                },
                ..Default::default()
            },
            reference_series: None,
            random_indices: None,
            dtw_distances: None,
            kernel_bandwidth: 1.0,
        }
    }

    /// Set kernel bandwidth
    pub fn bandwidth(mut self, bandwidth: f64) -> Self {
        self.kernel_bandwidth = bandwidth;
        self
    }

    /// Set DTW window size
    pub fn window_size(mut self, new_window_size: Option<usize>) -> Self {
        if let TimeSeriesKernelType::DTW {
            ref mut window_size,
            ..
        } = self.config.kernel_type
        {
            *window_size = new_window_size;
        }
        self
    }

    /// Set configuration
    pub fn with_config(mut self, config: TimeSeriesKernelConfig) -> Self {
        self.config = config;
        self
    }

    /// Fit the DTW kernel approximation
    pub fn fit(&mut self, time_series: &Array3<f64>) -> Result<()> {
        let (n_series, n_timepoints, n_features) = time_series.dim();

        // Select random reference series for approximation
        let mut rng = if let Some(seed) = self.config.random_state {
            RealStdRng::seed_from_u64(seed)
        } else {
            RealStdRng::from_seed(thread_rng().gen())
        };

        let n_references = std::cmp::min(self.config.n_components, n_series);
        let mut indices: Vec<usize> = (0..n_series).collect();
        indices.sort_by_key(|_| rng.gen::<u32>());
        indices.truncate(n_references);
        self.random_indices = Some(indices.clone());

        // Extract reference series
        let mut reference_series = Array2::zeros((n_references, n_timepoints * n_features));
        for (i, &idx) in indices.iter().enumerate() {
            let series = time_series.slice(s![idx, .., ..]);
            let flattened = series.into_shape((n_timepoints * n_features,)).unwrap();
            reference_series.row_mut(i).assign(&flattened);
        }
        self.reference_series = Some(reference_series);

        Ok(())
    }

    /// Transform time series using DTW kernel features
    pub fn transform(&self, time_series: &Array3<f64>) -> Result<Array2<f64>> {
        let reference_series = self.reference_series.as_ref().ok_or("Model not fitted")?;
        let (n_series, n_timepoints, n_features) = time_series.dim();
        let n_references = reference_series.nrows();

        let mut features = Array2::zeros((n_series, n_references));

        // Compute DTW distances to reference series
        for i in 0..n_series {
            let series = time_series.slice(s![i, .., ..]);
            let series_flat = series.into_shape((n_timepoints * n_features,)).unwrap();

            for j in 0..n_references {
                let reference = reference_series.row(j);
                let distance = self.compute_dtw_distance(&series_flat, &reference)?;

                // Convert distance to kernel value using RBF kernel
                let kernel_value = (-distance / (2.0 * self.kernel_bandwidth.powi(2))).exp();
                features[[i, j]] = kernel_value;
            }
        }

        Ok(features)
    }

    /// Compute DTW distance between two time series
    fn compute_dtw_distance(
        &self,
        series1: &ArrayView1<f64>,
        series2: &ArrayView1<f64>,
    ) -> Result<f64> {
        let n1 = series1.len();
        let n2 = series2.len();

        // Initialize DTW matrix
        let mut dtw_matrix = Array2::from_elem((n1 + 1, n2 + 1), f64::INFINITY);
        dtw_matrix[[0, 0]] = 0.0;

        // Apply window constraint if specified
        let window_constraint = self.get_window_constraint(n1, n2);

        for i in 1..=n1 {
            for j in 1..=n2 {
                if self.is_within_window(i - 1, j - 1, &window_constraint) {
                    let cost = self.compute_point_distance(series1[i - 1], series2[j - 1]);

                    let candidates = vec![
                        dtw_matrix[[i - 1, j]] + cost,     // Insertion
                        dtw_matrix[[i, j - 1]] + cost,     // Deletion
                        dtw_matrix[[i - 1, j - 1]] + cost, // Match
                    ];

                    dtw_matrix[[i, j]] = candidates
                        .into_iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap();
                }
            }
        }

        let distance = dtw_matrix[[n1, n2]];

        // Normalize by path length if requested
        if self
            .config
            .dtw_config
            .as_ref()
            .map_or(true, |cfg| cfg.normalize)
        {
            Ok(distance / (n1 + n2) as f64)
        } else {
            Ok(distance)
        }
    }

    /// Get window constraint for DTW
    fn get_window_constraint(&self, n1: usize, n2: usize) -> Option<Vec<(usize, usize)>> {
        if let Some(dtw_config) = &self.config.dtw_config {
            match &dtw_config.window_type {
                DTWWindowType::SakoeChiba { window_size } => {
                    let mut constraints = Vec::new();
                    for i in 0..n1 {
                        let j_start = (i as i32 - *window_size as i32).max(0) as usize;
                        let j_end = (i + window_size).min(n2 - 1);
                        for j in j_start..=j_end {
                            constraints.push((i, j));
                        }
                    }
                    Some(constraints)
                }
                DTWWindowType::Custom { window_func } => Some(window_func.clone()),
                _ => None,
            }
        } else {
            None
        }
    }

    /// Check if point is within window constraint
    fn is_within_window(
        &self,
        i: usize,
        j: usize,
        window_constraint: &Option<Vec<(usize, usize)>>,
    ) -> bool {
        match window_constraint {
            Some(constraints) => constraints.contains(&(i, j)),
            None => true,
        }
    }

    /// Compute distance between two points
    fn compute_point_distance(&self, x1: f64, x2: f64) -> f64 {
        let metric = self
            .config
            .dtw_config
            .as_ref()
            .map(|cfg| &cfg.distance_metric)
            .unwrap_or(&DTWDistanceMetric::Euclidean);

        match metric {
            DTWDistanceMetric::Euclidean => (x1 - x2).powi(2),
            DTWDistanceMetric::Manhattan => (x1 - x2).abs(),
            DTWDistanceMetric::Cosine => 1.0 - (x1 * x2) / ((x1.powi(2) + x2.powi(2)).sqrt()),
            DTWDistanceMetric::Custom => (x1 - x2).powi(2), // Default to Euclidean
        }
    }
}

/// Autoregressive kernel approximation
pub struct AutoregressiveKernelApproximation {
    config: TimeSeriesKernelConfig,
    ar_coefficients: Option<Array2<f64>>,
    reference_models: Option<Vec<Array1<f64>>>,
    random_features: Option<Array2<f64>>,
}

impl AutoregressiveKernelApproximation {
    /// Create a new autoregressive kernel approximation
    pub fn new(n_components: usize, order: usize) -> Self {
        Self {
            config: TimeSeriesKernelConfig {
                n_components,
                kernel_type: TimeSeriesKernelType::Autoregressive { order, lambda: 0.1 },
                ..Default::default()
            },
            ar_coefficients: None,
            reference_models: None,
            random_features: None,
        }
    }

    /// Set regularization parameter
    pub fn lambda(mut self, new_lambda: f64) -> Self {
        if let TimeSeriesKernelType::Autoregressive { ref mut lambda, .. } = self.config.kernel_type
        {
            *lambda = new_lambda;
        }
        self
    }

    /// Fit the autoregressive kernel approximation
    pub fn fit(&mut self, time_series: &Array3<f64>) -> Result<()> {
        let (n_series, n_timepoints, n_features) = time_series.dim();

        if let TimeSeriesKernelType::Autoregressive { order, lambda } = &self.config.kernel_type {
            // Fit AR models to each time series
            let mut ar_coefficients = Array2::zeros((n_series, order * n_features));

            for i in 0..n_series {
                let series = time_series.slice(s![i, .., ..]);
                let coeffs = self.fit_ar_model(&series, *order, *lambda)?;
                ar_coefficients.row_mut(i).assign(&coeffs);
            }

            self.ar_coefficients = Some(ar_coefficients);

            // Generate random features based on AR coefficients
            self.generate_random_features()?;
        }

        Ok(())
    }

    /// Transform time series using AR kernel features
    pub fn transform(&self, time_series: &Array3<f64>) -> Result<Array2<f64>> {
        let ar_coefficients = self.ar_coefficients.as_ref().ok_or("Model not fitted")?;
        let random_features = self
            .random_features
            .as_ref()
            .ok_or("Random features not generated")?;

        let (n_series, n_timepoints, n_features) = time_series.dim();
        let n_components = self.config.n_components;

        let mut features = Array2::zeros((n_series, n_components));

        if let TimeSeriesKernelType::Autoregressive { order, lambda } = &self.config.kernel_type {
            for i in 0..n_series {
                let series = time_series.slice(s![i, .., ..]);
                let coeffs = self.fit_ar_model(&series, *order, *lambda)?;

                // Compute random features
                for j in 0..n_components {
                    let random_proj = coeffs.dot(&random_features.row(j));
                    features[[i, j]] = random_proj.cos();
                }
            }
        }

        Ok(features)
    }

    /// Fit AR model to a single time series
    fn fit_ar_model(
        &self,
        series: &ArrayView2<f64>,
        order: usize,
        lambda: f64,
    ) -> Result<Array1<f64>> {
        let (n_timepoints, n_features) = series.dim();

        if n_timepoints <= order {
            return Err("Time series too short for specified AR order".into());
        }

        // Create design matrix X and target vector y
        let n_samples = n_timepoints - order;
        let mut x_matrix = Array2::zeros((n_samples, order * n_features));
        let mut y_vector = Array2::zeros((n_samples, n_features));

        for t in order..n_timepoints {
            let sample_idx = t - order;

            // Fill design matrix with lagged values
            for lag in 1..=order {
                let lag_idx = t - lag;
                for feat in 0..n_features {
                    x_matrix[[sample_idx, (lag - 1) * n_features + feat]] = series[[lag_idx, feat]];
                }
            }

            // Fill target vector
            for feat in 0..n_features {
                y_vector[[sample_idx, feat]] = series[[t, feat]];
            }
        }

        // Solve least squares with regularization: (X^T X + λI)β = X^T y
        let xtx = x_matrix.t().dot(&x_matrix);
        let xtx_reg = xtx + Array2::<f64>::eye(order * n_features) * lambda;
        let xty = x_matrix.t().dot(&y_vector);

        // Simplified solution (in practice, use proper linear algebra)
        let coeffs = self.solve_linear_system(&xtx_reg, &xty)?;

        Ok(coeffs)
    }

    /// Solve linear system (simplified implementation)
    fn solve_linear_system(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array1<f64>> {
        // This is a simplified implementation - in practice use proper solvers
        let n = a.nrows();
        let n_features = b.ncols();
        let mut solution = Array1::zeros(n);

        // Use diagonal approximation for simplicity - average across features
        for i in 0..n {
            if a[[i, i]].abs() > 1e-12 {
                let avg_target =
                    (0..n_features).map(|j| b[[i, j]]).sum::<f64>() / n_features as f64;
                solution[i] = avg_target / a[[i, i]];
            }
        }

        Ok(solution)
    }

    /// Generate random features for AR kernel approximation
    fn generate_random_features(&mut self) -> Result<()> {
        if let TimeSeriesKernelType::Autoregressive { .. } = &self.config.kernel_type {
            let ar_coefficients = self.ar_coefficients.as_ref().unwrap();
            let (_, n_ar_features) = ar_coefficients.dim();

            let mut rng = if let Some(seed) = self.config.random_state {
                RealStdRng::seed_from_u64(seed)
            } else {
                RealStdRng::from_seed(thread_rng().gen())
            };

            let normal = RandNormal::new(0.0, 1.0).unwrap();
            let random_features =
                Array2::from_shape_fn((self.config.n_components, n_ar_features), |_| {
                    rng.sample(normal)
                });

            self.random_features = Some(random_features);
        }

        Ok(())
    }
}

/// Spectral kernel approximation for time series
pub struct SpectralKernelApproximation {
    config: TimeSeriesKernelConfig,
    frequency_features: Option<Array2<f64>>,
    reference_spectra: Option<Array2<f64>>,
}

impl SpectralKernelApproximation {
    /// Create a new spectral kernel approximation
    pub fn new(n_components: usize, n_frequencies: usize) -> Self {
        Self {
            config: TimeSeriesKernelConfig {
                n_components,
                kernel_type: TimeSeriesKernelType::Spectral {
                    n_frequencies,
                    magnitude_only: true,
                },
                ..Default::default()
            },
            frequency_features: None,
            reference_spectra: None,
        }
    }

    /// Set whether to use magnitude only
    pub fn magnitude_only(mut self, new_magnitude_only: bool) -> Self {
        if let TimeSeriesKernelType::Spectral {
            ref mut magnitude_only,
            ..
        } = self.config.kernel_type
        {
            *magnitude_only = new_magnitude_only;
        }
        self
    }

    /// Fit the spectral kernel approximation
    pub fn fit(&mut self, time_series: &Array3<f64>) -> Result<()> {
        let (n_series, n_timepoints, n_features) = time_series.dim();

        if let TimeSeriesKernelType::Spectral {
            n_frequencies,
            magnitude_only,
        } = &self.config.kernel_type
        {
            // Compute frequency domain representation
            let mut spectra = Array2::zeros((n_series, *n_frequencies * n_features));

            for i in 0..n_series {
                let series = time_series.slice(s![i, .., ..]);
                let spectrum =
                    self.compute_frequency_features(&series, *n_frequencies, *magnitude_only)?;
                spectra.row_mut(i).assign(&spectrum);
            }

            self.reference_spectra = Some(spectra);

            // Generate random projection features
            self.generate_spectral_features(*n_frequencies * n_features)?;
        }

        Ok(())
    }

    /// Transform time series using spectral features
    pub fn transform(&self, time_series: &Array3<f64>) -> Result<Array2<f64>> {
        let frequency_features = self.frequency_features.as_ref().ok_or("Model not fitted")?;

        let (n_series, n_timepoints, n_features) = time_series.dim();
        let n_components = self.config.n_components;

        let mut features = Array2::zeros((n_series, n_components));

        if let TimeSeriesKernelType::Spectral {
            n_frequencies,
            magnitude_only,
        } = &self.config.kernel_type
        {
            for i in 0..n_series {
                let series = time_series.slice(s![i, .., ..]);
                let spectrum =
                    self.compute_frequency_features(&series, *n_frequencies, *magnitude_only)?;

                // Apply random projection
                for j in 0..n_components {
                    let projection = spectrum.dot(&frequency_features.row(j));
                    features[[i, j]] = projection.cos();
                }
            }
        }

        Ok(features)
    }

    /// Compute frequency domain features using FFT
    fn compute_frequency_features(
        &self,
        series: &ArrayView2<f64>,
        n_frequencies: usize,
        magnitude_only: bool,
    ) -> Result<Array1<f64>> {
        let (n_timepoints, n_features) = series.dim();
        let mut features = Vec::new();

        for feat in 0..n_features {
            let signal = series.column(feat);

            // Simple discrete Fourier transform approximation
            for k in 0..n_frequencies {
                let freq = 2.0 * std::f64::consts::PI * k as f64 / n_timepoints as f64;

                let mut real_part = 0.0;
                let mut imag_part = 0.0;

                for t in 0..n_timepoints {
                    let angle = freq * t as f64;
                    real_part += signal[t] * angle.cos();
                    imag_part += signal[t] * angle.sin();
                }

                if magnitude_only {
                    features.push((real_part.powi(2) + imag_part.powi(2)).sqrt());
                } else {
                    features.push(real_part);
                    features.push(imag_part);
                }
            }
        }

        Ok(Array1::from(features))
    }

    /// Generate random spectral features
    fn generate_spectral_features(&mut self, n_spectrum_features: usize) -> Result<()> {
        let mut rng = if let Some(seed) = self.config.random_state {
            RealStdRng::seed_from_u64(seed)
        } else {
            RealStdRng::from_seed(thread_rng().gen())
        };

        let normal = RandNormal::new(0.0, 1.0).unwrap();
        let frequency_features =
            Array2::from_shape_fn((self.config.n_components, n_spectrum_features), |_| {
                rng.sample(normal)
            });

        self.frequency_features = Some(frequency_features);
        Ok(())
    }
}

/// Global Alignment Kernel (GAK) approximation
pub struct GlobalAlignmentKernelApproximation {
    config: TimeSeriesKernelConfig,
    reference_series: Option<Array2<f64>>,
    sigma: f64,
}

impl GlobalAlignmentKernelApproximation {
    /// Create a new GAK approximation
    pub fn new(n_components: usize, sigma: f64) -> Self {
        Self {
            config: TimeSeriesKernelConfig {
                n_components,
                kernel_type: TimeSeriesKernelType::GlobalAlignment {
                    sigma,
                    triangular: false,
                },
                ..Default::default()
            },
            reference_series: None,
            sigma,
        }
    }

    /// Fit the GAK approximation
    pub fn fit(&mut self, time_series: &Array3<f64>) -> Result<()> {
        let (n_series, n_timepoints, n_features) = time_series.dim();

        // Select random reference series
        let mut rng = if let Some(seed) = self.config.random_state {
            RealStdRng::seed_from_u64(seed)
        } else {
            RealStdRng::from_seed(thread_rng().gen())
        };

        let n_references = std::cmp::min(self.config.n_components, n_series);
        let mut indices: Vec<usize> = (0..n_series).collect();
        indices.sort_by_key(|_| rng.gen::<u32>());
        indices.truncate(n_references);

        let mut reference_series = Array2::zeros((n_references, n_timepoints * n_features));
        for (i, &idx) in indices.iter().enumerate() {
            let series = time_series.slice(s![idx, .., ..]);
            let flattened = series.into_shape((n_timepoints * n_features,)).unwrap();
            reference_series.row_mut(i).assign(&flattened);
        }

        self.reference_series = Some(reference_series);
        Ok(())
    }

    /// Transform using GAK features
    pub fn transform(&self, time_series: &Array3<f64>) -> Result<Array2<f64>> {
        let reference_series = self.reference_series.as_ref().ok_or("Model not fitted")?;

        let (n_series, n_timepoints, n_features) = time_series.dim();
        let n_references = reference_series.nrows();

        let mut features = Array2::zeros((n_series, n_references));

        for i in 0..n_series {
            let series = time_series.slice(s![i, .., ..]);
            let series_flat = series.into_shape((n_timepoints * n_features,)).unwrap();

            for j in 0..n_references {
                let reference = reference_series.row(j);
                let gak_value = self.compute_gak(&series_flat, &reference)?;
                features[[i, j]] = gak_value;
            }
        }

        Ok(features)
    }

    /// Compute Global Alignment Kernel
    fn compute_gak(&self, series1: &ArrayView1<f64>, series2: &ArrayView1<f64>) -> Result<f64> {
        let n1 = series1.len();
        let n2 = series2.len();

        // Initialize GAK matrix with exponential of negative squared distances
        let mut gak_matrix = Array2::zeros((n1 + 1, n2 + 1));

        for i in 1..=n1 {
            for j in 1..=n2 {
                let dist_sq = (series1[i - 1] - series2[j - 1]).powi(2);
                let kernel_val = (-dist_sq / (2.0 * self.sigma.powi(2))).exp();

                // GAK recurrence relation
                let max_alignment = vec![
                    gak_matrix[[i - 1, j]] * kernel_val,
                    gak_matrix[[i, j - 1]] * kernel_val,
                    gak_matrix[[i - 1, j - 1]] * kernel_val,
                ]
                .into_iter()
                .max_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);

                gak_matrix[[i, j]] = max_alignment;
            }
        }

        Ok(gak_matrix[[n1, n2]])
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    fn create_test_time_series() -> Array3<f64> {
        // Create simple test time series data
        let mut ts = Array3::zeros((5, 10, 2));

        for i in 0..5 {
            for t in 0..10 {
                ts[[i, t, 0]] = (t as f64 + i as f64).sin();
                ts[[i, t, 1]] = (t as f64 + i as f64).cos();
            }
        }

        ts
    }

    #[test]
    fn test_dtw_kernel_approximation() {
        let time_series = create_test_time_series();

        let mut dtw_kernel = DTWKernelApproximation::new(3)
            .bandwidth(1.0)
            .window_size(Some(2));

        dtw_kernel.fit(&time_series).unwrap();
        let features = dtw_kernel.transform(&time_series).unwrap();

        assert_eq!(features.shape(), &[5, 3]);

        // Features should be positive (RBF kernel values)
        assert!(features.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }

    #[test]
    fn test_autoregressive_kernel_approximation() {
        let time_series = create_test_time_series();

        let mut ar_kernel = AutoregressiveKernelApproximation::new(4, 2).lambda(0.1);

        ar_kernel.fit(&time_series).unwrap();
        let features = ar_kernel.transform(&time_series).unwrap();

        assert_eq!(features.shape(), &[5, 4]);
    }

    #[test]
    fn test_spectral_kernel_approximation() {
        let time_series = create_test_time_series();

        let mut spectral_kernel = SpectralKernelApproximation::new(6, 5).magnitude_only(true);

        spectral_kernel.fit(&time_series).unwrap();
        let features = spectral_kernel.transform(&time_series).unwrap();

        assert_eq!(features.shape(), &[5, 6]);
    }

    #[test]
    fn test_global_alignment_kernel() {
        let time_series = create_test_time_series();

        let mut gak = GlobalAlignmentKernelApproximation::new(3, 1.0);

        gak.fit(&time_series).unwrap();
        let features = gak.transform(&time_series).unwrap();

        assert_eq!(features.shape(), &[5, 3]);
        assert!(features.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_dtw_distance_computation() {
        let series1 = Array1::from(vec![1.0, 2.0, 3.0, 2.0, 1.0]);
        let series2 = Array1::from(vec![1.0, 3.0, 2.0, 1.0]);

        let dtw_kernel = DTWKernelApproximation::new(1);
        let distance = dtw_kernel
            .compute_dtw_distance(&series1.view(), &series2.view())
            .unwrap();

        assert!(distance >= 0.0);
        assert!(distance.is_finite());
    }

    #[test]
    fn test_ar_model_fitting() {
        let time_series = create_test_time_series();
        let series = time_series.slice(s![0, .., ..]);

        let ar_kernel = AutoregressiveKernelApproximation::new(10, 2);
        let coeffs = ar_kernel.fit_ar_model(&series, 2, 0.1).unwrap();

        assert_eq!(coeffs.len(), 4); // 2 lags * 2 features
    }

    #[test]
    fn test_frequency_features() {
        let time_series = create_test_time_series();
        let series = time_series.slice(s![0, .., ..]);

        let spectral_kernel = SpectralKernelApproximation::new(10, 5);
        let features = spectral_kernel
            .compute_frequency_features(&series, 5, true)
            .unwrap();

        assert_eq!(features.len(), 10); // 5 frequencies * 2 features
        assert!(features.iter().all(|&x| x >= 0.0)); // Magnitudes are non-negative
    }

    #[test]
    fn test_time_series_kernel_config() {
        let config = TimeSeriesKernelConfig::default();

        assert_eq!(config.n_components, 100);
        assert!(matches!(
            config.kernel_type,
            TimeSeriesKernelType::DTW { .. }
        ));
        assert!(config.normalize_series);
        assert!(config.dtw_config.is_some());
    }

    #[test]
    fn test_dtw_window_constraints() {
        let dtw_kernel = DTWKernelApproximation::new(1);
        let constraints = dtw_kernel.get_window_constraint(5, 5);

        assert!(constraints.is_none()); // No window constraint by default

        let window_constraint = Some(vec![(0, 0), (1, 1), (2, 2)]);
        assert!(dtw_kernel.is_within_window(1, 1, &window_constraint));
        assert!(!dtw_kernel.is_within_window(0, 2, &window_constraint));
    }

    #[test]
    fn test_reproducibility_with_random_state() {
        let time_series = create_test_time_series();

        let config1 = TimeSeriesKernelConfig {
            random_state: Some(42),
            ..Default::default()
        };
        let config2 = TimeSeriesKernelConfig {
            random_state: Some(42),
            ..Default::default()
        };

        let mut dtw1 = DTWKernelApproximation::new(3).with_config(config1);
        let mut dtw2 = DTWKernelApproximation::new(3).with_config(config2);

        dtw1.fit(&time_series).unwrap();
        dtw2.fit(&time_series).unwrap();

        let features1 = dtw1.transform(&time_series).unwrap();
        let features2 = dtw2.transform(&time_series).unwrap();

        // Should be approximately equal due to same random state
        for i in 0..features1.len() {
            assert!(
                (features1.as_slice().unwrap()[i] - features2.as_slice().unwrap()[i]).abs() < 1e-10
            );
        }
    }
}
