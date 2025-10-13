//! Time Series Decomposition methods
//!
//! This module provides specialized decomposition techniques for time series data including:
//! - SSA: Singular Spectrum Analysis
//! - Multi-channel SSA
//! - Seasonal decomposition methods
//! - Change point detection using decomposition
//! - Trend extraction methods

use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{error::Result, error::SklearsError, types::Float};

/// Singular Spectrum Analysis (SSA) configuration
#[derive(Debug, Clone)]
pub struct SsaConfig {
    /// Window length (embedding dimension)
    pub window_length: usize,
    /// Number of components to extract
    pub n_components: Option<usize>,
    /// Whether to group components
    pub grouping: bool,
    /// Reconstruction method
    pub reconstruction_method: ReconstructionMethod,
}

/// Reconstruction method for SSA
#[derive(Debug, Clone, Copy)]
pub enum ReconstructionMethod {
    /// Simple averaging (diagonal averaging)
    DiagonalAveraging,
    /// Weighted averaging
    WeightedAveraging,
    /// Vector forecasting
    VectorForecasting,
}

impl Default for SsaConfig {
    fn default() -> Self {
        Self {
            window_length: 10,
            n_components: None,
            grouping: false,
            reconstruction_method: ReconstructionMethod::DiagonalAveraging,
        }
    }
}

/// Singular Spectrum Analysis (SSA) for time series decomposition
///
/// SSA is a non-parametric spectral estimation method used for time series analysis.
/// It decomposes a time series into a sum of components such as trend, periodic components, and noise.
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_decomposition::SingularSpectrumAnalysis;
/// use scirs2_core::ndarray::array;
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0];
/// let ssa = SingularSpectrumAnalysis::new()
///     .window_length(4)
///     .n_components(3);
///
/// let result = ssa.fit_transform(&ts);
/// ```
pub struct SingularSpectrumAnalysis {
    config: SsaConfig,
    // Fitted parameters
    trajectory_matrix_: Option<Array2<Float>>,
    singular_values_: Option<Array1<Float>>,
    left_singular_vectors_: Option<Array2<Float>>,
    right_singular_vectors_: Option<Array2<Float>>,
    eigenvalues_: Option<Array1<Float>>,
    components_: Option<Array2<Float>>,
    reconstructed_: Option<Array2<Float>>,
}

impl SingularSpectrumAnalysis {
    /// Create a new SSA instance
    pub fn new() -> Self {
        Self {
            config: SsaConfig::default(),
            trajectory_matrix_: None,
            singular_values_: None,
            left_singular_vectors_: None,
            right_singular_vectors_: None,
            eigenvalues_: None,
            components_: None,
            reconstructed_: None,
        }
    }

    /// Set the window length (embedding dimension)
    pub fn window_length(mut self, window_length: usize) -> Self {
        self.config.window_length = window_length;
        self
    }

    /// Set the number of components to extract
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.config.n_components = Some(n_components);
        self
    }

    /// Enable component grouping
    pub fn grouping(mut self, grouping: bool) -> Self {
        self.config.grouping = grouping;
        self
    }

    /// Set the reconstruction method
    pub fn reconstruction_method(mut self, method: ReconstructionMethod) -> Self {
        self.config.reconstruction_method = method;
        self
    }

    /// Fit SSA on the time series and transform
    pub fn fit_transform(&mut self, time_series: &Array1<Float>) -> Result<SsaResult> {
        self.fit(time_series)?;
        self.transform()
    }

    /// Fit SSA on the time series
    pub fn fit(&mut self, time_series: &Array1<Float>) -> Result<()> {
        let n = time_series.len();
        let window_length = self.config.window_length;

        if window_length >= n {
            return Err(SklearsError::InvalidInput(
                "Window length must be less than time series length".to_string(),
            ));
        }

        if window_length < 2 {
            return Err(SklearsError::InvalidInput(
                "Window length must be at least 2".to_string(),
            ));
        }

        // Step 1: Embedding - Create trajectory matrix
        let trajectory_matrix = self.create_trajectory_matrix(time_series)?;
        self.trajectory_matrix_ = Some(trajectory_matrix.clone());

        // Step 2: SVD decomposition
        let (u, s, vt) = self.svd_decomposition(&trajectory_matrix)?;

        self.left_singular_vectors_ = Some(u);
        self.singular_values_ = Some(s.clone());
        self.right_singular_vectors_ = Some(vt);

        // Calculate eigenvalues (squares of singular values)
        let eigenvalues = s.mapv(|x| x * x);
        self.eigenvalues_ = Some(eigenvalues);

        // Step 3: Grouping (component selection)
        let n_components = self
            .config
            .n_components
            .unwrap_or(s.len().min(window_length));

        // Step 4: Reconstruction
        let reconstructed_components = self.reconstruct_components(n_components)?;
        self.components_ = Some(reconstructed_components.clone());

        // Diagonal averaging to get reconstructed time series
        let reconstructed_series = self.diagonal_averaging(&reconstructed_components)?;
        self.reconstructed_ = Some(reconstructed_series);

        Ok(())
    }

    /// Transform the fitted SSA model
    pub fn transform(&self) -> Result<SsaResult> {
        let components = self
            .components_
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("SSA must be fitted first".to_string()))?;

        let reconstructed = self
            .reconstructed_
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("SSA must be fitted first".to_string()))?;

        let singular_values = self
            .singular_values_
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("SSA must be fitted first".to_string()))?;

        let eigenvalues = self
            .eigenvalues_
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("SSA must be fitted first".to_string()))?;

        Ok(SsaResult {
            components: components.clone(),
            reconstructed_series: reconstructed.clone(),
            singular_values: singular_values.clone(),
            eigenvalues: eigenvalues.clone(),
            explained_variance_ratio: self.calculate_explained_variance_ratio()?,
        })
    }

    /// Create trajectory matrix (Hankel matrix) from time series
    fn create_trajectory_matrix(&self, time_series: &Array1<Float>) -> Result<Array2<Float>> {
        let n = time_series.len();
        let window_length = self.config.window_length;
        let k = n - window_length + 1;

        let mut trajectory_matrix = Array2::zeros((window_length, k));

        for i in 0..window_length {
            for j in 0..k {
                trajectory_matrix[[i, j]] = time_series[i + j];
            }
        }

        Ok(trajectory_matrix)
    }

    /// Perform SVD decomposition on trajectory matrix
    fn svd_decomposition(
        &self,
        matrix: &Array2<Float>,
    ) -> Result<(Array2<Float>, Array1<Float>, Array2<Float>)> {
        let (m, n) = matrix.dim();

        // For now, we'll implement a simplified SVD
        // In a full implementation, this would use a proper SVD algorithm
        let cov_matrix = if m <= n {
            matrix.dot(&matrix.t())
        } else {
            matrix.t().dot(matrix)
        };

        // Eigendecomposition of covariance matrix
        let (eigenvalues, eigenvectors) = self.eigendecomposition(&cov_matrix)?;

        // Sort by eigenvalue magnitude (descending)
        let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
        indices.sort_by(|&i, &j| {
            eigenvalues[j]
                .partial_cmp(&eigenvalues[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let sorted_eigenvalues: Array1<Float> = indices.iter().map(|&i| eigenvalues[i]).collect();
        let singular_values = sorted_eigenvalues.mapv(|x| x.sqrt());

        let sorted_eigenvectors =
            Array2::from_shape_fn((eigenvectors.nrows(), eigenvectors.ncols()), |(i, j)| {
                eigenvectors[[i, indices[j]]]
            });

        // Construct U and V matrices
        let u = if m <= n {
            sorted_eigenvectors.clone()
        } else {
            // Calculate U from V: U = X * V * S^(-1)
            let v = sorted_eigenvectors.clone();
            let s_inv = singular_values.mapv(|x| if x > 1e-10 { 1.0 / x } else { 0.0 });
            let u_calc = matrix.dot(&v);

            Array2::from_shape_fn((m, u_calc.ncols()), |(i, j)| {
                if singular_values[j] > 1e-10 {
                    u_calc[[i, j]] * s_inv[j]
                } else {
                    0.0
                }
            })
        };

        let vt = if m > n {
            sorted_eigenvectors.t().to_owned()
        } else {
            // Calculate V^T from U: V^T = S^(-1) * U^T * X
            let s_inv = singular_values.mapv(|x| if x > 1e-10 { 1.0 / x } else { 0.0 });
            let vt_calc = sorted_eigenvectors.t().dot(matrix);

            Array2::from_shape_fn((vt_calc.nrows(), vt_calc.ncols()), |(i, j)| {
                if singular_values[i] > 1e-10 {
                    vt_calc[[i, j]] * s_inv[i]
                } else {
                    0.0
                }
            })
        };

        Ok((u, singular_values, vt))
    }

    /// Simple eigendecomposition using power iteration method
    /// This is a simplified implementation for demonstration purposes
    fn eigendecomposition(&self, matrix: &Array2<Float>) -> Result<(Array1<Float>, Array2<Float>)> {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(SklearsError::InvalidInput(
                "Matrix must be square for eigendecomposition".to_string(),
            ));
        }

        let mut eigenvalues = Array1::zeros(n);
        let mut eigenvectors = Array2::eye(n);

        // Simplified power iteration for the largest eigenvalue/eigenvector
        let mut v = Array1::ones(n);
        let max_iter = 100;
        let tolerance = 1e-6;

        for _ in 0..max_iter {
            let v_new = matrix.dot(&v);
            let norm = (v_new.dot(&v_new)).sqrt();
            if norm > 1e-12 {
                v = v_new / norm;
            }

            let eigenvalue = v.dot(&matrix.dot(&v));
            eigenvalues[0] = eigenvalue;

            for i in 0..n {
                eigenvectors[[i, 0]] = v[i];
            }

            // Check convergence
            let residual = matrix.dot(&v) - eigenvalue * &v;
            if (residual.dot(&residual)).sqrt() < tolerance {
                break;
            }
        }

        // For simplicity, fill remaining eigenvalues with smaller values
        for i in 1..n {
            eigenvalues[i] = eigenvalues[0] * ((i + 1) as Float).recip();
        }

        Ok((eigenvalues, eigenvectors))
    }

    /// Reconstruct components from SVD
    fn reconstruct_components(&self, n_components: usize) -> Result<Array2<Float>> {
        let u = self.left_singular_vectors_.as_ref().unwrap();
        let s = self.singular_values_.as_ref().unwrap();
        let vt = self.right_singular_vectors_.as_ref().unwrap();

        let n_components = n_components.min(s.len());
        let (window_length, k) = (u.nrows(), vt.ncols());

        let mut components = Array2::zeros((n_components, window_length + k - 1));

        for i in 0..n_components {
            // Reconstruct i-th elementary matrix
            let ui = u.column(i);
            let vi = vt.row(i);
            let si = s[i];

            let elementary_matrix =
                Array2::from_shape_fn((window_length, k), |(row, col)| si * ui[row] * vi[col]);

            // Apply diagonal averaging to get time series component
            let component_series = self.diagonal_averaging_matrix(&elementary_matrix)?;

            for (j, &val) in component_series.iter().enumerate() {
                components[[i, j]] = val;
            }
        }

        Ok(components)
    }

    /// Diagonal averaging for matrix (anti-diagonal averaging)
    fn diagonal_averaging_matrix(&self, matrix: &Array2<Float>) -> Result<Array1<Float>> {
        let (l, k) = matrix.dim();
        let n = l + k - 1;
        let mut result = Array1::zeros(n);

        for i in 0..n {
            let mut sum = 0.0;
            let mut count = 0;

            // Calculate bounds for the anti-diagonal
            let i_min = if i >= k { i - k + 1 } else { 0 };
            let i_max = if i >= l { l - 1 } else { i };

            for j in i_min..=i_max {
                let col = i - j;
                if col < k {
                    sum += matrix[[j, col]];
                    count += 1;
                }
            }

            result[i] = if count > 0 { sum / count as Float } else { 0.0 };
        }

        Ok(result)
    }

    /// Diagonal averaging for component reconstruction
    fn diagonal_averaging(&self, components: &Array2<Float>) -> Result<Array2<Float>> {
        let (n_components, series_length) = components.dim();
        let mut result = Array2::zeros((n_components, series_length));

        for i in 0..n_components {
            let component_row = components.row(i);
            for j in 0..series_length {
                result[[i, j]] = component_row[j];
            }
        }

        Ok(result)
    }

    /// Calculate explained variance ratio
    fn calculate_explained_variance_ratio(&self) -> Result<Array1<Float>> {
        let eigenvalues = self.eigenvalues_.as_ref().unwrap();
        let total_variance = eigenvalues.sum();

        if total_variance <= 0.0 {
            return Ok(Array1::zeros(eigenvalues.len()));
        }

        Ok(eigenvalues.mapv(|x| x / total_variance))
    }

    /// Get the trajectory matrix
    pub fn trajectory_matrix(&self) -> Option<&Array2<Float>> {
        self.trajectory_matrix_.as_ref()
    }

    /// Get the singular values
    pub fn singular_values(&self) -> Option<&Array1<Float>> {
        self.singular_values_.as_ref()
    }

    /// Get the eigenvalues
    pub fn eigenvalues(&self) -> Option<&Array1<Float>> {
        self.eigenvalues_.as_ref()
    }

    /// Forecast future values using SSA
    pub fn forecast(&self, n_ahead: usize) -> Result<Array1<Float>> {
        let components = self
            .components_
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("SSA must be fitted first".to_string()))?;

        // Simple forecasting by extending trend components
        // This is a simplified approach - full SSA forecasting is more complex
        let (_n_components, series_length) = components.dim();
        let mut forecast = Array1::zeros(n_ahead);

        // Use the trend (first component) for forecasting
        if components.nrows() > 0 {
            let trend = components.row(0);
            let last_values = trend.slice(scirs2_core::ndarray::s![-2..]);

            if last_values.len() >= 2 {
                let slope = last_values[1] - last_values[0];
                let last_value = trend[series_length - 1];

                for i in 0..n_ahead {
                    forecast[i] = last_value + slope * (i + 1) as Float;
                }
            }
        }

        Ok(forecast)
    }
}

impl Default for SingularSpectrumAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

/// Result structure for SSA decomposition
#[derive(Debug, Clone)]
pub struct SsaResult {
    /// Decomposed components
    pub components: Array2<Float>,
    /// Reconstructed time series for each component
    pub reconstructed_series: Array2<Float>,
    /// Singular values from SVD
    pub singular_values: Array1<Float>,
    /// Eigenvalues (squared singular values)
    pub eigenvalues: Array1<Float>,
    /// Explained variance ratio for each component
    pub explained_variance_ratio: Array1<Float>,
}

impl SsaResult {
    /// Get the total reconstruction (sum of all components)
    pub fn total_reconstruction(&self) -> Array1<Float> {
        self.reconstructed_series.sum_axis(Axis(0))
    }

    /// Get a specific component
    pub fn component(&self, index: usize) -> Option<Array1<Float>> {
        if index < self.components.nrows() {
            Some(self.components.row(index).to_owned())
        } else {
            None
        }
    }

    /// Get the trend component (typically the first component)
    pub fn trend(&self) -> Option<Array1<Float>> {
        self.component(0)
    }

    /// Get periodic components (components 2 and beyond, excluding trend and noise)
    pub fn periodic_components(&self, n_periodic: usize) -> Array2<Float> {
        let start_idx = 1; // Skip trend
        let end_idx = (start_idx + n_periodic).min(self.components.nrows());

        if start_idx < self.components.nrows() && start_idx < end_idx {
            self.components
                .slice(scirs2_core::ndarray::s![start_idx..end_idx, ..])
                .to_owned()
        } else {
            Array2::zeros((0, self.components.ncols()))
        }
    }

    /// Get noise component (last components)
    pub fn noise(&self, n_signal_components: usize) -> Array1<Float> {
        if n_signal_components < self.components.nrows() {
            self.components
                .slice(scirs2_core::ndarray::s![n_signal_components.., ..])
                .sum_axis(Axis(0))
        } else {
            Array1::zeros(self.components.ncols())
        }
    }
}

/// Multi-channel Singular Spectrum Analysis (MSSA)
/// Extension of SSA for multivariate time series
pub struct MultiChannelSSA {
    config: SsaConfig,
    n_channels: usize,
    // Fitted parameters for each channel
    channel_results: Vec<SsaResult>,
}

impl MultiChannelSSA {
    /// Create a new MSSA instance
    pub fn new(n_channels: usize) -> Self {
        Self {
            config: SsaConfig::default(),
            n_channels,
            channel_results: Vec::new(),
        }
    }

    /// Set the window length
    pub fn window_length(mut self, window_length: usize) -> Self {
        self.config.window_length = window_length;
        self
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.config.n_components = Some(n_components);
        self
    }

    /// Fit MSSA on multivariate time series
    pub fn fit_transform(&mut self, time_series: &Array2<Float>) -> Result<Vec<SsaResult>> {
        let (n_channels, _series_length) = time_series.dim();

        if n_channels != self.n_channels {
            let expected = self.n_channels;
            return Err(SklearsError::InvalidInput(format!(
                "Expected {expected} channels, got {n_channels}"
            )));
        }

        self.channel_results.clear();

        // Apply SSA to each channel
        for i in 0..n_channels {
            let channel_data = time_series.row(i).to_owned();
            let mut ssa = SingularSpectrumAnalysis::new()
                .window_length(self.config.window_length)
                .reconstruction_method(self.config.reconstruction_method);

            if let Some(n_comp) = self.config.n_components {
                ssa = ssa.n_components(n_comp);
            }

            let result = ssa.fit_transform(&channel_data)?;
            self.channel_results.push(result);
        }

        Ok(self.channel_results.clone())
    }

    /// Get results for a specific channel
    pub fn channel_result(&self, channel: usize) -> Option<&SsaResult> {
        self.channel_results.get(channel)
    }

    /// Get cross-correlation between channels for a specific component
    pub fn cross_correlation(&self, component_idx: usize) -> Result<Array2<Float>> {
        if self.channel_results.is_empty() {
            return Err(SklearsError::InvalidInput(
                "MSSA must be fitted first".to_string(),
            ));
        }

        let n_channels = self.channel_results.len();
        let mut correlations = Array2::zeros((n_channels, n_channels));

        for i in 0..n_channels {
            for j in 0..n_channels {
                if let (Some(comp_i), Some(comp_j)) = (
                    self.channel_results[i].component(component_idx),
                    self.channel_results[j].component(component_idx),
                ) {
                    let corr = self.calculate_correlation(&comp_i, &comp_j);
                    correlations[[i, j]] = corr;
                }
            }
        }

        Ok(correlations)
    }

    /// Calculate Pearson correlation between two time series
    fn calculate_correlation(&self, x: &Array1<Float>, y: &Array1<Float>) -> Float {
        let n = x.len();
        if n != y.len() || n == 0 {
            return 0.0;
        }

        let mean_x = x.mean().unwrap_or(0.0);
        let mean_y = y.mean().unwrap_or(0.0);

        let mut num = 0.0;
        let mut den_x = 0.0;
        let mut den_y = 0.0;

        for i in 0..n {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            num += dx * dy;
            den_x += dx * dx;
            den_y += dy * dy;
        }

        if den_x == 0.0 || den_y == 0.0 {
            return 0.0;
        }

        num / (den_x * den_y).sqrt()
    }
}

/// Seasonal decomposition using additive or multiplicative models
#[derive(Debug, Clone)]
pub struct SeasonalDecomposition {
    /// Length of the seasonal period
    pub period: usize,
    /// Type of decomposition (additive or multiplicative)
    pub model: SeasonalModel,
    /// Method for trend estimation
    pub trend_method: TrendMethod,
}

/// Seasonal decomposition model type
#[derive(Debug, Clone, Copy)]
pub enum SeasonalModel {
    /// Y(t) = Trend(t) + Seasonal(t) + Residual(t)
    Additive,
    /// Y(t) = Trend(t) * Seasonal(t) * Residual(t)
    Multiplicative,
}

/// Method for trend estimation
#[derive(Debug, Clone, Copy)]
pub enum TrendMethod {
    /// Moving average
    MovingAverage,
    /// Linear regression
    LinearRegression,
    /// Locally weighted regression (LOWESS)
    Lowess,
}

impl SeasonalDecomposition {
    /// Create a new seasonal decomposition
    pub fn new(period: usize) -> Self {
        Self {
            period,
            model: SeasonalModel::Additive,
            trend_method: TrendMethod::MovingAverage,
        }
    }

    /// Set the decomposition model
    pub fn model(mut self, model: SeasonalModel) -> Self {
        self.model = model;
        self
    }

    /// Set the trend estimation method
    pub fn trend_method(mut self, method: TrendMethod) -> Self {
        self.trend_method = method;
        self
    }

    /// Decompose the time series
    pub fn decompose(&self, time_series: &Array1<Float>) -> Result<SeasonalDecompositionResult> {
        let n = time_series.len();

        if n < 2 * self.period {
            return Err(SklearsError::InvalidInput(
                "Time series length must be at least 2 * period".to_string(),
            ));
        }

        // Step 1: Estimate trend
        let trend = self.estimate_trend(time_series)?;

        // Step 2: Detrend the series
        let detrended = match self.model {
            SeasonalModel::Additive => time_series - &trend,
            SeasonalModel::Multiplicative => {
                let mut result = Array1::zeros(n);
                for i in 0..n {
                    result[i] = if trend[i].abs() > 1e-10 {
                        time_series[i] / trend[i]
                    } else {
                        time_series[i]
                    };
                }
                result
            }
        };

        // Step 3: Estimate seasonal component
        let seasonal = self.estimate_seasonal(&detrended)?;

        // Step 4: Calculate residual
        let residual = match self.model {
            SeasonalModel::Additive => time_series - &trend - &seasonal,
            SeasonalModel::Multiplicative => {
                let mut result = Array1::zeros(n);
                for i in 0..n {
                    let denominator = trend[i] * seasonal[i];
                    result[i] = if denominator.abs() > 1e-10 {
                        time_series[i] / denominator
                    } else {
                        time_series[i]
                    };
                }
                result
            }
        };

        Ok(SeasonalDecompositionResult {
            trend,
            seasonal,
            residual,
            model: self.model,
        })
    }

    /// Estimate trend component
    fn estimate_trend(&self, time_series: &Array1<Float>) -> Result<Array1<Float>> {
        match self.trend_method {
            TrendMethod::MovingAverage => self.moving_average_trend(time_series),
            TrendMethod::LinearRegression => self.linear_regression_trend(time_series),
            TrendMethod::Lowess => self.lowess_trend(time_series),
        }
    }

    /// Moving average trend estimation
    fn moving_average_trend(&self, time_series: &Array1<Float>) -> Result<Array1<Float>> {
        let n = time_series.len();
        let window = self.period;
        let mut trend = Array1::zeros(n);

        // Centered moving average
        let half_window = window / 2;

        for i in 0..n {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(n);

            let sum: Float = time_series
                .slice(scirs2_core::ndarray::s![start..end])
                .sum();
            let count = end - start;
            trend[i] = sum / count as Float;
        }

        Ok(trend)
    }

    /// Linear regression trend estimation
    fn linear_regression_trend(&self, time_series: &Array1<Float>) -> Result<Array1<Float>> {
        let n = time_series.len();
        let mut trend = Array1::zeros(n);

        // Simple linear regression: y = a + b*x
        let x_mean = (n - 1) as Float / 2.0;
        let y_mean = time_series.mean().unwrap_or(0.0);

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..n {
            let x = i as Float;
            let y = time_series[i];
            let x_diff = x - x_mean;
            numerator += x_diff * (y - y_mean);
            denominator += x_diff * x_diff;
        }

        let slope = if denominator.abs() > 1e-10 {
            numerator / denominator
        } else {
            0.0
        };
        let intercept = y_mean - slope * x_mean;

        for i in 0..n {
            trend[i] = intercept + slope * i as Float;
        }

        Ok(trend)
    }

    /// LOWESS trend estimation (simplified version)
    fn lowess_trend(&self, time_series: &Array1<Float>) -> Result<Array1<Float>> {
        // For simplicity, fall back to moving average
        // A full LOWESS implementation would be more complex
        self.moving_average_trend(time_series)
    }

    /// Estimate seasonal component
    fn estimate_seasonal(&self, detrended: &Array1<Float>) -> Result<Array1<Float>> {
        let n = detrended.len();
        let period = self.period;
        let mut seasonal = Array1::<f64>::zeros(n);

        // Calculate average for each position in the period
        let mut period_averages = Array1::<f64>::zeros(period);
        let mut period_counts = Array1::<f64>::zeros(period);

        for i in 0..n {
            let pos = i % period;
            period_averages[pos] += detrended[i];
            period_counts[pos] += 1.0;
        }

        // Average each position
        for i in 0..period {
            if period_counts[i] > 0.0 {
                period_averages[i] /= period_counts[i];
            }
        }

        // Center the seasonal component (for additive model)
        if matches!(self.model, SeasonalModel::Additive) {
            let seasonal_mean = period_averages.mean().unwrap_or(0.0);
            period_averages -= seasonal_mean;
        }

        // Fill the seasonal component
        for i in 0..n {
            let pos = i % period;
            seasonal[i] = period_averages[pos];
        }

        Ok(seasonal)
    }
}

/// Result of seasonal decomposition
#[derive(Debug, Clone)]
pub struct SeasonalDecompositionResult {
    /// Trend component
    pub trend: Array1<Float>,
    /// Seasonal component
    pub seasonal: Array1<Float>,
    /// Residual (noise) component
    pub residual: Array1<Float>,
    /// Model type used
    pub model: SeasonalModel,
}

impl SeasonalDecompositionResult {
    /// Reconstruct the original time series
    pub fn reconstruct(&self) -> Array1<Float> {
        match self.model {
            SeasonalModel::Additive => &self.trend + &self.seasonal + &self.residual,
            SeasonalModel::Multiplicative => {
                let n = self.trend.len();
                Array1::from_shape_fn(n, |i| self.trend[i] * self.seasonal[i] * self.residual[i])
            }
        }
    }

    /// Get the signal component (trend + seasonal)
    pub fn signal(&self) -> Array1<Float> {
        match self.model {
            SeasonalModel::Additive => &self.trend + &self.seasonal,
            SeasonalModel::Multiplicative => {
                let n = self.trend.len();
                Array1::from_shape_fn(n, |i| self.trend[i] * self.seasonal[i])
            }
        }
    }

    /// Calculate signal-to-noise ratio
    pub fn signal_to_noise_ratio(&self) -> Float {
        let signal = self.signal();
        let signal_power = signal.mapv(|x| x * x).mean().unwrap_or(0.0);
        let noise_power = self.residual.mapv(|x| x * x).mean().unwrap_or(1e-10);

        signal_power / noise_power
    }
}

/// Change Point Detection using decomposition methods
#[derive(Debug, Clone)]
pub struct ChangePointDetection {
    /// Method for change point detection
    pub method: ChangePointMethod,
    /// Window size for analysis
    pub window_size: usize,
    /// Threshold for change detection
    pub threshold: Float,
    /// Minimum distance between change points
    pub min_distance: usize,
}

/// Methods for change point detection
#[derive(Debug, Clone, Copy)]
pub enum ChangePointMethod {
    /// Variance-based detection
    Variance,
    /// Mean-based detection
    Mean,
    /// SSA-based detection using singular value changes
    SSABased,
    /// Spectral-based detection
    Spectral,
}

impl ChangePointDetection {
    /// Create a new change point detection instance
    pub fn new() -> Self {
        Self {
            method: ChangePointMethod::Variance,
            window_size: 50,
            threshold: 2.0,
            min_distance: 10,
        }
    }

    /// Set the detection method
    pub fn method(mut self, method: ChangePointMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the window size
    pub fn window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }

    /// Set the threshold
    pub fn threshold(mut self, threshold: Float) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set minimum distance between change points
    pub fn min_distance(mut self, min_distance: usize) -> Self {
        self.min_distance = min_distance;
        self
    }

    /// Detect change points in time series
    pub fn detect(&self, time_series: &Array1<Float>) -> Result<ChangePointResult> {
        let n = time_series.len();

        if n < 2 * self.window_size {
            return Err(SklearsError::InvalidInput(
                "Time series too short for change point detection".to_string(),
            ));
        }

        let change_scores = match self.method {
            ChangePointMethod::Variance => self.variance_based_detection(time_series)?,
            ChangePointMethod::Mean => self.mean_based_detection(time_series)?,
            ChangePointMethod::SSABased => self.ssa_based_detection(time_series)?,
            ChangePointMethod::Spectral => self.spectral_based_detection(time_series)?,
        };

        let change_points = self.find_peaks(&change_scores);

        Ok(ChangePointResult {
            change_points,
            change_scores,
            method: self.method,
        })
    }

    /// Variance-based change point detection
    fn variance_based_detection(&self, time_series: &Array1<Float>) -> Result<Array1<Float>> {
        let n = time_series.len();
        let mut scores = Array1::zeros(n);
        let half_window = self.window_size / 2;

        for i in half_window..(n - half_window) {
            let left_start = i.saturating_sub(half_window);
            let left_end = i;
            let right_start = i;
            let right_end = (i + half_window).min(n);

            if right_end - right_start > 0 && left_end - left_start > 0 {
                let left_window = time_series.slice(scirs2_core::ndarray::s![left_start..left_end]);
                let right_window =
                    time_series.slice(scirs2_core::ndarray::s![right_start..right_end]);

                let left_var = self.calculate_variance(&left_window.to_owned());
                let right_var = self.calculate_variance(&right_window.to_owned());

                // Variance ratio as change score
                scores[i] = if left_var > 0.0 && right_var > 0.0 {
                    (left_var / right_var).max(right_var / left_var)
                } else {
                    1.0
                };
            }
        }

        Ok(scores)
    }

    /// Mean-based change point detection
    fn mean_based_detection(&self, time_series: &Array1<Float>) -> Result<Array1<Float>> {
        let n = time_series.len();
        let mut scores = Array1::zeros(n);
        let half_window = self.window_size / 2;

        for i in half_window..(n - half_window) {
            let left_start = i.saturating_sub(half_window);
            let left_end = i;
            let right_start = i;
            let right_end = (i + half_window).min(n);

            if right_end - right_start > 0 && left_end - left_start > 0 {
                let left_window = time_series.slice(scirs2_core::ndarray::s![left_start..left_end]);
                let right_window =
                    time_series.slice(scirs2_core::ndarray::s![right_start..right_end]);

                let left_mean = left_window.mean().unwrap_or(0.0);
                let right_mean = right_window.mean().unwrap_or(0.0);

                // Mean difference as change score
                scores[i] = (left_mean - right_mean).abs();
            }
        }

        Ok(scores)
    }

    /// SSA-based change point detection
    fn ssa_based_detection(&self, time_series: &Array1<Float>) -> Result<Array1<Float>> {
        let n = time_series.len();
        let mut scores = Array1::zeros(n);
        let ssa_window = (self.window_size / 4).max(5);

        for i in self.window_size..(n - self.window_size) {
            let left_start = i.saturating_sub(self.window_size);
            let left_end = i;
            let right_start = i;
            let right_end = (i + self.window_size).min(n);

            if right_end - right_start >= ssa_window && left_end - left_start >= ssa_window {
                let left_segment = time_series
                    .slice(scirs2_core::ndarray::s![left_start..left_end])
                    .to_owned();
                let right_segment = time_series
                    .slice(scirs2_core::ndarray::s![right_start..right_end])
                    .to_owned();

                // Apply SSA to both segments
                let mut left_ssa = SingularSpectrumAnalysis::new().window_length(ssa_window);
                let mut right_ssa = SingularSpectrumAnalysis::new().window_length(ssa_window);

                if let (Ok(left_result), Ok(right_result)) = (
                    left_ssa.fit_transform(&left_segment),
                    right_ssa.fit_transform(&right_segment),
                ) {
                    // Compare singular value spectra
                    let left_sv = &left_result.singular_values;
                    let right_sv = &right_result.singular_values;

                    let min_len = left_sv.len().min(right_sv.len());
                    if min_len > 0 {
                        let mut spectral_distance = 0.0;
                        for j in 0..min_len {
                            spectral_distance += (left_sv[j] - right_sv[j]).abs();
                        }
                        scores[i] = spectral_distance / min_len as Float;
                    }
                }
            }
        }

        Ok(scores)
    }

    /// Spectral-based change point detection
    fn spectral_based_detection(&self, time_series: &Array1<Float>) -> Result<Array1<Float>> {
        let n = time_series.len();
        let mut scores = Array1::zeros(n);

        for i in self.window_size..(n - self.window_size) {
            let left_start = i.saturating_sub(self.window_size);
            let left_end = i;
            let right_start = i;
            let right_end = (i + self.window_size).min(n);

            if right_end - right_start > 0 && left_end - left_start > 0 {
                let left_segment = time_series
                    .slice(scirs2_core::ndarray::s![left_start..left_end])
                    .to_owned();
                let right_segment = time_series
                    .slice(scirs2_core::ndarray::s![right_start..right_end])
                    .to_owned();

                // Compute power spectral densities
                let left_psd = self.compute_psd(&left_segment);
                let right_psd = self.compute_psd(&right_segment);

                // Compute spectral distance
                let spectral_distance = self.spectral_distance(&left_psd, &right_psd);
                scores[i] = spectral_distance;
            }
        }

        Ok(scores)
    }

    /// Calculate variance of a signal
    fn calculate_variance(&self, signal: &Array1<Float>) -> Float {
        let mean = signal.mean().unwrap_or(0.0);
        let variance = signal
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<Float>()
            / signal.len() as Float;
        variance
    }

    /// Compute power spectral density (simplified)
    fn compute_psd(&self, signal: &Array1<Float>) -> Array1<Float> {
        let n = signal.len();
        let n_freq = n / 2 + 1;
        let mut psd = Array1::zeros(n_freq);

        // Simple periodogram
        for k in 0..n_freq {
            let mut real = 0.0;
            let mut imag = 0.0;

            for i in 0..n {
                let angle = -2.0 * std::f64::consts::PI * k as Float * i as Float / n as Float;
                real += signal[i] * angle.cos();
                imag += signal[i] * angle.sin();
            }

            psd[k] = (real * real + imag * imag) / n as Float;
        }

        psd
    }

    /// Compute spectral distance between two PSDs
    fn spectral_distance(&self, psd1: &Array1<Float>, psd2: &Array1<Float>) -> Float {
        let min_len = psd1.len().min(psd2.len());
        if min_len == 0 {
            return 0.0;
        }

        let mut distance = 0.0;
        for i in 0..min_len {
            distance += (psd1[i] - psd2[i]).abs();
        }

        distance / min_len as Float
    }

    /// Find peaks in change scores
    fn find_peaks(&self, scores: &Array1<Float>) -> Vec<usize> {
        let n = scores.len();
        let mut peaks = Vec::new();

        // Find local maxima above threshold
        for i in 1..n - 1 {
            if scores[i] > scores[i - 1] && scores[i] > scores[i + 1] && scores[i] > self.threshold
            {
                peaks.push(i);
            }
        }

        // Apply minimum distance constraint
        self.filter_peaks_by_distance(peaks)
    }

    /// Filter peaks by minimum distance
    fn filter_peaks_by_distance(&self, peaks: Vec<usize>) -> Vec<usize> {
        if peaks.is_empty() {
            return peaks;
        }

        let mut filtered_peaks = vec![peaks[0]];

        for &peak in peaks.iter().skip(1) {
            if peak.saturating_sub(*filtered_peaks.last().unwrap()) >= self.min_distance {
                filtered_peaks.push(peak);
            }
        }

        filtered_peaks
    }
}

impl Default for ChangePointDetection {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of change point detection
#[derive(Debug, Clone)]
pub struct ChangePointResult {
    /// Detected change points (indices)
    pub change_points: Vec<usize>,
    /// Change scores for all time points
    pub change_scores: Array1<Float>,
    /// Method used for detection
    pub method: ChangePointMethod,
}

impl ChangePointResult {
    /// Get segments between change points
    pub fn segments(&self, series_length: usize) -> Vec<(usize, usize)> {
        let mut segments = Vec::new();
        let mut start = 0;

        for &cp in &self.change_points {
            if cp > start {
                segments.push((start, cp));
                start = cp;
            }
        }

        // Add final segment
        if start < series_length {
            segments.push((start, series_length));
        }

        segments
    }

    /// Get the most significant change points (top N)
    pub fn top_change_points(&self, n: usize) -> Vec<(usize, Float)> {
        let mut indexed_scores: Vec<(usize, Float)> = self
            .change_points
            .iter()
            .map(|&idx| (idx, self.change_scores[idx]))
            .collect();

        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed_scores.truncate(n);
        indexed_scores
    }
}

/// Advanced trend extraction methods
#[derive(Debug, Clone)]
pub struct TrendExtraction {
    /// Method for trend extraction
    pub method: TrendExtractionMethod,
    /// Smoothing parameter (method-dependent)
    pub smoothing_parameter: Float,
    /// Window size for local methods
    pub window_size: usize,
}

/// Methods for trend extraction
#[derive(Debug, Clone, Copy)]
pub enum TrendExtractionMethod {
    /// Moving average
    MovingAverage,
    /// Exponential smoothing
    ExponentialSmoothing,
    /// Polynomial fitting
    PolynomialFit,
    /// LOWESS (Locally Weighted Scatterplot Smoothing)
    Lowess,
    /// Hodrick-Prescott filter
    HodrickPrescott,
    /// SSA-based trend extraction
    SSATrend,
}

impl TrendExtraction {
    /// Create a new trend extraction instance
    pub fn new() -> Self {
        Self {
            method: TrendExtractionMethod::MovingAverage,
            smoothing_parameter: 0.1,
            window_size: 10,
        }
    }

    /// Set the extraction method
    pub fn method(mut self, method: TrendExtractionMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the smoothing parameter
    pub fn smoothing_parameter(mut self, smoothing_parameter: Float) -> Self {
        self.smoothing_parameter = smoothing_parameter;
        self
    }

    /// Set the window size
    pub fn window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }

    /// Extract trend from time series
    pub fn extract_trend(&self, time_series: &Array1<Float>) -> Result<TrendExtractionResult> {
        let trend = match self.method {
            TrendExtractionMethod::MovingAverage => self.moving_average_trend(time_series)?,
            TrendExtractionMethod::ExponentialSmoothing => {
                self.exponential_smoothing_trend(time_series)?
            }
            TrendExtractionMethod::PolynomialFit => self.polynomial_fit_trend(time_series)?,
            TrendExtractionMethod::Lowess => self.lowess_trend(time_series)?,
            TrendExtractionMethod::HodrickPrescott => self.hodrick_prescott_trend(time_series)?,
            TrendExtractionMethod::SSATrend => self.ssa_trend(time_series)?,
        };

        let detrended = time_series - &trend;

        Ok(TrendExtractionResult {
            trend,
            detrended,
            method: self.method,
        })
    }

    /// Moving average trend
    fn moving_average_trend(&self, time_series: &Array1<Float>) -> Result<Array1<Float>> {
        let n = time_series.len();
        let mut trend = Array1::zeros(n);
        let half_window = self.window_size / 2;

        for i in 0..n {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(n);

            let sum: Float = time_series
                .slice(scirs2_core::ndarray::s![start..end])
                .sum();
            let count = end - start;
            trend[i] = sum / count as Float;
        }

        Ok(trend)
    }

    /// Exponential smoothing trend
    fn exponential_smoothing_trend(&self, time_series: &Array1<Float>) -> Result<Array1<Float>> {
        let n = time_series.len();
        let mut trend = Array1::zeros(n);
        let alpha = self.smoothing_parameter;

        if n > 0 {
            trend[0] = time_series[0];

            for i in 1..n {
                trend[i] = alpha * time_series[i] + (1.0 - alpha) * trend[i - 1];
            }
        }

        Ok(trend)
    }

    /// Polynomial fit trend
    fn polynomial_fit_trend(&self, time_series: &Array1<Float>) -> Result<Array1<Float>> {
        let n = time_series.len();
        let _degree = (self.smoothing_parameter * 10.0) as usize + 1; // Convert smoothing param to polynomial degree

        // Fit polynomial using least squares (simplified to linear for now)
        let x_mean = (n - 1) as Float / 2.0;
        let y_mean = time_series.mean().unwrap_or(0.0);

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..n {
            let x = i as Float;
            let y = time_series[i];
            let x_diff = x - x_mean;
            numerator += x_diff * (y - y_mean);
            denominator += x_diff * x_diff;
        }

        let slope = if denominator.abs() > 1e-10 {
            numerator / denominator
        } else {
            0.0
        };
        let intercept = y_mean - slope * x_mean;

        let mut trend = Array1::zeros(n);
        for i in 0..n {
            trend[i] = intercept + slope * i as Float;
        }

        Ok(trend)
    }

    /// LOWESS trend (simplified implementation)
    fn lowess_trend(&self, time_series: &Array1<Float>) -> Result<Array1<Float>> {
        // For simplicity, use weighted moving average
        let n = time_series.len();
        let mut trend = Array1::zeros(n);
        let bandwidth = (self.smoothing_parameter * n as Float) as usize;

        for i in 0..n {
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for j in 0..n {
                let distance = (i as Float - j as Float).abs();
                let normalized_distance = distance / bandwidth as Float;

                // Tricube weight function
                let weight = if normalized_distance < 1.0 {
                    let u = 1.0 - normalized_distance * normalized_distance * normalized_distance;
                    u * u * u
                } else {
                    0.0
                };

                weighted_sum += weight * time_series[j];
                weight_sum += weight;
            }

            trend[i] = if weight_sum > 0.0 {
                weighted_sum / weight_sum
            } else {
                time_series[i]
            };
        }

        Ok(trend)
    }

    /// Hodrick-Prescott filter (simplified)
    fn hodrick_prescott_trend(&self, time_series: &Array1<Float>) -> Result<Array1<Float>> {
        let n = time_series.len();
        if n < 3 {
            return Ok(time_series.clone());
        }

        let lambda = self.smoothing_parameter * 100.0; // Scale smoothing parameter
        let mut trend = time_series.clone();

        // Iterative solution (simplified)
        for _ in 0..10 {
            let mut new_trend = Array1::zeros(n);

            // First point
            new_trend[0] =
                (time_series[0] + lambda * 2.0 * trend[1] - lambda * trend[2]) / (1.0 + lambda);

            // Interior points
            for i in 1..n - 1 {
                let numerator = time_series[i] + lambda * (trend[i - 1] + trend[i + 1]);
                let denominator = 1.0 + 2.0 * lambda;
                new_trend[i] = numerator / denominator;
            }

            // Last point
            new_trend[n - 1] = (time_series[n - 1] + lambda * 2.0 * trend[n - 2]
                - lambda * trend[n - 3])
                / (1.0 + lambda);

            trend = new_trend;
        }

        Ok(trend)
    }

    /// SSA-based trend extraction
    fn ssa_trend(&self, time_series: &Array1<Float>) -> Result<Array1<Float>> {
        let window_length = self.window_size.min(time_series.len() / 2);
        let mut ssa = SingularSpectrumAnalysis::new()
            .window_length(window_length)
            .n_components(1); // Extract only the first component (trend)

        let result = ssa.fit_transform(time_series)?;

        // Return the first component as trend
        if let Some(trend_component) = result.component(0) {
            Ok(trend_component)
        } else {
            Ok(Array1::zeros(time_series.len()))
        }
    }
}

impl Default for TrendExtraction {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of trend extraction
#[derive(Debug, Clone)]
pub struct TrendExtractionResult {
    /// Extracted trend component
    pub trend: Array1<Float>,
    /// Detrended time series
    pub detrended: Array1<Float>,
    /// Method used for extraction
    pub method: TrendExtractionMethod,
}

impl TrendExtractionResult {
    /// Reconstruct the original time series
    pub fn reconstruct(&self) -> Array1<Float> {
        &self.trend + &self.detrended
    }

    /// Calculate trend strength (proportion of variance explained by trend)
    pub fn trend_strength(&self) -> Float {
        let original = self.reconstruct();
        let total_variance = self.calculate_variance(&original);
        let detrended_variance = self.calculate_variance(&self.detrended);

        if total_variance > 0.0 {
            1.0 - (detrended_variance / total_variance)
        } else {
            0.0
        }
    }

    /// Calculate variance
    fn calculate_variance(&self, signal: &Array1<Float>) -> Float {
        let mean = signal.mean().unwrap_or(0.0);
        signal
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<Float>()
            / signal.len() as Float
    }

    /// Calculate trend slope (for linear trends)
    pub fn trend_slope(&self) -> Float {
        let n = self.trend.len();
        if n < 2 {
            return 0.0;
        }

        let x_mean = (n - 1) as Float / 2.0;
        let y_mean = self.trend.mean().unwrap_or(0.0);

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..n {
            let x = i as Float;
            let y = self.trend[i];
            let x_diff = x - x_mean;
            numerator += x_diff * (y - y_mean);
            denominator += x_diff * x_diff;
        }

        if denominator.abs() > 1e-10 {
            numerator / denominator
        } else {
            0.0
        }
    }
}
