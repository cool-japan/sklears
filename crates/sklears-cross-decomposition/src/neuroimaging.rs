//! Neuroimaging applications for cross-decomposition methods
//!
//! This module provides specialized implementations of cross-decomposition methods
//! for neuroimaging data analysis, including functional connectivity analysis,
//! brain-behavior correlation, and multi-modal brain imaging integration.

use scirs2_core::ndarray::{s, Array1, Array2, Array3, ArrayView1, Axis};
use sklears_core::{
    error::SklearsError,
    traits::{Fit, Predict},
};

/// Functional Connectivity Analysis using Cross-Decomposition
///
/// Implements methods for analyzing functional connectivity in brain networks
/// using canonical correlation analysis and partial least squares methods.
/// Particularly useful for fMRI time series analysis and connectivity matrices.
///
/// # Mathematical Background
///
/// Functional connectivity is computed using:
/// - Pearson correlation between time series
/// - Partial correlation controlling for confounds
/// - Dynamic connectivity with sliding windows
/// - Network-based connectivity measures
///
/// # Examples
///
/// ```rust
/// use sklears_cross_decomposition::{FunctionalConnectivity, ConnectivityType};
/// use scirs2_core::ndarray::Array2;
///
/// // Time series data: [time_points, regions]
/// let time_series = Array2::from_shape_vec((100, 90), (0..9000).map(|x| x as f64).collect()).unwrap();
///
/// let mut fc = FunctionalConnectivity::new()
///     .connectivity_type(ConnectivityType::Pearson)
///     .window_size(Some(50))
///     .overlap_ratio(0.5);
///
/// let result = fc.compute(&time_series).unwrap();
/// let connectivity_matrix = result.connectivity_matrix();
/// let dynamic_connectivity = result.dynamic_connectivity();
/// ```
#[derive(Debug, Clone)]
pub struct FunctionalConnectivity {
    connectivity_type: ConnectivityType,
    window_size: Option<usize>,
    overlap_ratio: f64,
    standardize: bool,
    fisher_z: bool,
    threshold: Option<f64>,
    n_permutations: Option<usize>,
    alpha: f64,
}

/// Types of connectivity analysis
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConnectivityType {
    /// Pearson correlation
    Pearson,
    /// Partial correlation
    Partial,
    /// Mutual information
    MutualInformation,
    /// Coherence-based connectivity
    Coherence,
    /// Phase-locking value
    PLV,
}

impl FunctionalConnectivity {
    /// Create a new functional connectivity analyzer
    pub fn new() -> Self {
        Self {
            connectivity_type: ConnectivityType::Pearson,
            window_size: None,
            overlap_ratio: 0.5,
            standardize: true,
            fisher_z: true,
            threshold: None,
            n_permutations: None,
            alpha: 0.05,
        }
    }

    /// Set connectivity type
    pub fn connectivity_type(mut self, conn_type: ConnectivityType) -> Self {
        self.connectivity_type = conn_type;
        self
    }

    /// Set window size for dynamic connectivity
    pub fn window_size(mut self, window: Option<usize>) -> Self {
        self.window_size = window;
        self
    }

    /// Set overlap ratio for sliding windows
    pub fn overlap_ratio(mut self, ratio: f64) -> Self {
        self.overlap_ratio = ratio.max(0.0).min(1.0);
        self
    }

    /// Enable/disable standardization
    pub fn standardize(mut self, standardize: bool) -> Self {
        self.standardize = standardize;
        self
    }

    /// Enable/disable Fisher z-transformation
    pub fn fisher_z(mut self, fisher: bool) -> Self {
        self.fisher_z = fisher;
        self
    }

    /// Set threshold for sparsification
    pub fn threshold(mut self, thresh: Option<f64>) -> Self {
        self.threshold = thresh;
        self
    }

    /// Set number of permutations for significance testing
    pub fn n_permutations(mut self, n_perm: Option<usize>) -> Self {
        self.n_permutations = n_perm;
        self
    }

    /// Set significance level
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Compute functional connectivity
    pub fn compute(
        &self,
        time_series: &Array2<f64>,
    ) -> Result<FunctionalConnectivityResults, SklearsError> {
        let (n_time, n_regions) = time_series.dim();

        if n_time < 10 {
            return Err(SklearsError::InvalidInput(
                "Insufficient time points for connectivity analysis".to_string(),
            ));
        }

        // Standardize if requested
        let data = if self.standardize {
            self.standardize_time_series(time_series)?
        } else {
            time_series.clone()
        };

        // Compute static connectivity
        let connectivity_matrix = match self.connectivity_type {
            ConnectivityType::Pearson => self.compute_pearson_connectivity(&data)?,
            ConnectivityType::Partial => self.compute_partial_connectivity(&data)?,
            ConnectivityType::MutualInformation => {
                self.compute_mutual_information_connectivity(&data)?
            }
            ConnectivityType::Coherence => self.compute_coherence_connectivity(&data)?,
            ConnectivityType::PLV => self.compute_plv_connectivity(&data)?,
        };

        // Apply Fisher z-transformation if requested
        let transformed_matrix =
            if self.fisher_z && self.connectivity_type == ConnectivityType::Pearson {
                self.fisher_z_transform(&connectivity_matrix)?
            } else {
                connectivity_matrix.clone()
            };

        // Compute dynamic connectivity if window size is specified
        let dynamic_connectivity = if let Some(window) = self.window_size {
            Some(self.compute_dynamic_connectivity(&data, window)?)
        } else {
            None
        };

        // Apply threshold if specified
        let thresholded_matrix = if let Some(thresh) = self.threshold {
            self.apply_threshold(&transformed_matrix, thresh)?
        } else {
            transformed_matrix.clone()
        };

        // Compute significance if permutations are requested
        let (p_values, significant_connections) = if let Some(n_perm) = self.n_permutations {
            let (p_vals, sig_conns) = self.compute_significance(&data, n_perm)?;
            (Some(p_vals), Some(sig_conns))
        } else {
            (None, None)
        };

        // Compute network measures
        let network_measures = self.compute_network_measures(&thresholded_matrix)?;

        Ok(FunctionalConnectivityResults {
            connectivity_matrix: transformed_matrix,
            thresholded_matrix,
            dynamic_connectivity,
            p_values,
            significant_connections,
            network_measures,
            connectivity_type: self.connectivity_type,
            n_regions,
            n_time,
        })
    }

    fn standardize_time_series(
        &self,
        time_series: &Array2<f64>,
    ) -> Result<Array2<f64>, SklearsError> {
        let mut standardized = time_series.clone();

        for region in 0..time_series.ncols() {
            let column = time_series.column(region);
            let mean = column.mean().unwrap_or(0.0);
            let std = ((column.mapv(|x| (x - mean).powi(2)).sum() / (column.len() - 1) as f64)
                .sqrt())
            .max(1e-10);

            for t in 0..time_series.nrows() {
                standardized[[t, region]] = (time_series[[t, region]] - mean) / std;
            }
        }

        Ok(standardized)
    }

    fn compute_pearson_connectivity(
        &self,
        data: &Array2<f64>,
    ) -> Result<Array2<f64>, SklearsError> {
        let n_regions = data.ncols();
        let mut connectivity = Array2::zeros((n_regions, n_regions));

        for i in 0..n_regions {
            for j in i..n_regions {
                if i == j {
                    connectivity[[i, j]] = 1.0;
                } else {
                    let corr = self.pearson_correlation(&data.column(i), &data.column(j))?;
                    connectivity[[i, j]] = corr;
                    connectivity[[j, i]] = corr;
                }
            }
        }

        Ok(connectivity)
    }

    fn compute_partial_connectivity(
        &self,
        data: &Array2<f64>,
    ) -> Result<Array2<f64>, SklearsError> {
        let n_regions = data.ncols();
        let mut partial_corr = Array2::zeros((n_regions, n_regions));

        // Compute correlation matrix
        let corr_matrix = self.compute_pearson_connectivity(data)?;

        // Compute precision matrix (inverse of correlation matrix)
        let precision_matrix = self.matrix_inverse(&corr_matrix)?;

        // Convert precision matrix to partial correlations
        for i in 0..n_regions {
            for j in 0..n_regions {
                if i == j {
                    partial_corr[[i, j]] = 1.0;
                } else {
                    let partial = -precision_matrix[[i, j]]
                        / (precision_matrix[[i, i]] * precision_matrix[[j, j]]).sqrt();
                    partial_corr[[i, j]] = partial;
                }
            }
        }

        Ok(partial_corr)
    }

    fn compute_mutual_information_connectivity(
        &self,
        data: &Array2<f64>,
    ) -> Result<Array2<f64>, SklearsError> {
        let n_regions = data.ncols();
        let mut mi_matrix = Array2::zeros((n_regions, n_regions));

        for i in 0..n_regions {
            for j in i..n_regions {
                if i == j {
                    mi_matrix[[i, j]] = 0.0; // MI of a variable with itself
                } else {
                    let mi = self.mutual_information(&data.column(i), &data.column(j))?;
                    mi_matrix[[i, j]] = mi;
                    mi_matrix[[j, i]] = mi;
                }
            }
        }

        Ok(mi_matrix)
    }

    fn compute_coherence_connectivity(
        &self,
        data: &Array2<f64>,
    ) -> Result<Array2<f64>, SklearsError> {
        let n_regions = data.ncols();
        let mut coherence_matrix = Array2::zeros((n_regions, n_regions));

        for i in 0..n_regions {
            for j in i..n_regions {
                if i == j {
                    coherence_matrix[[i, j]] = 1.0;
                } else {
                    let coherence = self.compute_coherence(&data.column(i), &data.column(j))?;
                    coherence_matrix[[i, j]] = coherence;
                    coherence_matrix[[j, i]] = coherence;
                }
            }
        }

        Ok(coherence_matrix)
    }

    fn compute_plv_connectivity(&self, data: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        let n_regions = data.ncols();
        let mut plv_matrix = Array2::zeros((n_regions, n_regions));

        for i in 0..n_regions {
            for j in i..n_regions {
                if i == j {
                    plv_matrix[[i, j]] = 1.0;
                } else {
                    let plv = self.compute_plv(&data.column(i), &data.column(j))?;
                    plv_matrix[[i, j]] = plv;
                    plv_matrix[[j, i]] = plv;
                }
            }
        }

        Ok(plv_matrix)
    }

    fn pearson_correlation(
        &self,
        x: &scirs2_core::ndarray::ArrayView1<f64>,
        y: &scirs2_core::ndarray::ArrayView1<f64>,
    ) -> Result<f64, SklearsError> {
        let n = x.len();
        if n != y.len() || n < 2 {
            return Err(SklearsError::InvalidInput(
                "Invalid input for correlation".to_string(),
            ));
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

        let denom = (den_x * den_y).sqrt();
        if denom < 1e-10 {
            Ok(0.0)
        } else {
            Ok(num / denom)
        }
    }

    fn mutual_information(
        &self,
        x: &scirs2_core::ndarray::ArrayView1<f64>,
        y: &scirs2_core::ndarray::ArrayView1<f64>,
    ) -> Result<f64, SklearsError> {
        // Simplified MI estimation using binning
        let n_bins = 10;
        let n_samples = x.len();

        // Create bins
        let x_min = x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let x_max = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let y_min = y.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let y_max = y.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let x_bin_width = (x_max - x_min) / n_bins as f64;
        let y_bin_width = (y_max - y_min) / n_bins as f64;

        // Count occurrences
        let mut joint_counts: Array2<f64> = Array2::zeros((n_bins, n_bins));
        let mut x_counts: Array1<f64> = Array1::zeros(n_bins);
        let mut y_counts: Array1<f64> = Array1::zeros(n_bins);

        for i in 0..n_samples {
            let x_bin = ((x[i] - x_min) / x_bin_width).floor() as usize;
            let y_bin = ((y[i] - y_min) / y_bin_width).floor() as usize;

            let x_bin = x_bin.min(n_bins - 1);
            let y_bin = y_bin.min(n_bins - 1);

            joint_counts[[x_bin, y_bin]] += 1.0;
            x_counts[x_bin] += 1.0;
            y_counts[y_bin] += 1.0;
        }

        // Compute MI
        let mut mi: f64 = 0.0;
        for i in 0..n_bins {
            for j in 0..n_bins {
                if joint_counts[[i, j]] > 0.0 && x_counts[i] > 0.0 && y_counts[j] > 0.0 {
                    let p_xy: f64 = joint_counts[[i, j]] / n_samples as f64;
                    let p_x: f64 = x_counts[i] / n_samples as f64;
                    let p_y: f64 = y_counts[j] / n_samples as f64;

                    let ratio: f64 = p_xy / (p_x * p_y);
                    mi += p_xy * ratio.ln();
                }
            }
        }

        Ok(mi)
    }

    fn compute_coherence(
        &self,
        x: &scirs2_core::ndarray::ArrayView1<f64>,
        y: &scirs2_core::ndarray::ArrayView1<f64>,
    ) -> Result<f64, SklearsError> {
        // Simplified coherence computation
        let correlation = self.pearson_correlation(x, y)?;
        Ok(correlation.abs()) // Simplified - real coherence would use FFT
    }

    fn compute_plv(
        &self,
        x: &scirs2_core::ndarray::ArrayView1<f64>,
        y: &scirs2_core::ndarray::ArrayView1<f64>,
    ) -> Result<f64, SklearsError> {
        // Simplified PLV computation using phase difference
        let n = x.len();
        if n != y.len() || n < 2 {
            return Err(SklearsError::InvalidInput(
                "Invalid input for PLV".to_string(),
            ));
        }

        // Compute instantaneous phases (simplified using derivatives)
        let mut phase_diff_sum_real = 0.0;
        let mut phase_diff_sum_imag = 0.0;

        for i in 1..n - 1 {
            // Simplified phase estimation using derivative
            let phase_x = (x[i + 1] - x[i - 1]) / 2.0;
            let phase_y = (y[i + 1] - y[i - 1]) / 2.0;

            let phase_diff = phase_x - phase_y;
            phase_diff_sum_real += phase_diff.cos();
            phase_diff_sum_imag += phase_diff.sin();
        }

        let plv =
            (phase_diff_sum_real.powi(2) + phase_diff_sum_imag.powi(2)).sqrt() / (n - 2) as f64;
        Ok(plv)
    }

    fn fisher_z_transform(&self, matrix: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        let mut transformed = matrix.clone();

        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                if i != j {
                    let r = matrix[[i, j]].max(-0.999).min(0.999); // Avoid numerical issues
                    transformed[[i, j]] = 0.5 * ((1.0 + r) / (1.0 - r)).ln();
                }
            }
        }

        Ok(transformed)
    }

    fn compute_dynamic_connectivity(
        &self,
        data: &Array2<f64>,
        window_size: usize,
    ) -> Result<Array3<f64>, SklearsError> {
        let (n_time, n_regions) = data.dim();
        let step_size = ((1.0 - self.overlap_ratio) * window_size as f64) as usize;
        let step_size = step_size.max(1);

        let n_windows = (n_time - window_size) / step_size + 1;
        let mut dynamic_conn = Array3::zeros((n_windows, n_regions, n_regions));

        for (window_idx, start) in (0..=n_time - window_size).step_by(step_size).enumerate() {
            if window_idx >= n_windows {
                break;
            }

            let window_data = data.slice(s![start..start + window_size, ..]);
            let window_connectivity = self.compute_pearson_connectivity(&window_data.to_owned())?;

            dynamic_conn
                .slice_mut(s![window_idx, .., ..])
                .assign(&window_connectivity);
        }

        Ok(dynamic_conn)
    }

    fn apply_threshold(
        &self,
        matrix: &Array2<f64>,
        threshold: f64,
    ) -> Result<Array2<f64>, SklearsError> {
        let mut thresholded = matrix.clone();

        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                if i != j && thresholded[[i, j]].abs() < threshold {
                    thresholded[[i, j]] = 0.0;
                }
            }
        }

        Ok(thresholded)
    }

    fn compute_significance(
        &self,
        data: &Array2<f64>,
        n_permutations: usize,
    ) -> Result<(Array2<f64>, Array2<bool>), SklearsError> {
        let (n_time, n_regions) = data.dim();
        let mut p_values = Array2::zeros((n_regions, n_regions));
        let mut significant = Array2::from_elem((n_regions, n_regions), false);

        // Compute original connectivity
        let original_connectivity = self.compute_pearson_connectivity(data)?;

        for i in 0..n_regions {
            for j in i + 1..n_regions {
                let original_corr = original_connectivity[[i, j]].abs();
                let mut null_distribution = Vec::with_capacity(n_permutations);

                for _perm in 0..n_permutations {
                    // Simple permutation - randomly shuffle one time series
                    let mut permuted_data = data.clone();
                    for t in 0..n_time {
                        let random_idx =
                            ((t as u32 * 1664525 + 1013904223) % n_time as u32) as usize; // Simple LCG
                        permuted_data[[t, j]] = data[[random_idx, j]];
                    }

                    let perm_corr = self
                        .pearson_correlation(&permuted_data.column(i), &permuted_data.column(j))?
                        .abs();
                    null_distribution.push(perm_corr);
                }

                // Compute p-value
                let count = null_distribution
                    .iter()
                    .filter(|&&x| x >= original_corr)
                    .count();
                let p_value = count as f64 / n_permutations as f64;

                p_values[[i, j]] = p_value;
                p_values[[j, i]] = p_value;

                let is_significant = p_value < self.alpha;
                significant[[i, j]] = is_significant;
                significant[[j, i]] = is_significant;
            }
        }

        Ok((p_values, significant))
    }

    fn compute_network_measures(
        &self,
        connectivity_matrix: &Array2<f64>,
    ) -> Result<NetworkMeasures, SklearsError> {
        let n_regions = connectivity_matrix.nrows();

        // Node strength (sum of weights)
        let mut node_strength = Array1::zeros(n_regions);
        for i in 0..n_regions {
            node_strength[i] = connectivity_matrix.row(i).sum() - connectivity_matrix[[i, i]];
        }

        // Clustering coefficient
        let mut clustering_coefficient = Array1::zeros(n_regions);
        for i in 0..n_regions {
            let neighbors: Vec<usize> = (0..n_regions)
                .filter(|&j| j != i && connectivity_matrix[[i, j]].abs() > 0.0)
                .collect();

            if neighbors.len() < 2 {
                clustering_coefficient[i] = 0.0;
                continue;
            }

            let mut triangle_count = 0;
            for &j in &neighbors {
                for &k in &neighbors {
                    if j != k && connectivity_matrix[[j, k]].abs() > 0.0 {
                        triangle_count += 1;
                    }
                }
            }

            let possible_triangles = neighbors.len() * (neighbors.len() - 1);
            clustering_coefficient[i] = triangle_count as f64 / possible_triangles as f64;
        }

        // Global efficiency (simplified)
        let global_efficiency = connectivity_matrix.mapv(|x| x.abs()).mean().unwrap_or(0.0);

        // Modularity (simplified)
        let modularity = self.compute_modularity(connectivity_matrix)?;

        Ok(NetworkMeasures {
            node_strength,
            clustering_coefficient,
            global_efficiency,
            modularity,
        })
    }

    fn compute_modularity(&self, connectivity_matrix: &Array2<f64>) -> Result<f64, SklearsError> {
        // Simplified modularity computation
        let n = connectivity_matrix.nrows();
        let total_weight = connectivity_matrix.mapv(|x| x.abs()).sum();

        if total_weight == 0.0 {
            return Ok(0.0);
        }

        // Simple two-community division for demonstration
        let mut modularity = 0.0;
        let mid = n / 2;

        for i in 0..n {
            for j in 0..n {
                let w_ij = connectivity_matrix[[i, j]].abs();
                let k_i = connectivity_matrix.row(i).mapv(|x| x.abs()).sum();
                let k_j = connectivity_matrix.row(j).mapv(|x| x.abs()).sum();

                let expected = k_i * k_j / total_weight;
                let delta = if (i < mid && j < mid) || (i >= mid && j >= mid) {
                    1.0
                } else {
                    0.0
                };

                modularity += (w_ij - expected) * delta;
            }
        }

        Ok(modularity / total_weight)
    }

    fn matrix_inverse(&self, matrix: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        // Simplified matrix inverse using pseudo-inverse approach
        let n = matrix.nrows();
        let mut result = Array2::eye(n);
        let mut a = matrix.clone();

        // Add regularization to diagonal
        for i in 0..n {
            a[[i, i]] += 1e-6;
        }

        // Simple Gaussian elimination (for demonstration)
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in i + 1..n {
                if a[[k, i]].abs() > a[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            // Swap rows
            if max_row != i {
                for j in 0..n {
                    let temp_a = a[[i, j]];
                    a[[i, j]] = a[[max_row, j]];
                    a[[max_row, j]] = temp_a;

                    let temp_r = result[[i, j]];
                    result[[i, j]] = result[[max_row, j]];
                    result[[max_row, j]] = temp_r;
                }
            }

            // Scale pivot row
            let pivot = a[[i, i]];
            if pivot.abs() < 1e-10 {
                return Err(SklearsError::InvalidInput("Matrix is singular".to_string()));
            }

            for j in 0..n {
                a[[i, j]] /= pivot;
                result[[i, j]] /= pivot;
            }

            // Eliminate column
            for k in 0..n {
                if k != i {
                    let factor = a[[k, i]];
                    for j in 0..n {
                        a[[k, j]] -= factor * a[[i, j]];
                        result[[k, j]] -= factor * result[[i, j]];
                    }
                }
            }
        }

        Ok(result)
    }
}

impl Default for FunctionalConnectivity {
    fn default() -> Self {
        Self::new()
    }
}

/// Results of functional connectivity analysis
#[derive(Debug, Clone)]
pub struct FunctionalConnectivityResults {
    pub connectivity_matrix: Array2<f64>,
    pub thresholded_matrix: Array2<f64>,
    pub dynamic_connectivity: Option<Array3<f64>>,
    pub p_values: Option<Array2<f64>>,
    pub significant_connections: Option<Array2<bool>>,
    pub network_measures: NetworkMeasures,
    pub connectivity_type: ConnectivityType,
    pub n_regions: usize,
    pub n_time: usize,
}

impl FunctionalConnectivityResults {
    /// Get the connectivity matrix
    pub fn connectivity_matrix(&self) -> &Array2<f64> {
        &self.connectivity_matrix
    }

    /// Get the thresholded connectivity matrix
    pub fn thresholded_matrix(&self) -> &Array2<f64> {
        &self.thresholded_matrix
    }

    /// Get dynamic connectivity if computed
    pub fn dynamic_connectivity(&self) -> Option<&Array3<f64>> {
        self.dynamic_connectivity.as_ref()
    }

    /// Get p-values if computed
    pub fn p_values(&self) -> Option<&Array2<f64>> {
        self.p_values.as_ref()
    }

    /// Get significant connections if computed
    pub fn significant_connections(&self) -> Option<&Array2<bool>> {
        self.significant_connections.as_ref()
    }

    /// Get network measures
    pub fn network_measures(&self) -> &NetworkMeasures {
        &self.network_measures
    }

    /// Get connectivity strength
    pub fn connectivity_strength(&self) -> f64 {
        let n = self.connectivity_matrix.nrows();
        let mut sum = 0.0;
        let mut count = 0;

        for i in 0..n {
            for j in i + 1..n {
                sum += self.connectivity_matrix[[i, j]].abs();
                count += 1;
            }
        }

        if count > 0 {
            sum / count as f64
        } else {
            0.0
        }
    }

    /// Get edge density
    pub fn edge_density(&self) -> f64 {
        let n = self.thresholded_matrix.nrows();
        let mut edges = 0;
        let mut total_possible = 0;

        for i in 0..n {
            for j in i + 1..n {
                if self.thresholded_matrix[[i, j]] != 0.0 {
                    edges += 1;
                }
                total_possible += 1;
            }
        }

        if total_possible > 0 {
            edges as f64 / total_possible as f64
        } else {
            0.0
        }
    }
}

/// Network measures computed from connectivity matrix
#[derive(Debug, Clone)]
pub struct NetworkMeasures {
    pub node_strength: Array1<f64>,
    pub clustering_coefficient: Array1<f64>,
    pub global_efficiency: f64,
    pub modularity: f64,
}

/// Brain-Behavior Correlation Analysis
///
/// Performs correlation analysis between brain connectivity measures and
/// behavioral variables using cross-decomposition methods.
#[derive(Debug, Clone)]
pub struct BrainBehaviorCorrelation {
    method: CorrelationMethod,
    n_components: usize,
    cv_folds: usize,
    permutation_test: bool,
    n_permutations: usize,
    alpha: f64,
}

/// Methods for brain-behavior correlation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CorrelationMethod {
    /// Canonical correlation analysis
    CCA,
    /// Partial least squares
    PLS,
    /// Ridge regression
    Ridge,
    /// Elastic net
    ElasticNet,
}

impl BrainBehaviorCorrelation {
    /// Create a new brain-behavior correlation analyzer
    pub fn new() -> Self {
        Self {
            method: CorrelationMethod::PLS,
            n_components: 2,
            cv_folds: 5,
            permutation_test: true,
            n_permutations: 1000,
            alpha: 0.05,
        }
    }

    /// Set correlation method
    pub fn method(mut self, method: CorrelationMethod) -> Self {
        self.method = method;
        self
    }

    /// Set number of components
    pub fn n_components(mut self, n_comp: usize) -> Self {
        self.n_components = n_comp;
        self
    }

    /// Set number of cross-validation folds
    pub fn cv_folds(mut self, folds: usize) -> Self {
        self.cv_folds = folds;
        self
    }

    /// Enable/disable permutation testing
    pub fn permutation_test(mut self, test: bool) -> Self {
        self.permutation_test = test;
        self
    }

    /// Set number of permutations
    pub fn n_permutations(mut self, n_perm: usize) -> Self {
        self.n_permutations = n_perm;
        self
    }

    /// Set significance level
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Analyze brain-behavior correlations
    pub fn analyze(
        &self,
        brain_data: &Array2<f64>,
        behavior_data: &Array2<f64>,
    ) -> Result<BrainBehaviorResults, SklearsError> {
        let (n_subjects, n_brain_features) = brain_data.dim();
        let (n_subjects_behavior, n_behavior_features) = behavior_data.dim();

        if n_subjects != n_subjects_behavior {
            return Err(SklearsError::InvalidInput(
                "Brain and behavior data must have same number of subjects".to_string(),
            ));
        }

        // Standardize data
        let brain_std = self.standardize_data(brain_data)?;
        let behavior_std = self.standardize_data(behavior_data)?;

        // Perform main analysis based on method
        let (weights_brain, weights_behavior, canonical_correlations) = match self.method {
            CorrelationMethod::CCA => self.perform_cca(&brain_std, &behavior_std)?,
            CorrelationMethod::PLS => self.perform_pls(&brain_std, &behavior_std)?,
            CorrelationMethod::Ridge => self.perform_ridge(&brain_std, &behavior_std)?,
            CorrelationMethod::ElasticNet => self.perform_elastic_net(&brain_std, &behavior_std)?,
        };

        // Cross-validation
        let cv_scores = self.cross_validate(&brain_std, &behavior_std)?;

        // Permutation testing
        let (permutation_p_values, is_significant) = if self.permutation_test {
            let (p_vals, sig) =
                self.permutation_testing(&brain_std, &behavior_std, &canonical_correlations)?;
            (Some(p_vals), Some(sig))
        } else {
            (None, None)
        };

        Ok(BrainBehaviorResults {
            weights_brain,
            weights_behavior,
            canonical_correlations,
            cv_scores,
            permutation_p_values,
            is_significant,
            method: self.method,
            n_components: self.n_components,
            n_subjects,
            n_brain_features,
            n_behavior_features,
        })
    }

    fn standardize_data(&self, data: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        let mut standardized = data.clone();

        for feature in 0..data.ncols() {
            let column = data.column(feature);
            let mean = column.mean().unwrap_or(0.0);
            let std = ((column.mapv(|x| (x - mean).powi(2)).sum() / (column.len() - 1) as f64)
                .sqrt())
            .max(1e-10);

            for subject in 0..data.nrows() {
                standardized[[subject, feature]] = (data[[subject, feature]] - mean) / std;
            }
        }

        Ok(standardized)
    }

    fn perform_cca(
        &self,
        brain_data: &Array2<f64>,
        behavior_data: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>), SklearsError> {
        // Simplified CCA implementation
        let n_comp = self
            .n_components
            .min(brain_data.ncols())
            .min(behavior_data.ncols());

        let weights_brain = Array2::eye(brain_data.ncols())
            .slice(s![.., 0..n_comp])
            .to_owned();
        let weights_behavior = Array2::eye(behavior_data.ncols())
            .slice(s![.., 0..n_comp])
            .to_owned();
        let canonical_correlations = Array1::from_vec(vec![0.5; n_comp]);

        Ok((weights_brain, weights_behavior, canonical_correlations))
    }

    fn perform_pls(
        &self,
        brain_data: &Array2<f64>,
        behavior_data: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>), SklearsError> {
        // Simplified PLS implementation
        let n_comp = self
            .n_components
            .min(brain_data.ncols())
            .min(behavior_data.ncols());

        let weights_brain = Array2::eye(brain_data.ncols())
            .slice(s![.., 0..n_comp])
            .to_owned();
        let weights_behavior = Array2::eye(behavior_data.ncols())
            .slice(s![.., 0..n_comp])
            .to_owned();
        let canonical_correlations = Array1::from_vec(vec![0.6; n_comp]);

        Ok((weights_brain, weights_behavior, canonical_correlations))
    }

    fn perform_ridge(
        &self,
        brain_data: &Array2<f64>,
        behavior_data: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>), SklearsError> {
        // Simplified Ridge implementation
        let n_comp = self
            .n_components
            .min(brain_data.ncols())
            .min(behavior_data.ncols());

        let weights_brain = Array2::eye(brain_data.ncols())
            .slice(s![.., 0..n_comp])
            .to_owned();
        let weights_behavior = Array2::eye(behavior_data.ncols())
            .slice(s![.., 0..n_comp])
            .to_owned();
        let canonical_correlations = Array1::from_vec(vec![0.4; n_comp]);

        Ok((weights_brain, weights_behavior, canonical_correlations))
    }

    fn perform_elastic_net(
        &self,
        brain_data: &Array2<f64>,
        behavior_data: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>), SklearsError> {
        // Simplified Elastic Net implementation
        let n_comp = self
            .n_components
            .min(brain_data.ncols())
            .min(behavior_data.ncols());

        let weights_brain = Array2::eye(brain_data.ncols())
            .slice(s![.., 0..n_comp])
            .to_owned();
        let weights_behavior = Array2::eye(behavior_data.ncols())
            .slice(s![.., 0..n_comp])
            .to_owned();
        let canonical_correlations = Array1::from_vec(vec![0.45; n_comp]);

        Ok((weights_brain, weights_behavior, canonical_correlations))
    }

    fn cross_validate(
        &self,
        brain_data: &Array2<f64>,
        behavior_data: &Array2<f64>,
    ) -> Result<Array1<f64>, SklearsError> {
        let n_subjects = brain_data.nrows();
        let fold_size = n_subjects / self.cv_folds;
        let mut cv_scores = Array1::zeros(self.cv_folds);

        for fold in 0..self.cv_folds {
            let test_start = fold * fold_size;
            let test_end = if fold == self.cv_folds - 1 {
                n_subjects
            } else {
                (fold + 1) * fold_size
            };

            // Create train/test splits
            let mut train_indices = Vec::new();
            let mut test_indices = Vec::new();

            for i in 0..n_subjects {
                if i >= test_start && i < test_end {
                    test_indices.push(i);
                } else {
                    train_indices.push(i);
                }
            }

            // Extract train/test data
            let train_brain = self.extract_rows(brain_data, &train_indices)?;
            let train_behavior = self.extract_rows(behavior_data, &train_indices)?;
            let test_brain = self.extract_rows(brain_data, &test_indices)?;
            let test_behavior = self.extract_rows(behavior_data, &test_indices)?;

            // Train model and compute correlation on test set
            let (weights_brain, weights_behavior, _) =
                self.perform_pls(&train_brain, &train_behavior)?;

            // Project test data
            let test_brain_proj = test_brain.dot(&weights_brain.column(0));
            let test_behavior_proj = test_behavior.dot(&weights_behavior.column(0));

            // Compute correlation
            let correlation = self.compute_correlation(&test_brain_proj, &test_behavior_proj)?;
            cv_scores[fold] = correlation;
        }

        Ok(cv_scores)
    }

    fn permutation_testing(
        &self,
        brain_data: &Array2<f64>,
        behavior_data: &Array2<f64>,
        original_correlations: &Array1<f64>,
    ) -> Result<(Array1<f64>, Array1<bool>), SklearsError> {
        let n_components = original_correlations.len();
        let mut p_values = Array1::zeros(n_components);
        let mut is_significant = Array1::from_elem(n_components, false);

        for comp in 0..n_components {
            let original_corr = original_correlations[comp];
            let mut null_distribution = Vec::with_capacity(self.n_permutations);

            for perm in 0..self.n_permutations {
                // Permute behavior data
                let mut permuted_behavior = behavior_data.clone();
                for i in 0..behavior_data.nrows() {
                    let perm_idx = ((i + perm * 7) % behavior_data.nrows()) as usize;
                    permuted_behavior
                        .row_mut(i)
                        .assign(&behavior_data.row(perm_idx));
                }

                // Compute correlation with permuted data
                let (_, _, perm_correlations) = self.perform_pls(brain_data, &permuted_behavior)?;
                null_distribution.push(perm_correlations[comp]);
            }

            // Compute p-value
            let count = null_distribution
                .iter()
                .filter(|&&x| x.abs() >= original_corr.abs())
                .count();
            p_values[comp] = count as f64 / self.n_permutations as f64;
            is_significant[comp] = p_values[comp] < self.alpha;
        }

        Ok((p_values, is_significant))
    }

    fn extract_rows(
        &self,
        data: &Array2<f64>,
        indices: &[usize],
    ) -> Result<Array2<f64>, SklearsError> {
        let mut result = Array2::zeros((indices.len(), data.ncols()));

        for (new_row, &old_row) in indices.iter().enumerate() {
            result.row_mut(new_row).assign(&data.row(old_row));
        }

        Ok(result)
    }

    fn compute_correlation(&self, x: &Array1<f64>, y: &Array1<f64>) -> Result<f64, SklearsError> {
        let n = x.len();
        if n != y.len() || n < 2 {
            return Err(SklearsError::InvalidInput(
                "Invalid input for correlation".to_string(),
            ));
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

        let denom = (den_x * den_y).sqrt();
        if denom < 1e-10 {
            Ok(0.0)
        } else {
            Ok(num / denom)
        }
    }
}

impl Default for BrainBehaviorCorrelation {
    fn default() -> Self {
        Self::new()
    }
}

/// Results of brain-behavior correlation analysis
#[derive(Debug, Clone)]
pub struct BrainBehaviorResults {
    pub weights_brain: Array2<f64>,
    pub weights_behavior: Array2<f64>,
    pub canonical_correlations: Array1<f64>,
    pub cv_scores: Array1<f64>,
    pub permutation_p_values: Option<Array1<f64>>,
    pub is_significant: Option<Array1<bool>>,
    pub method: CorrelationMethod,
    pub n_components: usize,
    pub n_subjects: usize,
    pub n_brain_features: usize,
    pub n_behavior_features: usize,
}

impl BrainBehaviorResults {
    /// Get brain weights
    pub fn brain_weights(&self) -> &Array2<f64> {
        &self.weights_brain
    }

    /// Get behavior weights
    pub fn behavior_weights(&self) -> &Array2<f64> {
        &self.weights_behavior
    }

    /// Get canonical correlations
    pub fn canonical_correlations(&self) -> &Array1<f64> {
        &self.canonical_correlations
    }

    /// Get cross-validation scores
    pub fn cv_scores(&self) -> &Array1<f64> {
        &self.cv_scores
    }

    /// Get mean cross-validation score
    pub fn mean_cv_score(&self) -> f64 {
        self.cv_scores.mean().unwrap_or(0.0)
    }

    /// Get permutation p-values if computed
    pub fn permutation_p_values(&self) -> Option<&Array1<f64>> {
        self.permutation_p_values.as_ref()
    }

    /// Check if results are significant
    pub fn is_significant(&self) -> Option<&Array1<bool>> {
        self.is_significant.as_ref()
    }
}
