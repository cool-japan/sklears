//! Diffusion distance computation for manifold analysis
//!
//! This module implements diffusion distances based on diffusion processes on manifolds:
//! - **Diffusion maps**: Spectral embedding using diffusion distances
//! - **Diffusion distance**: Distance metric based on random walk transitions
//! - **Multi-scale analysis**: Distance computation at multiple time scales
//! - **Adaptive kernels**: Automatic bandwidth selection for diffusion kernels
//! - **Robust diffusion**: Handling of outliers and noise in diffusion processes

use scirs2_core::ndarray::ndarray_linalg::{Eigh, UPLO};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::types::Float;
use std::collections::HashMap;

/// Kernel functions for diffusion maps
#[derive(Debug, Clone, Copy)]
pub enum DiffusionKernel {
    /// Gaussian kernel: K(x,y) = exp(-||x-y||²/(2σ²))
    Gaussian { sigma: Float },
    /// Adaptive Gaussian with automatic bandwidth selection
    AdaptiveGaussian { k_neighbors: usize },
    /// Polynomial kernel: K(x,y) = (x·y + c)^d
    Polynomial { degree: usize, coeff: Float },
    /// RBF kernel with custom bandwidth
    Rbf { gamma: Float },
}

/// Normalization methods for diffusion operators
#[derive(Debug, Clone, Copy)]
pub enum NormalizationMethod {
    /// No normalization (raw kernel matrix)
    None,
    /// Row normalization: D^(-1) * K
    Row,
    /// Symmetric normalization: D^(-1/2) * K * D^(-1/2)
    Symmetric,
    /// Laplacian normalization: I - D^(-1) * K
    Laplacian,
    /// Random walk normalization with alpha parameter
    RandomWalk { alpha: Float },
}

/// Compute pairwise distances for kernel construction
fn compute_pairwise_distances(data: &ArrayView2<Float>) -> Array2<Float> {
    let n = data.nrows();
    let mut distances = Array2::zeros((n, n));

    for i in 0..n {
        for j in i + 1..n {
            let dist_sq = data
                .row(i)
                .iter()
                .zip(data.row(j).iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<Float>();
            let dist = dist_sq.sqrt();
            distances[(i, j)] = dist;
            distances[(j, i)] = dist;
        }
    }
    distances
}

/// Estimate adaptive bandwidth using k-nearest neighbors
fn estimate_adaptive_bandwidth(
    distances: &Array2<Float>,
    k_neighbors: usize,
) -> Result<Array1<Float>, String> {
    let n = distances.nrows();
    if k_neighbors >= n {
        return Err("k_neighbors must be less than number of samples".to_string());
    }

    let mut bandwidths = Array1::zeros(n);

    for i in 0..n {
        let mut row_distances: Vec<Float> = (0..n)
            .filter(|&j| i != j)
            .map(|j| distances[(i, j)])
            .collect();

        row_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Use k-th nearest neighbor distance as bandwidth
        if row_distances.len() >= k_neighbors {
            bandwidths[i] = row_distances[k_neighbors - 1];
        } else {
            bandwidths[i] = row_distances.last().copied().unwrap_or(1.0);
        }
    }

    Ok(bandwidths)
}

/// Construct diffusion kernel matrix
pub fn build_diffusion_kernel(
    data: &ArrayView2<Float>,
    kernel_type: DiffusionKernel,
) -> Result<Array2<Float>, String> {
    let n = data.nrows();
    let distances = compute_pairwise_distances(data);
    let mut kernel = Array2::zeros((n, n));

    match kernel_type {
        DiffusionKernel::Gaussian { sigma } => {
            let two_sigma_sq = 2.0 * sigma * sigma;
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        kernel[(i, j)] = 1.0;
                    } else {
                        let dist_sq = distances[(i, j)].powi(2);
                        kernel[(i, j)] = (-dist_sq / two_sigma_sq).exp();
                    }
                }
            }
        }

        DiffusionKernel::AdaptiveGaussian { k_neighbors } => {
            let bandwidths = estimate_adaptive_bandwidth(&distances, k_neighbors)?;

            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        kernel[(i, j)] = 1.0;
                    } else {
                        let sigma_i = bandwidths[i];
                        let sigma_j = bandwidths[j];
                        let sigma_ij = (sigma_i * sigma_j).sqrt();
                        let two_sigma_sq = 2.0 * sigma_ij * sigma_ij;
                        let dist_sq = distances[(i, j)].powi(2);
                        kernel[(i, j)] = (-dist_sq / two_sigma_sq).exp();
                    }
                }
            }
        }

        DiffusionKernel::Polynomial { degree, coeff } => {
            // Compute dot products for polynomial kernel
            for i in 0..n {
                for j in 0..n {
                    let dot_product = data
                        .row(i)
                        .iter()
                        .zip(data.row(j).iter())
                        .map(|(a, b)| a * b)
                        .sum::<Float>();
                    kernel[(i, j)] = (dot_product + coeff).powi(degree as i32);
                }
            }
        }

        DiffusionKernel::Rbf { gamma } => {
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        kernel[(i, j)] = 1.0;
                    } else {
                        let dist_sq = distances[(i, j)].powi(2);
                        kernel[(i, j)] = (-gamma * dist_sq).exp();
                    }
                }
            }
        }
    }

    Ok(kernel)
}

/// Normalize kernel matrix to create diffusion operator
pub fn normalize_kernel(
    kernel: &Array2<Float>,
    method: NormalizationMethod,
) -> Result<Array2<Float>, String> {
    let n = kernel.nrows();

    match method {
        NormalizationMethod::None => Ok(kernel.clone()),

        NormalizationMethod::Row => {
            let mut normalized = kernel.clone();
            for i in 0..n {
                let row_sum: Float = kernel.row(i).sum();
                if row_sum > 0.0 {
                    for j in 0..n {
                        normalized[(i, j)] /= row_sum;
                    }
                }
            }
            Ok(normalized)
        }

        NormalizationMethod::Symmetric => {
            // Compute degree vector
            let mut degrees = Array1::zeros(n);
            for i in 0..n {
                degrees[i] = kernel.row(i).sum().sqrt();
            }

            let mut normalized = Array2::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    if degrees[i] > 0.0 && degrees[j] > 0.0 {
                        normalized[(i, j)] = kernel[(i, j)] / (degrees[i] * degrees[j]);
                    }
                }
            }
            Ok(normalized)
        }

        NormalizationMethod::Laplacian => {
            let row_normalized = normalize_kernel(kernel, NormalizationMethod::Row)?;
            let mut laplacian = Array2::eye(n);
            for i in 0..n {
                for j in 0..n {
                    laplacian[(i, j)] -= row_normalized[(i, j)];
                }
            }
            Ok(laplacian)
        }

        NormalizationMethod::RandomWalk { alpha } => {
            // Degree-corrected random walk normalization
            let mut degrees = Array1::zeros(n);
            for i in 0..n {
                degrees[i] = kernel.row(i).sum();
            }

            let mut normalized = Array2::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    if degrees[i] > 0.0 && degrees[j] > 0.0 {
                        let degree_factor = (degrees[i] / degrees[j]).powf(alpha);
                        normalized[(i, j)] = kernel[(i, j)] * degree_factor / degrees[i];
                    }
                }
            }
            Ok(normalized)
        }
    }
}

/// Compute diffusion distance at a given time scale
pub fn diffusion_distance(
    diffusion_operator: &Array2<Float>,
    time_scale: Float,
    n_components: Option<usize>,
) -> Result<Array2<Float>, String> {
    let n = diffusion_operator.nrows();
    let k = n_components.unwrap_or(n.min(50)); // Default to 50 or n components

    // Compute eigendecomposition
    let (eigenvals, eigenvecs) = diffusion_operator
        .eigh(UPLO::Upper)
        .map_err(|e| format!("Eigendecomposition failed: {:?}", e))?;

    // Sort eigenvalues and eigenvectors in descending order
    let mut eigen_pairs: Vec<(Float, ArrayView1<Float>)> = eigenvals
        .iter()
        .zip(eigenvecs.columns())
        .map(|(&val, vec)| (val, vec))
        .collect();

    eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // Take top k eigenvectors and eigenvalues
    let selected_eigenvals: Vec<Float> = eigen_pairs.iter().take(k).map(|(val, _)| *val).collect();

    let mut selected_eigenvecs = Array2::zeros((n, k));
    for (i, (_, eigenvec)) in eigen_pairs.iter().take(k).enumerate() {
        for (j, &val) in eigenvec.iter().enumerate() {
            selected_eigenvecs[(j, i)] = val;
        }
    }

    // Compute diffusion coordinates at time t
    let mut diffusion_coords = Array2::zeros((n, k));
    for i in 0..n {
        for j in 0..k {
            let eigenval = selected_eigenvals[j];
            let powered_eigenval = eigenval.powf(time_scale);
            diffusion_coords[(i, j)] = selected_eigenvecs[(i, j)] * powered_eigenval;
        }
    }

    // Compute pairwise diffusion distances
    let mut distances = Array2::zeros((n, n));
    for i in 0..n {
        for j in i..n {
            if i == j {
                distances[(i, j)] = 0.0;
            } else {
                let dist_sq = (0..k)
                    .map(|d| (diffusion_coords[(i, d)] - diffusion_coords[(j, d)]).powi(2))
                    .sum::<Float>();
                let dist = dist_sq.sqrt();
                distances[(i, j)] = dist;
                distances[(j, i)] = dist;
            }
        }
    }

    Ok(distances)
}

/// Multi-scale diffusion distance analysis
pub fn multiscale_diffusion_distance(
    data: &ArrayView2<Float>,
    kernel_type: DiffusionKernel,
    normalization: NormalizationMethod,
    time_scales: &[Float],
    n_components: Option<usize>,
) -> Result<Vec<Array2<Float>>, String> {
    // Build and normalize kernel
    let kernel = build_diffusion_kernel(data, kernel_type)?;
    let diffusion_op = normalize_kernel(&kernel, normalization)?;

    // Compute diffusion distances at multiple time scales
    let mut distances_at_scales = Vec::new();
    for &t in time_scales {
        let dist = diffusion_distance(&diffusion_op, t, n_components)?;
        distances_at_scales.push(dist);
    }

    Ok(distances_at_scales)
}

/// Diffusion distance computer with configurable parameters
pub struct DiffusionDistanceComputer {
    kernel_type: DiffusionKernel,
    normalization: NormalizationMethod,
    n_components: Option<usize>,
    time_scales: Vec<Float>,
}

impl Default for DiffusionDistanceComputer {
    fn default() -> Self {
        Self::new()
    }
}

impl DiffusionDistanceComputer {
    /// Create a new diffusion distance computer
    pub fn new() -> Self {
        Self {
            kernel_type: DiffusionKernel::AdaptiveGaussian { k_neighbors: 10 },
            normalization: NormalizationMethod::RandomWalk { alpha: 0.5 },
            n_components: None,
            time_scales: vec![1.0],
        }
    }

    /// Set the kernel type
    pub fn kernel_type(mut self, kernel_type: DiffusionKernel) -> Self {
        self.kernel_type = kernel_type;
        self
    }

    /// Set the normalization method
    pub fn normalization(mut self, normalization: NormalizationMethod) -> Self {
        self.normalization = normalization;
        self
    }

    /// Set number of eigenvector components to use
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = Some(n_components);
        self
    }

    /// Set time scales for diffusion
    pub fn time_scales(mut self, time_scales: Vec<Float>) -> Self {
        self.time_scales = time_scales;
        self
    }

    /// Compute diffusion distances
    pub fn compute(&self, data: &ArrayView2<Float>) -> Result<Vec<Array2<Float>>, String> {
        multiscale_diffusion_distance(
            data,
            self.kernel_type,
            self.normalization,
            &self.time_scales,
            self.n_components,
        )
    }

    /// Compute diffusion distances at a single time scale
    pub fn compute_single_scale(
        &self,
        data: &ArrayView2<Float>,
        time_scale: Float,
    ) -> Result<Array2<Float>, String> {
        let kernel = build_diffusion_kernel(data, self.kernel_type)?;
        let diffusion_op = normalize_kernel(&kernel, self.normalization)?;
        diffusion_distance(&diffusion_op, time_scale, self.n_components)
    }

    /// Get the diffusion operator matrix
    pub fn get_diffusion_operator(
        &self,
        data: &ArrayView2<Float>,
    ) -> Result<Array2<Float>, String> {
        let kernel = build_diffusion_kernel(data, self.kernel_type)?;
        normalize_kernel(&kernel, self.normalization)
    }
}

/// Utilities for diffusion distance analysis
pub mod diffusion_utils {
    use super::*;

    /// Estimate optimal time scale using spectral gap
    pub fn estimate_optimal_time_scale(
        diffusion_operator: &Array2<Float>,
    ) -> Result<Float, String> {
        // Compute eigenvalues
        let (eigenvals, _) = diffusion_operator
            .eigh(UPLO::Upper)
            .map_err(|e| format!("Eigendecomposition failed: {:?}", e))?;

        let mut sorted_eigenvals: Vec<Float> = eigenvals.to_vec();
        sorted_eigenvals.sort_by(|a, b| b.partial_cmp(a).unwrap());

        // Find spectral gap (largest gap between consecutive eigenvalues)
        let mut max_gap = 0.0;
        let mut gap_index = 0;

        for i in 0..sorted_eigenvals.len() - 1 {
            let gap = sorted_eigenvals[i] - sorted_eigenvals[i + 1];
            if gap > max_gap {
                max_gap = gap;
                gap_index = i;
            }
        }

        // Optimal time scale is related to the inverse of the spectral gap
        let optimal_t = if max_gap > 0.0 {
            1.0 / max_gap
        } else {
            1.0 // Default fallback
        };

        Ok(optimal_t)
    }

    /// Compute diffusion entropy at different time scales
    pub fn diffusion_entropy(
        diffusion_operator: &Array2<Float>,
        time_scales: &[Float],
    ) -> Result<Vec<Float>, String> {
        let (eigenvals, eigenvecs) = diffusion_operator
            .eigh(UPLO::Upper)
            .map_err(|e| format!("Eigendecomposition failed: {:?}", e))?;

        let mut entropies = Vec::new();

        for &t in time_scales {
            let mut entropy = 0.0;
            let n = diffusion_operator.nrows();

            // Compute transition probabilities at time t
            for i in 0..n {
                let mut prob_sum = 0.0;

                for j in 0..eigenvals.len() {
                    let eigenval = eigenvals[j];
                    let eigenvec_i = eigenvecs[(i, j)];
                    prob_sum += eigenvec_i * eigenvec_i * eigenval.powf(t);
                }

                if prob_sum > 0.0 {
                    entropy -= prob_sum * prob_sum.ln();
                }
            }

            entropies.push(entropy);
        }

        Ok(entropies)
    }

    /// Detect clusters using diffusion coordinates
    pub fn diffusion_clustering(
        data: &ArrayView2<Float>,
        kernel_type: DiffusionKernel,
        time_scale: Float,
        n_clusters: usize,
    ) -> Result<Vec<usize>, String> {
        // Build diffusion operator
        let kernel = build_diffusion_kernel(data, kernel_type)?;
        let diffusion_op = normalize_kernel(&kernel, NormalizationMethod::Row)?;

        // Compute eigendecomposition
        let (eigenvals, eigenvecs) = diffusion_op
            .eigh(UPLO::Upper)
            .map_err(|e| format!("Eigendecomposition failed: {:?}", e))?;

        // Sort and take top eigenvectors
        let mut eigen_pairs: Vec<(Float, ArrayView1<Float>)> = eigenvals
            .iter()
            .zip(eigenvecs.columns())
            .map(|(&val, vec)| (val, vec))
            .collect();

        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Use top n_clusters eigenvectors as features
        let n = data.nrows();
        let mut features = Array2::zeros((n, n_clusters));

        for (i, (eigenval, eigenvec)) in eigen_pairs.iter().take(n_clusters).enumerate() {
            let powered_eigenval = eigenval.powf(time_scale);
            for j in 0..n {
                features[(j, i)] = eigenvec[j] * powered_eigenval;
            }
        }

        // Simple k-means clustering on diffusion coordinates
        // This is a placeholder - in practice, you'd use a proper clustering algorithm
        let mut cluster_assignments = vec![0; n];

        // Initialize cluster centers randomly
        let mut centers = Array2::zeros((n_clusters, n_clusters));
        for i in 0..n_clusters {
            for j in 0..n_clusters {
                centers[(i, j)] = features[(i % n, j)];
            }
        }

        // Simple assignment based on closest center
        for i in 0..n {
            let mut min_dist = Float::INFINITY;
            let mut best_cluster = 0;

            for k in 0..n_clusters {
                let dist: Float = (0..n_clusters)
                    .map(|d| (features[(i, d)] - centers[(k, d)]).powi(2))
                    .sum::<Float>()
                    .sqrt();

                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = k;
                }
            }

            cluster_assignments[i] = best_cluster;
        }

        Ok(cluster_assignments)
    }

    /// Compute persistent diffusion distance across multiple time scales
    pub fn persistent_diffusion_analysis(
        data: &ArrayView2<Float>,
        kernel_type: DiffusionKernel,
        min_time: Float,
        max_time: Float,
        n_time_steps: usize,
    ) -> Result<HashMap<String, Vec<Float>>, String> {
        let time_step = (max_time - min_time) / (n_time_steps - 1) as Float;
        let time_scales: Vec<Float> = (0..n_time_steps)
            .map(|i| min_time + i as Float * time_step)
            .collect();

        let distances_at_scales = multiscale_diffusion_distance(
            data,
            kernel_type,
            NormalizationMethod::RandomWalk { alpha: 0.5 },
            &time_scales,
            Some(10),
        )?;

        let mut analysis = HashMap::new();
        let mut avg_distances = Vec::new();
        let mut max_distances = Vec::new();
        let mut min_distances = Vec::new();

        for distances in &distances_at_scales {
            let n = distances.nrows();
            let mut all_distances = Vec::new();

            for i in 0..n {
                for j in i + 1..n {
                    all_distances.push(distances[(i, j)]);
                }
            }

            if !all_distances.is_empty() {
                let avg = all_distances.iter().sum::<Float>() / all_distances.len() as Float;
                let max_dist = all_distances.iter().fold(0.0 as Float, |a, &b| a.max(b));
                let min_dist = all_distances.iter().fold(Float::INFINITY, |a, &b| a.min(b));

                avg_distances.push(avg);
                max_distances.push(max_dist);
                min_distances.push(min_dist);
            }
        }

        analysis.insert("time_scales".to_string(), time_scales);
        analysis.insert("average_distances".to_string(), avg_distances);
        analysis.insert("maximum_distances".to_string(), max_distances);
        analysis.insert("minimum_distances".to_string(), min_distances);

        Ok(analysis)
    }
}
