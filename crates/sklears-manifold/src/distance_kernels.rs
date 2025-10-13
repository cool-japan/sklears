//! Distance methods and kernel functions for manifold learning
//!
//! This module provides various distance measures and kernel functions specifically
//! designed for manifold learning applications, including commute time distance,
//! resistance distance, manifold kernels, and graph kernels.

use scirs2_core::ndarray::ndarray_linalg::{Eigh, SVD, UPLO};
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::hash::{Hash, Hasher};

// DISTANCE METHODS
// =====================================================================================

/// Commute Time Distance
///
/// The commute time distance between two nodes in a graph is the expected number
/// of steps in a random walk starting from one node before reaching the other
/// and returning to the starting node.
pub fn commute_time_distance(adjacency: &Array2<f64>) -> SklResult<Array2<f64>> {
    let n_nodes = adjacency.nrows();

    // Compute Laplacian matrix
    let degrees: Array1<f64> = adjacency.sum_axis(Axis(1));
    let mut laplacian = -adjacency.clone();
    for i in 0..n_nodes {
        laplacian[(i, i)] += degrees[i];
    }

    // Compute pseudo-inverse of Laplacian (Moore-Penrose inverse)
    let laplacian_pinv = compute_pseudoinverse(&laplacian)?;

    // Commute time distance formula: C(i,j) = vol(G) * (L+[i,i] + L+[j,j] - 2*L+[i,j])
    let vol_g: f64 = degrees.sum();
    let mut commute_times = Array2::zeros((n_nodes, n_nodes));

    for i in 0..n_nodes {
        for j in 0..n_nodes {
            if i != j {
                let ct = vol_g
                    * (laplacian_pinv[(i, i)] + laplacian_pinv[(j, j)]
                        - 2.0 * laplacian_pinv[(i, j)]);
                commute_times[(i, j)] = ct.max(0.0); // Ensure non-negative
            }
        }
    }

    Ok(commute_times)
}

/// Resistance Distance
///
/// The resistance distance (also called effective resistance) between two nodes
/// is the electrical resistance between them when the graph is viewed as an
/// electrical network with unit resistors on each edge.
pub fn resistance_distance(adjacency: &Array2<f64>) -> SklResult<Array2<f64>> {
    let n_nodes = adjacency.nrows();

    // Compute Laplacian matrix
    let degrees: Array1<f64> = adjacency.sum_axis(Axis(1));
    let mut laplacian = -adjacency.clone();
    for i in 0..n_nodes {
        laplacian[(i, i)] += degrees[i];
    }

    // Compute pseudo-inverse of Laplacian
    let laplacian_pinv = compute_pseudoinverse(&laplacian)?;

    // Resistance distance formula: R(i,j) = L+[i,i] + L+[j,j] - 2*L+[i,j]
    let mut resistance_distances = Array2::zeros((n_nodes, n_nodes));

    for i in 0..n_nodes {
        for j in 0..n_nodes {
            if i != j {
                let rd =
                    laplacian_pinv[(i, i)] + laplacian_pinv[(j, j)] - 2.0 * laplacian_pinv[(i, j)];
                resistance_distances[(i, j)] = rd.max(0.0); // Ensure non-negative
            }
        }
    }

    Ok(resistance_distances)
}

/// Procrustes Distance
///
/// The Procrustes distance measures the dissimilarity between two configurations
/// of points after optimal translation, rotation, and scaling.
pub fn procrustes_distance(x1: &Array2<f64>, x2: &Array2<f64>) -> SklResult<f64> {
    if x1.dim() != x2.dim() {
        return Err(SklearsError::InvalidInput(
            "Input matrices must have the same dimensions".to_string(),
        ));
    }

    let (n, m) = x1.dim();

    // Center the configurations
    let mean1 = x1.mean_axis(Axis(0)).unwrap();
    let mean2 = x2.mean_axis(Axis(0)).unwrap();

    let x1_centered = x1 - &mean1;
    let x2_centered = x2 - &mean2;

    // Scale to unit norm
    let norm1 = x1_centered.mapv(|x| x * x).sum().sqrt();
    let norm2 = x2_centered.mapv(|x| x * x).sum().sqrt();

    if norm1 < 1e-10 || norm2 < 1e-10 {
        return Ok(0.0);
    }

    let x1_scaled = &x1_centered / norm1;
    let x2_scaled = &x2_centered / norm2;

    // Find optimal rotation using SVD
    let h = x1_scaled.t().dot(&x2_scaled);
    let (u, _s, vt) = h
        .svd(true, true)
        .map_err(|e| SklearsError::NumericalError(format!("SVD failed: {:?}", e)))?;

    let u = u.unwrap();
    let vt = vt.unwrap();

    // Optimal rotation matrix
    let r = vt.t().dot(&u.t());

    // Apply rotation to x1_scaled
    let x1_rotated = x1_scaled.dot(&r);

    // Compute Procrustes distance
    let diff = &x1_rotated - &x2_scaled;
    let distance = diff.mapv(|x| x * x).sum().sqrt();

    Ok(distance)
}

// =====================================================================================
// MANIFOLD KERNELS
// =====================================================================================

/// Manifold Kernels for Kernel-based Manifold Learning
///
/// These are specialized kernel functions designed for manifold learning that
/// capture geometric properties and local structure preservation.

/// Geodesic Kernel
///
/// Computes a kernel based on geodesic distances on the manifold.
/// K(x_i, x_j) = exp(-γ * d_geo(x_i, x_j)^2)
pub fn geodesic_kernel(
    x: &Array2<f64>,
    geodesic_distances: &Array2<f64>,
    gamma: f64,
) -> SklResult<Array2<f64>> {
    let (n_samples, _) = x.dim();
    let mut kernel = Array2::zeros((n_samples, n_samples));

    for i in 0..n_samples {
        for j in 0..n_samples {
            let geo_dist = geodesic_distances[(i, j)];
            kernel[(i, j)] = (-gamma * geo_dist.powi(2)).exp();
        }
    }

    Ok(kernel)
}

/// Heat Kernel on Manifolds
///
/// Computes the heat kernel which models heat diffusion on the manifold.
/// K_t(x_i, x_j) = exp(-t * L)(i, j) where L is the Laplacian operator
pub fn heat_kernel(laplacian: &Array2<f64>, t: f64) -> SklResult<Array2<f64>> {
    // Compute matrix exponential: exp(-t * L)
    // For simplicity, we use eigendecomposition
    let (eigenvalues, eigenvectors) = laplacian
        .eigh(UPLO::Lower)
        .map_err(|e| SklearsError::NumericalError(format!("Eigendecomposition failed: {:?}", e)))?;

    // Compute exp(-t * eigenvalues)
    let exp_eigenvalues: Array1<f64> = eigenvalues.mapv(|λ| (-t * λ).exp());

    // Reconstruct: V * exp(-t * Λ) * V^T
    let exp_diag = Array2::from_diag(&exp_eigenvalues);
    let heat_kernel = eigenvectors.dot(&exp_diag).dot(&eigenvectors.t());

    Ok(heat_kernel)
}

/// Local Tangent Space Kernel
///
/// Kernel based on local tangent space approximations at each point.
/// Measures similarity based on how well one point fits in the tangent space of another.
pub fn local_tangent_space_kernel(
    x: &Array2<f64>,
    k_neighbors: usize,
    gamma: f64,
) -> SklResult<Array2<f64>> {
    let (n_samples, n_features) = x.dim();
    let mut kernel = Array2::zeros((n_samples, n_samples));

    for i in 0..n_samples {
        // Find k nearest neighbors of point i
        let mut distances: Vec<(usize, f64)> = Vec::new();
        for j in 0..n_samples {
            if i != j {
                let dist = x
                    .row(i)
                    .iter()
                    .zip(x.row(j).iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                distances.push((j, dist));
            }
        }
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let neighbors: Vec<usize> = distances
            .iter()
            .take(k_neighbors)
            .map(|(idx, _)| *idx)
            .collect();

        // Compute local tangent space using PCA
        if neighbors.len() >= 2 {
            let mut neighbor_matrix = Array2::zeros((neighbors.len(), n_features));
            for (k, &neighbor_idx) in neighbors.iter().enumerate() {
                for j in 0..n_features {
                    neighbor_matrix[(k, j)] = x[(neighbor_idx, j)] - x[(i, j)];
                }
            }

            // SVD to get tangent space basis
            if let Ok((u, s, _)) = neighbor_matrix.svd(true, false) {
                if let Some(u) = u {
                    // Project all points onto this tangent space and compute kernel
                    for j in 0..n_samples {
                        let diff = &x.row(j) - &x.row(i);
                        let projection_norm = u.t().dot(&diff).mapv(|x| x * x).sum();
                        let tangent_dist = diff.mapv(|x| x * x).sum() - projection_norm;
                        kernel[(i, j)] = (-gamma * tangent_dist).exp();
                    }
                }
            }
        }
    }

    Ok(kernel)
}

/// Diffusion Kernel with Adaptive Bandwidth
///
/// Computes diffusion kernels with locally adaptive bandwidth based on
/// local density estimation.
pub fn adaptive_diffusion_kernel(
    x: &Array2<f64>,
    k_neighbors: usize,
    alpha: f64,
) -> SklResult<Array2<f64>> {
    let (n_samples, _) = x.dim();

    // Step 1: Compute local densities (using k-NN distances)
    let mut local_densities = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let mut distances: Vec<f64> = Vec::new();
        for j in 0..n_samples {
            if i != j {
                let dist = x
                    .row(i)
                    .iter()
                    .zip(x.row(j).iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                distances.push(dist);
            }
        }
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Use k-th nearest neighbor distance as density measure
        if distances.len() >= k_neighbors {
            local_densities[i] = distances[k_neighbors - 1];
        }
    }

    // Step 2: Compute adaptive kernel
    let mut kernel = Array2::zeros((n_samples, n_samples));

    for i in 0..n_samples {
        for j in 0..n_samples {
            let dist = x
                .row(i)
                .iter()
                .zip(x.row(j).iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            // Adaptive bandwidth based on local densities
            let adaptive_bandwidth =
                (local_densities[i].powf(alpha) * local_densities[j].powf(alpha)).sqrt();

            if adaptive_bandwidth > 1e-10 {
                kernel[(i, j)] = (-dist.powi(2) / (2.0 * adaptive_bandwidth.powi(2))).exp();
            }
        }
    }

    Ok(kernel)
}

/// Manifold Distance Kernel
///
/// Kernel that combines multiple distance measures on the manifold
/// including Euclidean, geodesic, and curvature-aware distances.
pub fn manifold_distance_kernel(
    x: &Array2<f64>,
    euclidean_weight: f64,
    geodesic_weight: f64,
    curvature_weight: f64,
    gamma: f64,
) -> SklResult<Array2<f64>> {
    let (n_samples, _) = x.dim();

    // Compute Euclidean distances
    let mut euclidean_dists = Array2::zeros((n_samples, n_samples));
    for i in 0..n_samples {
        for j in 0..n_samples {
            euclidean_dists[(i, j)] = x
                .row(i)
                .iter()
                .zip(x.row(j).iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
        }
    }

    // Compute geodesic distances (using Isomap-style approximation)
    let k = 10; // Number of neighbors for geodesic approximation
    let mut adjacency = Array2::zeros((n_samples, n_samples));

    for i in 0..n_samples {
        let mut distances: Vec<(usize, f64)> = Vec::new();
        for j in 0..n_samples {
            if i != j {
                distances.push((j, euclidean_dists[(i, j)]));
            }
        }
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for &(j, dist) in distances.iter().take(k) {
            adjacency[(i, j)] = dist;
            adjacency[(j, i)] = dist;
        }
    }

    // Floyd-Warshall for geodesic distances
    let mut geodesic_dists = adjacency.clone();
    for i in 0..n_samples {
        for j in 0..n_samples {
            if i != j && geodesic_dists[(i, j)] == 0.0 {
                geodesic_dists[(i, j)] = f64::INFINITY;
            }
        }
    }

    for k in 0..n_samples {
        for i in 0..n_samples {
            for j in 0..n_samples {
                if geodesic_dists[(i, k)] + geodesic_dists[(k, j)] < geodesic_dists[(i, j)] {
                    geodesic_dists[(i, j)] = geodesic_dists[(i, k)] + geodesic_dists[(k, j)];
                }
            }
        }
    }

    // Estimate local curvature (simplified measure)
    let mut curvature_dists = Array2::zeros((n_samples, n_samples));
    for i in 0..n_samples {
        for j in 0..n_samples {
            if i != j {
                // Simple curvature approximation based on deviation from straight line
                let euclidean = euclidean_dists[(i, j)];
                let geodesic = if geodesic_dists[(i, j)].is_finite() {
                    geodesic_dists[(i, j)]
                } else {
                    euclidean * 2.0
                };
                curvature_dists[(i, j)] = (geodesic - euclidean).abs();
            }
        }
    }

    // Combine distances with weights
    let mut combined_dists = Array2::zeros((n_samples, n_samples));
    for i in 0..n_samples {
        for j in 0..n_samples {
            combined_dists[(i, j)] = euclidean_weight * euclidean_dists[(i, j)]
                + geodesic_weight * geodesic_dists[(i, j)]
                + curvature_weight * curvature_dists[(i, j)];
        }
    }

    // Apply kernel function
    let mut kernel = Array2::zeros((n_samples, n_samples));
    for i in 0..n_samples {
        for j in 0..n_samples {
            kernel[(i, j)] = (-gamma * combined_dists[(i, j)].powi(2)).exp();
        }
    }

    Ok(kernel)
}

/// Spectral Kernel
///
/// Kernel based on spectral embedding and eigenspace projections.
/// Uses the spectrum of the data's Laplacian matrix.
pub fn spectral_kernel(x: &Array2<f64>, n_components: usize, gamma: f64) -> SklResult<Array2<f64>> {
    let (n_samples, _) = x.dim();

    // Create affinity matrix
    let mut affinity = Array2::zeros((n_samples, n_samples));
    for i in 0..n_samples {
        for j in 0..n_samples {
            if i != j {
                let dist = x
                    .row(i)
                    .iter()
                    .zip(x.row(j).iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                affinity[(i, j)] = (-dist.powi(2) / 2.0).exp();
            }
        }
    }

    // Compute normalized Laplacian
    let degrees: Array1<f64> = affinity.sum_axis(Axis(1));
    let mut laplacian = Array2::eye(n_samples);

    for i in 0..n_samples {
        for j in 0..n_samples {
            if i != j && affinity[(i, j)] > 0.0 {
                let normalization = (degrees[i] * degrees[j]).sqrt();
                if normalization > 1e-10 {
                    laplacian[(i, j)] = -affinity[(i, j)] / normalization;
                }
            }
        }
    }

    // Eigendecomposition
    if let Ok((eigenvalues, eigenvectors)) = laplacian.eigh(UPLO::Lower) {
        // Use first n_components eigenvectors as features
        let features = eigenvectors.slice(s![.., 1..=n_components.min(n_samples - 1)]);

        // Compute kernel in eigenspace
        let mut kernel = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in 0..n_samples {
                let mut dist_sq = 0.0;
                for k in 0..features.ncols() {
                    let diff = features[(i, k)] - features[(j, k)];
                    dist_sq += diff * diff;
                }
                kernel[(i, j)] = (-gamma * dist_sq).exp();
            }
        }

        Ok(kernel)
    } else {
        Err(SklearsError::NumericalError(
            "Eigendecomposition failed".to_string(),
        ))
    }
}

/// Helper function to compute Moore-Penrose pseudoinverse
fn compute_pseudoinverse(matrix: &Array2<f64>) -> SklResult<Array2<f64>> {
    let (u, s, vt) = matrix
        .svd(true, true)
        .map_err(|e| SklearsError::NumericalError(format!("SVD failed: {:?}", e)))?;

    let u = u.unwrap();
    let vt = vt.unwrap();

    // Compute reciprocal of non-zero singular values
    let tolerance = 1e-10;
    let s_inv: Array1<f64> = s.mapv(|x| if x > tolerance { 1.0 / x } else { 0.0 });

    // Construct pseudoinverse: V * S^+ * U^T
    let s_inv_diag = Array2::from_diag(&s_inv);
    let pinv = vt.t().dot(&s_inv_diag).dot(&u.t());

    Ok(pinv)
}

// =====================================================================================
// GRAPH KERNELS
// =====================================================================================

/// Graph Kernels for Graph-based Manifold Learning
///
/// These are specialized kernel functions designed for graph data structures,
/// enabling manifold learning on graph-structured data.

/// Random Walk Kernel
///
/// Computes a kernel based on random walks on graphs. The kernel value between
/// two graphs is the sum over all possible random walks of the product of their
/// probabilities.
pub fn random_walk_kernel(
    adjacency1: &Array2<f64>,
    adjacency2: &Array2<f64>,
    walk_length: usize,
    lambda: f64,
) -> SklResult<f64> {
    let n1 = adjacency1.nrows();
    let n2 = adjacency2.nrows();

    // Normalize adjacency matrices to transition matrices
    let mut trans1 = adjacency1.clone();
    let mut trans2 = adjacency2.clone();

    for i in 0..n1 {
        let row_sum = trans1.row(i).sum();
        if row_sum > 1e-10 {
            for j in 0..n1 {
                trans1[(i, j)] /= row_sum;
            }
        }
    }

    for i in 0..n2 {
        let row_sum = trans2.row(i).sum();
        if row_sum > 1e-10 {
            for j in 0..n2 {
                trans2[(i, j)] /= row_sum;
            }
        }
    }

    // Compute direct product graph for synchronized random walks
    let mut kernel_sum = 0.0;

    // Initialize with uniform distribution over starting nodes
    let mut current_prob = Array2::from_elem((n1, n2), 1.0 / (n1 * n2) as f64);

    for step in 0..walk_length {
        // Add contribution of this step
        let step_contribution: f64 = current_prob.sum() * lambda.powi(step as i32);
        kernel_sum += step_contribution;

        // Update probabilities for next step
        let mut next_prob = Array2::zeros((n1, n2));
        for i in 0..n1 {
            for j in 0..n2 {
                for ii in 0..n1 {
                    for jj in 0..n2 {
                        next_prob[(ii, jj)] +=
                            current_prob[(i, j)] * trans1[(i, ii)] * trans2[(j, jj)];
                    }
                }
            }
        }
        current_prob = next_prob;
    }

    Ok(kernel_sum)
}

/// Shortest Path Kernel
///
/// Computes a kernel based on shortest path distances in graphs.
/// Compares the distribution of shortest path lengths between all pairs of nodes.
pub fn shortest_path_kernel(adjacency1: &Array2<f64>, adjacency2: &Array2<f64>) -> SklResult<f64> {
    let n1 = adjacency1.nrows();
    let n2 = adjacency2.nrows();

    // Compute shortest path matrices using Floyd-Warshall
    let sp1 = floyd_warshall_shortest_paths(adjacency1)?;
    let sp2 = floyd_warshall_shortest_paths(adjacency2)?;

    // Extract shortest path histograms
    let hist1 = compute_path_histogram(&sp1);
    let hist2 = compute_path_histogram(&sp2);

    // Compute kernel as dot product of normalized histograms
    let norm1 = hist1.mapv(|x| x * x).sum().sqrt();
    let norm2 = hist2.mapv(|x| x * x).sum().sqrt();

    if norm1 < 1e-10 || norm2 < 1e-10 {
        return Ok(0.0);
    }

    let normalized_hist1 = &hist1 / norm1;
    let normalized_hist2 = &hist2 / norm2;

    let kernel_value = normalized_hist1
        .iter()
        .zip(normalized_hist2.iter())
        .map(|(a, b)| a * b)
        .sum();

    Ok(kernel_value)
}

/// Graph Laplacian Kernel
///
/// Computes a kernel based on the spectral properties of graph Laplacians.
/// Uses the eigenvalues and eigenvectors of the normalized Laplacian matrix.
pub fn graph_laplacian_kernel(
    adjacency1: &Array2<f64>,
    adjacency2: &Array2<f64>,
    n_eigenvalues: usize,
) -> SklResult<f64> {
    // Compute normalized Laplacians
    let laplacian1 = compute_normalized_laplacian(adjacency1)?;
    let laplacian2 = compute_normalized_laplacian(adjacency2)?;

    // Eigendecomposition
    let (eigenvals1, _) = laplacian1
        .eigh(UPLO::Lower)
        .map_err(|e| SklearsError::NumericalError(format!("Eigendecomposition failed: {:?}", e)))?;
    let (eigenvals2, _) = laplacian2
        .eigh(UPLO::Lower)
        .map_err(|e| SklearsError::NumericalError(format!("Eigendecomposition failed: {:?}", e)))?;

    // Compare first n_eigenvalues eigenvalues
    let n_vals = n_eigenvalues.min(eigenvals1.len()).min(eigenvals2.len());

    let mut kernel_value = 0.0;
    for i in 0..n_vals {
        let diff = eigenvals1[i] - eigenvals2[i];
        kernel_value += (-diff * diff).exp();
    }

    Ok(kernel_value)
}

/// Weisfeiler-Lehman Kernel
///
/// Implements a simplified version of the Weisfeiler-Lehman graph kernel
/// that iteratively refines node labels based on neighborhood structure.
pub fn weisfeiler_lehman_kernel(
    adjacency1: &Array2<f64>,
    adjacency2: &Array2<f64>,
    iterations: usize,
) -> SklResult<f64> {
    let n1 = adjacency1.nrows();
    let n2 = adjacency2.nrows();

    // Initialize node labels (using degree as initial label)
    let mut labels1: Vec<usize> = Vec::new();
    let mut labels2: Vec<usize> = Vec::new();

    for i in 0..n1 {
        let degree = adjacency1.row(i).iter().filter(|&&x| x > 0.0).count();
        labels1.push(degree);
    }

    for i in 0..n2 {
        let degree = adjacency2.row(i).iter().filter(|&&x| x > 0.0).count();
        labels2.push(degree);
    }

    let mut total_kernel = 0.0;

    for iter in 0..iterations {
        // Compute label histograms
        let hist1 = compute_label_histogram(&labels1);
        let hist2 = compute_label_histogram(&labels2);

        // Add kernel contribution from this iteration
        total_kernel += compute_histogram_intersection(&hist1, &hist2);

        // Update labels based on neighborhood
        labels1 = update_wl_labels(&labels1, adjacency1);
        labels2 = update_wl_labels(&labels2, adjacency2);
    }

    Ok(total_kernel)
}

/// Graphlet Kernel
///
/// Computes a kernel based on graphlet (small subgraph) counts.
/// This is a simplified version that counts triangles and stars.
pub fn graphlet_kernel(adjacency1: &Array2<f64>, adjacency2: &Array2<f64>) -> SklResult<f64> {
    // Count graphlets in both graphs
    let graphlets1 = count_graphlets(adjacency1)?;
    let graphlets2 = count_graphlets(adjacency2)?;

    // Compute kernel as normalized dot product
    let norm1 = graphlets1.mapv(|x| x * x).sum().sqrt();
    let norm2 = graphlets2.mapv(|x| x * x).sum().sqrt();

    if norm1 < 1e-10 || norm2 < 1e-10 {
        return Ok(0.0);
    }

    let kernel_value = graphlets1
        .iter()
        .zip(graphlets2.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>()
        / (norm1 * norm2);

    Ok(kernel_value)
}

/// Graph Kernel Matrix Computation
///
/// Computes a kernel matrix for a collection of graphs using a specified graph kernel.
pub enum GraphKernelType {
    /// RandomWalk
    RandomWalk { walk_length: usize, lambda: f64 },
    /// ShortestPath
    ShortestPath,
    /// Laplacian
    Laplacian { n_eigenvalues: usize },
    /// WeisfeilerLehman
    WeisfeilerLehman { iterations: usize },
    /// Graphlet
    Graphlet,
}

pub fn compute_graph_kernel_matrix(
    graphs: &[Array2<f64>],
    kernel_type: GraphKernelType,
) -> SklResult<Array2<f64>> {
    let n_graphs = graphs.len();
    let mut kernel_matrix = Array2::zeros((n_graphs, n_graphs));

    for i in 0..n_graphs {
        for j in i..n_graphs {
            let kernel_value = match kernel_type {
                GraphKernelType::RandomWalk {
                    walk_length,
                    lambda,
                } => random_walk_kernel(&graphs[i], &graphs[j], walk_length, lambda)?,
                GraphKernelType::ShortestPath => shortest_path_kernel(&graphs[i], &graphs[j])?,
                GraphKernelType::Laplacian { n_eigenvalues } => {
                    graph_laplacian_kernel(&graphs[i], &graphs[j], n_eigenvalues)?
                }
                GraphKernelType::WeisfeilerLehman { iterations } => {
                    weisfeiler_lehman_kernel(&graphs[i], &graphs[j], iterations)?
                }
                GraphKernelType::Graphlet => graphlet_kernel(&graphs[i], &graphs[j])?,
            };

            kernel_matrix[(i, j)] = kernel_value;
            kernel_matrix[(j, i)] = kernel_value; // Symmetric
        }
    }

    Ok(kernel_matrix)
}

// Helper functions for graph kernels

fn floyd_warshall_shortest_paths(adjacency: &Array2<f64>) -> SklResult<Array2<f64>> {
    let n = adjacency.nrows();
    let mut distances = Array2::from_elem((n, n), f64::INFINITY);

    // Initialize with adjacency matrix
    for i in 0..n {
        distances[(i, i)] = 0.0;
        for j in 0..n {
            if adjacency[(i, j)] > 0.0 {
                distances[(i, j)] = 1.0; // Unweighted graphs
            }
        }
    }

    // Floyd-Warshall algorithm
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                if distances[(i, k)] + distances[(k, j)] < distances[(i, j)] {
                    distances[(i, j)] = distances[(i, k)] + distances[(k, j)];
                }
            }
        }
    }

    Ok(distances)
}

fn compute_path_histogram(shortest_paths: &Array2<f64>) -> Array1<f64> {
    let mut histogram = Array1::zeros(20); // Max path length of 19

    for &distance in shortest_paths.iter() {
        if distance.is_finite() && distance > 0.0 {
            let bin = (distance as usize).min(19);
            histogram[bin] += 1.0;
        }
    }

    histogram
}

fn compute_normalized_laplacian(adjacency: &Array2<f64>) -> SklResult<Array2<f64>> {
    let n = adjacency.nrows();
    let degrees: Array1<f64> = adjacency.sum_axis(Axis(1));

    let mut laplacian = Array2::eye(n);

    for i in 0..n {
        for j in 0..n {
            if i != j && adjacency[(i, j)] > 0.0 {
                let normalization = (degrees[i] * degrees[j]).sqrt();
                if normalization > 1e-10 {
                    laplacian[(i, j)] = -adjacency[(i, j)] / normalization;
                }
            }
        }
    }

    Ok(laplacian)
}

fn compute_label_histogram(labels: &[usize]) -> std::collections::HashMap<usize, usize> {
    let mut histogram = std::collections::HashMap::new();
    for &label in labels {
        *histogram.entry(label).or_insert(0) += 1;
    }
    histogram
}

fn compute_histogram_intersection(
    hist1: &std::collections::HashMap<usize, usize>,
    hist2: &std::collections::HashMap<usize, usize>,
) -> f64 {
    let mut intersection = 0.0;

    for (&label, &count1) in hist1 {
        if let Some(&count2) = hist2.get(&label) {
            intersection += (count1.min(count2)) as f64;
        }
    }

    intersection
}

fn update_wl_labels(labels: &[usize], adjacency: &Array2<f64>) -> Vec<usize> {
    let n = labels.len();
    let mut new_labels = Vec::new();

    for i in 0..n {
        let mut neighbor_labels = Vec::new();
        for j in 0..n {
            if adjacency[(i, j)] > 0.0 {
                neighbor_labels.push(labels[j]);
            }
        }
        neighbor_labels.sort_unstable();

        // Create new label by hashing current label with sorted neighbor labels
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        labels[i].hash(&mut hasher);
        neighbor_labels.hash(&mut hasher);
        new_labels.push(hasher.finish() as usize);
    }

    new_labels
}

fn count_graphlets(adjacency: &Array2<f64>) -> SklResult<Array1<f64>> {
    let n = adjacency.nrows();
    let mut graphlet_counts = Array1::zeros(4); // [edges, triangles, stars, 4-cliques]

    // Count edges
    let mut edge_count = 0.0;
    for i in 0..n {
        for j in i + 1..n {
            if adjacency[(i, j)] > 0.0 {
                edge_count += 1.0;
            }
        }
    }
    graphlet_counts[0] = edge_count;

    // Count triangles
    let mut triangle_count = 0.0;
    for i in 0..n {
        for j in i + 1..n {
            for k in j + 1..n {
                if adjacency[(i, j)] > 0.0 && adjacency[(j, k)] > 0.0 && adjacency[(i, k)] > 0.0 {
                    triangle_count += 1.0;
                }
            }
        }
    }
    graphlet_counts[1] = triangle_count;

    // Count 3-stars (node with 3+ neighbors)
    let mut star_count = 0.0;
    for i in 0..n {
        let degree = adjacency.row(i).iter().filter(|&&x| x > 0.0).count();
        if degree >= 3 {
            // Choose 3 neighbors from degree neighbors
            star_count += (degree * (degree - 1) * (degree - 2)) as f64 / 6.0;
        }
    }
    graphlet_counts[2] = star_count;

    // Count 4-cliques (simplified)
    let mut clique4_count = 0.0;
    for i in 0..n {
        for j in i + 1..n {
            for k in j + 1..n {
                for l in k + 1..n {
                    if adjacency[(i, j)] > 0.0
                        && adjacency[(i, k)] > 0.0
                        && adjacency[(i, l)] > 0.0
                        && adjacency[(j, k)] > 0.0
                        && adjacency[(j, l)] > 0.0
                        && adjacency[(k, l)] > 0.0
                    {
                        clique4_count += 1.0;
                    }
                }
            }
        }
    }
    graphlet_counts[3] = clique4_count;

    Ok(graphlet_counts)
}
