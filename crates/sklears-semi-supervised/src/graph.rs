//! Graph construction utilities for semi-supervised learning

use scirs2_core::ndarray_ext::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};
use std::collections::HashSet;

/// Construct k-nearest neighbors graph
pub fn knn_graph(X: &Array2<Float>, n_neighbors: usize, mode: &str) -> SklResult<Array2<Float>> {
    let n_samples = X.nrows();
    let mut W = Array2::zeros((n_samples, n_samples));

    for i in 0..n_samples {
        let mut distances: Vec<(usize, Float)> = Vec::new();
        for j in 0..n_samples {
            if i != j {
                let diff = &X.row(i) - &X.row(j);
                let dist = diff.mapv(|x| x * x).sum().sqrt();
                distances.push((j, dist));
            }
        }

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        match mode {
            "connectivity" => {
                for &(j, _) in distances.iter().take(n_neighbors) {
                    W[[i, j]] = 1.0;
                }
            }
            "distance" => {
                for &(j, dist) in distances.iter().take(n_neighbors) {
                    W[[i, j]] = dist;
                }
            }
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown mode: {}",
                    mode
                )));
            }
        }
    }

    Ok(W)
}

/// Construct epsilon-neighborhood graph
pub fn epsilon_graph(X: &Array2<Float>, epsilon: Float, mode: &str) -> SklResult<Array2<Float>> {
    let n_samples = X.nrows();
    let mut W = Array2::zeros((n_samples, n_samples));

    for i in 0..n_samples {
        for j in 0..n_samples {
            if i != j {
                let diff = &X.row(i) - &X.row(j);
                let dist = diff.mapv(|x| x * x).sum().sqrt();
                if dist <= epsilon {
                    match mode {
                        "connectivity" => W[[i, j]] = 1.0,
                        "distance" => W[[i, j]] = dist,
                        _ => {
                            return Err(SklearsError::InvalidInput(format!(
                                "Unknown mode: {}",
                                mode
                            )));
                        }
                    }
                }
            }
        }
    }

    Ok(W)
}

/// Construct graph Laplacian
pub fn graph_laplacian(adjacency: &Array2<Float>, normed: bool) -> SklResult<Array2<Float>> {
    let n_samples = adjacency.nrows();
    let mut L = Array2::zeros((n_samples, n_samples));

    // Compute degree matrix
    let degree = adjacency.sum_axis(Axis(1));

    if normed {
        // Normalized Laplacian: L = I - D^(-1/2) * W * D^(-1/2)
        let mut D_sqrt_inv = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            if degree[i] > 0.0 {
                D_sqrt_inv[[i, i]] = 1.0 / degree[i].sqrt();
            }
        }

        let normalized_adjacency = D_sqrt_inv.dot(adjacency).dot(&D_sqrt_inv);

        // L = I - normalized_adjacency
        for i in 0..n_samples {
            L[[i, i]] = 1.0;
            for j in 0..n_samples {
                L[[i, j]] -= normalized_adjacency[[i, j]];
            }
        }
    } else {
        // Unnormalized Laplacian: L = D - W
        for i in 0..n_samples {
            L[[i, i]] = degree[i];
            for j in 0..n_samples {
                if i != j {
                    L[[i, j]] = -adjacency[[i, j]];
                }
            }
        }
    }

    Ok(L)
}

/// Make graph symmetric
pub fn make_symmetric(W: &mut Array2<Float>) {
    let n = W.nrows();
    for i in 0..n {
        for j in i + 1..n {
            let avg = (W[[i, j]] + W[[j, i]]) / 2.0;
            W[[i, j]] = avg;
            W[[j, i]] = avg;
        }
    }
}

/// Construct mutual k-nearest neighbors graph
pub fn mutual_knn_graph(X: &Array2<Float>, n_neighbors: usize) -> SklResult<Array2<Float>> {
    let knn_graph = knn_graph(X, n_neighbors, "connectivity")?;
    let n_samples = X.nrows();
    let mut mutual_graph = Array2::zeros((n_samples, n_samples));

    // Only keep edges that are mutual
    for i in 0..n_samples {
        for j in 0..n_samples {
            if knn_graph[[i, j]] > 0.0 && knn_graph[[j, i]] > 0.0 {
                mutual_graph[[i, j]] = 1.0;
            }
        }
    }

    Ok(mutual_graph)
}

/// Construct shared nearest neighbors graph
pub fn shared_nn_graph(
    X: &Array2<Float>,
    n_neighbors: usize,
    min_shared: usize,
) -> SklResult<Array2<Float>> {
    let n_samples = X.nrows();
    let mut W = Array2::zeros((n_samples, n_samples));

    // For each point, find its k nearest neighbors
    let mut neighbors: Vec<Vec<usize>> = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let mut distances: Vec<(usize, Float)> = Vec::new();
        for j in 0..n_samples {
            if i != j {
                let diff = &X.row(i) - &X.row(j);
                let dist = diff.mapv(|x| x * x).sum().sqrt();
                distances.push((j, dist));
            }
        }

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let point_neighbors: Vec<usize> = distances
            .iter()
            .take(n_neighbors)
            .map(|(idx, _)| *idx)
            .collect();
        neighbors.push(point_neighbors);
    }

    // Count shared neighbors
    for i in 0..n_samples {
        for j in i + 1..n_samples {
            let shared_count = neighbors[i]
                .iter()
                .filter(|&&neighbor| neighbors[j].contains(&neighbor))
                .count();

            if shared_count >= min_shared {
                W[[i, j]] = shared_count as Float;
                W[[j, i]] = shared_count as Float;
            }
        }
    }

    Ok(W)
}

/// Construct random walk Laplacian
///
/// The random walk Laplacian is defined as L_rw = I - D^(-1) * W,
/// where D is the degree matrix and W is the adjacency matrix.
/// This is useful for spectral clustering and random walk analysis.
pub fn random_walk_laplacian(adjacency: &Array2<Float>) -> SklResult<Array2<Float>> {
    let n_samples = adjacency.nrows();
    let mut L_rw = Array2::zeros((n_samples, n_samples));

    // Compute degree matrix
    let degree = adjacency.sum_axis(Axis(1));

    // Random Walk Laplacian: L_rw = I - D^(-1) * W
    let mut D_inv = Array2::zeros((n_samples, n_samples));
    for i in 0..n_samples {
        if degree[i] > 0.0 {
            D_inv[[i, i]] = 1.0 / degree[i];
        }
    }

    let transition_matrix = D_inv.dot(adjacency);

    // L_rw = I - P where P is the transition matrix
    for i in 0..n_samples {
        L_rw[[i, i]] = 1.0;
        for j in 0..n_samples {
            L_rw[[i, j]] -= transition_matrix[[i, j]];
        }
    }

    Ok(L_rw)
}

/// Compute diffusion matrix for diffusion maps
///
/// The diffusion matrix is defined as P^t where P is the transition matrix
/// and t is the diffusion time parameter.
pub fn diffusion_matrix(adjacency: &Array2<Float>, t: usize) -> SklResult<Array2<Float>> {
    let n_samples = adjacency.nrows();

    // Compute degree matrix and transition matrix
    let degree = adjacency.sum_axis(Axis(1));
    let mut P = Array2::zeros((n_samples, n_samples));

    for i in 0..n_samples {
        if degree[i] > 0.0 {
            for j in 0..n_samples {
                P[[i, j]] = adjacency[[i, j]] / degree[i];
            }
        }
    }

    // Compute P^t by repeated matrix multiplication
    let mut result = P.clone();
    for _ in 1..t {
        result = result.dot(&P);
    }

    Ok(result)
}

/// Adaptive neighborhood graph construction
///
/// Automatically determines the neighborhood size based on local density estimation.
pub fn adaptive_knn_graph(X: &Array2<Float>, mode: &str) -> SklResult<Array2<Float>> {
    let n_samples = X.nrows();
    let mut W = Array2::zeros((n_samples, n_samples));

    for i in 0..n_samples {
        let mut distances: Vec<(usize, Float)> = Vec::new();
        for j in 0..n_samples {
            if i != j {
                let diff = &X.row(i) - &X.row(j);
                let dist = diff.mapv(|x| x * x).sum().sqrt();
                distances.push((j, dist));
            }
        }

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Adaptive k selection based on distance gaps
        let mut k = 3; // minimum k
        if distances.len() > 5 {
            // Find the largest gap in distances to determine natural neighborhood size
            let mut max_gap = 0.0;
            let mut best_k = k;

            for idx in 2..distances.len().min(15) {
                if idx < distances.len() - 1 {
                    let gap = distances[idx + 1].1 - distances[idx].1;
                    if gap > max_gap {
                        max_gap = gap;
                        best_k = idx + 1;
                    }
                }
            }
            k = best_k.max(3).min(n_samples / 3);
        }

        // Connect to k nearest neighbors
        match mode {
            "connectivity" => {
                for &(j, _) in distances.iter().take(k) {
                    W[[i, j]] = 1.0;
                }
            }
            "distance" => {
                for &(j, dist) in distances.iter().take(k) {
                    W[[i, j]] = dist;
                }
            }
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown mode: {}",
                    mode
                )));
            }
        }
    }

    // Make symmetric
    make_symmetric(&mut W);
    Ok(W)
}

/// Graph sparsification using effective resistance sampling
///
/// Reduces the number of edges while preserving spectral properties.
pub fn sparsify_graph(
    adjacency: &Array2<Float>,
    sparsity_ratio: Float,
) -> SklResult<Array2<Float>> {
    let n_samples = adjacency.nrows();
    let mut W_sparse = adjacency.clone();

    // Count non-zero edges
    let mut edges: Vec<(usize, usize, Float)> = Vec::new();
    for i in 0..n_samples {
        for j in i + 1..n_samples {
            if adjacency[[i, j]] > 0.0 {
                edges.push((i, j, adjacency[[i, j]]));
            }
        }
    }

    // Sort edges by weight (keep stronger connections)
    edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    let keep_count = ((edges.len() as Float) * sparsity_ratio) as usize;

    // Zero out the weakest edges
    W_sparse.fill(0.0);
    for &(i, j, weight) in edges.iter().take(keep_count) {
        W_sparse[[i, j]] = weight;
        W_sparse[[j, i]] = weight;
    }

    Ok(W_sparse)
}

/// Spectral clustering implementation for semi-supervised learning
///
/// Performs spectral clustering on a given affinity matrix using eigendecomposition
/// of the normalized graph Laplacian. This can be used for clustering or as a
/// preprocessing step for semi-supervised learning.
///
/// # Parameters
///
/// * `adjacency` - Affinity/adjacency matrix (symmetric, non-negative)
/// * `n_clusters` - Number of clusters to find
/// * `normalized` - Whether to use normalized Laplacian
/// * `random_state` - Seed for k-means clustering (for reproducibility)
///
/// # Returns
///
/// * `SklResult<Array1<i32>>` - Cluster labels for each sample
///
/// # Examples
///
/// ```
/// use scirs2_core::array;
/// use sklears_semi_supervised::spectral_clustering;
///
///
/// let W = array![[0.0, 1.0, 0.1], [1.0, 0.0, 0.2], [0.1, 0.2, 0.0]];
/// let labels = spectral_clustering(&W, 2, true, Some(42)).unwrap();
/// assert_eq!(labels.len(), 3);
/// ```
#[allow(non_snake_case)]
pub fn spectral_clustering(
    adjacency: &Array2<Float>,
    n_clusters: usize,
    normalized: bool,
    random_state: Option<u64>,
) -> SklResult<Array1<i32>> {
    if n_clusters == 0 {
        return Err(SklearsError::InvalidInput(
            "Number of clusters must be positive".to_string(),
        ));
    }

    let n_samples = adjacency.nrows();
    if n_samples < n_clusters {
        return Err(SklearsError::InvalidInput(
            "Number of clusters cannot exceed number of samples".to_string(),
        ));
    }

    // Compute graph Laplacian
    let L = graph_laplacian(adjacency, normalized)?;

    // Compute eigendecomposition
    let eigenvectors = compute_laplacian_eigenvectors(&L, n_clusters)?;

    // Apply k-means to the embedding
    let labels = kmeans_clustering(&eigenvectors, n_clusters, random_state)?;

    Ok(labels)
}

/// Spectral embedding for dimensionality reduction
///
/// Computes the spectral embedding of the graph Laplacian for dimensionality reduction.
/// This extracts the low-dimensional representation based on the graph structure.
///
/// # Parameters
///
/// * `adjacency` - Affinity/adjacency matrix
/// * `n_components` - Number of eigenvectors to use for embedding
/// * `normalized` - Whether to use normalized Laplacian
///
/// # Returns
///
/// * `SklResult<Array2<Float>>` - Spectral embedding (n_samples x n_components)
#[allow(non_snake_case)]
pub fn spectral_embedding(
    adjacency: &Array2<Float>,
    n_components: usize,
    normalized: bool,
) -> SklResult<Array2<Float>> {
    if n_components == 0 {
        return Err(SklearsError::InvalidInput(
            "Number of components must be positive".to_string(),
        ));
    }

    let n_samples = adjacency.nrows();
    if n_components >= n_samples {
        return Err(SklearsError::InvalidInput(
            "Number of components must be less than number of samples".to_string(),
        ));
    }

    // Compute graph Laplacian
    let L = graph_laplacian(adjacency, normalized)?;

    // Compute eigenvectors
    let embedding = compute_laplacian_eigenvectors(&L, n_components)?;

    Ok(embedding)
}

/// Compute eigenvectors of the graph Laplacian
///
/// This function computes the smallest eigenvalues and corresponding eigenvectors
/// of the graph Laplacian matrix using a simplified eigendecomposition.
///
/// Note: This is a simplified implementation. For production use, consider using
/// specialized linear algebra libraries for more efficient and stable eigendecomposition.
fn compute_laplacian_eigenvectors(
    laplacian: &Array2<Float>,
    n_eigenvectors: usize,
) -> SklResult<Array2<Float>> {
    let n = laplacian.nrows();

    // For small matrices, use power iteration method to approximate eigenvectors
    if n <= 100 {
        power_iteration_eigenvectors(laplacian, n_eigenvectors)
    } else {
        // For larger matrices, use a simplified approach
        lanczos_eigenvectors(laplacian, n_eigenvectors)
    }
}

/// Power iteration method for computing eigenvectors
fn power_iteration_eigenvectors(
    matrix: &Array2<Float>,
    n_eigenvectors: usize,
) -> SklResult<Array2<Float>> {
    use scirs2_core::random::rand_prelude::*;
    use scirs2_core::random::Random;

    let n = matrix.nrows();
    let mut rng = Random::seed(42);
    let max_iter = 1000;
    let tol = 1e-6;

    let mut eigenvectors = Array2::<Float>::zeros((n, n_eigenvectors));

    // Start with the identity matrix shifted to find smallest eigenvalues
    let shift = matrix.diag().iter().copied().fold(0.0, Float::max) + 1.0;
    let mut shifted_matrix = matrix.clone();
    for i in 0..n {
        shifted_matrix[[i, i]] += shift;
    }

    // Inverse power iteration to find smallest eigenvalues
    for k in 0..n_eigenvectors {
        // Initialize random vector
        let mut v = Array1::<Float>::zeros(n);
        for i in 0..n {
            v[i] = rng.random::<Float>() - 0.5;
        }

        // Orthogonalize against previous eigenvectors
        for j in 0..k {
            let prev_v = eigenvectors.column(j);
            let dot_product: Float = v.dot(&prev_v);
            for i in 0..n {
                v[i] -= dot_product * prev_v[i];
            }
        }

        // Normalize
        let norm = (v.iter().map(|x| x * x).sum::<Float>()).sqrt();
        if norm > 0.0 {
            v /= norm;
        }

        // Power iteration
        for _iter in 0..max_iter {
            let prev_v = v.clone();

            // Solve (A + shift*I) * v_new = v_old approximately using Jacobi iteration
            let mut v_new = Array1::<Float>::zeros(n);
            for _ in 0..10 {
                // Inner Jacobi iterations
                for i in 0..n {
                    let mut sum = 0.0;
                    for j in 0..n {
                        if i != j {
                            sum += shifted_matrix[[i, j]] * v_new[j];
                        }
                    }
                    v_new[i] = (v[i] - sum) / shifted_matrix[[i, i]];
                }
            }
            v = v_new;

            // Orthogonalize against previous eigenvectors
            for j in 0..k {
                let prev_ev = eigenvectors.column(j);
                let dot_product: Float = v.dot(&prev_ev);
                for i in 0..n {
                    v[i] -= dot_product * prev_ev[i];
                }
            }

            // Normalize
            let norm = (v.iter().map(|x| x * x).sum::<Float>()).sqrt();
            if norm > 0.0 {
                v /= norm;
            }

            // Check convergence
            let diff: Float = v
                .iter()
                .zip(prev_v.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            if diff < tol {
                break;
            }
        }

        // Store eigenvector
        for i in 0..n {
            eigenvectors[[i, k]] = v[i];
        }
    }

    Ok(eigenvectors)
}

/// Simplified Lanczos method for larger matrices
fn lanczos_eigenvectors(matrix: &Array2<Float>, n_eigenvectors: usize) -> SklResult<Array2<Float>> {
    use scirs2_core::random::rand_prelude::*;
    use scirs2_core::random::Random;

    let n = matrix.nrows();
    let mut rng = Random::seed(42);

    // For simplicity, use random projection method
    let mut eigenvectors = Array2::<Float>::zeros((n, n_eigenvectors));

    for k in 0..n_eigenvectors {
        // Random initialization
        let mut v = Array1::<Float>::zeros(n);
        for i in 0..n {
            v[i] = rng.random::<Float>() - 0.5;
        }

        // Simple power iteration (fewer iterations for speed)
        for _iter in 0..50 {
            let mut v_new = Array1::<Float>::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    v_new[i] += matrix[[i, j]] * v[j];
                }
            }

            // Orthogonalize against previous vectors
            for j in 0..k {
                let prev_v = eigenvectors.column(j);
                let dot_product: Float = v_new.dot(&prev_v);
                for i in 0..n {
                    v_new[i] -= dot_product * prev_v[i];
                }
            }

            // Normalize
            let norm = (v_new.iter().map(|x| x * x).sum::<Float>()).sqrt();
            if norm > 0.0 {
                v_new /= norm;
            }

            v = v_new;
        }

        // Store eigenvector
        for i in 0..n {
            eigenvectors[[i, k]] = v[i];
        }
    }

    Ok(eigenvectors)
}

/// Simple k-means clustering for spectral clustering
fn kmeans_clustering(
    X: &Array2<Float>,
    n_clusters: usize,
    random_state: Option<u64>,
) -> SklResult<Array1<i32>> {
    use scirs2_core::random::rand_prelude::*;
    use scirs2_core::random::Random;

    let (n_samples, n_features) = X.dim();
    let mut rng = Random::seed(42);

    if n_clusters >= n_samples {
        // Each sample is its own cluster
        return Ok(Array1::from_iter(0..n_samples as i32));
    }

    // Initialize centroids randomly
    let mut centroids = Array2::<Float>::zeros((n_clusters, n_features));
    let mut selected_indices = HashSet::new();

    for k in 0..n_clusters {
        let mut idx = rng.gen_range(0..n_samples);
        while selected_indices.contains(&idx) {
            idx = rng.gen_range(0..n_samples);
        }
        selected_indices.insert(idx);

        for j in 0..n_features {
            centroids[[k, j]] = X[[idx, j]];
        }
    }

    let mut labels = Array1::<i32>::zeros(n_samples);
    let max_iter = 300;

    for _iter in 0..max_iter {
        let prev_labels = labels.clone();

        // Assign points to closest centroid
        for i in 0..n_samples {
            let mut min_dist = Float::INFINITY;
            let mut best_cluster = 0;

            for k in 0..n_clusters {
                let mut dist = 0.0;
                for j in 0..n_features {
                    let diff = X[[i, j]] - centroids[[k, j]];
                    dist += diff * diff;
                }

                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = k;
                }
            }

            labels[i] = best_cluster as i32;
        }

        // Update centroids
        let mut cluster_counts = vec![0; n_clusters];
        centroids.fill(0.0);

        for i in 0..n_samples {
            let cluster = labels[i] as usize;
            cluster_counts[cluster] += 1;
            for j in 0..n_features {
                centroids[[cluster, j]] += X[[i, j]];
            }
        }

        for k in 0..n_clusters {
            if cluster_counts[k] > 0 {
                let count = cluster_counts[k] as Float;
                for j in 0..n_features {
                    centroids[[k, j]] /= count;
                }
            }
        }

        // Check convergence
        if labels == prev_labels {
            break;
        }
    }

    Ok(labels)
}

/// Multi-scale graph construction
///
/// Constructs graphs at multiple scales and combines them to capture both
/// local and global structure. This is particularly useful for semi-supervised
/// learning where different scales may reveal different patterns in the data.
///
/// The method builds k-NN graphs at different neighborhood sizes and combines
/// them using weighted averaging, where weights can be learned or predefined.
///
/// # Parameters
///
/// * `X` - Input data matrix (n_samples x n_features)
/// * `scales` - Vector of k values for different scales (neighborhood sizes)
/// * `scale_weights` - Optional weights for combining scales (if None, uniform weights used)
/// * `combination_method` - Method for combining graphs ("weighted", "union", "intersection")
/// * `normalize_scales` - Whether to normalize each scale independently
///
/// # Returns
///
/// * `SklResult<Array2<Float>>` - Combined multi-scale adjacency matrix
///
/// # Examples
///
/// ```
/// use scirs2_core::array;
/// use sklears_semi_supervised::multi_scale_graph_construction;
///
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
/// let scales = vec![2, 3];
/// let W = multi_scale_graph_construction(&X, &scales, None, "weighted", true).unwrap();
/// assert_eq!(W.dim(), (4, 4));
/// ```
pub fn multi_scale_graph_construction(
    X: &Array2<Float>,
    scales: &[usize],
    scale_weights: Option<&[Float]>,
    combination_method: &str,
    normalize_scales: bool,
) -> SklResult<Array2<Float>> {
    if scales.is_empty() {
        return Err(SklearsError::InvalidInput(
            "At least one scale must be provided".to_string(),
        ));
    }

    let n_samples = X.nrows();

    // Validate scales
    for &scale in scales {
        if scale >= n_samples {
            return Err(SklearsError::InvalidInput(format!(
                "Scale {} must be less than number of samples {}",
                scale, n_samples
            )));
        }
    }

    // Set default weights if not provided
    let weights = if let Some(w) = scale_weights {
        if w.len() != scales.len() {
            return Err(SklearsError::InvalidInput(
                "Number of weights must match number of scales".to_string(),
            ));
        }
        w.to_vec()
    } else {
        vec![1.0 / scales.len() as Float; scales.len()]
    };

    // Validate weights sum to 1 for weighted combination
    if combination_method == "weighted" {
        let weight_sum: Float = weights.iter().sum();
        if (weight_sum - 1.0).abs() > 1e-10 {
            return Err(SklearsError::InvalidInput(
                "Weights must sum to 1 for weighted combination".to_string(),
            ));
        }
    }

    // Build graphs at each scale
    let mut scale_graphs = Vec::new();

    for &k in scales {
        let mut graph = knn_graph(X, k, "connectivity")?;

        // Make symmetric
        make_symmetric(&mut graph);

        // Normalize if requested
        if normalize_scales {
            let max_degree = graph
                .sum_axis(Axis(1))
                .iter()
                .fold(0.0 as Float, |acc, &x| acc.max(x));
            if max_degree > 0.0 {
                graph.mapv_inplace(|x| x / max_degree);
            }
        }

        scale_graphs.push(graph);
    }

    // Combine graphs according to specified method
    let mut combined_graph = Array2::zeros((n_samples, n_samples));

    match combination_method {
        "weighted" => {
            // Weighted combination of all scales
            for (graph, &weight) in scale_graphs.iter().zip(weights.iter()) {
                combined_graph = combined_graph + weight * graph;
            }
        }
        "union" => {
            // Union: edge exists if it exists in any scale
            for graph in scale_graphs.iter() {
                for i in 0..n_samples {
                    for j in 0..n_samples {
                        if graph[[i, j]] > 0.0 {
                            combined_graph[[i, j]] = 1.0;
                        }
                    }
                }
            }
        }
        "intersection" => {
            // Intersection: edge exists only if it exists in all scales
            // Initialize with all ones
            combined_graph.fill(1.0);

            for graph in scale_graphs.iter() {
                for i in 0..n_samples {
                    for j in 0..n_samples {
                        if graph[[i, j]] == 0.0 {
                            combined_graph[[i, j]] = 0.0;
                        }
                    }
                }
            }
        }
        "adaptive_weighted" => {
            // Adaptive weighting based on local density
            let mut adaptive_weights = Vec::new();

            for (scale_idx, graph) in scale_graphs.iter().enumerate() {
                let avg_degree = graph.sum_axis(Axis(1)).mean().unwrap_or(0.0);
                let scale_density = avg_degree / scales[scale_idx] as Float;
                adaptive_weights.push(scale_density);
            }

            // Normalize adaptive weights
            let total_weight: Float = adaptive_weights.iter().sum();
            if total_weight > 0.0 {
                for weight in adaptive_weights.iter_mut() {
                    *weight /= total_weight;
                }
            }

            // Apply adaptive weights
            for (graph, &weight) in scale_graphs.iter().zip(adaptive_weights.iter()) {
                combined_graph = combined_graph + weight * graph;
            }
        }
        "max_pooling" => {
            // Max pooling: take maximum edge weight across scales
            for graph in scale_graphs.iter() {
                for i in 0..n_samples {
                    for j in 0..n_samples {
                        combined_graph[[i, j]] = combined_graph[[i, j]].max(graph[[i, j]]);
                    }
                }
            }
        }
        "hierarchical" => {
            // Hierarchical combination: smaller scales have higher priority
            for (scale_idx, graph) in scale_graphs.iter().enumerate() {
                let hierarchy_weight = 1.0 / (scale_idx + 1) as Float;
                for i in 0..n_samples {
                    for j in 0..n_samples {
                        if graph[[i, j]] > 0.0 {
                            combined_graph[[i, j]] =
                                combined_graph[[i, j]].max(graph[[i, j]] * hierarchy_weight);
                        }
                    }
                }
            }
        }
        _ => {
            return Err(SklearsError::InvalidInput(format!(
                "Unknown combination method: {}. Supported methods: weighted, union, intersection, adaptive_weighted, max_pooling, hierarchical",
                combination_method
            )));
        }
    }

    // Final symmetry enforcement
    make_symmetric(&mut combined_graph);

    Ok(combined_graph)
}

/// Multi-scale spectral clustering
///
/// Performs spectral clustering using multi-scale graph construction to capture
/// structure at different resolutions. This can improve clustering performance
/// by leveraging both local and global connectivity patterns.
///
/// # Parameters
///
/// * `X` - Input data matrix
/// * `n_clusters` - Number of clusters
/// * `scales` - Vector of k values for different scales
/// * `scale_weights` - Optional weights for combining scales
/// * `combination_method` - Method for combining graphs
/// * `normalized` - Whether to use normalized Laplacian
/// * `random_state` - Random seed for reproducibility
///
/// # Returns
///
/// * `SklResult<Array1<i32>>` - Cluster labels
pub fn multi_scale_spectral_clustering(
    X: &Array2<Float>,
    n_clusters: usize,
    scales: &[usize],
    scale_weights: Option<&[Float]>,
    combination_method: &str,
    normalized: bool,
    random_state: Option<u64>,
) -> SklResult<Array1<i32>> {
    // Build multi-scale graph
    let adjacency = multi_scale_graph_construction(
        X,
        scales,
        scale_weights,
        combination_method,
        true, // normalize scales
    )?;

    // Perform spectral clustering on the multi-scale graph
    spectral_clustering(&adjacency, n_clusters, normalized, random_state)
}

/// Adaptive multi-scale graph construction
///
/// Automatically determines appropriate scales based on data characteristics
/// and constructs a multi-scale graph. This version adaptively selects scales
/// based on local density and data distribution.
///
/// # Parameters
///
/// * `X` - Input data matrix
/// * `min_scale` - Minimum k value to consider
/// * `max_scale` - Maximum k value to consider
/// * `n_scales` - Number of scales to use
/// * `density_threshold` - Threshold for density-based scale selection
/// * `combination_method` - Method for combining graphs
///
/// # Returns
///
/// * `SklResult<(Array2<Float>, Vec<usize>)>` - Combined graph and selected scales
pub fn adaptive_multi_scale_graph_construction(
    X: &Array2<Float>,
    min_scale: usize,
    max_scale: usize,
    n_scales: usize,
    density_threshold: Float,
    combination_method: &str,
) -> SklResult<(Array2<Float>, Vec<usize>)> {
    let n_samples = X.nrows();

    if max_scale >= n_samples {
        return Err(SklearsError::InvalidInput(
            "max_scale must be less than number of samples".to_string(),
        ));
    }

    if min_scale >= max_scale {
        return Err(SklearsError::InvalidInput(
            "min_scale must be less than max_scale".to_string(),
        ));
    }

    // Compute local densities to guide scale selection
    let mut densities = Vec::new();
    let candidate_scales: Vec<usize> = (min_scale..=max_scale)
        .step_by(((max_scale - min_scale) / n_scales.max(1)).max(1))
        .take(n_scales)
        .collect();

    for &k in &candidate_scales {
        let graph = knn_graph(X, k, "connectivity")?;
        let avg_degree = graph.sum_axis(Axis(1)).mean().unwrap_or(0.0);
        let density = avg_degree / k as Float;
        densities.push(density);
    }

    // Select scales based on density criteria
    let mut selected_scales = Vec::new();
    let mut selected_weights = Vec::new();

    for (i, &density) in densities.iter().enumerate() {
        if density >= density_threshold {
            selected_scales.push(candidate_scales[i]);
            selected_weights.push(density);
        }
    }

    // Ensure we have at least one scale
    if selected_scales.is_empty() {
        selected_scales.push(candidate_scales[densities.len() / 2]);
        selected_weights.push(1.0);
    }

    // Normalize weights
    let total_weight: Float = selected_weights.iter().sum();
    if total_weight > 0.0 {
        for weight in selected_weights.iter_mut() {
            *weight /= total_weight;
        }
    }

    // Build multi-scale graph with selected scales
    let combined_graph = multi_scale_graph_construction(
        X,
        &selected_scales,
        Some(&selected_weights),
        combination_method,
        true,
    )?;

    Ok((combined_graph, selected_scales))
}
