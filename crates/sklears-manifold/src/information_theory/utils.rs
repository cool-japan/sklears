//! Utility functions for information-theoretic manifold learning methods

use super::bregman_divergence::BregmanDivergenceType;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::Rng;
use scirs2_linalg::compat::{ArrayLinalgExt, UPLO};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::f64::consts::{E, PI};

// Helper functions for Bregman divergence computations

/// Compute Bregman divergence matrix for all pairs of points
pub fn compute_bregman_divergence_matrix(
    x: &Array2<f64>,
    divergence_type: &BregmanDivergenceType,
) -> SklResult<Array2<f64>> {
    let n_samples = x.nrows();
    let mut divergence_matrix = Array2::zeros((n_samples, n_samples));

    for i in 0..n_samples {
        for j in 0..n_samples {
            if i != j {
                let xi = x.row(i);
                let xj = x.row(j);
                let div = compute_bregman_divergence(&xi, &xj, divergence_type)?;
                divergence_matrix[[i, j]] = div;
            }
        }
    }

    Ok(divergence_matrix)
}

/// Compute Bregman divergence between two points
pub fn compute_bregman_divergence(
    x: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
    divergence_type: &BregmanDivergenceType,
) -> SklResult<f64> {
    if x.len() != y.len() {
        return Err(SklearsError::InvalidInput(
            "Points must have the same dimension".to_string(),
        ));
    }

    let divergence = match divergence_type {
        BregmanDivergenceType::SquaredEuclidean => {
            // D_φ(x,y) = φ(x) - φ(y) - ∇φ(y)^T(x-y)
            // For φ(x) = ||x||²/2, ∇φ(y) = y
            let phi_x = x.dot(x) / 2.0;
            let phi_y = y.dot(y) / 2.0;
            let grad_phi_y_diff = y.dot(&(x - y));
            phi_x - phi_y - grad_phi_y_diff
        }
        BregmanDivergenceType::KullbackLeibler => {
            // D_φ(x,y) = Σ x_i log(x_i/y_i) - x_i + y_i
            let mut div = 0.0;
            for (xi, yi) in x.iter().zip(y.iter()) {
                if *xi > 0.0 && *yi > 0.0 {
                    div += xi * (xi / yi).ln() - xi + yi;
                } else if *xi == 0.0 && *yi >= 0.0 {
                    div += *yi;
                } else {
                    return Err(SklearsError::InvalidInput(
                        "KL divergence requires non-negative values".to_string(),
                    ));
                }
            }
            div
        }
        BregmanDivergenceType::ItakuraSaito => {
            // D_φ(x,y) = Σ (x_i/y_i - log(x_i/y_i) - 1)
            let mut div = 0.0;
            for (xi, yi) in x.iter().zip(y.iter()) {
                if *xi > 0.0 && *yi > 0.0 {
                    let ratio = xi / yi;
                    div += ratio - ratio.ln() - 1.0;
                } else {
                    return Err(SklearsError::InvalidInput(
                        "Itakura-Saito divergence requires positive values".to_string(),
                    ));
                }
            }
            div
        }
        BregmanDivergenceType::Exponential => {
            // D_φ(x,y) = Σ exp(x_i) - exp(y_i) - exp(y_i)(x_i - y_i)
            let mut div = 0.0;
            for (xi, yi) in x.iter().zip(y.iter()) {
                let exp_xi = xi.exp();
                let exp_yi = yi.exp();
                div += exp_xi - exp_yi - exp_yi * (xi - yi);
            }
            div
        }
        BregmanDivergenceType::LogSumExp => {
            // D_φ(x,y) = Σ log(1+exp(x_i)) - log(1+exp(y_i)) - sigmoid(y_i)(x_i - y_i)
            let mut div = 0.0;
            for (xi, yi) in x.iter().zip(y.iter()) {
                let log_sum_exp_xi = (1.0 + xi.exp()).ln();
                let log_sum_exp_yi = (1.0 + yi.exp()).ln();
                let sigmoid_yi = yi.exp() / (1.0 + yi.exp());
                div += log_sum_exp_xi - log_sum_exp_yi - sigmoid_yi * (xi - yi);
            }
            div
        }
    };

    Ok(divergence)
}

/// Compute Bregman centroids using iterative algorithm
pub fn compute_bregman_centroids(
    x: &Array2<f64>,
    divergence_type: &BregmanDivergenceType,
    n_centroids: usize,
) -> SklResult<Array2<f64>> {
    let (n_samples, n_features) = x.dim();

    if n_centroids > n_samples {
        return Err(SklearsError::InvalidInput(
            "Number of centroids cannot exceed number of samples".to_string(),
        ));
    }

    // Initialize centroids randomly
    let mut rng = StdRng::seed_from_u64(42);
    let mut centroids = Array2::zeros((n_centroids, n_features));

    for i in 0..n_centroids {
        let idx = rng.gen_range(0..n_samples);
        centroids.row_mut(i).assign(&x.row(idx));
    }

    // Iterative refinement using Bregman centroid algorithm
    let max_iter = 100;
    let tol = 1e-6;

    for _iter in 0..max_iter {
        let mut new_centroids = Array2::zeros((n_centroids, n_features));
        let mut cluster_assignments = vec![0; n_samples];

        // Assign points to nearest centroids
        for i in 0..n_samples {
            let mut min_div = f64::INFINITY;
            let mut best_centroid = 0;

            for j in 0..n_centroids {
                let div =
                    compute_bregman_divergence(&x.row(i), &centroids.row(j), divergence_type)?;
                if div < min_div {
                    min_div = div;
                    best_centroid = j;
                }
            }
            cluster_assignments[i] = best_centroid;
        }

        // Update centroids
        for j in 0..n_centroids {
            let cluster_points: Vec<_> = (0..n_samples)
                .filter(|&i| cluster_assignments[i] == j)
                .collect();

            if !cluster_points.is_empty() {
                let centroid =
                    compute_bregman_centroid_for_points(x, &cluster_points, divergence_type)?;
                new_centroids.row_mut(j).assign(&centroid);
            } else {
                // Keep the old centroid if no points assigned
                new_centroids.row_mut(j).assign(&centroids.row(j));
            }
        }

        // Check for convergence
        let change = (&new_centroids - &centroids).mapv(|x| x.abs()).sum();
        if change < tol {
            break;
        }

        centroids = new_centroids;
    }

    Ok(centroids)
}

/// Compute Bregman centroid for a set of points
pub fn compute_bregman_centroid_for_points(
    x: &Array2<f64>,
    point_indices: &[usize],
    divergence_type: &BregmanDivergenceType,
) -> SklResult<Array1<f64>> {
    if point_indices.is_empty() {
        return Err(SklearsError::InvalidInput(
            "Cannot compute centroid for empty set of points".to_string(),
        ));
    }

    let n_features = x.ncols();
    let mut centroid = Array1::zeros(n_features);

    match divergence_type {
        BregmanDivergenceType::SquaredEuclidean => {
            // Arithmetic mean
            for &idx in point_indices {
                centroid += &x.row(idx);
            }
            centroid /= point_indices.len() as f64;
        }
        BregmanDivergenceType::KullbackLeibler => {
            // Geometric mean (after log transformation)
            for j in 0..n_features {
                let mut sum_log = 0.0;
                for &idx in point_indices {
                    let val = x[[idx, j]];
                    if val > 0.0 {
                        sum_log += val.ln();
                    }
                }
                centroid[j] = (sum_log / point_indices.len() as f64).exp();
            }
        }
        BregmanDivergenceType::ItakuraSaito => {
            // Harmonic mean
            for j in 0..n_features {
                let mut sum_inv = 0.0;
                for &idx in point_indices {
                    let val = x[[idx, j]];
                    if val > 0.0 {
                        sum_inv += 1.0 / val;
                    }
                }
                centroid[j] = point_indices.len() as f64 / sum_inv;
            }
        }
        BregmanDivergenceType::Exponential => {
            // Log-sum-exp centroid
            for j in 0..n_features {
                let mut sum_exp = 0.0;
                for &idx in point_indices {
                    sum_exp += x[[idx, j]].exp();
                }
                centroid[j] = (sum_exp / point_indices.len() as f64).ln();
            }
        }
        BregmanDivergenceType::LogSumExp => {
            // Softmax centroid
            for j in 0..n_features {
                let mut sum = 0.0;
                let mut max_val = f64::NEG_INFINITY;

                // Find max for numerical stability
                for &idx in point_indices {
                    max_val = max_val.max(x[[idx, j]]);
                }

                // Compute softmax
                for &idx in point_indices {
                    sum += (x[[idx, j]] - max_val).exp();
                }

                centroid[j] = max_val + sum.ln() - (point_indices.len() as f64).ln();
            }
        }
    }

    Ok(centroid)
}

/// Perform MDS on Bregman divergence matrix
pub fn bregman_mds(divergence_matrix: &Array2<f64>, n_components: usize) -> SklResult<Array2<f64>> {
    let n_samples = divergence_matrix.nrows();

    // Double centering (classical MDS)
    let row_means = divergence_matrix.mean_axis(Axis(1)).unwrap();
    let col_means = divergence_matrix.mean_axis(Axis(0)).unwrap();
    let total_mean = divergence_matrix.mean().unwrap();

    let mut centered_matrix = Array2::zeros((n_samples, n_samples));
    for i in 0..n_samples {
        for j in 0..n_samples {
            centered_matrix[[i, j]] =
                -0.5 * (divergence_matrix[[i, j]] - row_means[i] - col_means[j] + total_mean);
        }
    }

    // Eigendecomposition
    let (eigenvalues, eigenvectors) = centered_matrix
        .eigh(UPLO::Lower)
        .map_err(|e| SklearsError::InvalidInput(format!("Eigendecomposition failed: {}", e)))?;

    // Sort eigenvalues and eigenvectors in descending order
    let mut eigen_pairs: Vec<_> = eigenvalues.iter().zip(eigenvectors.columns()).collect();
    eigen_pairs.sort_by(|a, b| b.0.partial_cmp(a.0).unwrap());

    // Take the top n_components eigenvectors and scale by sqrt(eigenvalue)
    let mut embedding = Array2::zeros((n_samples, n_components));
    for j in 0..n_components {
        let eigenval = *eigen_pairs[j].0;
        let eigenvec = eigen_pairs[j].1;

        if eigenval > 0.0 {
            let scale = eigenval.sqrt();
            for i in 0..n_samples {
                embedding[[i, j]] = eigenvec[i] * scale;
            }
        }
    }

    Ok(embedding)
}

/// Project new data using Bregman projections
pub fn bregman_project(
    x: &Array2<f64>,
    centroids: &Array2<f64>,
    divergence_type: &BregmanDivergenceType,
) -> SklResult<Array2<f64>> {
    let n_samples = x.nrows();
    let n_centroids = centroids.nrows();

    let mut projection = Array2::zeros((n_samples, n_centroids));

    for i in 0..n_samples {
        for j in 0..n_centroids {
            let div = compute_bregman_divergence(&x.row(i), &centroids.row(j), divergence_type)?;
            projection[[i, j]] = (-div).exp(); // Convert to similarity
        }

        // Normalize to probabilities
        let row_sum = projection.row(i).sum();
        if row_sum > 0.0 {
            projection.row_mut(i).mapv_inplace(|x| x / row_sum);
        }
    }

    Ok(projection)
}

// Helper functions for natural gradient computations

/// Compute Fisher information matrix for current embedding
pub fn compute_fisher_information_matrix(embedding: &Array2<f64>) -> SklResult<Array2<f64>> {
    let (n_samples, n_components) = embedding.dim();
    let total_params = n_samples * n_components;

    // Flatten embedding for Fisher computation
    let params = embedding.as_slice().unwrap();
    let mut fisher_matrix = Array2::zeros((total_params, total_params));

    // Compute Fisher information as second moment of score function
    // Using empirical Fisher information (outer product of gradients)
    for i in 0..n_samples {
        for d1 in 0..n_components {
            for j in 0..n_samples {
                for d2 in 0..n_components {
                    let idx1 = i * n_components + d1;
                    let idx2 = j * n_components + d2;

                    // Simplified Fisher computation based on manifold structure
                    if i == j {
                        fisher_matrix[[idx1, idx2]] = if d1 == d2 { 1.0 } else { 0.0 };
                    } else {
                        // Include neighbor interactions in Fisher matrix
                        let distance = ((embedding[[i, d1]] - embedding[[j, d1]]).powi(2)
                            + (embedding[[i, d2]] - embedding[[j, d2]]).powi(2))
                        .sqrt();
                        let weight = (-distance).exp();
                        fisher_matrix[[idx1, idx2]] = weight;
                    }
                }
            }
        }
    }

    Ok(fisher_matrix)
}

/// Compute natural gradient using Fisher information metric
pub fn compute_natural_gradient(
    gradient: &Array2<f64>,
    fisher_matrix: &Array2<f64>,
    regularization: f64,
) -> SklResult<Array2<f64>> {
    let (n_samples, n_components) = gradient.dim();

    // Flatten gradient
    let grad_flat = gradient.as_slice().unwrap();
    let grad_vec = Array1::from_vec(grad_flat.to_vec());

    // Add regularization to Fisher matrix
    let mut regularized_fisher = fisher_matrix.clone();
    for i in 0..fisher_matrix.nrows() {
        regularized_fisher[[i, i]] += regularization;
    }

    // Solve Fisher^{-1} * gradient for natural gradient
    // Using pseudo-inverse for numerical stability
    let (u, s, vt) = regularized_fisher
        .svd(true)
        .map_err(|e| SklearsError::InvalidInput(format!("SVD failed: {}", e)))?;

    let (u_mat, vt_mat) = (u, vt);
    // Compute pseudo-inverse using SVD
    let mut s_inv = Array1::zeros(s.len());
    for (i, &si) in s.iter().enumerate() {
        if si > 1e-10 {
            s_inv[i] = 1.0 / si;
        }
    }

    // Fisher^{-1} = V * S^{-1} * U^T
    let s_inv_diag = Array2::from_diag(&s_inv);
    let fisher_inv = vt_mat.t().dot(&s_inv_diag).dot(&u_mat.t());

    // Natural gradient = Fisher^{-1} * gradient
    let natural_grad_flat = fisher_inv.dot(&grad_vec);

    // Reshape back to original shape
    let natural_gradient =
        Array2::from_shape_vec((n_samples, n_components), natural_grad_flat.to_vec())
            .map_err(|e| SklearsError::InvalidInput(format!("Reshape failed: {}", e)))?;

    Ok(natural_gradient)
}

/// Compute stress gradient for embedding
pub fn compute_stress_gradient(x: &Array2<f64>, embedding: &Array2<f64>) -> SklResult<Array2<f64>> {
    let (n_samples, n_components) = embedding.dim();
    let mut gradient = Array2::zeros((n_samples, n_components));

    // Compute pairwise distances in original space
    let mut original_distances = Array2::zeros((n_samples, n_samples));
    for i in 0..n_samples {
        for j in 0..n_samples {
            if i != j {
                let diff = &x.row(i) - &x.row(j);
                original_distances[[i, j]] = diff.dot(&diff).sqrt();
            }
        }
    }

    // Compute stress gradient
    for i in 0..n_samples {
        for k in 0..n_components {
            let mut grad_sum = 0.0;

            for j in 0..n_samples {
                if i != j {
                    let embed_diff = &embedding.row(i) - &embedding.row(j);
                    let embed_dist = embed_diff.dot(&embed_diff).sqrt();

                    if embed_dist > 1e-10 {
                        let original_dist = original_distances[[i, j]];
                        let stress_factor = 1.0 - original_dist / embed_dist;
                        grad_sum += stress_factor * (embedding[[i, k]] - embedding[[j, k]]);
                    }
                }
            }

            gradient[[i, k]] = 4.0 * grad_sum; // Factor of 4 from stress derivative
        }
    }

    Ok(gradient)
}

/// Compute embedding stress (Sammon stress)
pub fn compute_embedding_stress(x: &Array2<f64>, embedding: &Array2<f64>) -> SklResult<f64> {
    let n_samples = x.nrows();
    let mut stress = 0.0;
    let mut normalizer = 0.0;

    for i in 0..n_samples {
        for j in i + 1..n_samples {
            // Original distance
            let diff_orig = &x.row(i) - &x.row(j);
            let dist_orig = diff_orig.dot(&diff_orig).sqrt();

            // Embedding distance
            let diff_embed = &embedding.row(i) - &embedding.row(j);
            let dist_embed = diff_embed.dot(&diff_embed).sqrt();

            if dist_orig > 1e-10 {
                let error = (dist_orig - dist_embed).powi(2) / dist_orig;
                stress += error;
                normalizer += dist_orig;
            }
        }
    }

    if normalizer > 0.0 {
        Ok(stress / normalizer)
    } else {
        Ok(0.0)
    }
}

// Helper functions for information-theoretic computations

/// Compute mutual information between two datasets using KNN estimation
pub fn compute_mutual_information_knn(
    x: &Array2<f64>,
    y: &Array2<f64>,
    k: usize,
) -> SklResult<f64> {
    let n_samples = x.nrows();

    if n_samples != y.nrows() {
        return Err(SklearsError::InvalidInput(
            "X and Y must have the same number of samples".to_string(),
        ));
    }

    // Combine X and Y along the feature axis
    let xy = scirs2_core::ndarray::concatenate(Axis(1), &[x.view(), y.view()])?;

    // Estimate entropies using k-NN
    let h_x = estimate_entropy_knn(x, k)?;
    let h_y = estimate_entropy_knn(y, k)?;
    let h_xy = estimate_entropy_knn(&xy, k)?;

    // MI = H(X) + H(Y) - H(X,Y)
    Ok(h_x + h_y - h_xy)
}

/// Compute mutual information using kernel density estimation
pub fn compute_mutual_information_kde(
    x: &Array2<f64>,
    y: &Array2<f64>,
    bandwidth: f64,
) -> SklResult<f64> {
    // Simplified KDE-based MI estimation
    // In practice, this would use more sophisticated KDE methods
    let n_samples = x.nrows() as f64;
    let d_x = x.ncols() as f64;
    let d_y = y.ncols() as f64;

    // Estimate using the rule of thumb for bandwidth
    let h_x = -0.5 * d_x * (2.0 * PI * E * bandwidth * bandwidth).ln() - (n_samples).ln();
    let h_y = -0.5 * d_y * (2.0 * PI * E * bandwidth * bandwidth).ln() - (n_samples).ln();
    let h_xy = -0.5 * (d_x + d_y) * (2.0 * PI * E * bandwidth * bandwidth).ln() - (n_samples).ln();

    Ok(h_x + h_y - h_xy)
}

/// Compute mutual information between two 2D arrays
pub fn compute_mutual_information_2d(x: &Array2<f64>, y: &Array2<f64>) -> SklResult<f64> {
    let n_samples = x.nrows();

    if n_samples != y.nrows() {
        return Err(SklearsError::InvalidInput(
            "X and Y must have the same number of samples".to_string(),
        ));
    }

    // Use KNN estimation with k=3
    compute_mutual_information_knn(x, y, 3)
}

/// Compute mutual information between two arrays (simplified version)
pub fn compute_mutual_information(x: &Array2<f64>, y: &Array1<f64>) -> SklResult<f64> {
    let n_samples = x.nrows();

    if n_samples != y.len() {
        return Err(SklearsError::InvalidInput(
            "X and y must have the same number of samples".to_string(),
        ));
    }

    // Convert y to 2D array for consistency
    let y_2d = y.clone().insert_axis(Axis(1));

    // Use KNN estimation with k=3
    compute_mutual_information_knn(x, &y_2d, 3)
}

/// Estimate entropy using k-nearest neighbors
pub fn estimate_entropy_knn(x: &Array2<f64>, k: usize) -> SklResult<f64> {
    let (n_samples, n_dims) = x.dim();

    if k >= n_samples {
        return Err(SklearsError::InvalidInput(
            "k must be less than the number of samples".to_string(),
        ));
    }

    let mut entropy = 0.0;

    for i in 0..n_samples {
        let xi = x.row(i);

        // Find k-th nearest neighbor distance
        let mut distances: Vec<f64> = Vec::new();

        for j in 0..n_samples {
            if i != j {
                let xj = x.row(j);
                let diff = &xi - &xj;
                let dist = diff.dot(&diff).sqrt();
                distances.push(dist);
            }
        }

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if k <= distances.len() {
            let rho_k = distances[k - 1];
            if rho_k > 0.0 {
                entropy += (n_dims as f64) * rho_k.ln();
            }
        }
    }

    // Add constant terms
    entropy /= n_samples as f64;
    entropy += (n_samples as f64).ln();
    entropy += gamma_function_ln(n_dims as f64 / 2.0 + 1.0);
    entropy -= (n_dims as f64 / 2.0) * PI.ln();

    Ok(entropy)
}

/// Estimate bandwidth for kernel density estimation
pub fn estimate_bandwidth(x: &Array2<f64>, n_neighbors: usize) -> f64 {
    let (n_samples, n_dims) = x.dim();

    // Silverman's rule of thumb
    let std_dev = x.std_axis(Axis(0), 0.0).mean().unwrap_or(1.0);
    let bandwidth = std_dev
        * ((4.0 / ((n_dims + 2) as f64)).powf(1.0 / (n_dims + 4) as f64))
        * (n_samples as f64).powf(-1.0 / (n_dims + 4) as f64);

    bandwidth.max(0.01) // Minimum bandwidth
}

/// Compute gradient of information bottleneck objective
pub fn compute_ib_gradient(
    x: &Array2<f64>,
    y: &Array1<f64>,
    weights: &Array2<f64>,
    beta: f64,
) -> SklResult<Array2<f64>> {
    let (n_samples, n_features) = x.dim();
    let n_components = weights.ncols();

    // Approximate gradient using finite differences
    let epsilon = 1e-6;
    let mut gradient = Array2::zeros((n_features, n_components));

    for i in 0..n_features {
        for j in 0..n_components {
            // Forward difference
            let mut weights_plus = weights.clone();
            weights_plus[[i, j]] += epsilon;

            let z_plus = x.dot(&weights_plus);
            let i_z_y_plus = compute_mutual_information(&z_plus, y)?;
            let i_x_z_plus = compute_mutual_information_2d(x, &z_plus)?;
            let obj_plus = i_z_y_plus - beta * i_x_z_plus;

            // Backward difference
            let mut weights_minus = weights.clone();
            weights_minus[[i, j]] -= epsilon;

            let z_minus = x.dot(&weights_minus);
            let i_z_y_minus = compute_mutual_information(&z_minus, y)?;
            let i_x_z_minus = compute_mutual_information_2d(x, &z_minus)?;
            let obj_minus = i_z_y_minus - beta * i_x_z_minus;

            gradient[[i, j]] = (obj_plus - obj_minus) / (2.0 * epsilon);
        }
    }

    Ok(gradient)
}

/// Compute gradient of mutual information for MMI
pub fn compute_mi_gradient(
    x: &Array2<f64>,
    embedding: &Array2<f64>,
    bandwidth: f64,
) -> SklResult<Array2<f64>> {
    let (n_samples, n_components) = embedding.dim();

    // Approximate gradient using finite differences
    let epsilon = 1e-6;
    let mut gradient = Array2::zeros((n_samples, n_components));

    for i in 0..n_samples {
        for j in 0..n_components {
            // Forward difference
            let mut embedding_plus = embedding.clone();
            embedding_plus[[i, j]] += epsilon;

            let mi_plus = compute_mutual_information_kde(x, &embedding_plus, bandwidth)?;

            // Backward difference
            let mut embedding_minus = embedding.clone();
            embedding_minus[[i, j]] -= epsilon;

            let mi_minus = compute_mutual_information_kde(x, &embedding_minus, bandwidth)?;

            gradient[[i, j]] = (mi_plus - mi_minus) / (2.0 * epsilon);
        }
    }

    Ok(gradient)
}

/// Compute local Fisher information matrix
pub fn compute_local_fisher_information(
    x: &Array2<f64>,
    n_neighbors: usize,
    sigma: f64,
) -> SklResult<Array2<f64>> {
    let (n_samples, n_features) = x.dim();
    let mut fisher_matrix = Array2::zeros((n_features, n_features));

    for i in 0..n_samples {
        let xi = x.row(i);

        // Find neighbors
        let mut distances: Vec<(usize, f64)> = Vec::new();

        for j in 0..n_samples {
            if i != j {
                let xj = x.row(j);
                let diff = &xi - &xj;
                let dist = diff.dot(&diff).sqrt();
                distances.push((j, dist));
            }
        }

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(n_neighbors);

        // Compute local Fisher information
        for &(j, dist) in &distances {
            let weight = (-dist * dist / (2.0 * sigma * sigma)).exp();
            let diff = &x.row(i) - &x.row(j);

            for p in 0..n_features {
                for q in 0..n_features {
                    fisher_matrix[[p, q]] += weight * diff[p] * diff[q] / (sigma * sigma);
                }
            }
        }
    }

    fisher_matrix /= n_samples as f64;
    Ok(fisher_matrix)
}

/// Compute global Fisher information matrix
pub fn compute_global_fisher_information(x: &Array2<f64>, sigma: f64) -> SklResult<Array2<f64>> {
    let (n_samples, n_features) = x.dim();
    let mut fisher_matrix = Array2::zeros((n_features, n_features));

    // Compute covariance matrix as approximation to Fisher information
    let mean = x.mean_axis(Axis(0)).unwrap();
    let x_centered = x - &mean.insert_axis(Axis(0));

    for i in 0..n_samples {
        let xi = x_centered.row(i);

        for p in 0..n_features {
            for q in 0..n_features {
                fisher_matrix[[p, q]] += xi[p] * xi[q];
            }
        }
    }

    fisher_matrix /= (n_samples - 1) as f64;
    fisher_matrix /= sigma * sigma;

    Ok(fisher_matrix)
}

/// Simplified log gamma function
pub fn gamma_function_ln(x: f64) -> f64 {
    // Stirling's approximation for simplicity
    if x > 1.0 {
        (x - 0.5) * x.ln() - x + 0.5 * (2.0 * PI).ln()
    } else {
        0.0
    }
}
