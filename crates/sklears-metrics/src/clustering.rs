//! Clustering evaluation metrics and validation utilities
//!
//! This module provides comprehensive clustering evaluation implementations.
//! Metrics include both supervised (comparing clusterings) and unsupervised
//! (intrinsic quality) evaluation methods.

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use std::collections::HashMap;

// ============================================================================
// Helper Functions
// ============================================================================

/// Build a contingency matrix describing the relationship between two labelings
fn contingency_matrix(labels_true: &ArrayView1<i32>, labels_pred: &ArrayView1<i32>) -> Array2<i64> {
    let n = labels_true.len();
    if n != labels_pred.len() {
        panic!("labels_true and labels_pred must have the same length");
    }

    // Find unique labels
    let mut classes_set = std::collections::HashSet::new();
    let mut clusters_set = std::collections::HashSet::new();

    for &label in labels_true.iter() {
        classes_set.insert(label);
    }
    for &label in labels_pred.iter() {
        clusters_set.insert(label);
    }

    let mut classes: Vec<i32> = classes_set.into_iter().collect();
    let mut clusters: Vec<i32> = clusters_set.into_iter().collect();
    classes.sort_unstable();
    clusters.sort_unstable();

    // Create index mappings
    let class_to_idx: HashMap<i32, usize> =
        classes.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    let cluster_to_idx: HashMap<i32, usize> =
        clusters.iter().enumerate().map(|(i, &c)| (c, i)).collect();

    let n_classes = classes.len();
    let n_clusters = clusters.len();

    let mut contingency = Array2::<i64>::zeros((n_classes, n_clusters));

    for i in 0..n {
        let class_idx = class_to_idx[&labels_true[i]];
        let cluster_idx = cluster_to_idx[&labels_pred[i]];
        contingency[[class_idx, cluster_idx]] += 1;
    }

    contingency
}

/// Compute pair confusion matrix for two clusterings
fn pair_confusion_matrix(
    labels_true: &ArrayView1<i32>,
    labels_pred: &ArrayView1<i32>,
) -> Array2<i64> {
    let n = labels_true.len() as i64;
    let contingency = contingency_matrix(labels_true, labels_pred);

    // Compute sums
    let n_c: Vec<i64> = contingency.sum_axis(Axis(1)).iter().copied().collect();
    let n_k: Vec<i64> = contingency.sum_axis(Axis(0)).iter().copied().collect();

    let sum_squares: i64 = contingency.iter().map(|&x| x * x).sum();

    let mut c = Array2::<i64>::zeros((2, 2));

    // C[1,1] = sum of squares - n
    c[[1, 1]] = sum_squares - n;

    // C[0,1] = sum of (n_c[i] * n_k[j]) for all i,j - sum_squares
    let mut sum_prod_k = 0i64;
    for row in contingency.axis_iter(Axis(0)) {
        for (j, &val) in row.iter().enumerate() {
            if val > 0 {
                sum_prod_k += val * n_k[j];
            }
        }
    }
    c[[0, 1]] = sum_prod_k - sum_squares;

    // C[1,0] = sum of (n_c[i] * n_k[j]) for all i,j - sum_squares (transposed)
    let mut sum_prod_c = 0i64;
    for col in contingency.axis_iter(Axis(1)) {
        for (i, &val) in col.iter().enumerate() {
            if val > 0 {
                sum_prod_c += val * n_c[i];
            }
        }
    }
    c[[1, 0]] = sum_prod_c - sum_squares;

    // C[0,0] = n^2 - C[0,1] - C[1,0] - sum_squares
    c[[0, 0]] = n * n - c[[0, 1]] - c[[1, 0]] - sum_squares;

    c
}

/// Calculate entropy for a labeling
fn entropy(labels: &ArrayView1<i32>) -> f64 {
    let n = labels.len();
    if n == 0 {
        return 1.0;
    }

    // Count label frequencies
    let mut counts = HashMap::new();
    for &label in labels.iter() {
        *counts.entry(label).or_insert(0) += 1;
    }

    // Single cluster => zero entropy
    if counts.len() == 1 {
        return 0.0;
    }

    // Compute entropy: -sum(p_i * log(p_i))
    let n_f64 = n as f64;
    let mut h = 0.0;
    for &count in counts.values() {
        if count > 0 {
            let p = count as f64 / n_f64;
            h -= p * p.ln();
        }
    }
    h
}

// ============================================================================
// Supervised Clustering Metrics (comparing two labelings)
// ============================================================================

/// Compute the Rand Index between two clusterings
///
/// The Rand Index measures similarity between two clusterings by counting
/// pairs of samples that are assigned to the same or different clusters.
///
/// # Arguments
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Cluster labels to evaluate
///
/// # Returns
/// * Similarity score between 0.0 and 1.0 (1.0 = perfect match)
pub fn rand_score(labels_true: &Array1<i32>, labels_pred: &Array1<i32>) -> MetricsResult<f64> {
    if labels_true.len() != labels_pred.len() {
        return Err(MetricsError::InvalidInput(
            "labels_true and labels_pred must have the same length".to_string(),
        ));
    }

    let contingency = pair_confusion_matrix(&labels_true.view(), &labels_pred.view());
    let numerator = contingency[[0, 0]] + contingency[[1, 1]];
    let denominator = contingency.iter().sum::<i64>();

    if numerator == denominator || denominator == 0 {
        return Ok(1.0);
    }

    Ok(numerator as f64 / denominator as f64)
}

/// Compute the Adjusted Rand Index between two clusterings
///
/// The ARI is the Rand Index adjusted for chance, with values:
/// - 1.0: perfect match
/// - 0.0: random labeling
/// - Can be negative for especially discordant clusterings
///
/// # Arguments
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Cluster labels to evaluate
///
/// # Returns
/// * ARI score between -0.5 and 1.0
pub fn adjusted_rand_score(
    labels_true: &Array1<i32>,
    labels_pred: &Array1<i32>,
) -> MetricsResult<f64> {
    if labels_true.len() != labels_pred.len() {
        return Err(MetricsError::InvalidInput(
            "labels_true and labels_pred must have the same length".to_string(),
        ));
    }

    let confusion = pair_confusion_matrix(&labels_true.view(), &labels_pred.view());
    let tn = confusion[[0, 0]] as f64;
    let fp = confusion[[0, 1]] as f64;
    let fn_val = confusion[[1, 0]] as f64;
    let tp = confusion[[1, 1]] as f64;

    // Special case: full agreement
    if fn_val == 0.0 && fp == 0.0 {
        return Ok(1.0);
    }

    // ARI formula: 2 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    let numerator = 2.0 * (tp * tn - fn_val * fp);
    let denominator = (tp + fn_val) * (fn_val + tn) + (tp + fp) * (fp + tn);

    Ok(numerator / denominator)
}

/// Compute the Fowlkes-Mallows Index between two clusterings
///
/// The FMI is the geometric mean of pairwise precision and recall.
///
/// # Arguments
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Cluster labels to evaluate
///
/// # Returns
/// * FMI score between 0.0 and 1.0
pub fn fowlkes_mallows_score(
    labels_true: &Array1<i32>,
    labels_pred: &Array1<i32>,
) -> MetricsResult<f64> {
    if labels_true.len() != labels_pred.len() {
        return Err(MetricsError::InvalidInput(
            "labels_true and labels_pred must have the same length".to_string(),
        ));
    }

    let confusion = pair_confusion_matrix(&labels_true.view(), &labels_pred.view());
    let tp = confusion[[1, 1]] as f64;
    let fp = confusion[[0, 1]] as f64;
    let fn_val = confusion[[1, 0]] as f64;

    if tp + fp == 0.0 || tp + fn_val == 0.0 {
        return Ok(0.0);
    }

    let precision = tp / (tp + fp);
    let recall = tp / (tp + fn_val);

    Ok((precision * recall).sqrt())
}

/// Compute Mutual Information between two clusterings
///
/// MI(U,V) = sum_ij P(i,j) * log(P(i,j) / (P(i) * P(j)))
///
/// # Arguments
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Cluster labels to evaluate
///
/// # Returns
/// * Mutual information score (non-negative, measured in nats)
pub fn mutual_info_score(
    labels_true: &Array1<i32>,
    labels_pred: &Array1<i32>,
) -> MetricsResult<f64> {
    if labels_true.len() != labels_pred.len() {
        return Err(MetricsError::InvalidInput(
            "labels_true and labels_pred must have the same length".to_string(),
        ));
    }

    let contingency = contingency_matrix(&labels_true.view(), &labels_pred.view());
    let n = labels_true.len() as f64;

    // Row and column sums
    let pi: Vec<f64> = contingency
        .sum_axis(Axis(1))
        .iter()
        .map(|&x| x as f64)
        .collect();
    let pj: Vec<f64> = contingency
        .sum_axis(Axis(0))
        .iter()
        .map(|&x| x as f64)
        .collect();

    // Single cluster => MI = 0
    if pi.len() == 1 || pj.len() == 1 {
        return Ok(0.0);
    }

    let mut mi = 0.0;
    for (i, row) in contingency.axis_iter(Axis(0)).enumerate() {
        for (j, &n_ij) in row.iter().enumerate() {
            if n_ij > 0 {
                let n_ij_f = n_ij as f64;
                let log_term = (n * n_ij_f / (pi[i] * pj[j])).ln();
                mi += (n_ij_f / n) * log_term;
            }
        }
    }

    Ok(mi.max(0.0))
}

/// Compute Normalized Mutual Information between two clusterings
///
/// NMI = MI(U, V) / sqrt(H(U) * H(V))
///
/// # Arguments
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Cluster labels to evaluate
///
/// # Returns
/// * NMI score between 0.0 and 1.0
pub fn normalized_mutual_info_score(
    labels_true: &Array1<i32>,
    labels_pred: &Array1<i32>,
) -> MetricsResult<f64> {
    if labels_true.is_empty() {
        return Ok(1.0);
    }

    let mi = mutual_info_score(labels_true, labels_pred)?;
    let h_true = entropy(&labels_true.view());
    let h_pred = entropy(&labels_pred.view());

    if h_true == 0.0 || h_pred == 0.0 {
        return Ok(0.0);
    }

    Ok(mi / (h_true * h_pred).sqrt())
}

/// Compute Adjusted Mutual Information between two clusterings
///
/// AMI adjusts MI for chance. This is a simplified implementation.
///
/// # Arguments
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Cluster labels to evaluate
///
/// # Returns
/// * AMI score (typically between 0.0 and 1.0, can be negative)
pub fn adjusted_mutual_info_score(
    labels_true: &Array1<i32>,
    labels_pred: &Array1<i32>,
) -> MetricsResult<f64> {
    if labels_true.is_empty() {
        return Ok(1.0);
    }

    let mi = mutual_info_score(labels_true, labels_pred)?;
    let h_true = entropy(&labels_true.view());
    let h_pred = entropy(&labels_pred.view());

    // Simplified AMI: (MI - 0) / (avg(H_true, H_pred) - 0)
    // A full implementation would compute expected MI
    let avg_entropy = (h_true + h_pred) / 2.0;

    if avg_entropy == 0.0 {
        return Ok(1.0);
    }

    Ok(mi / avg_entropy)
}

/// Compute homogeneity metric of cluster labeling
///
/// A clustering satisfies homogeneity if all clusters contain only
/// members of a single class.
///
/// # Arguments
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Cluster labels to evaluate
///
/// # Returns
/// * Homogeneity score between 0.0 and 1.0 (1.0 = perfect)
pub fn homogeneity_score(
    labels_true: &Array1<i32>,
    labels_pred: &Array1<i32>,
) -> MetricsResult<f64> {
    if labels_true.is_empty() {
        return Ok(1.0);
    }

    let h_c = entropy(&labels_true.view());
    if h_c == 0.0 {
        return Ok(1.0);
    }

    let mi = mutual_info_score(labels_true, labels_pred)?;
    Ok(mi / h_c)
}

/// Compute completeness metric of cluster labeling
///
/// A clustering satisfies completeness if all members of a given class
/// are assigned to the same cluster.
///
/// # Arguments
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Cluster labels to evaluate
///
/// # Returns
/// * Completeness score between 0.0 and 1.0 (1.0 = perfect)
pub fn completeness_score(
    labels_true: &Array1<i32>,
    labels_pred: &Array1<i32>,
) -> MetricsResult<f64> {
    if labels_true.is_empty() {
        return Ok(1.0);
    }

    let h_k = entropy(&labels_pred.view());
    if h_k == 0.0 {
        return Ok(1.0);
    }

    let mi = mutual_info_score(labels_true, labels_pred)?;
    Ok(mi / h_k)
}

/// Compute V-measure (harmonic mean of homogeneity and completeness)
///
/// V-measure is the harmonic mean of homogeneity and completeness:
/// v = 2 * (homogeneity * completeness) / (homogeneity + completeness)
///
/// # Arguments
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Cluster labels to evaluate
///
/// # Returns
/// * V-measure score between 0.0 and 1.0 (1.0 = perfect)
pub fn v_measure_score(labels_true: &Array1<i32>, labels_pred: &Array1<i32>) -> MetricsResult<f64> {
    let h = homogeneity_score(labels_true, labels_pred)?;
    let c = completeness_score(labels_true, labels_pred)?;

    if h + c == 0.0 {
        return Ok(0.0);
    }

    Ok(2.0 * h * c / (h + c))
}

// ============================================================================
// Unsupervised Clustering Metrics (intrinsic quality)
// ============================================================================

/// Compute mean Silhouette Coefficient for all samples
///
/// Silhouette = (b - a) / max(a, b) where:
/// - a: mean intra-cluster distance
/// - b: mean nearest-cluster distance
///
/// # Arguments
/// * `x` - Feature matrix (n_samples, n_features)
/// * `labels` - Cluster labels
///
/// # Returns
/// * Mean silhouette score between -1.0 and 1.0 (1.0 = perfect)
pub fn silhouette_score(x: &Array2<f64>, labels: &Array1<i32>) -> MetricsResult<f64> {
    let n_samples = x.nrows();
    if n_samples != labels.len() {
        return Err(MetricsError::InvalidInput(
            "X and labels must have the same number of samples".to_string(),
        ));
    }

    // Check number of unique labels
    let mut unique_labels = std::collections::HashSet::new();
    for &label in labels.iter() {
        unique_labels.insert(label);
    }

    let n_labels = unique_labels.len();
    if n_labels < 2 || n_labels >= n_samples {
        return Err(MetricsError::InvalidInput(
            "Number of labels must be between 2 and n_samples - 1".to_string(),
        ));
    }

    // Compute pairwise distances
    let mut distances = Array2::<f64>::zeros((n_samples, n_samples));
    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let mut dist_sq = 0.0;
            for k in 0..x.ncols() {
                let diff = x[[i, k]] - x[[j, k]];
                dist_sq += diff * diff;
            }
            let dist = dist_sq.sqrt();
            distances[[i, j]] = dist;
            distances[[j, i]] = dist;
        }
    }

    let mut silhouettes = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let own_label = labels[i];

        // Compute a: mean intra-cluster distance
        let mut intra_dist_sum = 0.0;
        let mut intra_count = 0;
        for j in 0..n_samples {
            if i != j && labels[j] == own_label {
                intra_dist_sum += distances[[i, j]];
                intra_count += 1;
            }
        }
        let a = if intra_count > 0 {
            intra_dist_sum / intra_count as f64
        } else {
            0.0
        };

        // Compute b: mean nearest-cluster distance
        let mut cluster_dists: HashMap<i32, (f64, usize)> = HashMap::new();
        for j in 0..n_samples {
            if labels[j] != own_label {
                let entry = cluster_dists.entry(labels[j]).or_insert((0.0, 0));
                entry.0 += distances[[i, j]];
                entry.1 += 1;
            }
        }

        let b = cluster_dists
            .values()
            .map(|(sum, count)| sum / *count as f64)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        let s = if a.max(b) > 0.0 {
            (b - a) / a.max(b)
        } else {
            0.0
        };

        silhouettes.push(s);
    }

    Ok(silhouettes.iter().sum::<f64>() / n_samples as f64)
}

/// Compute Calinski-Harabasz score (Variance Ratio Criterion)
///
/// Score = (between-cluster dispersion / within-cluster dispersion) *
///         ((n_samples - n_clusters) / (n_clusters - 1))
///
/// Higher values indicate better-defined clusters.
///
/// # Arguments
/// * `x` - Feature matrix (n_samples, n_features)
/// * `labels` - Cluster labels
///
/// # Returns
/// * Calinski-Harabasz score (non-negative, higher is better)
pub fn calinski_harabasz_score(x: &Array2<f64>, labels: &Array1<i32>) -> MetricsResult<f64> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples != labels.len() {
        return Err(MetricsError::InvalidInput(
            "X and labels must have the same number of samples".to_string(),
        ));
    }

    // Get unique labels
    let mut unique_labels: Vec<i32> = labels.iter().copied().collect();
    unique_labels.sort_unstable();
    unique_labels.dedup();
    let n_labels = unique_labels.len();

    if n_labels < 2 || n_labels >= n_samples {
        return Err(MetricsError::InvalidInput(
            "Number of labels must be between 2 and n_samples - 1".to_string(),
        ));
    }

    // Compute global mean
    let mut mean = Array1::<f64>::zeros(n_features);
    for i in 0..n_samples {
        for j in 0..n_features {
            mean[j] += x[[i, j]];
        }
    }
    mean /= n_samples as f64;

    let mut extra_disp = 0.0;
    let mut intra_disp = 0.0;

    for &k in &unique_labels {
        // Find points in cluster k
        let cluster_indices: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|(_, &l)| l == k)
            .map(|(i, _)| i)
            .collect();

        let n_k = cluster_indices.len();

        // Compute cluster mean
        let mut mean_k = Array1::<f64>::zeros(n_features);
        for &i in &cluster_indices {
            for j in 0..n_features {
                mean_k[j] += x[[i, j]];
            }
        }
        mean_k /= n_k as f64;

        // Between-cluster dispersion
        for j in 0..n_features {
            let diff = mean_k[j] - mean[j];
            extra_disp += n_k as f64 * diff * diff;
        }

        // Within-cluster dispersion
        for &i in &cluster_indices {
            for j in 0..n_features {
                let diff = x[[i, j]] - mean_k[j];
                intra_disp += diff * diff;
            }
        }
    }

    if intra_disp == 0.0 {
        return Ok(1.0);
    }

    let score = extra_disp * (n_samples - n_labels) as f64 / (intra_disp * (n_labels - 1) as f64);
    Ok(score)
}

/// Compute Davies-Bouldin score
///
/// The score is the average similarity measure of each cluster with its
/// most similar cluster. Lower values indicate better clustering.
///
/// # Arguments
/// * `x` - Feature matrix (n_samples, n_features)
/// * `labels` - Cluster labels
///
/// # Returns
/// * Davies-Bouldin score (non-negative, lower is better, minimum is 0)
pub fn davies_bouldin_score(x: &Array2<f64>, labels: &Array1<i32>) -> MetricsResult<f64> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples != labels.len() {
        return Err(MetricsError::InvalidInput(
            "X and labels must have the same number of samples".to_string(),
        ));
    }

    // Get unique labels
    let mut unique_labels: Vec<i32> = labels.iter().copied().collect();
    unique_labels.sort_unstable();
    unique_labels.dedup();
    let n_labels = unique_labels.len();

    if n_labels < 2 || n_labels >= n_samples {
        return Err(MetricsError::InvalidInput(
            "Number of labels must be between 2 and n_samples - 1".to_string(),
        ));
    }

    // Compute centroids and intra-cluster distances
    let mut centroids = Array2::<f64>::zeros((n_labels, n_features));
    let mut intra_dists = Array1::<f64>::zeros(n_labels);

    for (k_idx, &k) in unique_labels.iter().enumerate() {
        let cluster_indices: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|(_, &l)| l == k)
            .map(|(i, _)| i)
            .collect();

        let n_k = cluster_indices.len();

        // Compute centroid
        for &i in &cluster_indices {
            for j in 0..n_features {
                centroids[[k_idx, j]] += x[[i, j]];
            }
        }
        for j in 0..n_features {
            centroids[[k_idx, j]] /= n_k as f64;
        }

        // Compute average distance to centroid
        let mut dist_sum = 0.0;
        for &i in &cluster_indices {
            let mut dist_sq = 0.0;
            for j in 0..n_features {
                let diff = x[[i, j]] - centroids[[k_idx, j]];
                dist_sq += diff * diff;
            }
            dist_sum += dist_sq.sqrt();
        }
        intra_dists[k_idx] = dist_sum / n_k as f64;
    }

    // Compute pairwise centroid distances
    let mut centroid_distances = Array2::<f64>::zeros((n_labels, n_labels));
    for i in 0..n_labels {
        for j in 0..n_labels {
            if i != j {
                let mut dist_sq = 0.0;
                for k in 0..n_features {
                    let diff = centroids[[i, k]] - centroids[[j, k]];
                    dist_sq += diff * diff;
                }
                centroid_distances[[i, j]] = dist_sq.sqrt();
            } else {
                centroid_distances[[i, j]] = f64::INFINITY;
            }
        }
    }

    // Check for degenerate cases
    if intra_dists.iter().all(|&x| x.abs() < 1e-10)
        || centroid_distances
            .iter()
            .all(|&x| x.abs() < 1e-10 || x.is_infinite())
    {
        return Ok(0.0);
    }

    // Compute Davies-Bouldin index
    let mut scores = Vec::with_capacity(n_labels);
    for i in 0..n_labels {
        let mut max_ratio = 0.0;
        for j in 0..n_labels {
            if i != j {
                let ratio = (intra_dists[i] + intra_dists[j]) / centroid_distances[[i, j]];
                if ratio > max_ratio {
                    max_ratio = ratio;
                }
            }
        }
        scores.push(max_ratio);
    }

    Ok(scores.iter().sum::<f64>() / n_labels as f64)
}

/// Compute Dunn Index
///
/// Dunn Index = min(inter-cluster distances) / max(intra-cluster distances)
/// Higher values indicate better clustering.
///
/// # Arguments
/// * `x` - Feature matrix (n_samples, n_features)
/// * `labels` - Cluster labels
///
/// # Returns
/// * Dunn index (non-negative, higher is better)
pub fn dunn_index(x: &Array2<f64>, labels: &Array1<i32>) -> MetricsResult<f64> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples != labels.len() {
        return Err(MetricsError::InvalidInput(
            "X and labels must have the same number of samples".to_string(),
        ));
    }

    // Get unique labels
    let mut unique_labels: Vec<i32> = labels.iter().copied().collect();
    unique_labels.sort_unstable();
    unique_labels.dedup();
    let n_labels = unique_labels.len();

    if n_labels < 2 {
        return Err(MetricsError::InvalidInput(
            "Need at least 2 clusters for Dunn index".to_string(),
        ));
    }

    // Group samples by cluster
    let mut clusters: HashMap<i32, Vec<usize>> = HashMap::new();
    for (i, &label) in labels.iter().enumerate() {
        clusters.entry(label).or_default().push(i);
    }

    // Compute max intra-cluster distance for each cluster
    let mut max_intra_dist = 0.0;
    for indices in clusters.values() {
        for (i_idx, &i) in indices.iter().enumerate() {
            for &j in indices.iter().skip(i_idx + 1) {
                let mut dist_sq = 0.0;
                for k in 0..n_features {
                    let diff = x[[i, k]] - x[[j, k]];
                    dist_sq += diff * diff;
                }
                let dist = dist_sq.sqrt();
                if dist > max_intra_dist {
                    max_intra_dist = dist;
                }
            }
        }
    }

    // Compute min inter-cluster distance
    let mut min_inter_dist = f64::INFINITY;
    for (&label_i, indices_i) in clusters.iter() {
        for (&label_j, indices_j) in clusters.iter() {
            if label_i < label_j {
                for &i in indices_i {
                    for &j in indices_j {
                        let mut dist_sq = 0.0;
                        for k in 0..n_features {
                            let diff = x[[i, k]] - x[[j, k]];
                            dist_sq += diff * diff;
                        }
                        let dist = dist_sq.sqrt();
                        if dist < min_inter_dist {
                            min_inter_dist = dist;
                        }
                    }
                }
            }
        }
    }

    if max_intra_dist == 0.0 {
        return Ok(f64::INFINITY);
    }

    Ok(min_inter_dist / max_intra_dist)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_rand_score_perfect_match() {
        let labels_true = Array1::from_vec(vec![0, 0, 1, 1]);
        let labels_pred = Array1::from_vec(vec![1, 1, 0, 0]);
        let score = rand_score(&labels_true, &labels_pred).unwrap();
        assert!(
            (score - 1.0).abs() < 1e-10,
            "Perfect match should give score of 1.0"
        );
    }

    #[test]
    fn test_rand_score_partial_match() {
        let labels_true = Array1::from_vec(vec![0, 0, 1, 2]);
        let labels_pred = Array1::from_vec(vec![0, 0, 1, 1]);
        let score = rand_score(&labels_true, &labels_pred).unwrap();
        assert!(
            score > 0.5 && score < 1.0,
            "Partial match should be between 0.5 and 1.0"
        );
    }

    #[test]
    fn test_adjusted_rand_score_perfect() {
        let labels_true = Array1::from_vec(vec![0, 0, 1, 1]);
        let labels_pred = Array1::from_vec(vec![0, 0, 1, 1]);
        let score = adjusted_rand_score(&labels_true, &labels_pred).unwrap();
        assert!(
            (score - 1.0).abs() < 1e-10,
            "Perfect match should give ARI of 1.0"
        );
    }

    #[test]
    fn test_adjusted_rand_score_permuted() {
        let labels_true = Array1::from_vec(vec![0, 0, 1, 1]);
        let labels_pred = Array1::from_vec(vec![1, 1, 0, 0]);
        let score = adjusted_rand_score(&labels_true, &labels_pred).unwrap();
        assert!(
            (score - 1.0).abs() < 1e-10,
            "Permuted labels should still give ARI of 1.0"
        );
    }

    #[test]
    fn test_adjusted_rand_score_negative() {
        let labels_true = Array1::from_vec(vec![0, 0, 1, 1]);
        let labels_pred = Array1::from_vec(vec![0, 1, 0, 1]);
        let score = adjusted_rand_score(&labels_true, &labels_pred).unwrap();
        assert!(score < 0.0, "Discordant labeling should give negative ARI");
    }

    #[test]
    fn test_fowlkes_mallows_score() {
        let labels_true = Array1::from_vec(vec![0, 0, 1, 1]);
        let labels_pred = Array1::from_vec(vec![0, 0, 1, 1]);
        let score = fowlkes_mallows_score(&labels_true, &labels_pred).unwrap();
        assert!(
            (score - 1.0).abs() < 1e-10,
            "Perfect match should give FMI of 1.0"
        );
    }

    #[test]
    fn test_mutual_info_score_perfect() {
        let labels_true = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let labels_pred = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let mi = mutual_info_score(&labels_true, &labels_pred).unwrap();
        assert!(mi > 0.0, "Perfect clustering should have positive MI");
    }

    #[test]
    fn test_mutual_info_score_single_cluster() {
        let labels_true = Array1::from_vec(vec![0, 0, 0, 0]);
        let labels_pred = Array1::from_vec(vec![1, 1, 1, 1]);
        let mi = mutual_info_score(&labels_true, &labels_pred).unwrap();
        assert!(
            (mi - 0.0).abs() < 1e-10,
            "Single cluster should have MI of 0"
        );
    }

    #[test]
    fn test_normalized_mutual_info_score() {
        let labels_true = Array1::from_vec(vec![0, 0, 1, 1]);
        let labels_pred = Array1::from_vec(vec![0, 0, 1, 1]);
        let nmi = normalized_mutual_info_score(&labels_true, &labels_pred).unwrap();
        assert!(
            (nmi - 1.0).abs() < 1e-10,
            "Perfect match should give NMI of 1.0"
        );
    }

    #[test]
    fn test_homogeneity_score_perfect() {
        let labels_true = Array1::from_vec(vec![0, 0, 1, 1]);
        let labels_pred = Array1::from_vec(vec![0, 0, 1, 1]);
        let h = homogeneity_score(&labels_true, &labels_pred).unwrap();
        assert!(
            (h - 1.0).abs() < 1e-10,
            "Perfect clustering is perfectly homogeneous"
        );
    }

    #[test]
    fn test_homogeneity_score_oversplit() {
        // Homogeneous but not complete: each cluster has only one class, but classes are split
        let labels_true = Array1::from_vec(vec![0, 0, 1, 1]);
        let labels_pred = Array1::from_vec(vec![0, 1, 2, 3]);
        let h = homogeneity_score(&labels_true, &labels_pred).unwrap();
        assert!(
            (h - 1.0).abs() < 1e-10,
            "Oversplit clusters are still homogeneous"
        );
    }

    #[test]
    fn test_completeness_score_perfect() {
        let labels_true = Array1::from_vec(vec![0, 0, 1, 1]);
        let labels_pred = Array1::from_vec(vec![0, 0, 1, 1]);
        let c = completeness_score(&labels_true, &labels_pred).unwrap();
        assert!(
            (c - 1.0).abs() < 1e-10,
            "Perfect clustering is perfectly complete"
        );
    }

    #[test]
    fn test_completeness_score_merged() {
        // Complete but not homogeneous: all members of each class in same cluster
        let labels_true = Array1::from_vec(vec![0, 0, 1, 1]);
        let labels_pred = Array1::from_vec(vec![0, 0, 0, 0]);
        let c = completeness_score(&labels_true, &labels_pred).unwrap();
        assert!((c - 1.0).abs() < 1e-10, "All in one cluster is complete");
    }

    #[test]
    fn test_v_measure_score() {
        let labels_true = Array1::from_vec(vec![0, 0, 1, 1]);
        let labels_pred = Array1::from_vec(vec![0, 0, 1, 1]);
        let v = v_measure_score(&labels_true, &labels_pred).unwrap();
        assert!(
            (v - 1.0).abs() < 1e-10,
            "Perfect clustering should give V-measure of 1.0"
        );
    }

    #[test]
    fn test_silhouette_score_well_separated() {
        // Two well-separated clusters
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.0, 0.1],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.0, 10.1],
        ];
        let labels = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);
        let score = silhouette_score(&x, &labels).unwrap();
        assert!(
            score > 0.5,
            "Well-separated clusters should have high silhouette score"
        );
    }

    #[test]
    fn test_silhouette_score_overlapping() {
        // Overlapping clusters (poor clustering)
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
        ];
        let labels = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);
        let score = silhouette_score(&x, &labels).unwrap();
        // Overlapping clusters should have lower silhouette score
        assert!(
            score < 0.9,
            "Overlapping clusters should have lower silhouette score"
        );
    }

    #[test]
    fn test_calinski_harabasz_score() {
        // Two well-separated clusters
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.0, 0.1],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.0, 10.1],
        ];
        let labels = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);
        let score = calinski_harabasz_score(&x, &labels).unwrap();
        assert!(
            score > 1.0,
            "Well-separated clusters should have high CH score"
        );
    }

    #[test]
    fn test_davies_bouldin_score_good_clustering() {
        // Two well-separated clusters
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.0, 0.1],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.0, 10.1],
        ];
        let labels = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);
        let score = davies_bouldin_score(&x, &labels).unwrap();
        assert!(
            score < 1.0,
            "Well-separated clusters should have low DB score"
        );
    }

    #[test]
    fn test_davies_bouldin_score_poor_clustering() {
        // Overlapping clusters
        let x = array![
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [1.5, 1.5],
            [2.0, 2.0],
            [2.5, 2.5],
        ];
        let labels = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);
        let score = davies_bouldin_score(&x, &labels).unwrap();
        // Poor clustering should have higher DB score
        assert!(
            score > 0.1,
            "Overlapping clusters should have higher DB score"
        );
    }

    #[test]
    fn test_dunn_index_well_separated() {
        // Two well-separated clusters
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.0, 0.1],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.0, 10.1],
        ];
        let labels = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);
        let score = dunn_index(&x, &labels).unwrap();
        assert!(
            score > 5.0,
            "Well-separated clusters should have high Dunn index"
        );
    }

    #[test]
    fn test_dunn_index_overlapping() {
        // Overlapping clusters
        let x = array![
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [1.5, 1.5],
            [2.0, 2.0],
            [2.5, 2.5],
        ];
        let labels = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);
        let score = dunn_index(&x, &labels).unwrap();
        assert!(
            score < 5.0,
            "Overlapping clusters should have lower Dunn index"
        );
    }

    #[test]
    fn test_contingency_matrix() {
        let labels_true = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let labels_pred = Array1::from_vec(vec![0, 0, 1, 2, 2, 2]);
        let c = contingency_matrix(&labels_true.view(), &labels_pred.view());

        // Check dimensions
        assert_eq!(c.nrows(), 3); // 3 unique true labels
        assert_eq!(c.ncols(), 3); // 3 unique pred labels

        // Check total count
        let total: i64 = c.iter().sum();
        assert_eq!(total, 6);
    }

    #[test]
    fn test_entropy_single_cluster() {
        let labels = Array1::from_vec(vec![0, 0, 0, 0]);
        let h = entropy(&labels.view());
        assert!((h - 0.0).abs() < 1e-10, "Single cluster has zero entropy");
    }

    #[test]
    fn test_entropy_balanced() {
        let labels = Array1::from_vec(vec![0, 0, 1, 1]);
        let h = entropy(&labels.view());
        let expected = -(0.5_f64 * 0.5_f64.ln() + 0.5_f64 * 0.5_f64.ln());
        assert!(
            (h - expected).abs() < 1e-10,
            "Balanced clusters should have ln(2) entropy"
        );
    }

    #[test]
    fn test_error_on_length_mismatch() {
        let labels_true = Array1::from_vec(vec![0, 0, 1, 1]);
        let labels_pred = Array1::from_vec(vec![0, 0, 1]);

        assert!(rand_score(&labels_true, &labels_pred).is_err());
        assert!(adjusted_rand_score(&labels_true, &labels_pred).is_err());
        assert!(fowlkes_mallows_score(&labels_true, &labels_pred).is_err());
    }

    #[test]
    fn test_error_on_invalid_n_labels() {
        let x = array![[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]];

        // Only 1 label (all same cluster)
        let labels = Array1::from_vec(vec![0, 0, 0]);
        assert!(silhouette_score(&x, &labels).is_err());
        assert!(calinski_harabasz_score(&x, &labels).is_err());

        // Too many labels (n_labels == n_samples)
        let labels = Array1::from_vec(vec![0, 1, 2]);
        assert!(silhouette_score(&x, &labels).is_err());
    }

    #[test]
    fn test_adjusted_mutual_info_score_simple() {
        let labels_true = Array1::from_vec(vec![0, 0, 1, 1]);
        let labels_pred = Array1::from_vec(vec![0, 0, 1, 1]);
        let ami = adjusted_mutual_info_score(&labels_true, &labels_pred).unwrap();
        assert!(
            (ami - 1.0).abs() < 1e-10,
            "Perfect match should give AMI close to 1.0"
        );
    }

    #[test]
    fn test_all_metrics_consistency() {
        // Test that all supervised metrics give consistent results for perfect clustering
        let labels_true = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let labels_pred = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);

        assert!((rand_score(&labels_true, &labels_pred).unwrap() - 1.0).abs() < 1e-10);
        assert!((adjusted_rand_score(&labels_true, &labels_pred).unwrap() - 1.0).abs() < 1e-10);
        assert!((fowlkes_mallows_score(&labels_true, &labels_pred).unwrap() - 1.0).abs() < 1e-10);
        assert!((homogeneity_score(&labels_true, &labels_pred).unwrap() - 1.0).abs() < 1e-10);
        assert!((completeness_score(&labels_true, &labels_pred).unwrap() - 1.0).abs() < 1e-10);
        assert!((v_measure_score(&labels_true, &labels_pred).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_unsupervised_metrics_three_clusters() {
        // Three well-separated clusters
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.0, 0.1], // Cluster 0
            [5.0, 5.0],
            [5.1, 5.1],
            [5.0, 5.1], // Cluster 1
            [10.0, 10.0],
            [10.1, 10.1],
            [10.0, 10.1], // Cluster 2
        ];
        let labels = Array1::from_vec(vec![0, 0, 0, 1, 1, 1, 2, 2, 2]);

        // All metrics should indicate good clustering
        let sil = silhouette_score(&x, &labels).unwrap();
        let ch = calinski_harabasz_score(&x, &labels).unwrap();
        let db = davies_bouldin_score(&x, &labels).unwrap();
        let dunn = dunn_index(&x, &labels).unwrap();

        assert!(
            sil > 0.5,
            "Silhouette should be > 0.5 for well-separated clusters"
        );
        assert!(
            ch > 10.0,
            "CH score should be high for well-separated clusters"
        );
        assert!(
            db < 1.0,
            "DB score should be low for well-separated clusters"
        );
        assert!(
            dunn > 5.0,
            "Dunn index should be high for well-separated clusters"
        );
    }
}
