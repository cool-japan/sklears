//! Quality metrics for evaluating manifold learning algorithms
//!
//! This module provides comprehensive metrics to evaluate the quality of manifold embeddings:
//! - **Trustworthiness**: How well k-nearest neighbors in original space are preserved in embedding
//! - **Continuity**: How well k-nearest neighbors in embedding were neighbors in original space
//! - **Neighborhood Hit Rate**: Percentage of original neighbors that remain neighbors in embedding
//! - **Local Continuity Meta-Criterion (LCMC)**: Combined trustworthiness and continuity metric
//! - **Normalized Stress**: Global distance preservation metric
//! - **Mean Relative Rank Error (MRRE)**: Rank-based preservation metric

use scirs2_core::ndarray::{Array2, ArrayView2};
use sklears_core::types::Float;
use std::collections::HashMap;

/// Compute pairwise Euclidean distances between all points
fn pairwise_distances(x: &ArrayView2<Float>) -> Array2<Float> {
    let n = x.nrows();
    let mut distances = Array2::zeros((n, n));

    for i in 0..n {
        for j in i + 1..n {
            let dist = x
                .row(i)
                .iter()
                .zip(x.row(j).iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<Float>()
                .sqrt();
            distances[(i, j)] = dist;
            distances[(j, i)] = dist;
        }
    }
    distances
}

/// Find k-nearest neighbors for each point (excluding self)
fn find_k_nearest_neighbors(distances: &Array2<Float>, k: usize) -> Vec<Vec<usize>> {
    let n = distances.nrows();
    let mut neighbors = Vec::with_capacity(n);

    for i in 0..n {
        let mut indexed_distances: Vec<(usize, Float)> = (0..n)
            .filter(|&j| i != j)
            .map(|j| (j, distances[(i, j)]))
            .collect();

        indexed_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let k_neighbors: Vec<usize> = indexed_distances
            .into_iter()
            .take(k.min(n - 1))
            .map(|(idx, _)| idx)
            .collect();

        neighbors.push(k_neighbors);
    }
    neighbors
}

/// Compute trustworthiness of an embedding
///
/// Trustworthiness measures how well the k-nearest neighbors in the original
/// high-dimensional space are preserved as neighbors in the embedding.
///
/// # Arguments
/// * `x_original` - Original high-dimensional data
/// * `x_embedded` - Low-dimensional embedding
/// * `k` - Number of nearest neighbors to consider
///
/// # Returns
/// Trustworthiness score between 0 and 1 (higher is better)
pub fn trustworthiness(
    x_original: &ArrayView2<Float>,
    x_embedded: &ArrayView2<Float>,
    k: usize,
) -> Float {
    if x_original.nrows() != x_embedded.nrows() {
        panic!("Original and embedded data must have same number of samples");
    }

    let n = x_original.nrows();
    if k >= n {
        return 1.0; // Perfect trustworthiness if k is too large
    }

    // Compute distances and find neighbors in both spaces
    let orig_distances = pairwise_distances(x_original);
    let emb_distances = pairwise_distances(x_embedded);

    let orig_neighbors = find_k_nearest_neighbors(&orig_distances, k);
    let emb_neighbors = find_k_nearest_neighbors(&emb_distances, k);

    // Create rank maps for efficient lookup
    let mut orig_ranks: Vec<HashMap<usize, usize>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut rank_map = HashMap::new();
        let mut indexed_distances: Vec<(usize, Float)> = (0..n)
            .filter(|&j| i != j)
            .map(|j| (j, orig_distances[(i, j)]))
            .collect();
        indexed_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for (rank, (idx, _)) in indexed_distances.iter().enumerate() {
            rank_map.insert(*idx, rank + 1);
        }
        orig_ranks.push(rank_map);
    }

    let mut trustworthiness_sum = 0.0;

    for i in 0..n {
        let orig_k_neighbors: std::collections::HashSet<usize> =
            orig_neighbors[i].iter().cloned().collect();

        for &j in &emb_neighbors[i] {
            if !orig_k_neighbors.contains(&j) {
                // j is a neighbor in embedding but not in original space
                let rank_j = orig_ranks[i].get(&j).unwrap_or(&n);
                trustworthiness_sum += (*rank_j as Float - k as Float).max(0.0);
            }
        }
    }

    let max_sum = (n as Float) * k as Float * (2.0 * n as Float - 3.0 * k as Float - 1.0) / 2.0;
    1.0 - (2.0 / max_sum) * trustworthiness_sum
}

/// Compute continuity of an embedding
///
/// Continuity measures how well the k-nearest neighbors in the embedding
/// were neighbors in the original high-dimensional space.
///
/// # Arguments
/// * `x_original` - Original high-dimensional data
/// * `x_embedded` - Low-dimensional embedding
/// * `k` - Number of nearest neighbors to consider
///
/// # Returns
/// Continuity score between 0 and 1 (higher is better)
pub fn continuity(
    x_original: &ArrayView2<Float>,
    x_embedded: &ArrayView2<Float>,
    k: usize,
) -> Float {
    if x_original.nrows() != x_embedded.nrows() {
        panic!("Original and embedded data must have same number of samples");
    }

    let n = x_original.nrows();
    if k >= n {
        return 1.0; // Perfect continuity if k is too large
    }

    // Compute distances and find neighbors in both spaces
    let orig_distances = pairwise_distances(x_original);
    let emb_distances = pairwise_distances(x_embedded);

    let orig_neighbors = find_k_nearest_neighbors(&orig_distances, k);
    let emb_neighbors = find_k_nearest_neighbors(&emb_distances, k);

    // Create rank maps for embedded space
    let mut emb_ranks: Vec<HashMap<usize, usize>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut rank_map = HashMap::new();
        let mut indexed_distances: Vec<(usize, Float)> = (0..n)
            .filter(|&j| i != j)
            .map(|j| (j, emb_distances[(i, j)]))
            .collect();
        indexed_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for (rank, (idx, _)) in indexed_distances.iter().enumerate() {
            rank_map.insert(*idx, rank + 1);
        }
        emb_ranks.push(rank_map);
    }

    let mut continuity_sum = 0.0;

    for i in 0..n {
        let emb_k_neighbors: std::collections::HashSet<usize> =
            emb_neighbors[i].iter().cloned().collect();

        for &j in &orig_neighbors[i] {
            if !emb_k_neighbors.contains(&j) {
                // j is a neighbor in original space but not in embedding
                let rank_j = emb_ranks[i].get(&j).unwrap_or(&n);
                continuity_sum += (*rank_j as Float - k as Float).max(0.0);
            }
        }
    }

    let max_sum = (n as Float) * k as Float * (2.0 * n as Float - 3.0 * k as Float - 1.0) / 2.0;
    1.0 - (2.0 / max_sum) * continuity_sum
}

/// Compute neighborhood hit rate
///
/// The neighborhood hit rate measures the percentage of k-nearest neighbors
/// in the original space that remain as k-nearest neighbors in the embedding.
///
/// # Arguments
/// * `x_original` - Original high-dimensional data
/// * `x_embedded` - Low-dimensional embedding
/// * `k` - Number of nearest neighbors to consider
///
/// # Returns
/// Hit rate between 0 and 1 (higher is better)
pub fn neighborhood_hit_rate(
    x_original: &ArrayView2<Float>,
    x_embedded: &ArrayView2<Float>,
    k: usize,
) -> Float {
    if x_original.nrows() != x_embedded.nrows() {
        panic!("Original and embedded data must have same number of samples");
    }

    let n = x_original.nrows();
    if k >= n {
        return 1.0; // Perfect hit rate if k is too large
    }

    // Compute distances and find neighbors in both spaces
    let orig_distances = pairwise_distances(x_original);
    let emb_distances = pairwise_distances(x_embedded);

    let orig_neighbors = find_k_nearest_neighbors(&orig_distances, k);
    let emb_neighbors = find_k_nearest_neighbors(&emb_distances, k);

    let mut total_hits = 0;
    let total_possible = n * k;

    for i in 0..n {
        let orig_set: std::collections::HashSet<usize> =
            orig_neighbors[i].iter().cloned().collect();
        let emb_set: std::collections::HashSet<usize> = emb_neighbors[i].iter().cloned().collect();

        let intersection_size = orig_set.intersection(&emb_set).count();
        total_hits += intersection_size;
    }

    total_hits as Float / total_possible as Float
}

/// Compute Local Continuity Meta-Criterion (LCMC)
///
/// LCMC combines trustworthiness and continuity into a single metric.
/// It's the harmonic mean of trustworthiness and continuity.
///
/// # Arguments
/// * `x_original` - Original high-dimensional data
/// * `x_embedded` - Low-dimensional embedding
/// * `k` - Number of nearest neighbors to consider
///
/// # Returns
/// LCMC score between 0 and 1 (higher is better)
pub fn local_continuity_meta_criterion(
    x_original: &ArrayView2<Float>,
    x_embedded: &ArrayView2<Float>,
    k: usize,
) -> Float {
    let trust = trustworthiness(x_original, x_embedded, k);
    let cont = continuity(x_original, x_embedded, k);

    if trust + cont == 0.0 {
        0.0
    } else {
        2.0 * trust * cont / (trust + cont)
    }
}

/// Compute normalized stress
///
/// Normalized stress measures how well the pairwise distances are preserved
/// in the embedding relative to the original space.
///
/// # Arguments
/// * `x_original` - Original high-dimensional data
/// * `x_embedded` - Low-dimensional embedding
///
/// # Returns
/// Normalized stress (lower is better, 0 is perfect)
pub fn normalized_stress(x_original: &ArrayView2<Float>, x_embedded: &ArrayView2<Float>) -> Float {
    if x_original.nrows() != x_embedded.nrows() {
        panic!("Original and embedded data must have same number of samples");
    }

    let orig_distances = pairwise_distances(x_original);
    let emb_distances = pairwise_distances(x_embedded);

    let n = x_original.nrows();
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..n {
        for j in i + 1..n {
            let d_orig = orig_distances[(i, j)];
            let d_emb = emb_distances[(i, j)];

            numerator += (d_orig - d_emb).powi(2);
            denominator += d_orig.powi(2);
        }
    }

    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Compute Mean Relative Rank Error (MRRE)
///
/// MRRE measures how much the relative ranking of distances is preserved.
/// Lower values indicate better preservation of the distance ranking.
///
/// # Arguments
/// * `x_original` - Original high-dimensional data
/// * `x_embedded` - Low-dimensional embedding
///
/// # Returns
/// MRRE score (lower is better, 0 is perfect)
pub fn mean_relative_rank_error(
    x_original: &ArrayView2<Float>,
    x_embedded: &ArrayView2<Float>,
) -> Float {
    if x_original.nrows() != x_embedded.nrows() {
        panic!("Original and embedded data must have same number of samples");
    }

    let orig_distances = pairwise_distances(x_original);
    let emb_distances = pairwise_distances(x_embedded);
    let n = x_original.nrows();

    let mut total_error = 0.0;
    let mut total_pairs = 0;

    for i in 0..n {
        // Get rankings for original space
        let mut orig_pairs: Vec<(usize, Float)> = (0..n)
            .filter(|&j| i != j)
            .map(|j| (j, orig_distances[(i, j)]))
            .collect();
        orig_pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut orig_rank_map = HashMap::new();
        for (rank, (idx, _)) in orig_pairs.iter().enumerate() {
            orig_rank_map.insert(*idx, rank);
        }

        // Get rankings for embedded space
        let mut emb_pairs: Vec<(usize, Float)> = (0..n)
            .filter(|&j| i != j)
            .map(|j| (j, emb_distances[(i, j)]))
            .collect();
        emb_pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut emb_rank_map = HashMap::new();
        for (rank, (idx, _)) in emb_pairs.iter().enumerate() {
            emb_rank_map.insert(*idx, rank);
        }

        // Compute rank differences
        for j in 0..n {
            if i != j {
                let orig_rank = *orig_rank_map.get(&j).unwrap() as Float;
                let emb_rank = *emb_rank_map.get(&j).unwrap() as Float;
                let n_minus_1 = (n - 1) as Float;

                total_error += (orig_rank - emb_rank).abs() / n_minus_1;
                total_pairs += 1;
            }
        }
    }

    total_error / total_pairs as Float
}

/// Comprehensive quality report for a manifold embedding
///
/// This function computes all quality metrics and returns them in a structured format.
///
/// # Arguments
/// * `x_original` - Original high-dimensional data
/// * `x_embedded` - Low-dimensional embedding
/// * `k` - Number of nearest neighbors for local metrics (default: min(10, n-1))
///
/// # Returns
/// A structure containing all quality metrics
#[derive(Debug, Clone)]
pub struct QualityReport {
    /// trustworthiness
    pub trustworthiness: Float,
    /// continuity
    pub continuity: Float,
    /// neighborhood_hit_rate
    pub neighborhood_hit_rate: Float,
    /// local_continuity_meta_criterion
    pub local_continuity_meta_criterion: Float,
    /// normalized_stress
    pub normalized_stress: Float,
    /// mean_relative_rank_error
    pub mean_relative_rank_error: Float,
    /// k_neighbors
    pub k_neighbors: usize,
}

impl std::fmt::Display for QualityReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Manifold Embedding Quality Report")?;
        writeln!(f, "=================================")?;
        writeln!(f, "Local Metrics (k={})", self.k_neighbors)?;
        writeln!(
            f,
            "  Trustworthiness:              {:.4}",
            self.trustworthiness
        )?;
        writeln!(f, "  Continuity:                   {:.4}", self.continuity)?;
        writeln!(
            f,
            "  Neighborhood Hit Rate:        {:.4}",
            self.neighborhood_hit_rate
        )?;
        writeln!(
            f,
            "  LCMC (Harmonic Mean):         {:.4}",
            self.local_continuity_meta_criterion
        )?;
        writeln!(f, "Global Metrics")?;
        writeln!(
            f,
            "  Normalized Stress:            {:.4}",
            self.normalized_stress
        )?;
        writeln!(
            f,
            "  Mean Relative Rank Error:     {:.4}",
            self.mean_relative_rank_error
        )?;
        Ok(())
    }
}

/// Generate a comprehensive quality report
pub fn quality_report(
    x_original: &ArrayView2<Float>,
    x_embedded: &ArrayView2<Float>,
    k: Option<usize>,
) -> QualityReport {
    let n = x_original.nrows();
    let k_neighbors = k.unwrap_or_else(|| std::cmp::min(10, n - 1).max(1));

    QualityReport {
        trustworthiness: trustworthiness(x_original, x_embedded, k_neighbors),
        continuity: continuity(x_original, x_embedded, k_neighbors),
        neighborhood_hit_rate: neighborhood_hit_rate(x_original, x_embedded, k_neighbors),
        local_continuity_meta_criterion: local_continuity_meta_criterion(
            x_original,
            x_embedded,
            k_neighbors,
        ),
        normalized_stress: normalized_stress(x_original, x_embedded),
        mean_relative_rank_error: mean_relative_rank_error(x_original, x_embedded),
        k_neighbors,
    }
}
