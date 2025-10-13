//! Spatial Utility Functions
//!
//! This module provides core spatial computation utilities for spatial mixture models.
//! These functions handle distance calculations, nearest neighbor finding, and other
//! fundamental spatial operations required by spatial mixture modeling algorithms.

use scirs2_core::ndarray::Array2;

/// Compute Euclidean distance between two points
///
/// # Arguments
/// * `p1` - First point coordinates
/// * `p2` - Second point coordinates
///
/// # Returns
/// Euclidean distance between the points
pub fn euclidean_distance(p1: &[f64], p2: &[f64]) -> f64 {
    p1.iter()
        .zip(p2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Compute pairwise distances between all points
///
/// # Arguments
/// * `coords` - Matrix of coordinates where each row is a point
///
/// # Returns
/// Symmetric distance matrix
pub fn pairwise_distances(coords: &Array2<f64>) -> Array2<f64> {
    let n_samples = coords.nrows();
    let mut distances = Array2::zeros((n_samples, n_samples));

    for i in 0..n_samples {
        for j in i + 1..n_samples {
            let dist = euclidean_distance(
                &coords.row(i).to_owned().into_raw_vec(),
                &coords.row(j).to_owned().into_raw_vec(),
            );
            distances[[i, j]] = dist;
            distances[[j, i]] = dist;
        }
    }

    distances
}

/// Find k nearest neighbors for each point
///
/// # Arguments
/// * `coords` - Matrix of coordinates where each row is a point
/// * `k` - Number of nearest neighbors to find
///
/// # Returns
/// Vector where each element contains the indices of k nearest neighbors for that point
pub fn k_nearest_neighbors(coords: &Array2<f64>, k: usize) -> Vec<Vec<usize>> {
    let n_samples = coords.nrows();
    let distances = pairwise_distances(coords);

    (0..n_samples)
        .map(|i| {
            let mut neighbors: Vec<(f64, usize)> = (0..n_samples)
                .filter(|&j| j != i)
                .map(|j| (distances[[i, j]], j))
                .collect();

            neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            neighbors.iter().take(k).map(|(_, idx)| *idx).collect()
        })
        .collect()
}

/// Compute spatial weights matrix based on distance threshold
///
/// # Arguments
/// * `coords` - Matrix of coordinates where each row is a point
/// * `threshold` - Distance threshold for connectivity
///
/// # Returns
/// Binary spatial weights matrix (1 if within threshold, 0 otherwise)
pub fn spatial_weights_matrix(coords: &Array2<f64>, threshold: f64) -> Array2<f64> {
    let n_samples = coords.nrows();
    let mut weights = Array2::zeros((n_samples, n_samples));
    let distances = pairwise_distances(coords);

    for i in 0..n_samples {
        for j in 0..n_samples {
            if i != j && distances[[i, j]] <= threshold {
                weights[[i, j]] = 1.0;
            }
        }
    }

    weights
}

/// Compute spatial lag for a given variable
///
/// # Arguments
/// * `values` - Variable values for each spatial unit
/// * `weights` - Spatial weights matrix
///
/// # Returns
/// Spatial lag vector
pub fn spatial_lag(values: &[f64], weights: &Array2<f64>) -> Vec<f64> {
    let n = values.len();
    let mut lag = vec![0.0; n];

    for i in 0..n {
        let mut sum_weights = 0.0;
        let mut weighted_sum = 0.0;

        for j in 0..n {
            if weights[[i, j]] > 0.0 {
                weighted_sum += weights[[i, j]] * values[j];
                sum_weights += weights[[i, j]];
            }
        }

        if sum_weights > 0.0 {
            lag[i] = weighted_sum / sum_weights;
        }
    }

    lag
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_euclidean_distance() {
        let p1 = vec![0.0, 0.0];
        let p2 = vec![3.0, 4.0];
        let dist = euclidean_distance(&p1, &p2);
        assert!((dist - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_pairwise_distances() {
        let coords = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let distances = pairwise_distances(&coords);

        assert!((distances[[0, 1]] - 1.0).abs() < 1e-10);
        assert!((distances[[0, 2]] - 1.0).abs() < 1e-10);
        assert!((distances[[1, 2]] - 2.0_f64.sqrt()).abs() < 1e-10);
        assert_eq!(distances[[0, 0]], 0.0);
    }

    #[test]
    fn test_k_nearest_neighbors() {
        let coords = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]];
        let neighbors = k_nearest_neighbors(&coords, 2);

        assert_eq!(neighbors[0].len(), 2);
        assert!(neighbors[0].contains(&1)); // Point 1 should be closest to point 0
        assert!(neighbors[0].contains(&3) || neighbors[0].contains(&2)); // Either point 2 or 3 as second closest
    }

    #[test]
    fn test_spatial_weights_matrix() {
        let coords = array![[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]];
        let weights = spatial_weights_matrix(&coords, 1.5);

        assert_eq!(weights[[0, 1]], 1.0); // Distance = 1.0, within threshold
        assert_eq!(weights[[1, 0]], 1.0); // Symmetric
        assert_eq!(weights[[0, 2]], 0.0); // Distance = 3.0, outside threshold
        assert_eq!(weights[[1, 2]], 0.0); // Distance = 2.0, outside threshold
    }

    #[test]
    fn test_spatial_lag() {
        let values = vec![1.0, 2.0, 3.0];
        let weights = array![[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]];
        let lag = spatial_lag(&values, &weights);

        assert_eq!(lag[0], 2.0); // Only neighbor is point 1 with value 2.0
        assert_eq!(lag[1], 2.0); // Average of points 0 and 2: (1.0 + 3.0) / 2 = 2.0
        assert_eq!(lag[2], 2.0); // Only neighbor is point 1 with value 2.0
    }
}
