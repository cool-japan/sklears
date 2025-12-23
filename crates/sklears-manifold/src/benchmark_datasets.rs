//! Standard benchmark datasets for manifold learning evaluation
//! This module provides implementations of widely-used benchmark datasets
//! in manifold learning research, enabling fair comparison with reference
//! implementations and standardized evaluation of algorithm performance.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::Rng;
use scirs2_core::random::SeedableRng;
use std::f64::consts::PI;
pub struct BenchmarkDatasets;

impl BenchmarkDatasets {
    /// Generate Swiss Roll dataset
    ///
    /// The Swiss Roll is a classic 2D manifold embedded in 3D space,
    /// commonly used to test manifold learning algorithms.
    ///
    /// # Parameters
    ///
    /// * `n_samples` - Number of data points to generate
    /// * `noise` - Standard deviation of Gaussian noise added to the data
    /// * `random_state` - Random seed for reproducibility
    ///
    /// # Returns
    ///
    /// Returns a tuple of (data, colors) where:
    /// * `data` is an Array2<f64> of shape (n_samples, 3)
    /// * `colors` is an Array1<f64> representing the intrinsic 1D coordinate
    ///
    /// # Examples
    ///
    /// ```
    /// use sklears_manifold::benchmark_datasets::BenchmarkDatasets;
    ///
    /// let (data, colors) = BenchmarkDatasets::swiss_roll(1000, 0.1, 42);
    /// assert_eq!(data.shape(), &[1000, 3]);
    /// assert_eq!(colors.len(), 1000);
    /// ```
    pub fn swiss_roll(
        n_samples: usize,
        noise: f64,
        random_state: u64,
    ) -> (Array2<f64>, Array1<f64>) {
        let mut rng = StdRng::seed_from_u64(random_state);
        let mut data = Array2::zeros((n_samples, 3));
        let mut colors = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let t: f64 = 1.5 * PI * (1.0 + 2.0 * rng.gen::<f64>());
            let height: f64 = 21.0 * rng.gen::<f64>();

            data[[i, 0]] = t * t.cos() + noise * rng.sample::<f64, _>(scirs2_core::StandardNormal);
            data[[i, 1]] = height + noise * rng.sample::<f64, _>(scirs2_core::StandardNormal);
            data[[i, 2]] = t * t.sin() + noise * rng.sample::<f64, _>(scirs2_core::StandardNormal);

            colors[i] = t; // Intrinsic coordinate
        }

        (data, colors)
    }

    /// Generate S-Curve dataset
    ///
    /// The S-Curve is another classic 2D manifold embedded in 3D space,
    /// featuring a more complex curvature than the Swiss Roll.
    ///
    /// # Parameters
    ///
    /// * `n_samples` - Number of data points to generate
    /// * `noise` - Standard deviation of Gaussian noise added to the data
    /// * `random_state` - Random seed for reproducibility
    ///
    /// # Returns
    ///
    /// Returns a tuple of (data, colors) where:
    /// * `data` is an Array2<f64> of shape (n_samples, 3)
    /// * `colors` is an Array1<f64> representing the intrinsic 1D coordinate
    pub fn s_curve(n_samples: usize, noise: f64, random_state: u64) -> (Array2<f64>, Array1<f64>) {
        let mut rng = StdRng::seed_from_u64(random_state);
        let mut data = Array2::zeros((n_samples, 3));
        let mut colors = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let t: f64 = 3.0 * PI * rng.gen::<f64>();
            let height: f64 = 2.0 * rng.gen::<f64>();

            data[[i, 0]] = t.sin() + noise * rng.sample::<f64, _>(scirs2_core::StandardNormal);
            data[[i, 1]] = height + noise * rng.sample::<f64, _>(scirs2_core::StandardNormal);
            data[[i, 2]] =
                (t / 2.0).sin() + noise * rng.sample::<f64, _>(scirs2_core::StandardNormal);

            colors[i] = t; // Intrinsic coordinate
        }

        (data, colors)
    }

    /// Generate Twin Peaks dataset
    ///
    /// A 2D manifold with two peaks, useful for testing algorithms on
    /// more complex topological structures.
    ///
    /// # Parameters
    ///
    /// * `n_samples` - Number of data points to generate
    /// * `noise` - Standard deviation of Gaussian noise added to the data
    /// * `random_state` - Random seed for reproducibility
    ///
    /// # Returns
    ///
    /// Returns a tuple of (data, colors) where:
    /// * `data` is an Array2<f64> of shape (n_samples, 3)
    /// * `colors` is an Array1<f64> for visualization
    pub fn twin_peaks(
        n_samples: usize,
        noise: f64,
        random_state: u64,
    ) -> (Array2<f64>, Array1<f64>) {
        let mut rng = StdRng::seed_from_u64(random_state);
        let mut data = Array2::zeros((n_samples, 3));
        let mut colors = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let x = rng.gen_range(-3.0..3.0);
            let y = rng.gen_range(-3.0..3.0);

            // Twin peaks function
            let z = 3.0_f64 * (1.0_f64 - x).powi(2) * (-x.powi(2) - (y + 1.0_f64).powi(2)).exp()
                - 10.0_f64 * (x / 5.0_f64 - x.powi(3) - y.powi(5)) * (-x.powi(2) - y.powi(2)).exp()
                - (1.0_f64 / 3.0_f64) * (-(x + 1.0_f64).powi(2) - y.powi(2)).exp();

            data[[i, 0]] = x + noise * rng.sample::<f64, _>(scirs2_core::StandardNormal);
            data[[i, 1]] = y + noise * rng.sample::<f64, _>(scirs2_core::StandardNormal);
            data[[i, 2]] = z + noise * rng.sample::<f64, _>(scirs2_core::StandardNormal);

            colors[i] = z; // Height as color
        }

        (data, colors)
    }

    /// Generate Severed Sphere dataset
    ///
    /// A sphere with a section removed, creating a manifold with boundary.
    /// Useful for testing how algorithms handle boundaries and discontinuities.
    ///
    /// # Parameters
    ///
    /// * `n_samples` - Number of data points to generate
    /// * `noise` - Standard deviation of Gaussian noise added to the data
    /// * `random_state` - Random seed for reproducibility
    ///
    /// # Returns
    ///
    /// Returns a tuple of (data, colors) where:
    /// * `data` is an Array2<f64> of shape (n_samples, 3)
    /// * `colors` is an Array1<f64> representing the azimuthal angle
    pub fn severed_sphere(
        n_samples: usize,
        noise: f64,
        random_state: u64,
    ) -> (Array2<f64>, Array1<f64>) {
        let mut rng = StdRng::seed_from_u64(random_state);
        let mut data = Array2::zeros((n_samples, 3));
        let mut colors = Array1::zeros(n_samples);

        for i in 0..n_samples {
            // Generate points on sphere, but exclude a section
            let mut phi = rng.gen_range(0.0..2.0 * PI);
            let mut theta = rng.gen_range(0.0..PI);

            // Remove a "slice" from the sphere
            while phi > PI / 4.0
                && phi < 3.0 * PI / 4.0
                && theta > PI / 3.0
                && theta < 2.0 * PI / 3.0
            {
                phi = rng.gen_range(0.0..2.0 * PI);
                theta = rng.gen_range(0.0..PI);
            }

            let radius = 1.0 + noise * 0.1 * rng.sample::<f64, _>(scirs2_core::StandardNormal);

            data[[i, 0]] = radius * theta.sin() * phi.cos()
                + noise * rng.sample::<f64, _>(scirs2_core::StandardNormal);
            data[[i, 1]] = radius * theta.sin() * phi.sin()
                + noise * rng.sample::<f64, _>(scirs2_core::StandardNormal);
            data[[i, 2]] =
                radius * theta.cos() + noise * rng.sample::<f64, _>(scirs2_core::StandardNormal);

            colors[i] = phi; // Azimuthal angle
        }

        (data, colors)
    }

    /// Generate Möbius Strip dataset
    ///
    /// A non-orientable surface embedded in 3D, providing a challenging
    /// test case for manifold learning algorithms.
    ///
    /// # Parameters
    ///
    /// * `n_samples` - Number of data points to generate
    /// * `noise` - Standard deviation of Gaussian noise added to the data
    /// * `random_state` - Random seed for reproducibility
    ///
    /// # Returns
    ///
    /// Returns a tuple of (data, colors) where:
    /// * `data` is an Array2<f64> of shape (n_samples, 3)
    /// * `colors` is an Array1<f64> representing the parameter along the strip
    pub fn mobius_strip(
        n_samples: usize,
        noise: f64,
        random_state: u64,
    ) -> (Array2<f64>, Array1<f64>) {
        let mut rng = StdRng::seed_from_u64(random_state);
        let mut data = Array2::zeros((n_samples, 3));
        let mut colors = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let u = rng.gen_range(0.0..2.0 * PI); // Parameter along the strip
            let v = rng.gen_range(-1.0..1.0); // Parameter across the strip

            let radius = 1.0 + v * (u / 2.0).cos();

            data[[i, 0]] =
                radius * u.cos() + noise * rng.sample::<f64, _>(scirs2_core::StandardNormal);
            data[[i, 1]] =
                radius * u.sin() + noise * rng.sample::<f64, _>(scirs2_core::StandardNormal);
            data[[i, 2]] =
                v * (u / 2.0).sin() + noise * rng.sample::<f64, _>(scirs2_core::StandardNormal);

            colors[i] = u; // Parameter along the strip
        }

        (data, colors)
    }

    /// Generate Torus dataset
    ///
    /// A torus (donut shape) embedded in 3D, representing a manifold
    /// with genus 1 (one hole).
    ///
    /// # Parameters
    ///
    /// * `n_samples` - Number of data points to generate
    /// * `major_radius` - Major radius of the torus
    /// * `minor_radius` - Minor radius of the torus
    /// * `noise` - Standard deviation of Gaussian noise added to the data
    /// * `random_state` - Random seed for reproducibility
    ///
    /// # Returns
    ///
    /// Returns a tuple of (data, colors) where:
    /// * `data` is an Array2<f64> of shape (n_samples, 3)
    /// * `colors` is an Array1<f64> representing the major angle
    pub fn torus(
        n_samples: usize,
        major_radius: f64,
        minor_radius: f64,
        noise: f64,
        random_state: u64,
    ) -> (Array2<f64>, Array1<f64>) {
        let mut rng = StdRng::seed_from_u64(random_state);
        let mut data = Array2::zeros((n_samples, 3));
        let mut colors = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let u = rng.gen_range(0.0..2.0 * PI); // Major angle
            let v = rng.gen_range(0.0..2.0 * PI); // Minor angle

            let x = (major_radius + minor_radius * v.cos()) * u.cos();
            let y = (major_radius + minor_radius * v.cos()) * u.sin();
            let z = minor_radius * v.sin();

            data[[i, 0]] = x + noise * rng.sample::<f64, _>(scirs2_core::StandardNormal);
            data[[i, 1]] = y + noise * rng.sample::<f64, _>(scirs2_core::StandardNormal);
            data[[i, 2]] = z + noise * rng.sample::<f64, _>(scirs2_core::StandardNormal);

            colors[i] = u; // Major angle
        }

        (data, colors)
    }

    /// Generate Helix dataset
    ///
    /// A helical curve in 3D space, representing a 1D manifold.
    ///
    /// # Parameters
    ///
    /// * `n_samples` - Number of data points to generate
    /// * `n_turns` - Number of complete turns of the helix
    /// * `noise` - Standard deviation of Gaussian noise added to the data
    /// * `random_state` - Random seed for reproducibility
    ///
    /// # Returns
    ///
    /// Returns a tuple of (data, colors) where:
    /// * `data` is an Array2<f64> of shape (n_samples, 3)
    /// * `colors` is an Array1<f64> representing position along the helix
    pub fn helix(
        n_samples: usize,
        n_turns: f64,
        noise: f64,
        random_state: u64,
    ) -> (Array2<f64>, Array1<f64>) {
        let mut rng = StdRng::seed_from_u64(random_state);
        let mut data = Array2::zeros((n_samples, 3));
        let mut colors = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let t = (i as f64 / (n_samples - 1) as f64) * n_turns * 2.0 * PI;

            data[[i, 0]] = t.cos() + noise * rng.sample::<f64, _>(scirs2_core::StandardNormal);
            data[[i, 1]] = t.sin() + noise * rng.sample::<f64, _>(scirs2_core::StandardNormal);
            data[[i, 2]] =
                t / (2.0 * PI) + noise * rng.sample::<f64, _>(scirs2_core::StandardNormal);

            colors[i] = t; // Position along helix
        }

        (data, colors)
    }

    /// Generate Hyperellipsoid dataset
    ///
    /// An ellipsoid in high-dimensional space, useful for testing
    /// dimensionality reduction on curved manifolds.
    ///
    /// # Parameters
    ///
    /// * `n_samples` - Number of data points to generate
    /// * `n_dimensions` - Dimensionality of the embedding space
    /// * `axes_lengths` - Length of each axis of the ellipsoid
    /// * `noise` - Standard deviation of Gaussian noise added to the data
    /// * `random_state` - Random seed for reproducibility
    ///
    /// # Returns
    ///
    /// Returns a tuple of (data, colors) where:
    /// * `data` is an Array2<f64> of shape (n_samples, n_dimensions)
    /// * `colors` is an Array1<f64> representing the first spherical coordinate
    pub fn hyperellipsoid(
        n_samples: usize,
        n_dimensions: usize,
        axes_lengths: &[f64],
        noise: f64,
        random_state: u64,
    ) -> (Array2<f64>, Array1<f64>) {
        assert_eq!(
            axes_lengths.len(),
            n_dimensions,
            "Number of axes lengths must match number of dimensions"
        );

        let mut rng = StdRng::seed_from_u64(random_state);
        let mut data = Array2::zeros((n_samples, n_dimensions));
        let mut colors = Array1::zeros(n_samples);

        for i in 0..n_samples {
            // Generate point on unit hypersphere
            let mut point = Array1::zeros(n_dimensions);
            for j in 0..n_dimensions {
                point[j] = rng.sample::<f64, _>(scirs2_core::StandardNormal);
            }

            // Normalize to unit sphere
            let norm = point.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                point /= norm;
            }

            // Scale by axes lengths to create ellipsoid
            for j in 0..n_dimensions {
                data[[i, j]] = point[j] * axes_lengths[j]
                    + noise * rng.sample::<f64, _>(scirs2_core::StandardNormal);
            }

            colors[i] = point[0].atan2(point[1]); // First spherical coordinate
        }

        (data, colors)
    }

    /// Generate Gaussian Mixture on Manifold
    ///
    /// Multiple Gaussian clusters positioned on a manifold, useful for
    /// testing clustering and manifold learning together.
    ///
    /// # Parameters
    ///
    /// * `n_samples` - Number of data points to generate
    /// * `n_clusters` - Number of Gaussian clusters
    /// * `manifold_type` - Type of underlying manifold ("swiss_roll", "s_curve", "sphere")
    /// * `cluster_std` - Standard deviation of each cluster
    /// * `noise` - Additional noise added to the manifold
    /// * `random_state` - Random seed for reproducibility
    ///
    /// # Returns
    ///
    /// Returns a tuple of (data, labels) where:
    /// * `data` is an Array2<f64> of shape (n_samples, 3)
    /// * `labels` is an Array1<usize> indicating cluster membership
    pub fn gaussian_mixture_on_manifold(
        n_samples: usize,
        n_clusters: usize,
        manifold_type: &str,
        cluster_std: f64,
        noise: f64,
        random_state: u64,
    ) -> (Array2<f64>, Array1<usize>) {
        let mut rng = StdRng::seed_from_u64(random_state);

        // First generate the base manifold
        let (mut base_data, _) = match manifold_type {
            "swiss_roll" => Self::swiss_roll(n_samples, noise, random_state),
            "s_curve" => Self::s_curve(n_samples, noise, random_state),
            "sphere" => {
                // Generate sphere
                let mut data = Array2::zeros((n_samples, 3));
                for i in 0..n_samples {
                    let phi = rng.gen_range(0.0..2.0 * PI);
                    let theta = rng.gen_range(0.0..PI);

                    data[[i, 0]] = theta.sin() * phi.cos();
                    data[[i, 1]] = theta.sin() * phi.sin();
                    data[[i, 2]] = theta.cos();
                }
                (data, Array1::zeros(n_samples))
            }
            _ => panic!("Unknown manifold type: {}", manifold_type),
        };

        // Assign cluster labels and add cluster-specific noise
        let mut labels = Array1::zeros(n_samples);
        let samples_per_cluster = n_samples / n_clusters;

        for cluster in 0..n_clusters {
            let start_idx = cluster * samples_per_cluster;
            let end_idx = if cluster == n_clusters - 1 {
                n_samples
            } else {
                (cluster + 1) * samples_per_cluster
            };

            // Add cluster-specific Gaussian noise
            for i in start_idx..end_idx {
                labels[i] = cluster;

                for j in 0..3 {
                    base_data[[i, j]] +=
                        cluster_std * rng.sample::<f64, _>(scirs2_core::StandardNormal);
                }
            }
        }

        (base_data, labels)
    }
}

/// Performance evaluation utilities for benchmark datasets
pub struct PerformanceEvaluator;

impl PerformanceEvaluator {
    /// Evaluate trustworthiness of an embedding
    ///
    /// Trustworthiness measures how well the local neighborhood
    /// structure is preserved in the embedding.
    ///
    /// # Parameters
    ///
    /// * `original_data` - Original high-dimensional data
    /// * `embedded_data` - Low-dimensional embedding
    /// * `k` - Neighborhood size for evaluation
    ///
    /// # Returns
    ///
    /// Trustworthiness score between 0 and 1 (higher is better)
    pub fn trustworthiness(
        original_data: &Array2<f64>,
        embedded_data: &Array2<f64>,
        k: usize,
    ) -> f64 {
        let n = original_data.nrows();
        assert_eq!(n, embedded_data.nrows());

        let mut trustworthiness = 0.0;

        for i in 0..n {
            // Find k-nearest neighbors in original space
            let mut orig_distances: Vec<(f64, usize)> = Vec::new();
            for j in 0..n {
                if i != j {
                    let diff = &original_data.row(i) - &original_data.row(j);
                    let dist = diff.iter().map(|x| x * x).sum::<f64>().sqrt();
                    orig_distances.push((dist, j));
                }
            }
            orig_distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let orig_neighbors: Vec<usize> =
                orig_distances.iter().take(k).map(|(_, idx)| *idx).collect();

            // Find k-nearest neighbors in embedded space
            let mut embed_distances: Vec<(f64, usize)> = Vec::new();
            for j in 0..n {
                if i != j {
                    let diff = &embedded_data.row(i) - &embedded_data.row(j);
                    let dist = diff.iter().map(|x| x * x).sum::<f64>().sqrt();
                    embed_distances.push((dist, j));
                }
            }
            embed_distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let embed_neighbors: Vec<usize> = embed_distances
                .iter()
                .take(k)
                .map(|(_, idx)| *idx)
                .collect();

            // Count preserved neighbors
            let preserved = orig_neighbors
                .iter()
                .filter(|&&x| embed_neighbors.contains(&x))
                .count();
            trustworthiness += preserved as f64 / k as f64;
        }

        trustworthiness / n as f64
    }

    /// Evaluate continuity of an embedding
    ///
    /// Continuity measures how well neighbors in the embedding
    /// correspond to neighbors in the original space.
    ///
    /// # Parameters
    ///
    /// * `original_data` - Original high-dimensional data
    /// * `embedded_data` - Low-dimensional embedding
    /// * `k` - Neighborhood size for evaluation
    ///
    /// # Returns
    ///
    /// Continuity score between 0 and 1 (higher is better)
    pub fn continuity(original_data: &Array2<f64>, embedded_data: &Array2<f64>, k: usize) -> f64 {
        // Continuity is the reverse of trustworthiness
        Self::trustworthiness(embedded_data, original_data, k)
    }

    /// Compute normalized stress
    ///
    /// Stress measures how well pairwise distances are preserved.
    ///
    /// # Parameters
    ///
    /// * `original_data` - Original high-dimensional data
    /// * `embedded_data` - Low-dimensional embedding
    ///
    /// # Returns
    ///
    /// Normalized stress (lower is better)
    pub fn normalized_stress(original_data: &Array2<f64>, embedded_data: &Array2<f64>) -> f64 {
        let n = original_data.nrows();
        assert_eq!(n, embedded_data.nrows());

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..n {
            for j in (i + 1)..n {
                let orig_diff = &original_data.row(i) - &original_data.row(j);
                let orig_dist = orig_diff.iter().map(|x| x * x).sum::<f64>().sqrt();

                let embed_diff = &embedded_data.row(i) - &embedded_data.row(j);
                let embed_dist = embed_diff.iter().map(|x| x * x).sum::<f64>().sqrt();

                let diff = orig_dist - embed_dist;
                numerator += diff * diff;
                denominator += orig_dist * orig_dist;
            }
        }

        if denominator > 0.0 {
            (numerator / denominator).sqrt()
        } else {
            0.0
        }
    }

    /// Compute neighborhood hit rate
    ///
    /// Measures what fraction of k-nearest neighbors are preserved.
    ///
    /// # Parameters
    ///
    /// * `original_data` - Original high-dimensional data
    /// * `embedded_data` - Low-dimensional embedding
    /// * `k` - Neighborhood size for evaluation
    ///
    /// # Returns
    ///
    /// Neighborhood hit rate between 0 and 1 (higher is better)
    pub fn neighborhood_hit_rate(
        original_data: &Array2<f64>,
        embedded_data: &Array2<f64>,
        k: usize,
    ) -> f64 {
        Self::trustworthiness(original_data, embedded_data, k)
    }

    /// Comprehensive evaluation report
    ///
    /// Computes multiple quality metrics and returns a structured report.
    ///
    /// # Parameters
    ///
    /// * `original_data` - Original high-dimensional data
    /// * `embedded_data` - Low-dimensional embedding
    /// * `k_values` - Different neighborhood sizes to evaluate
    ///
    /// # Returns
    ///
    /// A structured evaluation report
    pub fn comprehensive_evaluation(
        original_data: &Array2<f64>,
        embedded_data: &Array2<f64>,
        k_values: &[usize],
    ) -> EvaluationReport {
        let mut trustworthiness_scores = Vec::new();
        let mut continuity_scores = Vec::new();
        let mut neighborhood_hit_rates = Vec::new();

        for &k in k_values {
            let trust = Self::trustworthiness(original_data, embedded_data, k);
            let cont = Self::continuity(original_data, embedded_data, k);
            let nhr = Self::neighborhood_hit_rate(original_data, embedded_data, k);

            trustworthiness_scores.push((k, trust));
            continuity_scores.push((k, cont));
            neighborhood_hit_rates.push((k, nhr));
        }

        let stress = Self::normalized_stress(original_data, embedded_data);

        EvaluationReport {
            trustworthiness_scores,
            continuity_scores,
            neighborhood_hit_rates,
            normalized_stress: stress,
            n_samples: original_data.nrows(),
            original_dim: original_data.ncols(),
            embedded_dim: embedded_data.ncols(),
        }
    }
}

/// Structured evaluation report for embedding quality
#[derive(Debug, Clone)]
pub struct EvaluationReport {
    /// trustworthiness_scores
    pub trustworthiness_scores: Vec<(usize, f64)>,
    /// continuity_scores
    pub continuity_scores: Vec<(usize, f64)>,
    /// neighborhood_hit_rates
    pub neighborhood_hit_rates: Vec<(usize, f64)>,
    /// normalized_stress
    pub normalized_stress: f64,
    /// n_samples
    pub n_samples: usize,
    /// original_dim
    pub original_dim: usize,
    /// embedded_dim
    pub embedded_dim: usize,
}

impl EvaluationReport {
    /// Get average trustworthiness across all k values
    pub fn average_trustworthiness(&self) -> f64 {
        self.trustworthiness_scores
            .iter()
            .map(|(_, score)| score)
            .sum::<f64>()
            / self.trustworthiness_scores.len() as f64
    }

    /// Get average continuity across all k values
    pub fn average_continuity(&self) -> f64 {
        self.continuity_scores
            .iter()
            .map(|(_, score)| score)
            .sum::<f64>()
            / self.continuity_scores.len() as f64
    }

    /// Generate a summary string of the evaluation
    pub fn summary(&self) -> String {
        format!(
            "Embedding Evaluation Report\n\
             Samples: {}, Original Dim: {}, Embedded Dim: {}\n\
             Average Trustworthiness: {:.4}\n\
             Average Continuity: {:.4}\n\
             Normalized Stress: {:.4}",
            self.n_samples,
            self.original_dim,
            self.embedded_dim,
            self.average_trustworthiness(),
            self.average_continuity(),
            self.normalized_stress
        )
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_swiss_roll_generation() {
        let (data, colors) = BenchmarkDatasets::swiss_roll(100, 0.1, 42);
        assert_eq!(data.shape(), &[100, 3]);
        assert_eq!(colors.len(), 100);

        // Check that colors are reasonable (between ~4.71 and ~14.14 for swiss roll)
        assert!(colors.iter().all(|&x| x >= 4.0 && x <= 15.0));
    }

    #[test]
    fn test_s_curve_generation() {
        let (data, colors) = BenchmarkDatasets::s_curve(50, 0.05, 123);
        assert_eq!(data.shape(), &[50, 3]);
        assert_eq!(colors.len(), 50);
    }

    #[test]
    fn test_torus_generation() {
        let (data, colors) = BenchmarkDatasets::torus(200, 2.0, 0.5, 0.1, 456);
        assert_eq!(data.shape(), &[200, 3]);
        assert_eq!(colors.len(), 200);

        // Check that colors are angles (between 0 and 2π)
        assert!(colors.iter().all(|&x| x >= 0.0 && x <= 2.0 * PI));
    }

    #[test]
    fn test_hyperellipsoid_generation() {
        let axes = vec![1.0, 2.0, 0.5, 1.5];
        let (data, colors) = BenchmarkDatasets::hyperellipsoid(100, 4, &axes, 0.1, 789);
        assert_eq!(data.shape(), &[100, 4]);
        assert_eq!(colors.len(), 100);
    }

    #[test]
    fn test_trustworthiness_perfect_embedding() {
        // Create identical embeddings - should have perfect trustworthiness
        let original =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let embedded = original.clone();

        let trust = PerformanceEvaluator::trustworthiness(&original, &embedded, 2);
        assert_abs_diff_eq!(trust, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_comprehensive_evaluation() {
        let (original, _) = BenchmarkDatasets::swiss_roll(100, 0.1, 42);
        let embedded = original
            .slice(scirs2_core::ndarray::s![.., 0..2])
            .to_owned(); // Project to first 2 dimensions

        let k_values = vec![5, 10, 15];
        let report =
            PerformanceEvaluator::comprehensive_evaluation(&original, &embedded, &k_values);

        assert_eq!(report.trustworthiness_scores.len(), 3);
        assert_eq!(report.continuity_scores.len(), 3);
        assert_eq!(report.n_samples, 100);
        assert_eq!(report.original_dim, 3);
        assert_eq!(report.embedded_dim, 2);

        // All scores should be between 0 and 1
        assert!(report.average_trustworthiness() >= 0.0 && report.average_trustworthiness() <= 1.0);
        assert!(report.average_continuity() >= 0.0 && report.average_continuity() <= 1.0);
    }
}
