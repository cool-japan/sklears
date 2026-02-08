//! Evolutionary and bio-inspired clustering algorithms.
//!
//! This module provides clustering algorithms inspired by biological and evolutionary processes,
//! including particle swarm optimization, genetic algorithms, ant colony optimization,
//! artificial bee colony, and differential evolution.

use scirs2_core::ndarray::{s, Array, Array1, Array2, Axis};
use scirs2_core::random::{Random, Rng};
use sklears_core::error::{Result, SklearsError};
use sklears_core::traits::{Estimator, Fit, Predict, Untrained};
use std::f64;

/// Particle Swarm Optimization (PSO) clustering algorithm.
///
/// PSO clustering is a bio-inspired metaheuristic algorithm that simulates the social
/// behavior of bird flocking or fish schooling to find optimal cluster centers.
/// Each particle represents a potential solution (set of cluster centers) and moves
/// through the solution space guided by its personal best and global best positions.
///
/// # Mathematical Foundation
///
/// The algorithm maintains a swarm of particles, where each particle i has:
/// - Position: x_i representing cluster centers
/// - Velocity: v_i controlling movement direction and magnitude
/// - Personal best: p_i (best position found by this particle)
/// - Global best: g (best position found by entire swarm)
///
/// Update equations:
/// ```text
/// v_i(t+1) = w * v_i(t) + c1 * r1 * (p_i - x_i(t)) + c2 * r2 * (g - x_i(t))
/// x_i(t+1) = x_i(t) + v_i(t+1)
/// ```
///
/// Where:
/// - w: inertia weight (exploration vs exploitation balance)
/// - c1, c2: acceleration coefficients (cognitive and social parameters)
/// - r1, r2: random numbers in \[0,1\]
///
/// # Example
///
/// ```rust
/// use sklears_clustering::evolutionary::PSOClustering;
/// use sklears_core::prelude::*;
///
/// let data = Array2::from_shape_vec(
///     (4, 2),
///     vec![1.0, 2.0, 1.1, 2.1, 5.0, 6.0, 5.1, 6.1],
/// )
/// .unwrap();
///
/// let pso = PSOClustering::builder()
///     .n_clusters(2)
///     .n_particles(30)
///     .max_iterations(100)
///     .build();
///
/// let fitted = pso.fit(&data, &()).unwrap();
/// let labels = fitted.predict(&data).unwrap();
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PSOClustering {
    /// Number of clusters to form
    pub n_clusters: usize,
    /// Number of particles in the swarm
    pub n_particles: usize,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Inertia weight for velocity update
    pub inertia_weight: f64,
    /// Cognitive acceleration coefficient
    pub cognitive_coeff: f64,
    /// Social acceleration coefficient
    pub social_coeff: f64,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for PSOClustering {
    fn default() -> Self {
        Self {
            n_clusters: 2,
            n_particles: 30,
            max_iterations: 100,
            inertia_weight: 0.9,
            cognitive_coeff: 2.0,
            social_coeff: 2.0,
            tolerance: 1e-4,
            random_seed: None,
        }
    }
}

impl PSOClustering {
    /// Creates a new PSO clustering instance with default parameters.
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            ..Default::default()
        }
    }

    /// Creates a builder for PSO clustering configuration.
    pub fn builder() -> PSOClusteringBuilder {
        PSOClusteringBuilder::new()
    }

    /// Computes fitness (within-cluster sum of squares) for given cluster centers.
    fn compute_fitness(&self, centers: &Array2<f64>, data: &Array2<f64>) -> f64 {
        let n_samples = data.nrows();
        let mut total_wcss = 0.0;

        for i in 0..n_samples {
            let sample = data.row(i);
            let mut min_distance = f64::INFINITY;

            // Find closest cluster center
            for j in 0..self.n_clusters {
                let center = centers.row(j);
                let distance: f64 = sample
                    .iter()
                    .zip(center.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();

                if distance < min_distance {
                    min_distance = distance;
                }
            }

            total_wcss += min_distance;
        }

        total_wcss
    }

    /// Assigns data points to closest cluster centers.
    fn assign_clusters(&self, centers: &Array2<f64>, data: &Array2<f64>) -> Array1<usize> {
        let n_samples = data.nrows();
        let mut labels = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let sample = data.row(i);
            let mut min_distance = f64::INFINITY;
            let mut best_cluster = 0;

            for j in 0..self.n_clusters {
                let center = centers.row(j);
                let distance: f64 = sample
                    .iter()
                    .zip(center.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();

                if distance < min_distance {
                    min_distance = distance;
                    best_cluster = j;
                }
            }

            labels[i] = best_cluster;
        }

        labels
    }
}

/// Builder for PSO clustering configuration.
#[derive(Debug, Clone)]
pub struct PSOClusteringBuilder {
    config: PSOClustering,
}

impl PSOClusteringBuilder {
    /// Creates a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: PSOClustering::default(),
        }
    }

    /// Sets the number of clusters.
    pub fn n_clusters(mut self, n_clusters: usize) -> Self {
        self.config.n_clusters = n_clusters;
        self
    }

    /// Sets the number of particles in the swarm.
    pub fn n_particles(mut self, n_particles: usize) -> Self {
        self.config.n_particles = n_particles;
        self
    }

    /// Sets the maximum number of iterations.
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.config.max_iterations = max_iterations;
        self
    }

    /// Sets the inertia weight for velocity updates.
    pub fn inertia_weight(mut self, weight: f64) -> Self {
        self.config.inertia_weight = weight;
        self
    }

    /// Sets the cognitive acceleration coefficient.
    pub fn cognitive_coeff(mut self, coeff: f64) -> Self {
        self.config.cognitive_coeff = coeff;
        self
    }

    /// Sets the social acceleration coefficient.
    pub fn social_coeff(mut self, coeff: f64) -> Self {
        self.config.social_coeff = coeff;
        self
    }

    /// Sets the convergence tolerance.
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.config.tolerance = tolerance;
        self
    }

    /// Sets the random seed for reproducibility.
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.config.random_seed = Some(seed);
        self
    }

    /// Builds the PSO clustering configuration.
    pub fn build(self) -> PSOClustering {
        self.config
    }
}

impl Default for PSOClusteringBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted PSO clustering model containing the optimized cluster centers.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PSOClusteringFitted {
    /// Final optimized cluster centers
    pub cluster_centers: Array2<f64>,
    /// Final fitness value (WCSS)
    pub fitness: f64,
    /// Number of iterations performed
    pub n_iterations: usize,
    /// Convergence status
    pub converged: bool,
    /// Original algorithm configuration
    pub config: PSOClustering,
}

impl PSOClusteringFitted {
    /// Returns the cluster centers.
    pub fn cluster_centers(&self) -> &Array2<f64> {
        &self.cluster_centers
    }

    /// Returns the final fitness value.
    pub fn fitness(&self) -> f64 {
        self.fitness
    }

    /// Returns whether the algorithm converged.
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// Returns the number of iterations performed.
    pub fn n_iterations(&self) -> usize {
        self.n_iterations
    }
}

impl Estimator for PSOClustering {
    type Config = PSOClustering;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        self
    }
}

impl PSOClustering {
    pub fn fit(&self, X: &Array2<f64>) -> Result<PSOClusteringFitted> {
        let (n_samples, n_features) = X.dim();

        if self.n_clusters > n_samples {
            return Err(SklearsError::InvalidInput(
                "Number of clusters cannot exceed number of samples".to_string(),
            ));
        }

        // Initialize random number generator
        let mut rng = match self.random_seed {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42), // Use a default seed for consistency
        };

        // Find data bounds for initialization
        let min_vals = X.fold_axis(Axis(0), f64::INFINITY, |acc, &x| acc.min(x));
        let max_vals = X.fold_axis(Axis(0), f64::NEG_INFINITY, |acc, &x| acc.max(x));

        // Initialize particles: positions (cluster centers) and velocities
        let mut positions: Array<f64, _> =
            Array::zeros((self.n_particles, self.n_clusters, n_features));
        let mut velocities: Array<f64, _> =
            Array::zeros((self.n_particles, self.n_clusters, n_features));
        let mut personal_best_positions: Array<f64, _> =
            Array::zeros((self.n_particles, self.n_clusters, n_features));
        let mut personal_best_fitness = vec![f64::INFINITY; self.n_particles];
        let mut global_best_position: Array<f64, _> = Array::zeros((self.n_clusters, n_features));
        let mut global_best_fitness = f64::INFINITY;

        // Initialize particle positions randomly within data bounds
        for p in 0..self.n_particles {
            for c in 0..self.n_clusters {
                for f in 0..n_features {
                    let range = max_vals[f] - min_vals[f];
                    positions[[p, c, f]] = min_vals[f] + rng.gen::<f64>() * range;
                    velocities[[p, c, f]] = (rng.gen::<f64>() - 0.5) * range * 0.1;
                }
            }

            // Evaluate initial fitness
            let particle_centers = positions.slice(s![p, .., ..]).to_owned();
            let fitness = self.compute_fitness(&particle_centers, X);

            if fitness < personal_best_fitness[p] {
                personal_best_fitness[p] = fitness;
                personal_best_positions
                    .slice_mut(s![p, .., ..])
                    .assign(&particle_centers);

                if fitness < global_best_fitness {
                    global_best_fitness = fitness;
                    global_best_position.assign(&particle_centers);
                }
            }
        }

        let mut converged = false;
        let mut iteration = 0;
        let mut prev_global_best = global_best_fitness;

        // PSO main loop
        while iteration < self.max_iterations && !converged {
            for p in 0..self.n_particles {
                // Update velocity and position for each particle
                for c in 0..self.n_clusters {
                    for f in 0..n_features {
                        let r1 = rng.gen::<f64>();
                        let r2 = rng.gen::<f64>();

                        let cognitive_component = self.cognitive_coeff
                            * r1
                            * (personal_best_positions[[p, c, f]] - positions[[p, c, f]]);
                        let social_component = self.social_coeff
                            * r2
                            * (global_best_position[[c, f]] - positions[[p, c, f]]);

                        // Update velocity
                        velocities[[p, c, f]] = self.inertia_weight * velocities[[p, c, f]]
                            + cognitive_component
                            + social_component;

                        // Update position
                        positions[[p, c, f]] += velocities[[p, c, f]];

                        // Boundary constraint - keep within data bounds
                        positions[[p, c, f]] =
                            positions[[p, c, f]].max(min_vals[f]).min(max_vals[f]);
                    }
                }

                // Evaluate fitness of updated position
                let particle_centers = positions.slice(s![p, .., ..]).to_owned();
                let fitness = self.compute_fitness(&particle_centers, X);

                // Update personal best
                if fitness < personal_best_fitness[p] {
                    personal_best_fitness[p] = fitness;
                    personal_best_positions
                        .slice_mut(s![p, .., ..])
                        .assign(&particle_centers);

                    // Update global best
                    if fitness < global_best_fitness {
                        global_best_fitness = fitness;
                        global_best_position.assign(&particle_centers);
                    }
                }
            }

            // Check convergence
            if (prev_global_best - global_best_fitness).abs() < self.tolerance {
                converged = true;
            }

            prev_global_best = global_best_fitness;
            iteration += 1;
        }

        Ok(PSOClusteringFitted {
            cluster_centers: global_best_position,
            fitness: global_best_fitness,
            n_iterations: iteration,
            converged,
            config: self.clone(),
        })
    }
}

impl Fit<Array2<f64>, (), Untrained> for PSOClustering {
    type Fitted = PSOClusteringFitted;

    fn fit(self, X: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        PSOClustering::fit(&self, X)
    }
}

impl Predict<Array2<f64>, Array1<usize>> for PSOClusteringFitted {
    fn predict(&self, X: &Array2<f64>) -> Result<Array1<usize>> {
        Ok(self.config.assign_clusters(&self.cluster_centers, X))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn create_simple_dataset() -> Array2<f64> {
        array![
            [1.0, 1.0],
            [1.1, 1.1],
            [1.2, 1.0],
            [8.0, 8.0],
            [8.1, 8.1],
            [8.2, 8.0]
        ]
    }

    #[test]
    fn test_pso_clustering_basic() {
        let data = create_simple_dataset();

        let pso = PSOClustering::builder()
            .n_clusters(2)
            .n_particles(20)
            .max_iterations(50)
            .random_seed(42)
            .build();

        let fitted = pso.fit(&data, &()).unwrap();
        let labels = fitted.predict(&data).unwrap();

        assert_eq!(fitted.cluster_centers.nrows(), 2);
        assert_eq!(fitted.cluster_centers.ncols(), 2);
        assert_eq!(labels.len(), 6);

        // Check that labels are binary (0 or 1)
        for &label in labels.iter() {
            assert!(label < 2);
        }
    }

    #[test]
    fn test_pso_clustering_convergence() {
        let data = create_simple_dataset();

        let pso = PSOClustering::builder()
            .n_clusters(2)
            .n_particles(30)
            .max_iterations(100)
            .tolerance(1e-6)
            .random_seed(42)
            .build();

        let fitted = pso.fit(&data, &()).unwrap();

        assert!(fitted.fitness() < 10.0); // Should find good solution
        assert!(fitted.n_iterations() <= 100);
    }

    #[test]
    fn test_pso_builder_pattern() {
        let pso = PSOClustering::builder()
            .n_clusters(3)
            .n_particles(40)
            .max_iterations(80)
            .inertia_weight(0.8)
            .cognitive_coeff(1.5)
            .social_coeff(2.5)
            .tolerance(1e-5)
            .random_seed(123)
            .build();

        assert_eq!(pso.n_clusters, 3);
        assert_eq!(pso.n_particles, 40);
        assert_eq!(pso.max_iterations, 80);
        assert_eq!(pso.inertia_weight, 0.8);
        assert_eq!(pso.cognitive_coeff, 1.5);
        assert_eq!(pso.social_coeff, 2.5);
        assert_eq!(pso.tolerance, 1e-5);
        assert_eq!(pso.random_seed, Some(123));
    }

    #[test]
    fn test_invalid_cluster_count() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];

        let pso = PSOClustering::new(5); // More clusters than samples
        assert!(pso.fit(&data, &()).is_err());
    }
}
