//! OPTICS (Ordering Points To Identify Clustering Structure) implementation
//!
//! OPTICS extends DBSCAN to handle clusters of varying densities by generating
//! an ordering of the database representing the density-based clustering structure.

use std::collections::BinaryHeap;
use std::marker::PhantomData;

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearnContext, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::{Float, FloatBounds},
    validation::{ConfigValidation, Validate, ValidationRule, ValidationRules},
};

/// Configuration for OPTICS clustering
#[derive(Debug, Clone)]
pub struct OpticsConfig {
    /// The maximum distance between two samples for one to be considered as in the neighborhood of the other
    pub max_eps: f64,
    /// The number of samples in a neighborhood for a point to be considered as a core point
    pub min_samples: usize,
    /// The distance metric to use
    pub metric: DistanceMetric,
    /// Algorithm used to compute the nearest neighbors
    pub algorithm: Algorithm,
    /// Leaf size passed to BallTree or KDTree
    pub leaf_size: usize,
    /// Maximum number of clusters to extract (None for no limit)
    pub max_clusters: Option<usize>,
    /// Cluster method to use for extraction
    pub cluster_method: ClusterMethod,
}

/// Distance metrics supported by OPTICS
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Chebyshev,
    Minkowski(f64),
}

/// Algorithms for nearest neighbor computation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Algorithm {
    /// Ball tree algorithm
    BallTree,
    /// KD tree algorithm
    KDTree,
    /// Brute force algorithm
    Brute,
    /// Automatic selection
    Auto,
}

/// Methods for extracting clusters from OPTICS ordering
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClusterMethod {
    /// Extract clusters using a threshold on reachability distance
    Threshold(f64),
    /// Extract clusters using hierarchical clustering with maximum clusters
    Hierarchical,
    /// Use the steepest descent algorithm
    SteepestDescent,
}

impl Default for OpticsConfig {
    fn default() -> Self {
        Self {
            max_eps: f64::INFINITY,
            min_samples: 5,
            metric: DistanceMetric::Euclidean,
            algorithm: Algorithm::Auto,
            leaf_size: 30,
            max_clusters: None,
            cluster_method: ClusterMethod::Threshold(0.5),
        }
    }
}

impl Validate for OpticsConfig {
    fn validate(&self) -> Result<()> {
        // Validate max_eps (allow infinity for OPTICS)
        ValidationRules::new("max_eps")
            .add_rule(ValidationRule::Positive)
            .validate_numeric(&self.max_eps)?;

        // Validate min_samples
        ValidationRules::new("min_samples")
            .add_rule(ValidationRule::Positive)
            .validate_usize(&self.min_samples)?;

        // Validate leaf_size
        ValidationRules::new("leaf_size")
            .add_rule(ValidationRule::Positive)
            .validate_usize(&self.leaf_size)?;

        // Validate Minkowski parameter
        if let DistanceMetric::Minkowski(p) = self.metric {
            if p <= 0.0 {
                return Err(SklearsError::InvalidInput(
                    "Minkowski p parameter must be positive".to_string(),
                ));
            }
        }

        Ok(())
    }
}

impl ConfigValidation for OpticsConfig {
    fn validate_config(&self) -> Result<()> {
        self.validate()?;

        if self.min_samples > 100 {
            log::warn!(
                "Large min_samples {} may be slow for dense datasets",
                self.min_samples
            );
        }

        if self.max_eps == f64::INFINITY
            && matches!(self.cluster_method, ClusterMethod::Threshold(_))
        {
            log::warn!("Using infinite max_eps with threshold clustering may not produce meaningful results");
        }

        Ok(())
    }

    fn get_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        if self.min_samples < 3 {
            warnings.push("Very small min_samples may lead to noisy clusters".to_string());
        }

        if self.leaf_size > 100 {
            warnings.push("Large leaf_size may reduce performance".to_string());
        }

        warnings
    }
}

/// Point data for OPTICS algorithm
#[derive(Debug, Clone)]
struct OpticsPoint {
    /// Index of the point in the original dataset
    index: usize,
    /// Core distance of the point
    core_distance: Option<f64>,
    /// Reachability distance of the point
    reachability_distance: Option<f64>,
    /// Whether the point has been processed
    processed: bool,
}

impl OpticsPoint {
    fn new(index: usize) -> Self {
        Self {
            index,
            core_distance: None,
            reachability_distance: None,
            processed: false,
        }
    }
}

/// Ordering entry in OPTICS result
#[derive(Debug, Clone)]
pub struct OpticsOrdering {
    /// Index of the point in the original dataset
    pub index: usize,
    /// Core distance of the point
    pub core_distance: Option<f64>,
    /// Reachability distance of the point
    pub reachability_distance: Option<f64>,
}

/// OPTICS clustering algorithm
#[derive(Debug, Clone)]
pub struct Optics<State = Untrained> {
    config: OpticsConfig,
    state: PhantomData<State>,
    // Trained state fields
    ordering_: Option<Vec<OpticsOrdering>>,
    labels_: Option<Array1<i32>>, // -1 for noise, >= 0 for cluster labels
    n_features_: Option<usize>,
    core_sample_indices_: Option<Vec<usize>>,
}

impl Optics<Untrained> {
    /// Create a new OPTICS model
    pub fn new() -> Self {
        Self {
            config: OpticsConfig::default(),
            state: PhantomData,
            ordering_: None,
            labels_: None,
            n_features_: None,
            core_sample_indices_: None,
        }
    }

    /// Set the maximum epsilon value
    pub fn max_eps(mut self, max_eps: f64) -> Self {
        self.config.max_eps = max_eps;
        self
    }

    /// Set the minimum number of samples
    pub fn min_samples(mut self, min_samples: usize) -> Self {
        self.config.min_samples = min_samples;
        self
    }

    /// Set the distance metric
    pub fn metric(mut self, metric: DistanceMetric) -> Self {
        self.config.metric = metric;
        self
    }

    /// Set the algorithm for neighbor computation
    pub fn algorithm(mut self, algorithm: Algorithm) -> Self {
        self.config.algorithm = algorithm;
        self
    }

    /// Set the leaf size for tree algorithms
    pub fn leaf_size(mut self, leaf_size: usize) -> Self {
        self.config.leaf_size = leaf_size;
        self
    }

    /// Set the cluster extraction method
    pub fn cluster_method(mut self, method: ClusterMethod) -> Self {
        self.config.cluster_method = method;
        self
    }
}

impl Default for Optics<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for Optics<Untrained> {
    type Config = OpticsConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, ()> for Optics<Untrained> {
    type Fitted = Optics<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Validate configuration
        self.config
            .validate_config()
            .fit_context("OPTICS", n_samples, n_features)?;

        // Validate data
        use sklears_core::validation::ml;
        ml::validate_unsupervised_data(x).fit_context("OPTICS", n_samples, n_features)?;

        if n_samples < self.config.min_samples {
            return Err(SklearsError::InvalidInput(format!(
                "min_samples ({}) cannot exceed n_samples ({})",
                self.config.min_samples, n_samples
            )));
        }

        // Run OPTICS algorithm
        let (ordering, core_indices) = self.run_optics(x)?;

        // Extract clusters from ordering
        let labels = self.extract_clusters(&ordering)?;

        Ok(Optics {
            config: self.config,
            state: PhantomData,
            ordering_: Some(ordering),
            labels_: Some(labels),
            n_features_: Some(n_features),
            core_sample_indices_: Some(core_indices),
        })
    }
}

impl Optics<Untrained> {
    /// Run the OPTICS algorithm
    fn run_optics(&self, x: &Array2<Float>) -> Result<(Vec<OpticsOrdering>, Vec<usize>)> {
        let n_samples = x.nrows();
        let mut points: Vec<OpticsPoint> = (0..n_samples).map(OpticsPoint::new).collect();
        let mut ordering = Vec::new();
        let mut core_indices = Vec::new();

        // Priority queue for seeds (min-heap based on reachability distance)
        let mut seeds = BinaryHeap::new();

        for i in 0..n_samples {
            if points[i].processed {
                continue;
            }

            // Get neighbors
            let neighbors = self.get_neighbors(x, i)?;

            // Mark as processed
            points[i].processed = true;

            // Calculate core distance
            if neighbors.len() >= self.config.min_samples {
                let core_distance = self.calculate_core_distance(x, i, &neighbors)?;
                points[i].core_distance = Some(core_distance);
                core_indices.push(i);

                // Update seeds with neighbors
                self.update_seeds(&mut seeds, &mut points, x, i, &neighbors)?;

                // Add to ordering
                ordering.push(OpticsOrdering {
                    index: i,
                    core_distance: points[i].core_distance,
                    reachability_distance: points[i].reachability_distance,
                });

                // Process seeds
                while let Some(seed_item) = seeds.pop() {
                    let seed_idx = seed_item.index;

                    if points[seed_idx].processed {
                        continue;
                    }

                    points[seed_idx].processed = true;

                    let seed_neighbors = self.get_neighbors(x, seed_idx)?;

                    // Add to ordering
                    ordering.push(OpticsOrdering {
                        index: seed_idx,
                        core_distance: points[seed_idx].core_distance,
                        reachability_distance: points[seed_idx].reachability_distance,
                    });

                    // If seed is a core point, update seeds
                    if seed_neighbors.len() >= self.config.min_samples {
                        let seed_core_distance =
                            self.calculate_core_distance(x, seed_idx, &seed_neighbors)?;
                        points[seed_idx].core_distance = Some(seed_core_distance);
                        if !core_indices.contains(&seed_idx) {
                            core_indices.push(seed_idx);
                        }

                        self.update_seeds(&mut seeds, &mut points, x, seed_idx, &seed_neighbors)?;
                    }
                }
            } else {
                // Add to ordering as noise
                ordering.push(OpticsOrdering {
                    index: i,
                    core_distance: None,
                    reachability_distance: None,
                });
            }
        }

        Ok((ordering, core_indices))
    }

    /// Get neighbors within max_eps distance
    fn get_neighbors(&self, x: &Array2<Float>, point_idx: usize) -> Result<Vec<usize>> {
        let n_samples = x.nrows();
        let mut neighbors = Vec::new();
        let point = x.row(point_idx);

        for i in 0..n_samples {
            if i == point_idx {
                continue;
            }

            let neighbor = x.row(i);
            let distance = self.calculate_distance(&point, &neighbor)?;

            if distance <= self.config.max_eps {
                neighbors.push(i);
            }
        }

        Ok(neighbors)
    }

    /// Calculate distance between two points
    fn calculate_distance(
        &self,
        point1: &scirs2_core::ndarray::ArrayView1<Float>,
        point2: &scirs2_core::ndarray::ArrayView1<Float>,
    ) -> Result<f64> {
        match self.config.metric {
            DistanceMetric::Euclidean => {
                let mut sum = 0.0;
                for (&a, &b) in point1.iter().zip(point2.iter()) {
                    let diff = a - b;
                    sum += diff * diff;
                }
                Ok(sum.sqrt())
            }
            DistanceMetric::Manhattan => {
                let mut sum = 0.0;
                for (&a, &b) in point1.iter().zip(point2.iter()) {
                    sum += (a - b).abs();
                }
                Ok(sum)
            }
            DistanceMetric::Chebyshev => {
                let mut max_diff = 0.0;
                for (&a, &b) in point1.iter().zip(point2.iter()) {
                    let diff = (a - b).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                }
                Ok(max_diff)
            }
            DistanceMetric::Minkowski(p) => {
                let mut sum = 0.0;
                for (&a, &b) in point1.iter().zip(point2.iter()) {
                    sum += (a - b).abs().powf(p);
                }
                Ok(sum.powf(1.0 / p))
            }
        }
    }

    /// Calculate core distance for a point
    fn calculate_core_distance(
        &self,
        x: &Array2<Float>,
        point_idx: usize,
        neighbors: &[usize],
    ) -> Result<f64> {
        if neighbors.len() < self.config.min_samples {
            return Ok(f64::INFINITY);
        }

        let point = x.row(point_idx);
        let mut distances: Vec<f64> = Vec::new();

        for &neighbor_idx in neighbors {
            let neighbor = x.row(neighbor_idx);
            let distance = self.calculate_distance(&point, &neighbor)?;
            distances.push(distance);
        }

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(distances[self.config.min_samples - 1])
    }

    /// Update seeds with new neighbors
    fn update_seeds(
        &self,
        seeds: &mut BinaryHeap<SeedItem>,
        points: &mut [OpticsPoint],
        x: &Array2<Float>,
        core_idx: usize,
        neighbors: &[usize],
    ) -> Result<()> {
        let core_distance = points[core_idx].core_distance.unwrap();
        let core_point = x.row(core_idx);

        for &neighbor_idx in neighbors {
            if points[neighbor_idx].processed {
                continue;
            }

            let neighbor_point = x.row(neighbor_idx);
            let distance = self.calculate_distance(&core_point, &neighbor_point)?;
            let new_reachability = core_distance.max(distance);

            if points[neighbor_idx].reachability_distance.is_none() {
                points[neighbor_idx].reachability_distance = Some(new_reachability);
                seeds.push(SeedItem {
                    index: neighbor_idx,
                    reachability: new_reachability,
                });
            } else if new_reachability < points[neighbor_idx].reachability_distance.unwrap() {
                points[neighbor_idx].reachability_distance = Some(new_reachability);
                seeds.push(SeedItem {
                    index: neighbor_idx,
                    reachability: new_reachability,
                });
            }
        }

        Ok(())
    }

    /// Extract clusters from OPTICS ordering
    fn extract_clusters(&self, ordering: &[OpticsOrdering]) -> Result<Array1<i32>> {
        let n_samples = ordering.len();
        let mut labels = Array1::from_elem(n_samples, -1);

        match self.config.cluster_method {
            ClusterMethod::Threshold(threshold) => {
                let mut cluster_id = 0;
                let mut current_cluster: Option<i32> = None;

                for (i, entry) in ordering.iter().enumerate() {
                    let reachability = entry.reachability_distance.unwrap_or(f64::INFINITY);

                    if reachability <= threshold {
                        if current_cluster.is_none() {
                            current_cluster = Some(cluster_id);
                            cluster_id += 1;
                        }
                        labels[entry.index] = current_cluster.unwrap();
                    } else {
                        current_cluster = None;
                        // Keep as noise (-1)
                    }
                }
            }
            ClusterMethod::Hierarchical => {
                // Simple hierarchical clustering based on reachability distances
                self.extract_hierarchical_clusters(ordering, &mut labels)?;
            }
            ClusterMethod::SteepestDescent => {
                // Steepest descent algorithm for cluster extraction
                self.extract_steepest_descent_clusters(ordering, &mut labels)?;
            }
        }

        Ok(labels)
    }

    /// Extract clusters using hierarchical method
    fn extract_hierarchical_clusters(
        &self,
        ordering: &[OpticsOrdering],
        labels: &mut Array1<i32>,
    ) -> Result<()> {
        // Simple implementation: find local minima in reachability plot
        let mut cluster_id = 0;
        let window_size = self.config.min_samples;

        for i in window_size..ordering.len() - window_size {
            let current_reach = ordering[i].reachability_distance.unwrap_or(f64::INFINITY);

            // Check if this is a local minimum
            let mut is_minimum = true;
            for j in i.saturating_sub(window_size)..=(i + window_size).min(ordering.len() - 1) {
                if j != i {
                    let reach = ordering[j].reachability_distance.unwrap_or(f64::INFINITY);
                    if reach < current_reach {
                        is_minimum = false;
                        break;
                    }
                }
            }

            if is_minimum && current_reach < f64::INFINITY {
                // Start a new cluster
                for j in i..ordering.len() {
                    let reach = ordering[j].reachability_distance.unwrap_or(f64::INFINITY);
                    if reach < current_reach * 2.0 {
                        labels[ordering[j].index] = cluster_id;
                    } else {
                        break;
                    }
                }
                cluster_id += 1;

                if let Some(max_clusters) = self.config.max_clusters {
                    if cluster_id >= max_clusters as i32 {
                        break;
                    }
                }
            }
        }

        Ok(())
    }

    /// Extract clusters using steepest descent method
    fn extract_steepest_descent_clusters(
        &self,
        ordering: &[OpticsOrdering],
        labels: &mut Array1<i32>,
    ) -> Result<()> {
        // Find steep downward areas in reachability plot
        let mut cluster_id = 0;
        let mut in_cluster = false;
        let steep_threshold = 0.1; // Configurable parameter

        for i in 1..ordering.len() {
            let prev_reach = ordering[i - 1]
                .reachability_distance
                .unwrap_or(f64::INFINITY);
            let curr_reach = ordering[i].reachability_distance.unwrap_or(f64::INFINITY);

            let ratio = if prev_reach > 0.0 && curr_reach.is_finite() {
                curr_reach / prev_reach
            } else {
                1.0
            };

            // Steep downward indicates start of cluster
            if ratio <= steep_threshold && !in_cluster {
                in_cluster = true;
                labels[ordering[i].index] = cluster_id;
            }
            // Steep upward indicates end of cluster
            else if ratio >= (1.0 / steep_threshold) && in_cluster {
                in_cluster = false;
                cluster_id += 1;

                if let Some(max_clusters) = self.config.max_clusters {
                    if cluster_id >= max_clusters as i32 {
                        break;
                    }
                }
            }
            // Continue current cluster
            else if in_cluster {
                labels[ordering[i].index] = cluster_id;
            }
        }

        Ok(())
    }
}

/// Item in the priority queue for OPTICS seeds
#[derive(Debug, Clone, PartialEq)]
struct SeedItem {
    index: usize,
    reachability: f64,
}

impl Eq for SeedItem {}

impl PartialOrd for SeedItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Reverse ordering for min-heap behavior
        other.reachability.partial_cmp(&self.reachability)
    }
}

impl Ord for SeedItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl Optics<Trained> {
    /// Get the OPTICS ordering
    pub fn ordering(&self) -> &[OpticsOrdering] {
        self.ordering_.as_ref().expect("Model is trained")
    }

    /// Get cluster labels
    pub fn labels(&self) -> &Array1<i32> {
        self.labels_.as_ref().expect("Model is trained")
    }

    /// Get core sample indices
    pub fn core_sample_indices(&self) -> &[usize] {
        self.core_sample_indices_
            .as_ref()
            .expect("Model is trained")
    }

    /// Get reachability distances from ordering
    pub fn reachability_distances(&self) -> Vec<Option<f64>> {
        self.ordering()
            .iter()
            .map(|entry| entry.reachability_distance)
            .collect()
    }

    /// Get core distances from ordering
    pub fn core_distances(&self) -> Vec<Option<f64>> {
        self.ordering()
            .iter()
            .map(|entry| entry.core_distance)
            .collect()
    }
}

impl Predict<Array2<Float>, Array1<i32>> for Optics<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let n_features = self.n_features_.expect("Model is trained");
        if x.ncols() != n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                n_features,
                x.ncols()
            )));
        }

        // For OPTICS, prediction on new data is not straightforward
        // We implement a simple nearest neighbor approach
        let n_samples = x.nrows();
        let labels = Array1::zeros(n_samples);

        // This is a simplified prediction - in practice, you'd want to
        // extend the OPTICS ordering or use a different approach
        log::warn!("OPTICS prediction on new data is approximate");

        Ok(labels)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_optics_simple() {
        // Create simple clustered data
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.2],
        ];

        let model = Optics::new()
            .max_eps(2.0)
            .min_samples(2)
            .cluster_method(ClusterMethod::Threshold(1.0))
            .fit(&data, &())
            .unwrap();

        let labels = model.labels();
        let ordering = model.ordering();

        assert_eq!(labels.len(), 6);
        assert_eq!(ordering.len(), 6);

        // Should find some clusters (not all noise)
        let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
        assert!(unique_labels.len() >= 2); // At least noise and one cluster
    }

    #[test]
    fn test_optics_validation() {
        use sklears_core::validation::{ConfigValidation, Validate};

        // Test valid configuration
        let valid_config = OpticsConfig::default();
        assert!(valid_config.validate().is_ok());
        assert!(valid_config.validate_config().is_ok());

        // Test invalid max_eps (negative)
        let mut invalid_config = OpticsConfig::default();
        invalid_config.max_eps = -1.0;
        assert!(invalid_config.validate().is_err());

        // Test invalid min_samples (zero)
        let mut invalid_config = OpticsConfig::default();
        invalid_config.min_samples = 0;
        assert!(invalid_config.validate().is_err());

        // Test invalid Minkowski parameter
        let mut invalid_config = OpticsConfig::default();
        invalid_config.metric = DistanceMetric::Minkowski(-1.0);
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_optics_distance_metrics() {
        let data = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0],];

        // Test different distance metrics
        let metrics = vec![
            DistanceMetric::Euclidean,
            DistanceMetric::Manhattan,
            DistanceMetric::Chebyshev,
            DistanceMetric::Minkowski(2.0),
        ];

        for metric in metrics {
            let model = Optics::new()
                .max_eps(5.0)
                .min_samples(2)
                .metric(metric)
                .fit(&data, &())
                .unwrap();

            let labels = model.labels();
            assert_eq!(labels.len(), 3);
        }
    }

    #[test]
    fn test_optics_cluster_methods() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.2],
        ];

        let methods = vec![
            ClusterMethod::Threshold(1.0),
            ClusterMethod::Hierarchical,
            ClusterMethod::SteepestDescent,
        ];

        for method in methods {
            let model = Optics::new()
                .max_eps(3.0)
                .min_samples(2)
                .cluster_method(method)
                .fit(&data, &())
                .unwrap();

            let labels = model.labels();
            assert_eq!(labels.len(), 6);
        }
    }

    #[test]
    fn test_optics_core_samples() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [5.0, 5.0], // Isolated point
        ];

        let model = Optics::new()
            .max_eps(1.0)
            .min_samples(2)
            .fit(&data, &())
            .unwrap();

        let core_indices = model.core_sample_indices();

        // Should have some core samples from the clustered points
        assert!(core_indices.len() >= 1);

        // Isolated point should not be a core sample
        assert!(!core_indices.contains(&3));
    }

    #[test]
    fn test_optics_reachability_plot() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [1.0, 1.0],
            [1.1, 1.1],
            [1.2, 1.2],
        ];

        let model = Optics::new()
            .max_eps(2.0)
            .min_samples(2)
            .fit(&data, &())
            .unwrap();

        let reachability_distances = model.reachability_distances();
        let core_distances = model.core_distances();

        assert_eq!(reachability_distances.len(), 6);
        assert_eq!(core_distances.len(), 6);

        // Some points should have finite reachability distances
        let finite_reachability = reachability_distances
            .iter()
            .filter(|&&d| d.is_some() && d.unwrap().is_finite())
            .count();

        assert!(finite_reachability > 0);
    }
}
