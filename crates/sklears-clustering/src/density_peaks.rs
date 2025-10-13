//! Density Peaks Clustering algorithm
//!
//! This algorithm automatically finds cluster centers based on the observation that
//! cluster centers are characterized by a higher density than their neighbors and by
//! a relatively large distance from points with higher densities.

use std::marker::PhantomData;

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Configuration for Density Peaks clustering
#[derive(Debug, Clone)]
pub struct DensityPeaksConfig {
    /// Distance metric to use
    pub metric: DistanceMetric,
    /// Cutoff distance (dc) for density calculation - auto-computed if None
    pub cutoff_distance: Option<Float>,
    /// Percentile used to compute cutoff distance (typically 1-2%)
    pub cutoff_percentile: Float,
    /// Minimum density for cluster centers
    pub min_density: Option<Float>,
    /// Minimum delta (distance to higher density point) for cluster centers
    pub min_delta: Option<Float>,
    /// Whether to use Gaussian kernel for density calculation
    pub use_gaussian_kernel: bool,
}

impl Default for DensityPeaksConfig {
    fn default() -> Self {
        Self {
            metric: DistanceMetric::Euclidean,
            cutoff_distance: None,
            cutoff_percentile: 2.0,
            min_density: None,
            min_delta: None,
            use_gaussian_kernel: false,
        }
    }
}

/// Distance metrics for density peaks clustering
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Chebyshev,
}

/// Density Peaks clustering algorithm
#[derive(Debug, Clone)]
pub struct DensityPeaks<State = Untrained> {
    config: DensityPeaksConfig,
    state: PhantomData<State>,
    // Trained state fields
    training_data_: Option<Array2<Float>>,
    cluster_centers_: Option<Array2<Float>>,
    center_indices_: Option<Vec<usize>>,
    labels_: Option<Array1<i32>>,
    densities_: Option<Array1<Float>>,
    deltas_: Option<Array1<Float>>,
    cutoff_distance_: Option<Float>,
    n_features_: Option<usize>,
}

impl DensityPeaks<Untrained> {
    /// Create a new Density Peaks model
    pub fn new() -> Self {
        Self {
            config: DensityPeaksConfig::default(),
            state: PhantomData,
            training_data_: None,
            cluster_centers_: None,
            center_indices_: None,
            labels_: None,
            densities_: None,
            deltas_: None,
            cutoff_distance_: None,
            n_features_: None,
        }
    }

    /// Set distance metric
    pub fn metric(mut self, metric: DistanceMetric) -> Self {
        self.config.metric = metric;
        self
    }

    /// Set explicit cutoff distance
    pub fn cutoff_distance(mut self, cutoff_distance: Float) -> Self {
        self.config.cutoff_distance = Some(cutoff_distance);
        self
    }

    /// Set cutoff percentile for auto-computing cutoff distance
    pub fn cutoff_percentile(mut self, cutoff_percentile: Float) -> Self {
        self.config.cutoff_percentile = cutoff_percentile;
        self
    }

    /// Set minimum density for cluster centers
    pub fn min_density(mut self, min_density: Float) -> Self {
        self.config.min_density = Some(min_density);
        self
    }

    /// Set minimum delta for cluster centers
    pub fn min_delta(mut self, min_delta: Float) -> Self {
        self.config.min_delta = Some(min_delta);
        self
    }

    /// Use Gaussian kernel for density calculation
    pub fn use_gaussian_kernel(mut self, use_gaussian: bool) -> Self {
        self.config.use_gaussian_kernel = use_gaussian;
        self
    }
}

impl Default for DensityPeaks<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for DensityPeaks<Untrained> {
    type Config = DensityPeaksConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, ()> for DensityPeaks<Untrained> {
    type Fitted = DensityPeaks<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput("Empty dataset".to_string()));
        }

        // Step 1: Compute distance matrix
        let distances = self.compute_distance_matrix(x)?;

        // Step 2: Determine cutoff distance
        let cutoff_distance = match self.config.cutoff_distance {
            Some(dc) => dc,
            None => self.compute_cutoff_distance(&distances)?,
        };

        // Step 3: Compute local densities
        let densities = self.compute_densities(&distances, cutoff_distance)?;

        // Step 4: Compute deltas (distance to nearest higher density point)
        let deltas = self.compute_deltas(&distances, &densities)?;

        // Step 5: Find cluster centers
        let center_indices = self.find_cluster_centers(&densities, &deltas)?;

        // Step 6: Assign points to clusters
        let labels = self.assign_clusters(&distances, &densities, &center_indices)?;

        // Extract cluster centers
        let cluster_centers = if center_indices.is_empty() {
            Array2::zeros((0, n_features))
        } else {
            let centers: Vec<_> = center_indices
                .iter()
                .map(|&idx| x.row(idx).to_owned())
                .collect();
            Array2::from_shape_vec(
                (centers.len(), n_features),
                centers.into_iter().flatten().collect(),
            )
            .map_err(|e| SklearsError::Other(format!("Failed to create centers array: {}", e)))?
        };

        Ok(DensityPeaks {
            config: self.config,
            state: PhantomData,
            training_data_: Some(x.clone()),
            cluster_centers_: Some(cluster_centers),
            center_indices_: Some(center_indices),
            labels_: Some(labels),
            densities_: Some(densities),
            deltas_: Some(deltas),
            cutoff_distance_: Some(cutoff_distance),
            n_features_: Some(n_features),
        })
    }
}

impl DensityPeaks<Untrained> {
    /// Compute distance matrix between all pairs of points
    fn compute_distance_matrix(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let mut distances = Array2::zeros((n_samples, n_samples));

        #[cfg(feature = "parallel")]
        {
            distances
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(i, mut row)| {
                    for j in 0..n_samples {
                        if i != j {
                            let dist = self.distance(&x.row(i), &x.row(j));
                            row[j] = dist;
                        }
                    }
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..n_samples {
                for j in 0..n_samples {
                    if i != j {
                        let dist = self.distance(&x.row(i), &x.row(j));
                        distances[[i, j]] = dist;
                    }
                }
            }
        }

        Ok(distances)
    }

    /// Compute cutoff distance as percentile of all pairwise distances
    fn compute_cutoff_distance(&self, distances: &Array2<Float>) -> Result<Float> {
        let n_samples = distances.nrows();
        let mut all_distances = Vec::new();

        // Collect upper triangle of distance matrix (excluding diagonal)
        for i in 0..n_samples {
            for j in i + 1..n_samples {
                all_distances.push(distances[[i, j]]);
            }
        }

        if all_distances.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Not enough points to compute cutoff distance".to_string(),
            ));
        }

        all_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let percentile_idx =
            ((self.config.cutoff_percentile / 100.0) * all_distances.len() as Float) as usize;
        let cutoff_idx = percentile_idx.min(all_distances.len() - 1);

        Ok(all_distances[cutoff_idx])
    }

    /// Compute local densities for each point
    fn compute_densities(
        &self,
        distances: &Array2<Float>,
        cutoff_distance: Float,
    ) -> Result<Array1<Float>> {
        let n_samples = distances.nrows();
        let mut densities = Array1::zeros(n_samples);

        if self.config.use_gaussian_kernel {
            // Gaussian kernel density
            for i in 0..n_samples {
                let mut density = 0.0;
                for j in 0..n_samples {
                    if i != j {
                        let dist = distances[[i, j]];
                        density += (-((dist / cutoff_distance).powi(2))).exp();
                    }
                }
                densities[i] = density;
            }
        } else {
            // Simple cutoff density
            for i in 0..n_samples {
                let mut density = 0.0;
                for j in 0..n_samples {
                    if i != j && distances[[i, j]] < cutoff_distance {
                        density += 1.0;
                    }
                }
                densities[i] = density;
            }
        }

        Ok(densities)
    }

    /// Compute delta values (distance to nearest higher density point)
    fn compute_deltas(
        &self,
        distances: &Array2<Float>,
        densities: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let n_samples = distances.nrows();
        let mut deltas = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut min_dist = Float::INFINITY;
            let mut found_higher = false;

            for j in 0..n_samples {
                if i != j && densities[j] > densities[i] {
                    found_higher = true;
                    if distances[[i, j]] < min_dist {
                        min_dist = distances[[i, j]];
                    }
                }
            }

            if found_higher {
                deltas[i] = min_dist;
            } else {
                // This is the highest density point, set delta to max distance
                let mut max_dist = 0.0;
                for j in 0..n_samples {
                    if i != j && distances[[i, j]] > max_dist {
                        max_dist = distances[[i, j]];
                    }
                }
                deltas[i] = max_dist;
            }
        }

        Ok(deltas)
    }

    /// Find cluster centers based on density and delta values
    fn find_cluster_centers(
        &self,
        densities: &Array1<Float>,
        deltas: &Array1<Float>,
    ) -> Result<Vec<usize>> {
        let n_samples = densities.len();
        let mut centers = Vec::new();

        // Determine thresholds
        let density_threshold = match self.config.min_density {
            Some(threshold) => threshold,
            None => {
                let mut sorted_densities = densities.to_vec();
                sorted_densities.sort_by(|a, b| b.partial_cmp(a).unwrap());
                // Use 80th percentile as default
                let idx = (0.2 * sorted_densities.len() as Float) as usize;
                sorted_densities[idx.min(sorted_densities.len() - 1)]
            }
        };

        let delta_threshold = match self.config.min_delta {
            Some(threshold) => threshold,
            None => {
                let mut sorted_deltas = deltas.to_vec();
                sorted_deltas.sort_by(|a, b| b.partial_cmp(a).unwrap());
                // Use 80th percentile as default
                let idx = (0.2 * sorted_deltas.len() as Float) as usize;
                sorted_deltas[idx.min(sorted_deltas.len() - 1)]
            }
        };

        // Find points that exceed both thresholds
        for i in 0..n_samples {
            if densities[i] >= density_threshold && deltas[i] >= delta_threshold {
                centers.push(i);
            }
        }

        // If no centers found, use the point with highest product of density and delta
        if centers.is_empty() {
            let mut max_product = 0.0;
            let mut best_idx = 0;

            for i in 0..n_samples {
                let product = densities[i] * deltas[i];
                if product > max_product {
                    max_product = product;
                    best_idx = i;
                }
            }
            centers.push(best_idx);
        }

        Ok(centers)
    }

    /// Assign points to clusters based on density and centers
    fn assign_clusters(
        &self,
        distances: &Array2<Float>,
        densities: &Array1<Float>,
        center_indices: &[usize],
    ) -> Result<Array1<i32>> {
        let n_samples = distances.nrows();
        let mut labels = Array1::from_elem(n_samples, -1i32); // Initialize as noise

        // Assign centers to their own clusters
        for (cluster_id, &center_idx) in center_indices.iter().enumerate() {
            labels[center_idx] = cluster_id as i32;
        }

        // Create sorted indices by density (descending)
        let mut density_order: Vec<usize> = (0..n_samples).collect();
        density_order.sort_by(|&a, &b| densities[b].partial_cmp(&densities[a]).unwrap());

        // Assign points in order of decreasing density
        for &i in &density_order {
            if labels[i] != -1 {
                continue; // Already assigned (cluster center)
            }

            // Find nearest higher density point
            let mut min_dist = Float::INFINITY;
            let mut nearest_higher = None;

            for j in 0..n_samples {
                if i != j && densities[j] > densities[i] && distances[[i, j]] < min_dist {
                    min_dist = distances[[i, j]];
                    nearest_higher = Some(j);
                }
            }

            // Assign same label as nearest higher density point
            if let Some(j) = nearest_higher {
                if labels[j] != -1 {
                    labels[i] = labels[j];
                }
            }
        }

        Ok(labels)
    }

    /// Calculate distance between two points
    fn distance(&self, point1: &ArrayView1<Float>, point2: &ArrayView1<Float>) -> Float {
        match self.config.metric {
            DistanceMetric::Euclidean => {
                let diff = point1.to_owned() - point2;
                diff.dot(&diff).sqrt()
            }
            DistanceMetric::Manhattan => point1
                .iter()
                .zip(point2.iter())
                .map(|(&a, &b)| (a - b).abs())
                .sum(),
            DistanceMetric::Chebyshev => point1
                .iter()
                .zip(point2.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0, |max, val| if val > max { val } else { max }),
        }
    }
}

impl DensityPeaks<Trained> {
    /// Get cluster labels
    pub fn labels(&self) -> &Array1<i32> {
        self.labels_.as_ref().expect("Model is trained")
    }

    /// Get cluster centers
    pub fn cluster_centers(&self) -> &Array2<Float> {
        self.cluster_centers_.as_ref().expect("Model is trained")
    }

    /// Get cluster center indices in original data
    pub fn center_indices(&self) -> &[usize] {
        self.center_indices_.as_ref().expect("Model is trained")
    }

    /// Get density values for training points
    pub fn densities(&self) -> &Array1<Float> {
        self.densities_.as_ref().expect("Model is trained")
    }

    /// Get delta values for training points
    pub fn deltas(&self) -> &Array1<Float> {
        self.deltas_.as_ref().expect("Model is trained")
    }

    /// Get the computed cutoff distance
    pub fn cutoff_distance(&self) -> Float {
        self.cutoff_distance_.expect("Model is trained")
    }

    /// Get number of clusters found
    pub fn n_clusters(&self) -> usize {
        self.cluster_centers().nrows()
    }

    /// Get number of noise points
    pub fn n_noise_points(&self) -> usize {
        let labels = self.labels();
        labels.iter().filter(|&&label| label == -1).count()
    }
}

impl Predict<Array2<Float>, Array1<i32>> for DensityPeaks<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let n_features = self.n_features_.expect("Model is trained");
        if x.ncols() != n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                n_features,
                x.ncols()
            )));
        }

        let centers = self.cluster_centers();
        let n_samples = x.nrows();
        let mut labels = Array1::zeros(n_samples);

        // Assign each point to nearest cluster center
        for i in 0..n_samples {
            let point = x.row(i);
            let mut min_distance = Float::INFINITY;
            let mut best_cluster = 0;

            for (j, center) in centers.axis_iter(Axis(0)).enumerate() {
                let dist = self.distance_to_center(&point, &center);
                if dist < min_distance {
                    min_distance = dist;
                    best_cluster = j;
                }
            }

            labels[i] = best_cluster as i32;
        }

        Ok(labels)
    }
}

impl DensityPeaks<Trained> {
    /// Calculate distance to cluster center
    fn distance_to_center(&self, point: &ArrayView1<Float>, center: &ArrayView1<Float>) -> Float {
        match self.config.metric {
            DistanceMetric::Euclidean => {
                let diff = point.to_owned() - center;
                diff.dot(&diff).sqrt()
            }
            DistanceMetric::Manhattan => point
                .iter()
                .zip(center.iter())
                .map(|(&a, &b)| (a - b).abs())
                .sum(),
            DistanceMetric::Chebyshev => point
                .iter()
                .zip(center.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0, |max, val| if val > max { val } else { max }),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_density_peaks_basic() {
        let data = array![
            // Cluster 1
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [0.0, 0.2],
            // Cluster 2
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.0],
            [5.0, 5.2],
            // Outlier
            [10.0, 10.0],
        ];

        let model = DensityPeaks::new()
            .cutoff_percentile(20.0) // Use larger percentile for small dataset
            .fit(&data, &())
            .unwrap();

        let labels = model.labels();
        let n_clusters = model.n_clusters();

        assert!(n_clusters >= 1);
        assert!(n_clusters <= 3);
        assert_eq!(labels.len(), 9);

        // Check that we found some cluster structure
        let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
        assert!(unique_labels.len() >= 2);
    }

    #[test]
    fn test_density_peaks_predict() {
        let train_data = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1],];

        let model = DensityPeaks::new()
            .cutoff_percentile(20.0)
            .fit(&train_data, &())
            .unwrap();

        let test_data = array![
            [0.05, 0.05], // Close to first cluster
            [5.05, 5.05], // Close to second cluster
        ];

        let predictions = model.predict(&test_data).unwrap();
        assert_eq!(predictions.len(), 2);
    }

    #[test]
    fn test_density_peaks_gaussian_kernel() {
        let data = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1],];

        let model = DensityPeaks::new()
            .use_gaussian_kernel(true)
            .cutoff_percentile(20.0)
            .fit(&data, &())
            .unwrap();

        let densities = model.densities();
        assert!(densities.iter().all(|&d| d >= 0.0));
    }

    #[test]
    fn test_density_peaks_different_metrics() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0],];

        for metric in [
            DistanceMetric::Euclidean,
            DistanceMetric::Manhattan,
            DistanceMetric::Chebyshev,
        ] {
            let model = DensityPeaks::new()
                .metric(metric)
                .cutoff_percentile(30.0)
                .fit(&data, &())
                .unwrap();

            assert!(model.n_clusters() >= 1);
        }
    }
}
