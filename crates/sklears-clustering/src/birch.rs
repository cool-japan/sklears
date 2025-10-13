//! BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) implementation
//!
//! BIRCH is designed for clustering large datasets efficiently. It builds a tree-like
//! data structure called CF-Tree (Clustering Feature Tree) that summarizes cluster
//! information and enables incremental clustering.

use std::marker::PhantomData;

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};

/// Configuration for BIRCH clustering
#[derive(Debug, Clone)]
pub struct BIRCHConfig {
    /// Maximum number of CF entries in each leaf node
    pub threshold: Float,
    /// Branching factor (maximum number of child nodes)
    pub branching_factor: usize,
    /// Number of final clusters (None for no final clustering)
    pub n_clusters: Option<usize>,
    /// Memory limit for CF-Tree (in number of nodes)
    pub memory_limit: Option<usize>,
}

impl Default for BIRCHConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            branching_factor: 50,
            n_clusters: None,
            memory_limit: None,
        }
    }
}

/// Clustering Feature (CF) - summarizes a cluster
#[derive(Debug, Clone)]
pub struct ClusteringFeature {
    /// Number of points in the cluster
    n: usize,
    /// Linear sum of all points
    linear_sum: Array1<Float>,
    /// Sum of squares of all points
    sum_of_squares: Float,
}

impl ClusteringFeature {
    /// Create a new CF from a single point
    pub fn from_point(point: &ArrayView1<Float>) -> Self {
        let n = 1;
        let linear_sum = point.to_owned();
        let sum_of_squares = point.dot(point);

        Self {
            n,
            linear_sum,
            sum_of_squares,
        }
    }

    /// Create an empty CF
    pub fn empty(dimensions: usize) -> Self {
        Self {
            n: 0,
            linear_sum: Array1::zeros(dimensions),
            sum_of_squares: 0.0,
        }
    }

    /// Add another CF to this one
    pub fn add(&mut self, other: &ClusteringFeature) {
        self.n += other.n;
        self.linear_sum = &self.linear_sum + &other.linear_sum;
        self.sum_of_squares += other.sum_of_squares;
    }

    /// Compute centroid of this CF
    pub fn centroid(&self) -> Array1<Float> {
        if self.n > 0 {
            &self.linear_sum / (self.n as Float)
        } else {
            self.linear_sum.clone()
        }
    }

    /// Compute radius (average distance from centroid)
    pub fn radius(&self) -> Float {
        if self.n <= 1 {
            return 0.0;
        }

        let centroid = self.centroid();
        let centroid_norm_sq = centroid.dot(&centroid);
        let variance =
            (self.sum_of_squares - (self.n as Float) * centroid_norm_sq) / (self.n as Float);
        variance.max(0.0).sqrt()
    }

    /// Compute diameter (maximum pairwise distance within cluster)
    pub fn diameter(&self) -> Float {
        if self.n <= 1 {
            return 0.0;
        }

        // For efficiency, use an approximation based on variance
        2.0 * self.radius()
    }

    /// Distance between this CF and another CF (centroid distance)
    pub fn distance_to(&self, other: &ClusteringFeature) -> Float {
        let centroid1 = self.centroid();
        let centroid2 = other.centroid();
        let diff = &centroid1 - &centroid2;
        diff.dot(&diff).sqrt()
    }
}

/// Node in the CF-Tree
#[derive(Debug, Clone)]
pub struct CFNode {
    /// Clustering features stored in this node
    cfs: Vec<ClusteringFeature>,
    /// Child nodes (None for leaf nodes)
    children: Option<Vec<CFNode>>,
    /// Whether this is a leaf node
    is_leaf: bool,
    /// Maximum number of CFs this node can hold
    max_entries: usize,
}

impl CFNode {
    /// Create a new leaf node
    pub fn new_leaf(max_entries: usize) -> Self {
        Self {
            cfs: Vec::new(),
            children: None,
            is_leaf: true,
            max_entries,
        }
    }

    /// Create a new internal node
    pub fn new_internal(max_entries: usize) -> Self {
        Self {
            cfs: Vec::new(),
            children: Some(Vec::new()),
            is_leaf: false,
            max_entries,
        }
    }

    /// Check if node is full
    pub fn is_full(&self) -> bool {
        self.cfs.len() >= self.max_entries
    }

    /// Insert a point into this subtree
    pub fn insert(
        &mut self,
        point: &ArrayView1<Float>,
        threshold: Float,
    ) -> Result<Option<CFNode>> {
        if self.is_leaf {
            self.insert_into_leaf(point, threshold)
        } else {
            self.insert_into_internal(point, threshold)
        }
    }

    /// Insert into leaf node
    fn insert_into_leaf(
        &mut self,
        point: &ArrayView1<Float>,
        threshold: Float,
    ) -> Result<Option<CFNode>> {
        let point_cf = ClusteringFeature::from_point(point);

        // Find closest CF that can absorb this point
        let mut best_idx = None;
        let mut min_distance = Float::INFINITY;

        for (i, cf) in self.cfs.iter().enumerate() {
            let distance = cf.distance_to(&point_cf);
            if distance < min_distance {
                min_distance = distance;
                best_idx = Some(i);
            }
        }

        if let Some(idx) = best_idx {
            // Check if we can merge with the closest CF
            let mut test_cf = self.cfs[idx].clone();
            test_cf.add(&point_cf);

            if test_cf.radius() <= threshold {
                // Merge with existing CF
                self.cfs[idx].add(&point_cf);
                return Ok(None);
            }
        }

        // Create new CF
        if !self.is_full() {
            self.cfs.push(point_cf);
            Ok(None)
        } else {
            // Node is full, need to split
            self.cfs.push(point_cf);
            self.split_leaf()
        }
    }

    /// Insert into internal node
    fn insert_into_internal(
        &mut self,
        point: &ArrayView1<Float>,
        threshold: Float,
    ) -> Result<Option<CFNode>> {
        let point_cf = ClusteringFeature::from_point(point);

        // Find closest child
        let mut best_idx = 0;
        let mut min_distance = Float::INFINITY;

        for (i, cf) in self.cfs.iter().enumerate() {
            let distance = cf.distance_to(&point_cf);
            if distance < min_distance {
                min_distance = distance;
                best_idx = i;
            }
        }

        // Check if we have children first
        if self.children.is_none() {
            return Err(SklearsError::Other(
                "Internal node has no children".to_string(),
            ));
        }

        // Insert into closest child
        let split_result = {
            let children = self.children.as_mut().unwrap();
            children[best_idx].insert(point, threshold)?
        };

        if let Some(new_node) = split_result {
            // Child split, need to handle the new node
            self.cfs[best_idx].add(&point_cf);

            let is_full = self.cfs.len() >= self.max_entries;
            if !is_full {
                // Add new CF and child
                self.cfs.push(new_node.cfs.iter().fold(
                    ClusteringFeature::empty(point.len()),
                    |mut acc, cf| {
                        acc.add(cf);
                        acc
                    },
                ));
                self.children.as_mut().unwrap().push(new_node);
                Ok(None)
            } else {
                // This node is also full, need to split
                self.cfs.push(new_node.cfs.iter().fold(
                    ClusteringFeature::empty(point.len()),
                    |mut acc, cf| {
                        acc.add(cf);
                        acc
                    },
                ));
                self.children.as_mut().unwrap().push(new_node);
                self.split_internal()
            }
        } else {
            // No split needed, just update CF
            self.cfs[best_idx].add(&point_cf);
            Ok(None)
        }
    }

    /// Split a leaf node
    fn split_leaf(&self) -> Result<Option<CFNode>> {
        // Find the two most distant CFs
        let (idx1, idx2) = self.find_farthest_pair()?;

        // Create two new leaf nodes
        let mut node1 = CFNode::new_leaf(self.max_entries);
        let mut node2 = CFNode::new_leaf(self.max_entries);

        // Distribute CFs between the two nodes
        for (i, cf) in self.cfs.iter().enumerate() {
            if i == idx1 {
                node1.cfs.push(cf.clone());
            } else if i == idx2 {
                node2.cfs.push(cf.clone());
            } else {
                // Assign to closer node
                let dist1 = cf.distance_to(&self.cfs[idx1]);
                let dist2 = cf.distance_to(&self.cfs[idx2]);

                if dist1 <= dist2 {
                    node1.cfs.push(cf.clone());
                } else {
                    node2.cfs.push(cf.clone());
                }
            }
        }

        // This is a simplified implementation - in practice, we would need to
        // update the parent node structure
        Ok(Some(node2))
    }

    /// Split an internal node
    fn split_internal(&self) -> Result<Option<CFNode>> {
        // Similar to split_leaf but handles children
        let (idx1, idx2) = self.find_farthest_pair()?;

        let mut node1 = CFNode::new_internal(self.max_entries);
        let mut node2 = CFNode::new_internal(self.max_entries);

        if let Some(ref children) = self.children {
            for (i, (cf, child)) in self.cfs.iter().zip(children.iter()).enumerate() {
                if i == idx1 {
                    node1.cfs.push(cf.clone());
                    node1.children.as_mut().unwrap().push(child.clone());
                } else if i == idx2 {
                    node2.cfs.push(cf.clone());
                    node2.children.as_mut().unwrap().push(child.clone());
                } else {
                    let dist1 = cf.distance_to(&self.cfs[idx1]);
                    let dist2 = cf.distance_to(&self.cfs[idx2]);

                    if dist1 <= dist2 {
                        node1.cfs.push(cf.clone());
                        node1.children.as_mut().unwrap().push(child.clone());
                    } else {
                        node2.cfs.push(cf.clone());
                        node2.children.as_mut().unwrap().push(child.clone());
                    }
                }
            }
        }

        Ok(Some(node2))
    }

    /// Find the pair of CFs with maximum distance
    fn find_farthest_pair(&self) -> Result<(usize, usize)> {
        if self.cfs.len() < 2 {
            return Err(SklearsError::Other(
                "Need at least 2 CFs to find farthest pair".to_string(),
            ));
        }

        let mut max_distance = 0.0;
        let mut best_pair = (0, 1);

        for i in 0..self.cfs.len() {
            for j in i + 1..self.cfs.len() {
                let distance = self.cfs[i].distance_to(&self.cfs[j]);
                if distance > max_distance {
                    max_distance = distance;
                    best_pair = (i, j);
                }
            }
        }

        Ok(best_pair)
    }

    /// Collect all leaf CFs from this subtree
    pub fn collect_leaf_cfs(&self) -> Vec<ClusteringFeature> {
        if self.is_leaf {
            self.cfs.clone()
        } else {
            let mut leaf_cfs = Vec::new();
            if let Some(ref children) = self.children {
                for child in children {
                    leaf_cfs.extend(child.collect_leaf_cfs());
                }
            }
            leaf_cfs
        }
    }
}

/// BIRCH clustering algorithm
#[derive(Debug, Clone)]
pub struct BIRCH<State = Untrained> {
    config: BIRCHConfig,
    state: PhantomData<State>,
    // Trained state fields
    root_: Option<CFNode>,
    labels_: Option<Array1<i32>>,
    cluster_centers_: Option<Array2<Float>>,
    n_features_: Option<usize>,
}

impl BIRCH<Untrained> {
    /// Create a new BIRCH model
    pub fn new() -> Self {
        Self {
            config: BIRCHConfig::default(),
            state: PhantomData,
            root_: None,
            labels_: None,
            cluster_centers_: None,
            n_features_: None,
        }
    }

    /// Set threshold
    pub fn threshold(mut self, threshold: Float) -> Self {
        self.config.threshold = threshold;
        self
    }

    /// Set branching factor
    pub fn branching_factor(mut self, branching_factor: usize) -> Self {
        self.config.branching_factor = branching_factor;
        self
    }

    /// Set number of clusters
    pub fn n_clusters(mut self, n_clusters: usize) -> Self {
        self.config.n_clusters = Some(n_clusters);
        self
    }

    /// Set memory limit
    pub fn memory_limit(mut self, memory_limit: usize) -> Self {
        self.config.memory_limit = Some(memory_limit);
        self
    }
}

impl Default for BIRCH<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for BIRCH<Untrained> {
    type Config = BIRCHConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, ()> for BIRCH<Untrained> {
    type Fitted = BIRCH<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput("Empty dataset".to_string()));
        }

        // Initialize CF-Tree with root node
        let mut root = CFNode::new_leaf(self.config.branching_factor);

        // Phase 1: Build CF-Tree by inserting all points
        for i in 0..n_samples {
            let point = x.row(i);
            if let Some(new_root) = root.insert(&point, self.config.threshold)? {
                // Root split, create new root
                let mut new_root_node = CFNode::new_internal(self.config.branching_factor);

                // Add both old and new nodes as children
                let old_root_cf =
                    root.cfs
                        .iter()
                        .fold(ClusteringFeature::empty(n_features), |mut acc, cf| {
                            acc.add(cf);
                            acc
                        });
                let new_node_cf = new_root.cfs.iter().fold(
                    ClusteringFeature::empty(n_features),
                    |mut acc, cf| {
                        acc.add(cf);
                        acc
                    },
                );

                new_root_node.cfs.push(old_root_cf);
                new_root_node.cfs.push(new_node_cf);
                new_root_node.children.as_mut().unwrap().push(root);
                new_root_node.children.as_mut().unwrap().push(new_root);

                root = new_root_node;
            }
        }

        // Phase 2: Collect leaf CFs
        let leaf_cfs = root.collect_leaf_cfs();

        // Phase 3: Apply final clustering if requested
        let (labels, cluster_centers) = if let Some(n_clusters) = self.config.n_clusters {
            self.final_clustering(&leaf_cfs, n_clusters, x)?
        } else {
            // Use leaf CFs as final clusters
            let labels = self.assign_points_to_cfs(x, &leaf_cfs)?;
            let centers = leaf_cfs.iter().map(|cf| cf.centroid()).collect::<Vec<_>>();
            let centers_array = Array2::from_shape_vec(
                (centers.len(), n_features),
                centers.into_iter().flatten().collect(),
            )
            .map_err(|e| SklearsError::Other(format!("Failed to create centers array: {}", e)))?;
            (labels, centers_array)
        };

        Ok(BIRCH {
            config: self.config,
            state: PhantomData,
            root_: Some(root),
            labels_: Some(labels),
            cluster_centers_: Some(cluster_centers),
            n_features_: Some(n_features),
        })
    }
}

impl BIRCH<Untrained> {
    /// Apply final clustering using K-means on CF centroids
    fn final_clustering(
        &self,
        leaf_cfs: &[ClusteringFeature],
        n_clusters: usize,
        original_data: &Array2<Float>,
    ) -> Result<(Array1<i32>, Array2<Float>)> {
        if leaf_cfs.len() <= n_clusters {
            // Fewer CFs than requested clusters, just use CF centroids
            let labels = self.assign_points_to_cfs(original_data, leaf_cfs)?;
            let centers = leaf_cfs.iter().map(|cf| cf.centroid()).collect::<Vec<_>>();
            let centers_array = Array2::from_shape_vec(
                (centers.len(), original_data.ncols()),
                centers.into_iter().flatten().collect(),
            )
            .map_err(|e| SklearsError::Other(format!("Failed to create centers array: {}", e)))?;
            return Ok((labels, centers_array));
        }

        // Simple K-means on CF centroids (simplified implementation)
        let cf_centroids: Vec<Array1<Float>> = leaf_cfs.iter().map(|cf| cf.centroid()).collect();

        // Initialize cluster centers randomly from CF centroids
        let mut centers = Vec::with_capacity(n_clusters);
        for i in 0..n_clusters.min(cf_centroids.len()) {
            centers.push(cf_centroids[i % cf_centroids.len()].clone());
        }

        // Simple K-means iterations
        for _ in 0..10 {
            // Assign CFs to clusters
            let mut cf_assignments = vec![0; cf_centroids.len()];
            for (i, centroid) in cf_centroids.iter().enumerate() {
                let mut min_dist = Float::INFINITY;
                let mut best_cluster = 0;

                for (j, center) in centers.iter().enumerate() {
                    let diff = centroid - center;
                    let dist = diff.dot(&diff).sqrt();
                    if dist < min_dist {
                        min_dist = dist;
                        best_cluster = j;
                    }
                }
                cf_assignments[i] = best_cluster;
            }

            // Update centers
            for k in 0..n_clusters {
                let cluster_cfs: Vec<_> = cf_assignments
                    .iter()
                    .enumerate()
                    .filter(|(_, &assignment)| assignment == k)
                    .map(|(i, _)| i)
                    .collect();

                if !cluster_cfs.is_empty() {
                    let mut sum = Array1::zeros(original_data.ncols());
                    for &i in &cluster_cfs {
                        sum = &sum + &cf_centroids[i];
                    }
                    centers[k] = &sum / (cluster_cfs.len() as Float);
                }
            }
        }

        // Assign original points to final clusters
        let labels = self.assign_points_to_centers(original_data, &centers)?;
        let centers_array = Array2::from_shape_vec(
            (centers.len(), original_data.ncols()),
            centers.into_iter().flatten().collect(),
        )
        .map_err(|e| SklearsError::Other(format!("Failed to create centers array: {}", e)))?;

        Ok((labels, centers_array))
    }

    /// Assign points to CFs
    fn assign_points_to_cfs(
        &self,
        x: &Array2<Float>,
        cfs: &[ClusteringFeature],
    ) -> Result<Array1<i32>> {
        let n_samples = x.nrows();
        let mut labels = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let point = x.row(i);
            let point_cf = ClusteringFeature::from_point(&point);

            let mut min_distance = Float::INFINITY;
            let mut best_cf = 0;

            for (j, cf) in cfs.iter().enumerate() {
                let distance = cf.distance_to(&point_cf);
                if distance < min_distance {
                    min_distance = distance;
                    best_cf = j;
                }
            }

            labels[i] = best_cf as i32;
        }

        Ok(labels)
    }

    /// Assign points to cluster centers
    fn assign_points_to_centers(
        &self,
        x: &Array2<Float>,
        centers: &[Array1<Float>],
    ) -> Result<Array1<i32>> {
        let n_samples = x.nrows();
        let mut labels = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let point = x.row(i);

            let mut min_distance = Float::INFINITY;
            let mut best_cluster = 0;

            for (j, center) in centers.iter().enumerate() {
                let diff = &point.to_owned() - center;
                let distance = diff.dot(&diff).sqrt();
                if distance < min_distance {
                    min_distance = distance;
                    best_cluster = j;
                }
            }

            labels[i] = best_cluster as i32;
        }

        Ok(labels)
    }
}

impl BIRCH<Trained> {
    /// Get cluster labels
    pub fn labels(&self) -> &Array1<i32> {
        self.labels_.as_ref().expect("Model is trained")
    }

    /// Get cluster centers
    pub fn cluster_centers(&self) -> &Array2<Float> {
        self.cluster_centers_.as_ref().expect("Model is trained")
    }

    /// Get number of clusters
    pub fn n_clusters(&self) -> usize {
        self.cluster_centers().nrows()
    }

    /// Get the CF-Tree root
    pub fn cf_tree(&self) -> &CFNode {
        self.root_.as_ref().expect("Model is trained")
    }
}

impl Predict<Array2<Float>, Array1<i32>> for BIRCH<Trained> {
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

        for i in 0..n_samples {
            let point = x.row(i);
            let mut min_distance = Float::INFINITY;
            let mut best_cluster = 0;

            for (j, center) in centers.axis_iter(Axis(0)).enumerate() {
                let diff = &point.to_owned() - &center.to_owned();
                let distance = diff.dot(&diff).sqrt();
                if distance < min_distance {
                    min_distance = distance;
                    best_cluster = j;
                }
            }

            labels[i] = best_cluster as i32;
        }

        Ok(labels)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_clustering_feature() {
        let point1 = array![1.0, 2.0];
        let point2 = array![3.0, 4.0];

        let mut cf1 = ClusteringFeature::from_point(&point1.view());
        let cf2 = ClusteringFeature::from_point(&point2.view());

        assert_eq!(cf1.n, 1);
        assert_eq!(cf1.centroid(), point1);

        cf1.add(&cf2);
        assert_eq!(cf1.n, 2);
        assert_eq!(cf1.centroid(), array![2.0, 3.0]);
    }

    #[test]
    fn test_birch_basic() {
        let data = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1],];

        let model = BIRCH::new()
            .threshold(1.0)
            .n_clusters(2)
            .fit(&data, &())
            .unwrap();

        let labels = model.labels();
        assert_eq!(labels.len(), 4);
        assert_eq!(model.n_clusters(), 2);

        // Points should be grouped into 2 clusters
        assert_eq!(labels[0], labels[1]); // First two points in same cluster
        assert_eq!(labels[2], labels[3]); // Last two points in same cluster
        assert_ne!(labels[0], labels[2]); // Different clusters
    }

    #[test]
    fn test_birch_predict() {
        let train_data = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1],];

        let model = BIRCH::new()
            .threshold(1.0)
            .n_clusters(2)
            .fit(&train_data, &())
            .unwrap();

        let test_data = array![
            [0.05, 0.05], // Should be close to first cluster
            [5.05, 5.05], // Should be close to second cluster
        ];

        let predictions = model.predict(&test_data).unwrap();
        assert_eq!(predictions.len(), 2);
        assert_ne!(predictions[0], predictions[1]); // Should predict different clusters
    }
}
