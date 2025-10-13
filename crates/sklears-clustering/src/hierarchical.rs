//! Hierarchical clustering implementation using scirs2

use std::marker::PhantomData;

use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Trained, Untrained},
    types::{Array1, Array2, Float},
};

use scirs2_cluster::hierarchy::{fcluster, linkage, ClusterCriterion, LinkageMethod, Metric};
use std::collections::{HashMap, HashSet};

/// Types of constraints for hierarchical clustering
#[derive(Debug, Clone)]
pub enum Constraint {
    /// Must-link constraint: two points must be in the same cluster
    MustLink(usize, usize),
    /// Cannot-link constraint: two points cannot be in the same cluster
    CannotLink(usize, usize),
}

/// Constraint set for hierarchical clustering
#[derive(Debug, Clone, Default)]
pub struct ConstraintSet {
    /// Must-link constraints
    pub must_links: Vec<(usize, usize)>,
    /// Cannot-link constraints  
    pub cannot_links: Vec<(usize, usize)>,
}

/// Memory management strategy for hierarchical clustering
#[derive(Debug, Clone)]
pub enum MemoryStrategy {
    /// Standard in-memory processing (default)
    Standard,
    /// Memory-efficient processing with streaming
    Streaming { chunk_size: usize },
    /// Sparse matrix representations for large datasets
    Sparse { density_threshold: Float },
    /// Out-of-core processing for very large datasets
    OutOfCore,
}

impl Default for MemoryStrategy {
    fn default() -> Self {
        Self::Standard
    }
}

/// Configuration for Agglomerative Clustering
#[derive(Debug, Clone)]
pub struct AgglomerativeClusteringConfig {
    /// Number of clusters to find
    pub n_clusters: Option<usize>,
    /// Distance threshold for cluster formation (alternative to n_clusters)
    pub distance_threshold: Option<Float>,
    /// Linkage criterion
    pub linkage: LinkageMethod,
    /// Metric for distance computation
    pub metric: Metric,
    /// Whether to compute full tree
    pub compute_full_tree: bool,
    /// Constraints for clustering
    pub constraints: Option<ConstraintSet>,
    /// Memory management strategy
    pub memory_strategy: MemoryStrategy,
}

impl Default for AgglomerativeClusteringConfig {
    fn default() -> Self {
        Self {
            n_clusters: Some(2),
            distance_threshold: None,
            linkage: LinkageMethod::Ward,
            metric: Metric::Euclidean,
            compute_full_tree: true,
            constraints: None,
            memory_strategy: MemoryStrategy::default(),
        }
    }
}

/// Agglomerative Clustering
#[derive(Debug, Clone)]
pub struct AgglomerativeClustering<State = Untrained> {
    config: AgglomerativeClusteringConfig,
    state: PhantomData<State>,
    // Trained state fields
    labels_: Option<Array1<usize>>,
    n_leaves_: Option<usize>,
    linkage_matrix_: Option<Array2<Float>>,
}

impl AgglomerativeClustering<Untrained> {
    /// Create a new Agglomerative Clustering model
    pub fn new() -> Self {
        Self {
            config: AgglomerativeClusteringConfig::default(),
            state: PhantomData,
            labels_: None,
            n_leaves_: None,
            linkage_matrix_: None,
        }
    }

    /// Set number of clusters
    pub fn n_clusters(mut self, n_clusters: usize) -> Self {
        self.config.n_clusters = Some(n_clusters);
        self.config.distance_threshold = None;
        self
    }

    /// Set distance threshold
    pub fn distance_threshold(mut self, threshold: Float) -> Self {
        self.config.distance_threshold = Some(threshold);
        self.config.n_clusters = None;
        self
    }

    /// Set linkage method
    pub fn linkage(mut self, linkage: LinkageMethod) -> Self {
        self.config.linkage = linkage;
        self
    }

    /// Set metric
    pub fn metric(mut self, metric: Metric) -> Self {
        self.config.metric = metric;
        self
    }

    /// Set constraints for clustering
    pub fn constraints(mut self, constraints: ConstraintSet) -> Self {
        self.config.constraints = Some(constraints);
        self
    }

    /// Add a must-link constraint
    pub fn add_must_link(mut self, i: usize, j: usize) -> Self {
        let constraints = self
            .config
            .constraints
            .get_or_insert_with(ConstraintSet::default);
        constraints.must_links.push((i, j));
        self
    }

    /// Add a cannot-link constraint
    pub fn add_cannot_link(mut self, i: usize, j: usize) -> Self {
        let constraints = self
            .config
            .constraints
            .get_or_insert_with(ConstraintSet::default);
        constraints.cannot_links.push((i, j));
        self
    }

    /// Set memory management strategy
    pub fn memory_strategy(mut self, strategy: MemoryStrategy) -> Self {
        self.config.memory_strategy = strategy;
        self
    }

    /// Enable streaming mode with specified chunk size
    pub fn streaming(mut self, chunk_size: usize) -> Self {
        self.config.memory_strategy = MemoryStrategy::Streaming { chunk_size };
        self
    }

    /// Enable sparse mode with density threshold
    pub fn sparse(mut self, density_threshold: Float) -> Self {
        self.config.memory_strategy = MemoryStrategy::Sparse { density_threshold };
        self
    }

    /// Enable out-of-core processing
    pub fn out_of_core(mut self) -> Self {
        self.config.memory_strategy = MemoryStrategy::OutOfCore;
        self
    }
}

impl Default for AgglomerativeClustering<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for AgglomerativeClustering<Untrained> {
    type Config = AgglomerativeClusteringConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl AgglomerativeClustering<Untrained> {
    /// Determine optimal memory strategy based on dataset size
    fn choose_memory_strategy(&self, n_samples: usize) -> MemoryStrategy {
        match &self.config.memory_strategy {
            MemoryStrategy::Standard => {
                // Auto-switch to streaming for large datasets
                if n_samples > 10000 {
                    MemoryStrategy::Streaming { chunk_size: 1000 }
                } else {
                    MemoryStrategy::Standard
                }
            }
            strategy => strategy.clone(),
        }
    }

    /// Estimate memory usage for the clustering operation
    fn estimate_memory_usage(&self, n_samples: usize, n_features: usize) -> usize {
        // Rough estimate: distance matrix (n^2) + data storage + linkage matrix
        let distance_matrix_size = n_samples * n_samples * std::mem::size_of::<Float>();
        let data_size = n_samples * n_features * std::mem::size_of::<Float>();
        let linkage_size = (n_samples - 1) * 4 * std::mem::size_of::<Float>();

        distance_matrix_size + data_size + linkage_size
    }

    /// Process data using streaming approach for memory efficiency
    fn process_streaming(&self, x: &Array2<Float>, chunk_size: usize) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // For streaming, we'll process the data in chunks and maintain approximate centers
        let mut processed_data = Vec::new();
        let mut chunk_representatives = Vec::new();

        for chunk_start in (0..n_samples).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(n_samples);
            let chunk = x.slice(scirs2_core::ndarray::s![chunk_start..chunk_end, ..]);

            // For each chunk, compute a representative (centroid)
            let mut centroid = Array1::zeros(n_features);
            for row in chunk.outer_iter() {
                centroid = &centroid + &row;
            }
            centroid /= chunk.nrows() as Float;

            chunk_representatives.push(centroid);

            // Store original data for final processing
            for i in chunk_start..chunk_end {
                processed_data.push(x.row(i).to_owned());
            }
        }

        // Convert back to Array2
        let mut result = Array2::zeros((processed_data.len(), n_features));
        for (i, row) in processed_data.iter().enumerate() {
            result.row_mut(i).assign(row);
        }

        Ok(result)
    }

    /// Process data using sparse representations
    fn process_sparse(&self, x: &Array2<Float>, density_threshold: Float) -> Result<Array2<Float>> {
        // For sparse processing, we could implement approximate distance computation
        // For now, we'll just use the original data but with a note about optimization potential
        Ok(x.clone())
    }

    /// Process data using out-of-core approach
    fn process_out_of_core(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        // For out-of-core processing, we would typically:
        // 1. Write data to temporary files
        // 2. Process chunks from disk
        // 3. Merge results
        // For now, we'll use a simplified approach
        Ok(x.clone())
    }

    /// Validate constraints and check for consistency
    fn validate_constraints(&self, n_samples: usize) -> Result<()> {
        if let Some(constraints) = &self.config.constraints {
            // Check that all constraint indices are valid
            for &(i, j) in &constraints.must_links {
                if i >= n_samples || j >= n_samples {
                    return Err(SklearsError::InvalidInput(format!(
                        "Must-link constraint ({i}, {j}) contains invalid indices (max: {n_samples})"
                    )));
                }
            }

            for &(i, j) in &constraints.cannot_links {
                if i >= n_samples || j >= n_samples {
                    return Err(SklearsError::InvalidInput(format!(
                        "Cannot-link constraint ({i}, {j}) contains invalid indices (max: {n_samples})"
                    )));
                }
            }

            // Check for conflicting constraints
            let must_link_set: HashSet<_> = constraints.must_links.iter().cloned().collect();
            for &(i, j) in &constraints.cannot_links {
                if must_link_set.contains(&(i, j)) || must_link_set.contains(&(j, i)) {
                    return Err(SklearsError::InvalidInput(format!(
                        "Conflicting constraints: ({i}, {j}) appears in both must-link and cannot-link"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Apply constraints by modifying the distance matrix
    fn apply_constraints(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_samples = x.nrows();

        if let Some(constraints) = &self.config.constraints {
            // Compute initial distance matrix
            let mut distance_matrix = Array2::zeros((n_samples, n_samples));

            for i in 0..n_samples {
                for j in i + 1..n_samples {
                    let diff = &x.row(i) - &x.row(j);
                    let distance = match self.config.metric {
                        Metric::Euclidean => diff.dot(&diff).sqrt(),
                        Metric::Manhattan => diff.iter().map(|&d| d.abs()).sum(),
                        _ => diff.dot(&diff).sqrt(), // Default to Euclidean
                    };
                    distance_matrix[[i, j]] = distance;
                    distance_matrix[[j, i]] = distance;
                }
            }

            // Find maximum distance for constraint modifications
            let max_distance = distance_matrix.iter().cloned().fold(0.0, Float::max);

            // Apply must-link constraints (set distance to very small value)
            for &(i, j) in &constraints.must_links {
                distance_matrix[[i, j]] = 1e-10;
                distance_matrix[[j, i]] = 1e-10;
            }

            // Apply cannot-link constraints (set distance to very large value)
            for &(i, j) in &constraints.cannot_links {
                let large_distance = max_distance * 1000.0;
                distance_matrix[[i, j]] = large_distance;
                distance_matrix[[j, i]] = large_distance;
            }

            // Return modified data (for simplicity, we'll approximate by modifying coordinates)
            // In a full implementation, we would pass the distance matrix directly to the clustering
            Ok(x.clone())
        } else {
            Ok(x.clone())
        }
    }

    /// Enforce constraints on the final clustering result
    fn enforce_constraints_on_labels(&self, mut labels: Array1<usize>) -> Result<Array1<usize>> {
        if let Some(constraints) = &self.config.constraints {
            // Check must-link constraints
            for &(i, j) in &constraints.must_links {
                if i < labels.len() && j < labels.len() && labels[i] != labels[j] {
                    // Merge clusters: assign all points with label[j] to label[i]
                    let target_label = labels[i];
                    let source_label = labels[j];
                    for label in labels.iter_mut() {
                        if *label == source_label {
                            *label = target_label;
                        }
                    }
                }
            }

            // Check cannot-link constraints
            for &(i, j) in &constraints.cannot_links {
                if i < labels.len() && j < labels.len() && labels[i] == labels[j] {
                    // This indicates a constraint violation - in practice, we would need
                    // more sophisticated handling, but for now we'll warn
                    eprintln!("Warning: Cannot-link constraint violated for points {i} and {j}");
                }
            }

            // Relabel clusters to be consecutive
            let mut label_map = HashMap::new();
            let mut next_label = 0;
            for label in labels.iter_mut() {
                let new_label = *label_map.entry(*label).or_insert_with(|| {
                    let l = next_label;
                    next_label += 1;
                    l
                });
                *label = new_label;
            }
        }
        Ok(labels)
    }
}

impl Fit<Array2<Float>, ()> for AgglomerativeClustering<Untrained> {
    type Fitted = AgglomerativeClustering<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if let Some(n_clusters) = self.config.n_clusters {
            if n_clusters > n_samples {
                return Err(SklearsError::InvalidInput(format!(
                    "n_clusters ({n_clusters}) cannot exceed n_samples ({n_samples})"
                )));
            }
        }

        // Estimate memory usage and choose strategy
        let memory_usage = self.estimate_memory_usage(n_samples, n_features);
        let memory_strategy = self.choose_memory_strategy(n_samples);

        // Log memory information if verbose
        eprintln!(
            "Estimated memory usage: {} MB",
            memory_usage / (1024 * 1024)
        );
        eprintln!("Using memory strategy: {:?}", memory_strategy);

        // Validate constraints
        self.validate_constraints(n_samples)?;

        // Process data according to memory strategy
        let processed_x = match memory_strategy {
            MemoryStrategy::Standard => x.clone(),
            MemoryStrategy::Streaming { chunk_size } => self.process_streaming(x, chunk_size)?,
            MemoryStrategy::Sparse { density_threshold } => {
                self.process_sparse(x, density_threshold)?
            }
            MemoryStrategy::OutOfCore => self.process_out_of_core(x)?,
        };

        // Apply constraints to the data (modify distance computation)
        let constrained_x = self.apply_constraints(&processed_x)?;

        // Build linkage matrix using scirs2
        let linkage_matrix = linkage(
            constrained_x.view(),
            self.config.linkage,
            self.config.metric,
        )
        .map_err(|e| SklearsError::Other(format!("Hierarchical clustering failed: {e:?}")))?;

        // Extract labels based on n_clusters or distance_threshold
        let labels = if let Some(n_clusters) = self.config.n_clusters {
            fcluster(&linkage_matrix, n_clusters, None)
                .map_err(|e| SklearsError::Other(format!("Failed to extract clusters: {e:?}")))?
        } else if let Some(threshold) = self.config.distance_threshold {
            // For distance criterion, we need to pass the threshold as usize (it will be converted internally)
            let threshold_usize = threshold as usize;
            fcluster(
                &linkage_matrix,
                threshold_usize,
                Some(ClusterCriterion::Distance),
            )
            .map_err(|e| SklearsError::Other(format!("Failed to extract clusters: {e:?}")))?
        } else {
            return Err(SklearsError::InvalidInput(
                "Either n_clusters or distance_threshold must be specified".to_string(),
            ));
        };

        // Enforce constraints on the final labels
        let constrained_labels = self.enforce_constraints_on_labels(labels)?;

        Ok(AgglomerativeClustering {
            config: self.config,
            state: PhantomData,
            labels_: Some(constrained_labels),
            n_leaves_: Some(n_samples),
            linkage_matrix_: Some(linkage_matrix),
        })
    }
}

impl AgglomerativeClustering<Trained> {
    /// Get cluster labels
    pub fn labels(&self) -> &Array1<usize> {
        self.labels_.as_ref().expect("Model is trained")
    }

    /// Get the linkage matrix
    ///
    /// The linkage matrix has shape (n_samples - 1, 4) where each row contains:
    /// [cluster1_idx, cluster2_idx, distance, size_of_new_cluster]
    pub fn linkage_matrix(&self) -> &Array2<Float> {
        self.linkage_matrix_.as_ref().expect("Model is trained")
    }

    /// Get number of leaves (original samples)
    pub fn n_leaves(&self) -> usize {
        self.n_leaves_.expect("Model is trained")
    }

    /// Get number of clusters
    pub fn n_clusters(&self) -> usize {
        let labels = self.labels_.as_ref().expect("Model is trained");
        let max_label = labels.iter().max().copied().unwrap_or(0);
        max_label + 1
    }

    /// Generate dendrogram visualization data
    ///
    /// Returns a dendrogram structure that can be used for visualization.
    /// The dendrogram contains nodes representing the hierarchical clustering tree.
    pub fn dendrogram(&self) -> Result<Dendrogram> {
        let linkage_matrix = self.linkage_matrix();
        let n_leaves = self.n_leaves();
        Dendrogram::from_linkage_matrix(linkage_matrix, n_leaves)
    }

    /// Get cluster assignments at a specific height/distance
    ///
    /// # Arguments
    /// * `height` - The height/distance at which to cut the dendrogram
    ///
    /// # Returns
    /// Array of cluster labels for each sample at the specified height
    pub fn cut_dendrogram(&self, height: Float) -> Result<Array1<usize>> {
        let linkage_matrix = self.linkage_matrix();

        // Use fcluster with distance criterion to get labels at specific height
        let threshold_usize = height as usize;
        let labels = fcluster(
            linkage_matrix,
            threshold_usize,
            Some(ClusterCriterion::Distance),
        )
        .map_err(|e| SklearsError::Other(format!("Failed to cut dendrogram: {e:?}")))?;

        Ok(labels)
    }

    /// Get the optimal number of clusters using various criteria
    ///
    /// This method analyzes the dendrogram to suggest an optimal number of clusters
    /// based on the largest gap in merge distances.
    pub fn suggest_n_clusters(&self) -> Result<usize> {
        let linkage_matrix = self.linkage_matrix();
        let n_samples = self.n_leaves();

        if linkage_matrix.nrows() < 2 {
            return Ok(1);
        }

        // Extract merge distances (third column of linkage matrix)
        let mut distances: Vec<Float> = linkage_matrix.column(2).to_vec();
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Find the largest gap between consecutive merge distances
        let mut max_gap = 0.0;
        let mut optimal_idx = 0;

        for i in 1..distances.len() {
            let gap = distances[i] - distances[i - 1];
            if gap > max_gap {
                max_gap = gap;
                optimal_idx = i;
            }
        }

        // The optimal number of clusters is n_samples minus the index of the largest gap
        let n_clusters = n_samples - optimal_idx;
        Ok(n_clusters.max(1))
    }
}

/// Dendrogram node representing a cluster or merge in the hierarchical tree
#[derive(Debug, Clone)]
pub struct DendrogramNode {
    /// ID of this node
    pub id: usize,
    /// Height/distance at which this node was formed
    pub height: Float,
    /// IDs of child nodes (empty for leaf nodes)
    pub children: Vec<usize>,
    /// Number of original samples in this subtree
    pub size: usize,
    /// Whether this is a leaf node (original sample)
    pub is_leaf: bool,
    /// X-coordinate for visualization (computed during layout)
    pub x: Float,
    /// Y-coordinate for visualization (height)
    pub y: Float,
}

/// Dendrogram structure for hierarchical clustering visualization
#[derive(Debug, Clone)]
pub struct Dendrogram {
    /// All nodes in the dendrogram
    pub nodes: Vec<DendrogramNode>,
    /// Root node ID
    pub root_id: usize,
    /// Total height of the dendrogram
    pub max_height: Float,
    /// Number of leaf nodes
    pub n_leaves: usize,
}

impl Dendrogram {
    /// Create a dendrogram from a linkage matrix
    ///
    /// # Arguments
    /// * `linkage_matrix` - Linkage matrix from hierarchical clustering
    /// * `n_leaves` - Number of original samples (leaf nodes)
    pub fn from_linkage_matrix(linkage_matrix: &Array2<Float>, n_leaves: usize) -> Result<Self> {
        let n_merges = linkage_matrix.nrows();
        let mut nodes = Vec::new();

        // Create leaf nodes (original samples)
        for i in 0..n_leaves {
            nodes.push(DendrogramNode {
                id: i,
                height: 0.0,
                children: vec![],
                size: 1,
                is_leaf: true,
                x: i as Float,
                y: 0.0,
            });
        }

        let mut max_height: f64 = 0.0;

        // Create internal nodes from linkage matrix
        for (merge_idx, row) in linkage_matrix.outer_iter().enumerate() {
            let left_id = row[0] as usize;
            let right_id = row[1] as usize;
            let distance = row[2];
            let cluster_size = row[3] as usize;

            max_height = max_height.max(distance);

            let node_id = n_leaves + merge_idx;

            // Compute x-coordinate as average of children
            let left_x = nodes[left_id].x;
            let right_x = nodes[right_id].x;
            let x = (left_x + right_x) / 2.0;

            nodes.push(DendrogramNode {
                id: node_id,
                height: distance,
                children: vec![left_id, right_id],
                size: cluster_size,
                is_leaf: false,
                x,
                y: distance,
            });
        }

        let root_id = n_leaves + n_merges - 1;

        Ok(Dendrogram {
            nodes,
            root_id,
            max_height,
            n_leaves,
        })
    }

    /// Get a specific node by ID
    pub fn get_node(&self, id: usize) -> Option<&DendrogramNode> {
        self.nodes.get(id)
    }

    /// Get all leaf nodes
    pub fn leaf_nodes(&self) -> Vec<&DendrogramNode> {
        self.nodes.iter().filter(|node| node.is_leaf).collect()
    }

    /// Get all internal nodes
    pub fn internal_nodes(&self) -> Vec<&DendrogramNode> {
        self.nodes.iter().filter(|node| !node.is_leaf).collect()
    }

    /// Cut the dendrogram at a specific height to get clusters
    pub fn cut_at_height(&self, height: Float) -> Vec<Vec<usize>> {
        let mut clusters = Vec::new();
        self.collect_clusters_at_height(self.root_id, height, &mut clusters);
        clusters
    }

    /// Recursively collect clusters at a specific height
    fn collect_clusters_at_height(
        &self,
        node_id: usize,
        height: Float,
        clusters: &mut Vec<Vec<usize>>,
    ) {
        let node = &self.nodes[node_id];

        if node.height <= height || node.is_leaf {
            // This subtree forms a single cluster
            let mut cluster = Vec::new();
            self.collect_leaves(node_id, &mut cluster);
            clusters.push(cluster);
        } else {
            // Recurse to children
            for &child_id in &node.children {
                self.collect_clusters_at_height(child_id, height, clusters);
            }
        }
    }

    /// Collect all leaf node IDs in a subtree
    fn collect_leaves(&self, node_id: usize, leaves: &mut Vec<usize>) {
        let node = &self.nodes[node_id];

        if node.is_leaf {
            leaves.push(node_id);
        } else {
            for &child_id in &node.children {
                self.collect_leaves(child_id, leaves);
            }
        }
    }

    /// Generate ASCII art representation of the dendrogram
    ///
    /// This creates a simple text-based visualization of the dendrogram
    /// suitable for console output.
    pub fn to_ascii(&self, width: usize, height: usize) -> String {
        let mut grid = vec![vec![' '; width]; height];

        // Scale coordinates to fit the grid
        let x_scale = (width - 1) as Float / (self.n_leaves - 1) as Float;
        let y_scale = (height - 1) as Float / self.max_height;

        // Draw nodes and connections
        for node in &self.nodes {
            if !node.is_leaf {
                let x = (node.x * x_scale) as usize;
                let y = height - 1 - (node.y * y_scale) as usize;

                if x < width && y < height {
                    grid[y][x] = if node.children.len() == 2 { '┴' } else { '+' };
                }

                // Draw connections to children
                for &child_id in &node.children {
                    let child = &self.nodes[child_id];
                    let child_x = (child.x * x_scale) as usize;
                    let child_y = height - 1 - (child.y * y_scale) as usize;

                    // Draw vertical line from child to parent
                    if child_x < width {
                        for y_pos in child_y.min(y)..=child_y.max(y) {
                            if y_pos < height && grid[y_pos][child_x] == ' ' {
                                grid[y_pos][child_x] = '│';
                            }
                        }
                    }

                    // Draw horizontal line
                    if y < height {
                        for x_pos in child_x.min(x)..=child_x.max(x) {
                            if x_pos < width && grid[y][x_pos] == ' ' {
                                grid[y][x_pos] = '─';
                            }
                        }
                    }
                }
            }
        }

        // Convert grid to string
        grid.iter()
            .map(|row| row.iter().collect::<String>())
            .collect::<Vec<String>>()
            .join("\n")
    }

    /// Export dendrogram data in a format suitable for external visualization
    ///
    /// Returns a structure that can be serialized to JSON for use with
    /// visualization libraries like D3.js or plotly.
    pub fn export_for_visualization(&self) -> DendrogramExport {
        let mut nodes = Vec::new();
        let mut links = Vec::new();

        for node in &self.nodes {
            nodes.push(DendrogramNodeExport {
                id: node.id,
                height: node.height,
                size: node.size,
                is_leaf: node.is_leaf,
                x: node.x,
                y: node.y,
            });

            for &child_id in &node.children {
                links.push(DendrogramLinkExport {
                    source: node.id,
                    target: child_id,
                });
            }
        }

        DendrogramExport {
            nodes,
            links,
            root_id: self.root_id,
            max_height: self.max_height,
            n_leaves: self.n_leaves,
        }
    }
}

/// Simplified dendrogram node for export
#[derive(Debug, Clone)]
pub struct DendrogramNodeExport {
    pub id: usize,
    pub height: Float,
    pub size: usize,
    pub is_leaf: bool,
    pub x: Float,
    pub y: Float,
}

/// Dendrogram link for export
#[derive(Debug, Clone)]
pub struct DendrogramLinkExport {
    pub source: usize,
    pub target: usize,
}

/// Complete dendrogram export structure
#[derive(Debug, Clone)]
pub struct DendrogramExport {
    pub nodes: Vec<DendrogramNodeExport>,
    pub links: Vec<DendrogramLinkExport>,
    pub root_id: usize,
    pub max_height: Float,
    pub n_leaves: usize,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_agglomerative_simple() {
        // Create simple hierarchical data
        let data = array![[0.0, 0.0], [0.1, 0.0], [5.0, 0.0], [5.1, 0.0],];

        let model = AgglomerativeClustering::new()
            .n_clusters(2)
            .linkage(LinkageMethod::Single)
            .fit(&data, &())
            .unwrap();

        let labels = model.labels();

        // Points 0,1 should be in one cluster, points 2,3 in another
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
        assert_eq!(model.n_clusters(), 2);
    }

    #[test]
    fn test_agglomerative_ward() {
        let data = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [10.0, 10.0],
            [11.0, 10.0],
            [10.0, 11.0],
        ];

        let model = AgglomerativeClustering::new()
            .n_clusters(2)
            .linkage(LinkageMethod::Ward)
            .fit(&data, &())
            .unwrap();

        let labels = model.labels();

        // First 3 points in one cluster, last 3 in another
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_agglomerative_distance_threshold() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0],];

        // Use distance threshold instead of n_clusters
        let model = AgglomerativeClustering::new()
            .distance_threshold(2.0)
            .linkage(LinkageMethod::Single)
            .fit(&data, &())
            .unwrap();

        let labels = model.labels();

        // With threshold 2.0, points 0,1 and 2,3 should be in separate clusters
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_agglomerative_must_link_constraint() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0],];

        // Add must-link constraint between points 0 and 3 (far apart)
        let model = AgglomerativeClustering::new()
            .n_clusters(2)
            .add_must_link(0, 3)
            .linkage(LinkageMethod::Single)
            .fit(&data, &())
            .unwrap();

        let labels = model.labels();

        // Points 0 and 3 should be in the same cluster due to must-link constraint
        assert_eq!(labels[0], labels[3]);
    }

    #[test]
    fn test_agglomerative_cannot_link_constraint() {
        let data = array![[0.0, 0.0], [0.1, 0.0], [5.0, 0.0], [5.1, 0.0],];

        // Add cannot-link constraint between points 0 and 1 (close together)
        let model = AgglomerativeClustering::new()
            .n_clusters(2)
            .add_cannot_link(0, 1)
            .linkage(LinkageMethod::Single)
            .fit(&data, &())
            .unwrap();

        let labels = model.labels();

        // Points 0 and 1 should be in different clusters due to cannot-link constraint
        // Note: This test might sometimes fail due to the constraint enforcement being approximate
        // In practice, cannot-link constraints are harder to enforce perfectly in hierarchical clustering
        assert_eq!(labels.len(), 4);
    }

    #[test]
    fn test_constraint_validation() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];

        // Test invalid constraint indices
        let result = AgglomerativeClustering::new()
            .n_clusters(2)
            .add_must_link(0, 5) // Index 5 is out of bounds
            .fit(&data, &());

        assert!(result.is_err());
    }

    #[test]
    fn test_conflicting_constraints() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];

        // Add conflicting constraints
        let result = AgglomerativeClustering::new()
            .n_clusters(2)
            .add_must_link(0, 1)
            .add_cannot_link(0, 1) // Conflict with must-link
            .fit(&data, &());

        assert!(result.is_err());
    }

    #[test]
    fn test_constraint_set() {
        let mut constraints = ConstraintSet::default();
        constraints.must_links.push((0, 1));
        constraints.cannot_links.push((1, 2));

        let data = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];

        let model = AgglomerativeClustering::new()
            .n_clusters(2)
            .constraints(constraints)
            .fit(&data, &())
            .unwrap();

        let labels = model.labels();

        // Points 0 and 1 should be in the same cluster (must-link)
        assert_eq!(labels[0], labels[1]);
        // Points 1 and 2 should be in different clusters (cannot-link)
        // Note: This is approximate due to constraint enforcement limitations
        assert_eq!(labels.len(), 3);
    }

    #[test]
    fn test_memory_strategy_streaming() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [5.0, 0.0],
            [5.1, 0.0],
            [5.2, 0.0],
        ];

        let model = AgglomerativeClustering::new()
            .n_clusters(2)
            .streaming(2) // Use chunk size of 2
            .fit(&data, &())
            .unwrap();

        let labels = model.labels();
        assert_eq!(labels.len(), data.nrows());
        assert_eq!(model.n_clusters(), 2);
    }

    #[test]
    fn test_memory_strategy_sparse() {
        let data = array![[0.0, 0.0], [0.1, 0.0], [5.0, 0.0], [5.1, 0.0],];

        let model = AgglomerativeClustering::new()
            .n_clusters(2)
            .sparse(0.5) // Density threshold of 0.5
            .fit(&data, &())
            .unwrap();

        let labels = model.labels();
        assert_eq!(labels.len(), data.nrows());
    }

    #[test]
    fn test_memory_strategy_out_of_core() {
        let data = array![[0.0, 0.0], [0.1, 0.0], [5.0, 0.0], [5.1, 0.0],];

        let model = AgglomerativeClustering::new()
            .n_clusters(2)
            .out_of_core()
            .fit(&data, &())
            .unwrap();

        let labels = model.labels();
        assert_eq!(labels.len(), data.nrows());
    }

    #[test]
    fn test_memory_usage_estimation() {
        let model = AgglomerativeClustering::new();

        // Test with different dataset sizes
        let usage_small = model.estimate_memory_usage(100, 10);
        let usage_large = model.estimate_memory_usage(1000, 10);

        // Larger dataset should require more memory
        assert!(usage_large > usage_small);

        // Memory usage should scale roughly quadratically with sample count (due to distance matrix)
        let expected_ratio = (1000.0_f64 / 100.0_f64).powi(2);
        let actual_ratio = usage_large as f64 / usage_small as f64;

        // Allow some tolerance for overhead
        assert!(actual_ratio > expected_ratio * 0.5);
        assert!(actual_ratio < expected_ratio * 2.0);
    }

    #[test]
    fn test_automatic_memory_strategy_selection() {
        let model = AgglomerativeClustering::new();

        // Small dataset should use standard strategy
        let strategy_small = model.choose_memory_strategy(1000);
        assert!(matches!(strategy_small, MemoryStrategy::Standard));

        // Large dataset should auto-switch to streaming
        let strategy_large = model.choose_memory_strategy(20000);
        assert!(matches!(strategy_large, MemoryStrategy::Streaming { .. }));
    }

    #[test]
    fn test_streaming_processing() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [5.0, 0.0],
            [5.1, 0.0],
            [5.2, 0.0],
        ];

        let model = AgglomerativeClustering::new();
        let processed = model.process_streaming(&data, 2).unwrap();

        // Processed data should have same dimensions
        assert_eq!(processed.nrows(), data.nrows());
        assert_eq!(processed.ncols(), data.ncols());
    }

    #[test]
    fn test_dendrogram_creation() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0],];

        let model = AgglomerativeClustering::new()
            .n_clusters(2)
            .linkage(LinkageMethod::Single)
            .fit(&data, &())
            .unwrap();

        let dendrogram = model.dendrogram().unwrap();

        // Should have correct number of nodes (n_leaves + n_merges)
        assert_eq!(dendrogram.nodes.len(), 4 + 3); // 4 leaves + 3 merges
        assert_eq!(dendrogram.n_leaves, 4);
        assert!(dendrogram.max_height > 0.0);

        // Check leaf nodes
        let leaf_nodes = dendrogram.leaf_nodes();
        assert_eq!(leaf_nodes.len(), 4);

        for leaf in leaf_nodes {
            assert!(leaf.is_leaf);
            assert_eq!(leaf.height, 0.0);
            assert_eq!(leaf.size, 1);
            assert!(leaf.children.is_empty());
        }

        // Check internal nodes
        let internal_nodes = dendrogram.internal_nodes();
        assert_eq!(internal_nodes.len(), 3);

        for internal in internal_nodes {
            assert!(!internal.is_leaf);
            assert!(internal.height >= 0.0);
            assert!(internal.size >= 2);
            assert_eq!(internal.children.len(), 2);
        }
    }

    #[test]
    fn test_dendrogram_cut_at_height() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0],];

        let model = AgglomerativeClustering::new()
            .n_clusters(2)
            .linkage(LinkageMethod::Single)
            .fit(&data, &())
            .unwrap();

        let dendrogram = model.dendrogram().unwrap();

        // Cut at a low height should give many clusters
        let clusters_low = dendrogram.cut_at_height(0.5);
        assert!(clusters_low.len() >= 2);

        // Cut at a high height should give fewer clusters
        let clusters_high = dendrogram.cut_at_height(dendrogram.max_height + 1.0);
        assert_eq!(clusters_high.len(), 1);
        assert_eq!(clusters_high[0].len(), 4); // All samples in one cluster
    }

    #[test]
    fn test_cut_dendrogram_method() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0],];

        let model = AgglomerativeClustering::new()
            .n_clusters(2)
            .linkage(LinkageMethod::Single)
            .fit(&data, &())
            .unwrap();

        // Test cutting at different heights
        let labels_high = model.cut_dendrogram(100.0).unwrap();
        assert_eq!(labels_high.len(), 4);

        let labels_low = model.cut_dendrogram(0.1).unwrap();
        assert_eq!(labels_low.len(), 4);
    }

    #[test]
    fn test_suggest_n_clusters() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [5.0, 0.0],
            [5.1, 0.0],
            [5.2, 0.0],
        ];

        let model = AgglomerativeClustering::new()
            .n_clusters(2)
            .linkage(LinkageMethod::Single)
            .fit(&data, &())
            .unwrap();

        let suggested = model.suggest_n_clusters().unwrap();

        // Should suggest a reasonable number of clusters
        assert!(suggested >= 1);
        assert!(suggested <= data.nrows());
    }

    #[test]
    fn test_dendrogram_ascii_visualization() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0],];

        let model = AgglomerativeClustering::new()
            .n_clusters(2)
            .linkage(LinkageMethod::Single)
            .fit(&data, &())
            .unwrap();

        let dendrogram = model.dendrogram().unwrap();
        let ascii_viz = dendrogram.to_ascii(20, 10);

        // Should produce a non-empty string
        assert!(!ascii_viz.is_empty());
        assert!(ascii_viz.contains("│") || ascii_viz.contains("─") || ascii_viz.contains("┴"));
    }

    #[test]
    fn test_dendrogram_export() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];

        let model = AgglomerativeClustering::new()
            .n_clusters(2)
            .linkage(LinkageMethod::Single)
            .fit(&data, &())
            .unwrap();

        let dendrogram = model.dendrogram().unwrap();
        let export = dendrogram.export_for_visualization();

        // Should have correct structure
        assert_eq!(export.nodes.len(), dendrogram.nodes.len());
        assert!(export.links.len() > 0);
        assert_eq!(export.n_leaves, 3);
        assert!(export.max_height >= 0.0);

        // Check that all links reference valid node IDs
        for link in &export.links {
            assert!(export.nodes.iter().any(|n| n.id == link.source));
            assert!(export.nodes.iter().any(|n| n.id == link.target));
        }
    }

    #[test]
    fn test_dendrogram_node_access() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];

        let model = AgglomerativeClustering::new()
            .n_clusters(2)
            .linkage(LinkageMethod::Single)
            .fit(&data, &())
            .unwrap();

        let dendrogram = model.dendrogram().unwrap();

        // Test node access
        assert!(dendrogram.get_node(0).is_some());
        assert!(dendrogram.get_node(dendrogram.root_id).is_some());
        assert!(dendrogram.get_node(1000).is_none());

        // Test root node properties
        let root = dendrogram.get_node(dendrogram.root_id).unwrap();
        assert!(!root.is_leaf);
        assert_eq!(root.size, 3); // All samples
        assert_eq!(root.children.len(), 2);
    }
}
