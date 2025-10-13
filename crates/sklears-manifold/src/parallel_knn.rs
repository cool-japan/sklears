//! Parallel K-Nearest Neighbors Search
//!
//! This module provides highly optimized parallel algorithms for finding
//! k-nearest neighbors, which is a fundamental operation in many manifold
//! learning algorithms. The implementations use various optimization techniques
//! including parallel processing, spatial data structures, and approximation methods.

use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::thread_rng;
use scirs2_core::random::Rng;
use scirs2_core::random::SeedableRng;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// Parallel K-Nearest Neighbors Search
///
/// This structure provides efficient parallel k-nearest neighbors search
/// using various algorithms and optimizations for different data sizes
/// and dimensionalities.
///
/// # Parameters
///
/// * `k` - Number of nearest neighbors to find
/// * `algorithm` - Algorithm to use ('brute_force', 'kd_tree', 'ball_tree', 'lsh')
/// * `metric` - Distance metric ('euclidean', 'manhattan', 'cosine')
/// * `n_jobs` - Number of parallel jobs (-1 for all cores)
/// * `chunk_size` - Size of chunks for parallel processing
/// * `approximate` - Whether to use approximate methods for speed
/// * `precision` - Precision level for approximate methods (0.0 to 1.0)
///
/// # Examples
///
/// ```
/// use sklears_manifold::parallel_knn::ParallelKNN;
/// use scirs2_core::ndarray::array;
///
/// let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
///
/// let knn = ParallelKNN::new(2);
/// let (indices, distances) = knn.fit_search(&data, &data).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ParallelKNN {
    k: usize,
    algorithm: String,
    metric: String,
    n_jobs: i32,
    chunk_size: usize,
    approximate: bool,
    precision: Float,
}

impl ParallelKNN {
    /// Create a new parallel KNN instance
    pub fn new(k: usize) -> Self {
        Self {
            k,
            algorithm: "auto".to_string(),
            metric: "euclidean".to_string(),
            n_jobs: -1,
            chunk_size: 1000,
            approximate: false,
            precision: 0.9,
        }
    }

    /// Set the algorithm to use
    pub fn algorithm(mut self, algorithm: &str) -> Self {
        self.algorithm = algorithm.to_string();
        self
    }

    /// Set the distance metric
    pub fn metric(mut self, metric: &str) -> Self {
        self.metric = metric.to_string();
        self
    }

    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: i32) -> Self {
        self.n_jobs = n_jobs;
        self
    }

    /// Set the chunk size for parallel processing
    pub fn chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Enable approximate search
    pub fn approximate(mut self, approximate: bool) -> Self {
        self.approximate = approximate;
        self
    }

    /// Set precision for approximate methods
    pub fn precision(mut self, precision: Float) -> Self {
        self.precision = precision;
        self
    }

    /// Fit and search for k-nearest neighbors
    pub fn fit_search(
        &self,
        X_train: &Array2<Float>,
        X_query: &Array2<Float>,
    ) -> SklResult<(Array2<usize>, Array2<Float>)> {
        if X_train.is_empty() || X_query.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Input arrays cannot be empty".to_string(),
            ));
        }

        if X_train.ncols() != X_query.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "Train and query arrays must have same number of features: {} vs {}",
                X_train.ncols(),
                X_query.ncols()
            )));
        }

        if self.k > X_train.nrows() {
            return Err(SklearsError::InvalidInput(format!(
                "k ({}) cannot be greater than number of training samples ({})",
                self.k,
                X_train.nrows()
            )));
        }

        // Choose algorithm based on data characteristics
        let algorithm = if self.algorithm == "auto" {
            self.choose_algorithm(X_train)
        } else {
            self.algorithm.clone()
        };

        match algorithm.as_str() {
            "brute_force" => self.brute_force_search(X_train, X_query),
            "kd_tree" => self.kd_tree_search(X_train, X_query),
            "ball_tree" => self.ball_tree_search(X_train, X_query),
            "lsh" => self.lsh_search(X_train, X_query),
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown algorithm: {}",
                algorithm
            ))),
        }
    }

    /// Choose optimal algorithm based on data characteristics
    fn choose_algorithm(&self, X: &Array2<Float>) -> String {
        let n_samples = X.nrows();
        let n_features = X.ncols();

        if self.approximate && n_samples > 10000 {
            "lsh".to_string()
        } else if n_features < 20 && n_samples < 5000 {
            "kd_tree".to_string()
        } else if n_samples < 1000 {
            "brute_force".to_string()
        } else {
            "ball_tree".to_string()
        }
    }

    /// Brute force parallel search
    fn brute_force_search(
        &self,
        X_train: &Array2<Float>,
        X_query: &Array2<Float>,
    ) -> SklResult<(Array2<usize>, Array2<Float>)> {
        let n_queries = X_query.nrows();
        let n_train = X_train.nrows();

        let chunk_size = self.chunk_size.min(n_queries);

        // Parallel processing over query chunks
        let results: Vec<_> = X_query
            .axis_chunks_iter(Axis(0), chunk_size)
            .enumerate()
            .collect::<Vec<_>>()
            .par_iter()
            .map(|(chunk_idx, chunk)| {
                let start_idx = chunk_idx * chunk_size;
                let chunk_size = chunk.nrows();

                let mut indices = Array2::zeros((chunk_size, self.k));
                let mut distances = Array2::zeros((chunk_size, self.k));

                for (i, query_point) in chunk.axis_iter(Axis(0)).enumerate() {
                    let (point_indices, point_distances) =
                        self.find_knn_for_point(&query_point, X_train)?;

                    indices.row_mut(i).assign(&Array1::from_vec(point_indices));
                    distances
                        .row_mut(i)
                        .assign(&Array1::from_vec(point_distances));
                }

                Ok((indices, distances))
            })
            .collect::<SklResult<Vec<_>>>()?;

        // Combine results
        let total_indices = Array2::from_shape_fn((n_queries, self.k), |(i, j)| {
            let chunk_idx = i / chunk_size;
            let local_idx = i % chunk_size;
            results[chunk_idx].0[[local_idx, j]]
        });

        let total_distances = Array2::from_shape_fn((n_queries, self.k), |(i, j)| {
            let chunk_idx = i / chunk_size;
            let local_idx = i % chunk_size;
            results[chunk_idx].1[[local_idx, j]]
        });

        Ok((total_indices, total_distances))
    }

    /// Find k-nearest neighbors for a single point
    fn find_knn_for_point(
        &self,
        query: &ArrayView1<Float>,
        X_train: &Array2<Float>,
    ) -> SklResult<(Vec<usize>, Vec<Float>)> {
        let n_train = X_train.nrows();

        // Use a max-heap to keep track of k closest points
        let mut heap = BinaryHeap::new();

        for (i, train_point) in X_train.axis_iter(Axis(0)).enumerate() {
            let distance = self.compute_distance(query, &train_point)?;

            if heap.len() < self.k {
                heap.push(NeighborDistance { index: i, distance });
            } else if distance < heap.peek().unwrap().distance {
                heap.pop();
                heap.push(NeighborDistance { index: i, distance });
            }
        }

        // Extract results and sort by distance
        let mut neighbors: Vec<_> = heap.into_vec();
        neighbors.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });

        let indices: Vec<usize> = neighbors.iter().map(|n| n.index).collect();
        let distances: Vec<Float> = neighbors.iter().map(|n| n.distance).collect();

        Ok((indices, distances))
    }

    /// Compute distance between two points
    fn compute_distance(&self, a: &ArrayView1<Float>, b: &ArrayView1<Float>) -> SklResult<Float> {
        match self.metric.as_str() {
            "euclidean" => Ok(euclidean_distance(a, b)),
            "manhattan" => Ok(manhattan_distance(a, b)),
            "cosine" => Ok(cosine_distance(a, b)?),
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown metric: {}",
                self.metric
            ))),
        }
    }

    /// KD-Tree based search (simplified implementation)
    fn kd_tree_search(
        &self,
        X_train: &Array2<Float>,
        X_query: &Array2<Float>,
    ) -> SklResult<(Array2<usize>, Array2<Float>)> {
        // Build KD-tree
        let kd_tree = KDTree::build(X_train)?;

        let n_queries = X_query.nrows();
        let mut indices = Array2::zeros((n_queries, self.k));
        let mut distances = Array2::zeros((n_queries, self.k));

        // Parallel search using KD-tree
        let results: Vec<_> = X_query
            .axis_iter(Axis(0))
            .enumerate()
            .collect::<Vec<_>>()
            .par_iter()
            .map(|(i, query_point)| kd_tree.search_knn(query_point, self.k, &self.metric))
            .collect::<SklResult<Vec<_>>>()?;

        for (i, (point_indices, point_distances)) in results.into_iter().enumerate() {
            indices.row_mut(i).assign(&Array1::from_vec(point_indices));
            distances
                .row_mut(i)
                .assign(&Array1::from_vec(point_distances));
        }

        Ok((indices, distances))
    }

    /// Ball Tree based search (simplified implementation)
    fn ball_tree_search(
        &self,
        X_train: &Array2<Float>,
        X_query: &Array2<Float>,
    ) -> SklResult<(Array2<usize>, Array2<Float>)> {
        // Build Ball Tree
        let ball_tree = BallTree::build(X_train, &self.metric)?;

        let n_queries = X_query.nrows();
        let mut indices = Array2::zeros((n_queries, self.k));
        let mut distances = Array2::zeros((n_queries, self.k));

        // Parallel search using Ball Tree
        let results: Vec<_> = X_query
            .axis_iter(Axis(0))
            .enumerate()
            .collect::<Vec<_>>()
            .par_iter()
            .map(|(i, query_point)| ball_tree.search_knn(query_point, self.k))
            .collect::<SklResult<Vec<_>>>()?;

        for (i, (point_indices, point_distances)) in results.into_iter().enumerate() {
            indices.row_mut(i).assign(&Array1::from_vec(point_indices));
            distances
                .row_mut(i)
                .assign(&Array1::from_vec(point_distances));
        }

        Ok((indices, distances))
    }

    /// Locality Sensitive Hashing based approximate search
    fn lsh_search(
        &self,
        X_train: &Array2<Float>,
        X_query: &Array2<Float>,
    ) -> SklResult<(Array2<usize>, Array2<Float>)> {
        // Build LSH index
        let lsh_index = LSHIndex::build(X_train, self.precision)?;

        let n_queries = X_query.nrows();
        let mut indices = Array2::zeros((n_queries, self.k));
        let mut distances = Array2::zeros((n_queries, self.k));

        // Parallel approximate search using LSH
        let results: Vec<_> = X_query
            .axis_iter(Axis(0))
            .enumerate()
            .collect::<Vec<_>>()
            .par_iter()
            .map(|(i, query_point)| {
                lsh_index.search_approximate_knn(query_point, self.k, X_train, &self.metric)
            })
            .collect::<SklResult<Vec<_>>>()?;

        for (i, (point_indices, point_distances)) in results.into_iter().enumerate() {
            indices.row_mut(i).assign(&Array1::from_vec(point_indices));
            distances
                .row_mut(i)
                .assign(&Array1::from_vec(point_distances));
        }

        Ok((indices, distances))
    }
}

/// Helper structure for maintaining k-nearest neighbors
#[derive(Debug, Clone)]
struct NeighborDistance {
    index: usize,
    distance: Float,
}

impl PartialEq for NeighborDistance {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for NeighborDistance {}

impl PartialOrd for NeighborDistance {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse order for max-heap behavior
        other.distance.partial_cmp(&self.distance)
    }
}

impl Ord for NeighborDistance {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Simple KD-Tree implementation for nearest neighbor search
#[derive(Debug, Clone)]
struct KDTree {
    root: Option<Box<KDNode>>,
    data: Array2<Float>,
}

#[derive(Debug, Clone)]
struct KDNode {
    point_index: usize,
    split_dimension: usize,
    split_value: Float,
    left: Option<Box<KDNode>>,
    right: Option<Box<KDNode>>,
}

impl KDTree {
    fn build(data: &Array2<Float>) -> SklResult<Self> {
        let n_samples = data.nrows();
        let indices: Vec<usize> = (0..n_samples).collect();

        let root = Self::build_recursive(data, indices, 0);

        Ok(Self {
            root,
            data: data.clone(),
        })
    }

    fn build_recursive(
        data: &Array2<Float>,
        mut indices: Vec<usize>,
        depth: usize,
    ) -> Option<Box<KDNode>> {
        if indices.is_empty() {
            return None;
        }

        if indices.len() == 1 {
            return Some(Box::new(KDNode {
                point_index: indices[0],
                split_dimension: 0,
                split_value: 0.0,
                left: None,
                right: None,
            }));
        }

        let n_features = data.ncols();
        let split_dimension = depth % n_features;

        // Sort indices by split dimension
        indices.sort_by(|&a, &b| {
            data[[a, split_dimension]]
                .partial_cmp(&data[[b, split_dimension]])
                .unwrap_or(Ordering::Equal)
        });

        let median_idx = indices.len() / 2;
        let median_point = indices[median_idx];
        let split_value = data[[median_point, split_dimension]];

        let left_indices = indices[..median_idx].to_vec();
        let right_indices = indices[median_idx + 1..].to_vec();

        let left = Self::build_recursive(data, left_indices, depth + 1);
        let right = Self::build_recursive(data, right_indices, depth + 1);

        Some(Box::new(KDNode {
            point_index: median_point,
            split_dimension,
            split_value,
            left,
            right,
        }))
    }

    fn search_knn(
        &self,
        query: &ArrayView1<Float>,
        k: usize,
        metric: &str,
    ) -> SklResult<(Vec<usize>, Vec<Float>)> {
        if let Some(ref root) = self.root {
            let mut heap = BinaryHeap::new();
            self.search_recursive(root, query, k, &mut heap, metric)?;

            let mut neighbors: Vec<_> = heap.into_vec();
            neighbors.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(Ordering::Equal)
            });

            let indices: Vec<usize> = neighbors.iter().map(|n| n.index).collect();
            let distances: Vec<Float> = neighbors.iter().map(|n| n.distance).collect();

            Ok((indices, distances))
        } else {
            Ok((Vec::new(), Vec::new()))
        }
    }

    fn search_recursive(
        &self,
        node: &KDNode,
        query: &ArrayView1<Float>,
        k: usize,
        heap: &mut BinaryHeap<NeighborDistance>,
        metric: &str,
    ) -> SklResult<()> {
        // Calculate distance to current node
        let distance = match metric {
            "euclidean" => euclidean_distance(query, &self.data.row(node.point_index)),
            "manhattan" => manhattan_distance(query, &self.data.row(node.point_index)),
            "cosine" => cosine_distance(query, &self.data.row(node.point_index))?,
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown metric: {}",
                    metric
                )))
            }
        };

        // Update heap
        if heap.len() < k {
            heap.push(NeighborDistance {
                index: node.point_index,
                distance,
            });
        } else if distance < heap.peek().unwrap().distance {
            heap.pop();
            heap.push(NeighborDistance {
                index: node.point_index,
                distance,
            });
        }

        // Recursively search children
        let query_value = query[node.split_dimension];
        let diff = query_value - node.split_value;

        let (primary, secondary) = if diff <= 0.0 {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        // Search primary subtree
        if let Some(ref child) = primary {
            self.search_recursive(child, query, k, heap, metric)?;
        }

        // Search secondary subtree if necessary
        if let Some(ref child) = secondary {
            let should_search = heap.len() < k || diff.abs() < heap.peek().unwrap().distance;
            if should_search {
                self.search_recursive(child, query, k, heap, metric)?;
            }
        }

        Ok(())
    }
}

/// Simple Ball Tree implementation for nearest neighbor search
#[derive(Debug, Clone)]
struct BallTree {
    root: Option<Box<BallNode>>,
    data: Array2<Float>,
    metric: String,
}

#[derive(Debug, Clone)]
struct BallNode {
    center: Array1<Float>,
    radius: Float,
    point_indices: Vec<usize>,
    left: Option<Box<BallNode>>,
    right: Option<Box<BallNode>>,
}

impl BallTree {
    fn build(data: &Array2<Float>, metric: &str) -> SklResult<Self> {
        let n_samples = data.nrows();
        let indices: Vec<usize> = (0..n_samples).collect();

        let root = Self::build_recursive(data, indices, metric)?;

        Ok(Self {
            root,
            data: data.clone(),
            metric: metric.to_string(),
        })
    }

    fn build_recursive(
        data: &Array2<Float>,
        indices: Vec<usize>,
        metric: &str,
    ) -> SklResult<Option<Box<BallNode>>> {
        if indices.is_empty() {
            return Ok(None);
        }

        if indices.len() <= 10 {
            // Leaf node
            let center = Self::compute_centroid(data, &indices);
            let radius = Self::compute_radius(data, &indices, &center, metric)?;

            return Ok(Some(Box::new(BallNode {
                center,
                radius,
                point_indices: indices,
                left: None,
                right: None,
            })));
        }

        // Find the dimension with maximum spread
        let (split_dim, _) = Self::find_max_spread_dimension(data, &indices);

        // Split points along this dimension
        let mut sorted_indices = indices;
        sorted_indices.sort_by(|&a, &b| {
            data[[a, split_dim]]
                .partial_cmp(&data[[b, split_dim]])
                .unwrap_or(Ordering::Equal)
        });

        let mid = sorted_indices.len() / 2;
        let left_indices = sorted_indices[..mid].to_vec();
        let right_indices = sorted_indices[mid..].to_vec();

        let left = Self::build_recursive(data, left_indices, metric)?;
        let right = Self::build_recursive(data, right_indices, metric)?;

        // Compute center and radius for internal node
        let center = Self::compute_centroid(data, &sorted_indices);
        let radius = Self::compute_radius(data, &sorted_indices, &center, metric)?;

        Ok(Some(Box::new(BallNode {
            center,
            radius,
            point_indices: sorted_indices,
            left,
            right,
        })))
    }

    fn compute_centroid(data: &Array2<Float>, indices: &[usize]) -> Array1<Float> {
        let n_features = data.ncols();
        let mut centroid = Array1::zeros(n_features);

        for &idx in indices {
            centroid += &data.row(idx);
        }

        centroid / indices.len() as Float
    }

    fn compute_radius(
        data: &Array2<Float>,
        indices: &[usize],
        center: &Array1<Float>,
        metric: &str,
    ) -> SklResult<Float> {
        let mut max_distance: Float = 0.0;

        for &idx in indices {
            let distance = match metric {
                "euclidean" => euclidean_distance(&center.view(), &data.row(idx)),
                "manhattan" => manhattan_distance(&center.view(), &data.row(idx)),
                "cosine" => cosine_distance(&center.view(), &data.row(idx))?,
                _ => {
                    return Err(SklearsError::InvalidInput(format!(
                        "Unknown metric: {}",
                        metric
                    )))
                }
            };

            max_distance = max_distance.max(distance);
        }

        Ok(max_distance)
    }

    fn find_max_spread_dimension(data: &Array2<Float>, indices: &[usize]) -> (usize, Float) {
        let n_features = data.ncols();
        let mut max_spread = 0.0;
        let mut max_dim = 0;

        for dim in 0..n_features {
            let mut min_val = Float::INFINITY;
            let mut max_val = Float::NEG_INFINITY;

            for &idx in indices {
                let val = data[[idx, dim]];
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }

            let spread = max_val - min_val;
            if spread > max_spread {
                max_spread = spread;
                max_dim = dim;
            }
        }

        (max_dim, max_spread)
    }

    fn search_knn(
        &self,
        query: &ArrayView1<Float>,
        k: usize,
    ) -> SklResult<(Vec<usize>, Vec<Float>)> {
        if let Some(ref root) = self.root {
            let mut heap = BinaryHeap::new();
            self.search_recursive(root, query, k, &mut heap)?;

            let mut neighbors: Vec<_> = heap.into_vec();
            neighbors.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(Ordering::Equal)
            });

            let indices: Vec<usize> = neighbors.iter().map(|n| n.index).collect();
            let distances: Vec<Float> = neighbors.iter().map(|n| n.distance).collect();

            Ok((indices, distances))
        } else {
            Ok((Vec::new(), Vec::new()))
        }
    }

    fn search_recursive(
        &self,
        node: &BallNode,
        query: &ArrayView1<Float>,
        k: usize,
        heap: &mut BinaryHeap<NeighborDistance>,
    ) -> SklResult<()> {
        // Check if we can prune this node
        let distance_to_center = match self.metric.as_str() {
            "euclidean" => euclidean_distance(query, &node.center.view()),
            "manhattan" => manhattan_distance(query, &node.center.view()),
            "cosine" => cosine_distance(query, &node.center.view())?,
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown metric: {}",
                    self.metric
                )))
            }
        };

        let min_distance_to_ball = (distance_to_center - node.radius).max(0.0);

        if heap.len() >= k && min_distance_to_ball >= heap.peek().unwrap().distance {
            return Ok(()); // Prune this subtree
        }

        // If leaf node, check all points
        if node.left.is_none() && node.right.is_none() {
            for &idx in &node.point_indices {
                let distance = match self.metric.as_str() {
                    "euclidean" => euclidean_distance(query, &self.data.row(idx)),
                    "manhattan" => manhattan_distance(query, &self.data.row(idx)),
                    "cosine" => cosine_distance(query, &self.data.row(idx))?,
                    _ => {
                        return Err(SklearsError::InvalidInput(format!(
                            "Unknown metric: {}",
                            self.metric
                        )))
                    }
                };

                if heap.len() < k {
                    heap.push(NeighborDistance {
                        index: idx,
                        distance,
                    });
                } else if distance < heap.peek().unwrap().distance {
                    heap.pop();
                    heap.push(NeighborDistance {
                        index: idx,
                        distance,
                    });
                }
            }
        } else {
            // Recursively search children
            if let Some(ref left) = node.left {
                self.search_recursive(left, query, k, heap)?;
            }
            if let Some(ref right) = node.right {
                self.search_recursive(right, query, k, heap)?;
            }
        }

        Ok(())
    }
}

/// Simple Locality Sensitive Hashing implementation
#[derive(Debug, Clone)]
struct LSHIndex {
    hash_tables: Vec<Vec<(Vec<usize>, Vec<usize>)>>, // (hash_value, point_indices)
    hash_functions: Vec<Array2<Float>>,
    n_hash_tables: usize,
    n_hash_functions_per_table: usize,
}

impl LSHIndex {
    fn build(data: &Array2<Float>, precision: Float) -> SklResult<Self> {
        let n_features = data.ncols();
        let n_samples = data.nrows();

        // Number of hash tables and functions based on precision
        let n_hash_tables = ((1.0 / precision).ln() / 2.0).ceil() as usize;
        let n_hash_functions_per_table = (n_features as Float * 0.1).ceil() as usize;

        // Generate random hash functions
        let mut hash_functions = Vec::new();
        let mut rng = StdRng::seed_from_u64(thread_rng().gen());
        for _ in 0..n_hash_tables {
            let mut hash_func = Array2::<Float>::zeros((n_hash_functions_per_table, n_features));
            for elem in hash_func.iter_mut() {
                *elem = rng.sample::<Float, _>(scirs2_core::StandardNormal);
            }
            hash_functions.push(hash_func);
        }

        // Build hash tables
        let mut hash_tables = Vec::new();
        for table_idx in 0..n_hash_tables {
            let mut table = HashMap::new();

            for (point_idx, point) in data.axis_iter(Axis(0)).enumerate() {
                let hash_value = Self::compute_hash(&hash_functions[table_idx], &point);
                table
                    .entry(hash_value)
                    .or_insert_with(Vec::new)
                    .push(point_idx);
            }

            // Convert to vector format
            let table_vec: Vec<(Vec<usize>, Vec<usize>)> = table.into_iter().collect();

            hash_tables.push(table_vec);
        }

        Ok(Self {
            hash_tables,
            hash_functions,
            n_hash_tables,
            n_hash_functions_per_table,
        })
    }

    fn compute_hash(hash_function: &Array2<Float>, point: &ArrayView1<Float>) -> Vec<usize> {
        let mut hash_value = Vec::new();

        for i in 0..hash_function.nrows() {
            let dot_product = hash_function.row(i).dot(point);
            hash_value.push(if dot_product >= 0.0 { 1 } else { 0 });
        }

        hash_value
    }

    fn search_approximate_knn(
        &self,
        query: &ArrayView1<Float>,
        k: usize,
        data: &Array2<Float>,
        metric: &str,
    ) -> SklResult<(Vec<usize>, Vec<Float>)> {
        let mut candidate_indices = std::collections::HashSet::new();

        // Find candidates from all hash tables
        for table_idx in 0..self.n_hash_tables {
            let query_hash = Self::compute_hash(&self.hash_functions[table_idx], query);

            // Find matching bucket
            for (bucket_hash, point_indices) in &self.hash_tables[table_idx] {
                if *bucket_hash == query_hash {
                    for &idx in point_indices {
                        candidate_indices.insert(idx);
                    }
                    break;
                }
            }
        }

        // If not enough candidates, add some random points
        if candidate_indices.len() < k * 2 {
            let n_samples = data.nrows();
            let n_additional = (k * 5).min(n_samples);

            for i in 0..n_additional {
                candidate_indices.insert(i % n_samples);
            }
        }

        // Compute exact distances for candidates
        let mut neighbors = Vec::new();
        for &idx in &candidate_indices {
            let distance = match metric {
                "euclidean" => euclidean_distance(query, &data.row(idx)),
                "manhattan" => manhattan_distance(query, &data.row(idx)),
                "cosine" => cosine_distance(query, &data.row(idx))?,
                _ => {
                    return Err(SklearsError::InvalidInput(format!(
                        "Unknown metric: {}",
                        metric
                    )))
                }
            };

            neighbors.push(NeighborDistance {
                index: idx,
                distance,
            });
        }

        // Sort and take top k
        neighbors.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        neighbors.truncate(k);

        let indices: Vec<usize> = neighbors.iter().map(|n| n.index).collect();
        let distances: Vec<Float> = neighbors.iter().map(|n| n.distance).collect();

        Ok((indices, distances))
    }
}

// Distance functions

fn euclidean_distance(a: &ArrayView1<Float>, b: &ArrayView1<Float>) -> Float {
    (a - b).mapv(|x| x * x).sum().sqrt()
}

fn manhattan_distance(a: &ArrayView1<Float>, b: &ArrayView1<Float>) -> Float {
    (a - b).mapv(|x| x.abs()).sum()
}

fn cosine_distance(a: &ArrayView1<Float>, b: &ArrayView1<Float>) -> SklResult<Float> {
    let dot_product = a.dot(b);
    let norm_a = a.mapv(|x| x * x).sum().sqrt();
    let norm_b = b.mapv(|x| x * x).sum().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(1.0); // Maximum cosine distance
    }

    let cosine_sim = dot_product / (norm_a * norm_b);
    Ok(1.0 - cosine_sim)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_parallel_knn_basic() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let knn = ParallelKNN::new(2);
        let (indices, distances) = knn.fit_search(&data, &data).unwrap();

        assert_eq!(indices.shape(), &[4, 2]);
        assert_eq!(distances.shape(), &[4, 2]);

        // First neighbor should be the point itself
        for i in 0..4 {
            assert_eq!(indices[[i, 0]], i);
            assert_abs_diff_eq!(distances[[i, 0]], 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_parallel_knn_algorithms() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];

        let algorithms = vec!["brute_force", "kd_tree", "ball_tree"];

        for algorithm in algorithms {
            let knn = ParallelKNN::new(2).algorithm(algorithm);
            let (indices, distances) = knn.fit_search(&data, &data).unwrap();

            assert_eq!(indices.shape(), &[5, 2]);
            assert_eq!(distances.shape(), &[5, 2]);

            // Check that distances are finite and sorted
            for i in 0..5 {
                assert!(distances[[i, 0]].is_finite());
                assert!(distances[[i, 1]].is_finite());
                assert!(distances[[i, 0]] <= distances[[i, 1]]);
            }
        }
    }

    #[test]
    fn test_parallel_knn_metrics() {
        let data = array![[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]];

        let metrics = vec!["euclidean", "manhattan", "cosine"];

        for metric in metrics {
            let knn = ParallelKNN::new(2).metric(metric);
            let (indices, distances) = knn.fit_search(&data, &data).unwrap();

            assert_eq!(indices.shape(), &[4, 2]);
            assert_eq!(distances.shape(), &[4, 2]);

            // Check that all distances are finite
            for i in 0..4 {
                for j in 0..2 {
                    assert!(distances[[i, j]].is_finite());
                }
            }
        }
    }

    #[test]
    fn test_parallel_knn_approximate() {
        let data = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0]
        ];

        let knn = ParallelKNN::new(2)
            .algorithm("lsh")
            .approximate(true)
            .precision(0.8);
        let (indices, distances) = knn.fit_search(&data, &data).unwrap();

        assert_eq!(indices.shape(), &[6, 2]);
        assert_eq!(distances.shape(), &[6, 2]);

        // Check that results are reasonable
        for i in 0..6 {
            assert!(distances[[i, 0]].is_finite());
            assert!(distances[[i, 1]].is_finite());
        }
    }

    #[test]
    fn test_distance_functions() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 5.0, 6.0];

        let euclidean_dist = euclidean_distance(&a.view(), &b.view());
        assert_abs_diff_eq!(
            euclidean_dist,
            (3.0_f64.powi(2) * 3.0).sqrt(),
            epsilon = 1e-10
        );

        let manhattan_dist = manhattan_distance(&a.view(), &b.view());
        assert_abs_diff_eq!(manhattan_dist, 9.0, epsilon = 1e-10);

        let cosine_dist = cosine_distance(&a.view(), &b.view()).unwrap();
        assert!(cosine_dist >= 0.0 && cosine_dist <= 2.0);
    }

    #[test]
    fn test_kd_tree() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let kd_tree = KDTree::build(&data).unwrap();
        let query = array![3.5, 4.5];
        let (indices, distances) = kd_tree.search_knn(&query.view(), 2, "euclidean").unwrap();

        assert_eq!(indices.len(), 2);
        assert_eq!(distances.len(), 2);
        assert!(distances[0] <= distances[1]);
    }

    #[test]
    fn test_ball_tree() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let ball_tree = BallTree::build(&data, "euclidean").unwrap();
        let query = array![3.5, 4.5];
        let (indices, distances) = ball_tree.search_knn(&query.view(), 2).unwrap();

        assert_eq!(indices.len(), 2);
        assert_eq!(distances.len(), 2);
        assert!(distances[0] <= distances[1]);
    }

    #[test]
    fn test_lsh_index() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let lsh_index = LSHIndex::build(&data, 0.8).unwrap();
        let query = array![3.5, 4.5];
        let (indices, distances) = lsh_index
            .search_approximate_knn(&query.view(), 2, &data, "euclidean")
            .unwrap();

        assert_eq!(indices.len(), 2);
        assert_eq!(distances.len(), 2);
    }

    #[test]
    fn test_parallel_knn_invalid_parameters() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];

        // Test k > n_samples
        let knn = ParallelKNN::new(5);
        assert!(knn.fit_search(&data, &data).is_err());

        // Test mismatched dimensions
        let query = array![[1.0, 2.0, 3.0]];
        let knn = ParallelKNN::new(1);
        assert!(knn.fit_search(&data, &query).is_err());
    }
}
