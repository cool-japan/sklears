//! Multi-view graph learning methods for semi-supervised learning
//!
//! This module provides advanced graph learning algorithms that can handle
//! multiple views or modalities of data, enabling more robust semi-supervised
//! learning on complex, multi-modal datasets.

use scirs2_core::ndarray_ext::{Array2, ArrayView1, ArrayView2};
use scirs2_core::random::rand_prelude::*;
use scirs2_core::random::Random;
use sklears_core::error::SklearsError;
use std::collections::HashMap;

/// Multi-view graph learning that constructs graphs from multiple data views
#[derive(Clone)]
pub struct MultiViewGraphLearning {
    /// Number of neighbors for k-NN graph construction
    pub k_neighbors: usize,
    /// Weights for combining different views
    pub view_weights: Vec<f64>,
    /// Method for combining views: "weighted", "union", "intersection", "adaptive"
    pub combination_method: String,
    /// Regularization parameter for graph structure learning
    pub regularization: f64,
    /// Maximum iterations for optimization
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl MultiViewGraphLearning {
    /// Create a new multi-view graph learning instance
    pub fn new() -> Self {
        Self {
            k_neighbors: 5,
            view_weights: vec![],
            combination_method: "weighted".to_string(),
            regularization: 0.1,
            max_iter: 100,
            tolerance: 1e-6,
            random_state: None,
        }
    }

    /// Set the number of neighbors for k-NN graph construction
    pub fn k_neighbors(mut self, k: usize) -> Self {
        self.k_neighbors = k;
        self
    }

    /// Set the weights for combining different views
    pub fn view_weights(mut self, weights: Vec<f64>) -> Self {
        self.view_weights = weights;
        self
    }

    /// Set the method for combining views
    pub fn combination_method(mut self, method: String) -> Self {
        self.combination_method = method;
        self
    }

    /// Set the regularization parameter
    pub fn regularization(mut self, reg: f64) -> Self {
        self.regularization = reg;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Learn a unified graph from multiple views of data
    pub fn fit(&self, views: &[ArrayView2<f64>]) -> Result<Array2<f64>, SklearsError> {
        if views.is_empty() {
            return Err(SklearsError::InvalidInput("No views provided".to_string()));
        }

        let n_samples = views[0].nrows();

        // Validate that all views have the same number of samples
        for view in views.iter() {
            if view.nrows() != n_samples {
                return Err(SklearsError::ShapeMismatch {
                    expected: format!("All views should have {} samples", n_samples),
                    actual: format!("View has {} samples", view.nrows()),
                });
            }
        }

        // Construct graphs for each view
        let view_graphs = self.construct_view_graphs(views)?;

        // Combine graphs according to the specified method
        let combined_graph = self.combine_graphs(&view_graphs)?;

        Ok(combined_graph)
    }

    /// Construct k-NN graphs for each view
    fn construct_view_graphs(
        &self,
        views: &[ArrayView2<f64>],
    ) -> Result<Vec<Array2<f64>>, SklearsError> {
        let mut graphs = Vec::new();

        for view in views.iter() {
            let graph = self.construct_knn_graph(view)?;
            graphs.push(graph);
        }

        Ok(graphs)
    }

    /// Construct a k-NN graph from a single view
    fn construct_knn_graph(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>, SklearsError> {
        let n_samples = X.nrows();
        let mut graph = Array2::<f64>::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            let mut distances: Vec<(f64, usize)> = Vec::new();

            for j in 0..n_samples {
                if i != j {
                    let dist = self.euclidean_distance(&X.row(i), &X.row(j));
                    distances.push((dist, j));
                }
            }

            // Sort by distance and take k nearest neighbors
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            for k in 0..self.k_neighbors.min(distances.len()) {
                let j = distances[k].1;
                let weight = (-distances[k].0.powi(2) / 2.0).exp(); // RBF kernel
                graph[[i, j]] = weight;
            }
        }

        // Make graph symmetric
        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let avg_weight = (graph[[i, j]] + graph[[j, i]]) / 2.0;
                graph[[i, j]] = avg_weight;
                graph[[j, i]] = avg_weight;
            }
        }

        Ok(graph)
    }

    /// Combine multiple view graphs into a unified graph
    fn combine_graphs(&self, graphs: &[Array2<f64>]) -> Result<Array2<f64>, SklearsError> {
        if graphs.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No graphs to combine".to_string(),
            ));
        }

        let n_samples = graphs[0].nrows();
        let mut combined = Array2::<f64>::zeros((n_samples, n_samples));

        match self.combination_method.as_str() {
            "weighted" => {
                let weights = if self.view_weights.is_empty() {
                    vec![1.0 / graphs.len() as f64; graphs.len()]
                } else {
                    self.view_weights.clone()
                };

                if weights.len() != graphs.len() {
                    return Err(SklearsError::InvalidInput(
                        "Number of weights must match number of views".to_string(),
                    ));
                }

                for (i, graph) in graphs.iter().enumerate() {
                    combined += &(graph * weights[i]);
                }
            }
            "union" => {
                for graph in graphs.iter() {
                    for i in 0..n_samples {
                        for j in 0..n_samples {
                            combined[[i, j]] = combined[[i, j]].max(graph[[i, j]]);
                        }
                    }
                }
            }
            "intersection" => {
                combined = graphs[0].clone();
                for graph in graphs.iter().skip(1) {
                    for i in 0..n_samples {
                        for j in 0..n_samples {
                            combined[[i, j]] = combined[[i, j]].min(graph[[i, j]]);
                        }
                    }
                }
            }
            "adaptive" => {
                combined = self.adaptive_combination(graphs)?;
            }
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown combination method: {}",
                    self.combination_method
                )));
            }
        }

        Ok(combined)
    }

    /// Adaptive combination that learns optimal weights for views
    fn adaptive_combination(&self, graphs: &[Array2<f64>]) -> Result<Array2<f64>, SklearsError> {
        let n_views = graphs.len();
        let n_samples = graphs[0].nrows();

        // Initialize weights uniformly
        let mut weights = vec![1.0 / n_views as f64; n_views];

        for _iter in 0..self.max_iter {
            let old_weights = weights.clone();

            // Compute current combined graph
            let mut combined = Array2::<f64>::zeros((n_samples, n_samples));
            for (i, graph) in graphs.iter().enumerate() {
                combined += &(graph * weights[i]);
            }

            // Update weights based on agreement with combined graph
            for i in 0..n_views {
                let agreement = self.compute_graph_agreement(&graphs[i], &combined);
                weights[i] = agreement;
            }

            // Normalize weights
            let weight_sum: f64 = weights.iter().sum();
            if weight_sum > 0.0 {
                for w in weights.iter_mut() {
                    *w /= weight_sum;
                }
            }

            // Check convergence
            let weight_change: f64 = weights
                .iter()
                .zip(old_weights.iter())
                .map(|(w1, w2)| (w1 - w2).abs())
                .sum();

            if weight_change < self.tolerance {
                break;
            }
        }

        // Compute final combined graph
        let mut combined = Array2::<f64>::zeros((n_samples, n_samples));
        for (i, graph) in graphs.iter().enumerate() {
            combined += &(graph * weights[i]);
        }

        Ok(combined)
    }

    /// Compute agreement between two graphs
    fn compute_graph_agreement(&self, graph1: &Array2<f64>, graph2: &Array2<f64>) -> f64 {
        let mut agreement = 0.0;
        let mut total = 0.0;

        for i in 0..graph1.nrows() {
            for j in 0..graph1.ncols() {
                let diff = (graph1[[i, j]] - graph2[[i, j]]).abs();
                agreement += 1.0 / (1.0 + diff);
                total += 1.0;
            }
        }

        if total > 0.0 {
            agreement / total
        } else {
            0.0
        }
    }

    /// Compute Euclidean distance between two vectors
    fn euclidean_distance(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        x1.iter()
            .zip(x2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

impl Default for MultiViewGraphLearning {
    fn default() -> Self {
        Self::new()
    }
}

/// Heterogeneous graph learning for mixed data types
#[derive(Clone)]
pub struct HeterogeneousGraphLearning {
    /// Node types in the heterogeneous graph
    pub node_types: Vec<String>,
    /// Edge types connecting different node types
    pub edge_types: Vec<(String, String)>,
    /// Weights for different edge types
    pub edge_weights: HashMap<(String, String), f64>,
    /// Embedding dimensions for each node type
    pub embedding_dims: HashMap<String, usize>,
    /// Number of neighbors for each edge type
    pub k_neighbors: HashMap<(String, String), usize>,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl HeterogeneousGraphLearning {
    /// Create a new heterogeneous graph learning instance
    pub fn new() -> Self {
        Self {
            node_types: vec![],
            edge_types: vec![],
            edge_weights: HashMap::new(),
            embedding_dims: HashMap::new(),
            k_neighbors: HashMap::new(),
            random_state: None,
        }
    }

    /// Set node types
    pub fn node_types(mut self, types: Vec<String>) -> Self {
        self.node_types = types;
        self
    }

    /// Set edge types
    pub fn edge_types(mut self, types: Vec<(String, String)>) -> Self {
        self.edge_types = types;
        self
    }

    /// Set weights for edge types
    pub fn edge_weights(mut self, weights: HashMap<(String, String), f64>) -> Self {
        self.edge_weights = weights;
        self
    }

    /// Set embedding dimensions for node types
    pub fn embedding_dims(mut self, dims: HashMap<String, usize>) -> Self {
        self.embedding_dims = dims;
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Learn embeddings for heterogeneous graph
    pub fn fit(
        &self,
        data: &HashMap<String, ArrayView2<f64>>,
    ) -> Result<HashMap<String, Array2<f64>>, SklearsError> {
        if data.is_empty() {
            return Err(SklearsError::InvalidInput("No data provided".to_string()));
        }

        let mut embeddings = HashMap::new();
        let mut rng = if let Some(seed) = self.random_state {
            Random::seed(42)
        } else {
            Random::seed(42) // Use a default seed instead of from_entropy
        };

        // Initialize embeddings for each node type
        for (node_type, node_data) in data.iter() {
            let embed_dim = self.embedding_dims.get(node_type).unwrap_or(&64);
            let n_nodes = node_data.nrows();

            // Initialize random embeddings
            let mut embedding = Array2::<f64>::zeros((n_nodes, *embed_dim));
            for i in 0..n_nodes {
                for j in 0..*embed_dim {
                    embedding[[i, j]] = rng.gen_range(-1.0..1.0);
                }
            }

            embeddings.insert(node_type.clone(), embedding);
        }

        // Simple implementation: use input features as embeddings
        // In practice, this would involve more sophisticated learning
        for (node_type, node_data) in data.iter() {
            let features = node_data.to_owned();
            embeddings.insert(node_type.clone(), features);
        }

        Ok(embeddings)
    }
}

impl Default for HeterogeneousGraphLearning {
    fn default() -> Self {
        Self::new()
    }
}

/// Temporal graph learning for time-evolving graphs
#[derive(Clone)]
pub struct TemporalGraphLearning {
    /// Window size for temporal analysis
    pub window_size: usize,
    /// Decay factor for temporal weighting
    pub temporal_decay: f64,
    /// Method for temporal aggregation: "mean", "weighted", "attention"
    pub aggregation_method: String,
    /// Number of neighbors for graph construction
    pub k_neighbors: usize,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl TemporalGraphLearning {
    /// Create a new temporal graph learning instance
    pub fn new() -> Self {
        Self {
            window_size: 5,
            temporal_decay: 0.9,
            aggregation_method: "weighted".to_string(),
            k_neighbors: 5,
            random_state: None,
        }
    }

    /// Set window size
    pub fn window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Set temporal decay factor
    pub fn temporal_decay(mut self, decay: f64) -> Self {
        self.temporal_decay = decay;
        self
    }

    /// Set aggregation method
    pub fn aggregation_method(mut self, method: String) -> Self {
        self.aggregation_method = method;
        self
    }

    /// Set number of neighbors
    pub fn k_neighbors(mut self, k: usize) -> Self {
        self.k_neighbors = k;
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Learn from temporal graph snapshots
    pub fn fit(&self, snapshots: &[ArrayView2<f64>]) -> Result<Array2<f64>, SklearsError> {
        if snapshots.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No snapshots provided".to_string(),
            ));
        }

        let n_samples = snapshots[0].nrows();

        // Validate that all snapshots have the same dimensions
        for snapshot in snapshots.iter() {
            if snapshot.nrows() != n_samples {
                return Err(SklearsError::ShapeMismatch {
                    expected: format!("All snapshots should have {} samples", n_samples),
                    actual: format!("Snapshot has {} samples", snapshot.nrows()),
                });
            }
        }

        // Construct graphs for each snapshot
        let graphs = self.construct_temporal_graphs(snapshots)?;

        // Aggregate temporal graphs
        let aggregated_graph = self.aggregate_temporal_graphs(&graphs)?;

        Ok(aggregated_graph)
    }

    /// Construct graphs for temporal snapshots
    fn construct_temporal_graphs(
        &self,
        snapshots: &[ArrayView2<f64>],
    ) -> Result<Vec<Array2<f64>>, SklearsError> {
        let mut graphs = Vec::new();

        for snapshot in snapshots.iter() {
            let graph = self.construct_knn_graph(snapshot)?;
            graphs.push(graph);
        }

        Ok(graphs)
    }

    /// Construct k-NN graph from snapshot data
    fn construct_knn_graph(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>, SklearsError> {
        let n_samples = X.nrows();
        let mut graph = Array2::<f64>::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            let mut distances: Vec<(f64, usize)> = Vec::new();

            for j in 0..n_samples {
                if i != j {
                    let dist = self.euclidean_distance(&X.row(i), &X.row(j));
                    distances.push((dist, j));
                }
            }

            // Sort by distance and take k nearest neighbors
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            for k in 0..self.k_neighbors.min(distances.len()) {
                let j = distances[k].1;
                let weight = (-distances[k].0.powi(2) / 2.0).exp(); // RBF kernel
                graph[[i, j]] = weight;
            }
        }

        // Make graph symmetric
        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let avg_weight = (graph[[i, j]] + graph[[j, i]]) / 2.0;
                graph[[i, j]] = avg_weight;
                graph[[j, i]] = avg_weight;
            }
        }

        Ok(graph)
    }

    /// Aggregate temporal graphs based on the aggregation method
    fn aggregate_temporal_graphs(
        &self,
        graphs: &[Array2<f64>],
    ) -> Result<Array2<f64>, SklearsError> {
        if graphs.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No graphs to aggregate".to_string(),
            ));
        }

        let n_samples = graphs[0].nrows();
        let mut aggregated = Array2::<f64>::zeros((n_samples, n_samples));

        match self.aggregation_method.as_str() {
            "mean" => {
                for graph in graphs.iter() {
                    aggregated += graph;
                }
                aggregated /= graphs.len() as f64;
            }
            "weighted" => {
                let total_weight: f64 = (0..graphs.len())
                    .map(|i| self.temporal_decay.powi(i as i32))
                    .sum();

                for (i, graph) in graphs.iter().enumerate() {
                    let weight = self.temporal_decay.powi(i as i32) / total_weight;
                    aggregated += &(graph * weight);
                }
            }
            "attention" => {
                // Simple attention mechanism - in practice this would be more sophisticated
                let weights = self.compute_attention_weights(graphs)?;
                for (i, graph) in graphs.iter().enumerate() {
                    aggregated += &(graph * weights[i]);
                }
            }
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown aggregation method: {}",
                    self.aggregation_method
                )));
            }
        }

        Ok(aggregated)
    }

    /// Compute attention weights for temporal graphs
    fn compute_attention_weights(&self, graphs: &[Array2<f64>]) -> Result<Vec<f64>, SklearsError> {
        let n_graphs = graphs.len();
        let mut weights = vec![1.0 / n_graphs as f64; n_graphs];

        // Simple implementation: weight by graph density
        let mut densities = Vec::new();
        for graph in graphs.iter() {
            let density = graph.iter().filter(|&&x| x > 0.0).count() as f64 / (graph.len() as f64);
            densities.push(density);
        }

        let total_density: f64 = densities.iter().sum();
        if total_density > 0.0 {
            for (i, density) in densities.iter().enumerate() {
                weights[i] = density / total_density;
            }
        }

        Ok(weights)
    }

    /// Compute Euclidean distance between two vectors
    fn euclidean_distance(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        x1.iter()
            .zip(x2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

impl Default for TemporalGraphLearning {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::array;

    #[test]
    fn test_multi_view_graph_learning() {
        let view1 = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let view2 = array![[2.0, 1.0], [3.0, 2.0], [4.0, 3.0]];
        let views = vec![view1.view(), view2.view()];

        let mvgl = MultiViewGraphLearning::new()
            .k_neighbors(2)
            .combination_method("weighted".to_string());

        let result = mvgl.fit(&views);
        assert!(result.is_ok());

        let graph = result.unwrap();
        assert_eq!(graph.dim(), (3, 3));

        // Check that diagonal is zero (no self-loops)
        assert_eq!(graph[[0, 0]], 0.0);
        assert_eq!(graph[[1, 1]], 0.0);
        assert_eq!(graph[[2, 2]], 0.0);

        // Check symmetry
        assert_abs_diff_eq!(graph[[0, 1]], graph[[1, 0]], epsilon = 1e-10);
        assert_abs_diff_eq!(graph[[0, 2]], graph[[2, 0]], epsilon = 1e-10);
        assert_abs_diff_eq!(graph[[1, 2]], graph[[2, 1]], epsilon = 1e-10);
    }

    #[test]
    fn test_multi_view_graph_union() {
        let view1 = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let view2 = array![[2.0, 1.0], [3.0, 2.0], [4.0, 3.0]];
        let views = vec![view1.view(), view2.view()];

        let mvgl = MultiViewGraphLearning::new()
            .k_neighbors(2)
            .combination_method("union".to_string());

        let result = mvgl.fit(&views);
        assert!(result.is_ok());

        let graph = result.unwrap();
        assert_eq!(graph.dim(), (3, 3));
    }

    #[test]
    fn test_multi_view_graph_adaptive() {
        let view1 = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let view2 = array![[2.0, 1.0], [3.0, 2.0], [4.0, 3.0]];
        let views = vec![view1.view(), view2.view()];

        let mvgl = MultiViewGraphLearning::new()
            .k_neighbors(2)
            .combination_method("adaptive".to_string())
            .max_iter(10)
            .tolerance(1e-4);

        let result = mvgl.fit(&views);
        assert!(result.is_ok());

        let graph = result.unwrap();
        assert_eq!(graph.dim(), (3, 3));
    }

    #[test]
    fn test_heterogeneous_graph_learning() {
        let type1_data = array![[1.0, 2.0], [2.0, 3.0]];
        let type2_data = array![[3.0, 4.0], [4.0, 5.0]];
        let mut data = HashMap::new();
        data.insert("type1".to_string(), type1_data.view());
        data.insert("type2".to_string(), type2_data.view());

        let hgl = HeterogeneousGraphLearning::new()
            .node_types(vec!["type1".to_string(), "type2".to_string()]);

        let result = hgl.fit(&data);
        assert!(result.is_ok());

        let embeddings = result.unwrap();
        assert!(embeddings.contains_key("type1"));
        assert!(embeddings.contains_key("type2"));
        assert_eq!(embeddings["type1"].dim(), (2, 2));
        assert_eq!(embeddings["type2"].dim(), (2, 2));
    }

    #[test]
    fn test_temporal_graph_learning() {
        let snapshot1 = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let snapshot2 = array![[1.1, 2.1], [2.1, 3.1], [3.1, 4.1]];
        let snapshot3 = array![[1.2, 2.2], [2.2, 3.2], [3.2, 4.2]];
        let snapshots = vec![snapshot1.view(), snapshot2.view(), snapshot3.view()];

        let tgl = TemporalGraphLearning::new()
            .window_size(3)
            .temporal_decay(0.9)
            .aggregation_method("weighted".to_string())
            .k_neighbors(2);

        let result = tgl.fit(&snapshots);
        assert!(result.is_ok());

        let graph = result.unwrap();
        assert_eq!(graph.dim(), (3, 3));

        // Check that diagonal is zero (no self-loops)
        assert_eq!(graph[[0, 0]], 0.0);
        assert_eq!(graph[[1, 1]], 0.0);
        assert_eq!(graph[[2, 2]], 0.0);
    }

    #[test]
    fn test_temporal_graph_attention() {
        let snapshot1 = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let snapshot2 = array![[1.1, 2.1], [2.1, 3.1], [3.1, 4.1]];
        let snapshots = vec![snapshot1.view(), snapshot2.view()];

        let tgl = TemporalGraphLearning::new()
            .aggregation_method("attention".to_string())
            .k_neighbors(2);

        let result = tgl.fit(&snapshots);
        assert!(result.is_ok());

        let graph = result.unwrap();
        assert_eq!(graph.dim(), (3, 3));
    }

    #[test]
    fn test_multi_view_graph_error_cases() {
        let mvgl = MultiViewGraphLearning::new();

        // Test with empty views
        let result = mvgl.fit(&[]);
        assert!(result.is_err());

        // Test with mismatched dimensions
        let view1 = array![[1.0, 2.0], [2.0, 3.0]];
        let view2 = array![[3.0, 4.0], [4.0, 5.0], [5.0, 6.0]];
        let views = vec![view1.view(), view2.view()];

        let result = mvgl.fit(&views);
        assert!(result.is_err());
    }

    #[test]
    fn test_temporal_graph_error_cases() {
        let tgl = TemporalGraphLearning::new();

        // Test with empty snapshots
        let result = tgl.fit(&[]);
        assert!(result.is_err());

        // Test with mismatched dimensions
        let snapshot1 = array![[1.0, 2.0], [2.0, 3.0]];
        let snapshot2 = array![[3.0, 4.0], [4.0, 5.0], [5.0, 6.0]];
        let snapshots = vec![snapshot1.view(), snapshot2.view()];

        let result = tgl.fit(&snapshots);
        assert!(result.is_err());
    }
}
