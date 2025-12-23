//! Composable graph construction methods
//!
//! This module provides a flexible, composable framework for constructing graphs
//! used in semi-supervised learning. It allows combining different graph construction
//! strategies and applying transformations in a pipeline.

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView2};
use sklears_core::error::{Result as SklResult, SklearsError};
use sklears_core::types::Float;

/// Trait for graph construction strategies
pub trait GraphBuilder: Clone {
    /// Build a graph from data
    fn build(&self, X: &ArrayView2<Float>) -> SklResult<Array2<f64>>;
}

/// Trait for graph transformations
pub trait GraphTransform: Clone {
    /// Transform a graph
    fn transform(&self, graph: &Array2<f64>) -> SklResult<Array2<f64>>;
}

/// K-Nearest Neighbors graph builder
#[derive(Debug, Clone)]
pub struct KNNGraphBuilder {
    n_neighbors: usize,
    weighted: bool,
    sigma: f64,
}

impl KNNGraphBuilder {
    /// Create a new KNN graph builder
    pub fn new(n_neighbors: usize) -> Self {
        Self {
            n_neighbors,
            weighted: true,
            sigma: 1.0,
        }
    }

    /// Set whether to use weighted edges
    pub fn weighted(mut self, weighted: bool) -> Self {
        self.weighted = weighted;
        self
    }

    /// Set the kernel bandwidth
    pub fn sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
        self
    }
}

impl GraphBuilder for KNNGraphBuilder {
    fn build(&self, X: &ArrayView2<Float>) -> SklResult<Array2<f64>> {
        let n_samples = X.nrows();
        let mut graph = Array2::<f64>::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            let mut distances: Vec<(usize, f64)> = Vec::new();

            for j in 0..n_samples {
                if i != j {
                    let diff = &X.row(i) - &X.row(j);
                    let dist = diff.mapv(|x| x * x).sum().sqrt();
                    distances.push((j, dist));
                }
            }

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            for &(j, dist) in distances.iter().take(self.n_neighbors) {
                if self.weighted {
                    let weight = (-dist * dist / (2.0 * self.sigma * self.sigma)).exp();
                    graph[[i, j]] = weight;
                } else {
                    graph[[i, j]] = 1.0;
                }
            }
        }

        Ok(graph)
    }
}

/// Epsilon-ball graph builder
#[derive(Debug, Clone)]
pub struct EpsilonGraphBuilder {
    epsilon: f64,
    weighted: bool,
    sigma: f64,
}

impl EpsilonGraphBuilder {
    /// Create a new epsilon graph builder
    pub fn new(epsilon: f64) -> Self {
        Self {
            epsilon,
            weighted: true,
            sigma: 1.0,
        }
    }

    /// Set whether to use weighted edges
    pub fn weighted(mut self, weighted: bool) -> Self {
        self.weighted = weighted;
        self
    }

    /// Set the kernel bandwidth
    pub fn sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
        self
    }
}

impl GraphBuilder for EpsilonGraphBuilder {
    fn build(&self, X: &ArrayView2<Float>) -> SklResult<Array2<f64>> {
        let n_samples = X.nrows();
        let mut graph = Array2::<f64>::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in 0..n_samples {
                if i != j {
                    let diff = &X.row(i) - &X.row(j);
                    let dist = diff.mapv(|x| x * x).sum().sqrt();

                    if dist < self.epsilon {
                        if self.weighted {
                            let weight = (-dist * dist / (2.0 * self.sigma * self.sigma)).exp();
                            graph[[i, j]] = weight;
                        } else {
                            graph[[i, j]] = 1.0;
                        }
                    }
                }
            }
        }

        Ok(graph)
    }
}

/// Symmetrize graph transformation
#[derive(Debug, Clone)]
pub struct SymmetrizeTransform {
    method: String,
}

impl SymmetrizeTransform {
    /// Create a new symmetrize transform
    pub fn new(method: String) -> Self {
        Self { method }
    }
}

impl GraphTransform for SymmetrizeTransform {
    fn transform(&self, graph: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n = graph.nrows();
        let mut symmetric = graph.clone();

        match self.method.as_str() {
            "max" => {
                for i in 0..n {
                    for j in (i + 1)..n {
                        let value = graph[[i, j]].max(graph[[j, i]]);
                        symmetric[[i, j]] = value;
                        symmetric[[j, i]] = value;
                    }
                }
            }
            "average" => {
                for i in 0..n {
                    for j in (i + 1)..n {
                        let value = (graph[[i, j]] + graph[[j, i]]) / 2.0;
                        symmetric[[i, j]] = value;
                        symmetric[[j, i]] = value;
                    }
                }
            }
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown symmetrization method: {}",
                    self.method
                )));
            }
        }

        Ok(symmetric)
    }
}

/// Normalize graph transformation
#[derive(Debug, Clone)]
pub struct NormalizeTransform {
    method: String,
}

impl NormalizeTransform {
    /// Create a new normalize transform
    pub fn new(method: String) -> Self {
        Self { method }
    }
}

impl GraphTransform for NormalizeTransform {
    fn transform(&self, graph: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n = graph.nrows();
        let mut normalized = graph.clone();

        match self.method.as_str() {
            "row" => {
                for i in 0..n {
                    let row_sum: f64 = graph.row(i).sum();
                    if row_sum > 0.0 {
                        for j in 0..n {
                            normalized[[i, j]] /= row_sum;
                        }
                    }
                }
            }
            "symmetric" => {
                // D^{-1/2} A D^{-1/2}
                let mut degrees = Array1::<f64>::zeros(n);
                for i in 0..n {
                    degrees[i] = graph.row(i).sum();
                }

                for i in 0..n {
                    for j in 0..n {
                        if degrees[i] > 0.0 && degrees[j] > 0.0 {
                            normalized[[i, j]] = graph[[i, j]] / (degrees[i] * degrees[j]).sqrt();
                        }
                    }
                }
            }
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown normalization method: {}",
                    self.method
                )));
            }
        }

        Ok(normalized)
    }
}

/// Sparsify graph transformation
#[derive(Debug, Clone)]
pub struct SparsifyTransform {
    threshold: f64,
}

impl SparsifyTransform {
    /// Create a new sparsify transform
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl GraphTransform for SparsifyTransform {
    fn transform(&self, graph: &Array2<f64>) -> SklResult<Array2<f64>> {
        let mut sparse = graph.clone();
        let n = graph.nrows();

        for i in 0..n {
            for j in 0..n {
                if sparse[[i, j]] < self.threshold {
                    sparse[[i, j]] = 0.0;
                }
            }
        }

        Ok(sparse)
    }
}

/// Composable graph pipeline
#[derive(Clone)]
pub struct GraphPipeline {
    builder: Box<dyn GraphBuilderTrait>,
    transforms: Vec<Box<dyn GraphTransformTrait>>,
}

impl std::fmt::Debug for GraphPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphPipeline")
            .field("builder", &"Box<dyn GraphBuilderTrait>")
            .field(
                "transforms",
                &format!("{} transforms", self.transforms.len()),
            )
            .finish()
    }
}

// Helper traits with object safety
trait GraphBuilderTrait {
    fn build_graph(&self, X: &ArrayView2<Float>) -> SklResult<Array2<f64>>;
    fn clone_box(&self) -> Box<dyn GraphBuilderTrait>;
}

trait GraphTransformTrait {
    fn transform_graph(&self, graph: &Array2<f64>) -> SklResult<Array2<f64>>;
    fn clone_box(&self) -> Box<dyn GraphTransformTrait>;
}

impl<T: GraphBuilder + 'static> GraphBuilderTrait for T {
    fn build_graph(&self, X: &ArrayView2<Float>) -> SklResult<Array2<f64>> {
        self.build(X)
    }

    fn clone_box(&self) -> Box<dyn GraphBuilderTrait> {
        Box::new(self.clone())
    }
}

impl<T: GraphTransform + 'static> GraphTransformTrait for T {
    fn transform_graph(&self, graph: &Array2<f64>) -> SklResult<Array2<f64>> {
        self.transform(graph)
    }

    fn clone_box(&self) -> Box<dyn GraphTransformTrait> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn GraphBuilderTrait> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl Clone for Box<dyn GraphTransformTrait> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl GraphPipeline {
    /// Create a new graph pipeline
    pub fn new<B: GraphBuilder + 'static>(builder: B) -> Self {
        Self {
            builder: Box::new(builder),
            transforms: Vec::new(),
        }
    }

    /// Add a transformation to the pipeline
    pub fn add_transform<T: GraphTransform + 'static>(mut self, transform: T) -> Self {
        self.transforms.push(Box::new(transform));
        self
    }

    /// Build the graph with all transformations
    pub fn build(&self, X: &ArrayView2<Float>) -> SklResult<Array2<f64>> {
        let mut graph = self.builder.build_graph(X)?;

        for transform in &self.transforms {
            graph = transform.transform_graph(&graph)?;
        }

        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::array;

    #[test]
    #[allow(non_snake_case)]
    fn test_knn_graph_builder() {
        let X = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        let builder = KNNGraphBuilder::new(2).weighted(true).sigma(1.0);

        let graph = builder.build(&X.view()).unwrap();

        assert_eq!(graph.dim(), (4, 4));
        // Each node should be connected to 2 neighbors
        for i in 0..4 {
            let row_nonzero = graph.row(i).iter().filter(|&&x| x > 0.0).count();
            assert_eq!(row_nonzero, 2);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_epsilon_graph_builder() {
        let X = array![[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]];
        let builder = EpsilonGraphBuilder::new(2.0).weighted(false);

        let graph = builder.build(&X.view()).unwrap();

        assert_eq!(graph.dim(), (3, 3));
        // Nodes 0 and 1 should be connected (distance 1.0 < 2.0)
        assert_eq!(graph[[0, 1]], 1.0);
        assert_eq!(graph[[1, 0]], 1.0);
        // Nodes 0 and 2 should not be connected (distance 10.0 > 2.0)
        assert_eq!(graph[[0, 2]], 0.0);
    }

    #[test]
    fn test_symmetrize_transform() {
        let mut graph = Array2::<f64>::zeros((3, 3));
        graph[[0, 1]] = 1.0;
        graph[[1, 0]] = 2.0;
        graph[[1, 2]] = 3.0;
        graph[[2, 1]] = 4.0;

        let transform = SymmetrizeTransform::new("max".to_string());
        let symmetric = transform.transform(&graph).unwrap();

        assert_eq!(symmetric[[0, 1]], 2.0);
        assert_eq!(symmetric[[1, 0]], 2.0);
        assert_eq!(symmetric[[1, 2]], 4.0);
        assert_eq!(symmetric[[2, 1]], 4.0);
    }

    #[test]
    fn test_normalize_transform() {
        let mut graph = Array2::<f64>::zeros((3, 3));
        graph[[0, 1]] = 2.0;
        graph[[0, 2]] = 2.0;
        graph[[1, 0]] = 1.0;

        let transform = NormalizeTransform::new("row".to_string());
        let normalized = transform.transform(&graph).unwrap();

        // Row 0 should sum to 1.0
        let row_sum: f64 = normalized.row(0).sum();
        assert!((row_sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparsify_transform() {
        let mut graph = Array2::<f64>::zeros((3, 3));
        graph[[0, 1]] = 0.1;
        graph[[0, 2]] = 0.5;
        graph[[1, 2]] = 0.3;

        let transform = SparsifyTransform::new(0.2);
        let sparse = transform.transform(&graph).unwrap();

        assert_eq!(sparse[[0, 1]], 0.0); // Below threshold
        assert_eq!(sparse[[0, 2]], 0.5); // Above threshold
        assert_eq!(sparse[[1, 2]], 0.3); // Above threshold
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_graph_pipeline() {
        let X = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];

        let pipeline = GraphPipeline::new(KNNGraphBuilder::new(2).weighted(true))
            .add_transform(SymmetrizeTransform::new("average".to_string()))
            .add_transform(NormalizeTransform::new("row".to_string()));

        let graph = pipeline.build(&X.view()).unwrap();

        assert_eq!(graph.dim(), (4, 4));

        // Check that each row sums to approximately 1.0 (normalized)
        for i in 0..4 {
            let row_sum: f64 = graph.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-6 || row_sum == 0.0);
        }

        // Note: row normalization breaks symmetry, so we don't check for it
        // Let's check connectivity instead
        let total_edges: usize = graph.iter().filter(|&&x| x > 0.0).count();
        assert!(total_edges > 0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_symmetric_pipeline() {
        let X = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];

        let pipeline = GraphPipeline::new(KNNGraphBuilder::new(1).weighted(true))
            .add_transform(SymmetrizeTransform::new("max".to_string()));

        let graph = pipeline.build(&X.view()).unwrap();

        assert_eq!(graph.dim(), (3, 3));

        // Check symmetry (without row normalization)
        for i in 0..3 {
            for j in 0..3 {
                assert!((graph[[i, j]] - graph[[j, i]]).abs() < 1e-10);
            }
        }
    }
}
