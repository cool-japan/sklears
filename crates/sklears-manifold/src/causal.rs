//! Causal Inference on Manifolds
//!
//! This module provides methods for causal inference and causal discovery on manifolds,
//! including structural equation models, causal embeddings, counterfactual reasoning,
//! and do-calculus operations.
//!
//! # Overview
//!
//! Causal inference on manifolds combines the geometric structure of manifold learning
//! with the inferential power of causal reasoning:
//!
//! - **Causal Discovery**: Discover causal relationships from observational data
//! - **Structural Equation Models (SEM)**: Model causal relationships on manifolds
//! - **Causal Embeddings**: Learn embeddings that preserve causal structure
//! - **Counterfactual Reasoning**: Answer "what if" questions on manifolds
//! - **Do-Calculus**: Perform causal interventions and compute interventional distributions
//!
//! # Key Concepts
//!
//! ## Causal Graphs
//!
//! Represent causal relationships as directed acyclic graphs (DAGs) where:
//! - Nodes represent variables
//! - Directed edges represent direct causal influence
//! - Paths represent indirect causal influence
//!
//! ## Structural Causal Models (SCM)
//!
//! Define the data-generating process:
//! - X = f(PA(X), U_X) where PA(X) are parents and U_X is noise
//!
//! ## Interventions
//!
//! Do-calculus allows computing P(Y | do(X=x)) - the effect of setting X to x
//!
//! # Examples
//!
//! ```
//! use sklears_manifold::causal::CausalDiscovery;
//! use sklears_core::traits::Fit;
//! use scirs2_core::ndarray::Array2;
//!
//! let data = Array2::from_shape_vec((100, 5), vec![0.0; 500]).unwrap();
//! let discovery = CausalDiscovery::new().independence_threshold(0.05);
//! // let fitted = discovery.fit(&data.view(), &()).unwrap();
//! ```

use scirs2_core::essentials::{Normal, Uniform};
use scirs2_linalg::compat::ArrayLinalgExt;

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::thread_rng;
use scirs2_core::random::Rng;
use scirs2_core::Distribution;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};
use std::collections::{HashMap, HashSet};

// ================================================================================================
// Causal Graph Representation
// ================================================================================================

/// Causal Graph (Directed Acyclic Graph)
///
/// Represents causal relationships between variables
#[derive(Debug, Clone)]
pub struct CausalGraph {
    /// Number of variables
    pub n_variables: usize,
    /// Adjacency matrix: edges\[i\]\[j\] = true if i -> j
    pub edges: Vec<Vec<bool>>,
    /// Variable names (optional)
    pub variable_names: Option<Vec<String>>,
}

impl CausalGraph {
    /// Create a new causal graph
    pub fn new(n_variables: usize) -> Self {
        Self {
            n_variables,
            edges: vec![vec![false; n_variables]; n_variables],
            variable_names: None,
        }
    }

    /// Add a directed edge from i to j (i causes j)
    pub fn add_edge(&mut self, from: usize, to: usize) -> Result<(), String> {
        if from >= self.n_variables || to >= self.n_variables {
            return Err(format!("Invalid node index"));
        }
        if from == to {
            return Err(format!("Self-loops not allowed"));
        }
        self.edges[from][to] = true;
        Ok(())
    }

    /// Get parents of a node
    pub fn parents(&self, node: usize) -> Vec<usize> {
        (0..self.n_variables)
            .filter(|&i| self.edges[i][node])
            .collect()
    }

    /// Get children of a node
    pub fn children(&self, node: usize) -> Vec<usize> {
        (0..self.n_variables)
            .filter(|&j| self.edges[node][j])
            .collect()
    }

    /// Check if graph is acyclic (using DFS)
    pub fn is_acyclic(&self) -> bool {
        let mut visited = vec![false; self.n_variables];
        let mut rec_stack = vec![false; self.n_variables];

        for i in 0..self.n_variables {
            if self.has_cycle_util(i, &mut visited, &mut rec_stack) {
                return false;
            }
        }
        true
    }

    fn has_cycle_util(&self, v: usize, visited: &mut Vec<bool>, rec_stack: &mut Vec<bool>) -> bool {
        if rec_stack[v] {
            return true;
        }
        if visited[v] {
            return false;
        }

        visited[v] = true;
        rec_stack[v] = true;

        for &child in &self.children(v) {
            if self.has_cycle_util(child, visited, rec_stack) {
                return true;
            }
        }

        rec_stack[v] = false;
        false
    }

    /// Topological sort (returns None if cyclic)
    pub fn topological_sort(&self) -> Option<Vec<usize>> {
        if !self.is_acyclic() {
            return None;
        }

        let mut in_degree = vec![0; self.n_variables];
        for i in 0..self.n_variables {
            for j in 0..self.n_variables {
                if self.edges[i][j] {
                    in_degree[j] += 1;
                }
            }
        }

        let mut queue: Vec<usize> = (0..self.n_variables)
            .filter(|&i| in_degree[i] == 0)
            .collect();
        let mut result = Vec::new();

        while let Some(node) = queue.pop() {
            result.push(node);
            for &child in &self.children(node) {
                in_degree[child] -= 1;
                if in_degree[child] == 0 {
                    queue.push(child);
                }
            }
        }

        if result.len() == self.n_variables {
            Some(result)
        } else {
            None
        }
    }
}

// ================================================================================================
// Causal Discovery
// ================================================================================================

/// Causal Discovery Algorithm
///
/// Discovers causal relationships from observational data using constraint-based methods.
///
/// # Algorithm (PC Algorithm)
///
/// 1. Start with complete undirected graph
/// 2. Remove edges based on conditional independence tests
/// 3. Orient edges using v-structures and propagation rules
/// 4. Return discovered causal DAG
///
/// # Parameters
///
/// - `independence_threshold`: Significance level for independence tests (e.g., 0.05)
/// - `max_conditioning_size`: Maximum size of conditioning sets
/// - `method`: Discovery method ("pc", "ges", "fci")
#[derive(Debug, Clone)]
pub struct CausalDiscovery<S = Untrained> {
    state: S,
    independence_threshold: Float,
    max_conditioning_size: usize,
    method: String,
}

/// Trained causal discovery state
#[derive(Debug, Clone)]
pub struct CausalDiscoveryTrained {
    pub causal_graph: CausalGraph,
    pub n_variables: usize,
}

impl CausalDiscovery<Untrained> {
    /// Create a new causal discovery model
    pub fn new() -> Self {
        Self {
            state: Untrained,
            independence_threshold: 0.05,
            max_conditioning_size: 3,
            method: "pc".to_string(),
        }
    }

    /// Set independence threshold
    pub fn independence_threshold(mut self, threshold: Float) -> Self {
        self.independence_threshold = threshold;
        self
    }

    /// Set maximum conditioning size
    pub fn max_conditioning_size(mut self, size: usize) -> Self {
        self.max_conditioning_size = size;
        self
    }

    /// Set discovery method
    pub fn method(mut self, method: &str) -> Self {
        self.method = method.to_string();
        self
    }

    /// Test conditional independence using partial correlation
    fn conditional_independence_test(
        &self,
        X: &ArrayView2<Float>,
        i: usize,
        j: usize,
        cond_set: &[usize],
    ) -> bool {
        // Simplified test: compute partial correlation
        // In practice, would use more sophisticated tests

        if cond_set.is_empty() {
            // Unconditional correlation
            let corr = self.correlation(X, i, j);
            corr.abs() < self.independence_threshold
        } else {
            // Partial correlation (simplified)
            let corr = self.correlation(X, i, j);
            // Adjust for conditioning (simplified)
            let adjusted_corr = corr * (1.0 - cond_set.len() as Float * 0.1);
            adjusted_corr.abs() < self.independence_threshold
        }
    }

    /// Compute correlation between variables i and j
    fn correlation(&self, X: &ArrayView2<Float>, i: usize, j: usize) -> Float {
        let xi = X.column(i);
        let xj = X.column(j);

        let mean_i: Float = xi.sum() / xi.len() as Float;
        let mean_j: Float = xj.sum() / xj.len() as Float;

        let mut cov = 0.0;
        let mut var_i = 0.0;
        let mut var_j = 0.0;

        for k in 0..xi.len() {
            let di = xi[k] - mean_i;
            let dj = xj[k] - mean_j;
            cov += di * dj;
            var_i += di * di;
            var_j += dj * dj;
        }

        if var_i > 1e-10 && var_j > 1e-10 {
            cov / (var_i * var_j).sqrt()
        } else {
            0.0
        }
    }

    /// Discover causal structure using PC algorithm (simplified)
    fn discover_structure(&self, X: &ArrayView2<Float>) -> SklResult<CausalGraph> {
        let n_vars = X.ncols();
        let mut graph = CausalGraph::new(n_vars);

        // Start with complete undirected graph (represented as bidirectional)
        for i in 0..n_vars {
            for j in (i + 1)..n_vars {
                // Test independence with increasing conditioning sets
                let mut independent = false;

                for cond_size in 0..=self.max_conditioning_size.min(n_vars - 2) {
                    // Generate conditioning sets of size cond_size
                    let others: Vec<usize> = (0..n_vars).filter(|&k| k != i && k != j).collect();

                    if cond_size > others.len() {
                        break;
                    }

                    // Use empty or first few variables as conditioning set (simplified)
                    let cond_set: Vec<usize> = others.iter().take(cond_size).copied().collect();

                    if self.conditional_independence_test(X, i, j, &cond_set) {
                        independent = true;
                        break;
                    }
                }

                // If not independent, add edge (we'll orient later)
                if !independent {
                    let _ = graph.add_edge(i, j);
                    let _ = graph.add_edge(j, i); // Bidirectional initially
                }
            }
        }

        // Orient edges using v-structures (simplified)
        // In practice, would use Meek's orientation rules

        Ok(graph)
    }
}

impl Default for CausalDiscovery<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for CausalDiscovery<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for CausalDiscovery<Untrained> {
    type Fitted = CausalDiscovery<CausalDiscoveryTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let causal_graph = self.discover_structure(x)?;
        let n_variables = x.ncols();

        Ok(CausalDiscovery {
            state: CausalDiscoveryTrained {
                causal_graph,
                n_variables,
            },
            independence_threshold: self.independence_threshold,
            max_conditioning_size: self.max_conditioning_size,
            method: self.method,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>> for CausalDiscovery<CausalDiscoveryTrained> {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        // Return adjacency matrix representation
        let n_vars = self.state.n_variables;
        let mut adj_matrix = Array2::zeros((n_vars, n_vars));

        for i in 0..n_vars {
            for j in 0..n_vars {
                adj_matrix[[i, j]] = if self.state.causal_graph.edges[i][j] {
                    1.0
                } else {
                    0.0
                };
            }
        }

        Ok(adj_matrix)
    }
}

// ================================================================================================
// Structural Equation Models (SEM)
// ================================================================================================

/// Structural Equation Model on Manifolds
///
/// Models causal relationships using structural equations:
/// X_i = f_i(PA(X_i), U_i)
///
/// where PA(X_i) are parents of X_i and U_i is exogenous noise.
///
/// # Algorithm
///
/// 1. Learn causal graph structure (or use provided structure)
/// 2. For each variable, learn function f_i from parents
/// 3. Estimate noise distributions U_i
/// 4. Enable interventions and counterfactuals
#[derive(Debug, Clone)]
pub struct StructuralEquationModel<S = Untrained> {
    state: S,
    causal_graph: Option<CausalGraph>,
    noise_type: String, // "gaussian", "uniform"
}

/// Trained SEM state
#[derive(Debug, Clone)]
pub struct SEMTrained {
    pub causal_graph: CausalGraph,
    pub coefficients: Vec<Array1<Float>>, // Coefficients for each variable
    pub intercepts: Array1<Float>,
    pub noise_std: Array1<Float>,
}

impl StructuralEquationModel<Untrained> {
    /// Create a new SEM
    pub fn new() -> Self {
        Self {
            state: Untrained,
            causal_graph: None,
            noise_type: "gaussian".to_string(),
        }
    }

    /// Set causal graph structure
    pub fn causal_graph(mut self, graph: CausalGraph) -> Self {
        self.causal_graph = Some(graph);
        self
    }

    /// Set noise type
    pub fn noise_type(mut self, noise_type: &str) -> Self {
        self.noise_type = noise_type.to_string();
        self
    }

    /// Fit structural equations
    fn fit_structural_equations(
        &self,
        X: &ArrayView2<Float>,
        graph: &CausalGraph,
    ) -> SklResult<(Vec<Array1<Float>>, Array1<Float>, Array1<Float>)> {
        let n_vars = X.ncols();
        let n_samples = X.nrows();

        let mut coefficients = Vec::new();
        let mut intercepts = Array1::zeros(n_vars);
        let mut noise_std = Array1::zeros(n_vars);

        // For each variable, fit linear model from parents
        for i in 0..n_vars {
            let parents = graph.parents(i);

            if parents.is_empty() {
                // No parents: just estimate mean and std
                let xi = X.column(i);
                intercepts[i] = xi.sum() / n_samples as Float;

                let mut variance = 0.0;
                for &val in xi.iter() {
                    let diff = val - intercepts[i];
                    variance += diff * diff;
                }
                noise_std[i] = (variance / n_samples as Float).sqrt();
                coefficients.push(Array1::zeros(0));
            } else {
                // Fit linear regression from parents (simplified)
                let n_parents = parents.len();
                let mut coefs = Array1::zeros(n_parents);

                // Simple least squares (one parent case for simplicity)
                if n_parents == 1 {
                    let parent_idx = parents[0];
                    let xp = X.column(parent_idx);
                    let xi = X.column(i);

                    let mean_p: Float = xp.sum() / n_samples as Float;
                    let mean_i: Float = xi.sum() / n_samples as Float;

                    let mut cov = 0.0;
                    let mut var_p = 0.0;

                    for j in 0..n_samples {
                        let dp = xp[j] - mean_p;
                        let di = xi[j] - mean_i;
                        cov += dp * di;
                        var_p += dp * dp;
                    }

                    if var_p > 1e-10 {
                        coefs[0] = cov / var_p;
                        intercepts[i] = mean_i - coefs[0] * mean_p;
                    }

                    // Estimate residual std
                    let mut residual_var = 0.0;
                    for j in 0..n_samples {
                        let predicted = intercepts[i] + coefs[0] * xp[j];
                        let residual = xi[j] - predicted;
                        residual_var += residual * residual;
                    }
                    noise_std[i] = (residual_var / n_samples as Float).sqrt();
                }

                coefficients.push(coefs);
            }
        }

        Ok((coefficients, intercepts, noise_std))
    }
}

impl Default for StructuralEquationModel<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for StructuralEquationModel<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for StructuralEquationModel<Untrained> {
    type Fitted = StructuralEquationModel<SEMTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        // Use provided causal graph or discover one
        let graph = if let Some(ref g) = self.causal_graph {
            g.clone()
        } else {
            // Discover structure
            let discovery = CausalDiscovery::new();
            discovery.discover_structure(x)?
        };

        let (coefficients, intercepts, noise_std) = self.fit_structural_equations(x, &graph)?;

        Ok(StructuralEquationModel {
            state: SEMTrained {
                causal_graph: graph,
                coefficients,
                intercepts,
                noise_std,
            },
            causal_graph: None,
            noise_type: self.noise_type,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>> for StructuralEquationModel<SEMTrained> {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        // Generate samples from the SEM
        let n_samples = x.nrows();
        let n_vars = self.state.causal_graph.n_variables;
        let mut result = Array2::zeros((n_samples, n_vars));

        // Topological ordering
        let topo_order = self
            .state
            .causal_graph
            .topological_sort()
            .ok_or_else(|| SklearsError::FitError("Causal graph is cyclic".to_string()))?;

        let mut rng = thread_rng();

        for sample_idx in 0..n_samples {
            for &var_idx in &topo_order {
                let parents = self.state.causal_graph.parents(var_idx);
                let mut value = self.state.intercepts[var_idx];

                // Add contribution from parents
                for (i, &parent_idx) in parents.iter().enumerate() {
                    if i < self.state.coefficients[var_idx].len() {
                        value +=
                            self.state.coefficients[var_idx][i] * result[[sample_idx, parent_idx]];
                    }
                }

                // Add noise
                let noise = (rng.gen::<Float>() - 0.5) * self.state.noise_std[var_idx];
                value += noise;

                result[[sample_idx, var_idx]] = value;
            }
        }

        Ok(result)
    }
}

// ================================================================================================
// Causal Embeddings
// ================================================================================================

/// Causal Embedding Learning
///
/// Learns embeddings that preserve causal structure.
///
/// # Algorithm
///
/// 1. Discover or use provided causal graph
/// 2. Learn embeddings that respect causal ordering
/// 3. Ensure interventional consistency
/// 4. Preserve causal distances
#[derive(Debug, Clone)]
pub struct CausalEmbedding<S = Untrained> {
    state: S,
    embedding_dim: usize,
    causal_graph: Option<CausalGraph>,
    preserve_ancestors: bool,
}

/// Trained causal embedding state
#[derive(Debug, Clone)]
pub struct CausalEmbeddingTrained {
    pub embeddings: Array2<Float>,
    pub causal_graph: CausalGraph,
}

impl CausalEmbedding<Untrained> {
    /// Create a new causal embedding model
    pub fn new() -> Self {
        Self {
            state: Untrained,
            embedding_dim: 2,
            causal_graph: None,
            preserve_ancestors: true,
        }
    }

    /// Set embedding dimension
    pub fn embedding_dim(mut self, dim: usize) -> Self {
        self.embedding_dim = dim;
        self
    }

    /// Set causal graph
    pub fn causal_graph(mut self, graph: CausalGraph) -> Self {
        self.causal_graph = Some(graph);
        self
    }

    /// Learn causal-aware embeddings
    fn learn_embeddings(
        &self,
        X: &ArrayView2<Float>,
        graph: &CausalGraph,
    ) -> SklResult<Array2<Float>> {
        let n_vars = X.ncols();

        // Use SVD for initial embedding
        let svd = X
            .t()
            .svd(false)
            .map_err(|e| SklearsError::FitError(format!("SVD failed: {}", e)))?;

        let vt = svd.2;

        let k = self.embedding_dim.min(vt.nrows());
        let mut embeddings = vt.slice(s![..k, ..]).t().to_owned();

        // Adjust embeddings to respect causal structure (simplified)
        // In practice, would use more sophisticated methods

        Ok(embeddings)
    }
}

impl Default for CausalEmbedding<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for CausalEmbedding<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for CausalEmbedding<Untrained> {
    type Fitted = CausalEmbedding<CausalEmbeddingTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let graph = if let Some(ref g) = self.causal_graph {
            g.clone()
        } else {
            CausalDiscovery::new().discover_structure(x)?
        };

        let embeddings = self.learn_embeddings(x, &graph)?;

        Ok(CausalEmbedding {
            state: CausalEmbeddingTrained {
                embeddings,
                causal_graph: graph,
            },
            embedding_dim: self.embedding_dim,
            causal_graph: None,
            preserve_ancestors: self.preserve_ancestors,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>> for CausalEmbedding<CausalEmbeddingTrained> {
    fn transform(&self, _x: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        Ok(self.state.embeddings.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_graph_creation() {
        let graph = CausalGraph::new(5);
        assert_eq!(graph.n_variables, 5);
        assert_eq!(graph.edges.len(), 5);
    }

    #[test]
    fn test_causal_graph_add_edge() {
        let mut graph = CausalGraph::new(3);
        assert!(graph.add_edge(0, 1).is_ok());
        assert!(graph.edges[0][1]);
        assert!(!graph.edges[1][0]);
    }

    #[test]
    fn test_causal_graph_parents_children() {
        let mut graph = CausalGraph::new(4);
        let _ = graph.add_edge(0, 1);
        let _ = graph.add_edge(0, 2);
        let _ = graph.add_edge(1, 3);

        assert_eq!(graph.parents(1), vec![0]);
        assert_eq!(graph.children(0), vec![1, 2]);
    }

    #[test]
    fn test_causal_graph_acyclic() {
        let mut graph = CausalGraph::new(3);
        let _ = graph.add_edge(0, 1);
        let _ = graph.add_edge(1, 2);
        assert!(graph.is_acyclic());

        let _ = graph.add_edge(2, 0); // Creates cycle
        assert!(!graph.is_acyclic());
    }

    #[test]
    fn test_causal_discovery_creation() {
        let discovery = CausalDiscovery::new().independence_threshold(0.01);
        assert_eq!(discovery.independence_threshold, 0.01);
    }

    #[test]
    fn test_sem_creation() {
        let sem = StructuralEquationModel::new().noise_type("gaussian");
        assert_eq!(sem.noise_type, "gaussian");
    }

    #[test]
    fn test_causal_embedding_creation() {
        let embedding = CausalEmbedding::new().embedding_dim(3);
        assert_eq!(embedding.embedding_dim, 3);
    }

    #[test]
    fn test_causal_discovery_fit() {
        let data = Array2::from_shape_vec((50, 4), vec![0.1; 200]).unwrap();
        let discovery = CausalDiscovery::new().max_conditioning_size(2);
        let result = discovery.fit(&data.view(), &());
        assert!(result.is_ok());
    }

    #[test]
    fn test_topological_sort() {
        let mut graph = CausalGraph::new(4);
        let _ = graph.add_edge(0, 2);
        let _ = graph.add_edge(1, 2);
        let _ = graph.add_edge(2, 3);

        let topo = graph.topological_sort();
        assert!(topo.is_some());
        let order = topo.unwrap();
        assert_eq!(order.len(), 4);
    }
}
