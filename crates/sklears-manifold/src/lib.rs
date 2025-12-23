#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(unused_mut)]
#![allow(unused_doc_comments)]
#![allow(unused_comparisons)]
#![allow(unused_must_use)]
#![allow(mismatched_lifetime_syntaxes)]
#![allow(ambiguous_glob_reexports)]
#![allow(unexpected_cfgs)]
//! Manifold learning algorithms (t-SNE, Isomap, etc.)
//!
//! This module is part of sklears, providing scikit-learn compatible
//! machine learning algorithms in Rust.

// #![warn(missing_docs)]

use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::thread_rng;
use scirs2_core::random::Rng;
use scirs2_core::random::SeedableRng;
use scirs2_core::Distribution;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};

// TSNE moved to tsne.rs module

// Isomap moved to isomap.rs module

// LocallyLinearEmbedding moved to lle.rs module

// LaplacianEigenmaps moved to laplacian_eigenmaps.rs module

// LaplacianEigenmaps implementations moved to laplacian_eigenmaps.rs module

// LaplacianEigenmaps implementations and LaplacianTrained moved to laplacian_eigenmaps.rs module

// UMAP moved to umap.rs module

// UMAP implementations and UmapTrained moved to umap.rs module
// DiffusionMaps implementation and DiffusionMapsTrained moved to diffusion_maps.rs module
// HLLE implementation and HessianLleTrained moved to hessian_lle.rs module

#[allow(non_snake_case)]
#[cfg(test)]
mod tests;

pub mod quality_metrics;

/// Re-export quality metrics for convenience
pub use quality_metrics::*;

// =====================================================================================
// STRESS TESTING AND SCALABILITY MODULE
// =====================================================================================

pub mod stress_testing;
pub use stress_testing::*;

// =====================================================================================
// GEODESIC DISTANCE COMPUTATION MODULE
// =====================================================================================

pub mod geodesic_distance;
pub use geodesic_distance::*;

// =====================================================================================
// DIFFUSION DISTANCE MODULE
// =====================================================================================

pub mod diffusion_distance;
pub use diffusion_distance::*;

// =====================================================================================
// RIEMANNIAN GEOMETRY MODULE
// =====================================================================================

pub mod riemannian;
pub use riemannian::*;

// =====================================================================================
// TOPOLOGICAL DATA ANALYSIS MODULE
// =====================================================================================

pub mod topological;
pub use topological::*;

// =====================================================================================
// RANDOM WALK EMBEDDINGS
// =====================================================================================

// =====================================================================================
// NODE2VEC AND DEEPWALK ALGORITHMS
// =====================================================================================

// =====================================================================================

/// Sparse Coding for manifold learning
///
/// Sparse coding learns a dictionary of basis vectors such that each data point
/// can be represented as a sparse linear combination of these basis vectors.
/// This is particularly useful for manifold learning when the data lies on
/// a low-dimensional manifold that can be sparsely represented.
///
/// # Parameters
///
/// * `n_components` - Number of dictionary atoms
/// * `alpha` - Sparsity regularization parameter
/// * `max_iter` - Maximum number of iterations
/// * `tol` - Tolerance for convergence
/// * `random_state` - Random seed for reproducibility
///
/// # Examples
///
/// ```
/// use sklears_manifold::SparseCoding;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
///
/// let sc = SparseCoding::new()
///     .n_components(2)
///     .alpha(0.1);
///
/// let fitted = sc.fit(&x.view(), &()).unwrap();
/// let embedded = fitted.transform(&x.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SparseCoding<S = Untrained> {
    state: S,
    n_components: usize,
    alpha: f64,
    max_iter: usize,
    tol: f64,
    random_state: Option<u64>,
}

impl Default for SparseCoding<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseCoding<Untrained> {
    /// Create a new SparseCoding instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 100,
            alpha: 1.0,
            max_iter: 1000,
            tol: 1e-8,
            random_state: None,
        }
    }

    /// Set the number of dictionary atoms
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the sparsity regularization parameter
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the tolerance for convergence
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Soft thresholding function for sparse coding
    fn soft_threshold(x: f64, lambda: f64) -> f64 {
        if x > lambda {
            x - lambda
        } else if x < -lambda {
            x + lambda
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone)]
pub struct SCTrained {
    dictionary: Array2<f64>,
    mean: Array1<f64>,
}

impl Estimator for SparseCoding<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for SparseCoding<Untrained> {
    type Fitted = SparseCoding<SCTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if self.n_components > n_features {
            return Err(SklearsError::InvalidInput(
                "n_components cannot be larger than n_features".to_string(),
            ));
        }

        // Convert to f64 and center the data
        let x_f64 = x.mapv(|v| v);
        let mean = x_f64.mean_axis(Axis(0)).unwrap();
        let x_centered = &x_f64 - &mean.view().broadcast(x_f64.dim()).unwrap();

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().random::<u64>())
        };

        // Initialize dictionary with random normalized vectors
        let mut dictionary = Array2::<f64>::zeros((n_features, self.n_components));
        for mut col in dictionary.columns_mut() {
            for elem in col.iter_mut() {
                *elem = rng.sample(scirs2_core::StandardNormal);
            }
            // Normalize the column
            let norm = col.dot(&col).sqrt();
            if norm > 0.0 {
                col /= norm;
            }
        }

        // Iterative dictionary learning using coordinate descent
        for iter in 0..self.max_iter {
            let mut max_change = 0.0f64;

            // Update dictionary atoms one at a time
            for k in 0..self.n_components {
                // Compute residual without atom k
                let mut residual = x_centered.clone();
                for j in 0..self.n_components {
                    if j != k {
                        let atom_j = dictionary.column(j);
                        // Compute sparse codes for atom j
                        let mut codes_j = Array1::zeros(n_samples);
                        for i in 0..n_samples {
                            let dot_product = residual.row(i).dot(&atom_j);
                            codes_j[i] = Self::soft_threshold(dot_product, self.alpha);
                        }

                        // Subtract contribution of atom j
                        for i in 0..n_samples {
                            let mut row = residual.row_mut(i);
                            row.scaled_add(-codes_j[i], &atom_j);
                        }
                    }
                }

                // Update atom k
                let mut new_atom = Array1::zeros(n_features);
                let mut total_code = 0.0;

                for i in 0..n_samples {
                    let code_k = Self::soft_threshold(
                        residual.row(i).dot(&dictionary.column(k)),
                        self.alpha,
                    );
                    if code_k.abs() > 1e-12 {
                        new_atom.scaled_add(code_k, &residual.row(i));
                        total_code += code_k * code_k;
                    }
                }

                if total_code > 1e-12 {
                    new_atom /= total_code;
                    // Normalize
                    let norm = new_atom.dot(&new_atom).sqrt();
                    if norm > 1e-12 {
                        new_atom /= norm;
                    }

                    // Check convergence
                    let change = (&new_atom - &dictionary.column(k)).mapv(|x| x.abs()).sum();
                    max_change = max_change.max(change);

                    // Update dictionary
                    dictionary.column_mut(k).assign(&new_atom);
                }
            }

            // Check convergence
            if max_change < self.tol {
                break;
            }
        }

        Ok(SparseCoding {
            state: SCTrained { dictionary, mean },
            n_components: self.n_components,
            alpha: self.alpha,
            max_iter: self.max_iter,
            tol: self.tol,
            random_state: self.random_state,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<f64>> for SparseCoding<SCTrained> {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let (n_samples, _) = x.dim();
        let x_f64 = x.mapv(|v| v);
        let x_centered = &x_f64 - &self.state.mean.view().broadcast(x_f64.dim()).unwrap();

        // Compute sparse codes using coordinate descent
        let mut codes = Array2::zeros((n_samples, self.n_components));

        for i in 0..n_samples {
            let sample = x_centered.row(i);
            let mut code = Array1::<f64>::zeros(self.n_components);

            // Coordinate descent for sparse coding
            for _ in 0..100 {
                // Limited iterations for transform
                let mut max_change = 0.0f64;

                for k in 0..self.n_components {
                    // Compute residual without component k
                    let mut residual = sample.to_owned();
                    for j in 0..self.n_components {
                        if j != k {
                            let atom_j = self.state.dictionary.column(j);
                            residual.scaled_add(-code[j], &atom_j);
                        }
                    }

                    // Update component k
                    let atom_k = self.state.dictionary.column(k);
                    let new_code_k =
                        SparseCoding::soft_threshold(residual.dot(&atom_k), self.alpha);
                    let change = (new_code_k - code[k]).abs();
                    max_change = max_change.max(change);
                    code[k] = new_code_k;
                }

                if max_change < 1e-6 {
                    break;
                }
            }

            codes.row_mut(i).assign(&code);
        }

        Ok(codes)
    }
}

// MINI-BATCH EMBEDDING METHODS FOR SCALABILITY
// =====================================================================================

/// t-SNE (t-distributed Stochastic Neighbor Embedding) module
pub mod tsne;

/// Isomap (Isometric Mapping) module
pub mod isomap;

/// LLE (Locally Linear Embedding) module
pub mod lle;

/// MDS (Multidimensional Scaling) module
pub mod mds;

/// Laplacian Eigenmaps module
pub mod laplacian_eigenmaps;

/// UMAP module
pub mod umap;

/// Diffusion Maps module
pub mod diffusion_maps;

/// HLLE (Hessian LLE) module
pub mod hessian_lle;

/// LTSA (Local Tangent Space Alignment) module
pub mod ltsa;

/// MVU (Maximum Variance Unfolding) module
pub mod mvu;

/// SNE (Stochastic Neighbor Embedding) module
pub mod sne;

/// SymmetricSNE (Symmetric Stochastic Neighbor Embedding) module
pub mod symmetric_sne;

/// ParametricTSNE (Parametric t-SNE) module
pub mod parametric_tsne;

/// HeavyTailedSymmetricSNE (Heavy-Tailed Symmetric SNE) module
pub mod heavy_tailed_symmetric_sne;

/// Spectral Embedding module
pub mod spectral_embedding;

/// Random Walk Embedding module
pub mod random_walk_embedding;

/// Node2Vec algorithm module
pub mod node2vec;

/// DeepWalk algorithm module
pub mod deepwalk;

/// Dictionary Learning module
pub mod dictionary_learning;

/// Mini-batch t-SNE module
pub mod minibatch_tsne;

/// Mini-batch UMAP module
pub mod minibatch_umap;

/// Distance methods and kernel functions module
pub mod distance_kernels;

/// Graph Neural Networks module
pub mod graph_neural_networks;

/// Random projection methods module
pub mod random_projections;

/// Similarity learning module
pub mod similarity;

/// Hierarchical manifold learning module
pub mod hierarchical;

/// Temporal manifold learning module
pub mod temporal;

/// Robust manifold learning module
pub mod robust;

/// Re-export t-SNE utilities for convenience
pub use tsne::{TsneTrained, TSNE};

/// Re-export Isomap utilities for convenience
pub use isomap::{Isomap, IsomapTrained};

/// Re-export LLE utilities for convenience
pub use lle::{LleTrained, LocallyLinearEmbedding};

/// Re-export MDS utilities for convenience
pub use mds::{MdsTrained, MDS};

/// Re-export Laplacian Eigenmaps utilities for convenience
pub use laplacian_eigenmaps::{LaplacianEigenmaps, LaplacianTrained};

/// Re-export UMAP utilities for convenience
pub use umap::{UmapTrained, UMAP};

/// Re-export Diffusion Maps utilities for convenience
pub use diffusion_maps::{DiffusionMaps, DiffusionMapsTrained};

/// Re-export DeepWalk types for convenience
pub use deepwalk::{DeepWalk, DeepWalkTrained};
/// Re-export DictionaryLearning types for convenience
pub use dictionary_learning::{DLTrained, DictionaryLearning};
/// Re-export HeavyTailedSymmetricSNE types for convenience
pub use heavy_tailed_symmetric_sne::{HeavyTailedSymmetricSNE, HeavyTailedSymmetricSneTrained};
/// Re-export HLLE utilities for convenience
pub use hessian_lle::{HessianLLE, HessianLleTrained};
/// Re-export LTSA types for convenience
pub use ltsa::{LtsaTrained, LTSA};
/// Re-export MiniBatchTSNE types for convenience
pub use minibatch_tsne::{MBTSNETrained, MiniBatchTSNE};
/// Re-export MiniBatchUMAP types for convenience
pub use minibatch_umap::{MBUMAPTrained, MiniBatchUMAP};
/// Re-export MVU types for convenience
pub use mvu::{MvuTrained, MVU};
/// Re-export Node2Vec types for convenience
pub use node2vec::{Node2Vec, Node2VecTrained};
/// Re-export ParametricTSNE types for convenience
pub use parametric_tsne::{ParametricTSNE, ParametricTsneTrained};
/// Re-export RandomWalkEmbedding types for convenience
pub use random_walk_embedding::{RandomWalkEmbedding, RandomWalkEmbeddingTrained};
/// Re-export SNE types for convenience
pub use sne::{SneTrained, SNE};
/// Re-export SpectralEmbedding types for convenience
pub use spectral_embedding::{SpectralEmbedding, SpectralEmbeddingTrained};
/// Re-export SymmetricSNE types for convenience
pub use symmetric_sne::{SymmetricSNE, SymmetricSneTrained};

/// Re-export distance methods and kernel functions for convenience
pub use distance_kernels::*;

/// Re-export Graph Neural Networks for convenience
pub use graph_neural_networks::*;

/// Re-export random projection methods for convenience
pub use random_projections::*;

/// Re-export similarity learning utilities for convenience
pub use similarity::*;

/// Re-export hierarchical manifold learning utilities for convenience
pub use hierarchical::*;

/// Re-export temporal manifold learning utilities for convenience
pub use temporal::*;

/// Re-export robust manifold learning utilities for convenience
pub use robust::*;

/// Multi-view learning module
pub mod multi_view;

/// Nyström approximation module
pub mod nystrom;

/// Compressed sensing module
pub mod compressed_sensing;

/// Parallel k-nearest neighbors module
pub mod parallel_knn;

/// Stochastic manifold learning module
pub mod stochastic;

/// Re-export multi-view learning utilities for convenience
pub use multi_view::*;

/// Re-export Nyström approximation utilities for convenience
pub use nystrom::*;

/// Re-export compressed sensing utilities for convenience
pub use compressed_sensing::*;

/// Re-export parallel KNN utilities for convenience
pub use parallel_knn::*;

/// Re-export stochastic manifold learning utilities for convenience
pub use stochastic::*;

/// Benchmark datasets module
pub mod benchmark_datasets;

/// Timing utilities module
pub mod timing_utilities;

/// Memory profiler module
pub mod memory_profiler;

/// SIMD-optimized distance computations
pub mod simd_distance;

/// Validation framework for hyperparameter tuning
pub mod validation;

/// Visualization integration utilities
pub mod visualization;

/// GPU-accelerated methods for manifold learning
#[cfg(feature = "gpu")]
pub mod gpu_acceleration;

/// Type-safe manifold abstractions with phantom types
pub mod type_safe_manifolds;

/// Zero-cost abstractions for manifold learning
pub mod zero_cost_abstractions;

/// Comparison tests against reference implementations
pub mod reference_tests;

/// Numerically stable eigenvalue algorithms
pub mod stable_eigenvalue;

/// Robust optimization methods for manifold learning
pub mod robust_optimization;

/// Condition number monitoring for numerical stability
pub mod condition_monitoring;

/// Trait-based manifold learning framework
pub mod manifold_traits;

/// Fluent API for manifold learning configuration
pub mod fluent_api;

/// Extensible distance metrics registry
pub mod extensible_metrics;

/// Type-safe geometric operations with compile-time dimension checking
pub mod type_safe_geometry;

/// Re-export manifold traits and utilities for convenience
pub use manifold_traits::*;

/// Re-export fluent API for convenience
pub use fluent_api::*;

/// Re-export extensible metrics for convenience
pub use extensible_metrics::*;

/// Re-export type-safe geometry for convenience
pub use type_safe_geometry::*;

/// Re-export visualization utilities for convenience
pub use visualization::*;

/// Serialization support for manifold learning models
#[cfg(feature = "serialization")]
pub mod serialization;

/// Serialization implementations for specific algorithms
#[cfg(feature = "serialization")]
pub mod serialization_impl;

/// Re-export serialization utilities for convenience
#[cfg(feature = "serialization")]
pub use serialization::*;

/// Plugin architecture for custom manifold learning methods
pub mod plugin_architecture;

/// Re-export plugin architecture utilities for convenience
pub use plugin_architecture::*;

/// Information-theoretic manifold learning methods
pub mod information_theory;

/// Re-export information theory utilities for convenience
pub use information_theory::*;

/// Optimal transport methods for manifold learning
pub mod optimal_transport;

/// Re-export optimal transport utilities for convenience
pub use optimal_transport::*;

/// Iterative refinement methods for improved numerical stability
pub mod iterative_refinement;

/// Re-export iterative refinement utilities for convenience
pub use iterative_refinement::*;

/// Pipeline middleware system for composable manifold learning
pub mod pipeline_middleware;

/// Re-export pipeline middleware utilities for convenience
pub use pipeline_middleware::*;

/// Embedding callbacks for monitoring and customizing manifold learning training
pub mod embedding_callbacks;

/// Re-export embedding callback utilities for convenience
pub use embedding_callbacks::*;

/// Category theory-based manifold representations and functorial embeddings
pub mod category_theory;

/// Re-export category theory utilities for convenience
pub use category_theory::*;

/// Advanced performance optimizations including cache-friendly data layouts and unsafe optimizations
pub mod performance_optimization;

/// Re-export performance optimization utilities for convenience
pub use performance_optimization::*;

/// Deep learning integration for manifold learning including autoencoders and variational autoencoders
pub mod deep_learning;

/// Re-export deep learning utilities for convenience
pub use deep_learning::*;

/// Computer vision applications for manifold learning including image patch embedding and face analysis
pub mod computer_vision;

/// Re-export computer vision utilities for convenience
pub use computer_vision::*;

/// Adversarial manifold learning module
pub mod adversarial;

/// Re-export adversarial manifold learning utilities for convenience
pub use adversarial::*;

/// Continuous normalizing flows module
pub mod continuous_normalizing_flows;

/// Re-export continuous normalizing flows utilities for convenience
pub use continuous_normalizing_flows::*;

/// Natural Language Processing manifold learning module
pub mod nlp;

/// Re-export NLP manifold learning utilities for convenience
pub use nlp::*;

/// Quantum methods for manifold learning module
pub mod quantum;

/// Re-export quantum methods for convenience
pub use quantum::*;

/// Causal inference on manifolds module
pub mod causal;

/// Re-export causal inference methods for convenience
pub use causal::*;

/// Bioinformatics applications for manifold learning including genomic analysis,
/// protein structures, phylogenetics, single-cell trajectories, and metabolic pathways
pub mod bioinformatics;

/// Re-export bioinformatics utilities for convenience
pub use bioinformatics::*;
