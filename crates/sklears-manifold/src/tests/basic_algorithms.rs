//! Basic algorithm tests for core manifold learning methods
//!
//! This module contains fundamental tests for the main manifold learning algorithms
//! including t-SNE, Isomap, LLE, Laplacian Eigenmaps, MDS, UMAP, Diffusion Maps,
//! Hessian LLE, LTSA, MVU, SNE, and Symmetric SNE.

use crate::*;
use scirs2_core::ndarray::array;

#[test]
fn test_tsne_basic() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

    let tsne = TSNE::new()
        .n_components(2)
        .perplexity(1.0)
        .n_iter(50)
        .verbose(false);

    let fitted = tsne.fit(&x.view(), &()).unwrap();
    let embedding = fitted.embedding();

    assert_eq!(embedding.dim(), (4, 2));
}

#[test]
fn test_isomap_basic() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

    let isomap = Isomap::new().n_neighbors(2).n_components(2);

    let fitted = isomap.fit(&x.view(), &()).unwrap();
    let embedding = fitted.embedding();

    assert_eq!(embedding.dim(), (4, 2));
}

#[test]
fn test_tsne_transform() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

    let tsne = TSNE::new().n_components(2).perplexity(1.0).n_iter(50);

    let fitted = tsne.fit(&x.view(), &()).unwrap();
    let transformed = fitted.transform(&x.view()).unwrap();

    assert_eq!(transformed.dim(), (4, 2));
}

#[test]
fn test_isomap_transform() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

    let isomap = Isomap::new().n_neighbors(2).n_components(2);

    let fitted = isomap.fit(&x.view(), &()).unwrap();
    let transformed = fitted.transform(&x.view()).unwrap();

    assert_eq!(transformed.dim(), (4, 2));
}

#[test]
fn test_tsne_perplexity_validation() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];

    let tsne = TSNE::new().perplexity(5.0); // Too high for 2 samples

    let result = tsne.fit(&x.view(), &());
    assert!(result.is_err());
}

#[test]
fn test_lle_basic() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];

    let lle = LocallyLinearEmbedding::new().n_neighbors(2).n_components(2);

    let fitted = lle.fit(&x.view(), &()).unwrap();
    let embedding = fitted.embedding();

    assert_eq!(embedding.dim(), (5, 2));
}

#[test]
fn test_lle_transform() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];

    let lle = LocallyLinearEmbedding::new().n_neighbors(2).n_components(2);

    let fitted = lle.fit(&x.view(), &()).unwrap();
    let transformed = fitted.transform(&x.view()).unwrap();

    assert_eq!(transformed.dim(), (5, 2));
}

#[test]
fn test_lle_neighbors_validation() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];

    let lle = LocallyLinearEmbedding::new().n_neighbors(5); // Too high for 2 samples

    let result = lle.fit(&x.view(), &());
    assert!(result.is_err());
}

#[test]
fn test_laplacian_eigenmaps_basic() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];

    let laplacian = LaplacianEigenmaps::new().n_neighbors(2).n_components(2);

    let fitted = laplacian.fit(&x.view(), &()).unwrap();
    let embedding = fitted.embedding();

    assert_eq!(embedding.dim(), (5, 2));
}

#[test]
fn test_laplacian_eigenmaps_transform() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];

    let laplacian = LaplacianEigenmaps::new().n_neighbors(2).n_components(2);

    let fitted = laplacian.fit(&x.view(), &()).unwrap();
    let transformed = fitted.transform(&x.view()).unwrap();

    assert_eq!(transformed.dim(), (5, 2));
}

#[test]
fn test_laplacian_eigenmaps_neighbors_validation() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];

    let laplacian = LaplacianEigenmaps::new().n_neighbors(5); // Too high for 2 samples

    let result = laplacian.fit(&x.view(), &());
    assert!(result.is_err());
}

#[test]
fn test_mds_basic() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];

    let mds = MDS::new().n_components(2);

    let fitted = mds.fit(&x.view(), &()).unwrap();
    let embedding = fitted.embedding();

    assert_eq!(embedding.dim(), (5, 2));
}

#[test]
fn test_mds_transform() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];

    let mds = MDS::new().n_components(2);

    let fitted = mds.fit(&x.view(), &()).unwrap();
    let transformed = fitted.transform(&x.view()).unwrap();

    assert_eq!(transformed.dim(), (5, 2));
}

#[test]
fn test_mds_stress() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];

    let mds = MDS::new().n_components(2);

    let fitted = mds.fit(&x.view(), &()).unwrap();
    let stress = fitted.stress();

    assert!(stress >= 0.0); // Stress should be non-negative
}

#[test]
fn test_umap_basic() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];

    let umap = UMAP::new()
        .n_neighbors(3)
        .n_components(2)
        .n_epochs(Some(10)) // Short for testing
        .random_state(Some(42));

    let fitted = umap.fit(&x.view(), &()).unwrap();
    let embedding = fitted.embedding();

    assert_eq!(embedding.dim(), (5, 2));
}

#[test]
fn test_umap_transform() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];

    let umap = UMAP::new()
        .n_neighbors(3)
        .n_components(2)
        .n_epochs(Some(10))
        .random_state(Some(42));

    let fitted = umap.fit(&x.view(), &()).unwrap();
    let transformed = fitted.transform(&x.view()).unwrap();

    assert_eq!(transformed.dim(), (5, 2));
}

#[test]
fn test_umap_neighbors_validation() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];

    let umap = UMAP::new().n_neighbors(5); // Too high for 2 samples

    let result = umap.fit(&x.view(), &());
    assert!(result.is_err());
}

#[test]
fn test_umap_params() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];

    let umap = UMAP::new()
        .n_neighbors(3)
        .n_components(2)
        .min_dist(0.5)
        .spread(2.0)
        .learning_rate(0.5)
        .n_epochs(Some(10))
        .random_state(Some(42));

    let fitted = umap.fit(&x.view(), &()).unwrap();

    assert!(fitted.a() > 0.0);
    assert!(fitted.b() > 0.0);
}

#[test]
fn test_tsne_barnes_hut() {
    // Create a smaller dataset for faster testing
    let mut x_data = Vec::new();
    for i in 0..100 {
        // Reduced from 300
        x_data.push([i as f64 / 100.0, (i as f64 / 100.0).sin()]);
    }
    let x = Array2::from_shape_vec((100, 2), x_data.into_iter().flatten().collect()).unwrap();

    let tsne = TSNE::new()
        .n_components(2)
        .perplexity(10.0) // Reduced perplexity
        .method("barnes_hut")
        .angle(0.5)
        .n_iter(20) // Reduced for faster testing
        .random_state(Some(42));

    let fitted = tsne.fit(&x.view(), &()).unwrap();
    let embedding = fitted.embedding();

    assert_eq!(embedding.dim(), (100, 2));
}

#[test]
fn test_tsne_exact_vs_barnes_hut() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];

    // Test exact method
    let tsne_exact = TSNE::new()
        .n_components(2)
        .perplexity(2.0)
        .method("exact")
        .n_iter(50)
        .random_state(Some(42));

    let fitted_exact = tsne_exact.fit(&x.view(), &()).unwrap();

    // Test Barnes-Hut method (should fall back to exact for small datasets)
    let tsne_bh = TSNE::new()
        .n_components(2)
        .perplexity(2.0)
        .method("barnes_hut")
        .n_iter(50)
        .random_state(Some(42));

    let fitted_bh = tsne_bh.fit(&x.view(), &()).unwrap();

    // Both should produce valid embeddings
    assert_eq!(fitted_exact.embedding().dim(), (5, 2));
    assert_eq!(fitted_bh.embedding().dim(), (5, 2));
}

#[test]
fn test_diffusion_maps_basic() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];

    let dm = DiffusionMaps::new()
        .n_components(2)
        .epsilon(2.0)
        .diffusion_time(1)
        .alpha(1.0);

    let fitted = dm.fit(&x.view(), &()).unwrap();
    let embedding = fitted.embedding();

    assert_eq!(embedding.dim(), (5, 2));
    assert!(fitted.epsilon() > 0.0);
}

#[test]
fn test_diffusion_maps_transform() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];

    let dm = DiffusionMaps::new().n_components(2).epsilon(1.5);

    let fitted = dm.fit(&x.view(), &()).unwrap();
    let transformed = fitted.transform(&x.view()).unwrap();

    assert_eq!(transformed.dim(), (5, 2));
}

#[test]
fn test_diffusion_maps_auto_epsilon() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];

    let dm = DiffusionMaps::new().n_components(2).diffusion_time(2);

    let fitted = dm.fit(&x.view(), &()).unwrap();

    // Should automatically estimate epsilon
    assert!(fitted.epsilon() > 0.0);
    assert_eq!(fitted.embedding().dim(), (5, 2));
}

#[test]
fn test_diffusion_maps_validation() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];

    let dm = DiffusionMaps::new().n_components(5); // Too many components

    let result = dm.fit(&x.view(), &());
    assert!(result.is_err());
}

#[test]
fn test_diffusion_maps_eigenvalues() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];

    let dm = DiffusionMaps::new().n_components(2).epsilon(1.0);

    let fitted = dm.fit(&x.view(), &()).unwrap();

    assert_eq!(fitted.eigenvalues().dim(), (5, 1)); // All eigenvalues are stored, not just n_components
    assert_eq!(fitted.eigenvectors().dim(), (5, 5)); // All eigenvectors are stored
    assert_eq!(fitted.affinity_matrix().dim(), (5, 5));
}
