//! Advanced algorithm tests for specialized manifold learning methods
//!
//! This module contains tests for advanced algorithms like Hessian LLE, LTSA,
//! MVU, SNE, and Symmetric SNE, including validation and property tests.

use crate::*;
use scirs2_core::ndarray::{array, Array2};

#[test]
fn test_hessian_lle_basic() {
    let x = array![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0]
    ];

    let hlle = HessianLLE::new().n_neighbors(4).n_components(2);

    let fitted = hlle.fit(&x.view(), &()).unwrap();
    let embedding = fitted.embedding();

    assert_eq!(embedding.dim(), (6, 2));
}

#[test]
fn test_hessian_lle_validation_neighbors() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];

    let hlle = HessianLLE::new().n_neighbors(5); // Too many neighbors

    let result = hlle.fit(&x.view(), &());
    assert!(result.is_err());
}

#[test]
fn test_hessian_lle_validation_components() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];

    let hlle = HessianLLE::new().n_components(5); // Too many components

    let result = hlle.fit(&x.view(), &());
    assert!(result.is_err());
}

#[test]
fn test_hessian_lle_validation_dimensions() {
    let x = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
    ];

    // HLLE requires n_neighbors > n_features
    let hlle = HessianLLE::new()
        .n_neighbors(3) // Equal to n_features
        .n_components(2);

    let result = hlle.fit(&x.view(), &());
    assert!(result.is_err());
}

#[test]
fn test_hessian_lle_transform_error() {
    let x = array![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0]
    ];

    let hlle = HessianLLE::new().n_neighbors(4).n_components(2);

    let fitted = hlle.fit(&x.view(), &()).unwrap();

    // HLLE doesn't support transforming new data
    let result = fitted.transform(&x.view());
    assert!(result.is_err());
}

#[test]
fn test_hessian_lle_properties() {
    let x = array![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0]
    ];

    let hlle = HessianLLE::new().n_neighbors(4).n_components(2);

    let fitted = hlle.fit(&x.view(), &()).unwrap();

    assert_eq!(fitted.embedding().dim(), (6, 2));
    assert_eq!(fitted.hessian_matrix().dim(), (6, 6));
}

#[test]
fn test_ltsa_basic() {
    let x = array![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0]
    ];

    let ltsa = LTSA::new().n_neighbors(4).n_components(2);

    let fitted = ltsa.fit(&x.view(), &()).unwrap();
    let embedding = fitted.embedding();

    assert_eq!(embedding.dim(), (6, 2));
}

#[test]
fn test_ltsa_validation_neighbors() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];

    let ltsa = LTSA::new().n_neighbors(5); // Too many neighbors

    let result = ltsa.fit(&x.view(), &());
    assert!(result.is_err());
}

#[test]
fn test_ltsa_validation_components() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];

    let ltsa = LTSA::new().n_components(5); // Too many components

    let result = ltsa.fit(&x.view(), &());
    assert!(result.is_err());
}

#[test]
fn test_ltsa_validation_dimensions() {
    let x = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
    ];

    // LTSA requires n_neighbors > n_components
    let ltsa = LTSA::new()
        .n_neighbors(2) // Equal to n_components
        .n_components(2);

    let result = ltsa.fit(&x.view(), &());
    assert!(result.is_err());
}

#[test]
fn test_ltsa_transform_error() {
    let x = array![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0]
    ];

    let ltsa = LTSA::new().n_neighbors(4).n_components(2);

    let fitted = ltsa.fit(&x.view(), &()).unwrap();

    // LTSA doesn't support transforming new data
    let result = fitted.transform(&x.view());
    assert!(result.is_err());
}

#[test]
fn test_ltsa_properties() {
    let x = array![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0]
    ];

    let ltsa = LTSA::new().n_neighbors(4).n_components(2);

    let fitted = ltsa.fit(&x.view(), &()).unwrap();

    assert_eq!(fitted.embedding().dim(), (6, 2));
    assert_eq!(fitted.alignment_matrix().dim(), (6, 6));
    assert_eq!(fitted.local_tangent_spaces().len(), 6);
}

#[test]
fn test_mvu_basic() {
    let x = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0]
    ];

    let mvu = MVU::new().n_components(2).n_neighbors(2).max_iter(10); // Reduced for testing

    let fitted = mvu.fit(&x.view(), &()).unwrap();
    let embedding = fitted.embedding();

    assert_eq!(embedding.dim(), (5, 2));
}

#[test]
fn test_mvu_transform() {
    let x = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0]
    ];

    let mvu = MVU::new().n_components(2).n_neighbors(2).max_iter(10);

    let fitted = mvu.fit(&x.view(), &()).unwrap();
    let transformed = fitted.transform(&x.view()).unwrap();

    assert_eq!(transformed.dim(), (5, 2));
}

#[test]
fn test_mvu_neighbors_validation() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];

    let mvu = MVU::new().n_neighbors(5); // Too many neighbors for 2 samples

    let result = mvu.fit(&x.view(), &());
    assert!(result.is_err());
}

#[test]
fn test_mvu_components_validation() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

    let mvu = MVU::new().n_components(3); // n_components >= n_features

    let result = mvu.fit(&x.view(), &());
    assert!(result.is_err());
}

#[test]
fn test_mvu_properties() {
    let x = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0]
    ];

    let mvu = MVU::new().n_components(2).n_neighbors(3).max_iter(10);

    let fitted = mvu.fit(&x.view(), &()).unwrap();

    assert_eq!(fitted.embedding().dim(), (5, 2));
    assert_eq!(fitted.kernel_matrix().dim(), (5, 5));
    assert_eq!(fitted.neighbors().len(), 5);
}

#[test]
fn test_sne_basic() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

    let sne = SNE::new()
        .n_components(2)
        .perplexity(1.0)
        .n_iter(50)
        .random_state(Some(42));

    let fitted = sne.fit(&x.view(), &()).unwrap();
    let embedding = fitted.embedding();

    assert_eq!(embedding.dim(), (4, 2));
}

#[test]
fn test_sne_transform() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

    let sne = SNE::new()
        .n_components(2)
        .perplexity(1.0)
        .n_iter(50)
        .random_state(Some(42));

    let fitted = sne.fit(&x.view(), &()).unwrap();
    let transformed = fitted.transform(&x.view()).unwrap();

    assert_eq!(transformed.dim(), (4, 2));
}

#[test]
fn test_sne_perplexity_validation() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];

    let sne = SNE::new().perplexity(3.0); // perplexity >= n_samples

    let result = sne.fit(&x.view(), &());
    assert!(result.is_err());
}

#[test]
fn test_sne_min_samples_validation() {
    let x = array![[1.0, 2.0]]; // Only 1 sample

    let sne = SNE::new();

    let result = sne.fit(&x.view(), &());
    assert!(result.is_err());
}

#[test]
fn test_sne_properties() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

    let sne = SNE::new()
        .n_components(2)
        .perplexity(1.0)
        .n_iter(50)
        .random_state(Some(42));

    let fitted = sne.fit(&x.view(), &()).unwrap();

    assert_eq!(fitted.embedding().dim(), (4, 2));
    assert_eq!(fitted.conditional_probabilities().dim(), (4, 4));
}

#[test]
fn test_symmetric_sne_basic() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

    let ssne = SymmetricSNE::new()
        .n_components(2)
        .perplexity(1.0)
        .n_iter(50)
        .random_state(Some(42));

    let fitted = ssne.fit(&x.view(), &()).unwrap();
    let embedding = fitted.embedding();

    assert_eq!(embedding.dim(), (4, 2));
}

#[test]
fn test_symmetric_sne_transform() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

    let ssne = SymmetricSNE::new()
        .n_components(2)
        .perplexity(1.0)
        .n_iter(50)
        .random_state(Some(42));

    let fitted = ssne.fit(&x.view(), &()).unwrap();
    let transformed = fitted.transform(&x.view()).unwrap();

    assert_eq!(transformed.dim(), (4, 2));
}

#[test]
fn test_symmetric_sne_perplexity_validation() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];

    let ssne = SymmetricSNE::new().perplexity(3.0); // perplexity >= n_samples

    let result = ssne.fit(&x.view(), &());
    assert!(result.is_err());
}

#[test]
fn test_symmetric_sne_min_samples_validation() {
    let x = array![[1.0, 2.0]]; // Only 1 sample

    let ssne = SymmetricSNE::new();

    let result = ssne.fit(&x.view(), &());
    assert!(result.is_err());
}

#[test]
fn test_symmetric_sne_properties() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

    let ssne = SymmetricSNE::new()
        .n_components(2)
        .perplexity(1.0)
        .n_iter(50)
        .random_state(Some(42));

    let fitted = ssne.fit(&x.view(), &()).unwrap();

    assert_eq!(fitted.embedding().dim(), (4, 2));
    assert_eq!(fitted.joint_probabilities().dim(), (4, 4));

    // Verify symmetry of joint probabilities
    let p_joint = fitted.joint_probabilities();
    for i in 0..4 {
        for j in 0..4 {
            let diff = (p_joint[[i, j]] - p_joint[[j, i]]).abs();
            assert!(diff < 1e-10, "Joint probabilities should be symmetric");
        }
    }
}

#[test]
fn test_reproducibility_new_algorithms() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
    let seed = 42u64;

    // Test SNE reproducibility
    let sne1 = SNE::new()
        .n_components(2)
        .perplexity(1.0)
        .n_iter(20)
        .random_state(Some(seed));

    let sne2 = SNE::new()
        .n_components(2)
        .perplexity(1.0)
        .n_iter(20)
        .random_state(Some(seed));

    let fitted1 = sne1.fit(&x.view(), &()).unwrap();
    let fitted2 = sne2.fit(&x.view(), &()).unwrap();

    let embedding1 = fitted1.embedding();
    let embedding2 = fitted2.embedding();

    // Check if embeddings are approximately equal
    let max_diff = embedding1
        .iter()
        .zip(embedding2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    assert!(
        max_diff < 1e-6,
        "SNE should be reproducible with same random state"
    );

    // Test Symmetric SNE reproducibility
    let ssne1 = SymmetricSNE::new()
        .n_components(2)
        .perplexity(1.0)
        .n_iter(20)
        .random_state(Some(seed));

    let ssne2 = SymmetricSNE::new()
        .n_components(2)
        .perplexity(1.0)
        .n_iter(20)
        .random_state(Some(seed));

    let fitted1 = ssne1.fit(&x.view(), &()).unwrap();
    let fitted2 = ssne2.fit(&x.view(), &()).unwrap();

    let embedding1 = fitted1.embedding();
    let embedding2 = fitted2.embedding();

    // Check if embeddings are approximately equal
    let max_diff = embedding1
        .iter()
        .zip(embedding2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    assert!(
        max_diff < 1e-6,
        "SymmetricSNE should be reproducible with same random state"
    );
}
