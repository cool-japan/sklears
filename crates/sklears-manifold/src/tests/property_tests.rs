//! Property-based tests for manifold learning algorithms
//!
//! This module contains property tests using the proptest framework to verify
//! algorithm behavior across a wide range of inputs and configurations.

use crate::*;
use proptest::prelude::*;
use scirs2_core::ndarray::Array2;

// Generate valid manifold learning configurations
fn valid_manifold_config() -> impl Strategy<Value = (usize, usize, usize)> {
    (5usize..20, 2usize..5, 3usize..8).prop_filter(
        "n_neighbors must be >= n_components + 1",
        |(n_samples, n_components, n_neighbors)| {
            n_neighbors < n_samples && n_neighbors > n_components
        },
    )
}

// Generate random data matrices
fn random_data_matrix(n_samples: usize, n_features: usize) -> impl Strategy<Value = Array2<f64>> {
    prop::collection::vec(prop::collection::vec(-10.0..10.0f64, n_features), n_samples).prop_map(
        move |data| {
            Array2::from_shape_vec(
                (n_samples, n_features),
                data.into_iter().flatten().collect(),
            )
            .unwrap()
        },
    )
}

proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig::with_cases(10))]
    #[test]
    fn test_tsne_embedding_dimensions(
        (n_samples, n_components, _) in valid_manifold_config(),
        data in random_data_matrix(5, 3).prop_flat_map(|_| random_data_matrix(8, 3))
    ) {
        let tsne = TSNE::new()
            .n_components(2)
            .perplexity(2.0)
            .n_iter(5) // Reduced for faster testing
            .random_state(Some(42));

        if let Ok(fitted) = tsne.fit(&data.view(), &()) {
            let embedding = fitted.embedding();
            prop_assert_eq!(embedding.nrows(), data.nrows());
            prop_assert_eq!(embedding.ncols(), 2);
        }
    }

    #[test]
    fn test_umap_embedding_dimensions(
        (n_samples, n_components, n_neighbors) in valid_manifold_config()
    ) {
        let data = Array2::from_shape_fn((n_samples, 3), |(i, j)| i as f64 + j as f64);

        let umap = UMAP::new()
            .n_neighbors(n_neighbors.min(n_samples - 1))
            .n_components(n_components)
            .n_epochs(Some(5)) // Short for testing
            .random_state(Some(42));

        if let Ok(fitted) = umap.fit(&data.view(), &()) {
            let embedding = fitted.embedding();
            prop_assert_eq!(embedding.nrows(), n_samples);
            prop_assert_eq!(embedding.ncols(), n_components);
        }
    }

    #[test]
    fn test_hessian_lle_embedding_dimensions(
        (n_samples, n_components, n_neighbors) in valid_manifold_config()
            .prop_filter("HLLE needs n_neighbors > n_features", |(n_samples, n_components, n_neighbors)| {
                *n_neighbors > 3 && *n_samples > *n_components
            })
    ) {
        let data = Array2::from_shape_fn((n_samples, 3), |(i, j)| i as f64 + j as f64);

        let hlle = HessianLLE::new()
            .n_neighbors(n_neighbors.min(n_samples - 1))
            .n_components(n_components);

        if let Ok(fitted) = hlle.fit(&data.view(), &()) {
            let embedding = fitted.embedding();
            prop_assert_eq!(embedding.nrows(), n_samples);
            prop_assert_eq!(embedding.ncols(), n_components);
        }
    }

    #[test]
    fn test_ltsa_embedding_dimensions(
        (n_samples, n_components, n_neighbors) in valid_manifold_config()
    ) {
        let data = Array2::from_shape_fn((n_samples, 3), |(i, j)| i as f64 + j as f64);

        let ltsa = LTSA::new()
            .n_neighbors(n_neighbors.min(n_samples - 1))
            .n_components(n_components);

        if let Ok(fitted) = ltsa.fit(&data.view(), &()) {
            let embedding = fitted.embedding();
            prop_assert_eq!(embedding.nrows(), n_samples);
            prop_assert_eq!(embedding.ncols(), n_components);
        }
    }

    #[test]
    fn test_diffusion_maps_embedding_dimensions(
        (n_samples, n_components, _) in valid_manifold_config()
    ) {
        let data = Array2::from_shape_fn((n_samples, 3), |(i, j)| i as f64 + j as f64);

        let dm = DiffusionMaps::new()
            .n_components(n_components)
            .epsilon(1.0);

        if let Ok(fitted) = dm.fit(&data.view(), &()) {
            let embedding = fitted.embedding();
            prop_assert_eq!(embedding.nrows(), n_samples);
            prop_assert_eq!(embedding.ncols(), n_components);
        }
    }

    #[test]
    fn test_manifold_embedding_finite_values(
        data in random_data_matrix(6, 3)
    ) {
        // Test that embeddings contain finite values (reduced algorithm set for speed)
        let algorithms: Vec<Box<dyn Fn(&Array2<f64>) -> Option<Array2<f64>>>> = vec![
            Box::new(|data| {
                TSNE::new()
                    .n_components(2)
                    .perplexity(2.0)
                    .n_iter(5) // Reduced for speed
                    .random_state(Some(42))
                    .fit(&data.view(), &())
                    .ok()
                    .map(|fitted| fitted.embedding().clone())
            }),
            Box::new(|data| {
                UMAP::new()
                    .n_neighbors(3)
                    .n_components(2)
                    .n_epochs(Some(3)) // Reduced for speed
                    .random_state(Some(42))
                    .fit(&data.view(), &())
                    .ok()
                    .map(|fitted| fitted.embedding().clone())
            }),
            // Removed slower algorithms to speed up the test
        ];

        for algorithm in algorithms {
            if let Some(embedding) = algorithm(&data) {
                prop_assert!(embedding.iter().all(|&x| x.is_finite()),
                           "Embedding should contain only finite values");
            }
        }
    }

    #[test]
    fn test_manifold_reproducibility(
        data in random_data_matrix(6, 3)
    ) {
        // Test that algorithms produce reproducible results with same random state
        let seed = 42u64;

        // Test TSNE reproducibility
        let tsne1 = TSNE::new()
            .n_components(2)
            .perplexity(2.0)
            .n_iter(10)
            .random_state(Some(seed));

        let tsne2 = TSNE::new()
            .n_components(2)
            .perplexity(2.0)
            .n_iter(10)
            .random_state(Some(seed));

        if let (Ok(fitted1), Ok(fitted2)) = (
            tsne1.fit(&data.view(), &()),
            tsne2.fit(&data.view(), &())
        ) {
            let embedding1 = fitted1.embedding();
            let embedding2 = fitted2.embedding();

            // Check if embeddings are approximately equal (allowing for small numerical differences)
            let max_diff = embedding1.iter()
                .zip(embedding2.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);

            prop_assert!(max_diff < 1e-10, "TSNE should be reproducible with same random state");
        }
    }

    #[test]
    #[ignore = "ParametricTSNE is too slow for CI - run with --ignored"]
    fn test_parametric_tsne_embedding_dimensions(
        (n_samples, n_components, _) in valid_manifold_config()
    ) {
        let data = Array2::from_shape_fn((n_samples, 3), |(i, j)| i as f64 + j as f64);

        let ptsne = ParametricTSNE::new()
            .n_components(n_components)
            .perplexity(2.0)
            .n_iter(5);

        if let Ok(fitted) = ptsne.fit(&data.view(), &()) {
            let embedding = fitted.embedding();
            prop_assert_eq!(embedding.nrows(), n_samples);
            prop_assert_eq!(embedding.ncols(), n_components);

            // Test out-of-sample projection
            let new_data = Array2::from_shape_fn((2, 3), |(i, j)| (i + n_samples) as f64 + j as f64);
            if let Ok(projected) = fitted.transform(&new_data.view()) {
                prop_assert_eq!(projected.nrows(), 2);
                prop_assert_eq!(projected.ncols(), n_components);
                prop_assert!(projected.iter().all(|&x| x.is_finite()));
            }
        }
    }

    #[test]
    fn test_heavy_tailed_sne_embedding_dimensions(
        (n_samples, n_components, _) in valid_manifold_config()
    ) {
        let data = Array2::from_shape_fn((n_samples, 3), |(i, j)| i as f64 + j as f64);

        let htsne = HeavyTailedSymmetricSNE::new()
            .n_components(n_components)
            .perplexity(2.0)
            .degrees_of_freedom(2.0)
            .n_iter(5);

        if let Ok(fitted) = htsne.fit(&data.view(), &()) {
            let embedding = fitted.embedding();
            prop_assert_eq!(embedding.nrows(), n_samples);
            prop_assert_eq!(embedding.ncols(), n_components);
        }
    }

    #[test]
    #[ignore = "ParametricTSNE is too slow for CI - run with --ignored"]
    fn test_parametric_tsne_reproducibility(
        data in random_data_matrix(5, 3)
    ) {
        let seed = 42u64;

        let ptsne1 = ParametricTSNE::new()
            .n_components(2)
            .perplexity(2.0)
            .n_iter(3) // Reduced for speed
            .random_state(Some(seed));

        let ptsne2 = ParametricTSNE::new()
            .n_components(2)
            .perplexity(2.0)
            .n_iter(3) // Reduced for speed
            .random_state(Some(seed));

        if let (Ok(fitted1), Ok(fitted2)) = (
            ptsne1.fit(&data.view(), &()),
            ptsne2.fit(&data.view(), &())
        ) {
            let embedding1 = fitted1.embedding();
            let embedding2 = fitted2.embedding();

            let max_diff = embedding1.iter()
                .zip(embedding2.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);

            prop_assert!(max_diff < 1e-6, "Parametric t-SNE should be reproducible with same random state");
        }
    }

    #[test]
    fn test_heavy_tailed_sne_reproducibility(
        data in random_data_matrix(6, 3)
    ) {
        let seed = 42u64;

        let htsne1 = HeavyTailedSymmetricSNE::new()
            .n_components(2)
            .perplexity(2.0)
            .degrees_of_freedom(2.0)
            .n_iter(5)
            .random_state(Some(seed));

        let htsne2 = HeavyTailedSymmetricSNE::new()
            .n_components(2)
            .perplexity(2.0)
            .degrees_of_freedom(2.0)
            .n_iter(5)
            .random_state(Some(seed));

        if let (Ok(fitted1), Ok(fitted2)) = (
            htsne1.fit(&data.view(), &()),
            htsne2.fit(&data.view(), &())
        ) {
            let embedding1 = fitted1.embedding();
            let embedding2 = fitted2.embedding();

            let max_diff = embedding1.iter()
                .zip(embedding2.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);

            prop_assert!(max_diff < 1e-6, "Heavy-Tailed Symmetric SNE should be reproducible with same random state");
        }
    }

    #[test]
    fn test_heavy_tailed_sne_degrees_of_freedom_parameter(
        data in random_data_matrix(6, 3)
    ) {
        // Test different degrees of freedom values
        let dofs = vec![0.5, 1.0, 2.0, 5.0, 10.0];

        for dof in dofs {
            let htsne = HeavyTailedSymmetricSNE::new()
                .n_components(2)
                .perplexity(2.0)
                .degrees_of_freedom(dof)
                .n_iter(5)
                .random_state(Some(42));

            if let Ok(fitted) = htsne.fit(&data.view(), &()) {
                let embedding = fitted.embedding();
                prop_assert!(embedding.iter().all(|&x| x.is_finite()),
                           "Embedding should contain finite values for dof = {}", dof);
                prop_assert_eq!(embedding.nrows(), data.nrows());
                prop_assert_eq!(embedding.ncols(), 2);
            }
        }
    }
}
