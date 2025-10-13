//! Information theory, topological data analysis, and comprehensive feature extraction tests
//!
//! This module contains tests for information theory operations, persistent homology, topological data analysis,
//! biological sequence features, image processing, and engineering feature extractors including polynomial features,
//! RBF sampling, Nyström approximation, and additive chi-squared sampling.

use crate::time_series_features::WaveletFeatureType;
use crate::{biological, engineering, image, information_theory, neural};
use scirs2_core::ndarray::{array, s, Array2};

// ===== TOPOLOGICAL DATA ANALYSIS TESTS =====

#[test]
fn test_topological_persistence_diagram() {
    // Create a simple point cloud with clear topological features
    let points = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5], // Center point
        [2.0, 2.0], // Isolated point
    ];

    let tda = information_theory::TopologicalDataAnalysis::new()
        .max_dimension(1)
        .threshold(2.0);

    let persistence = tda.compute_persistence(&points).unwrap();

    // Should have birth-death pairs for 0-dimensional and 1-dimensional features
    assert!(persistence.len() > 0);

    // All persistence values should be finite and birth <= death
    for diagram in persistence.iter() {
        for &(birth, death) in diagram.iter() {
            assert!(birth.is_finite());
            assert!(death.is_finite());
            assert!(birth <= death);
        }
    }
}

#[test]
fn test_topological_betti_numbers() {
    let points = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.866], // Triangle vertices
        [0.25, 0.25], // Point inside triangle
    ];

    let tda = information_theory::TopologicalDataAnalysis::new()
        .max_dimension(2)
        .threshold(1.5);

    let persistence = tda.compute_persistence(&points).unwrap();
    let betti = tda.compute_betti_numbers(&persistence);

    // Should have Betti numbers for dimensions 0, 1, 2
    assert_eq!(betti.len(), 3);

    // All Betti numbers should be non-negative integers
    for &b in betti.iter() {
        assert!(b <= points.nrows());
    }

    // For a connected set, β_0 should be 1
    assert_eq!(betti[0], 1);
}

#[test]
fn test_persistence_entropy() {
    let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],];

    let tda = information_theory::TopologicalDataAnalysis::new()
        .max_dimension(1)
        .threshold(2.0);

    let entropy = tda.persistence_entropy(&points).unwrap();

    // Entropy should be finite and non-negative
    assert!(entropy.is_finite());
    assert!(entropy >= 0.0);
}

#[test]
fn test_bottleneck_distance() {
    let points1 = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let points2 = array![[0.1, 0.1], [1.1, 0.1], [0.1, 1.1]]; // Slightly perturbed

    let tda = information_theory::TopologicalDataAnalysis::new()
        .max_dimension(1)
        .threshold(2.0);

    let distance = tda.bottleneck_distance(&points1, &points2).unwrap();

    // Distance should be finite and non-negative
    assert!(distance.is_finite());
    assert!(distance >= 0.0);

    // For similar point clouds, distance should be small
    assert!(distance < 1.0);
}

// ===== BIOLOGICAL SEQUENCE TESTS =====

#[test]
fn test_protein_sequence_features() {
    let sequence = "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTE";

    let extractor = biological::ProteinSequenceFeatures::new()
        .include_amino_acid_composition(true)
        .include_dipeptide_composition(true)
        .include_physicochemical_properties(true);

    let features = extractor.extract_features(sequence).unwrap();

    // Should have features for:
    // - 20 amino acids
    // - 400 dipeptides (20^2)
    // - physicochemical properties (varies, but let's say around 10)
    assert!(features.len() >= 430); // At least 20 + 400 + some physicochemical

    // All features should be finite and non-negative (for compositions)
    for &feat in features.iter() {
        assert!(feat.is_finite());
    }

    // Amino acid composition should sum to approximately 1.0
    let aa_composition_sum: f64 = features[..20].iter().sum();
    assert!((aa_composition_sum - 1.0).abs() < 1e-10);
}

#[test]
fn test_dna_sequence_features() {
    let sequence = "ATCGATCGATCGATCGTAGCTAGCTAGCT";

    let extractor = biological::DNASequenceFeatures::new()
        .include_nucleotide_composition(true)
        .include_dinucleotide_composition(true)
        .include_trinucleotide_composition(true);

    let features = extractor.extract_features(sequence).unwrap();

    // Should have features for:
    // - 4 nucleotides
    // - 16 dinucleotides (4^2)
    // - 64 trinucleotides (4^3)
    assert_eq!(features.len(), 4 + 16 + 64);

    // All features should be finite and non-negative
    for &feat in features.iter() {
        assert!(feat.is_finite());
        assert!(feat >= 0.0);
    }

    // Nucleotide composition should sum to 1.0
    let nucleotide_sum: f64 = features[..4].iter().sum();
    assert!((nucleotide_sum - 1.0).abs() < 1e-10);
}

#[test]
fn test_gc_content_features() {
    let sequence = "ATCGATCGATCGAAAAAAAATTTTTTTT"; // Mixed GC content

    let extractor = biological::GCContentFeatures::new()
        .window_size(10)
        .step_size(5);

    let features = extractor.extract_features(sequence).unwrap();

    // Should have GC content values for overlapping windows
    let expected_windows = (sequence.len() - 10) / 5 + 1;
    assert_eq!(features.len(), expected_windows);

    // All GC content values should be between 0 and 1
    for &gc in features.iter() {
        assert!(gc.is_finite());
        assert!(gc >= 0.0 && gc <= 1.0);
    }
}

// ===== IMAGE PROCESSING TESTS =====

#[test]
fn test_patch_extractor() {
    let image = Array2::from_shape_vec((6, 6), (0..36).map(|x| x as f64).collect()).unwrap();

    let extractor = image::PatchExtractor::new()
        .patch_size((3, 3))
        .max_patches(Some(4));

    let patches = extractor.extract(&image.view()).unwrap();
    assert_eq!(patches.dim(), (4, 3, 3));
}

#[test]
fn test_wavelet_feature_extractor() {
    let image =
        Array2::from_shape_vec((32, 32), (0..1024).map(|x| x as f64 / 1024.0).collect()).unwrap();

    let wavelet = image::WaveletFeatureExtractor::new()
        .wavelet_levels(3)
        .wavelet_type(image::WaveletType::Haar)
        .feature_type(WaveletFeatureType::Basic);

    let features = wavelet.extract_features(&image.view()).unwrap();

    // Should have features for each decomposition level + approximation
    // Each level contributes 4 features (mean, std, energy, entropy) for 3 detail bands
    // Plus final approximation: (3 levels * 3 bands + 1 approx) * 4 features = 40 features
    assert_eq!(features.len(), 40);

    // Test extended features
    let wavelet_extended = image::WaveletFeatureExtractor::new()
        .wavelet_levels(2)
        .feature_type(WaveletFeatureType::Extended);

    let features_extended = wavelet_extended.extract_features(&image.view()).unwrap();

    // Extended features include 8 features per band instead of 4
    // (2 levels * 3 bands + 1 approx) * 8 features = 56 features
    assert_eq!(features_extended.len(), 56);
}

// ===== ENGINEERING FEATURES TESTS =====

#[test]
fn test_polynomial_features() {
    let X = array![[1.0, 2.0], [3.0, 4.0]];

    let poly = crate::basic_features::PolynomialFeatures::new().degree(2);
    let features = poly.transform(&X.view()).unwrap();

    // Should include: bias, x1, x2, x1^2, x1*x2, x2^2
    assert_eq!(features.ncols(), 6);
    assert_eq!(features.nrows(), 2);
}

#[test]
fn test_rbf_sampler() {
    let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

    let mut rbf_sampler = engineering::RBFSampler::new()
        .gamma(1.0)
        .n_components(10)
        .random_state(Some(42));

    let features = rbf_sampler.fit_transform(&X.view()).unwrap();

    assert_eq!(features.dim(), (4, 10)); // 4 samples, 10 components

    // Check that features are normalized appropriately
    let normalization = (2.0 / 10.0_f64).sqrt();
    for &val in features.iter() {
        assert!(val.abs() <= normalization + 1e-10); // Cosine values should be bounded
    }

    // Test transform on new data
    let X_new = array![[2.0, 3.0], [4.0, 5.0]];
    let features_new = rbf_sampler.transform(&X_new.view()).unwrap();
    assert_eq!(features_new.dim(), (2, 10));
}

#[test]
fn test_nystroem() {
    let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];

    let mut nystroem = engineering::Nystroem::new()
        .kernel(engineering::NystromKernel::RBF)
        .gamma(1.0)
        .n_components(3)
        .random_state(Some(42));

    let features = nystroem.fit_transform(&X.view()).unwrap();

    assert_eq!(features.dim(), (5, 3)); // 5 samples, 3 components

    // Test transform on new data
    let X_new = array![[2.0, 3.0], [6.0, 7.0]];
    let features_new = nystroem.transform(&X_new.view()).unwrap();
    assert_eq!(features_new.dim(), (2, 3));

    // Test different kernels
    let mut nystroem_poly = engineering::Nystroem::new()
        .kernel(engineering::NystromKernel::Polynomial)
        .degree(2)
        .gamma(1.0)
        .coef0(1.0)
        .n_components(3);

    let features_poly = nystroem_poly.fit_transform(&X.view()).unwrap();
    assert_eq!(features_poly.dim(), (5, 3));
}

#[test]
fn test_additive_chi2_sampler() {
    let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

    let sampler = engineering::AdditiveChi2Sampler::new()
        .sample_steps(2)
        .sample_interval(0.5);

    let features = sampler.transform(&X.view()).unwrap();

    // Number of features: 2 features * 2 steps * 2 (cos + sin) = 8
    let expected_features = sampler.get_n_output_features(2);
    assert_eq!(expected_features, 8);
    assert_eq!(features.dim(), (3, 8)); // 3 samples, 8 features

    // Test that features are bounded (cosine and sine values)
    let normalization = (2.0 / std::f64::consts::PI).sqrt();
    for &val in features.iter() {
        assert!(val.abs() <= normalization + 1e-10);
    }
}

// ===== NEURAL NETWORK TESTS =====

#[test]
fn test_neural_autoencoder() {
    let X = array![
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0],
        [4.0, 5.0, 6.0, 7.0]
    ];

    let autoencoder = neural::Autoencoder::new()
        .input_dim(4)
        .hidden_dims(vec![3, 2, 3]) // Encoder: 4->3->2, Decoder: 2->3->4
        .activation("relu".to_string())
        .learning_rate(0.01)
        .n_epochs(10); // Reduced for testing

    let fitted = autoencoder.fit(&X.view(), &()).unwrap();
    let encoded = fitted.transform(&X.view()).unwrap();

    assert_eq!(encoded.dim(), (4, 2)); // Bottleneck dimension

    // All encoded values should be finite
    for &val in encoded.iter() {
        assert!(val.is_finite());
    }

    // Test reconstruction
    let reconstructed = fitted.inverse_transform(&encoded.view()).unwrap();
    assert_eq!(reconstructed.dim(), X.dim());

    for &val in reconstructed.iter() {
        assert!(val.is_finite());
    }
}

// ===== ERROR HANDLING AND EDGE CASES =====

#[test]
fn test_tda_empty_input() {
    let empty_points = Array2::<f64>::zeros((0, 2));
    let tda = information_theory::TopologicalDataAnalysis::new();

    // Should handle empty input gracefully
    let result = tda.compute_persistence(&empty_points);
    // This might error or return empty results, both are acceptable
    // We just ensure it doesn't panic
    match result {
        Ok(diagrams) => {
            let betti = tda.compute_betti_numbers(&diagrams);
            assert!(betti.is_empty() || betti.iter().all(|&b| b == 0));
        }
        Err(_) => (), // Error is acceptable for empty input
    }
}

#[test]
fn test_tda_single_point() {
    let single_point = array![[0.0, 0.0]];
    let tda = information_theory::TopologicalDataAnalysis::new()
        .max_dimension(1)
        .threshold(1.0);

    let diagrams = tda.compute_persistence(&single_point).unwrap();
    let betti = tda.compute_betti_numbers(&diagrams);

    // Single point should have β_0 = 1, β_1 = 0
    assert_eq!(betti[0], 1); // One connected component
    assert_eq!(betti[1], 0); // No loops
}

#[test]
fn test_engineering_error_cases() {
    let mut rbf_sampler = engineering::RBFSampler::new();
    let X = array![[1.0, 2.0], [3.0, 4.0]];

    // Test transform before fit
    let result = rbf_sampler.transform(&X.view());
    assert!(result.is_err()); // Should fail when not fitted

    // Test dimension mismatch
    rbf_sampler.fit(&X.view()).unwrap();
    let X_wrong = array![[1.0, 2.0, 3.0]]; // Wrong number of features
    let result = rbf_sampler.transform(&X_wrong.view());
    assert!(result.is_err()); // Should fail with dimension mismatch
}

// ===== PERFORMANCE AND CONSISTENCY TESTS =====

#[test]
fn test_tda_performance_large_dataset() {
    // Test with a moderately large dataset to ensure reasonable performance
    let n_points = 100;
    let points: Vec<f64> = (0..n_points * 2).map(|i| (i as f64 * 0.1) % 10.0).collect();

    let points_array = Array2::from_shape_vec((n_points, 2), points).unwrap();

    let tda = information_theory::TopologicalDataAnalysis::new()
        .max_dimension(1)
        .threshold(2.0);

    let start = std::time::Instant::now();
    let features = tda.extract_topological_features(&points_array).unwrap();
    let duration = start.elapsed();

    // Should complete in reasonable time (less than 10 seconds for this size)
    assert!(duration.as_secs() < 10);

    // Should produce some features
    assert!(features.len() > 0);

    // All features should be finite
    for &feat in features.iter() {
        assert!(feat.is_finite());
    }
}

#[test]
fn test_tda_features_consistency() {
    let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5],];

    let tda = information_theory::TopologicalDataAnalysis::new()
        .max_dimension(1)
        .threshold(2.0);

    // Extract features multiple times
    let features1 = tda.extract_topological_features(&points).unwrap();
    let features2 = tda.extract_topological_features(&points).unwrap();

    // Results should be consistent
    assert_eq!(features1.len(), features2.len());
    for (i, (&f1, &f2)) in features1.iter().zip(features2.iter()).enumerate() {
        assert!(
            (f1 - f2).abs() < 1e-10,
            "Inconsistent feature at index {}: {} != {}",
            i,
            f1,
            f2
        );
    }
}

// ===== INTEGRATION TESTS =====

#[test]
fn test_multi_domain_integration() {
    // Test workflow combining multiple feature extraction domains

    // 1. Start with biological sequence
    let sequence = "ATCGATCGATCG";
    let bio_extractor = biological::DNASequenceFeatures::new().include_nucleotide_composition(true);
    let bio_features = bio_extractor.extract_features(sequence).unwrap();

    // 2. Use biological features for polynomial expansion
    let bio_array = Array2::from_shape_vec((1, bio_features.len()), bio_features).unwrap();
    let poly = crate::basic_features::PolynomialFeatures::new().degree(2);
    let poly_features = poly.transform(&bio_array.view()).unwrap();

    // 3. Apply RBF sampling to polynomial features
    let mut rbf = engineering::RBFSampler::new()
        .n_components(8)
        .random_state(Some(42));
    let rbf_features = rbf.fit_transform(&poly_features.view()).unwrap();

    // All results should be finite and well-formed
    assert!(rbf_features.nrows() > 0);
    assert!(rbf_features.ncols() > 0);

    for &val in rbf_features.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_comprehensive_feature_pipeline() {
    // Create synthetic data that can be processed through multiple extractors
    let data = Array2::from_shape_fn((10, 6), |(i, j)| (i * j) as f64);

    // 1. Polynomial features
    let poly = crate::basic_features::PolynomialFeatures::new().degree(2);
    let poly_features = poly.transform(&data.view()).unwrap();

    // 2. RBF sampling on polynomial features
    let mut rbf = engineering::RBFSampler::new()
        .n_components(20)
        .random_state(Some(123));
    let rbf_features = rbf.fit_transform(&poly_features.view()).unwrap();

    // 3. Additive chi-squared sampling on subset
    let subset = rbf_features.slice(s![.., ..10]).to_owned();
    let chi2 = engineering::AdditiveChi2Sampler::new()
        .sample_steps(3)
        .sample_interval(1.0);
    let chi2_features = chi2.transform(&subset.view()).unwrap();

    // Verify pipeline produces valid results
    assert!(chi2_features.nrows() == data.nrows());
    assert!(chi2_features.ncols() > 0);

    for &val in chi2_features.iter() {
        assert!(val.is_finite());
    }
}
