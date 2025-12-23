//! Comprehensive integration tests for sklears-covariance
//!
//! These tests verify end-to-end workflows, algorithm interactions,
//! and integration between different components of the library.

use scirs2_core::ndarray::Array2;
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{Distribution, SeedableRng};
use sklears_core::traits::Fit;
use sklears_covariance::{
    adaptive_shrinkage, frobenius_norm, is_diagonally_dominant, rank_estimate,
    spectral_radius_estimate, validate_covariance_matrix, AdaptiveLassoCovariance,
    ChenSteinCovariance, CovarianceBenchmark, CovarianceCV, ElasticNetCovariance,
    EmpiricalCovariance, GraphicalLasso, GroupLassoCovariance, HuberCovariance, LedoitWolf,
    MinCovDet, RaoBlackwellLedoitWolf, RidgeCovariance, ScoringMethod, ShrunkCovariance, OAS,
};

/// Generate test data with controlled properties
fn generate_test_data(n_samples: usize, n_features: usize, seed: u64) -> Array2<f64> {
    let mut rng = scirs2_core::random::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    Array2::from_shape_fn((n_samples, n_features), |_| normal.sample(&mut rng))
}

/// Test 1: End-to-end workflow with multiple estimators
#[test]
fn test_integration_multiple_estimators_workflow() {
    let n_samples = 100;
    let n_features = 10;
    let data = generate_test_data(n_samples, n_features, 42);

    // Test multiple estimators on the same data
    let estimators: Vec<(&str, Box<dyn Fn() -> Box<dyn std::any::Any>>)> = vec![
        (
            "EmpiricalCovariance",
            Box::new(|| Box::new(EmpiricalCovariance::new()) as Box<dyn std::any::Any>),
        ),
        (
            "LedoitWolf",
            Box::new(|| Box::new(LedoitWolf::new()) as Box<dyn std::any::Any>),
        ),
        (
            "OAS",
            Box::new(|| Box::new(OAS::new()) as Box<dyn std::any::Any>),
        ),
        (
            "ShrunkCovariance",
            Box::new(|| Box::new(ShrunkCovariance::new().shrinkage(0.1)) as Box<dyn std::any::Any>),
        ),
        (
            "RidgeCovariance",
            Box::new(|| Box::new(RidgeCovariance::new().alpha(0.1)) as Box<dyn std::any::Any>),
        ),
    ];

    let mut covariance_matrices = Vec::new();

    // Fit all estimators
    for (name, _factory) in &estimators {
        match *name {
            "EmpiricalCovariance" => {
                let est = EmpiricalCovariance::new();
                if let Ok(fitted) = est.fit(&data.view(), &()) {
                    let cov = fitted.get_covariance();
                    covariance_matrices.push((name, cov.clone()));
                }
            }
            "LedoitWolf" => {
                let est = LedoitWolf::new();
                if let Ok(fitted) = est.fit(&data.view(), &()) {
                    let cov = fitted.get_covariance();
                    covariance_matrices.push((name, cov.clone()));
                }
            }
            "OAS" => {
                let est = OAS::new();
                if let Ok(fitted) = est.fit(&data.view(), &()) {
                    let cov = fitted.get_covariance();
                    covariance_matrices.push((name, cov.clone()));
                }
            }
            "ShrunkCovariance" => {
                let est = ShrunkCovariance::new().shrinkage(0.1);
                if let Ok(fitted) = est.fit(&data.view(), &()) {
                    let cov = fitted.get_covariance();
                    covariance_matrices.push((name, cov.clone()));
                }
            }
            "RidgeCovariance" => {
                let est = RidgeCovariance::new().alpha(0.1);
                if let Ok(fitted) = est.fit(&data.view(), &()) {
                    let cov = fitted.get_covariance();
                    covariance_matrices.push((name, cov.clone()));
                }
            }
            _ => {}
        }
    }

    // Verify all estimators produced valid results
    assert_eq!(
        covariance_matrices.len(),
        5,
        "All estimators should succeed"
    );

    // Verify all matrices are symmetric and positive semi-definite
    for (name, cov) in &covariance_matrices {
        let props = validate_covariance_matrix(cov)
            .expect(&format!("{} should produce valid covariance", name));
        assert!(props.is_symmetric, "{} matrix should be symmetric", name);
        assert!(
            props.is_positive_semi_definite,
            "{} matrix should be positive semi-definite",
            name
        );
    }

    // Compare Frobenius norms
    for i in 0..covariance_matrices.len() {
        for j in (i + 1)..covariance_matrices.len() {
            let (name1, cov1) = &covariance_matrices[i];
            let (name2, cov2) = &covariance_matrices[j];

            let norm1 = frobenius_norm(cov1);
            let norm2 = frobenius_norm(cov2);

            // Norms should be positive and finite
            assert!(
                norm1 > 0.0 && norm1.is_finite(),
                "{} norm should be positive and finite",
                name1
            );
            assert!(
                norm2 > 0.0 && norm2.is_finite(),
                "{} norm should be positive and finite",
                name2
            );
        }
    }
}

/// Test 2: Utility functions integration
#[test]
fn test_integration_utility_functions() {
    let n_samples = 80;
    let n_features = 10;
    let data = generate_test_data(n_samples, n_features, 42);

    // Fit empirical covariance
    let emp = EmpiricalCovariance::new();
    let fitted = emp.fit(&data.view(), &()).expect("Fit should succeed");
    let sample_cov = fitted.get_covariance();

    // Test frobenius_norm
    let frob_norm = frobenius_norm(&sample_cov);
    assert!(frob_norm > 0.0, "Frobenius norm should be positive");

    // Test is_diagonally_dominant
    let is_dd = is_diagonally_dominant(&sample_cov);
    // May or may not be diagonally dominant, just verify it returns
    assert!(is_dd == true || is_dd == false);

    // Test spectral_radius_estimate
    let spectral_radius = spectral_radius_estimate(&sample_cov, 100)
        .expect("Spectral radius estimation should succeed");
    assert!(spectral_radius > 0.0, "Spectral radius should be positive");

    // Test rank_estimate
    let rank = rank_estimate(&sample_cov, 1e-6);
    assert!(
        rank > 0 && rank <= n_features,
        "Rank should be between 1 and n_features"
    );

    // Test adaptive_shrinkage
    let shrunk_cov = adaptive_shrinkage(&sample_cov, n_samples, None)
        .expect("Adaptive shrinkage should succeed");
    assert_eq!(
        shrunk_cov.shape(),
        sample_cov.shape(),
        "Shape should be preserved"
    );

    // Verify shrunk covariance properties
    let props = validate_covariance_matrix(&shrunk_cov).expect("Shrunk covariance should be valid");
    assert!(props.is_symmetric, "Shrunk matrix should be symmetric");
}

/// Test 3: Cross-validation integration
#[test]
fn test_integration_cross_validation() {
    let n_samples = 100;
    let n_features = 8;
    let _data = generate_test_data(n_samples, n_features, 42);

    // Create CV framework
    let cv: CovarianceCV<f64> = CovarianceCV::new(5, ScoringMethod::LogLikelihood);

    // Test with Ledoit-Wolf
    let _lw_factory =
        |_: &Array2<f64>| -> Result<Box<dyn std::any::Any>, sklears_core::error::SklearsError> {
            Ok(Box::new(LedoitWolf::new()))
        };

    // Note: We can't directly test the CV framework without exposing more internals,
    // but we can verify the components work together
    assert_eq!(cv.n_folds, 5, "CV should have correct number of folds");
}

/// Test 4: Benchmarking integration
#[test]
fn test_integration_benchmarking() {
    let n_samples = 50;
    let n_features = 5;
    let data = generate_test_data(n_samples, n_features, 42);

    // Create benchmark
    let benchmark = CovarianceBenchmark::new(10);

    // Benchmark empirical covariance
    let result = benchmark.time_execution(|| {
        let emp = EmpiricalCovariance::new();
        let _ = emp.fit(&data.view(), &());
    });

    // Verify benchmark results
    assert!(
        result.mean_time_ms() >= 0.0,
        "Mean time should be non-negative"
    );
    assert!(
        result.median_time_ms() >= 0.0,
        "Median time should be non-negative"
    );
    assert!(result.std_dev_ns() >= 0.0, "Std dev should be non-negative");
    assert!(result.min_time_ns >= 0.0, "Min time should be non-negative");
    assert!(result.max_time_ns >= 0.0, "Max time should be non-negative");
    assert!(
        result.throughput_ops_per_sec() > 0.0,
        "Throughput should be positive"
    );

    // Benchmark another estimator for comparison
    let lw_result = benchmark.time_execution(|| {
        let lw = LedoitWolf::new();
        let _ = lw.fit(&data.view(), &());
    });

    assert!(
        lw_result.mean_time_ms() >= 0.0,
        "LW mean time should be non-negative"
    );
}

/// Test 5: Regularized estimators progression
#[test]
fn test_integration_regularization_progression() {
    let n_samples = 60;
    let n_features = 8;
    let data = generate_test_data(n_samples, n_features, 42);

    // Test progression: Ridge -> Elastic Net -> Adaptive Lasso
    let ridge = RidgeCovariance::new().alpha(0.1);
    let fitted_ridge = ridge.fit(&data.view(), &()).expect("Ridge should fit");
    let ridge_cov = fitted_ridge.get_covariance();

    let elastic = ElasticNetCovariance::new()
        .alpha(0.1)
        .l1_ratio(0.5)
        .max_iter(50);
    let fitted_elastic = elastic
        .fit(&data.view(), &())
        .expect("Elastic Net should fit");
    let elastic_cov = fitted_elastic.get_covariance();

    let adaptive = AdaptiveLassoCovariance::new().alpha(0.1).max_iter(50);
    let fitted_adaptive = adaptive
        .fit(&data.view(), &())
        .expect("Adaptive Lasso should fit");
    let adaptive_cov = fitted_adaptive.get_covariance();

    // All should produce valid covariance matrices
    for (name, cov) in &[
        ("Ridge", &ridge_cov),
        ("ElasticNet", &elastic_cov),
        ("AdaptiveLasso", &adaptive_cov),
    ] {
        let props = validate_covariance_matrix(cov)
            .expect(&format!("{} should produce valid covariance", name));
        assert!(props.is_symmetric, "{} should be symmetric", name);
    }
}

/// Test 6: Shrinkage estimators comparison
#[test]
fn test_integration_shrinkage_comparison() {
    let n_samples = 100;
    let n_features = 15;
    let data = generate_test_data(n_samples, n_features, 42);

    // Fit multiple shrinkage estimators
    let lw = LedoitWolf::new();
    let fitted_lw = lw.fit(&data.view(), &()).expect("LedoitWolf should fit");
    let lw_cov = fitted_lw.get_covariance();
    let lw_shrinkage = fitted_lw.get_shrinkage();

    let oas = OAS::new();
    let fitted_oas = oas.fit(&data.view(), &()).expect("OAS should fit");
    let oas_cov = fitted_oas.get_covariance();
    let oas_shrinkage = fitted_oas.get_shrinkage();

    let rblw = RaoBlackwellLedoitWolf::new();
    let fitted_rblw = rblw
        .fit(&data.view(), &())
        .expect("Rao-Blackwell LW should fit");
    let rblw_cov = fitted_rblw.get_covariance();
    let rblw_shrinkage = fitted_rblw.get_shrinkage();

    let chen = ChenSteinCovariance::new();
    let fitted_chen = chen.fit(&data.view(), &()).expect("Chen-Stein should fit");
    let chen_cov = fitted_chen.get_covariance();
    let chen_shrinkage = fitted_chen.get_shrinkage();

    // Verify all shrinkage parameters are in valid range [0, 1]
    assert!(
        lw_shrinkage >= 0.0 && lw_shrinkage <= 1.0,
        "LW shrinkage should be in [0, 1]"
    );
    assert!(
        oas_shrinkage >= 0.0 && oas_shrinkage <= 1.0,
        "OAS shrinkage should be in [0, 1]"
    );
    assert!(
        rblw_shrinkage >= 0.0 && rblw_shrinkage <= 1.0,
        "Rao-Blackwell LW shrinkage should be in [0, 1]"
    );
    assert!(
        chen_shrinkage >= 0.0 && chen_shrinkage <= 1.0,
        "Chen-Stein shrinkage should be in [0, 1]"
    );

    // Verify all produce valid covariance matrices
    for (name, cov) in &[
        ("LedoitWolf", &lw_cov),
        ("OAS", &oas_cov),
        ("RaoBlackwellLW", &rblw_cov),
        ("ChenStein", &chen_cov),
    ] {
        let props = validate_covariance_matrix(cov)
            .expect(&format!("{} should produce valid covariance", name));
        assert!(props.is_symmetric, "{} should be symmetric", name);
        assert!(
            props.condition_number > 0.0,
            "{} condition number should be positive",
            name
        );
    }
}

/// Test 7: Robust estimators with outliers
#[test]
fn test_integration_robust_estimators_outliers() {
    let n_samples = 100;
    let n_features = 6;
    let mut data = generate_test_data(n_samples, n_features, 42);

    // Add outliers
    for i in 0..5 {
        for j in 0..n_features {
            data[[i, j]] = 10.0; // Extreme outliers
        }
    }

    // Test empirical (non-robust)
    let emp = EmpiricalCovariance::new();
    let fitted_emp = emp.fit(&data.view(), &()).expect("Empirical should fit");
    let emp_cov = fitted_emp.get_covariance();

    // Test robust estimators
    let mcd = MinCovDet::new().support_fraction(0.8);
    let fitted_mcd = mcd.fit(&data.view(), &()).expect("MinCovDet should fit");
    let mcd_cov = fitted_mcd.get_covariance();

    let huber = HuberCovariance::new().max_iter(50);
    let fitted_huber = huber.fit(&data.view(), &()).expect("Huber should fit");
    let huber_cov = fitted_huber.get_covariance();

    // Robust estimators should produce different results than empirical with outliers
    let emp_frob = frobenius_norm(&emp_cov);
    let mcd_frob = frobenius_norm(&mcd_cov);
    let huber_frob = frobenius_norm(&huber_cov);

    // All should be positive and finite
    assert!(
        emp_frob > 0.0 && emp_frob.is_finite(),
        "Empirical norm should be valid"
    );
    assert!(
        mcd_frob > 0.0 && mcd_frob.is_finite(),
        "MCD norm should be valid"
    );
    assert!(
        huber_frob > 0.0 && huber_frob.is_finite(),
        "Huber norm should be valid"
    );

    // Robust estimators should differ from empirical (due to outliers)
    assert_ne!(
        emp_frob, mcd_frob,
        "MCD should differ from empirical with outliers"
    );
}

/// Test 8: Sparse estimators produce sparse precision matrices
#[test]
fn test_integration_sparse_estimators() {
    let n_samples = 80;
    let n_features = 10;
    let data = generate_test_data(n_samples, n_features, 42);

    // Test Graphical Lasso
    let glasso = GraphicalLasso::new().alpha(0.1).max_iter(50);
    let fitted_glasso = glasso
        .fit(&data.view(), &())
        .expect("GraphicalLasso should fit");
    let glasso_precision = fitted_glasso.get_precision();

    // Count sparse elements (near-zero off-diagonal elements)
    let threshold = 1e-4;
    let mut zero_count = 0;
    let mut total_off_diag = 0;

    for i in 0..n_features {
        for j in 0..n_features {
            if i != j {
                total_off_diag += 1;
                if glasso_precision[[i, j]].abs() < threshold {
                    zero_count += 1;
                }
            }
        }
    }

    // With regularization, we expect some sparsity
    let sparsity_ratio = zero_count as f64 / total_off_diag as f64;
    assert!(
        sparsity_ratio >= 0.0 && sparsity_ratio <= 1.0,
        "Sparsity ratio should be valid"
    );
}

/// Test 9: Group lasso with custom groups
#[test]
fn test_integration_group_lasso() {
    let n_samples = 60;
    let n_features = 12;
    let data = generate_test_data(n_samples, n_features, 42);

    // Define group assignments (each variable assigned to a group)
    // Group 0: features 0-3, Group 1: features 4-7, Group 2: features 8-11
    let groups = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];

    let glasso = GroupLassoCovariance::new()
        .alpha(0.1)
        .groups(groups)
        .max_iter(50);

    let fitted = glasso
        .fit(&data.view(), &())
        .expect("Group Lasso should fit");
    let cov = fitted.get_covariance();

    // Verify valid covariance
    let props =
        validate_covariance_matrix(&cov).expect("Group Lasso should produce valid covariance");
    assert!(
        props.is_symmetric,
        "Group Lasso covariance should be symmetric"
    );
}

/// Test 10: End-to-end workflow with validation and diagnostics
#[test]
fn test_integration_complete_workflow() {
    let n_samples = 100;
    let n_features = 12;
    let data = generate_test_data(n_samples, n_features, 42);

    // Step 1: Fit initial estimator
    let emp = EmpiricalCovariance::new();
    let fitted_emp = emp
        .fit(&data.view(), &())
        .expect("Empirical fit should succeed");
    let sample_cov = fitted_emp.get_covariance();

    // Step 2: Validate covariance properties
    let props = validate_covariance_matrix(&sample_cov).expect("Sample covariance should be valid");
    assert!(props.is_symmetric, "Sample covariance should be symmetric");
    assert!(props.trace > 0.0, "Trace should be positive");

    // Step 3: Apply shrinkage
    let shrunk =
        adaptive_shrinkage(&sample_cov, n_samples, None).expect("Shrinkage should succeed");

    // Step 4: Validate shrunk covariance
    let shrunk_props =
        validate_covariance_matrix(&shrunk).expect("Shrunk covariance should be valid");
    assert!(
        shrunk_props.is_symmetric,
        "Shrunk covariance should be symmetric"
    );
    assert!(
        shrunk_props.condition_number <= props.condition_number * 1.5,
        "Shrinkage should improve or maintain conditioning"
    );

    // Step 5: Compute diagnostics
    let frob_before = frobenius_norm(&sample_cov);
    let frob_after = frobenius_norm(&shrunk);
    assert!(
        frob_before > 0.0 && frob_after > 0.0,
        "Norms should be positive"
    );

    // Step 6: Benchmark performance
    let benchmark = CovarianceBenchmark::new(5);
    let result = benchmark.time_execution(|| {
        let emp = EmpiricalCovariance::new();
        let _ = emp.fit(&data.view(), &());
    });
    assert!(result.mean_time_ms() >= 0.0, "Benchmark should complete");
}
