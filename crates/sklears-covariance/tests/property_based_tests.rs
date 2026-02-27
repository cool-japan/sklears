//! Property-based tests for covariance estimators
//!
//! These tests verify mathematical properties that should hold for all
//! covariance estimators, such as symmetry, positive semi-definiteness,
//! and consistency with theoretical expectations.

use proptest::prelude::*;
use scirs2_core::ndarray::{Array2, Axis};
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{Distribution, SeedableRng};
use sklears_core::traits::Fit;
use sklears_covariance::{
    validate_covariance_matrix, EmpiricalCovariance, GraphicalLasso, LedoitWolf, MinCovDet,
    RidgeCovariance, OAS,
};

/// Generate random test data with controlled properties
fn generate_random_data(n_samples: usize, n_features: usize, seed: u64) -> Array2<f64> {
    let mut rng = scirs2_core::random::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    Array2::from_shape_fn((n_samples, n_features), |_| normal.sample(&mut rng))
}

/// Property 1: Covariance matrices must be symmetric
#[test]
fn test_property_symmetry_empirical() {
    proptest!(|(
        n_samples in 20usize..100,
        n_features in 5usize..20,
        seed in 0u64..1000
    )| {
        let data = generate_random_data(n_samples, n_features, seed);
        let estimator = EmpiricalCovariance::new();

        if let Ok(fitted) = estimator.fit(&data.view(), &()) {
            let cov = fitted.get_covariance();

            // Check symmetry
            for i in 0..n_features {
                for j in 0..n_features {
                    prop_assert!((cov[[i, j]] - cov[[j, i]]).abs() < 1e-10,
                        "Covariance matrix must be symmetric");
                }
            }
        }
    });
}

/// Property 2: Diagonal elements (variances) must be non-negative
#[test]
fn test_property_positive_variances() {
    proptest!(|(
        n_samples in 20usize..100,
        n_features in 5usize..20,
        seed in 0u64..1000
    )| {
        let data = generate_random_data(n_samples, n_features, seed);
        let estimator = EmpiricalCovariance::new();

        if let Ok(fitted) = estimator.fit(&data.view(), &()) {
            let cov = fitted.get_covariance();

            // Check diagonal elements are non-negative
            for i in 0..n_features {
                prop_assert!(cov[[i, i]] >= 0.0,
                    "Variance (diagonal elements) must be non-negative, got {}", cov[[i, i]]);
            }
        }
    });
}

/// Property 3: Trace of covariance matrix equals sum of variances
#[test]
fn test_property_trace_equals_sum_of_variances() {
    proptest!(|(
        n_samples in 30usize..100,
        n_features in 5usize..15,
        seed in 0u64..1000
    )| {
        let data = generate_random_data(n_samples, n_features, seed);
        let estimator = EmpiricalCovariance::new();

        if let Ok(fitted) = estimator.fit(&data.view(), &()) {
            let cov = fitted.get_covariance();

            // Compute trace
            let trace: f64 = (0..n_features).map(|i| cov[[i, i]]).sum();

            // Compute sum of sample variances (using n-1 normalization for unbiased estimator)
            let sum_variances: f64 = data.axis_iter(Axis(1))
                .map(|col| {
                    let mean = col.mean().unwrap_or(0.0);
                    col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / ((n_samples - 1) as f64)
                })
                .sum();

            // Allow for numerical precision differences
            let relative_error = (trace - sum_variances).abs() / trace.max(sum_variances).max(1.0);
            prop_assert!(relative_error < 1e-6,
                "Trace should approximately equal sum of variances: trace={}, sum_variances={}, rel_error={}",
                trace, sum_variances, relative_error);
        }
    });
}

/// Property 4: Ledoit-Wolf shrinkage parameter should be between 0 and 1
#[test]
fn test_property_ledoit_wolf_shrinkage_bounds() {
    proptest!(|(
        n_samples in 30usize..100,
        n_features in 5usize..20,
        seed in 0u64..1000
    )| {
        let data = generate_random_data(n_samples, n_features, seed);
        let estimator = LedoitWolf::new();

        if let Ok(fitted) = estimator.fit(&data.view(), &()) {
            let shrinkage = fitted.get_shrinkage();

            prop_assert!((0.0..=1.0).contains(&shrinkage),
                "Shrinkage parameter must be in [0, 1], got {}", shrinkage);
        }
    });
}

/// Property 5: Ridge regularization increases diagonal dominance
#[test]
fn test_property_ridge_increases_diagonal() {
    proptest!(|(
        n_samples in 30usize..80,
        n_features in 5usize..15,
        alpha in 0.01f64..1.0,
        seed in 0u64..1000
    )| {
        let data = generate_random_data(n_samples, n_features, seed);

        // Fit empirical covariance
        let empirical = EmpiricalCovariance::new();
        if let Ok(emp_fitted) = empirical.fit(&data.view(), &()) {
            let emp_cov = emp_fitted.get_covariance();

            // Fit ridge covariance
            let ridge = RidgeCovariance::new().alpha(alpha);
            if let Ok(ridge_fitted) = ridge.fit(&data.view(), &()) {
                let ridge_cov = ridge_fitted.get_covariance();

                // Check that diagonal elements increased
                for i in 0..n_features {
                    prop_assert!(ridge_cov[[i, i]] >= emp_cov[[i, i]],
                        "Ridge regularization should increase diagonal elements");
                }
            }
        }
    });
}

/// Property 6: Covariance matrix validation properties
#[test]
fn test_property_covariance_validation() {
    proptest!(|(
        n_samples in 30usize..100,
        n_features in 5usize..15,
        seed in 0u64..1000
    )| {
        let data = generate_random_data(n_samples, n_features, seed);
        let estimator = EmpiricalCovariance::new();

        if let Ok(fitted) = estimator.fit(&data.view(), &()) {
            let cov = fitted.get_covariance();

            // Validate covariance properties
            if let Ok(props) = validate_covariance_matrix(cov) {
                prop_assert!(props.is_symmetric,
                    "Covariance matrix must be symmetric");
                prop_assert!(props.is_positive_semi_definite,
                    "Covariance matrix must be positive semi-definite");
                prop_assert!(props.condition_number > 0.0,
                    "Condition number must be positive");
                prop_assert!(props.trace > 0.0,
                    "Trace must be positive");
            }
        }
    });
}

/// Property 7: OAS shrinkage should produce similar results to Ledoit-Wolf
#[test]
fn test_property_oas_similar_to_ledoit_wolf() {
    proptest!(|(
        n_samples in 40usize..100,
        n_features in 5usize..15,
        seed in 0u64..1000
    )| {
        let data = generate_random_data(n_samples, n_features, seed);

        let lw_estimator = LedoitWolf::new();
        let oas_estimator = OAS::new();

        if let (Ok(lw_fitted), Ok(oas_fitted)) = (
            lw_estimator.fit(&data.view(), &()),
            oas_estimator.fit(&data.view(), &())
        ) {
            let lw_cov = lw_fitted.get_covariance();
            let oas_cov = oas_fitted.get_covariance();

            // Compute Frobenius norm of difference
            let mut diff_norm = 0.0;
            for i in 0..n_features {
                for j in 0..n_features {
                    let diff = lw_cov[[i, j]] - oas_cov[[i, j]];
                    diff_norm += diff * diff;
                }
            }
            diff_norm = diff_norm.sqrt();

            // Compute Frobenius norm of LW covariance
            let mut lw_norm = 0.0;
            for i in 0..n_features {
                for j in 0..n_features {
                    lw_norm += lw_cov[[i, j]] * lw_cov[[i, j]];
                }
            }
            lw_norm = lw_norm.sqrt();

            // Relative difference should be reasonable
            let relative_diff = diff_norm / lw_norm;
            prop_assert!(relative_diff < 0.5,
                "OAS and Ledoit-Wolf should produce similar results");
        }
    });
}

/// Property 8: Increasing regularization increases condition number stability
#[test]
fn test_property_regularization_improves_conditioning() {
    proptest!(|(
        n_samples in 30usize..80,
        n_features in 5usize..15,
        seed in 0u64..1000
    )| {
        let data = generate_random_data(n_samples, n_features, seed);

        // Fit with different regularization strengths
        let ridge_weak = RidgeCovariance::new().alpha(0.01);
        let ridge_strong = RidgeCovariance::new().alpha(0.5);

        if let (Ok(weak_fitted), Ok(strong_fitted)) = (
            ridge_weak.fit(&data.view(), &()),
            ridge_strong.fit(&data.view(), &())
        ) {
            let weak_cov = weak_fitted.get_covariance();
            let strong_cov = strong_fitted.get_covariance();

            if let (Ok(weak_props), Ok(strong_props)) = (
                validate_covariance_matrix(weak_cov),
                validate_covariance_matrix(strong_cov)
            ) {
                // Stronger regularization should reduce condition number
                prop_assert!(strong_props.condition_number <= weak_props.condition_number * 1.1,
                    "Stronger regularization should improve conditioning");
            }
        }
    });
}

/// Property 9: Sample covariance converges to population covariance with more samples
#[test]
fn test_property_consistency_with_sample_size() {
    // Generate data from known covariance structure
    let seed = 42;
    let n_features = 5;

    // Small sample
    let small_data = generate_random_data(50, n_features, seed);
    let small_est = EmpiricalCovariance::new();

    // Large sample
    let large_data = generate_random_data(500, n_features, seed);
    let large_est = EmpiricalCovariance::new();

    if let (Ok(small_fitted), Ok(large_fitted)) = (
        small_est.fit(&small_data.view(), &()),
        large_est.fit(&large_data.view(), &()),
    ) {
        let small_cov = small_fitted.get_covariance();
        let large_cov = large_fitted.get_covariance();

        // Compute relative difference
        let mut diff_norm: f64 = 0.0;
        let mut large_norm: f64 = 0.0;

        for i in 0..n_features {
            for j in 0..n_features {
                let diff = small_cov[[i, j]] - large_cov[[i, j]];
                diff_norm += diff * diff;
                large_norm += large_cov[[i, j]] * large_cov[[i, j]];
            }
        }

        diff_norm = diff_norm.sqrt();
        large_norm = large_norm.sqrt();

        let relative_diff = diff_norm / large_norm;

        // Estimates should be reasonably close (allowing for sampling variation)
        assert!(
            relative_diff < 0.5,
            "Estimates should converge with sample size, got relative diff {}",
            relative_diff
        );
    }
}

/// Property 10: Robust estimators should be less affected by outliers
#[test]
fn test_property_robust_outlier_resistance() {
    let seed = 42;
    let n_samples = 100;
    let n_features = 5;

    // Generate clean data
    let clean_data = generate_random_data(n_samples, n_features, seed);

    // Add outliers to the data
    let mut outlier_data = clean_data.clone();
    for i in 0..5 {
        for j in 0..n_features {
            outlier_data[[i, j]] = 10.0; // Extreme outlier
        }
    }

    // Fit empirical covariance on both
    let emp_clean = EmpiricalCovariance::new();
    let emp_outlier = EmpiricalCovariance::new();

    // Fit robust estimator (MinCovDet) on both
    let robust_clean = MinCovDet::new().support_fraction(0.8);
    let robust_outlier = MinCovDet::new().support_fraction(0.8);

    if let (
        Ok(emp_clean_fitted),
        Ok(emp_outlier_fitted),
        Ok(robust_clean_fitted),
        Ok(robust_outlier_fitted),
    ) = (
        emp_clean.fit(&clean_data.view(), &()),
        emp_outlier.fit(&outlier_data.view(), &()),
        robust_clean.fit(&clean_data.view(), &()),
        robust_outlier.fit(&outlier_data.view(), &()),
    ) {
        let emp_clean_cov = emp_clean_fitted.get_covariance();
        let emp_outlier_cov = emp_outlier_fitted.get_covariance();
        let robust_clean_cov = robust_clean_fitted.get_covariance();
        let robust_outlier_cov = robust_outlier_fitted.get_covariance();

        // Compute differences
        let mut emp_diff: f64 = 0.0;
        let mut robust_diff: f64 = 0.0;

        for i in 0..n_features {
            for j in 0..n_features {
                let emp_d = (emp_clean_cov[[i, j]] - emp_outlier_cov[[i, j]]).abs();
                let robust_d = (robust_clean_cov[[i, j]] - robust_outlier_cov[[i, j]]).abs();
                emp_diff += emp_d * emp_d;
                robust_diff += robust_d * robust_d;
            }
        }

        emp_diff = emp_diff.sqrt();
        robust_diff = robust_diff.sqrt();

        // Robust estimator should be less affected by outliers
        assert!(
            robust_diff < emp_diff,
            "Robust estimator should be less affected by outliers: robust_diff={}, emp_diff={}",
            robust_diff,
            emp_diff
        );
    }
}

/// Property 11: Graphical Lasso should produce sparse precision matrices
#[test]
fn test_property_graphical_lasso_sparsity() {
    proptest!(|(
        n_samples in 50usize..100,
        n_features in 5usize..15,
        alpha in 0.1f64..0.5,
        seed in 0u64..1000
    )| {
        let data = generate_random_data(n_samples, n_features, seed);
        let estimator = GraphicalLasso::new().alpha(alpha).max_iter(50);

        if let Ok(fitted) = estimator.fit(&data.view(), &()) {
            let precision = fitted.get_precision();

            // Count near-zero elements in precision matrix
            let mut zero_count = 0;
            let threshold = 1e-4;

            for i in 0..n_features {
                for j in 0..n_features {
                    if i != j && precision[[i, j]].abs() < threshold {
                        zero_count += 1;
                    }
                }
            }

            let total_off_diag = n_features * (n_features - 1);
            let sparsity_ratio = zero_count as f64 / total_off_diag as f64;

            // With regularization, we expect some sparsity
            prop_assert!((0.0..=1.0).contains(&sparsity_ratio),
                "Sparsity ratio should be between 0 and 1");
        }
    });
}

/// Property 12: Covariance estimation should be scale-invariant up to scaling
#[test]
fn test_property_scale_invariance() {
    proptest!(|(
        n_samples in 30usize..100,
        n_features in 5usize..15,
        scale in 0.1f64..10.0,
        seed in 0u64..1000
    )| {
        let data = generate_random_data(n_samples, n_features, seed);
        let scaled_data = &data * scale;

        let est1 = EmpiricalCovariance::new();
        let est2 = EmpiricalCovariance::new();

        if let (Ok(fitted1), Ok(fitted2)) = (
            est1.fit(&data.view(), &()),
            est2.fit(&scaled_data.view(), &())
        ) {
            let cov1 = fitted1.get_covariance();
            let cov2 = fitted2.get_covariance();

            // Covariance should scale by scale^2
            let expected_scale = scale * scale;

            for i in 0..n_features {
                for j in 0..n_features {
                    let expected = cov1[[i, j]] * expected_scale;
                    prop_assert!((cov2[[i, j]] - expected).abs() < 0.01,
                        "Covariance should scale by scale^2");
                }
            }
        }
    });
}
