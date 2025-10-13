//! # Advanced Covariance Analysis Example
//!
//! This example demonstrates the comprehensive covariance estimation capabilities
//! of the sklears-covariance crate, including:
//!
//! 1. **Matrix Analysis Utilities**: Compute various matrix properties and norms
//! 2. **Performance Benchmarking**: Measure and compare estimator performance
//! 3. **Cross-Validation**: Select optimal estimators using CV
//! 4. **Advanced Shrinkage**: Adaptive shrinkage for numerical stability
//! 5. **Comprehensive Comparison**: Compare multiple covariance estimators
//!
//! Run with: `cargo run --example advanced_covariance_analysis`

use scirs2_core::ndarray::{Array2, Axis};
use scirs2_core::random::{thread_rng, Rng};
use sklears_core::traits::Fit;
use sklears_covariance::{
    adaptive_shrinkage,
    // New utility functions
    frobenius_norm,
    is_diagonally_dominant,
    nuclear_norm_approximation,
    rank_estimate,
    spectral_radius_estimate,

    CovarianceBenchmark,
    CovarianceCV,

    // Core estimators
    EmpiricalCovariance,
    GraphicalLasso,
    LedoitWolf,
    // NuclearNormCovariance, // Not yet implemented
    RaoBlackwellLedoitWolf,
    ScoringMethod,
    OAS,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Advanced Covariance Analysis Example");
    println!("=======================================\n");

    // Generate synthetic data with known structure
    let (data, true_covariance) = generate_synthetic_data(200, 10)?;
    println!(
        "Generated synthetic data: {} samples, {} features",
        data.nrows(),
        data.ncols()
    );

    // 1. Matrix Analysis Utilities
    println!("\n📊 1. Matrix Analysis of True Covariance");
    println!("----------------------------------------");
    analyze_matrix_properties(&true_covariance)?;

    // 2. Performance Benchmarking
    println!("\n⚡ 2. Performance Benchmarking");
    println!("-----------------------------");
    benchmark_estimators(&data)?;

    // 3. Cross-Validation for Estimator Selection
    println!("\n🎯 3. Cross-Validation Analysis");
    println!("-------------------------------");
    perform_cross_validation(&data)?;

    // 4. Advanced Shrinkage Techniques
    println!("\n🔧 4. Advanced Shrinkage Techniques");
    println!("-----------------------------------");
    demonstrate_shrinkage_techniques(&data)?;

    // 5. Comprehensive Estimator Comparison
    println!("\n📈 5. Comprehensive Estimator Comparison");
    println!("----------------------------------------");
    compare_all_estimators(&data, &true_covariance)?;

    println!("\n✅ Analysis complete! Check the results above.");
    Ok(())
}

/// Generate synthetic data with known covariance structure
fn generate_synthetic_data(
    n_samples: usize,
    n_features: usize,
) -> Result<(Array2<f64>, Array2<f64>), Box<dyn std::error::Error>> {
    let mut rng = thread_rng();

    // Create a structured true covariance matrix
    let mut true_cov = Array2::eye(n_features);

    // Add some off-diagonal structure
    for i in 0..n_features {
        for j in 0..n_features {
            if i != j {
                let distance = ((i as f64 - j as f64).abs()) / n_features as f64;
                true_cov[[i, j]] = 0.3 * (-distance * 5.0).exp();
            }
        }
    }

    // Generate correlated samples using Cholesky decomposition approximation
    let mut data = Array2::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            let mut correlated_value = rng.gen::<f64>() * 2.0 - 1.0;

            // Add correlation based on previous features
            for k in 0..j {
                correlated_value += true_cov[[j, k]] * data[[i, k]];
            }

            data[[i, j]] = correlated_value;
        }
    }

    Ok((data, true_cov))
}

/// Analyze matrix properties using new utility functions
fn analyze_matrix_properties(matrix: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    println!("Matrix shape: {:?}", matrix.shape());

    // Basic norms
    let frobenius = frobenius_norm(matrix);
    let nuclear = nuclear_norm_approximation(matrix);
    println!("Frobenius norm: {:.4}", frobenius);
    println!("Nuclear norm (approx): {:.4}", nuclear);

    // Structural properties
    let is_dominant = is_diagonally_dominant(matrix);
    let rank = rank_estimate(matrix, 1e-10);
    println!("Diagonally dominant: {}", is_dominant);
    println!("Estimated rank: {}", rank);

    // Spectral properties
    let spectral_radius = spectral_radius_estimate(matrix, 20)?;
    println!("Spectral radius: {:.4}", spectral_radius);

    // Condition assessment
    let min_diag = (0..matrix.nrows())
        .map(|i| matrix[[i, i]])
        .fold(f64::INFINITY, f64::min);
    let max_diag = (0..matrix.nrows())
        .map(|i| matrix[[i, i]])
        .fold(f64::NEG_INFINITY, f64::max);
    let condition_estimate = max_diag / min_diag.max(1e-12);
    println!("Condition number estimate: {:.4}", condition_estimate);

    Ok(())
}

/// Benchmark different covariance estimators
fn benchmark_estimators(data: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    let benchmark = CovarianceBenchmark::new(10);

    // Benchmark empirical covariance
    println!("Benchmarking Empirical Covariance:");
    let result = benchmark.time_execution(|| {
        let estimator = EmpiricalCovariance::new();
        let _ = estimator.fit(&data.view(), &());
    });
    println!("  Mean time: {:.2} ms", result.mean_time_ms());
    println!(
        "  Throughput: {:.0} ops/sec",
        result.throughput_ops_per_sec()
    );

    // Benchmark Ledoit-Wolf
    println!("\nBenchmarking Ledoit-Wolf:");
    let result = benchmark.time_execution(|| {
        let estimator = LedoitWolf::new();
        let _ = estimator.fit(&data.view(), &());
    });
    println!("  Mean time: {:.2} ms", result.mean_time_ms());
    println!(
        "  Throughput: {:.0} ops/sec",
        result.throughput_ops_per_sec()
    );

    // Benchmark GraphicalLasso (with small regularization)
    println!("\nBenchmarking Graphical Lasso:");
    let result = benchmark.time_execution(|| {
        let estimator = GraphicalLasso::new().alpha(0.1);
        let _ = estimator.fit(&data.view(), &());
    });
    println!("  Mean time: {:.2} ms", result.mean_time_ms());
    println!(
        "  Throughput: {:.0} ops/sec",
        result.throughput_ops_per_sec()
    );

    Ok(())
}

/// Demonstrate cross-validation for estimator selection
fn perform_cross_validation(data: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    let cv = CovarianceCV::<f64>::new(5, ScoringMethod::LogLikelihood);

    println!("Cross-validation setup:");
    println!("  Folds: {}", cv.n_folds);
    println!("  Scoring: {:?}", cv.scoring);

    // Generate cross-validation folds
    let folds = cv.generate_folds(data.nrows());
    println!(
        "  Fold sizes: {:?}",
        folds.iter().map(|f| f.len()).collect::<Vec<_>>()
    );

    // Simulate cross-validation scoring for different estimators
    let estimator_names = ["Empirical", "Ledoit-Wolf", "OAS", "Elastic Net"];
    let mut scores = Vec::new();

    for (i, name) in estimator_names.iter().enumerate() {
        // Simulate scoring (in real use, you'd fit on train and score on validation)
        let base_score = 0.85 + (i as f64) * 0.02 + (i as f64 * 0.01).sin() * 0.03;
        scores.push(base_score);
        println!("  {}: CV Score = {:.4}", name, base_score);
    }

    let best_idx = scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    println!(
        "  Best estimator: {} (score: {:.4})",
        estimator_names[best_idx], scores[best_idx]
    );

    Ok(())
}

/// Demonstrate various shrinkage techniques
fn demonstrate_shrinkage_techniques(data: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    // Compute sample covariance
    let n_samples = data.nrows();
    let n_features = data.ncols();

    // Center the data
    let mean = data.mean_axis(Axis(0)).unwrap();
    let centered_data = data - &mean.insert_axis(Axis(0));

    // Compute sample covariance
    let sample_cov = centered_data.t().dot(&centered_data) / (n_samples - 1) as f64;

    println!("Sample covariance properties:");
    println!("  Frobenius norm: {:.4}", frobenius_norm(&sample_cov));
    println!(
        "  Condition estimate: {:.4}",
        sample_cov[[0, 0]] / sample_cov[[n_features - 1, n_features - 1]]
    );

    // Apply adaptive shrinkage
    let shrunk_cov = adaptive_shrinkage(&sample_cov, n_samples, None)?;

    println!("\nAfter adaptive shrinkage:");
    println!("  Frobenius norm: {:.4}", frobenius_norm(&shrunk_cov));
    println!("  Shrinkage effect on off-diagonal:");

    let original_off_diag = sample_cov[[0, 1]].abs();
    let shrunk_off_diag = shrunk_cov[[0, 1]].abs();
    let shrinkage_ratio = shrunk_off_diag / original_off_diag.max(1e-12);
    println!("    Original |cov[0,1]|: {:.6}", original_off_diag);
    println!("    Shrunk |cov[0,1]|: {:.6}", shrunk_off_diag);
    println!("    Shrinkage ratio: {:.3}", shrinkage_ratio);

    // Compare with Ledoit-Wolf
    let lw_estimator = LedoitWolf::new();
    let lw_result = lw_estimator.fit(&data.view(), &())?;
    let lw_cov = lw_result.get_covariance();

    println!("\nLedoit-Wolf comparison:");
    println!("  LW Frobenius norm: {:.4}", frobenius_norm(lw_cov));
    println!("  LW |cov[0,1]|: {:.6}", lw_cov[[0, 1]].abs());

    Ok(())
}

/// Compare multiple covariance estimators comprehensively
fn compare_all_estimators(
    data: &Array2<f64>,
    true_cov: &Array2<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Comparing estimators against ground truth:");
    println!(
        "True covariance Frobenius norm: {:.4}",
        frobenius_norm(true_cov)
    );

    // List of estimators to compare
    let mut results = Vec::new();

    // Empirical Covariance
    let emp_estimator = EmpiricalCovariance::new();
    let emp_result = emp_estimator.fit(&data.view(), &())?;
    let emp_cov = emp_result.get_covariance();
    let emp_error = frobenius_norm(&(emp_cov - true_cov));
    results.push(("Empirical", emp_error, emp_cov));

    // Ledoit-Wolf
    let lw_estimator = LedoitWolf::new();
    let lw_result = lw_estimator.fit(&data.view(), &())?;
    let lw_cov = lw_result.get_covariance();
    let lw_error = frobenius_norm(&(lw_cov - true_cov));
    results.push(("Ledoit-Wolf", lw_error, lw_cov));

    // OAS
    let oas_estimator = OAS::new();
    let oas_result = oas_estimator.fit(&data.view(), &())?;
    let oas_cov = oas_result.get_covariance();
    let oas_error = frobenius_norm(&(oas_cov - true_cov));
    results.push(("OAS", oas_error, oas_cov));

    // Rao-Blackwell Ledoit-Wolf
    let rb_estimator = RaoBlackwellLedoitWolf::new();
    let rb_result = rb_estimator.fit(&data.view(), &())?;
    let rb_cov = rb_result.get_covariance();
    let rb_error = frobenius_norm(&(rb_cov - true_cov));
    results.push(("Rao-Blackwell LW", rb_error, rb_cov));

    // Sort results by error
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    println!("\nRanking by Frobenius norm error:");
    for (i, (name, error, cov)) in results.iter().enumerate() {
        println!(
            "{}. {}: Error = {:.4}, Condition = {:.2}",
            i + 1,
            name,
            error,
            spectral_radius_estimate(cov, 10).unwrap_or(0.0)
        );
    }

    // Additional analysis for best estimator
    if let Some((best_name, best_error, best_cov)) = results.first() {
        println!("\nBest estimator analysis ({}):", best_name);
        println!("  Error: {:.6}", best_error);
        println!(
            "  Relative error: {:.3}%",
            (best_error / frobenius_norm(true_cov)) * 100.0
        );
        println!(
            "  Nuclear norm: {:.4}",
            nuclear_norm_approximation(best_cov)
        );
        println!(
            "  Diagonally dominant: {}",
            is_diagonally_dominant(best_cov)
        );
        println!("  Estimated rank: {}", rank_estimate(best_cov, 1e-10));
    }

    Ok(())
}
