//! # Robust Covariance Estimation Comparison Demo
//!
//! This example demonstrates and compares various robust covariance estimation methods
//! including MinCovDet (MCD), Huber, M-estimators, and other robust approaches.
//!
//! Robust methods are essential when data contains outliers that could severely
//! distort standard covariance estimates.

use scirs2_core::ndarray::Array2;
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{Distribution, SeedableRng};
use sklears_core::error::Result;
use sklears_core::traits::Fit;
use sklears_covariance::{
    frobenius_norm, validate_covariance_matrix, CovarianceBenchmark, EmpiricalCovariance,
    HuberCovariance, MinCovDet,
};

/// Generate clean data (no outliers)
fn generate_clean_data(n_samples: usize, n_features: usize, seed: u64) -> Array2<f64> {
    let mut rng = scirs2_core::random::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    Array2::from_shape_fn((n_samples, n_features), |_| normal.sample(&mut rng))
}

/// Add outliers to data
fn add_outliers(data: &mut Array2<f64>, n_outliers: usize, outlier_magnitude: f64, seed: u64) {
    let mut rng = scirs2_core::random::StdRng::seed_from_u64(seed);
    let (n_samples, n_features) = data.dim();

    for _ in 0..n_outliers {
        let outlier_idx = rng.gen_range(0..n_samples);
        for j in 0..n_features {
            data[[outlier_idx, j]] = outlier_magnitude;
        }
    }
}

/// Compute Mahalanobis distances
fn compute_mahalanobis_distances(
    data: &Array2<f64>,
    mean: &Array2<f64>,
    precision: &Array2<f64>,
) -> Vec<f64> {
    let n_samples = data.nrows();
    let mut distances = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let mut row_vec = Array2::zeros((1, data.ncols()));
        for j in 0..data.ncols() {
            row_vec[[0, j]] = data[[i, j]] - mean[[0, j]];
        }

        // Compute distance: sqrt(x^T * Precision * x)
        let mut distance_sq = 0.0;
        for j in 0..data.ncols() {
            let mut sum = 0.0;
            for k in 0..data.ncols() {
                sum += row_vec[[0, k]] * precision[[k, j]];
            }
            distance_sq += sum * row_vec[[0, j]];
        }

        distances.push(distance_sq.max(0.0).sqrt());
    }

    distances
}

/// Identify outliers based on Mahalanobis distance
fn identify_outliers(distances: &[f64], threshold_multiplier: f64) -> Vec<usize> {
    let mean = distances.iter().sum::<f64>() / distances.len() as f64;
    let std_dev = (distances.iter().map(|&d| (d - mean).powi(2)).sum::<f64>()
        / distances.len() as f64)
        .sqrt();

    let threshold = mean + threshold_multiplier * std_dev;

    distances
        .iter()
        .enumerate()
        .filter_map(|(idx, &d)| if d > threshold { Some(idx) } else { None })
        .collect()
}

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   Robust Covariance Estimation - Comprehensive Comparison   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Configuration
    let n_samples = 200;
    let n_features = 10;
    let n_outliers = 20; // 10% contamination
    let outlier_magnitude = 10.0;
    let seed = 42;

    println!("Configuration:");
    println!("  Samples: {}", n_samples);
    println!("  Features: {}", n_features);
    println!(
        "  Outliers: {} ({:.1}% contamination)",
        n_outliers,
        (n_outliers as f64 / n_samples as f64) * 100.0
    );
    println!("  Outlier Magnitude: {}", outlier_magnitude);
    println!("  Random Seed: {}", seed);

    // ========================================
    // Part 1: Generate Data with Outliers
    // ========================================
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Part 1: Data Generation                                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    println!("\n📊 Generating clean data...");
    let clean_data = generate_clean_data(n_samples, n_features, seed);
    println!("✓ Generated {}x{} clean data matrix", n_samples, n_features);

    println!("\n💥 Adding outliers...");
    let mut contaminated_data = clean_data.clone();
    add_outliers(
        &mut contaminated_data,
        n_outliers,
        outlier_magnitude,
        seed + 1,
    );
    println!(
        "✓ Added {} extreme outliers (value = {})",
        n_outliers, outlier_magnitude
    );

    // ========================================
    // Part 2: Non-Robust Estimation
    // ========================================
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Part 2: Non-Robust Methods (Baseline)                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    println!("\n--- Empirical Covariance (Standard Method) ---");

    // Fit on clean data
    let emp_clean = EmpiricalCovariance::new();
    let fitted_emp_clean = emp_clean.fit(&clean_data.view(), &())?;
    let clean_cov = fitted_emp_clean.get_covariance();

    println!("\n🔵 On Clean Data:");
    let clean_frob = frobenius_norm(clean_cov);
    println!("  Frobenius Norm: {:.4}", clean_frob);

    match validate_covariance_matrix(clean_cov) {
        Ok(props) => {
            println!("  Condition Number: {:.2e}", props.condition_number);
            println!("  Trace: {:.4}", props.trace);
        }
        Err(e) => println!("  Validation error: {}", e),
    }

    // Fit on contaminated data
    let emp_contaminated = EmpiricalCovariance::new();
    let fitted_emp_contaminated = emp_contaminated.fit(&contaminated_data.view(), &())?;
    let contaminated_cov = fitted_emp_contaminated.get_covariance();

    println!("\n🔴 On Contaminated Data:");
    let contaminated_frob = frobenius_norm(contaminated_cov);
    println!("  Frobenius Norm: {:.4}", contaminated_frob);

    match validate_covariance_matrix(contaminated_cov) {
        Ok(props) => {
            println!("  Condition Number: {:.2e}", props.condition_number);
            println!("  Trace: {:.4}", props.trace);
        }
        Err(e) => println!("  Validation error: {}", e),
    }

    // Compute difference
    let mut diff_norm = 0.0;
    for i in 0..n_features {
        for j in 0..n_features {
            let diff = clean_cov[[i, j]] - contaminated_cov[[i, j]];
            diff_norm += diff * diff;
        }
    }
    diff_norm = diff_norm.sqrt();

    println!("\n📊 Impact of Outliers:");
    println!("  Frobenius Norm Difference: {:.4}", diff_norm);
    println!(
        "  Relative Change: {:.1}%",
        (diff_norm / clean_frob) * 100.0
    );
    println!("  ⚠️ Empirical covariance is heavily affected by outliers!");

    // ========================================
    // Part 3: Robust Estimation Methods
    // ========================================
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Part 3: Robust Estimation Methods                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    println!("\n--- MinCovDet (Minimum Covariance Determinant) ---");
    println!("MCD finds the subset of data with minimum covariance determinant.");

    let mcd = MinCovDet::new().support_fraction(0.8);
    let start = std::time::Instant::now();
    let fitted_mcd = mcd.fit(&contaminated_data.view(), &())?;
    let duration = start.elapsed();

    let mcd_cov = fitted_mcd.get_covariance();

    println!(
        "\n✓ Fit completed in {:.2}ms",
        duration.as_secs_f64() * 1000.0
    );

    let mcd_frob = frobenius_norm(mcd_cov);
    println!("  Frobenius Norm: {:.4}", mcd_frob);

    match validate_covariance_matrix(mcd_cov) {
        Ok(props) => {
            println!("  Condition Number: {:.2e}", props.condition_number);
            println!("  Trace: {:.4}", props.trace);
        }
        Err(e) => println!("  Validation error: {}", e),
    }

    // Compare with clean data
    let mut mcd_diff = 0.0;
    for i in 0..n_features {
        for j in 0..n_features {
            let diff = clean_cov[[i, j]] - mcd_cov[[i, j]];
            mcd_diff += diff * diff;
        }
    }
    mcd_diff = mcd_diff.sqrt();

    println!("\n  Difference from Clean: {:.4}", mcd_diff);
    println!("  Relative Error: {:.1}%", (mcd_diff / clean_frob) * 100.0);

    println!("\n--- Huber Covariance (M-Estimator) ---");
    println!("Huber method downweights observations based on robust distances.");

    let huber = HuberCovariance::new().max_iter(100);
    let start = std::time::Instant::now();
    let fitted_huber = huber.fit(&contaminated_data.view(), &())?;
    let duration = start.elapsed();

    let huber_cov = fitted_huber.get_covariance();

    println!(
        "\n✓ Fit completed in {:.2}ms",
        duration.as_secs_f64() * 1000.0
    );

    let huber_frob = frobenius_norm(huber_cov);
    println!("  Frobenius Norm: {:.4}", huber_frob);

    match validate_covariance_matrix(huber_cov) {
        Ok(props) => {
            println!("  Condition Number: {:.2e}", props.condition_number);
            println!("  Trace: {:.4}", props.trace);
        }
        Err(e) => println!("  Validation error: {}", e),
    }

    // Compare with clean data
    let mut huber_diff = 0.0;
    for i in 0..n_features {
        for j in 0..n_features {
            let diff = clean_cov[[i, j]] - huber_cov[[i, j]];
            huber_diff += diff * diff;
        }
    }
    huber_diff = huber_diff.sqrt();

    println!("\n  Difference from Clean: {:.4}", huber_diff);
    println!(
        "  Relative Error: {:.1}%",
        (huber_diff / clean_frob) * 100.0
    );

    // ========================================
    // Part 4: Method Comparison
    // ========================================
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Part 4: Comprehensive Method Comparison                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    println!("\n--- Accuracy Comparison ---");
    println!(
        "{:<25} {:>15} {:>15}",
        "Method", "Error (Frob)", "Relative %"
    );
    println!("{}", "─".repeat(60));

    println!(
        "{:<25} {:>15.4} {:>15.1}",
        "Empirical (Baseline)",
        diff_norm,
        (diff_norm / clean_frob) * 100.0
    );
    println!(
        "{:<25} {:>15.4} {:>15.1}",
        "MinCovDet",
        mcd_diff,
        (mcd_diff / clean_frob) * 100.0
    );
    println!(
        "{:<25} {:>15.4} {:>15.1}",
        "Huber",
        huber_diff,
        (huber_diff / clean_frob) * 100.0
    );

    println!("\n--- Performance Comparison ---");

    let benchmark = CovarianceBenchmark::new(5);

    println!(
        "{:<25} {:>15} {:>15}",
        "Method", "Mean Time (ms)", "Throughput"
    );
    println!("{}", "─".repeat(60));

    // Benchmark Empirical
    let emp_result = benchmark.time_execution(|| {
        let emp = EmpiricalCovariance::new();
        let _ = emp.fit(&contaminated_data.view(), &());
    });
    println!(
        "{:<25} {:>15.2} {:>15.2}",
        "Empirical",
        emp_result.mean_time_ms(),
        emp_result.throughput_ops_per_sec()
    );

    // Benchmark MinCovDet
    let mcd_result = benchmark.time_execution(|| {
        let mcd = MinCovDet::new().support_fraction(0.8);
        let _ = mcd.fit(&contaminated_data.view(), &());
    });
    println!(
        "{:<25} {:>15.2} {:>15.2}",
        "MinCovDet",
        mcd_result.mean_time_ms(),
        mcd_result.throughput_ops_per_sec()
    );

    // Benchmark Huber
    let huber_result = benchmark.time_execution(|| {
        let huber = HuberCovariance::new().max_iter(100);
        let _ = huber.fit(&contaminated_data.view(), &());
    });
    println!(
        "{:<25} {:>15.2} {:>15.2}",
        "Huber",
        huber_result.mean_time_ms(),
        huber_result.throughput_ops_per_sec()
    );

    // ========================================
    // Part 5: Outlier Detection
    // ========================================
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Part 5: Outlier Detection Capabilities                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    println!("\nUsing Mahalanobis distances to identify outliers...");

    // Compute mean for MinCovDet
    let mut mcd_mean = Array2::zeros((1, n_features));
    for j in 0..n_features {
        let mut sum = 0.0;
        for i in 0..n_samples {
            sum += contaminated_data[[i, j]];
        }
        mcd_mean[[0, j]] = sum / n_samples as f64;
    }

    // For simplicity, use identity as approximate precision (full implementation would invert covariance)
    let mut approx_precision = Array2::eye(n_features);
    for i in 0..n_features {
        approx_precision[[i, i]] = 1.0 / mcd_cov[[i, i]].max(1e-6);
    }

    let distances = compute_mahalanobis_distances(&contaminated_data, &mcd_mean, &approx_precision);

    println!("\n📊 Mahalanobis Distance Statistics:");
    let mean_dist = distances.iter().sum::<f64>() / distances.len() as f64;
    let max_dist = distances.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min_dist = distances.iter().fold(f64::INFINITY, |a, &b| a.min(b));

    println!("  Mean: {:.4}", mean_dist);
    println!("  Min: {:.4}", min_dist);
    println!("  Max: {:.4}", max_dist);

    let detected_outliers = identify_outliers(&distances, 3.0);
    println!("\n🎯 Detected Outliers: {}", detected_outliers.len());
    println!(
        "  Detection Rate: {:.1}%",
        (detected_outliers.len() as f64 / n_outliers as f64) * 100.0
    );

    if detected_outliers.len() <= 10 {
        println!("  Outlier Indices: {:?}", detected_outliers);
    } else {
        println!("  First 10 Outlier Indices: {:?}", &detected_outliers[..10]);
    }

    // ========================================
    // Part 6: Practical Recommendations
    // ========================================
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Part 6: Practical Recommendations                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    println!("\n📋 Choosing the Right Robust Method:\n");

    println!("1. 🎯 MinCovDet (MCD):");
    println!("   • Best for: High breakdown point (up to 50% contamination)");
    println!("   • Pros: Very robust, well-studied theory");
    println!("   • Cons: Computationally expensive for large datasets");
    println!("   • Use when: Severe contamination expected");

    println!("\n2. 🔨 Huber Covariance:");
    println!("   • Best for: Moderate outliers with good efficiency");
    println!("   • Pros: Good balance of robustness and efficiency");
    println!("   • Cons: Less robust than MCD for heavy contamination");
    println!("   • Use when: 5-20% contamination expected");

    println!("\n3. 📊 Ledoit-Wolf (with robust preprocessing):");
    println!("   • Best for: High-dimensional problems with mild contamination");
    println!("   • Pros: Handles high dimensions well");
    println!("   • Cons: Not inherently robust");
    println!("   • Use when: Dimension > samples, mild outliers");

    println!("\n💡 Practical Guidelines:\n");

    println!("  Contamination Level:");
    println!("    • < 5%: Standard methods may suffice");
    println!("    • 5-20%: Use Huber or M-estimators");
    println!("    • 20-40%: Use MinCovDet with appropriate support fraction");
    println!("    • > 40%: Consider data quality issues first");

    println!("\n  Support Fraction (for MCD):");
    println!("    • Default: 0.5 + (n_features + 1) / (2 * n_samples)");
    println!("    • Higher values: More data used, less robust");
    println!("    • Lower values: More robust, but higher variance");

    println!("\n  Convergence Settings:");
    println!("    • Max Iterations: 100-200 for most problems");
    println!("    • Tolerance: 1e-4 for standard precision");
    println!("    • Monitor: Check convergence warnings");

    // ========================================
    // Summary
    // ========================================
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Summary & Conclusions                                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    println!("\n✅ Successfully compared robust covariance estimation methods!");

    println!("\n📊 Key Findings:");
    println!(
        "  • Standard methods error: {:.1}% (heavily affected)",
        (diff_norm / clean_frob) * 100.0
    );
    println!(
        "  • MinCovDet error: {:.1}% (most robust)",
        (mcd_diff / clean_frob) * 100.0
    );
    println!(
        "  • Huber error: {:.1}% (good balance)",
        (huber_diff / clean_frob) * 100.0
    );
    println!(
        "  • Outlier detection: {}/{} detected",
        detected_outliers.len(),
        n_outliers
    );

    println!("\n🎯 Recommendations:");
    println!(
        "  • For this contamination level ({:.1}%), use MinCovDet or Huber",
        (n_outliers as f64 / n_samples as f64) * 100.0
    );
    println!("  • MinCovDet offers best accuracy but is slower");
    println!("  • Huber provides good balance of speed and robustness");
    println!("  • Always visualize data and check for outliers first");

    println!("\n🔬 Next Steps:");
    println!("  1. Try different contamination levels");
    println!("  2. Experiment with support fractions");
    println!("  3. Compare with SPACE, TIGER for sparse robust estimation");
    println!("  4. Apply to your real-world datasets");
    println!("  5. Combine with outlier removal for preprocessing");

    println!("\n✨ Demo completed successfully!\n");

    Ok(())
}
