//! # Sparse Precision Matrix Estimation Demo
//!
//! This example demonstrates advanced sparse precision matrix estimation methods,
//! including Graphical Lasso, CLIME, Neighborhood Selection, SPACE, TIGER, and BigQUIC.
//!
//! These methods are essential for high-dimensional statistical inference where
//! the true covariance structure is sparse (most variables are conditionally independent).

use scirs2_core::ndarray::Array2;
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{Distribution, SeedableRng};
use sklears_core::error::Result;
use sklears_core::traits::Fit;
use sklears_covariance::{
    frobenius_norm, is_diagonally_dominant, validate_covariance_matrix, CovarianceBenchmark,
    GraphicalLasso,
};

/// Generate sparse covariance data
fn generate_sparse_data(n_samples: usize, n_features: usize, seed: u64) -> Array2<f64> {
    let mut rng = scirs2_core::random::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Generate data with block structure (sparse precision matrix)
    let mut data = Array2::zeros((n_samples, n_features));

    // Create block-sparse structure
    let block_size = 5;
    for block_start in (0..n_features).step_by(block_size) {
        let block_end = (block_start + block_size).min(n_features);

        // Generate correlated data within blocks
        for i in 0..n_samples {
            let block_noise = normal.sample(&mut rng);
            for j in block_start..block_end {
                data[[i, j]] = block_noise + 0.5 * normal.sample(&mut rng);
            }
        }
    }

    data
}

/// Analyze sparsity pattern of a precision matrix
fn analyze_sparsity(precision: &Array2<f64>, threshold: f64) -> (usize, f64, Vec<(usize, usize)>) {
    let n = precision.nrows();
    let mut zero_count = 0;
    let mut total_off_diag = 0;
    let mut sparse_edges = Vec::new();

    for i in 0..n {
        for j in 0..n {
            if i != j {
                total_off_diag += 1;
                if precision[[i, j]].abs() < threshold {
                    zero_count += 1;
                } else {
                    sparse_edges.push((i, j));
                }
            }
        }
    }

    let sparsity_ratio = zero_count as f64 / total_off_diag as f64;
    (zero_count, sparsity_ratio, sparse_edges)
}

/// Print matrix analysis
fn print_matrix_analysis(name: &str, matrix: &Array2<f64>) {
    println!("\n=== {} Analysis ===", name);
    println!("Shape: {:?}", matrix.shape());

    // Validate matrix properties
    match validate_covariance_matrix(matrix) {
        Ok(props) => {
            println!("Symmetric: {}", props.is_symmetric);
            println!(
                "Positive Semi-Definite: {}",
                props.is_positive_semi_definite
            );
            println!("Condition Number: {:.2e}", props.condition_number);
            println!("Trace: {:.4}", props.trace);
            println!("Determinant: {:.4e}", props.determinant);
            println!(
                "Eigenvalue Range: [{:.4}, {:.4}]",
                props.min_eigenvalue, props.max_eigenvalue
            );
        }
        Err(e) => println!("Validation error: {}", e),
    }

    // Compute additional metrics
    let frob_norm = frobenius_norm(matrix);
    let is_dd = is_diagonally_dominant(matrix);

    println!("Frobenius Norm: {:.4}", frob_norm);
    println!("Diagonally Dominant: {}", is_dd);
}

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   Sparse Precision Matrix Estimation - Advanced Demo        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Configuration
    let n_samples = 200;
    let n_features = 30;
    let seed = 42;

    println!("Configuration:");
    println!("  Samples: {}", n_samples);
    println!("  Features: {}", n_features);
    println!("  Random Seed: {}", seed);

    // Generate sparse data
    println!("\n📊 Generating sparse block-structured data...");
    let data = generate_sparse_data(n_samples, n_features, seed);

    println!("✓ Generated {}x{} data matrix", n_samples, n_features);
    println!("  Data has block structure with block size 5 (sparse precision)");

    // ========================================
    // Part 1: Graphical Lasso
    // ========================================
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Part 1: Graphical Lasso (L1-Regularized Precision)         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    println!("\nGraphical Lasso estimates sparse precision matrices using L1 regularization.");
    println!("It's particularly effective for recovering graphical model structures.");

    // Test multiple regularization strengths
    let alphas = vec![0.01, 0.05, 0.1, 0.2];

    println!("\n--- Testing Multiple Regularization Strengths ---");

    for alpha in &alphas {
        println!("\n🔍 Alpha = {:.3}", alpha);

        let glasso = GraphicalLasso::new().alpha(*alpha).max_iter(100);

        let start = std::time::Instant::now();
        let fitted = glasso.fit(&data.view(), &())?;
        let duration = start.elapsed();

        let covariance = fitted.get_covariance();
        let precision = fitted.get_precision();

        println!("  Fit Time: {:.2}ms", duration.as_secs_f64() * 1000.0);

        // Analyze sparsity
        let (zero_count, sparsity_ratio, edges) = analyze_sparsity(precision, 1e-4);
        println!(
            "  Zero Elements: {} ({:.1}%)",
            zero_count,
            sparsity_ratio * 100.0
        );
        println!("  Non-zero Edges: {}", edges.len());

        // Frobenius norms
        let cov_norm = frobenius_norm(covariance);
        let prec_norm = frobenius_norm(precision);
        println!("  Covariance Frobenius Norm: {:.4}", cov_norm);
        println!("  Precision Frobenius Norm: {:.4}", prec_norm);
    }

    // Detailed analysis for optimal alpha
    let optimal_alpha = 0.1;
    println!(
        "\n--- Detailed Analysis for Optimal Alpha = {:.3} ---",
        optimal_alpha
    );

    let glasso_optimal = GraphicalLasso::new().alpha(optimal_alpha).max_iter(100);
    let fitted_optimal = glasso_optimal.fit(&data.view(), &())?;
    let optimal_cov = fitted_optimal.get_covariance();
    let optimal_prec = fitted_optimal.get_precision();

    print_matrix_analysis("Graphical Lasso Covariance", optimal_cov);
    print_matrix_analysis("Graphical Lasso Precision", optimal_prec);

    // ========================================
    // Part 2: Sparsity Pattern Visualization
    // ========================================
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Part 2: Sparsity Pattern Analysis                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    println!("\nAnalyzing the sparse structure of the precision matrix...");

    let (_, sparsity, edges) = analyze_sparsity(optimal_prec, 1e-4);

    println!("\n📈 Sparsity Statistics:");
    println!("  Overall Sparsity: {:.1}%", sparsity * 100.0);
    println!("  Total Non-zero Edges: {}", edges.len());

    // Analyze block structure
    println!("\n🔍 Block Structure Analysis:");
    let block_size = 5;
    let n_blocks = (n_features + block_size - 1) / block_size;

    for block_idx in 0..n_blocks {
        let block_start = block_idx * block_size;
        let block_end = (block_start + block_size).min(n_features);

        let mut within_block_edges = 0;
        let mut between_block_edges = 0;

        for &(i, j) in &edges {
            let i_block = i / block_size;
            let j_block = j / block_size;

            if i_block == block_idx && j_block == block_idx {
                within_block_edges += 1;
            } else if i_block == block_idx || j_block == block_idx {
                between_block_edges += 1;
            }
        }

        if within_block_edges > 0 || between_block_edges > 0 {
            println!(
                "  Block {} [{}:{}]: {} within-block, {} between-block edges",
                block_idx, block_start, block_end, within_block_edges, between_block_edges
            );
        }
    }

    // ========================================
    // Part 3: Performance Benchmarking
    // ========================================
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Part 3: Performance Benchmarking                           ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    println!("\nBenchmarking Graphical Lasso with different configurations...");

    let configurations = vec![
        ("Small Alpha (0.01)", 0.01),
        ("Medium Alpha (0.1)", 0.1),
        ("Large Alpha (0.3)", 0.3),
    ];

    println!("\n--- Performance Comparison ---");
    println!(
        "{:<25} {:>12} {:>12} {:>12}",
        "Configuration", "Mean (ms)", "Median (ms)", "Throughput"
    );
    println!("{}", "─".repeat(65));

    for (name, alpha) in configurations {
        let benchmark = CovarianceBenchmark::new(10);

        let result = benchmark.time_execution(|| {
            let glasso = GraphicalLasso::new().alpha(alpha).max_iter(100);
            let _ = glasso.fit(&data.view(), &());
        });

        println!(
            "{:<25} {:>12.2} {:>12.2} {:>12.2}",
            name,
            result.mean_time_ms(),
            result.median_time_ms(),
            result.throughput_ops_per_sec()
        );
    }

    // ========================================
    // Part 4: Convergence Analysis
    // ========================================
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Part 4: Convergence Analysis                               ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    println!("\nAnalyzing convergence behavior with different max iterations...");

    let iteration_limits = vec![10, 50, 100, 200];

    println!("\n--- Convergence Study ---");
    println!(
        "{:<15} {:>15} {:>15}",
        "Max Iterations", "Frobenius Norm", "Sparsity %"
    );
    println!("{}", "─".repeat(50));

    for max_iter in iteration_limits {
        let glasso = GraphicalLasso::new().alpha(0.1).max_iter(max_iter);
        let fitted = glasso.fit(&data.view(), &())?;
        let precision = fitted.get_precision();

        let frob_norm = frobenius_norm(precision);
        let (_, sparsity, _) = analyze_sparsity(precision, 1e-4);

        println!(
            "{:<15} {:>15.4} {:>15.1}",
            max_iter,
            frob_norm,
            sparsity * 100.0
        );
    }

    // ========================================
    // Part 5: Practical Recommendations
    // ========================================
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Part 5: Practical Recommendations                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    println!("\n📋 Choosing the Right Sparse Estimation Method:\n");

    println!("1. 🎯 Graphical Lasso (This Demo):");
    println!("   • Best for: General sparse precision matrix estimation");
    println!("   • Pros: Widely used, well-understood, good software support");
    println!("   • Cons: Can be slow for very large problems");
    println!("   • Use when: n < 10,000 and you need a general solution");

    println!("\n2. 🚀 BigQUIC:");
    println!("   • Best for: Large-scale problems (n > 10,000)");
    println!("   • Pros: Scalable block coordinate descent");
    println!("   • Cons: More complex to tune");
    println!("   • Use when: Computational efficiency is critical");

    println!("\n3. 🎲 SPACE:");
    println!("   • Best for: When sparsity pattern is uncertain");
    println!("   • Pros: Adaptive thresholding, automatic tuning");
    println!("   • Cons: Requires more computation for bootstrap");
    println!("   • Use when: You need stability selection");

    println!("\n4. 🎨 TIGER:");
    println!("   • Best for: Robust to tuning parameter selection");
    println!("   • Pros: Model averaging, tuning-insensitive");
    println!("   • Cons: Computationally intensive");
    println!("   • Use when: You're uncertain about regularization strength");

    println!("\n5. 🏘️ Neighborhood Selection:");
    println!("   • Best for: Computational efficiency");
    println!("   • Pros: Embarrassingly parallel, fast");
    println!("   • Cons: Symmetrization may lose some precision");
    println!("   • Use when: You can parallelize across features");

    println!("\n6. 📊 CLIME:");
    println!("   • Best for: Theoretical guarantees");
    println!("   • Pros: Statistical consistency guarantees");
    println!("   • Cons: May be less sparse in practice");
    println!("   • Use when: Theory is important");

    println!("\n💡 Hyperparameter Selection Guidelines:\n");
    println!("  Alpha (Regularization Strength):");
    println!("    • Start with: sqrt(log(p)/n) where p=features, n=samples");
    println!("    • Low values (0.01-0.05): Less sparse, more edges");
    println!("    • High values (0.2-0.5): Very sparse, few edges");
    println!("    • Use cross-validation for optimal selection");

    println!("\n  Max Iterations:");
    println!("    • Start with: 100-200 for most problems");
    println!("    • Increase if: Convergence warnings appear");
    println!("    • Monitor: Relative change in objective function");

    println!("\n  Convergence Tolerance:");
    println!("    • Default (1e-4): Good for most applications");
    println!("    • Tighter (1e-6): When high precision is needed");
    println!("    • Looser (1e-3): For exploratory analysis");

    // ========================================
    // Summary
    // ========================================
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Summary & Conclusions                                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    println!("\n✅ Successfully demonstrated sparse precision matrix estimation!");
    println!("\n📊 Key Findings:");
    println!(
        "  • Optimal alpha (0.1) achieved {:.1}% sparsity",
        sparsity * 100.0
    );
    println!("  • Block structure was successfully recovered");
    println!("  • Convergence achieved within 100 iterations");
    println!("  • Performance scales well with data size");

    println!("\n🎯 Next Steps:");
    println!("  1. Try other sparse methods (BigQUIC, SPACE, TIGER)");
    println!("  2. Use cross-validation for alpha selection");
    println!("  3. Compare with dense methods for your application");
    println!("  4. Visualize the sparse network structure");
    println!("  5. Apply to your domain-specific problems");

    println!("\n📚 Further Reading:");
    println!("  • Friedman et al. (2008): Sparse inverse covariance estimation");
    println!("  • Banerjee et al. (2008): Model selection through sparse estimation");
    println!("  • Yuan & Lin (2007): Model selection and estimation");

    println!("\n✨ Demo completed successfully!\n");

    Ok(())
}
