//! Demonstration of the sklearn benchmarking framework
//!
//! This example shows how to use the comprehensive benchmarking framework
//! to compare sklears dummy estimators against reference implementations.

use sklears_dummy::{
    BenchmarkConfig, DatasetConfig, DatasetProperties, DatasetSize, DatasetType,
    SklearnBenchmarkFramework,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Sklears Dummy Estimators - Sklearn Benchmark Demo");
    println!("=====================================================\n");

    // Create a custom benchmark configuration
    let config = BenchmarkConfig {
        tolerance: 1e-12,
        n_runs: 3, // Reduced for demo speed
        random_state: Some(42),
        include_performance: true,
        include_memory: false,
        test_reproducibility: true,
        datasets: vec![
            // Small balanced classification dataset
            DatasetConfig {
                name: "balanced_classification".to_string(),
                data_type: DatasetType::Classification { n_classes: 3 },
                size: DatasetSize {
                    n_samples: 200,
                    n_features: 5,
                },
                properties: DatasetProperties {
                    noise_level: 0.1,
                    correlation: 0.0,
                    outlier_fraction: 0.0,
                    random_state: Some(42),
                },
            },
            // Imbalanced classification dataset
            DatasetConfig {
                name: "imbalanced_classification".to_string(),
                data_type: DatasetType::Imbalanced {
                    majority_ratio: 0.8,
                },
                size: DatasetSize {
                    n_samples: 150,
                    n_features: 4,
                },
                properties: DatasetProperties {
                    noise_level: 0.05,
                    correlation: 0.1,
                    outlier_fraction: 0.02,
                    random_state: Some(123),
                },
            },
            // Regression dataset
            DatasetConfig {
                name: "basic_regression".to_string(),
                data_type: DatasetType::Regression,
                size: DatasetSize {
                    n_samples: 100,
                    n_features: 3,
                },
                properties: DatasetProperties {
                    noise_level: 0.2,
                    correlation: 0.0,
                    outlier_fraction: 0.05,
                    random_state: Some(456),
                },
            },
        ],
    };

    // Create benchmark framework with custom configuration
    let framework = SklearnBenchmarkFramework::with_config(config);

    println!("📊 Running classifier benchmarks...");
    let classifier_results = framework.benchmark_dummy_classifier()?;

    println!(
        "   ✅ Completed {} classifier benchmark(s)",
        classifier_results.len()
    );

    println!("\n📈 Running regressor benchmarks...");
    let regressor_results = framework.benchmark_dummy_regressor()?;

    println!(
        "   ✅ Completed {} regressor benchmark(s)",
        regressor_results.len()
    );

    // Combine all results
    let mut all_results = classifier_results;
    all_results.extend(regressor_results);

    println!("\n📋 Generating comprehensive report...");
    let report = framework.generate_report(&all_results);

    println!("\n{}", report);

    // Show some specific performance metrics
    println!("🎯 Performance Summary:");
    println!("====================");

    let total_benchmarks = all_results.len();
    let within_tolerance = all_results
        .iter()
        .filter(|r| r.accuracy_comparison.within_tolerance)
        .count();
    let tolerance_rate = within_tolerance as f64 / total_benchmarks as f64 * 100.0;

    println!("📈 Total benchmarks run: {}", total_benchmarks);
    println!(
        "✅ Within tolerance: {}/{} ({:.1}%)",
        within_tolerance, total_benchmarks, tolerance_rate
    );

    // Show average performance metrics
    let avg_fit_time: f64 = all_results
        .iter()
        .map(|r| r.performance_metrics.fit_time_sklears.as_secs_f64())
        .sum::<f64>()
        / total_benchmarks as f64;

    let avg_predict_time: f64 = all_results
        .iter()
        .map(|r| r.performance_metrics.predict_time_sklears.as_secs_f64())
        .sum::<f64>()
        / total_benchmarks as f64;

    println!("⏱️  Average fit time: {:.4} ms", avg_fit_time * 1000.0);
    println!(
        "⚡ Average predict time: {:.4} ms",
        avg_predict_time * 1000.0
    );

    // Show accuracy distribution
    let avg_accuracy = all_results
        .iter()
        .map(|r| r.accuracy_comparison.sklears_score)
        .sum::<f64>()
        / total_benchmarks as f64;

    println!("🎯 Average accuracy/R²: {:.4}", avg_accuracy);

    // Show numerical accuracy metrics
    let avg_correlation = all_results
        .iter()
        .map(|r| r.numerical_accuracy.correlation)
        .sum::<f64>()
        / total_benchmarks as f64;

    let max_error = all_results
        .iter()
        .map(|r| r.numerical_accuracy.max_absolute_error)
        .fold(0.0f64, f64::max);

    println!("🔍 Average correlation: {:.6}", avg_correlation);
    println!("⚠️  Maximum absolute error: {:.2e}", max_error);

    // Highlight any problematic results
    let problematic_results: Vec<_> = all_results
        .iter()
        .filter(|r| !r.accuracy_comparison.within_tolerance)
        .collect();

    if !problematic_results.is_empty() {
        println!("\n⚠️  Results requiring attention:");
        for result in problematic_results {
            println!(
                "   • {} on {}: difference = {:.2e}",
                result.strategy,
                result.dataset_info.name,
                result.accuracy_comparison.absolute_difference
            );
        }
    } else {
        println!("\n🎉 All benchmarks passed within tolerance!");
    }

    println!("\n✨ Benchmark demo completed successfully!");

    Ok(())
}
