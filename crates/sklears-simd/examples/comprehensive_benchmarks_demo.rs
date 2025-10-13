//! Comprehensive Benchmarking Suite Demo
//!
//! This example demonstrates the full capabilities of the comprehensive benchmarking
//! suite including performance testing, energy efficiency analysis, scaling benchmarks,
//! and automated reporting.

use sklears_simd::comprehensive_benchmarks::{
    BenchmarkConfig, ComprehensiveBenchmarkSuite, QuickBenchmark,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Comprehensive Benchmarking Suite Demo");
    println!("=========================================\n");

    // Demo 1: Quick CI Benchmarks
    println!("📊 Demo 1: Quick CI Benchmarks");
    println!("------------------------------");
    run_quick_ci_benchmarks()?;

    // Demo 2: Full Comprehensive Benchmarks
    println!("\n📊 Demo 2: Full Comprehensive Benchmarks");
    println!("----------------------------------------");
    run_comprehensive_benchmarks()?;

    // Demo 3: Custom Configuration Benchmarks
    println!("\n📊 Demo 3: Custom Configuration Benchmarks");
    println!("------------------------------------------");
    run_custom_benchmarks()?;

    println!("\n✨ All benchmark demos completed successfully!");
    println!("🎯 Key Features Demonstrated:");
    println!("   ✅ Performance benchmarking with detailed metrics");
    println!("   ✅ Energy efficiency analysis");
    println!("   ✅ Scaling analysis across different problem sizes");
    println!("   ✅ Regression detection and alerting");
    println!("   ✅ Automated report generation");
    println!("   ✅ CI/CD integration support");
    println!("   ✅ CSV export for external analysis");

    Ok(())
}

fn run_quick_ci_benchmarks() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Running quick benchmarks suitable for CI/CD...");

    let start = std::time::Instant::now();
    let results = QuickBenchmark::run_ci_benchmarks()?;
    let duration = start.elapsed();

    println!("  ⏱️  Execution time: {:.2?}", duration);
    println!("  📈 Performance results:");

    for result in &results.performance_results {
        let throughput = result.iterations as f64 / result.duration.as_secs_f64();
        println!(
            "    {}: {:.2?} ({:.0} ops/sec)",
            result.name, result.duration, throughput
        );
    }

    println!("  🎯 Summary:");
    println!("    Total tests: {}", results.summary.total_tests);
    println!("    Passed: {}", results.summary.passed_tests);
    println!("    Failed: {}", results.summary.failed_tests);
    println!(
        "    Performance score: {:.1}/100",
        results.summary.performance_score
    );
    println!("    Recommendation: {}", results.summary.recommendation);

    let ci_summary = QuickBenchmark::generate_ci_summary(&results);
    println!("  🏁 CI Summary: {}", ci_summary);

    Ok(())
}

fn run_comprehensive_benchmarks() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Running comprehensive benchmarks with all features...");

    let mut config = BenchmarkConfig::default();
    config.test_sizes = vec![128, 256, 512, 1024]; // Moderate sizes for demo
    config.iterations = 500; // Moderate iterations for demo
    config.warmup_iterations = 50;
    config.enable_detailed_reporting = true;

    let mut suite = ComprehensiveBenchmarkSuite::new(config);

    let start = std::time::Instant::now();
    let results = suite.run_comprehensive_benchmarks()?;
    let duration = start.elapsed();

    println!("  ⏱️  Execution time: {:.2?}", duration);

    // Show performance results
    println!("  📈 Performance Results:");
    for result in &results.performance_results {
        let throughput = result.iterations as f64 / result.duration.as_secs_f64();
        println!(
            "    {}: {:.2?} ({:.0} ops/sec, {}x{} SIMD)",
            result.name, result.duration, throughput, result.simd_width, result.architecture
        );
    }

    // Show energy efficiency results
    if !results.energy_results.is_empty() {
        println!("  ⚡ Energy Efficiency Results:");
        for result in &results.energy_results {
            println!(
                "    {}: {:.2}x energy efficiency, {:.2}x performance/watt",
                result.operation_name,
                result.energy_efficiency_ratio,
                result.performance_per_watt_ratio
            );
        }
    }

    // Show scaling results
    if !results.scaling_results.is_empty() {
        println!("  📊 Scaling Results:");
        for (operation, scaling_data) in &results.scaling_results {
            println!("    {}:", operation);
            for (size, result) in scaling_data {
                let throughput = *size as f64 / result.duration.as_secs_f64();
                println!(
                    "      Size {}: {:.2?} ({:.0} elements/sec)",
                    size, result.duration, throughput
                );
            }
        }
    }

    // Show regression alerts
    if !results.regression_alerts.is_empty() {
        println!("  🚨 Regression Alerts:");
        for alert in &results.regression_alerts {
            println!(
                "    {}: {:.1}% change - {}",
                alert.operation, alert.change_percent, alert.recommendation
            );
        }
    }

    println!("  🎯 Summary:");
    println!("    Total tests: {}", results.summary.total_tests);
    println!(
        "    Average speedup: {:.2}x",
        results.summary.average_speedup
    );
    println!("    Best speedup: {:.2}x", results.summary.best_speedup);
    println!(
        "    Energy efficiency: {:.2}x",
        results.summary.average_energy_efficiency
    );
    println!(
        "    Performance score: {:.1}/100",
        results.summary.performance_score
    );
    println!("    Recommendation: {}", results.summary.recommendation);

    // Generate and show sample report
    println!("  📋 Sample Report (first 500 chars):");
    let report = results.generate_report();
    let preview = if report.len() > 500 {
        format!("{}...", &report[..500])
    } else {
        report
    };
    println!("{}", preview);

    // Show CSV export capability
    println!("  📊 CSV Export Sample:");
    let csv = results.export_csv();
    let csv_lines: Vec<&str> = csv.lines().take(3).collect();
    for line in csv_lines {
        println!("    {}", line);
    }
    if csv.lines().count() > 3 {
        println!("    ... ({} more lines)", csv.lines().count() - 3);
    }

    Ok(())
}

fn run_custom_benchmarks() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Running benchmarks with custom configuration...");

    let mut config = BenchmarkConfig::default();
    config.test_sizes = vec![64, 256, 1024]; // Specific sizes
    config.iterations = 200; // Fewer iterations
    config.enable_energy_tests = true;
    config.enable_scaling_tests = true;
    config.enable_regression_tests = false; // No regression testing
    config.cpu_tdp = 95.0; // High-end CPU
    config.energy_budget = 15.0; // Higher energy budget

    let mut suite = ComprehensiveBenchmarkSuite::new(config);

    let start = std::time::Instant::now();
    let results = suite.run_comprehensive_benchmarks()?;
    let duration = start.elapsed();

    println!("  ⏱️  Execution time: {:.2?}", duration);
    println!("  📈 Custom Configuration Results:");

    // Focus on key metrics
    println!(
        "    Performance tests: {}",
        results.performance_results.len()
    );
    println!("    Energy tests: {}", results.energy_results.len());
    println!("    Scaling tests: {}", results.scaling_results.len());
    println!("    Regression alerts: {}", results.regression_alerts.len());

    // Show best performing operations
    if !results.performance_results.is_empty() {
        let mut sorted_results = results.performance_results.clone();
        sorted_results.sort_by(|a, b| a.duration.cmp(&b.duration));

        println!("  🏆 Top 3 Fastest Operations:");
        for (i, result) in sorted_results.iter().take(3).enumerate() {
            let throughput = result.iterations as f64 / result.duration.as_secs_f64();
            println!(
                "    {}. {}: {:.2?} ({:.0} ops/sec)",
                i + 1,
                result.name,
                result.duration,
                throughput
            );
        }
    }

    // Show energy efficiency leaders
    if !results.energy_results.is_empty() {
        let mut sorted_energy = results.energy_results.clone();
        sorted_energy.sort_by(|a, b| {
            b.energy_efficiency_ratio
                .partial_cmp(&a.energy_efficiency_ratio)
                .unwrap()
        });

        println!("  ⚡ Most Energy Efficient Operations:");
        for (i, result) in sorted_energy.iter().take(2).enumerate() {
            println!(
                "    {}. {}: {:.2}x energy efficiency",
                i + 1,
                result.operation_name,
                result.energy_efficiency_ratio
            );
        }
    }

    println!(
        "  🎯 Final Score: {:.1}/100 - {}",
        results.summary.performance_score, results.summary.recommendation
    );

    // Demonstrate pass/fail logic
    if results.passed() {
        println!("  ✅ Benchmarks PASSED - Ready for deployment!");
    } else {
        println!("  ❌ Benchmarks FAILED - Further optimization needed");
    }

    Ok(())
}

fn _demonstrate_scaling_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("  🔍 Scaling Analysis Demo:");

    let sizes = vec![64, 128, 256, 512, 1024, 2048];
    let mut timings = Vec::new();

    for size in &sizes {
        let start = std::time::Instant::now();

        // Simulate scaling benchmark
        let data: Vec<f32> = (0..*size).map(|i| i as f32).collect();
        let _result: f32 = data.iter().sum();

        let duration = start.elapsed();
        timings.push(duration);

        let throughput = *size as f64 / duration.as_secs_f64();
        println!(
            "    Size {}: {:.2?} ({:.0} elements/sec)",
            size, duration, throughput
        );
    }

    // Analyze scaling behavior
    println!("  📊 Scaling Analysis:");
    let first_throughput = sizes[0] as f64 / timings[0].as_secs_f64();
    let last_throughput = sizes[sizes.len() - 1] as f64 / timings[timings.len() - 1].as_secs_f64();
    let scaling_efficiency = last_throughput / first_throughput;

    println!("    Scaling efficiency: {:.2}x", scaling_efficiency);

    if scaling_efficiency > 0.8 {
        println!("    ✅ Excellent scaling - near linear performance");
    } else if scaling_efficiency > 0.6 {
        println!("    ⚠️  Good scaling - some overhead at larger sizes");
    } else {
        println!("    ❌ Poor scaling - significant overhead at larger sizes");
    }

    Ok(())
}
