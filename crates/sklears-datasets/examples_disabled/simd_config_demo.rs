//! Example demonstrating SIMD optimization and configuration management
//!
//! This example shows how to:
//! 1. Use the configuration system to define datasets
//! 2. Enable SIMD optimization for performance
//! 3. Generate multiple datasets with different parameters
//! 4. Export results in multiple formats

use sklears_datasets::{
    generate_example_config, get_simd_info, make_classification, make_regression,
    make_simd_classification, make_simd_regression, ConfigLoader, SimdConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== sklears-datasets SIMD and Configuration Demo ===\n");

    // Display SIMD capabilities
    println!("SIMD Information:");
    println!("{}\n", get_simd_info());

    // Create SIMD configuration
    let simd_config = SimdConfig {
        use_simd: true,
        simd_threshold: 100, // Lower threshold for demo
        chunk_size: 64,
        force_simd_level: None, // Auto-detect
    };

    // Generate datasets using SIMD-optimized generators
    println!("=== Direct SIMD Dataset Generation ===");

    println!("Generating classification dataset with SIMD...");
    let start = std::time::Instant::now();
    let (features_cls, targets_cls) = match make_simd_classification(
        1000,     // n_samples
        10,       // n_features
        3,        // n_classes
        Some(8),  // n_informative
        Some(42), // random_state
        Some(simd_config.clone()),
    ) {
        Ok(result) => result,
        Err(_) => {
            println!("  SIMD not available, falling back to standard generation...");
            make_classification(1000, 10, 8, 0, 3, Some(42))?
        }
    };
    let duration_cls = start.elapsed();
    println!(
        "✓ Classification dataset: {} samples, {} features in {:?}",
        features_cls.nrows(),
        features_cls.ncols(),
        duration_cls
    );

    println!("Generating regression dataset with SIMD...");
    let start = std::time::Instant::now();
    let (features_reg, targets_reg) = match make_simd_regression(
        1000,     // n_samples
        15,       // n_features
        0.1,      // noise
        Some(42), // random_state
        Some(simd_config),
    ) {
        Ok(result) => result,
        Err(_) => {
            println!("  SIMD not available, falling back to standard generation...");
            make_regression(1000, 15, 1, 0.1, Some(42))?
        }
    };
    let duration_reg = start.elapsed();
    println!(
        "✓ Regression dataset: {} samples, {} features in {:?}",
        features_reg.nrows(),
        features_reg.ncols(),
        duration_reg
    );

    // Demonstrate configuration-based generation
    #[cfg(feature = "serde")]
    {
        println!("\n=== Configuration-Based Dataset Generation ===");

        // Generate example configuration
        let config = generate_example_config();
        println!(
            "Generated configuration with {} datasets:",
            config.datasets.len()
        );
        for (idx, dataset) in config.datasets.iter().enumerate() {
            match dataset {
                sklears_datasets::DatasetSpec::Classification(cls_config) => {
                    println!(
                        "  {}. Classification: '{}' - {} samples, {} features, {} classes",
                        idx + 1,
                        cls_config.name,
                        cls_config.n_samples,
                        cls_config.n_features,
                        cls_config.n_classes
                    );
                }
                sklears_datasets::DatasetSpec::Regression(reg_config) => {
                    println!(
                        "  {}. Regression: '{}' - {} samples, {} features",
                        idx + 1,
                        reg_config.name,
                        reg_config.n_samples,
                        reg_config.n_features
                    );
                }
                _ => {
                    println!("  {}. Other dataset type", idx + 1);
                }
            }
        }

        // Save configuration to YAML
        println!("\n=== Configuration Serialization ===");
        let yaml_config = ConfigLoader::to_yaml(&config)?;
        println!(
            "Configuration serialized to YAML ({} bytes)",
            yaml_config.len()
        );

        // Show a snippet of the YAML
        let lines: Vec<&str> = yaml_config.lines().take(15).collect();
        println!("YAML snippet:");
        for line in lines {
            println!("  {}", line);
        }
        if yaml_config.lines().count() > 15 {
            println!("  ... ({} more lines)", yaml_config.lines().count() - 15);
        }

        // Save configuration to JSON
        let json_config = ConfigLoader::to_json(&config)?;
        println!(
            "\nConfiguration serialized to JSON ({} bytes)",
            json_config.len()
        );

        // Test round-trip serialization
        let config_from_yaml = ConfigLoader::load_from_yaml(&yaml_config)?;
        let config_from_json = ConfigLoader::load_from_json(&json_config)?;

        println!("\n✓ Round-trip serialization test:");
        println!(
            "  YAML: {} datasets loaded",
            config_from_yaml.datasets.len()
        );
        println!(
            "  JSON: {} datasets loaded",
            config_from_json.datasets.len()
        );
    }

    #[cfg(not(feature = "serde"))]
    {
        println!("\n=== Configuration System (Disabled) ===");
        println!("Configuration management requires the 'serde' feature.");
        println!("Run with: cargo run --example simd_config_demo --features serde");
    }

    // Performance comparison (SIMD vs standard)
    println!("\n=== Performance Comparison ===");

    println!("Benchmarking dataset generation performance...");
    let sizes = vec![500, 1000, 2000];

    for size in sizes {
        println!("\nDataset size: {} samples", size);

        // SIMD-optimized generation
        let simd_config = SimdConfig {
            use_simd: true,
            simd_threshold: 100,
            chunk_size: 128,
            force_simd_level: None,
        };

        let start = std::time::Instant::now();
        let simd_duration =
            match make_simd_classification(size, 20, 5, Some(15), Some(42), Some(simd_config)) {
                Ok(_) => start.elapsed(),
                Err(_) => {
                    // SIMD not available, use standard generator
                    let _ = make_classification(size, 20, 15, 0, 5, Some(42))?;
                    start.elapsed()
                }
            };

        // Standard generation (by setting high threshold)
        let standard_config = SimdConfig {
            use_simd: true,
            simd_threshold: size + 1, // Forces fallback to standard
            chunk_size: 128,
            force_simd_level: None,
        };

        let start = std::time::Instant::now();
        let standard_duration = match make_simd_classification(
            size,
            20,
            5,
            Some(15),
            Some(42),
            Some(standard_config),
        ) {
            Ok(_) => start.elapsed(),
            Err(_) => {
                // SIMD not available, use standard generator
                let _ = make_classification(size, 20, 15, 0, 5, Some(42))?;
                start.elapsed()
            }
        };

        let speedup = standard_duration.as_secs_f64() / simd_duration.as_secs_f64();
        println!("  SIMD:     {:8.3}ms", simd_duration.as_secs_f64() * 1000.0);
        println!(
            "  Standard: {:8.3}ms",
            standard_duration.as_secs_f64() * 1000.0
        );
        println!("  Speedup:  {:8.2}x", speedup);
    }

    println!("\n=== Data Quality Validation ===");

    // Basic validation of generated data
    println!("Validating classification dataset:");
    println!("  Features shape: {:?}", features_cls.dim());
    println!("  Targets shape: {}", targets_cls.len());
    println!("  Unique classes: {:?}", {
        let mut classes: Vec<i32> = targets_cls.to_vec();
        classes.sort_unstable();
        classes.dedup();
        classes
    });

    // Check for reasonable feature ranges
    let feature_stats = features_cls.column(0);
    let min_val = feature_stats.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = feature_stats
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let mean_val = feature_stats.mean().unwrap_or(0.0);
    println!(
        "  Feature 0 stats: min={:.3}, max={:.3}, mean={:.3}",
        min_val, max_val, mean_val
    );

    println!("\nValidating regression dataset:");
    println!("  Features shape: {:?}", features_reg.dim());
    println!("  Targets shape: {}", targets_reg.len());

    let target_min = targets_reg.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let target_max = targets_reg.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let target_mean = targets_reg.mean().unwrap_or(0.0);
    println!(
        "  Target stats: min={:.3}, max={:.3}, mean={:.3}",
        target_min, target_max, target_mean
    );

    println!("\n✓ Demo completed successfully!");
    println!("\nNext steps:");
    println!("  - Try running with different SIMD configurations");
    println!("  - Experiment with custom configuration files");
    println!("  - Enable the 'serde' feature for full configuration support");
    println!("  - Use the 'visualization' feature to plot generated datasets");

    Ok(())
}
