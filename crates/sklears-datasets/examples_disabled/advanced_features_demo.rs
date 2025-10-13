//! Advanced Features Demo for sklears-datasets
//!
//! This example demonstrates the advanced features implemented in sklears-datasets:
//! 1. Memory-mapped dataset storage for large datasets
//! 2. Zero-copy dataset views for efficient data access
//! 3. Trait-based dataset framework for extensibility
//! 4. Enhanced arena allocation with memory management
//! 5. SIMD-optimized generation (when available)

use scirs2_autograd::ndarray::{Array, Array1, Array2};
use sklears_datasets::{
    create_default_registry,
    // SIMD support
    get_simd_info,
    // Legacy generators for comparison
    make_classification,
    make_regression,
    make_simd_classification,
    make_simd_regression,
    // Memory management
    ArenaAllocation,
    ArenaUsageStats,
    // Trait-based framework
    ClassificationGenerator,
    Dataset,
    DatasetArena,
    DatasetGenerator,
    // Zero-copy views
    DatasetView,
    DatasetViewMut,
    FilteredDatasetView,
    GeneratorConfig,
    GeneratorRegistry,
    InMemoryDataset,
    MmapDataset,
    MmapDatasetMut,
    RegressionGenerator,
    SimdConfig,
    ZeroCopyResult,
};
use std::env;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== sklears-datasets Advanced Features Demo ===\n");

    // 1. Demonstrate SIMD capabilities
    demo_simd_capabilities()?;

    // 2. Demonstrate memory-mapped storage
    demo_memory_mapped_storage()?;

    // 3. Demonstrate zero-copy views
    demo_zero_copy_views()?;

    // 4. Demonstrate trait-based framework
    demo_trait_framework()?;

    // 5. Demonstrate enhanced arena allocation
    demo_arena_allocation()?;

    // 6. Performance comparison
    demo_performance_comparison()?;

    println!("Demo completed successfully!");
    Ok(())
}

fn demo_simd_capabilities() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. SIMD Capabilities Demo");
    println!("========================");

    // Show SIMD information
    println!("SIMD Information:");
    println!("{}\n", get_simd_info());

    // Compare SIMD vs standard generation (if SIMD is available)
    let n_samples = 10000;
    let n_features = 20;

    // Standard generation
    let start = Instant::now();
    let (features_std, targets_std) = make_classification(
        n_samples,
        n_features,
        Some(3),  // n_classes
        None,     // n_informative
        None,     // n_redundant
        None,     // n_clusters_per_class
        Some(42), // random_state
    )?;
    let std_time = start.elapsed();

    println!(
        "Standard generation: {} samples x {} features in {:?}",
        n_samples, n_features, std_time
    );

    // SIMD generation (if available)
    #[cfg(feature = "simd")]
    {
        let config = SimdConfig {
            simd_threshold: 1000,
            ..Default::default()
        };

        let start = Instant::now();
        match make_simd_classification(
            n_samples,
            n_features,
            3,
            Some(n_features),
            Some(42),
            Some(config),
        ) {
            Ok((features_simd, targets_simd)) => {
                let simd_time = start.elapsed();
                println!(
                    "SIMD generation: {} samples x {} features in {:?}",
                    features_simd.nrows(),
                    features_simd.ncols(),
                    simd_time
                );
                println!(
                    "SIMD speedup: {:.2}x\n",
                    std_time.as_nanos() as f64 / simd_time.as_nanos() as f64
                );
            }
            Err(e) => println!("SIMD not available: {}\n", e),
        }
    }

    #[cfg(not(feature = "simd"))]
    {
        println!("SIMD feature not enabled. Enable with --features simd\n");
    }

    Ok(())
}

fn demo_memory_mapped_storage() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Memory-Mapped Storage Demo");
    println!("=============================");

    let temp_dir = env::temp_dir();
    let dataset_path = temp_dir.join("demo_dataset.mmap");

    // Generate a large dataset
    let (features, targets) = make_regression(5000, 10, Some(0.1), Some(42))?;

    println!(
        "Created dataset: {} samples x {} features",
        features.nrows(),
        features.ncols()
    );

    // Create memory-mapped dataset
    let start = Instant::now();
    let mmap_dataset = MmapDataset::create(&dataset_path, &features, Some(&targets))?;
    let creation_time = start.elapsed();

    println!("Memory-mapped creation time: {:?}", creation_time);
    println!(
        "File size: {} MB",
        mmap_dataset.file_size() as f64 / 1024.0 / 1024.0
    );

    // Demonstrate efficient access
    let start = Instant::now();
    let sample_100 = mmap_dataset.sample(100)?;
    let batch_50_100 = mmap_dataset.batch(50, 100)?;
    let access_time = start.elapsed();

    println!(
        "Sample 100 first 3 features: {:?}",
        &sample_100.as_slice().unwrap()[..3]
    );
    println!(
        "Batch access (50 samples): {} x {}",
        batch_50_100.nrows(),
        batch_50_100.ncols()
    );
    println!("Access time: {:?}", access_time);

    // Demonstrate batch iteration
    println!("Processing in batches of 1000:");
    let mut total_processed = 0;
    for (i, batch_result) in mmap_dataset.batches(1000).enumerate() {
        let batch = batch_result?;
        total_processed += batch.nrows();
        if i < 3 {
            println!("  Batch {}: {} samples", i, batch.nrows());
        }
    }
    println!("Total samples processed: {}\n", total_processed);

    // Clean up
    std::fs::remove_file(&dataset_path).ok();

    Ok(())
}

fn demo_zero_copy_views() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Zero-Copy Views Demo");
    println!("=======================");

    // Create a dataset
    let features = Array::from_shape_vec((1000, 5), (0..5000).map(|x| x as f64 / 100.0).collect())?;
    let targets = Array1::from_shape_vec(1000, (0..1000).map(|x| (x % 3) as f64).collect())?;

    // Create a zero-copy view
    let view = DatasetView::new(features.view(), Some(targets.view()));

    println!(
        "Original dataset: {} samples x {} features",
        view.n_samples(),
        view.n_features()
    );

    // Demonstrate filtered view (samples where first feature > 25)
    let filtered = view.filter(|sample| sample[0] > 25.0)?;
    println!(
        "Filtered view (first feature > 25): {} samples",
        filtered.n_samples()
    );

    // Demonstrate feature selection (select features 0, 2, 4)
    let selected_features = view.select_features(&[0, 2, 4])?;
    println!(
        "Selected features view: {} features",
        selected_features.n_features()
    );

    // Demonstrate strided view (every 10th sample)
    let strided = view.strided(10, 0)?;
    println!(
        "Strided view (every 10th sample): {} samples",
        strided.n_samples()
    );

    // Demonstrate window view
    let windows = view.windows(50, 25)?;
    println!(
        "Window view (size 50, step 25): {} windows",
        windows.n_windows()
    );

    // Show some sample data
    println!("\nSample data from views:");
    println!(
        "Original sample 100: {:?}",
        &view.sample(100)?.as_slice().unwrap()[..3]
    );

    if filtered.n_samples() > 10 {
        let filtered_sample = filtered.sample(10)?;
        println!(
            "Filtered sample 10: {:?}",
            &filtered_sample.as_slice().unwrap()[..3]
        );
    }

    // Demonstrate batch processing with zero-copy
    println!("\nBatch processing with zero-copy:");
    let mut batch_count = 0;
    for batch_result in view.batches(200) {
        let batch = batch_result?;
        batch_count += 1;
        if batch_count <= 3 {
            println!(
                "  Batch {}: {} x {}",
                batch_count,
                batch.nrows(),
                batch.ncols()
            );
        }
    }
    println!("Total batches processed: {}\n", batch_count);

    Ok(())
}

fn demo_trait_framework() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Trait-Based Framework Demo");
    println!("==============================");

    // Create a generator registry
    let registry = create_default_registry();

    println!("Available generators: {:?}", registry.list());

    // Generate classification dataset
    let mut config = GeneratorConfig::new(1000, 8);
    config.set_parameter("n_classes".to_string(), 4i64);
    config = config.with_random_state(42);

    let classification_dataset = registry.generate("classification", config)?;
    println!(
        "Classification dataset: {} samples x {} features",
        classification_dataset.n_samples(),
        classification_dataset.n_features()
    );
    println!("Has targets: {}", classification_dataset.has_targets());

    // Generate regression dataset
    let mut reg_config = GeneratorConfig::new(800, 6);
    reg_config.set_parameter("noise".to_string(), 0.05);
    reg_config = reg_config.with_random_state(42);

    let regression_dataset = registry.generate("regression", reg_config)?;
    println!(
        "Regression dataset: {} samples x {} features",
        regression_dataset.n_samples(),
        regression_dataset.n_features()
    );

    // Show metadata
    println!(
        "Classification metadata: {:?}",
        classification_dataset.metadata()
    );
    println!("Regression metadata: {:?}", regression_dataset.metadata());

    // Demonstrate direct generator usage
    let generator = ClassificationGenerator;
    println!("Generator name: {}", generator.name());
    println!("Generator description: {}", generator.description());

    let mut direct_config = GeneratorConfig::new(500, 4);
    direct_config.set_parameter("n_classes".to_string(), 2i64);

    // Validate configuration
    generator.validate_config(&direct_config)?;

    let direct_dataset = generator.generate(direct_config)?;
    println!(
        "Direct generation: {} samples x {} features\n",
        direct_dataset.n_samples(),
        direct_dataset.n_features()
    );

    Ok(())
}

fn demo_arena_allocation() -> Result<(), Box<dyn std::error::Error>> {
    println!("5. Enhanced Arena Allocation Demo");
    println!("=================================");

    // Create an arena with 10MB capacity
    let mut arena = DatasetArena::new(10 * 1024 * 1024);

    println!(
        "Arena capacity: {} MB",
        arena.usage().total_capacity / 1024 / 1024
    );

    // Allocate several datasets
    let datasets = vec![
        ("small_dataset", 1000 * 10 * 8),  // 1000x10 f64 matrix
        ("medium_dataset", 5000 * 20 * 8), // 5000x20 f64 matrix
        ("large_dataset", 10000 * 15 * 8), // 10000x15 f64 matrix
    ];

    let mut allocation_ids = Vec::new();

    for (name, size) in datasets {
        if let Some(allocation_id) = arena.allocate_named(size, Some(name.to_string())) {
            allocation_ids.push(allocation_id);
            println!("Allocated {}: {} bytes (ID: {})", name, size, allocation_id);
        } else {
            println!("Failed to allocate {}: {} bytes", name, size);
        }
    }

    // Show usage statistics
    let stats = arena.usage();
    println!("\nArena Usage Statistics:");
    println!("  Active size: {} MB", stats.active_size / 1024 / 1024);
    println!("  Free size: {} MB", stats.free_size / 1024 / 1024);
    println!("  Utilization: {:.2}%", stats.utilization * 100.0);
    println!("  Fragmentation: {:.2}%", stats.fragmentation * 100.0);
    println!("  Active allocations: {}", stats.active_allocations);

    // Show allocation details
    println!("\nActive Allocations:");
    for allocation in arena.active_allocations() {
        println!(
            "  ID {}: {} bytes at offset {} ({})",
            allocation.id,
            allocation.size,
            allocation.offset,
            allocation.name.as_ref().unwrap_or(&"unnamed".to_string())
        );
    }

    // Deallocate the medium dataset
    if allocation_ids.len() > 1 {
        let medium_id = allocation_ids[1];
        if arena.deallocate(medium_id) {
            println!("\nDeallocated allocation ID: {}", medium_id);
        }
    }

    // Show updated statistics
    let updated_stats = arena.usage();
    println!(
        "Updated utilization: {:.2}%",
        updated_stats.utilization * 100.0
    );
    println!("Free blocks: {}", updated_stats.free_blocks);

    // Demonstrate compaction
    arena.compact();
    let compacted_stats = arena.usage();
    println!(
        "After compaction - utilization: {:.2}%",
        compacted_stats.utilization * 100.0
    );

    // Try to create a dataset view from an allocation
    if let Some(&first_id) = allocation_ids.first() {
        match arena.dataset_view(first_id, 1000, 10) {
            Ok(view) => {
                println!("Created dataset view: {} x {}", view.nrows(), view.ncols());
            }
            Err(e) => {
                println!("Failed to create dataset view: {}", e);
            }
        }
    }

    println!();
    Ok(())
}

fn demo_performance_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("6. Performance Comparison");
    println!("=========================");

    let sizes = vec![(1000, 10), (5000, 20), (10000, 50)];

    for (n_samples, n_features) in sizes {
        println!("Dataset size: {} x {}", n_samples, n_features);

        // Standard generation
        let start = Instant::now();
        let (features, targets) = make_regression(n_samples, n_features, Some(0.1), Some(42))?;
        let standard_time = start.elapsed();

        // Trait-based generation
        let start = Instant::now();
        let generator = RegressionGenerator;
        let mut config = GeneratorConfig::new(n_samples, n_features);
        config.set_parameter("noise".to_string(), 0.1);
        config = config.with_random_state(42);
        let trait_dataset = generator.generate(config)?;
        let trait_time = start.elapsed();

        // Zero-copy view creation
        let start = Instant::now();
        let view = DatasetView::new(features.view(), Some(targets.view()));
        let filtered = view.filter(|sample| sample[0] > 0.0)?;
        let view_time = start.elapsed();

        println!("  Standard generation: {:?}", standard_time);
        println!("  Trait-based generation: {:?}", trait_time);
        println!("  Zero-copy view + filter: {:?}", view_time);
        println!(
            "  Filtered samples: {} / {}",
            filtered.n_samples(),
            view.n_samples()
        );
        println!();
    }

    Ok(())
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_functions() {
        // Test that demo functions don't panic
        // In a real test environment, you might want to capture output
        // and verify specific behaviors
        assert!(demo_simd_capabilities().is_ok());
        assert!(demo_trait_framework().is_ok());
        assert!(demo_arena_allocation().is_ok());
    }
}
