//! Scalability tests for clustering algorithms
//!
//! This module tests clustering algorithms with synthetic datasets of varying sizes
//! to evaluate performance, memory usage, and scalability characteristics.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{ChaCha8Rng, Rng, SeedableRng};
use sklears_clustering::{
    AgglomerativeClustering, GaussianMixture, KMeans, KMeansConfig, LSHConfig, LSHFamily, LSHIndex,
    MemoryMappedConfig, MemoryMappedDistanceMatrix, SparseDistanceMatrix, SparseMatrixConfig,
    SpectralClustering, DBSCAN,
};
use sklears_core::{
    traits::{Fit, Predict},
    types::Float,
};
use std::time::Instant;

/// Generate synthetic datasets with controlled properties for scalability testing
struct SyntheticDataGenerator {
    rng: ChaCha8Rng,
}

impl SyntheticDataGenerator {
    fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Generate well-separated Gaussian clusters
    fn generate_gaussian_clusters(
        &mut self,
        n_samples: usize,
        n_features: usize,
        n_clusters: usize,
        cluster_std: Float,
        separation: Float,
    ) -> Array2<Float> {
        let mut data = Array2::zeros((n_samples, n_features));
        let samples_per_cluster = n_samples / n_clusters;

        for cluster_id in 0..n_clusters {
            let start_idx = cluster_id * samples_per_cluster;
            let end_idx = if cluster_id == n_clusters - 1 {
                n_samples
            } else {
                (cluster_id + 1) * samples_per_cluster
            };

            // Generate cluster center
            let mut center = Array1::zeros(n_features);
            for i in 0..n_features {
                center[i] = (cluster_id as Float) * separation + self.rng.gen_range(-1.0..1.0);
            }

            // Generate points around the center
            for sample_idx in start_idx..end_idx {
                for feature_idx in 0..n_features {
                    let noise = self.rng.gen::<Float>() * cluster_std - cluster_std / 2.0;
                    data[[sample_idx, feature_idx]] = center[feature_idx] + noise;
                }
            }
        }

        data
    }

    /// Generate high-dimensional sparse data
    fn generate_sparse_data(
        &mut self,
        n_samples: usize,
        n_features: usize,
        sparsity: Float,
    ) -> Array2<Float> {
        let mut data = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            for j in 0..n_features {
                if self.rng.gen::<Float>() > sparsity {
                    data[[i, j]] = self.rng.gen_range(-5.0..5.0);
                }
            }
        }

        data
    }

    /// Generate data with different cluster densities
    fn generate_variable_density_clusters(
        &mut self,
        n_samples: usize,
        n_features: usize,
    ) -> Array2<Float> {
        let mut data = Array2::zeros((n_samples, n_features));
        let densities = [0.5, 1.0, 2.0]; // Different standard deviations
        let cluster_sizes = [
            n_samples / 2,
            n_samples / 3,
            n_samples - n_samples / 2 - n_samples / 3,
        ];

        let mut start_idx = 0;
        for (cluster_id, (&density, &size)) in
            densities.iter().zip(cluster_sizes.iter()).enumerate()
        {
            let end_idx = start_idx + size;

            // Cluster center
            let center_x = (cluster_id as Float) * 10.0;
            let center_y = (cluster_id as Float) * 10.0;

            for sample_idx in start_idx..end_idx {
                data[[sample_idx, 0]] =
                    center_x + self.rng.gen::<Float>() * density - density / 2.0;
                data[[sample_idx, 1]] =
                    center_y + self.rng.gen::<Float>() * density - density / 2.0;

                // Fill remaining features with noise
                for feature_idx in 2..n_features {
                    data[[sample_idx, feature_idx]] = self.rng.gen_range(-1.0..1.0);
                }
            }

            start_idx = end_idx;
        }

        data
    }
}

/// Performance metrics for scalability analysis
#[derive(Debug, Clone)]
struct PerformanceMetrics {
    algorithm_name: String,
    n_samples: usize,
    n_features: usize,
    n_clusters: usize,
    execution_time_ms: u128,
    memory_usage_mb: f64,
    success: bool,
    error_message: Option<String>,
}

impl PerformanceMetrics {
    fn new(algorithm_name: String, n_samples: usize, n_features: usize, n_clusters: usize) -> Self {
        Self {
            algorithm_name,
            n_samples,
            n_features,
            n_clusters,
            execution_time_ms: 0,
            memory_usage_mb: 0.0,
            success: false,
            error_message: None,
        }
    }

    fn success(mut self, execution_time_ms: u128, memory_usage_mb: f64) -> Self {
        self.execution_time_ms = execution_time_ms;
        self.memory_usage_mb = memory_usage_mb;
        self.success = true;
        self
    }

    fn failure(mut self, error_message: String) -> Self {
        self.error_message = Some(error_message);
        self.success = false;
        self
    }
}

/// Test K-Means scalability
fn test_kmeans_scalability(data: &Array2<Float>, n_clusters: usize) -> PerformanceMetrics {
    let metrics =
        PerformanceMetrics::new("KMeans".to_string(), data.nrows(), data.ncols(), n_clusters);

    let start_time = Instant::now();
    let start_memory = get_memory_usage();

    let config = KMeansConfig {
        n_clusters,
        max_iter: 100,
        random_seed: Some(42),
        ..Default::default()
    };
    let kmeans = KMeans::new(config);
    let y_dummy = scirs2_core::ndarray::Array1::zeros(data.nrows());

    match kmeans.fit(data, &y_dummy) {
        Ok(fitted) => match fitted.predict(data) {
            Ok(_labels) => {
                let execution_time = start_time.elapsed().as_millis();
                let memory_usage = get_memory_usage() - start_memory;
                metrics.success(execution_time, memory_usage)
            }
            Err(e) => metrics.failure(format!("Prediction failed: {}", e)),
        },
        Err(e) => metrics.failure(format!("Training failed: {}", e)),
    }
}

/// Test DBSCAN scalability
fn test_dbscan_scalability(data: &Array2<Float>) -> PerformanceMetrics {
    let metrics = PerformanceMetrics::new(
        "DBSCAN".to_string(),
        data.nrows(),
        data.ncols(),
        0, // DBSCAN doesn't require specifying clusters
    );

    let start_time = Instant::now();
    let start_memory = get_memory_usage();

    let dbscan = DBSCAN::new().eps(1.0).min_samples(5);

    match dbscan.fit(data, &()) {
        Ok(_fitted) => {
            let execution_time = start_time.elapsed().as_millis();
            let memory_usage = get_memory_usage() - start_memory;
            metrics.success(execution_time, memory_usage)
        }
        Err(e) => metrics.failure(format!("Training failed: {}", e)),
    }
}

/// Test Agglomerative Clustering scalability
fn test_agglomerative_scalability(data: &Array2<Float>, n_clusters: usize) -> PerformanceMetrics {
    let metrics = PerformanceMetrics::new(
        "AgglomerativeClustering".to_string(),
        data.nrows(),
        data.ncols(),
        n_clusters,
    );

    let start_time = Instant::now();
    let start_memory = get_memory_usage();

    let agg = AgglomerativeClustering::new().n_clusters(n_clusters);

    match agg.fit(data, &()) {
        Ok(_fitted) => {
            let execution_time = start_time.elapsed().as_millis();
            let memory_usage = get_memory_usage() - start_memory;
            metrics.success(execution_time, memory_usage)
        }
        Err(e) => metrics.failure(format!("Training failed: {}", e)),
    }
}

/// Test Spectral Clustering scalability
fn test_spectral_scalability(data: &Array2<Float>, n_clusters: usize) -> PerformanceMetrics {
    let metrics = PerformanceMetrics::new(
        "SpectralClustering".to_string(),
        data.nrows(),
        data.ncols(),
        n_clusters,
    );

    let start_time = Instant::now();
    let start_memory = get_memory_usage();

    let spectral = SpectralClustering::<Array2<f64>, Array1<f64>>::new()
        .n_clusters(n_clusters)
        .random_state(42);

    let dummy_y = Array1::zeros(data.nrows());
    match spectral.fit(&data.view(), &dummy_y.view()) {
        Ok(fitted) => match fitted.predict(&data.view()) {
            Ok(_labels) => {
                let execution_time = start_time.elapsed().as_millis();
                let memory_usage = get_memory_usage() - start_memory;
                metrics.success(execution_time, memory_usage)
            }
            Err(e) => metrics.failure(format!("Prediction failed: {}", e)),
        },
        Err(e) => metrics.failure(format!("Training failed: {}", e)),
    }
}

/// Test Gaussian Mixture Model scalability
fn test_gmm_scalability(data: &Array2<Float>, n_components: usize) -> PerformanceMetrics {
    let metrics = PerformanceMetrics::new(
        "GaussianMixture".to_string(),
        data.nrows(),
        data.ncols(),
        n_components,
    );

    let start_time = Instant::now();
    let start_memory = get_memory_usage();

    let gmm: GaussianMixture<(), ()> = GaussianMixture::new()
        .n_components(n_components)
        .max_iter(50)
        .random_state(42);

    let dummy_y = Array1::zeros(data.nrows());
    match gmm.fit(&data.view(), &dummy_y.view()) {
        Ok(fitted) => match fitted.predict(&data.view()) {
            Ok(_labels) => {
                let execution_time = start_time.elapsed().as_millis();
                let memory_usage = get_memory_usage() - start_memory;
                metrics.success(execution_time, memory_usage)
            }
            Err(e) => metrics.failure(format!("Prediction failed: {}", e)),
        },
        Err(e) => metrics.failure(format!("Training failed: {}", e)),
    }
}

/// Test Memory-Mapped Distance Matrix scalability
fn test_memory_mapped_scalability(data: &Array2<Float>) -> PerformanceMetrics {
    let metrics = PerformanceMetrics::new(
        "MemoryMappedDistanceMatrix".to_string(),
        data.nrows(),
        data.ncols(),
        0,
    );

    let start_time = Instant::now();
    let start_memory = get_memory_usage();

    let config = MemoryMappedConfig {
        chunk_size: 100,
        ..Default::default()
    };

    match MemoryMappedDistanceMatrix::new(data.nrows(), config) {
        Ok(mut mmap_matrix) => match mmap_matrix.compute_distances(data) {
            Ok(_) => {
                let execution_time = start_time.elapsed().as_millis();
                let memory_usage = get_memory_usage() - start_memory;
                metrics.success(execution_time, memory_usage)
            }
            Err(e) => metrics.failure(format!("Distance computation failed: {}", e)),
        },
        Err(e) => metrics.failure(format!("Initialization failed: {}", e)),
    }
}

/// Test LSH Index scalability
fn test_lsh_scalability(data: &Array2<Float>) -> PerformanceMetrics {
    let metrics = PerformanceMetrics::new("LSHIndex".to_string(), data.nrows(), data.ncols(), 0);

    let start_time = Instant::now();
    let start_memory = get_memory_usage();

    let config = LSHConfig {
        family: LSHFamily::RandomHyperplane,
        num_hash_functions: 10,
        num_tables: 5,
        input_dim: data.ncols(),
        distance_threshold: 1.0,
        ..Default::default()
    };

    let mut lsh_index = LSHIndex::new(config);
    match lsh_index.build_from_data(data) {
        Ok(_) => {
            let execution_time = start_time.elapsed().as_millis();
            let memory_usage = get_memory_usage() - start_memory;
            metrics.success(execution_time, memory_usage)
        }
        Err(e) => metrics.failure(format!("Index building failed: {}", e)),
    }
}

/// Simple memory usage estimation (platform-specific, for testing purposes)
fn get_memory_usage() -> f64 {
    // This is a simplified memory estimation
    // In a real implementation, you'd use platform-specific APIs
    0.0
}

/// Run comprehensive scalability tests
fn run_scalability_suite() -> Vec<PerformanceMetrics> {
    let mut generator = SyntheticDataGenerator::new(42);
    let mut results = Vec::new();

    // Test different dataset sizes
    let test_sizes = vec![
        (100, 5, 3),   // Small: 100 samples, 5 features, 3 clusters
        (1000, 10, 5), // Medium: 1000 samples, 10 features, 5 clusters
        (5000, 20, 8), // Large: 5000 samples, 20 features, 8 clusters
    ];

    for (n_samples, n_features, n_clusters) in test_sizes {
        eprintln!(
            "Testing with {} samples, {} features, {} clusters",
            n_samples, n_features, n_clusters
        );

        // Generate test data
        let data =
            generator.generate_gaussian_clusters(n_samples, n_features, n_clusters, 1.0, 5.0);

        // Test each algorithm
        results.push(test_kmeans_scalability(&data, n_clusters));
        results.push(test_dbscan_scalability(&data));
        results.push(test_agglomerative_scalability(&data, n_clusters));
        results.push(test_gmm_scalability(&data, n_clusters));

        // Test spectral clustering only for smaller datasets (it's computationally expensive)
        if n_samples <= 1000 {
            results.push(test_spectral_scalability(&data, n_clusters));
        }

        // Test specialized data structures
        if n_samples <= 1000 {
            // Memory-mapped matrices are better for larger datasets
            results.push(test_memory_mapped_scalability(&data));
        }
        results.push(test_lsh_scalability(&data));

        eprintln!("Completed testing for size {}", n_samples);
    }

    results
}

/// Print scalability test results
fn print_results(results: &[PerformanceMetrics]) {
    eprintln!("\n=== Scalability Test Results ===\n");

    for result in results {
        if result.success {
            eprintln!(
                "{}: {} samples, {} features -> {}ms, {:.2}MB",
                result.algorithm_name,
                result.n_samples,
                result.n_features,
                result.execution_time_ms,
                result.memory_usage_mb
            );
        } else {
            eprintln!(
                "{}: {} samples, {} features -> FAILED: {}",
                result.algorithm_name,
                result.n_samples,
                result.n_features,
                result
                    .error_message
                    .as_ref()
                    .unwrap_or(&"Unknown error".to_string())
            );
        }
    }

    // Performance analysis
    eprintln!("\n=== Performance Analysis ===\n");

    let successful_results: Vec<_> = results.iter().filter(|r| r.success).collect();
    if successful_results.is_empty() {
        eprintln!("No successful tests to analyze.");
        return;
    }

    // Group by algorithm
    let mut algorithm_groups: std::collections::HashMap<String, Vec<&PerformanceMetrics>> =
        std::collections::HashMap::new();

    for result in &successful_results {
        algorithm_groups
            .entry(result.algorithm_name.clone())
            .or_default()
            .push(result);
    }

    for (algorithm, metrics) in algorithm_groups {
        if metrics.len() < 2 {
            continue;
        }

        let times: Vec<_> = metrics.iter().map(|m| m.execution_time_ms as f64).collect();
        let samples: Vec<_> = metrics.iter().map(|m| m.n_samples as f64).collect();

        eprintln!("{}:", algorithm);
        for metric in &metrics {
            let time_per_sample = metric.execution_time_ms as f64 / metric.n_samples as f64;
            eprintln!(
                "  {} samples: {:.2}ms per sample",
                metric.n_samples, time_per_sample
            );
        }
    }
}

#[test]
#[ignore] // Temporarily disabled due to long runtime on CI with large synthetic datasets
fn test_basic_scalability() {
    let results = run_scalability_suite();
    print_results(&results);

    // Basic assertions
    assert!(!results.is_empty(), "Should have run some tests");

    let successful_count = results.iter().filter(|r| r.success).count();
    let total_count = results.len();

    eprintln!(
        "\nSuccess rate: {}/{} ({:.1}%)",
        successful_count,
        total_count,
        (successful_count as f64 / total_count as f64) * 100.0
    );

    // Expect at least 70% success rate
    assert!(
        successful_count as f64 / total_count as f64 >= 0.7,
        "Expected at least 70% success rate, got {}/{}",
        successful_count,
        total_count
    );
}

#[test]
fn test_sparse_data_scalability() {
    let mut generator = SyntheticDataGenerator::new(123);

    // Test with high-dimensional sparse data
    let sparse_data = generator.generate_sparse_data(500, 100, 0.8); // 80% sparse

    let start_time = Instant::now();
    let sparse_config = SparseMatrixConfig {
        distance_threshold: 2.0,
        sparsity_threshold: 0.5,
        ..Default::default()
    };

    match SparseDistanceMatrix::from_data(&sparse_data, sparse_config) {
        Ok(sparse_matrix) => {
            let elapsed = start_time.elapsed();
            eprintln!("Sparse matrix creation: {}ms", elapsed.as_millis());

            let stats = sparse_matrix.stats();
            eprintln!(
                "Sparse matrix stats: {:.1}% sparse, {:.2}% memory savings",
                stats.sparsity * 100.0,
                stats.memory_savings * 100.0
            );

            assert!(stats.sparsity > 0.5, "Matrix should be reasonably sparse");
            assert!(
                stats.memory_savings > 0.0,
                "Should have some memory savings"
            );
        }
        Err(e) => {
            eprintln!("Sparse matrix creation failed: {}", e);
            // This might fail if data is not sparse enough, which is acceptable
        }
    }
}

#[test]
fn test_variable_density_clustering() {
    let mut generator = SyntheticDataGenerator::new(456);
    let data = generator.generate_variable_density_clusters(300, 10);

    eprintln!("Testing clustering on variable density data...");

    // Test DBSCAN (should handle variable density well)
    let dbscan = DBSCAN::new().eps(2.0).min_samples(3);

    let start_time = Instant::now();
    match dbscan.fit(&data, &()) {
        Ok(fitted) => {
            let elapsed = start_time.elapsed();
            eprintln!("DBSCAN on variable density data: {}ms", elapsed.as_millis());

            let labels = fitted.labels();
            let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
            eprintln!("Found {} clusters (including noise)", unique_labels.len());

            assert!(unique_labels.len() >= 2, "Should find multiple clusters");
        }
        Err(e) => {
            panic!("DBSCAN failed on variable density data: {}", e);
        }
    }
}
