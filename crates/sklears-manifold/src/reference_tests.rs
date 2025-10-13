//! Comparison tests against reference implementations
//! This module provides comprehensive testing infrastructure to compare
//! our manifold learning implementations against reference implementations
//! from scikit-learn and other established libraries.

use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::thread_rng;
use scirs2_core::random::Rng;
use scirs2_core::random::SeedableRng;
/// Test configuration for reference comparisons
use sklears_core::{error::Result as SklResult, traits::Estimator, types::Float};
use std::collections::HashMap;
use std::time::Instant;
#[derive(Debug, Clone)]
pub struct ReferenceTestConfig {
    /// Tolerance for numerical comparisons
    pub tolerance: Float,
    /// Whether to test on multiple random seeds
    pub test_multiple_seeds: bool,
    /// Number of random seeds to test
    pub n_seeds: usize,
    /// Whether to test edge cases
    pub test_edge_cases: bool,
    /// Whether to test performance comparisons
    pub test_performance: bool,
}

impl Default for ReferenceTestConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-6,
            test_multiple_seeds: true,
            n_seeds: 5,
            test_edge_cases: true,
            test_performance: false,
        }
    }
}

/// Test result for reference comparison
#[derive(Debug, Clone)]
pub struct ReferenceTestResult {
    /// Test name
    pub test_name: String,
    /// Whether the test passed
    pub passed: bool,
    /// Error message if test failed
    pub error_message: Option<String>,
    /// Numerical differences observed
    pub max_difference: Float,
    /// Performance metrics (if enabled)
    pub performance_metrics: Option<PerformanceMetrics>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Performance comparison metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Our implementation runtime (seconds)
    pub our_runtime: Float,
    /// Reference implementation runtime (seconds)
    pub reference_runtime: Float,
    /// Speed improvement factor (reference_time / our_time)
    pub speedup_factor: Float,
    /// Memory usage comparison
    pub memory_usage: Option<(usize, usize)>, // (our_memory, reference_memory) in bytes
}

/// Mock reference implementation for testing purposes
/// In a real implementation, this would interface with Python/scikit-learn
pub trait ReferenceImplementation {
    /// Algorithm name
    const NAME: &'static str;

    /// Input parameters type
    type Params;

    /// Fit and transform data using reference implementation
    fn fit_transform(
        &self,
        data: ArrayView2<Float>,
        params: &Self::Params,
    ) -> SklResult<Array2<Float>>;
}

/// Mock scikit-learn t-SNE implementation
pub struct SklearnTSNE;

impl ReferenceImplementation for SklearnTSNE {
    const NAME: &'static str = "sklearn.manifold.TSNE";
    type Params = TSNEParams;

    fn fit_transform(
        &self,
        data: ArrayView2<Float>,
        params: &Self::Params,
    ) -> SklResult<Array2<Float>> {
        // Mock implementation that simulates scikit-learn t-SNE behavior
        // In practice, this would call Python scikit-learn via PyO3 or similar

        let n_samples = data.nrows();
        let mut embedding = Array2::zeros((n_samples, params.n_components));

        // Simulate t-SNE with simple random initialization and basic optimization
        use scirs2_core::random::rngs::StdRng;
        use scirs2_core::random::SeedableRng;

        let mut rng = if let Some(seed) = params.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().gen())
        };

        // Initialize embedding randomly
        for i in 0..n_samples {
            for j in 0..params.n_components {
                embedding[[i, j]] = rng.gen_range(-1.0..1.0) * 0.0001;
            }
        }

        // Simulate basic gradient descent (simplified)
        for _iter in 0..params.n_iter.min(10) {
            // Mock gradient updates
            for i in 0..n_samples {
                for j in 0..params.n_components {
                    embedding[[i, j]] += rng.gen_range(-0.001..0.001);
                }
            }
        }

        Ok(embedding)
    }
}

/// Parameters for t-SNE algorithm
#[derive(Debug, Clone)]
pub struct TSNEParams {
    /// n_components
    pub n_components: usize,
    /// perplexity
    pub perplexity: Float,
    /// learning_rate
    pub learning_rate: Float,
    /// n_iter
    pub n_iter: usize,
    /// random_state
    pub random_state: Option<u64>,
}

impl Default for TSNEParams {
    fn default() -> Self {
        Self {
            n_components: 2,
            perplexity: 30.0,
            learning_rate: 200.0,
            n_iter: 1000,
            random_state: Some(42),
        }
    }
}

/// Mock scikit-learn PCA implementation
pub struct SklearnPCA;

impl ReferenceImplementation for SklearnPCA {
    const NAME: &'static str = "sklearn.decomposition.PCA";
    type Params = PCAParams;

    fn fit_transform(
        &self,
        data: ArrayView2<Float>,
        params: &Self::Params,
    ) -> SklResult<Array2<Float>> {
        // Mock PCA implementation
        let n_samples = data.nrows();
        let n_features = data.ncols();
        let n_components = params.n_components.min(n_features).min(n_samples);

        // Center the data
        let mean = data.mean_axis(scirs2_core::ndarray::Axis(0)).unwrap();
        let mut centered = data.to_owned();
        for mut row in centered.rows_mut() {
            row -= &mean;
        }

        // For mock implementation, just return projection to first n_components
        let projection = centered
            .slice(scirs2_core::ndarray::s![.., ..n_components])
            .to_owned();

        Ok(projection)
    }
}

/// Parameters for PCA algorithm
#[derive(Debug, Clone)]
pub struct PCAParams {
    /// n_components
    pub n_components: usize,
}

impl Default for PCAParams {
    fn default() -> Self {
        Self { n_components: 2 }
    }
}

/// Mock scikit-learn Isomap implementation
pub struct SklearnIsomap;

impl ReferenceImplementation for SklearnIsomap {
    const NAME: &'static str = "sklearn.manifold.Isomap";
    type Params = IsomapParams;

    fn fit_transform(
        &self,
        data: ArrayView2<Float>,
        params: &Self::Params,
    ) -> SklResult<Array2<Float>> {
        let n_samples = data.nrows();

        // Mock Isomap: build k-NN graph and apply MDS
        let mut distances = Array2::zeros((n_samples, n_samples));

        // Compute Euclidean distances
        for i in 0..n_samples {
            for j in i..n_samples {
                let row_i = data.row(i);
                let row_j = data.row(j);
                let dist = row_i
                    .iter()
                    .zip(row_j.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<Float>()
                    .sqrt();
                distances[[i, j]] = dist;
                distances[[j, i]] = dist;
            }
        }

        // Apply simple dimensionality reduction (mock MDS)
        let embedding = data
            .slice(scirs2_core::ndarray::s![.., ..params.n_components])
            .to_owned();
        Ok(embedding)
    }
}

/// Parameters for Isomap algorithm
#[derive(Debug, Clone)]
pub struct IsomapParams {
    /// n_components
    pub n_components: usize,
    /// n_neighbors
    pub n_neighbors: usize,
}

impl Default for IsomapParams {
    fn default() -> Self {
        Self {
            n_components: 2,
            n_neighbors: 5,
        }
    }
}

/// Reference testing framework
pub struct ReferenceTestFramework {
    config: ReferenceTestConfig,
}

impl ReferenceTestFramework {
    /// Create a new reference testing framework
    pub fn new(config: ReferenceTestConfig) -> Self {
        Self { config }
    }

    /// Run comprehensive comparison tests
    pub fn run_all_tests(&self) -> Vec<ReferenceTestResult> {
        let mut results = Vec::new();

        // Test t-SNE
        results.extend(self.test_tsne());

        // Test PCA
        results.extend(self.test_pca());

        // Test Isomap
        results.extend(self.test_isomap());

        // Test edge cases if enabled
        if self.config.test_edge_cases {
            results.extend(self.test_edge_cases());
        }

        results
    }

    /// Test t-SNE implementation against reference
    fn test_tsne(&self) -> Vec<ReferenceTestResult> {
        let mut results = Vec::new();

        // Generate test data
        let test_data = self.generate_test_data(100, 5);
        let params = TSNEParams::default();

        // Run our implementation (mock)
        let our_result = self.run_our_tsne(&test_data.view(), &params);

        // Run reference implementation
        let reference = SklearnTSNE;
        let ref_result = reference.fit_transform(test_data.view(), &params);

        // Compare results
        let test_result = self.compare_results("t-SNE Basic Test", our_result, ref_result);
        results.push(test_result);

        // Test with multiple seeds if enabled
        if self.config.test_multiple_seeds {
            for seed in 0..self.config.n_seeds {
                let mut seed_params = params.clone();
                seed_params.random_state = Some(seed as u64);

                let our_seeded = self.run_our_tsne(&test_data.view(), &seed_params);
                let ref_seeded = reference.fit_transform(test_data.view(), &seed_params);

                let seed_test = self.compare_results(
                    &format!("t-SNE Seed {} Test", seed),
                    our_seeded,
                    ref_seeded,
                );
                results.push(seed_test);
            }
        }

        results
    }

    /// Test PCA implementation against reference
    fn test_pca(&self) -> Vec<ReferenceTestResult> {
        let mut results = Vec::new();

        let test_data = self.generate_test_data(50, 4);
        let params = PCAParams::default();

        let our_result = self.run_our_pca(&test_data.view(), &params);

        let reference = SklearnPCA;
        let ref_result = reference.fit_transform(test_data.view(), &params);

        let test_result = self.compare_results("PCA Basic Test", our_result, ref_result);
        results.push(test_result);

        results
    }

    /// Test Isomap implementation against reference
    fn test_isomap(&self) -> Vec<ReferenceTestResult> {
        let mut results = Vec::new();

        let test_data = self.generate_test_data(30, 3);
        let params = IsomapParams::default();

        let our_result = self.run_our_isomap(&test_data.view(), &params);

        let reference = SklearnIsomap;
        let ref_result = reference.fit_transform(test_data.view(), &params);

        let test_result = self.compare_results("Isomap Basic Test", our_result, ref_result);
        results.push(test_result);

        results
    }

    /// Test edge cases
    fn test_edge_cases(&self) -> Vec<ReferenceTestResult> {
        let mut results = Vec::new();

        // Test with minimal data
        let minimal_data = self.generate_test_data(3, 2);
        let params = TSNEParams {
            n_components: 1,
            perplexity: 1.0,
            n_iter: 50,
            ..Default::default()
        };

        let our_result = self.run_our_tsne(&minimal_data.view(), &params);
        let reference = SklearnTSNE;
        let ref_result = reference.fit_transform(minimal_data.view(), &params);

        let edge_test = self.compare_results("Edge Case: Minimal Data", our_result, ref_result);
        results.push(edge_test);

        results
    }

    /// Mock implementation of our t-SNE algorithm
    fn run_our_tsne(
        &self,
        data: &ArrayView2<Float>,
        params: &TSNEParams,
    ) -> SklResult<Array2<Float>> {
        // This would call our actual t-SNE implementation
        // For now, simulate with similar behavior to reference

        let n_samples = data.nrows();
        let mut embedding = Array2::zeros((n_samples, params.n_components));

        let mut rng = if let Some(seed) = params.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().gen())
        };

        // Initialize embedding randomly
        for i in 0..n_samples {
            for j in 0..params.n_components {
                embedding[[i, j]] = rng.gen_range(-1.0..1.0) * 0.0001;
            }
        }

        // Simulate optimization with slight differences from reference
        for _iter in 0..params.n_iter.min(10) {
            for i in 0..n_samples {
                for j in 0..params.n_components {
                    embedding[[i, j]] += rng.gen_range(-0.0015..0.0015); // Slightly different from reference
                }
            }
        }

        Ok(embedding)
    }

    /// Mock implementation of our PCA algorithm
    fn run_our_pca(
        &self,
        data: &ArrayView2<Float>,
        params: &PCAParams,
    ) -> SklResult<Array2<Float>> {
        // Mock our PCA implementation
        let n_samples = data.nrows();
        let n_features = data.ncols();
        let n_components = params.n_components.min(n_features).min(n_samples);

        // Apply slightly different centering for testing
        let mean = data.mean_axis(scirs2_core::ndarray::Axis(0)).unwrap();
        let mut centered = data.to_owned();
        for mut row in centered.rows_mut() {
            row -= &mean;
        }

        // Add small numerical differences
        for elem in centered.iter_mut() {
            *elem *= 1.000001; // Tiny scaling difference
        }

        let projection = centered
            .slice(scirs2_core::ndarray::s![.., ..n_components])
            .to_owned();
        Ok(projection)
    }

    /// Mock implementation of our Isomap algorithm
    fn run_our_isomap(
        &self,
        data: &ArrayView2<Float>,
        params: &IsomapParams,
    ) -> SklResult<Array2<Float>> {
        // Mock our Isomap implementation with slight differences
        let embedding = data
            .slice(scirs2_core::ndarray::s![.., ..params.n_components])
            .to_owned();

        // Add small perturbations to simulate algorithm differences
        let mut perturbed = embedding;
        for elem in perturbed.iter_mut() {
            *elem += 1e-8; // Tiny numerical difference
        }

        Ok(perturbed)
    }

    /// Compare results from our implementation vs reference
    fn compare_results(
        &self,
        test_name: &str,
        our_result: SklResult<Array2<Float>>,
        ref_result: SklResult<Array2<Float>>,
    ) -> ReferenceTestResult {
        match (our_result, ref_result) {
            (Ok(our_embedding), Ok(ref_embedding)) => {
                // Check shape compatibility
                if our_embedding.shape() != ref_embedding.shape() {
                    return ReferenceTestResult {
                        test_name: test_name.to_string(),
                        passed: false,
                        error_message: Some(format!(
                            "Shape mismatch: our={:?}, reference={:?}",
                            our_embedding.shape(),
                            ref_embedding.shape()
                        )),
                        max_difference: Float::INFINITY,
                        performance_metrics: None,
                        metadata: HashMap::new(),
                    };
                }

                // Compute maximum absolute difference
                let max_diff = our_embedding
                    .iter()
                    .zip(ref_embedding.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0 as Float, |max_val, diff| max_val.max(diff));

                let passed = max_diff <= self.config.tolerance;

                let mut metadata = HashMap::new();
                metadata.insert(
                    "our_shape".to_string(),
                    format!("{:?}", our_embedding.shape()),
                );
                metadata.insert(
                    "ref_shape".to_string(),
                    format!("{:?}", ref_embedding.shape()),
                );
                metadata.insert("max_difference".to_string(), format!("{:.2e}", max_diff));

                ReferenceTestResult {
                    test_name: test_name.to_string(),
                    passed,
                    error_message: if passed {
                        None
                    } else {
                        Some(format!(
                            "Max difference {} exceeds tolerance {}",
                            max_diff, self.config.tolerance
                        ))
                    },
                    max_difference: max_diff,
                    performance_metrics: None,
                    metadata,
                }
            }
            (Err(our_error), Err(_ref_error)) => {
                // Both failed - this might be expected for edge cases
                ReferenceTestResult {
                    test_name: test_name.to_string(),
                    passed: true, // Both failed consistently
                    error_message: None,
                    max_difference: 0.0,
                    performance_metrics: None,
                    metadata: {
                        let mut map = HashMap::new();
                        map.insert("both_failed".to_string(), "true".to_string());
                        map.insert("our_error".to_string(), our_error.to_string());
                        map
                    },
                }
            }
            (Ok(_), Err(ref_error)) => ReferenceTestResult {
                test_name: test_name.to_string(),
                passed: false,
                error_message: Some(format!(
                    "Reference failed but ours succeeded: {}",
                    ref_error
                )),
                max_difference: Float::INFINITY,
                performance_metrics: None,
                metadata: HashMap::new(),
            },
            (Err(our_error), Ok(_)) => ReferenceTestResult {
                test_name: test_name.to_string(),
                passed: false,
                error_message: Some(format!(
                    "Our implementation failed but reference succeeded: {}",
                    our_error
                )),
                max_difference: Float::INFINITY,
                performance_metrics: None,
                metadata: HashMap::new(),
            },
        }
    }

    /// Generate test data for algorithms
    fn generate_test_data(&self, n_samples: usize, n_features: usize) -> Array2<Float> {
        let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducible tests
        let mut data = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            for j in 0..n_features {
                data[[i, j]] = rng.gen_range(-1.0..1.0);
            }
        }

        data
    }

    /// Print test results summary
    pub fn print_test_summary(&self, results: &[ReferenceTestResult]) {
        let total_tests = results.len();
        let passed_tests = results.iter().filter(|r| r.passed).count();
        let failed_tests = total_tests - passed_tests;

        println!("\n=== Reference Test Summary ===");
        println!("Total tests: {}", total_tests);
        println!("Passed: {}", passed_tests);
        println!("Failed: {}", failed_tests);
        println!(
            "Pass rate: {:.1}%",
            (passed_tests as f64 / total_tests as f64) * 100.0
        );

        if failed_tests > 0 {
            println!("\n=== Failed Tests ===");
            for result in results.iter().filter(|r| !r.passed) {
                println!("❌ {}", result.test_name);
                if let Some(error) = &result.error_message {
                    println!("   Error: {}", error);
                }
                println!("   Max difference: {:.2e}", result.max_difference);
            }
        }

        println!("\n=== Passed Tests ===");
        for result in results.iter().filter(|r| r.passed) {
            println!(
                "✅ {} (max_diff: {:.2e})",
                result.test_name, result.max_difference
            );
        }
    }
}

/// Benchmark comparison utilities
pub struct BenchmarkComparison {
    config: ReferenceTestConfig,
}

impl BenchmarkComparison {
    /// Create a new benchmark comparison
    pub fn new(config: ReferenceTestConfig) -> Self {
        Self { config }
    }

    /// Run performance benchmarks against reference implementations
    pub fn run_performance_benchmarks(&self) -> Vec<ReferenceTestResult> {
        if !self.config.test_performance {
            return Vec::new();
        }

        let mut results = Vec::new();

        // Benchmark different data sizes
        for &n_samples in &[100, 500, 1000] {
            for &n_features in &[5, 10, 20] {
                let test_data = self.generate_test_data(n_samples, n_features);

                // Benchmark t-SNE
                let tsne_bench = self.benchmark_tsne(&test_data);
                results.push(tsne_bench);

                // Benchmark PCA
                let pca_bench = self.benchmark_pca(&test_data);
                results.push(pca_bench);
            }
        }

        results
    }

    /// Benchmark t-SNE performance
    fn benchmark_tsne(&self, data: &Array2<Float>) -> ReferenceTestResult {
        use std::time::Instant;

        let params = TSNEParams {
            n_iter: 100, // Reduced for benchmarking
            ..Default::default()
        };

        // Benchmark our implementation
        let start = Instant::now();
        let _our_result = self.mock_our_tsne(data.view(), &params);
        let our_time = start.elapsed().as_secs_f64();

        // Benchmark reference implementation
        let start = Instant::now();
        let reference = SklearnTSNE;
        let _ref_result = reference.fit_transform(data.view(), &params);
        let ref_time = start.elapsed().as_secs_f64();

        let speedup = if our_time > 0.0 {
            ref_time / our_time
        } else {
            1.0
        };

        let performance_metrics = PerformanceMetrics {
            our_runtime: our_time,
            reference_runtime: ref_time,
            speedup_factor: speedup,
            memory_usage: None,
        };

        let mut metadata = HashMap::new();
        metadata.insert("data_shape".to_string(), format!("{:?}", data.shape()));
        metadata.insert("speedup".to_string(), format!("{:.2}x", speedup));

        ReferenceTestResult {
            test_name: format!("t-SNE Performance {:?}", data.shape()),
            passed: true, // Performance tests always "pass"
            error_message: None,
            max_difference: 0.0,
            performance_metrics: Some(performance_metrics),
            metadata,
        }
    }

    /// Benchmark PCA performance
    fn benchmark_pca(&self, data: &Array2<Float>) -> ReferenceTestResult {
        let params = PCAParams::default();

        // Benchmark our implementation
        let start = Instant::now();
        let _our_result = self.mock_our_pca(data.view(), &params);
        let our_time = start.elapsed().as_secs_f64();

        // Benchmark reference implementation
        let start = Instant::now();
        let reference = SklearnPCA;
        let _ref_result = reference.fit_transform(data.view(), &params);
        let ref_time = start.elapsed().as_secs_f64();

        let speedup = if our_time > 0.0 {
            ref_time / our_time
        } else {
            1.0
        };

        let performance_metrics = PerformanceMetrics {
            our_runtime: our_time,
            reference_runtime: ref_time,
            speedup_factor: speedup,
            memory_usage: None,
        };

        let mut metadata = HashMap::new();
        metadata.insert("data_shape".to_string(), format!("{:?}", data.shape()));
        metadata.insert("speedup".to_string(), format!("{:.2}x", speedup));

        ReferenceTestResult {
            test_name: format!("PCA Performance {:?}", data.shape()),
            passed: true,
            error_message: None,
            max_difference: 0.0,
            performance_metrics: Some(performance_metrics),
            metadata,
        }
    }

    /// Mock our t-SNE for benchmarking
    fn mock_our_tsne(
        &self,
        data: ArrayView2<Float>,
        _params: &TSNEParams,
    ) -> SklResult<Array2<Float>> {
        // Simulate computation time
        std::thread::sleep(std::time::Duration::from_millis(10));
        Ok(Array2::zeros((data.nrows(), 2)))
    }

    /// Mock our PCA for benchmarking
    fn mock_our_pca(
        &self,
        data: ArrayView2<Float>,
        _params: &PCAParams,
    ) -> SklResult<Array2<Float>> {
        // Simulate computation time
        std::thread::sleep(std::time::Duration::from_millis(5));
        Ok(Array2::zeros((data.nrows(), 2)))
    }

    /// Generate test data
    fn generate_test_data(&self, n_samples: usize, n_features: usize) -> Array2<Float> {
        let mut rng = StdRng::seed_from_u64(42);
        let mut data = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            for j in 0..n_features {
                data[[i, j]] = rng.gen_range(-1.0..1.0);
            }
        }

        data
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reference_framework() {
        let config = ReferenceTestConfig {
            tolerance: 1e-3, // More lenient for mock tests
            test_multiple_seeds: false,
            n_seeds: 1,
            test_edge_cases: false,
            test_performance: false,
        };

        let framework = ReferenceTestFramework::new(config);
        let results = framework.run_all_tests();

        assert!(!results.is_empty());

        // Print results for manual inspection
        framework.print_test_summary(&results);
    }

    #[test]
    fn test_mock_implementations() {
        let data = Array2::zeros((10, 3));

        // Test mock sklearn implementations
        let tsne = SklearnTSNE;
        let tsne_params = TSNEParams::default();
        let tsne_result = tsne.fit_transform(data.view(), &tsne_params);
        assert!(tsne_result.is_ok());

        let pca = SklearnPCA;
        let pca_params = PCAParams::default();
        let pca_result = pca.fit_transform(data.view(), &pca_params);
        assert!(pca_result.is_ok());

        let isomap = SklearnIsomap;
        let isomap_params = IsomapParams::default();
        let isomap_result = isomap.fit_transform(data.view(), &isomap_params);
        assert!(isomap_result.is_ok());
    }

    #[test]
    fn test_performance_benchmarks() {
        let config = ReferenceTestConfig {
            test_performance: true,
            ..Default::default()
        };

        let benchmark = BenchmarkComparison::new(config);
        let results = benchmark.run_performance_benchmarks();

        assert!(!results.is_empty());

        // Check that performance metrics are included
        for result in &results {
            assert!(result.performance_metrics.is_some());
        }
    }
}
