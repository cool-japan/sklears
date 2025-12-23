//! Compression Management Module
//!
//! This module provides comprehensive compression management capabilities including
//! multiple compression algorithms, performance benchmarking, adaptive selection,
//! and compression optimization for notification channel performance.

use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use serde::{Deserialize, Serialize};

use super::performance_core::CompressionAlgorithm;

/// Compression manager for data optimization
#[derive(Debug, Clone)]
pub struct CompressionManager {
    /// Compression engines
    pub engines: HashMap<CompressionAlgorithm, CompressionEngine>,
    /// Compression statistics
    pub statistics: CompressionStatistics,
    /// Compression configuration
    pub config: CompressionManagerConfig,
    /// Compression benchmarks
    pub benchmarks: CompressionBenchmarks,
}

/// Compression engine
#[derive(Debug, Clone)]
pub struct CompressionEngine {
    /// Engine identifier
    pub engine_id: String,
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Engine configuration
    pub config: CompressionEngineConfig,
    /// Engine statistics
    pub statistics: CompressionEngineStatistics,
    /// Performance metrics
    pub performance: CompressionPerformance,
}

/// Compression engine configuration
#[derive(Debug, Clone)]
pub struct CompressionEngineConfig {
    /// Default compression level
    pub default_level: u8,
    /// Minimum size threshold
    pub min_size_threshold: usize,
    /// Maximum size threshold
    pub max_size_threshold: Option<usize>,
    /// Enable parallel compression
    pub parallel_compression: bool,
    /// Compression buffer size
    pub buffer_size: usize,
}

/// Compression engine statistics
#[derive(Debug, Clone)]
pub struct CompressionEngineStatistics {
    /// Total compressions
    pub total_compressions: u64,
    /// Total decompressions
    pub total_decompressions: u64,
    /// Bytes compressed
    pub bytes_compressed: u64,
    /// Bytes decompressed
    pub bytes_decompressed: u64,
    /// Average compression ratio
    pub avg_compression_ratio: f64,
    /// Compression errors
    pub compression_errors: u64,
}

/// Compression performance metrics
#[derive(Debug, Clone)]
pub struct CompressionPerformance {
    /// Average compression time
    pub avg_compression_time: Duration,
    /// Average decompression time
    pub avg_decompression_time: Duration,
    /// Compression throughput (bytes/sec)
    pub compression_throughput: f64,
    /// Decompression throughput (bytes/sec)
    pub decompression_throughput: f64,
    /// CPU usage
    pub cpu_usage: f64,
    /// Memory usage
    pub memory_usage: usize,
}

/// Compression manager statistics
#[derive(Debug, Clone)]
pub struct CompressionStatistics {
    /// Total data processed
    pub total_data_processed: u64,
    /// Total space saved
    pub total_space_saved: u64,
    /// Global compression ratio
    pub global_compression_ratio: f64,
    /// Performance by algorithm
    pub performance_by_algorithm: HashMap<CompressionAlgorithm, CompressionPerformance>,
    /// Usage statistics
    pub usage_statistics: HashMap<CompressionAlgorithm, u64>,
}

/// Compression manager configuration
#[derive(Debug, Clone)]
pub struct CompressionManagerConfig {
    /// Enable compression
    pub enabled: bool,
    /// Auto-select best algorithm
    pub auto_select_algorithm: bool,
    /// Benchmark interval
    pub benchmark_interval: Duration,
    /// Enable adaptive compression
    pub adaptive_compression: bool,
    /// Performance monitoring
    pub performance_monitoring: bool,
}

/// Compression benchmarks
#[derive(Debug, Clone)]
pub struct CompressionBenchmarks {
    /// Benchmark results
    pub results: HashMap<CompressionAlgorithm, BenchmarkResult>,
    /// Last benchmark timestamp
    pub last_benchmark: SystemTime,
    /// Benchmark configuration
    pub config: BenchmarkConfig,
}

/// Benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Algorithm tested
    pub algorithm: CompressionAlgorithm,
    /// Compression ratio achieved
    pub compression_ratio: f64,
    /// Compression speed (bytes/sec)
    pub compression_speed: f64,
    /// Decompression speed (bytes/sec)
    pub decompression_speed: f64,
    /// Memory usage
    pub memory_usage: usize,
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Benchmark score
    pub score: f64,
}

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Test data size
    pub test_data_size: usize,
    /// Number of iterations
    pub iterations: u32,
    /// Benchmark timeout
    pub timeout: Duration,
    /// Include memory profiling
    pub memory_profiling: bool,
    /// Include CPU profiling
    pub cpu_profiling: bool,
}

/// Compression request for processing
#[derive(Debug, Clone)]
pub struct CompressionRequest {
    /// Request identifier
    pub request_id: String,
    /// Input data
    pub data: Vec<u8>,
    /// Preferred algorithm
    pub preferred_algorithm: Option<CompressionAlgorithm>,
    /// Compression level
    pub level: Option<u8>,
    /// Maximum compression time
    pub max_time: Option<Duration>,
    /// Quality requirements
    pub quality_requirements: CompressionQualityRequirements,
}

/// Compression quality requirements
#[derive(Debug, Clone)]
pub struct CompressionQualityRequirements {
    /// Minimum compression ratio
    pub min_compression_ratio: Option<f64>,
    /// Maximum compression time
    pub max_compression_time: Option<Duration>,
    /// Maximum memory usage
    pub max_memory_usage: Option<usize>,
    /// Quality priority
    pub priority: CompressionPriority,
}

/// Compression priority levels
#[derive(Debug, Clone)]
pub enum CompressionPriority {
    Speed,
    Ratio,
    Memory,
    Balanced,
}

/// Compression result
#[derive(Debug, Clone)]
pub struct CompressionResult {
    /// Request identifier
    pub request_id: String,
    /// Compressed data
    pub compressed_data: Vec<u8>,
    /// Algorithm used
    pub algorithm_used: CompressionAlgorithm,
    /// Compression ratio achieved
    pub compression_ratio: f64,
    /// Compression time
    pub compression_time: Duration,
    /// Original size
    pub original_size: usize,
    /// Compressed size
    pub compressed_size: usize,
    /// Compression metadata
    pub metadata: CompressionMetadata,
}

/// Compression metadata
#[derive(Debug, Clone)]
pub struct CompressionMetadata {
    /// Compression timestamp
    pub compressed_at: SystemTime,
    /// Compression parameters used
    pub parameters: HashMap<String, String>,
    /// Performance metrics
    pub performance_metrics: CompressionPerformance,
    /// Quality metrics
    pub quality_metrics: CompressionQualityMetrics,
}

/// Compression quality metrics
#[derive(Debug, Clone)]
pub struct CompressionQualityMetrics {
    /// Compression efficiency score
    pub efficiency_score: f64,
    /// Speed score
    pub speed_score: f64,
    /// Memory efficiency score
    pub memory_score: f64,
    /// Overall quality score
    pub overall_score: f64,
}

impl CompressionManager {
    /// Create a new compression manager
    pub fn new() -> Self {
        let mut manager = Self {
            engines: HashMap::new(),
            statistics: CompressionStatistics::default(),
            config: CompressionManagerConfig::default(),
            benchmarks: CompressionBenchmarks::default(),
        };

        // Initialize default engines
        manager.initialize_default_engines();
        manager
    }

    /// Initialize default compression engines
    fn initialize_default_engines(&mut self) {
        let algorithms = vec![
            CompressionAlgorithm::Gzip,
            CompressionAlgorithm::Deflate,
            CompressionAlgorithm::Brotli,
            CompressionAlgorithm::LZ4,
            CompressionAlgorithm::Zstd,
        ];

        for algorithm in algorithms {
            let engine = CompressionEngine::new(algorithm.clone());
            self.engines.insert(algorithm, engine);
        }
    }

    /// Compress data using specified algorithm
    pub fn compress(&mut self, data: &[u8], algorithm: CompressionAlgorithm) -> Result<Vec<u8>, String> {
        if !self.config.enabled {
            return Ok(data.to_vec());
        }

        let engine = self.engines.get_mut(&algorithm)
            .ok_or_else(|| format!("Compression engine for {:?} not found", algorithm))?;

        let start_time = SystemTime::now();
        let result = self.perform_compression(data, engine);
        let compression_time = start_time.elapsed().unwrap_or(Duration::from_millis(0));

        // Update statistics
        engine.statistics.total_compressions += 1;
        engine.statistics.bytes_compressed += data.len() as u64;

        if let Ok(compressed_data) = &result {
            let ratio = data.len() as f64 / compressed_data.len() as f64;
            engine.statistics.avg_compression_ratio =
                (engine.statistics.avg_compression_ratio + ratio) / 2.0;

            // Update performance metrics
            engine.performance.avg_compression_time =
                (engine.performance.avg_compression_time + compression_time) / 2;
        } else {
            engine.statistics.compression_errors += 1;
        }

        result
    }

    /// Decompress data
    pub fn decompress(&mut self, data: &[u8], algorithm: CompressionAlgorithm) -> Result<Vec<u8>, String> {
        let engine = self.engines.get_mut(&algorithm)
            .ok_or_else(|| format!("Compression engine for {:?} not found", algorithm))?;

        let start_time = SystemTime::now();
        let result = self.perform_decompression(data, engine);
        let decompression_time = start_time.elapsed().unwrap_or(Duration::from_millis(0));

        // Update statistics
        engine.statistics.total_decompressions += 1;
        engine.statistics.bytes_decompressed += data.len() as u64;

        // Update performance metrics
        engine.performance.avg_decompression_time =
            (engine.performance.avg_decompression_time + decompression_time) / 2;

        result
    }

    /// Compress with automatic algorithm selection
    pub fn compress_auto(&mut self, data: &[u8]) -> Result<CompressionResult, String> {
        if !self.config.enabled {
            return Ok(CompressionResult {
                request_id: format!("auto_{}", SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos()),
                compressed_data: data.to_vec(),
                algorithm_used: CompressionAlgorithm::Gzip, // Default fallback
                compression_ratio: 1.0,
                compression_time: Duration::from_millis(0),
                original_size: data.len(),
                compressed_size: data.len(),
                metadata: CompressionMetadata::default(),
            });
        }

        let start_time = SystemTime::now();
        let best_algorithm = self.select_best_algorithm(data)?;
        let compressed_data = self.compress(data, best_algorithm.clone())?;
        let compression_time = start_time.elapsed().unwrap_or(Duration::from_millis(0));

        let compression_ratio = data.len() as f64 / compressed_data.len() as f64;
        let request_id = format!("auto_{}", SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos());

        Ok(CompressionResult {
            request_id,
            compressed_data,
            algorithm_used: best_algorithm,
            compression_ratio,
            compression_time,
            original_size: data.len(),
            compressed_size: compressed_data.len(),
            metadata: CompressionMetadata::new(compression_time),
        })
    }

    /// Process compression request
    pub fn process_request(&mut self, request: CompressionRequest) -> Result<CompressionResult, String> {
        let algorithm = if let Some(preferred) = request.preferred_algorithm {
            preferred
        } else if self.config.auto_select_algorithm {
            self.select_best_algorithm(&request.data)?
        } else {
            CompressionAlgorithm::Gzip // Default
        };

        let start_time = SystemTime::now();
        let compressed_data = self.compress(&request.data, algorithm.clone())?;
        let compression_time = start_time.elapsed().unwrap_or(Duration::from_millis(0));

        let compression_ratio = request.data.len() as f64 / compressed_data.len() as f64;

        Ok(CompressionResult {
            request_id: request.request_id,
            compressed_data,
            algorithm_used: algorithm,
            compression_ratio,
            compression_time,
            original_size: request.data.len(),
            compressed_size: compressed_data.len(),
            metadata: CompressionMetadata::new(compression_time),
        })
    }

    /// Select best compression algorithm for data
    pub fn select_best_algorithm(&self, data: &[u8]) -> Result<CompressionAlgorithm, String> {
        if data.len() < 1024 {
            // For small data, prefer speed
            return Ok(CompressionAlgorithm::LZ4);
        }

        // Analyze data characteristics
        let entropy = self.calculate_entropy(data);

        if entropy > 0.8 {
            // High entropy data - prefer faster algorithms
            Ok(CompressionAlgorithm::LZ4)
        } else if entropy < 0.3 {
            // Low entropy data - prefer better compression
            Ok(CompressionAlgorithm::Brotli)
        } else {
            // Balanced approach
            Ok(CompressionAlgorithm::Zstd)
        }
    }

    /// Run compression benchmarks
    pub fn run_benchmarks(&mut self) -> Result<(), String> {
        let test_data = self.generate_test_data();
        let mut results = HashMap::new();

        for algorithm in self.engines.keys().cloned().collect::<Vec<_>>() {
            let result = self.benchmark_algorithm(&algorithm, &test_data)?;
            results.insert(algorithm, result);
        }

        self.benchmarks.results = results;
        self.benchmarks.last_benchmark = SystemTime::now();
        Ok(())
    }

    /// Benchmark specific algorithm
    pub fn benchmark_algorithm(&mut self, algorithm: &CompressionAlgorithm, test_data: &[u8]) -> Result<BenchmarkResult, String> {
        let iterations = self.benchmarks.config.iterations;
        let mut total_compression_time = Duration::from_millis(0);
        let mut total_decompression_time = Duration::from_millis(0);
        let mut total_compressed_size = 0;

        for _ in 0..iterations {
            // Compression benchmark
            let start = SystemTime::now();
            let compressed = self.compress(test_data, algorithm.clone())?;
            total_compression_time += start.elapsed().unwrap_or(Duration::from_millis(0));
            total_compressed_size += compressed.len();

            // Decompression benchmark
            let start = SystemTime::now();
            let _ = self.decompress(&compressed, algorithm.clone())?;
            total_decompression_time += start.elapsed().unwrap_or(Duration::from_millis(0));
        }

        let avg_compression_time = total_compression_time / iterations;
        let avg_decompression_time = total_decompression_time / iterations;
        let avg_compressed_size = total_compressed_size / iterations as usize;
        let compression_ratio = test_data.len() as f64 / avg_compressed_size as f64;

        let compression_speed = test_data.len() as f64 / avg_compression_time.as_secs_f64();
        let decompression_speed = test_data.len() as f64 / avg_decompression_time.as_secs_f64();

        // Calculate overall score (balanced between ratio and speed)
        let ratio_score = compression_ratio.min(10.0) / 10.0; // Normalize to 0-1
        let speed_score = (compression_speed / 1_000_000.0).min(1.0); // Normalize to 0-1 (1MB/s = 1.0)
        let score = (ratio_score + speed_score) / 2.0;

        Ok(BenchmarkResult {
            algorithm: algorithm.clone(),
            compression_ratio,
            compression_speed,
            decompression_speed,
            memory_usage: 0, // TODO: Implement memory profiling
            cpu_usage: 0.0,  // TODO: Implement CPU profiling
            score,
        })
    }

    /// Get compression statistics
    pub fn get_statistics(&mut self) -> &CompressionStatistics {
        self.update_global_statistics();
        &self.statistics
    }

    /// Update global statistics from all engines
    fn update_global_statistics(&mut self) {
        let mut total_processed = 0;
        let mut total_saved = 0;
        let mut total_ratio = 0.0;
        let mut engine_count = 0;

        for engine in self.engines.values() {
            total_processed += engine.statistics.bytes_compressed;
            let saved = engine.statistics.bytes_compressed -
                (engine.statistics.bytes_compressed as f64 / engine.statistics.avg_compression_ratio) as u64;
            total_saved += saved;
            total_ratio += engine.statistics.avg_compression_ratio;
            engine_count += 1;
        }

        self.statistics.total_data_processed = total_processed;
        self.statistics.total_space_saved = total_saved;
        if engine_count > 0 {
            self.statistics.global_compression_ratio = total_ratio / engine_count as f64;
        }
    }

    fn perform_compression(&self, data: &[u8], _engine: &CompressionEngine) -> Result<Vec<u8>, String> {
        // Placeholder implementation - in reality, this would use actual compression libraries
        if data.len() < 10 {
            return Ok(data.to_vec());
        }

        // Simulate compression by returning smaller data
        let compressed_size = (data.len() as f64 * 0.7) as usize;
        Ok(data[..compressed_size.min(data.len())].to_vec())
    }

    fn perform_decompression(&self, data: &[u8], _engine: &CompressionEngine) -> Result<Vec<u8>, String> {
        // Placeholder implementation - in reality, this would use actual decompression libraries
        Ok(data.to_vec())
    }

    fn calculate_entropy(&self, data: &[u8]) -> f64 {
        // Simple entropy calculation
        let mut frequency = [0u32; 256];
        for &byte in data {
            frequency[byte as usize] += 1;
        }

        let len = data.len() as f64;
        let mut entropy = 0.0;

        for &count in &frequency {
            if count > 0 {
                let probability = count as f64 / len;
                entropy -= probability * probability.log2();
            }
        }

        entropy / 8.0 // Normalize to 0-1 range
    }

    fn generate_test_data(&self) -> Vec<u8> {
        // Generate test data for benchmarking
        let size = self.benchmarks.config.test_data_size;
        let mut data = Vec::with_capacity(size);

        // Create data with varying patterns
        for i in 0..size {
            data.push((i % 256) as u8);
        }

        data
    }
}

impl CompressionEngine {
    /// Create a new compression engine
    pub fn new(algorithm: CompressionAlgorithm) -> Self {
        Self {
            engine_id: format!("{:?}_engine", algorithm),
            algorithm,
            config: CompressionEngineConfig::default(),
            statistics: CompressionEngineStatistics::default(),
            performance: CompressionPerformance::default(),
        }
    }

    /// Get compression efficiency score
    pub fn get_efficiency_score(&self) -> f64 {
        let ratio_score = (self.statistics.avg_compression_ratio - 1.0).max(0.0) / 9.0; // Normalize 1-10 to 0-1
        let speed_score = (self.performance.compression_throughput / 1_000_000.0).min(1.0); // 1MB/s = 1.0
        let error_rate = if self.statistics.total_compressions > 0 {
            self.statistics.compression_errors as f64 / self.statistics.total_compressions as f64
        } else {
            0.0
        };
        let reliability_score = (1.0 - error_rate).max(0.0);

        (ratio_score + speed_score + reliability_score) / 3.0
    }
}

impl CompressionMetadata {
    /// Create new compression metadata
    pub fn new(compression_time: Duration) -> Self {
        Self {
            compressed_at: SystemTime::now(),
            parameters: HashMap::new(),
            performance_metrics: CompressionPerformance::default(),
            quality_metrics: CompressionQualityMetrics::default(),
        }
    }
}

// Default implementations
impl Default for CompressionStatistics {
    fn default() -> Self {
        Self {
            total_data_processed: 0,
            total_space_saved: 0,
            global_compression_ratio: 1.0,
            performance_by_algorithm: HashMap::new(),
            usage_statistics: HashMap::new(),
        }
    }
}

impl Default for CompressionManagerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_select_algorithm: true,
            benchmark_interval: Duration::from_secs(3600), // 1 hour
            adaptive_compression: true,
            performance_monitoring: true,
        }
    }
}

impl Default for CompressionBenchmarks {
    fn default() -> Self {
        Self {
            results: HashMap::new(),
            last_benchmark: SystemTime::UNIX_EPOCH,
            config: BenchmarkConfig::default(),
        }
    }
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            test_data_size: 1_048_576, // 1MB
            iterations: 10,
            timeout: Duration::from_secs(60),
            memory_profiling: false,
            cpu_profiling: false,
        }
    }
}

impl Default for CompressionEngineConfig {
    fn default() -> Self {
        Self {
            default_level: 6,
            min_size_threshold: 128,
            max_size_threshold: None,
            parallel_compression: true,
            buffer_size: 65536, // 64KB
        }
    }
}

impl Default for CompressionEngineStatistics {
    fn default() -> Self {
        Self {
            total_compressions: 0,
            total_decompressions: 0,
            bytes_compressed: 0,
            bytes_decompressed: 0,
            avg_compression_ratio: 1.0,
            compression_errors: 0,
        }
    }
}

impl Default for CompressionPerformance {
    fn default() -> Self {
        Self {
            avg_compression_time: Duration::from_millis(0),
            avg_decompression_time: Duration::from_millis(0),
            compression_throughput: 0.0,
            decompression_throughput: 0.0,
            cpu_usage: 0.0,
            memory_usage: 0,
        }
    }
}

impl Default for CompressionMetadata {
    fn default() -> Self {
        Self {
            compressed_at: SystemTime::now(),
            parameters: HashMap::new(),
            performance_metrics: CompressionPerformance::default(),
            quality_metrics: CompressionQualityMetrics::default(),
        }
    }
}

impl Default for CompressionQualityMetrics {
    fn default() -> Self {
        Self {
            efficiency_score: 0.0,
            speed_score: 0.0,
            memory_score: 0.0,
            overall_score: 0.0,
        }
    }
}