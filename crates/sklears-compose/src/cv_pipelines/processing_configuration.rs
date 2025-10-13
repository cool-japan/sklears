//! Processing configuration and quality settings
//!
//! This module provides comprehensive configuration structures for computer vision
//! processing including quality settings, performance optimization, parallel processing,
//! memory management, and caching strategies.

use super::types_config::{
    CacheEvictionPolicy, LoadBalancingAlgorithm, MemoryOptimizationLevel, ParallelStrategy,
    ProcessingComplexity, RecoveryStrategy, TransformParameter,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Quality settings for computer vision processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitySettings {
    /// Overall quality level
    pub quality_level: ProcessingComplexity,
    /// Compression configuration
    pub compression: CompressionConfig,
    /// Noise reduction settings
    pub noise_reduction: NoiseReductionConfig,
    /// Sharpening configuration
    pub sharpening: SharpeningConfig,
    /// Color correction settings
    pub color_correction: ColorCorrectionConfig,
    /// Quality assurance settings
    pub quality_assurance: QualityAssuranceConfig,
}

impl Default for QualitySettings {
    fn default() -> Self {
        Self {
            quality_level: ProcessingComplexity::Medium,
            compression: CompressionConfig::default(),
            noise_reduction: NoiseReductionConfig::default(),
            sharpening: SharpeningConfig::default(),
            color_correction: ColorCorrectionConfig::default(),
            quality_assurance: QualityAssuranceConfig::default(),
        }
    }
}

impl QualitySettings {
    /// Create high-quality settings
    #[must_use]
    pub fn high_quality() -> Self {
        Self {
            quality_level: ProcessingComplexity::High,
            compression: CompressionConfig::lossless(),
            noise_reduction: NoiseReductionConfig::aggressive(),
            sharpening: SharpeningConfig::enhanced(),
            color_correction: ColorCorrectionConfig::accurate(),
            quality_assurance: QualityAssuranceConfig::strict(),
        }
    }

    /// Create performance-optimized settings
    #[must_use]
    pub fn performance_optimized() -> Self {
        Self {
            quality_level: ProcessingComplexity::Low,
            compression: CompressionConfig::fast(),
            noise_reduction: NoiseReductionConfig::basic(),
            sharpening: SharpeningConfig::minimal(),
            color_correction: ColorCorrectionConfig::basic(),
            quality_assurance: QualityAssuranceConfig::lenient(),
        }
    }

    /// Create balanced settings
    #[must_use]
    pub fn balanced() -> Self {
        Self::default()
    }
}

/// Compression configuration for image processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Enable compression
    pub enabled: bool,
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Quality level (0-100, higher is better quality)
    pub quality: u8,
    /// Enable progressive encoding
    pub progressive: bool,
    /// Optimize for file size
    pub optimize_size: bool,
    /// Custom compression parameters
    pub custom_params: HashMap<String, TransformParameter>,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: CompressionAlgorithm::JPEG,
            quality: 85,
            progressive: true,
            optimize_size: false,
            custom_params: HashMap::new(),
        }
    }
}

impl CompressionConfig {
    /// Create lossless compression configuration
    #[must_use]
    pub fn lossless() -> Self {
        Self {
            enabled: true,
            algorithm: CompressionAlgorithm::PNG,
            quality: 100,
            progressive: false,
            optimize_size: false,
            custom_params: HashMap::new(),
        }
    }

    /// Create fast compression configuration
    #[must_use]
    pub fn fast() -> Self {
        Self {
            enabled: true,
            algorithm: CompressionAlgorithm::JPEG,
            quality: 70,
            progressive: false,
            optimize_size: true,
            custom_params: HashMap::new(),
        }
    }
}

/// Compression algorithms for image processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// JPEG compression
    JPEG,
    /// PNG compression
    PNG,
    /// WebP compression
    WebP,
    /// HEIF compression
    HEIF,
    /// AVIF compression
    AVIF,
    /// Custom compression
    Custom,
}

impl Default for CompressionAlgorithm {
    fn default() -> Self {
        Self::JPEG
    }
}

/// Noise reduction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseReductionConfig {
    /// Enable noise reduction
    pub enabled: bool,
    /// Denoising algorithm
    pub algorithm: DenoisingAlgorithm,
    /// Noise reduction strength (0.0-1.0)
    pub strength: f64,
    /// Preserve edges while denoising
    pub preserve_edges: bool,
    /// Preserve fine details
    pub preserve_details: bool,
    /// Custom denoising parameters
    pub custom_params: HashMap<String, TransformParameter>,
}

impl Default for NoiseReductionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: DenoisingAlgorithm::BilateralFilter,
            strength: 0.5,
            preserve_edges: true,
            preserve_details: true,
            custom_params: HashMap::new(),
        }
    }
}

impl NoiseReductionConfig {
    /// Create aggressive noise reduction configuration
    #[must_use]
    pub fn aggressive() -> Self {
        Self {
            enabled: true,
            algorithm: DenoisingAlgorithm::NonLocalMeans,
            strength: 0.8,
            preserve_edges: true,
            preserve_details: false,
            custom_params: HashMap::new(),
        }
    }

    /// Create basic noise reduction configuration
    #[must_use]
    pub fn basic() -> Self {
        Self {
            enabled: true,
            algorithm: DenoisingAlgorithm::GaussianBlur,
            strength: 0.3,
            preserve_edges: false,
            preserve_details: true,
            custom_params: HashMap::new(),
        }
    }
}

/// Denoising algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DenoisingAlgorithm {
    /// Gaussian blur denoising
    GaussianBlur,
    /// Bilateral filter
    BilateralFilter,
    /// Non-local means denoising
    NonLocalMeans,
    /// Wiener filter
    WienerFilter,
    /// Wavelet denoising
    Wavelet,
    /// Deep learning denoising
    DeepLearning,
    /// Custom denoising
    Custom,
}

impl Default for DenoisingAlgorithm {
    fn default() -> Self {
        Self::BilateralFilter
    }
}

/// Sharpening configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharpeningConfig {
    /// Enable sharpening
    pub enabled: bool,
    /// Sharpening strength (0.0-2.0)
    pub strength: f64,
    /// Sharpening radius
    pub radius: f64,
    /// Threshold for edge detection
    pub threshold: f64,
    /// Unsharp mask settings
    pub unsharp_mask: bool,
    /// Custom sharpening parameters
    pub custom_params: HashMap<String, TransformParameter>,
}

impl Default for SharpeningConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            strength: 1.0,
            radius: 1.0,
            threshold: 0.1,
            unsharp_mask: true,
            custom_params: HashMap::new(),
        }
    }
}

impl SharpeningConfig {
    /// Create enhanced sharpening configuration
    #[must_use]
    pub fn enhanced() -> Self {
        Self {
            enabled: true,
            strength: 1.5,
            radius: 1.5,
            threshold: 0.05,
            unsharp_mask: true,
            custom_params: HashMap::new(),
        }
    }

    /// Create minimal sharpening configuration
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            enabled: true,
            strength: 0.5,
            radius: 0.5,
            threshold: 0.2,
            unsharp_mask: false,
            custom_params: HashMap::new(),
        }
    }
}

/// Color correction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorCorrectionConfig {
    /// Enable color correction
    pub enabled: bool,
    /// Brightness adjustment (-1.0 to 1.0)
    pub brightness: f64,
    /// Contrast adjustment (0.0 to 2.0)
    pub contrast: f64,
    /// Saturation adjustment (0.0 to 2.0)
    pub saturation: f64,
    /// Hue adjustment (-180 to 180 degrees)
    pub hue: f64,
    /// Gamma correction (0.1 to 3.0)
    pub gamma: f64,
    /// White balance correction
    pub white_balance: WhiteBalanceConfig,
    /// Color temperature adjustment
    pub color_temperature: Option<f64>,
}

impl Default for ColorCorrectionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            brightness: 0.0,
            contrast: 1.0,
            saturation: 1.0,
            hue: 0.0,
            gamma: 1.0,
            white_balance: WhiteBalanceConfig::default(),
            color_temperature: None,
        }
    }
}

impl ColorCorrectionConfig {
    /// Create accurate color correction configuration
    #[must_use]
    pub fn accurate() -> Self {
        Self {
            enabled: true,
            brightness: 0.0,
            contrast: 1.0,
            saturation: 1.0,
            hue: 0.0,
            gamma: 2.2, // Standard gamma for sRGB
            white_balance: WhiteBalanceConfig::auto(),
            color_temperature: Some(6500.0), // Daylight
        }
    }

    /// Create basic color correction configuration
    #[must_use]
    pub fn basic() -> Self {
        Self {
            enabled: true,
            brightness: 0.0,
            contrast: 1.0,
            saturation: 1.0,
            hue: 0.0,
            gamma: 1.0,
            white_balance: WhiteBalanceConfig::none(),
            color_temperature: None,
        }
    }
}

/// White balance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhiteBalanceConfig {
    /// Enable white balance correction
    pub enabled: bool,
    /// White balance mode
    pub mode: WhiteBalanceMode,
    /// Custom white point (R, G, B multipliers)
    pub custom_white_point: Option<(f64, f64, f64)>,
}

impl Default for WhiteBalanceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: WhiteBalanceMode::Auto,
            custom_white_point: None,
        }
    }
}

impl WhiteBalanceConfig {
    /// Create auto white balance configuration
    #[must_use]
    pub fn auto() -> Self {
        Self {
            enabled: true,
            mode: WhiteBalanceMode::Auto,
            custom_white_point: None,
        }
    }

    /// Create disabled white balance configuration
    #[must_use]
    pub fn none() -> Self {
        Self {
            enabled: false,
            mode: WhiteBalanceMode::None,
            custom_white_point: None,
        }
    }
}

/// White balance modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WhiteBalanceMode {
    /// No white balance correction
    None,
    /// Automatic white balance
    Auto,
    /// Daylight white balance
    Daylight,
    /// Tungsten white balance
    Tungsten,
    /// Fluorescent white balance
    Fluorescent,
    /// Custom white balance
    Custom,
}

/// Quality assurance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssuranceConfig {
    /// Enable quality checks
    pub enabled: bool,
    /// Minimum acceptable quality score
    pub min_quality_score: f64,
    /// Maximum acceptable processing time
    pub max_processing_time: Duration,
    /// Enable automatic quality adjustment
    pub auto_adjust: bool,
    /// Quality metrics to monitor
    pub monitored_metrics: Vec<String>,
}

impl Default for QualityAssuranceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_quality_score: 0.7,
            max_processing_time: Duration::from_secs(30),
            auto_adjust: true,
            monitored_metrics: vec![
                "sharpness".to_string(),
                "contrast".to_string(),
                "brightness".to_string(),
            ],
        }
    }
}

impl QualityAssuranceConfig {
    /// Create strict quality assurance configuration
    #[must_use]
    pub fn strict() -> Self {
        Self {
            enabled: true,
            min_quality_score: 0.9,
            max_processing_time: Duration::from_secs(60),
            auto_adjust: false,
            monitored_metrics: vec![
                "sharpness".to_string(),
                "contrast".to_string(),
                "brightness".to_string(),
                "noise_level".to_string(),
                "color_accuracy".to_string(),
            ],
        }
    }

    /// Create lenient quality assurance configuration
    #[must_use]
    pub fn lenient() -> Self {
        Self {
            enabled: true,
            min_quality_score: 0.5,
            max_processing_time: Duration::from_secs(10),
            auto_adjust: true,
            monitored_metrics: vec!["sharpness".to_string()],
        }
    }
}

/// Performance configuration for computer vision processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Memory optimization level
    pub memory_optimization: MemoryOptimizationLevel,
    /// Caching strategy
    pub caching: CachingStrategy,
    /// Parallel processing configuration
    pub parallel_processing: ParallelProcessingConfig,
    /// Resource limits
    pub resource_limits: ResourceLimits,
    /// Error handling strategy
    pub error_handling: ErrorHandlingConfig,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            memory_optimization: MemoryOptimizationLevel::Medium,
            caching: CachingStrategy::default(),
            parallel_processing: ParallelProcessingConfig::default(),
            resource_limits: ResourceLimits::default(),
            error_handling: ErrorHandlingConfig::default(),
        }
    }
}

/// Caching strategy for performance optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingStrategy {
    /// Enable caching
    pub enabled: bool,
    /// Cache size in bytes
    pub cache_size: usize,
    /// Cache eviction policy
    pub eviction_policy: CacheEvictionPolicy,
    /// Cache TTL for entries
    pub ttl: Duration,
    /// Enable persistent cache
    pub persistent: bool,
    /// Cache location
    pub cache_location: String,
}

impl Default for CachingStrategy {
    fn default() -> Self {
        Self {
            enabled: true,
            cache_size: 100 * 1024 * 1024, // 100MB
            eviction_policy: CacheEvictionPolicy::LRU,
            ttl: Duration::from_secs(3600), // 1 hour
            persistent: false,
            cache_location: "/tmp/cv_cache".to_string(),
        }
    }
}

/// Parallel processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelProcessingConfig {
    /// Enable parallel processing
    pub enabled: bool,
    /// Parallel strategy
    pub strategy: ParallelStrategy,
    /// Number of worker threads/processes
    pub num_workers: usize,
    /// Load balancing algorithm
    pub load_balancing: LoadBalancingAlgorithm,
    /// Batch size for parallel processing
    pub batch_size: usize,
    /// Work stealing configuration
    pub work_stealing: WorkStealingConfig,
}

impl Default for ParallelProcessingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strategy: ParallelStrategy::Threading,
            num_workers: num_cpus::get(),
            load_balancing: LoadBalancingAlgorithm::WorkStealing,
            batch_size: 32,
            work_stealing: WorkStealingConfig::default(),
        }
    }
}

/// Work stealing configuration for load balancing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkStealingConfig {
    /// Enable work stealing
    pub enabled: bool,
    /// Steal threshold (when to steal work)
    pub steal_threshold: usize,
    /// Maximum steal attempts
    pub max_steal_attempts: usize,
    /// Steal ratio (fraction of work to steal)
    pub steal_ratio: f64,
}

impl Default for WorkStealingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            steal_threshold: 5,
            max_steal_attempts: 3,
            steal_ratio: 0.5,
        }
    }
}

/// Resource limits for processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory usage in bytes
    pub max_memory: usize,
    /// Maximum CPU usage percentage
    pub max_cpu_usage: f64,
    /// Maximum GPU memory usage in bytes
    pub max_gpu_memory: Option<usize>,
    /// Maximum processing time per item
    pub max_processing_time: Duration,
    /// Maximum concurrent operations
    pub max_concurrent_operations: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory: 1024 * 1024 * 1024, // 1GB
            max_cpu_usage: 80.0,            // 80%
            max_gpu_memory: None,
            max_processing_time: Duration::from_secs(30),
            max_concurrent_operations: 10,
        }
    }
}

/// Error handling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorHandlingConfig {
    /// Recovery strategy for failures
    pub recovery_strategy: RecoveryStrategy,
    /// Maximum retry attempts
    pub max_retries: usize,
    /// Retry delay
    pub retry_delay: Duration,
    /// Enable graceful degradation
    pub graceful_degradation: bool,
    /// Fallback quality level
    pub fallback_quality: ProcessingComplexity,
}

impl Default for ErrorHandlingConfig {
    fn default() -> Self {
        Self {
            recovery_strategy: RecoveryStrategy::Skip,
            max_retries: 3,
            retry_delay: Duration::from_millis(500),
            graceful_degradation: true,
            fallback_quality: ProcessingComplexity::Low,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_settings_presets() {
        let high_quality = QualitySettings::high_quality();
        assert_eq!(high_quality.quality_level, ProcessingComplexity::High);

        let performance = QualitySettings::performance_optimized();
        assert_eq!(performance.quality_level, ProcessingComplexity::Low);

        let balanced = QualitySettings::balanced();
        assert_eq!(balanced.quality_level, ProcessingComplexity::Medium);
    }

    #[test]
    fn test_compression_config() {
        let lossless = CompressionConfig::lossless();
        assert_eq!(lossless.algorithm, CompressionAlgorithm::PNG);
        assert_eq!(lossless.quality, 100);

        let fast = CompressionConfig::fast();
        assert_eq!(fast.algorithm, CompressionAlgorithm::JPEG);
        assert!(fast.optimize_size);
    }

    #[test]
    fn test_noise_reduction_config() {
        let aggressive = NoiseReductionConfig::aggressive();
        assert_eq!(aggressive.algorithm, DenoisingAlgorithm::NonLocalMeans);
        assert_eq!(aggressive.strength, 0.8);

        let basic = NoiseReductionConfig::basic();
        assert_eq!(basic.algorithm, DenoisingAlgorithm::GaussianBlur);
        assert_eq!(basic.strength, 0.3);
    }

    #[test]
    fn test_color_correction_config() {
        let accurate = ColorCorrectionConfig::accurate();
        assert!(accurate.enabled);
        assert_eq!(accurate.gamma, 2.2);
        assert_eq!(accurate.color_temperature, Some(6500.0));

        let basic = ColorCorrectionConfig::basic();
        assert!(basic.enabled);
        assert_eq!(basic.gamma, 1.0);
        assert_eq!(basic.color_temperature, None);
    }

    #[test]
    fn test_quality_assurance_config() {
        let strict = QualityAssuranceConfig::strict();
        assert_eq!(strict.min_quality_score, 0.9);
        assert!(!strict.auto_adjust);
        assert_eq!(strict.monitored_metrics.len(), 5);

        let lenient = QualityAssuranceConfig::lenient();
        assert_eq!(lenient.min_quality_score, 0.5);
        assert!(lenient.auto_adjust);
        assert_eq!(lenient.monitored_metrics.len(), 1);
    }

    #[test]
    fn test_performance_config_defaults() {
        let config = PerformanceConfig::default();
        assert_eq!(config.memory_optimization, MemoryOptimizationLevel::Medium);
        assert!(config.caching.enabled);
        assert!(config.parallel_processing.enabled);
    }

    #[test]
    fn test_resource_limits() {
        let limits = ResourceLimits::default();
        assert_eq!(limits.max_memory, 1024 * 1024 * 1024);
        assert_eq!(limits.max_cpu_usage, 80.0);
        assert_eq!(limits.max_concurrent_operations, 10);
    }

    #[test]
    fn test_error_handling_config() {
        let config = ErrorHandlingConfig::default();
        assert_eq!(config.recovery_strategy, RecoveryStrategy::Skip);
        assert_eq!(config.max_retries, 3);
        assert!(config.graceful_degradation);
        assert_eq!(config.fallback_quality, ProcessingComplexity::Low);
    }
}
