//! Memory Optimization Features for Multiclass Classification
//!
//! This module provides memory-efficient storage, compression, and quantization
//! for multiclass models.

pub mod compression;
pub mod quantization;
pub mod storage;

pub use compression::*;
pub use quantization::*;
pub use storage::*;

use sklears_core::error::Result as SklResult;

/// Memory optimization configuration
#[derive(Debug, Clone)]
pub struct MemoryOptimizationConfig {
    /// Enable model compression
    pub enable_compression: bool,
    /// Enable weight quantization
    pub enable_quantization: bool,
    /// Quantization bits (4, 8, or 16)
    pub quantization_bits: u8,
    /// Enable sparse storage for matrices
    pub enable_sparse: bool,
    /// Sparsity threshold (values below this are set to zero)
    pub sparsity_threshold: f64,
}

impl Default for MemoryOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_compression: false,
            enable_quantization: false,
            quantization_bits: 8,
            enable_sparse: false,
            sparsity_threshold: 1e-6,
        }
    }
}

/// Memory statistics for a model
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total memory usage in bytes
    pub total_bytes: usize,
    /// Uncompressed size in bytes
    pub uncompressed_bytes: usize,
    /// Compression ratio (uncompressed / compressed)
    pub compression_ratio: f64,
    /// Number of parameters
    pub num_parameters: usize,
    /// Number of non-zero parameters (for sparse models)
    pub num_nonzero: usize,
}

impl MemoryStats {
    /// Create new memory statistics
    pub fn new() -> Self {
        Self {
            total_bytes: 0,
            uncompressed_bytes: 0,
            compression_ratio: 1.0,
            num_parameters: 0,
            num_nonzero: 0,
        }
    }

    /// Calculate sparsity ratio
    pub fn sparsity_ratio(&self) -> f64 {
        if self.num_parameters == 0 {
            return 0.0;
        }
        1.0 - (self.num_nonzero as f64 / self.num_parameters as f64)
    }

    /// Format size in human-readable format
    pub fn format_size(&self) -> String {
        format_bytes(self.total_bytes)
    }

    /// Print memory statistics
    pub fn print(&self) {
        println!("Memory Statistics:");
        println!("  Total size: {}", self.format_size());
        println!("  Uncompressed: {}", format_bytes(self.uncompressed_bytes));
        println!("  Compression ratio: {:.2}x", self.compression_ratio);
        println!("  Parameters: {}", self.num_parameters);
        println!("  Non-zero: {}", self.num_nonzero);
        println!("  Sparsity: {:.2}%", self.sparsity_ratio() * 100.0);
    }
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Format bytes in human-readable format
fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Trait for memory-optimizable models
pub trait MemoryOptimizable {
    /// Get memory statistics
    fn memory_stats(&self) -> MemoryStats;

    /// Optimize memory usage
    fn optimize_memory(&mut self, config: &MemoryOptimizationConfig) -> SklResult<()>;

    /// Estimate memory savings from optimization
    fn estimate_memory_savings(&self, config: &MemoryOptimizationConfig) -> SklResult<usize>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_optimization_config_default() {
        let config = MemoryOptimizationConfig::default();
        assert!(!config.enable_compression);
        assert!(!config.enable_quantization);
        assert_eq!(config.quantization_bits, 8);
        assert!(!config.enable_sparse);
    }

    #[test]
    fn test_memory_stats_new() {
        let stats = MemoryStats::new();
        assert_eq!(stats.total_bytes, 0);
        assert_eq!(stats.num_parameters, 0);
        assert_eq!(stats.sparsity_ratio(), 0.0);
    }

    #[test]
    fn test_memory_stats_sparsity() {
        let stats = MemoryStats {
            total_bytes: 1000,
            uncompressed_bytes: 2000,
            compression_ratio: 2.0,
            num_parameters: 100,
            num_nonzero: 25,
        };
        assert_eq!(stats.sparsity_ratio(), 0.75);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(2048), "2.00 KB");
        assert_eq!(format_bytes(2 * 1024 * 1024), "2.00 MB");
        assert_eq!(format_bytes(3 * 1024 * 1024 * 1024), "3.00 GB");
    }

    #[test]
    fn test_memory_stats_format() {
        let stats = MemoryStats {
            total_bytes: 1024 * 1024,
            uncompressed_bytes: 2 * 1024 * 1024,
            compression_ratio: 2.0,
            num_parameters: 1000,
            num_nonzero: 800,
        };
        let formatted = stats.format_size();
        assert_eq!(formatted, "1.00 MB");
    }
}
