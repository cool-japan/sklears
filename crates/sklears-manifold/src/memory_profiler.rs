//! Memory profiling utilities for manifold learning algorithms
//!
//! This module provides comprehensive memory usage tracking and analysis
//! for manifold learning algorithms. It includes memory allocation tracking,
//! peak usage monitoring, and memory efficiency analysis capabilities.

use scirs2_core::ndarray::Array2;
use std::fmt;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryStats {
    /// Current memory usage in bytes
    pub current_bytes: usize,
    /// Peak memory usage in bytes
    pub peak_bytes: usize,
    /// Total allocations made
    pub total_allocations: usize,
    /// Total deallocations made
    pub total_deallocations: usize,
    /// Currently active allocations
    pub active_allocations: usize,
}

impl MemoryStats {
    /// Create new memory stats with zero values
    pub fn new() -> Self {
        Self {
            current_bytes: 0,
            peak_bytes: 0,
            total_allocations: 0,
            total_deallocations: 0,
            active_allocations: 0,
        }
    }

    /// Get memory usage in different units
    pub fn current_kb(&self) -> f64 {
        self.current_bytes as f64 / 1024.0
    }

    pub fn current_mb(&self) -> f64 {
        self.current_bytes as f64 / (1024.0 * 1024.0)
    }

    pub fn current_gb(&self) -> f64 {
        self.current_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn peak_kb(&self) -> f64 {
        self.peak_bytes as f64 / 1024.0
    }

    pub fn peak_mb(&self) -> f64 {
        self.peak_bytes as f64 / (1024.0 * 1024.0)
    }

    pub fn peak_gb(&self) -> f64 {
        self.peak_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Calculate allocation efficiency (ratio of current to peak)
    pub fn allocation_efficiency(&self) -> f64 {
        if self.peak_bytes > 0 {
            self.current_bytes as f64 / self.peak_bytes as f64
        } else {
            0.0
        }
    }

    /// Calculate allocation-deallocation balance
    pub fn allocation_balance(&self) -> i64 {
        self.total_allocations as i64 - self.total_deallocations as i64
    }
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for MemoryStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Memory Stats: Current: {:.2} MB, Peak: {:.2} MB, Active Allocs: {}, Efficiency: {:.1}%",
            self.current_mb(),
            self.peak_mb(),
            self.active_allocations,
            self.allocation_efficiency() * 100.0
        )
    }
}

/// Memory profiler for tracking algorithm memory usage
#[derive(Debug)]
pub struct MemoryProfiler {
    stats: Arc<Mutex<MemoryStats>>,
    checkpoints: HashMap<String, MemoryStats>,
    algorithm_name: String,
    dataset_info: DatasetInfo,
    enabled: bool,
}

impl MemoryProfiler {
    /// Create a new memory profiler
    pub fn new(algorithm_name: &str, dataset_info: DatasetInfo) -> Self {
        Self {
            stats: Arc::new(Mutex::new(MemoryStats::new())),
            checkpoints: HashMap::new(),
            algorithm_name: algorithm_name.to_string(),
            dataset_info,
            enabled: true,
        }
    }

    /// Record memory allocation
    pub fn record_allocation(&mut self, bytes: usize) {
        if !self.enabled {
            return;
        }

        if let Ok(mut stats) = self.stats.lock() {
            stats.current_bytes += bytes;
            stats.peak_bytes = stats.peak_bytes.max(stats.current_bytes);
            stats.total_allocations += 1;
            stats.active_allocations += 1;
        }
    }

    /// Record memory deallocation
    pub fn record_deallocation(&mut self, bytes: usize) {
        if !self.enabled {
            return;
        }

        if let Ok(mut stats) = self.stats.lock() {
            stats.current_bytes = stats.current_bytes.saturating_sub(bytes);
            stats.total_deallocations += 1;
            stats.active_allocations = stats.active_allocations.saturating_sub(1);
        }
    }

    /// Create a memory checkpoint
    pub fn checkpoint(&mut self, label: &str) {
        if !self.enabled {
            return;
        }

        if let Ok(stats) = self.stats.lock() {
            self.checkpoints.insert(label.to_string(), *stats);
        }
    }

    /// Get current memory statistics
    pub fn current_stats(&self) -> MemoryStats {
        if let Ok(stats) = self.stats.lock() {
            *stats
        } else {
            MemoryStats::new()
        }
    }

    /// Get memory stats at a specific checkpoint
    pub fn checkpoint_stats(&self, label: &str) -> Option<MemoryStats> {
        self.checkpoints.get(label).copied()
    }

    /// Calculate memory difference between checkpoints
    pub fn memory_diff(&self, from: &str, to: &str) -> Option<i64> {
        match (self.checkpoints.get(from), self.checkpoints.get(to)) {
            (Some(from_stats), Some(to_stats)) => {
                Some(to_stats.current_bytes as i64 - from_stats.current_bytes as i64)
            }
            _ => None,
        }
    }

    /// Generate memory profile report
    pub fn generate_report(&self) -> MemoryProfileReport {
        let final_stats = self.current_stats();

        let mut checkpoint_data = HashMap::new();
        for (label, stats) in &self.checkpoints {
            checkpoint_data.insert(label.clone(), *stats);
        }

        /// MemoryProfileReport
        MemoryProfileReport {
            algorithm_name: self.algorithm_name.clone(),
            dataset_info: self.dataset_info.clone(),
            final_stats,
            checkpoints: checkpoint_data,
            theoretical_minimum: self.calculate_theoretical_minimum(),
            efficiency_score: self.calculate_efficiency_score(),
        }
    }

    /// Enable or disable profiling
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        if let Ok(mut stats) = self.stats.lock() {
            *stats = MemoryStats::new();
        }
        self.checkpoints.clear();
    }

    /// Calculate theoretical minimum memory required
    fn calculate_theoretical_minimum(&self) -> usize {
        let n = self.dataset_info.n_samples;
        let d = self.dataset_info.n_features;
        let k = self.dataset_info.n_components;

        // Base data: input + output
        let base_memory = (n * d + n * k) * 8; // f64 = 8 bytes

        // Algorithm-specific overhead
        let algorithm_overhead = match self.algorithm_name.as_str() {
            "pca" => d * d * 8,                 // Covariance matrix
            "tsne" | "tsne_exact" => n * n * 8, // Distance matrix
            "tsne_barnes_hut" => n * 100,       // Approximate tree overhead
            "isomap" => n * n * 8 * 2,          // Distance + shortest path matrices
            "lle" => n * n * 8,                 // Weight matrix
            "umap" => n * 20 * 8,               // Sparse graph approximation
            _ => n * k * 8,                     // Default: output size
        };

        base_memory + algorithm_overhead
    }

    /// Calculate memory efficiency score (0-1, higher is better)
    fn calculate_efficiency_score(&self) -> f64 {
        let current_stats = self.current_stats();
        let theoretical_min = self.calculate_theoretical_minimum();

        if current_stats.peak_bytes > 0 {
            theoretical_min as f64 / current_stats.peak_bytes as f64
        } else {
            0.0
        }
    }
}

/// Dataset information for memory profiling
#[derive(Debug, Clone)]
pub struct DatasetInfo {
    /// name
    pub name: String,
    /// n_samples
    pub n_samples: usize,
    /// n_features
    pub n_features: usize,
    /// n_components
    pub n_components: usize,
}

impl DatasetInfo {
    pub fn new(name: &str, n_samples: usize, n_features: usize, n_components: usize) -> Self {
        Self {
            name: name.to_string(),
            n_samples,
            n_features,
            n_components,
        }
    }

    /// Calculate base data size (input + output arrays)
    pub fn base_data_size_bytes(&self) -> usize {
        (self.n_samples * self.n_features + self.n_samples * self.n_components) * 8
    }

    /// Get human-readable base data size
    pub fn base_data_size_mb(&self) -> f64 {
        self.base_data_size_bytes() as f64 / (1024.0 * 1024.0)
    }
}

/// Comprehensive memory profile report
#[derive(Debug, Clone)]
pub struct MemoryProfileReport {
    /// algorithm_name
    pub algorithm_name: String,
    /// dataset_info
    pub dataset_info: DatasetInfo,
    /// final_stats
    pub final_stats: MemoryStats,
    /// checkpoints
    pub checkpoints: HashMap<String, MemoryStats>,
    /// theoretical_minimum
    pub theoretical_minimum: usize,
    /// efficiency_score
    pub efficiency_score: f64,
}

impl MemoryProfileReport {
    /// Generate summary string
    pub fn summary(&self) -> String {
        let mut summary = format!(
            "Memory Profile Report: {}\n\
             Dataset: {} ({}×{} → {})\n\
             Peak Memory: {:.2} MB\n\
             Current Memory: {:.2} MB\n\
             Theoretical Minimum: {:.2} MB\n\
             Efficiency Score: {:.1}%\n\
             Total Allocations: {}\n\
             Active Allocations: {}\n",
            self.algorithm_name,
            self.dataset_info.name,
            self.dataset_info.n_samples,
            self.dataset_info.n_features,
            self.dataset_info.n_components,
            self.final_stats.peak_mb(),
            self.final_stats.current_mb(),
            self.theoretical_minimum as f64 / (1024.0 * 1024.0),
            self.efficiency_score * 100.0,
            self.final_stats.total_allocations,
            self.final_stats.active_allocations
        );

        if !self.checkpoints.is_empty() {
            summary.push_str("\nMemory Checkpoints:\n");
            let mut checkpoints: Vec<_> = self.checkpoints.iter().collect();
            checkpoints.sort_by_key(|(_, stats)| stats.peak_bytes);

            for (label, stats) in checkpoints {
                summary.push_str(&format!(
                    "  {}: {:.2} MB (peak), {:.2} MB (current)\n",
                    label,
                    stats.peak_mb(),
                    stats.current_mb()
                ));
            }
        }

        summary
    }

    /// Calculate memory overhead factor
    pub fn overhead_factor(&self) -> f64 {
        let base_size = self.dataset_info.base_data_size_bytes();
        if base_size > 0 {
            self.final_stats.peak_bytes as f64 / base_size as f64
        } else {
            0.0
        }
    }

    /// Check if memory usage is reasonable
    pub fn is_memory_efficient(&self, threshold: f64) -> bool {
        self.efficiency_score >= threshold
    }

    /// Get memory usage category
    pub fn memory_category(&self) -> MemoryCategory {
        let peak_mb = self.final_stats.peak_mb();

        if peak_mb < 10.0 {
            MemoryCategory::VeryLow
        } else if peak_mb < 100.0 {
            MemoryCategory::Low
        } else if peak_mb < 1000.0 {
            MemoryCategory::Medium
        } else if peak_mb < 10000.0 {
            MemoryCategory::High
        } else {
            MemoryCategory::VeryHigh
        }
    }
}

impl fmt::Display for MemoryProfileReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

/// Memory usage categories
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryCategory {
    /// VeryLow
    VeryLow, // < 10 MB
    /// Low
    Low, // 10-100 MB
    /// Medium
    Medium, // 100 MB - 1 GB
    /// High
    High, // 1-10 GB
    /// VeryHigh
    VeryHigh, // > 10 GB
}

impl fmt::Display for MemoryCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryCategory::VeryLow => write!(f, "Very Low (< 10 MB)"),
            MemoryCategory::Low => write!(f, "Low (10-100 MB)"),
            MemoryCategory::Medium => write!(f, "Medium (100 MB - 1 GB)"),
            MemoryCategory::High => write!(f, "High (1-10 GB)"),
            MemoryCategory::VeryHigh => write!(f, "Very High (> 10 GB)"),
        }
    }
}

/// Memory analysis utilities
pub struct MemoryAnalyzer;

impl MemoryAnalyzer {
    /// Estimate memory requirements for an algorithm
    pub fn estimate_memory_requirements(
        algorithm: &str,
        n_samples: usize,
        n_features: usize,
        n_components: usize,
    ) -> MemoryEstimate {
        let base_input = n_samples * n_features * 8; // f64 input data
        let base_output = n_samples * n_components * 8; // f64 output data

        let (algorithm_overhead, complexity_note) = match algorithm {
            "pca" => {
                let covariance = n_features * n_features * 8;
                let eigenvectors = n_features * n_components * 8;
                (covariance + eigenvectors, "O(d²) for covariance matrix")
            }
            "tsne" | "tsne_exact" => {
                let distance_matrix = n_samples * n_samples * 8;
                let gradients = n_samples * n_components * 8;
                (distance_matrix + gradients, "O(n²) for distance matrix")
            }
            "tsne_barnes_hut" => {
                let tree_overhead = n_samples * 100; // Approximate
                let gradients = n_samples * n_components * 8;
                (tree_overhead + gradients, "O(n) for spatial tree")
            }
            "isomap" => {
                let distance_matrix = n_samples * n_samples * 8;
                let shortest_paths = n_samples * n_samples * 8;
                let eigenvectors = n_samples * n_components * 8;
                (
                    distance_matrix + shortest_paths + eigenvectors,
                    "O(n²) for distance matrices",
                )
            }
            "lle" => {
                let neighbors = n_samples * 20 * 8; // Assume k=20 neighbors
                let weights = n_samples * n_samples * 8;
                let eigenvectors = n_samples * n_components * 8;
                (
                    neighbors + weights + eigenvectors,
                    "O(n²) for weight matrix",
                )
            }
            "umap" => {
                let graph = n_samples * 20 * 8 * 2; // Sparse graph with ~20 edges per node
                let embeddings = n_samples * n_components * 8 * 2; // Current + gradient
                (graph + embeddings, "O(n) for sparse graph")
            }
            "laplacian_eigenmaps" => {
                let affinity = n_samples * n_samples * 8;
                let laplacian = n_samples * n_samples * 8;
                let eigenvectors = n_samples * n_components * 8;
                (
                    affinity + laplacian + eigenvectors,
                    "O(n²) for affinity matrix",
                )
            }
            _ => {
                let buffer = n_samples * n_components * 8;
                (buffer, "O(nk) working memory")
            }
        };

        let total_bytes = base_input + base_output + algorithm_overhead;

        /// MemoryEstimate
        MemoryEstimate {
            algorithm: algorithm.to_string(),
            dataset_size: (n_samples, n_features, n_components),
            base_input_bytes: base_input,
            base_output_bytes: base_output,
            algorithm_overhead_bytes: algorithm_overhead,
            total_bytes,
            complexity_note: complexity_note.to_string(),
        }
    }

    /// Compare memory estimates for multiple algorithms
    pub fn compare_algorithms(
        algorithms: &[&str],
        n_samples: usize,
        n_features: usize,
        n_components: usize,
    ) -> Vec<MemoryEstimate> {
        algorithms
            .iter()
            .map(|&algo| {
                Self::estimate_memory_requirements(algo, n_samples, n_features, n_components)
            })
            .collect()
    }

    /// Find most memory-efficient algorithm
    pub fn most_efficient_algorithm(estimates: &[MemoryEstimate]) -> Option<&MemoryEstimate> {
        estimates.iter().min_by_key(|est| est.total_bytes)
    }

    /// Calculate array memory usage
    pub fn array_memory_usage<T>(array: &Array2<T>) -> usize {
        array.len() * std::mem::size_of::<T>()
    }

    /// Calculate vector memory usage
    pub fn vector_memory_usage<T>(vec: &[T]) -> usize {
        std::mem::size_of_val(vec)
    }

    /// Analyze memory usage pattern over time
    pub fn analyze_memory_pattern(checkpoints: &HashMap<String, MemoryStats>) -> MemoryPattern {
        if checkpoints.is_empty() {
            return MemoryPattern::Stable;
        }

        let mut peak_values: Vec<_> = checkpoints.values().map(|s| s.peak_bytes).collect();
        peak_values.sort_unstable();

        let min_peak = peak_values[0];
        let max_peak = peak_values[peak_values.len() - 1];

        if max_peak == 0 {
            return MemoryPattern::Stable;
        }

        let variation_ratio = (max_peak - min_peak) as f64 / max_peak as f64;

        if variation_ratio < 0.1 {
            MemoryPattern::Stable
        } else if variation_ratio < 0.3 {
            MemoryPattern::GradualIncrease
        } else if variation_ratio < 0.6 {
            MemoryPattern::Moderate
        } else {
            MemoryPattern::HighVariation
        }
    }
}

/// Memory usage estimate
#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    /// algorithm
    pub algorithm: String,
    /// dataset_size
    pub dataset_size: (usize, usize, usize), // (n_samples, n_features, n_components)
    /// base_input_bytes
    pub base_input_bytes: usize,
    /// base_output_bytes
    pub base_output_bytes: usize,
    /// algorithm_overhead_bytes
    pub algorithm_overhead_bytes: usize,
    /// total_bytes
    pub total_bytes: usize,
    /// complexity_note
    pub complexity_note: String,
}

impl MemoryEstimate {
    /// Get total memory in different units
    pub fn total_kb(&self) -> f64 {
        self.total_bytes as f64 / 1024.0
    }

    pub fn total_mb(&self) -> f64 {
        self.total_bytes as f64 / (1024.0 * 1024.0)
    }

    pub fn total_gb(&self) -> f64 {
        self.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Calculate overhead ratio
    pub fn overhead_ratio(&self) -> f64 {
        let base_total = self.base_input_bytes + self.base_output_bytes;
        if base_total > 0 {
            self.algorithm_overhead_bytes as f64 / base_total as f64
        } else {
            0.0
        }
    }

    /// Check if estimate is reasonable for available memory
    pub fn is_feasible(&self, available_bytes: usize) -> bool {
        self.total_bytes <= available_bytes
    }

    /// Generate estimate summary
    pub fn summary(&self) -> String {
        format!(
            "Memory Estimate for {}\n\
             Dataset: {}×{} → {}\n\
             Input Data: {:.2} MB\n\
             Output Data: {:.2} MB\n\
             Algorithm Overhead: {:.2} MB ({:.1}x)\n\
             Total Estimated: {:.2} MB\n\
             Complexity: {}",
            self.algorithm,
            self.dataset_size.0,
            self.dataset_size.1,
            self.dataset_size.2,
            self.base_input_bytes as f64 / (1024.0 * 1024.0),
            self.base_output_bytes as f64 / (1024.0 * 1024.0),
            self.algorithm_overhead_bytes as f64 / (1024.0 * 1024.0),
            self.overhead_ratio(),
            self.total_mb(),
            self.complexity_note
        )
    }
}

impl fmt::Display for MemoryEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

/// Memory usage patterns
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryPattern {
    /// Stable
    Stable, // Consistent memory usage
    /// GradualIncrease
    GradualIncrease, // Slowly increasing
    /// Moderate
    Moderate, // Moderate variation
    /// HighVariation
    HighVariation, // High variation/spikes
}

impl fmt::Display for MemoryPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryPattern::Stable => write!(f, "Stable"),
            MemoryPattern::GradualIncrease => write!(f, "Gradual Increase"),
            MemoryPattern::Moderate => write!(f, "Moderate Variation"),
            MemoryPattern::HighVariation => write!(f, "High Variation"),
        }
    }
}

/// Macro for automatic memory tracking of array operations
#[macro_export]
macro_rules! track_array_allocation {
    ($profiler:expr, $array:expr, $label:expr) => {{
        let size = $crate::memory_profiler::MemoryAnalyzer::array_memory_usage(&$array);
        $profiler.record_allocation(size);
        $profiler.checkpoint($label);
    }};
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_stats() {
        let mut stats = MemoryStats::new();
        assert_eq!(stats.current_bytes, 0);
        assert_eq!(stats.peak_bytes, 0);

        // Simulate allocation
        stats.current_bytes = 1024 * 1024; // 1 MB
        stats.peak_bytes = 1024 * 1024;
        stats.total_allocations = 1;
        stats.active_allocations = 1;

        assert_eq!(stats.current_mb(), 1.0);
        assert_eq!(stats.peak_mb(), 1.0);
        assert_eq!(stats.allocation_efficiency(), 1.0);
    }

    #[test]
    fn test_memory_profiler() {
        let dataset_info = DatasetInfo::new("test", 100, 10, 2);
        let mut profiler = MemoryProfiler::new("test_algo", dataset_info);

        profiler.record_allocation(1024);
        profiler.checkpoint("after_allocation");

        let stats = profiler.current_stats();
        assert_eq!(stats.current_bytes, 1024);
        assert_eq!(stats.total_allocations, 1);

        let checkpoint_stats = profiler.checkpoint_stats("after_allocation").unwrap();
        assert_eq!(checkpoint_stats.current_bytes, 1024);
    }

    #[test]
    fn test_memory_analyzer() {
        let estimate = MemoryAnalyzer::estimate_memory_requirements("pca", 100, 10, 2);

        assert_eq!(estimate.algorithm, "pca");
        assert_eq!(estimate.dataset_size, (100, 10, 2));
        assert!(estimate.total_bytes > 0);
        assert!(estimate.total_mb() > 0.0);
    }

    #[test]
    fn test_memory_estimate_comparison() {
        let algorithms = vec!["pca", "tsne", "umap"];
        let estimates = MemoryAnalyzer::compare_algorithms(&algorithms, 100, 10, 2);

        assert_eq!(estimates.len(), 3);

        let most_efficient = MemoryAnalyzer::most_efficient_algorithm(&estimates).unwrap();
        assert!(estimates
            .iter()
            .all(|est| est.total_bytes >= most_efficient.total_bytes));
    }

    #[test]
    fn test_array_memory_calculation() {
        let array = Array2::<f64>::zeros((100, 10));
        let memory_usage = MemoryAnalyzer::array_memory_usage(&array);

        // 100 * 10 * 8 bytes = 8000 bytes
        assert_eq!(memory_usage, 8000);
    }

    #[test]
    fn test_memory_pattern_analysis() {
        let mut checkpoints = HashMap::new();

        // Stable pattern
        checkpoints.insert(
            "step1".to_string(),
            /// MemoryStats
            MemoryStats {
                peak_bytes: 1000,
                ..MemoryStats::new()
            },
        );
        checkpoints.insert(
            "step2".to_string(),
            /// MemoryStats
            MemoryStats {
                peak_bytes: 1100,
                ..MemoryStats::new()
            },
        );
        checkpoints.insert(
            "step3".to_string(),
            /// MemoryStats
            MemoryStats {
                peak_bytes: 1050,
                ..MemoryStats::new()
            },
        );

        let pattern = MemoryAnalyzer::analyze_memory_pattern(&checkpoints);
        assert_eq!(pattern, MemoryPattern::Stable);
    }
}
