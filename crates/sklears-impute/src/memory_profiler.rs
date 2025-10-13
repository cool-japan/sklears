//! Memory profiling and monitoring for imputation operations
//!
//! This module provides tools for profiling memory usage during imputation
//! to help optimize performance and identify memory bottlenecks.

use std::alloc::{GlobalAlloc, Layout};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Global memory tracking statistics
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static TOTAL_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static PEAK_MEMORY: AtomicUsize = AtomicUsize::new(0);

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Current allocated bytes
    pub current_bytes: usize,
    /// Total number of allocations
    pub total_allocations: usize,
    /// Peak memory usage
    pub peak_bytes: usize,
    /// Memory usage over time
    pub timeline: Vec<(Instant, usize)>,
}

impl MemoryStats {
    /// Get current memory statistics
    pub fn current() -> Self {
        Self {
            current_bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
            total_allocations: TOTAL_ALLOCATIONS.load(Ordering::Relaxed),
            peak_bytes: PEAK_MEMORY.load(Ordering::Relaxed),
            timeline: Vec::new(),
        }
    }

    /// Reset memory statistics
    pub fn reset() {
        ALLOCATED_BYTES.store(0, Ordering::Relaxed);
        TOTAL_ALLOCATIONS.store(0, Ordering::Relaxed);
        PEAK_MEMORY.store(0, Ordering::Relaxed);
    }

    /// Format memory size in human-readable format
    pub fn format_bytes(bytes: usize) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = bytes as f64;
        let mut unit_idx = 0;

        while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
            size /= 1024.0;
            unit_idx += 1;
        }

        if unit_idx == 0 {
            format!("{} {}", bytes, UNITS[unit_idx])
        } else {
            format!("{:.2} {}", size, UNITS[unit_idx])
        }
    }

    /// Get memory usage summary
    pub fn summary(&self) -> String {
        format!(
            "Memory Stats:\n\
             Current: {}\n\
             Peak: {}\n\
             Total Allocations: {}",
            Self::format_bytes(self.current_bytes),
            Self::format_bytes(self.peak_bytes),
            self.total_allocations
        )
    }
}

/// Memory profiler for tracking memory usage during imputation
pub struct MemoryProfiler {
    start_time: Instant,
    timeline: Vec<(Instant, usize)>,
    sampling_interval: Duration,
    operation_name: String,
}

impl MemoryProfiler {
    /// Create a new memory profiler
    pub fn new(operation_name: impl Into<String>) -> Self {
        Self {
            start_time: Instant::now(),
            timeline: Vec::new(),
            sampling_interval: Duration::from_millis(100),
            operation_name: operation_name.into(),
        }
    }

    /// Set sampling interval for memory timeline
    pub fn with_sampling_interval(mut self, interval: Duration) -> Self {
        self.sampling_interval = interval;
        self
    }

    /// Start profiling
    pub fn start(&mut self) {
        self.start_time = Instant::now();
        self.timeline.clear();
        self.record_sample();
    }

    /// Record a memory usage sample
    pub fn record_sample(&mut self) {
        let current_memory = ALLOCATED_BYTES.load(Ordering::Relaxed);
        self.timeline.push((Instant::now(), current_memory));
    }

    /// Stop profiling and return results
    pub fn stop(mut self) -> MemoryProfilingResult {
        self.record_sample();

        let duration = self.start_time.elapsed();
        let peak_during_profiling = self
            .timeline
            .iter()
            .map(|(_, bytes)| *bytes)
            .max()
            .unwrap_or(0);

        let memory_over_time = self
            .timeline
            .into_iter()
            .map(|(time, bytes)| (time.duration_since(self.start_time), bytes))
            .collect();

        MemoryProfilingResult {
            operation_name: self.operation_name,
            duration,
            peak_memory: peak_during_profiling,
            memory_timeline: memory_over_time,
            final_stats: MemoryStats::current(),
        }
    }
}

/// Results of memory profiling
#[derive(Debug)]
pub struct MemoryProfilingResult {
    /// Name of the operation profiled
    pub operation_name: String,
    /// Total duration of the operation
    pub duration: Duration,
    /// Peak memory usage during the operation
    pub peak_memory: usize,
    /// Memory usage timeline
    pub memory_timeline: Vec<(Duration, usize)>,
    /// Final memory statistics
    pub final_stats: MemoryStats,
}

impl MemoryProfilingResult {
    /// Generate a memory usage report
    pub fn report(&self) -> String {
        let avg_memory = if !self.memory_timeline.is_empty() {
            self.memory_timeline
                .iter()
                .map(|(_, bytes)| *bytes)
                .sum::<usize>()
                / self.memory_timeline.len()
        } else {
            0
        };

        let memory_growth = if self.memory_timeline.len() >= 2 {
            let start_memory = self.memory_timeline[0].1;
            let end_memory = self.memory_timeline[self.memory_timeline.len() - 1].1;
            end_memory.saturating_sub(start_memory)
        } else {
            0
        };

        format!(
            "Memory Profile Report: {}\n\
             Duration: {:.2}s\n\
             Peak Memory: {}\n\
             Average Memory: {}\n\
             Memory Growth: {}\n\
             Samples: {}\n\
             {}",
            self.operation_name,
            self.duration.as_secs_f64(),
            MemoryStats::format_bytes(self.peak_memory),
            MemoryStats::format_bytes(avg_memory),
            MemoryStats::format_bytes(memory_growth),
            self.memory_timeline.len(),
            self.final_stats.summary()
        )
    }

    /// Export timeline data as CSV
    pub fn export_csv(&self) -> String {
        let mut csv = String::from("timestamp_ms,memory_bytes\n");
        for (duration, bytes) in &self.memory_timeline {
            csv.push_str(&format!("{},{}\n", duration.as_millis(), bytes));
        }
        csv
    }

    /// Check for potential memory leaks
    pub fn check_memory_leaks(&self) -> Option<String> {
        if self.memory_timeline.len() < 2 {
            return None;
        }

        let start_memory = self.memory_timeline[0].1;
        let end_memory = self.memory_timeline[self.memory_timeline.len() - 1].1;
        let growth = end_memory.saturating_sub(start_memory);

        // If memory grew by more than 10MB, flag as potential leak
        if growth > 10 * 1024 * 1024 {
            Some(format!(
                "Potential memory leak detected: {} growth during {}",
                MemoryStats::format_bytes(growth),
                self.operation_name
            ))
        } else {
            None
        }
    }
}

/// Convenience macro for profiling memory usage
#[macro_export]
macro_rules! profile_memory {
    ($operation_name:expr, $code:block) => {{
        let mut profiler = $crate::memory_profiler::MemoryProfiler::new($operation_name);
        profiler.start();
        let result = $code;
        let profile_result = profiler.stop();
        (result, profile_result)
    }};
}

/// Track memory allocations (would need to be enabled as global allocator)
pub struct TrackingAllocator<A: GlobalAlloc> {
    inner: A,
}

impl<A: GlobalAlloc> TrackingAllocator<A> {
    pub const fn new(inner: A) -> Self {
        Self { inner }
    }
}

unsafe impl<A: GlobalAlloc> GlobalAlloc for TrackingAllocator<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = self.inner.alloc(layout);
        if !ptr.is_null() {
            let size = layout.size();
            let current = ALLOCATED_BYTES.fetch_add(size, Ordering::Relaxed) + size;
            TOTAL_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);

            // Update peak memory
            let mut peak = PEAK_MEMORY.load(Ordering::Relaxed);
            while peak < current {
                match PEAK_MEMORY.compare_exchange_weak(
                    peak,
                    current,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(x) => peak = x,
                }
            }
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        ALLOCATED_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        self.inner.dealloc(ptr, layout);
    }
}

/// Benchmark memory usage of different imputation methods
pub struct ImputationMemoryBenchmark {
    results: HashMap<String, MemoryProfilingResult>,
}

impl ImputationMemoryBenchmark {
    /// Create new benchmark
    pub fn new() -> Self {
        Self {
            results: HashMap::new(),
        }
    }

    /// Add a profiling result
    pub fn add_result(&mut self, name: String, result: MemoryProfilingResult) {
        self.results.insert(name, result);
    }

    /// Generate comparison report
    pub fn comparison_report(&self) -> String {
        let mut report = String::from("Memory Usage Comparison:\n");
        report.push_str("Method\tPeak Memory\tAvg Memory\tDuration\tGrowth\n");

        for (name, result) in &self.results {
            let avg_memory = if !result.memory_timeline.is_empty() {
                result
                    .memory_timeline
                    .iter()
                    .map(|(_, bytes)| *bytes)
                    .sum::<usize>()
                    / result.memory_timeline.len()
            } else {
                0
            };

            let growth = if result.memory_timeline.len() >= 2 {
                let start = result.memory_timeline[0].1;
                let end = result.memory_timeline[result.memory_timeline.len() - 1].1;
                end.saturating_sub(start)
            } else {
                0
            };

            report.push_str(&format!(
                "{}\t{}\t{}\t{:.2}s\t{}\n",
                name,
                MemoryStats::format_bytes(result.peak_memory),
                MemoryStats::format_bytes(avg_memory),
                result.duration.as_secs_f64(),
                MemoryStats::format_bytes(growth),
            ));
        }

        report
    }

    /// Find the most memory-efficient method
    pub fn most_efficient(&self) -> Option<(&String, &MemoryProfilingResult)> {
        self.results
            .iter()
            .min_by_key(|(_, result)| result.peak_memory)
    }
}

impl Default for ImputationMemoryBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_memory_stats_formatting() {
        assert_eq!(MemoryStats::format_bytes(500), "500 B");
        assert_eq!(MemoryStats::format_bytes(1536), "1.50 KB");
        assert_eq!(MemoryStats::format_bytes(1048576), "1.00 MB");
        assert_eq!(MemoryStats::format_bytes(2147483648), "2.00 GB");
    }

    #[test]
    fn test_memory_profiler() {
        let mut profiler = MemoryProfiler::new("test_operation");
        profiler.start();

        // Simulate some work
        thread::sleep(Duration::from_millis(10));
        profiler.record_sample();

        let result = profiler.stop();

        assert_eq!(result.operation_name, "test_operation");
        assert!(result.duration.as_millis() >= 10);
        assert!(!result.memory_timeline.is_empty());
    }

    #[test]
    fn test_profiling_result_report() {
        let result = MemoryProfilingResult {
            operation_name: "test".to_string(),
            duration: Duration::from_secs(1),
            peak_memory: 1024,
            memory_timeline: vec![
                (Duration::from_millis(0), 512),
                (Duration::from_millis(500), 1024),
                (Duration::from_millis(1000), 768),
            ],
            final_stats: MemoryStats::current(),
        };

        let report = result.report();
        assert!(report.contains("test"));
        assert!(report.contains("1.00s"));
    }

    #[test]
    fn test_csv_export() {
        let result = MemoryProfilingResult {
            operation_name: "test".to_string(),
            duration: Duration::from_secs(1),
            peak_memory: 1024,
            memory_timeline: vec![
                (Duration::from_millis(0), 512),
                (Duration::from_millis(1000), 1024),
            ],
            final_stats: MemoryStats::current(),
        };

        let csv = result.export_csv();
        assert!(csv.contains("timestamp_ms,memory_bytes"));
        assert!(csv.contains("0,512"));
        assert!(csv.contains("1000,1024"));
    }

    #[test]
    fn test_memory_leak_detection() {
        let result = MemoryProfilingResult {
            operation_name: "test".to_string(),
            duration: Duration::from_secs(1),
            peak_memory: 20 * 1024 * 1024, // 20MB
            memory_timeline: vec![
                (Duration::from_millis(0), 1024),
                (Duration::from_millis(1000), 20 * 1024 * 1024), // Growth of ~20MB
            ],
            final_stats: MemoryStats::current(),
        };

        let leak_check = result.check_memory_leaks();
        assert!(leak_check.is_some());
        assert!(leak_check.unwrap().contains("Potential memory leak"));
    }
}
