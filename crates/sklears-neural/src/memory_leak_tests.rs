//! Memory leak detection tests for neural network training and inference.
//!
//! This module provides utilities to detect and test for memory leaks in
//! neural network operations, including training loops, batch processing,
//! and model inference. It monitors memory usage patterns and identifies
//! potential memory leaks.

use crate::activation::Activation;
use crate::mlp_classifier::MLPClassifier;
use crate::mlp_regressor::MLPRegressor;
use crate::solvers::Solver;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::traits::{Fit, Predict};
use std::time::{Duration, Instant};

/// Memory usage snapshot
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    /// Virtual memory size in bytes
    pub virtual_memory: u64,
    /// Resident set size in bytes
    pub resident_memory: u64,
    /// Timestamp when snapshot was taken
    pub timestamp: Instant,
    /// Optional label for the snapshot
    pub label: Option<String>,
}

impl MemorySnapshot {
    /// Take a new memory snapshot
    pub fn take() -> Self {
        Self::take_with_label(None)
    }

    /// Take a memory snapshot with a label
    pub fn take_with_label(label: Option<String>) -> Self {
        let (virtual_memory, resident_memory) = get_memory_usage();
        Self {
            virtual_memory,
            resident_memory,
            timestamp: Instant::now(),
            label,
        }
    }

    /// Calculate memory difference from another snapshot
    pub fn diff_from(&self, other: &MemorySnapshot) -> MemoryDiff {
        MemoryDiff {
            virtual_memory_delta: self.virtual_memory as i64 - other.virtual_memory as i64,
            resident_memory_delta: self.resident_memory as i64 - other.resident_memory as i64,
            duration: self.timestamp.duration_since(other.timestamp),
        }
    }
}

/// Memory usage difference between snapshots
#[derive(Debug, Clone)]
pub struct MemoryDiff {
    /// Change in virtual memory (bytes)
    pub virtual_memory_delta: i64,
    /// Change in resident memory (bytes)
    pub resident_memory_delta: i64,
    /// Duration between snapshots
    pub duration: Duration,
}

impl MemoryDiff {
    /// Check if this represents a potential memory leak
    pub fn is_potential_leak(&self, threshold_bytes: u64) -> bool {
        self.virtual_memory_delta > threshold_bytes as i64
            || self.resident_memory_delta > threshold_bytes as i64
    }

    /// Get memory growth rate in bytes per second
    pub fn memory_growth_rate(&self) -> f64 {
        let duration_secs = self.duration.as_secs_f64();
        if duration_secs > 0.0 {
            self.resident_memory_delta as f64 / duration_secs
        } else {
            0.0
        }
    }
}

/// Memory leak detector for tracking memory usage over time
pub struct MemoryLeakDetector {
    /// History of memory snapshots
    snapshots: Vec<MemorySnapshot>,
    /// Threshold for leak detection (bytes)
    leak_threshold: u64,
    /// Maximum number of snapshots to keep
    max_snapshots: usize,
}

impl MemoryLeakDetector {
    /// Create a new memory leak detector
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
            leak_threshold: 10_000_000, // 10MB default threshold
            max_snapshots: 1000,
        }
    }

    /// Set the leak detection threshold in bytes
    pub fn with_threshold(mut self, threshold: u64) -> Self {
        self.leak_threshold = threshold;
        self
    }

    /// Set maximum number of snapshots to keep
    pub fn with_max_snapshots(mut self, max: usize) -> Self {
        self.max_snapshots = max;
        self
    }

    /// Take a new memory snapshot and add to history
    pub fn snapshot(&mut self) -> &MemorySnapshot {
        self.snapshot_with_label(None)
    }

    /// Take a labeled memory snapshot
    pub fn snapshot_with_label(&mut self, label: Option<String>) -> &MemorySnapshot {
        let snapshot = MemorySnapshot::take_with_label(label);
        self.snapshots.push(snapshot);

        // Keep only the most recent snapshots
        if self.snapshots.len() > self.max_snapshots {
            self.snapshots.remove(0);
        }

        self.snapshots.last().unwrap()
    }

    /// Get all memory snapshots
    pub fn get_snapshots(&self) -> &[MemorySnapshot] {
        &self.snapshots
    }

    /// Check for potential memory leaks between consecutive snapshots
    pub fn detect_leaks(&self) -> Vec<(usize, MemoryDiff)> {
        let mut leaks = Vec::new();

        for i in 1..self.snapshots.len() {
            let diff = self.snapshots[i].diff_from(&self.snapshots[i - 1]);
            if diff.is_potential_leak(self.leak_threshold) {
                leaks.push((i, diff));
            }
        }

        leaks
    }

    /// Get memory usage statistics
    pub fn get_statistics(&self) -> MemoryStats {
        if self.snapshots.is_empty() {
            return MemoryStats::default();
        }

        let virtual_memories: Vec<u64> = self.snapshots.iter().map(|s| s.virtual_memory).collect();
        let resident_memories: Vec<u64> =
            self.snapshots.iter().map(|s| s.resident_memory).collect();

        let min_virtual = *virtual_memories.iter().min().unwrap();
        let max_virtual = *virtual_memories.iter().max().unwrap();
        let avg_virtual =
            virtual_memories.iter().sum::<u64>() as f64 / virtual_memories.len() as f64;

        let min_resident = *resident_memories.iter().min().unwrap();
        let max_resident = *resident_memories.iter().max().unwrap();
        let avg_resident =
            resident_memories.iter().sum::<u64>() as f64 / resident_memories.len() as f64;

        MemoryStats {
            virtual_memory_min: min_virtual,
            virtual_memory_max: max_virtual,
            virtual_memory_avg: avg_virtual,
            resident_memory_min: min_resident,
            resident_memory_max: max_resident,
            resident_memory_avg: avg_resident,
            total_snapshots: self.snapshots.len(),
            potential_leaks: self.detect_leaks().len(),
        }
    }

    /// Clear all snapshots
    pub fn clear(&mut self) {
        self.snapshots.clear();
    }

    /// Get the current leak threshold
    pub fn get_leak_threshold(&self) -> u64 {
        self.leak_threshold
    }
}

impl Default for MemoryLeakDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub virtual_memory_min: u64,
    pub virtual_memory_max: u64,
    pub virtual_memory_avg: f64,
    pub resident_memory_min: u64,
    pub resident_memory_max: u64,
    pub resident_memory_avg: f64,
    pub total_snapshots: usize,
    pub potential_leaks: usize,
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self {
            virtual_memory_min: 0,
            virtual_memory_max: 0,
            virtual_memory_avg: 0.0,
            resident_memory_min: 0,
            resident_memory_max: 0,
            resident_memory_avg: 0.0,
            total_snapshots: 0,
            potential_leaks: 0,
        }
    }
}

/// Get current memory usage (platform-specific implementation)
#[cfg(target_os = "linux")]
fn get_memory_usage() -> (u64, u64) {
    use std::fs;

    if let Ok(contents) = fs::read_to_string("/proc/self/status") {
        let mut vm_size = 0u64;
        let mut vm_rss = 0u64;

        for line in contents.lines() {
            if line.starts_with("VmSize:") {
                if let Some(size_str) = line.split_whitespace().nth(1) {
                    vm_size = size_str.parse::<u64>().unwrap_or(0) * 1024; // Convert kB to bytes
                }
            } else if line.starts_with("VmRSS:") {
                if let Some(rss_str) = line.split_whitespace().nth(1) {
                    vm_rss = rss_str.parse::<u64>().unwrap_or(0) * 1024; // Convert kB to bytes
                }
            }
        }

        (vm_size, vm_rss)
    } else {
        (0, 0)
    }
}

/// Get current memory usage (macOS implementation)
#[cfg(target_os = "macos")]
fn get_memory_usage() -> (u64, u64) {
    use std::process::Command;

    // Use ps command to get memory info
    if let Ok(output) = Command::new("ps")
        .args(&["-o", "vsz,rss", "-p"])
        .arg(std::process::id().to_string())
        .output()
    {
        if let Ok(output_str) = String::from_utf8(output.stdout) {
            let lines: Vec<&str> = output_str.trim().lines().collect();
            if lines.len() >= 2 {
                let values: Vec<&str> = lines[1].split_whitespace().collect();
                if values.len() >= 2 {
                    let vsz = values[0].parse::<u64>().unwrap_or(0) * 1024; // Convert kB to bytes
                    let rss = values[1].parse::<u64>().unwrap_or(0) * 1024; // Convert kB to bytes
                    return (vsz, rss);
                }
            }
        }
    }

    (0, 0)
}

/// Fallback implementation for other platforms
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn get_memory_usage() -> (u64, u64) {
    // Return zeros for unsupported platforms
    (0, 0)
}

/// Memory leak test suite for neural networks
pub struct MemoryLeakTestSuite;

impl MemoryLeakTestSuite {
    /// Test for memory leaks during MLP classifier training
    pub fn test_mlp_classifier_training() -> Result<MemoryStats, Box<dyn std::error::Error>> {
        let mut detector = MemoryLeakDetector::new().with_threshold(5_000_000); // 5MB threshold

        // Generate test data
        let n_samples = 1000;
        let n_features = 20;
        let mut x = Array2::zeros((n_samples, n_features));
        let mut y = vec![0; n_samples];

        // Simple binary classification data
        for i in 0..n_samples {
            for j in 0..n_features {
                x[[i, j]] = if j % 2 == 0 { 1.0 } else { -1.0 };
            }
            y[i] = if i % 2 == 0 { 0 } else { 1 };
        }

        detector.snapshot_with_label(Some("Initial".to_string()));

        // Test multiple training runs to detect leaks
        for iteration in 0..10 {
            let classifier = MLPClassifier::new()
                .hidden_layer_sizes(&[50, 30])
                .activation(Activation::Relu)
                .solver(Solver::Adam)
                .learning_rate_init(0.001)
                .max_iter(50)
                .random_state(42);

            let _trained = classifier.fit(&x, &y)?;

            detector.snapshot_with_label(Some(format!("Training iteration {}", iteration + 1)));

            // Force garbage collection if available
            #[cfg(feature = "force_gc")]
            {
                std::gc::collect();
            }

            // Small delay to allow cleanup
            std::thread::sleep(Duration::from_millis(100));
        }

        detector.snapshot_with_label(Some("Final".to_string()));

        let stats = detector.get_statistics();
        let leaks = detector.detect_leaks();

        if !leaks.is_empty() {
            println!("Potential memory leaks detected in MLP classifier training:");
            for (idx, diff) in &leaks {
                println!(
                    "  Snapshot {}: +{} bytes virtual, +{} bytes resident ({:.2} bytes/sec)",
                    idx,
                    diff.virtual_memory_delta,
                    diff.resident_memory_delta,
                    diff.memory_growth_rate()
                );
            }
        }

        Ok(stats)
    }

    /// Test for memory leaks during MLP regressor training
    pub fn test_mlp_regressor_training() -> Result<MemoryStats, Box<dyn std::error::Error>> {
        let mut detector = MemoryLeakDetector::new().with_threshold(5_000_000);

        // Generate test data
        let n_samples = 1000;
        let n_features = 20;
        let mut x = Array2::zeros((n_samples, n_features));
        let mut y = Array2::zeros((n_samples, 1));

        // Simple regression data
        for i in 0..n_samples {
            for j in 0..n_features {
                x[[i, j]] = (i as f64) / 100.0 + (j as f64) * 0.1;
            }
            y[[i, 0]] = x.row(i).sum();
        }

        detector.snapshot_with_label(Some("Initial".to_string()));

        // Test multiple training runs
        for iteration in 0..10 {
            let regressor = MLPRegressor::new()
                .hidden_layer_sizes(&[50, 30])
                .activation(Activation::Relu)
                .solver(Solver::Adam)
                .learning_rate_init(0.001)
                .max_iter(50)
                .random_state(42);

            let _trained = regressor.fit(&x, &y)?;

            detector.snapshot_with_label(Some(format!("Training iteration {}", iteration + 1)));

            std::thread::sleep(Duration::from_millis(100));
        }

        detector.snapshot_with_label(Some("Final".to_string()));
        Ok(detector.get_statistics())
    }

    /// Test for memory leaks during batch prediction
    pub fn test_batch_prediction() -> Result<MemoryStats, Box<dyn std::error::Error>> {
        let mut detector = MemoryLeakDetector::new().with_threshold(2_000_000);

        // Create a trained classifier
        let n_samples = 500;
        let n_features = 10;
        let mut x_train = Array2::zeros((n_samples, n_features));
        let mut y_train = vec![0; n_samples];

        for i in 0..n_samples {
            for j in 0..n_features {
                x_train[[i, j]] = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
            }
            y_train[i] = if i % 2 == 0 { 0 } else { 1 };
        }

        let classifier = MLPClassifier::new()
            .hidden_layer_sizes(&[20, 10])
            .max_iter(10)
            .random_state(42);

        let trained = classifier.fit(&x_train, &y_train)?;

        detector.snapshot_with_label(Some("After training".to_string()));

        // Test multiple batch predictions
        let batch_size = 1000;
        for batch in 0..20 {
            let mut x_batch = Array2::zeros((batch_size, n_features));
            for i in 0..batch_size {
                for j in 0..n_features {
                    x_batch[[i, j]] = ((batch * batch_size + i) as f64) * 0.01;
                }
            }

            let _predictions = trained.predict(&x_batch)?;

            detector.snapshot_with_label(Some(format!("Batch prediction {}", batch + 1)));

            std::thread::sleep(Duration::from_millis(50));
        }

        detector.snapshot_with_label(Some("Final".to_string()));
        Ok(detector.get_statistics())
    }

    /// Run all memory leak tests
    pub fn run_all_tests() -> Result<(), Box<dyn std::error::Error>> {
        println!("Running memory leak detection tests...\n");

        println!("1. Testing MLP classifier training for memory leaks:");
        let classifier_stats = Self::test_mlp_classifier_training()?;
        println!(
            "   Memory usage: {:.2} MB avg virtual, {:.2} MB avg resident",
            classifier_stats.virtual_memory_avg / 1_000_000.0,
            classifier_stats.resident_memory_avg / 1_000_000.0
        );
        if classifier_stats.potential_leaks > 0 {
            println!(
                "   ⚠️  {} potential leaks detected!",
                classifier_stats.potential_leaks
            );
        } else {
            println!("   ✅ No memory leaks detected");
        }

        println!("\n2. Testing MLP regressor training for memory leaks:");
        let regressor_stats = Self::test_mlp_regressor_training()?;
        println!(
            "   Memory usage: {:.2} MB avg virtual, {:.2} MB avg resident",
            regressor_stats.virtual_memory_avg / 1_000_000.0,
            regressor_stats.resident_memory_avg / 1_000_000.0
        );
        if regressor_stats.potential_leaks > 0 {
            println!(
                "   ⚠️  {} potential leaks detected!",
                regressor_stats.potential_leaks
            );
        } else {
            println!("   ✅ No memory leaks detected");
        }

        println!("\n3. Testing batch prediction for memory leaks:");
        let prediction_stats = Self::test_batch_prediction()?;
        println!(
            "   Memory usage: {:.2} MB avg virtual, {:.2} MB avg resident",
            prediction_stats.virtual_memory_avg / 1_000_000.0,
            prediction_stats.resident_memory_avg / 1_000_000.0
        );
        if prediction_stats.potential_leaks > 0 {
            println!(
                "   ⚠️  {} potential leaks detected!",
                prediction_stats.potential_leaks
            );
        } else {
            println!("   ✅ No memory leaks detected");
        }

        println!("\nMemory leak detection tests completed.");
        Ok(())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_snapshot() {
        let snapshot1 = MemorySnapshot::take();
        std::thread::sleep(Duration::from_millis(10));
        let snapshot2 = MemorySnapshot::take();

        let diff = snapshot2.diff_from(&snapshot1);
        assert!(diff.duration > Duration::from_millis(5));
    }

    #[test]
    fn test_memory_leak_detector() {
        let mut detector = MemoryLeakDetector::new().with_threshold(1_000_000);

        detector.snapshot_with_label(Some("Start".to_string()));
        detector.snapshot_with_label(Some("End".to_string()));

        let stats = detector.get_statistics();
        assert_eq!(stats.total_snapshots, 2);
    }

    #[test]
    fn test_memory_diff() {
        let snapshot1 = MemorySnapshot {
            virtual_memory: 1_000_000,
            resident_memory: 500_000,
            timestamp: Instant::now(),
            label: None,
        };

        std::thread::sleep(Duration::from_millis(10));

        let snapshot2 = MemorySnapshot {
            virtual_memory: 1_100_000,
            resident_memory: 550_000,
            timestamp: Instant::now(),
            label: None,
        };

        let diff = snapshot2.diff_from(&snapshot1);
        assert_eq!(diff.virtual_memory_delta, 100_000);
        assert_eq!(diff.resident_memory_delta, 50_000);
        assert!(diff.duration > Duration::from_millis(5));
    }

    #[test]
    fn test_leak_detection() {
        let diff = MemoryDiff {
            virtual_memory_delta: 15_000_000, // 15MB
            resident_memory_delta: 5_000_000, // 5MB
            duration: Duration::from_secs(1),
        };

        assert!(diff.is_potential_leak(10_000_000)); // 10MB threshold
        assert!(!diff.is_potential_leak(20_000_000)); // 20MB threshold

        let growth_rate = diff.memory_growth_rate();
        assert!((growth_rate - 5_000_000.0).abs() < 1.0); // ~5MB/sec
    }

    #[test]
    fn test_memory_usage_function() {
        let (virtual_mem, resident_mem) = get_memory_usage();
        // On supported platforms, we should get some memory usage
        // On unsupported platforms, both will be 0
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            assert!(virtual_mem > 0 || resident_mem > 0); // At least one should be non-zero
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            assert_eq!(virtual_mem, 0);
            assert_eq!(resident_mem, 0);
        }
    }
}
