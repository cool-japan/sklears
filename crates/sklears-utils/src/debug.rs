//! Debug utilities for development and troubleshooting
//!
//! This module provides debugging helper functions, assertion macros,
//! test data generation, and diagnostic utilities for ML development.

use crate::{UtilsError, UtilsResult};
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::time::{Duration, Instant};

/// Debug context for collecting debugging information
#[derive(Debug, Clone)]
pub struct DebugContext {
    pub module: String,
    pub function: String,
    pub line: u32,
    pub timestamp: Instant,
    pub metadata: HashMap<String, String>,
}

impl DebugContext {
    /// Create a new debug context
    pub fn new(module: &str, function: &str, line: u32) -> Self {
        Self {
            module: module.to_string(),
            function: function.to_string(),
            line,
            timestamp: Instant::now(),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the debug context
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Format the debug context as a string
    pub fn format(&self) -> String {
        let mut result = format!("{}::{}:{}", self.module, self.function, self.line);
        if !self.metadata.is_empty() {
            result.push_str(" {");
            let metadata_strs: Vec<String> = self
                .metadata
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect();
            result.push_str(&metadata_strs.join(", "));
            result.push('}');
        }
        result
    }
}

/// Macro for creating debug contexts easily
#[macro_export]
macro_rules! debug_context {
    () => {
        DebugContext::new(module_path!(), function_name!(), line!())
    };
    ($($key:expr => $value:expr),*) => {
        DebugContext::new(module_path!(), function_name!(), line!())
            $(.with_metadata($key, &$value.to_string()))*
    };
}

/// Enhanced assertion macro with debugging information
#[macro_export]
macro_rules! debug_assert_msg {
    ($condition:expr, $msg:expr) => {
        if !$condition {
            panic!(
                "Assertion failed at {}:{}:{}: {}\nCondition: {}",
                module_path!(),
                file!(),
                line!(),
                $msg,
                stringify!($condition)
            );
        }
    };
    ($condition:expr, $msg:expr, $($arg:tt)*) => {
        if !$condition {
            panic!(
                "Assertion failed at {}:{}:{}: {}\nCondition: {}",
                module_path!(),
                file!(),
                line!(),
                format!($msg, $($arg)*),
                stringify!($condition)
            );
        }
    };
}

/// Array debugging utilities
pub struct ArrayDebugger;

impl ArrayDebugger {
    /// Debug array statistics
    pub fn array_stats<T: fmt::Debug + Clone>(array: &[T]) -> String {
        format!(
            "Array[len={}, type={}]",
            array.len(),
            std::any::type_name::<T>()
        )
    }

    /// Debug array shape for multidimensional arrays
    pub fn array_shape_info(shape: &[usize]) -> String {
        format!(
            "Shape: {:?}, Total elements: {}",
            shape,
            shape.iter().product::<usize>()
        )
    }

    /// Find array differences for debugging
    pub fn compare_arrays<T: PartialEq + fmt::Debug>(a: &[T], b: &[T]) -> Vec<String> {
        let mut differences = Vec::new();

        if a.len() != b.len() {
            differences.push(format!("Length mismatch: {} vs {}", b.len(), a.len()));
        }

        let min_len = a.len().min(b.len());
        for i in 0..min_len {
            if a[i] != b[i] {
                differences.push(format!("Element {} differs: {:?} vs {:?}", i, a[i], b[i]));
            }
        }

        differences
    }

    /// Check for NaN and infinite values in float arrays
    pub fn check_float_array(array: &[f64]) -> Vec<String> {
        let mut issues = Vec::new();

        for (i, &value) in array.iter().enumerate() {
            if value.is_nan() {
                issues.push(format!("NaN at index {i}"));
            } else if value.is_infinite() {
                issues.push(format!("Infinite value at index {i}: {value}"));
            }
        }

        issues
    }
}

/// Memory debugging utilities
pub struct MemoryDebugger;

impl MemoryDebugger {
    /// Get memory usage information (simplified version)
    pub fn memory_info() -> String {
        // Note: In a real implementation, you'd use system calls or external crates
        // for actual memory information. This is a placeholder.
        "Memory debugging not fully implemented in this version".to_string()
    }

    /// Debug heap allocations (placeholder)
    pub fn heap_info() -> String {
        "Heap debugging requires external dependencies".to_string()
    }

    /// Check for potential memory leaks (placeholder)
    pub fn leak_check() -> Vec<String> {
        vec!["Memory leak detection requires runtime support".to_string()]
    }
}

/// Performance debugging utilities
#[derive(Debug, Clone)]
pub struct PerformanceDebugger {
    timers: HashMap<String, Instant>,
    durations: HashMap<String, Vec<Duration>>,
}

impl Default for PerformanceDebugger {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceDebugger {
    /// Create a new performance debugger
    pub fn new() -> Self {
        Self {
            timers: HashMap::new(),
            durations: HashMap::new(),
        }
    }

    /// Start timing an operation
    pub fn start_timer(&mut self, name: &str) {
        self.timers.insert(name.to_string(), Instant::now());
    }

    /// Stop timing and record duration
    pub fn stop_timer(&mut self, name: &str) -> Option<Duration> {
        if let Some(start) = self.timers.remove(name) {
            let duration = start.elapsed();
            self.durations
                .entry(name.to_string())
                .or_default()
                .push(duration);
            Some(duration)
        } else {
            None
        }
    }

    /// Get timing statistics
    pub fn timing_stats(&self, name: &str) -> Option<TimingStats> {
        self.durations.get(name).map(|durations| {
            let count = durations.len();
            let total: Duration = durations.iter().sum();
            let avg = total / count as u32;
            let min = *durations.iter().min().unwrap();
            let max = *durations.iter().max().unwrap();

            TimingStats {
                name: name.to_string(),
                count,
                total,
                average: avg,
                min,
                max,
            }
        })
    }

    /// Get all timing statistics
    pub fn all_stats(&self) -> Vec<TimingStats> {
        self.durations
            .keys()
            .filter_map(|name| self.timing_stats(name))
            .collect()
    }

    /// Format timing report
    pub fn timing_report(&self) -> String {
        let mut report = String::from("Performance Report:\n");

        for stats in self.all_stats() {
            report.push_str(&format!(
                "  {}: {} calls, avg: {:?}, min: {:?}, max: {:?}, total: {:?}\n",
                stats.name, stats.count, stats.average, stats.min, stats.max, stats.total
            ));
        }

        report
    }
}

/// Timing statistics for performance analysis
#[derive(Debug, Clone)]
pub struct TimingStats {
    pub name: String,
    pub count: usize,
    pub total: Duration,
    pub average: Duration,
    pub min: Duration,
    pub max: Duration,
}

/// Test data generation utilities
pub struct TestDataGenerator;

impl TestDataGenerator {
    /// Generate random integers in a range
    pub fn random_integers(count: usize, min: i32, max: i32) -> Vec<i32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut result = Vec::with_capacity(count);
        let range = (max - min + 1) as u64;

        for i in 0..count {
            let mut hasher = DefaultHasher::new();
            (i as u64 + 12345).hash(&mut hasher); // Add seed for better distribution
            let hash_value = hasher.finish();
            let value = min + ((hash_value % range) as i32);
            result.push(value);
        }

        result
    }

    /// Generate random floats in a range
    pub fn random_floats(count: usize, min: f64, max: f64) -> Vec<f64> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut result = Vec::with_capacity(count);

        for i in 0..count {
            let mut hasher = DefaultHasher::new();
            (i as u64 + 54321).hash(&mut hasher); // Add different seed
            let hash_value = hasher.finish();
            let normalized = (hash_value as f64) / (u64::MAX as f64);
            let value = min + normalized * (max - min);
            result.push(value);
        }

        result
    }

    /// Generate test matrix data
    pub fn test_matrix(rows: usize, cols: usize) -> Vec<Vec<f64>> {
        let mut matrix = Vec::with_capacity(rows);

        for i in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for j in 0..cols {
                let value = (i * cols + j) as f64 / (rows * cols) as f64;
                row.push(value);
            }
            matrix.push(row);
        }

        matrix
    }

    /// Generate test strings
    pub fn test_strings(count: usize) -> Vec<String> {
        let prefixes = ["test", "data", "sample", "debug", "item"];
        let mut result = Vec::with_capacity(count);

        for i in 0..count {
            let prefix = prefixes[i % prefixes.len()];
            result.push(format!("{prefix}_{i}"));
        }

        result
    }

    /// Generate pathological test cases
    pub fn pathological_cases() -> HashMap<String, Vec<f64>> {
        let mut cases = HashMap::new();

        cases.insert("empty".to_string(), vec![]);
        cases.insert("single".to_string(), vec![1.0]);
        cases.insert("zeros".to_string(), vec![0.0; 10]);
        cases.insert("ones".to_string(), vec![1.0; 10]);
        cases.insert("alternating".to_string(), vec![1.0, -1.0, 1.0, -1.0, 1.0]);
        cases.insert("large_values".to_string(), vec![1e10, 1e11, 1e12]);
        cases.insert("small_values".to_string(), vec![1e-10, 1e-11, 1e-12]);
        cases.insert(
            "mixed_signs".to_string(),
            vec![-100.0, -1.0, 0.0, 1.0, 100.0],
        );

        // Add special float values
        cases.insert(
            "special_floats".to_string(),
            vec![
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NAN,
                f64::MIN,
                f64::MAX,
                f64::EPSILON,
            ],
        );

        cases
    }
}

/// Diagnostic utilities for troubleshooting
pub struct DiagnosticTools;

impl DiagnosticTools {
    /// Run basic system diagnostics
    pub fn system_check() -> Vec<String> {
        let mut checks = Vec::new();

        // Check basic Rust environment
        checks.push(format!("Target architecture: {}", std::env::consts::ARCH));
        checks.push(format!("Operating system: {}", std::env::consts::OS));
        checks.push(format!(
            "Profile: {}",
            if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            }
        ));

        // Check available features
        checks.push(format!(
            "Float support: {}",
            if f64::INFINITY.is_infinite() {
                "OK"
            } else {
                "FAIL"
            }
        ));
        checks.push(format!(
            "Threading: {}",
            if std::thread::available_parallelism().is_ok() {
                "OK"
            } else {
                "LIMITED"
            }
        ));

        checks
    }

    /// Validate algorithm inputs
    pub fn validate_ml_inputs(
        features: &[Vec<f64>],
        targets: Option<&[f64]>,
    ) -> UtilsResult<Vec<String>> {
        let mut warnings = Vec::new();

        if features.is_empty() {
            return Err(UtilsError::InvalidParameter(
                "Empty feature matrix".to_string(),
            ));
        }

        let feature_count = features[0].len();

        // Check feature consistency
        for (i, row) in features.iter().enumerate() {
            if row.len() != feature_count {
                warnings.push(format!(
                    "Row {} has {} features, expected {}",
                    i,
                    row.len(),
                    feature_count
                ));
            }

            // Check for NaN/infinite values
            for (j, &value) in row.iter().enumerate() {
                if value.is_nan() {
                    warnings.push(format!("NaN found at row {i}, col {j}"));
                } else if value.is_infinite() {
                    warnings.push(format!("Infinite value found at row {i}, col {j}: {value}"));
                }
            }
        }

        // Check targets if provided
        if let Some(targets) = targets {
            if targets.len() != features.len() {
                warnings.push(format!(
                    "Target count ({}) doesn't match sample count ({})",
                    targets.len(),
                    features.len()
                ));
            }

            for (i, &target) in targets.iter().enumerate() {
                if target.is_nan() {
                    warnings.push(format!("NaN target at index {i}"));
                } else if target.is_infinite() {
                    warnings.push(format!("Infinite target at index {i}: {target}"));
                }
            }
        }

        Ok(warnings)
    }

    /// Visualize data distribution (simplified text histogram)
    pub fn text_histogram(data: &[f64], bins: usize) -> String {
        if data.is_empty() {
            return "No data to visualize".to_string();
        }

        let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if min == max {
            return format!("All values equal: {min}");
        }

        let bin_width = (max - min) / bins as f64;
        let mut counts = vec![0; bins];

        for &value in data {
            let bin = ((value - min) / bin_width).floor() as usize;
            let bin = bin.min(bins - 1);
            counts[bin] += 1;
        }

        let max_count = *counts.iter().max().unwrap();
        let scale = 50.0 / max_count as f64;

        let mut result = String::from("Data Distribution:\n");
        for (i, &count) in counts.iter().enumerate() {
            let bin_start = min + i as f64 * bin_width;
            let bin_end = bin_start + bin_width;
            let bar_length = (count as f64 * scale) as usize;
            let bar = "█".repeat(bar_length);

            result.push_str(&format!(
                "[{bin_start:8.2} - {bin_end:8.2}]: {count:4} {bar}\n"
            ));
        }

        result
    }
}

/// Macro for quick performance timing
#[macro_export]
macro_rules! time_it {
    ($name:expr, $code:block) => {{
        let start = std::time::Instant::now();
        let result = $code;
        let duration = start.elapsed();
        println!("{}: {:?}", $name, duration);
        result
    }};
}

/// Macro for conditional debugging output
#[macro_export]
macro_rules! debug_println {
    ($($arg:tt)*) => {
        if cfg!(debug_assertions) {
            println!("[DEBUG] {}", format!($($arg)*));
        }
    };
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debug_context() {
        let ctx = DebugContext::new("test_module", "test_function", 42)
            .with_metadata("key1", "value1")
            .with_metadata("key2", "value2");

        let formatted = ctx.format();
        assert!(formatted.contains("test_module::test_function:42"));
        assert!(formatted.contains("key1=value1"));
        assert!(formatted.contains("key2=value2"));
    }

    #[test]
    fn test_array_debugger() {
        let array = vec![1, 2, 3, 4, 5];
        let stats = ArrayDebugger::array_stats(&array);
        assert!(stats.contains("len=5"));

        let shape_info = ArrayDebugger::array_shape_info(&[2, 3, 4]);
        assert!(shape_info.contains("Total elements: 24"));

        let diffs = ArrayDebugger::compare_arrays(&[1, 2, 3], &[1, 2, 4]);
        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].contains("Element 2 differs"));
    }

    #[test]
    fn test_float_array_check() {
        let array = vec![1.0, 2.0, f64::NAN, f64::INFINITY, 5.0];
        let issues = ArrayDebugger::check_float_array(&array);
        assert_eq!(issues.len(), 2);
        assert!(issues.iter().any(|s| s.contains("NaN")));
        assert!(issues.iter().any(|s| s.contains("Infinite")));
    }

    #[test]
    fn test_performance_debugger() {
        let mut debugger = PerformanceDebugger::new();

        debugger.start_timer("test_operation");
        std::thread::sleep(std::time::Duration::from_millis(1));
        let duration = debugger.stop_timer("test_operation");

        assert!(duration.is_some());
        assert!(duration.unwrap().as_millis() >= 1);

        let stats = debugger.timing_stats("test_operation");
        assert!(stats.is_some());
        assert_eq!(stats.unwrap().count, 1);
    }

    #[test]
    fn test_test_data_generator() {
        let integers = TestDataGenerator::random_integers(10, 1, 100);
        assert_eq!(integers.len(), 10);
        assert!(integers.iter().all(|&x| (1..=100).contains(&x)));

        let floats = TestDataGenerator::random_floats(10, 0.0, 1.0);
        assert_eq!(floats.len(), 10);
        assert!(floats.iter().all(|&x| (0.0..=1.0).contains(&x)));

        let matrix = TestDataGenerator::test_matrix(3, 4);
        assert_eq!(matrix.len(), 3);
        assert!(matrix.iter().all(|row| row.len() == 4));

        let strings = TestDataGenerator::test_strings(5);
        assert_eq!(strings.len(), 5);
        assert!(strings.iter().all(|s| !s.is_empty()));
    }

    #[test]
    fn test_pathological_cases() {
        let cases = TestDataGenerator::pathological_cases();
        assert!(cases.contains_key("empty"));
        assert!(cases.contains_key("special_floats"));
        assert_eq!(cases["empty"].len(), 0);
        assert_eq!(cases["single"].len(), 1);
    }

    #[test]
    fn test_diagnostic_tools() {
        let checks = DiagnosticTools::system_check();
        assert!(!checks.is_empty());
        assert!(checks.iter().any(|s| s.contains("Target architecture")));

        let features = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let targets = vec![1.0, 2.0];

        let warnings = DiagnosticTools::validate_ml_inputs(&features, Some(&targets)).unwrap();
        assert!(warnings.is_empty());

        // Test with inconsistent data
        let bad_features = vec![vec![1.0, 2.0], vec![3.0]];
        let warnings = DiagnosticTools::validate_ml_inputs(&bad_features, Some(&targets)).unwrap();
        assert!(!warnings.is_empty());
    }

    #[test]
    fn test_text_histogram() {
        let data = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0];
        let histogram = DiagnosticTools::text_histogram(&data, 5);
        assert!(histogram.contains("Data Distribution:"));
        assert!(histogram.contains("█"));

        let empty_data = vec![];
        let empty_histogram = DiagnosticTools::text_histogram(&empty_data, 5);
        assert!(empty_histogram.contains("No data"));

        let uniform_data = vec![1.0, 1.0, 1.0];
        let uniform_histogram = DiagnosticTools::text_histogram(&uniform_data, 5);
        assert!(uniform_histogram.contains("All values equal"));
    }
}
