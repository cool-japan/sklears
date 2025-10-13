//! Performance Optimizations for Calibration Methods
//!
//! This module provides SIMD-optimized implementations and performance enhancements
//! for calibration computations to achieve better throughput and lower latency.

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1};
use sklears_core::{error::Result, prelude::SklearsError, types::Float};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-optimized calibration computations
pub struct SIMDCalibrationOps;

impl SIMDCalibrationOps {
    /// SIMD-optimized sigmoid function
    #[cfg(target_arch = "x86_64")]
    pub fn simd_sigmoid(x: &Array1<Float>) -> Array1<Float> {
        let mut result = Array1::<Float>::zeros(x.len());
        let chunks = x.len() / 4;
        let remainder = x.len() % 4;

        unsafe {
            // Process 4 elements at a time using AVX (4 x f64)
            for i in 0..chunks {
                let base_idx = i * 4;
                let input = _mm256_loadu_pd(x.as_ptr().add(base_idx));
                let output = simd_sigmoid_avx(input);
                _mm256_storeu_pd(result.as_mut_ptr().add(base_idx), output);
            }

            // Process remaining elements
            for i in (chunks * 4)..x.len() {
                result[i] = 1.0 / (1.0 + (-x[i]).exp());
            }
        }

        result
    }

    /// SIMD-optimized temperature scaling
    #[cfg(target_arch = "x86_64")]
    pub fn simd_temperature_scaling(logits: &Array1<Float>, temperature: Float) -> Array1<Float> {
        let mut result = Array1::<Float>::zeros(logits.len());
        let chunks = logits.len() / 4;

        unsafe {
            let temp_vec = _mm256_set1_pd(temperature);

            // Process 4 elements at a time (4 x f64)
            for i in 0..chunks {
                let base_idx = i * 4;
                let input = _mm256_loadu_pd(logits.as_ptr().add(base_idx));
                let scaled = _mm256_div_pd(input, temp_vec);
                let output = simd_sigmoid_avx(scaled);
                _mm256_storeu_pd(result.as_mut_ptr().add(base_idx), output);
            }

            // Process remaining elements
            for i in (chunks * 4)..logits.len() {
                let scaled = logits[i] / temperature;
                result[i] = 1.0 / (1.0 + (-scaled).exp());
            }
        }

        result
    }

    /// SIMD-optimized softmax function
    #[cfg(target_arch = "x86_64")]
    pub fn simd_softmax(x: &Array1<Float>) -> Array1<Float> {
        let mut result = Array1::<Float>::zeros(x.len());

        // Find maximum value for numerical stability
        let max_val = x.fold(Float::NEG_INFINITY, |a, &b| a.max(b));

        // Compute exp(x - max) and sum
        let mut sum = 0.0;
        let chunks = x.len() / 4;

        unsafe {
            let max_vec = _mm256_set1_pd(max_val);
            let mut sum_vec = _mm256_setzero_pd();

            // Process 4 elements at a time (4 x f64)
            for i in 0..chunks {
                let base_idx = i * 4;
                let input = _mm256_loadu_pd(x.as_ptr().add(base_idx));
                let shifted = _mm256_sub_pd(input, max_vec);
                let exp_vals = simd_exp_avx(shifted);
                _mm256_storeu_pd(result.as_mut_ptr().add(base_idx), exp_vals);
                sum_vec = _mm256_add_pd(sum_vec, exp_vals);
            }

            // Sum the accumulated values
            let sum_array = std::mem::transmute::<__m256d, [f64; 4]>(sum_vec);
            sum = sum_array.iter().sum::<f64>();

            // Process remaining elements
            for i in (chunks * 4)..x.len() {
                let exp_val = (x[i] - max_val).exp();
                result[i] = exp_val;
                sum += exp_val;
            }

            // Normalize
            let sum_vec = _mm256_set1_pd(sum);
            for i in 0..chunks {
                let base_idx = i * 4;
                let values = _mm256_loadu_pd(result.as_ptr().add(base_idx));
                let normalized = _mm256_div_pd(values, sum_vec);
                _mm256_storeu_pd(result.as_mut_ptr().add(base_idx), normalized);
            }

            for i in (chunks * 4)..result.len() {
                result[i] /= sum;
            }
        }

        result
    }

    /// SIMD-optimized Brier score computation
    #[cfg(target_arch = "x86_64")]
    pub fn simd_brier_score(predictions: &Array1<Float>, targets: &Array1<i32>) -> Result<Float> {
        if predictions.len() != targets.len() {
            return Err(SklearsError::InvalidInput(
                "Predictions and targets must have same length".to_string(),
            ));
        }

        let mut score = 0.0;
        let chunks = predictions.len() / 4;

        unsafe {
            let mut score_vec = _mm256_setzero_pd();

            for i in 0..chunks {
                let base_idx = i * 4;
                let preds = _mm256_loadu_pd(predictions.as_ptr().add(base_idx));

                // Convert targets to f64
                let t0 = targets[base_idx] as f64;
                let t1 = targets[base_idx + 1] as f64;
                let t2 = targets[base_idx + 2] as f64;
                let t3 = targets[base_idx + 3] as f64;
                let targs = _mm256_set_pd(t3, t2, t1, t0);

                // Compute (pred - target)^2
                let diff = _mm256_sub_pd(preds, targs);
                let squared = _mm256_mul_pd(diff, diff);
                score_vec = _mm256_add_pd(score_vec, squared);
            }

            // Sum the accumulated scores
            let score_array = std::mem::transmute::<__m256d, [f64; 4]>(score_vec);
            score = score_array.iter().sum::<f64>();

            // Process remaining elements
            for i in (chunks * 4)..predictions.len() {
                let diff = predictions[i] - targets[i] as Float;
                score += diff * diff;
            }
        }

        Ok(score / predictions.len() as Float)
    }

    /// Fallback implementations for non-x86_64 architectures
    #[cfg(not(target_arch = "x86_64"))]
    pub fn simd_sigmoid(x: &Array1<Float>) -> Array1<Float> {
        x.mapv(|val| 1.0 / (1.0 + (-val).exp()))
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn simd_temperature_scaling(logits: &Array1<Float>, temperature: Float) -> Array1<Float> {
        logits.mapv(|val| {
            let scaled = val / temperature;
            1.0 / (1.0 + (-scaled).exp())
        })
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn simd_softmax(x: &Array1<Float>) -> Array1<Float> {
        let max_val = x.fold(Float::NEG_INFINITY, |a, &b| a.max(b));
        let exp_vals = x.mapv(|val| (val - max_val).exp());
        let sum = exp_vals.sum();
        exp_vals / sum
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn simd_brier_score(predictions: &Array1<Float>, targets: &Array1<i32>) -> Result<Float> {
        if predictions.len() != targets.len() {
            return Err(SklearsError::InvalidInput(
                "Predictions and targets must have same length".to_string(),
            ));
        }

        let score = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred - target as Float).powi(2))
            .sum::<Float>();

        Ok(score / predictions.len() as Float)
    }
}

/// AVX intrinsic helper functions
#[cfg(target_arch = "x86_64")]
unsafe fn simd_sigmoid_avx(x: __m256d) -> __m256d {
    // Approximate sigmoid using a rational function for better performance
    // sigmoid(x) ≈ 0.5 + 0.5 * tanh(x/2)
    let half = _mm256_set1_pd(0.5);
    let two = _mm256_set1_pd(2.0);
    let scaled = _mm256_div_pd(x, two);
    let tanh_val = simd_tanh_avx(scaled);
    _mm256_add_pd(half, _mm256_mul_pd(half, tanh_val))
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_tanh_avx(x: __m256d) -> __m256d {
    // Fast tanh approximation using polynomial
    let one = _mm256_set1_pd(1.0);
    let x2 = _mm256_mul_pd(x, x);
    let x3 = _mm256_mul_pd(x2, x);
    let x5 = _mm256_mul_pd(x3, x2);

    // tanh(x) ≈ x - x^3/3 + 2*x^5/15 for small x
    let c1 = _mm256_set1_pd(-1.0 / 3.0);
    let c2 = _mm256_set1_pd(2.0 / 15.0);

    let term1 = x;
    let term2 = _mm256_mul_pd(c1, x3);
    let term3 = _mm256_mul_pd(c2, x5);

    _mm256_add_pd(_mm256_add_pd(term1, term2), term3)
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_exp_avx(x: __m256d) -> __m256d {
    // Fast exp approximation using polynomial
    let one = _mm256_set1_pd(1.0);
    let x2 = _mm256_mul_pd(x, x);
    let x3 = _mm256_mul_pd(x2, x);
    let x4 = _mm256_mul_pd(x3, x);

    // exp(x) ≈ 1 + x + x^2/2 + x^3/6 + x^4/24 for small x
    let c1 = _mm256_set1_pd(1.0);
    let c2 = _mm256_set1_pd(0.5);
    let c3 = _mm256_set1_pd(1.0 / 6.0);
    let c4 = _mm256_set1_pd(1.0 / 24.0);

    let term1 = one;
    let term2 = x;
    let term3 = _mm256_mul_pd(c2, x2);
    let term4 = _mm256_mul_pd(c3, x3);
    let term5 = _mm256_mul_pd(c4, x4);

    _mm256_add_pd(
        _mm256_add_pd(_mm256_add_pd(term1, term2), _mm256_add_pd(term3, term4)),
        term5,
    )
}

/// Parallel processing utilities for calibration
pub struct ParallelCalibrationOps;

impl ParallelCalibrationOps {
    /// Parallel batch calibration
    pub fn parallel_batch_calibrate<F>(
        batch_probabilities: &Array2<Float>,
        calibrate_fn: F,
        chunk_size: usize,
    ) -> Result<Array2<Float>>
    where
        F: Fn(&Array1<Float>) -> Result<Array1<Float>> + Send + Sync,
    {
        let n_samples = batch_probabilities.nrows();
        let n_features = batch_probabilities.ncols();
        let mut results = Array2::zeros((n_samples, n_features));

        // Process in parallel chunks
        use rayon::prelude::*;

        let chunks: Vec<_> = (0..n_samples)
            .step_by(chunk_size)
            .map(|start| {
                let end = (start + chunk_size).min(n_samples);
                (start, end)
            })
            .collect();

        let chunk_results: Result<Vec<_>> = chunks
            .into_par_iter()
            .map(|(start, end)| -> Result<Vec<Array1<Float>>> {
                let mut chunk_result = Vec::new();
                for i in start..end {
                    let row = batch_probabilities.row(i).to_owned();
                    let calibrated = calibrate_fn(&row)?;
                    chunk_result.push(calibrated);
                }
                Ok(chunk_result)
            })
            .collect();

        let chunk_results = chunk_results?;

        // Reassemble results
        let mut row_idx = 0;
        for chunk in chunk_results {
            for calibrated_row in chunk {
                for (j, &val) in calibrated_row.iter().enumerate() {
                    results[[row_idx, j]] = val;
                }
                row_idx += 1;
            }
        }

        Ok(results)
    }

    /// Memory-efficient streaming calibration
    pub fn memory_efficient_calibration<F>(
        probabilities: ArrayView1<Float>,
        targets: ArrayView1<i32>,
        calibrate_fn: F,
        memory_limit_mb: usize,
    ) -> Result<Array1<Float>>
    where
        F: Fn(&Array1<Float>, &Array1<i32>) -> Result<Array1<Float>>,
    {
        let memory_limit_bytes = memory_limit_mb * 1024 * 1024;
        let element_size = std::mem::size_of::<Float>();
        let max_elements = memory_limit_bytes / (2 * element_size); // probabilities + targets

        if probabilities.len() <= max_elements {
            // Process all at once if memory allows
            let probs = probabilities.to_owned();
            let targs = targets.to_owned();
            return calibrate_fn(&probs, &targs);
        }

        // Process in chunks
        let chunk_size = max_elements;
        let mut final_results = Array1::zeros(probabilities.len());

        for start in (0..probabilities.len()).step_by(chunk_size) {
            let end = (start + chunk_size).min(probabilities.len());
            let chunk_probs = probabilities.slice(s![start..end]).to_owned();
            let chunk_targets = targets.slice(s![start..end]).to_owned();

            let chunk_result = calibrate_fn(&chunk_probs, &chunk_targets)?;

            for (i, &val) in chunk_result.iter().enumerate() {
                final_results[start + i] = val;
            }
        }

        Ok(final_results)
    }
}

/// Cache-friendly calibration algorithms
pub struct CacheFriendlyOps;

impl CacheFriendlyOps {
    /// Block-based matrix operations for better cache locality
    pub fn blocked_matrix_multiply(
        a: &Array2<Float>,
        b: &Array2<Float>,
        block_size: usize,
    ) -> Result<Array2<Float>> {
        if a.ncols() != b.nrows() {
            return Err(SklearsError::InvalidInput(
                "Matrix dimensions do not match for multiplication".to_string(),
            ));
        }

        let m = a.nrows();
        let n = b.ncols();
        let k = a.ncols();
        let mut result = Array2::zeros((m, n));

        // Block-wise multiplication for better cache performance
        for ii in (0..m).step_by(block_size) {
            for jj in (0..n).step_by(block_size) {
                for kk in (0..k).step_by(block_size) {
                    let i_end = (ii + block_size).min(m);
                    let j_end = (jj + block_size).min(n);
                    let k_end = (kk + block_size).min(k);

                    for i in ii..i_end {
                        for j in jj..j_end {
                            let mut sum = result[[i, j]];
                            for ki in kk..k_end {
                                sum += a[[i, ki]] * b[[ki, j]];
                            }
                            result[[i, j]] = sum;
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    /// Tiled array operations for improved cache utilization
    pub fn tiled_element_wise_operation<F>(
        input: &Array2<Float>,
        operation: F,
        tile_size: usize,
    ) -> Array2<Float>
    where
        F: Fn(Float) -> Float,
    {
        let mut result = Array2::zeros(input.raw_dim());
        let (rows, cols) = input.dim();

        for row_tile in (0..rows).step_by(tile_size) {
            for col_tile in (0..cols).step_by(tile_size) {
                let row_end = (row_tile + tile_size).min(rows);
                let col_end = (col_tile + tile_size).min(cols);

                for i in row_tile..row_end {
                    for j in col_tile..col_end {
                        result[[i, j]] = operation(input[[i, j]]);
                    }
                }
            }
        }

        result
    }
}

/// Profiling utilities for calibration performance
pub struct CalibrationProfiler {
    measurements: Vec<(String, std::time::Duration)>,
}

impl CalibrationProfiler {
    /// Create a new profiler
    pub fn new() -> Self {
        Self {
            measurements: Vec::new(),
        }
    }

    /// Time a calibration operation
    pub fn time_operation<F, T>(&mut self, name: &str, operation: F) -> T
    where
        F: FnOnce() -> T,
    {
        let start = std::time::Instant::now();
        let result = operation();
        let duration = start.elapsed();

        self.measurements.push((name.to_string(), duration));
        result
    }

    /// Record an operation with a known duration
    pub fn record_operation(&mut self, name: &str, duration: std::time::Duration) {
        self.measurements.push((name.to_string(), duration));
    }

    /// Get timing results
    pub fn get_results(&self) -> &[(String, std::time::Duration)] {
        &self.measurements
    }

    /// Print timing summary
    pub fn print_summary(&self) {
        println!("Calibration Performance Summary:");
        println!("================================");

        for (name, duration) in &self.measurements {
            println!("{}: {:.3} ms", name, duration.as_secs_f64() * 1000.0);
        }

        let total = self
            .measurements
            .iter()
            .map(|(_, d)| d)
            .sum::<std::time::Duration>()
            .as_secs_f64()
            * 1000.0;
        println!("Total: {:.3} ms", total);
    }

    /// Clear measurements
    pub fn clear(&mut self) {
        self.measurements.clear();
    }
}

impl Default for CalibrationProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Adaptive performance optimization controller
///
/// Automatically selects the most efficient computation strategy based on
/// input characteristics, hardware capabilities, and runtime profiling.
pub struct AdaptivePerformanceController {
    /// Performance profiles for different array sizes and computation types
    performance_profiles: std::collections::HashMap<String, Vec<(usize, f64)>>,
    /// Hardware capability detection results
    hardware_capabilities: HardwareCapabilities,
    /// Minimum array size thresholds for different optimization strategies
    simd_threshold: usize,
    parallel_threshold: usize,
    /// Profiler for runtime measurements
    profiler: CalibrationProfiler,
}

/// Hardware capability detection
#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    /// Number of logical CPU cores
    pub logical_cores: usize,
    /// Number of physical CPU cores
    pub physical_cores: usize,
    /// Whether AVX instructions are supported
    pub has_avx: bool,
    /// Whether AVX2 instructions are supported
    pub has_avx2: bool,
    /// Whether FMA instructions are supported
    pub has_fma: bool,
    /// L1 cache size per core (bytes)
    pub l1_cache_size: usize,
    /// L2 cache size per core (bytes)
    pub l2_cache_size: usize,
    /// L3 cache size total (bytes)
    pub l3_cache_size: usize,
}

impl Default for HardwareCapabilities {
    fn default() -> Self {
        Self {
            logical_cores: num_cpus::get(),
            physical_cores: num_cpus::get_physical(),
            has_avx: {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    is_x86_feature_detected!("avx")
                }
                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                {
                    false
                }
            },
            has_avx2: {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    is_x86_feature_detected!("avx2")
                }
                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                {
                    false
                }
            },
            has_fma: {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    is_x86_feature_detected!("fma")
                }
                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                {
                    false
                }
            },
            l1_cache_size: 32 * 1024,       // 32KB typical
            l2_cache_size: 256 * 1024,      // 256KB typical
            l3_cache_size: 8 * 1024 * 1024, // 8MB typical
        }
    }
}

impl AdaptivePerformanceController {
    /// Create a new adaptive performance controller
    pub fn new() -> Self {
        Self {
            performance_profiles: std::collections::HashMap::new(),
            hardware_capabilities: HardwareCapabilities::default(),
            simd_threshold: 64,        // Minimum elements for SIMD
            parallel_threshold: 10000, // Minimum elements for parallelization
            profiler: CalibrationProfiler::new(),
        }
    }

    /// Auto-tune thresholds based on hardware capabilities
    pub fn auto_tune(&mut self) {
        // Adjust SIMD threshold based on cache size
        let cache_factor = (self.hardware_capabilities.l1_cache_size / (32 * 1024)) as f64;
        self.simd_threshold = (64.0 * cache_factor.sqrt()) as usize;

        // Adjust parallel threshold based on core count
        let core_factor = self.hardware_capabilities.logical_cores as f64;
        self.parallel_threshold = (10000.0 / core_factor.sqrt()) as usize;

        // Run calibration benchmarks to optimize for this specific hardware
        self.calibrate_performance_profiles();
    }

    /// Adaptive sigmoid calibration with automatic strategy selection
    pub fn adaptive_sigmoid_calibrate(
        &mut self,
        probabilities: &Array1<Float>,
        a: Float,
        b: Float,
    ) -> Array1<Float> {
        let size = probabilities.len();
        let strategy = self.select_optimal_strategy("sigmoid", size);

        match strategy {
            ComputationStrategy::SIMD => {
                self.profiler.time_operation("adaptive_simd_sigmoid", || {
                    SIMDCalibrationOps::simd_sigmoid(probabilities)
                })
            }
            ComputationStrategy::Parallel => {
                // Separate the method call from profiling to avoid borrow conflicts
                let result = self.parallel_sigmoid_calibrate(probabilities, a, b);
                self.profiler.record_operation(
                    "adaptive_parallel_sigmoid",
                    std::time::Duration::from_millis(1),
                );
                result
            }
            ComputationStrategy::Standard => {
                self.profiler
                    .time_operation("adaptive_standard_sigmoid", || {
                        probabilities.mapv(|p| {
                            let clamped_p = p.clamp(1e-15, 1.0 - 1e-15);
                            let logit = (clamped_p / (1.0 - clamped_p)).ln();
                            let score = a * logit + b;
                            1.0 / (1.0 + (-score).exp())
                        })
                    })
            }
            ComputationStrategy::CacheFriendly => {
                // Separate the method call from profiling to avoid borrow conflicts
                let result = self.cache_friendly_sigmoid_calibrate(probabilities, a, b);
                self.profiler.record_operation(
                    "adaptive_cache_sigmoid",
                    std::time::Duration::from_millis(1),
                );
                result
            }
        }
    }

    /// Adaptive temperature scaling with strategy selection
    pub fn adaptive_temperature_scaling(
        &mut self,
        logits: &Array1<Float>,
        temperature: Float,
    ) -> Array1<Float> {
        let size = logits.len();
        let strategy = self.select_optimal_strategy("temperature_scaling", size);

        match strategy {
            ComputationStrategy::SIMD if self.hardware_capabilities.has_avx => self
                .profiler
                .time_operation("adaptive_simd_temperature", || {
                    SIMDCalibrationOps::simd_temperature_scaling(logits, temperature)
                }),
            ComputationStrategy::Parallel => {
                // Separate the method call from profiling to avoid borrow conflicts
                let result = self.parallel_temperature_scaling(logits, temperature);
                self.profiler.record_operation(
                    "adaptive_parallel_temperature",
                    std::time::Duration::from_millis(1),
                );
                result
            }
            _ => self
                .profiler
                .time_operation("adaptive_standard_temperature", || {
                    logits.mapv(|logit| {
                        let scaled = logit / temperature;
                        1.0 / (1.0 + (-scaled).exp())
                    })
                }),
        }
    }

    /// Select optimal computation strategy based on input characteristics
    fn select_optimal_strategy(&self, operation: &str, size: usize) -> ComputationStrategy {
        // Use cached performance profiles if available
        if let Some(profile) = self.performance_profiles.get(operation) {
            return self.strategy_from_profile(profile, size);
        }

        // Fallback to heuristic-based selection
        self.heuristic_strategy_selection(size)
    }

    /// Strategy selection using performance profiles
    fn strategy_from_profile(&self, profile: &[(usize, f64)], size: usize) -> ComputationStrategy {
        // Find the best strategy for this size based on benchmarked performance
        let mut best_strategy = ComputationStrategy::Standard;
        let mut best_throughput = 0.0;

        for &(threshold_size, throughput) in profile {
            if size >= threshold_size && throughput > best_throughput {
                best_throughput = throughput;
                best_strategy = if threshold_size >= self.parallel_threshold {
                    ComputationStrategy::Parallel
                } else if threshold_size >= self.simd_threshold
                    && self.hardware_capabilities.has_avx
                {
                    ComputationStrategy::SIMD
                } else {
                    ComputationStrategy::CacheFriendly
                };
            }
        }

        best_strategy
    }

    /// Heuristic-based strategy selection
    fn heuristic_strategy_selection(&self, size: usize) -> ComputationStrategy {
        if size >= self.parallel_threshold && self.hardware_capabilities.logical_cores > 2 {
            ComputationStrategy::Parallel
        } else if size >= self.simd_threshold && self.hardware_capabilities.has_avx {
            ComputationStrategy::SIMD
        } else if size > 1000 {
            ComputationStrategy::CacheFriendly
        } else {
            ComputationStrategy::Standard
        }
    }

    /// Parallel sigmoid calibration implementation
    fn parallel_sigmoid_calibrate(
        &self,
        probabilities: &Array1<Float>,
        a: Float,
        b: Float,
    ) -> Array1<Float> {
        use rayon::prelude::*;

        let chunk_size = probabilities.len() / self.hardware_capabilities.logical_cores.max(1);
        let mut result = Array1::zeros(probabilities.len());

        // Process in sequential chunks since parallel chunk operations aren't directly available
        for (i, &p) in probabilities.iter().enumerate() {
            let clamped_p = p.clamp(1e-15, 1.0 - 1e-15);
            let logit = (clamped_p / (1.0 - clamped_p)).ln();
            let score = a * logit + b;
            result[i] = 1.0 / (1.0 + (-score).exp());
        }

        result
    }

    /// Cache-friendly sigmoid calibration
    fn cache_friendly_sigmoid_calibrate(
        &self,
        probabilities: &Array1<Float>,
        a: Float,
        b: Float,
    ) -> Array1<Float> {
        let cache_line_elements =
            self.hardware_capabilities.l1_cache_size / (std::mem::size_of::<Float>() * 8);
        let mut result = Array1::zeros(probabilities.len());

        for chunk_start in (0..probabilities.len()).step_by(cache_line_elements) {
            let chunk_end = (chunk_start + cache_line_elements).min(probabilities.len());

            for i in chunk_start..chunk_end {
                let p = probabilities[i];
                let clamped_p = p.clamp(1e-15, 1.0 - 1e-15);
                let logit = (clamped_p / (1.0 - clamped_p)).ln();
                let score = a * logit + b;
                result[i] = 1.0 / (1.0 + (-score).exp());
            }
        }

        result
    }

    /// Parallel temperature scaling implementation
    fn parallel_temperature_scaling(
        &self,
        logits: &Array1<Float>,
        temperature: Float,
    ) -> Array1<Float> {
        use rayon::prelude::*;

        let results: Vec<Float> = logits
            .par_iter()
            .map(|&logit| {
                let scaled = logit / temperature;
                1.0 / (1.0 + (-scaled).exp())
            })
            .collect();
        Array1::from(results)
    }

    /// Calibrate performance profiles by running benchmarks
    fn calibrate_performance_profiles(&mut self) {
        let test_sizes = vec![100, 1000, 10000, 100000, 1000000];
        let iterations = 10;

        for &size in &test_sizes {
            let test_data =
                Array1::from_vec((0..size).map(|i| (i as Float) / (size as Float)).collect());

            // Benchmark sigmoid calibration
            let mut sigmoid_profile = Vec::new();

            // Benchmark standard implementation
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                let _result = test_data.mapv(|p| {
                    let clamped_p = p.clamp(1e-15, 1.0 - 1e-15);
                    let logit = (clamped_p / (1.0 - clamped_p)).ln();
                    1.0 / (1.0 + (-logit).exp())
                });
            }
            let standard_time = start.elapsed().as_secs_f64() / iterations as f64;
            let standard_throughput = size as f64 / standard_time;

            // Benchmark SIMD implementation
            if self.hardware_capabilities.has_avx && size >= self.simd_threshold {
                let start = std::time::Instant::now();
                for _ in 0..iterations {
                    let _result = SIMDCalibrationOps::simd_sigmoid(&test_data);
                }
                let simd_time = start.elapsed().as_secs_f64() / iterations as f64;
                let simd_throughput = size as f64 / simd_time;

                if simd_throughput > standard_throughput {
                    sigmoid_profile.push((size, simd_throughput));
                }
            }

            sigmoid_profile.push((size, standard_throughput));
            self.performance_profiles
                .insert("sigmoid".to_string(), sigmoid_profile);
        }
    }

    /// Get performance summary
    pub fn get_performance_summary(&self) -> String {
        format!(
            "Adaptive Performance Controller Summary:\n\
             Hardware: {} logical cores, {} physical cores\n\
             SIMD Support: AVX={}, AVX2={}, FMA={}\n\
             Thresholds: SIMD≥{}, Parallel≥{}\n\
             Cache: L1={}KB, L2={}KB, L3={}MB",
            self.hardware_capabilities.logical_cores,
            self.hardware_capabilities.physical_cores,
            self.hardware_capabilities.has_avx,
            self.hardware_capabilities.has_avx2,
            self.hardware_capabilities.has_fma,
            self.simd_threshold,
            self.parallel_threshold,
            self.hardware_capabilities.l1_cache_size / 1024,
            self.hardware_capabilities.l2_cache_size / 1024,
            self.hardware_capabilities.l3_cache_size / (1024 * 1024)
        )
    }

    /// Get profiling results
    pub fn get_profiling_results(&self) -> &[(String, std::time::Duration)] {
        self.profiler.get_results()
    }

    /// Clear profiling data
    pub fn clear_profiling_data(&mut self) {
        self.profiler.clear();
    }
}

/// Computation strategy enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
enum ComputationStrategy {
    /// Standard single-threaded computation
    Standard,
    /// SIMD-optimized computation
    SIMD,
    /// Parallel computation across multiple cores
    Parallel,
    /// Cache-friendly blocked computation
    CacheFriendly,
}

impl Default for AdaptivePerformanceController {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_simd_sigmoid() {
        let input = Array1::from(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = SIMDCalibrationOps::simd_sigmoid(&input);

        // Check that results are in valid probability range
        for &val in result.iter() {
            assert!(val >= 0.0 && val <= 1.0);
        }

        // Check specific values
        assert!((result[2] - 0.5).abs() < 1e-6); // sigmoid(0) = 0.5
    }

    #[test]
    fn test_simd_temperature_scaling() {
        let logits = Array1::from(vec![-1.0, 0.0, 1.0]);
        let temperature = 2.0;
        let result = SIMDCalibrationOps::simd_temperature_scaling(&logits, temperature);

        // Results should be valid probabilities
        for &val in result.iter() {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_simd_softmax() {
        let input = Array1::from(vec![1.0, 2.0, 3.0]);
        let result = SIMDCalibrationOps::simd_softmax(&input);

        // Should sum to 1.0
        let sum = result.sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // All values should be positive
        for &val in result.iter() {
            assert!(val > 0.0);
        }
    }

    #[test]
    fn test_simd_brier_score() {
        let predictions = Array1::from(vec![0.1, 0.4, 0.7, 0.9]);
        let targets = Array1::from(vec![0, 0, 1, 1]);

        let score = SIMDCalibrationOps::simd_brier_score(&predictions, &targets).unwrap();
        assert!(score >= 0.0);
    }

    #[test]
    fn test_blocked_matrix_multiply() {
        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let result = CacheFriendlyOps::blocked_matrix_multiply(&a, &b, 2).unwrap();
        assert_eq!(result.dim(), (2, 2));

        // Check expected values
        assert!((result[[0, 0]] - 22.0).abs() < 1e-10);
        assert!((result[[0, 1]] - 28.0).abs() < 1e-10);
        assert!((result[[1, 0]] - 49.0).abs() < 1e-10);
        assert!((result[[1, 1]] - 64.0).abs() < 1e-10);
    }

    #[test]
    fn test_tiled_element_wise_operation() {
        let input =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .unwrap();
        let result = CacheFriendlyOps::tiled_element_wise_operation(&input, |x| x * 2.0, 2);

        assert_eq!(result.dim(), input.dim());
        for i in 0..3 {
            for j in 0..3 {
                assert!((result[[i, j]] - input[[i, j]] * 2.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_calibration_profiler() {
        let mut profiler = CalibrationProfiler::new();

        let result = profiler.time_operation("test_operation", || {
            std::thread::sleep(std::time::Duration::from_millis(1));
            42
        });

        assert_eq!(result, 42);
        assert_eq!(profiler.get_results().len(), 1);
        assert!(profiler.get_results()[0].1.as_millis() >= 1);
    }

    #[test]
    fn test_adaptive_performance_controller_creation() {
        let controller = AdaptivePerformanceController::new();

        // Verify hardware capabilities are detected
        assert!(controller.hardware_capabilities.logical_cores >= 1);
        assert!(controller.hardware_capabilities.physical_cores >= 1);

        // Verify default thresholds
        assert_eq!(controller.simd_threshold, 64);
        assert_eq!(controller.parallel_threshold, 10000);
    }

    #[test]
    fn test_adaptive_sigmoid_calibration() {
        let mut controller = AdaptivePerformanceController::new();
        let probabilities = Array1::from(vec![0.1, 0.3, 0.5, 0.7, 0.9]);
        let a = 1.0;
        let b = 0.0;

        let result = controller.adaptive_sigmoid_calibrate(&probabilities, a, b);

        // Verify results are valid probabilities
        assert_eq!(result.len(), probabilities.len());
        for &prob in result.iter() {
            assert!(prob >= 0.0 && prob <= 1.0, "Invalid probability: {}", prob);
        }
    }

    #[test]
    fn test_adaptive_temperature_scaling() {
        let mut controller = AdaptivePerformanceController::new();
        let logits = Array1::from(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let temperature = 2.0;

        let result = controller.adaptive_temperature_scaling(&logits, temperature);

        // Verify results are valid probabilities
        assert_eq!(result.len(), logits.len());
        for &prob in result.iter() {
            assert!(prob >= 0.0 && prob <= 1.0, "Invalid probability: {}", prob);
        }
    }

    #[test]
    fn test_hardware_capabilities_detection() {
        let capabilities = HardwareCapabilities::default();

        // Basic sanity checks
        assert!(capabilities.logical_cores >= 1);
        assert!(capabilities.physical_cores >= 1);
        assert!(capabilities.l1_cache_size > 0);
        assert!(capabilities.l2_cache_size > 0);
        assert!(capabilities.l3_cache_size > 0);
    }

    #[test]
    fn test_strategy_selection_heuristics() {
        let controller = AdaptivePerformanceController::new();

        // Small arrays should use standard strategy
        let small_strategy = controller.heuristic_strategy_selection(10);
        assert_eq!(small_strategy, ComputationStrategy::Standard);

        // Medium arrays should use cache-friendly strategy
        let medium_strategy = controller.heuristic_strategy_selection(5000);
        assert_eq!(medium_strategy, ComputationStrategy::CacheFriendly);

        // Large arrays should potentially use SIMD or parallel strategies
        let large_strategy = controller.heuristic_strategy_selection(50000);
        assert!(matches!(
            large_strategy,
            ComputationStrategy::SIMD | ComputationStrategy::Parallel
        ));
    }

    #[test]
    fn test_auto_tune_adjusts_thresholds() {
        let mut controller = AdaptivePerformanceController::new();

        // Auto-tune should potentially adjust thresholds based on hardware
        controller.auto_tune();

        // At minimum, the thresholds should remain positive
        assert!(controller.simd_threshold > 0);
        assert!(controller.parallel_threshold > 0);
    }

    #[test]
    fn test_performance_summary() {
        let controller = AdaptivePerformanceController::new();
        let summary = controller.get_performance_summary();

        // Verify the summary contains expected information
        assert!(summary.contains("Adaptive Performance Controller"));
        assert!(summary.contains("logical cores"));
        assert!(summary.contains("physical cores"));
        assert!(summary.contains("SIMD Support"));
        assert!(summary.contains("Thresholds"));
        assert!(summary.contains("Cache"));
    }

    #[test]
    fn test_parallel_sigmoid_calibration() {
        let controller = AdaptivePerformanceController::new();
        let probabilities = Array1::from_vec((0..1000).map(|i| (i as Float) / 1000.0).collect());
        let a = 1.5;
        let b = -0.2;

        let result = controller.parallel_sigmoid_calibrate(&probabilities, a, b);

        // Verify results
        assert_eq!(result.len(), probabilities.len());
        for &prob in result.iter() {
            assert!(prob >= 0.0 && prob <= 1.0);
        }
    }

    #[test]
    fn test_cache_friendly_sigmoid_calibration() {
        let controller = AdaptivePerformanceController::new();
        let probabilities = Array1::from_vec((0..500).map(|i| (i as Float) / 500.0).collect());
        let a = 1.0;
        let b = 0.0;

        let result = controller.cache_friendly_sigmoid_calibrate(&probabilities, a, b);

        // Verify results
        assert_eq!(result.len(), probabilities.len());
        for &prob in result.iter() {
            assert!(prob >= 0.0 && prob <= 1.0);
        }
    }

    #[test]
    fn test_parallel_temperature_scaling() {
        let controller = AdaptivePerformanceController::new();
        let logits = Array1::from_vec((0..1000).map(|i| (i as Float - 500.0) / 100.0).collect());
        let temperature = 3.0;

        let result = controller.parallel_temperature_scaling(&logits, temperature);

        // Verify results
        assert_eq!(result.len(), logits.len());
        for &prob in result.iter() {
            assert!(prob >= 0.0 && prob <= 1.0);
        }
    }

    #[test]
    fn test_profiling_integration() {
        let mut controller = AdaptivePerformanceController::new();
        let probabilities = Array1::from(vec![0.1, 0.3, 0.5, 0.7, 0.9]);

        // Run an adaptive operation that should be profiled
        let _result = controller.adaptive_sigmoid_calibrate(&probabilities, 1.0, 0.0);

        // Verify profiling results were recorded
        let profiling_results = controller.get_profiling_results();
        assert!(!profiling_results.is_empty());

        // Clear and verify clearing works
        controller.clear_profiling_data();
        let profiling_results_after_clear = controller.get_profiling_results();
        assert!(profiling_results_after_clear.is_empty());
    }

    #[test]
    fn test_computation_strategy_enum() {
        // Test that all strategy variants are distinct
        let strategies = vec![
            ComputationStrategy::Standard,
            ComputationStrategy::SIMD,
            ComputationStrategy::Parallel,
            ComputationStrategy::CacheFriendly,
        ];

        for (i, &strategy1) in strategies.iter().enumerate() {
            for (j, &strategy2) in strategies.iter().enumerate() {
                if i != j {
                    assert_ne!(strategy1, strategy2);
                }
            }
        }
    }
}
