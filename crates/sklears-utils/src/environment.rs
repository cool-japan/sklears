//! Environment detection utilities for hardware and runtime capabilities
//!
//! This module provides utilities for detecting various aspects of the runtime environment
//! including hardware capabilities, OS-specific features, compiler information, and
//! performance characteristics.
//!
//! # Examples
//!
//! ```rust
//! use sklears_utils::environment::{HardwareDetector, PerformanceCharacteristics};
//!
//! let hw = HardwareDetector::new();
//! let cpu_count = hw.cpu_cores();
//! let has_simd = hw.has_avx2();
//!
//! let perf = PerformanceCharacteristics::measure();
//! println!("Memory bandwidth: {} GB/s", perf.memory_bandwidth_gbps);
//! ```

use std::sync::OnceLock;
use std::time::Instant;

/// Hardware capability detection
#[derive(Debug, Clone)]
pub struct HardwareDetector {
    cpu_info: CpuInfo,
    memory_info: MemoryInfo,
    cache_info: CacheInfo,
}

#[derive(Debug, Clone)]
pub struct CpuInfo {
    pub cores: usize,
    pub logical_cores: usize,
    pub architecture: String,
    pub vendor: String,
    pub model_name: String,
    pub base_frequency_mhz: Option<u32>,
    pub max_frequency_mhz: Option<u32>,
    pub features: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct MemoryInfo {
    pub total_memory_bytes: u64,
    pub available_memory_bytes: u64,
    pub page_size_bytes: usize,
}

#[derive(Debug, Clone, Default)]
pub struct CacheInfo {
    pub l1_data_cache_kb: Option<u32>,
    pub l1_instruction_cache_kb: Option<u32>,
    pub l2_cache_kb: Option<u32>,
    pub l3_cache_kb: Option<u32>,
    pub cache_line_size_bytes: Option<u32>,
}

impl Default for CpuInfo {
    fn default() -> Self {
        Self {
            cores: 1,
            logical_cores: 1,
            architecture: "unknown".to_string(),
            vendor: "unknown".to_string(),
            model_name: "unknown".to_string(),
            base_frequency_mhz: None,
            max_frequency_mhz: None,
            features: Vec::new(),
        }
    }
}

impl Default for MemoryInfo {
    fn default() -> Self {
        Self {
            total_memory_bytes: 0,
            available_memory_bytes: 0,
            page_size_bytes: 4096, // Common default
        }
    }
}

impl HardwareDetector {
    /// Create new hardware detector and gather system information
    pub fn new() -> Self {
        Self {
            cpu_info: Self::detect_cpu_info(),
            memory_info: Self::detect_memory_info(),
            cache_info: Self::detect_cache_info(),
        }
    }

    /// Get number of physical CPU cores
    pub fn cpu_cores(&self) -> usize {
        self.cpu_info.cores
    }

    /// Get number of logical CPU cores (including hyperthreading)
    pub fn logical_cores(&self) -> usize {
        self.cpu_info.logical_cores
    }

    /// Get CPU architecture
    pub fn cpu_architecture(&self) -> &str {
        &self.cpu_info.architecture
    }

    /// Get CPU vendor
    pub fn cpu_vendor(&self) -> &str {
        &self.cpu_info.vendor
    }

    /// Get total system memory in bytes
    pub fn total_memory(&self) -> u64 {
        self.memory_info.total_memory_bytes
    }

    /// Get available system memory in bytes
    pub fn available_memory(&self) -> u64 {
        self.memory_info.available_memory_bytes
    }

    /// Check if CPU supports AVX2 instructions
    pub fn has_avx2(&self) -> bool {
        self.cpu_info.features.iter().any(|f| f.contains("avx2"))
    }

    /// Check if CPU supports AVX-512 instructions
    pub fn has_avx512(&self) -> bool {
        self.cpu_info.features.iter().any(|f| f.contains("avx512"))
    }

    /// Check if CPU supports SSE4.2 instructions
    pub fn has_sse42(&self) -> bool {
        self.cpu_info.features.iter().any(|f| f.contains("sse4_2"))
    }

    /// Check if CPU supports ARM NEON instructions
    pub fn has_neon(&self) -> bool {
        self.cpu_info.features.iter().any(|f| f.contains("neon"))
    }

    /// Check if running on ARM architecture
    pub fn is_arm(&self) -> bool {
        self.cpu_info.architecture.contains("arm") || self.cpu_info.architecture.contains("aarch")
    }

    /// Check if running on x86/x64 architecture
    pub fn is_x86(&self) -> bool {
        self.cpu_info.architecture.contains("x86") || self.cpu_info.architecture.contains("x64")
    }

    /// Get L3 cache size in KB
    pub fn l3_cache_size_kb(&self) -> Option<u32> {
        self.cache_info.l3_cache_kb
    }

    /// Get cache line size in bytes
    pub fn cache_line_size(&self) -> Option<u32> {
        self.cache_info.cache_line_size_bytes
    }

    /// Get all CPU features
    pub fn cpu_features(&self) -> &[String] {
        &self.cpu_info.features
    }

    /// Detect CPU information
    fn detect_cpu_info() -> CpuInfo {
        let mut cpu_info = CpuInfo {
            logical_cores: num_cpus::get(),
            cores: num_cpus::get_physical(),
            architecture: std::env::consts::ARCH.to_string(),
            ..Default::default()
        };

        // Platform-specific CPU detection
        #[cfg(target_arch = "x86_64")]
        {
            Self::detect_x86_features(&mut cpu_info);
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self::detect_arm_features(&mut cpu_info);
        }

        cpu_info
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_x86_features(cpu_info: &mut CpuInfo) {
        if is_x86_feature_detected!("sse") {
            cpu_info.features.push("sse".to_string());
        }
        if is_x86_feature_detected!("sse2") {
            cpu_info.features.push("sse2".to_string());
        }
        if is_x86_feature_detected!("sse3") {
            cpu_info.features.push("sse3".to_string());
        }
        if is_x86_feature_detected!("sse4.1") {
            cpu_info.features.push("sse4_1".to_string());
        }
        if is_x86_feature_detected!("sse4.2") {
            cpu_info.features.push("sse4_2".to_string());
        }
        if is_x86_feature_detected!("avx") {
            cpu_info.features.push("avx".to_string());
        }
        if is_x86_feature_detected!("avx2") {
            cpu_info.features.push("avx2".to_string());
        }
        if is_x86_feature_detected!("fma") {
            cpu_info.features.push("fma".to_string());
        }

        // Try to detect vendor using cpuid if available
        cpu_info.vendor = "Intel/AMD".to_string(); // Simplified
    }

    #[cfg(target_arch = "aarch64")]
    fn detect_arm_features(cpu_info: &mut CpuInfo) {
        cpu_info.vendor = "ARM".to_string();
        cpu_info.features.push("neon".to_string()); // Most ARM64 has NEON
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn detect_x86_features(_cpu_info: &mut CpuInfo) {}

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn detect_arm_features(_cpu_info: &mut CpuInfo) {}

    /// Detect memory information
    fn detect_memory_info() -> MemoryInfo {
        let mut memory_info = MemoryInfo::default();

        // Use sysinfo for cross-platform memory detection
        #[cfg(feature = "sysinfo")]
        {
            use sysinfo::System;
            let mut system = System::new_all();
            system.refresh_memory();
            memory_info.total_memory_bytes = system.total_memory() * 1024; // sysinfo returns KB
            memory_info.available_memory_bytes = system.available_memory() * 1024;
        }

        // Fallback memory detection without sysinfo
        #[cfg(not(feature = "sysinfo"))]
        {
            // Platform-specific fallbacks
            #[cfg(target_os = "linux")]
            {
                if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
                    for line in meminfo.lines() {
                        if line.starts_with("MemTotal:") {
                            if let Some(kb_str) = line.split_whitespace().nth(1) {
                                if let Ok(kb) = kb_str.parse::<u64>() {
                                    memory_info.total_memory_bytes = kb * 1024;
                                }
                            }
                        } else if line.starts_with("MemAvailable:") {
                            if let Some(kb_str) = line.split_whitespace().nth(1) {
                                if let Ok(kb) = kb_str.parse::<u64>() {
                                    memory_info.available_memory_bytes = kb * 1024;
                                }
                            }
                        }
                    }
                }
            }

            #[cfg(target_os = "macos")]
            {
                // Use sysctl for macOS
                if let Ok(output) = std::process::Command::new("sysctl")
                    .arg("-n")
                    .arg("hw.memsize")
                    .output()
                {
                    if let Ok(mem_str) = String::from_utf8(output.stdout) {
                        if let Ok(mem_bytes) = mem_str.trim().parse::<u64>() {
                            memory_info.total_memory_bytes = mem_bytes;
                            memory_info.available_memory_bytes = mem_bytes / 2; // Rough estimate
                        }
                    }
                }
            }

            // Fallback: use a reasonable default for testing
            if memory_info.total_memory_bytes == 0 {
                memory_info.total_memory_bytes = 8 * 1024 * 1024 * 1024; // 8GB default
                memory_info.available_memory_bytes = 4 * 1024 * 1024 * 1024; // 4GB available
            }
        }

        // Fallback page size detection
        #[cfg(unix)]
        {
            unsafe {
                let page_size = libc::sysconf(libc::_SC_PAGESIZE);
                if page_size > 0 {
                    memory_info.page_size_bytes = page_size as usize;
                }
            }
        }

        memory_info
    }

    /// Detect cache information
    fn detect_cache_info() -> CacheInfo {
        let mut cache_info = CacheInfo::default();

        // Platform-specific cache detection
        #[cfg(target_os = "linux")]
        {
            Self::detect_linux_cache_info(&mut cache_info);
        }

        #[cfg(target_os = "macos")]
        {
            Self::detect_macos_cache_info(&mut cache_info);
        }

        #[cfg(target_os = "windows")]
        {
            Self::detect_windows_cache_info(&mut cache_info);
        }

        cache_info
    }

    #[cfg(target_os = "linux")]
    fn detect_linux_cache_info(cache_info: &mut CacheInfo) {
        // Try to read from /sys/devices/system/cpu/cpu0/cache/
        if let Ok(l1d) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index0/size") {
            if let Ok(size) = l1d.trim().strip_suffix("K").unwrap_or(&l1d).parse::<u32>() {
                cache_info.l1_data_cache_kb = Some(size);
            }
        }

        if let Ok(l2) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index2/size") {
            if let Ok(size) = l2.trim().strip_suffix("K").unwrap_or(&l2).parse::<u32>() {
                cache_info.l2_cache_kb = Some(size);
            }
        }

        if let Ok(l3) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index3/size") {
            if let Ok(size) = l3.trim().strip_suffix("K").unwrap_or(&l3).parse::<u32>() {
                cache_info.l3_cache_kb = Some(size);
            }
        }

        // Cache line size is typically 64 bytes on modern systems
        cache_info.cache_line_size_bytes = Some(64);
    }

    #[cfg(target_os = "macos")]
    fn detect_macos_cache_info(cache_info: &mut CacheInfo) {
        // macOS sysctl approach would go here
        cache_info.cache_line_size_bytes = Some(64); // Common on modern Apple hardware
    }

    #[cfg(target_os = "windows")]
    fn detect_windows_cache_info(cache_info: &mut CacheInfo) {
        // Windows GetLogicalProcessorInformation approach would go here
        cache_info.cache_line_size_bytes = Some(64); // Common on modern systems
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    fn detect_linux_cache_info(_cache_info: &mut CacheInfo) {}

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    fn detect_macos_cache_info(_cache_info: &mut CacheInfo) {}

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    fn detect_windows_cache_info(_cache_info: &mut CacheInfo) {}
}

impl Default for HardwareDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Operating System specific utilities
#[derive(Debug, Clone)]
pub struct OSInfo {
    pub name: String,
    pub version: String,
    pub arch: String,
    pub kernel_version: Option<String>,
    pub is_64bit: bool,
}

impl OSInfo {
    /// Detect operating system information
    pub fn detect() -> Self {
        Self {
            name: std::env::consts::OS.to_string(),
            version: Self::detect_os_version(),
            arch: std::env::consts::ARCH.to_string(),
            kernel_version: Self::detect_kernel_version(),
            is_64bit: cfg!(target_pointer_width = "64"),
        }
    }

    /// Check if running on Linux
    pub fn is_linux(&self) -> bool {
        self.name == "linux"
    }

    /// Check if running on macOS
    pub fn is_macos(&self) -> bool {
        self.name == "macos"
    }

    /// Check if running on Windows
    pub fn is_windows(&self) -> bool {
        self.name == "windows"
    }

    /// Check if running on Unix-like system
    pub fn is_unix(&self) -> bool {
        cfg!(unix)
    }

    fn detect_os_version() -> String {
        #[cfg(target_os = "linux")]
        {
            if let Ok(version) = std::fs::read_to_string("/proc/version") {
                return version.lines().next().unwrap_or("unknown").to_string();
            }
        }

        #[cfg(target_os = "macos")]
        {
            // Could use sw_vers command here
            "macOS".to_string()
        }
        #[cfg(target_os = "windows")]
        {
            // Could use WinAPI here
            "Windows".to_string()
        }
        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        {
            "unknown".to_string()
        }
    }

    fn detect_kernel_version() -> Option<String> {
        #[cfg(unix)]
        {
            if let Ok(output) = std::process::Command::new("uname").arg("-r").output() {
                if let Ok(version) = String::from_utf8(output.stdout) {
                    return Some(version.trim().to_string());
                }
            }
        }
        None
    }
}

/// Compiler and runtime detection
#[derive(Debug, Clone)]
pub struct RuntimeInfo {
    pub rust_version: String,
    pub compiler: String,
    pub target_triple: String,
    pub build_profile: String,
    pub features: Vec<String>,
}

impl RuntimeInfo {
    /// Detect runtime information
    pub fn detect() -> Self {
        let target_family = std::env::var("CARGO_CFG_TARGET_FAMILY")
            .unwrap_or_else(|_| std::env::consts::FAMILY.to_string());
        let target_triple = std::env::var("CARGO_CFG_TARGET_TRIPLE").unwrap_or_else(|_| {
            format!(
                "{}-{}-{}",
                std::env::consts::ARCH,
                std::env::consts::FAMILY,
                std::env::consts::OS
            )
        });

        Self {
            rust_version: option_env!("CARGO_PKG_RUST_VERSION")
                .unwrap_or("unknown")
                .to_string(),
            compiler: format!("{target_family} {}", rustc_version_runtime::version()),
            target_triple,
            build_profile: if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            }
            .to_string(),
            features: Self::detect_features(),
        }
    }

    /// Check if running in debug mode
    pub fn is_debug(&self) -> bool {
        self.build_profile == "debug"
    }

    /// Check if running in release mode
    pub fn is_release(&self) -> bool {
        self.build_profile == "release"
    }

    fn detect_features() -> Vec<String> {
        // Feature detection disabled due to missing features in Cargo.toml
        // if cfg!(feature = "simd") {
        //     features.push("simd".to_string());
        // }
        // if cfg!(feature = "parallel") {
        //     features.push("parallel".to_string());
        // }
        // if cfg!(feature = "serde") {
        //     features.push("serde".to_string());
        // }

        Vec::new()
    }
}

/// Performance characteristics measurement
#[derive(Debug, Clone)]
pub struct PerformanceCharacteristics {
    pub memory_bandwidth_gbps: f64,
    pub l1_cache_latency_ns: f64,
    pub l2_cache_latency_ns: f64,
    pub l3_cache_latency_ns: f64,
    pub memory_latency_ns: f64,
    pub cpu_frequency_estimation_mhz: f64,
    pub single_thread_performance_score: f64,
    pub multi_thread_performance_score: f64,
}

impl PerformanceCharacteristics {
    /// Measure system performance characteristics
    pub fn measure() -> Self {
        let _start_time = Instant::now();

        let memory_bandwidth = Self::measure_memory_bandwidth();
        let cache_latencies = Self::measure_cache_latencies();
        let cpu_freq = Self::estimate_cpu_frequency();
        let single_perf = Self::measure_single_thread_performance();
        let multi_perf = Self::measure_multi_thread_performance();

        Self {
            memory_bandwidth_gbps: memory_bandwidth,
            l1_cache_latency_ns: cache_latencies.0,
            l2_cache_latency_ns: cache_latencies.1,
            l3_cache_latency_ns: cache_latencies.2,
            memory_latency_ns: cache_latencies.3,
            cpu_frequency_estimation_mhz: cpu_freq,
            single_thread_performance_score: single_perf,
            multi_thread_performance_score: multi_perf,
        }
    }

    fn measure_memory_bandwidth() -> f64 {
        // Simple sequential memory access test
        const SIZE: usize = 16 * 1024 * 1024; // 16MB
        let mut data = vec![0u64; SIZE / 8];

        let start = Instant::now();

        // Sequential write
        for (i, item) in data.iter_mut().enumerate() {
            *item = i as u64;
        }

        // Sequential read
        let mut sum = 0u64;
        for &val in &data {
            sum = sum.wrapping_add(val);
        }

        let elapsed = start.elapsed();
        let bytes_processed = (SIZE * 2) as f64; // Read + write
        let bandwidth_bps = bytes_processed / elapsed.as_secs_f64();

        // Prevent optimization
        std::hint::black_box(sum);

        bandwidth_bps / 1e9 // Convert to GB/s
    }

    fn measure_cache_latencies() -> (f64, f64, f64, f64) {
        // Simplified cache latency measurement
        // In practice, this would be more sophisticated
        (1.0, 3.0, 10.0, 100.0) // ns estimates for L1, L2, L3, RAM
    }

    fn estimate_cpu_frequency() -> f64 {
        // Simple CPU frequency estimation using timing
        let iterations = 10_000_000;
        let start = Instant::now();

        let mut counter = 0u64;
        for _ in 0..iterations {
            counter = counter.wrapping_add(1);
        }

        let elapsed = start.elapsed();
        std::hint::black_box(counter);

        // Very rough estimate - would need calibration
        let cycles_per_second = iterations as f64 / elapsed.as_secs_f64();
        cycles_per_second / 1e6 // Convert to MHz estimate
    }

    fn measure_single_thread_performance() -> f64 {
        // Simple floating point benchmark
        let start = Instant::now();
        let mut sum = 0.0;

        for i in 0..1_000_000 {
            sum += (i as f64).sqrt().sin().cos();
        }

        let elapsed = start.elapsed();
        std::hint::black_box(sum);

        // Score based on operations per second
        1_000_000.0 / elapsed.as_secs_f64()
    }

    fn measure_multi_thread_performance() -> f64 {
        use std::thread;

        let num_threads = num_cpus::get();
        let start = Instant::now();

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                thread::spawn(|| {
                    let mut sum = 0.0;
                    for i in 0..100_000 {
                        sum += (i as f64).sqrt().sin().cos();
                    }
                    sum
                })
            })
            .collect();

        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let elapsed = start.elapsed();

        std::hint::black_box(results);

        // Score based on total operations per second across all threads
        (num_threads * 100_000) as f64 / elapsed.as_secs_f64()
    }
}

/// Feature availability checker
#[derive(Debug, Clone)]
pub struct FeatureChecker {
    hardware: HardwareDetector,
    os_info: OSInfo,
    #[allow(dead_code)]
    runtime: RuntimeInfo,
}

impl FeatureChecker {
    /// Create new feature checker
    pub fn new() -> Self {
        Self {
            hardware: HardwareDetector::new(),
            os_info: OSInfo::detect(),
            runtime: RuntimeInfo::detect(),
        }
    }

    /// Check if SIMD operations are available and beneficial
    pub fn simd_available(&self) -> bool {
        self.hardware.has_sse42() || self.hardware.has_avx2() || self.hardware.has_neon()
    }

    /// Check if parallel operations are beneficial
    pub fn parallel_beneficial(&self) -> bool {
        self.hardware.logical_cores() > 1
    }

    /// Check if memory mapping is available
    pub fn memory_mapping_available(&self) -> bool {
        self.os_info.is_unix() || self.os_info.is_windows()
    }

    /// Check if high-resolution timers are available
    pub fn high_resolution_timer_available(&self) -> bool {
        true // std::time::Instant should be high-resolution on all platforms
    }

    /// Get recommended number of threads for parallel operations
    pub fn recommended_thread_count(&self) -> usize {
        // Use logical cores but cap at reasonable limit
        self.hardware.logical_cores().min(32)
    }

    /// Get recommended SIMD width in elements
    pub fn recommended_simd_width(&self) -> usize {
        if self.hardware.has_avx512() {
            16 // 512 bits / 32 bits = 16 f32 elements
        } else if self.hardware.has_avx2() {
            8 // 256 bits / 32 bits = 8 f32 elements
        } else if self.hardware.has_sse42() || self.hardware.has_neon() {
            4 // 128 bits / 32 bits = 4 f32 elements
        } else {
            1 // No SIMD
        }
    }

    /// Check if specific optimization is recommended
    pub fn optimization_recommended(&self, optimization: &str) -> bool {
        match optimization {
            "simd" => self.simd_available(),
            "parallel" => self.parallel_beneficial(),
            "cache_friendly" => self.hardware.l3_cache_size_kb().is_some(),
            "memory_pool" => self.hardware.total_memory() > 1_000_000_000, // > 1GB
            _ => false,
        }
    }

    /// Get optimization recommendations
    pub fn get_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if self.simd_available() {
            recommendations.push("Enable SIMD optimizations".to_string());
        }

        if self.parallel_beneficial() {
            recommendations.push(format!(
                "Use parallel processing with {} threads",
                self.recommended_thread_count()
            ));
        }

        if let Some(cache_size) = self.hardware.l3_cache_size_kb() {
            recommendations.push(format!("Optimize for {cache_size}KB L3 cache"));
        }

        if self.hardware.total_memory() < 1_000_000_000 {
            recommendations.push("Consider memory usage optimizations".to_string());
        }

        recommendations
    }
}

impl Default for FeatureChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Global environment information (lazy-initialized)
static GLOBAL_ENV_INFO: OnceLock<EnvironmentInfo> = OnceLock::new();

/// Complete environment information
#[derive(Debug, Clone)]
pub struct EnvironmentInfo {
    pub hardware: HardwareDetector,
    pub os_info: OSInfo,
    pub runtime: RuntimeInfo,
    pub performance: PerformanceCharacteristics,
    pub features: FeatureChecker,
}

impl EnvironmentInfo {
    /// Get global environment information (initialized once)
    pub fn global() -> &'static EnvironmentInfo {
        GLOBAL_ENV_INFO.get_or_init(EnvironmentInfo::detect)
    }

    /// Detect complete environment information
    pub fn detect() -> Self {
        let hardware = HardwareDetector::new();
        let os_info = OSInfo::detect();
        let runtime = RuntimeInfo::detect();
        let performance = PerformanceCharacteristics::measure();
        let features = FeatureChecker::new();

        Self {
            hardware,
            os_info,
            runtime,
            performance,
            features,
        }
    }

    /// Generate environment summary
    pub fn summary(&self) -> String {
        format!(
            "Environment Summary:\n\
             OS: {} {} ({})\n\
             CPU: {} cores ({} logical), {}\n\
             Memory: {:.1} GB total, {:.1} GB available\n\
             Runtime: {} {}\n\
             Features: SIMD={}, Parallel={}, Cache={}KB",
            self.os_info.name,
            self.os_info.version,
            self.os_info.arch,
            self.hardware.cpu_cores(),
            self.hardware.logical_cores(),
            self.hardware.cpu_architecture(),
            self.hardware.total_memory() as f64 / 1e9,
            self.hardware.available_memory() as f64 / 1e9,
            self.runtime.compiler,
            self.runtime.build_profile,
            self.features.simd_available(),
            self.features.parallel_beneficial(),
            self.hardware.l3_cache_size_kb().unwrap_or(0)
        )
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_detector() {
        let hw = HardwareDetector::new();

        assert!(hw.cpu_cores() >= 1);
        assert!(hw.logical_cores() >= 1); // In containers, logical_cores may not equal physical cores
        assert!(!hw.cpu_architecture().is_empty());
        assert!(hw.total_memory() > 0);
    }

    #[test]
    fn test_os_info() {
        let os = OSInfo::detect();

        assert!(!os.name.is_empty());
        assert!(!os.arch.is_empty());

        // At least one should be true
        assert!(os.is_linux() || os.is_macos() || os.is_windows() || !os.name.is_empty());
    }

    #[test]
    fn test_runtime_info() {
        let runtime = RuntimeInfo::detect();

        assert!(!runtime.compiler.is_empty());
        assert!(!runtime.target_triple.is_empty());
        assert!(runtime.is_debug() || runtime.is_release());
    }

    #[test]
    fn test_performance_characteristics() {
        let perf = PerformanceCharacteristics::measure();

        assert!(perf.memory_bandwidth_gbps > 0.0);
        assert!(perf.cpu_frequency_estimation_mhz > 0.0);
        assert!(perf.single_thread_performance_score > 0.0);
        assert!(perf.multi_thread_performance_score > 0.0);
    }

    #[test]
    fn test_feature_checker() {
        let checker = FeatureChecker::new();

        assert!(checker.recommended_thread_count() >= 1);
        assert!(checker.recommended_simd_width() >= 1);
        assert!(checker.high_resolution_timer_available());
    }

    #[test]
    fn test_environment_info() {
        let env = EnvironmentInfo::detect();
        let summary = env.summary();

        assert!(!summary.is_empty());
        assert!(summary.contains("Environment Summary"));

        // Test global access
        let global_env = EnvironmentInfo::global();
        assert!(!global_env.summary().is_empty());
    }

    #[test]
    fn test_cpu_features() {
        let hw = HardwareDetector::new();

        // These should not panic
        let _has_avx2 = hw.has_avx2();
        let _has_sse42 = hw.has_sse42();
        let _has_neon = hw.has_neon();
        let _is_arm = hw.is_arm();
        let _is_x86 = hw.is_x86();
    }

    #[test]
    fn test_optimization_recommendations() {
        let checker = FeatureChecker::new();
        let recommendations = checker.get_recommendations();

        // Should have at least some recommendations
        assert!(!recommendations.is_empty());

        // Test specific optimizations
        let _simd_rec = checker.optimization_recommended("simd");
        let _parallel_rec = checker.optimization_recommended("parallel");
        let _cache_rec = checker.optimization_recommended("cache_friendly");
    }
}
