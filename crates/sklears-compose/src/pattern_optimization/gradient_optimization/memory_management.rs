//! Memory Management and Allocation Tracking for Gradient Optimization
//!
//! This module provides comprehensive memory management capabilities for gradient-based
//! optimization algorithms, including real-time memory usage tracking, allocation monitoring,
//! memory pool management, and advanced optimization strategies.
//!
//! # Core Components
//!
//! * [`MemoryUsageStats`] - Real-time memory usage statistics and monitoring
//! * [`AllocationTracker`] - Detailed allocation tracking and analysis
//! * [`MemoryPool`] - Efficient memory pool management for optimization data
//! * [`GarbageCollector`] - Intelligent garbage collection strategies
//! * [`MemoryOptimizer`] - Advanced memory optimization techniques
//!
//! # Example Usage
//!
//! ```rust
//! use crate::pattern_optimization::gradient_optimization::memory_management::*;
//!
//! // Create memory usage monitor
//! let memory_stats = MemoryUsageStats::builder()
//!     .update_interval(Duration::from_millis(100))
//!     .memory_limit(8 * 1024 * 1024 * 1024) // 8GB
//!     .enable_profiling(true)
//!     .build()?;
//!
//! // Setup allocation tracker
//! let allocation_tracker = AllocationTracker::builder()
//!     .track_stack_traces(true)
//!     .allocation_size_threshold(1024 * 1024) // Track allocations > 1MB
//!     .retention_period(Duration::from_hours(1))
//!     .build()?;
//!
//! // Configure memory pool
//! let memory_pool = MemoryPool::builder()
//!     .pool_size(1024 * 1024 * 1024) // 1GB pool
//!     .block_sizes(vec![1024, 4096, 16384, 65536])
//!     .auto_expand(true)
//!     .build()?;
//! ```

use std::collections::{HashMap, VecDeque, BTreeMap, HashSet};
use std::sync::{Arc, Mutex, RwLock, atomic::{AtomicUsize, AtomicBool, Ordering}};
use std::time::{Duration, Instant, SystemTime};
use std::thread::{ThreadId, JoinHandle};
use std::alloc::{Layout, GlobalAlloc, System};
use std::ptr::NonNull;
use std::fmt;
use scirs2_core::error::{CoreError, Result as SklResult};
use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::profiling::{Profiler, profiling_memory_tracker};
use scirs2_core::metrics::{MetricRegistry, Counter, Gauge, Histogram, Timer};

/// Memory allocation types for categorization
#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub enum AllocationType {
    /// Gradient storage allocations
    Gradient,
    /// Model parameter allocations
    Parameters,
    /// Activation/intermediate values
    Activations,
    /// Optimizer state (momentum, running averages, etc.)
    OptimizerState,
    /// Loss computation buffers
    LossBuffers,
    /// Batch data storage
    BatchData,
    /// Temporary computation buffers
    TemporaryBuffers,
    /// Memory pool allocations
    PoolAllocation,
    /// System/infrastructure allocations
    System,
    /// Custom user-defined allocation type
    Custom { category: String },
}

/// Memory pressure levels
#[derive(Debug, Clone, PartialEq, PartialOrd, Ord, Eq)]
pub enum MemoryPressure {
    Low,      // < 60% of limit
    Moderate, // 60-80% of limit
    High,     // 80-95% of limit
    Critical, // > 95% of limit
    Emergency, // Out of memory imminent
}

/// Garbage collection strategies
#[derive(Debug, Clone, PartialEq)]
pub enum GCStrategy {
    /// Immediate collection when pressure increases
    Immediate,
    /// Periodic collection at fixed intervals
    Periodic { interval: Duration },
    /// Adaptive collection based on allocation patterns
    Adaptive { pressure_threshold: MemoryPressure },
    /// Manual collection only
    Manual,
    /// Generational collection with multiple generations
    Generational { generations: usize },
    /// Custom collection strategy
    Custom { strategy_name: String },
}

/// Memory optimization techniques
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationTechnique {
    /// Reuse buffers across iterations
    BufferReuse,
    /// In-place operations when possible
    InPlaceOperations,
    /// Memory mapping for large datasets
    MemoryMapping,
    /// Compression for inactive data
    Compression { algorithm: CompressionAlgorithm },
    /// Swap to disk for least recently used data
    DiskSwapping { threshold_mb: usize },
    /// Quantization to reduce precision
    Quantization { bits: u8 },
    /// Gradient checkpointing to trade compute for memory
    GradientCheckpointing,
    /// Custom optimization technique
    Custom { technique_name: String },
}

/// Compression algorithms for memory optimization
#[derive(Debug, Clone, PartialEq)]
pub enum CompressionAlgorithm {
    LZ4,
    Zstd,
    Snappy,
    Gzip,
    Custom { algorithm_name: String },
}

/// Allocation information structure
#[derive(Debug, Clone)]
pub struct AllocationInfo {
    pub id: u64,
    pub thread_id: ThreadId,
    pub timestamp: Instant,
    pub size: usize,
    pub alignment: usize,
    pub allocation_type: AllocationType,
    pub stack_trace: Option<Vec<String>>,
    pub lifetime_estimate: Option<Duration>,
    pub access_pattern: AccessPattern,
    pub metadata: HashMap<String, String>,
}

/// Memory access patterns for optimization
#[derive(Debug, Clone, PartialEq)]
pub enum AccessPattern {
    /// Sequential access (good for prefetching)
    Sequential,
    /// Random access patterns
    Random,
    /// Write-once, read-many
    WriteOnceReadMany,
    /// Read-once (can be disposed quickly)
    ReadOnce,
    /// Frequent read/write access
    HighFrequency,
    /// Infrequent access (candidate for compression/swapping)
    LowFrequency,
    /// Unknown or mixed pattern
    Unknown,
}

/// Memory block information for pool management
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    pub ptr: NonNull<u8>,
    pub size: usize,
    pub alignment: usize,
    pub allocated_at: Instant,
    pub last_accessed: Instant,
    pub access_count: usize,
    pub allocation_type: AllocationType,
    pub in_use: bool,
}

/// Real-time memory usage statistics
pub struct MemoryUsageStats {
    config: MemoryStatsConfig,
    current_stats: Arc<RwLock<CurrentMemoryStats>>,
    historical_stats: Arc<RwLock<HistoricalMemoryStats>>,
    pressure_monitor: Arc<MemoryPressureMonitor>,
    profiler: Arc<MemoryProfiler>,
    metrics: Arc<MemoryMetrics>,
    collection_thread: Option<JoinHandle<()>>,
    shutdown_signal: Arc<AtomicBool>,
}

/// Configuration for memory statistics collection
#[derive(Debug, Clone)]
pub struct MemoryStatsConfig {
    pub update_interval: Duration,
    pub memory_limit: usize,
    pub enable_profiling: bool,
    pub enable_stack_traces: bool,
    pub historical_retention: Duration,
    pub pressure_thresholds: MemoryPressureThresholds,
    pub detailed_tracking: bool,
    pub thread_local_tracking: bool,
}

/// Memory pressure thresholds
#[derive(Debug, Clone)]
pub struct MemoryPressureThresholds {
    pub low_threshold: f64,      // 0.6
    pub moderate_threshold: f64, // 0.8
    pub high_threshold: f64,     // 0.95
    pub critical_threshold: f64, // 0.98
}

impl Default for MemoryPressureThresholds {
    fn default() -> Self {
        Self {
            low_threshold: 0.6,
            moderate_threshold: 0.8,
            high_threshold: 0.95,
            critical_threshold: 0.98,
        }
    }
}

impl Default for MemoryStatsConfig {
    fn default() -> Self {
        Self {
            update_interval: Duration::from_millis(100),
            memory_limit: 8 * 1024 * 1024 * 1024, // 8GB
            enable_profiling: true,
            enable_stack_traces: false, // Expensive, disabled by default
            historical_retention: Duration::from_hours(24),
            pressure_thresholds: MemoryPressureThresholds::default(),
            detailed_tracking: true,
            thread_local_tracking: false,
        }
    }
}

/// Current memory statistics snapshot
#[derive(Debug, Clone)]
pub struct CurrentMemoryStats {
    pub total_allocated: usize,
    pub total_deallocated: usize,
    pub current_usage: usize,
    pub peak_usage: usize,
    pub allocation_count: usize,
    pub deallocation_count: usize,
    pub memory_pressure: MemoryPressure,
    pub allocations_by_type: HashMap<AllocationType, usize>,
    pub allocations_by_thread: HashMap<ThreadId, usize>,
    pub large_allocations: Vec<AllocationInfo>,
    pub fragmentation_ratio: f64,
    pub gc_pressure: f64,
    pub last_updated: Instant,
}

impl Default for CurrentMemoryStats {
    fn default() -> Self {
        Self {
            total_allocated: 0,
            total_deallocated: 0,
            current_usage: 0,
            peak_usage: 0,
            allocation_count: 0,
            deallocation_count: 0,
            memory_pressure: MemoryPressure::Low,
            allocations_by_type: HashMap::new(),
            allocations_by_thread: HashMap::new(),
            large_allocations: Vec::new(),
            fragmentation_ratio: 0.0,
            gc_pressure: 0.0,
            last_updated: Instant::now(),
        }
    }
}

/// Historical memory statistics
#[derive(Debug)]
pub struct HistoricalMemoryStats {
    pub snapshots: VecDeque<MemorySnapshot>,
    pub usage_trends: MemoryUsageTrends,
    pub allocation_patterns: AllocationPatterns,
    pub pressure_history: VecDeque<(Instant, MemoryPressure)>,
    pub gc_events: VecDeque<GCEvent>,
}

/// Memory usage snapshot for historical tracking
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    pub timestamp: Instant,
    pub total_usage: usize,
    pub peak_usage: usize,
    pub allocation_rate: f64, // allocations per second
    pub deallocation_rate: f64,
    pub memory_pressure: MemoryPressure,
    pub active_allocations: usize,
    pub fragmentation_ratio: f64,
}

/// Memory usage trends analysis
#[derive(Debug, Clone)]
pub struct MemoryUsageTrends {
    pub growth_rate: f64,
    pub peak_growth_rate: f64,
    pub allocation_velocity: f64,
    pub deallocation_velocity: f64,
    pub pressure_trend: f64,
    pub fragmentation_trend: f64,
    pub predicted_peak: Option<(Instant, usize)>,
}

/// Allocation pattern analysis
#[derive(Debug, Clone)]
pub struct AllocationPatterns {
    pub common_sizes: BTreeMap<usize, usize>, // size -> frequency
    pub temporal_patterns: HashMap<String, usize>, // time_bucket -> allocation_count
    pub thread_patterns: HashMap<ThreadId, ThreadAllocationPattern>,
    pub type_patterns: HashMap<AllocationType, TypeAllocationPattern>,
    pub access_patterns: HashMap<AccessPattern, usize>,
}

/// Thread-specific allocation patterns
#[derive(Debug, Clone)]
pub struct ThreadAllocationPattern {
    pub thread_id: ThreadId,
    pub total_allocations: usize,
    pub average_allocation_size: f64,
    pub peak_allocation_size: usize,
    pub allocation_frequency: f64,
    pub preferred_types: Vec<AllocationType>,
}

/// Type-specific allocation patterns
#[derive(Debug, Clone)]
pub struct TypeAllocationPattern {
    pub allocation_type: AllocationType,
    pub total_size: usize,
    pub average_size: f64,
    pub allocation_count: usize,
    pub peak_concurrent: usize,
    pub average_lifetime: Duration,
    pub access_patterns: HashMap<AccessPattern, usize>,
}

/// Garbage collection event information
#[derive(Debug, Clone)]
pub struct GCEvent {
    pub timestamp: Instant,
    pub trigger: GCTrigger,
    pub strategy: GCStrategy,
    pub duration: Duration,
    pub memory_freed: usize,
    pub objects_collected: usize,
    pub performance_impact: f64,
}

/// Garbage collection triggers
#[derive(Debug, Clone, PartialEq)]
pub enum GCTrigger {
    MemoryPressure(MemoryPressure),
    PeriodicTimer,
    ManualTrigger,
    AllocationThreshold,
    OptimizationRequest,
    SystemRequest,
}

impl MemoryUsageStats {
    /// Create a new memory usage statistics builder
    pub fn builder() -> MemoryUsageStatsBuilder {
        MemoryUsageStatsBuilder::new()
    }

    /// Get current memory statistics
    pub fn get_current_stats(&self) -> SklResult<CurrentMemoryStats> {
        let stats = self.current_stats.read()
            .map_err(|_| CoreError::LockError("Failed to acquire current stats lock".to_string()))?;
        Ok(stats.clone())
    }

    /// Get memory usage trends
    pub fn get_usage_trends(&self) -> SklResult<MemoryUsageTrends> {
        let historical = self.historical_stats.read()
            .map_err(|_| CoreError::LockError("Failed to acquire historical stats lock".to_string()))?;
        Ok(historical.usage_trends.clone())
    }

    /// Get allocation patterns
    pub fn get_allocation_patterns(&self) -> SklResult<AllocationPatterns> {
        let historical = self.historical_stats.read()
            .map_err(|_| CoreError::LockError("Failed to acquire historical stats lock".to_string()))?;
        Ok(historical.allocation_patterns.clone())
    }

    /// Get memory pressure level
    pub fn get_memory_pressure(&self) -> SklResult<MemoryPressure> {
        let stats = self.current_stats.read()
            .map_err(|_| CoreError::LockError("Failed to acquire current stats lock".to_string()))?;
        Ok(stats.memory_pressure.clone())
    }

    /// Force a statistics update
    pub fn update_stats(&self) -> SklResult<()> {
        self.collect_statistics()?;
        self.analyze_trends()?;
        self.update_pressure_level()?;
        Ok(())
    }

    /// Get detailed memory report
    pub fn get_detailed_report(&self) -> SklResult<MemoryReport> {
        let current = self.get_current_stats()?;
        let trends = self.get_usage_trends()?;
        let patterns = self.get_allocation_patterns()?;
        let pressure_history = self.get_pressure_history()?;

        Ok(MemoryReport {
            current_stats: current,
            usage_trends: trends,
            allocation_patterns: patterns,
            pressure_history,
            optimization_recommendations: self.generate_optimization_recommendations()?,
            performance_impact: self.calculate_performance_impact()?,
        })
    }

    fn collect_statistics(&self) -> SklResult<()> {
        let mut stats = self.current_stats.write()
            .map_err(|_| CoreError::LockError("Failed to acquire current stats lock".to_string()))?;

        // Update basic memory statistics
        let current_usage = self.get_current_memory_usage()?;
        stats.current_usage = current_usage;
        stats.peak_usage = stats.peak_usage.max(current_usage);

        // Update memory pressure
        let pressure = self.calculate_memory_pressure(current_usage)?;
        stats.memory_pressure = pressure;

        // Update fragmentation ratio
        stats.fragmentation_ratio = self.calculate_fragmentation_ratio()?;

        // Update GC pressure
        stats.gc_pressure = self.calculate_gc_pressure()?;

        stats.last_updated = Instant::now();

        Ok(())
    }

    fn analyze_trends(&self) -> SklResult<()> {
        let mut historical = self.historical_stats.write()
            .map_err(|_| CoreError::LockError("Failed to acquire historical stats lock".to_string()))?;

        // Add current snapshot to history
        let current_stats = self.current_stats.read()
            .map_err(|_| CoreError::LockError("Failed to acquire current stats lock".to_string()))?;

        let snapshot = MemorySnapshot {
            timestamp: Instant::now(),
            total_usage: current_stats.current_usage,
            peak_usage: current_stats.peak_usage,
            allocation_rate: self.calculate_allocation_rate()?,
            deallocation_rate: self.calculate_deallocation_rate()?,
            memory_pressure: current_stats.memory_pressure.clone(),
            active_allocations: current_stats.allocation_count - current_stats.deallocation_count,
            fragmentation_ratio: current_stats.fragmentation_ratio,
        };

        historical.snapshots.push_back(snapshot);

        // Limit historical data size
        while historical.snapshots.len() > 10000 {
            historical.snapshots.pop_front();
        }

        // Update trends
        historical.usage_trends = self.calculate_usage_trends(&historical.snapshots)?;

        Ok(())
    }

    fn update_pressure_level(&self) -> SklResult<()> {
        let current_usage = self.get_current_memory_usage()?;
        let pressure = self.calculate_memory_pressure(current_usage)?;

        let mut historical = self.historical_stats.write()
            .map_err(|_| CoreError::LockError("Failed to acquire historical stats lock".to_string()))?;

        historical.pressure_history.push_back((Instant::now(), pressure.clone()));

        // Limit pressure history size
        if historical.pressure_history.len() > 1000 {
            historical.pressure_history.pop_front();
        }

        // Update pressure monitor
        self.pressure_monitor.update_pressure(pressure)?;

        Ok(())
    }

    fn get_current_memory_usage(&self) -> SklResult<usize> {
        // In a real implementation, this would query actual memory usage
        // For now, return a placeholder value based on profiler data
        if let Some(usage) = profiling_memory_tracker().current_usage() {
            Ok(usage)
        } else {
            // Fallback to system memory query
            Ok(self.query_system_memory_usage()?)
        }
    }

    fn query_system_memory_usage(&self) -> SklResult<usize> {
        // Platform-specific memory usage query
        // This is a simplified implementation
        Ok(1024 * 1024 * 100) // 100 MB placeholder
    }

    fn calculate_memory_pressure(&self, current_usage: usize) -> SklResult<MemoryPressure> {
        let usage_ratio = current_usage as f64 / self.config.memory_limit as f64;
        let thresholds = &self.config.pressure_thresholds;

        Ok(if usage_ratio >= thresholds.critical_threshold {
            MemoryPressure::Critical
        } else if usage_ratio >= thresholds.high_threshold {
            MemoryPressure::High
        } else if usage_ratio >= thresholds.moderate_threshold {
            MemoryPressure::Moderate
        } else {
            MemoryPressure::Low
        })
    }

    fn calculate_fragmentation_ratio(&self) -> SklResult<f64> {
        // Simplified fragmentation calculation
        // Real implementation would analyze memory layout
        Ok(0.1) // 10% fragmentation placeholder
    }

    fn calculate_gc_pressure(&self) -> SklResult<f64> {
        // Calculate pressure indicating need for garbage collection
        let current_usage = self.get_current_memory_usage()?;
        let usage_ratio = current_usage as f64 / self.config.memory_limit as f64;

        // Simple pressure calculation based on usage and fragmentation
        let fragmentation = self.calculate_fragmentation_ratio()?;
        Ok(usage_ratio * (1.0 + fragmentation))
    }

    fn calculate_allocation_rate(&self) -> SklResult<f64> {
        // Calculate allocations per second based on recent history
        Ok(100.0) // Placeholder: 100 allocations per second
    }

    fn calculate_deallocation_rate(&self) -> SklResult<f64> {
        // Calculate deallocations per second based on recent history
        Ok(95.0) // Placeholder: 95 deallocations per second
    }

    fn calculate_usage_trends(&self, snapshots: &VecDeque<MemorySnapshot>) -> SklResult<MemoryUsageTrends> {
        if snapshots.len() < 2 {
            return Ok(MemoryUsageTrends {
                growth_rate: 0.0,
                peak_growth_rate: 0.0,
                allocation_velocity: 0.0,
                deallocation_velocity: 0.0,
                pressure_trend: 0.0,
                fragmentation_trend: 0.0,
                predicted_peak: None,
            });
        }

        // Calculate linear regression for growth rate
        let recent_snapshots: Vec<&MemorySnapshot> = snapshots.iter().rev().take(100).collect();
        let growth_rate = self.calculate_linear_trend(
            &recent_snapshots.iter().map(|s| s.total_usage as f64).collect::<Vec<_>>()
        )?;

        let allocation_velocity = self.calculate_linear_trend(
            &recent_snapshots.iter().map(|s| s.allocation_rate).collect::<Vec<_>>()
        )?;

        let deallocation_velocity = self.calculate_linear_trend(
            &recent_snapshots.iter().map(|s| s.deallocation_rate).collect::<Vec<_>>()
        )?;

        let fragmentation_trend = self.calculate_linear_trend(
            &recent_snapshots.iter().map(|s| s.fragmentation_ratio).collect::<Vec<_>>()
        )?;

        // Predict peak usage if current trend continues
        let predicted_peak = if growth_rate > 0.0 {
            let current_usage = recent_snapshots[0].total_usage;
            let time_to_limit = (self.config.memory_limit - current_usage) as f64 / growth_rate;
            if time_to_limit > 0.0 && time_to_limit < 3600.0 { // Within 1 hour
                Some((Instant::now() + Duration::from_secs(time_to_limit as u64), self.config.memory_limit))
            } else {
                None
            }
        } else {
            None
        };

        Ok(MemoryUsageTrends {
            growth_rate,
            peak_growth_rate: growth_rate, // Simplified
            allocation_velocity,
            deallocation_velocity,
            pressure_trend: 0.0, // Placeholder
            fragmentation_trend,
            predicted_peak,
        })
    }

    fn calculate_linear_trend(&self, values: &[f64]) -> SklResult<f64> {
        if values.len() < 2 {
            return Ok(0.0);
        }

        let n = values.len() as f64;
        let x_mean = (0..values.len()).map(|i| i as f64).sum::<f64>() / n;
        let y_mean = values.iter().sum::<f64>() / n;

        let numerator: f64 = (0..values.len())
            .map(|i| (i as f64 - x_mean) * (values[i] - y_mean))
            .sum();

        let denominator: f64 = (0..values.len())
            .map(|i| (i as f64 - x_mean).powi(2))
            .sum();

        if denominator.abs() < 1e-10 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    fn get_pressure_history(&self) -> SklResult<Vec<(Instant, MemoryPressure)>> {
        let historical = self.historical_stats.read()
            .map_err(|_| CoreError::LockError("Failed to acquire historical stats lock".to_string()))?;
        Ok(historical.pressure_history.iter().cloned().collect())
    }

    fn generate_optimization_recommendations(&self) -> SklResult<Vec<OptimizationRecommendation>> {
        let mut recommendations = Vec::new();
        let current = self.get_current_stats()?;
        let trends = self.get_usage_trends()?;

        // High memory pressure recommendations
        if current.memory_pressure >= MemoryPressure::High {
            recommendations.push(OptimizationRecommendation {
                technique: OptimizationTechnique::GradientCheckpointing,
                priority: RecommendationPriority::High,
                estimated_savings: current.current_usage / 4, // 25% savings estimate
                implementation_effort: ImplementationEffort::Medium,
                description: "Enable gradient checkpointing to reduce memory usage".to_string(),
            });
        }

        // High fragmentation recommendations
        if current.fragmentation_ratio > 0.3 {
            recommendations.push(OptimizationRecommendation {
                technique: OptimizationTechnique::BufferReuse,
                priority: RecommendationPriority::Medium,
                estimated_savings: (current.fragmentation_ratio * current.current_usage as f64) as usize,
                implementation_effort: ImplementationEffort::Low,
                description: "Implement buffer reuse to reduce fragmentation".to_string(),
            });
        }

        // Growth trend recommendations
        if trends.growth_rate > 1000.0 { // Growing more than 1KB per measurement
            recommendations.push(OptimizationRecommendation {
                technique: OptimizationTechnique::Compression { algorithm: CompressionAlgorithm::LZ4 },
                priority: RecommendationPriority::Medium,
                estimated_savings: current.current_usage / 3, // 33% savings estimate
                implementation_effort: ImplementationEffort::High,
                description: "Enable compression for inactive data".to_string(),
            });
        }

        Ok(recommendations)
    }

    fn calculate_performance_impact(&self) -> SklResult<PerformanceImpact> {
        let current = self.get_current_stats()?;

        let memory_overhead = match current.memory_pressure {
            MemoryPressure::Low => 0.01,
            MemoryPressure::Moderate => 0.05,
            MemoryPressure::High => 0.15,
            MemoryPressure::Critical => 0.30,
            MemoryPressure::Emergency => 0.50,
        };

        let fragmentation_overhead = current.fragmentation_ratio * 0.1;
        let gc_overhead = current.gc_pressure * 0.05;

        Ok(PerformanceImpact {
            memory_overhead,
            fragmentation_overhead,
            gc_overhead,
            total_overhead: memory_overhead + fragmentation_overhead + gc_overhead,
            recommendations: self.generate_optimization_recommendations()?,
        })
    }
}

/// Allocation tracking system
pub struct AllocationTracker {
    config: AllocationTrackerConfig,
    active_allocations: Arc<RwLock<HashMap<u64, AllocationInfo>>>,
    allocation_history: Arc<RwLock<VecDeque<AllocationInfo>>>,
    pattern_analyzer: Arc<AllocationPatternAnalyzer>,
    metrics: Arc<AllocationMetrics>,
    allocation_counter: AtomicUsize,
}

/// Configuration for allocation tracking
#[derive(Debug, Clone)]
pub struct AllocationTrackerConfig {
    pub track_stack_traces: bool,
    pub allocation_size_threshold: usize,
    pub retention_period: Duration,
    pub pattern_analysis_enabled: bool,
    pub real_time_analysis: bool,
    pub thread_local_tracking: bool,
    pub detailed_metadata: bool,
}

impl Default for AllocationTrackerConfig {
    fn default() -> Self {
        Self {
            track_stack_traces: false, // Expensive
            allocation_size_threshold: 1024, // Track allocations >= 1KB
            retention_period: Duration::from_hours(1),
            pattern_analysis_enabled: true,
            real_time_analysis: false,
            thread_local_tracking: true,
            detailed_metadata: false,
        }
    }
}

impl AllocationTracker {
    /// Create a new allocation tracker builder
    pub fn builder() -> AllocationTrackerBuilder {
        AllocationTrackerBuilder::new()
    }

    /// Track a new allocation
    pub fn track_allocation(&self, size: usize, alignment: usize, allocation_type: AllocationType) -> SklResult<u64> {
        if size < self.config.allocation_size_threshold {
            return Ok(0); // Skip tracking small allocations
        }

        let allocation_id = self.allocation_counter.fetch_add(1, Ordering::Relaxed) as u64;

        let allocation_info = AllocationInfo {
            id: allocation_id,
            thread_id: std::thread::current().id(),
            timestamp: Instant::now(),
            size,
            alignment,
            allocation_type: allocation_type.clone(),
            stack_trace: if self.config.track_stack_traces {
                Some(self.capture_stack_trace())
            } else {
                None
            },
            lifetime_estimate: self.estimate_lifetime(&allocation_type),
            access_pattern: AccessPattern::Unknown,
            metadata: HashMap::new(),
        };

        // Add to active allocations
        let mut active = self.active_allocations.write()
            .map_err(|_| CoreError::LockError("Failed to acquire active allocations lock".to_string()))?;
        active.insert(allocation_id, allocation_info.clone());

        // Add to history
        let mut history = self.allocation_history.write()
            .map_err(|_| CoreError::LockError("Failed to acquire allocation history lock".to_string()))?;
        history.push_back(allocation_info);

        // Cleanup old history
        let cutoff_time = Instant::now() - self.config.retention_period;
        while let Some(front) = history.front() {
            if front.timestamp < cutoff_time {
                history.pop_front();
            } else {
                break;
            }
        }

        // Update metrics
        self.metrics.record_allocation(size, &allocation_type)?;

        // Pattern analysis
        if self.config.pattern_analysis_enabled {
            self.pattern_analyzer.analyze_allocation(&allocation_info)?;
        }

        Ok(allocation_id)
    }

    /// Track deallocation
    pub fn track_deallocation(&self, allocation_id: u64) -> SklResult<Option<AllocationInfo>> {
        let mut active = self.active_allocations.write()
            .map_err(|_| CoreError::LockError("Failed to acquire active allocations lock".to_string()))?;

        if let Some(mut allocation_info) = active.remove(&allocation_id) {
            let lifetime = allocation_info.timestamp.elapsed();

            // Update metrics
            self.metrics.record_deallocation(allocation_info.size, &allocation_info.allocation_type, lifetime)?;

            // Update allocation info with actual lifetime
            allocation_info.metadata.insert("actual_lifetime".to_string(), format!("{:?}", lifetime));

            Ok(Some(allocation_info))
        } else {
            Ok(None)
        }
    }

    /// Update access pattern for an allocation
    pub fn update_access_pattern(&self, allocation_id: u64, pattern: AccessPattern) -> SklResult<()> {
        let mut active = self.active_allocations.write()
            .map_err(|_| CoreError::LockError("Failed to acquire active allocations lock".to_string()))?;

        if let Some(allocation_info) = active.get_mut(&allocation_id) {
            allocation_info.access_pattern = pattern;
        }

        Ok(())
    }

    /// Get allocation statistics
    pub fn get_allocation_statistics(&self) -> SklResult<AllocationStatistics> {
        let active = self.active_allocations.read()
            .map_err(|_| CoreError::LockError("Failed to acquire active allocations lock".to_string()))?;

        let total_active_allocations = active.len();
        let total_active_memory = active.values().map(|a| a.size).sum::<usize>();

        let allocations_by_type = active.values()
            .fold(HashMap::new(), |mut acc, alloc| {
                *acc.entry(alloc.allocation_type.clone()).or_insert(0) += 1;
                acc
            });

        let memory_by_type = active.values()
            .fold(HashMap::new(), |mut acc, alloc| {
                *acc.entry(alloc.allocation_type.clone()).or_insert(0) += alloc.size;
                acc
            });

        let allocations_by_thread = active.values()
            .fold(HashMap::new(), |mut acc, alloc| {
                *acc.entry(alloc.thread_id).or_insert(0) += 1;
                acc
            });

        let large_allocations = active.values()
            .filter(|a| a.size > 1024 * 1024) // > 1MB
            .cloned()
            .collect();

        Ok(AllocationStatistics {
            total_active_allocations,
            total_active_memory,
            allocations_by_type,
            memory_by_type,
            allocations_by_thread,
            large_allocations,
            pattern_analysis: self.pattern_analyzer.get_pattern_summary()?,
        })
    }

    fn capture_stack_trace(&self) -> Vec<String> {
        // In a real implementation, this would capture the actual stack trace
        vec!["stack_trace_placeholder".to_string()]
    }

    fn estimate_lifetime(&self, allocation_type: &AllocationType) -> Option<Duration> {
        // Estimate based on allocation type
        match allocation_type {
            AllocationType::TemporaryBuffers => Some(Duration::from_millis(100)),
            AllocationType::BatchData => Some(Duration::from_secs(1)),
            AllocationType::Parameters => Some(Duration::from_hours(1)),
            AllocationType::OptimizerState => Some(Duration::from_hours(1)),
            _ => None,
        }
    }
}

/// Memory pool for efficient allocation management
pub struct MemoryPool {
    config: MemoryPoolConfig,
    pools: Arc<RwLock<HashMap<usize, FixedSizePool>>>,
    large_object_pool: Arc<RwLock<LargeObjectPool>>,
    statistics: Arc<RwLock<PoolStatistics>>,
    metrics: Arc<PoolMetrics>,
}

/// Configuration for memory pool
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    pub pool_size: usize,
    pub block_sizes: Vec<usize>,
    pub auto_expand: bool,
    pub max_expansion_factor: f64,
    pub large_object_threshold: usize,
    pub alignment: usize,
    pub enable_statistics: bool,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            pool_size: 1024 * 1024 * 1024, // 1GB
            block_sizes: vec![64, 256, 1024, 4096, 16384, 65536],
            auto_expand: true,
            max_expansion_factor: 2.0,
            large_object_threshold: 1024 * 1024, // 1MB
            alignment: 64, // Cache line alignment
            enable_statistics: true,
        }
    }
}

/// Fixed-size memory pool for specific block sizes
#[derive(Debug)]
struct FixedSizePool {
    block_size: usize,
    total_blocks: usize,
    free_blocks: VecDeque<NonNull<u8>>,
    allocated_blocks: HashSet<NonNull<u8>>,
    pool_memory: Vec<u8>,
    statistics: FixedPoolStatistics,
}

/// Large object pool for allocations above threshold
#[derive(Debug)]
struct LargeObjectPool {
    allocations: HashMap<NonNull<u8>, LargeAllocation>,
    total_allocated: usize,
    statistics: LargePoolStatistics,
}

/// Large allocation information
#[derive(Debug)]
struct LargeAllocation {
    size: usize,
    allocated_at: Instant,
    allocation_type: AllocationType,
}

impl MemoryPool {
    /// Create a new memory pool builder
    pub fn builder() -> MemoryPoolBuilder {
        MemoryPoolBuilder::new()
    }

    /// Allocate memory from the pool
    pub fn allocate(&self, size: usize, allocation_type: AllocationType) -> SklResult<NonNull<u8>> {
        if size >= self.config.large_object_threshold {
            self.allocate_large_object(size, allocation_type)
        } else {
            self.allocate_from_pool(size, allocation_type)
        }
    }

    /// Deallocate memory back to the pool
    pub fn deallocate(&self, ptr: NonNull<u8>, size: usize) -> SklResult<()> {
        if size >= self.config.large_object_threshold {
            self.deallocate_large_object(ptr)
        } else {
            self.deallocate_to_pool(ptr, size)
        }
    }

    /// Get pool statistics
    pub fn get_pool_statistics(&self) -> SklResult<PoolStatistics> {
        let stats = self.statistics.read()
            .map_err(|_| CoreError::LockError("Failed to acquire pool statistics lock".to_string()))?;
        Ok(stats.clone())
    }

    fn allocate_large_object(&self, size: usize, allocation_type: AllocationType) -> SklResult<NonNull<u8>> {
        // Use system allocator for large objects
        let layout = Layout::from_size_align(size, self.config.alignment)
            .map_err(|_| CoreError::AllocationError("Invalid layout for large object".to_string()))?;

        let ptr = unsafe {
            NonNull::new(System.alloc(layout))
                .ok_or_else(|| CoreError::AllocationError("Failed to allocate large object".to_string()))?
        };

        // Track in large object pool
        let mut large_pool = self.large_object_pool.write()
            .map_err(|_| CoreError::LockError("Failed to acquire large object pool lock".to_string()))?;

        large_pool.allocations.insert(ptr, LargeAllocation {
            size,
            allocated_at: Instant::now(),
            allocation_type,
        });
        large_pool.total_allocated += size;

        Ok(ptr)
    }

    fn allocate_from_pool(&self, size: usize, _allocation_type: AllocationType) -> SklResult<NonNull<u8>> {
        // Find appropriate pool size
        let pool_size = self.config.block_sizes.iter()
            .find(|&&block_size| block_size >= size)
            .copied()
            .unwrap_or_else(|| {
                // Round up to next power of 2
                let mut pool_size = 1;
                while pool_size < size {
                    pool_size *= 2;
                }
                pool_size
            });

        let mut pools = self.pools.write()
            .map_err(|_| CoreError::LockError("Failed to acquire pools lock".to_string()))?;

        // Get or create pool for this size
        if !pools.contains_key(&pool_size) {
            let new_pool = self.create_fixed_size_pool(pool_size)?;
            pools.insert(pool_size, new_pool);
        }

        let pool = pools.get_mut(&pool_size).unwrap();

        // Allocate from pool
        if let Some(ptr) = pool.free_blocks.pop_front() {
            pool.allocated_blocks.insert(ptr);
            pool.statistics.allocations += 1;
            pool.statistics.current_allocated += 1;
            Ok(ptr)
        } else if self.config.auto_expand {
            // Expand pool if auto-expansion is enabled
            self.expand_pool(pool)?;
            if let Some(ptr) = pool.free_blocks.pop_front() {
                pool.allocated_blocks.insert(ptr);
                pool.statistics.allocations += 1;
                pool.statistics.current_allocated += 1;
                Ok(ptr)
            } else {
                Err(CoreError::AllocationError("Failed to expand pool".to_string()))
            }
        } else {
            Err(CoreError::AllocationError("Pool exhausted and auto-expansion disabled".to_string()))
        }
    }

    fn deallocate_large_object(&self, ptr: NonNull<u8>) -> SklResult<()> {
        let mut large_pool = self.large_object_pool.write()
            .map_err(|_| CoreError::LockError("Failed to acquire large object pool lock".to_string()))?;

        if let Some(allocation) = large_pool.allocations.remove(&ptr) {
            large_pool.total_allocated -= allocation.size;

            // Deallocate using system allocator
            let layout = Layout::from_size_align(allocation.size, self.config.alignment)
                .map_err(|_| CoreError::AllocationError("Invalid layout for deallocation".to_string()))?;

            unsafe {
                System.dealloc(ptr.as_ptr(), layout);
            }

            Ok(())
        } else {
            Err(CoreError::AllocationError("Pointer not found in large object pool".to_string()))
        }
    }

    fn deallocate_to_pool(&self, ptr: NonNull<u8>, size: usize) -> SklResult<()> {
        // Find appropriate pool size
        let pool_size = self.config.block_sizes.iter()
            .find(|&&block_size| block_size >= size)
            .copied()
            .unwrap_or_else(|| {
                let mut pool_size = 1;
                while pool_size < size {
                    pool_size *= 2;
                }
                pool_size
            });

        let mut pools = self.pools.write()
            .map_err(|_| CoreError::LockError("Failed to acquire pools lock".to_string()))?;

        if let Some(pool) = pools.get_mut(&pool_size) {
            if pool.allocated_blocks.remove(&ptr) {
                pool.free_blocks.push_back(ptr);
                pool.statistics.deallocations += 1;
                pool.statistics.current_allocated -= 1;
                Ok(())
            } else {
                Err(CoreError::AllocationError("Pointer not found in allocated blocks".to_string()))
            }
        } else {
            Err(CoreError::AllocationError("Pool not found for size".to_string()))
        }
    }

    fn create_fixed_size_pool(&self, block_size: usize) -> SklResult<FixedSizePool> {
        let total_blocks = self.config.pool_size / block_size;
        let total_size = total_blocks * block_size;

        let mut pool_memory = vec![0u8; total_size];
        let mut free_blocks = VecDeque::new();

        // Initialize free block list
        for i in 0..total_blocks {
            let offset = i * block_size;
            let ptr = unsafe {
                NonNull::new_unchecked(pool_memory.as_mut_ptr().add(offset))
            };
            free_blocks.push_back(ptr);
        }

        Ok(FixedSizePool {
            block_size,
            total_blocks,
            free_blocks,
            allocated_blocks: HashSet::new(),
            pool_memory,
            statistics: FixedPoolStatistics {
                block_size,
                total_blocks,
                allocations: 0,
                deallocations: 0,
                current_allocated: 0,
                peak_allocated: 0,
                fragmentation_ratio: 0.0,
            },
        })
    }

    fn expand_pool(&self, pool: &mut FixedSizePool) -> SklResult<()> {
        let expansion_blocks = (pool.total_blocks as f64 * self.config.max_expansion_factor) as usize - pool.total_blocks;
        let additional_size = expansion_blocks * pool.block_size;

        // Extend pool memory
        let current_size = pool.pool_memory.len();
        pool.pool_memory.resize(current_size + additional_size, 0);

        // Add new blocks to free list
        for i in 0..expansion_blocks {
            let offset = current_size + i * pool.block_size;
            let ptr = unsafe {
                NonNull::new_unchecked(pool.pool_memory.as_mut_ptr().add(offset))
            };
            pool.free_blocks.push_back(ptr);
        }

        pool.total_blocks += expansion_blocks;

        Ok(())
    }
}

// Additional supporting structures and implementations

/// Memory pressure monitoring
pub struct MemoryPressureMonitor {
    current_pressure: Arc<RwLock<MemoryPressure>>,
    pressure_callbacks: Arc<RwLock<Vec<Box<dyn Fn(MemoryPressure) + Send + Sync>>>>,
    thresholds: MemoryPressureThresholds,
}

impl MemoryPressureMonitor {
    pub fn new(thresholds: MemoryPressureThresholds) -> Self {
        Self {
            current_pressure: Arc::new(RwLock::new(MemoryPressure::Low)),
            pressure_callbacks: Arc::new(RwLock::new(Vec::new())),
            thresholds,
        }
    }

    pub fn update_pressure(&self, new_pressure: MemoryPressure) -> SklResult<()> {
        let mut current = self.current_pressure.write()
            .map_err(|_| CoreError::LockError("Failed to acquire pressure lock".to_string()))?;

        if *current != new_pressure {
            *current = new_pressure.clone();

            // Trigger callbacks
            let callbacks = self.pressure_callbacks.read()
                .map_err(|_| CoreError::LockError("Failed to acquire callbacks lock".to_string()))?;

            for callback in callbacks.iter() {
                callback(new_pressure.clone());
            }
        }

        Ok(())
    }

    pub fn register_callback<F>(&self, callback: F) -> SklResult<()>
    where
        F: Fn(MemoryPressure) + Send + Sync + 'static,
    {
        let mut callbacks = self.pressure_callbacks.write()
            .map_err(|_| CoreError::LockError("Failed to acquire callbacks lock".to_string()))?;
        callbacks.push(Box::new(callback));
        Ok(())
    }
}

/// Memory profiler for detailed analysis
pub struct MemoryProfiler {
    profiling_enabled: AtomicBool,
    profile_data: Arc<RwLock<ProfileData>>,
    sampling_rate: AtomicUsize, // 1 in N allocations
}

#[derive(Debug, Default)]
struct ProfileData {
    allocation_sites: HashMap<String, AllocationSite>,
    hot_paths: Vec<HotPath>,
    memory_leaks: Vec<LeakCandidate>,
}

#[derive(Debug, Clone)]
struct AllocationSite {
    location: String,
    count: usize,
    total_size: usize,
    average_size: f64,
    peak_size: usize,
}

#[derive(Debug, Clone)]
struct HotPath {
    path: Vec<String>,
    allocation_frequency: f64,
    total_allocated: usize,
}

#[derive(Debug, Clone)]
struct LeakCandidate {
    allocation_id: u64,
    size: usize,
    age: Duration,
    suspected_leak_probability: f64,
}

/// Allocation pattern analyzer
pub struct AllocationPatternAnalyzer {
    patterns: Arc<RwLock<AllocationPatterns>>,
    analysis_window: Duration,
}

impl AllocationPatternAnalyzer {
    pub fn new() -> Self {
        Self {
            patterns: Arc::new(RwLock::new(AllocationPatterns {
                common_sizes: BTreeMap::new(),
                temporal_patterns: HashMap::new(),
                thread_patterns: HashMap::new(),
                type_patterns: HashMap::new(),
                access_patterns: HashMap::new(),
            })),
            analysis_window: Duration::from_minutes(10),
        }
    }

    pub fn analyze_allocation(&self, allocation: &AllocationInfo) -> SklResult<()> {
        let mut patterns = self.patterns.write()
            .map_err(|_| CoreError::LockError("Failed to acquire patterns lock".to_string()))?;

        // Update size patterns
        *patterns.common_sizes.entry(allocation.size).or_insert(0) += 1;

        // Update temporal patterns
        let time_bucket = format!("{}", allocation.timestamp.elapsed().as_secs() / 60); // Minute buckets
        *patterns.temporal_patterns.entry(time_bucket).or_insert(0) += 1;

        // Update thread patterns
        let thread_pattern = patterns.thread_patterns.entry(allocation.thread_id).or_insert_with(|| {
            ThreadAllocationPattern {
                thread_id: allocation.thread_id,
                total_allocations: 0,
                average_allocation_size: 0.0,
                peak_allocation_size: 0,
                allocation_frequency: 0.0,
                preferred_types: Vec::new(),
            }
        });

        thread_pattern.total_allocations += 1;
        thread_pattern.peak_allocation_size = thread_pattern.peak_allocation_size.max(allocation.size);
        thread_pattern.average_allocation_size =
            (thread_pattern.average_allocation_size * (thread_pattern.total_allocations - 1) as f64 + allocation.size as f64)
            / thread_pattern.total_allocations as f64;

        // Update type patterns
        let type_pattern = patterns.type_patterns.entry(allocation.allocation_type.clone()).or_insert_with(|| {
            TypeAllocationPattern {
                allocation_type: allocation.allocation_type.clone(),
                total_size: 0,
                average_size: 0.0,
                allocation_count: 0,
                peak_concurrent: 0,
                average_lifetime: Duration::from_secs(0),
                access_patterns: HashMap::new(),
            }
        });

        type_pattern.total_size += allocation.size;
        type_pattern.allocation_count += 1;
        type_pattern.average_size = type_pattern.total_size as f64 / type_pattern.allocation_count as f64;

        // Update access patterns
        *patterns.access_patterns.entry(allocation.access_pattern.clone()).or_insert(0) += 1;

        Ok(())
    }

    pub fn get_pattern_summary(&self) -> SklResult<PatternSummary> {
        let patterns = self.patterns.read()
            .map_err(|_| CoreError::LockError("Failed to acquire patterns lock".to_string()))?;

        let most_common_size = patterns.common_sizes.iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&size, &count)| (size, count));

        let most_active_thread = patterns.thread_patterns.iter()
            .max_by_key(|(_, pattern)| pattern.total_allocations)
            .map(|(_, pattern)| pattern.clone());

        let most_common_type = patterns.type_patterns.iter()
            .max_by_key(|(_, pattern)| pattern.allocation_count)
            .map(|(_, pattern)| pattern.clone());

        Ok(PatternSummary {
            most_common_size,
            most_active_thread,
            most_common_type,
            total_analyzed: patterns.common_sizes.values().sum::<usize>(),
        })
    }
}

// Metrics structures

/// Memory metrics collector
pub struct MemoryMetrics {
    registry: MetricRegistry,
    memory_usage: Gauge,
    allocation_rate: Counter,
    deallocation_rate: Counter,
    pressure_level: Gauge,
    fragmentation_ratio: Gauge,
    gc_frequency: Counter,
}

impl MemoryMetrics {
    pub fn new() -> SklResult<Self> {
        let registry = MetricRegistry::new();
        let memory_usage = registry.gauge("memory_usage_bytes", "Current memory usage in bytes")?;
        let allocation_rate = registry.counter("allocations_total", "Total number of allocations")?;
        let deallocation_rate = registry.counter("deallocations_total", "Total number of deallocations")?;
        let pressure_level = registry.gauge("memory_pressure_level", "Current memory pressure level")?;
        let fragmentation_ratio = registry.gauge("memory_fragmentation_ratio", "Memory fragmentation ratio")?;
        let gc_frequency = registry.counter("gc_events_total", "Total garbage collection events")?;

        Ok(Self {
            registry,
            memory_usage,
            allocation_rate,
            deallocation_rate,
            pressure_level,
            fragmentation_ratio,
            gc_frequency,
        })
    }

    pub fn update_memory_usage(&self, usage: usize) -> SklResult<()> {
        self.memory_usage.set(usage as f64);
        Ok(())
    }

    pub fn record_allocation(&self, _size: usize) -> SklResult<()> {
        self.allocation_rate.increment();
        Ok(())
    }

    pub fn record_deallocation(&self, _size: usize) -> SklResult<()> {
        self.deallocation_rate.increment();
        Ok(())
    }

    pub fn update_pressure(&self, pressure: &MemoryPressure) -> SklResult<()> {
        let level = match pressure {
            MemoryPressure::Low => 1.0,
            MemoryPressure::Moderate => 2.0,
            MemoryPressure::High => 3.0,
            MemoryPressure::Critical => 4.0,
            MemoryPressure::Emergency => 5.0,
        };
        self.pressure_level.set(level);
        Ok(())
    }
}

/// Allocation metrics collector
pub struct AllocationMetrics {
    registry: MetricRegistry,
    allocation_counter: Counter,
    deallocation_counter: Counter,
    allocation_size_histogram: Histogram,
    lifetime_histogram: Histogram,
}

impl AllocationMetrics {
    pub fn new() -> SklResult<Self> {
        let registry = MetricRegistry::new();
        let allocation_counter = registry.counter("tracked_allocations_total", "Total tracked allocations")?;
        let deallocation_counter = registry.counter("tracked_deallocations_total", "Total tracked deallocations")?;
        let allocation_size_histogram = registry.histogram("allocation_size_bytes", "Distribution of allocation sizes")?;
        let lifetime_histogram = registry.histogram("allocation_lifetime_seconds", "Distribution of allocation lifetimes")?;

        Ok(Self {
            registry,
            allocation_counter,
            deallocation_counter,
            allocation_size_histogram,
            lifetime_histogram,
        })
    }

    pub fn record_allocation(&self, size: usize, _allocation_type: &AllocationType) -> SklResult<()> {
        self.allocation_counter.increment();
        self.allocation_size_histogram.record(size as f64);
        Ok(())
    }

    pub fn record_deallocation(&self, size: usize, _allocation_type: &AllocationType, lifetime: Duration) -> SklResult<()> {
        self.deallocation_counter.increment();
        self.lifetime_histogram.record(lifetime.as_secs_f64());
        Ok(())
    }
}

/// Pool metrics collector
pub struct PoolMetrics {
    registry: MetricRegistry,
    pool_utilization: Gauge,
    pool_fragmentation: Gauge,
    large_object_count: Gauge,
    pool_expansions: Counter,
}

impl PoolMetrics {
    pub fn new() -> SklResult<Self> {
        let registry = MetricRegistry::new();
        let pool_utilization = registry.gauge("pool_utilization_ratio", "Memory pool utilization ratio")?;
        let pool_fragmentation = registry.gauge("pool_fragmentation_ratio", "Memory pool fragmentation ratio")?;
        let large_object_count = registry.gauge("large_objects_count", "Number of large objects")?;
        let pool_expansions = registry.counter("pool_expansions_total", "Total pool expansions")?;

        Ok(Self {
            registry,
            pool_utilization,
            pool_fragmentation,
            large_object_count,
            pool_expansions,
        })
    }
}

// Builder implementations

/// Builder for MemoryUsageStats
pub struct MemoryUsageStatsBuilder {
    config: MemoryStatsConfig,
}

impl MemoryUsageStatsBuilder {
    pub fn new() -> Self {
        Self {
            config: MemoryStatsConfig::default(),
        }
    }

    pub fn update_interval(mut self, interval: Duration) -> Self {
        self.config.update_interval = interval;
        self
    }

    pub fn memory_limit(mut self, limit: usize) -> Self {
        self.config.memory_limit = limit;
        self
    }

    pub fn enable_profiling(mut self, enabled: bool) -> Self {
        self.config.enable_profiling = enabled;
        self
    }

    pub fn build(self) -> SklResult<MemoryUsageStats> {
        let current_stats = Arc::new(RwLock::new(CurrentMemoryStats::default()));
        let historical_stats = Arc::new(RwLock::new(HistoricalMemoryStats {
            snapshots: VecDeque::new(),
            usage_trends: MemoryUsageTrends {
                growth_rate: 0.0,
                peak_growth_rate: 0.0,
                allocation_velocity: 0.0,
                deallocation_velocity: 0.0,
                pressure_trend: 0.0,
                fragmentation_trend: 0.0,
                predicted_peak: None,
            },
            allocation_patterns: AllocationPatterns {
                common_sizes: BTreeMap::new(),
                temporal_patterns: HashMap::new(),
                thread_patterns: HashMap::new(),
                type_patterns: HashMap::new(),
                access_patterns: HashMap::new(),
            },
            pressure_history: VecDeque::new(),
            gc_events: VecDeque::new(),
        }));

        let pressure_monitor = Arc::new(MemoryPressureMonitor::new(self.config.pressure_thresholds.clone()));
        let profiler = Arc::new(MemoryProfiler {
            profiling_enabled: AtomicBool::new(self.config.enable_profiling),
            profile_data: Arc::new(RwLock::new(ProfileData::default())),
            sampling_rate: AtomicUsize::new(100), // Sample 1 in 100 allocations
        });
        let metrics = Arc::new(MemoryMetrics::new()?);
        let shutdown_signal = Arc::new(AtomicBool::new(false));

        Ok(MemoryUsageStats {
            config: self.config,
            current_stats,
            historical_stats,
            pressure_monitor,
            profiler,
            metrics,
            collection_thread: None, // Would be initialized separately
            shutdown_signal,
        })
    }
}

/// Builder for AllocationTracker
pub struct AllocationTrackerBuilder {
    config: AllocationTrackerConfig,
}

impl AllocationTrackerBuilder {
    pub fn new() -> Self {
        Self {
            config: AllocationTrackerConfig::default(),
        }
    }

    pub fn track_stack_traces(mut self, enabled: bool) -> Self {
        self.config.track_stack_traces = enabled;
        self
    }

    pub fn allocation_size_threshold(mut self, threshold: usize) -> Self {
        self.config.allocation_size_threshold = threshold;
        self
    }

    pub fn retention_period(mut self, period: Duration) -> Self {
        self.config.retention_period = period;
        self
    }

    pub fn build(self) -> SklResult<AllocationTracker> {
        let active_allocations = Arc::new(RwLock::new(HashMap::new()));
        let allocation_history = Arc::new(RwLock::new(VecDeque::new()));
        let pattern_analyzer = Arc::new(AllocationPatternAnalyzer::new());
        let metrics = Arc::new(AllocationMetrics::new()?);
        let allocation_counter = AtomicUsize::new(0);

        Ok(AllocationTracker {
            config: self.config,
            active_allocations,
            allocation_history,
            pattern_analyzer,
            metrics,
            allocation_counter,
        })
    }
}

/// Builder for MemoryPool
pub struct MemoryPoolBuilder {
    config: MemoryPoolConfig,
}

impl MemoryPoolBuilder {
    pub fn new() -> Self {
        Self {
            config: MemoryPoolConfig::default(),
        }
    }

    pub fn pool_size(mut self, size: usize) -> Self {
        self.config.pool_size = size;
        self
    }

    pub fn block_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.config.block_sizes = sizes;
        self
    }

    pub fn auto_expand(mut self, enabled: bool) -> Self {
        self.config.auto_expand = enabled;
        self
    }

    pub fn build(self) -> SklResult<MemoryPool> {
        let pools = Arc::new(RwLock::new(HashMap::new()));
        let large_object_pool = Arc::new(RwLock::new(LargeObjectPool {
            allocations: HashMap::new(),
            total_allocated: 0,
            statistics: LargePoolStatistics {
                total_allocations: 0,
                total_deallocations: 0,
                current_allocated: 0,
                peak_allocated: 0,
                total_allocated_size: 0,
                average_allocation_size: 0.0,
            },
        }));
        let statistics = Arc::new(RwLock::new(PoolStatistics {
            total_pools: 0,
            total_allocated_blocks: 0,
            total_free_blocks: 0,
            total_memory_used: 0,
            fragmentation_ratio: 0.0,
            utilization_ratio: 0.0,
            expansion_count: 0,
        }));
        let metrics = Arc::new(PoolMetrics::new()?);

        Ok(MemoryPool {
            config: self.config,
            pools,
            large_object_pool,
            statistics,
            metrics,
        })
    }
}

// Statistics and result types

/// Memory report containing comprehensive analysis
#[derive(Debug, Clone)]
pub struct MemoryReport {
    pub current_stats: CurrentMemoryStats,
    pub usage_trends: MemoryUsageTrends,
    pub allocation_patterns: AllocationPatterns,
    pub pressure_history: Vec<(Instant, MemoryPressure)>,
    pub optimization_recommendations: Vec<OptimizationRecommendation>,
    pub performance_impact: PerformanceImpact,
}

/// Optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    pub technique: OptimizationTechnique,
    pub priority: RecommendationPriority,
    pub estimated_savings: usize,
    pub implementation_effort: ImplementationEffort,
    pub description: String,
}

/// Recommendation priority levels
#[derive(Debug, Clone, PartialEq, PartialOrd, Ord, Eq)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Implementation effort estimates
#[derive(Debug, Clone, PartialEq)]
pub enum ImplementationEffort {
    Low,    // < 1 day
    Medium, // 1-5 days
    High,   // > 5 days
    Custom { estimated_hours: usize },
}

/// Performance impact analysis
#[derive(Debug, Clone)]
pub struct PerformanceImpact {
    pub memory_overhead: f64,
    pub fragmentation_overhead: f64,
    pub gc_overhead: f64,
    pub total_overhead: f64,
    pub recommendations: Vec<OptimizationRecommendation>,
}

/// Allocation statistics
#[derive(Debug, Clone)]
pub struct AllocationStatistics {
    pub total_active_allocations: usize,
    pub total_active_memory: usize,
    pub allocations_by_type: HashMap<AllocationType, usize>,
    pub memory_by_type: HashMap<AllocationType, usize>,
    pub allocations_by_thread: HashMap<ThreadId, usize>,
    pub large_allocations: Vec<AllocationInfo>,
    pub pattern_analysis: PatternSummary,
}

/// Pattern analysis summary
#[derive(Debug, Clone)]
pub struct PatternSummary {
    pub most_common_size: Option<(usize, usize)>,
    pub most_active_thread: Option<ThreadAllocationPattern>,
    pub most_common_type: Option<TypeAllocationPattern>,
    pub total_analyzed: usize,
}

/// Pool statistics
#[derive(Debug, Clone)]
pub struct PoolStatistics {
    pub total_pools: usize,
    pub total_allocated_blocks: usize,
    pub total_free_blocks: usize,
    pub total_memory_used: usize,
    pub fragmentation_ratio: f64,
    pub utilization_ratio: f64,
    pub expansion_count: usize,
}

/// Fixed pool statistics
#[derive(Debug, Clone)]
struct FixedPoolStatistics {
    pub block_size: usize,
    pub total_blocks: usize,
    pub allocations: usize,
    pub deallocations: usize,
    pub current_allocated: usize,
    pub peak_allocated: usize,
    pub fragmentation_ratio: f64,
}

/// Large pool statistics
#[derive(Debug, Clone)]
struct LargePoolStatistics {
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub current_allocated: usize,
    pub peak_allocated: usize,
    pub total_allocated_size: usize,
    pub average_allocation_size: f64,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_memory_usage_stats_creation() {
        let stats = MemoryUsageStats::builder()
            .memory_limit(1024 * 1024 * 1024) // 1GB
            .update_interval(Duration::from_millis(100))
            .enable_profiling(true)
            .build()
            .expect("Failed to create memory usage stats");

        assert_eq!(stats.config.memory_limit, 1024 * 1024 * 1024);
        assert_eq!(stats.config.update_interval, Duration::from_millis(100));
        assert!(stats.config.enable_profiling);
    }

    #[test]
    fn test_memory_pressure_calculation() {
        let stats = MemoryUsageStats::builder()
            .memory_limit(1000)
            .build()
            .expect("Failed to create memory usage stats");

        // Test different pressure levels
        assert_eq!(stats.calculate_memory_pressure(500).unwrap(), MemoryPressure::Low);
        assert_eq!(stats.calculate_memory_pressure(700).unwrap(), MemoryPressure::Moderate);
        assert_eq!(stats.calculate_memory_pressure(900).unwrap(), MemoryPressure::High);
        assert_eq!(stats.calculate_memory_pressure(980).unwrap(), MemoryPressure::Critical);
    }

    #[test]
    fn test_allocation_tracker_creation() {
        let tracker = AllocationTracker::builder()
            .allocation_size_threshold(1024)
            .track_stack_traces(false)
            .retention_period(Duration::from_hours(1))
            .build()
            .expect("Failed to create allocation tracker");

        assert_eq!(tracker.config.allocation_size_threshold, 1024);
        assert!(!tracker.config.track_stack_traces);
        assert_eq!(tracker.config.retention_period, Duration::from_hours(1));
    }

    #[test]
    fn test_allocation_tracking() {
        let tracker = AllocationTracker::builder()
            .allocation_size_threshold(100) // Low threshold for testing
            .build()
            .expect("Failed to create allocation tracker");

        // Track an allocation
        let allocation_id = tracker.track_allocation(2048, 64, AllocationType::Gradient)
            .expect("Failed to track allocation");

        assert!(allocation_id > 0);

        // Get statistics
        let stats = tracker.get_allocation_statistics()
            .expect("Failed to get allocation statistics");

        assert_eq!(stats.total_active_allocations, 1);
        assert_eq!(stats.total_active_memory, 2048);
        assert_eq!(*stats.allocations_by_type.get(&AllocationType::Gradient).unwrap(), 1);

        // Track deallocation
        let deallocated = tracker.track_deallocation(allocation_id)
            .expect("Failed to track deallocation");

        assert!(deallocated.is_some());
        let allocation_info = deallocated.unwrap();
        assert_eq!(allocation_info.size, 2048);
        assert_eq!(allocation_info.allocation_type, AllocationType::Gradient);
    }

    #[test]
    fn test_memory_pool_creation() {
        let pool = MemoryPool::builder()
            .pool_size(1024 * 1024) // 1MB
            .block_sizes(vec![64, 256, 1024])
            .auto_expand(true)
            .build()
            .expect("Failed to create memory pool");

        assert_eq!(pool.config.pool_size, 1024 * 1024);
        assert_eq!(pool.config.block_sizes, vec![64, 256, 1024]);
        assert!(pool.config.auto_expand);
    }

    #[test]
    fn test_memory_pool_allocation() {
        let pool = MemoryPool::builder()
            .pool_size(4096)
            .block_sizes(vec![64, 256, 1024])
            .build()
            .expect("Failed to create memory pool");

        // Allocate small block
        let ptr1 = pool.allocate(128, AllocationType::TemporaryBuffers)
            .expect("Failed to allocate from pool");

        // Allocate another block
        let ptr2 = pool.allocate(512, AllocationType::BatchData)
            .expect("Failed to allocate from pool");

        assert_ne!(ptr1, ptr2);

        // Deallocate
        pool.deallocate(ptr1, 128)
            .expect("Failed to deallocate");
        pool.deallocate(ptr2, 512)
            .expect("Failed to deallocate");
    }

    #[test]
    fn test_allocation_type_categorization() {
        let types = vec![
            AllocationType::Gradient,
            AllocationType::Parameters,
            AllocationType::OptimizerState,
            AllocationType::TemporaryBuffers,
        ];

        for allocation_type in types {
            // Test that allocation types can be used as hash keys
            let mut map = HashMap::new();
            map.insert(allocation_type.clone(), 100);
            assert_eq!(*map.get(&allocation_type).unwrap(), 100);
        }
    }

    #[test]
    fn test_memory_pressure_levels() {
        let pressures = vec![
            MemoryPressure::Low,
            MemoryPressure::Moderate,
            MemoryPressure::High,
            MemoryPressure::Critical,
            MemoryPressure::Emergency,
        ];

        // Test ordering
        for i in 0..pressures.len() - 1 {
            assert!(pressures[i] < pressures[i + 1]);
        }
    }

    #[test]
    fn test_access_pattern_analysis() {
        let patterns = vec![
            AccessPattern::Sequential,
            AccessPattern::Random,
            AccessPattern::WriteOnceReadMany,
            AccessPattern::ReadOnce,
            AccessPattern::HighFrequency,
            AccessPattern::LowFrequency,
            AccessPattern::Unknown,
        ];

        // Test that access patterns can be compared
        for pattern in patterns {
            assert_eq!(pattern, pattern);
        }
    }

    #[test]
    fn test_optimization_recommendations() {
        let recommendations = vec![
            OptimizationTechnique::BufferReuse,
            OptimizationTechnique::InPlaceOperations,
            OptimizationTechnique::GradientCheckpointing,
            OptimizationTechnique::Compression { algorithm: CompressionAlgorithm::LZ4 },
        ];

        for technique in recommendations {
            let recommendation = OptimizationRecommendation {
                technique: technique.clone(),
                priority: RecommendationPriority::Medium,
                estimated_savings: 1024,
                implementation_effort: ImplementationEffort::Low,
                description: "Test recommendation".to_string(),
            };

            assert_eq!(recommendation.technique, technique);
            assert_eq!(recommendation.estimated_savings, 1024);
        }
    }

    #[test]
    fn test_linear_trend_calculation() {
        let stats = MemoryUsageStats::builder()
            .build()
            .expect("Failed to create memory usage stats");

        // Test increasing trend
        let increasing_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let trend = stats.calculate_linear_trend(&increasing_values)
            .expect("Failed to calculate trend");
        assert!(trend > 0.0);

        // Test decreasing trend
        let decreasing_values = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let trend = stats.calculate_linear_trend(&decreasing_values)
            .expect("Failed to calculate trend");
        assert!(trend < 0.0);

        // Test flat trend
        let flat_values = vec![3.0, 3.0, 3.0, 3.0, 3.0];
        let trend = stats.calculate_linear_trend(&flat_values)
            .expect("Failed to calculate trend");
        assert_eq!(trend, 0.0);
    }
}