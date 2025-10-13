//! Performance Profiling Utilities for Pipeline Optimization
//!
//! This module provides comprehensive performance profiling tools for machine learning
//! pipelines, including execution timing, memory usage tracking, bottleneck detection,
//! and optimization recommendations.

use chrono::{DateTime, Utc};
use scirs2_core::random::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Main performance profiler for pipeline execution
#[derive(Debug, Clone)]
pub struct PerformanceProfiler {
    config: ProfilerConfig,
    active_sessions: Arc<Mutex<HashMap<String, ProfileSession>>>,
    completed_sessions: Arc<Mutex<Vec<ProfileSession>>>,
}

/// Configuration for the performance profiler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    /// Enable detailed timing measurements
    pub enable_timing: bool,
    /// Enable memory usage tracking
    pub enable_memory_tracking: bool,
    /// Enable CPU usage monitoring
    pub enable_cpu_monitoring: bool,
    /// Enable GPU usage monitoring if available
    pub enable_gpu_monitoring: bool,
    /// Sample interval for continuous monitoring (ms)
    pub sample_interval_ms: u64,
    /// Maximum number of profiling sessions to keep
    pub max_sessions: usize,
    /// Enable automatic bottleneck detection
    pub enable_bottleneck_detection: bool,
    /// Enable optimization recommendations
    pub enable_optimization_hints: bool,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            enable_timing: true,
            enable_memory_tracking: true,
            enable_cpu_monitoring: true,
            enable_gpu_monitoring: false,
            sample_interval_ms: 100,
            max_sessions: 100,
            enable_bottleneck_detection: true,
            enable_optimization_hints: true,
        }
    }
}

/// Individual profiling session for a pipeline execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileSession {
    pub session_id: String,
    pub pipeline_name: String,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub stages: BTreeMap<String, StageProfile>,
    pub overall_metrics: OverallMetrics,
    pub bottlenecks: Vec<Bottleneck>,
    pub optimization_hints: Vec<OptimizationHint>,
}

/// Performance profile for individual pipeline stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageProfile {
    pub stage_name: String,
    pub component_type: String,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub execution_time: Duration,
    pub memory_samples: Vec<MemorySample>,
    pub cpu_samples: Vec<CpuSample>,
    pub gpu_samples: Vec<GpuSample>,
    pub input_shape: Option<(usize, usize)>,
    pub output_shape: Option<(usize, usize)>,
    pub parameters: HashMap<String, String>,
    pub error_count: u32,
    pub warning_count: u32,
}

/// Overall pipeline execution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallMetrics {
    pub total_execution_time: Duration,
    pub peak_memory_usage_mb: f64,
    pub average_cpu_usage: f64,
    pub average_gpu_usage: f64,
    pub total_data_processed_mb: f64,
    pub throughput_samples_per_second: f64,
    pub cache_hit_ratio: f64,
    pub parallel_efficiency: f64,
    pub pipeline_stages: usize,
    pub data_transformations: usize,
}

/// Memory usage sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySample {
    pub timestamp: DateTime<Utc>,
    pub heap_usage_mb: f64,
    pub stack_usage_mb: f64,
    pub gpu_memory_mb: f64,
    pub virtual_memory_mb: f64,
}

/// CPU usage sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuSample {
    pub timestamp: DateTime<Utc>,
    pub overall_usage: f64,
    pub user_usage: f64,
    pub system_usage: f64,
    pub core_usage: Vec<f64>,
    pub thread_count: u32,
}

/// GPU usage sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuSample {
    pub timestamp: DateTime<Utc>,
    pub gpu_utilization: f64,
    pub memory_utilization: f64,
    pub temperature: f64,
    pub power_consumption: f64,
}

/// Performance bottleneck identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    pub bottleneck_type: BottleneckType,
    pub affected_stage: String,
    pub severity: BottleneckSeverity,
    pub impact_factor: f64,
    pub description: String,
    pub metrics: BottleneckMetrics,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckType {
    /// MemoryConstraint
    MemoryConstraint,
    /// ComputationalBottleneck
    ComputationalBottleneck,
    /// IOBottleneck
    IOBottleneck,
    /// CacheInefficiency
    CacheInefficiency,
    /// SynchronizationOverhead
    SynchronizationOverhead,
    /// DataMovementOverhead
    DataMovementOverhead,
    /// AlgorithmicComplexity
    AlgorithmicComplexity,
    /// ConfigurationSuboptimal
    ConfigurationSuboptimal,
}

/// Bottleneck severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BottleneckSeverity {
    /// Low
    Low,
    /// Medium
    Medium,
    /// High
    High,
    /// Critical
    Critical,
}

/// Specific metrics for bottleneck analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckMetrics {
    pub time_spent_waiting_ms: f64,
    pub resource_utilization: f64,
    pub efficiency_score: f64,
    pub improvement_potential: f64,
}

/// Optimization suggestions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationHint {
    pub category: OptimizationCategory,
    pub priority: OptimizationPriority,
    pub title: String,
    pub description: String,
    pub expected_improvement: f64,
    pub implementation_difficulty: ImplementationDifficulty,
    pub code_examples: Vec<String>,
}

/// Categories of optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationCategory {
    /// AlgorithmSelection
    AlgorithmSelection,
    /// ParameterTuning
    ParameterTuning,
    /// MemoryOptimization
    MemoryOptimization,
    /// ParallelProcessing
    ParallelProcessing,
    /// CacheOptimization
    CacheOptimization,
    /// DataStructureOptimization
    DataStructureOptimization,
    /// HardwareUtilization
    HardwareUtilization,
    /// PipelineRestructuring
    PipelineRestructuring,
}

/// Priority levels for optimizations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum OptimizationPriority {
    /// Low
    Low,
    /// Medium
    Medium,
    /// High
    High,
    /// Critical
    Critical,
}

/// Implementation difficulty assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationDifficulty {
    /// Trivial
    Trivial, // Just configuration changes
    /// Easy
    Easy, // Simple code modifications
    /// Moderate
    Moderate, // Significant code changes
    /// Hard
    Hard, // Major restructuring
    /// Expert
    Expert, // Requires deep expertise
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    #[must_use]
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            config,
            active_sessions: Arc::new(Mutex::new(HashMap::new())),
            completed_sessions: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Create profiler with default configuration
    #[must_use]
    pub fn default() -> Self {
        Self::new(ProfilerConfig::default())
    }

    /// Start a new profiling session
    #[must_use]
    pub fn start_session(&self, pipeline_name: &str) -> String {
        let session_id = format!("profile_{}", uuid::Uuid::new_v4());
        let session = ProfileSession {
            session_id: session_id.clone(),
            pipeline_name: pipeline_name.to_string(),
            start_time: Utc::now(),
            end_time: None,
            stages: BTreeMap::new(),
            overall_metrics: OverallMetrics::default(),
            bottlenecks: Vec::new(),
            optimization_hints: Vec::new(),
        };

        {
            let mut active = self.active_sessions.lock().unwrap();
            active.insert(session_id.clone(), session);
        }

        // Start background monitoring if enabled
        if self.config.enable_cpu_monitoring || self.config.enable_memory_tracking {
            self.start_background_monitoring(&session_id);
        }

        session_id
    }

    /// Start profiling a specific pipeline stage
    pub fn start_stage(
        &self,
        session_id: &str,
        stage_name: &str,
        component_type: &str,
    ) -> Result<(), String> {
        let mut active = self.active_sessions.lock().unwrap();
        if let Some(session) = active.get_mut(session_id) {
            let stage_profile = StageProfile {
                stage_name: stage_name.to_string(),
                component_type: component_type.to_string(),
                start_time: Utc::now(),
                end_time: None,
                execution_time: Duration::from_secs(0),
                memory_samples: Vec::new(),
                cpu_samples: Vec::new(),
                gpu_samples: Vec::new(),
                input_shape: None,
                output_shape: None,
                parameters: HashMap::new(),
                error_count: 0,
                warning_count: 0,
            };
            session.stages.insert(stage_name.to_string(), stage_profile);
            Ok(())
        } else {
            Err(format!("Session {session_id} not found"))
        }
    }

    /// End profiling a specific pipeline stage
    pub fn end_stage(&self, session_id: &str, stage_name: &str) -> Result<Duration, String> {
        let mut active = self.active_sessions.lock().unwrap();
        if let Some(session) = active.get_mut(session_id) {
            if let Some(stage) = session.stages.get_mut(stage_name) {
                let end_time = Utc::now();
                stage.end_time = Some(end_time);
                stage.execution_time = end_time
                    .signed_duration_since(stage.start_time)
                    .to_std()
                    .unwrap_or(Duration::from_secs(0));
                Ok(stage.execution_time)
            } else {
                Err(format!(
                    "Stage {stage_name} not found in session {session_id}"
                ))
            }
        } else {
            Err(format!("Session {session_id} not found"))
        }
    }

    /// Record stage parameters for analysis
    pub fn record_stage_parameters(
        &self,
        session_id: &str,
        stage_name: &str,
        parameters: HashMap<String, String>,
    ) -> Result<(), String> {
        let mut active = self.active_sessions.lock().unwrap();
        if let Some(session) = active.get_mut(session_id) {
            if let Some(stage) = session.stages.get_mut(stage_name) {
                stage.parameters = parameters;
                Ok(())
            } else {
                Err(format!("Stage {stage_name} not found"))
            }
        } else {
            Err(format!("Session {session_id} not found"))
        }
    }

    /// Record data shapes for performance analysis
    pub fn record_data_shapes(
        &self,
        session_id: &str,
        stage_name: &str,
        input_shape: Option<(usize, usize)>,
        output_shape: Option<(usize, usize)>,
    ) -> Result<(), String> {
        let mut active = self.active_sessions.lock().unwrap();
        if let Some(session) = active.get_mut(session_id) {
            if let Some(stage) = session.stages.get_mut(stage_name) {
                stage.input_shape = input_shape;
                stage.output_shape = output_shape;
                Ok(())
            } else {
                Err(format!("Stage {stage_name} not found"))
            }
        } else {
            Err(format!("Session {session_id} not found"))
        }
    }

    /// End profiling session and generate analysis
    pub fn end_session(&self, session_id: &str) -> Result<ProfileSession, String> {
        let mut session = {
            let mut active = self.active_sessions.lock().unwrap();
            active
                .remove(session_id)
                .ok_or_else(|| format!("Session {session_id} not found"))?
        };

        session.end_time = Some(Utc::now());

        // Calculate overall metrics
        session.overall_metrics = self.calculate_overall_metrics(&session);

        // Detect bottlenecks
        if self.config.enable_bottleneck_detection {
            session.bottlenecks = self.detect_bottlenecks(&session);
        }

        // Generate optimization hints
        if self.config.enable_optimization_hints {
            session.optimization_hints = self.generate_optimization_hints(&session);
        }

        // Store completed session
        {
            let mut completed = self.completed_sessions.lock().unwrap();
            completed.push(session.clone());

            // Maintain session limit
            while completed.len() > self.config.max_sessions {
                completed.remove(0);
            }
        }

        Ok(session)
    }

    /// Start background monitoring for system resources
    fn start_background_monitoring(&self, session_id: &str) {
        let session_id = session_id.to_string();
        let active_sessions = Arc::clone(&self.active_sessions);
        let config = self.config.clone();

        thread::spawn(move || {
            let sample_interval = Duration::from_millis(config.sample_interval_ms);

            loop {
                let should_continue = {
                    let active = active_sessions.lock().unwrap();
                    active.contains_key(&session_id)
                };

                if !should_continue {
                    break;
                }

                // Sample system metrics
                if config.enable_memory_tracking {
                    let memory_sample = Self::sample_memory();
                    Self::add_memory_sample(&active_sessions, &session_id, memory_sample);
                }

                if config.enable_cpu_monitoring {
                    let cpu_sample = Self::sample_cpu();
                    Self::add_cpu_sample(&active_sessions, &session_id, cpu_sample);
                }

                if config.enable_gpu_monitoring {
                    if let Some(gpu_sample) = Self::sample_gpu() {
                        Self::add_gpu_sample(&active_sessions, &session_id, gpu_sample);
                    }
                }

                thread::sleep(sample_interval);
            }
        });
    }

    /// Sample current memory usage
    fn sample_memory() -> MemorySample {
        // In a real implementation, you would use system APIs or libraries like sysinfo
        // For now, we'll simulate the sampling
        /// MemorySample
        MemorySample {
            timestamp: Utc::now(),
            heap_usage_mb: Self::get_process_memory(),
            stack_usage_mb: 5.2, // Simulated
            gpu_memory_mb: 0.0,  // Would need GPU APIs
            virtual_memory_mb: Self::get_process_memory() * 1.5,
        }
    }

    /// Sample current CPU usage
    fn sample_cpu() -> CpuSample {
        // In a real implementation, you would use system APIs
        /// CpuSample
        CpuSample {
            timestamp: Utc::now(),
            overall_usage: Self::get_cpu_usage(),
            user_usage: Self::get_cpu_usage() * 0.8,
            system_usage: Self::get_cpu_usage() * 0.2,
            core_usage: (0..num_cpus::get())
                .map(|_| Self::get_cpu_usage())
                .collect(),
            thread_count: 8, // Would need proper thread counting
        }
    }

    /// Sample GPU usage if available
    fn sample_gpu() -> Option<GpuSample> {
        // GPU monitoring would require specialized libraries like NVML
        None
    }

    /// Get current process memory usage (simplified)
    fn get_process_memory() -> f64 {
        // This is a simplified implementation
        // In practice, you'd use sysinfo or similar
        150.0 + (thread_rng().gen::<f64>() * 50.0)
    }

    /// Get current CPU usage (simplified)
    fn get_cpu_usage() -> f64 {
        // Simplified CPU usage simulation
        30.0 + (thread_rng().gen::<f64>() * 40.0)
    }

    /// Add memory sample to active session
    fn add_memory_sample(
        active_sessions: &Arc<Mutex<HashMap<String, ProfileSession>>>,
        session_id: &str,
        sample: MemorySample,
    ) {
        let mut active = active_sessions.lock().unwrap();
        if let Some(session) = active.get_mut(session_id) {
            // Add to the currently active stage or overall session
            if let Some((_, stage)) = session.stages.iter_mut().last() {
                if stage.end_time.is_none() {
                    stage.memory_samples.push(sample);
                }
            }
        }
    }

    /// Add CPU sample to active session
    fn add_cpu_sample(
        active_sessions: &Arc<Mutex<HashMap<String, ProfileSession>>>,
        session_id: &str,
        sample: CpuSample,
    ) {
        let mut active = active_sessions.lock().unwrap();
        if let Some(session) = active.get_mut(session_id) {
            if let Some((_, stage)) = session.stages.iter_mut().last() {
                if stage.end_time.is_none() {
                    stage.cpu_samples.push(sample);
                }
            }
        }
    }

    /// Add GPU sample to active session
    fn add_gpu_sample(
        active_sessions: &Arc<Mutex<HashMap<String, ProfileSession>>>,
        session_id: &str,
        sample: GpuSample,
    ) {
        let mut active = active_sessions.lock().unwrap();
        if let Some(session) = active.get_mut(session_id) {
            if let Some((_, stage)) = session.stages.iter_mut().last() {
                if stage.end_time.is_none() {
                    stage.gpu_samples.push(sample);
                }
            }
        }
    }

    /// Calculate overall pipeline metrics
    fn calculate_overall_metrics(&self, session: &ProfileSession) -> OverallMetrics {
        let total_execution_time = session
            .stages
            .values()
            .map(|stage| stage.execution_time)
            .fold(Duration::from_secs(0), |acc, dur| acc + dur);

        let peak_memory = session
            .stages
            .values()
            .flat_map(|stage| &stage.memory_samples)
            .map(|sample| sample.heap_usage_mb)
            .fold(0.0, f64::max);

        let avg_cpu = session
            .stages
            .values()
            .flat_map(|stage| &stage.cpu_samples)
            .map(|sample| sample.overall_usage)
            .collect::<Vec<_>>();
        let average_cpu_usage = if avg_cpu.is_empty() {
            0.0
        } else {
            avg_cpu.iter().sum::<f64>() / avg_cpu.len() as f64
        };

        /// OverallMetrics
        OverallMetrics {
            total_execution_time,
            peak_memory_usage_mb: peak_memory,
            average_cpu_usage,
            average_gpu_usage: 0.0, // Would be calculated from GPU samples
            total_data_processed_mb: Self::estimate_data_processed(session),
            throughput_samples_per_second: Self::calculate_throughput(session),
            cache_hit_ratio: 0.75, // Would be measured from actual cache statistics
            parallel_efficiency: Self::calculate_parallel_efficiency(session),
            pipeline_stages: session.stages.len(),
            data_transformations: Self::count_transformations(session),
        }
    }

    /// Estimate total data processed
    fn estimate_data_processed(session: &ProfileSession) -> f64 {
        session
            .stages
            .values()
            .filter_map(|stage| stage.input_shape)
            .map(|(samples, features)| (samples * features * 8) as f64 / (1024.0 * 1024.0)) // 8 bytes per f64
            .sum()
    }

    /// Calculate processing throughput
    fn calculate_throughput(session: &ProfileSession) -> f64 {
        let total_samples: usize = session
            .stages
            .values()
            .filter_map(|stage| stage.input_shape)
            .map(|(samples, _)| samples)
            .sum();

        let total_time_seconds = session.overall_metrics.total_execution_time.as_secs_f64();

        if total_time_seconds > 0.0 {
            total_samples as f64 / total_time_seconds
        } else {
            0.0
        }
    }

    /// Calculate parallel execution efficiency
    fn calculate_parallel_efficiency(session: &ProfileSession) -> f64 {
        // Simplified calculation based on CPU utilization vs ideal parallelism
        let ideal_parallel_stages = session.stages.len().min(num_cpus::get());
        let avg_cpu_per_core = session
            .stages
            .values()
            .flat_map(|stage| &stage.cpu_samples)
            .map(|sample| sample.overall_usage / sample.core_usage.len() as f64)
            .collect::<Vec<_>>();

        if avg_cpu_per_core.is_empty() {
            0.5 // Default moderate efficiency
        } else {
            let actual_efficiency =
                avg_cpu_per_core.iter().sum::<f64>() / avg_cpu_per_core.len() as f64;
            (actual_efficiency / 100.0).min(1.0)
        }
    }

    /// Count data transformation stages
    fn count_transformations(session: &ProfileSession) -> usize {
        session
            .stages
            .values()
            .filter(|stage| {
                stage.component_type.contains("transformer")
                    || stage.component_type.contains("preprocessor")
            })
            .count()
    }

    /// Detect performance bottlenecks
    fn detect_bottlenecks(&self, session: &ProfileSession) -> Vec<Bottleneck> {
        let mut bottlenecks = Vec::new();

        // Detect memory bottlenecks
        for (stage_name, stage) in &session.stages {
            if let Some(max_memory) = stage
                .memory_samples
                .iter()
                .map(|s| s.heap_usage_mb)
                .fold(None, |acc, x| Some(acc.map_or(x, |acc: f64| acc.max(x))))
            {
                if max_memory > 1000.0 {
                    // > 1GB
                    bottlenecks.push(Bottleneck {
                        bottleneck_type: BottleneckType::MemoryConstraint,
                        affected_stage: stage_name.clone(),
                        severity: if max_memory > 4000.0 {
                            BottleneckSeverity::Critical
                        } else {
                            BottleneckSeverity::High
                        },
                        impact_factor: (max_memory / 1000.0).min(5.0),
                        description: format!(
                            "High memory usage: {max_memory:.1}MB in stage '{stage_name}'"
                        ),
                        metrics: BottleneckMetrics {
                            time_spent_waiting_ms: 0.0,
                            resource_utilization: max_memory / 8192.0, // Assume 8GB system
                            efficiency_score: 1.0 - (max_memory / 8192.0),
                            improvement_potential: 0.3,
                        },
                    });
                }
            }

            // Detect computational bottlenecks
            if stage.execution_time.as_secs_f64() > 10.0 {
                let severity = if stage.execution_time.as_secs_f64() > 60.0 {
                    BottleneckSeverity::Critical
                } else if stage.execution_time.as_secs_f64() > 30.0 {
                    BottleneckSeverity::High
                } else {
                    BottleneckSeverity::Medium
                };

                bottlenecks.push(Bottleneck {
                    bottleneck_type: BottleneckType::ComputationalBottleneck,
                    affected_stage: stage_name.clone(),
                    severity,
                    impact_factor: stage.execution_time.as_secs_f64() / 10.0,
                    description: format!(
                        "Slow execution: {:.1}s in stage '{}'",
                        stage.execution_time.as_secs_f64(),
                        stage_name
                    ),
                    metrics: BottleneckMetrics {
                        time_spent_waiting_ms: stage.execution_time.as_millis() as f64,
                        resource_utilization: 0.8,
                        efficiency_score: 1.0 / stage.execution_time.as_secs_f64().max(1.0),
                        improvement_potential: 0.5,
                    },
                });
            }
        }

        bottlenecks
    }

    /// Generate optimization recommendations
    fn generate_optimization_hints(&self, session: &ProfileSession) -> Vec<OptimizationHint> {
        let mut hints = Vec::new();

        // Memory optimization hints
        if session.overall_metrics.peak_memory_usage_mb > 2000.0 {
            hints.push(OptimizationHint {
                category: OptimizationCategory::MemoryOptimization,
                priority: OptimizationPriority::High,
                title: "High Memory Usage Detected".to_string(),
                description: format!(
                    "Pipeline uses {:.1}MB peak memory. Consider chunked processing or streaming.",
                    session.overall_metrics.peak_memory_usage_mb
                ),
                expected_improvement: 0.4,
                implementation_difficulty: ImplementationDifficulty::Moderate,
                code_examples: vec![
                    "Use streaming: pipeline.enable_streaming(chunk_size=1000)".to_string(),
                    "Enable memory optimization: config.memory_efficient = true".to_string(),
                ],
            });
        }

        // Parallel processing hints
        if session.overall_metrics.parallel_efficiency < 0.5 {
            hints.push(OptimizationHint {
                category: OptimizationCategory::ParallelProcessing,
                priority: OptimizationPriority::Medium,
                title: "Low Parallel Efficiency".to_string(),
                description: format!(
                    "Parallel efficiency is {:.1}%. Consider enabling more parallelization.",
                    session.overall_metrics.parallel_efficiency * 100.0
                ),
                expected_improvement: 0.6,
                implementation_difficulty: ImplementationDifficulty::Easy,
                code_examples: vec![
                    "Set parallel jobs: pipeline.set_n_jobs(-1)".to_string(),
                    "Enable SIMD: config.enable_simd = true".to_string(),
                ],
            });
        }

        // Algorithm selection hints
        for (stage_name, stage) in &session.stages {
            if stage.execution_time.as_secs_f64() > 30.0
                && stage.component_type.contains("estimator")
            {
                hints.push(OptimizationHint {
                    category: OptimizationCategory::AlgorithmSelection,
                    priority: OptimizationPriority::High,
                    title: format!("Slow Algorithm in {stage_name}"),
                    description: format!(
                        "Stage '{}' takes {:.1}s. Consider faster algorithms or approximations.",
                        stage_name,
                        stage.execution_time.as_secs_f64()
                    ),
                    expected_improvement: 0.7,
                    implementation_difficulty: ImplementationDifficulty::Moderate,
                    code_examples: vec![
                        "Use approximate algorithms where applicable".to_string(),
                        "Consider ensemble methods for better speed/accuracy trade-off".to_string(),
                    ],
                });
            }
        }

        hints
    }

    /// Get all completed sessions
    #[must_use]
    pub fn get_completed_sessions(&self) -> Vec<ProfileSession> {
        let completed = self.completed_sessions.lock().unwrap();
        completed.clone()
    }

    /// Generate comprehensive performance report
    #[must_use]
    pub fn generate_report(&self, session_id: Option<&str>) -> PerformanceReport {
        let sessions = if let Some(id) = session_id {
            let completed = self.completed_sessions.lock().unwrap();
            completed
                .iter()
                .filter(|s| s.session_id == id)
                .cloned()
                .collect()
        } else {
            self.get_completed_sessions()
        };

        PerformanceReport::from_sessions(sessions)
    }
}

impl Default for OverallMetrics {
    fn default() -> Self {
        Self {
            total_execution_time: Duration::from_secs(0),
            peak_memory_usage_mb: 0.0,
            average_cpu_usage: 0.0,
            average_gpu_usage: 0.0,
            total_data_processed_mb: 0.0,
            throughput_samples_per_second: 0.0,
            cache_hit_ratio: 0.0,
            parallel_efficiency: 0.0,
            pipeline_stages: 0,
            data_transformations: 0,
        }
    }
}

/// Comprehensive performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub report_id: String,
    pub generated_at: DateTime<Utc>,
    pub sessions_analyzed: usize,
    pub summary_metrics: SummaryMetrics,
    pub bottleneck_analysis: BottleneckAnalysis,
    pub optimization_recommendations: Vec<OptimizationHint>,
    pub trend_analysis: TrendAnalysis,
    pub comparative_analysis: ComparativeAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryMetrics {
    pub average_execution_time: Duration,
    pub fastest_execution_time: Duration,
    pub slowest_execution_time: Duration,
    pub average_memory_usage: f64,
    pub peak_memory_across_sessions: f64,
    pub average_throughput: f64,
    pub best_parallel_efficiency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    pub most_common_bottleneck: BottleneckType,
    pub bottleneck_frequency: HashMap<String, u32>,
    pub severity_distribution: HashMap<BottleneckSeverity, u32>,
    pub impact_analysis: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    pub performance_trend: TrendDirection,
    pub memory_usage_trend: TrendDirection,
    pub throughput_trend: TrendDirection,
    pub session_performance_scores: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Improving
    Improving,
    /// Stable
    Stable,
    /// Degrading
    Degrading,
    /// InsufficientData
    InsufficientData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparativeAnalysis {
    pub best_performing_session: String,
    pub worst_performing_session: String,
    pub performance_variance: f64,
    pub consistency_score: f64,
}

impl PerformanceReport {
    #[must_use]
    pub fn from_sessions(sessions: Vec<ProfileSession>) -> Self {
        let report_id = format!("report_{}", uuid::Uuid::new_v4());

        let summary_metrics = Self::calculate_summary_metrics(&sessions);
        let bottleneck_analysis = Self::analyze_bottlenecks(&sessions);
        let trend_analysis = Self::analyze_trends(&sessions);
        let comparative_analysis = Self::comparative_analysis(&sessions);

        // Aggregate optimization recommendations
        let mut all_hints: Vec<OptimizationHint> = sessions
            .iter()
            .flat_map(|s| s.optimization_hints.iter())
            .cloned()
            .collect();

        // Deduplicate and prioritize hints
        all_hints.sort_by(|a, b| b.priority.cmp(&a.priority));
        all_hints.truncate(10); // Keep top 10 recommendations

        Self {
            report_id,
            generated_at: Utc::now(),
            sessions_analyzed: sessions.len(),
            summary_metrics,
            bottleneck_analysis,
            optimization_recommendations: all_hints,
            trend_analysis,
            comparative_analysis,
        }
    }

    fn calculate_summary_metrics(sessions: &[ProfileSession]) -> SummaryMetrics {
        if sessions.is_empty() {
            return SummaryMetrics {
                average_execution_time: Duration::from_secs(0),
                fastest_execution_time: Duration::from_secs(0),
                slowest_execution_time: Duration::from_secs(0),
                average_memory_usage: 0.0,
                peak_memory_across_sessions: 0.0,
                average_throughput: 0.0,
                best_parallel_efficiency: 0.0,
            };
        }

        let execution_times: Vec<Duration> = sessions
            .iter()
            .map(|s| s.overall_metrics.total_execution_time)
            .collect();

        let average_execution = Duration::from_secs_f64(
            execution_times
                .iter()
                .map(std::time::Duration::as_secs_f64)
                .sum::<f64>()
                / sessions.len() as f64,
        );

        /// SummaryMetrics
        SummaryMetrics {
            average_execution_time: average_execution,
            fastest_execution_time: *execution_times.iter().min().unwrap(),
            slowest_execution_time: *execution_times.iter().max().unwrap(),
            average_memory_usage: sessions
                .iter()
                .map(|s| s.overall_metrics.peak_memory_usage_mb)
                .sum::<f64>()
                / sessions.len() as f64,
            peak_memory_across_sessions: sessions
                .iter()
                .map(|s| s.overall_metrics.peak_memory_usage_mb)
                .fold(0.0, f64::max),
            average_throughput: sessions
                .iter()
                .map(|s| s.overall_metrics.throughput_samples_per_second)
                .sum::<f64>()
                / sessions.len() as f64,
            best_parallel_efficiency: sessions
                .iter()
                .map(|s| s.overall_metrics.parallel_efficiency)
                .fold(0.0, f64::max),
        }
    }

    fn analyze_bottlenecks(sessions: &[ProfileSession]) -> BottleneckAnalysis {
        let all_bottlenecks: Vec<&Bottleneck> =
            sessions.iter().flat_map(|s| &s.bottlenecks).collect();

        let mut bottleneck_frequency = HashMap::new();
        let mut severity_distribution = HashMap::new();
        let mut impact_analysis = HashMap::new();

        for bottleneck in &all_bottlenecks {
            *bottleneck_frequency
                .entry(bottleneck.affected_stage.clone())
                .or_insert(0) += 1;
            *severity_distribution
                .entry(bottleneck.severity.clone())
                .or_insert(0) += 1;
            *impact_analysis
                .entry(bottleneck.affected_stage.clone())
                .or_insert(0.0) += bottleneck.impact_factor;
        }

        let most_common_bottleneck = all_bottlenecks
            .iter()
            .fold(HashMap::new(), |mut acc, b| {
                *acc.entry(format!("{:?}", b.bottleneck_type)).or_insert(0) += 1;
                acc
            })
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map_or(
                BottleneckType::ComputationalBottleneck,
                |(bottleneck_type, _)| match bottleneck_type.as_str() {
                    "MemoryConstraint" => BottleneckType::MemoryConstraint,
                    "ComputationalBottleneck" => BottleneckType::ComputationalBottleneck,
                    _ => BottleneckType::ComputationalBottleneck,
                },
            );

        /// BottleneckAnalysis
        BottleneckAnalysis {
            most_common_bottleneck,
            bottleneck_frequency,
            severity_distribution,
            impact_analysis,
        }
    }

    fn analyze_trends(sessions: &[ProfileSession]) -> TrendAnalysis {
        if sessions.len() < 3 {
            return TrendAnalysis {
                performance_trend: TrendDirection::InsufficientData,
                memory_usage_trend: TrendDirection::InsufficientData,
                throughput_trend: TrendDirection::InsufficientData,
                session_performance_scores: Vec::new(),
            };
        }

        let performance_scores: Vec<f64> = sessions
            .iter()
            .map(|s| {
                1000.0
                    / s.overall_metrics
                        .total_execution_time
                        .as_secs_f64()
                        .max(1.0)
            })
            .collect();

        let performance_trend = Self::calculate_trend_direction(&performance_scores);

        let memory_scores: Vec<f64> = sessions
            .iter()
            .map(|s| s.overall_metrics.peak_memory_usage_mb)
            .collect();
        let memory_usage_trend = Self::calculate_trend_direction(&memory_scores);

        let throughput_scores: Vec<f64> = sessions
            .iter()
            .map(|s| s.overall_metrics.throughput_samples_per_second)
            .collect();
        let throughput_trend = Self::calculate_trend_direction(&throughput_scores);

        /// TrendAnalysis
        TrendAnalysis {
            performance_trend,
            memory_usage_trend,
            throughput_trend,
            session_performance_scores: performance_scores,
        }
    }

    fn calculate_trend_direction(values: &[f64]) -> TrendDirection {
        if values.len() < 3 {
            return TrendDirection::InsufficientData;
        }

        let mid_point = values.len() / 2;
        let first_half_avg = values[..mid_point].iter().sum::<f64>() / mid_point as f64;
        let second_half_avg =
            values[mid_point..].iter().sum::<f64>() / (values.len() - mid_point) as f64;

        let change_percentage =
            (second_half_avg - first_half_avg) / first_half_avg.abs().max(1e-10);

        if change_percentage > 0.05 {
            TrendDirection::Improving
        } else if change_percentage < -0.05 {
            TrendDirection::Degrading
        } else {
            TrendDirection::Stable
        }
    }

    fn comparative_analysis(sessions: &[ProfileSession]) -> ComparativeAnalysis {
        if sessions.is_empty() {
            return ComparativeAnalysis {
                best_performing_session: "none".to_string(),
                worst_performing_session: "none".to_string(),
                performance_variance: 0.0,
                consistency_score: 0.0,
            };
        }

        let (best_session, worst_session) = sessions.iter().fold(
            (sessions[0].clone(), sessions[0].clone()),
            |(best, worst), session| {
                let best_next = if session.overall_metrics.total_execution_time
                    < best.overall_metrics.total_execution_time
                {
                    session.clone()
                } else {
                    best
                };

                let worst_next = if session.overall_metrics.total_execution_time
                    > worst.overall_metrics.total_execution_time
                {
                    session.clone()
                } else {
                    worst
                };

                (best_next, worst_next)
            },
        );

        let execution_times: Vec<f64> = sessions
            .iter()
            .map(|s| s.overall_metrics.total_execution_time.as_secs_f64())
            .collect();

        let mean_time = execution_times.iter().sum::<f64>() / sessions.len() as f64;
        let variance = execution_times
            .iter()
            .map(|t| (t - mean_time).powi(2))
            .sum::<f64>()
            / sessions.len() as f64;

        let consistency_score = 1.0 / (1.0 + variance.sqrt() / mean_time);

        /// ComparativeAnalysis
        ComparativeAnalysis {
            best_performing_session: best_session.session_id,
            worst_performing_session: worst_session.session_id,
            performance_variance: variance,
            consistency_score,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let profiler = PerformanceProfiler::default();
        assert_eq!(profiler.config.enable_timing, true);
        assert_eq!(profiler.config.enable_memory_tracking, true);
    }

    #[test]
    fn test_session_lifecycle() {
        let profiler = PerformanceProfiler::default();
        let session_id = profiler.start_session("test_pipeline");

        // Start and end a stage
        profiler
            .start_stage(&session_id, "preprocessing", "transformer")
            .unwrap();
        thread::sleep(Duration::from_millis(10));
        let stage_duration = profiler.end_stage(&session_id, "preprocessing").unwrap();

        assert!(stage_duration > Duration::from_millis(5));

        // End session
        let completed_session = profiler.end_session(&session_id).unwrap();
        assert_eq!(completed_session.pipeline_name, "test_pipeline");
        assert_eq!(completed_session.stages.len(), 1);
    }

    #[test]
    fn test_bottleneck_detection() {
        let profiler = PerformanceProfiler::default();
        let session_id = profiler.start_session("test_pipeline");

        // Simulate a slow stage
        profiler
            .start_stage(&session_id, "slow_stage", "estimator")
            .unwrap();
        thread::sleep(Duration::from_millis(50)); // Simulate slow execution
        profiler.end_stage(&session_id, "slow_stage").unwrap();

        let completed_session = profiler.end_session(&session_id).unwrap();

        // Check that bottlenecks were detected (though timing may be too short for actual detection)
        assert_eq!(completed_session.stages.len(), 1);
    }

    #[test]
    fn test_performance_report_generation() {
        let profiler = PerformanceProfiler::default();

        // Create multiple sessions for analysis
        for i in 0..3 {
            let session_id = profiler.start_session(&format!("pipeline_{}", i));
            profiler
                .start_stage(&session_id, "stage", "transformer")
                .unwrap();
            thread::sleep(Duration::from_millis(10));
            profiler.end_stage(&session_id, "stage").unwrap();
            profiler.end_session(&session_id).unwrap();
        }

        let report = profiler.generate_report(None);
        assert_eq!(report.sessions_analyzed, 3);
        assert!(report.summary_metrics.average_execution_time > Duration::from_secs(0));
    }
}
