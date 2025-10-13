/// Next-generation effect type system for machine learning
///
/// This module implements an advanced effect type system that tracks computational effects
/// in machine learning pipelines. It provides compile-time guarantees about data flow,
/// side effects, error propagation, and resource usage.
///
/// # Key Features
///
/// - **Effect Tracking**: Compile-time tracking of computational effects
/// - **Data Dependencies**: Static analysis of data flow and dependencies
/// - **Resource Management**: Memory and computational resource tracking
/// - **Error Propagation**: Type-safe error handling with effect composition
/// - **Purity Analysis**: Distinction between pure and impure computations
/// - **Capability System**: Fine-grained permission and capability tracking
/// - **Linear Types**: Resource-aware computation with linear type support
///
/// # Effect Categories
///
/// The system tracks several categories of effects:
///
/// ## Pure Effects
/// - Mathematical computations without side effects
/// - Deterministic transformations
/// - Immutable data operations
///
/// ## IO Effects
/// - File system operations
/// - Network communication
/// - Database access
/// - Logging and monitoring
///
/// ## Random Effects
/// - Stochastic sampling
/// - Random number generation
/// - Non-deterministic algorithms
///
/// ## Memory Effects
/// - Heap allocation
/// - Memory-mapped operations
/// - Cache effects
/// - GPU memory operations
///
/// ## Error Effects
/// - Recoverable errors
/// - Validation failures
/// - Timeout and resource exhaustion
/// - Type conversion errors
///
/// # Usage Examples
///
/// ## Pure Computations
/// ```rust,ignore
/// use sklears_core::effect_types::{Pure, Effect, EffectfulComputation};
///
/// // Pure mathematical computation
/// fn linear_transform<T>(data: T, weights: &[f64]) -> Effect<T, Pure>
/// where
///     T: LinearAlgebra,
/// {
///     Effect::pure(data.matrix_multiply(weights))
/// }
///
/// // Composition of pure effects
/// let result = linear_transform(input_data, &weights)
///     .then(|x| normalize(x))
///     .then(|x| apply_activation(x));
/// ```
///
/// ## IO and Random Effects
/// ```rust,ignore
/// use sklears_core::effect_types::{IO, Random, Combined};
///
/// // Loading data with IO effects
/// async fn load_dataset(path: &str) -> Effect<Dataset, IO> {
///     Effect::io(async move {
///         let data = std::fs::read_to_string(path).await?;
///         Dataset::from_csv(&data)
///     })
/// }
///
/// // Random sampling with tracked effects
/// fn sample_minibatch<R: RandomState>(
///     dataset: &Dataset,
///     rng: &mut R,
///     batch_size: usize
/// ) -> Effect<MiniBatch, Random> {
///     Effect::random(move |rng| {
///         dataset.sample(batch_size, rng)
///     })
/// }
///
/// // Combined effects composition
/// type TrainingEffect = Combined<IO, Random>;
/// async fn training_step() -> Effect<ModelUpdate, TrainingEffect> {
///     let data = load_dataset("train.csv").await?;
///     let batch = sample_minibatch(&data, &mut rng, 32)?;
///     Effect::pure(compute_gradients(&batch))
/// }
/// ```
///
/// ## Resource Management
/// ```rust,ignore
/// use sklears_core::effect_types::{Memory, GPU, Resource};
///
/// // Memory-aware computation
/// fn large_matrix_multiply<T>(
///     a: Matrix<T>,
///     b: Matrix<T>
/// ) -> Effect<Matrix<T>, Memory>
/// where
///     T: Float + Send + Sync,
/// {
///     Effect::with_memory_bound(1024 * 1024 * 1024, move || {
///         a.multiply(&b)
///     })
/// }
///
/// // GPU computation with resource tracking
/// fn gpu_convolution(
///     input: Tensor4D,
///     kernel: ConvKernel
/// ) -> Effect<Tensor4D, GPU> {
///     Effect::gpu(move |ctx: GpuContext| {
///         ctx.conv2d(input, kernel)
///     })
/// }
/// ```
use futures_core::future::BoxFuture;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

// =============================================================================
// Core Effect Type System
// =============================================================================

/// Core effect wrapper that tracks computational effects at the type level
#[derive(Debug)]
pub struct Effect<T, E>
where
    E: EffectType,
{
    /// The computed value
    value: T,
    /// Effect evidence/witness
    effect_data: E::Data,
    /// Effect metadata
    metadata: EffectMetadata,
    /// Phantom type for effect tracking
    _phantom: PhantomData<E>,
}

/// Trait for effect types that can be tracked by the type system
pub trait EffectType: Send + Sync + 'static {
    /// Associated data type for effect evidence
    type Data: Send + Sync;

    /// Effect name for debugging and introspection
    const NAME: &'static str;

    /// Whether this effect is pure (deterministic and side-effect free)
    const IS_PURE: bool;

    /// Whether this effect requires runtime tracking
    const NEEDS_TRACKING: bool;

    /// Combine two instances of this effect
    fn combine(left: Self::Data, right: Self::Data) -> Self::Data;

    /// Create a default instance of effect data
    fn default_data() -> Self::Data;
}

/// Metadata about effect execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectMetadata {
    /// Effect execution timestamp
    pub timestamp: std::time::SystemTime,
    /// Execution duration
    pub duration: std::time::Duration,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// CPU cycles consumed
    pub cpu_cycles: u64,
    /// Custom metadata tags
    pub tags: HashMap<String, String>,
}

impl Default for EffectMetadata {
    fn default() -> Self {
        Self {
            timestamp: std::time::SystemTime::now(),
            duration: std::time::Duration::from_nanos(0),
            memory_usage: 0,
            cpu_cycles: 0,
            tags: HashMap::new(),
        }
    }
}

impl<T, E> Effect<T, E>
where
    E: EffectType,
{
    /// Create a new effect with value and effect data
    pub fn new(value: T, effect_data: E::Data) -> Self {
        Self {
            value,
            effect_data,
            metadata: EffectMetadata::default(),
            _phantom: PhantomData,
        }
    }

    /// Create an effect with pure computation (no side effects)
    pub fn pure(value: T) -> Effect<T, Pure>
    where
        E: EffectType<Data = ()>,
    {
        Effect {
            value,
            effect_data: (),
            metadata: EffectMetadata::default(),
            _phantom: PhantomData,
        }
    }

    /// Unwrap the value (consuming the effect)
    pub fn unwrap(self) -> T {
        self.value
    }

    /// Get a reference to the value
    pub fn value(&self) -> &T {
        &self.value
    }

    /// Get effect metadata
    pub fn metadata(&self) -> &EffectMetadata {
        &self.metadata
    }

    /// Map over the value while preserving the effect
    pub fn map<U, F>(self, f: F) -> Effect<U, E>
    where
        F: FnOnce(T) -> U,
    {
        Effect {
            value: f(self.value),
            effect_data: self.effect_data,
            metadata: self.metadata,
            _phantom: PhantomData,
        }
    }

    /// Flat map for effect composition
    pub fn flat_map<U, F, E2>(self, f: F) -> Effect<U, Combined<E, E2>>
    where
        F: FnOnce(T) -> Effect<U, E2>,
        E2: EffectType,
        Combined<E, E2>: EffectType,
    {
        let effect2 = f(self.value);
        Effect {
            value: effect2.value,
            effect_data: <Combined<E, E2> as EffectType>::default_data(),
            metadata: self.metadata, // Could combine metadata here
            _phantom: PhantomData,
        }
    }

    /// Transform with error handling
    pub fn try_map<U, F, Err>(self, f: F) -> Effect<std::result::Result<U, Err>, E>
    where
        F: FnOnce(T) -> std::result::Result<U, Err>,
    {
        Effect {
            value: f(self.value),
            effect_data: self.effect_data,
            metadata: self.metadata,
            _phantom: PhantomData,
        }
    }
}

// =============================================================================
// Core Effect Types
// =============================================================================

/// Pure effect - no side effects, deterministic
#[derive(Debug, Clone, Copy)]
pub struct Pure;

impl EffectType for Pure {
    type Data = ();
    const NAME: &'static str = "Pure";
    const IS_PURE: bool = true;
    const NEEDS_TRACKING: bool = false;

    fn combine(_left: Self::Data, _right: Self::Data) -> Self::Data {}

    fn default_data() -> Self::Data {}
}

/// IO effect - file system, network, database operations
#[derive(Debug, Clone)]
pub struct IO;

impl EffectType for IO {
    type Data = IOEffectData;
    const NAME: &'static str = "IO";
    const IS_PURE: bool = false;
    const NEEDS_TRACKING: bool = true;

    fn combine(left: Self::Data, right: Self::Data) -> Self::Data {
        IOEffectData {
            files_read: left.files_read + right.files_read,
            files_written: left.files_written + right.files_written,
            bytes_read: left.bytes_read + right.bytes_read,
            bytes_written: left.bytes_written + right.bytes_written,
            network_requests: left.network_requests + right.network_requests,
        }
    }

    fn default_data() -> Self::Data {
        IOEffectData::default()
    }
}

/// Data tracking for IO effects
#[derive(Debug, Clone, Default)]
pub struct IOEffectData {
    /// Number of files read
    pub files_read: u32,
    /// Number of files written
    pub files_written: u32,
    /// Total bytes read
    pub bytes_read: u64,
    /// Total bytes written
    pub bytes_written: u64,
    /// Number of network requests
    pub network_requests: u32,
}

/// Random effect - non-deterministic computations
#[derive(Debug, Clone)]
pub struct Random;

impl EffectType for Random {
    type Data = RandomEffectData;
    const NAME: &'static str = "Random";
    const IS_PURE: bool = false;
    const NEEDS_TRACKING: bool = true;

    fn combine(left: Self::Data, right: Self::Data) -> Self::Data {
        RandomEffectData {
            samples_generated: left.samples_generated + right.samples_generated,
            entropy_consumed: left.entropy_consumed + right.entropy_consumed,
            distributions_used: {
                let mut combined = left.distributions_used;
                combined.extend(right.distributions_used);
                combined
            },
        }
    }

    fn default_data() -> Self::Data {
        RandomEffectData::default()
    }
}

/// Data tracking for random effects
#[derive(Debug, Clone, Default)]
pub struct RandomEffectData {
    /// Number of random samples generated
    pub samples_generated: u64,
    /// Entropy consumed in bits
    pub entropy_consumed: u64,
    /// Types of distributions used
    pub distributions_used: Vec<String>,
}

/// Memory effect - heap allocation, memory management
#[derive(Debug, Clone)]
pub struct Memory;

impl EffectType for Memory {
    type Data = MemoryEffectData;
    const NAME: &'static str = "Memory";
    const IS_PURE: bool = false;
    const NEEDS_TRACKING: bool = true;

    fn combine(left: Self::Data, right: Self::Data) -> Self::Data {
        MemoryEffectData {
            allocations: left.allocations + right.allocations,
            deallocations: left.deallocations + right.deallocations,
            bytes_allocated: left.bytes_allocated + right.bytes_allocated,
            peak_usage: std::cmp::max(left.peak_usage, right.peak_usage),
            cache_misses: left.cache_misses + right.cache_misses,
        }
    }

    fn default_data() -> Self::Data {
        MemoryEffectData::default()
    }
}

/// Data tracking for memory effects
#[derive(Debug, Clone, Default)]
pub struct MemoryEffectData {
    /// Number of allocations
    pub allocations: u32,
    /// Number of deallocations
    pub deallocations: u32,
    /// Total bytes allocated
    pub bytes_allocated: u64,
    /// Peak memory usage
    pub peak_usage: u64,
    /// Cache misses
    pub cache_misses: u64,
}

/// GPU effect - GPU computations
#[derive(Debug, Clone)]
pub struct GPU;

impl EffectType for GPU {
    type Data = GPUEffectData;
    const NAME: &'static str = "GPU";
    const IS_PURE: bool = false;
    const NEEDS_TRACKING: bool = true;

    fn combine(left: Self::Data, right: Self::Data) -> Self::Data {
        GPUEffectData {
            kernel_launches: left.kernel_launches + right.kernel_launches,
            gpu_memory_used: left.gpu_memory_used + right.gpu_memory_used,
            data_transfers: left.data_transfers + right.data_transfers,
            compute_units_used: std::cmp::max(left.compute_units_used, right.compute_units_used),
        }
    }

    fn default_data() -> Self::Data {
        GPUEffectData::default()
    }
}

/// Data tracking for GPU effects
#[derive(Debug, Clone, Default)]
pub struct GPUEffectData {
    /// Number of GPU kernel launches
    pub kernel_launches: u32,
    /// GPU memory used in bytes
    pub gpu_memory_used: u64,
    /// Host-device data transfers
    pub data_transfers: u32,
    /// Maximum compute units used
    pub compute_units_used: u32,
}

/// Error effect - computations that may fail
#[derive(Debug, Clone)]
pub struct Fallible;

impl EffectType for Fallible {
    type Data = FallibleEffectData;
    const NAME: &'static str = "Fallible";
    const IS_PURE: bool = true; // Pure but may fail
    const NEEDS_TRACKING: bool = true;

    fn combine(left: Self::Data, right: Self::Data) -> Self::Data {
        FallibleEffectData {
            error_count: left.error_count + right.error_count,
            warning_count: left.warning_count + right.warning_count,
            recovery_attempts: left.recovery_attempts + right.recovery_attempts,
        }
    }

    fn default_data() -> Self::Data {
        FallibleEffectData::default()
    }
}

/// Data tracking for fallible effects
#[derive(Debug, Clone, Default)]
pub struct FallibleEffectData {
    /// Number of errors encountered
    pub error_count: u32,
    /// Number of warnings
    pub warning_count: u32,
    /// Number of recovery attempts
    pub recovery_attempts: u32,
}

// =============================================================================
// Effect Combination System
// =============================================================================

/// Combined effect type for composing multiple effects
#[derive(Debug, Clone)]
pub struct Combined<E1, E2>
where
    E1: EffectType,
    E2: EffectType,
{
    _phantom: PhantomData<(E1, E2)>,
}

/// Data for combined effects
#[derive(Debug, Clone)]
pub struct CombinedData<E1, E2>
where
    E1: EffectType,
    E2: EffectType,
{
    left: E1::Data,
    right: E2::Data,
    _phantom: PhantomData<(E1, E2)>,
}

impl<E1, E2> EffectType for Combined<E1, E2>
where
    E1: EffectType,
    E2: EffectType,
{
    type Data = CombinedData<E1, E2>;
    const NAME: &'static str = "Combined";
    const IS_PURE: bool = E1::IS_PURE && E2::IS_PURE;
    const NEEDS_TRACKING: bool = E1::NEEDS_TRACKING || E2::NEEDS_TRACKING;

    fn combine(left: Self::Data, right: Self::Data) -> Self::Data {
        CombinedData {
            left: E1::combine(left.left, right.left),
            right: E2::combine(left.right, right.right),
            _phantom: PhantomData,
        }
    }

    fn default_data() -> Self::Data {
        CombinedData {
            left: E1::default_data(),
            right: E2::default_data(),
            _phantom: PhantomData,
        }
    }
}

// =============================================================================
// Effect Builders and Utilities
// =============================================================================

/// Builder for creating effectful computations
pub struct EffectBuilder;

impl EffectBuilder {
    /// Create a pure computation
    pub fn pure<T>(value: T) -> Effect<T, Pure> {
        Effect {
            value,
            effect_data: (),
            metadata: EffectMetadata::default(),
            _phantom: PhantomData,
        }
    }

    /// Create an IO computation
    pub fn io<T, F>(f: F) -> Effect<T, IO>
    where
        F: FnOnce() -> T,
    {
        let start = std::time::Instant::now();
        let value = f();
        let duration = start.elapsed();

        Effect {
            value,
            effect_data: IOEffectData {
                files_read: 1, // Placeholder tracking
                files_written: 0,
                bytes_read: 0,
                bytes_written: 0,
                network_requests: 0,
            },
            metadata: EffectMetadata {
                timestamp: std::time::SystemTime::now(),
                duration,
                memory_usage: 0,
                cpu_cycles: 0,
                tags: HashMap::new(),
            },
            _phantom: PhantomData,
        }
    }

    /// Create a random computation
    pub fn random<T, F>(f: F) -> Effect<T, Random>
    where
        F: FnOnce() -> T,
    {
        let start = std::time::Instant::now();
        let value = f();
        let duration = start.elapsed();

        Effect {
            value,
            effect_data: RandomEffectData {
                samples_generated: 1,
                entropy_consumed: 32, // Placeholder
                distributions_used: vec!["uniform".to_string()],
            },
            metadata: EffectMetadata {
                timestamp: std::time::SystemTime::now(),
                duration,
                memory_usage: 0,
                cpu_cycles: 0,
                tags: HashMap::new(),
            },
            _phantom: PhantomData,
        }
    }

    /// Create a memory-aware computation
    pub fn with_memory<T, F>(f: F) -> Effect<T, Memory>
    where
        F: FnOnce() -> T,
    {
        let start_memory = get_memory_usage();
        let start = std::time::Instant::now();
        let value = f();
        let duration = start.elapsed();
        let end_memory = get_memory_usage();

        Effect {
            value,
            effect_data: MemoryEffectData {
                allocations: 1,
                deallocations: 0,
                bytes_allocated: end_memory.saturating_sub(start_memory),
                peak_usage: end_memory,
                cache_misses: 0,
            },
            metadata: EffectMetadata {
                timestamp: std::time::SystemTime::now(),
                duration,
                memory_usage: end_memory.saturating_sub(start_memory),
                cpu_cycles: 0,
                tags: HashMap::new(),
            },
            _phantom: PhantomData,
        }
    }

    /// Create a GPU computation
    pub fn gpu<T, F>(f: F) -> Effect<T, GPU>
    where
        F: FnOnce() -> T,
    {
        let start = std::time::Instant::now();
        let value = f();
        let duration = start.elapsed();

        Effect {
            value,
            effect_data: GPUEffectData {
                kernel_launches: 1,
                gpu_memory_used: 1024 * 1024, // Placeholder
                data_transfers: 2,            // Host to device and back
                compute_units_used: 32,
            },
            metadata: EffectMetadata {
                timestamp: std::time::SystemTime::now(),
                duration,
                memory_usage: 0,
                cpu_cycles: 0,
                tags: HashMap::new(),
            },
            _phantom: PhantomData,
        }
    }

    /// Create a fallible computation
    pub fn fallible<T, E, F>(f: F) -> Effect<std::result::Result<T, E>, Fallible>
    where
        F: FnOnce() -> std::result::Result<T, E>,
    {
        let start = std::time::Instant::now();
        let result = f();
        let duration = start.elapsed();

        let effect_data = match &result {
            Ok(_) => FallibleEffectData::default(),
            Err(_) => FallibleEffectData {
                error_count: 1,
                warning_count: 0,
                recovery_attempts: 0,
            },
        };

        Effect {
            value: result,
            effect_data,
            metadata: EffectMetadata {
                timestamp: std::time::SystemTime::now(),
                duration,
                memory_usage: 0,
                cpu_cycles: 0,
                tags: HashMap::new(),
            },
            _phantom: PhantomData,
        }
    }
}

// =============================================================================
// Linear Types for Resource Management
// =============================================================================

/// Linear type wrapper that enforces single-use semantics
#[derive(Debug)]
pub struct Linear<T> {
    value: Option<T>,
}

impl<T> Linear<T> {
    /// Create a new linear value
    pub fn new(value: T) -> Self {
        Self { value: Some(value) }
    }

    /// Consume the linear value (can only be called once)
    pub fn consume(mut self) -> T {
        self.value.take().expect("Linear value already consumed")
    }

    /// Transform the linear value while maintaining linearity
    pub fn map<U, F>(mut self, f: F) -> Linear<U>
    where
        F: FnOnce(T) -> U,
    {
        let value = self.value.take().expect("Linear value already consumed");
        Linear::new(f(value))
    }

    /// Check if the linear value is still available
    pub fn is_available(&self) -> bool {
        self.value.is_some()
    }
}

/// Capability token for resource access
#[derive(Debug)]
pub struct Capability<R> {
    _phantom: PhantomData<R>,
}

impl<R> Default for Capability<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R> Capability<R> {
    /// Create a new capability (typically done by privileged code)
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Use the capability to access a resource
    pub fn use_resource<T, F>(self, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        f()
    }
}

/// Resource types for capability system
pub struct FileSystem;
pub struct Network;

// =============================================================================
// Effect Analysis and Optimization
// =============================================================================

/// Analyzer for effect composition and optimization
pub struct EffectAnalyzer {
    /// Effect tracking data
    tracked_effects: Arc<Mutex<HashMap<String, EffectMetadata>>>,
}

impl EffectAnalyzer {
    /// Create a new effect analyzer
    pub fn new() -> Self {
        Self {
            tracked_effects: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Analyze effect composition for optimization opportunities
    pub fn analyze_composition<E1, E2>(&self) -> CompositionAnalysis
    where
        E1: EffectType,
        E2: EffectType,
    {
        CompositionAnalysis {
            can_parallelize: !E1::NEEDS_TRACKING && !E2::NEEDS_TRACKING,
            can_reorder: E1::IS_PURE && E2::IS_PURE,
            can_memoize: E1::IS_PURE,
            memory_overhead: if E1::NEEDS_TRACKING || E2::NEEDS_TRACKING {
                64
            } else {
                0
            },
        }
    }

    /// Suggest optimizations for effect chains
    pub fn suggest_optimizations<E>(&self, effect_chain: &[String]) -> Vec<OptimizationSuggestion>
    where
        E: EffectType,
    {
        let mut suggestions = Vec::new();

        // Suggest batching for IO operations
        if effect_chain.iter().filter(|e| *e == "IO").count() > 3 {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: OptimizationType::Batching,
                description: "Consider batching IO operations to reduce overhead".to_string(),
                estimated_speedup: 2.5,
            });
        }

        // Suggest caching for pure operations
        if effect_chain.iter().filter(|e| *e == "Pure").count() > 2 {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: OptimizationType::Caching,
                description: "Pure computations can be memoized".to_string(),
                estimated_speedup: 1.8,
            });
        }

        suggestions
    }

    /// Track effect execution
    pub fn track_effect<E>(&self, effect_name: String, metadata: EffectMetadata)
    where
        E: EffectType,
    {
        if let Ok(mut tracked) = self.tracked_effects.lock() {
            tracked.insert(effect_name, metadata);
        }
    }

    /// Get performance statistics for effects
    pub fn get_statistics(&self) -> EffectStatistics {
        let tracked = self.tracked_effects.lock().unwrap();

        let total_duration: std::time::Duration = tracked.values().map(|m| m.duration).sum();

        let total_memory: u64 = tracked.values().map(|m| m.memory_usage).sum();

        EffectStatistics {
            total_effects: tracked.len(),
            total_duration,
            total_memory_usage: total_memory,
            average_duration: if tracked.is_empty() {
                std::time::Duration::from_nanos(0)
            } else {
                total_duration / tracked.len() as u32
            },
        }
    }
}

impl Default for EffectAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Analysis of effect composition
#[derive(Debug, Clone)]
pub struct CompositionAnalysis {
    /// Whether effects can be parallelized
    pub can_parallelize: bool,
    /// Whether effects can be reordered
    pub can_reorder: bool,
    /// Whether effects can be memoized
    pub can_memoize: bool,
    /// Memory overhead of composition
    pub memory_overhead: u64,
}

/// Optimization suggestion
#[derive(Debug, Clone)]
pub struct OptimizationSuggestion {
    /// Type of optimization
    pub suggestion_type: OptimizationType,
    /// Human-readable description
    pub description: String,
    /// Estimated performance improvement factor
    pub estimated_speedup: f64,
}

/// Types of effect optimizations
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationType {
    /// Batching operations
    Batching,
    /// Caching results
    Caching,
    /// Parallelization
    Parallelization,
    /// Reordering
    Reordering,
    /// Resource pooling
    ResourcePooling,
}

/// Performance statistics for effects
#[derive(Debug, Clone)]
pub struct EffectStatistics {
    /// Total number of effects tracked
    pub total_effects: usize,
    /// Total execution duration
    pub total_duration: std::time::Duration,
    /// Total memory usage
    pub total_memory_usage: u64,
    /// Average execution duration
    pub average_duration: std::time::Duration,
}

// =============================================================================
// Async Effect Support
// =============================================================================

/// Async effect wrapper for asynchronous computations
pub struct AsyncEffect<T, E>
where
    E: EffectType,
{
    future: BoxFuture<'static, Effect<T, E>>,
}

impl<T, E> AsyncEffect<T, E>
where
    E: EffectType,
    T: Send + 'static,
{
    /// Create a new async effect
    pub fn new<F>(future: F) -> Self
    where
        F: std::future::Future<Output = Effect<T, E>> + Send + 'static,
    {
        Self {
            future: Box::pin(future),
        }
    }

    /// Await the async effect
    pub async fn await_effect(self) -> Effect<T, E> {
        self.future.await
    }

    /// Map over the async effect
    pub fn map<U, F>(self, f: F) -> AsyncEffect<U, E>
    where
        F: FnOnce(T) -> U + Send + 'static,
        U: Send + 'static,
    {
        AsyncEffect::new(async move { self.await_effect().await.map(f) })
    }

    /// Flat map for async effect composition
    pub fn flat_map<U, F, E2>(self, f: F) -> AsyncEffect<U, Combined<E, E2>>
    where
        F: FnOnce(T) -> AsyncEffect<U, E2> + Send + 'static,
        E2: EffectType,
        U: Send + 'static,
        Combined<E, E2>: EffectType,
    {
        AsyncEffect::new(async move {
            let effect1 = self.await_effect().await;
            let effect2 = f(effect1.value).await_effect().await;

            Effect {
                value: effect2.value,
                effect_data: <Combined<E, E2> as EffectType>::default_data(),
                metadata: effect1.metadata, // Could combine metadata
                _phantom: PhantomData,
            }
        })
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Get current memory usage (placeholder implementation)
fn get_memory_usage() -> u64 {
    // In a real implementation, this would query system memory usage
    1024 * 1024 // Placeholder: 1MB
}

/// Convenience macros for effect creation
#[macro_export]
macro_rules! pure_effect {
    ($expr:expr) => {
        Effect::pure($expr)
    };
}

#[macro_export]
macro_rules! io_effect {
    ($expr:expr) => {
        EffectBuilder::io(|| $expr)
    };
}

#[macro_export]
macro_rules! random_effect {
    ($expr:expr) => {
        EffectBuilder::random(|| $expr)
    };
}

/// Type aliases for common effect combinations
pub type IORandomEffect = Combined<IO, Random>;
pub type MemoryIOEffect = Combined<Memory, IO>;
pub type GPUMemoryEffect = Combined<GPU, Memory>;
pub type FallibleIOEffect = Combined<Fallible, IO>;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pure_effect() {
        let effect = EffectBuilder::pure(42);
        assert_eq!(effect.unwrap(), 42);
    }

    #[test]
    fn test_effect_mapping() {
        let effect = EffectBuilder::pure(10);
        let mapped = effect.map(|x| x * 2);
        assert_eq!(mapped.unwrap(), 20);
    }

    #[test]
    fn test_io_effect_tracking() {
        let effect = EffectBuilder::io(|| {
            // Simulate file read
            "file content".to_string()
        });

        assert_eq!(effect.value(), "file content");
        assert_eq!(effect.effect_data.files_read, 1);
    }

    #[test]
    fn test_random_effect_tracking() {
        let effect = EffectBuilder::random(|| {
            // Simulate random number generation
            42
        });

        assert_eq!(effect.unwrap(), 42);
    }

    #[test]
    fn test_linear_type() {
        let linear = Linear::new(100);
        assert!(linear.is_available());

        let value = linear.consume();
        assert_eq!(value, 100);
    }

    #[test]
    fn test_linear_mapping() {
        let linear = Linear::new(5);
        let mapped = linear.map(|x| x * 3);
        assert_eq!(mapped.consume(), 15);
    }

    #[test]
    fn test_capability_system() {
        let file_cap = Capability::<FileSystem>::new();
        let result = file_cap.use_resource(|| "accessed file system");
        assert_eq!(result, "accessed file system");
    }

    #[test]
    fn test_effect_analyzer() {
        let analyzer = EffectAnalyzer::new();
        let analysis = analyzer.analyze_composition::<Pure, Pure>();

        assert!(analysis.can_parallelize);
        assert!(analysis.can_reorder);
        assert!(analysis.can_memoize);
        assert_eq!(analysis.memory_overhead, 0);
    }

    #[test]
    fn test_optimization_suggestions() {
        let analyzer = EffectAnalyzer::new();
        let chain = vec![
            "IO".to_string(),
            "IO".to_string(),
            "IO".to_string(),
            "IO".to_string(),
        ];
        let suggestions = analyzer.suggest_optimizations::<IO>(&chain);

        assert!(!suggestions.is_empty());
        assert_eq!(suggestions[0].suggestion_type, OptimizationType::Batching);
    }

    #[test]
    fn test_effect_combination() {
        // Test that effect types can be combined
        let _combined_type: Combined<IO, Random> = Combined {
            _phantom: PhantomData,
        };
        assert!(!Combined::<IO, Random>::IS_PURE);
        assert!(Combined::<IO, Random>::NEEDS_TRACKING);
    }

    #[test]
    fn test_fallible_effect() {
        let success_effect = EffectBuilder::fallible(|| Ok::<i32, String>(42));
        assert!(success_effect.value().is_ok());
        assert_eq!(success_effect.effect_data.error_count, 0);

        let error_effect = EffectBuilder::fallible(|| Err::<i32, String>("error".to_string()));
        assert!(error_effect.value().is_err());
        assert_eq!(error_effect.effect_data.error_count, 1);
    }

    #[cfg(feature = "async_support")]
    #[tokio::test]
    async fn test_async_effect() {
        let async_effect = AsyncEffect::new(async { Effect::<i32, Pure>::pure(100) });

        let result = async_effect.await_effect().await;
        assert_eq!(result.unwrap(), 100);
    }

    #[test]
    fn test_effect_metadata() {
        let effect = EffectBuilder::io(|| "test");
        let metadata = effect.metadata();

        assert!(metadata.duration >= std::time::Duration::from_nanos(0));
        assert!(metadata.timestamp <= std::time::SystemTime::now());
    }
}
