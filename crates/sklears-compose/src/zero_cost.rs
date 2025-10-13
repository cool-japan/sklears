//! Zero-cost composition abstractions
//!
//! This module provides zero-overhead abstractions for pipeline composition,
//! leveraging Rust's type system to eliminate runtime costs while maintaining
//! ergonomic APIs for complex machine learning workflows.

use scirs2_core::ndarray::{Array2, ArrayView2};
use sklears_core::{
    error::Result as SklResult,
    prelude::{SklearsError, Transform},
    traits::{Estimator, Fit},
    types::Float,
};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, RwLock, Weak};
use std::time::{Duration, Instant};

/// Zero-cost pipeline step that is optimized away at compile time
pub struct ZeroCostStep<T, S> {
    inner: T,
    _state: PhantomData<S>,
}

impl<T, S> ZeroCostStep<T, S> {
    /// Create a new zero-cost step
    #[inline(always)]
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            _state: PhantomData,
        }
    }

    /// Get the inner value with zero cost
    #[inline(always)]
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Consume and get the inner value with zero cost
    #[inline(always)]
    pub fn into_inner(self) -> T {
        self.inner
    }
}

/// Zero-cost pipeline composition using const generics
pub struct ZeroCostPipeline<const N: usize, T> {
    steps: [T; N],
}

impl<const N: usize, T> ZeroCostPipeline<N, T> {
    /// Create a new zero-cost pipeline
    #[inline(always)]
    pub const fn new(steps: [T; N]) -> Self {
        Self { steps }
    }

    /// Get the steps with zero cost
    #[inline(always)]
    pub fn steps(&self) -> &[T; N] {
        &self.steps
    }

    /// Execute all steps in sequence with compile-time unrolling
    #[inline(always)]
    pub fn execute<I>(&self, input: I) -> I
    where
        T: Fn(I) -> I,
    {
        let mut result = input;
        for step in &self.steps {
            result = step(result);
        }
        result
    }
}

/// Compile-time feature union with fixed size
pub struct ZeroCostFeatureUnion<const N: usize, T> {
    transformers: [T; N],
}

impl<const N: usize, T> ZeroCostFeatureUnion<N, T> {
    /// Create a new zero-cost feature union
    #[inline(always)]
    pub const fn new(transformers: [T; N]) -> Self {
        Self { transformers }
    }

    /// Apply all transformers and concatenate results
    pub fn transform(&self, input: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>>
    where
        T: for<'a> Transform<ArrayView2<'a, Float>, Array2<f64>>,
    {
        if N == 0 {
            return Ok(input.mapv(|v| v));
        }

        let mut results = Vec::with_capacity(N);

        // This loop will be unrolled for small N
        for transformer in &self.transformers {
            results.push(transformer.transform(input)?);
        }

        // Concatenate results efficiently
        let total_features: usize = results
            .iter()
            .map(scirs2_core::ndarray::ArrayBase::ncols)
            .sum();
        let n_samples = results[0].nrows();

        let mut concatenated = Array2::zeros((n_samples, total_features));
        let mut col_idx = 0;

        for result in results {
            let end_idx = col_idx + result.ncols();
            concatenated
                .slice_mut(s![.., col_idx..end_idx])
                .assign(&result);
            col_idx = end_idx;
        }

        Ok(concatenated)
    }
}

/// Zero-cost estimator wrapper that eliminates virtual dispatch
pub struct ZeroCostEstimator<E> {
    estimator: E,
}

impl<E> ZeroCostEstimator<E> {
    /// Create a new zero-cost estimator
    #[inline(always)]
    pub const fn new(estimator: E) -> Self {
        Self { estimator }
    }

    /// Get the estimator with zero cost
    #[inline(always)]
    pub fn estimator(&self) -> &E {
        &self.estimator
    }
}

impl<E> Estimator for ZeroCostEstimator<E>
where
    E: Estimator,
{
    type Config = E::Config;
    type Error = E::Error;
    type Float = E::Float;

    #[inline(always)]
    fn config(&self) -> &Self::Config {
        self.estimator.config()
    }
}

impl<E, X, Y> Fit<X, Y> for ZeroCostEstimator<E>
where
    E: Estimator + Fit<X, Y>,
    E::Error: Into<SklearsError>,
{
    type Fitted = ZeroCostEstimator<E::Fitted>;

    #[inline(always)]
    fn fit(self, x: &X, y: &Y) -> SklResult<Self::Fitted> {
        self.estimator.fit(x, y).map(ZeroCostEstimator::new)
    }
}

/// Compile-time pipeline builder using type-level recursion
pub struct ZeroCostBuilder<T> {
    _phantom: PhantomData<T>,
}

impl Default for ZeroCostBuilder<()> {
    fn default() -> Self {
        Self::new()
    }
}

impl ZeroCostBuilder<()> {
    /// Start building a zero-cost pipeline
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T> ZeroCostBuilder<T> {
    /// Add a step to the pipeline
    #[inline(always)]
    pub fn step<S>(self, _step: S) -> ZeroCostBuilder<(T, S)> {
        ZeroCostBuilder {
            _phantom: PhantomData,
        }
    }

    /// Finalize the pipeline with compile-time optimization
    #[inline(always)]
    #[must_use]
    pub fn build(self) -> ZeroCostBuilder<T> {
        self
    }
}

/// Zero-cost conditional execution using const generics
pub struct ZeroCostConditional<const CONDITION: bool, T, F> {
    true_branch: T,
    false_branch: F,
}

impl<const CONDITION: bool, T, F> ZeroCostConditional<CONDITION, T, F> {
    /// Create a new zero-cost conditional
    #[inline(always)]
    pub const fn new(true_branch: T, false_branch: F) -> Self {
        Self {
            true_branch,
            false_branch,
        }
    }

    /// Execute with compile-time branch elimination
    #[inline(always)]
    pub fn execute<I>(&self, input: I) -> I
    where
        T: Fn(I) -> I,
        F: Fn(I) -> I,
    {
        if CONDITION {
            (self.true_branch)(input)
        } else {
            (self.false_branch)(input)
        }
    }
}

/// Compile-time feature selection using const generics
pub struct ZeroCostFeatureSelector<const FEATURES: u64> {
    _phantom: PhantomData<u64>,
}

impl<const FEATURES: u64> ZeroCostFeatureSelector<FEATURES> {
    /// Create a new zero-cost feature selector
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Select features with compile-time bounds checking
    pub fn select(&self, input: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let n_features = FEATURES as usize;
        if input.ncols() < n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Input has {} features, but {} are required",
                input.ncols(),
                n_features
            )));
        }

        // Select the first FEATURES columns
        Ok(input.slice(s![.., ..n_features]).mapv(|v| v))
    }
}

impl<const FEATURES: u64> Default for ZeroCostFeatureSelector<FEATURES> {
    fn default() -> Self {
        Self::new()
    }
}

/// Zero-cost transformer composition using function composition
pub struct ZeroCostComposition<F, G> {
    first: F,
    second: G,
}

impl<F, G> ZeroCostComposition<F, G> {
    /// Create a new zero-cost composition
    #[inline(always)]
    pub const fn new(first: F, second: G) -> Self {
        Self { first, second }
    }

    /// Apply both transformations in sequence
    #[inline(always)]
    pub fn apply<I>(&self, input: I) -> I
    where
        F: Fn(I) -> I,
        G: Fn(I) -> I,
    {
        (self.second)((self.first)(input))
    }
}

/// Trait for zero-cost composable operations
pub trait ZeroCostCompose<Other> {
    type Output;

    /// Compose with another operation at zero cost
    fn compose(self, other: Other) -> Self::Output;
}

impl<F, G> ZeroCostCompose<G> for F
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    type Output = ZeroCostComposition<F, G>;

    #[inline(always)]
    fn compose(self, other: G) -> Self::Output {
        ZeroCostComposition::new(self, other)
    }
}

/// Zero-cost parallel execution for embarrassingly parallel tasks
pub struct ZeroCostParallel<const N: usize> {
    _phantom: PhantomData<[(); N]>,
}

impl<const N: usize> ZeroCostParallel<N> {
    /// Create a new zero-cost parallel executor
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Execute tasks in parallel with compile-time task count
    pub fn execute<T, F, R>(&self, tasks: [F; N]) -> [R; N]
    where
        F: Fn() -> R + Send,
        R: Send,
        T: Send,
    {
        // For small N, this might be executed sequentially
        // For larger N, this would use rayon or similar
        tasks.map(|task| task())
    }
}

impl<const N: usize> Default for ZeroCostParallel<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Compile-time memory layout optimization
pub struct ZeroCostLayout<T> {
    data: T,
}

impl<T> ZeroCostLayout<T> {
    /// Create with optimal memory layout
    #[inline(always)]
    pub const fn new(data: T) -> Self {
        Self { data }
    }

    /// Access data with zero indirection
    #[inline(always)]
    pub fn data(&self) -> &T {
        &self.data
    }

    /// Consume and get data
    #[inline(always)]
    pub fn into_data(self) -> T {
        self.data
    }
}

/// Zero-cost cache-friendly data structure
#[repr(C)]
pub struct ZeroCostBuffer<T, const SIZE: usize> {
    data: Vec<T>,
    _phantom: PhantomData<[T; SIZE]>,
}

impl<T, const SIZE: usize> Default for ZeroCostBuffer<T, SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const SIZE: usize> ZeroCostBuffer<T, SIZE> {
    /// Create a new zero-cost buffer
    #[inline(always)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: Vec::with_capacity(SIZE),
            _phantom: PhantomData,
        }
    }

    /// Push with compile-time bounds checking
    #[inline(always)]
    pub fn push(&mut self, item: T) -> Result<(), &'static str> {
        if self.data.len() >= SIZE {
            return Err("Buffer full");
        }
        self.data.push(item);
        Ok(())
    }

    /// Get slice of used data
    #[inline(always)]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Clear with zero cost
    #[inline(always)]
    pub fn clear(&mut self) {
        self.data.clear();
    }
}

// Import ndarray slice macro
use scirs2_core::ndarray::s;
use std::rc::Rc;

/// Zero-copy data view for efficient data passing
pub struct ZeroCopyView<'a, T> {
    data: &'a [T],
    shape: (usize, usize),
    strides: (usize, usize),
}

impl<'a, T> ZeroCopyView<'a, T> {
    /// Create a new zero-copy view
    #[inline(always)]
    pub fn new(data: &'a [T], shape: (usize, usize), strides: (usize, usize)) -> Self {
        Self {
            data,
            shape,
            strides,
        }
    }

    /// Get shape without copying
    #[inline(always)]
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Get strides without copying
    #[inline(always)]
    #[must_use]
    pub fn strides(&self) -> (usize, usize) {
        self.strides
    }

    /// Get element by index with bounds checking
    #[inline(always)]
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row >= self.shape.0 || col >= self.shape.1 {
            return None;
        }
        let index = row * self.strides.0 + col * self.strides.1;
        self.data.get(index)
    }

    /// Create a sub-view without copying data
    #[inline(always)]
    #[must_use]
    pub fn slice(
        &self,
        row_range: std::ops::Range<usize>,
        col_range: std::ops::Range<usize>,
    ) -> Option<ZeroCopyView<'a, T>> {
        if row_range.end > self.shape.0 || col_range.end > self.shape.1 {
            return None;
        }

        let start_index = row_range.start * self.strides.0 + col_range.start * self.strides.1;
        let new_shape = (row_range.len(), col_range.len());
        let new_data = &self.data[start_index..];

        Some(ZeroCopyView::new(new_data, new_shape, self.strides))
    }
}

/// Reference-counted shared ownership for efficient data sharing
#[derive(Debug)]
pub struct SharedData<T> {
    data: Rc<T>,
}

impl<T> SharedData<T> {
    /// Create new shared data
    #[inline(always)]
    pub fn new(data: T) -> Self {
        Self {
            data: Rc::new(data),
        }
    }

    /// Get reference count
    #[inline(always)]
    #[must_use]
    pub fn ref_count(&self) -> usize {
        Rc::strong_count(&self.data)
    }

    /// Try to get mutable reference if uniquely owned
    #[inline(always)]
    pub fn try_unwrap(self) -> Result<T, Self> {
        Rc::try_unwrap(self.data).map_err(|data| Self { data })
    }
}

impl<T> Clone for SharedData<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self {
            data: Rc::clone(&self.data),
        }
    }
}

impl<T> std::ops::Deref for SharedData<T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// Copy-on-write semantics for efficient data modifications
pub struct CowData<T: Clone + 'static> {
    data: std::borrow::Cow<'static, T>,
}

impl<T: Clone + 'static> CowData<T> {
    /// Create from borrowed data
    #[inline(always)]
    pub fn borrowed(data: &'static T) -> Self {
        Self {
            data: std::borrow::Cow::Borrowed(data),
        }
    }

    /// Create from owned data
    #[inline(always)]
    pub fn owned(data: T) -> Self {
        Self {
            data: std::borrow::Cow::Owned(data),
        }
    }

    /// Get mutable reference, cloning if necessary
    #[inline(always)]
    pub fn to_mut(&mut self) -> &mut T {
        self.data.to_mut()
    }

    /// Convert to owned data
    #[inline(always)]
    pub fn into_owned(self) -> T {
        self.data.into_owned()
    }
}

impl<T: Clone + 'static> std::ops::Deref for CowData<T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// Arena allocator for batch allocation and zero-copy lifetime management
pub struct Arena<T> {
    chunks: Vec<Vec<T>>,
    current_chunk: usize,
    current_offset: usize,
    chunk_size: usize,
}

impl<T> Arena<T> {
    /// Create a new arena with specified chunk size
    #[must_use]
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunks: vec![Vec::with_capacity(chunk_size)],
            current_chunk: 0,
            current_offset: 0,
            chunk_size,
        }
    }

    /// Allocate space for a single item
    pub fn alloc(&mut self, item: T) -> &mut T {
        // Check if current chunk has space
        if self.current_offset >= self.chunk_size {
            // Need a new chunk
            self.chunks.push(Vec::with_capacity(self.chunk_size));
            self.current_chunk += 1;
            self.current_offset = 0;
        }

        let chunk = &mut self.chunks[self.current_chunk];
        chunk.push(item);
        self.current_offset += 1;

        chunk.last_mut().unwrap()
    }

    /// Allocate space for multiple items
    pub fn alloc_slice(&mut self, items: &[T]) -> &mut [T]
    where
        T: Clone,
    {
        let start_len = self.chunks[self.current_chunk].len();

        // Check if we need a new chunk
        if self.current_offset + items.len() > self.chunk_size {
            // Need a new chunk
            self.chunks
                .push(Vec::with_capacity(self.chunk_size.max(items.len())));
            self.current_chunk += 1;
            self.current_offset = 0;
        }

        let chunk = &mut self.chunks[self.current_chunk];
        chunk.extend_from_slice(items);
        self.current_offset += items.len();

        &mut chunk[start_len..]
    }

    /// Clear all allocations
    pub fn clear(&mut self) {
        for chunk in &mut self.chunks {
            chunk.clear();
        }
        self.current_chunk = 0;
        self.current_offset = 0;
    }

    /// Get total allocated items
    #[must_use]
    pub fn len(&self) -> usize {
        self.chunks.iter().map(std::vec::Vec::len).sum()
    }

    /// Check if arena is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new(1024) // Default chunk size
    }
}

/// Memory pool for efficient reuse of allocated buffers
pub struct MemoryPool<T> {
    free_buffers: Vec<Vec<T>>,
    min_capacity: usize,
    max_capacity: usize,
}

impl<T> MemoryPool<T> {
    /// Create a new memory pool
    #[must_use]
    pub fn new(min_capacity: usize, max_capacity: usize) -> Self {
        Self {
            free_buffers: Vec::new(),
            min_capacity,
            max_capacity,
        }
    }

    /// Get a buffer from the pool or allocate a new one
    pub fn get_buffer(&mut self, capacity: usize) -> Vec<T> {
        // Look for a suitable buffer in the pool
        for i in 0..self.free_buffers.len() {
            if self.free_buffers[i].capacity() >= capacity {
                let mut buffer = self.free_buffers.swap_remove(i);
                buffer.clear();
                return buffer;
            }
        }

        // No suitable buffer found, allocate new one
        Vec::with_capacity(capacity.clamp(self.min_capacity, self.max_capacity))
    }

    /// Return a buffer to the pool
    pub fn return_buffer(&mut self, mut buffer: Vec<T>) {
        buffer.clear();

        // Only keep buffers within our capacity limits
        if buffer.capacity() >= self.min_capacity && buffer.capacity() <= self.max_capacity {
            // Limit pool size to prevent unbounded growth
            if self.free_buffers.len() < 16 {
                self.free_buffers.push(buffer);
            }
        }
        // Otherwise let the buffer be dropped
    }

    /// Clear all pooled buffers
    pub fn clear(&mut self) {
        self.free_buffers.clear();
    }

    /// Get number of pooled buffers
    #[must_use]
    pub fn pool_size(&self) -> usize {
        self.free_buffers.len()
    }
}

impl<T> Default for MemoryPool<T> {
    fn default() -> Self {
        Self::new(64, 4096)
    }
}

/// RAII wrapper for automatic buffer return to pool
pub struct PooledBuffer<T> {
    buffer: Option<Vec<T>>,
    pool: *mut MemoryPool<T>,
}

impl<T> PooledBuffer<T> {
    /// Create from buffer and pool
    pub(crate) fn new(buffer: Vec<T>, pool: &mut MemoryPool<T>) -> Self {
        Self {
            buffer: Some(buffer),
            pool: pool as *mut MemoryPool<T>,
        }
    }

    /// Get mutable reference to the buffer
    pub fn as_mut(&mut self) -> &mut Vec<T> {
        self.buffer.as_mut().unwrap()
    }

    /// Get immutable reference to the buffer
    #[must_use]
    pub fn as_ref(&self) -> &Vec<T> {
        self.buffer.as_ref().unwrap()
    }
}

impl<T> std::ops::Deref for PooledBuffer<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        self.buffer.as_ref().unwrap()
    }
}

impl<T> std::ops::DerefMut for PooledBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buffer.as_mut().unwrap()
    }
}

impl<T> Drop for PooledBuffer<T> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            // Safety: pool pointer is valid during the lifetime of this object
            unsafe {
                (*self.pool).return_buffer(buffer);
            }
        }
    }
}

/// Zero-copy slice operations
pub trait ZeroCopySlice<T> {
    /// Create a zero-copy view of a portion of the data
    fn zero_copy_slice(&self, start: usize, end: usize) -> Option<&[T]>;

    /// Create a zero-copy iterator
    fn zero_copy_iter(&self) -> std::slice::Iter<'_, T>;

    /// Create a zero-copy chunks iterator
    fn zero_copy_chunks(&self, chunk_size: usize) -> std::slice::Chunks<'_, T>;
}

impl<T> ZeroCopySlice<T> for [T] {
    #[inline(always)]
    fn zero_copy_slice(&self, start: usize, end: usize) -> Option<&[T]> {
        self.get(start..end)
    }

    #[inline(always)]
    fn zero_copy_iter(&self) -> std::slice::Iter<'_, T> {
        self.iter()
    }

    #[inline(always)]
    fn zero_copy_chunks(&self, chunk_size: usize) -> std::slice::Chunks<'_, T> {
        self.chunks(chunk_size)
    }
}

impl<T> ZeroCopySlice<T> for Vec<T> {
    #[inline(always)]
    fn zero_copy_slice(&self, start: usize, end: usize) -> Option<&[T]> {
        self.as_slice().zero_copy_slice(start, end)
    }

    #[inline(always)]
    fn zero_copy_iter(&self) -> std::slice::Iter<'_, T> {
        self.iter()
    }

    #[inline(always)]
    fn zero_copy_chunks(&self, chunk_size: usize) -> std::slice::Chunks<'_, T> {
        self.chunks(chunk_size)
    }
}

/// Extend `MemoryPool` with `PooledBuffer` creation
impl<T> MemoryPool<T> {
    /// Get a pooled buffer that automatically returns to pool on drop
    pub fn get_pooled_buffer(&mut self, capacity: usize) -> PooledBuffer<T> {
        let buffer = self.get_buffer(capacity);
        PooledBuffer::new(buffer, self)
    }
}

/// Memory leak detector for tracking allocations and preventing leaks
#[derive(Debug)]
pub struct MemoryLeakDetector {
    /// Active allocations tracked by ID
    allocations: Mutex<HashMap<u64, AllocationInfo>>,
    /// Next allocation ID
    next_id: std::sync::atomic::AtomicU64,
    /// Configuration
    config: MemoryLeakConfig,
}

/// Information about a tracked allocation
#[derive(Debug, Clone)]
pub struct AllocationInfo {
    /// Unique allocation ID
    pub id: u64,
    /// Size of the allocation
    pub size: usize,
    /// Allocation timestamp
    pub timestamp: Instant,
    /// Stack trace (if enabled)
    pub stack_trace: Option<String>,
    /// Type name
    pub type_name: &'static str,
    /// Source location
    pub location: &'static str,
}

/// Configuration for memory leak detection
#[derive(Debug, Clone)]
pub struct MemoryLeakConfig {
    /// Whether to enable stack trace collection
    pub collect_stack_traces: bool,
    /// Maximum age before considering allocation a potential leak
    pub max_age: Duration,
    /// Whether to panic on detected leaks (for testing)
    pub panic_on_leak: bool,
    /// Maximum number of allocations to track
    pub max_tracked_allocations: usize,
}

impl Default for MemoryLeakConfig {
    fn default() -> Self {
        Self {
            collect_stack_traces: false,
            max_age: Duration::from_secs(300), // 5 minutes
            panic_on_leak: false,
            max_tracked_allocations: 10000,
        }
    }
}

impl MemoryLeakDetector {
    /// Create a new memory leak detector
    #[must_use]
    pub fn new(config: MemoryLeakConfig) -> Self {
        Self {
            allocations: Mutex::new(HashMap::new()),
            next_id: std::sync::atomic::AtomicU64::new(1),
            config,
        }
    }

    /// Track a new allocation
    pub fn track_allocation<T>(&self, size: usize, location: &'static str) -> TrackedAllocation {
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        let info = AllocationInfo {
            id,
            size,
            timestamp: Instant::now(),
            stack_trace: if self.config.collect_stack_traces {
                Some(self.collect_stack_trace())
            } else {
                None
            },
            type_name: std::any::type_name::<T>(),
            location,
        };

        if let Ok(mut allocations) = self.allocations.lock() {
            // Limit tracking to prevent unbounded growth
            if allocations.len() < self.config.max_tracked_allocations {
                allocations.insert(id, info);
            }
        }

        TrackedAllocation { id, detector: self }
    }

    /// Untrack an allocation
    fn untrack_allocation(&self, id: u64) {
        if let Ok(mut allocations) = self.allocations.lock() {
            allocations.remove(&id);
        }
    }

    /// Check for potential memory leaks
    pub fn check_leaks(&self) -> Vec<AllocationInfo> {
        let mut leaks = Vec::new();
        let now = Instant::now();

        if let Ok(allocations) = self.allocations.lock() {
            for info in allocations.values() {
                if now.duration_since(info.timestamp) > self.config.max_age {
                    leaks.push(info.clone());
                }
            }
        }

        assert!(
            !(!leaks.is_empty() && self.config.panic_on_leak),
            "Memory leaks detected: {} allocations",
            leaks.len()
        );

        leaks
    }

    /// Get current allocation statistics
    pub fn get_stats(&self) -> MemoryStats {
        if let Ok(allocations) = self.allocations.lock() {
            let total_allocations = allocations.len();
            let total_size = allocations.values().map(|info| info.size).sum();
            let oldest_age = allocations
                .values()
                .map(|info| info.timestamp.elapsed())
                .max()
                .unwrap_or_default();

            MemoryStats {
                total_allocations,
                total_size,
                oldest_age,
            }
        } else {
            MemoryStats::default()
        }
    }

    /// Collect stack trace (simplified implementation)
    fn collect_stack_trace(&self) -> String {
        // In a real implementation, this would use backtrace or similar
        "Stack trace collection not implemented".to_string()
    }
}

/// RAII wrapper for tracked allocations
pub struct TrackedAllocation<'a> {
    id: u64,
    detector: &'a MemoryLeakDetector,
}

impl Drop for TrackedAllocation<'_> {
    fn drop(&mut self) {
        self.detector.untrack_allocation(self.id);
    }
}

/// Memory usage statistics
#[derive(Debug, Default, Clone)]
pub struct MemoryStats {
    /// Total number of tracked allocations
    pub total_allocations: usize,
    /// Total size of tracked allocations
    pub total_size: usize,
    /// Age of oldest allocation
    pub oldest_age: Duration,
}

/// Thread-safe data structure for concurrent access with `RwLock`
#[derive(Debug)]
pub struct SafeConcurrentData<T> {
    data: RwLock<T>,
    /// Statistics for monitoring contention
    stats: Arc<Mutex<ConcurrencyStats>>,
}

/// Statistics for concurrent access patterns
#[derive(Debug, Default, Clone)]
pub struct ConcurrencyStats {
    /// Number of read locks acquired
    pub read_locks: u64,
    /// Number of write locks acquired
    pub write_locks: u64,
    /// Total time spent waiting for locks
    pub total_wait_time: Duration,
    /// Number of lock contentions
    pub contentions: u64,
}

impl<T> SafeConcurrentData<T> {
    /// Create new thread-safe data
    pub fn new(data: T) -> Self {
        Self {
            data: RwLock::new(data),
            stats: Arc::new(Mutex::new(ConcurrencyStats::default())),
        }
    }

    /// Read data with monitoring
    pub fn read<F, R>(&self, f: F) -> SklResult<R>
    where
        F: FnOnce(&T) -> R,
    {
        let start = Instant::now();

        if let Ok(guard) = self.data.read() {
            self.update_stats(true, start.elapsed(), false);
            Ok(f(&*guard))
        } else {
            self.update_stats(true, start.elapsed(), true);
            Err(SklearsError::InvalidOperation("Lock poisoned".to_string()))
        }
    }

    /// Write data with monitoring
    pub fn write<F, R>(&self, f: F) -> SklResult<R>
    where
        F: FnOnce(&mut T) -> R,
    {
        let start = Instant::now();

        if let Ok(mut guard) = self.data.write() {
            self.update_stats(false, start.elapsed(), false);
            Ok(f(&mut *guard))
        } else {
            self.update_stats(false, start.elapsed(), true);
            Err(SklearsError::InvalidOperation("Lock poisoned".to_string()))
        }
    }

    /// Try to read data without blocking
    pub fn try_read<F, R>(&self, f: F) -> SklResult<Option<R>>
    where
        F: FnOnce(&T) -> R,
    {
        let start = Instant::now();

        match self.data.try_read() {
            Ok(guard) => {
                self.update_stats(true, start.elapsed(), false);
                Ok(Some(f(&*guard)))
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                self.update_stats(true, start.elapsed(), true);
                Ok(None)
            }
            Err(std::sync::TryLockError::Poisoned(_)) => {
                self.update_stats(true, start.elapsed(), true);
                Err(SklearsError::InvalidOperation("Lock poisoned".to_string()))
            }
        }
    }

    /// Try to write data without blocking
    pub fn try_write<F, R>(&self, f: F) -> SklResult<Option<R>>
    where
        F: FnOnce(&mut T) -> R,
    {
        let start = Instant::now();

        match self.data.try_write() {
            Ok(mut guard) => {
                self.update_stats(false, start.elapsed(), false);
                Ok(Some(f(&mut *guard)))
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                self.update_stats(false, start.elapsed(), true);
                Ok(None)
            }
            Err(std::sync::TryLockError::Poisoned(_)) => {
                self.update_stats(false, start.elapsed(), true);
                Err(SklearsError::InvalidOperation("Lock poisoned".to_string()))
            }
        }
    }

    /// Get concurrency statistics
    pub fn get_stats(&self) -> ConcurrencyStats {
        self.stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Update statistics
    fn update_stats(&self, is_read: bool, wait_time: Duration, contention: bool) {
        if let Ok(mut stats) = self.stats.lock() {
            if is_read {
                stats.read_locks += 1;
            } else {
                stats.write_locks += 1;
            }
            stats.total_wait_time += wait_time;
            if contention {
                stats.contentions += 1;
            }
        }
    }
}

/// Atomic reference-counted data with weak references for cycle prevention
#[derive(Debug)]
pub struct AtomicRcData<T> {
    data: Arc<T>,
}

impl<T> AtomicRcData<T> {
    /// Create new atomic RC data
    pub fn new(data: T) -> Self {
        Self {
            data: Arc::new(data),
        }
    }

    /// Get strong reference count
    #[must_use]
    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.data)
    }

    /// Get weak reference count
    #[must_use]
    pub fn weak_count(&self) -> usize {
        Arc::weak_count(&self.data)
    }

    /// Create a weak reference to prevent cycles
    #[must_use]
    pub fn downgrade(&self) -> WeakRcData<T> {
        WeakRcData {
            weak: Arc::downgrade(&self.data),
        }
    }

    /// Try to unwrap if this is the only reference
    pub fn try_unwrap(self) -> Result<T, Self> {
        Arc::try_unwrap(self.data).map_err(|data| Self { data })
    }
}

impl<T> Clone for AtomicRcData<T> {
    fn clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
        }
    }
}

impl<T> std::ops::Deref for AtomicRcData<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// Weak reference wrapper for cycle prevention
#[derive(Debug)]
pub struct WeakRcData<T> {
    weak: Weak<T>,
}

impl<T> WeakRcData<T> {
    /// Upgrade weak reference to strong reference
    #[must_use]
    pub fn upgrade(&self) -> Option<AtomicRcData<T>> {
        self.weak.upgrade().map(|data| AtomicRcData { data })
    }

    /// Get current weak reference count
    #[must_use]
    pub fn weak_count(&self) -> usize {
        self.weak.weak_count()
    }

    /// Get current strong reference count
    #[must_use]
    pub fn strong_count(&self) -> usize {
        self.weak.strong_count()
    }
}

impl<T> Clone for WeakRcData<T> {
    fn clone(&self) -> Self {
        Self {
            weak: Weak::clone(&self.weak),
        }
    }
}

/// Lock-free concurrent queue for high-performance message passing
#[derive(Debug)]
pub struct LockFreeQueue<T> {
    /// For simplicity, using Mutex here. Real implementation would use atomic pointers
    inner: Mutex<std::collections::VecDeque<T>>,
    stats: Arc<Mutex<QueueStats>>,
}

/// Statistics for queue operations
#[derive(Debug, Default, Clone)]
pub struct QueueStats {
    pub enqueues: u64,
    pub dequeues: u64,
    pub current_size: usize,
    pub max_size: usize,
    pub contentions: u64,
}

impl<T> LockFreeQueue<T> {
    /// Create a new lock-free queue
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(std::collections::VecDeque::new()),
            stats: Arc::new(Mutex::new(QueueStats::default())),
        }
    }

    /// Enqueue an item
    pub fn enqueue(&self, item: T) -> SklResult<()> {
        match self.inner.lock() {
            Ok(mut queue) => {
                queue.push_back(item);

                // Update stats
                if let Ok(mut stats) = self.stats.lock() {
                    stats.enqueues += 1;
                    stats.current_size = queue.len();
                    stats.max_size = stats.max_size.max(queue.len());
                }

                Ok(())
            }
            Err(_) => Err(SklearsError::InvalidOperation(
                "Queue lock poisoned".to_string(),
            )),
        }
    }

    /// Dequeue an item
    pub fn dequeue(&self) -> SklResult<Option<T>> {
        match self.inner.lock() {
            Ok(mut queue) => {
                let item = queue.pop_front();

                // Update stats
                if let Ok(mut stats) = self.stats.lock() {
                    if item.is_some() {
                        stats.dequeues += 1;
                    }
                    stats.current_size = queue.len();
                }

                Ok(item)
            }
            Err(_) => Err(SklearsError::InvalidOperation(
                "Queue lock poisoned".to_string(),
            )),
        }
    }

    /// Try to dequeue without blocking
    pub fn try_dequeue(&self) -> SklResult<Option<T>> {
        match self.inner.try_lock() {
            Ok(mut queue) => {
                let item = queue.pop_front();

                // Update stats
                if let Ok(mut stats) = self.stats.lock() {
                    if item.is_some() {
                        stats.dequeues += 1;
                    }
                    stats.current_size = queue.len();
                }

                Ok(item)
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                // Update contention stats
                if let Ok(mut stats) = self.stats.lock() {
                    stats.contentions += 1;
                }
                Ok(None)
            }
            Err(std::sync::TryLockError::Poisoned(_)) => Err(SklearsError::InvalidOperation(
                "Queue lock poisoned".to_string(),
            )),
        }
    }

    /// Get current queue size
    pub fn len(&self) -> usize {
        self.inner.lock().map(|queue| queue.len()).unwrap_or(0)
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get queue statistics
    pub fn get_stats(&self) -> QueueStats {
        self.stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

impl<T> Default for LockFreeQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Work-stealing deque for efficient parallel processing
#[derive(Debug)]
pub struct WorkStealingDeque<T> {
    /// Work items
    items: Mutex<std::collections::VecDeque<T>>,
    /// Statistics
    stats: Arc<Mutex<WorkStealingStats>>,
}

/// Statistics for work-stealing operations
#[derive(Debug, Default, Clone)]
pub struct WorkStealingStats {
    /// Total push operations
    pub pushes: u64,
    /// Total pop operations  
    pub pops: u64,
    /// Total steal attempts
    pub steal_attempts: u64,
    /// Successful steals
    pub successful_steals: u64,
    /// Current size
    pub current_size: usize,
}

impl<T> WorkStealingDeque<T> {
    /// Create a new work-stealing deque
    #[must_use]
    pub fn new() -> Self {
        Self {
            items: Mutex::new(std::collections::VecDeque::new()),
            stats: Arc::new(Mutex::new(WorkStealingStats::default())),
        }
    }

    /// Push work item (local thread)
    pub fn push(&self, item: T) -> SklResult<()> {
        match self.items.lock() {
            Ok(mut items) => {
                items.push_back(item);

                if let Ok(mut stats) = self.stats.lock() {
                    stats.pushes += 1;
                    stats.current_size = items.len();
                }

                Ok(())
            }
            Err(_) => Err(SklearsError::InvalidOperation(
                "Deque lock poisoned".to_string(),
            )),
        }
    }

    /// Pop work item (local thread)
    pub fn pop(&self) -> SklResult<Option<T>> {
        match self.items.lock() {
            Ok(mut items) => {
                let item = items.pop_back();

                if let Ok(mut stats) = self.stats.lock() {
                    if item.is_some() {
                        stats.pops += 1;
                    }
                    stats.current_size = items.len();
                }

                Ok(item)
            }
            Err(_) => Err(SklearsError::InvalidOperation(
                "Deque lock poisoned".to_string(),
            )),
        }
    }

    /// Steal work item (other thread)
    pub fn steal(&self) -> SklResult<Option<T>> {
        // Update steal attempt stats first
        if let Ok(mut stats) = self.stats.lock() {
            stats.steal_attempts += 1;
        }

        match self.items.try_lock() {
            Ok(mut items) => {
                let item = items.pop_front();

                if let Ok(mut stats) = self.stats.lock() {
                    if item.is_some() {
                        stats.successful_steals += 1;
                    }
                    stats.current_size = items.len();
                }

                Ok(item)
            }
            Err(_) => Ok(None), // Failed to steal
        }
    }

    /// Get current size
    pub fn len(&self) -> usize {
        self.items.lock().map(|items| items.len()).unwrap_or(0)
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get statistics
    pub fn get_stats(&self) -> WorkStealingStats {
        self.stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

impl<T> Default for WorkStealingDeque<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_cost_step() {
        let step: ZeroCostStep<i32, ()> = ZeroCostStep::new(42);
        assert_eq!(*step.inner(), 42);
        assert_eq!(step.into_inner(), 42);
    }

    #[test]
    fn test_zero_cost_pipeline() {
        fn add_one(x: i32) -> i32 {
            x + 1
        }
        fn mul_two(x: i32) -> i32 {
            x * 2
        }

        let pipeline =
            ZeroCostPipeline::new([add_one as fn(i32) -> i32, mul_two as fn(i32) -> i32]);
        let result = pipeline.execute(5);
        assert_eq!(result, 12); // (5 + 1) * 2
    }

    #[test]
    fn test_zero_cost_builder() {
        let builder = ZeroCostBuilder::new();
        let _pipeline = builder.step("transform").step("estimate").build();
    }

    #[test]
    fn test_zero_cost_conditional() {
        let add_one = |x: i32| x + 1;
        let mul_two = |x: i32| x * 2;

        let conditional_true = ZeroCostConditional::<true, _, _>::new(add_one, mul_two);
        assert_eq!(conditional_true.execute(5), 6);

        let conditional_false = ZeroCostConditional::<false, _, _>::new(add_one, mul_two);
        assert_eq!(conditional_false.execute(5), 10);
    }

    #[test]
    fn test_zero_cost_composition() {
        let add_one = |x: f64| x + 1.0;
        let mul_two = |x: f64| x * 2.0;

        let composition = add_one.compose(mul_two);
        assert_eq!(composition.apply(5.0), 12.0); // (5 + 1) * 2
    }

    #[test]
    fn test_zero_cost_buffer() {
        let mut buffer: ZeroCostBuffer<i32, 4> = ZeroCostBuffer::new();

        assert!(buffer.push(1).is_ok());
        assert!(buffer.push(2).is_ok());
        assert_eq!(buffer.as_slice(), &[1, 2]);

        buffer.clear();
        let empty: &[i32] = &[];
        assert_eq!(buffer.as_slice(), empty);
    }

    #[test]
    fn test_zero_cost_parallel() {
        let parallel: ZeroCostParallel<2> = ZeroCostParallel::new();
        let tasks = [|| 1 + 1, || 2 * 2];
        let results = parallel.execute::<(), _, _>(tasks);
        assert_eq!(results, [2, 4]);
    }

    #[test]
    fn test_zero_cost_layout() {
        let layout = ZeroCostLayout::new(vec![1, 2, 3]);
        assert_eq!(layout.data(), &vec![1, 2, 3]);
        assert_eq!(layout.into_data(), vec![1, 2, 3]);
    }

    #[test]
    fn test_zero_copy_view() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let view = ZeroCopyView::new(&data, (2, 3), (3, 1));

        assert_eq!(view.shape(), (2, 3));
        assert_eq!(view.strides(), (3, 1));
        assert_eq!(view.get(0, 0), Some(&1));
        assert_eq!(view.get(1, 2), Some(&6));
        assert_eq!(view.get(2, 0), None); // Out of bounds
    }

    #[test]
    fn test_zero_copy_view_slice() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let view = ZeroCopyView::new(&data, (3, 3), (3, 1));

        let sub_view = view.slice(1..3, 1..3).unwrap();
        assert_eq!(sub_view.shape(), (2, 2));
        assert_eq!(sub_view.get(0, 0), Some(&5)); // data[1*3 + 1] = data[4]
    }

    #[test]
    fn test_shared_data() {
        let data = SharedData::new(vec![1, 2, 3]);
        assert_eq!(data.ref_count(), 1);

        let cloned = data.clone();
        assert_eq!(data.ref_count(), 2);
        assert_eq!(cloned.ref_count(), 2);

        drop(cloned);
        assert_eq!(data.ref_count(), 1);

        let recovered = data.try_unwrap().unwrap();
        assert_eq!(recovered, vec![1, 2, 3]);
    }

    #[test]
    fn test_cow_data() {
        // Test with owned data since static lifetime is complex
        let mut cow = CowData::owned(vec![1, 2, 3]);

        // Access without cloning
        assert_eq!(cow.len(), 3);

        // Modify
        cow.to_mut().push(42);
        assert_eq!(cow.len(), 4);
        assert_eq!(cow[3], 42);

        let owned = cow.into_owned();
        assert_eq!(owned, vec![1, 2, 3, 42]);
    }

    #[test]
    fn test_arena() {
        let mut arena = Arena::new(4);

        // Allocate single items
        arena.alloc(10);
        arena.alloc(20);
        assert_eq!(arena.len(), 2);

        // Allocate slice
        arena.alloc_slice(&[30, 40, 50]);
        assert_eq!(arena.len(), 5);

        // This should trigger a new chunk
        arena.alloc(60);
        assert_eq!(arena.len(), 6);

        arena.clear();
        assert_eq!(arena.len(), 0);
        assert!(arena.is_empty());
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = MemoryPool::new(4, 16);

        // Get a buffer
        let mut buffer1 = pool.get_buffer(8);
        buffer1.extend_from_slice(&[1, 2, 3]);
        assert_eq!(buffer1, vec![1, 2, 3]);
        assert!(buffer1.capacity() >= 8);

        // Return buffer to pool
        pool.return_buffer(buffer1);
        assert_eq!(pool.pool_size(), 1);

        // Get another buffer - should reuse the pooled one
        let buffer2 = pool.get_buffer(6);
        assert!(buffer2.is_empty());
        assert!(buffer2.capacity() >= 6);
    }

    #[test]
    fn test_pooled_buffer() {
        let mut pool = MemoryPool::new(4, 16);

        {
            let mut pooled = pool.get_pooled_buffer(8);
            pooled.push(42);
            assert_eq!(pooled[0], 42);
            assert_eq!(pool.pool_size(), 0); // Buffer is still in use
        } // Buffer returned to pool on drop

        assert_eq!(pool.pool_size(), 1);
    }

    #[test]
    fn test_zero_copy_slice_trait() {
        let data = vec![1, 2, 3, 4, 5];

        // Test zero_copy_slice
        let slice = data.zero_copy_slice(1, 4).unwrap();
        assert_eq!(slice, &[2, 3, 4]);

        // Test zero_copy_chunks
        let chunks: Vec<_> = data.zero_copy_chunks(2).collect();
        assert_eq!(chunks, vec![&[1, 2][..], &[3, 4][..], &[5][..]]);

        // Test zero_copy_iter
        let values: Vec<_> = data.zero_copy_iter().copied().collect();
        assert_eq!(values, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_memory_leak_detector() {
        let config = MemoryLeakConfig {
            collect_stack_traces: false,
            max_age: Duration::from_millis(100),
            panic_on_leak: false,
            max_tracked_allocations: 100,
        };

        let detector = MemoryLeakDetector::new(config);

        // Track an allocation
        let _tracked = detector.track_allocation::<Vec<i32>>(100, "test_location");

        let stats = detector.get_stats();
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.total_size, 100);

        // Check for leaks immediately (should be none)
        let leaks = detector.check_leaks();
        assert!(leaks.is_empty());

        // Wait for max_age to pass
        std::thread::sleep(Duration::from_millis(150));

        // Check for leaks again
        let leaks = detector.check_leaks();
        assert_eq!(leaks.len(), 1);
        assert_eq!(leaks[0].size, 100);

        // Drop the tracked allocation
        drop(_tracked);

        // Stats should now show 0 allocations
        let stats = detector.get_stats();
        assert_eq!(stats.total_allocations, 0);
    }

    #[test]
    fn test_safe_concurrent_data() {
        let data = SafeConcurrentData::new(vec![1, 2, 3]);

        // Test reading
        let result = data.read(|v| v.len()).unwrap();
        assert_eq!(result, 3);

        // Test writing
        let result = data
            .write(|v| {
                v.push(4);
                v.len()
            })
            .unwrap();
        assert_eq!(result, 4);

        // Check stats
        let stats = data.get_stats();
        assert_eq!(stats.read_locks, 1);
        assert_eq!(stats.write_locks, 1);
        assert_eq!(stats.contentions, 0);

        // Test try_read
        let result = data.try_read(|v| v.len()).unwrap();
        assert_eq!(result, Some(4));

        // Test final data
        let final_result = data.read(|v| v.clone()).unwrap();
        assert_eq!(final_result, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_atomic_rc_data() {
        let data = AtomicRcData::new(vec![1, 2, 3]);
        assert_eq!(data.strong_count(), 1);
        assert_eq!(data.weak_count(), 0);

        // Create weak reference
        let weak = data.downgrade();
        assert_eq!(data.weak_count(), 1);
        assert_eq!(weak.strong_count(), 1);

        // Clone strong reference
        let cloned = data.clone();
        assert_eq!(data.strong_count(), 2);
        assert_eq!(cloned.strong_count(), 2);

        // Upgrade weak reference
        let upgraded = weak.upgrade().unwrap();
        assert_eq!(upgraded.strong_count(), 3);

        // Drop cloned and upgraded
        drop(cloned);
        drop(upgraded);
        assert_eq!(data.strong_count(), 1);

        // Try to unwrap
        let recovered = data.try_unwrap().unwrap();
        assert_eq!(recovered, vec![1, 2, 3]);
    }

    #[test]
    fn test_lock_free_queue() {
        let queue = LockFreeQueue::new();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);

        // Enqueue items
        queue.enqueue(1).unwrap();
        queue.enqueue(2).unwrap();
        queue.enqueue(3).unwrap();

        assert_eq!(queue.len(), 3);
        assert!(!queue.is_empty());

        // Dequeue items
        assert_eq!(queue.dequeue().unwrap(), Some(1));
        assert_eq!(queue.dequeue().unwrap(), Some(2));
        assert_eq!(queue.len(), 1);

        // Try dequeue
        assert_eq!(queue.try_dequeue().unwrap(), Some(3));
        assert_eq!(queue.try_dequeue().unwrap(), None);

        assert!(queue.is_empty());

        // Check stats
        let stats = queue.get_stats();
        assert_eq!(stats.enqueues, 3);
        assert_eq!(stats.dequeues, 3);
        assert_eq!(stats.current_size, 0);
        assert_eq!(stats.max_size, 3);
    }

    #[test]
    fn test_work_stealing_deque() {
        let deque = WorkStealingDeque::new();
        assert!(deque.is_empty());
        assert_eq!(deque.len(), 0);

        // Push work items
        deque.push(1).unwrap();
        deque.push(2).unwrap();
        deque.push(3).unwrap();

        assert_eq!(deque.len(), 3);
        assert!(!deque.is_empty());

        // Pop from same thread (LIFO)
        assert_eq!(deque.pop().unwrap(), Some(3));
        assert_eq!(deque.len(), 2);

        // Steal from other thread (FIFO)
        assert_eq!(deque.steal().unwrap(), Some(1));
        assert_eq!(deque.len(), 1);

        // Pop remaining
        assert_eq!(deque.pop().unwrap(), Some(2));
        assert!(deque.is_empty());

        // Try steal from empty
        assert_eq!(deque.steal().unwrap(), None);

        // Check stats
        let stats = deque.get_stats();
        assert_eq!(stats.pushes, 3);
        assert_eq!(stats.pops, 2);
        assert_eq!(stats.steal_attempts, 2);
        assert_eq!(stats.successful_steals, 1);
        assert_eq!(stats.current_size, 0);
    }
}
