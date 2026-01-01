/// Memory Safety Guarantees for sklears Machine Learning Library
///
/// This module documents and validates the memory safety guarantees provided by
/// the sklears library, leveraging Rust's ownership system and type safety to
/// eliminate entire classes of memory-related bugs common in machine learning codebases.
///
/// # Memory Safety Guarantees
///
/// ## 1. Memory Leak Prevention
///
/// Rust's ownership system ensures automatic memory management without garbage collection:
/// - All heap allocations are automatically freed when owners go out of scope
/// - RAII (Resource Acquisition Is Initialization) patterns prevent resource leaks
/// - No manual memory management required for safe operation
///
/// ## 2. Buffer Overflow Protection
///
/// Array and matrix operations are bounds-checked by default:
/// - Index operations panic on out-of-bounds access in debug builds
/// - Release builds may use unchecked access for performance (documented per function)
/// - ndarray provides comprehensive bounds checking for all operations
///
/// ## 3. Use-After-Free Elimination
///
/// The ownership system prevents accessing freed memory:
/// - Borrowed references ensure data outlives all uses
/// - Move semantics transfer ownership explicitly
/// - Lifetime parameters document and enforce temporal dependencies
///
/// ## 4. Data Race Prevention
///
/// Concurrent access is controlled by the type system:
/// - `Send` and `Sync` traits control thread safety
/// - Mutex and RwLock provide safe shared mutable access
/// - Atomic operations for lock-free data structures
///
/// ## 5. Null Pointer Dereference Prevention
///
/// Optional values are explicit and checked:
/// - `Option<T>` replaces null pointers
/// - Pattern matching enforces null checks
/// - Safe references that cannot be null by construction
///
/// # Implementation Details
///
/// ## Safe Array Operations
///
/// ```rust
/// use scirs2_core::ndarray::Array2;
/// use sklears_core::memory_safety::SafeArrayOps;
///
/// fn safe_matrix_access() -> Result<f64, &'static str> {
///     let matrix = Array2::zeros((1000, 1000));
///     
///     // Bounds-checked access - will return error for out-of-bounds
///     matrix.get((999, 999))
///         .copied()
///         .ok_or("Index out of bounds")
/// }
/// ```
///
/// ## Memory Pool Safety
///
/// ```rust
/// use sklears_core::memory_safety::SafeMemoryPool;
///
/// fn pooled_allocation_example() {
///     let mut pool = SafeMemoryPool::`<f64>`::new();
///     
///     // Safe allocation with automatic cleanup
///     let buffer = pool.allocate(1000);
///     // Buffer is automatically returned to pool when dropped
/// }
/// ```
// SciRS2 Policy: Using scirs2_core::ndarray for unified access (COMPLIANT)
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, RwLock};

/// Memory safety documentation and validation utilities
pub struct MemorySafety;

impl MemorySafety {
    /// Document memory safety guarantees for a given operation
    pub fn document_safety(operation: &str) -> MemorySafetyGuarantee {
        match operation {
            "array_indexing" => MemorySafetyGuarantee {
                operation: operation.to_string(),
                guarantees: vec![
                    "Bounds checking prevents buffer overflows".to_string(),
                    "Panic on out-of-bounds access in debug mode".to_string(),
                    "Optional bounds checking in release mode for performance".to_string(),
                ],
                unsafe_blocks: vec![],
                mitigation_strategies: vec![
                    "Use checked indexing methods when bounds are uncertain".to_string(),
                    "Validate input dimensions before processing".to_string(),
                ],
            },
            "parallel_processing" => MemorySafetyGuarantee {
                operation: operation.to_string(),
                guarantees: vec![
                    "Send and Sync traits prevent data races".to_string(),
                    "Rayon provides work-stealing without data races".to_string(),
                    "Immutable borrows allow safe parallel reading".to_string(),
                ],
                unsafe_blocks: vec![],
                mitigation_strategies: vec![
                    "Use Arc<T> for shared ownership across threads".to_string(),
                    "Use Mutex<T> or RwLock<T> for shared mutable access".to_string(),
                ],
            },
            "gpu_operations" => MemorySafetyGuarantee {
                operation: operation.to_string(),
                guarantees: vec![
                    "CUDA memory is managed through RAII wrappers".to_string(),
                    "GPU pointers are opaque and cannot be dereferenced on CPU".to_string(),
                    "Automatic cleanup of GPU resources on drop".to_string(),
                ],
                unsafe_blocks: vec![
                    "CUDA FFI calls require unsafe blocks".to_string(),
                    "Memory transfers between CPU and GPU use unsafe operations".to_string(),
                ],
                mitigation_strategies: vec![
                    "Wrap all CUDA operations in safe abstractions".to_string(),
                    "Validate GPU memory allocation success".to_string(),
                    "Use typed GPU pointers to prevent type confusion".to_string(),
                ],
            },
            _ => MemorySafetyGuarantee {
                operation: operation.to_string(),
                guarantees: vec!["General Rust memory safety guarantees apply".to_string()],
                unsafe_blocks: vec![],
                mitigation_strategies: vec![],
            },
        }
    }

    /// Validate that unsafe code follows safety guidelines
    pub fn validate_unsafe_usage(code_block: &str) -> UnsafeValidationResult {
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        // Check for common unsafe patterns
        if code_block.contains("transmute") {
            issues.push("transmute operations can break type safety".to_string());
            recommendations.push("Consider using safe casting alternatives".to_string());
        }

        if code_block.contains("from_raw_parts") {
            issues.push("Raw pointer operations require careful validation".to_string());
            recommendations.push("Ensure pointer validity and proper alignment".to_string());
        }

        if code_block.contains("assume_init") {
            issues.push("Uninitialized memory access detected".to_string());
            recommendations
                .push("Use MaybeUninit for safer uninitialized memory handling".to_string());
        }

        let safety_score = if issues.is_empty() {
            100
        } else {
            std::cmp::max(0, 100 - (issues.len() * 20)) as u8
        };

        UnsafeValidationResult {
            safety_score,
            issues,
            recommendations,
            requires_review: safety_score < 80,
        }
    }
}

/// Memory safety guarantee documentation
#[derive(Debug, Clone)]
pub struct MemorySafetyGuarantee {
    pub operation: String,
    pub guarantees: Vec<String>,
    pub unsafe_blocks: Vec<String>,
    pub mitigation_strategies: Vec<String>,
}

/// Result of unsafe code validation
#[derive(Debug, Clone)]
pub struct UnsafeValidationResult {
    pub safety_score: u8, // 0-100 safety score
    pub issues: Vec<String>,
    pub recommendations: Vec<String>,
    pub requires_review: bool,
}

/// Safe array operations trait
pub trait SafeArrayOps<T> {
    /// Safe element access with bounds checking
    fn safe_get(&self, index: &[usize]) -> Option<&T>;

    /// Safe mutable element access with bounds checking
    fn safe_get_mut(&mut self, index: &[usize]) -> Option<&mut T>;

    /// Validate array dimensions and return error if invalid
    fn validate_dimensions(&self) -> Result<(), String>;

    /// Check if index is within bounds
    fn is_valid_index(&self, index: &[usize]) -> bool;
}

impl<T> SafeArrayOps<T> for Array2<T> {
    fn safe_get(&self, index: &[usize]) -> Option<&T> {
        if index.len() != 2 {
            return None;
        }
        self.get((index[0], index[1]))
    }

    fn safe_get_mut(&mut self, index: &[usize]) -> Option<&mut T> {
        if index.len() != 2 {
            return None;
        }
        self.get_mut((index[0], index[1]))
    }

    fn validate_dimensions(&self) -> Result<(), String> {
        if self.nrows() == 0 || self.ncols() == 0 {
            Err("Array has zero-sized dimension".to_string())
        } else if self.nrows() > isize::MAX as usize || self.ncols() > isize::MAX as usize {
            Err("Array dimension exceeds maximum safe size".to_string())
        } else {
            Ok(())
        }
    }

    fn is_valid_index(&self, index: &[usize]) -> bool {
        index.len() == 2 && index[0] < self.nrows() && index[1] < self.ncols()
    }
}

impl<T> SafeArrayOps<T> for Array1<T> {
    fn safe_get(&self, index: &[usize]) -> Option<&T> {
        if index.len() != 1 {
            return None;
        }
        self.get(index[0])
    }

    fn safe_get_mut(&mut self, index: &[usize]) -> Option<&mut T> {
        if index.len() != 1 {
            return None;
        }
        self.get_mut(index[0])
    }

    fn validate_dimensions(&self) -> Result<(), String> {
        if self.is_empty() {
            Err("Array is empty".to_string())
        } else if self.len() > isize::MAX as usize {
            Err("Array length exceeds maximum safe size".to_string())
        } else {
            Ok(())
        }
    }

    fn is_valid_index(&self, index: &[usize]) -> bool {
        index.len() == 1 && index[0] < self.len()
    }
}

/// Safe memory pool for efficient allocation with automatic cleanup
pub struct SafeMemoryPool<T> {
    pools: Arc<Mutex<HashMap<usize, Vec<Vec<T>>>>>,
    allocated_count: Arc<Mutex<usize>>,
    max_pool_size: usize,
}

impl<T> SafeMemoryPool<T> {
    /// Create a new safe memory pool
    pub fn new() -> Self {
        Self {
            pools: Arc::new(Mutex::new(HashMap::new())),
            allocated_count: Arc::new(Mutex::new(0)),
            max_pool_size: 1000, // Maximum number of pooled allocations
        }
    }

    /// Create a new safe memory pool with custom limits
    pub fn with_limits(max_pool_size: usize) -> Self {
        Self {
            pools: Arc::new(Mutex::new(HashMap::new())),
            allocated_count: Arc::new(Mutex::new(0)),
            max_pool_size,
        }
    }

    /// Allocate a vector with the specified capacity
    pub fn allocate(&self, capacity: usize) -> SafePooledBuffer<T> {
        let buffer = {
            let mut pools = self.pools.lock().unwrap();
            if let Some(pool) = pools.get_mut(&capacity) {
                if let Some(mut buffer) = pool.pop() {
                    buffer.clear();
                    buffer
                } else {
                    Vec::with_capacity(capacity)
                }
            } else {
                Vec::with_capacity(capacity)
            }
        };

        {
            let mut count = self.allocated_count.lock().unwrap();
            *count += 1;
        }

        SafePooledBuffer {
            buffer: Some(buffer),
            capacity,
            pool: self.pools.clone(),
            allocated_count: self.allocated_count.clone(),
            max_pool_size: self.max_pool_size,
        }
    }

    /// Get current allocation statistics
    pub fn stats(&self) -> MemoryPoolStats {
        let allocated_count = *self.allocated_count.lock().unwrap();
        let pools = self.pools.lock().unwrap();
        let pooled_count: usize = pools.values().map(|v| v.len()).sum();

        MemoryPoolStats {
            allocated_count,
            pooled_count,
            pool_sizes: pools.iter().map(|(&k, v)| (k, v.len())).collect(),
        }
    }
}

impl<T> Default for SafeMemoryPool<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for memory pool usage
#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    pub allocated_count: usize,
    pub pooled_count: usize,
    pub pool_sizes: Vec<(usize, usize)>, // (capacity, count) pairs
}

/// Safe pooled buffer with automatic return to pool on drop
pub struct SafePooledBuffer<T> {
    buffer: Option<Vec<T>>,
    capacity: usize,
    pool: Arc<Mutex<HashMap<usize, Vec<Vec<T>>>>>,
    allocated_count: Arc<Mutex<usize>>,
    max_pool_size: usize,
}

impl<T> SafePooledBuffer<T> {
    /// Get a mutable reference to the underlying buffer
    pub fn as_mut_vec(&mut self) -> &mut Vec<T> {
        self.buffer.as_mut().expect("Buffer has been consumed")
    }

    /// Get an immutable reference to the underlying buffer
    pub fn as_ref_vec(&self) -> &Vec<T> {
        self.buffer.as_ref().expect("Buffer has been consumed")
    }

    /// Consume the buffer and return the inner Vec
    pub fn into_inner(mut self) -> Vec<T> {
        self.buffer.take().expect("Buffer has been consumed")
    }
}

impl<T> Drop for SafePooledBuffer<T> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            // Only return to pool if we haven't exceeded the limit
            let mut pools = self.pool.lock().unwrap();
            let pool = pools.entry(self.capacity).or_default();

            if pool.len() < self.max_pool_size {
                pool.push(buffer);
            }
            // Otherwise, let the buffer be freed normally

            // Decrement allocation count
            let mut count = self.allocated_count.lock().unwrap();
            *count = count.saturating_sub(1);
        }
    }
}

impl<T> std::ops::Deref for SafePooledBuffer<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        self.as_ref_vec()
    }
}

impl<T> std::ops::DerefMut for SafePooledBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_vec()
    }
}

/// Safe pointer wrapper that prevents raw pointer dereference
#[derive(Debug)]
pub struct SafePtr<T> {
    ptr: NonNull<T>,
    _marker: PhantomData<T>,
}

impl<T> SafePtr<T> {
    /// Create a new safe pointer from a raw pointer
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - The pointer is valid and properly aligned
    /// - The memory is initialized for the lifetime of this pointer
    /// - No other mutable references exist to this memory
    pub unsafe fn new(ptr: NonNull<T>) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// Get the raw pointer value for FFI operations
    ///
    /// # Safety
    ///
    /// The returned pointer should only be used with appropriate safety checks
    pub unsafe fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Get a mutable raw pointer for FFI operations
    ///
    /// # Safety
    ///
    /// The returned pointer should only be used with appropriate safety checks
    pub unsafe fn as_mut_ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }
}

// SafePtr cannot be Send or Sync without additional guarantees
unsafe impl<T: Send> Send for SafePtr<T> {}
unsafe impl<T: Sync> Sync for SafePtr<T> {}

/// Thread-safe reference counting for shared machine learning models
pub struct SafeSharedModel<T> {
    inner: Arc<RwLock<T>>,
    id: String,
}

impl<T> SafeSharedModel<T> {
    /// Create a new shared model
    pub fn new(model: T, id: String) -> Self {
        Self {
            inner: Arc::new(RwLock::new(model)),
            id,
        }
    }

    /// Get a read lock on the model
    pub fn read(&self) -> std::sync::RwLockReadGuard<'_, T> {
        self.inner
            .read()
            .unwrap_or_else(|e| panic!("RwLock poisoned for model {}: {}", self.id, e))
    }

    /// Get a write lock on the model
    pub fn write(&self) -> std::sync::RwLockWriteGuard<'_, T> {
        self.inner
            .write()
            .unwrap_or_else(|e| panic!("RwLock poisoned for model {}: {}", self.id, e))
    }

    /// Try to get a read lock without blocking
    pub fn try_read(&self) -> Option<std::sync::RwLockReadGuard<'_, T>> {
        self.inner.try_read().ok()
    }

    /// Try to get a write lock without blocking
    pub fn try_write(&self) -> Option<std::sync::RwLockWriteGuard<'_, T>> {
        self.inner.try_write().ok()
    }

    /// Clone the shared model reference
    pub fn clone_ref(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            id: self.id.clone(),
        }
    }
}

impl<T: Clone> SafeSharedModel<T> {
    /// Create a deep copy of the model
    pub fn clone_model(&self) -> T {
        self.read().clone()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_memory_safety_documentation() {
        let guarantee = MemorySafety::document_safety("array_indexing");
        assert_eq!(guarantee.operation, "array_indexing");
        assert!(!guarantee.guarantees.is_empty());
    }

    #[test]
    fn test_unsafe_validation() {
        let safe_code = "let x = vec![1, 2, 3]; let y = &x[0];";
        let result = MemorySafety::validate_unsafe_usage(safe_code);
        assert_eq!(result.safety_score, 100);
        assert!(result.issues.is_empty());

        let unsafe_code = "let x = transmute::<i32, f32>(42);";
        let result = MemorySafety::validate_unsafe_usage(unsafe_code);
        assert!(result.safety_score < 100);
        assert!(!result.issues.is_empty());
    }

    #[test]
    fn test_safe_array_operations() {
        let array = Array2::<f64>::zeros((10, 10));

        // Test safe access
        assert!(array.safe_get(&[0, 0]).is_some());
        assert!(array.safe_get(&[10, 10]).is_none());
        assert!(array.safe_get(&[5]).is_none()); // Wrong number of indices

        // Test dimension validation
        assert!(array.validate_dimensions().is_ok());

        // Test index validation
        assert!(array.is_valid_index(&[5, 5]));
        assert!(!array.is_valid_index(&[10, 5]));
    }

    #[test]
    fn test_memory_pool() {
        let pool = SafeMemoryPool::<i32>::new();

        // Allocate buffer
        let buffer = pool.allocate(100);
        assert_eq!(buffer.capacity(), 100);

        let stats = pool.stats();
        assert_eq!(stats.allocated_count, 1);

        // Buffer should be returned to pool on drop
        drop(buffer);

        let stats = pool.stats();
        assert_eq!(stats.allocated_count, 0);
        assert_eq!(stats.pooled_count, 1);
    }

    #[test]
    fn test_shared_model() {
        let model = vec![1, 2, 3, 4, 5];
        let shared = SafeSharedModel::new(model, "test_model".to_string());

        // Test read access
        {
            let reader = shared.read();
            assert_eq!(reader.len(), 5);
        }

        // Test write access
        {
            let mut writer = shared.write();
            writer.push(6);
            assert_eq!(writer.len(), 6);
        }

        // Test cloning reference
        let shared2 = shared.clone_ref();
        let reader = shared2.read();
        assert_eq!(reader.len(), 6);
    }

    #[test]
    fn test_pooled_buffer_deref() {
        let pool = SafeMemoryPool::<i32>::new();
        let mut buffer = pool.allocate(10);

        // Test deref operations
        buffer.push(42);
        assert_eq!(buffer.len(), 1);
        assert_eq!(buffer[0], 42);

        // Test into_inner
        let inner = buffer.into_inner();
        assert_eq!(inner, vec![42]);
    }
}
