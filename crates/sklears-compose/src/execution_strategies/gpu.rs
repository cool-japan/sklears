//! GPU-accelerated execution strategy scheduling descriptors.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use super::core::{StrategyConfig, StrategyMetrics, StrategyState};


/// Memory allocation strategies
#[derive(Debug, Clone)]
pub enum AllocationStrategy {
    /// FirstFit
    FirstFit,
    /// BestFit
    BestFit,
    /// WorstFit
    WorstFit,
    /// Buddy
    Buddy,
    /// Slab
    Slab,
}
/// GPU execution context: a scheduling descriptor, not a live device
/// handle. See the [`GpuExecutionStrategy`] doc comment above for what
/// that means in practice.
#[derive(Debug)]
pub struct GpuContext {
    /// The devices.
    pub devices: HashMap<String, GpuDevice>,
    /// The memory pools.
    pub memory_pools: HashMap<String, MemoryPool>,
    /// The active kernels.
    pub active_kernels: HashMap<String, GpuKernel>,
}
/// GPU device information: a scheduling descriptor populated by the
/// caller, not a live-sampled hardware reading -- `utilization` and
/// `temperature` are inputs to this crate's scheduler, not measurements
/// this type takes itself. See the [`GpuExecutionStrategy`] doc comment
/// above.
#[derive(Debug, Clone)]
pub struct GpuDevice {
    /// Device ID
    pub id: String,
    /// Device name
    pub name: String,
    /// Compute capability
    pub compute_capability: String,
    /// Total memory
    pub total_memory: u64,
    /// Available memory
    pub available_memory: u64,
    /// Utilization percentage (scheduling metadata, not a live reading)
    pub utilization: f64,
    /// Temperature (scheduling metadata, not a live reading)
    pub temperature: f64,
}
/// GPU execution strategy for hardware-accelerated computation.
///
/// # Scheduling-metadata status
///
/// `GpuExecutionStrategy` and the `GpuContext` / `GpuDevice` / `GpuKernel`
/// types it composes (below) are **descriptors used by this crate's task
/// scheduler**, not a GPU compute backend: nothing in this file launches a
/// kernel or touches a real device. `devices`, `utilization`,
/// `temperature`, and friends are scheduling-input fields a caller (or a
/// future integration) can populate, not live hardware measurements this
/// strategy samples itself. Real device discovery/counting for this crate
/// lives in `resource_management::gpu_manager` and
/// `comprehensive_benchmarking::execution_engine`, both backed by
/// `sklears_core::gpu` (oxicuda-driver) behind the `gpu` feature. Backing
/// `GpuExecutionStrategy` itself with oxicuda-launch kernels is tracked as a
/// separate, larger future item.
#[derive(Debug)]
#[allow(dead_code)]
pub struct GpuExecutionStrategy {
    /// Strategy configuration
    pub(super) config: StrategyConfig,
    /// GPU devices to use
    pub(super) devices: Vec<String>,
    /// GPU memory pool size
    pub(super) memory_pool_size: u64,
    /// Memory optimization enabled
    pub(super) memory_optimization: bool,
    /// Mixed precision enabled
    pub(super) mixed_precision: bool,
    /// Number of compute streams per device
    pub(super) compute_streams: usize,
    /// GPU context manager
    pub(super) gpu_context: Arc<Mutex<GpuContext>>,
    /// Execution metrics
    pub(super) metrics: Arc<Mutex<StrategyMetrics>>,
    /// Strategy state
    pub(super) state: Arc<RwLock<StrategyState>>,
}
/// GPU kernel execution context: a scheduling descriptor recording the
/// launch parameters this crate's scheduler *would* use, not a handle to a
/// kernel actually launched on a device. See the [`GpuExecutionStrategy`]
/// doc comment above.
#[derive(Debug)]
pub struct GpuKernel {
    /// Kernel name
    pub name: String,
    /// Device ID
    pub device_id: String,
    /// Stream ID
    pub stream_id: String,
    /// Grid dimensions
    pub grid_dims: (u32, u32, u32),
    /// Block dimensions
    pub block_dims: (u32, u32, u32),
    /// Shared memory size
    pub shared_memory: u32,
}
/// GPU memory pool
#[derive(Debug)]
pub struct MemoryPool {
    /// Pool size
    pub size: u64,
    /// Used memory
    pub used: u64,
    /// Free memory
    pub free: u64,
    /// Allocation strategy
    pub allocation_strategy: AllocationStrategy,
}
