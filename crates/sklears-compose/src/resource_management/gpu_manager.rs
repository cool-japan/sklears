//! GPU resource management
//!
//! Device discovery/counting here is wired to `sklears_core::gpu`, which is
//! itself backed by `oxicuda-driver`. Without the `gpu` feature (which
//! forwards to `sklears-core/gpu_support`), every query below honestly
//! reports zero devices rather than fabricating hardware -- see
//! `GpuUtils::device_count` in `sklears-core`.

use super::resource_types::GpuAllocation;
#[cfg(feature = "gpu")]
use super::resource_types::GpuDeviceAllocation;
use sklears_core::error::Result as SklResult;
#[cfg(feature = "gpu")]
use sklears_core::gpu::GpuUtils;
use std::collections::HashMap;

/// GPU resource manager
#[derive(Debug)]
#[allow(dead_code)]
pub struct GpuResourceManager {
    /// GPU devices
    devices: Vec<GpuDevice>,
    /// Active allocations
    allocations: HashMap<String, GpuAllocation>,
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDevice {
    /// Device ID
    pub device_id: String,
    /// Device name
    pub name: String,
    /// Memory size
    pub memory_size: u64,
    /// Available memory
    pub available_memory: u64,
}

impl Default for GpuResourceManager {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuResourceManager {
    /// Create a new GPU resource manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            devices: Vec::new(),
            allocations: HashMap::new(),
        }
    }

    /// Allocate GPU resources.
    ///
    /// With the `gpu` feature enabled, this enumerates real devices through
    /// `sklears_core::gpu::GpuUtils`, which resolves to actual
    /// `oxicuda-driver` device queries (and to a zero-device count on a
    /// host without a working CUDA driver, even with the feature on).
    /// Without the `gpu` feature, this honestly returns an empty
    /// allocation rather than fabricating devices.
    #[cfg(feature = "gpu")]
    pub fn allocate_gpu(&mut self, device_count: usize) -> SklResult<GpuAllocation> {
        let available = GpuUtils::device_count();
        let requested = device_count.min(available);

        let mut devices = Vec::with_capacity(requested);
        for device_index in 0..requested {
            let props = GpuUtils::device_properties(device_index)?;
            devices.push(GpuDeviceAllocation {
                device_id: format!("gpu-{device_index}"),
                device_index,
                allocated_memory: props.free_memory as u64,
                compute_capability: format!(
                    "{}.{}",
                    props.compute_capability.0, props.compute_capability.1
                ),
                exclusive: false,
                p2p_enabled: false,
            });
        }

        Ok(GpuAllocation {
            devices,
            memory_pools: Vec::new(),
            compute_streams: Vec::new(),
            context: None,
        })
    }

    /// Allocate GPU resources (default build, no `gpu` feature).
    ///
    /// Always returns an empty allocation: without the `gpu` feature this
    /// crate has no way to query real devices, so it honestly reports none
    /// rather than fabricating hardware.
    #[cfg(not(feature = "gpu"))]
    pub fn allocate_gpu(&mut self, _device_count: usize) -> SklResult<GpuAllocation> {
        Ok(GpuAllocation {
            devices: Vec::new(),
            memory_pools: Vec::new(),
            compute_streams: Vec::new(),
            context: None,
        })
    }

    /// Release GPU allocation
    pub fn release_gpu(&mut self, _allocation: &GpuAllocation) -> SklResult<()> {
        Ok(())
    }
}
