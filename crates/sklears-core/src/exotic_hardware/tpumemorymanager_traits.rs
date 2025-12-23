//! Auto-generated trait implementations
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "async_support")]
use super::functions::*;
use super::types::*;
#[cfg(feature = "async_support")]
use crate::error::{Result, SklearsError};
#[cfg(feature = "async_support")]
use async_trait::async_trait;
#[cfg(feature = "async_support")]
use std::time::Duration;

impl Default for TpuMemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "async_support")]
#[async_trait]
impl HardwareMemoryManager for TpuMemoryManager {
    async fn allocate(&self, size_bytes: u64, alignment: u32) -> Result<MemoryHandle> {
        if self.used_memory + size_bytes > self.total_memory {
            return Err(SklearsError::HardwareError(
                "Insufficient TPU memory".to_string(),
            ));
        }
        Ok(MemoryHandle {
            id: self.next_handle_id,
            size: size_bytes,
            alignment,
        })
    }
    async fn deallocate(&self, _handle: MemoryHandle) -> Result<()> {
        Ok(())
    }
    async fn copy_to_device(&self, _handle: MemoryHandle, _data: &[u8]) -> Result<()> {
        tokio::time::sleep(Duration::from_micros(100)).await;
        Ok(())
    }
    async fn copy_from_device(&self, _handle: MemoryHandle, _data: &mut [u8]) -> Result<()> {
        tokio::time::sleep(Duration::from_micros(100)).await;
        Ok(())
    }
    async fn memory_stats(&self) -> Result<MemoryStats> {
        Ok(MemoryStats {
            total_bytes: self.total_memory,
            used_bytes: self.used_memory,
            free_bytes: self.total_memory - self.used_memory,
            fragmentation_ratio: 0.1,
            allocation_count: self.allocated_memory.len() as u64,
            peak_usage_bytes: self.used_memory,
        })
    }
    async fn synchronize(&self) -> Result<()> {
        tokio::time::sleep(Duration::from_micros(10)).await;
        Ok(())
    }
}
