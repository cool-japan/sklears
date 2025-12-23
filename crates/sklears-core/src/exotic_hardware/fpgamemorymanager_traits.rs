//! Auto-generated trait implementations
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "async_support")]
use super::functions::*;
use super::types::*;
#[cfg(feature = "async_support")]
use crate::error::Result;
#[cfg(feature = "async_support")]
use async_trait::async_trait;
#[cfg(feature = "async_support")]
use std::time::Duration;

impl Default for FpgaMemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "async_support")]
#[async_trait]
impl HardwareMemoryManager for FpgaMemoryManager {
    async fn allocate(&self, size_bytes: u64, alignment: u32) -> Result<MemoryHandle> {
        let _total_memory =
            (self.block_ram_mb * 1024 * 1024) + (self.external_ram_gb * 1024 * 1024 * 1024);
        Ok(MemoryHandle {
            id: self.next_handle_id,
            size: size_bytes,
            alignment,
        })
    }
    async fn deallocate(&self, _handle: MemoryHandle) -> Result<()> {
        Ok(())
    }
    async fn copy_to_device(&self, handle: MemoryHandle, _data: &[u8]) -> Result<()> {
        let delay_us = if handle.size < 1024 * 1024 { 50 } else { 500 };
        tokio::time::sleep(Duration::from_micros(delay_us)).await;
        Ok(())
    }
    async fn copy_from_device(&self, handle: MemoryHandle, _data: &mut [u8]) -> Result<()> {
        let delay_us = if handle.size < 1024 * 1024 { 50 } else { 500 };
        tokio::time::sleep(Duration::from_micros(delay_us)).await;
        Ok(())
    }
    async fn memory_stats(&self) -> Result<MemoryStats> {
        let total_memory =
            (self.block_ram_mb * 1024 * 1024) + (self.external_ram_gb * 1024 * 1024 * 1024);
        let used_memory = self.allocated_memory.values().map(|h| h.size).sum::<u64>();
        Ok(MemoryStats {
            total_bytes: total_memory,
            used_bytes: used_memory,
            free_bytes: total_memory - used_memory,
            fragmentation_ratio: 0.05,
            allocation_count: self.allocated_memory.len() as u64,
            peak_usage_bytes: used_memory,
        })
    }
    async fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}
