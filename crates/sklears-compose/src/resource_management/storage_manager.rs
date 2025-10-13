//! Storage resource management

use super::resource_types::{IOPriority, StorageAllocation, StoragePermissions, StorageType};
use sklears_core::error::Result as SklResult;

/// Storage resource manager
#[derive(Debug)]
pub struct StorageResourceManager {
    /// Storage devices
    devices: Vec<StorageDevice>,
}

/// Storage device information
#[derive(Debug, Clone)]
pub struct StorageDevice {
    /// Device path
    pub path: String,
    /// Total size
    pub total_size: u64,
    /// Available size
    pub available_size: u64,
    /// Storage type
    pub storage_type: StorageType,
}

impl Default for StorageResourceManager {
    fn default() -> Self {
        Self::new()
    }
}

impl StorageResourceManager {
    /// Create a new storage resource manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            devices: Vec::new(),
        }
    }

    /// Allocate storage resources
    pub fn allocate_storage(
        &mut self,
        size: u64,
        storage_type: StorageType,
    ) -> SklResult<StorageAllocation> {
        Ok(StorageAllocation {
            size,
            storage_type,
            io_priority: IOPriority::Normal,
            mount_point: None,
            permissions: StoragePermissions {
                read: true,
                write: true,
                execute: false,
            },
        })
    }

    /// Release storage allocation
    pub fn release_storage(&mut self, allocation: &StorageAllocation) -> SklResult<()> {
        Ok(())
    }
}
