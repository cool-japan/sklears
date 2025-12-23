//! # OutputManager - Trait Implementations
//!
//! This module contains trait implementations for `OutputManager`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for OutputManager {
    fn default() -> Self {
        Self {
            output_destinations: vec![
                OutputDestination::LocalFile(LocalFileDestination { base_path :
                "/tmp/sklears_exports".to_string(), create_directories : true,
                permissions : FilePermissions::default(), }),
            ],
            file_naming: FileNamingStrategy::default(),
            organization: OutputOrganization::default(),
            delivery_manager: DeliveryManager::default(),
            storage_manager: StorageManager::default(),
            backup_system: BackupSystem::default(),
            access_control: AccessControlManager::default(),
        }
    }
}

