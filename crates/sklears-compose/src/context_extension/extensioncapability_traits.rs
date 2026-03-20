//! # ExtensionCapability - Trait Implementations
//!
//! This module contains trait implementations for `ExtensionCapability`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ExtensionCapability;

impl Display for ExtensionCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtensionCapability::FileRead => write!(f, "file_read"),
            ExtensionCapability::FileWrite => write!(f, "file_write"),
            ExtensionCapability::NetworkClient => write!(f, "network_client"),
            ExtensionCapability::NetworkServer => write!(f, "network_server"),
            ExtensionCapability::DatabaseAccess => write!(f, "database_access"),
            ExtensionCapability::ProcessSpawn => write!(f, "process_spawn"),
            ExtensionCapability::EnvironmentAccess => write!(f, "environment_access"),
            ExtensionCapability::RegistryAccess => write!(f, "registry_access"),
            ExtensionCapability::IpcCommunication => write!(f, "ipc_communication"),
            ExtensionCapability::Custom(name) => write!(f, "custom_{}", name),
        }
    }
}

