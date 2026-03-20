//! # ExtensionSecuritySettings - Trait Implementations
//!
//! This module contains trait implementations for `ExtensionSecuritySettings`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::{
    collections::{HashMap, HashSet, VecDeque, BTreeMap},
    sync::{Arc, RwLock, Mutex},
    time::{Duration, SystemTime, Instant},
    fmt::{Debug, Display},
    path::PathBuf, any::Any, thread,
};

use super::types::{ExtensionCapability, ExtensionSecuritySettings, PermissionLevel};

impl Default for ExtensionSecuritySettings {
    fn default() -> Self {
        let mut allowed_capabilities = HashSet::new();
        allowed_capabilities.insert(ExtensionCapability::FileRead);
        allowed_capabilities.insert(ExtensionCapability::NetworkClient);
        Self {
            require_signed: false,
            trusted_publishers: HashSet::new(),
            allowed_capabilities,
            restricted_apis: HashSet::new(),
            enable_permissions: true,
            default_permission_level: PermissionLevel::Restricted,
        }
    }
}

