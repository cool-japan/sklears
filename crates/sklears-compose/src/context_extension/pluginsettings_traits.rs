//! # PluginSettings - Trait Implementations
//!
//! This module contains trait implementations for `PluginSettings`.
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

use super::types::{PluginCacheSettings, PluginFormat, PluginSettings, PluginVerificationSettings};

impl Default for PluginSettings {
    fn default() -> Self {
        let mut supported_formats = HashSet::new();
        supported_formats.insert(PluginFormat::DynamicLibrary);
        supported_formats.insert(PluginFormat::WebAssembly);
        Self {
            search_paths: vec![PathBuf::from("./plugins")],
            supported_formats,
            cache_settings: PluginCacheSettings::default(),
            auto_discovery: true,
            verification_settings: PluginVerificationSettings::default(),
        }
    }
}

