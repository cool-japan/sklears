//! Auto-generated module structure

pub mod exotichardwaremanager_traits;
pub mod fpgacompiler_traits;
pub mod fpgadevice_traits;
pub mod fpgadiscovery_traits;
pub mod fpgamemorymanager_traits;
pub mod functions;
pub mod hardwarecapabilities_traits;
pub mod hardwareid_traits;
pub mod hardwaretype_traits;
pub mod mockmlcomputation_traits;
pub mod quantumcompiler_traits;
pub mod quantumdevice_traits;
pub mod quantumdiscovery_traits;
pub mod tpucompiler_traits;
pub mod tpudevice_traits;
pub mod tpudiscovery_traits;
pub mod tpumemorymanager_traits;
pub mod types;

// Re-export all types
pub use types::*;

// Re-export traits from functions module (only with async_support feature)
#[cfg(feature = "async_support")]
pub use functions::{
    ExoticHardware, HardwareCompiler, HardwareComputation, HardwareDiscovery, HardwareMemoryManager,
};
