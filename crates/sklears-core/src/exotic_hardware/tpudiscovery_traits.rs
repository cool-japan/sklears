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

impl Default for TpuDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "async_support")]
#[async_trait]
impl HardwareDiscovery for TpuDiscovery {
    async fn discover(&self) -> Result<Vec<Box<dyn ExoticHardware>>> {
        let mut devices = Vec::new();
        for i in 0..2 {
            let mut device = TpuDevice::new(i, TpuVersion::V3);
            device.initialize().await?;
            devices.push(Box::new(device) as Box<dyn ExoticHardware>);
        }
        Ok(devices)
    }
    fn agent_name(&self) -> &str {
        "TPU Discovery Agent"
    }
}
