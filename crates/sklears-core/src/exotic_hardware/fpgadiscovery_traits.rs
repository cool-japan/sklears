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

impl Default for FpgaDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "async_support")]
#[async_trait]
impl HardwareDiscovery for FpgaDiscovery {
    async fn discover(&self) -> Result<Vec<Box<dyn ExoticHardware>>> {
        let mut devices = Vec::new();
        let mut xilinx_device = FpgaDevice::new(0, FpgaVendor::Xilinx);
        xilinx_device.initialize().await?;
        devices.push(Box::new(xilinx_device) as Box<dyn ExoticHardware>);
        let mut intel_device = FpgaDevice::new(1, FpgaVendor::Intel);
        intel_device.initialize().await?;
        devices.push(Box::new(intel_device) as Box<dyn ExoticHardware>);
        Ok(devices)
    }
    fn agent_name(&self) -> &str {
        "FPGA Discovery Agent"
    }
}
