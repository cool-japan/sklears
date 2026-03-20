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

impl Default for QuantumDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "async_support")]
#[async_trait]
impl HardwareDiscovery for QuantumDiscovery {
    async fn discover(&self) -> Result<Vec<Box<dyn ExoticHardware>>> {
        let mut devices = Vec::new();
        let mut ibm_device = QuantumDevice::new(0, QuantumBackend::Superconducting);
        ibm_device.initialize().await?;
        devices.push(Box::new(ibm_device) as Box<dyn ExoticHardware>);
        let mut ionq_device = QuantumDevice::new(1, QuantumBackend::IonTrap);
        ionq_device.initialize().await?;
        devices.push(Box::new(ionq_device) as Box<dyn ExoticHardware>);
        Ok(devices)
    }
    fn agent_name(&self) -> &str {
        "Quantum Discovery Agent"
    }
}
