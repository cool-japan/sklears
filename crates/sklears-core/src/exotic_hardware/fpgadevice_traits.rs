//! Auto-generated trait implementations
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "async_support")]
use super::functions::*;
#[cfg(feature = "async_support")]
use super::types::*;
#[cfg(feature = "async_support")]
use crate::error::{Result, SklearsError};
#[cfg(feature = "async_support")]
use async_trait::async_trait;
#[cfg(feature = "async_support")]
use std::collections::HashMap;
#[cfg(feature = "async_support")]
use std::time::Duration;

#[cfg(feature = "async_support")]
#[async_trait]
impl ExoticHardware for FpgaDevice {
    fn hardware_id(&self) -> &HardwareId {
        &self.id
    }
    fn capabilities(&self) -> &HardwareCapabilities {
        &self.capabilities
    }
    async fn initialize(&mut self) -> Result<()> {
        let default_bitstream = vec![0u8; 1024];
        self.reconfigure(&default_bitstream).await?;
        self.is_initialized = true;
        Ok(())
    }
    async fn shutdown(&mut self) -> Result<()> {
        self.configuration = None;
        self.is_initialized = false;
        Ok(())
    }
    async fn is_ready(&self) -> Result<bool> {
        Ok(self.is_initialized && self.configuration.is_some())
    }
    async fn status(&self) -> Result<HardwareStatus> {
        let config = self.configuration.as_ref();
        Ok(HardwareStatus {
            is_online: self.is_initialized,
            temperature_celsius: Some(55.0),
            power_usage_watts: config.map(|c| c.power_consumption_watts),
            memory_usage_percent: config.map(|c| c.memory_utilization).unwrap_or(0.0),
            compute_utilization_percent: config.map(|c| c.logic_utilization).unwrap_or(0.0),
            error_count: 0,
            uptime_seconds: 1800,
        })
    }
    async fn execute_computation(
        &self,
        computation: &dyn HardwareComputation,
    ) -> Result<ComputationResult> {
        if !self.is_ready().await? {
            return Err(SklearsError::HardwareError("FPGA not ready".to_string()));
        }
        let validation = computation.validate_for_hardware(self)?;
        if !validation.is_compatible {
            return Err(SklearsError::HardwareError(format!(
                "Computation not compatible with FPGA: {:?}",
                validation.errors
            )));
        }
        let start_time = std::time::Instant::now();
        tokio::time::sleep(Duration::from_millis(20)).await;
        let execution_time = start_time.elapsed().as_millis() as f32;
        Ok(ComputationResult {
            outputs: vec![],
            execution_time_ms: execution_time,
            memory_used_bytes: 512 * 1024,
            hardware_metrics: HardwareMetrics {
                compute_utilization: 85.0,
                memory_bandwidth_gbps: 460.0,
                energy_consumed_joules: execution_time * 0.075,
                hardware_specific: {
                    let mut metrics = HashMap::new();
                    if let Some(config) = &self.configuration {
                        metrics.insert(
                            "logic_utilization".to_string(),
                            serde_json::Value::Number(
                                serde_json::Number::from_f64(config.logic_utilization as f64)
                                    .unwrap(),
                            ),
                        );
                        metrics.insert(
                            "dsp_utilization".to_string(),
                            serde_json::Value::Number(
                                serde_json::Number::from_f64(config.dsp_utilization as f64)
                                    .unwrap(),
                            ),
                        );
                    }
                    metrics
                },
            },
        })
    }
    fn get_compiler(&self) -> Result<Box<dyn HardwareCompiler>> {
        Ok(Box::new(FpgaCompiler::new()))
    }
    fn get_memory_manager(&self) -> Result<Box<dyn HardwareMemoryManager>> {
        Ok(Box::new(FpgaMemoryManager::new()))
    }
}
