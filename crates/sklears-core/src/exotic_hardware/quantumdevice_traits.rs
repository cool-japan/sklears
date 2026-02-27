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
impl ExoticHardware for QuantumDevice {
    fn hardware_id(&self) -> &HardwareId {
        &self.id
    }
    fn capabilities(&self) -> &HardwareCapabilities {
        &self.capabilities
    }
    async fn initialize(&mut self) -> Result<()> {
        self.initialize_quantum_state(self.capabilities.compute_units as usize)
            .await?;
        self.is_initialized = true;
        Ok(())
    }
    async fn shutdown(&mut self) -> Result<()> {
        self.quantum_state = None;
        self.is_initialized = false;
        Ok(())
    }
    async fn is_ready(&self) -> Result<bool> {
        Ok(self.is_initialized && self.quantum_state.is_some())
    }
    async fn status(&self) -> Result<HardwareStatus> {
        Ok(HardwareStatus {
            is_online: self.is_initialized,
            temperature_celsius: Some(0.01),
            power_usage_watts: Some(25.0),
            memory_usage_percent: 0.0,
            compute_utilization_percent: 0.0,
            error_count: 0,
            uptime_seconds: 7200,
        })
    }
    async fn execute_computation(
        &self,
        computation: &dyn HardwareComputation,
    ) -> Result<ComputationResult> {
        if !self.is_ready().await? {
            return Err(SklearsError::HardwareError(
                "Quantum device not ready".to_string(),
            ));
        }
        let validation = computation.validate_for_hardware(self)?;
        if !validation.is_compatible {
            return Err(SklearsError::HardwareError(format!(
                "Computation not compatible with quantum device: {:?}",
                validation.errors
            )));
        }
        let start_time = std::time::Instant::now();
        tokio::time::sleep(Duration::from_millis(100)).await;
        let execution_time = start_time.elapsed().as_millis() as f32;
        Ok(ComputationResult {
            outputs: vec![],
            execution_time_ms: execution_time,
            memory_used_bytes: 0,
            hardware_metrics: HardwareMetrics {
                compute_utilization: 100.0,
                memory_bandwidth_gbps: 0.0,
                energy_consumed_joules: execution_time * 0.025,
                hardware_specific: {
                    let mut metrics = HashMap::new();
                    metrics.insert(
                        "gate_count".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(150)),
                    );
                    metrics.insert(
                        "circuit_depth".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(50)),
                    );
                    metrics.insert(
                        "decoherence_error".to_string(),
                        serde_json::Value::Number(
                            serde_json::Number::from_f64(0.01).expect("valid JSON operation"),
                        ),
                    );
                    metrics
                },
            },
        })
    }
    fn get_compiler(&self) -> Result<Box<dyn HardwareCompiler>> {
        Ok(Box::new(QuantumCompiler::new()))
    }
    fn get_memory_manager(&self) -> Result<Box<dyn HardwareMemoryManager>> {
        Err(SklearsError::HardwareError(
            "Quantum devices do not support traditional memory management".to_string(),
        ))
    }
}
