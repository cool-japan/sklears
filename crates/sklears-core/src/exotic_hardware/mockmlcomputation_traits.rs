//! Auto-generated trait implementations
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "async_support")]
use super::functions::*;
use super::types::*;
#[cfg(feature = "async_support")]
use crate::error::Result;

impl Default for MockMLComputation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "async_support")]
impl HardwareComputation for MockMLComputation {
    fn get_computation_graph(&self) -> Result<ComputationGraph> {
        Ok(self.graph.clone())
    }
    fn input_specs(&self) -> Vec<TensorSpec> {
        vec![TensorSpec {
            shape: vec![1024, 256],
            dtype: Precision::Float32,
            layout: MemoryLayout::RowMajor,
            sparsity: None,
        }]
    }
    fn output_specs(&self) -> Vec<TensorSpec> {
        vec![TensorSpec {
            shape: vec![1024, 512],
            dtype: Precision::Float32,
            layout: MemoryLayout::RowMajor,
            sparsity: None,
        }]
    }
    fn metadata(&self) -> ComputationMetadata {
        ComputationMetadata {
            name: "Mock ML Computation".to_string(),
            version: "1.0".to_string(),
            estimated_flops: 1024 * 256 * 512 * 2,
            memory_requirement_bytes: 1024 * 512 * 4,
            latency_requirement_ms: Some(10.0),
            throughput_requirement_ops_per_sec: Some(100.0),
        }
    }
    fn validate_for_hardware(&self, hardware: &dyn ExoticHardware) -> Result<ValidationReport> {
        let capabilities = hardware.capabilities();
        let is_compatible = match hardware.hardware_id().device_type {
            HardwareType::TPU => true,
            HardwareType::FPGA => true,
            HardwareType::Quantum => false,
            _ => false,
        };
        let estimated_performance = if is_compatible {
            Some(PerformanceEstimate {
                latency_ms: match hardware.hardware_id().device_type {
                    HardwareType::TPU => 2.0,
                    HardwareType::FPGA => 5.0,
                    _ => 100.0,
                },
                throughput_ops_per_sec: capabilities.peak_performance_ops as f32 * 0.8,
                memory_usage_bytes: 1024 * 512 * 4,
                power_usage_watts: match hardware.hardware_id().device_type {
                    HardwareType::TPU => 450.0,
                    HardwareType::FPGA => 75.0,
                    _ => 25.0,
                },
                confidence: 0.85,
            })
        } else {
            None
        };
        Ok(ValidationReport {
            is_compatible,
            warnings: vec![],
            errors: if is_compatible {
                vec![]
            } else {
                vec!["Hardware not suitable for this computation type".to_string()]
            },
            optimizations: vec!["Consider using operator fusion".to_string()],
            estimated_performance,
        })
    }
}
