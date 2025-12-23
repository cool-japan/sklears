//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "async_support")]
use super::types::*;
#[cfg(feature = "async_support")]
use crate::error::Result;
#[cfg(feature = "async_support")]
use async_trait::async_trait;
#[cfg(feature = "async_support")]
use std::time::Duration;
/// Core trait for exotic hardware devices
#[cfg(feature = "async_support")]
#[async_trait]
pub trait ExoticHardware: Send + Sync {
    /// Get hardware identification
    fn hardware_id(&self) -> &HardwareId;
    /// Get hardware capabilities
    fn capabilities(&self) -> &HardwareCapabilities;
    /// Initialize the hardware device
    async fn initialize(&mut self) -> Result<()>;
    /// Shutdown the hardware device
    async fn shutdown(&mut self) -> Result<()>;
    /// Check if hardware is available and ready
    async fn is_ready(&self) -> Result<bool>;
    /// Get current hardware status
    async fn status(&self) -> Result<HardwareStatus>;
    /// Execute a computation on the hardware
    async fn execute_computation(
        &self,
        computation: &dyn HardwareComputation,
    ) -> Result<ComputationResult>;
    /// Get hardware-specific compiler/optimizer
    fn get_compiler(&self) -> Result<Box<dyn HardwareCompiler>>;
    /// Get memory manager for this hardware
    fn get_memory_manager(&self) -> Result<Box<dyn HardwareMemoryManager>>;
}
/// Trait for computations that can be executed on exotic hardware
#[cfg(feature = "async_support")]
pub trait HardwareComputation: Send + Sync {
    /// Get the computation graph representation
    fn get_computation_graph(&self) -> Result<ComputationGraph>;
    /// Get required input specifications
    fn input_specs(&self) -> Vec<TensorSpec>;
    /// Get expected output specifications
    fn output_specs(&self) -> Vec<TensorSpec>;
    /// Get computation metadata
    fn metadata(&self) -> ComputationMetadata;
    /// Validate computation for specific hardware
    fn validate_for_hardware(&self, hardware: &dyn ExoticHardware) -> Result<ValidationReport>;
}
/// Hardware-specific compiler interface
#[cfg(feature = "async_support")]
pub trait HardwareCompiler: Send + Sync {
    /// Compile computation graph for target hardware
    fn compile(
        &self,
        graph: &ComputationGraph,
        options: &CompilationOptions,
    ) -> Result<CompiledProgram>;
    /// Optimize graph for hardware
    fn optimize(
        &self,
        graph: &ComputationGraph,
        target: &HardwareCapabilities,
    ) -> Result<ComputationGraph>;
    /// Get supported optimization passes
    fn supported_optimizations(&self) -> Vec<OptimizationPass>;
    /// Estimate compilation time
    fn estimate_compilation_time(&self, graph: &ComputationGraph) -> Result<Duration>;
}
/// Hardware-specific memory manager
#[cfg(feature = "async_support")]
#[async_trait]
pub trait HardwareMemoryManager: Send + Sync {
    /// Allocate memory on hardware
    async fn allocate(&self, size_bytes: u64, alignment: u32) -> Result<MemoryHandle>;
    /// Deallocate memory
    async fn deallocate(&self, handle: MemoryHandle) -> Result<()>;
    /// Copy data to hardware memory
    async fn copy_to_device(&self, handle: MemoryHandle, data: &[u8]) -> Result<()>;
    /// Copy data from hardware memory
    async fn copy_from_device(&self, handle: MemoryHandle, data: &mut [u8]) -> Result<()>;
    /// Get memory statistics
    async fn memory_stats(&self) -> Result<MemoryStats>;
    /// Synchronize memory operations
    async fn synchronize(&self) -> Result<()>;
}
/// Hardware discovery trait
#[cfg(feature = "async_support")]
#[async_trait]
pub trait HardwareDiscovery: Send + Sync {
    /// Discover available hardware of this type
    async fn discover(&self) -> Result<Vec<Box<dyn ExoticHardware>>>;
    /// Get discovery agent name
    fn agent_name(&self) -> &str;
}
/// Example of using exotic hardware for ML computation
#[cfg(feature = "async_support")]
pub async fn example_exotic_hardware_usage() -> Result<()> {
    let mut manager = ExoticHardwareManager::new();
    let devices = manager.discover_hardware().await?;
    println!("Discovered {} exotic hardware devices", devices.len());
    let computation = MockMLComputation::new();
    if let Some(best_device_id) = manager.find_best_device(&computation).await? {
        println!("Best device for computation: {}", best_device_id);
        if let Some(device) = manager.get_device(best_device_id) {
            let result = device.execute_computation(&computation).await?;
            println!("Computation completed in {}ms", result.execution_time_ms);
        }
    }
    Ok(())
}
#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::exotic_hardware::{HardwareId, HardwareType, Complex64};
    #[cfg(feature = "async_support")]
    #[tokio::test]
    async fn test_tpu_device_creation() {
        let mut tpu = TpuDevice::new(0, TpuVersion::V3);
        assert_eq!(tpu.hardware_id().device_type, HardwareType::TPU);
        assert!(!tpu.is_ready().await.unwrap());
        tpu.initialize().await.unwrap();
        assert!(tpu.is_ready().await.unwrap());
    }
    #[cfg(feature = "async_support")]
    #[tokio::test]
    async fn test_fpga_device_reconfiguration() {
        let mut fpga = FpgaDevice::new(0, FpgaVendor::Xilinx);
        assert!(!fpga.is_ready().await.unwrap());
        fpga.initialize().await.unwrap();
        assert!(fpga.is_ready().await.unwrap());
        let new_bitstream = vec![1u8; 2048];
        fpga.reconfigure(&new_bitstream).await.unwrap();
        assert!(fpga.configuration.is_some());
    }
    #[cfg(feature = "async_support")]
    #[tokio::test]
    async fn test_quantum_device_state() {
        let mut quantum = QuantumDevice::new(0, QuantumBackend::Superconducting);
        quantum.initialize().await.unwrap();
        let state = quantum.quantum_state().unwrap();
        assert_eq!(state.num_qubits, 4);
        assert_eq!(state.amplitudes.len(), 1 << 4);
    }
    #[cfg(feature = "async_support")]
    #[tokio::test]
    async fn test_hardware_manager_discovery() {
        let mut manager = ExoticHardwareManager::new();
        let devices = manager.discover_hardware().await.unwrap();
        assert!(!devices.is_empty());
        let device_types: std::collections::HashSet<_> =
            devices.iter().map(|id| id.device_type).collect();
        assert!(device_types.contains(&HardwareType::TPU));
        assert!(device_types.contains(&HardwareType::FPGA));
        assert!(device_types.contains(&HardwareType::Quantum));
    }
    #[test]
    fn test_hardware_id_display() {
        let id = HardwareId {
            device_type: HardwareType::TPU,
            device_index: 0,
            vendor: "Google".to_string(),
            model: "TPU-V3".to_string(),
        };
        assert_eq!(format!("{}", id), "TPU:Google-TPU-V3-0");
    }
    #[test]
    fn test_complex_number_operations() {
        let c1 = Complex64::new(3.0, 4.0);
        assert_eq!(c1.magnitude_squared(), 25.0);
    }
    #[cfg(feature = "async_support")]
    #[tokio::test]
    async fn test_mock_computation_validation() {
        let computation = MockMLComputation::new();
        let mut tpu = TpuDevice::new(0, TpuVersion::V3);
        tpu.initialize().await.unwrap();
        let validation = computation.validate_for_hardware(&tpu).unwrap();
        assert!(validation.is_compatible);
        assert!(validation.estimated_performance.is_some());
    }
}
