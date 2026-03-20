//! Auto-generated trait implementations
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "async_support")]
use super::functions::*;
use super::types::*;
#[cfg(feature = "async_support")]
use crate::error::Result;
#[cfg(feature = "async_support")]
use std::time::Duration;

impl Default for FpgaCompiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "async_support")]
impl HardwareCompiler for FpgaCompiler {
    fn compile(
        &self,
        _graph: &ComputationGraph,
        _options: &CompilationOptions,
    ) -> Result<CompiledProgram> {
        Ok(CompiledProgram {
            binary: vec![0u8; 8192],
            metadata: ProgramMetadata {
                compilation_time_ms: 180000.0,
                optimization_passes_applied: vec![
                    "HLS_Synthesis".to_string(),
                    "Place_And_Route".to_string(),
                    "Pipeline_Optimization".to_string(),
                ],
                estimated_performance: PerformanceEstimate {
                    latency_ms: 2.0,
                    throughput_ops_per_sec: 2000.0,
                    memory_usage_bytes: 512 * 1024,
                    power_usage_watts: 75.0,
                    confidence: 0.95,
                },
                checksum: "fpga_bitstream_checksum".to_string(),
            },
            resource_requirements: ResourceRequirements {
                memory_bytes: 512 * 1024,
                compute_units: 256,
                execution_time_estimate_ms: 2.0,
            },
        })
    }
    fn optimize(
        &self,
        graph: &ComputationGraph,
        _target: &HardwareCapabilities,
    ) -> Result<ComputationGraph> {
        let mut optimized = graph.clone();
        optimized.metadata.optimization_level = 3;
        Ok(optimized)
    }
    fn supported_optimizations(&self) -> Vec<OptimizationPass> {
        self.supported_ops.clone()
    }
    fn estimate_compilation_time(&self, graph: &ComputationGraph) -> Result<Duration> {
        let base_time_ms = 60000 + (graph.nodes.len() * 1000) as u64;
        Ok(Duration::from_millis(base_time_ms))
    }
}
