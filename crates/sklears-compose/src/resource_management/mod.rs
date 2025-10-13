//! Resource Management System for Composable Execution Engine
//!
//! This module provides comprehensive resource management capabilities for the composable
//! execution engine, including intelligent allocation, monitoring, optimization, and
//! constraint checking for CPU, memory, GPU, network, and storage resources. The system
//! is designed to maximize resource utilization while ensuring fair allocation and
//! preventing resource starvation.
//!
//! # Resource Management Architecture
//!
//! The resource management system is organized into specialized components:
//!
//! ```text
//! ResourceManager (main coordinator)
//! ├── ResourcePool              // Resource pool management
//! ├── ResourceAllocation        // Allocation tracking and management
//! ├── ResourceMonitor           // Real-time resource monitoring
//! ├── ResourceOptimizer         // Optimization algorithms
//! ├── ResourceConstraintChecker // Constraint validation
//! ├── Specialized Managers:
//! │   ├── CpuResourceManager    // CPU core management
//! │   ├── MemoryResourceManager // Memory allocation
//! │   ├── GpuResourceManager    // GPU device management
//! │   ├── NetworkResourceManager // Network bandwidth
//! │   └── StorageResourceManager // Storage I/O management
//! ├── ResourceUsageTracker      // Usage statistics and tracking
//! └── ResourcePredictionEngine  // Predictive resource planning
//! ```

pub mod constraints;
pub mod cpu_manager;
pub mod defaults;
pub mod gpu_manager;
pub mod memory_manager;
pub mod monitoring;
pub mod network_manager;
pub mod optimization;
pub mod resource_types;
pub mod simd_operations;
pub mod storage_manager;

// Re-export main types and functionality
pub use constraints::ResourceConstraintChecker;
pub use monitoring::{AlertSystem, ResourceMonitor};
pub use optimization::{ResourceOptimizer, ResourcePredictionEngine};
pub use resource_types::{
    ResourceAllocation, ResourceManager, ResourceManagerState, ResourceManagerStats,
    ResourceUtilization,
};

// Re-export specialized managers
pub use cpu_manager::CpuResourceManager;
pub use gpu_manager::GpuResourceManager;
pub use memory_manager::MemoryResourceManager;
pub use network_manager::NetworkResourceManager;
pub use storage_manager::StorageResourceManager;

// Re-export SIMD operations for performance-critical code
pub use simd_operations::*;

// Re-export default implementations
pub use defaults::*;
