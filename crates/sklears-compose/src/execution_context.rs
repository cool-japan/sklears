//! # Execution Context Module
//!
//! This module provides comprehensive execution context management for the composable
//! execution framework. It defines the runtime environment and state management for
//! task execution, including configuration, security context, environmental variables,
//! resource constraints, and execution metadata.
//!
//! # Execution Context Architecture
//!
//! The execution context system is organized around multiple context layers:
//!
//! ```text
//! ExecutionContext (main coordinator)
//! ├── RuntimeContext             // Runtime environment and configuration
//! ├── SecurityContext            // Security and authorization context
//! ├── ResourceContext            // Resource allocation and constraints
//! ├── ConfigurationContext       // Configuration and parameters
//! ├── MetadataContext            // Execution metadata and tracking
//! ├── EnvironmentContext         // Environment variables and settings
//! ├── SessionContext             // Session and user context
//! ├── DiagnosticContext          // Debugging and diagnostic information
//! ├── ComplianceContext          // Regulatory and compliance context
//! └── ExtensionContext           // Custom context extensions
//! ```
//!
//! # Modular Architecture
//!
//! This module coordinates specialized context modules:
//!
//! - **[context_core]**: Core traits, types, and the main ExecutionContext coordinator
//! - **[runtime_context]**: Runtime environment, process, thread, and memory management
//! - **[security_context]**: Authentication, authorization, encryption, and audit trails
//! - **[resource_context]**: Resource allocation, constraints, monitoring, and quotas
//! - **[configuration_context]**: Dynamic configuration, feature flags, and validation
//! - **[metadata_context]**: Execution metadata, tracking, lineage, and versioning
//! - **[environment_context]**: Environment variables, system settings, and isolation
//! - **[session_context]**: Session management, user context, and lifecycle tracking
//! - **[diagnostic_context]**: Debugging information, profiling, and observability
//! - **[extension_context]**: Custom context extensions and plugin architecture
//!
//! # Context Management Features
//!
//! ## Runtime Context
//! - **Execution Environment**: Runtime configuration and settings
//! - **Process Context**: Process-level information and state
//! - **Thread Context**: Thread-local storage and state
//! - **Memory Context**: Memory management and allocation tracking
//!
//! ## Security Context
//! - **Authentication**: User and service authentication
//! - **Authorization**: Permission and access control
//! - **Encryption**: Data encryption and key management
//! - **Audit Trail**: Security event logging and tracking
//!
//! ## Resource Context
//! - **Resource Allocation**: Allocated resources tracking
//! - **Resource Constraints**: Resource usage limits
//! - **Resource Monitoring**: Real-time resource utilization
//! - **Resource Quotas**: Usage quotas and billing
//!
//! ## Configuration Context
//! - **Dynamic Configuration**: Runtime configuration updates
//! - **Feature Flags**: Feature toggle management
//! - **Parameter Injection**: Dynamic parameter injection
//! - **Configuration Validation**: Config validation and defaults
//!
//! # Usage Examples
//!
//! ## Basic Execution Context
//! ```rust,no_run
//! use crate::execution_context::*;
//!
//! // Create execution context
//! let mut context = ExecutionContext::new("task-001")?;
//!
//! // Set runtime configuration
//! context.set_config("batch_size", ConfigValue::Integer(100))?;
//! context.set_config("timeout", ConfigValue::Duration(std::time::Duration::from_secs(300)))?;
//! context.set_config("parallel_workers", ConfigValue::Integer(4))?;
//!
//! // Set security context
//! let security_ctx = SecurityContext {
//!     user_id: Some("user123".to_string()),
//!     session_id: Some("session456".to_string()),
//!     permissions: vec!["read".to_string(), "write".to_string()],
//!     auth_method: AuthenticationMethod::Token,
//!     encryption_required: true,
//!     ..Default::default()
//! };
//! context.set_security_context(security_ctx)?;
//!
//! // Set resource constraints
//! let resource_ctx = ResourceContext {
//!     max_cpu_cores: Some(8),
//!     max_memory: Some(16 * 1024 * 1024 * 1024), // 16GB
//!     max_execution_time: Some(std::time::Duration::from_secs(7200)), // 2 hours
//!     ..Default::default()
//! };
//! context.set_resource_context(resource_ctx)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Advanced Context Management
//! ```rust,no_run
//! // Create context with custom configuration
//! let context_config = ExecutionContextConfig {
//!     enable_security: true,
//!     enable_audit_trail: true,
//!     enable_metrics: true,
//!     enable_tracing: true,
//!     enable_compliance: true,
//!     context_isolation: ContextIsolationLevel::Process,
//!     state_persistence: StatePersistence::Memory,
//!     ..Default::default()
//! };
//!
//! let mut context = ExecutionContext::with_config(
//!     "advanced-task",
//!     context_config
//! )?;
//!
//! // Set environment variables
//! context.set_env("MODEL_PATH", "/data/models/model.pkl")?;
//! context.set_env("LOG_LEVEL", "INFO")?;
//! context.set_env("CUDA_VISIBLE_DEVICES", "0,1")?;
//!
//! // Get execution metadata
//! let metadata = context.get_metadata();
//! println!("Execution ID: {}", metadata.execution_id);
//! println!("Start Time: {:?}", metadata.start_time);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

// Re-export specialized context modules
pub mod context_core;
pub mod runtime_context;
pub mod security_context;
pub mod resource_context;
pub mod configuration_context;
pub mod metadata_context;
pub mod environment_context;
pub mod session_context;
pub mod diagnostic_context;
pub mod extension_context;

// Re-export all public types for easy access
pub use context_core::*;
pub use runtime_context::*;
pub use security_context::*;
pub use resource_context::*;
pub use configuration_context::*;
pub use metadata_context::*;
pub use environment_context::*;
pub use session_context::*;
pub use diagnostic_context::*;
pub use extension_context::*;