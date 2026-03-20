//! Architecture improvements
//!
//! This module provides comprehensive architectural pattern implementations including
//! plugin architecture, middleware pipelines, event-driven systems, configuration management,
//! service locators, dependency injection, fluent APIs, hooks and interceptors, modular design
//! patterns, component lifecycle management, and architectural quality assurance tools.
//! All architectural patterns have been refactored into focused modules for better
//! maintainability and comply with SciRS2 Policy.

// Core architectural types and base structures
mod architecture_core;
pub use architecture_core::{
    ArchitecturalPattern, ArchitectureConfig, ArchitectureValidator, ArchitectureEstimator,
    ArchitectureTransformer, ArchitecturalAnalyzer, DesignPatternEngine, ModularityAnalyzer
};

// Plugin system and extensible architecture
mod plugin_system;
pub use plugin_system::{
    Plugin, PluginManager, PluginContext, PluginResult, PluginError, PluginState,
    PluginExecutionStats, PluginValidator, PluginRegistry, DynamicPluginLoader
};

// Middleware pipeline and request processing
mod middleware_pipeline;
pub use middleware_pipeline::{
    Middleware, MiddlewarePipeline, MiddlewareContext, MiddlewareError, MiddlewareValidator,
    MiddlewareExecutionStats, RequestProcessor, ResponseProcessor, PipelineOrchestrator
};

// Component registry and dependency injection
mod component_registry;
pub use component_registry::{
    ComponentRegistry, ComponentError, ComponentValidator, DependencyInjector,
    ServiceContainer, ComponentLifecycle, ComponentMetadata, ComponentResolver
};

// Service locator and service discovery
mod service_locator;
pub use service_locator::{
    ServiceLocator, ServiceMetadata, ServiceLifecycle, ServiceValidator,
    ServiceDiscovery, ServiceRegistry, ServiceProvider, ServiceConsumer
};

// Event bus and event-driven architecture
mod event_bus;
pub use event_bus::{
    EventBus, Event, EventHandler, EventError, EventValidator, EventHandlerStats,
    EventRouter, EventProcessor, EventSubscription, EventPublisher
};

// Fluent API builders and interfaces
mod fluent_api;
pub use fluent_api::{
    FluentApiBuilder, BuilderError, FluentApiValidator, ConfigurationBuilder,
    ValidationError, FluentInterfaceDesigner, APIPatternBuilder, FluentMethodChainer
};

// Configuration presets and management
mod configuration_presets;
pub use configuration_presets::{
    ConfigurationPreset, PresetRegistry, PresetApplicationResult, PresetValidationRule,
    PresetValidationType, PresetBuilder, PresetValidator, ConfigurationTemplate
};

// Hook system and interceptors
mod hook_system;
pub use hook_system::{
    Hook, HookRegistry, HookContext, HookResult, HookError, HookType, HookConfig,
    HookExecutionStats, HookErrorHandling, PipelineHookManager, UtilityHookManager
};

// Module registry and feature categorization
mod module_registry;
pub use module_registry::{
    ModuleRegistry, FeatureModule, ModuleConfig, ModuleValidator,
    ModuleCatalog, FeatureCategorizer, ModuleDependencyResolver, ModuleLifecycleManager
};

// Utility registry and function management
mod utility_registry;
pub use utility_registry::{
    UtilityRegistry, UtilityFunction, UtilityContext, UtilityResult, UtilityValue,
    UtilityError, UtilityExecutionStats, UtilityValidator, FunctionCatalog
};

// Fluent chain building and operation pipelines
mod fluent_chains;
pub use fluent_chains::{
    FluentChainBuilder, FluentChain, FluentOperation, FluentOperationType, FluentCondition,
    FluentErrorHandling, FluentRetryPolicy, FluentChainResult, ChainExecutionStats,
    ChainValidationRule, ChainValidationType, OperationPipeline, PipelineExecutor
};

// Architectural patterns and design templates
mod architectural_patterns;
pub use architectural_patterns::{
    ArchitecturalPatterns, DesignTemplate, PatternCatalog, ArchitecturalPrinciples,
    PatternValidator, DesignPatternApplicator, ArchitecturalGuide, PatternMatcher
};

// Performance optimization for architectural components
mod performance_optimization;
pub use performance_optimization::{
    ArchitecturePerformanceOptimizer, ComputationalEfficiency, MemoryOptimizer,
    AlgorithmicOptimizer, CacheOptimizer, ParallelArchitectureProcessor
};

// Utilities and helper functions
mod architecture_utilities;
pub use architecture_utilities::{
    ArchitectureUtilities, DesignPatternUtils, ArchitecturalMathUtils, ValidationUtils,
    ComputationalUtils, HelperFunctions, ArchitecturalAnalysisUtils, UtilityValidator
};

// Re-export main architectural classes for backwards compatibility
pub use plugin_system::{Plugin, PluginManager, PluginContext, PluginResult, PluginError};
pub use middleware_pipeline::{Middleware, MiddlewarePipeline, MiddlewareContext, MiddlewareError};
pub use component_registry::{ComponentRegistry, ComponentError};
pub use service_locator::{ServiceLocator, ServiceMetadata, ServiceLifecycle};
pub use event_bus::{EventBus, Event, EventHandler, EventError};
pub use fluent_api::{FluentApiBuilder, BuilderError, ConfigurationBuilder, ValidationError};

// Re-export common configurations and types
pub use architecture_core::ArchitectureConfig;
pub use module_registry::{FeatureModule, ModuleConfig};
pub use utility_registry::{UtilityValue, UtilityContext, UtilityResult};
pub use hook_system::{HookType, HookConfig, HookErrorHandling};
pub use fluent_chains::{FluentOperationType, FluentErrorHandling};