//! Metadata Context Module
//!
//! This module provides comprehensive execution metadata and tracking for execution contexts,
//! including lineage tracking, provenance, workflow dependencies, and execution history.
//!
//! This is the main orchestration module that coordinates all metadata management functionality
//! through specialized sub-modules for optimal organization and maintainability.
//!
//! # Architecture
//!
//! The metadata context system is built around several specialized modules:
//! - **metadata_storage**: Core data storage and retrieval functionality
//! - **lineage_tracking**: Lineage graph and provenance management
//! - **execution_history**: Execution tracking and historical analysis
//! - **search_engine**: Advanced metadata search and indexing capabilities
//! - **schema_validation**: Metadata schema validation and enforcement
//! - **dependency_tracker**: Dependency resolution and management
//!
//! # Features
//!
//! - **Comprehensive Metadata Management**: Complete metadata lifecycle management
//! - **Advanced Lineage Tracking**: Sophisticated data and execution lineage tracking
//! - **Provenance Management**: Detailed provenance information capture and analysis
//! - **Execution History**: Complete execution tracking and historical analysis
//! - **Powerful Search**: Advanced search capabilities with full-text indexing
//! - **Schema Validation**: Robust metadata schema validation and enforcement
//! - **Dependency Resolution**: Intelligent dependency tracking and resolution
//! - **Performance Optimization**: High-performance storage and retrieval
//!
//! # Usage
//!
//! ```rust,no_run
//! use crate::metadata_context::{MetadataContext, MetadataEntryType};
//! use std::collections::HashMap;
//!
//! // Create metadata context
//! let context = MetadataContext::new("execution-ctx".to_string())?;
//!
//! // Store metadata
//! let mut metadata = HashMap::new();
//! metadata.insert("key".to_string(), "value".to_string());
//! context.store_metadata("entry-1", MetadataEntryType::ExecutionStart, metadata)?;
//!
//! // Track lineage
//! context.track_lineage("parent-id", "child-id")?;
//!
//! // Search metadata
//! let results = context.search_metadata("execution")?;
//! ```

use std::{
    collections::{HashMap, HashSet, VecDeque, BTreeMap},
    sync::{Arc, RwLock, Mutex},
    time::{Duration, Instant, SystemTime},
    fmt::{Debug, Display},
    path::PathBuf,
};
use serde::{Serialize, Deserialize};
use uuid::Uuid;

use crate::context_core::{
    ExecutionContextTrait, ContextType, ContextState, ContextMetadata, ContextError, ContextResult,
    ContextEvent, IsolationLevel, ContextPriority
};

// Import specialized modules
pub mod metadata_storage;
pub mod lineage_tracking;
pub mod execution_history;
pub mod search_engine;
pub mod schema_validation;
pub mod dependency_tracker;

// Re-export key types from modules for convenience
pub use metadata_storage::{
    MetadataStore, MetadataStoreConfig, ExecutionMetadataEntry, MetadataPayload,
    MetadataEntryType, MetadataEntryStatus, MetadataSchema, TagDefinition,
    MetadataRelationship, MetadataStoreMetrics
};

pub use lineage_tracking::{
    LineageTracker, LineageGraph, LineageNode, LineageEdge, LineageNodeType,
    LineageEdgeType, ProvenanceManager, ProvenanceInfo, ProvenanceSource,
    ProvenanceAgent, ProvenanceActivity, LineageMetrics, ProvenanceMetrics
};

pub use execution_history::{
    ExecutionHistory, ExecutionRecord, ExecutionStatus, ExecutionError,
    ErrorCategory, ResourceUsageSnapshot, PerformanceMetrics,
    ExecutionHistoryConfig, ExecutionHistoryStats, ExecutionHistoryMetrics
};

pub use search_engine::{
    MetadataSearchEngine, SearchQuery, QueryType, SearchFilter, FilterOperator,
    SortCriteria, Pagination, Aggregation, SearchResult, SearchIndex,
    SearchEngineConfig, SearchMetrics
};

pub use schema_validation::{
    SchemaValidator, ValidationResult, ValidationRule, SchemaDefinition,
    FieldDefinition, FieldType, ValidationError, ValidationConfig,
    SchemaRegistry, SchemaMetrics
};

pub use dependency_tracker::{
    DependencyTracker, DependencyType, DependencyRelation, ResolutionResult,
    DependencyGraph, DependencyNode, CircularDependencyError,
    DependencyConfig, DependencyMetrics
};

/// Main metadata context orchestrating all metadata management functionality
#[derive(Debug)]
pub struct MetadataContext {
    /// Context identifier
    context_id: String,

    /// Core metadata storage system
    metadata_store: Arc<RwLock<metadata_storage::MetadataStore>>,

    /// Lineage and provenance tracking system
    lineage_tracker: Arc<lineage_tracking::LineageTracker>,

    /// Execution history management system
    execution_history: Arc<RwLock<execution_history::ExecutionHistory>>,

    /// Metadata search and indexing engine
    search_engine: Arc<search_engine::MetadataSearchEngine>,

    /// Schema validation system
    schema_validator: Arc<schema_validation::SchemaValidator>,

    /// Dependency tracking and resolution system
    dependency_tracker: Arc<dependency_tracker::DependencyTracker>,

    /// Context state
    state: Arc<RwLock<ContextState>>,

    /// Context metadata
    metadata: Arc<RwLock<ContextMetadata>>,

    /// Combined metrics from all subsystems
    metrics: Arc<Mutex<MetadataMetrics>>,

    /// Configuration
    config: MetadataContextConfig,
}

/// Comprehensive metrics aggregating all subsystem metrics
#[derive(Debug, Clone, Default)]
pub struct MetadataMetrics {
    /// Storage metrics
    pub storage_metrics: metadata_storage::MetadataStoreMetrics,

    /// Lineage tracking metrics
    pub lineage_metrics: lineage_tracking::LineageMetrics,

    /// Execution history metrics
    pub history_metrics: execution_history::ExecutionHistoryMetrics,

    /// Search engine metrics
    pub search_metrics: search_engine::SearchMetrics,

    /// Schema validation metrics
    pub schema_metrics: schema_validation::SchemaMetrics,

    /// Dependency tracking metrics
    pub dependency_metrics: dependency_tracker::DependencyMetrics,

    /// Overall system metrics
    pub system_metrics: SystemMetrics,
}

/// System-wide metrics
#[derive(Debug, Clone, Default)]
pub struct SystemMetrics {
    /// Total operations performed
    pub total_operations: u64,

    /// Average operation latency
    pub average_latency: Duration,

    /// System memory usage
    pub memory_usage_bytes: u64,

    /// System uptime
    pub uptime: Duration,

    /// Error rate
    pub error_rate: f64,

    /// Last updated timestamp
    pub last_updated: SystemTime,
}

/// Configuration for the metadata context system
#[derive(Debug, Clone)]
pub struct MetadataContextConfig {
    /// Storage configuration
    pub storage_config: metadata_storage::MetadataStoreConfig,

    /// Lineage tracking configuration
    pub lineage_config: lineage_tracking::LineageConfig,

    /// Execution history configuration
    pub history_config: execution_history::ExecutionHistoryConfig,

    /// Search engine configuration
    pub search_config: search_engine::SearchEngineConfig,

    /// Schema validation configuration
    pub validation_config: schema_validation::ValidationConfig,

    /// Dependency tracking configuration
    pub dependency_config: dependency_tracker::DependencyConfig,

    /// Enable performance optimizations
    pub enable_optimizations: bool,

    /// Enable comprehensive logging
    pub enable_logging: bool,

    /// Metrics collection interval
    pub metrics_interval: Duration,
}

impl Default for MetadataContextConfig {
    fn default() -> Self {
        Self {
            storage_config: metadata_storage::MetadataStoreConfig::default(),
            lineage_config: lineage_tracking::LineageConfig::default(),
            history_config: execution_history::ExecutionHistoryConfig::default(),
            search_config: search_engine::SearchEngineConfig::default(),
            validation_config: schema_validation::ValidationConfig::default(),
            dependency_config: dependency_tracker::DependencyConfig::default(),
            enable_optimizations: true,
            enable_logging: true,
            metrics_interval: Duration::from_secs(60),
        }
    }
}

impl MetadataContext {
    /// Create new metadata context with comprehensive functionality
    pub fn new(context_id: String) -> ContextResult<Self> {
        let config = MetadataContextConfig::default();
        Self::with_config(context_id, config)
    }

    /// Create metadata context with custom configuration
    pub fn with_config(context_id: String, config: MetadataContextConfig) -> ContextResult<Self> {
        // Initialize all subsystems
        let metadata_store = Arc::new(RwLock::new(
            metadata_storage::MetadataStore::with_config(config.storage_config.clone())?
        ));

        let lineage_tracker = Arc::new(
            lineage_tracking::LineageTracker::with_config(config.lineage_config.clone())?
        );

        let execution_history = Arc::new(RwLock::new(
            execution_history::ExecutionHistory::with_config(config.history_config.clone())?
        ));

        let search_engine = Arc::new(
            search_engine::MetadataSearchEngine::with_config(config.search_config.clone())?
        );

        let schema_validator = Arc::new(
            schema_validation::SchemaValidator::with_config(config.validation_config.clone())?
        );

        let dependency_tracker = Arc::new(
            dependency_tracker::DependencyTracker::with_config(config.dependency_config.clone())?
        );

        Ok(Self {
            context_id: context_id.clone(),
            metadata_store,
            lineage_tracker,
            execution_history,
            search_engine,
            schema_validator,
            dependency_tracker,
            state: Arc::new(RwLock::new(ContextState::Active)),
            metadata: Arc::new(RwLock::new(ContextMetadata {
                context_id: context_id.clone(),
                context_type: ContextType::Metadata,
                created_at: SystemTime::now(),
                properties: HashMap::new(),
                tags: HashSet::new(),
            })),
            metrics: Arc::new(Mutex::new(MetadataMetrics::default())),
            config,
        })
    }

    /// Store metadata entry with comprehensive tracking
    pub fn store_metadata(
        &self,
        entry_id: &str,
        entry_type: MetadataEntryType,
        metadata: HashMap<String, String>
    ) -> ContextResult<()> {
        // Validate metadata against schema if validation is enabled
        if let Err(validation_error) = self.schema_validator.validate_metadata(&metadata) {
            return Err(ContextError::InvalidData(
                format!("Schema validation failed: {}", validation_error)
            ));
        }

        // Create metadata entry
        let entry = ExecutionMetadataEntry {
            entry_id: entry_id.to_string(),
            context_id: self.context_id.clone(),
            entry_type,
            metadata: MetadataPayload {
                data: metadata.clone(),
                content_type: "application/json".to_string(),
                encoding: "utf-8".to_string(),
                size_bytes: serde_json::to_string(&metadata)
                    .map(|s| s.len())
                    .unwrap_or(0) as u64,
                checksum: None,
            },
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            version: "1.0".to_string(),
            status: MetadataEntryStatus::Active,
            tags: HashSet::new(),
            attributes: HashMap::new(),
            lineage: lineage_tracking::LineageInfo::new(),
            provenance: lineage_tracking::ProvenanceInfo::new(),
        };

        // Store in metadata store
        self.metadata_store.write().unwrap().store_entry(entry.clone())?;

        // Index for search
        self.search_engine.index_entry(&entry)?;

        // Update metrics
        self.update_storage_metrics();

        Ok(())
    }

    /// Retrieve metadata entry by ID
    pub fn get_metadata(&self, entry_id: &str) -> ContextResult<Option<ExecutionMetadataEntry>> {
        let store = self.metadata_store.read().unwrap();
        Ok(store.get_entry(entry_id))
    }

    /// Track lineage relationship between entities
    pub fn track_lineage(&self, parent_id: &str, child_id: &str) -> ContextResult<()> {
        self.lineage_tracker.add_lineage_edge(
            parent_id,
            child_id,
            lineage_tracking::LineageEdgeType::DataDependency
        )
    }

    /// Add provenance information for an entity
    pub fn add_provenance(&self, entity_id: &str, provenance: lineage_tracking::ProvenanceInfo) -> ContextResult<()> {
        self.lineage_tracker.add_provenance_record(entity_id, provenance)
    }

    /// Record execution event in history
    pub fn record_execution(&self, record: execution_history::ExecutionRecord) -> ContextResult<()> {
        let mut history = self.execution_history.write().unwrap();
        history.add_record(record)?;

        // Update metrics
        self.update_history_metrics();

        Ok(())
    }

    /// Get execution history with optional filtering
    pub fn get_execution_history(&self, limit: Option<usize>) -> ContextResult<Vec<execution_history::ExecutionRecord>> {
        let history = self.execution_history.read().unwrap();
        Ok(history.get_records(limit))
    }

    /// Search metadata with advanced query capabilities
    pub fn search_metadata(&self, query: &str) -> ContextResult<Vec<search_engine::SearchResult>> {
        let search_query = search_engine::SearchQuery {
            query: query.to_string(),
            query_type: QueryType::Match,
            fields: vec!["content".to_string()],
            filters: Vec::new(),
            sort: Vec::new(),
            pagination: search_engine::Pagination::default(),
            aggregations: Vec::new(),
        };

        self.search_engine.search(&search_query)
    }

    /// Perform advanced search with structured query
    pub fn advanced_search(&self, query: &search_engine::SearchQuery) -> ContextResult<Vec<search_engine::SearchResult>> {
        self.search_engine.search(query)
    }

    /// Add dependency relationship
    pub fn add_dependency(
        &self,
        source_id: &str,
        target_id: &str,
        dependency_type: dependency_tracker::DependencyType
    ) -> ContextResult<()> {
        self.dependency_tracker.add_dependency(source_id, target_id, dependency_type)
    }

    /// Resolve dependencies for a set of nodes
    pub fn resolve_dependencies(&self, node_ids: &[String]) -> ContextResult<dependency_tracker::ResolutionResult> {
        self.dependency_tracker.resolve_dependencies(node_ids)
    }

    /// Register metadata schema for validation
    pub fn register_schema(
        &self,
        schema_id: &str,
        schema: schema_validation::SchemaDefinition
    ) -> ContextResult<()> {
        self.schema_validator.register_schema(schema_id, schema)
    }

    /// Validate metadata against registered schemas
    pub fn validate_metadata(&self, metadata: &HashMap<String, String>) -> ContextResult<schema_validation::ValidationResult> {
        self.schema_validator.validate_metadata(metadata)
    }

    /// Get lineage graph for analysis
    pub fn get_lineage_graph(&self) -> ContextResult<lineage_tracking::LineageGraph> {
        self.lineage_tracker.get_lineage_graph()
    }

    /// Get comprehensive metrics from all subsystems
    pub fn get_comprehensive_metrics(&self) -> ContextResult<MetadataMetrics> {
        let mut metrics = self.metrics.lock().unwrap();

        // Collect metrics from all subsystems
        metrics.storage_metrics = self.metadata_store.read().unwrap().get_metrics();
        metrics.lineage_metrics = self.lineage_tracker.get_metrics();
        metrics.history_metrics = self.execution_history.read().unwrap().get_metrics();
        metrics.search_metrics = self.search_engine.get_metrics();
        metrics.schema_metrics = self.schema_validator.get_metrics();
        metrics.dependency_metrics = self.dependency_tracker.get_metrics();

        // Update system metrics
        metrics.system_metrics.last_updated = SystemTime::now();

        Ok(metrics.clone())
    }

    /// Export metadata context data for backup/migration
    pub fn export_context_data(&self) -> ContextResult<MetadataContextExport> {
        let store_data = self.metadata_store.read().unwrap().export_data()?;
        let lineage_data = self.lineage_tracker.export_lineage_data()?;
        let history_data = self.execution_history.read().unwrap().export_history_data()?;
        let schema_data = self.schema_validator.export_schema_data()?;
        let dependency_data = self.dependency_tracker.export_dependency_data()?;

        Ok(MetadataContextExport {
            context_id: self.context_id.clone(),
            exported_at: SystemTime::now(),
            storage_data: store_data,
            lineage_data,
            history_data,
            schema_data,
            dependency_data,
            config: self.config.clone(),
        })
    }

    /// Import metadata context data from backup/migration
    pub fn import_context_data(&self, export_data: MetadataContextExport) -> ContextResult<()> {
        // Import data to all subsystems
        self.metadata_store.write().unwrap().import_data(export_data.storage_data)?;
        self.lineage_tracker.import_lineage_data(export_data.lineage_data)?;
        self.execution_history.write().unwrap().import_history_data(export_data.history_data)?;
        self.schema_validator.import_schema_data(export_data.schema_data)?;
        self.dependency_tracker.import_dependency_data(export_data.dependency_data)?;

        // Rebuild search indices
        self.rebuild_search_indices()?;

        Ok(())
    }

    /// Rebuild search indices from current metadata
    pub fn rebuild_search_indices(&self) -> ContextResult<()> {
        let store = self.metadata_store.read().unwrap();
        let entries = store.get_all_entries();

        self.search_engine.clear_indices()?;

        for entry in entries {
            self.search_engine.index_entry(&entry)?;
        }

        Ok(())
    }

    /// Optimize metadata context performance
    pub fn optimize_performance(&self) -> ContextResult<()> {
        if !self.config.enable_optimizations {
            return Ok(());
        }

        // Optimize storage
        self.metadata_store.write().unwrap().optimize_storage()?;

        // Optimize search indices
        self.search_engine.optimize_indices()?;

        // Optimize lineage graph
        self.lineage_tracker.optimize_graph()?;

        // Clean up old history records
        self.execution_history.write().unwrap().cleanup_old_records()?;

        Ok(())
    }

    /// Get context health status
    pub fn get_health_status(&self) -> ContextResult<ContextHealthStatus> {
        let storage_health = self.metadata_store.read().unwrap().get_health_status();
        let lineage_health = self.lineage_tracker.get_health_status();
        let history_health = self.execution_history.read().unwrap().get_health_status();
        let search_health = self.search_engine.get_health_status();
        let schema_health = self.schema_validator.get_health_status();
        let dependency_health = self.dependency_tracker.get_health_status();

        Ok(ContextHealthStatus {
            overall_status: if storage_health.is_healthy && lineage_health.is_healthy &&
                              history_health.is_healthy && search_health.is_healthy &&
                              schema_health.is_healthy && dependency_health.is_healthy {
                HealthStatus::Healthy
            } else {
                HealthStatus::Degraded
            },
            storage_status: storage_health,
            lineage_status: lineage_health,
            history_status: history_health,
            search_status: search_health,
            schema_status: schema_health,
            dependency_status: dependency_health,
            last_checked: SystemTime::now(),
        })
    }

    // Helper methods for metrics updates
    fn update_storage_metrics(&self) {
        // Implementation would update storage-related metrics
    }

    fn update_history_metrics(&self) {
        // Implementation would update history-related metrics
    }
}

/// Export data structure for backup/migration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataContextExport {
    /// Context identifier
    pub context_id: String,

    /// Export timestamp
    pub exported_at: SystemTime,

    /// Storage system data
    pub storage_data: metadata_storage::MetadataStoreExport,

    /// Lineage tracking data
    pub lineage_data: lineage_tracking::LineageExport,

    /// Execution history data
    pub history_data: execution_history::HistoryExport,

    /// Schema definitions
    pub schema_data: schema_validation::SchemaExport,

    /// Dependency data
    pub dependency_data: dependency_tracker::DependencyExport,

    /// Configuration at time of export
    pub config: MetadataContextConfig,
}

/// Health status for the entire context
#[derive(Debug, Clone)]
pub struct ContextHealthStatus {
    /// Overall system health
    pub overall_status: HealthStatus,

    /// Storage subsystem health
    pub storage_status: ComponentHealthStatus,

    /// Lineage tracking health
    pub lineage_status: ComponentHealthStatus,

    /// History tracking health
    pub history_status: ComponentHealthStatus,

    /// Search engine health
    pub search_status: ComponentHealthStatus,

    /// Schema validation health
    pub schema_status: ComponentHealthStatus,

    /// Dependency tracking health
    pub dependency_status: ComponentHealthStatus,

    /// Last health check timestamp
    pub last_checked: SystemTime,
}

/// Component health status
#[derive(Debug, Clone)]
pub struct ComponentHealthStatus {
    /// Is component healthy
    pub is_healthy: bool,

    /// Health score (0.0 to 1.0)
    pub health_score: f64,

    /// Current status message
    pub status_message: String,

    /// Last error (if any)
    pub last_error: Option<String>,

    /// Performance metrics
    pub performance_metrics: HashMap<String, f64>,
}

/// Overall health status levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HealthStatus {
    /// All systems operating normally
    Healthy,

    /// Some systems experiencing issues but functional
    Degraded,

    /// Critical issues affecting functionality
    Unhealthy,

    /// System is down or not responsive
    Down,
}

// Implementation of ExecutionContextTrait for compatibility
impl ExecutionContextTrait for MetadataContext {
    fn get_context_id(&self) -> &str {
        &self.context_id
    }

    fn get_context_type(&self) -> ContextType {
        ContextType::Metadata
    }

    fn get_state(&self) -> ContextResult<ContextState> {
        Ok(self.state.read().unwrap().clone())
    }

    fn set_state(&self, state: ContextState) -> ContextResult<()> {
        *self.state.write().unwrap() = state;
        Ok(())
    }

    fn get_metadata(&self) -> ContextResult<ContextMetadata> {
        Ok(self.metadata.read().unwrap().clone())
    }

    fn set_property(&self, key: &str, value: String) -> ContextResult<()> {
        let mut metadata = self.metadata.write().unwrap();
        metadata.properties.insert(key.to_string(), value);
        Ok(())
    }

    fn get_property(&self, key: &str) -> ContextResult<Option<String>> {
        let metadata = self.metadata.read().unwrap();
        Ok(metadata.properties.get(key).cloned())
    }

    fn add_tag(&self, tag: String) -> ContextResult<()> {
        let mut metadata = self.metadata.write().unwrap();
        metadata.tags.insert(tag);
        Ok(())
    }

    fn has_tag(&self, tag: &str) -> ContextResult<bool> {
        let metadata = self.metadata.read().unwrap();
        Ok(metadata.tags.contains(tag))
    }
}

// Re-export all important types for backward compatibility and ease of use
pub use metadata_storage::{MetadataEntryType, MetadataEntryStatus, MetadataPayload};
pub use lineage_tracking::{LineageNodeType, LineageEdgeType, ProvenanceSourceType, ProvenanceAgentType};
pub use execution_history::{ExecutionStatus, ErrorCategory, ExecutionRecord};
pub use search_engine::{QueryType, FilterOperator, SearchResult};
pub use schema_validation::{FieldType, ValidationError, ValidationResult};
pub use dependency_tracker::{DependencyType, ResolutionResult, CircularDependencyError};

/// Module-level convenience functions for common operations
pub mod convenience {
    use super::*;

    /// Quick metadata storage with minimal setup
    pub fn quick_store_metadata(
        context: &MetadataContext,
        key: &str,
        value: &str,
        entry_type: MetadataEntryType,
    ) -> ContextResult<()> {
        let mut metadata = HashMap::new();
        metadata.insert(key.to_string(), value.to_string());
        context.store_metadata(&Uuid::new_v4().to_string(), entry_type, metadata)
    }

    /// Quick lineage tracking
    pub fn quick_track_lineage(
        context: &MetadataContext,
        parent: &str,
        child: &str,
    ) -> ContextResult<()> {
        context.track_lineage(parent, child)
    }

    /// Quick metadata search
    pub fn quick_search(
        context: &MetadataContext,
        query: &str,
    ) -> ContextResult<Vec<String>> {
        let results = context.search_metadata(query)?;
        Ok(results.into_iter().map(|r| r.entry_id).collect())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_context_creation() {
        let context = MetadataContext::new("test-context".to_string());
        assert!(context.is_ok());

        let ctx = context.unwrap();
        assert_eq!(ctx.get_context_id(), "test-context");
    }

    #[test]
    fn test_metadata_storage_and_retrieval() {
        let context = MetadataContext::new("test-storage".to_string()).unwrap();

        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), "value".to_string());

        let result = context.store_metadata("entry-1", MetadataEntryType::ExecutionStart, metadata);
        assert!(result.is_ok());

        let retrieved = context.get_metadata("entry-1").unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().entry_id, "entry-1");
    }

    #[test]
    fn test_lineage_tracking() {
        let context = MetadataContext::new("test-lineage".to_string()).unwrap();

        let result = context.track_lineage("parent-1", "child-1");
        assert!(result.is_ok());

        let graph = context.get_lineage_graph().unwrap();
        assert!(graph.nodes.len() >= 2);
        assert!(graph.edges.len() >= 1);
    }

    #[test]
    fn test_dependency_resolution() {
        let context = MetadataContext::new("test-deps".to_string()).unwrap();

        context.add_dependency("task-1", "task-2", dependency_tracker::DependencyType::Hard).unwrap();
        context.add_dependency("task-2", "task-3", dependency_tracker::DependencyType::Data).unwrap();

        let resolution = context.resolve_dependencies(&[
            "task-1".to_string(),
            "task-2".to_string(),
            "task-3".to_string()
        ]).unwrap();

        assert!(resolution.success);
        assert_eq!(resolution.execution_order.len(), 3);
    }

    #[test]
    fn test_health_status() {
        let context = MetadataContext::new("test-health".to_string()).unwrap();

        let health = context.get_health_status().unwrap();
        assert_eq!(health.overall_status, HealthStatus::Healthy);
    }

    #[test]
    fn test_metrics_collection() {
        let context = MetadataContext::new("test-metrics".to_string()).unwrap();

        let metrics = context.get_comprehensive_metrics().unwrap();
        assert!(metrics.system_metrics.last_updated <= SystemTime::now());
    }
}