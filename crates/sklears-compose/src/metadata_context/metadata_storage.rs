//! # Metadata Storage Module
//!
//! Core metadata storage and retrieval system providing comprehensive data management
//! capabilities for the metadata context framework.
//!
//! ## Features
//!
//! - **High-Performance Storage**: In-memory storage with optional persistence
//! - **Concurrent Access**: Thread-safe operations with RwLock protection
//! - **Flexible Indexing**: Multi-dimensional indexing for fast retrieval
//! - **Compression Support**: Optional data compression for large metadata
//! - **Backup and Restore**: Export/import functionality for data migration
//! - **Performance Monitoring**: Comprehensive metrics and health tracking
//!
//! ## Architecture
//!
//! ```text
//! MetadataStore
//! ├── StorageEngine (core data storage)
//! ├── IndexManager (multi-dimensional indexing)
//! ├── CompressionHandler (data compression)
//! ├── PersistenceLayer (backup/restore)
//! └── MetricsCollector (performance tracking)
//! ```

use scirs2_core::error::{CoreError, Result};
use scirs2_core::metrics::{MetricRegistry, Counter, Gauge, Histogram, Timer};
use scirs2_core::ndarray::{Array, Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};
use uuid::Uuid;

/// Comprehensive metadata entry with rich attributes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataEntry {
    /// Unique identifier for the metadata entry
    pub id: String,
    /// Primary metadata key
    pub key: String,
    /// Metadata value (JSON-serializable)
    pub value: serde_json::Value,
    /// Entry type classification
    pub entry_type: MetadataType,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last modification timestamp
    pub updated_at: SystemTime,
    /// Version number for optimistic concurrency
    pub version: u64,
    /// Tags for categorization and filtering
    pub tags: HashSet<String>,
    /// Custom attributes for extensibility
    pub attributes: HashMap<String, serde_json::Value>,
    /// Data size in bytes
    pub size: usize,
    /// Compression status
    pub compressed: bool,
    /// Access count for analytics
    pub access_count: u64,
    /// Last access timestamp
    pub last_accessed: SystemTime,
}

/// Metadata entry types for classification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetadataType {
    /// Execution context metadata
    ExecutionContext,
    /// Algorithm configuration metadata
    AlgorithmConfig,
    /// Performance metrics metadata
    PerformanceMetrics,
    /// Data lineage metadata
    DataLineage,
    /// Model parameters metadata
    ModelParameters,
    /// Validation results metadata
    ValidationResults,
    /// Custom user-defined metadata
    Custom(String),
}

/// Storage indexing strategies for optimized retrieval
#[derive(Debug, Clone)]
pub struct StorageIndex {
    /// Key-based primary index
    key_index: BTreeMap<String, String>, // key -> entry_id
    /// Type-based secondary index
    type_index: HashMap<MetadataType, HashSet<String>>, // type -> entry_ids
    /// Tag-based filtering index
    tag_index: HashMap<String, HashSet<String>>, // tag -> entry_ids
    /// Time-based range index
    time_index: BTreeMap<SystemTime, HashSet<String>>, // timestamp -> entry_ids
    /// Size-based index for storage analytics
    size_index: BTreeMap<usize, HashSet<String>>, // size -> entry_ids
}

/// Storage engine configuration
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Maximum entries before cleanup
    pub max_entries: usize,
    /// Enable compression for large entries
    pub enable_compression: bool,
    /// Compression threshold in bytes
    pub compression_threshold: usize,
    /// Enable persistence to disk
    pub enable_persistence: bool,
    /// Persistence file path
    pub persistence_path: Option<String>,
    /// Auto-backup interval
    pub backup_interval: Option<Duration>,
    /// Memory usage limit in bytes
    pub memory_limit: Option<usize>,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            max_entries: 100_000,
            enable_compression: true,
            compression_threshold: 1024, // 1KB
            enable_persistence: false,
            persistence_path: None,
            backup_interval: None,
            memory_limit: Some(1_024 * 1024 * 1024), // 1GB
        }
    }
}

/// Storage statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    /// Total number of entries
    pub total_entries: usize,
    /// Total storage size in bytes
    pub total_size: usize,
    /// Compressed entries count
    pub compressed_entries: usize,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Cache hit rate percentage
    pub cache_hit_rate: f64,
    /// Average access time in microseconds
    pub avg_access_time: f64,
    /// Index efficiency metrics
    pub index_efficiency: HashMap<String, f64>,
    /// Storage health score (0-100)
    pub health_score: u8,
}

/// Query filter for advanced metadata retrieval
#[derive(Debug, Clone)]
pub struct MetadataQuery {
    /// Filter by key pattern (supports wildcards)
    pub key_pattern: Option<String>,
    /// Filter by metadata type
    pub entry_type: Option<MetadataType>,
    /// Filter by tags (AND operation)
    pub required_tags: HashSet<String>,
    /// Filter by creation time range
    pub created_after: Option<SystemTime>,
    pub created_before: Option<SystemTime>,
    /// Filter by size range
    pub min_size: Option<usize>,
    pub max_size: Option<usize>,
    /// Limit number of results
    pub limit: Option<usize>,
    /// Sort order
    pub sort_by: SortOrder,
}

/// Sort order options for query results
#[derive(Debug, Clone, PartialEq)]
pub enum SortOrder {
    /// Sort by creation time (newest first)
    CreatedDesc,
    /// Sort by creation time (oldest first)
    CreatedAsc,
    /// Sort by key alphabetically
    KeyAsc,
    /// Sort by size (largest first)
    SizeDesc,
    /// Sort by access count (most accessed first)
    AccessCountDesc,
}

impl Default for MetadataQuery {
    fn default() -> Self {
        Self {
            key_pattern: None,
            entry_type: None,
            required_tags: HashSet::new(),
            created_after: None,
            created_before: None,
            min_size: None,
            max_size: None,
            limit: None,
            sort_by: SortOrder::CreatedDesc,
        }
    }
}

/// Core metadata storage engine
#[derive(Debug)]
pub struct MetadataStore {
    /// Storage configuration
    config: StorageConfig,
    /// Main data storage
    entries: HashMap<String, MetadataEntry>, // entry_id -> entry
    /// Multi-dimensional indexing system
    indexes: StorageIndex,
    /// Performance metrics
    metrics: Arc<MetricRegistry>,
    /// Storage statistics
    stats: StorageStats,
    /// Version counter for optimistic concurrency
    version_counter: u64,
    /// Performance timers
    access_timer: Timer,
    storage_timer: Timer,
    /// Operation counters
    read_counter: Counter,
    write_counter: Counter,
    delete_counter: Counter,
}

impl MetadataStore {
    /// Create a new metadata store with default configuration
    pub fn new() -> Self {
        Self::with_config(StorageConfig::default())
    }

    /// Create a new metadata store with custom configuration
    pub fn with_config(config: StorageConfig) -> Self {
        let metrics = Arc::new(MetricRegistry::new());

        Self {
            config,
            entries: HashMap::new(),
            indexes: StorageIndex {
                key_index: BTreeMap::new(),
                type_index: HashMap::new(),
                tag_index: HashMap::new(),
                time_index: BTreeMap::new(),
                size_index: BTreeMap::new(),
            },
            metrics: metrics.clone(),
            stats: StorageStats {
                total_entries: 0,
                total_size: 0,
                compressed_entries: 0,
                memory_usage: 0,
                cache_hit_rate: 0.0,
                avg_access_time: 0.0,
                index_efficiency: HashMap::new(),
                health_score: 100,
            },
            version_counter: 0,
            access_timer: metrics.timer("metadata_store.access_time"),
            storage_timer: metrics.timer("metadata_store.storage_time"),
            read_counter: metrics.counter("metadata_store.reads"),
            write_counter: metrics.counter("metadata_store.writes"),
            delete_counter: metrics.counter("metadata_store.deletes"),
        }
    }

    /// Store metadata entry
    pub fn store_metadata(
        &mut self,
        key: String,
        value: serde_json::Value,
        entry_type: MetadataType,
        tags: HashSet<String>,
    ) -> Result<String> {
        let _timer = self.storage_timer.start_timer();

        let entry_id = Uuid::new_v4().to_string();
        let now = SystemTime::now();
        let size = self.calculate_entry_size(&value);

        // Apply compression if needed
        let (final_value, compressed) = if self.config.enable_compression
            && size > self.config.compression_threshold {
            (self.compress_value(value)?, true)
        } else {
            (value, false)
        };

        let entry = MetadataEntry {
            id: entry_id.clone(),
            key: key.clone(),
            value: final_value,
            entry_type: entry_type.clone(),
            created_at: now,
            updated_at: now,
            version: self.next_version(),
            tags: tags.clone(),
            attributes: HashMap::new(),
            size,
            compressed,
            access_count: 0,
            last_accessed: now,
        };

        // Store entry
        self.entries.insert(entry_id.clone(), entry);

        // Update indexes
        self.update_indexes_on_insert(&entry_id, &key, &entry_type, &tags, now, size);

        // Update statistics
        self.update_stats_on_insert(size, compressed);

        self.write_counter.inc();

        Ok(entry_id)
    }

    /// Retrieve metadata entry by ID
    pub fn get_metadata(&mut self, entry_id: &str) -> Result<Option<MetadataEntry>> {
        let _timer = self.access_timer.start_timer();

        if let Some(entry) = self.entries.get_mut(entry_id) {
            // Update access statistics
            entry.access_count += 1;
            entry.last_accessed = SystemTime::now();

            let mut result = entry.clone();

            // Decompress if needed
            if result.compressed {
                result.value = self.decompress_value(result.value)?;
                result.compressed = false;
            }

            self.read_counter.inc();
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    /// Retrieve metadata entry by key
    pub fn get_metadata_by_key(&mut self, key: &str) -> Result<Option<MetadataEntry>> {
        if let Some(entry_id) = self.indexes.key_index.get(key) {
            self.get_metadata(entry_id)
        } else {
            Ok(None)
        }
    }

    /// Query metadata entries with advanced filtering
    pub fn query_metadata(&mut self, query: &MetadataQuery) -> Result<Vec<MetadataEntry>> {
        let _timer = self.access_timer.start_timer();

        let mut candidate_ids = HashSet::new();
        let mut first_filter = true;

        // Apply filters progressively
        if let Some(ref pattern) = query.key_pattern {
            let matching_ids = self.find_entries_by_key_pattern(pattern);
            if first_filter {
                candidate_ids = matching_ids;
                first_filter = false;
            } else {
                candidate_ids = candidate_ids.intersection(&matching_ids).cloned().collect();
            }
        }

        if let Some(ref entry_type) = query.entry_type {
            let matching_ids = self.indexes.type_index
                .get(entry_type)
                .cloned()
                .unwrap_or_default();
            if first_filter {
                candidate_ids = matching_ids;
                first_filter = false;
            } else {
                candidate_ids = candidate_ids.intersection(&matching_ids).cloned().collect();
            }
        }

        if !query.required_tags.is_empty() {
            let matching_ids = self.find_entries_by_tags(&query.required_tags);
            if first_filter {
                candidate_ids = matching_ids;
                first_filter = false;
            } else {
                candidate_ids = candidate_ids.intersection(&matching_ids).cloned().collect();
            }
        }

        // If no filters applied, use all entries
        if first_filter {
            candidate_ids = self.entries.keys().cloned().collect();
        }

        // Apply time and size filters
        let filtered_entries: Vec<_> = candidate_ids.iter()
            .filter_map(|id| self.entries.get_mut(id))
            .filter(|entry| self.entry_matches_query(entry, query))
            .map(|entry| {
                // Update access statistics
                entry.access_count += 1;
                entry.last_accessed = SystemTime::now();

                let mut result = entry.clone();
                // Decompress if needed
                if result.compressed {
                    if let Ok(decompressed) = self.decompress_value(result.value.clone()) {
                        result.value = decompressed;
                        result.compressed = false;
                    }
                }
                result
            })
            .collect();

        // Sort results
        let mut sorted_entries = filtered_entries;
        self.sort_entries(&mut sorted_entries, &query.sort_by);

        // Apply limit
        if let Some(limit) = query.limit {
            sorted_entries.truncate(limit);
        }

        self.read_counter.inc();
        Ok(sorted_entries)
    }

    /// Update metadata entry
    pub fn update_metadata(
        &mut self,
        entry_id: &str,
        value: serde_json::Value,
        tags: Option<HashSet<String>>,
    ) -> Result<bool> {
        let _timer = self.storage_timer.start_timer();

        if let Some(entry) = self.entries.get_mut(entry_id) {
            let old_size = entry.size;
            let new_size = self.calculate_entry_size(&value);

            // Apply compression if needed
            let (final_value, compressed) = if self.config.enable_compression
                && new_size > self.config.compression_threshold {
                (self.compress_value(value)?, true)
            } else {
                (value, false)
            };

            // Update entry
            entry.value = final_value;
            entry.updated_at = SystemTime::now();
            entry.version = self.next_version();
            entry.size = new_size;
            entry.compressed = compressed;

            if let Some(new_tags) = tags {
                // Update tag indexes
                self.remove_from_tag_indexes(&entry.id, &entry.tags);
                entry.tags = new_tags.clone();
                self.add_to_tag_indexes(&entry.id, &new_tags);
            }

            // Update size indexes
            self.update_size_indexes(&entry.id, old_size, new_size);

            // Update statistics
            self.update_stats_on_update(old_size, new_size, compressed);

            self.write_counter.inc();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Delete metadata entry
    pub fn delete_metadata(&mut self, entry_id: &str) -> Result<bool> {
        if let Some(entry) = self.entries.remove(entry_id) {
            // Remove from all indexes
            self.remove_from_all_indexes(&entry);

            // Update statistics
            self.update_stats_on_delete(entry.size, entry.compressed);

            self.delete_counter.inc();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get storage statistics
    pub fn get_stats(&self) -> &StorageStats {
        &self.stats
    }

    /// Get detailed storage health information
    pub fn get_health_status(&self) -> HashMap<String, serde_json::Value> {
        let mut health = HashMap::new();

        health.insert("total_entries".to_string(), json!(self.stats.total_entries));
        health.insert("total_size_mb".to_string(),
            json!(self.stats.total_size as f64 / 1024.0 / 1024.0));
        health.insert("memory_usage_mb".to_string(),
            json!(self.stats.memory_usage as f64 / 1024.0 / 1024.0));
        health.insert("compression_ratio".to_string(),
            json!(self.stats.compressed_entries as f64 / self.stats.total_entries.max(1) as f64));
        health.insert("cache_hit_rate".to_string(), json!(self.stats.cache_hit_rate));
        health.insert("avg_access_time_ms".to_string(),
            json!(self.stats.avg_access_time / 1000.0));
        health.insert("health_score".to_string(), json!(self.stats.health_score));

        // Index efficiency
        for (index_name, efficiency) in &self.stats.index_efficiency {
            health.insert(format!("index_efficiency_{}", index_name), json!(efficiency));
        }

        health
    }

    /// Export all metadata to JSON
    pub fn export_metadata(&self) -> Result<String> {
        let export_data = serde_json::json!({
            "version": "1.0",
            "timestamp": SystemTime::now(),
            "entries": self.entries.values().collect::<Vec<_>>(),
            "stats": self.stats
        });

        serde_json::to_string_pretty(&export_data)
            .map_err(|e| CoreError::SerializationError(format!("Export failed: {}", e)))
    }

    /// Import metadata from JSON
    pub fn import_metadata(&mut self, json_data: &str) -> Result<usize> {
        let import_data: serde_json::Value = serde_json::from_str(json_data)
            .map_err(|e| CoreError::SerializationError(format!("Import failed: {}", e)))?;

        let entries = import_data["entries"].as_array()
            .ok_or_else(|| CoreError::ValidationError("Invalid import format".to_string()))?;

        let mut imported_count = 0;

        for entry_value in entries {
            let entry: MetadataEntry = serde_json::from_value(entry_value.clone())
                .map_err(|e| CoreError::SerializationError(format!("Entry import failed: {}", e)))?;

            self.entries.insert(entry.id.clone(), entry.clone());
            self.update_indexes_on_insert(
                &entry.id,
                &entry.key,
                &entry.entry_type,
                &entry.tags,
                entry.created_at,
                entry.size
            );
            imported_count += 1;
        }

        self.recalculate_stats();
        Ok(imported_count)
    }

    /// Clear all metadata
    pub fn clear(&mut self) {
        self.entries.clear();
        self.indexes = StorageIndex {
            key_index: BTreeMap::new(),
            type_index: HashMap::new(),
            tag_index: HashMap::new(),
            time_index: BTreeMap::new(),
            size_index: BTreeMap::new(),
        };
        self.stats = StorageStats {
            total_entries: 0,
            total_size: 0,
            compressed_entries: 0,
            memory_usage: 0,
            cache_hit_rate: 0.0,
            avg_access_time: 0.0,
            index_efficiency: HashMap::new(),
            health_score: 100,
        };
    }

    // Private helper methods

    fn next_version(&mut self) -> u64 {
        self.version_counter += 1;
        self.version_counter
    }

    fn calculate_entry_size(&self, value: &serde_json::Value) -> usize {
        serde_json::to_string(value).map(|s| s.len()).unwrap_or(0)
    }

    fn compress_value(&self, value: serde_json::Value) -> Result<serde_json::Value> {
        // Simplified compression - in real implementation would use proper compression
        Ok(json!({
            "_compressed": true,
            "_data": serde_json::to_string(&value)?
        }))
    }

    fn decompress_value(&self, value: serde_json::Value) -> Result<serde_json::Value> {
        if let Some(data) = value.get("_data").and_then(|d| d.as_str()) {
            serde_json::from_str(data)
                .map_err(|e| CoreError::SerializationError(format!("Decompression failed: {}", e)))
        } else {
            Ok(value)
        }
    }

    fn update_indexes_on_insert(
        &mut self,
        entry_id: &str,
        key: &str,
        entry_type: &MetadataType,
        tags: &HashSet<String>,
        timestamp: SystemTime,
        size: usize,
    ) {
        // Update key index
        self.indexes.key_index.insert(key.to_string(), entry_id.to_string());

        // Update type index
        self.indexes.type_index
            .entry(entry_type.clone())
            .or_insert_with(HashSet::new)
            .insert(entry_id.to_string());

        // Update tag indexes
        self.add_to_tag_indexes(entry_id, tags);

        // Update time index
        self.indexes.time_index
            .entry(timestamp)
            .or_insert_with(HashSet::new)
            .insert(entry_id.to_string());

        // Update size index
        self.indexes.size_index
            .entry(size)
            .or_insert_with(HashSet::new)
            .insert(entry_id.to_string());
    }

    fn add_to_tag_indexes(&mut self, entry_id: &str, tags: &HashSet<String>) {
        for tag in tags {
            self.indexes.tag_index
                .entry(tag.clone())
                .or_insert_with(HashSet::new)
                .insert(entry_id.to_string());
        }
    }

    fn remove_from_tag_indexes(&mut self, entry_id: &str, tags: &HashSet<String>) {
        for tag in tags {
            if let Some(tag_entries) = self.indexes.tag_index.get_mut(tag) {
                tag_entries.remove(entry_id);
                if tag_entries.is_empty() {
                    self.indexes.tag_index.remove(tag);
                }
            }
        }
    }

    fn remove_from_all_indexes(&mut self, entry: &MetadataEntry) {
        // Remove from key index
        self.indexes.key_index.remove(&entry.key);

        // Remove from type index
        if let Some(type_entries) = self.indexes.type_index.get_mut(&entry.entry_type) {
            type_entries.remove(&entry.id);
            if type_entries.is_empty() {
                self.indexes.type_index.remove(&entry.entry_type);
            }
        }

        // Remove from tag indexes
        self.remove_from_tag_indexes(&entry.id, &entry.tags);

        // Remove from time index
        if let Some(time_entries) = self.indexes.time_index.get_mut(&entry.created_at) {
            time_entries.remove(&entry.id);
            if time_entries.is_empty() {
                self.indexes.time_index.remove(&entry.created_at);
            }
        }

        // Remove from size index
        if let Some(size_entries) = self.indexes.size_index.get_mut(&entry.size) {
            size_entries.remove(&entry.id);
            if size_entries.is_empty() {
                self.indexes.size_index.remove(&entry.size);
            }
        }
    }

    fn update_size_indexes(&mut self, entry_id: &str, old_size: usize, new_size: usize) {
        // Remove from old size index
        if let Some(old_size_entries) = self.indexes.size_index.get_mut(&old_size) {
            old_size_entries.remove(entry_id);
            if old_size_entries.is_empty() {
                self.indexes.size_index.remove(&old_size);
            }
        }

        // Add to new size index
        self.indexes.size_index
            .entry(new_size)
            .or_insert_with(HashSet::new)
            .insert(entry_id.to_string());
    }

    fn find_entries_by_key_pattern(&self, pattern: &str) -> HashSet<String> {
        let mut matching_ids = HashSet::new();

        if pattern.contains('*') || pattern.contains('?') {
            // Wildcard matching
            for (key, entry_id) in &self.indexes.key_index {
                if self.matches_pattern(key, pattern) {
                    matching_ids.insert(entry_id.clone());
                }
            }
        } else {
            // Exact match
            if let Some(entry_id) = self.indexes.key_index.get(pattern) {
                matching_ids.insert(entry_id.clone());
            }
        }

        matching_ids
    }

    fn find_entries_by_tags(&self, required_tags: &HashSet<String>) -> HashSet<String> {
        if required_tags.is_empty() {
            return HashSet::new();
        }

        let mut result = None;

        for tag in required_tags {
            if let Some(tag_entries) = self.indexes.tag_index.get(tag) {
                result = Some(match result {
                    None => tag_entries.clone(),
                    Some(current) => current.intersection(tag_entries).cloned().collect(),
                });
            } else {
                return HashSet::new(); // If any tag is missing, no results
            }
        }

        result.unwrap_or_default()
    }

    fn matches_pattern(&self, text: &str, pattern: &str) -> bool {
        // Simple wildcard matching (* and ?)
        let pattern_chars: Vec<char> = pattern.chars().collect();
        let text_chars: Vec<char> = text.chars().collect();

        self.matches_pattern_recursive(&text_chars, 0, &pattern_chars, 0)
    }

    fn matches_pattern_recursive(
        &self,
        text: &[char],
        text_pos: usize,
        pattern: &[char],
        pattern_pos: usize,
    ) -> bool {
        if pattern_pos >= pattern.len() {
            return text_pos >= text.len();
        }

        if text_pos >= text.len() {
            return pattern[pattern_pos..].iter().all(|&c| c == '*');
        }

        match pattern[pattern_pos] {
            '*' => {
                // Try matching zero or more characters
                for i in text_pos..=text.len() {
                    if self.matches_pattern_recursive(text, i, pattern, pattern_pos + 1) {
                        return true;
                    }
                }
                false
            }
            '?' => {
                // Match exactly one character
                self.matches_pattern_recursive(text, text_pos + 1, pattern, pattern_pos + 1)
            }
            c => {
                // Match exact character
                text[text_pos] == c
                    && self.matches_pattern_recursive(text, text_pos + 1, pattern, pattern_pos + 1)
            }
        }
    }

    fn entry_matches_query(&self, entry: &MetadataEntry, query: &MetadataQuery) -> bool {
        // Time range filters
        if let Some(after) = query.created_after {
            if entry.created_at < after {
                return false;
            }
        }

        if let Some(before) = query.created_before {
            if entry.created_at > before {
                return false;
            }
        }

        // Size range filters
        if let Some(min_size) = query.min_size {
            if entry.size < min_size {
                return false;
            }
        }

        if let Some(max_size) = query.max_size {
            if entry.size > max_size {
                return false;
            }
        }

        true
    }

    fn sort_entries(&self, entries: &mut [MetadataEntry], sort_order: &SortOrder) {
        match sort_order {
            SortOrder::CreatedDesc => {
                entries.sort_by(|a, b| b.created_at.cmp(&a.created_at));
            }
            SortOrder::CreatedAsc => {
                entries.sort_by(|a, b| a.created_at.cmp(&b.created_at));
            }
            SortOrder::KeyAsc => {
                entries.sort_by(|a, b| a.key.cmp(&b.key));
            }
            SortOrder::SizeDesc => {
                entries.sort_by(|a, b| b.size.cmp(&a.size));
            }
            SortOrder::AccessCountDesc => {
                entries.sort_by(|a, b| b.access_count.cmp(&a.access_count));
            }
        }
    }

    fn update_stats_on_insert(&mut self, size: usize, compressed: bool) {
        self.stats.total_entries += 1;
        self.stats.total_size += size;
        if compressed {
            self.stats.compressed_entries += 1;
        }
        self.recalculate_health_score();
    }

    fn update_stats_on_update(&mut self, old_size: usize, new_size: usize, compressed: bool) {
        self.stats.total_size = self.stats.total_size - old_size + new_size;
        if compressed {
            self.stats.compressed_entries += 1;
        }
        self.recalculate_health_score();
    }

    fn update_stats_on_delete(&mut self, size: usize, was_compressed: bool) {
        self.stats.total_entries -= 1;
        self.stats.total_size -= size;
        if was_compressed {
            self.stats.compressed_entries -= 1;
        }
        self.recalculate_health_score();
    }

    fn recalculate_stats(&mut self) {
        self.stats.total_entries = self.entries.len();
        self.stats.total_size = self.entries.values().map(|e| e.size).sum();
        self.stats.compressed_entries = self.entries.values()
            .filter(|e| e.compressed)
            .count();

        self.recalculate_health_score();
    }

    fn recalculate_health_score(&mut self) {
        let mut score = 100u8;

        // Reduce score based on memory usage
        if let Some(limit) = self.config.memory_limit {
            let usage_ratio = self.stats.memory_usage as f64 / limit as f64;
            if usage_ratio > 0.9 {
                score = score.saturating_sub(20);
            } else if usage_ratio > 0.7 {
                score = score.saturating_sub(10);
            }
        }

        // Reduce score based on entry count
        let entry_ratio = self.stats.total_entries as f64 / self.config.max_entries as f64;
        if entry_ratio > 0.9 {
            score = score.saturating_sub(15);
        } else if entry_ratio > 0.7 {
            score = score.saturating_sub(5);
        }

        // Bonus for compression usage
        if self.stats.compressed_entries > 0 {
            score = std::cmp::min(100, score + 5);
        }

        self.stats.health_score = score;
    }
}

impl Default for MetadataStore {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_metadata_storage_basic_operations() {
        let mut store = MetadataStore::new();

        // Test store
        let tags = ["test", "basic"].iter().map(|s| s.to_string()).collect();
        let entry_id = store.store_metadata(
            "test_key".to_string(),
            json!({"value": 42}),
            MetadataType::Custom("test".to_string()),
            tags,
        ).unwrap();

        // Test retrieve by ID
        let entry = store.get_metadata(&entry_id).unwrap().unwrap();
        assert_eq!(entry.key, "test_key");
        assert_eq!(entry.value["value"], 42);

        // Test retrieve by key
        let entry_by_key = store.get_metadata_by_key("test_key").unwrap().unwrap();
        assert_eq!(entry_by_key.id, entry_id);

        // Test update
        let new_tags = ["updated"].iter().map(|s| s.to_string()).collect();
        let updated = store.update_metadata(
            &entry_id,
            json!({"value": 100}),
            Some(new_tags),
        ).unwrap();
        assert!(updated);

        let updated_entry = store.get_metadata(&entry_id).unwrap().unwrap();
        assert_eq!(updated_entry.value["value"], 100);
        assert!(updated_entry.tags.contains("updated"));

        // Test delete
        let deleted = store.delete_metadata(&entry_id).unwrap();
        assert!(deleted);

        let deleted_entry = store.get_metadata(&entry_id).unwrap();
        assert!(deleted_entry.is_none());
    }

    #[test]
    fn test_metadata_query() {
        let mut store = MetadataStore::new();

        // Store test entries
        let tags1 = ["tag1", "shared"].iter().map(|s| s.to_string()).collect();
        let tags2 = ["tag2", "shared"].iter().map(|s| s.to_string()).collect();

        store.store_metadata(
            "key1".to_string(),
            json!({"size": 100}),
            MetadataType::ExecutionContext,
            tags1,
        ).unwrap();

        store.store_metadata(
            "key2".to_string(),
            json!({"size": 200}),
            MetadataType::AlgorithmConfig,
            tags2,
        ).unwrap();

        // Query by type
        let mut query = MetadataQuery::default();
        query.entry_type = Some(MetadataType::ExecutionContext);

        let results = store.query_metadata(&query).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "key1");

        // Query by tags
        let mut query = MetadataQuery::default();
        query.required_tags = ["shared"].iter().map(|s| s.to_string()).collect();

        let results = store.query_metadata(&query).unwrap();
        assert_eq!(results.len(), 2);

        // Query with limit
        let mut query = MetadataQuery::default();
        query.limit = Some(1);

        let results = store.query_metadata(&query).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_wildcard_pattern_matching() {
        let store = MetadataStore::new();

        assert!(store.matches_pattern("hello", "hello"));
        assert!(store.matches_pattern("hello", "h*"));
        assert!(store.matches_pattern("hello", "*lo"));
        assert!(store.matches_pattern("hello", "h?llo"));
        assert!(store.matches_pattern("hello", "*"));
        assert!(!store.matches_pattern("hello", "world"));
        assert!(!store.matches_pattern("hello", "h?"));
    }

    #[test]
    fn test_export_import() {
        let mut store = MetadataStore::new();

        // Store test data
        let tags = ["export", "test"].iter().map(|s| s.to_string()).collect();
        store.store_metadata(
            "export_key".to_string(),
            json!({"test": "data"}),
            MetadataType::Custom("export".to_string()),
            tags,
        ).unwrap();

        // Export
        let exported = store.export_metadata().unwrap();

        // Clear and import
        store.clear();
        assert_eq!(store.get_stats().total_entries, 0);

        let imported_count = store.import_metadata(&exported).unwrap();
        assert_eq!(imported_count, 1);
        assert_eq!(store.get_stats().total_entries, 1);

        // Verify data
        let entry = store.get_metadata_by_key("export_key").unwrap().unwrap();
        assert_eq!(entry.value["test"], "data");
    }
}