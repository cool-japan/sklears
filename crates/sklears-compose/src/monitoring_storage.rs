//! Storage Backends and Data Management
//!
//! This module provides comprehensive storage backends and data management capabilities
//! for the execution monitoring framework. It includes multiple storage backends,
//! data lifecycle management, archival, compression, indexing, and storage health monitoring.

use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, SystemTime};
use std::path::{Path, PathBuf};
use std::fs;
use std::io::{self, Write, Read};
use sklears_core::error::{Result as SklResult, SklearsError};
use crate::monitoring_config::*;
use crate::monitoring_metrics::*;
use crate::monitoring_events::*;
use crate::monitoring_core::*;

/// Universal storage backend trait
///
/// Provides a unified interface for different storage backends including
/// in-memory, file-based, database, and distributed storage systems.
pub trait StorageBackend: Send + Sync {
    /// Store data with key-value semantics
    fn store(&mut self, key: &str, data: &[u8]) -> SklResult<()>;

    /// Retrieve data by key
    fn retrieve(&self, key: &str) -> SklResult<Option<Vec<u8>>>;

    /// Delete data by key
    fn delete(&mut self, key: &str) -> SklResult<bool>;

    /// List all keys matching a pattern
    fn list_keys(&self, pattern: &str) -> SklResult<Vec<String>>;

    /// Get storage statistics
    fn get_stats(&self) -> SklResult<StorageStats>;

    /// Perform health check
    fn health_check(&self) -> SklResult<StorageHealth>;

    /// Cleanup old data
    fn cleanup(&mut self, older_than: SystemTime) -> SklResult<usize>;

    /// Compact storage (if supported)
    fn compact(&mut self) -> SklResult<()> {
        Ok(()) // Default implementation does nothing
    }

    /// Flush pending writes
    fn flush(&mut self) -> SklResult<()> {
        Ok(()) // Default implementation does nothing
    }
}

/// Storage statistics
#[derive(Debug, Clone)]
pub struct StorageStats {
    /// Total stored items
    pub total_items: u64,

    /// Total storage size in bytes
    pub total_size: u64,

    /// Available storage space
    pub available_space: u64,

    /// Read operations count
    pub read_operations: u64,

    /// Write operations count
    pub write_operations: u64,

    /// Delete operations count
    pub delete_operations: u64,

    /// Average read latency
    pub avg_read_latency: Duration,

    /// Average write latency
    pub avg_write_latency: Duration,

    /// Error count
    pub error_count: u64,

    /// Last operation timestamp
    pub last_operation: SystemTime,
}

/// In-memory storage backend
#[derive(Debug)]
pub struct InMemoryStorage {
    /// Data storage
    data: Arc<RwLock<HashMap<String, Vec<u8>>>>,

    /// Configuration
    config: InMemoryStorageConfig,

    /// Statistics
    stats: Arc<RwLock<StorageStats>>,
}

/// In-memory storage configuration
#[derive(Debug, Clone)]
pub struct InMemoryStorageConfig {
    /// Maximum memory usage in bytes
    pub max_memory: u64,

    /// Maximum number of items
    pub max_items: usize,

    /// Eviction policy
    pub eviction_policy: EvictionPolicy,

    /// Enable compression
    pub compression: bool,
}

/// Eviction policies for memory management
#[derive(Debug, Clone)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// First In First Out
    FIFO,
    /// Random eviction
    Random,
    /// Time-based eviction
    TTL { default_ttl: Duration },
}

impl InMemoryStorage {
    /// Create new in-memory storage
    pub fn new(config: InMemoryStorageConfig) -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
            config,
            stats: Arc::new(RwLock::new(StorageStats::default())),
        }
    }

    /// Get current memory usage
    fn get_memory_usage(&self) -> u64 {
        let data = self.data.read().unwrap();
        data.values().map(|v| v.len() as u64).sum()
    }

    /// Evict items if necessary
    fn evict_if_needed(&mut self) -> SklResult<()> {
        let mut data = self.data.write().unwrap();

        // Check if eviction is needed
        if data.len() >= self.config.max_items || self.get_memory_usage() >= self.config.max_memory {
            match self.config.eviction_policy {
                EvictionPolicy::FIFO => {
                    // Remove first inserted item (simplified)
                    if let Some(key) = data.keys().next().cloned() {
                        data.remove(&key);
                    }
                }
                EvictionPolicy::Random => {
                    // Remove random item
                    use std::collections::hash_map::RandomState;
                    use std::hash::{BuildHasher, Hasher};
                    let s = RandomState::new();
                    let mut hasher = s.build_hasher();
                    hasher.write_usize(data.len());
                    let index = (hasher.finish() as usize) % data.len();
                    if let Some(key) = data.keys().nth(index).cloned() {
                        data.remove(&key);
                    }
                }
                _ => {
                    // Default to FIFO for unimplemented policies
                    if let Some(key) = data.keys().next().cloned() {
                        data.remove(&key);
                    }
                }
            }
        }

        Ok(())
    }
}

impl StorageBackend for InMemoryStorage {
    fn store(&mut self, key: &str, data: &[u8]) -> SklResult<()> {
        let start = SystemTime::now();

        // Evict if needed
        self.evict_if_needed()?;

        // Compress data if enabled
        let stored_data = if self.config.compression {
            compress_data(data)?
        } else {
            data.to_vec()
        };

        // Store data
        self.data.write().unwrap().insert(key.to_string(), stored_data);

        // Update statistics
        let mut stats = self.stats.write().unwrap();
        stats.write_operations += 1;
        stats.total_items += 1;
        stats.total_size += data.len() as u64;
        stats.last_operation = SystemTime::now();
        if let Ok(elapsed) = start.elapsed() {
            stats.avg_write_latency = (stats.avg_write_latency * (stats.write_operations - 1) as u32 + elapsed) / stats.write_operations as u32;
        }

        Ok(())
    }

    fn retrieve(&self, key: &str) -> SklResult<Option<Vec<u8>>> {
        let start = SystemTime::now();

        let data = self.data.read().unwrap();
        let result = if let Some(stored_data) = data.get(key) {
            // Decompress if needed
            let decompressed = if self.config.compression {
                decompress_data(stored_data)?
            } else {
                stored_data.clone()
            };
            Some(decompressed)
        } else {
            None
        };

        // Update statistics
        let mut stats = self.stats.write().unwrap();
        stats.read_operations += 1;
        stats.last_operation = SystemTime::now();
        if let Ok(elapsed) = start.elapsed() {
            stats.avg_read_latency = (stats.avg_read_latency * (stats.read_operations - 1) as u32 + elapsed) / stats.read_operations as u32;
        }

        Ok(result)
    }

    fn delete(&mut self, key: &str) -> SklResult<bool> {
        let mut data = self.data.write().unwrap();
        let removed = data.remove(key).is_some();

        if removed {
            let mut stats = self.stats.write().unwrap();
            stats.delete_operations += 1;
            stats.total_items = stats.total_items.saturating_sub(1);
            stats.last_operation = SystemTime::now();
        }

        Ok(removed)
    }

    fn list_keys(&self, pattern: &str) -> SklResult<Vec<String>> {
        let data = self.data.read().unwrap();
        let keys: Vec<String> = data.keys()
            .filter(|key| key.contains(pattern))
            .cloned()
            .collect();
        Ok(keys)
    }

    fn get_stats(&self) -> SklResult<StorageStats> {
        let mut stats = self.stats.read().unwrap().clone();
        stats.total_size = self.get_memory_usage();
        stats.available_space = self.config.max_memory.saturating_sub(stats.total_size);
        Ok(stats)
    }

    fn health_check(&self) -> SklResult<StorageHealth> {
        let stats = self.get_stats()?;
        let memory_usage_percent = (stats.total_size as f64 / self.config.max_memory as f64) * 100.0;

        let status = if memory_usage_percent < 70.0 {
            StorageStatus::Healthy
        } else if memory_usage_percent < 90.0 {
            StorageStatus::Warning
        } else {
            StorageStatus::Critical
        };

        Ok(StorageHealth {
            status,
            used_capacity: stats.total_size,
            total_capacity: self.config.max_memory,
            metric_count: stats.total_items,
            performance: StoragePerformance {
                avg_write_latency: stats.avg_write_latency,
                avg_read_latency: stats.avg_read_latency,
                write_throughput: if stats.avg_write_latency.as_millis() > 0 {
                    1000.0 / stats.avg_write_latency.as_millis() as f64
                } else {
                    0.0
                },
                read_throughput: if stats.avg_read_latency.as_millis() > 0 {
                    1000.0 / stats.avg_read_latency.as_millis() as f64
                } else {
                    0.0
                },
                error_rate: stats.error_count as f64 / (stats.read_operations + stats.write_operations + stats.delete_operations) as f64,
            },
        })
    }

    fn cleanup(&mut self, older_than: SystemTime) -> SklResult<usize> {
        // In-memory storage doesn't have timestamps by default
        // This is a simplified implementation
        Ok(0)
    }
}

/// File-based storage backend
#[derive(Debug)]
pub struct FileStorage {
    /// Base directory for storage
    base_path: PathBuf,

    /// Configuration
    config: FileStorageConfig,

    /// Index for fast lookups
    index: Arc<RwLock<HashMap<String, FileIndex>>>,

    /// Statistics
    stats: Arc<RwLock<StorageStats>>,
}

/// File storage configuration
#[derive(Debug, Clone)]
pub struct FileStorageConfig {
    /// Base directory path
    pub base_path: String,

    /// File format
    pub format: FileFormat,

    /// Compression settings
    pub compression: CompressionConfig,

    /// Indexing configuration
    pub indexing: IndexingConfig,

    /// Maximum file size
    pub max_file_size: u64,

    /// Directory structure
    pub directory_structure: DirectoryStructure,
}

/// Directory structure strategies
#[derive(Debug, Clone)]
pub enum DirectoryStructure {
    /// Flat structure - all files in base directory
    Flat,
    /// Hierarchical by date (year/month/day)
    DateHierarchy,
    /// Hierarchical by hash (first 2 chars of key)
    HashHierarchy,
    /// Custom structure
    Custom { pattern: String },
}

/// File index entry
#[derive(Debug, Clone)]
pub struct FileIndex {
    /// File path
    pub file_path: PathBuf,

    /// File size
    pub size: u64,

    /// Creation timestamp
    pub created_at: SystemTime,

    /// Last modified timestamp
    pub modified_at: SystemTime,

    /// Compression used
    pub compressed: bool,

    /// Checksum for integrity
    pub checksum: String,
}

impl FileStorage {
    /// Create new file storage
    pub fn new(config: FileStorageConfig) -> SklResult<Self> {
        let base_path = PathBuf::from(&config.base_path);

        // Create base directory if it doesn't exist
        if !base_path.exists() {
            fs::create_dir_all(&base_path)
                .map_err(|e| SklearsError::IoError(format!("Failed to create storage directory: {}", e)))?;
        }

        let storage = Self {
            base_path,
            config,
            index: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(StorageStats::default())),
        };

        // Build index from existing files
        storage.rebuild_index()?;

        Ok(storage)
    }

    /// Get file path for a key
    fn get_file_path(&self, key: &str) -> PathBuf {
        match self.config.directory_structure {
            DirectoryStructure::Flat => {
                self.base_path.join(format!("{}.dat", key))
            }
            DirectoryStructure::HashHierarchy => {
                let hash = format!("{:x}", md5::compute(key.as_bytes()));
                let dir1 = &hash[0..2];
                let dir2 = &hash[2..4];
                self.base_path.join(dir1).join(dir2).join(format!("{}.dat", key))
            }
            DirectoryStructure::DateHierarchy => {
                let now = SystemTime::now();
                let since_epoch = now.duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default();
                let secs = since_epoch.as_secs();
                let datetime = time::OffsetDateTime::from_unix_timestamp(secs as i64).unwrap_or_else(|_| time::OffsetDateTime::now_utc());
                let year = datetime.year();
                let month = datetime.month() as u8;
                let day = datetime.day();
                self.base_path.join(format!("{}", year)).join(format!("{:02}", month)).join(format!("{:02}", day)).join(format!("{}.dat", key))
            }
            DirectoryStructure::Custom { .. } => {
                // Simplified custom implementation
                self.base_path.join(format!("{}.dat", key))
            }
        }
    }

    /// Rebuild index from existing files
    fn rebuild_index(&self) -> SklResult<()> {
        let mut index = self.index.write().unwrap();
        index.clear();

        self.scan_directory(&self.base_path, &mut index)?;

        Ok(())
    }

    /// Recursively scan directory and build index
    fn scan_directory(&self, dir: &Path, index: &mut HashMap<String, FileIndex>) -> SklResult<()> {
        if !dir.exists() {
            return Ok(());
        }

        let entries = fs::read_dir(dir)
            .map_err(|e| SklearsError::IoError(format!("Failed to read directory: {}", e)))?;

        for entry in entries {
            let entry = entry.map_err(|e| SklearsError::IoError(e.to_string()))?;
            let path = entry.path();

            if path.is_dir() {
                self.scan_directory(&path, index)?;
            } else if path.extension().and_then(|s| s.to_str()) == Some("dat") {
                let metadata = fs::metadata(&path)
                    .map_err(|e| SklearsError::IoError(e.to_string()))?;

                if let Some(file_name) = path.file_stem().and_then(|s| s.to_str()) {
                    let created_at = metadata.created().unwrap_or(SystemTime::now());
                    let modified_at = metadata.modified().unwrap_or(SystemTime::now());

                    // Calculate checksum
                    let checksum = self.calculate_file_checksum(&path)?;

                    let file_index = FileIndex {
                        file_path: path.clone(),
                        size: metadata.len(),
                        created_at,
                        modified_at,
                        compressed: self.config.compression.enabled,
                        checksum,
                    };

                    index.insert(file_name.to_string(), file_index);
                }
            }
        }

        Ok(())
    }

    /// Calculate file checksum
    fn calculate_file_checksum(&self, path: &Path) -> SklResult<String> {
        let mut file = fs::File::open(path)
            .map_err(|e| SklearsError::IoError(e.to_string()))?;

        let mut hasher = md5::Context::new();
        let mut buffer = [0; 8192];

        loop {
            let bytes_read = file.read(&mut buffer)
                .map_err(|e| SklearsError::IoError(e.to_string()))?;

            if bytes_read == 0 {
                break;
            }

            hasher.consume(&buffer[..bytes_read]);
        }

        Ok(format!("{:x}", hasher.compute()))
    }

    /// Ensure directory exists for file path
    fn ensure_directory(&self, file_path: &Path) -> SklResult<()> {
        if let Some(parent) = file_path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent)
                    .map_err(|e| SklearsError::IoError(format!("Failed to create directory: {}", e)))?;
            }
        }
        Ok(())
    }
}

impl StorageBackend for FileStorage {
    fn store(&mut self, key: &str, data: &[u8]) -> SklResult<()> {
        let start = SystemTime::now();
        let file_path = self.get_file_path(key);

        // Ensure directory exists
        self.ensure_directory(&file_path)?;

        // Compress data if enabled
        let stored_data = if self.config.compression.enabled {
            compress_data(data)?
        } else {
            data.to_vec()
        };

        // Write to file
        let mut file = fs::File::create(&file_path)
            .map_err(|e| SklearsError::IoError(format!("Failed to create file: {}", e)))?;

        file.write_all(&stored_data)
            .map_err(|e| SklearsError::IoError(format!("Failed to write file: {}", e)))?;

        file.flush()
            .map_err(|e| SklearsError::IoError(format!("Failed to flush file: {}", e)))?;

        // Calculate checksum
        let checksum = self.calculate_file_checksum(&file_path)?;

        // Update index
        let file_index = FileIndex {
            file_path: file_path.clone(),
            size: stored_data.len() as u64,
            created_at: SystemTime::now(),
            modified_at: SystemTime::now(),
            compressed: self.config.compression.enabled,
            checksum,
        };

        self.index.write().unwrap().insert(key.to_string(), file_index);

        // Update statistics
        let mut stats = self.stats.write().unwrap();
        stats.write_operations += 1;
        stats.total_items += 1;
        stats.total_size += data.len() as u64;
        stats.last_operation = SystemTime::now();
        if let Ok(elapsed) = start.elapsed() {
            stats.avg_write_latency = (stats.avg_write_latency * (stats.write_operations - 1) as u32 + elapsed) / stats.write_operations as u32;
        }

        Ok(())
    }

    fn retrieve(&self, key: &str) -> SklResult<Option<Vec<u8>>> {
        let start = SystemTime::now();

        let index = self.index.read().unwrap();
        let result = if let Some(file_index) = index.get(key) {
            // Read file
            let mut file = fs::File::open(&file_index.file_path)
                .map_err(|e| SklearsError::IoError(format!("Failed to open file: {}", e)))?;

            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)
                .map_err(|e| SklearsError::IoError(format!("Failed to read file: {}", e)))?;

            // Verify checksum
            let current_checksum = format!("{:x}", md5::compute(&buffer));
            if current_checksum != file_index.checksum {
                return Err(SklearsError::DataCorruption("File checksum mismatch".to_string()));
            }

            // Decompress if needed
            let data = if file_index.compressed {
                decompress_data(&buffer)?
            } else {
                buffer
            };

            Some(data)
        } else {
            None
        };

        // Update statistics
        let mut stats = self.stats.write().unwrap();
        stats.read_operations += 1;
        stats.last_operation = SystemTime::now();
        if let Ok(elapsed) = start.elapsed() {
            stats.avg_read_latency = (stats.avg_read_latency * (stats.read_operations - 1) as u32 + elapsed) / stats.read_operations as u32;
        }

        Ok(result)
    }

    fn delete(&mut self, key: &str) -> SklResult<bool> {
        let mut index = self.index.write().unwrap();

        if let Some(file_index) = index.remove(key) {
            // Delete file
            if file_index.file_path.exists() {
                fs::remove_file(&file_index.file_path)
                    .map_err(|e| SklearsError::IoError(format!("Failed to delete file: {}", e)))?;
            }

            // Update statistics
            let mut stats = self.stats.write().unwrap();
            stats.delete_operations += 1;
            stats.total_items = stats.total_items.saturating_sub(1);
            stats.total_size = stats.total_size.saturating_sub(file_index.size);
            stats.last_operation = SystemTime::now();

            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn list_keys(&self, pattern: &str) -> SklResult<Vec<String>> {
        let index = self.index.read().unwrap();
        let keys: Vec<String> = index.keys()
            .filter(|key| key.contains(pattern))
            .cloned()
            .collect();
        Ok(keys)
    }

    fn get_stats(&self) -> SklResult<StorageStats> {
        let stats = self.stats.read().unwrap().clone();
        Ok(stats)
    }

    fn health_check(&self) -> SklResult<StorageHealth> {
        let stats = self.get_stats()?;

        // Check disk space
        let available_space = self.get_available_disk_space()?;
        let usage_percent = if available_space > 0 {
            ((stats.total_size as f64) / (stats.total_size + available_space) as f64) * 100.0
        } else {
            100.0
        };

        let status = if usage_percent < 70.0 {
            StorageStatus::Healthy
        } else if usage_percent < 90.0 {
            StorageStatus::Warning
        } else {
            StorageStatus::Critical
        };

        Ok(StorageHealth {
            status,
            used_capacity: stats.total_size,
            total_capacity: stats.total_size + available_space,
            metric_count: stats.total_items,
            performance: StoragePerformance {
                avg_write_latency: stats.avg_write_latency,
                avg_read_latency: stats.avg_read_latency,
                write_throughput: if stats.avg_write_latency.as_millis() > 0 {
                    1000.0 / stats.avg_write_latency.as_millis() as f64
                } else {
                    0.0
                },
                read_throughput: if stats.avg_read_latency.as_millis() > 0 {
                    1000.0 / stats.avg_read_latency.as_millis() as f64
                } else {
                    0.0
                },
                error_rate: stats.error_count as f64 / (stats.read_operations + stats.write_operations + stats.delete_operations) as f64,
            },
        })
    }

    fn cleanup(&mut self, older_than: SystemTime) -> SklResult<usize> {
        let mut index = self.index.write().unwrap();
        let mut removed_count = 0;
        let mut keys_to_remove = Vec::new();

        // Identify old files
        for (key, file_index) in index.iter() {
            if file_index.created_at < older_than {
                keys_to_remove.push(key.clone());
            }
        }

        // Remove old files
        for key in keys_to_remove {
            if let Some(file_index) = index.remove(&key) {
                if file_index.file_path.exists() {
                    fs::remove_file(&file_index.file_path)
                        .map_err(|e| SklearsError::IoError(format!("Failed to delete file: {}", e)))?;
                }
                removed_count += 1;
            }
        }

        // Update statistics
        let mut stats = self.stats.write().unwrap();
        stats.total_items = stats.total_items.saturating_sub(removed_count);

        Ok(removed_count as usize)
    }

    fn compact(&mut self) -> SklResult<()> {
        // Rebuild index to clean up any inconsistencies
        self.rebuild_index()?;

        // Remove empty directories
        self.remove_empty_directories(&self.base_path)?;

        Ok(())
    }
}

impl FileStorage {
    /// Get available disk space
    fn get_available_disk_space(&self) -> SklResult<u64> {
        // Simplified implementation - in a real system, would use platform-specific APIs
        Ok(1_000_000_000) // 1GB placeholder
    }

    /// Remove empty directories recursively
    fn remove_empty_directories(&self, dir: &Path) -> SklResult<()> {
        if !dir.exists() || !dir.is_dir() {
            return Ok(());
        }

        let entries = fs::read_dir(dir)
            .map_err(|e| SklearsError::IoError(e.to_string()))?;

        let mut has_files = false;
        let mut subdirs = Vec::new();

        for entry in entries {
            let entry = entry.map_err(|e| SklearsError::IoError(e.to_string()))?;
            let path = entry.path();

            if path.is_dir() {
                subdirs.push(path);
            } else {
                has_files = true;
            }
        }

        // Recursively clean subdirectories
        for subdir in subdirs {
            self.remove_empty_directories(&subdir)?;

            // Check if subdir is now empty
            if fs::read_dir(&subdir).map_or(false, |mut entries| entries.next().is_none()) {
                fs::remove_dir(&subdir)
                    .map_err(|e| SklearsError::IoError(e.to_string()))?;
            } else {
                has_files = true;
            }
        }

        Ok(())
    }
}

/// Storage manager for coordinating multiple storage backends
#[derive(Debug)]
pub struct StorageManager {
    /// Primary storage backend
    primary: Box<dyn StorageBackend>,

    /// Secondary storage backends for replication
    secondaries: Vec<Box<dyn StorageBackend>>,

    /// Configuration
    config: StorageManagerConfig,

    /// Manager statistics
    stats: Arc<RwLock<ManagerStats>>,
}

/// Storage manager configuration
#[derive(Debug, Clone)]
pub struct StorageManagerConfig {
    /// Replication factor
    pub replication_factor: usize,

    /// Consistency level
    pub consistency_level: ConsistencyLevel,

    /// Failure handling
    pub failure_handling: FailureHandling,

    /// Load balancing for reads
    pub read_load_balancing: bool,

    /// Enable caching
    pub enable_caching: bool,

    /// Cache configuration
    pub cache_config: CacheConfig,
}

/// Consistency levels
#[derive(Debug, Clone)]
pub enum ConsistencyLevel {
    /// Eventual consistency
    Eventual,
    /// Strong consistency
    Strong,
    /// Session consistency
    Session,
    /// Bounded staleness
    BoundedStaleness { max_staleness: Duration },
}

/// Failure handling strategies
#[derive(Debug, Clone)]
pub enum FailureHandling {
    /// Fail fast on any error
    FailFast,
    /// Continue on partial failures
    BestEffort,
    /// Retry failed operations
    Retry { max_attempts: usize, backoff: Duration },
    /// Circuit breaker pattern
    CircuitBreaker { failure_threshold: usize, timeout: Duration },
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Cache size
    pub size: usize,

    /// Cache TTL
    pub ttl: Duration,

    /// Cache eviction policy
    pub eviction_policy: EvictionPolicy,

    /// Enable write-through caching
    pub write_through: bool,
}

/// Manager statistics
#[derive(Debug, Clone)]
pub struct ManagerStats {
    /// Total operations
    pub total_operations: u64,

    /// Successful operations
    pub successful_operations: u64,

    /// Failed operations
    pub failed_operations: u64,

    /// Cache hits
    pub cache_hits: u64,

    /// Cache misses
    pub cache_misses: u64,

    /// Replication lag
    pub replication_lag: Duration,

    /// Backend health scores
    pub backend_health: HashMap<String, f64>,
}

impl StorageManager {
    /// Create new storage manager
    pub fn new(
        primary: Box<dyn StorageBackend>,
        secondaries: Vec<Box<dyn StorageBackend>>,
        config: StorageManagerConfig,
    ) -> Self {
        Self {
            primary,
            secondaries,
            config,
            stats: Arc::new(RwLock::new(ManagerStats::default())),
        }
    }

    /// Store data with replication
    pub fn store_replicated(&mut self, key: &str, data: &[u8]) -> SklResult<()> {
        let mut successful_writes = 0;
        let mut last_error = None;

        // Write to primary
        match self.primary.store(key, data) {
            Ok(()) => successful_writes += 1,
            Err(e) => last_error = Some(e),
        }

        // Write to secondaries based on replication factor
        let replication_count = self.config.replication_factor.min(self.secondaries.len());
        for secondary in self.secondaries.iter_mut().take(replication_count) {
            match secondary.store(key, data) {
                Ok(()) => successful_writes += 1,
                Err(e) => last_error = Some(e),
            }
        }

        // Check consistency requirements
        let required_writes = match self.config.consistency_level {
            ConsistencyLevel::Strong => self.config.replication_factor + 1, // Primary + all replicas
            ConsistencyLevel::Eventual => 1, // At least one write
            _ => (self.config.replication_factor + 1) / 2 + 1, // Majority
        };

        if successful_writes >= required_writes {
            self.stats.write().unwrap().successful_operations += 1;
            Ok(())
        } else {
            self.stats.write().unwrap().failed_operations += 1;
            Err(last_error.unwrap_or_else(|| SklearsError::StorageError("Insufficient replicas".to_string())))
        }
    }

    /// Retrieve data with load balancing
    pub fn retrieve_balanced(&self, key: &str) -> SklResult<Option<Vec<u8>>> {
        if self.config.read_load_balancing {
            // Try reading from different backends in round-robin fashion
            // This is a simplified implementation
            if let Ok(result) = self.primary.retrieve(key) {
                if result.is_some() {
                    return Ok(result);
                }
            }

            for secondary in &self.secondaries {
                if let Ok(result) = secondary.retrieve(key) {
                    if result.is_some() {
                        return Ok(result);
                    }
                }
            }

            Ok(None)
        } else {
            self.primary.retrieve(key)
        }
    }

    /// Get aggregated health status
    pub fn get_health_status(&self) -> SklResult<StorageHealth> {
        let primary_health = self.primary.health_check()?;

        let mut total_capacity = primary_health.total_capacity;
        let mut used_capacity = primary_health.used_capacity;
        let mut metric_count = primary_health.metric_count;

        // Aggregate secondary storage health
        for secondary in &self.secondaries {
            if let Ok(health) = secondary.health_check() {
                total_capacity += health.total_capacity;
                used_capacity += health.used_capacity;
                metric_count += health.metric_count;
            }
        }

        // Determine overall status
        let usage_percent = (used_capacity as f64 / total_capacity as f64) * 100.0;
        let status = if usage_percent < 70.0 {
            StorageStatus::Healthy
        } else if usage_percent < 90.0 {
            StorageStatus::Warning
        } else {
            StorageStatus::Critical
        };

        Ok(StorageHealth {
            status,
            used_capacity,
            total_capacity,
            metric_count,
            performance: primary_health.performance, // Use primary performance metrics
        })
    }

    /// Get manager statistics
    pub fn get_stats(&self) -> ManagerStats {
        self.stats.read().unwrap().clone()
    }
}

impl Default for ManagerStats {
    fn default() -> Self {
        Self {
            total_operations: 0,
            successful_operations: 0,
            failed_operations: 0,
            cache_hits: 0,
            cache_misses: 0,
            replication_lag: Duration::ZERO,
            backend_health: HashMap::new(),
        }
    }
}

impl Default for StorageStats {
    fn default() -> Self {
        Self {
            total_items: 0,
            total_size: 0,
            available_space: 0,
            read_operations: 0,
            write_operations: 0,
            delete_operations: 0,
            avg_read_latency: Duration::ZERO,
            avg_write_latency: Duration::ZERO,
            error_count: 0,
            last_operation: SystemTime::now(),
        }
    }
}

// Utility functions for compression

/// Compress data using the specified algorithm
fn compress_data(data: &[u8]) -> SklResult<Vec<u8>> {
    // Simplified compression - in a real implementation would use actual compression libraries
    Ok(data.to_vec())
}

/// Decompress data
fn decompress_data(data: &[u8]) -> SklResult<Vec<u8>> {
    // Simplified decompression - in a real implementation would use actual compression libraries
    Ok(data.to_vec())
}

/// Data retention manager
#[derive(Debug)]
pub struct DataRetentionManager {
    storage: Box<dyn StorageBackend>,

    policies: HashMap<String, RetentionPolicy>,

    archive_manager: Option<ArchiveManager>,

    config: DataRetentionConfig,
}

/// Archive manager for long-term storage
#[derive(Debug)]
pub struct ArchiveManager {
    /// Archive storage location
    archive_path: PathBuf,

    /// Archive configuration
    config: ArchiveConfig,
}

impl DataRetentionManager {
    /// Create new data retention manager
    pub fn new(
        storage: Box<dyn StorageBackend>,
        config: DataRetentionConfig,
    ) -> Self {
        let archive_manager = if !config.archive.location.is_empty() {
            Some(ArchiveManager::new(config.archive.clone()))
        } else {
            None
        };

        Self {
            storage,
            policies: config.policies.clone(),
            archive_manager,
            config,
        }
    }

    /// Apply retention policies
    pub fn apply_retention_policies(&mut self) -> SklResult<RetentionReport> {
        let mut report = RetentionReport::default();

        // Get all keys
        let all_keys = self.storage.list_keys("")?;

        for key in all_keys {
            // Determine which policy applies
            let policy = self.get_applicable_policy(&key);

            // Check if data should be archived
            if let Some(archive_after) = policy.archive_after {
                let cutoff = SystemTime::now() - archive_after;
                if self.should_archive(&key, cutoff)? {
                    if let Some(ref mut archive_manager) = self.archive_manager {
                        archive_manager.archive_data(&key, &mut self.storage)?;
                        report.archived_items += 1;
                    }
                }
            }

            // Check if data should be deleted
            let delete_cutoff = SystemTime::now() - policy.delete_after;
            if self.should_delete(&key, delete_cutoff)? {
                self.storage.delete(&key)?;
                report.deleted_items += 1;
            }
        }

        Ok(report)
    }

    /// Get applicable retention policy for a key
    fn get_applicable_policy(&self, key: &str) -> &RetentionPolicy {
        // Try to find specific policy
        for (pattern, policy) in &self.policies {
            if key.contains(pattern) {
                return policy;
            }
        }

        // Return default policy
        &RetentionPolicy {
            duration: self.config.default_retention,
            compress_after: None,
            archive_after: None,
            delete_after: self.config.default_retention,
        }
    }

    fn should_archive(&self, _key: &str, _cutoff: SystemTime) -> SklResult<bool> {
        // Simplified implementation
        Ok(false)
    }

    fn should_delete(&self, _key: &str, _cutoff: SystemTime) -> SklResult<bool> {
        // Simplified implementation
        Ok(false)
    }
}

impl ArchiveManager {
    /// Create new archive manager
    pub fn new(config: ArchiveConfig) -> Self {
        Self {
            archive_path: PathBuf::from(&config.location),
            config,
        }
    }

    /// Archive data
    pub fn archive_data(&mut self, key: &str, storage: &mut Box<dyn StorageBackend>) -> SklResult<()> {
        // Retrieve data from storage
        if let Some(data) = storage.retrieve(key)? {
            // Create archive file path
            let archive_file = self.archive_path.join(format!("{}.archive", key));

            // Ensure archive directory exists
            if let Some(parent) = archive_file.parent() {
                fs::create_dir_all(parent)
                    .map_err(|e| SklearsError::IoError(e.to_string()))?;
            }

            // Compress data if enabled
            let archived_data = if self.config.compression.enabled {
                compress_data(&data)?
            } else {
                data
            };

            // Write to archive
            fs::write(&archive_file, archived_data)
                .map_err(|e| SklearsError::IoError(e.to_string()))?;

            // Remove from primary storage
            storage.delete(key)?;
        }

        Ok(())
    }
}

/// Retention report
#[derive(Debug, Default)]
pub struct RetentionReport {
    /// Number of items archived
    pub archived_items: usize,

    /// Number of items deleted
    pub deleted_items: usize,

    /// Number of items compressed
    pub compressed_items: usize,

    /// Space reclaimed
    pub space_reclaimed: u64,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_in_memory_storage() {
        let config = InMemoryStorageConfig {
            max_memory: 1024 * 1024, // 1MB
            max_items: 1000,
            eviction_policy: EvictionPolicy::FIFO,
            compression: false,
        };

        let mut storage = InMemoryStorage::new(config);

        // Test store and retrieve
        let key = "test_key";
        let data = b"test_data";

        storage.store(key, data).unwrap();
        let retrieved = storage.retrieve(key).unwrap();
        assert_eq!(retrieved, Some(data.to_vec()));

        // Test delete
        assert!(storage.delete(key).unwrap());
        let retrieved = storage.retrieve(key).unwrap();
        assert_eq!(retrieved, None);

        // Test stats
        let stats = storage.get_stats().unwrap();
        assert_eq!(stats.write_operations, 1);
        assert_eq!(stats.read_operations, 2);
        assert_eq!(stats.delete_operations, 1);
    }

    #[test]
    fn test_file_storage() {
        let temp_dir = TempDir::new().unwrap();
        let config = FileStorageConfig {
            base_path: temp_dir.path().to_string_lossy().to_string(),
            format: FileFormat::Binary,
            compression: CompressionConfig::default(),
            indexing: IndexingConfig::default(),
            max_file_size: 1024 * 1024,
            directory_structure: DirectoryStructure::Flat,
        };

        let mut storage = FileStorage::new(config).unwrap();

        // Test store and retrieve
        let key = "test_key";
        let data = b"test_data";

        storage.store(key, data).unwrap();
        let retrieved = storage.retrieve(key).unwrap();
        assert_eq!(retrieved, Some(data.to_vec()));

        // Test list keys
        let keys = storage.list_keys("test").unwrap();
        assert!(keys.contains(&key.to_string()));

        // Test delete
        assert!(storage.delete(key).unwrap());
        let retrieved = storage.retrieve(key).unwrap();
        assert_eq!(retrieved, None);
    }

    #[test]
    fn test_storage_manager() {
        let primary_config = InMemoryStorageConfig {
            max_memory: 1024 * 1024,
            max_items: 1000,
            eviction_policy: EvictionPolicy::FIFO,
            compression: false,
        };

        let secondary_config = InMemoryStorageConfig {
            max_memory: 1024 * 1024,
            max_items: 1000,
            eviction_policy: EvictionPolicy::FIFO,
            compression: false,
        };

        let primary: Box<dyn StorageBackend> = Box::new(InMemoryStorage::new(primary_config));
        let secondary: Box<dyn StorageBackend> = Box::new(InMemoryStorage::new(secondary_config));

        let manager_config = StorageManagerConfig {
            replication_factor: 1,
            consistency_level: ConsistencyLevel::Eventual,
            failure_handling: FailureHandling::BestEffort,
            read_load_balancing: true,
            enable_caching: false,
            cache_config: CacheConfig {
                size: 100,
                ttl: Duration::from_secs(300),
                eviction_policy: EvictionPolicy::LRU,
                write_through: false,
            },
        };

        let mut manager = StorageManager::new(primary, vec![secondary], manager_config);

        // Test replicated store
        let key = "test_key";
        let data = b"test_data";

        manager.store_replicated(key, data).unwrap();
        let retrieved = manager.retrieve_balanced(key).unwrap();
        assert_eq!(retrieved, Some(data.to_vec()));

        // Test health check
        let health = manager.get_health_status().unwrap();
        assert!(matches!(health.status, StorageStatus::Healthy));
    }

    #[test]
    fn test_data_retention() {
        let storage_config = InMemoryStorageConfig {
            max_memory: 1024 * 1024,
            max_items: 1000,
            eviction_policy: EvictionPolicy::FIFO,
            compression: false,
        };

        let storage: Box<dyn StorageBackend> = Box::new(InMemoryStorage::new(storage_config));

        let retention_config = DataRetentionConfig {
            default_retention: Duration::from_secs(3600),
            policies: HashMap::new(),
            archive: ArchiveConfig {
                location: "/tmp/archive".to_string(),
                format: ArchiveFormat::Tar,
                compression: CompressionConfig::default(),
            },
            cleanup: CleanupConfig {
                interval: Duration::from_secs(3600),
                strategies: vec![CleanupStrategy::Age],
                thresholds: CleanupThresholds::default(),
            },
        };

        let mut retention_manager = DataRetentionManager::new(storage, retention_config);
        let report = retention_manager.apply_retention_policies().unwrap();

        // With no data, nothing should be archived or deleted
        assert_eq!(report.archived_items, 0);
        assert_eq!(report.deleted_items, 0);
    }
}