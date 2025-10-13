//! Memory-mapped storage system for explanation results
//!
//! This module provides persistent storage capabilities for explanation data using
//! memory-mapped files for efficient access and optional compression support.

use crate::types::*;
use crate::SklResult;
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::SklearsError;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Memory-mapped explanation storage for large datasets
pub struct MemoryMappedStorage {
    /// Base directory for storage files
    base_dir: std::path::PathBuf,
    /// Memory map configurations
    config: MemoryMapConfig,
    /// Currently mapped files
    mapped_files: Arc<Mutex<HashMap<String, MappedFile>>>,
}

/// Configuration for memory-mapped storage
#[derive(Clone, Debug)]
pub struct MemoryMapConfig {
    /// Enable memory mapping (vs regular file I/O)
    pub enable_mmap: bool,
    /// Page size for mapping
    pub page_size: usize,
    /// Maximum file size before splitting
    pub max_file_size: usize,
    /// File compression
    pub compress: bool,
    /// Prefault pages on mapping
    pub prefault: bool,
}

/// Memory-mapped file handle
struct MappedFile {
    /// File handle
    file: File,
    /// Memory map (if enabled)
    #[cfg(feature = "mmap")]
    mmap: Option<memmap2::MmapMut>,
    /// File size
    size: usize,
    /// Is read-only
    read_only: bool,
}

/// Memory-mapped explanation result storage
#[derive(Debug, Clone)]
pub struct MappedExplanationResult {
    /// Storage identifier
    pub storage_id: String,
    /// Feature importance data location
    pub feature_importance_offset: usize,
    /// feature_importance_len
    pub feature_importance_len: usize,
    /// SHAP values data location
    pub shap_values_offset: usize,
    /// shap_values_len
    pub shap_values_len: usize,
    /// Metadata
    pub n_features: usize,
    /// n_samples
    pub n_samples: usize,
    /// Creation timestamp
    pub timestamp: u64,
}

/// Storage statistics
#[derive(Debug, Clone)]
pub struct StorageStats {
    /// Total number of files
    pub total_files: usize,
    /// Total size in bytes
    pub total_size: usize,
    /// Number of memory-mapped files
    pub memory_mapped_files: usize,
}

impl Default for MemoryMapConfig {
    fn default() -> Self {
        Self {
            enable_mmap: true,
            page_size: 4096,
            max_file_size: 1024 * 1024 * 1024, // 1GB
            compress: false,
            prefault: false,
        }
    }
}

impl MemoryMappedStorage {
    /// Create a new memory-mapped storage system
    pub fn new<P: AsRef<Path>>(base_dir: P, config: MemoryMapConfig) -> SklResult<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();

        // Create base directory if it doesn't exist
        if !base_dir.exists() {
            std::fs::create_dir_all(&base_dir).map_err(|e| {
                SklearsError::Other(format!("Failed to create storage directory: {}", e))
            })?;
        }

        Ok(Self {
            base_dir,
            config,
            mapped_files: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Store explanation results to memory-mapped storage
    pub fn store_explanation_results(
        &self,
        storage_id: &str,
        feature_importance: &Array1<Float>,
        shap_values: Option<&Array2<Float>>,
    ) -> SklResult<MappedExplanationResult> {
        let file_path = self.base_dir.join(format!("{}.expdata", storage_id));

        // Create or open file
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .truncate(true)
            .open(&file_path)
            .map_err(|e| SklearsError::Other(format!("Failed to create storage file: {}", e)))?;

        // Write feature importance data
        let feature_importance_offset = 0;
        let feature_importance_bytes = self.serialize_array1(feature_importance)?;
        file.write_all(&feature_importance_bytes).map_err(|e| {
            SklearsError::Other(format!("Failed to write feature importance: {}", e))
        })?;

        // Write SHAP values data if provided
        let (shap_values_offset, shap_values_len) = if let Some(shap_data) = shap_values {
            let offset = feature_importance_bytes.len();
            let shap_bytes = self.serialize_array2(shap_data)?;
            file.write_all(&shap_bytes)
                .map_err(|e| SklearsError::Other(format!("Failed to write SHAP values: {}", e)))?;
            (offset, shap_bytes.len())
        } else {
            (0, 0)
        };

        file.sync_all()
            .map_err(|e| SklearsError::Other(format!("Failed to sync file: {}", e)))?;

        // Create memory map if enabled
        #[cfg(feature = "mmap")]
        let mmap = if self.config.enable_mmap {
            let mmap = unsafe {
                memmap2::MmapOptions::new().map_mut(&file).map_err(|e| {
                    SklearsError::Other(format!("Failed to create memory map: {}", e))
                })?
            };
            Some(mmap)
        } else {
            None
        };

        let file_size = file
            .metadata()
            .map_err(|e| SklearsError::Other(format!("Failed to get file metadata: {}", e)))?
            .len() as usize;

        // Store mapped file handle
        let mapped_file = MappedFile {
            file,
            #[cfg(feature = "mmap")]
            mmap,
            size: file_size,
            read_only: false,
        };

        {
            let mut mapped_files = self.mapped_files.lock().unwrap();
            mapped_files.insert(storage_id.to_string(), mapped_file);
        }

        Ok(MappedExplanationResult {
            storage_id: storage_id.to_string(),
            feature_importance_offset,
            feature_importance_len: feature_importance_bytes.len(),
            shap_values_offset,
            shap_values_len,
            n_features: feature_importance.len(),
            n_samples: shap_values.map(|s| s.nrows()).unwrap_or(0),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        })
    }

    /// Load feature importance from memory-mapped storage
    pub fn load_feature_importance(
        &self,
        result: &MappedExplanationResult,
    ) -> SklResult<Array1<Float>> {
        let mapped_files = self.mapped_files.lock().unwrap();
        let mapped_file = mapped_files
            .get(&result.storage_id)
            .ok_or_else(|| SklearsError::Other("Storage file not found".to_string()))?;

        #[cfg(feature = "mmap")]
        if let Some(mmap) = &mapped_file.mmap {
            // Use memory map for direct access
            let start = result.feature_importance_offset;
            let end = start + result.feature_importance_len;
            let data_slice = &mmap[start..end];
            return self.deserialize_array1(data_slice, result.n_features);
        }

        // Fallback to regular file I/O
        drop(mapped_files); // Release lock
        self.load_feature_importance_file(&result.storage_id, result)
    }

    /// Load SHAP values from memory-mapped storage
    pub fn load_shap_values(&self, result: &MappedExplanationResult) -> SklResult<Array2<Float>> {
        if result.shap_values_len == 0 {
            return Err(SklearsError::Other("No SHAP values stored".to_string()));
        }

        let mapped_files = self.mapped_files.lock().unwrap();
        let mapped_file = mapped_files
            .get(&result.storage_id)
            .ok_or_else(|| SklearsError::Other("Storage file not found".to_string()))?;

        #[cfg(feature = "mmap")]
        if let Some(mmap) = &mapped_file.mmap {
            // Use memory map for direct access
            let start = result.shap_values_offset;
            let end = start + result.shap_values_len;
            let data_slice = &mmap[start..end];
            return self.deserialize_array2(data_slice, result.n_samples, result.n_features);
        }

        // Fallback to regular file I/O
        drop(mapped_files); // Release lock
        self.load_shap_values_file(&result.storage_id, result)
    }

    /// Serialize Array1 to bytes
    fn serialize_array1(&self, array: &Array1<Float>) -> SklResult<Vec<u8>> {
        let mut bytes = Vec::new();

        // Write dimensions
        bytes.extend_from_slice(&array.len().to_le_bytes());

        // Write data
        for &value in array.iter() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        if self.config.compress {
            // Simple compression using RLE for zeros (placeholder for real compression)
            self.compress_bytes(&bytes)
        } else {
            Ok(bytes)
        }
    }

    /// Serialize Array2 to bytes
    fn serialize_array2(&self, array: &Array2<Float>) -> SklResult<Vec<u8>> {
        let mut bytes = Vec::new();

        // Write dimensions
        bytes.extend_from_slice(&array.nrows().to_le_bytes());
        bytes.extend_from_slice(&array.ncols().to_le_bytes());

        // Write data in row-major order
        for row in array.rows() {
            for &value in row.iter() {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }

        if self.config.compress {
            self.compress_bytes(&bytes)
        } else {
            Ok(bytes)
        }
    }

    /// Deserialize bytes to Array1
    fn deserialize_array1(&self, data: &[u8], expected_len: usize) -> SklResult<Array1<Float>> {
        // Decompress if needed
        let data_to_use = if self.config.compress {
            self.decompress_bytes(data)?
        } else {
            data.to_vec()
        };
        let data = &data_to_use;

        if data.len() < 8 {
            return Err(SklearsError::Other("Invalid data format".to_string()));
        }

        // Read dimensions
        let len = usize::from_le_bytes(data[0..8].try_into().unwrap());
        if len != expected_len {
            return Err(SklearsError::Other("Dimension mismatch".to_string()));
        }

        // Read data
        let mut array_data = Vec::with_capacity(len);
        let float_size = std::mem::size_of::<Float>();

        for i in 0..len {
            let start = 8 + i * float_size;
            let end = start + float_size;

            if end > data.len() {
                return Err(SklearsError::Other("Insufficient data".to_string()));
            }

            let value = if float_size == 8 {
                f64::from_le_bytes(data[start..end].try_into().unwrap()) as Float
            } else {
                f32::from_le_bytes(data[start..end].try_into().unwrap()) as Float
            };
            array_data.push(value);
        }

        Ok(Array1::from_vec(array_data))
    }

    /// Deserialize bytes to Array2
    fn deserialize_array2(
        &self,
        data: &[u8],
        expected_rows: usize,
        expected_cols: usize,
    ) -> SklResult<Array2<Float>> {
        // Decompress if needed
        let data_to_use = if self.config.compress {
            self.decompress_bytes(data)?
        } else {
            data.to_vec()
        };
        let data = &data_to_use;

        if data.len() < 16 {
            return Err(SklearsError::Other("Invalid data format".to_string()));
        }

        // Read dimensions
        let nrows = usize::from_le_bytes(data[0..8].try_into().unwrap());
        let ncols = usize::from_le_bytes(data[8..16].try_into().unwrap());

        if nrows != expected_rows || ncols != expected_cols {
            return Err(SklearsError::Other("Dimension mismatch".to_string()));
        }

        // Read data
        let mut array_data = Vec::with_capacity(nrows * ncols);
        let float_size = std::mem::size_of::<Float>();

        for i in 0..(nrows * ncols) {
            let start = 16 + i * float_size;
            let end = start + float_size;

            if end > data.len() {
                return Err(SklearsError::Other("Insufficient data".to_string()));
            }

            let value = if float_size == 8 {
                f64::from_le_bytes(data[start..end].try_into().unwrap()) as Float
            } else {
                f32::from_le_bytes(data[start..end].try_into().unwrap()) as Float
            };
            array_data.push(value);
        }

        Array2::from_shape_vec((nrows, ncols), array_data)
            .map_err(|e| SklearsError::Other(format!("Failed to create array: {}", e)))
    }

    /// Simple compression placeholder (could be replaced with real compression)
    fn compress_bytes(&self, data: &[u8]) -> SklResult<Vec<u8>> {
        // For now, just return the original data
        // In a real implementation, you might use libraries like flate2, lz4, etc.
        Ok(data.to_vec())
    }

    /// Simple decompression placeholder
    fn decompress_bytes(&self, data: &[u8]) -> SklResult<Vec<u8>> {
        // For now, just return the original data
        Ok(data.to_vec())
    }

    /// Load feature importance using regular file I/O
    fn load_feature_importance_file(
        &self,
        storage_id: &str,
        result: &MappedExplanationResult,
    ) -> SklResult<Array1<Float>> {
        let file_path = self.base_dir.join(format!("{}.expdata", storage_id));
        let mut file = File::open(&file_path)
            .map_err(|e| SklearsError::Other(format!("Failed to open storage file: {}", e)))?;

        file.seek(SeekFrom::Start(result.feature_importance_offset as u64))
            .map_err(|e| SklearsError::Other(format!("Failed to seek: {}", e)))?;

        let mut buffer = vec![0u8; result.feature_importance_len];
        file.read_exact(&mut buffer)
            .map_err(|e| SklearsError::Other(format!("Failed to read data: {}", e)))?;

        self.deserialize_array1(&buffer, result.n_features)
    }

    /// Load SHAP values using regular file I/O
    fn load_shap_values_file(
        &self,
        storage_id: &str,
        result: &MappedExplanationResult,
    ) -> SklResult<Array2<Float>> {
        let file_path = self.base_dir.join(format!("{}.expdata", storage_id));
        let mut file = File::open(&file_path)
            .map_err(|e| SklearsError::Other(format!("Failed to open storage file: {}", e)))?;

        file.seek(SeekFrom::Start(result.shap_values_offset as u64))
            .map_err(|e| SklearsError::Other(format!("Failed to seek: {}", e)))?;

        let mut buffer = vec![0u8; result.shap_values_len];
        file.read_exact(&mut buffer)
            .map_err(|e| SklearsError::Other(format!("Failed to read data: {}", e)))?;

        self.deserialize_array2(&buffer, result.n_samples, result.n_features)
    }

    /// Remove storage file
    pub fn remove_storage(&self, storage_id: &str) -> SklResult<()> {
        // Remove from mapped files
        {
            let mut mapped_files = self.mapped_files.lock().unwrap();
            mapped_files.remove(storage_id);
        }

        // Remove file
        let file_path = self.base_dir.join(format!("{}.expdata", storage_id));
        if file_path.exists() {
            std::fs::remove_file(&file_path).map_err(|e| {
                SklearsError::Other(format!("Failed to remove storage file: {}", e))
            })?;
        }

        Ok(())
    }

    /// Get storage statistics
    pub fn get_storage_stats(&self) -> StorageStats {
        let mapped_files = self.mapped_files.lock().unwrap();
        let total_files = mapped_files.len();
        let total_size: usize = mapped_files.values().map(|f| f.size).sum();

        StorageStats {
            total_files,
            total_size,
            memory_mapped_files: mapped_files
                .values()
                .filter(|f| {
                    #[cfg(feature = "mmap")]
                    return f.mmap.is_some();
                    #[cfg(not(feature = "mmap"))]
                    return false;
                })
                .count(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;
    use tempfile::TempDir;

    #[test]
    fn test_memory_mapped_storage_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = MemoryMapConfig::default();

        let storage = MemoryMappedStorage::new(temp_dir.path(), config);
        assert!(storage.is_ok());
    }

    #[test]
    fn test_memory_mapped_storage_store_and_load() {
        let temp_dir = TempDir::new().unwrap();
        let config = MemoryMapConfig::default();
        let storage = MemoryMappedStorage::new(temp_dir.path(), config).unwrap();

        // Create test data
        let feature_importance = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let shap_values = array![[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]];

        // Store data
        let result = storage.store_explanation_results(
            "test_storage",
            &feature_importance,
            Some(&shap_values),
        );
        assert!(result.is_ok());
        let result = result.unwrap();

        // Load feature importance
        let loaded_importance = storage.load_feature_importance(&result);
        assert!(loaded_importance.is_ok());
        let loaded_importance = loaded_importance.unwrap();

        assert_eq!(loaded_importance.len(), feature_importance.len());
        for (a, b) in loaded_importance.iter().zip(feature_importance.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-6);
        }

        // Load SHAP values
        let loaded_shap = storage.load_shap_values(&result);
        assert!(loaded_shap.is_ok());
        let loaded_shap = loaded_shap.unwrap();

        assert_eq!(loaded_shap.shape(), shap_values.shape());
        for (a, b) in loaded_shap.iter().zip(shap_values.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_memory_mapped_storage_stats() {
        let temp_dir = TempDir::new().unwrap();
        let config = MemoryMapConfig::default();
        let storage = MemoryMappedStorage::new(temp_dir.path(), config).unwrap();

        // Initially no files
        let stats = storage.get_storage_stats();
        assert_eq!(stats.total_files, 0);
        assert_eq!(stats.total_size, 0);

        // Store some data
        let feature_importance = array![1.0, 2.0, 3.0];
        let _result = storage
            .store_explanation_results("test_stats", &feature_importance, None)
            .unwrap();

        // Check stats
        let stats = storage.get_storage_stats();
        assert_eq!(stats.total_files, 1);
        assert!(stats.total_size > 0);
    }

    #[test]
    fn test_memory_mapped_storage_remove() {
        let temp_dir = TempDir::new().unwrap();
        let config = MemoryMapConfig::default();
        let storage = MemoryMappedStorage::new(temp_dir.path(), config).unwrap();

        // Store data
        let feature_importance = array![1.0, 2.0, 3.0];
        let _result = storage
            .store_explanation_results("test_remove", &feature_importance, None)
            .unwrap();

        // Check it exists
        let stats = storage.get_storage_stats();
        assert_eq!(stats.total_files, 1);

        // Remove it
        let result = storage.remove_storage("test_remove");
        assert!(result.is_ok());

        // Check it's gone
        let stats = storage.get_storage_stats();
        assert_eq!(stats.total_files, 0);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let config = MemoryMapConfig::default();
        let storage = MemoryMappedStorage::new(temp_dir.path(), config).unwrap();

        // Test Array1 serialization
        let arr1 = array![1.5, 2.5, 3.5];
        let bytes1 = storage.serialize_array1(&arr1).unwrap();
        let recovered1 = storage.deserialize_array1(&bytes1, arr1.len()).unwrap();

        for (a, b) in arr1.iter().zip(recovered1.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-10);
        }

        // Test Array2 serialization
        let arr2 = array![[1.0, 2.0], [3.0, 4.0]];
        let bytes2 = storage.serialize_array2(&arr2).unwrap();
        let recovered2 = storage
            .deserialize_array2(&bytes2, arr2.nrows(), arr2.ncols())
            .unwrap();

        assert_eq!(arr2.shape(), recovered2.shape());
        for (a, b) in arr2.iter().zip(recovered2.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-10);
        }
    }
}
