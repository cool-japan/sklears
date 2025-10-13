//! Cloud storage utilities for machine learning data processing
//!
//! This module provides unified interfaces for working with cloud storage services
//! including AWS S3, Google Cloud Storage, and Azure Blob Storage.

use crate::{UtilsError, UtilsResult};
use std::collections::HashMap;
use std::fmt;

/// Cloud storage configuration
#[derive(Debug, Clone)]
pub struct CloudStorageConfig {
    pub provider: CloudProvider,
    pub endpoint: Option<String>,
    pub region: Option<String>,
    pub access_key: Option<String>,
    pub secret_key: Option<String>,
    pub bucket: String,
    pub timeout_seconds: Option<u64>,
    pub use_ssl: bool,
    pub custom_headers: HashMap<String, String>,
}

/// Supported cloud storage providers
#[derive(Debug, Clone, PartialEq)]
pub enum CloudProvider {
    AWS,
    GoogleCloud,
    Azure,
    MinIO,
    Custom(String),
}

impl fmt::Display for CloudProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CloudProvider::AWS => write!(f, "aws"),
            CloudProvider::GoogleCloud => write!(f, "gcp"),
            CloudProvider::Azure => write!(f, "azure"),
            CloudProvider::MinIO => write!(f, "minio"),
            CloudProvider::Custom(name) => write!(f, "{name}"),
        }
    }
}

impl Default for CloudStorageConfig {
    fn default() -> Self {
        Self {
            provider: CloudProvider::AWS,
            endpoint: None,
            region: Some("us-east-1".to_string()),
            access_key: None,
            secret_key: None,
            bucket: String::new(),
            timeout_seconds: Some(30),
            use_ssl: true,
            custom_headers: HashMap::new(),
        }
    }
}

/// Cloud storage client trait
pub trait CloudStorageClient {
    /// Upload data to cloud storage
    fn upload(&self, key: &str, data: &[u8]) -> UtilsResult<String>;

    /// Download data from cloud storage
    fn download(&self, key: &str) -> UtilsResult<Vec<u8>>;

    /// Delete object from cloud storage
    fn delete(&self, key: &str) -> UtilsResult<()>;

    /// List objects with prefix
    fn list_objects(&self, prefix: &str) -> UtilsResult<Vec<String>>;

    /// Check if object exists
    fn exists(&self, key: &str) -> UtilsResult<bool>;

    /// Get object metadata
    fn get_metadata(&self, key: &str) -> UtilsResult<ObjectMetadata>;

    /// Upload file from local path
    fn upload_file(&self, key: &str, local_path: &str) -> UtilsResult<String>;

    /// Download file to local path
    fn download_file(&self, key: &str, local_path: &str) -> UtilsResult<()>;
}

/// Object metadata
#[derive(Debug, Clone)]
pub struct ObjectMetadata {
    pub size: u64,
    pub etag: Option<String>,
    pub content_type: Option<String>,
    pub last_modified: Option<String>,
    pub custom_metadata: HashMap<String, String>,
}

/// Mock cloud storage client for testing
pub struct MockCloudStorageClient {
    storage: std::sync::Arc<std::sync::Mutex<HashMap<String, Vec<u8>>>>,
    metadata: std::sync::Arc<std::sync::Mutex<HashMap<String, ObjectMetadata>>>,
}

impl Default for MockCloudStorageClient {
    fn default() -> Self {
        Self::new()
    }
}

impl MockCloudStorageClient {
    pub fn new() -> Self {
        Self {
            storage: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
            metadata: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
        }
    }
}

impl CloudStorageClient for MockCloudStorageClient {
    fn upload(&self, key: &str, data: &[u8]) -> UtilsResult<String> {
        let mut storage = self.storage.lock().unwrap();
        let mut metadata = self.metadata.lock().unwrap();

        storage.insert(key.to_string(), data.to_vec());
        metadata.insert(
            key.to_string(),
            ObjectMetadata {
                size: data.len() as u64,
                etag: Some(format!("mock-etag-{key}")),
                content_type: Some("application/octet-stream".to_string()),
                last_modified: Some(chrono::Utc::now().to_rfc3339()),
                custom_metadata: HashMap::new(),
            },
        );

        Ok(format!("mock://bucket/{key}"))
    }

    fn download(&self, key: &str) -> UtilsResult<Vec<u8>> {
        let storage = self.storage.lock().unwrap();
        storage
            .get(key)
            .cloned()
            .ok_or_else(|| UtilsError::InvalidParameter(format!("Object not found: {key}")))
    }

    fn delete(&self, key: &str) -> UtilsResult<()> {
        let mut storage = self.storage.lock().unwrap();
        let mut metadata = self.metadata.lock().unwrap();

        storage.remove(key);
        metadata.remove(key);
        Ok(())
    }

    fn list_objects(&self, prefix: &str) -> UtilsResult<Vec<String>> {
        let storage = self.storage.lock().unwrap();
        let objects: Vec<String> = storage
            .keys()
            .filter(|key| key.starts_with(prefix))
            .cloned()
            .collect();
        Ok(objects)
    }

    fn exists(&self, key: &str) -> UtilsResult<bool> {
        let storage = self.storage.lock().unwrap();
        Ok(storage.contains_key(key))
    }

    fn get_metadata(&self, key: &str) -> UtilsResult<ObjectMetadata> {
        let metadata = self.metadata.lock().unwrap();
        metadata
            .get(key)
            .cloned()
            .ok_or_else(|| UtilsError::InvalidParameter(format!("Object not found: {key}")))
    }

    fn upload_file(&self, key: &str, local_path: &str) -> UtilsResult<String> {
        let data = std::fs::read(local_path)
            .map_err(|e| UtilsError::InvalidParameter(format!("Failed to read file: {e}")))?;
        self.upload(key, &data)
    }

    fn download_file(&self, key: &str, local_path: &str) -> UtilsResult<()> {
        let data = self.download(key)?;
        std::fs::write(local_path, data)
            .map_err(|e| UtilsError::InvalidParameter(format!("Failed to write file: {e}")))?;
        Ok(())
    }
}

/// Cloud storage factory
pub struct CloudStorageFactory;

impl CloudStorageFactory {
    /// Create a cloud storage client based on configuration
    pub fn create_client(config: &CloudStorageConfig) -> UtilsResult<Box<dyn CloudStorageClient>> {
        match config.provider {
            CloudProvider::AWS => {
                // In a real implementation, this would create an AWS S3 client
                // For now, we'll use the mock client
                Ok(Box::new(MockCloudStorageClient::new()))
            }
            CloudProvider::GoogleCloud => {
                // In a real implementation, this would create a GCS client
                Ok(Box::new(MockCloudStorageClient::new()))
            }
            CloudProvider::Azure => {
                // In a real implementation, this would create an Azure Blob client
                Ok(Box::new(MockCloudStorageClient::new()))
            }
            CloudProvider::MinIO => {
                // In a real implementation, this would create a MinIO client
                Ok(Box::new(MockCloudStorageClient::new()))
            }
            CloudProvider::Custom(_) => {
                // For custom providers, use mock client
                Ok(Box::new(MockCloudStorageClient::new()))
            }
        }
    }
}

/// Cloud storage utilities for ML data processing
pub struct CloudStorageUtils;

impl CloudStorageUtils {
    /// Upload ML dataset to cloud storage
    pub fn upload_dataset(
        client: &dyn CloudStorageClient,
        dataset_path: &str,
        key_prefix: &str,
    ) -> UtilsResult<Vec<String>> {
        let mut uploaded_keys = Vec::new();

        // Read dataset directory
        let entries = std::fs::read_dir(dataset_path)
            .map_err(|e| UtilsError::InvalidParameter(format!("Failed to read directory: {e}")))?;

        for entry in entries {
            let entry = entry
                .map_err(|e| UtilsError::InvalidParameter(format!("Failed to read entry: {e}")))?;
            let path = entry.path();

            if path.is_file() {
                let filename = path.file_name().unwrap().to_str().unwrap();
                let key = format!("{key_prefix}/{filename}");
                let local_path = path.to_str().unwrap();

                client.upload_file(&key, local_path)?;
                uploaded_keys.push(key);
            }
        }

        Ok(uploaded_keys)
    }

    /// Download ML dataset from cloud storage
    pub fn download_dataset(
        client: &dyn CloudStorageClient,
        key_prefix: &str,
        local_path: &str,
    ) -> UtilsResult<Vec<String>> {
        let objects = client.list_objects(key_prefix)?;
        let mut downloaded_files = Vec::new();

        // Create local directory if it doesn't exist
        std::fs::create_dir_all(local_path).map_err(|e| {
            UtilsError::InvalidParameter(format!("Failed to create directory: {e}"))
        })?;

        for object_key in objects {
            let filename = object_key.split('/').next_back().unwrap_or(&object_key);
            let local_file_path = format!("{local_path}/{filename}");

            client.download_file(&object_key, &local_file_path)?;
            downloaded_files.push(local_file_path);
        }

        Ok(downloaded_files)
    }

    /// Sync local dataset with cloud storage
    pub fn sync_dataset(
        client: &dyn CloudStorageClient,
        local_path: &str,
        key_prefix: &str,
        sync_mode: SyncMode,
    ) -> UtilsResult<SyncResult> {
        let mut result = SyncResult::default();

        match sync_mode {
            SyncMode::Upload => {
                let uploaded = Self::upload_dataset(client, local_path, key_prefix)?;
                result.uploaded = uploaded;
            }
            SyncMode::Download => {
                let downloaded = Self::download_dataset(client, key_prefix, local_path)?;
                result.downloaded = downloaded;
            }
            SyncMode::Bidirectional => {
                // Simple bidirectional sync: upload first, then download
                let uploaded = Self::upload_dataset(client, local_path, key_prefix)?;
                let downloaded = Self::download_dataset(client, key_prefix, local_path)?;
                result.uploaded = uploaded;
                result.downloaded = downloaded;
            }
        }

        Ok(result)
    }

    /// Batch upload multiple files with metadata
    pub fn batch_upload(
        client: &dyn CloudStorageClient,
        files: &[(String, String)], // (local_path, key)
    ) -> UtilsResult<Vec<String>> {
        let mut uploaded_keys = Vec::new();

        for (local_path, key) in files {
            let result = client.upload_file(key, local_path)?;
            uploaded_keys.push(result);
        }

        Ok(uploaded_keys)
    }

    /// Calculate storage metrics for ML datasets
    pub fn calculate_storage_metrics(
        client: &dyn CloudStorageClient,
        key_prefix: &str,
    ) -> UtilsResult<StorageMetrics> {
        let objects = client.list_objects(key_prefix)?;
        let mut total_size = 0;
        let mut total_objects = 0;
        let mut file_types = HashMap::new();

        for object_key in objects {
            if let Ok(metadata) = client.get_metadata(&object_key) {
                total_size += metadata.size;
                total_objects += 1;

                // Extract file extension
                if let Some(ext) = object_key.split('.').next_back() {
                    *file_types.entry(ext.to_string()).or_insert(0) += 1;
                }
            }
        }

        Ok(StorageMetrics {
            total_size_bytes: total_size,
            total_objects,
            file_types,
            average_file_size: if total_objects > 0 {
                total_size / total_objects
            } else {
                0
            },
        })
    }
}

/// Sync mode for dataset synchronization
#[derive(Debug, Clone)]
pub enum SyncMode {
    Upload,
    Download,
    Bidirectional,
}

/// Sync result
#[derive(Debug, Clone, Default)]
pub struct SyncResult {
    pub uploaded: Vec<String>,
    pub downloaded: Vec<String>,
    pub errors: Vec<String>,
}

/// Storage metrics
#[derive(Debug, Clone)]
pub struct StorageMetrics {
    pub total_size_bytes: u64,
    pub total_objects: u64,
    pub file_types: HashMap<String, usize>,
    pub average_file_size: u64,
}

impl fmt::Display for StorageMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Storage Metrics:")?;
        writeln!(
            f,
            "  Total Size: {:.2} MB",
            self.total_size_bytes as f64 / 1024.0 / 1024.0
        )?;
        writeln!(f, "  Total Objects: {}", self.total_objects)?;
        writeln!(
            f,
            "  Average File Size: {:.2} KB",
            self.average_file_size as f64 / 1024.0
        )?;
        writeln!(f, "  File Types:")?;
        for (ext, count) in &self.file_types {
            writeln!(f, "    .{ext}: {count}")?;
        }
        Ok(())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_cloud_storage_config() {
        let config = CloudStorageConfig {
            provider: CloudProvider::AWS,
            bucket: "test-bucket".to_string(),
            ..Default::default()
        };

        assert_eq!(config.provider, CloudProvider::AWS);
        assert_eq!(config.bucket, "test-bucket");
        assert_eq!(config.region, Some("us-east-1".to_string()));
        assert!(config.use_ssl);
    }

    #[test]
    fn test_cloud_provider_display() {
        assert_eq!(CloudProvider::AWS.to_string(), "aws");
        assert_eq!(CloudProvider::GoogleCloud.to_string(), "gcp");
        assert_eq!(CloudProvider::Azure.to_string(), "azure");
        assert_eq!(CloudProvider::MinIO.to_string(), "minio");
        assert_eq!(
            CloudProvider::Custom("test".to_string()).to_string(),
            "test"
        );
    }

    #[test]
    fn test_mock_client_upload_download() {
        let client = MockCloudStorageClient::new();
        let test_data = b"hello world";

        // Test upload
        let url = client.upload("test-key", test_data).unwrap();
        assert_eq!(url, "mock://bucket/test-key");

        // Test download
        let downloaded = client.download("test-key").unwrap();
        assert_eq!(downloaded, test_data);

        // Test exists
        assert!(client.exists("test-key").unwrap());
        assert!(!client.exists("nonexistent-key").unwrap());
    }

    #[test]
    fn test_mock_client_metadata() {
        let client = MockCloudStorageClient::new();
        let test_data = b"hello world";

        client.upload("test-key", test_data).unwrap();

        let metadata = client.get_metadata("test-key").unwrap();
        assert_eq!(metadata.size, test_data.len() as u64);
        assert_eq!(metadata.etag, Some("mock-etag-test-key".to_string()));
        assert_eq!(
            metadata.content_type,
            Some("application/octet-stream".to_string())
        );
    }

    #[test]
    fn test_mock_client_list_objects() {
        let client = MockCloudStorageClient::new();

        client.upload("data/file1.txt", b"content1").unwrap();
        client.upload("data/file2.txt", b"content2").unwrap();
        client.upload("other/file3.txt", b"content3").unwrap();

        let objects = client.list_objects("data/").unwrap();
        assert_eq!(objects.len(), 2);
        assert!(objects.contains(&"data/file1.txt".to_string()));
        assert!(objects.contains(&"data/file2.txt".to_string()));
    }

    #[test]
    fn test_mock_client_delete() {
        let client = MockCloudStorageClient::new();

        client.upload("test-key", b"hello").unwrap();
        assert!(client.exists("test-key").unwrap());

        client.delete("test-key").unwrap();
        assert!(!client.exists("test-key").unwrap());
    }

    #[test]
    fn test_cloud_storage_factory() {
        let config = CloudStorageConfig {
            provider: CloudProvider::AWS,
            bucket: "test-bucket".to_string(),
            ..Default::default()
        };

        let client = CloudStorageFactory::create_client(&config).unwrap();

        // Test that we can use the client
        client.upload("test", b"data").unwrap();
        let downloaded = client.download("test").unwrap();
        assert_eq!(downloaded, b"data");
    }

    #[test]
    fn test_storage_metrics_display() {
        let mut file_types = HashMap::new();
        file_types.insert("txt".to_string(), 5);
        file_types.insert("csv".to_string(), 3);

        let metrics = StorageMetrics {
            total_size_bytes: 1_048_576, // 1 MB
            total_objects: 8,
            file_types,
            average_file_size: 131_072, // 128 KB
        };

        let display = metrics.to_string();
        assert!(display.contains("Total Size: 1.00 MB"));
        assert!(display.contains("Total Objects: 8"));
        assert!(display.contains("Average File Size: 128.00 KB"));
        assert!(display.contains(".txt: 5"));
        assert!(display.contains(".csv: 3"));
    }

    #[test]
    fn test_sync_result_default() {
        let result = SyncResult::default();
        assert!(result.uploaded.is_empty());
        assert!(result.downloaded.is_empty());
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_file_upload_download() {
        let client = MockCloudStorageClient::new();
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test.txt");

        // Create test file
        fs::write(&file_path, b"test content").unwrap();

        // Upload file
        let url = client
            .upload_file("test.txt", file_path.to_str().unwrap())
            .unwrap();
        assert_eq!(url, "mock://bucket/test.txt");

        // Download file
        let download_path = temp_dir.path().join("downloaded.txt");
        client
            .download_file("test.txt", download_path.to_str().unwrap())
            .unwrap();

        // Verify content
        let downloaded_content = fs::read(&download_path).unwrap();
        assert_eq!(downloaded_content, b"test content");
    }

    #[test]
    fn test_calculate_storage_metrics() {
        let client = MockCloudStorageClient::new();

        // Upload test files
        client.upload("data/file1.txt", b"hello").unwrap();
        client.upload("data/file2.csv", b"world").unwrap();
        client.upload("data/file3.txt", b"test").unwrap();

        let metrics = CloudStorageUtils::calculate_storage_metrics(&client, "data/").unwrap();

        assert_eq!(metrics.total_objects, 3);
        assert_eq!(metrics.total_size_bytes, 14); // 5 + 5 + 4
        assert_eq!(metrics.file_types.get("txt"), Some(&2));
        assert_eq!(metrics.file_types.get("csv"), Some(&1));
    }

    #[test]
    fn test_batch_upload() {
        let client = MockCloudStorageClient::new();
        let temp_dir = tempfile::tempdir().unwrap();

        // Create test files
        let file1_path = temp_dir.path().join("file1.txt");
        let file2_path = temp_dir.path().join("file2.txt");
        fs::write(&file1_path, b"content1").unwrap();
        fs::write(&file2_path, b"content2").unwrap();

        let files = vec![
            (
                file1_path.to_str().unwrap().to_string(),
                "batch/file1.txt".to_string(),
            ),
            (
                file2_path.to_str().unwrap().to_string(),
                "batch/file2.txt".to_string(),
            ),
        ];

        let results = CloudStorageUtils::batch_upload(&client, &files).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0], "mock://bucket/batch/file1.txt");
        assert_eq!(results[1], "mock://bucket/batch/file2.txt");

        // Verify uploads
        let content1 = client.download("batch/file1.txt").unwrap();
        let content2 = client.download("batch/file2.txt").unwrap();
        assert_eq!(content1, b"content1");
        assert_eq!(content2, b"content2");
    }
}
