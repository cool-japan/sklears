use thiserror::Error;

#[derive(Debug, Error)]
pub enum DataStorageError {
    #[error("Backend not found: {0}")]
    BackendNotFound(String),

    #[error("Backend already exists: {0}")]
    BackendAlreadyExists(String),

    #[error("Backend unavailable: {0}")]
    BackendUnavailable(String),

    #[error("Storage operation failed: {0}")]
    StorageOperationFailed(String),

    #[error("Index operation failed: {0}")]
    IndexOperationFailed(String),

    #[error("Query execution failed: {0}")]
    QueryExecutionFailed(String),

    #[error("Compression failed: {0}")]
    CompressionFailed(String),

    #[error("Cache operation failed: {0}")]
    CacheOperationFailed(String),

    #[error("Backup operation failed: {0}")]
    BackupOperationFailed(String),

    #[error("Integrity check failed: {0}")]
    IntegrityCheckFailed(String),

    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type DataStorageResult<T> = Result<T, DataStorageError>;
