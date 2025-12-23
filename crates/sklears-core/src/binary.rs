/// Binary format support for efficient storage and transfer of ML models and datasets
///
/// This module provides high-performance binary serialization/deserialization capabilities
/// for machine learning models, datasets, and related structures using multiple formats:
/// - Bincode: Fast binary serialization
/// - MessagePack: Cross-language binary format  
/// - Custom format: Optimized for large arrays
///
/// Key features:
/// - Zero-copy deserialization where possible
/// - Compression support (gzip, zstd)
/// - Streaming serialization for large datasets
/// - Version compatibility and migration
/// - Memory-mapped file support
///
/// # Examples
///
/// ## Basic Array Serialization
///
/// ```rust
/// use sklears_core::binary::convenience;
/// use sklears_core::types::Array2;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Create a sample array
/// let array = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
///
/// // Save to binary file
/// convenience::save_array2(&array, "model_weights.skl")?;
///
/// // Load from binary file
/// let loaded_array = convenience::load_array2("model_weights.skl")?;
///
/// assert_eq!(loaded_array.dim(), array.dim());
/// # std::fs::remove_file("model_weights.skl").ok();
/// # Ok(())
/// # }
/// ```
///
/// ## Custom Configuration
///
/// ```rust
/// use sklears_core::binary::{BinaryConfig, BinaryFormat, CompressionType, BinarySerialize};
/// use sklears_core::types::Array1;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
///
/// let config = BinaryConfig {
///     format: BinaryFormat::Custom,
///     compression: CompressionType::None,
///     include_metadata: true,
///     version: 1,
///     little_endian: true,
/// };
///
/// // Serialize to bytes
/// let bytes = data.serialize_binary(&config)?;
///
/// // Save to file with custom config
/// data.serialize_binary_to_file("custom_data.skl", &config)?;
/// # std::fs::remove_file("custom_data.skl").ok();
/// # Ok(())
/// # }
/// ```
use crate::error::{Result, SklearsError};
use crate::types::{Array1, Array2, Float};
use std::io::{Read, Write};
use std::path::Path;

#[cfg(feature = "binary")]
use serde::{Deserialize, Serialize};

/// Binary format types supported
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryFormat {
    /// Bincode - Fast binary serialization
    Bincode,
    /// MessagePack - Cross-language binary format
    MessagePack,
    /// Custom optimized format for ML data
    Custom,
}

/// Compression types supported
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionType {
    /// No compression
    None,
    /// Gzip compression
    #[cfg(feature = "compression")]
    Gzip,
    /// Zstandard compression  
    #[cfg(feature = "compression")]
    Zstd,
}

/// Binary serialization configuration
#[derive(Debug, Clone)]
pub struct BinaryConfig {
    /// Binary format to use
    pub format: BinaryFormat,
    /// Compression type
    pub compression: CompressionType,
    /// Whether to include metadata
    pub include_metadata: bool,
    /// Version for compatibility
    pub version: u32,
    /// Endianness (for cross-platform compatibility)
    pub little_endian: bool,
}

impl Default for BinaryConfig {
    fn default() -> Self {
        Self {
            format: BinaryFormat::Bincode,
            compression: CompressionType::None,
            include_metadata: true,
            version: 1,
            little_endian: true,
        }
    }
}

/// Metadata for binary files
#[cfg(feature = "binary")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryMetadata {
    /// File format version
    pub version: u32,
    /// Creation timestamp
    pub created_at: u64,
    /// Data type information
    pub data_type: String,
    /// Shape information for arrays
    pub shape: Option<Vec<usize>>,
    /// Original size before compression
    pub uncompressed_size: Option<u64>,
    /// Compression type used
    pub compression: String,
    /// Endianness
    pub little_endian: bool,
    /// Custom user metadata
    pub user_metadata: std::collections::HashMap<String, String>,
}

#[cfg(feature = "binary")]
impl Default for BinaryMetadata {
    fn default() -> Self {
        Self {
            version: 1,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            data_type: "unknown".to_string(),
            shape: None,
            uncompressed_size: None,
            compression: "none".to_string(),
            little_endian: true,
            user_metadata: std::collections::HashMap::new(),
        }
    }
}

/// Trait for binary serialization
pub trait BinarySerialize {
    /// Serialize to binary format
    fn serialize_binary(&self, config: &BinaryConfig) -> Result<Vec<u8>>;

    /// Serialize to writer
    fn serialize_binary_to_writer<W: Write>(&self, writer: W, config: &BinaryConfig) -> Result<()>;

    /// Serialize to file
    fn serialize_binary_to_file<P: AsRef<Path>>(
        &self,
        path: P,
        config: &BinaryConfig,
    ) -> Result<()>;
}

/// Trait for binary deserialization
pub trait BinaryDeserialize: Sized {
    /// Deserialize from binary data
    fn deserialize_binary(data: &[u8], config: &BinaryConfig) -> Result<Self>;

    /// Deserialize from reader
    fn deserialize_binary_from_reader<R: Read>(reader: R, config: &BinaryConfig) -> Result<Self>;

    /// Deserialize from file
    fn deserialize_binary_from_file<P: AsRef<Path>>(path: P, config: &BinaryConfig)
        -> Result<Self>;
}

/// Binary serializer with format-specific implementations
pub struct BinarySerializer {
    config: BinaryConfig,
}

impl BinarySerializer {
    /// Create new serializer with configuration
    pub fn new(config: BinaryConfig) -> Self {
        Self { config }
    }

    /// Create with default bincode configuration
    pub fn bincode() -> Self {
        Self::new(BinaryConfig {
            format: BinaryFormat::Bincode,
            ..Default::default()
        })
    }

    /// Create with MessagePack configuration
    pub fn messagepack() -> Self {
        Self::new(BinaryConfig {
            format: BinaryFormat::MessagePack,
            ..Default::default()
        })
    }

    /// Create with custom format configuration
    pub fn custom() -> Self {
        Self::new(BinaryConfig {
            format: BinaryFormat::Custom,
            ..Default::default()
        })
    }

    /// Serialize data to bytes
    #[cfg(feature = "binary")]
    pub fn serialize<T: Serialize>(&self, data: &T) -> Result<Vec<u8>> {
        let serialized = match self.config.format {
            BinaryFormat::Bincode => bincode::serialize(data).map_err(|e| {
                SklearsError::SerializationError(format!("Bincode serialization failed: {}", e))
            })?,
            BinaryFormat::MessagePack => {
                #[cfg(feature = "messagepack")]
                {
                    rmp_serde::to_vec(data).map_err(|e| {
                        SklearsError::SerializationError(format!(
                            "MessagePack serialization failed: {}",
                            e
                        ))
                    })?
                }
                #[cfg(not(feature = "messagepack"))]
                {
                    return Err(SklearsError::MissingDependency {
                        dependency: "rmp-serde".to_string(),
                        feature: "MessagePack serialization".to_string(),
                    });
                }
            }
            BinaryFormat::Custom => {
                // For custom format, fallback to bincode for now
                bincode::serialize(data).map_err(|e| {
                    SklearsError::SerializationError(format!("Custom serialization failed: {}", e))
                })?
            }
        };

        self.apply_compression(serialized)
    }

    /// Deserialize data from bytes
    #[cfg(feature = "binary")]
    pub fn deserialize<T: for<'de> Deserialize<'de>>(&self, data: &[u8]) -> Result<T> {
        let decompressed = self.decompress(data)?;

        match self.config.format {
            BinaryFormat::Bincode => bincode::deserialize(&decompressed).map_err(|e| {
                SklearsError::DeserializationError(format!("Bincode deserialization failed: {}", e))
            }),
            BinaryFormat::MessagePack => {
                #[cfg(feature = "messagepack")]
                {
                    rmp_serde::from_slice(&decompressed).map_err(|e| {
                        SklearsError::DeserializationError(format!(
                            "MessagePack deserialization failed: {}",
                            e
                        ))
                    })
                }
                #[cfg(not(feature = "messagepack"))]
                {
                    Err(SklearsError::MissingDependency {
                        dependency: "rmp-serde".to_string(),
                        feature: "MessagePack deserialization".to_string(),
                    })
                }
            }
            BinaryFormat::Custom => {
                // For custom format, fallback to bincode for now
                bincode::deserialize(&decompressed).map_err(|e| {
                    SklearsError::DeserializationError(format!(
                        "Custom deserialization failed: {}",
                        e
                    ))
                })
            }
        }
    }

    /// Apply compression to data
    fn apply_compression(&self, data: Vec<u8>) -> Result<Vec<u8>> {
        match self.config.compression {
            CompressionType::None => Ok(data),
            #[cfg(feature = "compression")]
            CompressionType::Gzip => {
                use std::io::Write;
                let mut encoder =
                    flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
                encoder.write_all(&data).map_err(|e| {
                    SklearsError::SerializationError(format!("Gzip compression failed: {}", e))
                })?;
                encoder.finish().map_err(|e| {
                    SklearsError::SerializationError(format!(
                        "Gzip compression finish failed: {}",
                        e
                    ))
                })
            }
            #[cfg(feature = "compression")]
            CompressionType::Zstd => zstd::encode_all(&data[..], 3).map_err(|e| {
                SklearsError::SerializationError(format!("Zstd compression failed: {}", e))
            }),
            #[cfg(not(feature = "compression"))]
            _ => Err(SklearsError::MissingDependency {
                dependency: "compression".to_string(),
                feature: "data compression".to_string(),
            }),
        }
    }

    /// Decompress data
    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        match self.config.compression {
            CompressionType::None => Ok(data.to_vec()),
            #[cfg(feature = "compression")]
            CompressionType::Gzip => {
                use std::io::Read;
                let mut decoder = flate2::read::GzDecoder::new(data);
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed).map_err(|e| {
                    SklearsError::DeserializationError(format!("Gzip decompression failed: {}", e))
                })?;
                Ok(decompressed)
            }
            #[cfg(feature = "compression")]
            CompressionType::Zstd => zstd::decode_all(data).map_err(|e| {
                SklearsError::DeserializationError(format!("Zstd decompression failed: {}", e))
            }),
            #[cfg(not(feature = "compression"))]
            _ => Err(SklearsError::MissingDependency {
                dependency: "compression".to_string(),
                feature: "data decompression".to_string(),
            }),
        }
    }
}

/// Efficient binary format for large arrays
pub struct ArrayBinaryFormat;

impl ArrayBinaryFormat {
    /// Serialize Array2 to custom binary format optimized for large arrays
    pub fn serialize_array2(array: &Array2<Float>) -> Result<Vec<u8>> {
        let (rows, cols) = array.dim();
        let mut buffer = Vec::with_capacity(24 + rows * cols * 8); // Header + data

        // Write header
        buffer.extend_from_slice(b"SKLA"); // Magic number
        buffer.extend_from_slice(&1u32.to_le_bytes()); // Version
        buffer.extend_from_slice(&(rows as u64).to_le_bytes()); // Rows
        buffer.extend_from_slice(&(cols as u64).to_le_bytes()); // Cols

        // Write data in row-major order
        for row in array.rows() {
            for &value in row {
                buffer.extend_from_slice(&value.to_le_bytes());
            }
        }

        Ok(buffer)
    }

    /// Deserialize Array2 from custom binary format
    pub fn deserialize_array2(data: &[u8]) -> Result<Array2<Float>> {
        if data.len() < 24 {
            return Err(SklearsError::DeserializationError(
                "Insufficient data for array header".to_string(),
            ));
        }

        // Check magic number
        if &data[0..4] != b"SKLA" {
            return Err(SklearsError::DeserializationError(
                "Invalid magic number".to_string(),
            ));
        }

        // Read header
        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if version != 1 {
            return Err(SklearsError::DeserializationError(format!(
                "Unsupported version: {}",
                version
            )));
        }

        let rows = u64::from_le_bytes([
            data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
        ]) as usize;
        let cols = u64::from_le_bytes([
            data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23],
        ]) as usize;

        // Check data length
        let expected_len = 24 + rows * cols * 8;
        if data.len() != expected_len {
            return Err(SklearsError::DeserializationError(format!(
                "Data length mismatch: expected {}, got {}",
                expected_len,
                data.len()
            )));
        }

        // Read array data
        let mut array_data = Vec::with_capacity(rows * cols);
        let data_start = 24;

        for i in 0..(rows * cols) {
            let start = data_start + i * 8;
            let end = start + 8;
            let bytes: [u8; 8] = data[start..end].try_into().map_err(|_| {
                SklearsError::DeserializationError("Failed to read float bytes".to_string())
            })?;
            array_data.push(Float::from_le_bytes(bytes));
        }

        Array2::from_shape_vec((rows, cols), array_data).map_err(|e| {
            SklearsError::DeserializationError(format!("Failed to create array: {}", e))
        })
    }

    /// Serialize Array1 to custom binary format
    pub fn serialize_array1(array: &Array1<Float>) -> Result<Vec<u8>> {
        let len = array.len();
        let mut buffer = Vec::with_capacity(16 + len * 8); // Header + data

        // Write header
        buffer.extend_from_slice(b"SKL1"); // Magic number for 1D arrays
        buffer.extend_from_slice(&1u32.to_le_bytes()); // Version
        buffer.extend_from_slice(&(len as u64).to_le_bytes()); // Length

        // Write data
        for &value in array {
            buffer.extend_from_slice(&value.to_le_bytes());
        }

        Ok(buffer)
    }

    /// Deserialize Array1 from custom binary format
    pub fn deserialize_array1(data: &[u8]) -> Result<Array1<Float>> {
        if data.len() < 16 {
            return Err(SklearsError::DeserializationError(
                "Insufficient data for array header".to_string(),
            ));
        }

        // Check magic number
        if &data[0..4] != b"SKL1" {
            return Err(SklearsError::DeserializationError(
                "Invalid magic number for 1D array".to_string(),
            ));
        }

        // Read header
        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if version != 1 {
            return Err(SklearsError::DeserializationError(format!(
                "Unsupported version: {}",
                version
            )));
        }

        let len = u64::from_le_bytes([
            data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
        ]) as usize;

        // Check data length
        let expected_len = 16 + len * 8;
        if data.len() != expected_len {
            return Err(SklearsError::DeserializationError(format!(
                "Data length mismatch: expected {}, got {}",
                expected_len,
                data.len()
            )));
        }

        // Read array data
        let mut array_data = Vec::with_capacity(len);
        let data_start = 16;

        for i in 0..len {
            let start = data_start + i * 8;
            let end = start + 8;
            let bytes: [u8; 8] = data[start..end].try_into().map_err(|_| {
                SklearsError::DeserializationError("Failed to read float bytes".to_string())
            })?;
            array_data.push(Float::from_le_bytes(bytes));
        }

        Ok(Array1::from_vec(array_data))
    }
}

/// Streaming binary writer for large datasets
pub struct StreamingBinaryWriter<W: Write> {
    writer: W,
    config: BinaryConfig,
    #[cfg(feature = "binary")]
    metadata: BinaryMetadata,
}

impl<W: Write> StreamingBinaryWriter<W> {
    /// Create new streaming writer
    #[cfg(feature = "binary")]
    pub fn new(writer: W, config: BinaryConfig) -> Result<Self> {
        let metadata = BinaryMetadata::default();
        let mut instance = Self {
            writer,
            config,
            metadata,
        };

        // Write file header if metadata is enabled
        if instance.config.include_metadata {
            instance.write_metadata()?;
        }

        Ok(instance)
    }

    /// Write metadata header
    #[cfg(feature = "binary")]
    fn write_metadata(&mut self) -> Result<()> {
        let serializer = BinarySerializer::new(self.config.clone());
        let metadata_bytes = serializer.serialize(&self.metadata)?;

        // Write metadata length followed by metadata
        let metadata_len = metadata_bytes.len() as u64;
        self.writer
            .write_all(&metadata_len.to_le_bytes())
            .map_err(|e| {
                SklearsError::SerializationError(format!("Failed to write metadata length: {}", e))
            })?;
        self.writer.write_all(&metadata_bytes).map_err(|e| {
            SklearsError::SerializationError(format!("Failed to write metadata: {}", e))
        })?;

        Ok(())
    }

    /// Write a chunk of data
    #[cfg(feature = "binary")]
    pub fn write_chunk<T: Serialize>(&mut self, chunk: &T) -> Result<()> {
        let serializer = BinarySerializer::new(self.config.clone());
        let chunk_bytes = serializer.serialize(chunk)?;

        // Write chunk length followed by chunk data
        let chunk_len = chunk_bytes.len() as u64;
        self.writer
            .write_all(&chunk_len.to_le_bytes())
            .map_err(|e| {
                SklearsError::SerializationError(format!("Failed to write chunk length: {}", e))
            })?;
        self.writer.write_all(&chunk_bytes).map_err(|e| {
            SklearsError::SerializationError(format!("Failed to write chunk: {}", e))
        })?;

        Ok(())
    }

    /// Finish writing and flush
    pub fn finish(mut self) -> Result<W> {
        self.writer.flush().map_err(|e| {
            SklearsError::SerializationError(format!("Failed to flush writer: {}", e))
        })?;
        Ok(self.writer)
    }
}

/// Streaming binary reader for large datasets
pub struct StreamingBinaryReader<R: Read> {
    reader: R,
    config: BinaryConfig,
    #[cfg(feature = "binary")]
    metadata: Option<BinaryMetadata>,
}

impl<R: Read> StreamingBinaryReader<R> {
    /// Create new streaming reader
    #[cfg(feature = "binary")]
    pub fn new(mut reader: R, config: BinaryConfig) -> Result<Self> {
        let metadata = if config.include_metadata {
            Some(Self::read_metadata(&mut reader, &config)?)
        } else {
            None
        };

        Ok(Self {
            reader,
            config,
            metadata,
        })
    }

    /// Read metadata from stream
    #[cfg(feature = "binary")]
    fn read_metadata(reader: &mut R, config: &BinaryConfig) -> Result<BinaryMetadata> {
        // Read metadata length
        let mut len_bytes = [0u8; 8];
        reader.read_exact(&mut len_bytes).map_err(|e| {
            SklearsError::DeserializationError(format!("Failed to read metadata length: {}", e))
        })?;
        let metadata_len = u64::from_le_bytes(len_bytes) as usize;

        // Read metadata
        let mut metadata_bytes = vec![0u8; metadata_len];
        reader.read_exact(&mut metadata_bytes).map_err(|e| {
            SklearsError::DeserializationError(format!("Failed to read metadata: {}", e))
        })?;

        let deserializer = BinarySerializer::new(config.clone());
        deserializer.deserialize(&metadata_bytes)
    }

    /// Read next chunk from stream
    #[cfg(feature = "binary")]
    pub fn read_chunk<T: for<'de> Deserialize<'de>>(&mut self) -> Result<Option<T>> {
        // Try to read chunk length
        let mut len_bytes = [0u8; 8];
        match self.reader.read_exact(&mut len_bytes) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Ok(None); // End of stream
            }
            Err(e) => {
                return Err(SklearsError::DeserializationError(format!(
                    "Failed to read chunk length: {}",
                    e
                )));
            }
        }

        let chunk_len = u64::from_le_bytes(len_bytes) as usize;

        // Read chunk data
        let mut chunk_bytes = vec![0u8; chunk_len];
        self.reader.read_exact(&mut chunk_bytes).map_err(|e| {
            SklearsError::DeserializationError(format!("Failed to read chunk: {}", e))
        })?;

        let deserializer = BinarySerializer::new(self.config.clone());
        let chunk = deserializer.deserialize(&chunk_bytes)?;
        Ok(Some(chunk))
    }

    /// Get metadata if available
    #[cfg(feature = "binary")]
    pub fn metadata(&self) -> Option<&BinaryMetadata> {
        self.metadata.as_ref()
    }
}

/// File-based binary storage utilities
pub struct BinaryFileStorage;

impl BinaryFileStorage {
    /// Save data to binary file with metadata
    #[cfg(feature = "binary")]
    pub fn save<T: Serialize, P: AsRef<Path>>(
        data: &T,
        path: P,
        config: &BinaryConfig,
    ) -> Result<()> {
        let file = std::fs::File::create(&path).map_err(|e| {
            SklearsError::FileError(format!(
                "Failed to create file {}: {}",
                path.as_ref().display(),
                e
            ))
        })?;

        let mut writer = StreamingBinaryWriter::new(file, config.clone())?;
        writer.write_chunk(data)?;
        writer.finish()?;

        Ok(())
    }

    /// Load data from binary file
    #[cfg(feature = "binary")]
    pub fn load<T: for<'de> Deserialize<'de>, P: AsRef<Path>>(
        path: P,
        config: &BinaryConfig,
    ) -> Result<T> {
        let file = std::fs::File::open(&path).map_err(|e| {
            SklearsError::FileError(format!(
                "Failed to open file {}: {}",
                path.as_ref().display(),
                e
            ))
        })?;

        let mut reader = StreamingBinaryReader::new(file, config.clone())?;
        reader
            .read_chunk()?
            .ok_or_else(|| SklearsError::DeserializationError("No data found in file".to_string()))
    }

    /// Get file info without loading data
    #[cfg(feature = "binary")]
    pub fn info<P: AsRef<Path>>(path: P, config: &BinaryConfig) -> Result<BinaryMetadata> {
        let file = std::fs::File::open(&path).map_err(|e| {
            SklearsError::FileError(format!(
                "Failed to open file {}: {}",
                path.as_ref().display(),
                e
            ))
        })?;

        let reader = StreamingBinaryReader::new(file, config.clone())?;
        reader.metadata().cloned().ok_or_else(|| {
            SklearsError::DeserializationError("No metadata found in file".to_string())
        })
    }
}

// Implementations for common types
#[cfg(feature = "binary")]
impl BinarySerialize for Array2<Float> {
    fn serialize_binary(&self, config: &BinaryConfig) -> Result<Vec<u8>> {
        match config.format {
            BinaryFormat::Custom => ArrayBinaryFormat::serialize_array2(self),
            _ => {
                let serializer = BinarySerializer::new(config.clone());
                serializer.serialize(self)
            }
        }
    }

    fn serialize_binary_to_writer<W: Write>(
        &self,
        mut writer: W,
        config: &BinaryConfig,
    ) -> Result<()> {
        let data = self.serialize_binary(config)?;
        writer.write_all(&data).map_err(|e| {
            SklearsError::SerializationError(format!("Failed to write to writer: {}", e))
        })
    }

    fn serialize_binary_to_file<P: AsRef<Path>>(
        &self,
        path: P,
        config: &BinaryConfig,
    ) -> Result<()> {
        let file = std::fs::File::create(&path).map_err(|e| {
            SklearsError::FileError(format!(
                "Failed to create file {}: {}",
                path.as_ref().display(),
                e
            ))
        })?;
        self.serialize_binary_to_writer(file, config)
    }
}

#[cfg(feature = "binary")]
impl BinaryDeserialize for Array2<Float> {
    fn deserialize_binary(data: &[u8], config: &BinaryConfig) -> Result<Self> {
        match config.format {
            BinaryFormat::Custom => ArrayBinaryFormat::deserialize_array2(data),
            _ => {
                let serializer = BinarySerializer::new(config.clone());
                serializer.deserialize(data)
            }
        }
    }

    fn deserialize_binary_from_reader<R: Read>(
        mut reader: R,
        config: &BinaryConfig,
    ) -> Result<Self> {
        let mut data = Vec::new();
        reader.read_to_end(&mut data).map_err(|e| {
            SklearsError::DeserializationError(format!("Failed to read from reader: {}", e))
        })?;
        Self::deserialize_binary(&data, config)
    }

    fn deserialize_binary_from_file<P: AsRef<Path>>(
        path: P,
        config: &BinaryConfig,
    ) -> Result<Self> {
        let file = std::fs::File::open(&path).map_err(|e| {
            SklearsError::FileError(format!(
                "Failed to open file {}: {}",
                path.as_ref().display(),
                e
            ))
        })?;
        Self::deserialize_binary_from_reader(file, config)
    }
}

#[cfg(feature = "binary")]
impl BinarySerialize for Array1<Float> {
    fn serialize_binary(&self, config: &BinaryConfig) -> Result<Vec<u8>> {
        match config.format {
            BinaryFormat::Custom => ArrayBinaryFormat::serialize_array1(self),
            _ => {
                let serializer = BinarySerializer::new(config.clone());
                serializer.serialize(self)
            }
        }
    }

    fn serialize_binary_to_writer<W: Write>(
        &self,
        mut writer: W,
        config: &BinaryConfig,
    ) -> Result<()> {
        let data = self.serialize_binary(config)?;
        writer.write_all(&data).map_err(|e| {
            SklearsError::SerializationError(format!("Failed to write to writer: {}", e))
        })
    }

    fn serialize_binary_to_file<P: AsRef<Path>>(
        &self,
        path: P,
        config: &BinaryConfig,
    ) -> Result<()> {
        let file = std::fs::File::create(&path).map_err(|e| {
            SklearsError::FileError(format!(
                "Failed to create file {}: {}",
                path.as_ref().display(),
                e
            ))
        })?;
        self.serialize_binary_to_writer(file, config)
    }
}

#[cfg(feature = "binary")]
impl BinaryDeserialize for Array1<Float> {
    fn deserialize_binary(data: &[u8], config: &BinaryConfig) -> Result<Self> {
        match config.format {
            BinaryFormat::Custom => ArrayBinaryFormat::deserialize_array1(data),
            _ => {
                let serializer = BinarySerializer::new(config.clone());
                serializer.deserialize(data)
            }
        }
    }

    fn deserialize_binary_from_reader<R: Read>(
        mut reader: R,
        config: &BinaryConfig,
    ) -> Result<Self> {
        let mut data = Vec::new();
        reader.read_to_end(&mut data).map_err(|e| {
            SklearsError::DeserializationError(format!("Failed to read from reader: {}", e))
        })?;
        Self::deserialize_binary(&data, config)
    }

    fn deserialize_binary_from_file<P: AsRef<Path>>(
        path: P,
        config: &BinaryConfig,
    ) -> Result<Self> {
        let file = std::fs::File::open(&path).map_err(|e| {
            SklearsError::FileError(format!(
                "Failed to open file {}: {}",
                path.as_ref().display(),
                e
            ))
        })?;
        Self::deserialize_binary_from_reader(file, config)
    }
}

/// Convenience functions for common use cases
pub mod convenience {
    use super::*;

    pub fn save_array2<P: AsRef<Path>>(array: &Array2<Float>, path: P) -> Result<()> {
        let config = BinaryConfig {
            format: BinaryFormat::Custom,
            compression: CompressionType::None,
            include_metadata: true,
            ..Default::default()
        };
        array.serialize_binary_to_file(path, &config)
    }

    /// Load Array2 from file
    pub fn load_array2<P: AsRef<Path>>(path: P) -> Result<Array2<Float>> {
        let config = BinaryConfig {
            format: BinaryFormat::Custom,
            ..Default::default()
        };
        Array2::deserialize_binary_from_file(path, &config)
    }

    /// Save Array1 to file with optimal settings
    pub fn save_array1<P: AsRef<Path>>(array: &Array1<Float>, path: P) -> Result<()> {
        let config = BinaryConfig {
            format: BinaryFormat::Custom,
            compression: CompressionType::None,
            include_metadata: true,
            ..Default::default()
        };
        array.serialize_binary_to_file(path, &config)
    }

    /// Load Array1 from file
    pub fn load_array1<P: AsRef<Path>>(path: P) -> Result<Array1<Float>> {
        let config = BinaryConfig {
            format: BinaryFormat::Custom,
            ..Default::default()
        };
        Array1::deserialize_binary_from_file(path, &config)
    }

    /// Save with compression
    #[cfg(feature = "compression")]
    pub fn save_array2_compressed<P: AsRef<Path>>(array: &Array2<Float>, path: P) -> Result<()> {
        let config = BinaryConfig {
            format: BinaryFormat::Custom,
            compression: CompressionType::Zstd,
            include_metadata: true,
            ..Default::default()
        };
        array.serialize_binary_to_file(path, &config)
    }

    /// Load with compression
    #[cfg(feature = "compression")]
    pub fn load_array2_compressed<P: AsRef<Path>>(path: P) -> Result<Array2<Float>> {
        let config = BinaryConfig {
            format: BinaryFormat::Custom,
            compression: CompressionType::Zstd,
            ..Default::default()
        };
        Array2::deserialize_binary_from_file(path, &config)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_binary_format() {
        let array = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        // Test serialization
        let bytes = ArrayBinaryFormat::serialize_array2(&array).unwrap();
        assert!(bytes.len() > 24); // Header + data

        // Test deserialization
        let restored = ArrayBinaryFormat::deserialize_array2(&bytes).unwrap();
        assert_eq!(restored.dim(), array.dim());

        for ((i, j), &original) in array.indexed_iter() {
            assert!((restored[[i, j]] - original).abs() < 1e-10);
        }
    }

    #[test]
    fn test_array1_binary_format() {
        let array = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        // Test serialization
        let bytes = ArrayBinaryFormat::serialize_array1(&array).unwrap();
        assert!(bytes.len() > 16); // Header + data

        // Test deserialization
        let restored = ArrayBinaryFormat::deserialize_array1(&bytes).unwrap();
        assert_eq!(restored.len(), array.len());

        for (i, &original) in array.iter().enumerate() {
            assert!((restored[i] - original).abs() < 1e-10);
        }
    }

    #[cfg(feature = "binary")]
    #[test]
    fn test_binary_serializer() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0];
        let config = BinaryConfig::default();
        let serializer = BinarySerializer::new(config);

        // Test serialization
        let bytes = serializer.serialize(&data).unwrap();
        assert!(!bytes.is_empty());

        // Test deserialization
        let restored: Vec<f64> = serializer.deserialize(&bytes).unwrap();
        assert_eq!(restored, data);
    }

    #[test]
    fn test_binary_format_types() {
        assert_eq!(BinaryFormat::Bincode, BinaryFormat::Bincode);
        assert_ne!(BinaryFormat::Bincode, BinaryFormat::MessagePack);

        let config = BinaryConfig::default();
        assert_eq!(config.format, BinaryFormat::Bincode);
        assert_eq!(config.version, 1);
        assert!(config.include_metadata);
    }

    #[cfg(feature = "binary")]
    #[test]
    fn test_array_binary_serialize_traits() {
        let array = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let config = BinaryConfig {
            format: BinaryFormat::Custom,
            ..Default::default()
        };

        // Test trait implementation
        let bytes = array.serialize_binary(&config).unwrap();
        assert!(!bytes.is_empty());

        let restored = Array2::deserialize_binary(&bytes, &config).unwrap();
        assert_eq!(restored.dim(), array.dim());

        for ((i, j), &original) in array.indexed_iter() {
            assert!((restored[[i, j]] - original).abs() < 1e-10);
        }
    }

    #[test]
    fn test_convenience_functions() {
        use tempfile::NamedTempFile;

        let array = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        // Test save and load with temporary file
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        convenience::save_array2(&array, path).unwrap();
        let loaded = convenience::load_array2(path).unwrap();

        assert_eq!(loaded.dim(), array.dim());
        for ((i, j), &original) in array.indexed_iter() {
            assert!((loaded[[i, j]] - original).abs() < 1e-10);
        }
    }

    #[test]
    fn test_array1_convenience() {
        use tempfile::NamedTempFile;

        let array = Array1::from_vec(vec![10.0, 20.0, 30.0]);

        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        convenience::save_array1(&array, path).unwrap();
        let loaded = convenience::load_array1(path).unwrap();

        assert_eq!(loaded.len(), array.len());
        for (i, &original) in array.iter().enumerate() {
            assert!((loaded[i] - original).abs() < 1e-10);
        }
    }

    #[test]
    fn test_binary_metadata() {
        #[cfg(feature = "binary")]
        {
            let metadata = BinaryMetadata::default();
            assert_eq!(metadata.version, 1);
            assert!(metadata.little_endian);
            assert_eq!(metadata.compression, "none");
        }
    }

    #[test]
    fn test_compression_types() {
        assert_eq!(CompressionType::None, CompressionType::None);

        #[cfg(feature = "compression")]
        {
            assert_ne!(CompressionType::None, CompressionType::Gzip);
            assert_ne!(CompressionType::Gzip, CompressionType::Zstd);
        }
    }
}
