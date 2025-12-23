/// Memory-mapped dataset functionality for large datasets
///
/// This module provides memory-mapped dataset capabilities for handling
/// datasets too large to fit in memory. It includes file format definition,
/// streaming I/O, and batch processing capabilities.
use crate::dataset::core::Dataset;
use crate::error::Result;
use crate::types::Float;

#[cfg(feature = "mmap")]
use memmap2::MmapOptions;
// SciRS2 Policy: Target migration to scirs2_core::ndarray
// TODO: Replace with scirs2_core::ndarray::Array when stable
#[cfg(feature = "mmap")]
use scirs2_core::ndarray::Array;
#[cfg(feature = "mmap")]
use std::io::Write;
#[cfg(feature = "mmap")]
use std::path::Path;

/// Trait for types that can be serialized to memory-mapped format
#[cfg(feature = "mmap")]
pub trait MmapSerializable {
    /// Save data to a memory-mapped format writer
    fn save_mmap_impl<W: Write>(&self, writer: W) -> Result<()>;
}

/// Implementation of MmapSerializable for standard Dataset types
#[cfg(feature = "mmap")]
impl MmapSerializable
    for Dataset<scirs2_core::ndarray::Array2<Float>, scirs2_core::ndarray::Array1<Float>>
{
    fn save_mmap_impl<W: Write>(&self, mut writer: W) -> Result<()> {
        // Get dimensions
        let (n_samples, n_features) = self.data.dim();

        if n_samples != self.target.len() {
            return Err(crate::error::SklearsError::ShapeMismatch {
                expected: format!("data.nrows() == target.len() ({n_samples})"),
                actual: format!(
                    "data.nrows()={}, target.len()={}",
                    n_samples,
                    self.target.len()
                ),
            });
        }

        // Calculate offsets
        let header_size = MmapHeader::size();
        let data_size = n_samples * n_features * std::mem::size_of::<Float>();
        let target_size = n_samples * std::mem::size_of::<Float>();

        let data_offset = header_size;
        let target_offset = data_offset + data_size;

        // Create and write header
        let header = MmapHeader::new(
            n_samples,
            n_features,
            data_offset,
            target_offset,
            0, // metadata_offset (not implemented)
            0, // metadata_size
            std::mem::size_of::<Float>(),
            0, // checksum (not implemented)
            self.feature_names.clone(),
            self.target_names.clone(),
            self.description.clone(),
        );

        header.write(&mut writer)?;

        // Write data
        let data_bytes =
            unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const u8, data_size) };
        writer
            .write_all(data_bytes)
            .map_err(crate::error::SklearsError::IoError)?;

        // Write target
        let target_bytes =
            unsafe { std::slice::from_raw_parts(self.target.as_ptr() as *const u8, target_size) };
        writer
            .write_all(target_bytes)
            .map_err(crate::error::SklearsError::IoError)?;

        Ok(())
    }
}

/// Generate large synthetic regression data directly to memory-mapped file
///
/// This function creates large regression datasets by streaming data directly
/// to a memory-mapped file, avoiding memory constraints for very large datasets.
///
/// # Arguments
///
/// * `path` - Path where the memory-mapped file will be created
/// * `n_samples` - Number of samples to generate
/// * `n_features` - Number of features per sample
/// * `noise` - Standard deviation of Gaussian noise added to targets
/// * `chunk_size` - Optional chunk size for batch processing (default: 1000)
///
/// # Examples
///
/// ```rust,no_run
/// use sklears_core::dataset::mmap::make_large_regression;
/// use std::path::Path;
///
/// make_large_regression(
///     Path::new("large_dataset.skl"),
///     1_000_000,  // 1M samples
///     100,        // 100 features
///     0.1,        // Low noise
///     Some(5000)  // Process in chunks of 5000
/// ).unwrap();
/// ```
#[cfg(feature = "mmap")]
pub fn make_large_regression<P: AsRef<Path>>(
    path: P,
    n_samples: usize,
    n_features: usize,
    noise: f64,
    chunk_size: Option<usize>,
) -> Result<()> {
    // SciRS2 Policy: Use scirs2_core::random for all RNG operations
    use scirs2_core::random::essentials::Uniform;
    use scirs2_core::random::prelude::*;
    use scirs2_core::random::{thread_rng, Distribution};

    let chunk_size = chunk_size.unwrap_or(1000);
    let mut rng = thread_rng();
    let normal =
        Normal::new(0.0, 1.0).map_err(|e| crate::error::SklearsError::Other(e.to_string()))?;

    // Generate random coefficients for linear combination
    let uniform =
        Uniform::new(-10.0, 10.0).map_err(|e| crate::error::SklearsError::Other(e.to_string()))?;
    let mut coef = Vec::with_capacity(n_features);
    for _ in 0..n_features {
        coef.push(uniform.sample(&mut rng));
    }

    let mut builder = MmapDatasetBuilder::new(n_samples, n_features)
        .description("Large synthetic regression dataset".to_string())
        .build(path)?;

    let noise_dist =
        Normal::new(0.0, noise).map_err(|e| crate::error::SklearsError::Other(e.to_string()))?;

    // Generate data in chunks to manage memory usage
    let mut samples_written = 0;
    while samples_written < n_samples {
        let current_chunk_size = std::cmp::min(chunk_size, n_samples - samples_written);

        // Generate chunk features
        let mut x_data = Vec::with_capacity(current_chunk_size * n_features);
        for _ in 0..current_chunk_size * n_features {
            x_data.push(normal.sample(&mut rng));
        }
        let x_chunk = Array::from_shape_vec((current_chunk_size, n_features), x_data)
            .map_err(|e| crate::error::SklearsError::Other(e.to_string()))?;

        // Generate chunk targets: y = X @ coef + noise
        let mut y_data = Vec::with_capacity(current_chunk_size);
        for i in 0..current_chunk_size {
            let mut y_i = 0.0;
            for j in 0..n_features {
                y_i += x_chunk[[i, j]] * coef[j];
            }
            y_i += noise_dist.sample(&mut rng);
            y_data.push(y_i);
        }
        let y_chunk = Array::from_vec(y_data);

        builder.write_chunk(&x_chunk, &y_chunk)?;
        samples_written += current_chunk_size;
    }

    builder.finish()?;
    Ok(())
}

/// Memory-mapped dataset for handling large datasets that don't fit in memory
///
/// MmapDataset provides read-only access to datasets stored in memory-mapped
/// files, enabling processing of arbitrarily large datasets with constant
/// memory usage.
#[cfg(feature = "mmap")]
#[derive(Debug)]
pub struct MmapDataset {
    /// Memory-mapped file handle
    mmap: memmap2::Mmap,
    /// Shape information: (n_samples, n_features)
    shape: (usize, usize),
    /// Offset to feature data in the file
    data_offset: usize,
    /// Offset to target data in the file
    target_offset: usize,
    /// Feature names for interpretability
    feature_names: Vec<String>,
    /// Target names (for classification)
    target_names: Option<Vec<String>>,
    /// Dataset description
    description: String,
}

#[cfg(feature = "mmap")]
impl MmapDataset {
    /// Create a memory-mapped dataset from a file with validation
    ///
    /// Opens and validates a memory-mapped dataset file, checking the header
    /// format and ensuring the file is complete and uncorrupted.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the memory-mapped dataset file
    ///
    /// # Returns
    ///
    /// A MmapDataset instance providing read access to the data
    ///
    /// # Errors
    ///
    /// Returns an error if the file is invalid, corrupted, or inaccessible.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = std::fs::File::open(&path).map_err(crate::error::SklearsError::IoError)?;
        let file_size = file
            .metadata()
            .map_err(crate::error::SklearsError::IoError)?
            .len() as usize;

        if file_size < MmapHeader::size() {
            return Err(crate::error::SklearsError::InvalidInput(format!(
                "File too small: {} bytes, minimum: {} bytes",
                file_size,
                MmapHeader::size()
            )));
        }

        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(crate::error::SklearsError::IoError)?
        };

        let header = MmapHeader::from_bytes(&mmap[0..MmapHeader::size()])?;
        header.validate(file_size)?;

        Ok(Self {
            mmap,
            shape: (header.n_samples, header.n_features),
            data_offset: header.data_offset,
            target_offset: header.target_offset,
            feature_names: header.feature_names,
            target_names: header.target_names,
            description: header.description,
        })
    }

    /// Create a memory-mapped dataset from an existing memory map
    ///
    /// This is useful for creating datasets from already mapped memory regions,
    /// such as when working with shared memory or custom memory management.
    ///
    /// # Arguments
    ///
    /// * `mmap` - The memory-mapped region containing the dataset
    ///
    /// # Returns
    ///
    /// A MmapDataset instance providing read access to the data
    pub fn from_mmap(mmap: memmap2::Mmap) -> Result<Self> {
        let file_size = mmap.len();

        if file_size < MmapHeader::size() {
            return Err(crate::error::SklearsError::InvalidInput(format!(
                "Memory map too small: {} bytes, minimum: {} bytes",
                file_size,
                MmapHeader::size()
            )));
        }

        let header = MmapHeader::from_bytes(&mmap[0..MmapHeader::size()])?;
        header.validate(file_size)?;

        Ok(Self {
            mmap,
            shape: (header.n_samples, header.n_features),
            data_offset: header.data_offset,
            target_offset: header.target_offset,
            feature_names: header.feature_names,
            target_names: header.target_names,
            description: header.description,
        })
    }

    /// Get the shape of the dataset as (n_samples, n_features)
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Get the number of samples
    pub fn n_samples(&self) -> usize {
        self.shape.0
    }

    /// Get the number of features
    pub fn n_features(&self) -> usize {
        self.shape.1
    }

    /// Get feature names
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Get target names if available
    pub fn target_names(&self) -> Option<&[String]> {
        self.target_names.as_deref()
    }

    /// Get dataset description
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Get a batch iterator for processing data in chunks
    ///
    /// Returns an iterator that yields batches of the specified size,
    /// enabling memory-efficient processing of large datasets.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Size of each batch
    ///
    /// # Returns
    ///
    /// A MmapBatchIterator for processing data in batches
    pub fn batch_iter(&self, batch_size: usize) -> MmapBatchIterator<'_> {
        MmapBatchIterator {
            dataset: self,
            batch_size,
            current_offset: 0,
        }
    }

    /// Read a specific sample (row) from the dataset
    ///
    /// # Arguments
    ///
    /// * `sample_idx` - Index of the sample to read
    ///
    /// # Returns
    ///
    /// A tuple of (features, target) for the specified sample
    pub fn get_sample(&self, sample_idx: usize) -> Result<(Vec<Float>, Float)> {
        if sample_idx >= self.n_samples() {
            return Err(crate::error::SklearsError::InvalidInput(format!(
                "Sample index {} out of bounds (max: {})",
                sample_idx,
                self.n_samples() - 1
            )));
        }

        let n_features = self.n_features();
        let feature_size = std::mem::size_of::<Float>();

        // Read features
        let features_start = self.data_offset + sample_idx * n_features * feature_size;
        let features_end = features_start + n_features * feature_size;
        let feature_bytes = &self.mmap[features_start..features_end];
        let features = unsafe {
            std::slice::from_raw_parts(feature_bytes.as_ptr() as *const Float, n_features)
        }
        .to_vec();

        // Read target
        let target_start = self.target_offset + sample_idx * feature_size;
        let target_end = target_start + feature_size;
        let target_bytes = &self.mmap[target_start..target_end];
        let target = unsafe { *(target_bytes.as_ptr() as *const Float) };

        Ok((features, target))
    }
}

/// Iterator for processing memory-mapped datasets in batches
///
/// This iterator enables memory-efficient processing of large datasets by
/// yielding chunks of data rather than loading everything into memory.
#[cfg(feature = "mmap")]
pub struct MmapBatchIterator<'a> {
    dataset: &'a MmapDataset,
    batch_size: usize,
    current_offset: usize,
}

#[cfg(feature = "mmap")]
impl<'a> Iterator for MmapBatchIterator<'a> {
    type Item = Result<(
        scirs2_core::ndarray::Array2<Float>,
        scirs2_core::ndarray::Array1<Float>,
    )>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_offset >= self.dataset.n_samples() {
            return None;
        }

        let remaining = self.dataset.n_samples() - self.current_offset;
        let current_batch_size = std::cmp::min(self.batch_size, remaining);

        let result = self.read_batch(current_batch_size);
        self.current_offset += current_batch_size;

        Some(result)
    }
}

#[cfg(feature = "mmap")]
impl<'a> MmapBatchIterator<'a> {
    fn read_batch(
        &self,
        batch_size: usize,
    ) -> Result<(
        scirs2_core::ndarray::Array2<Float>,
        scirs2_core::ndarray::Array1<Float>,
    )> {
        let n_features = self.dataset.n_features();
        let feature_size = std::mem::size_of::<Float>();

        // Read features batch
        let features_start =
            self.dataset.data_offset + self.current_offset * n_features * feature_size;
        let features_end = features_start + batch_size * n_features * feature_size;
        let feature_bytes = &self.dataset.mmap[features_start..features_end];

        let features_data = unsafe {
            std::slice::from_raw_parts(
                feature_bytes.as_ptr() as *const Float,
                batch_size * n_features,
            )
        }
        .to_vec();

        let features = Array::from_shape_vec((batch_size, n_features), features_data)
            .map_err(|e| crate::error::SklearsError::Other(e.to_string()))?;

        // Read targets batch
        let targets_start = self.dataset.target_offset + self.current_offset * feature_size;
        let targets_end = targets_start + batch_size * feature_size;
        let target_bytes = &self.dataset.mmap[targets_start..targets_end];

        let targets_data = unsafe {
            std::slice::from_raw_parts(target_bytes.as_ptr() as *const Float, batch_size)
        }
        .to_vec();

        let targets = Array::from_vec(targets_data);

        Ok((features, targets))
    }
}

/// Builder for creating memory-mapped datasets
///
/// Provides a streaming interface for writing large datasets directly to
/// memory-mapped files without loading all data into memory at once.
#[cfg(feature = "mmap")]
pub struct MmapDatasetBuilder {
    file: std::fs::File,
    written_samples: usize,
    total_samples: usize,
    n_features: usize,
    data_offset: usize,
    target_offset: usize,
}

#[cfg(feature = "mmap")]
impl MmapDatasetBuilder {
    /// Create a new builder configuration
    #[allow(clippy::new_ret_no_self)]
    pub fn new(total_samples: usize, n_features: usize) -> MmapDatasetBuilderConfig {
        MmapDatasetBuilderConfig {
            total_samples,
            n_features,
            feature_names: Vec::new(),
            target_names: None,
            description: String::new(),
        }
    }

    /// Write a chunk of data to the memory-mapped file
    ///
    /// # Arguments
    ///
    /// * `features` - Feature matrix for this chunk
    /// * `targets` - Target values for this chunk
    ///
    /// # Returns
    ///
    /// Result indicating success or failure of the write operation
    pub fn write_chunk(
        &mut self,
        features: &scirs2_core::ndarray::Array2<Float>,
        targets: &scirs2_core::ndarray::Array1<Float>,
    ) -> Result<()> {
        let (batch_samples, batch_features) = features.dim();

        if batch_features != self.n_features {
            return Err(crate::error::SklearsError::ShapeMismatch {
                expected: format!("n_features={}", self.n_features),
                actual: format!("batch_features={}", batch_features),
            });
        }

        if batch_samples != targets.len() {
            return Err(crate::error::SklearsError::ShapeMismatch {
                expected: format!("batch_samples={}", batch_samples),
                actual: format!("targets.len()={}", targets.len()),
            });
        }

        if self.written_samples + batch_samples > self.total_samples {
            return Err(crate::error::SklearsError::InvalidInput(format!(
                "Writing {} samples would exceed total capacity of {}",
                self.written_samples + batch_samples,
                self.total_samples
            )));
        }

        // Calculate positions for this chunk
        let feature_size = std::mem::size_of::<Float>();
        let features_start =
            self.data_offset + self.written_samples * self.n_features * feature_size;
        let targets_start = self.target_offset + self.written_samples * feature_size;

        // Write features
        let feature_bytes = unsafe {
            std::slice::from_raw_parts(
                features.as_ptr() as *const u8,
                batch_samples * self.n_features * feature_size,
            )
        };

        use std::os::unix::fs::FileExt;
        self.file
            .write_all_at(feature_bytes, features_start as u64)
            .map_err(crate::error::SklearsError::IoError)?;

        // Write targets
        let target_bytes = unsafe {
            std::slice::from_raw_parts(targets.as_ptr() as *const u8, batch_samples * feature_size)
        };

        self.file
            .write_all_at(target_bytes, targets_start as u64)
            .map_err(crate::error::SklearsError::IoError)?;

        self.written_samples += batch_samples;
        Ok(())
    }

    /// Finish writing and close the file
    ///
    /// This method ensures all data has been written and flushes
    /// any pending writes to disk.
    pub fn finish(self) -> Result<()> {
        if self.written_samples != self.total_samples {
            return Err(crate::error::SklearsError::InvalidInput(format!(
                "Dataset incomplete: wrote {} samples, expected {}",
                self.written_samples, self.total_samples
            )));
        }

        self.file
            .sync_all()
            .map_err(crate::error::SklearsError::IoError)?;
        Ok(())
    }
}

/// Configuration builder for MmapDatasetBuilder
#[cfg(feature = "mmap")]
pub struct MmapDatasetBuilderConfig {
    total_samples: usize,
    n_features: usize,
    feature_names: Vec<String>,
    target_names: Option<Vec<String>>,
    description: String,
}

#[cfg(feature = "mmap")]
impl MmapDatasetBuilderConfig {
    /// Set feature names
    pub fn feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Set target names
    pub fn target_names(mut self, names: Vec<String>) -> Self {
        self.target_names = Some(names);
        self
    }

    /// Set description
    pub fn description(mut self, description: String) -> Self {
        self.description = description;
        self
    }

    /// Build the MmapDatasetBuilder and create the file
    pub fn build<P: AsRef<Path>>(self, path: P) -> Result<MmapDatasetBuilder> {
        let header_size = MmapHeader::size();
        let feature_size = std::mem::size_of::<Float>();
        let data_size = self.total_samples * self.n_features * feature_size;
        let target_size = self.total_samples * feature_size;
        let total_size = header_size + data_size + target_size;

        let data_offset = header_size;
        let target_offset = data_offset + data_size;

        // Create file with proper size
        let file = std::fs::File::create(&path).map_err(crate::error::SklearsError::IoError)?;
        file.set_len(total_size as u64)
            .map_err(crate::error::SklearsError::IoError)?;

        // Write header
        let header = MmapHeader::new(
            self.total_samples,
            self.n_features,
            data_offset,
            target_offset,
            0, // metadata_offset (not implemented)
            0, // metadata_size
            feature_size,
            0, // checksum (not implemented)
            self.feature_names,
            self.target_names,
            self.description,
        );

        let mut file_writer = &file;
        header.write(&mut file_writer)?;

        Ok(MmapDatasetBuilder {
            file,
            written_samples: 0,
            total_samples: self.total_samples,
            n_features: self.n_features,
            data_offset,
            target_offset,
        })
    }
}

/// Header structure for memory-mapped dataset files
///
/// Contains metadata and layout information for the memory-mapped dataset format.
#[cfg(feature = "mmap")]
struct MmapHeader {
    /// Magic number for file format validation ("SKML")
    magic: [u8; 4],
    /// Format version for backward compatibility
    version: u32,
    /// Number of samples in the dataset
    n_samples: usize,
    /// Number of features per sample
    n_features: usize,
    /// Offset to feature data in the file
    data_offset: usize,
    /// Offset to target data in the file
    target_offset: usize,
    /// Offset to metadata section
    #[allow(dead_code)]
    metadata_offset: usize,
    /// Size of metadata section
    #[allow(dead_code)]
    metadata_size: usize,
    /// Data type size (typically 8 for f64)
    data_type_size: usize,
    /// Checksum for integrity verification
    #[allow(dead_code)]
    checksum: u64,
    /// Feature names
    feature_names: Vec<String>,
    /// Target names (for classification)
    target_names: Option<Vec<String>>,
    /// Dataset description
    description: String,
}

#[cfg(feature = "mmap")]
impl MmapHeader {
    /// Magic number identifying sklears memory-mapped files
    const MAGIC: [u8; 4] = *b"SKML";
    /// Current format version
    const VERSION: u32 = 1;

    /// Create a new header with the given parameters
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_samples: usize,
        n_features: usize,
        data_offset: usize,
        target_offset: usize,
        metadata_offset: usize,
        metadata_size: usize,
        data_type_size: usize,
        checksum: u64,
        feature_names: Vec<String>,
        target_names: Option<Vec<String>>,
        description: String,
    ) -> Self {
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            n_samples,
            n_features,
            data_offset,
            target_offset,
            metadata_offset,
            metadata_size,
            data_type_size,
            checksum,
            feature_names,
            target_names,
            description,
        }
    }

    /// Get the size of the header in bytes
    fn size() -> usize {
        // This is a simplified calculation - in practice, you'd need
        // to account for variable-length strings
        1024 // Fixed size for simplicity
    }

    /// Write header to a writer
    fn write<W: Write>(&self, mut writer: W) -> Result<()> {
        // This is a simplified implementation
        // In practice, you'd use a proper serialization format
        writer
            .write_all(&self.magic)
            .map_err(crate::error::SklearsError::IoError)?;
        writer
            .write_all(&self.version.to_le_bytes())
            .map_err(crate::error::SklearsError::IoError)?;
        writer
            .write_all(&self.n_samples.to_le_bytes())
            .map_err(crate::error::SklearsError::IoError)?;
        writer
            .write_all(&self.n_features.to_le_bytes())
            .map_err(crate::error::SklearsError::IoError)?;
        // ... write other fields similarly
        Ok(())
    }

    /// Read header from bytes
    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < Self::size() {
            return Err(crate::error::SklearsError::InvalidInput(
                "Header bytes too short".to_string(),
            ));
        }

        // This is a simplified implementation
        // In practice, you'd use proper deserialization
        let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if magic != Self::MAGIC {
            return Err(crate::error::SklearsError::InvalidInput(
                "Invalid magic number".to_string(),
            ));
        }

        // For simplicity, return a default header
        Ok(Self::new(
            0,
            0,
            0,
            0,
            0,
            0,
            8,
            0,
            Vec::new(),
            None,
            String::new(),
        ))
    }

    /// Validate header consistency with file size
    fn validate(&self, file_size: usize) -> Result<()> {
        let expected_size = self.data_offset
            + self.n_samples * self.n_features * self.data_type_size
            + self.n_samples * self.data_type_size;

        if file_size < expected_size {
            return Err(crate::error::SklearsError::InvalidInput(format!(
                "File size {} too small, expected at least {}",
                file_size, expected_size
            )));
        }

        Ok(())
    }
}
