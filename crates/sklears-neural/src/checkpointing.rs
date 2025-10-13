//! Model checkpointing and state management for neural networks.
//!
//! This module provides functionality to save and load model states, training progress,
//! and optimizer states during training. This is essential for long-running training
//! jobs and experimentation.

use crate::NeuralResult;
use scirs2_core::ndarray::{Array1, Array2};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sklears_core::error::SklearsError;
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

#[cfg(feature = "serde")]
use serde_json;

#[cfg(feature = "serde")]
use bincode;

#[cfg(feature = "serde")]
use chrono;

#[cfg(feature = "mmap")]
use memmap2::{MmapMut, MmapOptions};
#[cfg(feature = "mmap")]
use std::io::{Seek, SeekFrom};

/// Format for saving checkpoints
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckpointFormat {
    /// JSON format (human-readable, larger files)
    Json,
    /// Binary format (compact, faster I/O)
    Binary,
    /// Custom format with compression
    Compressed,
    /// Memory-mapped format for large models (efficient random access)
    #[cfg(feature = "mmap")]
    MemoryMapped,
}

impl Default for CheckpointFormat {
    fn default() -> Self {
        CheckpointFormat::Binary
    }
}

/// Model weights and biases
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ModelWeights {
    /// Weight matrices for each layer
    pub weights: Vec<Array2<f64>>,
    /// Bias vectors for each layer
    pub biases: Vec<Array1<f64>>,
    /// Layer names (optional)
    pub layer_names: Option<Vec<String>>,
}

impl ModelWeights {
    /// Create new model weights
    pub fn new(weights: Vec<Array2<f64>>, biases: Vec<Array1<f64>>) -> Self {
        Self {
            weights,
            biases,
            layer_names: None,
        }
    }

    /// Create model weights with layer names
    pub fn with_names(
        weights: Vec<Array2<f64>>,
        biases: Vec<Array1<f64>>,
        layer_names: Vec<String>,
    ) -> Self {
        Self {
            weights,
            biases,
            layer_names: Some(layer_names),
        }
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.weights.len().min(self.biases.len())
    }

    /// Validate that weights and biases are consistent
    pub fn validate(&self) -> NeuralResult<()> {
        if self.weights.len() != self.biases.len() {
            return Err(SklearsError::InvalidParameter {
                name: "weights_biases".to_string(),
                reason: format!(
                    "Number of weight matrices ({}) doesn't match number of bias vectors ({})",
                    self.weights.len(),
                    self.biases.len()
                ),
            });
        }

        for (i, (weight, bias)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            if weight.ncols() != bias.len() {
                return Err(SklearsError::ShapeMismatch {
                    expected: format!("weight.ncols()={}", weight.ncols()),
                    actual: format!("bias.len()={} at layer {}", bias.len(), i),
                });
            }
        }

        if let Some(ref names) = self.layer_names {
            if names.len() != self.num_layers() {
                return Err(SklearsError::InvalidParameter {
                    name: "layer_names".to_string(),
                    reason: format!(
                        "Number of layer names ({}) doesn't match number of layers ({})",
                        names.len(),
                        self.num_layers()
                    ),
                });
            }
        }

        Ok(())
    }

    /// Get total number of parameters
    pub fn total_parameters(&self) -> usize {
        let weight_params: usize = self.weights.iter().map(|w| w.len()).sum();
        let bias_params: usize = self.biases.iter().map(|b| b.len()).sum();
        weight_params + bias_params
    }
}

/// Optimizer state for resuming training
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OptimizerState {
    /// Optimizer type identifier
    pub optimizer_type: String,
    /// Current learning rate
    pub learning_rate: f64,
    /// Momentum states (for momentum-based optimizers)
    pub momentum_states: Option<Vec<Array2<f64>>>,
    /// Velocity states (for Adam-like optimizers)
    pub velocity_states: Option<Vec<Array2<f64>>>,
    /// Bias correction terms
    pub bias_correction: Option<HashMap<String, f64>>,
    /// Additional optimizer-specific parameters
    pub parameters: HashMap<String, f64>,
    /// Step count
    pub step_count: usize,
}

impl OptimizerState {
    /// Create new optimizer state
    pub fn new(optimizer_type: String, learning_rate: f64) -> Self {
        Self {
            optimizer_type,
            learning_rate,
            momentum_states: None,
            velocity_states: None,
            bias_correction: None,
            parameters: HashMap::new(),
            step_count: 0,
        }
    }

    /// Add momentum states
    pub fn with_momentum(mut self, momentum_states: Vec<Array2<f64>>) -> Self {
        self.momentum_states = Some(momentum_states);
        self
    }

    /// Add velocity states
    pub fn with_velocity(mut self, velocity_states: Vec<Array2<f64>>) -> Self {
        self.velocity_states = Some(velocity_states);
        self
    }

    /// Add bias correction terms
    pub fn with_bias_correction(mut self, bias_correction: HashMap<String, f64>) -> Self {
        self.bias_correction = Some(bias_correction);
        self
    }

    /// Add parameter
    pub fn with_parameter(mut self, key: String, value: f64) -> Self {
        self.parameters.insert(key, value);
        self
    }

    /// Set step count
    pub fn with_step_count(mut self, step_count: usize) -> Self {
        self.step_count = step_count;
        self
    }
}

/// Training metrics and history
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TrainingHistory {
    /// Training loss history
    pub train_loss: Vec<f64>,
    /// Validation loss history
    pub val_loss: Option<Vec<f64>>,
    /// Training accuracy history
    pub train_accuracy: Option<Vec<f64>>,
    /// Validation accuracy history
    pub val_accuracy: Option<Vec<f64>>,
    /// Learning rate history
    pub learning_rates: Vec<f64>,
    /// Epoch numbers
    pub epochs: Vec<usize>,
    /// Additional custom metrics
    pub custom_metrics: HashMap<String, Vec<f64>>,
}

impl TrainingHistory {
    /// Create new training history
    pub fn new() -> Self {
        Self {
            train_loss: Vec::new(),
            val_loss: None,
            train_accuracy: None,
            val_accuracy: None,
            learning_rates: Vec::new(),
            epochs: Vec::new(),
            custom_metrics: HashMap::new(),
        }
    }

    /// Add training step
    pub fn add_epoch(&mut self, epoch: usize, train_loss: f64, learning_rate: f64) {
        self.epochs.push(epoch);
        self.train_loss.push(train_loss);
        self.learning_rates.push(learning_rate);
    }

    /// Add validation metrics
    pub fn add_validation(&mut self, val_loss: f64, val_accuracy: Option<f64>) {
        if self.val_loss.is_none() {
            self.val_loss = Some(Vec::new());
        }
        self.val_loss.as_mut().unwrap().push(val_loss);

        if let Some(acc) = val_accuracy {
            if self.val_accuracy.is_none() {
                self.val_accuracy = Some(Vec::new());
            }
            self.val_accuracy.as_mut().unwrap().push(acc);
        }
    }

    /// Add custom metric
    pub fn add_custom_metric(&mut self, name: &str, value: f64) {
        self.custom_metrics
            .entry(name.to_string())
            .or_insert_with(Vec::new)
            .push(value);
    }

    /// Get best epoch (minimum validation loss)
    pub fn best_epoch(&self) -> Option<usize> {
        self.val_loss.as_ref().and_then(|losses| {
            losses
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| self.epochs[idx])
        })
    }

    /// Check if training is improving
    pub fn is_improving(&self, patience: usize) -> bool {
        if let Some(ref val_losses) = self.val_loss {
            if val_losses.len() < patience {
                return true; // Too early to tell
            }

            let recent_best = val_losses[val_losses.len() - patience..]
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
            let overall_best = val_losses.iter().fold(f64::INFINITY, |a, &b| a.min(b));

            recent_best <= overall_best
        } else {
            // Use training loss if no validation loss available
            if self.train_loss.len() < patience {
                return true;
            }

            let recent_best = self.train_loss[self.train_loss.len() - patience..]
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
            let overall_best = self.train_loss.iter().fold(f64::INFINITY, |a, &b| a.min(b));

            recent_best <= overall_best
        }
    }
}

impl Default for TrainingHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete model checkpoint
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ModelCheckpoint {
    /// Model architecture identifier
    pub model_type: String,
    /// Model weights and biases
    pub weights: ModelWeights,
    /// Optimizer state
    pub optimizer_state: Option<OptimizerState>,
    /// Training history
    pub training_history: TrainingHistory,
    /// Current epoch
    pub current_epoch: usize,
    /// Model hyperparameters
    #[cfg(feature = "serde")]
    pub hyperparameters: HashMap<String, serde_json::Value>,
    /// Checkpoint metadata
    pub metadata: CheckpointMetadata,
    /// Model version
    pub version: String,
}

/// Checkpoint metadata
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CheckpointMetadata {
    /// Creation timestamp
    #[cfg(feature = "serde")]
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Training duration so far
    pub training_duration: Option<std::time::Duration>,
    /// Git commit hash (if available)
    pub git_commit: Option<String>,
    /// Environment information
    pub environment: HashMap<String, String>,
    /// Custom tags
    pub tags: Vec<String>,
    /// Description
    pub description: Option<String>,
}

impl Default for CheckpointMetadata {
    fn default() -> Self {
        let mut environment = HashMap::new();
        environment.insert(
            "sklears_version".to_string(),
            env!("CARGO_PKG_VERSION").to_string(),
        );

        #[cfg(feature = "serde")]
        {
            Self {
                created_at: chrono::Utc::now(),
                training_duration: None,
                git_commit: None,
                environment,
                tags: Vec::new(),
                description: None,
            }
        }
        #[cfg(not(feature = "serde"))]
        {
            Self {
                training_duration: None,
                git_commit: None,
                environment,
                tags: Vec::new(),
                description: None,
            }
        }
    }
}

impl ModelCheckpoint {
    /// Create new model checkpoint
    pub fn new(model_type: String, weights: ModelWeights, current_epoch: usize) -> Self {
        #[cfg(feature = "serde")]
        {
            Self {
                model_type,
                weights,
                optimizer_state: None,
                training_history: TrainingHistory::new(),
                current_epoch,
                hyperparameters: HashMap::new(),
                metadata: CheckpointMetadata::default(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            }
        }
        #[cfg(not(feature = "serde"))]
        {
            Self {
                model_type,
                weights,
                optimizer_state: None,
                training_history: TrainingHistory::new(),
                current_epoch,
                metadata: CheckpointMetadata::default(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            }
        }
    }

    /// Add optimizer state
    pub fn with_optimizer(mut self, optimizer_state: OptimizerState) -> Self {
        self.optimizer_state = Some(optimizer_state);
        self
    }

    /// Add training history
    pub fn with_history(mut self, training_history: TrainingHistory) -> Self {
        self.training_history = training_history;
        self
    }

    /// Add hyperparameters
    #[cfg(feature = "serde")]
    pub fn with_hyperparameters(
        mut self,
        hyperparameters: HashMap<String, serde_json::Value>,
    ) -> Self {
        self.hyperparameters = hyperparameters;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, metadata: CheckpointMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Validate checkpoint consistency
    pub fn validate(&self) -> NeuralResult<()> {
        self.weights.validate()?;

        // Validate optimizer state consistency
        if let Some(ref opt_state) = self.optimizer_state {
            if let Some(ref momentum) = opt_state.momentum_states {
                if momentum.len() != self.weights.num_layers() {
                    return Err(SklearsError::InvalidParameter {
                        name: "momentum_states".to_string(),
                        reason: "Momentum states count doesn't match number of layers".to_string(),
                    });
                }
            }

            if let Some(ref velocity) = opt_state.velocity_states {
                if velocity.len() != self.weights.num_layers() {
                    return Err(SklearsError::InvalidParameter {
                        name: "velocity_states".to_string(),
                        reason: "Velocity states count doesn't match number of layers".to_string(),
                    });
                }
            }
        }

        Ok(())
    }
}

/// Checkpoint manager for saving and loading model states
pub struct CheckpointManager {
    /// Base directory for checkpoints
    pub checkpoint_dir: std::path::PathBuf,
    /// Default checkpoint format
    pub format: CheckpointFormat,
    /// Maximum number of checkpoints to keep
    pub max_checkpoints: Option<usize>,
    /// Checkpoint naming pattern
    pub naming_pattern: String,
}

impl CheckpointManager {
    /// Create new checkpoint manager
    pub fn new<P: AsRef<Path>>(checkpoint_dir: P) -> NeuralResult<Self> {
        let checkpoint_dir = checkpoint_dir.as_ref().to_path_buf();

        // Create directory if it doesn't exist
        if !checkpoint_dir.exists() {
            fs::create_dir_all(&checkpoint_dir).map_err(|e| SklearsError::InvalidParameter {
                name: "checkpoint_dir".to_string(),
                reason: format!("Failed to create checkpoint directory: {}", e),
            })?;
        }

        Ok(Self {
            checkpoint_dir,
            format: CheckpointFormat::default(),
            max_checkpoints: Some(10),
            naming_pattern: "checkpoint_epoch_{:04d}".to_string(),
        })
    }

    /// Set checkpoint format
    pub fn with_format(mut self, format: CheckpointFormat) -> Self {
        self.format = format;
        self
    }

    /// Set maximum number of checkpoints to keep
    pub fn with_max_checkpoints(mut self, max_checkpoints: Option<usize>) -> Self {
        self.max_checkpoints = max_checkpoints;
        self
    }

    /// Set naming pattern
    pub fn with_naming_pattern(mut self, pattern: String) -> Self {
        self.naming_pattern = pattern;
        self
    }

    /// Save checkpoint
    #[cfg(feature = "serde")]
    pub fn save_checkpoint(
        &self,
        checkpoint: &ModelCheckpoint,
    ) -> NeuralResult<std::path::PathBuf> {
        checkpoint.validate()?;

        let filename = self
            .naming_pattern
            .replace("{:04d}", &format!("{:04}", checkpoint.current_epoch));
        let file_path = match self.format {
            CheckpointFormat::Json => {
                let path = self.checkpoint_dir.join(format!("{}.json", filename));
                let json_data = serde_json::to_string_pretty(checkpoint).map_err(|e| {
                    SklearsError::InvalidParameter {
                        name: "checkpoint".to_string(),
                        reason: format!("Failed to serialize checkpoint to JSON: {}", e),
                    }
                })?;
                fs::write(&path, json_data).map_err(|e| SklearsError::InvalidParameter {
                    name: "file_path".to_string(),
                    reason: format!("Failed to write checkpoint file: {}", e),
                })?;
                path
            }
            CheckpointFormat::Binary => {
                let path = self.checkpoint_dir.join(format!("{}.bin", filename));
                let binary_data =
                    bincode::serialize(checkpoint).map_err(|e| SklearsError::InvalidParameter {
                        name: "checkpoint".to_string(),
                        reason: format!("Failed to serialize checkpoint to binary: {}", e),
                    })?;
                fs::write(&path, binary_data).map_err(|e| SklearsError::InvalidParameter {
                    name: "file_path".to_string(),
                    reason: format!("Failed to write checkpoint file: {}", e),
                })?;
                path
            }
            CheckpointFormat::Compressed => {
                let path = self.checkpoint_dir.join(format!("{}.bin.gz", filename));
                let binary_data =
                    bincode::serialize(checkpoint).map_err(|e| SklearsError::InvalidParameter {
                        name: "checkpoint".to_string(),
                        reason: format!("Failed to serialize checkpoint: {}", e),
                    })?;

                use flate2::write::GzEncoder;
                use flate2::Compression;
                use std::io::Write;

                let file = fs::File::create(&path).map_err(|e| SklearsError::InvalidParameter {
                    name: "file_path".to_string(),
                    reason: format!("Failed to create checkpoint file: {}", e),
                })?;
                let mut encoder = GzEncoder::new(file, Compression::default());
                encoder
                    .write_all(&binary_data)
                    .map_err(|e| SklearsError::InvalidParameter {
                        name: "compression".to_string(),
                        reason: format!("Failed to compress checkpoint: {}", e),
                    })?;
                encoder
                    .finish()
                    .map_err(|e| SklearsError::InvalidParameter {
                        name: "compression".to_string(),
                        reason: format!("Failed to finish compression: {}", e),
                    })?;
                path
            }
            #[cfg(feature = "mmap")]
            CheckpointFormat::MemoryMapped => {
                let path = self.checkpoint_dir.join(format!("{}.mmap", filename));
                self.save_memory_mapped_checkpoint(checkpoint, &path)?;
                path
            }
        };

        // Clean up old checkpoints if max_checkpoints is set
        if let Some(max_checkpoints) = self.max_checkpoints {
            self.cleanup_old_checkpoints(max_checkpoints)?;
        }

        Ok(file_path)
    }

    /// Load checkpoint
    #[cfg(feature = "serde")]
    pub fn load_checkpoint<P: AsRef<Path>>(&self, file_path: P) -> NeuralResult<ModelCheckpoint> {
        let file_path = file_path.as_ref();
        let extension = file_path.extension().and_then(|s| s.to_str()).unwrap_or("");

        let checkpoint: ModelCheckpoint = match extension {
            "json" => {
                let json_data =
                    fs::read_to_string(file_path).map_err(|e| SklearsError::InvalidParameter {
                        name: "file_path".to_string(),
                        reason: format!("Failed to read checkpoint file: {}", e),
                    })?;
                serde_json::from_str(&json_data).map_err(|e| SklearsError::InvalidParameter {
                    name: "checkpoint".to_string(),
                    reason: format!("Failed to deserialize JSON checkpoint: {}", e),
                })?
            }
            "bin" => {
                let binary_data =
                    fs::read(file_path).map_err(|e| SklearsError::InvalidParameter {
                        name: "file_path".to_string(),
                        reason: format!("Failed to read checkpoint file: {}", e),
                    })?;
                bincode::deserialize(&binary_data).map_err(|e| SklearsError::InvalidParameter {
                    name: "checkpoint".to_string(),
                    reason: format!("Failed to deserialize binary checkpoint: {}", e),
                })?
            }
            "gz" => {
                use flate2::read::GzDecoder;
                use std::io::Read;

                let file =
                    fs::File::open(file_path).map_err(|e| SklearsError::InvalidParameter {
                        name: "file_path".to_string(),
                        reason: format!("Failed to open checkpoint file: {}", e),
                    })?;
                let mut decoder = GzDecoder::new(file);
                let mut binary_data = Vec::new();
                decoder.read_to_end(&mut binary_data).map_err(|e| {
                    SklearsError::InvalidParameter {
                        name: "decompression".to_string(),
                        reason: format!("Failed to decompress checkpoint: {}", e),
                    }
                })?;
                bincode::deserialize(&binary_data).map_err(|e| SklearsError::InvalidParameter {
                    name: "checkpoint".to_string(),
                    reason: format!("Failed to deserialize compressed checkpoint: {}", e),
                })?
            }
            #[cfg(feature = "mmap")]
            "mmap" => self.load_memory_mapped_checkpoint(file_path)?,
            _ => {
                return Err(SklearsError::InvalidParameter {
                    name: "file_extension".to_string(),
                    reason: format!("Unsupported checkpoint file extension: {}", extension),
                });
            }
        };

        checkpoint.validate()?;
        Ok(checkpoint)
    }

    /// List available checkpoints
    pub fn list_checkpoints(&self) -> NeuralResult<Vec<std::path::PathBuf>> {
        let mut checkpoints = Vec::new();

        let entries =
            fs::read_dir(&self.checkpoint_dir).map_err(|e| SklearsError::InvalidParameter {
                name: "checkpoint_dir".to_string(),
                reason: format!("Failed to read checkpoint directory: {}", e),
            })?;

        for entry in entries {
            let entry = entry.map_err(|e| SklearsError::InvalidParameter {
                name: "directory_entry".to_string(),
                reason: format!("Failed to read directory entry: {}", e),
            })?;

            let path = entry.path();
            if path.is_file() {
                if let Some(extension) = path.extension().and_then(|s| s.to_str()) {
                    if matches!(extension, "json" | "bin" | "gz") {
                        checkpoints.push(path);
                    }
                }
            }
        }

        // Sort by modification time (newest first)
        checkpoints.sort_by_key(|path| {
            fs::metadata(path)
                .and_then(|metadata| metadata.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
        });
        checkpoints.reverse();

        Ok(checkpoints)
    }

    /// Get latest checkpoint
    pub fn get_latest_checkpoint(&self) -> NeuralResult<Option<std::path::PathBuf>> {
        let checkpoints = self.list_checkpoints()?;
        Ok(checkpoints.into_iter().next())
    }

    /// Clean up old checkpoints
    fn cleanup_old_checkpoints(&self, max_checkpoints: usize) -> NeuralResult<()> {
        let checkpoints = self.list_checkpoints()?;

        if checkpoints.len() > max_checkpoints {
            for checkpoint_path in checkpoints.into_iter().skip(max_checkpoints) {
                fs::remove_file(&checkpoint_path).map_err(|e| SklearsError::InvalidParameter {
                    name: "cleanup".to_string(),
                    reason: format!("Failed to remove old checkpoint: {}", e),
                })?;
            }
        }

        Ok(())
    }

    /// Save checkpoint using memory-mapped storage
    #[cfg(feature = "mmap")]
    fn save_memory_mapped_checkpoint(
        &self,
        checkpoint: &ModelCheckpoint,
        path: &Path,
    ) -> NeuralResult<()> {
        use std::mem;

        // First, serialize the checkpoint to binary to get the size
        let binary_data =
            bincode::serialize(checkpoint).map_err(|e| SklearsError::InvalidParameter {
                name: "checkpoint".to_string(),
                reason: format!("Failed to serialize checkpoint: {}", e),
            })?;

        // Create a file with the required size and proper permissions
        let mut file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(|e| SklearsError::InvalidParameter {
                name: "file_path".to_string(),
                reason: format!("Failed to create memory-mapped file: {}", e),
            })?;

        // Write the data first to set proper file size
        file.write_all(&binary_data)
            .map_err(|e| SklearsError::InvalidParameter {
                name: "file_write".to_string(),
                reason: format!("Failed to write data to file: {}", e),
            })?;
        file.flush().map_err(|e| SklearsError::InvalidParameter {
            name: "file_flush".to_string(),
            reason: format!("Failed to flush file: {}", e),
        })?;

        // Optionally create a read-only memory map for verification
        // (In practice, memory-mapped files are more useful for reading large files)

        Ok(())
    }

    /// Load checkpoint from memory-mapped storage
    #[cfg(feature = "mmap")]
    fn load_memory_mapped_checkpoint(&self, path: &Path) -> NeuralResult<ModelCheckpoint> {
        use memmap2::Mmap;

        // Open file
        let file = fs::File::open(path).map_err(|e| SklearsError::InvalidParameter {
            name: "file_path".to_string(),
            reason: format!("Failed to open memory-mapped file: {}", e),
        })?;

        // Create read-only memory map
        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| SklearsError::InvalidParameter {
                name: "memory_map".to_string(),
                reason: format!("Failed to create memory map: {}", e),
            })?
        };

        // Deserialize from memory-mapped data
        let checkpoint: ModelCheckpoint =
            bincode::deserialize(&mmap).map_err(|e| SklearsError::InvalidParameter {
                name: "checkpoint".to_string(),
                reason: format!("Failed to deserialize memory-mapped checkpoint: {}", e),
            })?;

        Ok(checkpoint)
    }
}

/// Trait for models that support checkpointing
pub trait Checkpointable {
    /// Save current model state to checkpoint
    fn to_checkpoint(&self, current_epoch: usize) -> NeuralResult<ModelCheckpoint>;

    /// Load model state from checkpoint
    fn from_checkpoint(&mut self, checkpoint: &ModelCheckpoint) -> NeuralResult<()>;

    /// Get model type identifier
    fn model_type(&self) -> String;
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_model_weights_creation() {
        let weights = vec![
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
            Array2::from_shape_vec((3, 1), vec![7.0, 8.0, 9.0]).unwrap(),
        ];
        let biases = vec![
            Array1::from_vec(vec![0.1, 0.2, 0.3]),
            Array1::from_vec(vec![0.4]),
        ];

        let model_weights = ModelWeights::new(weights, biases);
        assert_eq!(model_weights.num_layers(), 2);
        assert_eq!(model_weights.total_parameters(), 13); // weights: (2×3) + (3×1) = 9, biases: 3 + 1 = 4, total = 13
        assert!(model_weights.validate().is_ok());
    }

    #[test]
    fn test_model_weights_validation() {
        let weights =
            vec![Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()];
        let biases = vec![
            Array1::from_vec(vec![0.1, 0.2]), // Wrong size!
        ];

        let model_weights = ModelWeights::new(weights, biases);
        assert!(model_weights.validate().is_err());
    }

    #[test]
    fn test_optimizer_state() {
        let opt_state = OptimizerState::new("Adam".to_string(), 0.001)
            .with_step_count(100)
            .with_parameter("beta1".to_string(), 0.9)
            .with_parameter("beta2".to_string(), 0.999);

        assert_eq!(opt_state.optimizer_type, "Adam");
        assert_abs_diff_eq!(opt_state.learning_rate, 0.001, epsilon = 1e-10);
        assert_eq!(opt_state.step_count, 100);
        assert_abs_diff_eq!(opt_state.parameters["beta1"], 0.9, epsilon = 1e-10);
    }

    #[test]
    fn test_training_history() {
        let mut history = TrainingHistory::new();

        history.add_epoch(0, 1.0, 0.01);
        history.add_validation(0.9, Some(0.85));

        history.add_epoch(1, 0.8, 0.01);
        history.add_validation(0.7, Some(0.88));

        history.add_custom_metric("f1_score", 0.82);
        history.add_custom_metric("f1_score", 0.86);

        assert_eq!(history.epochs.len(), 2);
        assert_eq!(history.train_loss, vec![1.0, 0.8]);
        assert_eq!(history.val_loss, Some(vec![0.9, 0.7]));
        assert_eq!(history.best_epoch(), Some(1)); // Epoch 1 has lowest val loss
        assert!(history.is_improving(2));
        assert_eq!(history.custom_metrics["f1_score"], vec![0.82, 0.86]);
    }

    #[test]
    fn test_checkpoint_creation() {
        let weights =
            vec![Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()];
        let biases = vec![Array1::from_vec(vec![0.1, 0.2, 0.3])];
        let model_weights = ModelWeights::new(weights, biases);

        let checkpoint = ModelCheckpoint::new("MLP".to_string(), model_weights, 10);

        assert_eq!(checkpoint.model_type, "MLP");
        assert_eq!(checkpoint.current_epoch, 10);
        assert!(checkpoint.validate().is_ok());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_checkpoint_manager() {
        use std::env;

        let temp_dir = env::temp_dir().join("sklears_checkpoint_test");
        let manager = CheckpointManager::new(&temp_dir).unwrap();

        assert!(temp_dir.exists());
        assert_eq!(manager.checkpoint_dir, temp_dir);

        // Clean up
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    #[cfg(all(feature = "mmap", feature = "serde"))]
    fn test_memory_mapped_checkpoint() {
        use std::env;

        let temp_dir = env::temp_dir().join("sklears_mmap_test");
        let checkpoint_manager = CheckpointManager::new(&temp_dir)
            .unwrap()
            .with_format(CheckpointFormat::MemoryMapped);

        // Create test checkpoint
        let weights = vec![
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
            Array2::from_shape_vec((3, 1), vec![7.0, 8.0, 9.0]).unwrap(),
        ];
        let biases = vec![
            Array1::from_vec(vec![0.1, 0.2, 0.3]),
            Array1::from_vec(vec![0.4]),
        ];
        let model_weights = ModelWeights::new(weights, biases);

        let optimizer_state = OptimizerState::new("Adam".to_string(), 0.001).with_step_count(100);

        let checkpoint = ModelCheckpoint::new("MLP".to_string(), model_weights, 42)
            .with_optimizer(optimizer_state);

        // Save checkpoint
        let saved_path = checkpoint_manager.save_checkpoint(&checkpoint).unwrap();
        assert!(saved_path.exists());
        assert_eq!(saved_path.extension().unwrap(), "mmap");

        // Load checkpoint
        let loaded_checkpoint = checkpoint_manager.load_checkpoint(&saved_path).unwrap();

        // Verify loaded checkpoint
        assert_eq!(loaded_checkpoint.current_epoch, 42);
        assert_eq!(loaded_checkpoint.weights.num_layers(), 2);
        assert_eq!(loaded_checkpoint.weights.total_parameters(), 13);

        if let Some(ref opt_state) = loaded_checkpoint.optimizer_state {
            assert_eq!(opt_state.optimizer_type, "Adam");
            assert_abs_diff_eq!(opt_state.learning_rate, 0.001, epsilon = 1e-10);
            assert_eq!(opt_state.step_count, 100);
        } else {
            panic!("Optimizer state should be present");
        }

        // Verify weight values
        assert_abs_diff_eq!(
            loaded_checkpoint.weights.weights[0][[0, 0]],
            1.0,
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            loaded_checkpoint.weights.weights[0][[1, 2]],
            6.0,
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(loaded_checkpoint.weights.biases[0][1], 0.2, epsilon = 1e-10);

        // Clean up
        let _ = fs::remove_dir_all(&temp_dir);
    }
}
