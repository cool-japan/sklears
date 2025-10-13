//! Model compression techniques for large ensembles
//!
//! This module provides various compression strategies to reduce memory usage
//! and improve inference speed for large ensemble models, including knowledge
//! distillation, pruning, quantization, and ensemble reduction techniques.

// ❌ REMOVED: rand_chacha::rand_core - use scirs2_core::random instead
// ❌ REMOVED: rand_chacha::scirs2_core::random::rngs::StdRng - use scirs2_core::random instead
use scirs2_core::ndarray::{Array1, Array2};
#[allow(unused_imports)]
use scirs2_core::random::SeedableRng;
use sklears_core::error::{Result, SklearsError};
use sklears_core::prelude::Predict;
use sklears_core::traits::{Estimator, PredictProba};
use sklears_core::types::{Float, Int};
use std::collections::HashMap;

/// Helper function to generate random f64 from scirs2_core::random::RngCore
fn gen_f64(rng: &mut impl scirs2_core::random::RngCore) -> f64 {
    let mut bytes = [0u8; 8];
    rng.fill_bytes(&mut bytes);
    f64::from_le_bytes(bytes) / f64::from_le_bytes([255u8; 8])
}

/// Helper function to generate random value in range from scirs2_core::random::RngCore
fn gen_range_usize(
    rng: &mut impl scirs2_core::random::RngCore,
    range: std::ops::Range<usize>,
) -> usize {
    let mut bytes = [0u8; 8];
    rng.fill_bytes(&mut bytes);
    let val = u64::from_le_bytes(bytes);
    range.start + (val as usize % (range.end - range.start))
}

/// Compression strategy enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionStrategy {
    /// Knowledge distillation - train a smaller model to mimic ensemble
    KnowledgeDistillation,
    /// Ensemble pruning - remove redundant or weak models
    EnsemblePruning,
    /// Model quantization - reduce precision of model parameters
    Quantization,
    /// Weight sharing - share parameters across models
    WeightSharing,
    /// Low-rank approximation - approximate weight matrices
    LowRankApproximation,
    /// Sparse representation - remove near-zero weights
    SparseRepresentation,
    /// Hierarchical compression - compress at multiple levels
    HierarchicalCompression,
    /// Bayesian optimization for ensemble size optimization
    BayesianOptimization,
}

/// Compression configuration
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Primary compression strategy
    pub strategy: CompressionStrategy,
    /// Target compression ratio (0.0 to 1.0)
    pub compression_ratio: Float,
    /// Quality threshold - minimum acceptable performance
    pub quality_threshold: Float,
    /// Number of bits for quantization
    pub quantization_bits: Option<u8>,
    /// Sparsity level for sparse representation
    pub sparsity_level: Option<Float>,
    /// Rank for low-rank approximation
    pub low_rank: Option<usize>,
    /// Distillation temperature for knowledge distillation
    pub distillation_temperature: Float,
    /// Number of distillation epochs
    pub distillation_epochs: usize,
    /// Learning rate for distillation
    pub distillation_lr: Float,
    /// Enable progressive compression
    pub progressive_compression: bool,
    /// Bayesian optimization configuration
    pub bayes_opt_n_calls: usize,
    pub bayes_opt_n_initial: usize,
    pub bayes_opt_acquisition_kappa: Float,
    pub bayes_opt_random_state: Option<u64>,
    /// Performance-cost trade-off weight
    pub performance_cost_trade_off: Float,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            strategy: CompressionStrategy::EnsemblePruning,
            compression_ratio: 0.5,
            quality_threshold: 0.95,
            quantization_bits: Some(8),
            sparsity_level: Some(0.1),
            low_rank: Some(64),
            distillation_temperature: 3.0,
            distillation_epochs: 100,
            distillation_lr: 0.01,
            progressive_compression: false,
            bayes_opt_n_calls: 50,
            bayes_opt_n_initial: 10,
            bayes_opt_acquisition_kappa: 2.576,
            bayes_opt_random_state: None,
            performance_cost_trade_off: 0.7,
        }
    }
}

/// Compression statistics
#[derive(Debug, Clone)]
pub struct CompressionStats {
    /// Original model size (bytes)
    pub original_size_bytes: usize,
    /// Compressed model size (bytes)
    pub compressed_size_bytes: usize,
    /// Compression ratio achieved
    pub compression_ratio: Float,
    /// Performance before compression
    pub original_accuracy: Float,
    /// Performance after compression
    pub compressed_accuracy: Float,
    /// Performance degradation
    pub accuracy_loss: Float,
    /// Compression time (seconds)
    pub compression_time_secs: Float,
    /// Inference speedup factor
    pub speedup_factor: Float,
    /// Memory reduction factor
    pub memory_reduction_factor: Float,
}

/// Ensemble compressor
pub struct EnsembleCompressor {
    config: CompressionConfig,
    stats: Option<CompressionStats>,
}

/// Compressed ensemble representation
#[derive(Debug, Clone)]
pub struct CompressedEnsemble<T> {
    /// Compressed models
    pub models: Vec<T>,
    /// Compression metadata
    pub metadata: CompressionMetadata,
    /// Model weights for voting/averaging
    pub weights: Option<Array1<Float>>,
}

/// Compression metadata
#[derive(Debug, Clone)]
pub struct CompressionMetadata {
    /// Original number of models
    pub original_count: usize,
    /// Compressed number of models
    pub compressed_count: usize,
    /// Compression strategy used
    pub strategy: CompressionStrategy,
    /// Model mapping (original -> compressed indices)
    pub model_mapping: HashMap<usize, usize>,
    /// Quantization parameters
    pub quantization_params: Option<QuantizationParams>,
    /// Sparsity information
    pub sparsity_info: Option<SparsityInfo>,
}

/// Quantization parameters
#[derive(Debug, Clone)]
pub struct QuantizationParams {
    /// Number of bits per parameter
    pub bits: u8,
    /// Scale factor for quantization
    pub scale: Float,
    /// Zero point for quantization
    pub zero_point: Int,
    /// Min and max values for clipping
    pub min_val: Float,
    pub max_val: Float,
}

/// Sparsity information
#[derive(Debug, Clone)]
pub struct SparsityInfo {
    /// Sparsity level (fraction of zero weights)
    pub sparsity_level: Float,
    /// Number of non-zero parameters
    pub non_zero_params: usize,
    /// Total number of parameters
    pub total_params: usize,
    /// Sparse representation indices
    pub sparse_indices: Vec<usize>,
}

/// Knowledge distillation trainer
pub struct KnowledgeDistillationTrainer {
    temperature: Float,
    alpha: Float, // Weight for distillation loss
    beta: Float,  // Weight for ground truth loss
}

/// Ensemble pruning algorithm
pub struct EnsemblePruner {
    /// Diversity threshold for pruning
    diversity_threshold: Float,
    /// Performance threshold
    performance_threshold: Float,
    /// Correlation threshold
    correlation_threshold: Float,
}

/// Bayesian optimization for ensemble size selection
#[derive(Debug)]
pub struct BayesianEnsembleOptimizer {
    /// Configuration for optimization
    config: CompressionConfig,
    /// Gaussian Process for surrogate modeling
    gp: SimpleGaussianProcess,
    /// Random number generator
    rng: scirs2_core::random::rngs::StdRng,
    /// Evaluation history (ensemble_size, accuracy, cost, objective)
    evaluations: Vec<(usize, Float, Float, Float)>,
    /// Best configuration found
    best_config: Option<(usize, Float)>,
}

/// Simple Gaussian Process for Bayesian optimization
#[derive(Debug)]
struct SimpleGaussianProcess {
    x_train: Array2<Float>,
    y_train: Array1<Float>,
    noise_level: Float,
    length_scale: Float,
    signal_variance: Float,
}

/// Acquisition function for Bayesian optimization
#[derive(Debug, Clone, Copy)]
pub enum AcquisitionFunction {
    /// Expected Improvement
    ExpectedImprovement,
    /// Upper Confidence Bound
    UpperConfidenceBound { kappa: Float },
    /// Probability of Improvement
    ProbabilityOfImprovement,
}

impl EnsembleCompressor {
    /// Create a new ensemble compressor
    pub fn new(config: CompressionConfig) -> Self {
        Self {
            config,
            stats: None,
        }
    }

    /// Create compressor with knowledge distillation
    pub fn knowledge_distillation(compression_ratio: Float, temperature: Float) -> Self {
        Self::new(CompressionConfig {
            strategy: CompressionStrategy::KnowledgeDistillation,
            compression_ratio,
            distillation_temperature: temperature,
            ..Default::default()
        })
    }

    /// Create compressor with ensemble pruning
    pub fn ensemble_pruning(compression_ratio: Float, quality_threshold: Float) -> Self {
        Self::new(CompressionConfig {
            strategy: CompressionStrategy::EnsemblePruning,
            compression_ratio,
            quality_threshold,
            ..Default::default()
        })
    }

    /// Create compressor with quantization
    pub fn quantization(bits: u8) -> Self {
        Self::new(CompressionConfig {
            strategy: CompressionStrategy::Quantization,
            quantization_bits: Some(bits),
            compression_ratio: 1.0 - (bits as Float / 32.0), // Assume 32-bit baseline
            ..Default::default()
        })
    }

    /// Create compressor with Bayesian optimization
    pub fn bayesian_optimization(
        performance_cost_trade_off: Float,
        n_calls: usize,
        random_state: Option<u64>,
    ) -> Self {
        Self::new(CompressionConfig {
            strategy: CompressionStrategy::BayesianOptimization,
            performance_cost_trade_off,
            bayes_opt_n_calls: n_calls,
            bayes_opt_n_initial: (n_calls / 5).max(5),
            bayes_opt_random_state: random_state,
            ..Default::default()
        })
    }

    /// Compress an ensemble using the configured strategy
    pub fn compress<T>(
        &mut self,
        ensemble: &[T],
        x_val: &Array2<Float>,
        y_val: &Array1<Int>,
    ) -> Result<CompressedEnsemble<T>>
    where
        T: Clone + Predict<Array2<Float>, Array1<Int>>,
    {
        let start_time = std::time::Instant::now();

        // Calculate original performance
        let original_accuracy = self.evaluate_ensemble_accuracy(ensemble, x_val, y_val)?;
        let original_size = self.estimate_ensemble_size(ensemble);

        let compressed = match self.config.strategy {
            CompressionStrategy::EnsemblePruning => {
                self.compress_by_pruning(ensemble, x_val, y_val)?
            }
            CompressionStrategy::Quantization => self.compress_by_quantization(ensemble)?,
            CompressionStrategy::WeightSharing => self.compress_by_weight_sharing(ensemble)?,
            CompressionStrategy::SparseRepresentation => {
                self.compress_by_sparsification(ensemble)?
            }
            CompressionStrategy::HierarchicalCompression => {
                self.compress_hierarchically(ensemble, x_val, y_val)?
            }
            CompressionStrategy::BayesianOptimization => {
                self.compress_by_bayesian_optimization(ensemble, x_val, y_val)?
            }
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Compression strategy {:?} not yet implemented",
                    self.config.strategy
                )));
            }
        };

        // Calculate compressed performance
        let compressed_accuracy = self.evaluate_compressed_accuracy(&compressed, x_val, y_val)?;
        let compressed_size = self.estimate_compressed_size(&compressed);

        // Update statistics
        self.stats = Some(CompressionStats {
            original_size_bytes: original_size,
            compressed_size_bytes: compressed_size,
            compression_ratio: 1.0 - (compressed_size as Float / original_size as Float),
            original_accuracy,
            compressed_accuracy,
            accuracy_loss: original_accuracy - compressed_accuracy,
            compression_time_secs: start_time.elapsed().as_secs_f64(),
            speedup_factor: original_size as Float / compressed_size as Float,
            memory_reduction_factor: original_size as Float / compressed_size as Float,
        });

        Ok(compressed)
    }

    /// Compress ensemble by pruning weak or redundant models
    fn compress_by_pruning<T>(
        &self,
        ensemble: &[T],
        x_val: &Array2<Float>,
        y_val: &Array1<Int>,
    ) -> Result<CompressedEnsemble<T>>
    where
        T: Clone + Predict<Array2<Float>, Array1<Int>>,
    {
        let target_count = (ensemble.len() as Float * self.config.compression_ratio) as usize;
        let target_count = target_count.max(1); // Keep at least one model

        // Calculate individual model performances
        let mut model_scores = Vec::new();
        for (i, model) in ensemble.iter().enumerate() {
            let predictions = model.predict(x_val)?;
            let accuracy = self.calculate_accuracy(&predictions, y_val);
            model_scores.push((i, accuracy));
        }

        // Sort by performance and keep top models
        model_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let selected_indices: Vec<usize> = model_scores
            .into_iter()
            .take(target_count)
            .map(|(i, _)| i)
            .collect();

        let mut selected_models = Vec::new();
        let mut model_mapping = HashMap::new();

        for (new_idx, &orig_idx) in selected_indices.iter().enumerate() {
            selected_models.push(ensemble[orig_idx].clone());
            model_mapping.insert(orig_idx, new_idx);
        }

        // Calculate uniform weights for selected models
        let weights =
            Array1::from_elem(selected_models.len(), 1.0 / selected_models.len() as Float);

        Ok(CompressedEnsemble {
            models: selected_models,
            metadata: CompressionMetadata {
                original_count: ensemble.len(),
                compressed_count: target_count,
                strategy: CompressionStrategy::EnsemblePruning,
                model_mapping,
                quantization_params: None,
                sparsity_info: None,
            },
            weights: Some(weights),
        })
    }

    /// Compress ensemble using quantization
    fn compress_by_quantization<T>(&self, ensemble: &[T]) -> Result<CompressedEnsemble<T>>
    where
        T: Clone,
    {
        let bits = self.config.quantization_bits.unwrap_or(8);

        // For demonstration, we'll create quantization parameters
        // In a real implementation, this would quantize the actual model weights
        let quantization_params = QuantizationParams {
            bits,
            scale: 1.0 / (2_i32.pow(bits as u32 - 1) as Float),
            zero_point: 0,
            min_val: -1.0,
            max_val: 1.0,
        };

        // Clone all models (in practice, these would be quantized versions)
        let compressed_models = ensemble.to_vec();

        Ok(CompressedEnsemble {
            models: compressed_models,
            metadata: CompressionMetadata {
                original_count: ensemble.len(),
                compressed_count: ensemble.len(),
                strategy: CompressionStrategy::Quantization,
                model_mapping: (0..ensemble.len()).map(|i| (i, i)).collect(),
                quantization_params: Some(quantization_params),
                sparsity_info: None,
            },
            weights: None,
        })
    }

    /// Compress ensemble using weight sharing
    fn compress_by_weight_sharing<T>(&self, ensemble: &[T]) -> Result<CompressedEnsemble<T>>
    where
        T: Clone,
    {
        // Simplified weight sharing - group similar models
        let num_groups = (ensemble.len() as Float * self.config.compression_ratio) as usize;
        let num_groups = num_groups.max(1);

        let models_per_group = ensemble.len() / num_groups;
        let mut compressed_models = Vec::new();
        let mut model_mapping = HashMap::new();

        for group_id in 0..num_groups {
            let start_idx = group_id * models_per_group;
            let end_idx = if group_id == num_groups - 1 {
                ensemble.len()
            } else {
                (group_id + 1) * models_per_group
            };

            // Use the first model in each group as representative
            if start_idx < ensemble.len() {
                compressed_models.push(ensemble[start_idx].clone());

                // Map all models in group to the representative
                for orig_idx in start_idx..end_idx {
                    model_mapping.insert(orig_idx, group_id);
                }
            }
        }

        let compressed_count = compressed_models.len();
        Ok(CompressedEnsemble {
            models: compressed_models,
            metadata: CompressionMetadata {
                original_count: ensemble.len(),
                compressed_count,
                strategy: CompressionStrategy::WeightSharing,
                model_mapping,
                quantization_params: None,
                sparsity_info: None,
            },
            weights: None,
        })
    }

    /// Compress ensemble using sparsification
    fn compress_by_sparsification<T>(&self, ensemble: &[T]) -> Result<CompressedEnsemble<T>>
    where
        T: Clone,
    {
        let sparsity_level = self.config.sparsity_level.unwrap_or(0.1);

        // For demonstration, create sparsity info
        // In practice, this would modify the actual model weights
        let total_params = ensemble.len() * 1000; // Assumed 1000 params per model
        let non_zero_params = (total_params as Float * (1.0 - sparsity_level)) as usize;

        let sparsity_info = SparsityInfo {
            sparsity_level,
            non_zero_params,
            total_params,
            sparse_indices: (0..non_zero_params).collect(),
        };

        // Clone all models (in practice, these would be sparsified versions)
        let compressed_models = ensemble.to_vec();

        Ok(CompressedEnsemble {
            models: compressed_models,
            metadata: CompressionMetadata {
                original_count: ensemble.len(),
                compressed_count: ensemble.len(),
                strategy: CompressionStrategy::SparseRepresentation,
                model_mapping: (0..ensemble.len()).map(|i| (i, i)).collect(),
                quantization_params: None,
                sparsity_info: Some(sparsity_info),
            },
            weights: None,
        })
    }

    /// Compress ensemble using hierarchical approach
    fn compress_hierarchically<T>(
        &self,
        ensemble: &[T],
        x_val: &Array2<Float>,
        y_val: &Array1<Int>,
    ) -> Result<CompressedEnsemble<T>>
    where
        T: Clone + Predict<Array2<Float>, Array1<Int>>,
    {
        // First, apply pruning
        let pruned = self.compress_by_pruning(ensemble, x_val, y_val)?;

        // Then, apply quantization to the pruned ensemble
        let quantized = self.compress_by_quantization(&pruned.models)?;

        let compressed_count = quantized.models.len();
        Ok(CompressedEnsemble {
            models: quantized.models,
            metadata: CompressionMetadata {
                original_count: ensemble.len(),
                compressed_count,
                strategy: CompressionStrategy::HierarchicalCompression,
                model_mapping: pruned.metadata.model_mapping,
                quantization_params: quantized.metadata.quantization_params,
                sparsity_info: None,
            },
            weights: pruned.weights,
        })
    }

    /// Compress ensemble using Bayesian optimization
    fn compress_by_bayesian_optimization<T>(
        &self,
        ensemble: &[T],
        x_val: &Array2<Float>,
        y_val: &Array1<Int>,
    ) -> Result<CompressedEnsemble<T>>
    where
        T: Clone + Predict<Array2<Float>, Array1<Int>>,
    {
        let mut optimizer = BayesianEnsembleOptimizer::new(
            self.config.clone(),
            self.config.bayes_opt_random_state.unwrap_or(42),
        );

        // Find optimal ensemble size using Bayesian optimization
        let optimal_size = optimizer.optimize_ensemble_size(ensemble, x_val, y_val)?;

        // Use the optimal size to perform intelligent pruning
        let mut pruning_config = self.config.clone();
        pruning_config.strategy = CompressionStrategy::EnsemblePruning;
        pruning_config.compression_ratio = optimal_size as Float / ensemble.len() as Float;

        let temp_compressor = EnsembleCompressor::new(pruning_config);
        let mut compressed = temp_compressor.compress_by_pruning(ensemble, x_val, y_val)?;

        // Update metadata to reflect Bayesian optimization strategy
        compressed.metadata.strategy = CompressionStrategy::BayesianOptimization;

        Ok(compressed)
    }

    /// Evaluate ensemble accuracy
    fn evaluate_ensemble_accuracy<T>(
        &self,
        ensemble: &[T],
        x_val: &Array2<Float>,
        y_val: &Array1<Int>,
    ) -> Result<Float>
    where
        T: Predict<Array2<Float>, Array1<Int>>,
    {
        let n_samples = x_val.nrows();
        let mut correct = 0;

        for i in 0..n_samples {
            let x_sample = x_val.row(i).insert_axis(scirs2_core::ndarray::Axis(0));
            let mut votes = HashMap::new();

            // Collect votes from all models
            for model in ensemble {
                let prediction = model.predict(&x_sample.to_owned())?;
                if !prediction.is_empty() {
                    *votes.entry(prediction[0]).or_insert(0) += 1;
                }
            }

            // Find majority vote
            if let Some((&predicted_class, _)) = votes.iter().max_by_key(|(_, &count)| count) {
                if predicted_class == y_val[i] {
                    correct += 1;
                }
            }
        }

        Ok(correct as Float / n_samples as Float)
    }

    /// Evaluate compressed ensemble accuracy
    fn evaluate_compressed_accuracy<T>(
        &self,
        compressed: &CompressedEnsemble<T>,
        x_val: &Array2<Float>,
        y_val: &Array1<Int>,
    ) -> Result<Float>
    where
        T: Predict<Array2<Float>, Array1<Int>>,
    {
        self.evaluate_ensemble_accuracy(&compressed.models, x_val, y_val)
    }

    /// Calculate accuracy for predictions
    fn calculate_accuracy(&self, predictions: &Array1<Int>, y_true: &Array1<Int>) -> Float {
        let correct = predictions
            .iter()
            .zip(y_true.iter())
            .map(|(pred, true_val)| if pred == true_val { 1 } else { 0 })
            .sum::<i32>();

        correct as Float / predictions.len() as Float
    }

    /// Estimate ensemble size in bytes
    fn estimate_ensemble_size<T>(&self, ensemble: &[T]) -> usize {
        // Simplified size estimation
        // In practice, this would calculate actual memory usage
        ensemble.len() * 1024 * 1024 // Assume 1MB per model
    }

    /// Estimate compressed ensemble size
    fn estimate_compressed_size<T>(&self, compressed: &CompressedEnsemble<T>) -> usize {
        let base_size = compressed.models.len() * 1024 * 1024;

        match compressed.metadata.strategy {
            CompressionStrategy::Quantization => {
                if let Some(ref params) = compressed.metadata.quantization_params {
                    (base_size as Float * (params.bits as Float / 32.0)) as usize
                } else {
                    base_size
                }
            }
            CompressionStrategy::SparseRepresentation => {
                if let Some(ref info) = compressed.metadata.sparsity_info {
                    (base_size as Float * (1.0 - info.sparsity_level)) as usize
                } else {
                    base_size
                }
            }
            _ => base_size,
        }
    }

    /// Get compression statistics
    pub fn stats(&self) -> Option<&CompressionStats> {
        self.stats.as_ref()
    }

    /// Reset compression statistics
    pub fn reset_stats(&mut self) {
        self.stats = None;
    }
}

impl KnowledgeDistillationTrainer {
    /// Create new knowledge distillation trainer
    pub fn new(temperature: Float, alpha: Float, beta: Float) -> Self {
        Self {
            temperature,
            alpha,
            beta,
        }
    }

    /// Train student model using teacher ensemble
    pub fn distill<Teacher, Student>(
        &self,
        teachers: &[Teacher],
        student: Student,
        x_train: &Array2<Float>,
        y_train: &Array1<Int>,
    ) -> Result<Student>
    where
        Teacher: PredictProba<Array2<Float>, Array2<Float>>,
        Student: Clone,
    {
        // Simplified distillation - in practice this would involve
        // actual gradient-based optimization

        // Get soft targets from teacher ensemble
        let _soft_targets = self.get_ensemble_soft_targets(teachers, x_train)?;

        // Here you would train the student model using both
        // hard targets (y_train) and soft targets from teachers
        // with the appropriate loss function combination

        Ok(student)
    }

    /// Get soft targets from teacher ensemble
    fn get_ensemble_soft_targets<T>(
        &self,
        teachers: &[T],
        x: &Array2<Float>,
    ) -> Result<Array2<Float>>
    where
        T: PredictProba<Array2<Float>, Array2<Float>>,
    {
        let n_samples = x.nrows();
        let first_proba = teachers[0].predict_proba(x)?;
        let n_classes = first_proba.ncols();

        let mut ensemble_proba = Array2::zeros((n_samples, n_classes));

        // Average probabilities from all teachers
        for teacher in teachers {
            let proba = teacher.predict_proba(x)?;
            ensemble_proba = ensemble_proba + proba;
        }

        ensemble_proba /= teachers.len() as Float;

        // Apply temperature scaling
        ensemble_proba.mapv_inplace(|p| (p / self.temperature).exp());

        // Normalize probabilities
        for mut row in ensemble_proba.rows_mut() {
            let sum = row.sum();
            if sum > 0.0 {
                row /= sum;
            }
        }

        Ok(ensemble_proba)
    }
}

impl EnsemblePruner {
    /// Create new ensemble pruner
    pub fn new(
        diversity_threshold: Float,
        performance_threshold: Float,
        correlation_threshold: Float,
    ) -> Self {
        Self {
            diversity_threshold,
            performance_threshold,
            correlation_threshold,
        }
    }

    /// Prune ensemble based on diversity and performance criteria
    pub fn prune<T>(
        &self,
        ensemble: &[T],
        x_val: &Array2<Float>,
        y_val: &Array1<Int>,
        target_size: usize,
    ) -> Result<Vec<usize>>
    where
        T: Predict<Array2<Float>, Array1<Int>>,
    {
        let n_models = ensemble.len();
        let target_size = target_size.min(n_models);

        // Calculate performance scores
        let mut model_scores = Vec::new();
        for (i, model) in ensemble.iter().enumerate() {
            let predictions = model.predict(x_val)?;
            let accuracy = self.calculate_accuracy(&predictions, y_val);
            model_scores.push((i, accuracy));
        }

        // Sort by performance
        model_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Select top performers that meet diversity criteria
        let mut selected = Vec::new();
        selected.push(model_scores[0].0); // Start with best model

        for (model_idx, score) in model_scores.iter().skip(1) {
            if selected.len() >= target_size {
                break;
            }

            if *score >= self.performance_threshold {
                // Check diversity with already selected models
                let is_diverse =
                    self.check_diversity_with_selected(*model_idx, &selected, ensemble, x_val)?;

                if is_diverse {
                    selected.push(*model_idx);
                }
            }
        }

        Ok(selected)
    }

    /// Check if a model is diverse enough compared to selected models
    fn check_diversity_with_selected<T>(
        &self,
        candidate_idx: usize,
        selected: &[usize],
        ensemble: &[T],
        x_val: &Array2<Float>,
    ) -> Result<bool>
    where
        T: Predict<Array2<Float>, Array1<Int>>,
    {
        let candidate_pred = ensemble[candidate_idx].predict(x_val)?;

        for &selected_idx in selected {
            let selected_pred = ensemble[selected_idx].predict(x_val)?;
            let correlation = self.calculate_correlation(&candidate_pred, &selected_pred);

            if correlation > self.correlation_threshold {
                return Ok(false); // Too similar to existing model
            }
        }

        Ok(true)
    }

    /// Calculate correlation between two prediction vectors
    fn calculate_correlation(&self, pred1: &Array1<Int>, pred2: &Array1<Int>) -> Float {
        if pred1.len() != pred2.len() {
            return 0.0;
        }

        let n = pred1.len() as Float;
        let agreements = pred1
            .iter()
            .zip(pred2.iter())
            .map(|(a, b)| if a == b { 1.0 } else { 0.0 })
            .sum::<Float>();

        agreements / n
    }

    /// Calculate accuracy
    fn calculate_accuracy(&self, predictions: &Array1<Int>, y_true: &Array1<Int>) -> Float {
        let correct = predictions
            .iter()
            .zip(y_true.iter())
            .map(|(pred, true_val)| if pred == true_val { 1 } else { 0 })
            .sum::<i32>();

        correct as Float / predictions.len() as Float
    }
}

impl BayesianEnsembleOptimizer {
    /// Create a new Bayesian ensemble optimizer
    pub fn new(config: CompressionConfig, random_state: u64) -> Self {
        Self {
            config,
            gp: SimpleGaussianProcess::new(0.01),
            rng: scirs2_core::random::rngs::StdRng::seed_from_u64(random_state),
            evaluations: Vec::new(),
            best_config: None,
        }
    }

    /// Optimize ensemble size using Bayesian optimization
    pub fn optimize_ensemble_size<T>(
        &mut self,
        ensemble: &[T],
        x_val: &Array2<Float>,
        y_val: &Array1<Int>,
    ) -> Result<usize>
    where
        T: Clone + Predict<Array2<Float>, Array1<Int>>,
    {
        let max_size = ensemble.len();
        let min_size = 1;

        // Phase 1: Random exploration
        for _ in 0..self.config.bayes_opt_n_initial {
            let ensemble_size = gen_range_usize(&mut self.rng, min_size..(max_size + 1));
            let objective =
                self.evaluate_ensemble_configuration(ensemble, x_val, y_val, ensemble_size)?;
            self.evaluations.push((ensemble_size, 0.0, 0.0, objective));
            self.update_best(ensemble_size, objective);
        }

        // Phase 2: Bayesian optimization
        let remaining_calls = self
            .config
            .bayes_opt_n_calls
            .saturating_sub(self.config.bayes_opt_n_initial);

        for _ in 0..remaining_calls {
            // Fit GP to current evaluations
            self.fit_surrogate_model()?;

            // Select next ensemble size using acquisition function
            let next_size = self.select_next_ensemble_size(min_size, max_size)?;
            let objective =
                self.evaluate_ensemble_configuration(ensemble, x_val, y_val, next_size)?;

            self.evaluations.push((next_size, 0.0, 0.0, objective));
            self.update_best(next_size, objective);
        }

        Ok(self
            .best_config
            .map(|(size, _)| size)
            .unwrap_or(max_size / 2))
    }

    /// Evaluate objective function for a given ensemble size
    fn evaluate_ensemble_configuration<T>(
        &self,
        ensemble: &[T],
        x_val: &Array2<Float>,
        y_val: &Array1<Int>,
        ensemble_size: usize,
    ) -> Result<Float>
    where
        T: Clone + Predict<Array2<Float>, Array1<Int>>,
    {
        if ensemble_size == 0 || ensemble_size > ensemble.len() {
            return Ok(Float::NEG_INFINITY);
        }

        // Select top performing models up to ensemble_size
        let mut model_scores = Vec::new();
        for (i, model) in ensemble.iter().enumerate() {
            let predictions = model.predict(x_val)?;
            let accuracy = self.calculate_accuracy(&predictions, y_val);
            model_scores.push((i, accuracy));
        }

        // Sort by performance and take top ensemble_size models
        model_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let selected_models: Vec<&T> = model_scores
            .iter()
            .take(ensemble_size)
            .map(|(i, _)| &ensemble[*i])
            .collect();

        // Evaluate ensemble performance
        let ensemble_accuracy = self.evaluate_subset_accuracy(&selected_models, x_val, y_val)?;

        // Calculate cost (inverse of ensemble size, normalized)
        let cost = ensemble_size as Float / ensemble.len() as Float;

        // Combine performance and cost with trade-off parameter
        let objective = self.config.performance_cost_trade_off * ensemble_accuracy
            - (1.0 - self.config.performance_cost_trade_off) * cost;

        Ok(objective)
    }

    /// Evaluate accuracy of a subset of models
    fn evaluate_subset_accuracy<T>(
        &self,
        models: &[&T],
        x_val: &Array2<Float>,
        y_val: &Array1<Int>,
    ) -> Result<Float>
    where
        T: Predict<Array2<Float>, Array1<Int>>,
    {
        let n_samples = x_val.nrows();
        let mut correct = 0;

        for i in 0..n_samples {
            let x_sample = x_val.row(i).insert_axis(scirs2_core::ndarray::Axis(0));
            let mut votes = HashMap::new();

            // Collect votes from selected models
            for model in models {
                let prediction = model.predict(&x_sample.to_owned())?;
                if !prediction.is_empty() {
                    *votes.entry(prediction[0]).or_insert(0) += 1;
                }
            }

            // Find majority vote
            if let Some((&predicted_class, _)) = votes.iter().max_by_key(|(_, &count)| count) {
                if predicted_class == y_val[i] {
                    correct += 1;
                }
            }
        }

        Ok(correct as Float / n_samples as Float)
    }

    /// Calculate accuracy for predictions
    fn calculate_accuracy(&self, predictions: &Array1<Int>, y_true: &Array1<Int>) -> Float {
        let correct = predictions
            .iter()
            .zip(y_true.iter())
            .map(|(pred, true_val)| if pred == true_val { 1 } else { 0 })
            .sum::<i32>();

        correct as Float / predictions.len() as Float
    }

    /// Fit surrogate model to current evaluations
    fn fit_surrogate_model(&mut self) -> Result<()> {
        if self.evaluations.is_empty() {
            return Ok(());
        }

        let n_points = self.evaluations.len();
        let mut x_train = Array2::zeros((n_points, 1));
        let mut y_train = Array1::zeros(n_points);

        for (i, &(ensemble_size, _, _, objective)) in self.evaluations.iter().enumerate() {
            x_train[[i, 0]] = ensemble_size as Float;
            y_train[i] = objective;
        }

        self.gp.fit(&x_train, &y_train)?;
        Ok(())
    }

    /// Select next ensemble size using acquisition function
    fn select_next_ensemble_size(&mut self, min_size: usize, max_size: usize) -> Result<usize> {
        let mut best_size = min_size;
        let mut best_acquisition = Float::NEG_INFINITY;

        // Evaluate acquisition function for all possible sizes
        for size in min_size..=max_size {
            let x_test = Array2::from_shape_vec((1, 1), vec![size as Float])
                .map_err(|_| SklearsError::InvalidInput("Invalid size".to_string()))?;

            let acquisition_value = self.compute_acquisition(&x_test)?;

            if acquisition_value > best_acquisition {
                best_acquisition = acquisition_value;
                best_size = size;
            }
        }

        Ok(best_size)
    }

    /// Compute acquisition function value
    fn compute_acquisition(&self, x: &Array2<Float>) -> Result<Float> {
        if self.evaluations.is_empty() {
            return Ok(0.0);
        }

        let (mean, std) = self.gp.predict(x)?;
        let mu = mean[0];
        let sigma = std[0];

        let acquisition_func = AcquisitionFunction::UpperConfidenceBound {
            kappa: self.config.bayes_opt_acquisition_kappa,
        };

        let acquisition = match acquisition_func {
            AcquisitionFunction::UpperConfidenceBound { kappa } => mu + kappa * sigma,
            AcquisitionFunction::ExpectedImprovement => {
                let best_score = self
                    .best_config
                    .map(|(_, score)| score)
                    .unwrap_or(Float::NEG_INFINITY);
                if sigma <= 1e-8 {
                    0.0
                } else {
                    let improvement = mu - best_score;
                    let z = improvement / sigma;
                    let phi = 0.5 * (1.0 + erf(z / 2.0_f64.sqrt()));
                    let density = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
                    improvement * phi + sigma * density
                }
            }
            AcquisitionFunction::ProbabilityOfImprovement => {
                let best_score = self
                    .best_config
                    .map(|(_, score)| score)
                    .unwrap_or(Float::NEG_INFINITY);
                if sigma <= 1e-8 {
                    0.0
                } else {
                    let z = (mu - best_score) / sigma;
                    0.5 * (1.0 + erf(z / 2.0_f64.sqrt()))
                }
            }
        };

        Ok(acquisition)
    }

    /// Update best configuration
    fn update_best(&mut self, ensemble_size: usize, objective: Float) {
        if let Some((_, best_obj)) = self.best_config {
            if objective > best_obj {
                self.best_config = Some((ensemble_size, objective));
            }
        } else {
            self.best_config = Some((ensemble_size, objective));
        }
    }

    /// Get best configuration found
    pub fn best_config(&self) -> Option<(usize, Float)> {
        self.best_config
    }

    /// Get all evaluations
    pub fn evaluations(&self) -> &[(usize, Float, Float, Float)] {
        &self.evaluations
    }
}

impl SimpleGaussianProcess {
    fn new(noise_level: Float) -> Self {
        Self {
            x_train: Array2::zeros((0, 0)),
            y_train: Array1::zeros(0),
            noise_level,
            length_scale: 1.0,
            signal_variance: 1.0,
        }
    }

    fn fit(&mut self, x: &Array2<Float>, y: &Array1<Float>) -> Result<()> {
        self.x_train = x.clone();
        self.y_train = y.clone();

        // Simple hyperparameter estimation
        if x.nrows() > 1 {
            // Estimate length scale as median pairwise distance
            let mut distances = Vec::new();
            for i in 0..x.nrows() {
                for j in (i + 1)..x.nrows() {
                    let mut dist_sq = 0.0;
                    for k in 0..x.ncols() {
                        let diff = x[[i, k]] - x[[j, k]];
                        dist_sq += diff * diff;
                    }
                    distances.push(dist_sq.sqrt());
                }
            }

            if !distances.is_empty() {
                distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
                self.length_scale = distances[distances.len() / 2].max(0.1);
            }

            // Estimate signal variance as variance of y
            let y_mean = y.mean().unwrap_or(0.0);
            self.signal_variance =
                y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<Float>() / y.len() as Float;
            self.signal_variance = self.signal_variance.max(0.01);
        }

        Ok(())
    }

    fn predict(&self, x: &Array2<Float>) -> Result<(Array1<Float>, Array1<Float>)> {
        let n_test = x.nrows();
        let mut mean = Array1::zeros(n_test);
        let mut std = Array1::zeros(n_test);

        if self.x_train.nrows() == 0 {
            return Ok((mean, Array1::from_elem(n_test, self.signal_variance.sqrt())));
        }

        // Simple GP prediction using RBF kernel
        for i in 0..n_test {
            let mut kernel_values = Array1::zeros(self.x_train.nrows());
            let mut total_weight = 0.0;

            for j in 0..self.x_train.nrows() {
                let mut dist_sq = 0.0;
                for k in 0..x.ncols() {
                    let diff = x[[i, k]] - self.x_train[[j, k]];
                    dist_sq += diff * diff;
                }

                let kernel_val = self.signal_variance
                    * (-dist_sq / (2.0 * self.length_scale * self.length_scale)).exp();
                kernel_values[j] = kernel_val;
                total_weight += kernel_val;
            }

            if total_weight > 1e-8 {
                kernel_values /= total_weight;
                mean[i] = kernel_values.dot(&self.y_train);

                let kernel_var: Float = kernel_values
                    .iter()
                    .zip(self.y_train.iter())
                    .map(|(&k, &y)| k * (y - mean[i]).powi(2))
                    .sum();

                let predictive_var = kernel_var + self.noise_level;
                std[i] = predictive_var.sqrt();
            } else {
                mean[i] = self.y_train.mean().unwrap_or(0.0);
                std[i] = self.signal_variance.sqrt();
            }
        }

        Ok((mean, std))
    }
}

// Simple error function approximation
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // Mock model for testing
    #[derive(Clone)]
    struct MockModel {
        prediction: Int,
    }

    impl Predict<Array2<Float>, Array1<Int>> for MockModel {
        fn predict(&self, x: &Array2<Float>) -> Result<Array1<Int>> {
            Ok(Array1::from_elem(x.nrows(), self.prediction))
        }
    }

    #[test]
    fn test_compression_config() {
        let config = CompressionConfig::default();
        assert_eq!(config.strategy, CompressionStrategy::EnsemblePruning);
        assert_eq!(config.compression_ratio, 0.5);
    }

    #[test]
    fn test_ensemble_compressor_creation() {
        let compressor = EnsembleCompressor::knowledge_distillation(0.3, 2.0);
        assert_eq!(
            compressor.config.strategy,
            CompressionStrategy::KnowledgeDistillation
        );
        assert_eq!(compressor.config.compression_ratio, 0.3);
        assert_eq!(compressor.config.distillation_temperature, 2.0);
    }

    #[test]
    fn test_ensemble_pruning() {
        let ensemble = vec![
            MockModel { prediction: 0 },
            MockModel { prediction: 1 },
            MockModel { prediction: 0 },
            MockModel { prediction: 1 },
        ];

        let x_val = Array2::zeros((10, 5));
        let y_val = Array1::zeros(10);

        let mut compressor = EnsembleCompressor::ensemble_pruning(0.5, 0.9);
        let compressed = compressor.compress(&ensemble, &x_val, &y_val).unwrap();

        assert_eq!(compressed.models.len(), 2); // 50% compression
        assert_eq!(
            compressed.metadata.strategy,
            CompressionStrategy::EnsemblePruning
        );
        assert!(compressed.weights.is_some());
    }

    #[test]
    fn test_quantization_compression() {
        let ensemble = vec![MockModel { prediction: 0 }];
        let mut compressor = EnsembleCompressor::quantization(8);

        let x_val = Array2::zeros((10, 5));
        let y_val = Array1::zeros(10);

        let compressed = compressor.compress(&ensemble, &x_val, &y_val).unwrap();

        assert_eq!(compressed.models.len(), 1);
        assert_eq!(
            compressed.metadata.strategy,
            CompressionStrategy::Quantization
        );
        assert!(compressed.metadata.quantization_params.is_some());
    }

    #[test]
    fn test_knowledge_distillation_trainer() {
        let trainer = KnowledgeDistillationTrainer::new(3.0, 0.7, 0.3);
        assert_eq!(trainer.temperature, 3.0);
        assert_eq!(trainer.alpha, 0.7);
        assert_eq!(trainer.beta, 0.3);
    }

    #[test]
    fn test_ensemble_pruner() {
        let pruner = EnsemblePruner::new(0.8, 0.9, 0.7);
        assert_eq!(pruner.diversity_threshold, 0.8);
        assert_eq!(pruner.performance_threshold, 0.9);
        assert_eq!(pruner.correlation_threshold, 0.7);
    }

    #[test]
    fn test_compression_stats() {
        let stats = CompressionStats {
            original_size_bytes: 1000,
            compressed_size_bytes: 500,
            compression_ratio: 0.5,
            original_accuracy: 0.95,
            compressed_accuracy: 0.92,
            accuracy_loss: 0.03,
            compression_time_secs: 1.5,
            speedup_factor: 2.0,
            memory_reduction_factor: 2.0,
        };

        assert_eq!(stats.compression_ratio, 0.5);
        assert_eq!(stats.speedup_factor, 2.0);
    }

    #[test]
    fn test_bayesian_optimization_compressor_creation() {
        let compressor = EnsembleCompressor::bayesian_optimization(0.8, 25, Some(42));
        assert_eq!(
            compressor.config.strategy,
            CompressionStrategy::BayesianOptimization
        );
        assert_eq!(compressor.config.performance_cost_trade_off, 0.8);
        assert_eq!(compressor.config.bayes_opt_n_calls, 25);
        assert_eq!(compressor.config.bayes_opt_random_state, Some(42));
    }

    #[test]
    fn test_bayesian_ensemble_optimizer() {
        let ensemble = vec![
            MockModel { prediction: 0 },
            MockModel { prediction: 1 },
            MockModel { prediction: 0 },
            MockModel { prediction: 1 },
            MockModel { prediction: 0 },
        ];

        let x_val = Array2::zeros((20, 3));
        let y_val = Array1::zeros(20);

        let config = CompressionConfig {
            strategy: CompressionStrategy::BayesianOptimization,
            performance_cost_trade_off: 0.7,
            bayes_opt_n_calls: 10,
            bayes_opt_n_initial: 3,
            bayes_opt_acquisition_kappa: 2.0,
            bayes_opt_random_state: Some(42),
            ..Default::default()
        };

        let mut optimizer = BayesianEnsembleOptimizer::new(config, 42);
        let optimal_size = optimizer
            .optimize_ensemble_size(&ensemble, &x_val, &y_val)
            .unwrap();

        // Should find some reasonable ensemble size
        assert!(optimal_size >= 1);
        assert!(optimal_size <= ensemble.len());
        assert!(optimizer.evaluations().len() == 10);
        assert!(optimizer.best_config().is_some());
    }

    #[test]
    fn test_bayesian_optimization_compression() {
        let ensemble = vec![
            MockModel { prediction: 0 },
            MockModel { prediction: 1 },
            MockModel { prediction: 0 },
            MockModel { prediction: 1 },
            MockModel { prediction: 0 },
            MockModel { prediction: 1 },
        ];

        let x_val = Array2::zeros((15, 4));
        let y_val = Array1::zeros(15);

        let mut compressor = EnsembleCompressor::bayesian_optimization(0.6, 8, Some(123));
        let compressed = compressor.compress(&ensemble, &x_val, &y_val).unwrap();

        assert_eq!(
            compressed.metadata.strategy,
            CompressionStrategy::BayesianOptimization
        );
        assert!(compressed.models.len() <= ensemble.len());
        assert!(compressed.models.len() >= 1);

        // Should have compression statistics
        let stats = compressor.stats().unwrap();
        assert!(stats.compression_ratio >= 0.0);
        assert!(stats.compression_ratio <= 1.0);
    }

    #[test]
    fn test_simple_gaussian_process() {
        let mut gp = SimpleGaussianProcess::new(0.01);

        // Test with no training data
        let x_test = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
        let (mean, std) = gp.predict(&x_test).unwrap();
        assert_eq!(mean.len(), 2);
        assert_eq!(std.len(), 2);

        // Train with some data
        let x_train = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let y_train = Array1::from_vec(vec![0.0, 1.0, 4.0]);
        gp.fit(&x_train, &y_train).unwrap();

        // Test prediction
        let (mean, std) = gp.predict(&x_test).unwrap();
        assert_eq!(mean.len(), 2);
        assert_eq!(std.len(), 2);

        // Predictions should be finite
        for &m in mean.iter() {
            assert!(m.is_finite());
        }
        for &s in std.iter() {
            assert!(s.is_finite() && s > 0.0);
        }
    }

    #[test]
    fn test_acquisition_functions() {
        let config = CompressionConfig {
            strategy: CompressionStrategy::BayesianOptimization,
            bayes_opt_acquisition_kappa: 1.96,
            performance_cost_trade_off: 0.5,
            ..Default::default()
        };

        let mut optimizer = BayesianEnsembleOptimizer::new(config, 42);

        // Add some mock evaluations
        optimizer.evaluations.push((3, 0.0, 0.0, 0.8));
        optimizer.evaluations.push((5, 0.0, 0.0, 0.6));
        optimizer.update_best(3, 0.8);

        // Test acquisition function computation
        optimizer.fit_surrogate_model().unwrap();
        let x_test = Array2::from_shape_vec((1, 1), vec![4.0]).unwrap();
        let acquisition = optimizer.compute_acquisition(&x_test).unwrap();

        // Should be finite
        assert!(acquisition.is_finite());
    }

    #[test]
    fn test_erf_approximation() {
        // Test known values
        assert!((erf(0.0) - 0.0).abs() < 1e-6);
        assert!((erf(1.0) - 0.8427).abs() < 1e-3);
        assert!((erf(-1.0) - (-0.8427)).abs() < 1e-3);

        // Test that erf is bounded between -1 and 1
        for x in [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0] {
            let result = erf(x);
            assert!(result >= -1.0 && result <= 1.0);
        }
    }
}
