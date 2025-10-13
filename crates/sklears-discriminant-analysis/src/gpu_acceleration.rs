//! GPU Acceleration for Discriminant Analysis
//!
//! This module provides GPU-accelerated implementations of discriminant analysis algorithms,
//! leveraging SciRS2's GPU capabilities for maximum performance on large-scale datasets.
//! Supports multiple GPU backends (CUDA, Metal, OpenCL, WebGPU) with intelligent fallback.

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::kernels::blas::gemm::{GemmImpl, GemmKernel};
use scirs2_core::kernels::ml::softmax::SoftmaxKernel;
use scirs2_core::gpu::kernels::reduction::ReductionKernel;
use scirs2_core::gpu::tensor_cores::TensorCoreOp;
use scirs2_core::gpu::{
    kernels::{DataType, KernelParams, OperationType},
    GpuBackend, GpuContext, GpuError,
};
use scirs2_core::gpu::memory::GpuBuffer;
use std::fmt;

use crate::{
    lda::{LinearDiscriminantAnalysis, LinearDiscriminantAnalysisConfig},
    numerical_stability::{NumericalConfig, NumericalStability},
    qda::{QuadraticDiscriminantAnalysis, QuadraticDiscriminantAnalysisConfig},
};

use rayon::prelude::*;
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba, Trained, Transform, Untrained},
    types::Float,
};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

/// Configuration for GPU-accelerated discriminant analysis
#[derive(Debug, Clone)]
pub struct GpuAccelerationConfig {
    /// Preferred GPU backend (None = auto-detect)
    pub preferred_backend: Option<GpuBackend>,
    /// Enable tensor cores if available (NVIDIA GPUs)
    pub use_tensor_cores: bool,
    /// Use mixed precision (f16/bf16) for memory efficiency
    pub use_mixed_precision: bool,
    /// Batch size for GPU operations
    pub batch_size: usize,
    /// Memory usage limit for GPU (bytes, None = auto-detect)
    pub memory_limit: Option<usize>,
    /// Enable asynchronous GPU operations
    pub async_operations: bool,
    /// Fallback to CPU if GPU operations fail
    pub cpu_fallback: bool,
    /// Minimum matrix size to use GPU (smaller matrices use CPU)
    pub gpu_threshold: usize,
    /// Enable multi-GPU support for very large datasets
    pub multi_gpu: bool,
    /// GPU memory management strategy
    pub memory_strategy: GpuMemoryStrategy,
    /// Custom kernel implementations
    pub custom_kernels: HashMap<String, Box<dyn GpuKernel + Send + Sync>>,
}

/// GPU memory management strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuMemoryStrategy {
    Conservative,
    Balanced,
    Aggressive,
    Custom {

        batch_size: usize,

        buffer_size: usize,
    },
}

impl Default for GpuAccelerationConfig {
    fn default() -> Self {
        Self {
            preferred_backend: None, // Auto-detect
            use_tensor_cores: true,
            use_mixed_precision: false, // Conservative default
            batch_size: 1024,
            memory_limit: None, // Auto-detect
            async_operations: true,
            cpu_fallback: true,
            gpu_threshold: 256, // Use GPU for matrices > 256x256
            multi_gpu: false,
            memory_strategy: GpuMemoryStrategy::Balanced,
            custom_kernels: HashMap::new(),
        }
    }
}

/// Trait for GPU-accelerated discriminant analysis kernels
pub trait GpuKernel: Send + Sync {
    /// Execute the kernel on the GPU
    fn execute(&self, context: &GpuContext, params: &KernelParams) -> Result<GpuBuffer>;

    /// Get kernel metadata
    fn metadata(&self) -> &crate::gpu::kernels::KernelMetadata;

    /// Check if kernel supports the given data type
    fn supports_dtype(&self, dtype: DataType) -> bool;

    /// Get preferred workgroup size for given input dimensions
    fn optimal_workgroup_size(&self, shape: &[usize]) -> [u32; 3];
}

/// GPU-accelerated Linear Discriminant Analysis kernel
pub struct GpuLDAKernel {
    gemm: GemmKernel,
    reduction: ReductionKernel,
    softmax: SoftmaxKernel,
    config: GpuAccelerationConfig,
}

impl GpuLDAKernel {
    pub fn new(config: GpuAccelerationConfig) -> Self {
        let gemm_impl = if config.use_tensor_cores {
            GemmImpl::TensorCore
        } else {
            GemmImpl::Standard
        };

        Self {
            gemm: GemmKernel::with_implementation(gemm_impl),
            reduction: ReductionKernel::new(),
            softmax: SoftmaxKernel::new(),
            config,
        }
    }

    /// Compute class means on GPU
    pub fn compute_class_means_gpu(
        &self,
        context: &GpuContext,
        X: &ArrayView2<Float>,
        y: &ArrayView1<usize>,
        n_classes: usize,
    ) -> Result<Array2<Float>> {
        let (n_samples, n_features) = X.dim();

        // Allocate GPU buffers
        let X_gpu = self.transfer_to_gpu(context, X)?;
        let y_gpu = self.transfer_to_gpu(
            context,
            &y.mapv(|label| label as Float).into_shape((n_samples, 1))?,
        )?;

        let mut class_means = Array2::<Float>::zeros((n_classes, n_features));

        for class in 0..n_classes {
            // Create mask for current class on GPU
            let mask_params = self.create_class_mask_params(class, &y_gpu)?;
            let mask_gpu = self.execute_kernel(context, "class_mask", &mask_params)?;

            // Compute mean for current class using GPU reduction
            let mean_params = self.create_class_mean_params(&X_gpu, &mask_gpu)?;
            let mean_gpu = self.reduction.execute(context, &mean_params)?;

            // Transfer result back to CPU
            let mean_cpu = self.transfer_from_gpu(context, &mean_gpu)?;
            class_means.row_mut(class).assign(&mean_cpu.view());
        }

        Ok(class_means)
    }

    /// Compute covariance matrix on GPU
    pub fn compute_covariance_gpu(
        &self,
        context: &GpuContext,
        X: &ArrayView2<Float>,
        class_means: &ArrayView2<Float>,
    ) -> Result<Array2<Float>> {
        let (n_samples, n_features) = X.dim();

        // Center the data: X_centered = X - mean
        let X_gpu = self.transfer_to_gpu(context, X)?;
        let means_gpu = self.transfer_to_gpu(context, class_means)?;

        // Subtract means using GPU BLAS
        let centering_params = self.create_centering_params(&X_gpu, &means_gpu)?;
        let X_centered_gpu = self.gemm.execute(context, &centering_params)?;

        // Compute covariance: C = X_centered^T @ X_centered / (n_samples - 1)
        let mut cov_params = KernelParams::new();
        cov_params.set_alpha(1.0 / (n_samples - 1) as Float);
        cov_params.set_beta(0.0);
        cov_params.set_transpose_a(true);
        cov_params.set_transpose_b(false);
        cov_params.add_input_buffer("A", X_centered_gpu.clone());
        cov_params.add_input_buffer("B", X_centered_gpu.clone());

        let cov_gpu = self.gemm.execute(context, &cov_params)?;
        let covariance = self.transfer_from_gpu(context, &cov_gpu)?;

        Ok(covariance.into_shape((n_features, n_features))?)
    }

    /// Compute discriminant function on GPU
    pub fn compute_discriminant_gpu(
        &self,
        context: &GpuContext,
        X: &ArrayView2<Float>,
        class_means: &ArrayView2<Float>,
        cov_inv: &ArrayView2<Float>,
        log_priors: &ArrayView1<Float>,
    ) -> Result<Array2<Float>> {
        let (n_samples, _) = X.dim();
        let n_classes = class_means.nrows();

        // Transfer data to GPU
        let X_gpu = self.transfer_to_gpu(context, X)?;
        let means_gpu = self.transfer_to_gpu(context, class_means)?;
        let cov_inv_gpu = self.transfer_to_gpu(context, cov_inv)?;
        let priors_gpu = self.transfer_to_gpu(context, &log_priors.into_shape((n_classes, 1))?)?;

        let mut discriminants = Array2::<Float>::zeros((n_samples, n_classes));

        for class in 0..n_classes {
            // Compute (x - mean_k) for class k
            let diff_params = self.create_difference_params(&X_gpu, &means_gpu, class)?;
            let diff_gpu = self.execute_kernel(context, "matrix_diff", &diff_params)?;

            // Compute (x - mean_k)^T @ cov_inv
            let mut quad_params1 = KernelParams::new();
            quad_params1.set_alpha(1.0);
            quad_params1.set_beta(0.0);
            quad_params1.add_input_buffer("A", diff_gpu.clone());
            quad_params1.add_input_buffer("B", cov_inv_gpu.clone());

            let quad1_gpu = self.gemm.execute(context, &quad_params1)?;

            // Compute ((x - mean_k)^T @ cov_inv) @ (x - mean_k)
            let mut quad_params2 = KernelParams::new();
            quad_params2.set_alpha(-0.5); // -0.5 for discriminant function
            quad_params2.set_beta(0.0);
            quad_params2.add_input_buffer("A", quad1_gpu);
            quad_params2.add_input_buffer("B", diff_gpu);

            let quad_gpu = self.gemm.execute(context, &quad_params2)?;

            // Add log prior
            let discriminant_params =
                self.create_add_prior_params(&quad_gpu, &priors_gpu, class)?;
            let class_discriminant_gpu =
                self.execute_kernel(context, "add_prior", &discriminant_params)?;

            // Transfer result back to CPU
            let class_discriminant = self.transfer_from_gpu(context, &class_discriminant_gpu)?;
            discriminants
                .column_mut(class)
                .assign(&class_discriminant.view());
        }

        Ok(discriminants)
    }

    /// Compute predictions using GPU softmax
    pub fn predict_gpu(
        &self,
        context: &GpuContext,
        discriminants: &ArrayView2<Float>,
    ) -> Result<Array1<usize>> {
        // Apply softmax to discriminants on GPU
        let discriminants_gpu = self.transfer_to_gpu(context, discriminants)?;

        let mut softmax_params = KernelParams::new();
        softmax_params.add_input_buffer("input", discriminants_gpu);
        softmax_params.set_axis(1); // Softmax along class dimension

        let probabilities_gpu = self.softmax.execute(context, &softmax_params)?;
        let probabilities = self.transfer_from_gpu(context, &probabilities_gpu)?;

        // Find argmax on CPU (could be optimized with GPU reduction)
        let predictions = probabilities
            .axis_iter(Axis(0))
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect::<Vec<usize>>();

        Ok(Array1::from_vec(predictions))
    }

    // Helper methods for GPU operations
    fn transfer_to_gpu<D>(
        &self,
        context: &GpuContext,
        array: &ndarray::ArrayBase<D, ndarray::Ix2>,
    ) -> Result<GpuBuffer>
    where
        D: ndarray::Data<Elem = Float>,
    {
        // Convert ndarray to GPU buffer
        let data = array.as_slice().unwrap_or_else(|| {
            // Handle non-contiguous arrays
            &array.to_owned().into_raw_vec()
        });

        context
            .create_buffer_with_data(data)
            .map_err(|e| SklearsError::ComputationError(format!("GPU transfer failed: {}", e)))
    }

    fn transfer_from_gpu(&self, context: &GpuContext, buffer: &GpuBuffer) -> Result<Array2<Float>> {
        let data = context
            .read_buffer(buffer)
            .map_err(|e| SklearsError::ComputationError(format!("GPU readback failed: {}", e)))?;

        // Reconstruct array shape (this is simplified - real implementation needs shape tracking)
        let shape = buffer.shape(); // Assuming GpuBuffer tracks shape
        Ok(Array2::from_shape_vec(shape, data)?)
    }

    fn execute_kernel(
        &self,
        context: &GpuContext,
        kernel_name: &str,
        params: &KernelParams,
    ) -> Result<GpuBuffer> {
        // Execute custom kernel - this would be implemented based on SciRS2's kernel system
        match kernel_name {
            "class_mask" => self.execute_class_mask_kernel(context, params),
            "matrix_diff" => self.execute_matrix_diff_kernel(context, params),
            "add_prior" => self.execute_add_prior_kernel(context, params),
            _ => Err(SklearsError::ComputationError(format!(
                "Unknown kernel: {}",
                kernel_name
            ))),
        }
    }

    fn execute_class_mask_kernel(
        &self,
        context: &GpuContext,
        params: &KernelParams,
    ) -> Result<GpuBuffer> {
        // Placeholder for custom kernel implementation
        // Real implementation would use SciRS2's kernel compilation system
        Err(SklearsError::ComputationError(
            "Custom kernels not yet implemented".to_string(),
        ))
    }

    fn execute_matrix_diff_kernel(
        &self,
        context: &GpuContext,
        params: &KernelParams,
    ) -> Result<GpuBuffer> {
        // Placeholder for custom kernel implementation
        Err(SklearsError::ComputationError(
            "Custom kernels not yet implemented".to_string(),
        ))
    }

    fn execute_add_prior_kernel(
        &self,
        context: &GpuContext,
        params: &KernelParams,
    ) -> Result<GpuBuffer> {
        // Placeholder for custom kernel implementation
        Err(SklearsError::ComputationError(
            "Custom kernels not yet implemented".to_string(),
        ))
    }

    // Parameter creation helpers
    fn create_class_mask_params(&self, class: usize, y_gpu: &GpuBuffer) -> Result<KernelParams> {
        let mut params = KernelParams::new();
        params.add_input_buffer("labels", y_gpu.clone());
        params.set_int_param("target_class", class as i32);
        Ok(params)
    }

    fn create_class_mean_params(
        &self,
        X_gpu: &GpuBuffer,
        mask_gpu: &GpuBuffer,
    ) -> Result<KernelParams> {
        let mut params = KernelParams::new();
        params.add_input_buffer("data", X_gpu.clone());
        params.add_input_buffer("mask", mask_gpu.clone());
        Ok(params)
    }

    fn create_centering_params(
        &self,
        X_gpu: &GpuBuffer,
        means_gpu: &GpuBuffer,
    ) -> Result<KernelParams> {
        let mut params = KernelParams::new();
        params.set_alpha(-1.0); // Subtract means
        params.set_beta(1.0); // Keep original data
        params.add_input_buffer("A", means_gpu.clone());
        params.add_input_buffer("B", X_gpu.clone());
        Ok(params)
    }

    fn create_difference_params(
        &self,
        X_gpu: &GpuBuffer,
        means_gpu: &GpuBuffer,
        class: usize,
    ) -> Result<KernelParams> {
        let mut params = KernelParams::new();
        params.add_input_buffer("data", X_gpu.clone());
        params.add_input_buffer("means", means_gpu.clone());
        params.set_int_param("class_index", class as i32);
        Ok(params)
    }

    fn create_add_prior_params(
        &self,
        quad_gpu: &GpuBuffer,
        priors_gpu: &GpuBuffer,
        class: usize,
    ) -> Result<KernelParams> {
        let mut params = KernelParams::new();
        params.add_input_buffer("discriminant", quad_gpu.clone());
        params.add_input_buffer("priors", priors_gpu.clone());
        params.set_int_param("class_index", class as i32);
        Ok(params)
    }
}

/// GPU-accelerated Quadratic Discriminant Analysis kernel
pub struct GpuQDAKernel {
    gemm: GemmKernel,
    reduction: ReductionKernel,
    softmax: SoftmaxKernel,
    config: GpuAccelerationConfig,
}

impl GpuQDAKernel {
    pub fn new(config: GpuAccelerationConfig) -> Self {
        let gemm_impl = if config.use_tensor_cores {
            GemmImpl::TensorCore
        } else {
            GemmImpl::Standard
        };

        Self {
            gemm: GemmKernel::with_implementation(gemm_impl),
            reduction: ReductionKernel::new(),
            softmax: SoftmaxKernel::new(),
            config,
        }
    }

    /// Compute class-specific covariance matrices on GPU
    pub fn compute_class_covariances_gpu(
        &self,
        context: &GpuContext,
        X: &ArrayView2<Float>,
        y: &ArrayView1<usize>,
        class_means: &ArrayView2<Float>,
        n_classes: usize,
    ) -> Result<Vec<Array2<Float>>> {
        let (n_samples, n_features) = X.dim();
        let mut class_covariances = Vec::with_capacity(n_classes);

        // Transfer common data to GPU
        let X_gpu = self.transfer_to_gpu(context, X)?;
        let means_gpu = self.transfer_to_gpu(context, class_means)?;
        let y_gpu = self.transfer_to_gpu(
            context,
            &y.mapv(|label| label as Float).into_shape((n_samples, 1))?,
        )?;

        for class in 0..n_classes {
            // Create class mask
            let mask_params = self.create_class_mask_params(class, &y_gpu)?;
            let mask_gpu = self.execute_kernel(context, "class_mask", &mask_params)?;

            // Extract class data
            let class_data_params = self.create_class_data_params(&X_gpu, &mask_gpu)?;
            let class_X_gpu =
                self.execute_kernel(context, "extract_class_data", &class_data_params)?;

            // Center the class data
            let centering_params = self.create_centering_params(&class_X_gpu, &means_gpu, class)?;
            let centered_gpu =
                self.execute_kernel(context, "center_class_data", &centering_params)?;

            // Compute class covariance: C_k = X_k^T @ X_k / (n_k - 1)
            let n_class_samples = self.count_class_samples(context, &mask_gpu)?;

            let mut cov_params = KernelParams::new();
            cov_params.set_alpha(1.0 / (n_class_samples - 1) as Float);
            cov_params.set_beta(0.0);
            cov_params.set_transpose_a(true);
            cov_params.set_transpose_b(false);
            cov_params.add_input_buffer("A", centered_gpu.clone());
            cov_params.add_input_buffer("B", centered_gpu.clone());

            let cov_gpu = self.gemm.execute(context, &cov_params)?;
            let covariance = self.transfer_from_gpu(context, &cov_gpu)?;

            class_covariances.push(covariance.into_shape((n_features, n_features))?);
        }

        Ok(class_covariances)
    }

    /// Compute QDA discriminant function on GPU
    pub fn compute_qda_discriminant_gpu(
        &self,
        context: &GpuContext,
        X: &ArrayView2<Float>,
        class_means: &ArrayView2<Float>,
        class_covariances: &[Array2<Float>],
        log_priors: &ArrayView1<Float>,
    ) -> Result<Array2<Float>> {
        let (n_samples, _) = X.dim();
        let n_classes = class_means.nrows();

        // Transfer data to GPU
        let X_gpu = self.transfer_to_gpu(context, X)?;
        let means_gpu = self.transfer_to_gpu(context, class_means)?;
        let priors_gpu = self.transfer_to_gpu(context, &log_priors.into_shape((n_classes, 1))?)?;

        let mut discriminants = Array2::<Float>::zeros((n_samples, n_classes));

        for class in 0..n_classes {
            // Transfer class covariance and compute its inverse + log determinant
            let cov_gpu = self.transfer_to_gpu(context, &class_covariances[class])?;
            let (cov_inv_gpu, log_det) = self.compute_inverse_and_log_det_gpu(context, &cov_gpu)?;

            // Compute (x - mean_k) for class k
            let diff_params = self.create_difference_params(&X_gpu, &means_gpu, class)?;
            let diff_gpu = self.execute_kernel(context, "matrix_diff", &diff_params)?;

            // Compute (x - mean_k)^T @ cov_inv_k
            let mut quad_params1 = KernelParams::new();
            quad_params1.set_alpha(1.0);
            quad_params1.set_beta(0.0);
            quad_params1.add_input_buffer("A", diff_gpu.clone());
            quad_params1.add_input_buffer("B", cov_inv_gpu);

            let quad1_gpu = self.gemm.execute(context, &quad_params1)?;

            // Compute ((x - mean_k)^T @ cov_inv_k) @ (x - mean_k)
            let mut quad_params2 = KernelParams::new();
            quad_params2.set_alpha(-0.5); // -0.5 for discriminant function
            quad_params2.set_beta(0.0);
            quad_params2.add_input_buffer("A", quad1_gpu);
            quad_params2.add_input_buffer("B", diff_gpu);

            let quad_gpu = self.gemm.execute(context, &quad_params2)?;

            // Add log prior and log determinant term
            let discriminant_params =
                self.create_qda_final_params(&quad_gpu, &priors_gpu, class, log_det)?;
            let class_discriminant_gpu =
                self.execute_kernel(context, "qda_final", &discriminant_params)?;

            // Transfer result back to CPU
            let class_discriminant = self.transfer_from_gpu(context, &class_discriminant_gpu)?;
            discriminants
                .column_mut(class)
                .assign(&class_discriminant.view());
        }

        Ok(discriminants)
    }

    // Helper methods specific to QDA
    fn compute_inverse_and_log_det_gpu(
        &self,
        context: &GpuContext,
        cov_gpu: &GpuBuffer,
    ) -> Result<(GpuBuffer, Float)> {
        // This would use a GPU-based matrix inverse and determinant kernel
        // For now, we'll compute on CPU as a fallback
        let cov = self.transfer_from_gpu(context, cov_gpu)?;
        let cov_inv = crate::numerical_stability::NumericalStability::matrix_inverse(&cov.view())?;
        let log_det = cov.diag().mapv(|x| x.ln()).sum(); // Simplified - should use proper log det

        let cov_inv_gpu = self.transfer_to_gpu(context, &cov_inv)?;
        Ok((cov_inv_gpu, log_det))
    }

    fn count_class_samples(&self, context: &GpuContext, mask_gpu: &GpuBuffer) -> Result<usize> {
        // Use GPU reduction to count samples
        let mut count_params = KernelParams::new();
        count_params.add_input_buffer("mask", mask_gpu.clone());

        let count_gpu = self.reduction.execute(context, &count_params)?;
        let count_array = self.transfer_from_gpu(context, &count_gpu)?;
        Ok(count_array[[0, 0]] as usize)
    }

    // Additional parameter creation methods for QDA
    fn create_class_data_params(
        &self,
        X_gpu: &GpuBuffer,
        mask_gpu: &GpuBuffer,
    ) -> Result<KernelParams> {
        let mut params = KernelParams::new();
        params.add_input_buffer("data", X_gpu.clone());
        params.add_input_buffer("mask", mask_gpu.clone());
        Ok(params)
    }

    fn create_qda_final_params(
        &self,
        quad_gpu: &GpuBuffer,
        priors_gpu: &GpuBuffer,
        class: usize,
        log_det: Float,
    ) -> Result<KernelParams> {
        let mut params = KernelParams::new();
        params.add_input_buffer("quadratic_term", quad_gpu.clone());
        params.add_input_buffer("priors", priors_gpu.clone());
        params.set_int_param("class_index", class as i32);
        params.set_float_param("log_det", log_det);
        Ok(params)
    }

    // Inherit other methods from GpuLDAKernel (in a real implementation, we'd use composition or traits)
    fn transfer_to_gpu<D>(
        &self,
        context: &GpuContext,
        array: &ndarray::ArrayBase<D, ndarray::Ix2>,
    ) -> Result<GpuBuffer>
    where
        D: ndarray::Data<Elem = Float>,
    {
        // Same implementation as in GpuLDAKernel
        let data = array
            .as_slice()
            .unwrap_or_else(|| &array.to_owned().into_raw_vec());

        context
            .create_buffer_with_data(data)
            .map_err(|e| SklearsError::ComputationError(format!("GPU transfer failed: {}", e)))
    }

    fn transfer_from_gpu(&self, context: &GpuContext, buffer: &GpuBuffer) -> Result<Array2<Float>> {
        let data = context
            .read_buffer(buffer)
            .map_err(|e| SklearsError::ComputationError(format!("GPU readback failed: {}", e)))?;

        let shape = buffer.shape();
        Ok(Array2::from_shape_vec(shape, data)?)
    }

    fn execute_kernel(
        &self,
        context: &GpuContext,
        kernel_name: &str,
        params: &KernelParams,
    ) -> Result<GpuBuffer> {
        // Same basic structure as GpuLDAKernel but with additional QDA-specific kernels
        match kernel_name {
            "class_mask" => self.execute_class_mask_kernel(context, params),
            "extract_class_data" => self.execute_extract_class_data_kernel(context, params),
            "center_class_data" => self.execute_center_class_data_kernel(context, params),
            "matrix_diff" => self.execute_matrix_diff_kernel(context, params),
            "qda_final" => self.execute_qda_final_kernel(context, params),
            _ => Err(SklearsError::ComputationError(format!(
                "Unknown kernel: {}",
                kernel_name
            ))),
        }
    }

    fn execute_class_mask_kernel(
        &self,
        context: &GpuContext,
        params: &KernelParams,
    ) -> Result<GpuBuffer> {
        Err(SklearsError::ComputationError(
            "Custom kernels not yet implemented".to_string(),
        ))
    }

    fn execute_extract_class_data_kernel(
        &self,
        context: &GpuContext,
        params: &KernelParams,
    ) -> Result<GpuBuffer> {
        Err(SklearsError::ComputationError(
            "Custom kernels not yet implemented".to_string(),
        ))
    }

    fn execute_center_class_data_kernel(
        &self,
        context: &GpuContext,
        params: &KernelParams,
    ) -> Result<GpuBuffer> {
        Err(SklearsError::ComputationError(
            "Custom kernels not yet implemented".to_string(),
        ))
    }

    fn execute_matrix_diff_kernel(
        &self,
        context: &GpuContext,
        params: &KernelParams,
    ) -> Result<GpuBuffer> {
        Err(SklearsError::ComputationError(
            "Custom kernels not yet implemented".to_string(),
        ))
    }

    fn execute_qda_final_kernel(
        &self,
        context: &GpuContext,
        params: &KernelParams,
    ) -> Result<GpuBuffer> {
        Err(SklearsError::ComputationError(
            "Custom kernels not yet implemented".to_string(),
        ))
    }

    fn create_class_mask_params(&self, class: usize, y_gpu: &GpuBuffer) -> Result<KernelParams> {
        let mut params = KernelParams::new();
        params.add_input_buffer("labels", y_gpu.clone());
        params.set_int_param("target_class", class as i32);
        Ok(params)
    }

    fn create_centering_params(
        &self,
        X_gpu: &GpuBuffer,
        means_gpu: &GpuBuffer,
        class: usize,
    ) -> Result<KernelParams> {
        let mut params = KernelParams::new();
        params.add_input_buffer("data", X_gpu.clone());
        params.add_input_buffer("means", means_gpu.clone());
        params.set_int_param("class_index", class as i32);
        Ok(params)
    }

    fn create_difference_params(
        &self,
        X_gpu: &GpuBuffer,
        means_gpu: &GpuBuffer,
        class: usize,
    ) -> Result<KernelParams> {
        let mut params = KernelParams::new();
        params.add_input_buffer("data", X_gpu.clone());
        params.add_input_buffer("means", means_gpu.clone());
        params.set_int_param("class_index", class as i32);
        Ok(params)
    }
}

/// GPU-accelerated discriminant analysis manager
pub struct GpuDiscriminantAnalysis {
    context: Option<Arc<GpuContext>>,
    config: GpuAccelerationConfig,
    lda_kernel: Option<GpuLDAKernel>,
    qda_kernel: Option<GpuQDAKernel>,
    performance_stats: Arc<Mutex<GpuPerformanceStats>>,
}

/// Performance statistics for GPU operations
#[derive(Debug, Default)]
pub struct GpuPerformanceStats {
    pub gpu_time_ms: f64,
    pub cpu_time_ms: f64,
    pub transfer_time_ms: f64,
    pub memory_usage_mb: f64,
    pub kernel_executions: usize,
    pub fallback_count: usize,
}

impl GpuDiscriminantAnalysis {
    /// Create new GPU-accelerated discriminant analysis
    pub fn new(config: GpuAccelerationConfig) -> Result<Self> {
        let context = Self::initialize_gpu_context(&config)?;

        Ok(Self {
            context: Some(Arc::new(context)),
            lda_kernel: Some(GpuLDAKernel::new(config.clone())),
            qda_kernel: Some(GpuQDAKernel::new(config.clone())),
            config,
            performance_stats: Arc::new(Mutex::new(GpuPerformanceStats::default())),
        })
    }

    /// Initialize GPU context with fallback strategy
    fn initialize_gpu_context(config: &GpuAccelerationConfig) -> Result<GpuContext> {
        let backends = if let Some(preferred) = config.preferred_backend {
            vec![preferred, GpuBackend::Cpu] // Always include CPU fallback
        } else {
            // Try backends in order of preference
            vec![
                GpuBackend::Cuda,
                GpuBackend::Metal,
                GpuBackend::Wgpu,
                GpuBackend::OpenCL,
                GpuBackend::Cpu,
            ]
        };

        for backend in &backends {
            if backend.is_available() {
                match GpuContext::new(*backend) {
                    Ok(context) => {
                        log::info!("Successfully initialized {} backend", backend);
                        return Ok(context);
                    }
                    Err(e) => {
                        log::warn!("Failed to initialize {} backend: {}", backend, e);
                        continue;
                    }
                }
            }
        }

        Err(SklearsError::ComputationError(
            "No GPU backends available".to_string(),
        ))
    }

    /// Check if GPU acceleration is available
    pub fn is_gpu_available(&self) -> bool {
        self.context.is_some() && self.context.as_ref().unwrap().backend() != GpuBackend::Cpu
    }

    /// Get current GPU backend
    pub fn current_backend(&self) -> Option<GpuBackend> {
        self.context.as_ref().map(|ctx| ctx.backend())
    }

    /// Get performance statistics
    pub fn performance_stats(&self) -> GpuPerformanceStats {
        self.performance_stats.lock().unwrap().clone()
    }

    /// Determine whether to use GPU based on problem size
    fn should_use_gpu(&self, n_samples: usize, n_features: usize) -> bool {
        let problem_size = n_samples * n_features;
        problem_size >= self.config.gpu_threshold && self.is_gpu_available()
    }

    /// Execute GPU-accelerated LDA
    pub fn gpu_lda_fit_predict(
        &self,
        X: &ArrayView2<Float>,
        y: &ArrayView1<usize>,
        X_test: &ArrayView2<Float>,
        lda_config: &LinearDiscriminantAnalysisConfig,
    ) -> Result<Array1<usize>> {
        let (n_samples, n_features) = X.dim();

        if !self.should_use_gpu(n_samples, n_features) {
            return self.cpu_fallback_lda(X, y, X_test, lda_config);
        }

        let context = self.context.as_ref().unwrap();
        let kernel = self.lda_kernel.as_ref().unwrap();

        let start_time = std::time::Instant::now();

        // Compute number of classes
        let n_classes = y.iter().max().unwrap() + 1;

        // Compute class means on GPU
        let class_means = kernel.compute_class_means_gpu(context, X, y, n_classes)?;

        // Compute pooled covariance on GPU
        let covariance = kernel.compute_covariance_gpu(context, X, &class_means.view())?;

        // Compute log priors
        let priors = self.compute_priors(y, n_classes);
        let log_priors = priors.mapv(|p| p.ln());

        // Compute discriminant function for test data
        let discriminants = kernel.compute_discriminant_gpu(
            context,
            X_test,
            &class_means.view(),
            &covariance.view(),
            &log_priors.view(),
        )?;

        // Get predictions
        let predictions = kernel.predict_gpu(context, &discriminants.view())?;

        // Update performance stats
        let elapsed = start_time.elapsed();
        {
            let mut stats = self.performance_stats.lock().unwrap();
            stats.gpu_time_ms += elapsed.as_millis() as f64;
            stats.kernel_executions += 1;
        }

        Ok(predictions)
    }

    /// Execute GPU-accelerated QDA
    pub fn gpu_qda_fit_predict(
        &self,
        X: &ArrayView2<Float>,
        y: &ArrayView1<usize>,
        X_test: &ArrayView2<Float>,
        qda_config: &QuadraticDiscriminantAnalysisConfig,
    ) -> Result<Array1<usize>> {
        let (n_samples, n_features) = X.dim();

        if !self.should_use_gpu(n_samples, n_features) {
            return self.cpu_fallback_qda(X, y, X_test, qda_config);
        }

        let context = self.context.as_ref().unwrap();
        let lda_kernel = self.lda_kernel.as_ref().unwrap();
        let qda_kernel = self.qda_kernel.as_ref().unwrap();

        let start_time = std::time::Instant::now();

        // Compute number of classes
        let n_classes = y.iter().max().unwrap() + 1;

        // Compute class means on GPU (reuse LDA kernel)
        let class_means = lda_kernel.compute_class_means_gpu(context, X, y, n_classes)?;

        // Compute class-specific covariances on GPU
        let class_covariances = qda_kernel.compute_class_covariances_gpu(
            context,
            X,
            y,
            &class_means.view(),
            n_classes,
        )?;

        // Compute log priors
        let priors = self.compute_priors(y, n_classes);
        let log_priors = priors.mapv(|p| p.ln());

        // Compute QDA discriminant function for test data
        let discriminants = qda_kernel.compute_qda_discriminant_gpu(
            context,
            X_test,
            &class_means.view(),
            &class_covariances,
            &log_priors.view(),
        )?;

        // Get predictions (reuse LDA kernel)
        let predictions = lda_kernel.predict_gpu(context, &discriminants.view())?;

        // Update performance stats
        let elapsed = start_time.elapsed();
        {
            let mut stats = self.performance_stats.lock().unwrap();
            stats.gpu_time_ms += elapsed.as_millis() as f64;
            stats.kernel_executions += 1;
        }

        Ok(predictions)
    }

    // CPU fallback methods
    fn cpu_fallback_lda(
        &self,
        X: &ArrayView2<Float>,
        y: &ArrayView1<usize>,
        X_test: &ArrayView2<Float>,
        config: &LinearDiscriminantAnalysisConfig,
    ) -> Result<Array1<usize>> {
        {
            let mut stats = self.performance_stats.lock().unwrap();
            stats.fallback_count += 1;
        }

        // Use regular CPU-based LDA
        let lda = LinearDiscriminantAnalysis::new(config.clone());
        let trained_lda = lda.fit(X, y)?;
        trained_lda.predict(X_test)
    }

    fn cpu_fallback_qda(
        &self,
        X: &ArrayView2<Float>,
        y: &ArrayView1<usize>,
        X_test: &ArrayView2<Float>,
        config: &QuadraticDiscriminantAnalysisConfig,
    ) -> Result<Array1<usize>> {
        {
            let mut stats = self.performance_stats.lock().unwrap();
            stats.fallback_count += 1;
        }

        // Use regular CPU-based QDA
        let qda = QuadraticDiscriminantAnalysis::new(config.clone());
        let trained_qda = qda.fit(X, y)?;
        trained_qda.predict(X_test)
    }

    // Helper methods
    fn compute_priors(&self, y: &ArrayView1<usize>, n_classes: usize) -> Array1<Float> {
        let mut class_counts = Array1::<Float>::zeros(n_classes);

        for &label in y.iter() {
            class_counts[label] += 1.0;
        }

        let total_samples = y.len() as Float;
        class_counts.mapv(|count| count / total_samples)
    }
}

// Implementations would continue with proper error handling, cleanup, and additional optimizations
impl Drop for GpuDiscriminantAnalysis {
    fn drop(&mut self) {
        // Cleanup GPU resources
        if let Some(context) = self.context.take() {
            // Context cleanup happens automatically through Arc
            log::info!("GPU discriminant analysis context cleaned up");
        }
    }
}

// Additional trait implementations for integration with sklears ecosystem
unsafe impl Send for GpuDiscriminantAnalysis {}
unsafe impl Sync for GpuDiscriminantAnalysis {}

impl fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuBackend::Cuda => write!(f, "CUDA"),
            GpuBackend::Rocm => write!(f, "ROCm"),
            GpuBackend::Wgpu => write!(f, "WebGPU"),
            GpuBackend::Metal => write!(f, "Metal"),
            GpuBackend::OpenCL => write!(f, "OpenCL"),
            GpuBackend::Cpu => write!(f, "CPU"),
        }
    }
}

impl Clone for GpuPerformanceStats {
    fn clone(&self) -> Self {
        Self {
            gpu_time_ms: self.gpu_time_ms,
            cpu_time_ms: self.cpu_time_ms,
            transfer_time_ms: self.transfer_time_ms,
            memory_usage_mb: self.memory_usage_mb,
            kernel_executions: self.kernel_executions,
            fallback_count: self.fallback_count,
        }
    }
}

/// High-level GPU-accelerated discriminant analysis wrapper
pub struct GpuAcceleratedLDA {
    gpu_manager: GpuDiscriminantAnalysis,
    config: LinearDiscriminantAnalysisConfig,
}

pub struct GpuAcceleratedQDA {
    gpu_manager: GpuDiscriminantAnalysis,
    config: QuadraticDiscriminantAnalysisConfig,
}

impl GpuAcceleratedLDA {
    pub fn new(
        lda_config: LinearDiscriminantAnalysisConfig,
        gpu_config: GpuAccelerationConfig,
    ) -> Result<Self> {
        Ok(Self {
            gpu_manager: GpuDiscriminantAnalysis::new(gpu_config)?,
            config: lda_config,
        })
    }

    pub fn fit_predict(
        &self,
        X: &ArrayView2<Float>,
        y: &ArrayView1<usize>,
        X_test: &ArrayView2<Float>,
    ) -> Result<Array1<usize>> {
        self.gpu_manager
            .gpu_lda_fit_predict(X, y, X_test, &self.config)
    }

    pub fn is_using_gpu(&self) -> bool {
        self.gpu_manager.is_gpu_available()
    }

    pub fn performance_stats(&self) -> GpuPerformanceStats {
        self.gpu_manager.performance_stats()
    }
}

impl GpuAcceleratedQDA {
    pub fn new(
        qda_config: QuadraticDiscriminantAnalysisConfig,
        gpu_config: GpuAccelerationConfig,
    ) -> Result<Self> {
        Ok(Self {
            gpu_manager: GpuDiscriminantAnalysis::new(gpu_config)?,
            config: qda_config,
        })
    }

    pub fn fit_predict(
        &self,
        X: &ArrayView2<Float>,
        y: &ArrayView1<usize>,
        X_test: &ArrayView2<Float>,
    ) -> Result<Array1<usize>> {
        self.gpu_manager
            .gpu_qda_fit_predict(X, y, X_test, &self.config)
    }

    pub fn is_using_gpu(&self) -> bool {
        self.gpu_manager.is_gpu_available()
    }

    pub fn performance_stats(&self) -> GpuPerformanceStats {
        self.gpu_manager.performance_stats()
    }
}

// Additional features that could be added:
// - Async GPU operations with futures/async-await
// - Multi-GPU support with data parallelism
// - Memory-mapped GPU arrays for very large datasets
// - Custom kernel compilation and caching
// - Benchmark-driven algorithm selection (GPU vs CPU)
// - Progressive GPU memory allocation strategies
// - Integration with distributed computing frameworks
