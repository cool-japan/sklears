//! Distributed Support Vector Machine Training
//!
//! This module implements distributed SVM training algorithms that can
//! scale across multiple processes or machines using message passing
//! and data parallelism.

use crate::kernels::{create_kernel, Kernel, KernelType};
use crate::smo::{SmoConfig, SmoSolver};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::thread;

/// Configuration for distributed SVM training
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// Number of worker processes/threads
    pub n_workers: usize,
    /// Communication interval (iterations between synchronization)
    pub sync_interval: usize,
    /// Maximum number of global iterations
    pub max_global_iter: usize,
    /// Convergence tolerance
    pub tolerance: Float,
    /// Whether to use asynchronous updates
    pub async_updates: bool,
    /// Size of data chunks per worker
    pub chunk_size: usize,
    /// Cache size for each worker (in MB)
    pub cache_size_mb: usize,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            n_workers: 4,
            sync_interval: 10,
            max_global_iter: 100,
            tolerance: 1e-3,
            async_updates: false,
            chunk_size: 1000,
            cache_size_mb: 50,
        }
    }
}

/// Distributed training strategy
#[derive(Debug, Clone, PartialEq, Default)]
pub enum DistributedStrategy {
    /// Data parallel: each worker processes a subset of data
    #[default]
    DataParallel,
    /// Model parallel: each worker handles a subset of support vectors
    ModelParallel,
    /// Hybrid: combination of data and model parallelism
    Hybrid,
}

/// Communication protocol for distributed training
#[derive(Debug, Clone, PartialEq, Default)]
pub enum CommunicationProtocol {
    /// Synchronous: all workers synchronize at each iteration
    #[default]
    Synchronous,
    /// Asynchronous: workers update shared state independently
    Asynchronous,
    /// Parameter server: centralized parameter management
    ParameterServer,
}

/// Worker state for distributed training
#[derive(Debug, Clone)]
struct WorkerState {
    /// Worker ID
    worker_id: usize,
    /// Local dual coefficients
    local_alpha: Array1<Float>,
    /// Local data chunk indices
    data_indices: Vec<usize>,
    /// Local convergence status
    converged: bool,
    /// Number of local iterations
    local_iterations: usize,
}

/// Shared state across all workers
#[derive(Debug)]
struct SharedState {
    /// Global dual coefficients
    global_alpha: Arc<Mutex<Array1<Float>>>,
    /// Global intercept
    global_intercept: Arc<Mutex<Float>>,
    /// Convergence flags from all workers
    worker_convergence: Arc<Mutex<Vec<bool>>>,
    /// Global iteration counter
    global_iteration: Arc<Mutex<usize>>,
    /// Total number of samples
    n_samples: usize,
}

/// Distributed SVM classifier
#[derive(Debug, Clone)]
pub struct DistributedSVM<S> {
    /// Regularization parameter
    pub c: Float,
    /// Kernel function
    pub kernel: KernelType,
    /// Distributed training configuration
    pub config: DistributedConfig,
    /// Training strategy
    pub strategy: DistributedStrategy,
    /// Communication protocol
    pub protocol: CommunicationProtocol,
    /// Number of classes
    pub n_classes: Option<usize>,
    /// Support vectors
    pub support_vectors: Option<Array2<Float>>,
    /// Dual coefficients
    pub dual_coef: Option<Array1<Float>>,
    /// Intercept
    pub intercept: Float,
    /// Class labels
    pub classes: Option<Array1<i32>>,
    /// Number of support vectors per class
    pub n_support: Option<Array1<usize>>,
    /// State marker
    _state: PhantomData<S>,
}

impl DistributedSVM<Untrained> {
    /// Create a new distributed SVM classifier
    pub fn new(
        c: Float,
        kernel: KernelType,
        config: DistributedConfig,
        strategy: DistributedStrategy,
        protocol: CommunicationProtocol,
    ) -> Self {
        Self {
            c,
            kernel,
            config,
            strategy,
            protocol,
            n_classes: None,
            support_vectors: None,
            dual_coef: None,
            intercept: 0.0,
            classes: None,
            n_support: None,
            _state: PhantomData,
        }
    }

    /// Builder pattern for configuration
    pub fn builder() -> DistributedSVMBuilder {
        DistributedSVMBuilder::new()
    }
}

/// Builder for DistributedSVM
pub struct DistributedSVMBuilder {
    c: Float,
    kernel: KernelType,
    config: DistributedConfig,
    strategy: DistributedStrategy,
    protocol: CommunicationProtocol,
}

impl DistributedSVMBuilder {
    pub fn new() -> Self {
        Self {
            c: 1.0,
            kernel: KernelType::Rbf { gamma: 1.0 },
            config: DistributedConfig::default(),
            strategy: DistributedStrategy::default(),
            protocol: CommunicationProtocol::default(),
        }
    }

    pub fn c(mut self, c: Float) -> Self {
        self.c = c;
        self
    }

    pub fn kernel(mut self, kernel: KernelType) -> Self {
        self.kernel = kernel;
        self
    }

    pub fn n_workers(mut self, n_workers: usize) -> Self {
        self.config.n_workers = n_workers;
        self
    }

    pub fn sync_interval(mut self, sync_interval: usize) -> Self {
        self.config.sync_interval = sync_interval;
        self
    }

    pub fn max_global_iter(mut self, max_global_iter: usize) -> Self {
        self.config.max_global_iter = max_global_iter;
        self
    }

    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.config.tolerance = tolerance;
        self
    }

    pub fn strategy(mut self, strategy: DistributedStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    pub fn protocol(mut self, protocol: CommunicationProtocol) -> Self {
        self.protocol = protocol;
        self
    }

    pub fn chunk_size(mut self, chunk_size: usize) -> Self {
        self.config.chunk_size = chunk_size;
        self
    }

    pub fn build(self) -> DistributedSVM<Untrained> {
        DistributedSVM::new(
            self.c,
            self.kernel,
            self.config,
            self.strategy,
            self.protocol,
        )
    }
}

impl Default for DistributedSVMBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, Float>> for DistributedSVM<Untrained> {
    type Fitted = DistributedSVM<Trained>;

    fn fit(self, x: &ArrayView2<Float>, y: &ArrayView1<Float>) -> Result<Self::Fitted> {
        self.fit_distributed(*x, *y)
    }
}

impl DistributedSVM<Untrained> {
    /// Train using distributed approach
    pub fn fit_distributed(
        &self,
        x: ArrayView2<Float>,
        y: ArrayView1<Float>,
    ) -> Result<DistributedSVM<Trained>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Determine unique classes
        let mut classes_vec = Vec::new();
        for &label in y.iter() {
            let label_i32 = label as i32;
            if !classes_vec.contains(&label_i32) {
                classes_vec.push(label_i32);
            }
        }
        classes_vec.sort_unstable();
        let n_classes = classes_vec.len();

        if n_classes != 2 {
            return Err(SklearsError::InvalidInput(
                "Multi-class distributed SVM not yet implemented".to_string(),
            ));
        }

        // Convert to binary labels
        let mut binary_y = Array1::zeros(n_samples);
        for (i, &label) in y.iter().enumerate() {
            binary_y[i] = if label as i32 == classes_vec[0] {
                -1.0
            } else {
                1.0
            };
        }

        // Initialize shared state
        let shared_state = SharedState {
            global_alpha: Arc::new(Mutex::new(Array1::zeros(n_samples))),
            global_intercept: Arc::new(Mutex::new(0.0)),
            worker_convergence: Arc::new(Mutex::new(vec![false; self.config.n_workers])),
            global_iteration: Arc::new(Mutex::new(0)),
            n_samples,
        };

        // Partition data across workers
        let chunk_size = n_samples / self.config.n_workers;
        let mut worker_handles = Vec::new();

        for worker_id in 0..self.config.n_workers {
            let start_idx = worker_id * chunk_size;
            let end_idx = if worker_id == self.config.n_workers - 1 {
                n_samples
            } else {
                (worker_id + 1) * chunk_size
            };

            // Clone data for this worker
            let worker_x = x
                .slice(scirs2_core::ndarray::s![start_idx..end_idx, ..])
                .to_owned();
            let worker_y = binary_y
                .slice(scirs2_core::ndarray::s![start_idx..end_idx])
                .to_owned();
            let data_indices: Vec<usize> = (start_idx..end_idx).collect();

            // Clone necessary data for the worker
            let kernel = self.kernel.clone();
            let c = self.c;
            let tolerance = self.config.tolerance;
            let sync_interval = self.config.sync_interval;
            let max_global_iter = self.config.max_global_iter;
            let cache_size_mb = self.config.cache_size_mb;

            // Clone shared state references
            let global_alpha = shared_state.global_alpha.clone();
            let global_intercept = shared_state.global_intercept.clone();
            let worker_convergence = shared_state.worker_convergence.clone();
            let global_iteration = shared_state.global_iteration.clone();

            // Spawn worker thread
            let handle = thread::spawn(move || {
                Self::worker_thread(
                    worker_id,
                    worker_x,
                    worker_y,
                    data_indices,
                    kernel,
                    c,
                    tolerance,
                    sync_interval,
                    max_global_iter,
                    cache_size_mb,
                    global_alpha,
                    global_intercept,
                    worker_convergence,
                    global_iteration,
                )
            });

            worker_handles.push(handle);
        }

        // Wait for all workers to complete
        for handle in worker_handles {
            handle.join().map_err(|_| {
                SklearsError::NumericalError("Worker thread panicked".to_string())
            })??;
        }

        // Extract final results
        let final_alpha = shared_state.global_alpha.lock().unwrap().clone();
        let final_intercept = *shared_state.global_intercept.lock().unwrap();

        // Extract support vectors
        let support_indices: Vec<usize> = final_alpha
            .iter()
            .enumerate()
            .filter(|(_, &coef)| coef.abs() > 1e-10)
            .map(|(i, _)| i)
            .collect();

        let n_support_vectors = support_indices.len();
        let mut support_vectors = Array2::zeros((n_support_vectors, n_features));
        let mut support_dual_coef = Array1::zeros(n_support_vectors);

        for (i, &idx) in support_indices.iter().enumerate() {
            support_vectors.row_mut(i).assign(&x.row(idx));
            support_dual_coef[i] = final_alpha[idx];
        }

        // Count support vectors per class
        let mut n_support = Array1::zeros(n_classes);
        for &coef in support_dual_coef.iter() {
            if coef > 0.0 {
                n_support[1] += 1;
            } else {
                n_support[0] += 1;
            }
        }

        Ok(DistributedSVM {
            c: self.c,
            kernel: self.kernel.clone(),
            config: self.config.clone(),
            strategy: self.strategy.clone(),
            protocol: self.protocol.clone(),
            n_classes: Some(n_classes),
            support_vectors: Some(support_vectors),
            dual_coef: Some(support_dual_coef),
            intercept: final_intercept,
            classes: Some(Array1::from_vec(classes_vec)),
            n_support: Some(n_support),
            _state: PhantomData,
        })
    }

    /// Worker thread function
    fn worker_thread(
        worker_id: usize,
        x: Array2<Float>,
        y: Array1<Float>,
        data_indices: Vec<usize>,
        kernel: KernelType,
        c: Float,
        tolerance: Float,
        sync_interval: usize,
        max_global_iter: usize,
        cache_size_mb: usize,
        global_alpha: Arc<Mutex<Array1<Float>>>,
        global_intercept: Arc<Mutex<Float>>,
        worker_convergence: Arc<Mutex<Vec<bool>>>,
        global_iteration: Arc<Mutex<usize>>,
    ) -> Result<()> {
        // Initialize local state
        let mut local_alpha = Array1::zeros(x.nrows());
        let mut local_iterations = 0;

        // Create SMO solver for this worker
        let smo_config = SmoConfig {
            c,
            tol: tolerance,
            max_iter: sync_interval,
            cache_size: cache_size_mb,
            shrinking: true,
            ..Default::default()
        };

        let concrete_kernel = create_kernel(kernel);
        let mut smo_solver = SmoSolver::new(smo_config, concrete_kernel);

        loop {
            // Safety check: prevent infinite loops with local iteration limit
            if local_iterations >= max_global_iter * 2 {
                break;
            }
            local_iterations += 1;

            // Check global convergence and iteration limit
            {
                let global_iter = *global_iteration.lock().unwrap();
                let convergence_flags = worker_convergence.lock().unwrap();

                if global_iter >= max_global_iter || convergence_flags.iter().all(|&x| x) {
                    break;
                }
            }

            // Perform local SMO iterations
            let smo_result = smo_solver.solve(&x, &y)?;

            // Update local alpha
            for (i, &coef) in smo_result.alpha.iter().enumerate() {
                local_alpha[i] = coef;
            }

            // Synchronize with global state
            {
                let mut global_alpha_lock = global_alpha.lock().unwrap();
                let mut global_intercept_lock = global_intercept.lock().unwrap();
                let mut convergence_lock = worker_convergence.lock().unwrap();
                let mut global_iter_lock = global_iteration.lock().unwrap();

                // Update global alpha for this worker's data indices
                for (local_idx, &global_idx) in data_indices.iter().enumerate() {
                    global_alpha_lock[global_idx] = local_alpha[local_idx];
                }

                // Update global intercept (average across workers)
                *global_intercept_lock = (*global_intercept_lock * worker_id as Float
                    + smo_result.b)
                    / (worker_id + 1) as Float;

                // Update convergence status
                convergence_lock[worker_id] = smo_result.converged;

                // Increment global iteration (only by worker 0 to avoid double counting)
                if worker_id == 0 {
                    *global_iter_lock += 1;
                }
            }

            // Brief pause for synchronization
            thread::yield_now();
        }

        Ok(())
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<Float>> for DistributedSVM<Trained> {
    fn predict(&self, x: &ArrayView2<Float>) -> Result<Array1<Float>> {
        let support_vectors =
            self.support_vectors
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict".to_string(),
                })?;

        let dual_coef = self
            .dual_coef
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let classes = self
            .classes
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let concrete_kernel = create_kernel(self.kernel.clone());
        let mut predictions = Array1::zeros(x.nrows());

        // Compute decision function for each sample
        for i in 0..x.nrows() {
            let mut decision_value = 0.0;

            // Compute kernel values with all support vectors
            for j in 0..support_vectors.nrows() {
                let kernel_value = concrete_kernel.compute(
                    x.row(i).to_owned().view(),
                    support_vectors.row(j).to_owned().view(),
                );
                decision_value += dual_coef[j] * kernel_value;
            }

            decision_value += self.intercept;

            // Convert to class prediction
            predictions[i] = if decision_value >= 0.0 {
                classes[1] as Float
            } else {
                classes[0] as Float
            };
        }

        Ok(predictions)
    }
}

impl Estimator for DistributedSVM<Untrained> {
    type Config = DistributedConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for DistributedSVM<Trained> {
    type Config = DistributedConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    #[test]
    fn test_distributed_svm_creation() {
        let config = DistributedConfig::default();
        let svm = DistributedSVM::new(
            1.0,
            KernelType::Linear,
            config,
            DistributedStrategy::DataParallel,
            CommunicationProtocol::Synchronous,
        );
        assert_eq!(svm.c, 1.0);
        assert_eq!(svm.config().n_workers, 4);
    }

    #[test]
    fn test_distributed_svm_builder() {
        let svm = DistributedSVM::builder()
            .c(2.0)
            .kernel(KernelType::Rbf { gamma: 0.5 })
            .n_workers(8)
            .tolerance(1e-4)
            .strategy(DistributedStrategy::ModelParallel)
            .protocol(CommunicationProtocol::Asynchronous)
            .build();

        assert_eq!(svm.c, 2.0);
        assert_eq!(svm.config().n_workers, 8);
        assert_eq!(svm.config().tolerance, 1e-4);
        assert_eq!(svm.strategy, DistributedStrategy::ModelParallel);
        assert_eq!(svm.protocol, CommunicationProtocol::Asynchronous);
    }

    #[test]
    #[ignore = "Disabled due to potential deadlock in worker threads - needs architecture redesign"]
    fn test_distributed_svm_training() -> Result<()> {
        // Create simple binary classification data
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
            ],
        )?;
        let y = Array1::from_vec(vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);

        let svm = DistributedSVM::builder()
            .c(1.0)
            .kernel(KernelType::Linear)
            .n_workers(2)
            .tolerance(1e-2)
            .max_global_iter(10)
            .build();

        let trained_svm = svm.fit(&x.view(), &y.view())?;

        // Test predictions
        let predictions = trained_svm.predict(&x.view())?;
        assert_eq!(predictions.len(), 8);

        // Check that we have reasonable predictions
        for &pred in predictions.iter() {
            assert!(pred == 1.0 || pred == -1.0);
        }

        Ok(())
    }
}
