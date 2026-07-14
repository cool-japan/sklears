//! Optional GPU-acceleration wrappers for calibration methods.
//!
//! These wrappers expose a stable API for GPU-accelerated calibration.
//!
//! * **Default (`gpu` feature off) build**: pure Rust, no GPU runtime linked.
//!   `GpuUtils` reports no devices and every wrapper transparently delegates
//!   to the wrapped CPU calibrator, producing identical, correct results.
//! * **`gpu` feature on**: device discovery, memory statistics, and the
//!   temperature-scaling prediction fast path route through the real
//!   `oxicuda-driver` / `oxicuda-blas` stack via [`sklears_core::gpu`]. On a
//!   host with no CUDA-capable GPU (e.g. this workspace's own macOS
//!   development machine) [`sklears_core::gpu::GpuBackend::detect`] honestly
//!   returns `Ok(None)` and every wrapper still falls back to the CPU path --
//!   the `gpu` feature changes *how a real device would be found and used*,
//!   it never fabricates one.
//!
//! The wrappers never fabricate a result: when no device is available they
//! run the real CPU calibration.

use crate::isotonic::IsotonicCalibrator;
use crate::temperature::TemperatureScalingCalibrator;
use crate::CalibrationEstimator;
use scirs2_core::ndarray::Array1;
use sklears_core::{error::SklearsError, types::Float};
use std::sync::{Arc, Mutex};

#[cfg(feature = "gpu")]
use sklears_core::gpu::GpuBackend;

/// Error raised by the GPU utility layer.
#[derive(Debug)]
pub struct GpuError {
    message: String,
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "GPU Error: {}", self.message)
    }
}

impl std::error::Error for GpuError {}

/// Handle to a GPU device.
///
/// Without the `gpu` feature no backend is linked, so this can never be
/// constructed (all `GpuUtils` lookups return `None`). With the `gpu` feature
/// it wraps a real [`GpuBackend`] returned by
/// [`GpuBackend::detect`]/[`GpuBackend::with_device_id`], so holding one is
/// existence-proof of a real, initialised CUDA device.
#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub id: u32,
    #[cfg(feature = "gpu")]
    backend: GpuBackend,
}

impl GpuDevice {
    /// The real backend behind this device handle (only exists under the
    /// `gpu` feature: without it no `GpuDevice` can ever be constructed).
    #[cfg(feature = "gpu")]
    fn backend(&self) -> &GpuBackend {
        &self.backend
    }
}

/// Memory statistics.
///
/// With no GPU backend linked (or none detected) these report host RAM. With
/// the `gpu` feature enabled and a device present, they report real device
/// memory queried via `cuMemGetInfo` (through
/// [`sklears_core::gpu::GpuBackend::memory_info`]).
#[derive(Debug)]
pub struct GpuMemoryStats {
    pub total: u64,
    pub used: u64,
    pub free: u64,
}

/// Device-discovery utility.
///
/// Honest by construction: without the `gpu` feature (or with it but no CUDA
/// device present) it discovers no devices and reports host memory. It never
/// fabricates a "simulated" device.
#[derive(Debug, Default)]
pub struct GpuUtils {
    #[cfg(feature = "gpu")]
    backend: Option<GpuBackend>,
}

impl GpuUtils {
    pub fn new() -> Self {
        Self::default()
    }

    /// Initialise device discovery.
    ///
    /// Without the `gpu` feature this is a no-op (there is nothing to
    /// discover). With it, this calls [`GpuBackend::detect`], which performs
    /// real CUDA driver initialisation, device enumeration (picking the
    /// device with the most free memory), context creation, and BLAS handle
    /// setup. On a host with no CUDA driver/device this returns `Ok(None)`
    /// internally (not an error) and every subsequent lookup below honestly
    /// reports "no device".
    pub fn init_devices(&mut self) -> Result<(), GpuError> {
        #[cfg(feature = "gpu")]
        {
            self.backend = GpuBackend::detect().map_err(|e| GpuError {
                message: e.to_string(),
            })?;
        }
        Ok(())
    }

    /// Look up a device by id.
    ///
    /// Without the `gpu` feature this is always `None`. With it, `Some` only
    /// when [`init_devices`](Self::init_devices) found a real device whose
    /// ordinal matches `id` -- there is a single detected device per
    /// `GpuUtils` instance (the best one found by `detect()`), so this is
    /// effectively "is the detected device's id equal to `id`".
    pub fn get_device(&self, id: u32) -> Option<GpuDevice> {
        #[cfg(not(feature = "gpu"))]
        {
            let _ = id;
            None
        }
        #[cfg(feature = "gpu")]
        {
            let backend = self.backend.as_ref()?;
            if backend.device_id() as u32 == id {
                return Some(GpuDevice {
                    id,
                    backend: backend.clone(),
                });
            }
            None
        }
    }

    /// Select the best available device (the one `detect()` chose: highest
    /// free memory). `None` without the `gpu` feature, or when no CUDA
    /// device was found.
    pub fn get_best_device(&self) -> Option<GpuDevice> {
        #[cfg(feature = "gpu")]
        {
            let backend = self.backend.as_ref()?;
            Some(GpuDevice {
                id: backend.device_id() as u32,
                backend: backend.clone(),
            })
        }
        #[cfg(not(feature = "gpu"))]
        {
            None
        }
    }

    /// Report memory statistics.
    ///
    /// With the `gpu` feature enabled and a device detected, this queries
    /// live device memory via `cuMemGetInfo`
    /// ([`GpuBackend::memory_info`]). Otherwise (no `gpu` feature, or no
    /// device found) it reports host RAM read from `/proc/meminfo` (Linux);
    /// zeros if the query fails or the platform has no such file -- never an
    /// invented constant.
    pub fn get_memory_stats(&self) -> GpuMemoryStats {
        #[cfg(feature = "gpu")]
        {
            if let Some(backend) = &self.backend {
                if let Ok(info) = backend.memory_info() {
                    return GpuMemoryStats {
                        total: info.total as u64,
                        used: info.used as u64,
                        free: info.free as u64,
                    };
                }
            }
        }
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                let mut total_kb: Option<u64> = None;
                let mut avail_kb: Option<u64> = None;
                for line in content.lines() {
                    if let Some(rest) = line.strip_prefix("MemTotal:") {
                        total_kb = rest
                            .split_whitespace()
                            .next()
                            .and_then(|s| s.parse::<u64>().ok());
                    } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
                        avail_kb = rest
                            .split_whitespace()
                            .next()
                            .and_then(|s| s.parse::<u64>().ok());
                    }
                    if total_kb.is_some() && avail_kb.is_some() {
                        break;
                    }
                }
                if let (Some(t), Some(a)) = (total_kb, avail_kb) {
                    let used_kb = t.saturating_sub(a);
                    return GpuMemoryStats {
                        total: t * 1024,
                        used: used_kb * 1024,
                        free: a * 1024,
                    };
                }
            }
        }
        GpuMemoryStats {
            total: 0,
            used: 0,
            free: 0,
        }
    }

    /// Device utilisation.
    ///
    /// Always `0.0`. Without the `gpu` feature there is no device to measure.
    /// With the `gpu` feature, `oxicuda-driver` 0.4.0 wraps the CUDA *driver*
    /// API only -- SM/compute occupancy sampling is an NVML API
    /// (`nvmlDeviceGetUtilizationRates`), which has no `oxicuda-driver`
    /// binding, so there is no real occupancy figure to report here.
    /// Returning device *memory* utilization instead (which the driver API
    /// can answer via `cuMemGetInfo`) would silently redefine what
    /// "utilization" means and is deliberately not done; callers that want a
    /// device memory occupancy figure should use
    /// [`get_memory_stats`](Self::get_memory_stats) instead. (deferred
    /// 2026-07-06: no NVML utilization-rate binding in oxicuda-driver 0.4.0)
    pub fn get_utilization(&self) -> Float {
        0.0
    }
}

/// GPU-accelerated calibration framework.
///
/// Wraps any [`CalibrationEstimator`]. When a GPU device is available and the
/// data exceeds the configured threshold the GPU path would be used; with no
/// backend linked (or none detected) the wrapper delegates to the wrapped CPU
/// calibrator, so the results are exactly those of the underlying method.
#[derive(Debug)]
pub struct GpuCalibratedClassifier {
    calibrator: Box<dyn CalibrationEstimator>,
    gpu_utils: Arc<Mutex<GpuUtils>>,
    device_id: u32,
    batch_size: usize,
    use_gpu_threshold: usize,
    config: GpuCalibrationConfig,
}

/// Configuration for GPU calibration.
#[derive(Debug, Clone)]
pub struct GpuCalibrationConfig {
    /// Preferred GPU device ID (None for auto-selection).
    pub device_id: Option<u32>,
    /// Batch size for GPU operations.
    pub batch_size: usize,
    /// Minimum data size to prefer GPU acceleration.
    pub gpu_threshold: usize,
    /// Enable performance profiling (effective only with a GPU backend).
    pub enable_profiling: bool,
    /// Use mixed precision (FP16/FP32) for memory efficiency.
    pub use_mixed_precision: bool,
    /// Memory limit per GPU operation (in bytes).
    pub memory_limit: u64,
}

impl Default for GpuCalibrationConfig {
    fn default() -> Self {
        Self {
            device_id: None,
            batch_size: 1024,
            gpu_threshold: 10000,
            enable_profiling: false,
            use_mixed_precision: false,
            memory_limit: 1_073_741_824, // 1 GB
        }
    }
}

impl GpuCalibratedClassifier {
    /// Create a new GPU-accelerated calibrated classifier.
    ///
    /// Succeeds even when no GPU is present: the wrapper falls back to the CPU
    /// calibrator. An explicit `device_id` that does not exist is reported as an
    /// honest error rather than silently ignored.
    pub fn new(
        calibrator: Box<dyn CalibrationEstimator>,
        config: GpuCalibrationConfig,
    ) -> Result<Self, SklearsError> {
        let mut gpu_utils = GpuUtils::new();
        gpu_utils.init_devices().map_err(|e| {
            SklearsError::InvalidOperation(format!("Failed to initialize GPU devices: {}", e))
        })?;

        let device_id = match config.device_id {
            Some(id) => {
                if gpu_utils.get_device(id).is_none() {
                    return Err(SklearsError::InvalidOperation(format!(
                        "GPU device {} not found; no GPU backend is linked",
                        id
                    )));
                }
                id
            }
            None => gpu_utils.get_best_device().map(|d| d.id).unwrap_or(0),
        };

        Ok(Self {
            calibrator,
            gpu_utils: Arc::new(Mutex::new(gpu_utils)),
            device_id,
            batch_size: config.batch_size,
            use_gpu_threshold: config.gpu_threshold,
            config,
        })
    }

    /// Whether a usable GPU device is currently available.
    fn gpu_available(&self) -> bool {
        self.gpu_utils
            .lock()
            .map(|g| g.get_best_device().is_some())
            .unwrap_or(false)
    }

    /// Whether `data_size` exceeds the configured GPU threshold.
    pub fn exceeds_gpu_threshold(&self, data_size: usize) -> bool {
        data_size >= self.use_gpu_threshold
    }

    /// Whether the GPU path would actually run for `data_size`.
    ///
    /// Requires both a usable device and a data size over the threshold. With no
    /// GPU backend linked this is always `false`, and calibration runs on CPU.
    pub fn should_use_gpu(&self, data_size: usize) -> bool {
        self.gpu_available() && self.exceeds_gpu_threshold(data_size)
    }

    /// Configured batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get GPU device information.
    pub fn get_device_info(&self) -> Result<String, SklearsError> {
        let gpu_utils = self
            .gpu_utils
            .lock()
            .map_err(|e| SklearsError::Other(format!("mutex lock poisoned: {}", e)))?;
        if let Some(device) = gpu_utils.get_device(self.device_id) {
            Ok(format!(
                "GPU Device ID: {} (GPU acceleration available)",
                device.id
            ))
        } else {
            Err(SklearsError::InvalidOperation(
                "GPU device not available; calibration runs on CPU".to_string(),
            ))
        }
    }

    /// Get memory statistics. Reports host RAM when no GPU is present.
    pub fn get_memory_stats(&self) -> Result<String, SklearsError> {
        let gpu_utils = self
            .gpu_utils
            .lock()
            .map_err(|e| SklearsError::Other(format!("mutex lock poisoned: {}", e)))?;
        let stats = gpu_utils.get_memory_stats();
        let label = if gpu_utils.get_best_device().is_some() {
            "GPU Memory"
        } else {
            "Host Memory (no GPU)"
        };

        Ok(format!(
            "{} - Total: {:.1} GB, Used: {:.1} GB, Free: {:.1} GB",
            label,
            stats.total as f64 / 1_073_741_824.0,
            stats.used as f64 / 1_073_741_824.0,
            stats.free as f64 / 1_073_741_824.0
        ))
    }

    /// Get GPU utilization (0.0 when no GPU is present).
    pub fn get_utilization(&self) -> Result<f64, SklearsError> {
        let gpu_utils = self
            .gpu_utils
            .lock()
            .map_err(|e| SklearsError::Other(format!("mutex lock poisoned: {}", e)))?;
        Ok(gpu_utils.get_utilization())
    }
}

impl CalibrationEstimator for GpuCalibratedClassifier {
    fn fit(
        &mut self,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
    ) -> sklears_core::error::Result<()> {
        // No GPU kernel is linked; run the real calibration on CPU. This is the
        // genuine fit of the wrapped calibrator, not a stand-in.
        self.calibrator.fit(probabilities, y_true)
    }

    fn predict_proba(
        &self,
        probabilities: &Array1<Float>,
    ) -> sklears_core::error::Result<Array1<Float>> {
        self.calibrator.predict_proba(probabilities)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(GpuCalibratedClassifier {
            calibrator: self.calibrator.clone_box(),
            gpu_utils: self.gpu_utils.clone(),
            device_id: self.device_id,
            batch_size: self.batch_size,
            use_gpu_threshold: self.use_gpu_threshold,
            config: self.config.clone(),
        })
    }
}

/// GPU-accelerated isotonic calibration (delegates to CPU isotonic when no GPU).
#[derive(Debug)]
pub struct GpuIsotonicCalibrator {
    gpu_calibrator: GpuCalibratedClassifier,
}

impl GpuIsotonicCalibrator {
    /// Create new GPU isotonic calibrator.
    pub fn new(config: GpuCalibrationConfig) -> Result<Self, SklearsError> {
        let cpu_calibrator = IsotonicCalibrator::new();
        let gpu_calibrator = GpuCalibratedClassifier::new(Box::new(cpu_calibrator), config)?;
        Ok(Self { gpu_calibrator })
    }
}

impl CalibrationEstimator for GpuIsotonicCalibrator {
    fn fit(
        &mut self,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
    ) -> sklears_core::error::Result<()> {
        self.gpu_calibrator.fit(probabilities, y_true)
    }

    fn predict_proba(
        &self,
        probabilities: &Array1<Float>,
    ) -> sklears_core::error::Result<Array1<Float>> {
        self.gpu_calibrator.predict_proba(probabilities)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(GpuIsotonicCalibrator {
            gpu_calibrator: GpuCalibratedClassifier {
                calibrator: self.gpu_calibrator.calibrator.clone_box(),
                gpu_utils: self.gpu_calibrator.gpu_utils.clone(),
                device_id: self.gpu_calibrator.device_id,
                batch_size: self.gpu_calibrator.batch_size,
                use_gpu_threshold: self.gpu_calibrator.use_gpu_threshold,
                config: self.gpu_calibrator.config.clone(),
            },
        })
    }
}

/// GPU-accelerated temperature scaling.
///
/// Fitting always runs on the CPU (the optimal temperature is found via a
/// small grid + line search over at most a few dozen scalar loss
/// evaluations -- not a workload worth a device round-trip). Prediction
/// applies the learned temperature to a batch of logits as `sigmoid(logit /
/// T)`, an embarrassingly-parallel elementwise transform.
///
/// The device sigmoid is built from the `ex2.approx` special-function unit,
/// which PTX only defines for `.f32`; there is no faithful f64 sigmoid kernel
/// in the oxicuda stack. The wrapper's native precision is f64, so the device
/// fast path is taken only when the caller explicitly opts into mixed
/// precision (`use_mixed_precision`): the scale and sigmoid step then runs as
/// two real f32 `oxicuda-blas` device kernels (`elementwise::scale`,
/// `elementwise::sigmoid`) on a present device above the configured GPU
/// threshold. Without that opt-in -- and whenever no `gpu` feature, no device,
/// or a small batch applies -- prediction runs on the CPU at full f64
/// precision via [`TemperatureScalingCalibrator::predict_proba`].
#[derive(Debug)]
pub struct GpuTemperatureScalingCalibrator {
    cpu: TemperatureScalingCalibrator,
    gpu_calibrator: GpuCalibratedClassifier,
}

impl GpuTemperatureScalingCalibrator {
    /// Create new GPU temperature scaling calibrator.
    pub fn new(config: GpuCalibrationConfig) -> Result<Self, SklearsError> {
        let cpu = TemperatureScalingCalibrator::new();
        let gpu_calibrator = GpuCalibratedClassifier::new(Box::new(cpu.clone()), config)?;
        Ok(Self {
            cpu,
            gpu_calibrator,
        })
    }

    /// Whether the GPU device fast path would run for a batch of this size:
    /// requires both a real detected device and a batch at/above the
    /// configured threshold.
    pub fn should_use_gpu(&self, data_size: usize) -> bool {
        self.gpu_calibrator.should_use_gpu(data_size)
    }

    /// Runs `sigmoid(logits / T)` for the already-fitted temperature on the
    /// detected device, in f32 (the only precision at which the device sigmoid
    /// exists).
    ///
    /// Returns `Ok(None)` (rather than an error) when the device fast path is
    /// declined -- because mixed precision was not requested, or no device is
    /// currently available -- so callers can transparently fall back to the
    /// exact f64 CPU path; device/kernel failures are still surfaced as `Err`.
    #[cfg(feature = "gpu")]
    fn try_gpu_predict(&self, logits: &[Float]) -> sklears_core::error::Result<Option<Vec<Float>>> {
        // At native f64 precision the device sigmoid is not available (see the
        // detailed note on the kernels below); decline the fast path so the
        // caller falls back to the exact CPU calibrator. The genuine device
        // path is taken only under an explicit mixed-precision opt-in.
        if !self.gpu_calibrator.config.use_mixed_precision {
            return Ok(None);
        }
        let device = {
            let gpu_utils = self
                .gpu_calibrator
                .gpu_utils
                .lock()
                .map_err(|e| SklearsError::Other(format!("mutex lock poisoned: {}", e)))?;
            gpu_utils.get_best_device()
        };
        let Some(device) = device else {
            return Ok(None);
        };
        let backend = device.backend();
        backend
            .context()
            .set_current()
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;

        // The activation is `sigmoid`, which on the device is built from the
        // `ex2.approx` special-function unit -- an instruction PTX only defines
        // for `.f32`. There is no faithful f64 sigmoid kernel in the oxicuda
        // stack, so at the wrapper's native f64 precision the device cannot
        // reproduce the CPU result to full precision. We run the genuine f32
        // device kernels only when the caller has explicitly opted into mixed
        // precision; otherwise we decline the fast path (`Ok(None)`) and let
        // `predict_proba` fall back to the exact f64 CPU calibrator.
        let n = logits.len();
        let host_f32: Vec<f32> = logits.iter().map(|&x| x as f32).collect();
        let input = oxicuda_memory::DeviceBuffer::from_host(&host_f32)
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
        let mut scaled = oxicuda_memory::DeviceBuffer::<f32>::zeroed(n)
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
        let inv_temperature = (1.0 / self.cpu.temperature()) as f32;
        oxicuda_blas::elementwise::scale(
            backend.blas(),
            n as u32,
            inv_temperature,
            &input,
            &mut scaled,
        )
        .map_err(|e| SklearsError::NumericalError(e.to_string()))?;

        let mut activated = oxicuda_memory::DeviceBuffer::<f32>::zeroed(n)
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
        oxicuda_blas::elementwise::sigmoid(backend.blas(), n as u32, &scaled, &mut activated)
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;

        let mut out_f32 = vec![0.0f32; n];
        activated
            .copy_to_host(&mut out_f32)
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
        let out: Vec<Float> = out_f32.into_iter().map(Float::from).collect();
        Ok(Some(out))
    }
}

impl CalibrationEstimator for GpuTemperatureScalingCalibrator {
    fn fit(
        &mut self,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
    ) -> sklears_core::error::Result<()> {
        CalibrationEstimator::fit(&mut self.cpu, probabilities, y_true)?;
        // Keep the device-utility wrapper's boxed copy in sync too, so
        // `get_device_info` / `get_memory_stats` / `get_utilization` continue
        // to reflect a fitted calibrator.
        self.gpu_calibrator.fit(probabilities, y_true)
    }

    fn predict_proba(
        &self,
        probabilities: &Array1<Float>,
    ) -> sklears_core::error::Result<Array1<Float>> {
        #[cfg(feature = "gpu")]
        {
            if self.should_use_gpu(probabilities.len()) {
                let logits: Vec<Float> = probabilities
                    .iter()
                    .map(|&p| {
                        let clamped = p.clamp(1e-15, 1.0 - 1e-15);
                        (clamped / (1.0 - clamped)).ln()
                    })
                    .collect();
                if let Some(calibrated) = self.try_gpu_predict(&logits)? {
                    return Ok(Array1::from_vec(calibrated));
                }
                // No device actually available despite `should_use_gpu`
                // returning true a moment ago (e.g. a race with another
                // process); fall through to the CPU path below.
            }
        }
        CalibrationEstimator::predict_proba(&self.cpu, probabilities)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(GpuTemperatureScalingCalibrator {
            cpu: self.cpu.clone(),
            gpu_calibrator: GpuCalibratedClassifier {
                calibrator: self.gpu_calibrator.calibrator.clone_box(),
                gpu_utils: self.gpu_calibrator.gpu_utils.clone(),
                device_id: self.gpu_calibrator.device_id,
                batch_size: self.gpu_calibrator.batch_size,
                use_gpu_threshold: self.gpu_calibrator.use_gpu_threshold,
                config: self.gpu_calibrator.config.clone(),
            },
        })
    }
}

/// Builder for GPU calibration configuration.
pub struct GpuCalibrationBuilder {
    config: GpuCalibrationConfig,
}

impl GpuCalibrationBuilder {
    /// Create new builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: GpuCalibrationConfig::default(),
        }
    }

    /// Set preferred GPU device ID.
    pub fn device_id(mut self, device_id: u32) -> Self {
        self.config.device_id = Some(device_id);
        self
    }

    /// Set batch size for GPU operations.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    /// Set minimum data size to prefer GPU acceleration.
    pub fn gpu_threshold(mut self, threshold: usize) -> Self {
        self.config.gpu_threshold = threshold;
        self
    }

    /// Enable performance profiling.
    pub fn enable_profiling(mut self) -> Self {
        self.config.enable_profiling = true;
        self
    }

    /// Enable mixed precision for memory efficiency.
    pub fn use_mixed_precision(mut self) -> Self {
        self.config.use_mixed_precision = true;
        self
    }

    /// Set memory limit per GPU operation.
    pub fn memory_limit(mut self, limit: u64) -> Self {
        self.config.memory_limit = limit;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> GpuCalibrationConfig {
        self.config
    }
}

impl Default for GpuCalibrationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_gpu_calibration_config() {
        let config = GpuCalibrationBuilder::new()
            .batch_size(512)
            .gpu_threshold(5000)
            .enable_profiling()
            .build();

        assert_eq!(config.batch_size, 512);
        assert_eq!(config.gpu_threshold, 5000);
        assert!(config.enable_profiling);
    }

    #[test]
    fn test_gpu_calibration_creation_succeeds_on_cpu() {
        let config = GpuCalibrationConfig::default();

        // Construction succeeds even without a GPU: it falls back to CPU.
        let isotonic = GpuIsotonicCalibrator::new(config.clone());
        assert!(isotonic.is_ok());

        let temp = GpuTemperatureScalingCalibrator::new(config);
        assert!(temp.is_ok());
    }

    #[test]
    fn test_explicit_missing_device_is_honest_error() {
        // Requesting a device that does not exist must fail loudly, not silently
        // fall back to a fabricated device.
        let config = GpuCalibrationConfig {
            device_id: Some(3),
            ..Default::default()
        };
        let result = GpuTemperatureScalingCalibrator::new(config);
        assert!(result.is_err());
        let msg = format!("{:?}", result.expect_err("result must be an error"));
        assert!(msg.contains("not found"));
    }

    #[test]
    fn test_threshold_and_gpu_dispatch_logic() {
        let config = GpuCalibrationConfig {
            gpu_threshold: 1000,
            ..Default::default()
        };

        let cpu_calibrator = Box::new(IsotonicCalibrator::new()) as Box<dyn CalibrationEstimator>;
        let gpu_calibrator = GpuCalibratedClassifier::new(cpu_calibrator, config)
            .expect("CPU fallback construction must succeed");

        // Pure threshold check is independent of device availability.
        assert!(!gpu_calibrator.exceeds_gpu_threshold(500));
        assert!(gpu_calibrator.exceeds_gpu_threshold(1500));

        // Dispatch requires BOTH a usable device and a size over the threshold.
        // A batch under the threshold never dispatches, regardless of hardware.
        assert!(!gpu_calibrator.should_use_gpu(500));
        // A batch over the threshold dispatches exactly when a device is
        // actually present -- so the answer must track real detection rather
        // than a hard-coded no-GPU assumption.
        assert_eq!(
            gpu_calibrator.should_use_gpu(1500),
            gpu_calibrator.gpu_available()
        );
    }

    #[test]
    fn test_gpu_wrapper_matches_plain_cpu_calibrator() {
        // The wrapper must produce EXACTLY the wrapped calibrator's output —
        // proving it delegates real calibration rather than fabricating values.
        let probabilities = array![0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.55];
        let y = array![0, 0, 1, 1, 1, 0, 1, 0];

        // Use the trait methods explicitly: the wrapper delegates through the
        // `CalibrationEstimator` trait, so the reference must too (the inherent
        // consuming `fit` would otherwise be selected).
        let mut reference = IsotonicCalibrator::new();
        CalibrationEstimator::fit(&mut reference, &probabilities, &y).expect("reference fit");
        let reference_out = CalibrationEstimator::predict_proba(&reference, &probabilities)
            .expect("reference predict");

        let config = GpuCalibrationConfig {
            gpu_threshold: 1, // would force GPU if any device existed
            ..Default::default()
        };
        let mut wrapper =
            GpuIsotonicCalibrator::new(config).expect("wrapper construction must succeed");
        wrapper.fit(&probabilities, &y).expect("wrapper fit");
        let wrapper_out = wrapper
            .predict_proba(&probabilities)
            .expect("wrapper predict");

        assert_eq!(reference_out.len(), wrapper_out.len());
        for (r, w) in reference_out.iter().zip(wrapper_out.iter()) {
            assert!(
                (r - w).abs() < 1e-12,
                "wrapper diverged from CPU calibrator: {r} vs {w}"
            );
        }
    }

    /// Same cross-check as above, for temperature scaling: the GPU wrapper
    /// (which, on this GPU-less test machine, always takes the CPU path
    /// regardless of the `gpu` feature) must match the plain CPU calibrator
    /// exactly.
    #[test]
    fn test_gpu_temperature_wrapper_matches_plain_cpu_calibrator() {
        let probabilities = array![0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.55];
        let y = array![0, 0, 1, 1, 1, 0, 1, 0];

        let mut reference = TemperatureScalingCalibrator::new();
        CalibrationEstimator::fit(&mut reference, &probabilities, &y).expect("reference fit");
        let reference_out = CalibrationEstimator::predict_proba(&reference, &probabilities)
            .expect("reference predict");

        let config = GpuCalibrationConfig {
            gpu_threshold: 1, // would force GPU if any device existed
            ..Default::default()
        };
        let mut wrapper = GpuTemperatureScalingCalibrator::new(config)
            .expect("wrapper construction must succeed");
        wrapper.fit(&probabilities, &y).expect("wrapper fit");
        let wrapper_out = wrapper
            .predict_proba(&probabilities)
            .expect("wrapper predict");

        assert_eq!(reference_out.len(), wrapper_out.len());
        for (r, w) in reference_out.iter().zip(wrapper_out.iter()) {
            assert!(
                (r - w).abs() < 1e-9,
                "wrapper diverged from CPU calibrator: {r} vs {w}"
            );
        }
    }

    /// With mixed precision enabled the temperature wrapper takes the genuine
    /// f32 device path when a GPU is present (and falls back to the exact CPU
    /// path when it is not). Either way the calibrated probabilities must track
    /// the plain CPU calibrator to within f32 accuracy.
    #[test]
    fn test_gpu_temperature_wrapper_mixed_precision_tracks_cpu() {
        let probabilities = array![0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.55];
        let y = array![0, 0, 1, 1, 1, 0, 1, 0];

        let mut reference = TemperatureScalingCalibrator::new();
        CalibrationEstimator::fit(&mut reference, &probabilities, &y).expect("reference fit");
        let reference_out = CalibrationEstimator::predict_proba(&reference, &probabilities)
            .expect("reference predict");

        let config = GpuCalibrationConfig {
            gpu_threshold: 1,
            use_mixed_precision: true,
            ..Default::default()
        };
        let mut wrapper = GpuTemperatureScalingCalibrator::new(config)
            .expect("wrapper construction must succeed");
        wrapper.fit(&probabilities, &y).expect("wrapper fit");
        let wrapper_out = wrapper
            .predict_proba(&probabilities)
            .expect("wrapper predict");

        assert_eq!(reference_out.len(), wrapper_out.len());
        for (r, w) in reference_out.iter().zip(wrapper_out.iter()) {
            assert!(
                (r - w).abs() < 1e-4,
                "mixed-precision wrapper diverged from CPU calibrator: {r} vs {w}"
            );
        }
    }

    #[test]
    fn test_builder_pattern() {
        let config = GpuCalibrationBuilder::new()
            .device_id(0)
            .batch_size(256)
            .gpu_threshold(2000)
            .enable_profiling()
            .use_mixed_precision()
            .memory_limit(2_147_483_648) // 2 GB
            .build();

        assert_eq!(config.device_id, Some(0));
        assert_eq!(config.batch_size, 256);
        assert_eq!(config.gpu_threshold, 2000);
        assert!(config.enable_profiling);
        assert!(config.use_mixed_precision);
        assert_eq!(config.memory_limit, 2_147_483_648);
    }

    #[test]
    fn test_default_configuration() {
        let config = GpuCalibrationConfig::default();

        assert_eq!(config.device_id, None);
        assert_eq!(config.batch_size, 1024);
        assert_eq!(config.gpu_threshold, 10000);
        assert!(!config.enable_profiling);
        assert!(!config.use_mixed_precision);
        assert_eq!(config.memory_limit, 1_073_741_824); // 1 GB
    }
}
