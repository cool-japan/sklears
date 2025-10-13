//! External SIMD library integration framework
//!
//! This module provides integration capabilities with external high-performance
//! SIMD libraries such as Intel MKL, OpenBLAS, BLIS, and others.

use crate::traits::SimdError;

#[cfg(feature = "no-std")]
use alloc::{
    collections::BTreeMap as HashMap,
    format,
    string::{String, ToString},
    sync::Arc,
    vec,
    vec::Vec,
};
#[cfg(not(feature = "no-std"))]
use std::collections::HashMap;
#[cfg(not(feature = "no-std"))]
use std::string::ToString;
#[cfg(not(feature = "no-std"))]
use std::sync::Arc;

#[cfg(feature = "no-std")]
use spin::Mutex;
#[cfg(not(feature = "no-std"))]
use std::sync::Mutex;

/// Result type for external library operations
pub type ExternalResult<T> = Result<T, SimdError>;

/// Trait for external library adapters
pub trait ExternalLibrary: Send + Sync {
    /// Get the library name
    fn name(&self) -> &str;

    /// Get the library version
    fn version(&self) -> &str;

    /// Check if the library is available on the system
    fn is_available(&self) -> bool;

    /// Initialize the library (if needed)
    fn initialize(&mut self) -> ExternalResult<()>;

    /// Get supported operations
    fn supported_operations(&self) -> Vec<String>;

    /// Check if a specific operation is supported
    fn supports_operation(&self, operation: &str) -> bool {
        self.supported_operations().contains(&operation.to_string())
    }
}

/// Trait for BLAS-like operations from external libraries
pub trait ExternalBlas: ExternalLibrary {
    /// Vector dot product (SDOT)
    fn dot(&self, x: &[f32], y: &[f32]) -> ExternalResult<f32>;

    /// Vector-scalar multiplication (SSCAL)
    fn scal(&self, alpha: f32, x: &mut [f32]) -> ExternalResult<()>;

    /// Vector addition (SAXPY): y = alpha * x + y
    fn axpy(&self, alpha: f32, x: &[f32], y: &mut [f32]) -> ExternalResult<()>;

    /// Matrix-vector multiplication (SGEMV)
    fn gemv(
        &self,
        alpha: f32,
        a: &[f32],
        m: usize,
        n: usize,
        x: &[f32],
        beta: f32,
        y: &mut [f32],
    ) -> ExternalResult<()>;

    /// Matrix-matrix multiplication (SGEMM)
    fn gemm(
        &self,
        alpha: f32,
        a: &[f32],
        m: usize,
        k: usize,
        b: &[f32],
        n: usize,
        beta: f32,
        c: &mut [f32],
    ) -> ExternalResult<()>;
}

/// Trait for LAPACK-like operations from external libraries
pub trait ExternalLapack: ExternalLibrary {
    /// LU decomposition
    fn lu_decomposition(&self, a: &mut [f32], m: usize, n: usize) -> ExternalResult<Vec<i32>>;

    /// QR decomposition
    fn qr_decomposition(&self, a: &mut [f32], m: usize, n: usize) -> ExternalResult<Vec<f32>>;

    /// Singular Value Decomposition
    fn svd(
        &self,
        a: &mut [f32],
        m: usize,
        n: usize,
    ) -> ExternalResult<(Vec<f32>, Vec<f32>, Vec<f32>)>;

    /// Eigenvalue decomposition
    fn eigenvalues(&self, a: &mut [f32], n: usize) -> ExternalResult<Vec<f32>>;
}

/// Trait for FFT operations from external libraries
pub trait ExternalFft: ExternalLibrary {
    /// Real-to-complex FFT
    fn rfft(&self, input: &[f32]) -> ExternalResult<Vec<f32>>;

    /// Complex-to-real inverse FFT
    fn irfft(&self, input: &[f32]) -> ExternalResult<Vec<f32>>;

    /// Complex-to-complex FFT
    fn cfft(&self, real: &[f32], imag: &[f32]) -> ExternalResult<(Vec<f32>, Vec<f32>)>;

    /// Complex-to-complex inverse FFT
    fn icfft(&self, real: &[f32], imag: &[f32]) -> ExternalResult<(Vec<f32>, Vec<f32>)>;
}

/// Mock implementation for Intel MKL adapter (demonstration)
#[derive(Debug, Clone)]
pub struct MklAdapter {
    initialized: bool,
}

impl MklAdapter {
    pub fn new() -> Self {
        Self { initialized: false }
    }
}

impl Default for MklAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl ExternalLibrary for MklAdapter {
    fn name(&self) -> &str {
        "Intel MKL"
    }

    fn version(&self) -> &str {
        "2024.0"
    }

    fn is_available(&self) -> bool {
        // In a real implementation, this would check for MKL libraries
        false // Mock: not available in test environment
    }

    fn initialize(&mut self) -> ExternalResult<()> {
        if !self.is_available() {
            return Err(SimdError::ExternalLibraryError(
                "MKL not available".to_string(),
            ));
        }
        self.initialized = true;
        Ok(())
    }

    fn supported_operations(&self) -> Vec<String> {
        vec![
            "dot".to_string(),
            "scal".to_string(),
            "axpy".to_string(),
            "gemv".to_string(),
            "gemm".to_string(),
            "lu".to_string(),
            "qr".to_string(),
            "svd".to_string(),
            "fft".to_string(),
        ]
    }
}

impl ExternalBlas for MklAdapter {
    fn dot(&self, x: &[f32], y: &[f32]) -> ExternalResult<f32> {
        if !self.initialized {
            return Err(SimdError::ExternalLibraryError(
                "MKL not initialized".to_string(),
            ));
        }

        if x.len() != y.len() {
            return Err(SimdError::InvalidInput(
                "Vector lengths must match".to_string(),
            ));
        }

        // Mock implementation - in reality would call cblas_sdot
        Ok(x.iter().zip(y.iter()).map(|(a, b)| a * b).sum())
    }

    fn scal(&self, alpha: f32, x: &mut [f32]) -> ExternalResult<()> {
        if !self.initialized {
            return Err(SimdError::ExternalLibraryError(
                "MKL not initialized".to_string(),
            ));
        }

        // Mock implementation - in reality would call cblas_sscal
        x.iter_mut().for_each(|v| *v *= alpha);
        Ok(())
    }

    fn axpy(&self, alpha: f32, x: &[f32], y: &mut [f32]) -> ExternalResult<()> {
        if !self.initialized {
            return Err(SimdError::ExternalLibraryError(
                "MKL not initialized".to_string(),
            ));
        }

        if x.len() != y.len() {
            return Err(SimdError::InvalidInput(
                "Vector lengths must match".to_string(),
            ));
        }

        // Mock implementation - in reality would call cblas_saxpy
        for (yi, &xi) in y.iter_mut().zip(x.iter()) {
            *yi += alpha * xi;
        }
        Ok(())
    }

    fn gemv(
        &self,
        alpha: f32,
        a: &[f32],
        m: usize,
        n: usize,
        x: &[f32],
        beta: f32,
        y: &mut [f32],
    ) -> ExternalResult<()> {
        if !self.initialized {
            return Err(SimdError::ExternalLibraryError(
                "MKL not initialized".to_string(),
            ));
        }

        if a.len() != m * n || x.len() != n || y.len() != m {
            return Err(SimdError::InvalidInput(
                "Matrix/vector dimension mismatch".to_string(),
            ));
        }

        // Mock implementation - in reality would call cblas_sgemv
        for i in 0..m {
            let mut sum = 0.0;
            for j in 0..n {
                sum += a[i * n + j] * x[j];
            }
            y[i] = alpha * sum + beta * y[i];
        }
        Ok(())
    }

    fn gemm(
        &self,
        alpha: f32,
        a: &[f32],
        m: usize,
        k: usize,
        b: &[f32],
        n: usize,
        beta: f32,
        c: &mut [f32],
    ) -> ExternalResult<()> {
        if !self.initialized {
            return Err(SimdError::ExternalLibraryError(
                "MKL not initialized".to_string(),
            ));
        }

        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(SimdError::InvalidInput(
                "Matrix dimension mismatch".to_string(),
            ));
        }

        // Mock implementation - in reality would call cblas_sgemm
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = alpha * sum + beta * c[i * n + j];
            }
        }
        Ok(())
    }
}

/// Mock implementation for OpenBLAS adapter
#[derive(Debug, Clone)]
pub struct OpenBlasAdapter {
    initialized: bool,
}

impl OpenBlasAdapter {
    pub fn new() -> Self {
        Self { initialized: false }
    }
}

impl Default for OpenBlasAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl ExternalLibrary for OpenBlasAdapter {
    fn name(&self) -> &str {
        "OpenBLAS"
    }

    fn version(&self) -> &str {
        "0.3.21"
    }

    fn is_available(&self) -> bool {
        // Mock: assume available for demonstration
        true
    }

    fn initialize(&mut self) -> ExternalResult<()> {
        self.initialized = true;
        Ok(())
    }

    fn supported_operations(&self) -> Vec<String> {
        vec![
            "dot".to_string(),
            "scal".to_string(),
            "axpy".to_string(),
            "gemv".to_string(),
            "gemm".to_string(),
        ]
    }
}

impl ExternalBlas for OpenBlasAdapter {
    fn dot(&self, x: &[f32], y: &[f32]) -> ExternalResult<f32> {
        if !self.initialized {
            return Err(SimdError::ExternalLibraryError(
                "OpenBLAS not initialized".to_string(),
            ));
        }

        if x.len() != y.len() {
            return Err(SimdError::InvalidInput(
                "Vector lengths must match".to_string(),
            ));
        }

        // Mock implementation using our internal SIMD dot product
        Ok(crate::vector::dot_product(x, y))
    }

    fn scal(&self, alpha: f32, x: &mut [f32]) -> ExternalResult<()> {
        if !self.initialized {
            return Err(SimdError::ExternalLibraryError(
                "OpenBLAS not initialized".to_string(),
            ));
        }

        // Mock implementation using our internal SIMD scale
        crate::vector::scale(x, alpha);
        Ok(())
    }

    fn axpy(&self, alpha: f32, x: &[f32], y: &mut [f32]) -> ExternalResult<()> {
        if !self.initialized {
            return Err(SimdError::ExternalLibraryError(
                "OpenBLAS not initialized".to_string(),
            ));
        }

        if x.len() != y.len() {
            return Err(SimdError::InvalidInput(
                "Vector lengths must match".to_string(),
            ));
        }

        // Mock implementation
        for (yi, &xi) in y.iter_mut().zip(x.iter()) {
            *yi += alpha * xi;
        }
        Ok(())
    }

    fn gemv(
        &self,
        alpha: f32,
        a: &[f32],
        m: usize,
        n: usize,
        x: &[f32],
        beta: f32,
        y: &mut [f32],
    ) -> ExternalResult<()> {
        if !self.initialized {
            return Err(SimdError::ExternalLibraryError(
                "OpenBLAS not initialized".to_string(),
            ));
        }

        if a.len() != m * n || x.len() != n || y.len() != m {
            return Err(SimdError::InvalidInput(
                "Matrix/vector dimension mismatch".to_string(),
            ));
        }

        // Mock implementation using our internal matrix operations
        use scirs2_autograd::ndarray::{Array1, Array2};
        let a_matrix = Array2::from_shape_vec((m, n), a.to_vec())
            .map_err(|_| SimdError::ExternalLibraryError("Invalid matrix shape".to_string()))?;
        let x_vector = Array1::from_vec(x.to_vec());
        let result = crate::matrix::matrix_vector_multiply_f32(&a_matrix, &x_vector);

        for (yi, &ri) in y.iter_mut().zip(result.iter()) {
            *yi = alpha * ri + beta * (*yi);
        }
        Ok(())
    }

    fn gemm(
        &self,
        alpha: f32,
        a: &[f32],
        m: usize,
        k: usize,
        b: &[f32],
        n: usize,
        beta: f32,
        c: &mut [f32],
    ) -> ExternalResult<()> {
        if !self.initialized {
            return Err(SimdError::ExternalLibraryError(
                "OpenBLAS not initialized".to_string(),
            ));
        }

        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(SimdError::InvalidInput(
                "Matrix dimension mismatch".to_string(),
            ));
        }

        // Mock implementation using our internal matrix operations
        use scirs2_autograd::ndarray::Array2;
        let a_matrix = Array2::from_shape_vec((m, k), a.to_vec())
            .map_err(|_| SimdError::ExternalLibraryError("Invalid matrix A shape".to_string()))?;
        let b_matrix = Array2::from_shape_vec((k, n), b.to_vec())
            .map_err(|_| SimdError::ExternalLibraryError("Invalid matrix B shape".to_string()))?;
        let result = crate::matrix::matrix_multiply_f32_simd(&a_matrix, &b_matrix);

        for (ci, &ri) in c.iter_mut().zip(result.iter()) {
            *ci = alpha * ri + beta * (*ci);
        }
        Ok(())
    }
}

/// External library registry and management
pub struct ExternalLibraryRegistry {
    /// Registered BLAS libraries
    blas_libraries: HashMap<String, Arc<Mutex<dyn ExternalBlas>>>,
    /// Registered LAPACK libraries
    lapack_libraries: HashMap<String, Arc<Mutex<dyn ExternalLapack>>>,
    /// Registered FFT libraries
    fft_libraries: HashMap<String, Arc<Mutex<dyn ExternalFft>>>,
    /// Preferred library for each operation type
    preferences: HashMap<String, String>,
}

impl ExternalLibraryRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            blas_libraries: HashMap::new(),
            lapack_libraries: HashMap::new(),
            fft_libraries: HashMap::new(),
            preferences: HashMap::new(),
        }
    }

    /// Register a BLAS library
    pub fn register_blas<T: ExternalBlas + 'static>(&mut self, library: T) {
        let name = library.name().to_string();
        self.blas_libraries
            .insert(name.clone(), Arc::new(Mutex::new(library)));

        // Set as default if none set
        if !self.preferences.contains_key("blas") {
            self.preferences.insert("blas".to_string(), name);
        }
    }

    /// Register a LAPACK library
    pub fn register_lapack<T: ExternalLapack + 'static>(&mut self, library: T) {
        let name = library.name().to_string();
        self.lapack_libraries
            .insert(name.clone(), Arc::new(Mutex::new(library)));

        // Set as default if none set
        if !self.preferences.contains_key("lapack") {
            self.preferences.insert("lapack".to_string(), name);
        }
    }

    /// Register an FFT library
    pub fn register_fft<T: ExternalFft + 'static>(&mut self, library: T) {
        let name = library.name().to_string();
        self.fft_libraries
            .insert(name.clone(), Arc::new(Mutex::new(library)));

        // Set as default if none set
        if !self.preferences.contains_key("fft") {
            self.preferences.insert("fft".to_string(), name);
        }
    }

    /// Set preferred library for operation type
    pub fn set_preference(
        &mut self,
        operation_type: &str,
        library_name: &str,
    ) -> ExternalResult<()> {
        match operation_type {
            "blas" => {
                if !self.blas_libraries.contains_key(library_name) {
                    return Err(SimdError::ExternalLibraryError(format!(
                        "BLAS library '{}' not registered",
                        library_name
                    )));
                }
            }
            "lapack" => {
                if !self.lapack_libraries.contains_key(library_name) {
                    return Err(SimdError::ExternalLibraryError(format!(
                        "LAPACK library '{}' not registered",
                        library_name
                    )));
                }
            }
            "fft" => {
                if !self.fft_libraries.contains_key(library_name) {
                    return Err(SimdError::ExternalLibraryError(format!(
                        "FFT library '{}' not registered",
                        library_name
                    )));
                }
            }
            _ => {
                return Err(SimdError::InvalidInput(format!(
                    "Unknown operation type: {}",
                    operation_type
                )));
            }
        }

        self.preferences
            .insert(operation_type.to_string(), library_name.to_string());
        Ok(())
    }

    /// Get preferred BLAS library
    pub fn get_blas(&self) -> Option<Arc<Mutex<dyn ExternalBlas>>> {
        self.preferences
            .get("blas")
            .and_then(|name| self.blas_libraries.get(name))
            .cloned()
    }

    /// Get preferred LAPACK library
    pub fn get_lapack(&self) -> Option<Arc<Mutex<dyn ExternalLapack>>> {
        self.preferences
            .get("lapack")
            .and_then(|name| self.lapack_libraries.get(name))
            .cloned()
    }

    /// Get preferred FFT library
    pub fn get_fft(&self) -> Option<Arc<Mutex<dyn ExternalFft>>> {
        self.preferences
            .get("fft")
            .and_then(|name| self.fft_libraries.get(name))
            .cloned()
    }

    /// List all registered libraries
    pub fn list_libraries(&self) -> Vec<String> {
        let mut libraries = Vec::new();
        libraries.extend(self.blas_libraries.keys().cloned());
        libraries.extend(self.lapack_libraries.keys().cloned());
        libraries.extend(self.fft_libraries.keys().cloned());
        libraries.sort();
        libraries.dedup();
        libraries
    }

    /// Check library availability
    pub fn check_availability(&self) -> HashMap<String, bool> {
        let mut availability = HashMap::new();

        #[cfg(not(feature = "no-std"))]
        {
            for (name, library) in &self.blas_libraries {
                availability.insert(name.clone(), library.lock().unwrap().is_available());
            }

            for (name, library) in &self.lapack_libraries {
                availability.insert(name.clone(), library.lock().unwrap().is_available());
            }

            for (name, library) in &self.fft_libraries {
                availability.insert(name.clone(), library.lock().unwrap().is_available());
            }
        }

        #[cfg(feature = "no-std")]
        {
            for (name, library) in &self.blas_libraries {
                availability.insert(name.clone(), library.lock().is_available());
            }

            for (name, library) in &self.lapack_libraries {
                availability.insert(name.clone(), library.lock().is_available());
            }

            for (name, library) in &self.fft_libraries {
                availability.insert(name.clone(), library.lock().is_available());
            }
        }

        availability
    }
}

impl Default for ExternalLibraryRegistry {
    fn default() -> Self {
        let mut registry = Self::new();

        // Register default adapters
        registry.register_blas(OpenBlasAdapter::new());
        registry.register_blas(MklAdapter::new());

        registry
    }
}

/// Global external library registry
#[cfg(not(feature = "no-std"))]
static EXTERNAL_REGISTRY: std::sync::LazyLock<std::sync::Mutex<ExternalLibraryRegistry>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(ExternalLibraryRegistry::default()));

#[cfg(feature = "no-std")]
static EXTERNAL_REGISTRY: spin::Once<spin::Mutex<ExternalLibraryRegistry>> = spin::Once::new();

/// Get the global external library registry
#[cfg(not(feature = "no-std"))]
pub fn get_registry() -> &'static std::sync::Mutex<ExternalLibraryRegistry> {
    &EXTERNAL_REGISTRY
}

#[cfg(feature = "no-std")]
pub fn get_registry() -> &'static spin::Mutex<ExternalLibraryRegistry> {
    EXTERNAL_REGISTRY.call_once(|| spin::Mutex::new(ExternalLibraryRegistry::default()))
}

/// Perform dot product using external BLAS if available, fallback to internal
pub fn external_dot(x: &[f32], y: &[f32]) -> ExternalResult<f32> {
    #[cfg(not(feature = "no-std"))]
    {
        if let Some(blas) = get_registry().lock().unwrap().get_blas() {
            // Try to use external BLAS, but fallback to internal if it fails
            match blas.lock().unwrap().dot(x, y) {
                Ok(result) => Ok(result),
                Err(_) => {
                    // Fall back to internal implementation if external library fails
                    Ok(crate::vector::dot_product(x, y))
                }
            }
        } else {
            Ok(crate::vector::dot_product(x, y))
        }
    }
    #[cfg(feature = "no-std")]
    {
        if let Some(blas) = get_registry().lock().get_blas() {
            // Try to use external BLAS, but fallback to internal if it fails
            match blas.lock().dot(x, y) {
                Ok(result) => Ok(result),
                Err(_) => {
                    // Fall back to internal implementation if external library fails
                    Ok(crate::vector::dot_product(x, y))
                }
            }
        } else {
            Ok(crate::vector::dot_product(x, y))
        }
    }
}

/// Perform matrix-vector multiplication using external BLAS if available
pub fn external_gemv(
    alpha: f32,
    a: &[f32],
    m: usize,
    n: usize,
    x: &[f32],
    beta: f32,
    y: &mut [f32],
) -> ExternalResult<()> {
    #[cfg(not(feature = "no-std"))]
    {
        if let Some(blas) = get_registry().lock().unwrap().get_blas() {
            // Try to use external BLAS, but fallback to internal if it fails
            match blas.lock().unwrap().gemv(alpha, a, m, n, x, beta, y) {
                Ok(()) => Ok(()),
                Err(_) => {
                    // Fall back to internal implementation if external library fails
                    use scirs2_autograd::ndarray::{Array1, Array2};
                    let a_matrix = Array2::from_shape_vec((m, n), a.to_vec()).map_err(|_| {
                        SimdError::ExternalLibraryError("Invalid matrix shape".to_string())
                    })?;
                    let x_vector = Array1::from_vec(x.to_vec());
                    let result = crate::matrix::matrix_vector_multiply_f32(&a_matrix, &x_vector);

                    for (yi, &ri) in y.iter_mut().zip(result.iter()) {
                        *yi = alpha * ri + beta * (*yi);
                    }
                    Ok(())
                }
            }
        } else {
            // Fallback to internal implementation
            use scirs2_autograd::ndarray::{Array1, Array2};
            let a_matrix = Array2::from_shape_vec((m, n), a.to_vec())
                .map_err(|_| SimdError::ExternalLibraryError("Invalid matrix shape".to_string()))?;
            let x_vector = Array1::from_vec(x.to_vec());
            let result = crate::matrix::matrix_vector_multiply_f32(&a_matrix, &x_vector);

            for (yi, &ri) in y.iter_mut().zip(result.iter()) {
                *yi = alpha * ri + beta * (*yi);
            }
            Ok(())
        }
    }
    #[cfg(feature = "no-std")]
    {
        if let Some(blas) = get_registry().lock().get_blas() {
            // Try to use external BLAS, but fallback to internal if it fails
            match blas.lock().gemv(alpha, a, m, n, x, beta, y) {
                Ok(()) => Ok(()),
                Err(_) => {
                    // Fall back to internal implementation if external library fails
                    use scirs2_autograd::ndarray::{Array1, Array2};
                    let a_matrix = Array2::from_shape_vec((m, n), a.to_vec()).map_err(|_| {
                        SimdError::ExternalLibraryError("Invalid matrix shape".to_string())
                    })?;
                    let x_vector = Array1::from_vec(x.to_vec());
                    let result = crate::matrix::matrix_vector_multiply_f32(&a_matrix, &x_vector);

                    for (yi, &ri) in y.iter_mut().zip(result.iter()) {
                        *yi = alpha * ri + beta * (*yi);
                    }
                    Ok(())
                }
            }
        } else {
            // Fallback to internal implementation
            use scirs2_autograd::ndarray::{Array1, Array2};
            let a_matrix = Array2::from_shape_vec((m, n), a.to_vec())
                .map_err(|_| SimdError::ExternalLibraryError("Invalid matrix shape".to_string()))?;
            let x_vector = Array1::from_vec(x.to_vec());
            let result = crate::matrix::matrix_vector_multiply_f32(&a_matrix, &x_vector);

            for (yi, &ri) in y.iter_mut().zip(result.iter()) {
                *yi = alpha * ri + beta * (*yi);
            }
            Ok(())
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mkl_adapter_creation() {
        let adapter = MklAdapter::new();
        assert_eq!(adapter.name(), "Intel MKL");
        assert_eq!(adapter.version(), "2024.0");
        assert!(!adapter.is_available()); // Mock: not available
    }

    #[test]
    fn test_openblas_adapter_creation() {
        let adapter = OpenBlasAdapter::new();
        assert_eq!(adapter.name(), "OpenBLAS");
        assert_eq!(adapter.version(), "0.3.21");
        assert!(adapter.is_available()); // Mock: available
    }

    #[test]
    fn test_openblas_initialization() {
        let mut adapter = OpenBlasAdapter::new();
        assert!(adapter.initialize().is_ok());
    }

    #[test]
    fn test_openblas_dot_product() {
        let mut adapter = OpenBlasAdapter::new();
        adapter.initialize().unwrap();

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        let result = adapter.dot(&x, &y).unwrap();

        assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6
    }

    #[test]
    fn test_openblas_scal() {
        let mut adapter = OpenBlasAdapter::new();
        adapter.initialize().unwrap();

        let mut x = vec![1.0, 2.0, 3.0];
        adapter.scal(2.0, &mut x).unwrap();

        assert_eq!(x, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_registry_blas_registration() {
        let mut registry = ExternalLibraryRegistry::new();
        let adapter = OpenBlasAdapter::new();

        registry.register_blas(adapter);

        assert!(registry.get_blas().is_some());
        assert_eq!(registry.list_libraries(), vec!["OpenBLAS"]);
    }

    #[test]
    fn test_registry_preferences() {
        let mut registry = ExternalLibraryRegistry::new();
        let adapter1 = OpenBlasAdapter::new();
        let adapter2 = MklAdapter::new();

        registry.register_blas(adapter1);
        registry.register_blas(adapter2);

        // Set preference to MKL
        registry.set_preference("blas", "Intel MKL").unwrap();

        // Should fail for unknown library
        assert!(registry.set_preference("blas", "Unknown").is_err());
    }

    #[test]
    fn test_registry_availability_check() {
        let registry = ExternalLibraryRegistry::default();
        let availability = registry.check_availability();

        // OpenBLAS should be available (mock), MKL should not
        assert_eq!(availability.get("OpenBLAS"), Some(&true));
        assert_eq!(availability.get("Intel MKL"), Some(&false));
    }

    #[test]
    fn test_external_dot_fallback() {
        // This should fallback to internal implementation
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        let result = external_dot(&x, &y).unwrap();

        assert_eq!(result, 32.0);
    }

    #[test]
    fn test_invalid_dimensions() {
        let mut adapter = OpenBlasAdapter::new();
        adapter.initialize().unwrap();

        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0, 5.0];

        assert!(adapter.dot(&x, &y).is_err());
    }

    #[test]
    fn test_uninitialized_adapter() {
        let adapter = OpenBlasAdapter::new();
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        assert!(adapter.dot(&x, &y).is_err());
    }
}
