//! Kernel functions for Gaussian Process models
//!
//! This module provides various kernel functions that can be used with
//! Gaussian Process models. All implementations comply with SciRS2 Policy.

// SciRS2 Policy - Use scirs2-autograd for ndarray types and array operations
use scirs2_core::ndarray::{Array2, ArrayView1};
// SciRS2 Policy - Use scirs2-core for random operations
use sklears_core::error::{Result as SklResult, SklearsError};

// Core kernel trait - import and re-export from the kernel_trait module
pub use crate::kernel_trait::Kernel;

/// RBF (Radial Basis Function) Kernel
#[derive(Debug, Clone)]
pub struct RBF {
    length_scale: f64,
}

impl RBF {
    pub fn new(length_scale: f64) -> Self {
        Self { length_scale }
    }
}

impl Kernel for RBF {
    fn compute_kernel_matrix(
        &self,
        X1: &Array2<f64>,
        X2: Option<&Array2<f64>>,
    ) -> SklResult<Array2<f64>> {
        let X2 = X2.unwrap_or(X1);
        let n1 = X1.nrows();
        let n2 = X2.nrows();
        let mut K = Array2::<f64>::zeros((n1, n2));

        for i in 0..n1 {
            for j in 0..n2 {
                let x1 = X1.row(i);
                let x2 = X2.row(j);
                K[[i, j]] = self.kernel(&x1, &x2);
            }
        }
        Ok(K)
    }

    fn kernel(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        let mut sq_dist = 0.0;
        for (a, b) in x1.iter().zip(x2.iter()) {
            sq_dist += (a - b).powi(2);
        }
        (-sq_dist / (2.0 * self.length_scale.powi(2))).exp()
    }

    fn get_params(&self) -> Vec<f64> {
        vec![self.length_scale]
    }

    fn set_params(&mut self, params: &[f64]) -> SklResult<()> {
        if params.len() != 1 {
            return Err(SklearsError::InvalidInput(
                "RBF kernel requires exactly 1 parameter".to_string(),
            ));
        }
        self.length_scale = params[0];
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }
}

/// ARD RBF (Automatic Relevance Determination RBF) Kernel
///
/// This kernel is similar to RBF but has a separate length scale for each dimension,
/// allowing it to automatically determine which dimensions are most relevant for
/// the prediction task.
///
/// The kernel is defined as:
/// k(x, x') = exp(-0.5 * sum((x_i - x'_i)^2 / length_scales\[i\]^2))
///
/// # Examples
///
/// ```ignore
/// use sklears_gaussian_process::kernels::ARDRBF;
/// use scirs2_core::ndarray::Array1;
///
/// // Create ARD RBF kernel with different length scales for each dimension
/// let length_scales = Array1::from_vec(vec![1.0, 2.0, 0.5]);
/// let kernel = ARDRBF::new(length_scales);
/// ```
#[derive(Debug, Clone)]
pub struct ARDRBF {
    /// Length scale for each dimension
    length_scales: scirs2_core::ndarray::Array1<f64>,
}

impl ARDRBF {
    /// Create a new ARD RBF kernel with specified length scales for each dimension
    ///
    /// # Arguments
    ///
    /// * `length_scales` - Array of length scales, one for each dimension
    pub fn new(length_scales: scirs2_core::ndarray::Array1<f64>) -> Self {
        Self { length_scales }
    }

    /// Create a new ARD RBF kernel with uniform length scales
    ///
    /// # Arguments
    ///
    /// * `n_dims` - Number of dimensions
    /// * `length_scale` - Initial length scale value for all dimensions
    pub fn new_uniform(n_dims: usize, length_scale: f64) -> Self {
        Self {
            length_scales: scirs2_core::ndarray::Array1::from_elem(n_dims, length_scale),
        }
    }

    /// Get the number of dimensions
    pub fn n_dimensions(&self) -> usize {
        self.length_scales.len()
    }
}

impl Kernel for ARDRBF {
    fn compute_kernel_matrix(
        &self,
        X1: &Array2<f64>,
        X2: Option<&Array2<f64>>,
    ) -> SklResult<Array2<f64>> {
        let X2 = X2.unwrap_or(X1);
        let n1 = X1.nrows();
        let n2 = X2.nrows();

        // Validate dimensions
        if X1.ncols() != self.length_scales.len() {
            return Err(SklearsError::InvalidInput(format!(
                "X1 has {} dimensions but kernel expects {}",
                X1.ncols(),
                self.length_scales.len()
            )));
        }
        if X2.ncols() != self.length_scales.len() {
            return Err(SklearsError::InvalidInput(format!(
                "X2 has {} dimensions but kernel expects {}",
                X2.ncols(),
                self.length_scales.len()
            )));
        }

        let mut K = Array2::<f64>::zeros((n1, n2));

        for i in 0..n1 {
            for j in 0..n2 {
                let x1 = X1.row(i);
                let x2 = X2.row(j);
                K[[i, j]] = self.kernel(&x1, &x2);
            }
        }
        Ok(K)
    }

    fn kernel(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        // Compute weighted squared distance: sum((x_i - x'_i)^2 / l_i^2)
        let mut weighted_sq_dist = 0.0;
        for ((a, b), length_scale) in x1.iter().zip(x2.iter()).zip(self.length_scales.iter()) {
            let diff = a - b;
            weighted_sq_dist += (diff * diff) / (length_scale * length_scale);
        }
        (-0.5 * weighted_sq_dist).exp()
    }

    fn get_params(&self) -> Vec<f64> {
        self.length_scales.to_vec()
    }

    fn set_params(&mut self, params: &[f64]) -> SklResult<()> {
        if params.len() != self.length_scales.len() {
            return Err(SklearsError::InvalidInput(format!(
                "ARD RBF kernel requires exactly {} parameters (one per dimension), got {}",
                self.length_scales.len(),
                params.len()
            )));
        }
        for (i, &param) in params.iter().enumerate() {
            self.length_scales[i] = param;
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }
}

/// Matérn Kernel
#[derive(Debug, Clone)]
pub struct Matern {
    length_scale: f64,
    nu: f64,
}

impl Matern {
    pub fn new(length_scale: f64, nu: f64) -> Self {
        Self { length_scale, nu }
    }
}

impl Kernel for Matern {
    fn compute_kernel_matrix(
        &self,
        X1: &Array2<f64>,
        X2: Option<&Array2<f64>>,
    ) -> SklResult<Array2<f64>> {
        let X2 = X2.unwrap_or(X1);
        let n1 = X1.nrows();
        let n2 = X2.nrows();
        let mut K = Array2::<f64>::zeros((n1, n2));

        for i in 0..n1 {
            for j in 0..n2 {
                let x1 = X1.row(i);
                let x2 = X2.row(j);
                K[[i, j]] = self.kernel(&x1, &x2);
            }
        }
        Ok(K)
    }

    fn kernel(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        let mut sq_dist = 0.0;
        for (a, b) in x1.iter().zip(x2.iter()) {
            sq_dist += (a - b).powi(2);
        }
        let dist = sq_dist.sqrt();

        if dist == 0.0 {
            return 1.0;
        }

        let sqrt_3_dist = (3.0_f64).sqrt() * dist / self.length_scale;
        (1.0 + sqrt_3_dist) * (-sqrt_3_dist).exp()
    }

    fn get_params(&self) -> Vec<f64> {
        vec![self.length_scale, self.nu]
    }

    fn set_params(&mut self, params: &[f64]) -> SklResult<()> {
        if params.len() != 2 {
            return Err(SklearsError::InvalidInput(
                "Matern kernel requires exactly 2 parameters".to_string(),
            ));
        }
        self.length_scale = params[0];
        self.nu = params[1];
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }
}

/// Linear Kernel
#[derive(Debug, Clone)]
pub struct Linear {
    sigma_0_sq: f64,
    sigma_1_sq: f64,
}

impl Linear {
    pub fn new(sigma_0_sq: f64, sigma_1_sq: f64) -> Self {
        Self {
            sigma_0_sq,
            sigma_1_sq,
        }
    }
}

impl Kernel for Linear {
    fn compute_kernel_matrix(
        &self,
        X1: &Array2<f64>,
        X2: Option<&Array2<f64>>,
    ) -> SklResult<Array2<f64>> {
        let X2 = X2.unwrap_or(X1);
        let n1 = X1.nrows();
        let n2 = X2.nrows();
        let mut K = Array2::<f64>::zeros((n1, n2));

        for i in 0..n1 {
            for j in 0..n2 {
                let x1 = X1.row(i);
                let x2 = X2.row(j);
                K[[i, j]] = self.kernel(&x1, &x2);
            }
        }
        Ok(K)
    }

    fn kernel(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        let dot_product: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum();
        self.sigma_0_sq + self.sigma_1_sq * dot_product
    }

    fn get_params(&self) -> Vec<f64> {
        vec![self.sigma_0_sq, self.sigma_1_sq]
    }

    fn set_params(&mut self, params: &[f64]) -> SklResult<()> {
        if params.len() != 2 {
            return Err(SklearsError::InvalidInput(
                "Linear kernel requires exactly 2 parameters".to_string(),
            ));
        }
        self.sigma_0_sq = params[0];
        self.sigma_1_sq = params[1];
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }
}

/// Polynomial Kernel
#[derive(Debug, Clone)]
pub struct Polynomial {
    gamma: f64,
    coef0: f64,
    degree: f64,
}

impl Polynomial {
    pub fn new(gamma: f64, coef0: f64, degree: f64) -> Self {
        Self {
            gamma,
            coef0,
            degree,
        }
    }
}

impl Kernel for Polynomial {
    fn compute_kernel_matrix(
        &self,
        X1: &Array2<f64>,
        X2: Option<&Array2<f64>>,
    ) -> SklResult<Array2<f64>> {
        let X2 = X2.unwrap_or(X1);
        let n1 = X1.nrows();
        let n2 = X2.nrows();
        let mut K = Array2::<f64>::zeros((n1, n2));

        for i in 0..n1 {
            for j in 0..n2 {
                let x1 = X1.row(i);
                let x2 = X2.row(j);
                K[[i, j]] = self.kernel(&x1, &x2);
            }
        }
        Ok(K)
    }

    fn kernel(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        let dot_product: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum();
        (self.gamma * dot_product + self.coef0).powf(self.degree)
    }

    fn get_params(&self) -> Vec<f64> {
        vec![self.gamma, self.coef0, self.degree]
    }

    fn set_params(&mut self, params: &[f64]) -> SklResult<()> {
        if params.len() != 3 {
            return Err(SklearsError::InvalidInput(
                "Polynomial kernel requires exactly 3 parameters".to_string(),
            ));
        }
        self.gamma = params[0];
        self.coef0 = params[1];
        self.degree = params[2];
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }
}

/// Rational Quadratic Kernel
#[derive(Debug, Clone)]
pub struct RationalQuadratic {
    length_scale: f64,
    alpha: f64,
}

impl RationalQuadratic {
    pub fn new(length_scale: f64, alpha: f64) -> Self {
        Self {
            length_scale,
            alpha,
        }
    }
}

impl Kernel for RationalQuadratic {
    fn compute_kernel_matrix(
        &self,
        X1: &Array2<f64>,
        X2: Option<&Array2<f64>>,
    ) -> SklResult<Array2<f64>> {
        let X2 = X2.unwrap_or(X1);
        let n1 = X1.nrows();
        let n2 = X2.nrows();
        let mut K = Array2::<f64>::zeros((n1, n2));

        for i in 0..n1 {
            for j in 0..n2 {
                let x1 = X1.row(i);
                let x2 = X2.row(j);
                K[[i, j]] = self.kernel(&x1, &x2);
            }
        }
        Ok(K)
    }

    fn kernel(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        let mut sq_dist = 0.0;
        for (a, b) in x1.iter().zip(x2.iter()) {
            sq_dist += (a - b).powi(2);
        }
        (1.0 + sq_dist / (2.0 * self.alpha * self.length_scale.powi(2))).powf(-self.alpha)
    }

    fn get_params(&self) -> Vec<f64> {
        vec![self.length_scale, self.alpha]
    }

    fn set_params(&mut self, params: &[f64]) -> SklResult<()> {
        if params.len() != 2 {
            return Err(SklearsError::InvalidInput(
                "RationalQuadratic kernel requires exactly 2 parameters".to_string(),
            ));
        }
        self.length_scale = params[0];
        self.alpha = params[1];
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }
}

/// Exp-Sine-Squared (Periodic) Kernel
#[derive(Debug, Clone)]
pub struct ExpSineSquared {
    length_scale: f64,
    periodicity: f64,
}

impl ExpSineSquared {
    pub fn new(length_scale: f64, periodicity: f64) -> Self {
        Self {
            length_scale,
            periodicity,
        }
    }
}

impl Kernel for ExpSineSquared {
    fn compute_kernel_matrix(
        &self,
        X1: &Array2<f64>,
        X2: Option<&Array2<f64>>,
    ) -> SklResult<Array2<f64>> {
        let X2 = X2.unwrap_or(X1);
        let n1 = X1.nrows();
        let n2 = X2.nrows();
        let mut K = Array2::<f64>::zeros((n1, n2));

        for i in 0..n1 {
            for j in 0..n2 {
                let x1 = X1.row(i);
                let x2 = X2.row(j);
                K[[i, j]] = self.kernel(&x1, &x2);
            }
        }
        Ok(K)
    }

    fn kernel(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        let dist = x1
            .iter()
            .zip(x2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let sin_term = (std::f64::consts::PI * dist / self.periodicity).sin();
        (-2.0 * sin_term.powi(2) / self.length_scale.powi(2)).exp()
    }

    fn get_params(&self) -> Vec<f64> {
        vec![self.length_scale, self.periodicity]
    }

    fn set_params(&mut self, params: &[f64]) -> SklResult<()> {
        if params.len() != 2 {
            return Err(SklearsError::InvalidInput(
                "ExpSineSquared kernel requires exactly 2 parameters".to_string(),
            ));
        }
        self.length_scale = params[0];
        self.periodicity = params[1];
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }
}

/// White (noise) Kernel
#[derive(Debug, Clone)]
pub struct WhiteKernel {
    noise_level: f64,
}

impl WhiteKernel {
    pub fn new(noise_level: f64) -> Self {
        Self { noise_level }
    }
}

impl Kernel for WhiteKernel {
    fn compute_kernel_matrix(
        &self,
        X1: &Array2<f64>,
        X2: Option<&Array2<f64>>,
    ) -> SklResult<Array2<f64>> {
        let n1 = X1.nrows();
        let n2 = X2.map_or(n1, |x| x.nrows());
        let mut K = Array2::<f64>::zeros((n1, n2));

        // White kernel is only non-zero on the diagonal when X1 == X2
        if X2.is_none() {
            for i in 0..n1 {
                K[[i, i]] = self.noise_level;
            }
        }
        Ok(K)
    }

    fn kernel(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        // Check if vectors are identical
        let identical = x1.iter().zip(x2.iter()).all(|(a, b)| (a - b).abs() < 1e-10);
        if identical {
            self.noise_level
        } else {
            0.0
        }
    }

    fn get_params(&self) -> Vec<f64> {
        vec![self.noise_level]
    }

    fn set_params(&mut self, params: &[f64]) -> SklResult<()> {
        if params.len() != 1 {
            return Err(SklearsError::InvalidInput(
                "WhiteKernel requires exactly 1 parameter".to_string(),
            ));
        }
        self.noise_level = params[0];
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }
}

/// Constant Kernel
#[derive(Debug, Clone)]
pub struct ConstantKernel {
    constant_value: f64,
}

impl ConstantKernel {
    pub fn new(constant_value: f64) -> Self {
        Self { constant_value }
    }
}

impl Kernel for ConstantKernel {
    fn compute_kernel_matrix(
        &self,
        X1: &Array2<f64>,
        X2: Option<&Array2<f64>>,
    ) -> SklResult<Array2<f64>> {
        let n1 = X1.nrows();
        let n2 = X2.map_or(n1, |x| x.nrows());
        Ok(Array2::<f64>::from_elem((n1, n2), self.constant_value))
    }

    fn kernel(&self, _x1: &ArrayView1<f64>, _x2: &ArrayView1<f64>) -> f64 {
        self.constant_value
    }

    fn get_params(&self) -> Vec<f64> {
        vec![self.constant_value]
    }

    fn set_params(&mut self, params: &[f64]) -> SklResult<()> {
        if params.len() != 1 {
            return Err(SklearsError::InvalidInput(
                "ConstantKernel requires exactly 1 parameter".to_string(),
            ));
        }
        self.constant_value = params[0];
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }
}

/// Sum kernel for composing kernels additively
#[derive(Debug, Clone)]
pub struct SumKernel {
    kernels: Vec<Box<dyn Kernel>>,
}

impl SumKernel {
    pub fn new(kernels: Vec<Box<dyn Kernel>>) -> Self {
        Self { kernels }
    }
}

impl Kernel for SumKernel {
    fn compute_kernel_matrix(
        &self,
        X1: &Array2<f64>,
        X2: Option<&Array2<f64>>,
    ) -> SklResult<Array2<f64>> {
        if self.kernels.is_empty() {
            return Err(SklearsError::InvalidInput(
                "SumKernel requires at least one kernel".to_string(),
            ));
        }

        let mut result = self.kernels[0].compute_kernel_matrix(X1, X2)?;
        for kernel in &self.kernels[1..] {
            let k_matrix = kernel.compute_kernel_matrix(X1, X2)?;
            result = result + k_matrix;
        }
        Ok(result)
    }

    fn kernel(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        self.kernels.iter().map(|k| k.kernel(x1, x2)).sum()
    }

    fn get_params(&self) -> Vec<f64> {
        self.kernels.iter().flat_map(|k| k.get_params()).collect()
    }

    fn set_params(&mut self, params: &[f64]) -> SklResult<()> {
        let mut offset = 0;
        for kernel in &mut self.kernels {
            let n_params = kernel.get_params().len();
            if offset + n_params > params.len() {
                return Err(SklearsError::InvalidInput(
                    "Not enough parameters for SumKernel".to_string(),
                ));
            }
            kernel.set_params(&params[offset..offset + n_params])?;
            offset += n_params;
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Kernel> {
        Box::new(Self {
            kernels: self.kernels.iter().map(|k| k.clone_box()).collect(),
        })
    }
}

/// Product kernel for composing kernels multiplicatively
#[derive(Debug, Clone)]
pub struct ProductKernel {
    kernels: Vec<Box<dyn Kernel>>,
}

impl ProductKernel {
    pub fn new(kernels: Vec<Box<dyn Kernel>>) -> Self {
        Self { kernels }
    }
}

impl Kernel for ProductKernel {
    fn compute_kernel_matrix(
        &self,
        X1: &Array2<f64>,
        X2: Option<&Array2<f64>>,
    ) -> SklResult<Array2<f64>> {
        if self.kernels.is_empty() {
            return Err(SklearsError::InvalidInput(
                "ProductKernel requires at least one kernel".to_string(),
            ));
        }

        let mut result = self.kernels[0].compute_kernel_matrix(X1, X2)?;
        for kernel in &self.kernels[1..] {
            let k_matrix = kernel.compute_kernel_matrix(X1, X2)?;
            result = result * k_matrix;
        }
        Ok(result)
    }

    fn kernel(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        self.kernels.iter().map(|k| k.kernel(x1, x2)).product()
    }

    fn get_params(&self) -> Vec<f64> {
        self.kernels.iter().flat_map(|k| k.get_params()).collect()
    }

    fn set_params(&mut self, params: &[f64]) -> SklResult<()> {
        let mut offset = 0;
        for kernel in &mut self.kernels {
            let n_params = kernel.get_params().len();
            if offset + n_params > params.len() {
                return Err(SklearsError::InvalidInput(
                    "Not enough parameters for ProductKernel".to_string(),
                ));
            }
            kernel.set_params(&params[offset..offset + n_params])?;
            offset += n_params;
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Kernel> {
        Box::new(Self {
            kernels: self.kernels.iter().map(|k| k.clone_box()).collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // SciRS2 Policy - Use scirs2-autograd for ndarray types
    use scirs2_core::ndarray::{array, Array1};

    #[test]
    fn test_ardrbf_creation() {
        let length_scales = Array1::from_vec(vec![1.0, 2.0, 0.5]);
        let kernel = ARDRBF::new(length_scales.clone());
        assert_eq!(kernel.n_dimensions(), 3);
        assert_eq!(kernel.get_params(), vec![1.0, 2.0, 0.5]);
    }

    #[test]
    fn test_ardrbf_creation_uniform() {
        let kernel = ARDRBF::new_uniform(4, 1.5);
        assert_eq!(kernel.n_dimensions(), 4);
        assert_eq!(kernel.get_params(), vec![1.5, 1.5, 1.5, 1.5]);
    }

    #[test]
    fn test_ardrbf_kernel_identical_points() {
        let length_scales = Array1::from_vec(vec![1.0, 1.0]);
        let kernel = ARDRBF::new(length_scales);

        let x1 = array![1.0, 2.0];
        let x2 = array![1.0, 2.0];

        let k = kernel.kernel(&x1.view(), &x2.view());
        assert!(
            (k - 1.0).abs() < 1e-10,
            "Kernel of identical points should be 1.0"
        );
    }

    #[test]
    fn test_ardrbf_kernel_different_points() {
        let length_scales = Array1::from_vec(vec![1.0, 1.0]);
        let kernel = ARDRBF::new(length_scales);

        let x1 = array![0.0, 0.0];
        let x2 = array![1.0, 1.0];

        let k = kernel.kernel(&x1.view(), &x2.view());
        // Expected: exp(-0.5 * ((1.0^2 / 1.0^2) + (1.0^2 / 1.0^2))) = exp(-1.0) ≈ 0.368
        assert!(k > 0.0 && k < 1.0, "Kernel should be between 0 and 1");
        assert!((k - (-1.0f64).exp()).abs() < 1e-10, "Kernel value mismatch");
    }

    #[test]
    fn test_ardrbf_kernel_matrix() {
        let length_scales = Array1::from_vec(vec![1.0, 1.0]);
        let kernel = ARDRBF::new(length_scales);

        let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let k_matrix = kernel.compute_kernel_matrix(&x, None).unwrap();

        assert_eq!(k_matrix.dim(), (3, 3));
        // Diagonal should be 1.0
        assert!((k_matrix[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((k_matrix[[1, 1]] - 1.0).abs() < 1e-10);
        assert!((k_matrix[[2, 2]] - 1.0).abs() < 1e-10);
        // Matrix should be symmetric
        assert!((k_matrix[[0, 1]] - k_matrix[[1, 0]]).abs() < 1e-10);
        assert!((k_matrix[[0, 2]] - k_matrix[[2, 0]]).abs() < 1e-10);
        assert!((k_matrix[[1, 2]] - k_matrix[[2, 1]]).abs() < 1e-10);
    }

    #[test]
    fn test_ardrbf_relevance_determination() {
        // Test that different length scales weight dimensions differently
        let length_scales = Array1::from_vec(vec![0.1, 10.0]); // First dimension very relevant, second not
        let kernel = ARDRBF::new(length_scales);

        let x1 = array![0.0, 0.0];
        let x2_dim1 = array![1.0, 0.0]; // Change in first dimension
        let x2_dim2 = array![0.0, 1.0]; // Change in second dimension

        let k1 = kernel.kernel(&x1.view(), &x2_dim1.view());
        let k2 = kernel.kernel(&x1.view(), &x2_dim2.view());

        // Change in dimension 1 should have much more effect (smaller kernel value)
        assert!(
            k1 < k2,
            "Dimension with smaller length scale should have more effect"
        );
    }

    #[test]
    fn test_ardrbf_set_params() {
        let length_scales = Array1::from_vec(vec![1.0, 1.0]);
        let mut kernel = ARDRBF::new(length_scales);

        let new_params = vec![2.0, 3.0];
        kernel.set_params(&new_params).unwrap();

        assert_eq!(kernel.get_params(), vec![2.0, 3.0]);
    }

    #[test]
    fn test_ardrbf_set_params_wrong_size() {
        let length_scales = Array1::from_vec(vec![1.0, 1.0]);
        let mut kernel = ARDRBF::new(length_scales);

        let wrong_params = vec![1.0, 2.0, 3.0]; // Too many parameters
        let result = kernel.set_params(&wrong_params);

        assert!(
            result.is_err(),
            "Should error with wrong number of parameters"
        );
    }

    #[test]
    fn test_ardrbf_dimension_validation() {
        let length_scales = Array1::from_vec(vec![1.0, 1.0]);
        let kernel = ARDRBF::new(length_scales);

        let x_wrong_dim = array![[0.0, 0.0, 0.0]]; // 3 dimensions instead of 2
        let result = kernel.compute_kernel_matrix(&x_wrong_dim, None);

        assert!(
            result.is_err(),
            "Should error with wrong number of dimensions"
        );
    }

    #[test]
    fn test_ardrbf_clone() {
        let length_scales = Array1::from_vec(vec![1.0, 2.0]);
        let kernel = ARDRBF::new(length_scales);
        let cloned = kernel.clone();

        assert_eq!(kernel.get_params(), cloned.get_params());
        assert_eq!(kernel.n_dimensions(), cloned.n_dimensions());
    }

    #[test]
    fn test_ardrbf_vs_rbf_isotropic() {
        // When all length scales are equal, ARD RBF should behave like regular RBF
        let length_scale = 1.5;
        let ard_kernel = ARDRBF::new_uniform(2, length_scale);
        let rbf_kernel = RBF::new(length_scale);

        let x1 = array![1.0, 2.0];
        let x2 = array![3.0, 4.0];

        let k_ard = ard_kernel.kernel(&x1.view(), &x2.view());
        let k_rbf = rbf_kernel.kernel(&x1.view(), &x2.view());

        assert!(
            (k_ard - k_rbf).abs() < 1e-10,
            "ARD RBF with uniform length scales should match RBF"
        );
    }
}
