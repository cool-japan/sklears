//! Basic Kernel Ridge Regression Implementation
//!
//! This module contains the core KernelRidgeRegression implementation including:
//! - Basic kernel ridge regression with multiple approximation methods
//! - Support for different solvers (Direct, SVD, Conjugate Gradient)
//! - Online/Incremental learning capabilities
//! - Comprehensive numerical linear algebra implementations

use crate::{
    FastfoodTransform, Nystroem, RBFSampler, StructuredRandomFeatures, Trained, Untrained,
};
use scirs2_core::ndarray::ndarray_linalg::solve::Solve;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{thread_rng, Rng};
use sklears_core::error::{Result, SklearsError};
use sklears_core::prelude::{Estimator, Fit, Float, Predict};
use std::marker::PhantomData;

use super::core_types::*;

/// Basic Kernel Ridge Regression
///
/// This is the main kernel ridge regression implementation that supports various
/// kernel approximation methods and solvers for different use cases.
#[derive(Debug, Clone)]
pub struct KernelRidgeRegression<State = Untrained> {
    pub approximation_method: ApproximationMethod,
    pub alpha: Float,
    pub solver: Solver,
    pub random_state: Option<u64>,

    // Fitted parameters
    pub(crate) weights_: Option<Array1<Float>>,
    pub(crate) feature_transformer_: Option<FeatureTransformer>,

    pub(crate) _state: PhantomData<State>,
}

impl KernelRidgeRegression<Untrained> {
    /// Create a new kernel ridge regression model
    pub fn new(approximation_method: ApproximationMethod) -> Self {
        Self {
            approximation_method,
            alpha: 1.0,
            solver: Solver::Direct,
            random_state: None,
            weights_: None,
            feature_transformer_: None,
            _state: PhantomData,
        }
    }

    /// Set the regularization parameter
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the solver method
    pub fn solver(mut self, solver: Solver) -> Self {
        self.solver = solver;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl Estimator for KernelRidgeRegression<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for KernelRidgeRegression<Untrained> {
    type Fitted = KernelRidgeRegression<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let (n_samples, _) = x.dim();

        if y.len() != n_samples {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        // Fit the feature transformer based on approximation method
        let feature_transformer = self.fit_feature_transformer(x)?;

        // Transform features
        let x_transformed = feature_transformer.transform(x)?;

        // Solve the ridge regression problem: (X^T X + alpha * I) w = X^T y
        let weights = self.solve_ridge_regression(&x_transformed, y)?;

        Ok(KernelRidgeRegression {
            approximation_method: self.approximation_method,
            alpha: self.alpha,
            solver: self.solver,
            random_state: self.random_state,
            weights_: Some(weights),
            feature_transformer_: Some(feature_transformer),
            _state: PhantomData,
        })
    }
}

impl KernelRidgeRegression<Untrained> {
    /// Fit the feature transformer based on the approximation method
    fn fit_feature_transformer(&self, x: &Array2<Float>) -> Result<FeatureTransformer> {
        match &self.approximation_method {
            ApproximationMethod::Nystroem {
                kernel,
                n_components,
                sampling_strategy,
            } => {
                let mut nystroem = Nystroem::new(kernel.clone(), *n_components)
                    .sampling_strategy(sampling_strategy.clone());

                if let Some(seed) = self.random_state {
                    nystroem = nystroem.random_state(seed);
                }

                let fitted = nystroem.fit(x, &())?;
                Ok(FeatureTransformer::Nystroem(fitted))
            }
            ApproximationMethod::RandomFourierFeatures {
                n_components,
                gamma,
            } => {
                let mut rbf_sampler = RBFSampler::new(*n_components).gamma(*gamma);

                if let Some(seed) = self.random_state {
                    rbf_sampler = rbf_sampler.random_state(seed);
                }

                let fitted = rbf_sampler.fit(x, &())?;
                Ok(FeatureTransformer::RBFSampler(fitted))
            }
            ApproximationMethod::StructuredRandomFeatures {
                n_components,
                gamma,
            } => {
                let mut structured_rff = StructuredRandomFeatures::new(*n_components).gamma(*gamma);

                if let Some(seed) = self.random_state {
                    structured_rff = structured_rff.random_state(seed);
                }

                let fitted = structured_rff.fit(x, &())?;
                Ok(FeatureTransformer::StructuredRFF(fitted))
            }
            ApproximationMethod::Fastfood {
                n_components,
                gamma,
            } => {
                let mut fastfood = FastfoodTransform::new(*n_components).gamma(*gamma);

                if let Some(seed) = self.random_state {
                    fastfood = fastfood.random_state(seed);
                }

                let fitted = fastfood.fit(x, &())?;
                Ok(FeatureTransformer::Fastfood(fitted))
            }
        }
    }

    /// Solve the ridge regression problem
    fn solve_ridge_regression(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let (n_samples, n_features) = x.dim();

        match &self.solver {
            Solver::Direct => self.solve_direct(x, y),
            Solver::SVD => self.solve_svd(x, y),
            Solver::ConjugateGradient { max_iter, tol } => {
                self.solve_conjugate_gradient(x, y, *max_iter, *tol)
            }
        }
    }

    /// Direct solver using normal equations
    fn solve_direct(&self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Array1<Float>> {
        let (_, n_features) = x.dim();

        // SIMD-accelerated computation of X^T X + alpha * I using optimized operations
        let x_f64 = Array2::from_shape_fn(x.dim(), |(i, j)| x[[i, j]] as f64);
        let y_f64 = Array1::from_vec(y.iter().map(|&val| val as f64).collect());

        // #[cfg(feature = "nightly-simd")]
        // let (gram_matrix, xty_f64, weights_f64) = {
        //     let xtx_f64 = simd_kernel::simd_gram_matrix_from_data(&x_f64.view());
        //     let mut gram_matrix = xtx_f64;
        //
        //     // SIMD-accelerated diagonal regularization
        //     for i in 0..n_features {
        //         gram_matrix[[i, i]] += self.alpha as f64;
        //     }
        //
        //     // SIMD-accelerated computation of X^T y
        //     let xty_f64 =
        //         simd_kernel::simd_matrix_vector_multiply(&x_f64.t().view(), &y_f64.view())?;
        //
        //     // SIMD-accelerated linear system solving
        //     let weights_f64 =
        //         simd_kernel::simd_ridge_coefficients(&gram_matrix.view(), &xty_f64.view(), 0.0)?;
        //
        //     (gram_matrix, xty_f64, weights_f64)
        // };

        // Use standard implementation (SIMD temporarily disabled)
        let weights_f64 = {
            // Standard implementation without SIMD
            let gram_matrix = x_f64.t().dot(&x_f64);
            let mut regularized_gram = gram_matrix;
            for i in 0..n_features {
                regularized_gram[[i, i]] += self.alpha as f64;
            }
            let xty_f64 = x_f64.t().dot(&y_f64);

            // Solve the linear system (X^T X + αI) w = X^T y
            regularized_gram
                .solve(&xty_f64)
                .map_err(|e| SklearsError::InvalidParameter {
                    name: "regularization".to_string(),
                    reason: format!("Linear system solving failed: {:?}", e),
                })?
        };

        // Convert back to Float
        let weights = Array1::from_vec(weights_f64.iter().map(|&val| val as Float).collect());
        Ok(weights)
    }

    /// SVD-based solver (more numerically stable)
    fn solve_svd(&self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Array1<Float>> {
        // Use SVD of the design matrix X for numerical stability
        // Solve the regularized least squares: min ||Xw - y||² + α||w||²

        let (n_samples, n_features) = x.dim();

        // Compute SVD of X using power iteration method
        let (u, s, vt) = self.compute_svd(x)?;

        // Compute regularized pseudo-inverse
        let threshold = 1e-12;
        let mut s_reg_inv = Array1::zeros(s.len());
        for i in 0..s.len() {
            if s[i] > threshold {
                s_reg_inv[i] = s[i] / (s[i] * s[i] + self.alpha);
            }
        }

        // Solve: w = V * S_reg^(-1) * U^T * y
        let ut_y = u.t().dot(y);
        let mut temp = Array1::zeros(s.len());
        for i in 0..s.len() {
            temp[i] = s_reg_inv[i] * ut_y[i];
        }

        let weights = vt.t().dot(&temp);
        Ok(weights)
    }

    /// Conjugate Gradient solver (iterative, memory efficient)
    fn solve_conjugate_gradient(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        max_iter: usize,
        tol: Float,
    ) -> Result<Array1<Float>> {
        let (n_samples, n_features) = x.dim();

        // Initialize weights
        let mut w = Array1::zeros(n_features);

        // Compute initial residual: r = X^T y - (X^T X + alpha I) w
        let xty = x.t().dot(y);
        let mut r = xty.clone();

        let mut p = r.clone();
        let mut rsold = r.dot(&r);

        for _iter in 0..max_iter {
            // Compute A * p where A = X^T X + alpha I
            let xtxp = x.t().dot(&x.dot(&p));
            let mut ap = xtxp;
            for i in 0..n_features {
                ap[i] += self.alpha * p[i];
            }

            // Compute step size
            let alpha_cg = rsold / p.dot(&ap);

            // Update weights
            w = w + alpha_cg * &p;

            // Update residual
            r = r - alpha_cg * &ap;

            let rsnew = r.dot(&r);

            // Check convergence
            if rsnew.sqrt() < tol {
                break;
            }

            // Update search direction
            let beta = rsnew / rsold;
            p = &r + beta * &p;
            rsold = rsnew;
        }

        Ok(w)
    }

    /// Solve linear system Ax = b using Gaussian elimination with partial pivoting
    fn solve_linear_system(&self, a: &Array2<Float>, b: &Array1<Float>) -> Result<Array1<Float>> {
        let n = a.nrows();
        if n != a.ncols() || n != b.len() {
            return Err(SklearsError::InvalidInput(
                "Matrix dimensions must match for linear system solve".to_string(),
            ));
        }

        // Create augmented matrix [A|b]
        let mut aug = Array2::zeros((n, n + 1));
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = a[[i, j]];
            }
            aug[[i, n]] = b[i];
        }

        // Forward elimination with partial pivoting
        for k in 0..n {
            // Find pivot
            let mut max_row = k;
            for i in (k + 1)..n {
                if aug[[i, k]].abs() > aug[[max_row, k]].abs() {
                    max_row = i;
                }
            }

            // Swap rows if needed
            if max_row != k {
                for j in 0..=n {
                    let temp = aug[[k, j]];
                    aug[[k, j]] = aug[[max_row, j]];
                    aug[[max_row, j]] = temp;
                }
            }

            // Check for zero pivot
            if aug[[k, k]].abs() < 1e-12 {
                return Err(SklearsError::InvalidInput(
                    "Matrix is singular or nearly singular".to_string(),
                ));
            }

            // Eliminate column
            for i in (k + 1)..n {
                let factor = aug[[i, k]] / aug[[k, k]];
                for j in k..=n {
                    aug[[i, j]] -= factor * aug[[k, j]];
                }
            }
        }

        // Back substitution
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = aug[[i, n]];
            for j in (i + 1)..n {
                sum -= aug[[i, j]] * x[j];
            }
            x[i] = sum / aug[[i, i]];
        }

        Ok(x)
    }

    /// Compute SVD using power iteration method
    /// Returns (U, S, V^T) where X = U * S * V^T
    fn compute_svd(
        &self,
        x: &Array2<Float>,
    ) -> Result<(Array2<Float>, Array1<Float>, Array2<Float>)> {
        let (m, n) = x.dim();
        let min_dim = m.min(n);

        // For SVD, we compute eigendecomposition of X^T X (for V) and X X^T (for U)
        let xt = x.t();

        if n <= m {
            // Thin SVD: compute V from X^T X
            let xtx = xt.dot(x);
            let (eigenvals_v, eigenvecs_v) = self.compute_eigendecomposition_svd(&xtx)?;

            // Singular values are sqrt of eigenvalues
            let mut singular_vals = Array1::zeros(min_dim);
            let mut valid_indices = Vec::new();
            for i in 0..eigenvals_v.len() {
                if eigenvals_v[i] > 1e-12 {
                    singular_vals[valid_indices.len()] = eigenvals_v[i].sqrt();
                    valid_indices.push(i);
                    if valid_indices.len() >= min_dim {
                        break;
                    }
                }
            }

            // Construct V matrix
            let mut v = Array2::zeros((n, min_dim));
            for (new_idx, &old_idx) in valid_indices.iter().enumerate() {
                v.column_mut(new_idx).assign(&eigenvecs_v.column(old_idx));
            }

            // Compute U = X * V * S^(-1)
            let mut u = Array2::zeros((m, min_dim));
            for j in 0..valid_indices.len() {
                let v_col = v.column(j);
                let xv = x.dot(&v_col);
                let u_col = &xv / singular_vals[j];
                u.column_mut(j).assign(&u_col);
            }

            Ok((u, singular_vals, v.t().to_owned()))
        } else {
            // Wide matrix: compute U from X X^T
            let xxt = x.dot(&xt);
            let (eigenvals_u, eigenvecs_u) = self.compute_eigendecomposition_svd(&xxt)?;

            // Singular values are sqrt of eigenvalues
            let mut singular_vals = Array1::zeros(min_dim);
            let mut valid_indices = Vec::new();
            for i in 0..eigenvals_u.len() {
                if eigenvals_u[i] > 1e-12 {
                    singular_vals[valid_indices.len()] = eigenvals_u[i].sqrt();
                    valid_indices.push(i);
                    if valid_indices.len() >= min_dim {
                        break;
                    }
                }
            }

            // Construct U matrix
            let mut u = Array2::zeros((m, min_dim));
            for (new_idx, &old_idx) in valid_indices.iter().enumerate() {
                u.column_mut(new_idx).assign(&eigenvecs_u.column(old_idx));
            }

            // Compute V = X^T * U * S^(-1)
            let mut v = Array2::zeros((n, min_dim));
            for j in 0..valid_indices.len() {
                let u_col = u.column(j);
                let xtu = xt.dot(&u_col);
                let v_col = &xtu / singular_vals[j];
                v.column_mut(j).assign(&v_col);
            }

            Ok((u, singular_vals, v.t().to_owned()))
        }
    }

    /// Compute eigendecomposition for SVD computation
    fn compute_eigendecomposition_svd(
        &self,
        matrix: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        let n = matrix.nrows();

        if n != matrix.ncols() {
            return Err(SklearsError::InvalidInput(
                "Matrix must be square for eigendecomposition".to_string(),
            ));
        }

        let mut eigenvals = Array1::zeros(n);
        let mut eigenvecs = Array2::zeros((n, n));

        // Use deflation method to find multiple eigenvalues
        let mut deflated_matrix = matrix.clone();

        for k in 0..n {
            // Power iteration for k-th eigenvalue/eigenvector
            let (eigenval, eigenvec) = self.power_iteration_svd(&deflated_matrix, 100, 1e-8)?;

            eigenvals[k] = eigenval;
            eigenvecs.column_mut(k).assign(&eigenvec);

            // Deflate matrix: A_new = A - λ * v * v^T
            for i in 0..n {
                for j in 0..n {
                    deflated_matrix[[i, j]] -= eigenval * eigenvec[i] * eigenvec[j];
                }
            }
        }

        // Sort eigenvalues and eigenvectors in descending order
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| eigenvals[j].partial_cmp(&eigenvals[i]).unwrap());

        let mut sorted_eigenvals = Array1::zeros(n);
        let mut sorted_eigenvecs = Array2::zeros((n, n));

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            sorted_eigenvals[new_idx] = eigenvals[old_idx];
            sorted_eigenvecs
                .column_mut(new_idx)
                .assign(&eigenvecs.column(old_idx));
        }

        Ok((sorted_eigenvals, sorted_eigenvecs))
    }

    /// Power iteration method for SVD eigendecomposition
    fn power_iteration_svd(
        &self,
        matrix: &Array2<Float>,
        max_iter: usize,
        tol: Float,
    ) -> Result<(Float, Array1<Float>)> {
        let n = matrix.nrows();

        // Initialize random vector
        let mut v = Array1::from_shape_fn(n, |_| thread_rng().gen::<Float>() - 0.5);

        // Normalize
        let norm = v.dot(&v).sqrt();
        if norm < 1e-10 {
            return Err(SklearsError::InvalidInput(
                "Initial vector has zero norm".to_string(),
            ));
        }
        v /= norm;

        let mut eigenval = 0.0;

        for _iter in 0..max_iter {
            // Apply matrix
            let w = matrix.dot(&v);

            // Compute Rayleigh quotient
            let new_eigenval = v.dot(&w);

            // Normalize
            let w_norm = w.dot(&w).sqrt();
            if w_norm < 1e-10 {
                break;
            }
            let new_v = w / w_norm;

            // Check convergence
            let eigenval_change = (new_eigenval - eigenval).abs();
            let vector_change = (&new_v - &v).mapv(|x| x.abs()).sum();

            if eigenval_change < tol && vector_change < tol {
                return Ok((new_eigenval, new_v));
            }

            eigenval = new_eigenval;
            v = new_v;
        }

        Ok((eigenval, v))
    }
}

impl Predict<Array2<Float>, Array1<Float>> for KernelRidgeRegression<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let weights = self
            .weights_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let feature_transformer =
            self.feature_transformer_
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict".to_string(),
                })?;

        // Transform features
        let x_transformed = feature_transformer.transform(x)?;

        // SIMD-accelerated prediction computation using optimized matrix-vector multiplication
        let x_f64 =
            Array2::from_shape_fn(x_transformed.dim(), |(i, j)| x_transformed[[i, j]] as f64);
        let weights_f64 = Array1::from_vec(weights.iter().map(|&val| val as f64).collect());

        // #[cfg(feature = "nightly-simd")]
        // let predictions_f64 =
        //     simd_kernel::simd_matrix_vector_multiply(&x_f64.view(), &weights_f64.view())?;

        // Use standard implementation (SIMD temporarily disabled)
        let predictions_f64 = x_f64.dot(&weights_f64);

        // Convert back to Float
        let predictions =
            Array1::from_vec(predictions_f64.iter().map(|&val| val as Float).collect());

        Ok(predictions)
    }
}

/// Online/Incremental Kernel Ridge Regression
///
/// This variant allows for online updates to the model as new data arrives.
#[derive(Debug, Clone)]
pub struct OnlineKernelRidgeRegression<State = Untrained> {
    /// Base kernel ridge regression
    pub base_model: KernelRidgeRegression<State>,
    /// Forgetting factor for online updates
    pub forgetting_factor: Float,
    /// Update frequency
    pub update_frequency: usize,

    // Online state
    update_count_: usize,
    accumulated_data_: Option<(Array2<Float>, Array1<Float>)>,

    _state: PhantomData<State>,
}

impl OnlineKernelRidgeRegression<Untrained> {
    /// Create a new online kernel ridge regression model
    pub fn new(approximation_method: ApproximationMethod) -> Self {
        Self {
            base_model: KernelRidgeRegression::new(approximation_method),
            forgetting_factor: 0.99,
            update_frequency: 100,
            update_count_: 0,
            accumulated_data_: None,
            _state: PhantomData,
        }
    }

    /// Set forgetting factor
    pub fn forgetting_factor(mut self, factor: Float) -> Self {
        self.forgetting_factor = factor;
        self
    }

    /// Set update frequency
    pub fn update_frequency(mut self, frequency: usize) -> Self {
        self.update_frequency = frequency;
        self
    }

    /// Set alpha parameter
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.base_model = self.base_model.alpha(alpha);
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.base_model = self.base_model.random_state(seed);
        self
    }
}

impl Estimator for OnlineKernelRidgeRegression<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for OnlineKernelRidgeRegression<Untrained> {
    type Fitted = OnlineKernelRidgeRegression<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let fitted_base = self.base_model.fit(x, y)?;

        Ok(OnlineKernelRidgeRegression {
            base_model: fitted_base,
            forgetting_factor: self.forgetting_factor,
            update_frequency: self.update_frequency,
            update_count_: 0,
            accumulated_data_: None,
            _state: PhantomData,
        })
    }
}

impl OnlineKernelRidgeRegression<Trained> {
    /// Update the model with new data
    pub fn partial_fit(mut self, x_new: &Array2<Float>, y_new: &Array1<Float>) -> Result<Self> {
        // Accumulate new data
        match &self.accumulated_data_ {
            Some((x_acc, y_acc)) => {
                let x_combined =
                    scirs2_core::ndarray::concatenate![Axis(0), x_acc.clone(), x_new.clone()];
                let y_combined =
                    scirs2_core::ndarray::concatenate![Axis(0), y_acc.clone(), y_new.clone()];
                self.accumulated_data_ = Some((x_combined, y_combined));
            }
            None => {
                self.accumulated_data_ = Some((x_new.clone(), y_new.clone()));
            }
        }

        self.update_count_ += 1;

        // Check if it's time to update
        if self.update_count_ % self.update_frequency == 0 {
            if let Some((ref x_acc, ref y_acc)) = self.accumulated_data_ {
                // Refit the model with accumulated data
                // In practice, you might want to implement a more sophisticated
                // online update algorithm here
                let updated_base = self.base_model.clone().into_untrained().fit(x_acc, y_acc)?;
                self.base_model = updated_base;
                self.accumulated_data_ = None;
            }
        }

        Ok(self)
    }

    /// Get the number of updates performed
    pub fn update_count(&self) -> usize {
        self.update_count_
    }
}

impl Predict<Array2<Float>, Array1<Float>> for OnlineKernelRidgeRegression<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        self.base_model.predict(x)
    }
}

// Helper trait to convert trained model to untrained
pub trait IntoUntrained<T> {
    fn into_untrained(self) -> T;
}

impl IntoUntrained<KernelRidgeRegression<Untrained>> for KernelRidgeRegression<Trained> {
    fn into_untrained(self) -> KernelRidgeRegression<Untrained> {
        KernelRidgeRegression {
            approximation_method: self.approximation_method,
            alpha: self.alpha,
            solver: self.solver,
            random_state: self.random_state,
            weights_: None,
            feature_transformer_: None,
            _state: PhantomData,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_kernel_ridge_regression_rff() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![1.0, 4.0, 9.0, 16.0];

        let approximation = ApproximationMethod::RandomFourierFeatures {
            n_components: 50,
            gamma: 0.1,
        };

        let krr = KernelRidgeRegression::new(approximation).alpha(0.1);
        let fitted = krr.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        // Check that predictions are reasonable
        for pred in predictions.iter() {
            assert!(pred.is_finite());
        }
    }

    #[test]
    fn test_kernel_ridge_regression_nystroem() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![1.0, 2.0, 3.0];

        let approximation = ApproximationMethod::Nystroem {
            kernel: Kernel::Rbf { gamma: 1.0 },
            n_components: 3,
            sampling_strategy: SamplingStrategy::Random,
        };

        let krr = KernelRidgeRegression::new(approximation).alpha(1.0);
        let fitted = krr.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 3);
    }

    #[test]
    fn test_kernel_ridge_regression_fastfood() {
        let x = array![[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]];
        let y = array![1.0, 2.0];

        let approximation = ApproximationMethod::Fastfood {
            n_components: 8,
            gamma: 0.5,
        };

        let krr = KernelRidgeRegression::new(approximation).alpha(0.1);
        let fitted = krr.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 2);
    }

    #[test]
    fn test_different_solvers() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![1.0, 2.0, 3.0];

        let approximation = ApproximationMethod::RandomFourierFeatures {
            n_components: 10,
            gamma: 1.0,
        };

        // Test Direct solver
        let krr_direct = KernelRidgeRegression::new(approximation.clone())
            .solver(Solver::Direct)
            .alpha(0.1);
        let fitted_direct = krr_direct.fit(&x, &y).unwrap();
        let pred_direct = fitted_direct.predict(&x).unwrap();

        // Test SVD solver
        let krr_svd = KernelRidgeRegression::new(approximation.clone())
            .solver(Solver::SVD)
            .alpha(0.1);
        let fitted_svd = krr_svd.fit(&x, &y).unwrap();
        let pred_svd = fitted_svd.predict(&x).unwrap();

        // Test CG solver
        let krr_cg = KernelRidgeRegression::new(approximation)
            .solver(Solver::ConjugateGradient {
                max_iter: 100,
                tol: 1e-6,
            })
            .alpha(0.1);
        let fitted_cg = krr_cg.fit(&x, &y).unwrap();
        let pred_cg = fitted_cg.predict(&x).unwrap();

        assert_eq!(pred_direct.len(), 3);
        assert_eq!(pred_svd.len(), 3);
        assert_eq!(pred_cg.len(), 3);
    }

    #[test]
    fn test_online_kernel_ridge_regression() {
        let x_initial = array![[1.0, 2.0], [2.0, 3.0]];
        let y_initial = array![1.0, 2.0];
        let x_new = array![[3.0, 4.0], [4.0, 5.0]];
        let y_new = array![3.0, 4.0];

        let approximation = ApproximationMethod::RandomFourierFeatures {
            n_components: 20,
            gamma: 0.5,
        };

        let online_krr = OnlineKernelRidgeRegression::new(approximation)
            .alpha(0.1)
            .update_frequency(2);

        let fitted = online_krr.fit(&x_initial, &y_initial).unwrap();
        let updated = fitted.partial_fit(&x_new, &y_new).unwrap();

        assert_eq!(updated.update_count(), 1);

        let predictions = updated.predict(&x_initial).unwrap();
        assert_eq!(predictions.len(), 2);
    }

    #[test]
    fn test_reproducibility() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![1.0, 2.0, 3.0];

        let approximation = ApproximationMethod::RandomFourierFeatures {
            n_components: 10,
            gamma: 1.0,
        };

        let krr1 = KernelRidgeRegression::new(approximation.clone())
            .alpha(0.1)
            .random_state(42);
        let fitted1 = krr1.fit(&x, &y).unwrap();
        let pred1 = fitted1.predict(&x).unwrap();

        let krr2 = KernelRidgeRegression::new(approximation)
            .alpha(0.1)
            .random_state(42);
        let fitted2 = krr2.fit(&x, &y).unwrap();
        let pred2 = fitted2.predict(&x).unwrap();

        assert_eq!(pred1.len(), pred2.len());
        for i in 0..pred1.len() {
            assert!((pred1[i] - pred2[i]).abs() < 1e-10);
        }
    }
}
