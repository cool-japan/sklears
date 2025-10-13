//! Generalized Canonical Correlation Analysis (GCCA)
//!
//! Generalized CCA extends canonical correlation analysis to handle multiple datasets
//! simultaneously using generalized eigenvalue decomposition for more robust
//! optimization and theoretical guarantees.

use numrs2::prelude::*;
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Transform},
    types::Float,
};
use std::marker::PhantomData;

/// Generalized Canonical Correlation Analysis
///
/// GCCA extends traditional CCA to handle multiple datasets by formulating the problem
/// as a generalized eigenvalue decomposition, providing theoretical guarantees and
/// improved numerical stability compared to iterative multi-view approaches.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::array;
/// use sklears_cross_decomposition::GeneralizedCCA;
/// use sklears_core::traits::Fit;
///
/// let view1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let view2 = array![[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]];
/// let view3 = array![[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]];
/// let views = vec![view1, view2, view3];
///
/// let gcca = GeneralizedCCA::new(1).regularization(0.1);
/// let fitted = gcca.fit(&views, &()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct GeneralizedCCA<State = Untrained> {
    /// Number of canonical components
    pub n_components: usize,
    /// Scale input data
    pub scale: bool,
    /// Regularization parameter
    pub regularization: Float,
    /// Deflation method: "canonical" or "regression"
    pub deflation_method: String,
    /// Copy input data
    pub copy: bool,
    /// Algorithm for eigenvalue decomposition: "auto", "full", or "randomized"
    pub algorithm: String,
    /// Tolerance for eigenvalue decomposition
    pub tol: Float,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Canonical weights for each view
    weights_: Option<Vec<Array2<Float>>>,
    /// Canonical loadings for each view
    loadings_: Option<Vec<Array2<Float>>>,
    /// Canonical correlations
    canonical_correlations_: Option<Array1<Float>>,
    /// Mean of each view
    means_: Option<Vec<Array1<Float>>>,
    /// Standard deviation of each view
    stds_: Option<Vec<Array1<Float>>>,
    /// Explained variance for each component
    explained_variance_: Option<Array1<Float>>,
    /// Explained variance ratio for each component
    explained_variance_ratio_: Option<Array1<Float>>,
    /// Eigenvalues from generalized eigenvalue decomposition
    eigenvalues_: Option<Array1<Float>>,
    /// State marker
    _state: PhantomData<State>,
}

/// Marker type for untrained state
#[derive(Debug, Clone)]
pub struct Untrained;

/// Marker type for trained state
#[derive(Debug, Clone)]
pub struct Trained;

impl GeneralizedCCA<Untrained> {
    /// Create a new Generalized CCA with specified number of components
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            scale: true,
            regularization: 0.0,
            deflation_method: "canonical".to_string(),
            copy: true,
            algorithm: "auto".to_string(),
            tol: 1e-6,
            random_state: None,
            weights_: None,
            loadings_: None,
            canonical_correlations_: None,
            means_: None,
            stds_: None,
            explained_variance_: None,
            explained_variance_ratio_: None,
            eigenvalues_: None,
            _state: PhantomData,
        }
    }

    /// Set whether to scale the data
    pub fn scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }

    /// Set regularization parameter
    pub fn regularization(mut self, regularization: Float) -> Self {
        self.regularization = regularization;
        self
    }

    /// Set deflation method
    pub fn deflation_method(mut self, method: &str) -> Self {
        self.deflation_method = method.to_string();
        self
    }

    /// Set algorithm for eigenvalue decomposition
    pub fn algorithm(mut self, algorithm: &str) -> Self {
        self.algorithm = algorithm.to_string();
        self
    }

    /// Set tolerance for eigenvalue decomposition
    pub fn tol(mut self, tol: Float) -> Self {
        self.tol = tol;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set whether to copy input data
    pub fn copy(mut self, copy: bool) -> Self {
        self.copy = copy;
        self
    }
}

impl Estimator for GeneralizedCCA<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Vec<Array2<Float>>, ()> for GeneralizedCCA<Untrained> {
    type Fitted = GeneralizedCCA<Trained>;

    fn fit(self, views: &Vec<Array2<Float>>, _target: &()) -> Result<Self::Fitted> {
        self.fit_views(views)
    }
}

impl GeneralizedCCA<Untrained> {
    /// Fit the model to multiple views
    pub fn fit_views(self, views: &Vec<Array2<Float>>) -> Result<GeneralizedCCA<Trained>> {
        if views.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "At least 2 views are required for generalized CCA".to_string(),
            ));
        }

        let n_samples = views[0].nrows();

        // Validate that all views have the same number of samples
        for (i, view) in views.iter().enumerate() {
            if view.nrows() != n_samples {
                return Err(SklearsError::InvalidInput(
                    format!("All views must have the same number of samples. View {} has {} samples, expected {}", 
                            i, view.nrows(), n_samples)
                ));
            }
        }

        // Validate n_components
        let min_features = views.iter().map(|v| v.ncols()).min().unwrap();
        let max_components = (n_samples - 1).min(min_features);
        if self.n_components > max_components {
            return Err(SklearsError::InvalidInput(format!(
                "n_components ({}) cannot exceed min(n_samples-1, min_features) = {}",
                self.n_components, max_components
            )));
        }

        // Center and scale the data
        let mut scaled_views = Vec::new();
        let mut means = Vec::new();
        let mut stds = Vec::new();

        for view in views {
            let view_mean = view.mean_axis(Axis(0)).unwrap();
            let mut centered_view = view - &view_mean.view().insert_axis(Axis(0));

            let view_std = if self.scale {
                let std = centered_view.var_axis(Axis(0), 1.0).mapv(|v| v.sqrt());
                for (i, &s) in std.iter().enumerate() {
                    if s > self.tol {
                        centered_view.column_mut(i).mapv_inplace(|v| v / s);
                    }
                }
                std
            } else {
                Array1::ones(view.ncols())
            };

            scaled_views.push(centered_view);
            means.push(view_mean);
            stds.push(view_std);
        }

        // Construct the generalized CCA problem
        // We solve the generalized eigenvalue problem: K * v = lambda * H * v
        // where K is the between-view covariance and H is the within-view covariance

        let total_features: usize = scaled_views.iter().map(|v| v.ncols()).sum();
        let mut K = Array2::zeros((total_features, total_features));
        let mut H = Array2::zeros((total_features, total_features));

        // Construct block matrices
        let mut feature_offsets = vec![0];
        for view in &scaled_views {
            feature_offsets.push(feature_offsets.last().unwrap() + view.ncols());
        }

        // Between-view covariance matrix K
        for i in 0..scaled_views.len() {
            for j in 0..scaled_views.len() {
                if i != j {
                    let cov_ij =
                        scaled_views[i].t().dot(&scaled_views[j]) / (n_samples as Float - 1.0);
                    let i_start = feature_offsets[i];
                    let i_end = feature_offsets[i + 1];
                    let j_start = feature_offsets[j];
                    let j_end = feature_offsets[j + 1];

                    K.slice_mut(s![i_start..i_end, j_start..j_end])
                        .assign(&cov_ij);
                }
            }
        }

        // Within-view covariance matrix H
        for i in 0..scaled_views.len() {
            let cov_ii = scaled_views[i].t().dot(&scaled_views[i]) / (n_samples as Float - 1.0);
            let i_start = feature_offsets[i];
            let i_end = feature_offsets[i + 1];

            // Add regularization to diagonal
            let mut regularized_cov = cov_ii;
            for k in 0..regularized_cov.nrows() {
                regularized_cov[[k, k]] += self.regularization;
            }

            H.slice_mut(s![i_start..i_end, i_start..i_end])
                .assign(&regularized_cov);
        }

        // Solve generalized eigenvalue problem: K * v = lambda * H * v
        // This is equivalent to: H^(-1) * K * v = lambda * v
        let (eigenvalues, eigenvectors) = self.solve_generalized_eigenvalue(&K, &H)?;

        // Sort eigenvalues and eigenvectors in descending order
        let mut eigen_pairs: Vec<(Float, Array1<Float>)> = eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &val)| (val, eigenvectors.column(i).to_owned()))
            .collect();

        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Take the top n_components
        let selected_eigenvalues: Array1<Float> = eigen_pairs
            .iter()
            .take(self.n_components)
            .map(|(val, _)| *val)
            .collect();

        let selected_eigenvectors: Array2<Float> = {
            let mut mat = Array2::zeros((total_features, self.n_components));
            for (i, (_, vec)) in eigen_pairs.iter().take(self.n_components).enumerate() {
                mat.column_mut(i).assign(vec);
            }
            mat
        };

        // Extract weights for each view
        let mut weights = Vec::new();
        for i in 0..scaled_views.len() {
            let i_start = feature_offsets[i];
            let i_end = feature_offsets[i + 1];
            let view_weights = selected_eigenvectors
                .slice(s![i_start..i_end, ..])
                .to_owned();
            weights.push(view_weights);
        }

        // Compute loadings and other statistics
        let mut loadings = Vec::new();
        let mut explained_var = Array1::zeros(self.n_components);
        let mut total_var = 0.0;

        for view in views {
            total_var += view.var_axis(Axis(0), 1.0).sum();
        }

        for (view_idx, view) in scaled_views.iter().enumerate() {
            let view_loadings =
                view.t().dot(&view.dot(&weights[view_idx])) / (n_samples as Float - 1.0);
            loadings.push(view_loadings);

            // Compute explained variance for this view
            for comp in 0..self.n_components {
                let scores = view.dot(&weights[view_idx].column(comp));
                explained_var[comp] += scores.var(1.0);
            }
        }

        let explained_var_ratio = &explained_var / total_var;

        // Canonical correlations are related to eigenvalues but need proper normalization
        // For generalized CCA, correlations should be clipped to [0, 1]
        let canonical_correlations = selected_eigenvalues.mapv(|x| {
            let correlation = x.sqrt().abs();
            correlation.min(1.0).max(0.0)
        });

        Ok(GeneralizedCCA {
            n_components: self.n_components,
            scale: self.scale,
            regularization: self.regularization,
            deflation_method: self.deflation_method,
            copy: self.copy,
            algorithm: self.algorithm,
            tol: self.tol,
            random_state: self.random_state,
            weights_: Some(weights),
            loadings_: Some(loadings),
            canonical_correlations_: Some(canonical_correlations),
            means_: Some(means),
            stds_: Some(stds),
            explained_variance_: Some(explained_var),
            explained_variance_ratio_: Some(explained_var_ratio),
            eigenvalues_: Some(selected_eigenvalues),
            _state: PhantomData,
        })
    }

    /// Solve the generalized eigenvalue problem K*v = lambda*H*v
    fn solve_generalized_eigenvalue(
        &self,
        K: &Array2<Float>,
        H: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        // For numerical stability, we first decompose H = L*L^T using Cholesky
        // Then solve the regular eigenvalue problem: L^(-1)*K*L^(-T)*u = lambda*u
        // And recover v = L^(-T)*u

        let mut H_reg = H.clone();

        // Add small regularization to diagonal for numerical stability
        for i in 0..H_reg.nrows() {
            H_reg[[i, i]] += 1e-12;
        }

        // Compute Cholesky decomposition of H
        let L = self.cholesky_decomposition(&H_reg)?;

        // Solve L * L_inv = I to get L^(-1)
        let L_inv = self.invert_lower_triangular(&L)?;

        // Compute the regular eigenvalue problem matrix: L^(-1) * K * L^(-T)
        let K_transformed = L_inv.dot(K).dot(&L_inv.t());

        // Solve regular eigenvalue problem
        let (eigenvalues, eigenvectors) = self.eigenvalue_decomposition(&K_transformed)?;

        // Transform eigenvectors back: v = L^(-T) * u
        let original_eigenvectors = L_inv.t().dot(&eigenvectors);

        Ok((eigenvalues, original_eigenvectors))
    }

    /// Cholesky decomposition: A = L * L^T
    fn cholesky_decomposition(&self, A: &Array2<Float>) -> Result<Array2<Float>> {
        let n = A.nrows();
        let mut L = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    let sum: Float = (0..j).map(|k| L[[i, k]] * L[[i, k]]).sum();
                    let val = A[[i, i]] - sum;
                    if val <= 0.0 {
                        return Err(SklearsError::InvalidInput(
                            "Matrix is not positive definite for Cholesky decomposition"
                                .to_string(),
                        ));
                    }
                    L[[i, j]] = val.sqrt();
                } else {
                    let sum: Float = (0..j).map(|k| L[[i, k]] * L[[j, k]]).sum();
                    L[[i, j]] = (A[[i, j]] - sum) / L[[j, j]];
                }
            }
        }

        Ok(L)
    }

    /// Invert a lower triangular matrix
    fn invert_lower_triangular(&self, L: &Array2<Float>) -> Result<Array2<Float>> {
        let n = L.nrows();
        let mut L_inv = Array2::zeros((n, n));

        for i in 0..n {
            L_inv[[i, i]] = 1.0 / L[[i, i]];
            for j in 0..i {
                let sum: Float = (j..i).map(|k| L[[i, k]] * L_inv[[k, j]]).sum();
                L_inv[[i, j]] = -sum / L[[i, i]];
            }
        }

        Ok(L_inv)
    }

    /// Simple eigenvalue decomposition for symmetric matrices
    fn eigenvalue_decomposition(
        &self,
        A: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        // For now, we use a simple power iteration method for the largest eigenvalues
        // In a production system, this should use LAPACK eigenvalue routines

        let n = A.nrows();
        let mut eigenvalues = Array1::zeros(self.n_components.min(n));
        let mut eigenvectors = Array2::zeros((n, self.n_components.min(n)));

        let mut A_deflated = A.clone();

        for comp in 0..self.n_components.min(n) {
            // Power iteration to find dominant eigenvalue/eigenvector
            let mut v: Array1<Float> = Array1::ones(n);
            v /= (v.dot(&v)).sqrt();

            for _ in 0..1000 {
                // max iterations
                let v_new = A_deflated.dot(&v);
                let norm = (v_new.dot(&v_new)).sqrt();
                if norm < self.tol {
                    break;
                }
                v = v_new / norm;
            }

            let eigenvalue = v.dot(&A_deflated.dot(&v));
            eigenvalues[comp] = eigenvalue;
            eigenvectors.column_mut(comp).assign(&v);

            // Deflate the matrix for next eigenvalue
            let vv_t = v
                .view()
                .insert_axis(Axis(1))
                .dot(&v.view().insert_axis(Axis(0)));
            A_deflated = A_deflated - eigenvalue * vv_t;
        }

        Ok((eigenvalues, eigenvectors))
    }
}

impl Transform<Vec<Array2<Float>>, Vec<Array2<Float>>> for GeneralizedCCA<Trained> {
    /// Transform views to canonical space
    fn transform(&self, views: &Vec<Array2<Float>>) -> Result<Vec<Array2<Float>>> {
        let weights = self.weights_.as_ref().unwrap();
        let means = self.means_.as_ref().unwrap();
        let stds = self.stds_.as_ref().unwrap();

        if views.len() != weights.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} views, got {}",
                weights.len(),
                views.len()
            )));
        }

        let mut transformed_views = Vec::new();

        for (view_idx, view) in views.iter().enumerate() {
            // Center and scale
            let mut scaled_view = view - &means[view_idx].view().insert_axis(Axis(0));

            if self.scale {
                for (i, &std) in stds[view_idx].iter().enumerate() {
                    if std > self.tol {
                        scaled_view.column_mut(i).mapv_inplace(|v| v / std);
                    }
                }
            }

            // Transform to canonical space
            let transformed = scaled_view.dot(&weights[view_idx]);
            transformed_views.push(transformed);
        }

        Ok(transformed_views)
    }
}

impl GeneralizedCCA<Trained> {
    /// Get the canonical weights for each view
    pub fn weights(&self) -> &Vec<Array2<Float>> {
        self.weights_.as_ref().unwrap()
    }

    /// Get the canonical loadings for each view
    pub fn loadings(&self) -> &Vec<Array2<Float>> {
        self.loadings_.as_ref().unwrap()
    }

    /// Get the canonical correlations
    pub fn canonical_correlations(&self) -> &Array1<Float> {
        self.canonical_correlations_.as_ref().unwrap()
    }

    /// Get the eigenvalues from generalized eigenvalue decomposition
    pub fn eigenvalues(&self) -> &Array1<Float> {
        self.eigenvalues_.as_ref().unwrap()
    }

    /// Get the explained variance for each component
    pub fn explained_variance(&self) -> &Array1<Float> {
        self.explained_variance_.as_ref().unwrap()
    }

    /// Get the explained variance ratio for each component
    pub fn explained_variance_ratio(&self) -> &Array1<Float> {
        self.explained_variance_ratio_.as_ref().unwrap()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use sklears_core::traits::Fit;

    #[test]
    fn test_generalized_cca_basic() {
        let view1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let view2 = array![[2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]];
        let view3 = array![[1.5, 2.5], [3.5, 4.5], [5.5, 6.5], [7.5, 8.5]];
        let views = vec![view1.clone(), view2, view3];

        let gcca = GeneralizedCCA::new(1);
        let fitted = gcca.fit_views(&views).unwrap();

        // Check that weights were computed
        assert_eq!(fitted.weights().len(), 3);
        assert_eq!(fitted.weights()[0].shape(), &[2, 1]);
        assert_eq!(fitted.weights()[1].shape(), &[2, 1]);
        assert_eq!(fitted.weights()[2].shape(), &[2, 1]);

        // Check that canonical correlations were computed
        assert_eq!(fitted.canonical_correlations().len(), 1);
        assert!(fitted.canonical_correlations()[0] >= 0.0);
        assert!(fitted.canonical_correlations()[0] <= 1.0);
    }

    #[test]
    fn test_generalized_cca_transform() {
        let view1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let view2 = array![[2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]];
        let views = vec![view1.clone(), view2.clone()];

        let gcca = GeneralizedCCA::new(1);
        let fitted = gcca.fit_views(&views).unwrap();
        let transformed = fitted.transform(&views).unwrap();

        assert_eq!(transformed.len(), 2);
        assert_eq!(transformed[0].shape(), &[4, 1]);
        assert_eq!(transformed[1].shape(), &[4, 1]);
    }

    #[test]
    fn test_generalized_cca_error_cases() {
        // Test with insufficient views
        let view1 = array![[1.0, 2.0], [3.0, 4.0]];
        let views = vec![view1];
        let gcca = GeneralizedCCA::new(1);
        assert!(gcca.fit_views(&views).is_err());

        // Test with mismatched sample sizes
        let view1 = array![[1.0, 2.0], [3.0, 4.0]];
        let view2 = array![[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]];
        let views = vec![view1, view2];
        let gcca = GeneralizedCCA::new(1);
        assert!(gcca.fit_views(&views).is_err());
    }

    #[test]
    fn test_generalized_cca_regularization() {
        let view1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let view2 = array![[2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]];
        let view3 = array![[1.5, 2.5], [3.5, 4.5], [5.5, 6.5], [7.5, 8.5]];
        let views = vec![view1, view2, view3];

        let gcca = GeneralizedCCA::new(1).regularization(0.1);
        let fitted = gcca.fit_views(&views).unwrap();

        // Should work with regularization
        assert!(fitted.explained_variance()[0] >= 0.0);
        assert!(fitted.canonical_correlations()[0] >= 0.0);
    }

    #[test]
    fn test_cholesky_decomposition() {
        let gcca = GeneralizedCCA::new(1);

        // Test with a simple positive definite matrix
        let A = array![[4.0, 2.0], [2.0, 3.0]];
        let L = gcca.cholesky_decomposition(&A).unwrap();

        // Check that L * L^T = A
        let reconstructed = L.dot(&L.t());
        for i in 0..A.nrows() {
            for j in 0..A.ncols() {
                assert!((reconstructed[[i, j]] - A[[i, j]]).abs() < 1e-10);
            }
        }
    }
}
