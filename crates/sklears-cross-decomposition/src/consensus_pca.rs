//! Consensus Principal Component Analysis (Consensus PCA)
//!
//! Consensus PCA is a multi-view dimensionality reduction technique that finds
//! a consensus low-dimensional representation across multiple data views while
//! preserving the structure within each view.

use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Transform},
    types::Float,
};
use std::marker::PhantomData;

/// Consensus Principal Component Analysis
///
/// Consensus PCA finds a common low-dimensional representation across multiple views
/// by optimizing a consensus criterion that balances between-view agreement and
/// within-view variance preservation.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::array;
/// use sklears_cross_decomposition::ConsensusPCA;
/// use sklears_core::traits::Fit;
///
/// let view1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let view2 = array![[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]];
/// let view3 = array![[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]];
/// let views = vec![view1, view2, view3];
///
/// let cpca = ConsensusPCA::new(2).consensus_weight(0.5);
/// let fitted = cpca.fit(&views, &()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ConsensusPCA<State = Untrained> {
    /// Number of consensus components
    pub n_components: usize,
    /// Scale input data
    pub scale: bool,
    /// Weight for consensus vs individual variance (0.0 to 1.0)
    pub consensus_weight: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: Float,
    /// Copy input data
    pub copy: bool,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Consensus components (shared across views)
    consensus_components_: Option<Array2<Float>>,
    /// Individual components for each view
    individual_components_: Option<Vec<Array2<Float>>>,
    /// Transformation matrices for each view
    transformations_: Option<Vec<Array2<Float>>>,
    /// Mean of each view
    means_: Option<Vec<Array1<Float>>>,
    /// Standard deviation of each view
    stds_: Option<Vec<Array1<Float>>>,
    /// Consensus explained variance for each component
    consensus_explained_variance_: Option<Array1<Float>>,
    /// Individual explained variance for each view and component
    individual_explained_variance_: Option<Vec<Array1<Float>>>,
    /// Total explained variance ratio
    explained_variance_ratio_: Option<Array1<Float>>,
    /// Number of iterations for convergence
    n_iter_: Option<usize>,
    /// Reconstruction error for each view
    reconstruction_errors_: Option<Array1<Float>>,
    /// State marker
    _state: PhantomData<State>,
}

/// Marker type for untrained state
#[derive(Debug, Clone)]
pub struct Untrained;

/// Marker type for trained state
#[derive(Debug, Clone)]
pub struct Trained;

impl ConsensusPCA<Untrained> {
    /// Create a new Consensus PCA with specified number of components
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            scale: true,
            consensus_weight: 0.5,
            max_iter: 100,
            tol: 1e-6,
            copy: true,
            random_state: None,
            consensus_components_: None,
            individual_components_: None,
            transformations_: None,
            means_: None,
            stds_: None,
            consensus_explained_variance_: None,
            individual_explained_variance_: None,
            explained_variance_ratio_: None,
            n_iter_: None,
            reconstruction_errors_: None,
            _state: PhantomData,
        }
    }

    /// Set whether to scale the data
    pub fn scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }

    /// Set consensus weight (0.0 = only individual variance, 1.0 = only consensus)
    pub fn consensus_weight(mut self, weight: Float) -> Self {
        self.consensus_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
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

impl Estimator for ConsensusPCA<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Vec<Array2<Float>>, ()> for ConsensusPCA<Untrained> {
    type Fitted = ConsensusPCA<Trained>;

    fn fit(self, views: &Vec<Array2<Float>>, _target: &()) -> Result<Self::Fitted> {
        self.fit_views(views)
    }
}

impl ConsensusPCA<Untrained> {
    /// Fit the model to multiple views
    pub fn fit_views(self, views: &Vec<Array2<Float>>) -> Result<ConsensusPCA<Trained>> {
        if views.is_empty() {
            return Err(SklearsError::InvalidInput(
                "At least 1 view is required for consensus PCA".to_string(),
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

        // Compute individual PCAs for each view
        let mut individual_components = Vec::new();
        let mut individual_explained_vars = Vec::new();

        for view in &scaled_views {
            let (components, explained_var) = self.compute_pca(view)?;
            individual_components.push(components);
            individual_explained_vars.push(explained_var);
        }

        // Initialize consensus components randomly
        let mut consensus_components = self.initialize_consensus_components(&scaled_views)?;
        let mut converged = false;
        let mut n_iter = 0;

        // Alternating optimization for consensus and individual components
        while !converged && n_iter < self.max_iter {
            let old_consensus = consensus_components.clone();

            // Update consensus components by optimizing across all views
            consensus_components =
                self.update_consensus_components(&scaled_views, &individual_components)?;

            // Check convergence
            let consensus_change = (&consensus_components - &old_consensus)
                .mapv(|x| x.abs())
                .sum();

            if consensus_change < self.tol {
                converged = true;
            }

            n_iter += 1;
        }

        // Compute transformation matrices for each view
        let mut transformations = Vec::new();
        for (view_idx, view) in scaled_views.iter().enumerate() {
            let transformation = self.compute_transformation_matrix(
                view,
                &consensus_components,
                &individual_components[view_idx],
            )?;
            transformations.push(transformation);
        }

        // Compute explained variances
        let consensus_explained_var =
            self.compute_consensus_explained_variance(&scaled_views, &consensus_components)?;
        let total_variance: Float = scaled_views
            .iter()
            .map(|v| v.var_axis(Axis(0), 1.0).sum())
            .sum();

        let explained_var_ratio = &consensus_explained_var / total_variance;

        // Compute reconstruction errors
        let mut reconstruction_errors = Array1::zeros(views.len());
        for (view_idx, view) in scaled_views.iter().enumerate() {
            let transformed = view.dot(&transformations[view_idx]);
            let reconstructed = transformed.dot(&transformations[view_idx].t());
            let error = (view - &reconstructed).mapv(|x| x * x).sum();
            reconstruction_errors[view_idx] = error / (view.len() as Float);
        }

        Ok(ConsensusPCA {
            n_components: self.n_components,
            scale: self.scale,
            consensus_weight: self.consensus_weight,
            max_iter: self.max_iter,
            tol: self.tol,
            copy: self.copy,
            random_state: self.random_state,
            consensus_components_: Some(consensus_components),
            individual_components_: Some(individual_components),
            transformations_: Some(transformations),
            means_: Some(means),
            stds_: Some(stds),
            consensus_explained_variance_: Some(consensus_explained_var),
            individual_explained_variance_: Some(individual_explained_vars),
            explained_variance_ratio_: Some(explained_var_ratio),
            n_iter_: Some(n_iter),
            reconstruction_errors_: Some(reconstruction_errors),
            _state: PhantomData,
        })
    }

    /// Compute PCA for a single view
    fn compute_pca(&self, view: &Array2<Float>) -> Result<(Array2<Float>, Array1<Float>)> {
        let covariance = view.t().dot(view) / (view.nrows() as Float - 1.0);
        let (eigenvalues, eigenvectors) = self.eigenvalue_decomposition(&covariance)?;

        // Take the top n_components
        let components = eigenvectors.slice(s![.., 0..self.n_components]).to_owned();
        let explained_var = eigenvalues.slice(s![0..self.n_components]).to_owned();

        Ok((components, explained_var))
    }

    /// Initialize consensus components using concatenated PCA
    fn initialize_consensus_components(&self, views: &[Array2<Float>]) -> Result<Array2<Float>> {
        // Concatenate all views horizontally
        let mut concatenated_data = views[0].clone();
        for view in views.iter().skip(1) {
            concatenated_data =
                scirs2_core::ndarray::concatenate![Axis(1), concatenated_data, view.clone()];
        }

        // Compute PCA on concatenated data
        let (components, _) = self.compute_pca(&concatenated_data)?;

        // Project back to consensus space (average across views)
        let features_per_view: Vec<usize> = views.iter().map(|v| v.ncols()).collect();
        let total_features = features_per_view.iter().sum::<usize>();
        let mut consensus_components = Array2::zeros((features_per_view[0], self.n_components));

        let mut feature_offset = 0;
        for (view_idx, &n_features) in features_per_view.iter().enumerate() {
            let view_components =
                components.slice(s![feature_offset..feature_offset + n_features, ..]);

            if view_idx == 0 {
                consensus_components.assign(&view_components);
            } else {
                // For now, use the first view's feature space as consensus
                // In practice, this would require more sophisticated alignment
                break;
            }

            feature_offset += n_features;
        }

        Ok(consensus_components)
    }

    /// Update consensus components to maximize agreement across views
    fn update_consensus_components(
        &self,
        views: &[Array2<Float>],
        individual_components: &[Array2<Float>],
    ) -> Result<Array2<Float>> {
        let n_features = views[0].ncols();
        let mut new_consensus = Array2::zeros((n_features, self.n_components));

        for comp in 0..self.n_components {
            let mut consensus_component = Array1::zeros(n_features);
            let mut total_weight = 0.0;

            // Weighted average of individual components
            for (view_idx, view) in views.iter().enumerate() {
                let individual_comp = individual_components[view_idx].column(comp).to_owned();
                let weight = self.consensus_weight;

                // Project individual component to consensus space
                let projected = self.project_to_consensus_space(view, &individual_comp)?;

                consensus_component = consensus_component + weight * &projected;
                total_weight += weight;
            }

            // Normalize
            if total_weight > 0.0 {
                consensus_component /= total_weight;
                let norm = (consensus_component.dot(&consensus_component)).sqrt();
                if norm > self.tol {
                    consensus_component /= norm;
                }
            }

            new_consensus.column_mut(comp).assign(&consensus_component);
        }

        Ok(new_consensus)
    }

    /// Project individual component to consensus space
    fn project_to_consensus_space(
        &self,
        view: &Array2<Float>,
        component: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        // Simple identity projection for now
        // In practice, this would involve more sophisticated cross-view alignment
        Ok(component.clone())
    }

    /// Compute transformation matrix for a view
    fn compute_transformation_matrix(
        &self,
        view: &Array2<Float>,
        consensus_components: &Array2<Float>,
        individual_components: &Array2<Float>,
    ) -> Result<Array2<Float>> {
        // Combine consensus and individual components with weighting
        let mut transformation = Array2::zeros((view.ncols(), self.n_components));

        for comp in 0..self.n_components {
            let consensus_comp = consensus_components.column(comp).to_owned();
            let individual_comp = individual_components.column(comp).to_owned();

            let combined = self.consensus_weight * &consensus_comp
                + (1.0 - self.consensus_weight) * &individual_comp;

            transformation.column_mut(comp).assign(&combined);
        }

        Ok(transformation)
    }

    /// Compute consensus explained variance
    fn compute_consensus_explained_variance(
        &self,
        views: &[Array2<Float>],
        consensus_components: &Array2<Float>,
    ) -> Result<Array1<Float>> {
        let mut explained_var = Array1::zeros(self.n_components);

        for comp in 0..self.n_components {
            let mut total_var = 0.0;
            let consensus_comp = consensus_components.column(comp).to_owned();

            for view in views {
                let scores = view.dot(&consensus_comp);
                total_var += scores.var(1.0);
            }

            explained_var[comp] = total_var / views.len() as Float;
        }

        Ok(explained_var)
    }

    /// Simple eigenvalue decomposition for symmetric matrices
    fn eigenvalue_decomposition(
        &self,
        A: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        // Simple power iteration for demonstration
        // In production, use proper LAPACK routines

        let n = A.nrows();
        let mut eigenvalues = Array1::zeros(n);
        let mut eigenvectors = Array2::zeros((n, n));

        let mut A_deflated = A.clone();

        for comp in 0..n {
            // Power iteration
            let mut v: Array1<Float> = Array1::ones(n);
            v /= (v.dot(&v)).sqrt();

            for _ in 0..100 {
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

            // Deflate matrix
            let vv_t = v
                .view()
                .insert_axis(Axis(1))
                .dot(&v.view().insert_axis(Axis(0)));
            A_deflated = A_deflated - eigenvalue * vv_t;
        }

        // Sort eigenvalues and eigenvectors in descending order
        let mut eigen_pairs: Vec<(Float, Array1<Float>)> = eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &val)| (val, eigenvectors.column(i).to_owned()))
            .collect();

        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let sorted_eigenvalues: Array1<Float> = eigen_pairs.iter().map(|(val, _)| *val).collect();
        let sorted_eigenvectors: Array2<Float> = {
            let mut mat = Array2::zeros((n, n));
            for (i, (_, vec)) in eigen_pairs.iter().enumerate() {
                mat.column_mut(i).assign(vec);
            }
            mat
        };

        Ok((sorted_eigenvalues, sorted_eigenvectors))
    }
}

impl Transform<Vec<Array2<Float>>, Vec<Array2<Float>>> for ConsensusPCA<Trained> {
    /// Transform views to consensus space
    fn transform(&self, views: &Vec<Array2<Float>>) -> Result<Vec<Array2<Float>>> {
        let transformations = self.transformations_.as_ref().unwrap();
        let means = self.means_.as_ref().unwrap();
        let stds = self.stds_.as_ref().unwrap();

        if views.len() != transformations.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} views, got {}",
                transformations.len(),
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

            // Transform to consensus space
            let transformed = scaled_view.dot(&transformations[view_idx]);
            transformed_views.push(transformed);
        }

        Ok(transformed_views)
    }
}

impl ConsensusPCA<Trained> {
    /// Get the consensus components (shared across views)
    pub fn consensus_components(&self) -> &Array2<Float> {
        self.consensus_components_.as_ref().unwrap()
    }

    /// Get the individual components for each view
    pub fn individual_components(&self) -> &Vec<Array2<Float>> {
        self.individual_components_.as_ref().unwrap()
    }

    /// Get the transformation matrices for each view
    pub fn transformations(&self) -> &Vec<Array2<Float>> {
        self.transformations_.as_ref().unwrap()
    }

    /// Get the consensus explained variance for each component
    pub fn consensus_explained_variance(&self) -> &Array1<Float> {
        self.consensus_explained_variance_.as_ref().unwrap()
    }

    /// Get the individual explained variance for each view
    pub fn individual_explained_variance(&self) -> &Vec<Array1<Float>> {
        self.individual_explained_variance_.as_ref().unwrap()
    }

    /// Get the explained variance ratio
    pub fn explained_variance_ratio(&self) -> &Array1<Float> {
        self.explained_variance_ratio_.as_ref().unwrap()
    }

    /// Get the number of iterations for convergence
    pub fn n_iter(&self) -> usize {
        self.n_iter_.unwrap()
    }

    /// Get the reconstruction errors for each view
    pub fn reconstruction_errors(&self) -> &Array1<Float> {
        self.reconstruction_errors_.as_ref().unwrap()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use sklears_core::traits::Fit;

    #[test]
    fn test_consensus_pca_basic() {
        let view1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let view2 = array![[2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]];
        let view3 = array![[1.5, 2.5], [3.5, 4.5], [5.5, 6.5], [7.5, 8.5]];
        let views = vec![view1.clone(), view2, view3];

        let cpca = ConsensusPCA::new(1);
        let fitted = cpca.fit_views(&views).unwrap();

        // Check that components were computed
        assert_eq!(fitted.consensus_components().shape(), &[2, 1]);
        assert_eq!(fitted.individual_components().len(), 3);
        assert_eq!(fitted.transformations().len(), 3);

        // Check dimensions
        for i in 0..3 {
            assert_eq!(fitted.individual_components()[i].shape(), &[2, 1]);
            assert_eq!(fitted.transformations()[i].shape(), &[2, 1]);
        }
    }

    #[test]
    fn test_consensus_pca_transform() {
        let view1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let view2 = array![[2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]];
        let views = vec![view1.clone(), view2.clone()];

        let cpca = ConsensusPCA::new(1);
        let fitted = cpca.fit_views(&views).unwrap();
        let transformed = fitted.transform(&views).unwrap();

        assert_eq!(transformed.len(), 2);
        assert_eq!(transformed[0].shape(), &[4, 1]);
        assert_eq!(transformed[1].shape(), &[4, 1]);
    }

    #[test]
    fn test_consensus_pca_consensus_weight() {
        let view1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let view2 = array![[2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]];
        let views = vec![view1, view2];

        // Test with different consensus weights
        let cpca1 = ConsensusPCA::new(1).consensus_weight(0.0);
        let fitted1 = cpca1.fit_views(&views).unwrap();

        let cpca2 = ConsensusPCA::new(1).consensus_weight(1.0);
        let fitted2 = cpca2.fit_views(&views).unwrap();

        // Both should work but give different results
        assert!(fitted1.n_iter() > 0);
        assert!(fitted2.n_iter() > 0);
    }

    #[test]
    fn test_consensus_pca_error_cases() {
        // Test with empty views
        let views: Vec<Array2<Float>> = vec![];
        let cpca = ConsensusPCA::new(1);
        assert!(cpca.fit_views(&views).is_err());

        // Test with mismatched sample sizes
        let view1 = array![[1.0, 2.0], [3.0, 4.0]];
        let view2 = array![[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]];
        let views = vec![view1, view2];
        let cpca = ConsensusPCA::new(1);
        assert!(cpca.fit_views(&views).is_err());
    }

    #[test]
    fn test_consensus_pca_single_view() {
        let view1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let views = vec![view1];

        let cpca = ConsensusPCA::new(1);
        let fitted = cpca.fit_views(&views).unwrap();

        // Should work with single view (reduces to PCA)
        assert_eq!(fitted.consensus_components().shape(), &[2, 1]);
        assert_eq!(fitted.individual_components().len(), 1);
    }
}
