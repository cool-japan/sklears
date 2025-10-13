//! Multi-view Canonical Correlation Analysis (Multi-view CCA)
//!
//! Multi-view CCA is an extension of canonical correlation analysis to handle
//! more than two datasets (views) simultaneously, finding linear combinations
//! that maximize correlation across all views.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{thread_rng, Rng};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Transform},
    types::Float,
};
use std::marker::PhantomData;

/// Multi-view Canonical Correlation Analysis
///
/// Multi-view CCA extends traditional CCA to handle multiple datasets (views) by finding
/// linear transformations that maximize correlation across all views simultaneously.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::array;
/// use sklears_cross_decomposition::{MultiViewCCA};
/// use sklears_core::traits::Fit;
///
/// let view1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let view2 = array![[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]];
/// let view3 = array![[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]];
/// let other_views = vec![view2, view3].into();
///
/// let mv_cca = MultiViewCCA::new(1).regularization(0.1);
/// let fitted = mv_cca.fit(&view1, &other_views).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MultiViewCCA<State = Untrained> {
    /// Number of canonical components
    pub n_components: usize,
    /// Scale input data
    pub scale: bool,
    /// Regularization parameter
    pub regularization: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: Float,
    /// Copy input data
    pub copy: bool,
    /// Canonical weights for each view
    weights_: Option<Vec<Array2<Float>>>,
    /// Canonical loadings for each view
    loadings_: Option<Vec<Array2<Float>>>,
    /// Canonical scores for each view
    scores_: Option<Vec<Array2<Float>>>,
    /// Canonical correlations between views
    correlations_: Option<Array2<Float>>,
    /// Mean of each view
    means_: Option<Vec<Array1<Float>>>,
    /// Standard deviation of each view
    stds_: Option<Vec<Array1<Float>>>,
    /// Number of iterations for convergence
    n_iter_: Option<usize>,
    /// Explained variance for each component
    explained_variance_: Option<Array1<Float>>,
    /// Explained variance ratio for each component
    explained_variance_ratio_: Option<Array1<Float>>,
    /// State marker
    _state: PhantomData<State>,
}

/// Marker type for untrained state
#[derive(Debug, Clone)]
pub struct Untrained;

/// Marker type for trained state
#[derive(Debug, Clone)]
pub struct Trained;

impl MultiViewCCA<Untrained> {
    /// Create a new Multi-view CCA with specified number of components
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            scale: true,
            regularization: 0.0,
            max_iter: 500,
            tol: 1e-6,
            copy: true,
            weights_: None,
            loadings_: None,
            scores_: None,
            correlations_: None,
            means_: None,
            stds_: None,
            n_iter_: None,
            explained_variance_: None,
            explained_variance_ratio_: None,
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

    /// Set whether to copy input data
    pub fn copy(mut self, copy: bool) -> Self {
        self.copy = copy;
        self
    }
}

impl Estimator for MultiViewCCA<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

/// Wrapper type for multiple views
#[derive(Debug, Clone)]
pub struct MultipleViews {
    pub views: Vec<Array2<Float>>,
}

impl From<Vec<Array2<Float>>> for MultipleViews {
    fn from(views: Vec<Array2<Float>>) -> Self {
        Self { views }
    }
}

impl Fit<Array2<Float>, MultipleViews> for MultiViewCCA<Untrained> {
    type Fitted = MultiViewCCA<Trained>;

    fn fit(self, first_view: &Array2<Float>, other_views: &MultipleViews) -> Result<Self::Fitted> {
        let mut all_views = vec![first_view.clone()];
        all_views.extend(other_views.views.iter().cloned());
        self.fit_views(&all_views)
    }
}

impl MultiViewCCA<Untrained> {
    /// Fit the model to multiple views
    pub fn fit_views(self, views: &Vec<Array2<Float>>) -> Result<MultiViewCCA<Trained>> {
        if views.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "At least 2 views are required for multi-view CCA".to_string(),
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
        if self.n_components > min_features {
            return Err(SklearsError::InvalidInput(format!(
                "n_components ({}) cannot exceed minimum number of features ({})",
                self.n_components, min_features
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
                let std = centered_view.var_axis(Axis(0), 0.0).mapv(|v| v.sqrt());
                for (i, &s) in std.iter().enumerate() {
                    if s > 0.0 {
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

        // Initialize weights and scores
        let mut weights: Vec<Array2<Float>> = views
            .iter()
            .map(|view| Array2::zeros((view.ncols(), self.n_components)))
            .collect();

        let mut scores: Vec<Array2<Float>> = views
            .iter()
            .map(|_| Array2::zeros((n_samples, self.n_components)))
            .collect();

        let mut loadings: Vec<Array2<Float>> = views
            .iter()
            .map(|view| Array2::zeros((view.ncols(), self.n_components)))
            .collect();

        let mut correlations = Array2::zeros((views.len(), self.n_components));
        let mut n_iter = 0;

        // Iterate over components
        for comp in 0..self.n_components {
            let mut converged = false;
            let mut iter_count = 0;

            // Initialize weights randomly for this component
            for (view_idx, view) in scaled_views.iter().enumerate() {
                let mut weight =
                    Array1::from_iter((0..view.ncols()).map(|_| thread_rng().gen::<Float>()));
                let norm = (weight.dot(&weight)).sqrt();
                if norm > 0.0 {
                    weight /= norm;
                }
                weights[view_idx].column_mut(comp).assign(&weight);
            }

            while !converged && iter_count < self.max_iter {
                let mut old_weights = weights.clone();

                // Update weights for each view based on correlations with other views
                for view_idx in 0..scaled_views.len() {
                    let mut correlation_sum: Array1<Float> =
                        Array1::zeros(scaled_views[view_idx].ncols());

                    // Compute weighted sum of correlations with other views
                    for other_idx in 0..scaled_views.len() {
                        if other_idx != view_idx {
                            let other_score =
                                scaled_views[other_idx].dot(&weights[other_idx].column(comp));
                            let correlation = scaled_views[view_idx].t().dot(&other_score);
                            correlation_sum = correlation_sum + &correlation;
                        }
                    }

                    // Add regularization
                    if self.regularization > 0.0 {
                        let reg_term = &weights[view_idx].column(comp) * self.regularization;
                        correlation_sum = correlation_sum + &reg_term;
                    }

                    // Normalize the weight
                    let norm = (correlation_sum.dot(&correlation_sum)).sqrt();
                    if norm > 0.0 {
                        correlation_sum /= norm;
                    }

                    weights[view_idx].column_mut(comp).assign(&correlation_sum);
                }

                // Check convergence
                let mut max_change: Float = 0.0;
                for view_idx in 0..weights.len() {
                    let change = (&weights[view_idx].column(comp)
                        - &old_weights[view_idx].column(comp))
                        .mapv(|x| x.abs())
                        .sum();
                    max_change = max_change.max(change);
                }

                if max_change < self.tol {
                    converged = true;
                }

                iter_count += 1;
            }

            n_iter = iter_count;

            // Compute scores and loadings for this component
            for view_idx in 0..scaled_views.len() {
                let score = scaled_views[view_idx].dot(&weights[view_idx].column(comp));
                scores[view_idx].column_mut(comp).assign(&score);

                let loading = scaled_views[view_idx].t().dot(&score);
                loadings[view_idx].column_mut(comp).assign(&loading);
            }

            // Compute correlations between views for this component
            for view_idx in 0..scaled_views.len() {
                let mut total_correlation = 0.0;
                let mut count = 0;

                for other_idx in 0..scaled_views.len() {
                    if other_idx != view_idx {
                        let correlation = scores[view_idx]
                            .column(comp)
                            .dot(&scores[other_idx].column(comp))
                            / (n_samples as Float - 1.0);
                        total_correlation += correlation.abs();
                        count += 1;
                    }
                }

                correlations[[view_idx, comp]] = if count > 0 {
                    total_correlation / count as Float
                } else {
                    0.0
                };
            }

            // Deflate the data for next component
            if comp < self.n_components - 1 {
                for view_idx in 0..scaled_views.len() {
                    let deflation = scores[view_idx]
                        .column(comp)
                        .insert_axis(Axis(1))
                        .dot(&loadings[view_idx].column(comp).insert_axis(Axis(0)));
                    scaled_views[view_idx] = &scaled_views[view_idx] - &deflation;
                }
            }
        }

        // Compute explained variance
        let mut explained_var = Array1::zeros(self.n_components);
        let mut total_var = 0.0;

        for view in views {
            total_var += view.var_axis(Axis(0), 1.0).sum();
        }

        for comp in 0..self.n_components {
            let mut comp_var = 0.0;
            for view_idx in 0..views.len() {
                comp_var += scores[view_idx].column(comp).var(1.0);
            }
            explained_var[comp] = comp_var;
        }

        let explained_var_ratio = &explained_var / total_var;

        Ok(MultiViewCCA {
            n_components: self.n_components,
            scale: self.scale,
            regularization: self.regularization,
            max_iter: self.max_iter,
            tol: self.tol,
            copy: self.copy,
            weights_: Some(weights),
            loadings_: Some(loadings),
            scores_: Some(scores),
            correlations_: Some(correlations),
            means_: Some(means),
            stds_: Some(stds),
            n_iter_: Some(n_iter),
            explained_variance_: Some(explained_var),
            explained_variance_ratio_: Some(explained_var_ratio),
            _state: PhantomData,
        })
    }
}

impl Transform<Vec<Array2<Float>>, Vec<Array2<Float>>> for MultiViewCCA<Trained> {
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
                    if std > 0.0 {
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

impl MultiViewCCA<Trained> {
    /// Get the canonical weights for each view
    pub fn weights(&self) -> &Vec<Array2<Float>> {
        self.weights_.as_ref().unwrap()
    }

    /// Get the canonical loadings for each view
    pub fn loadings(&self) -> &Vec<Array2<Float>> {
        self.loadings_.as_ref().unwrap()
    }

    /// Get the canonical scores for each view
    pub fn scores(&self) -> &Vec<Array2<Float>> {
        self.scores_.as_ref().unwrap()
    }

    /// Get the canonical correlations between views
    pub fn correlations(&self) -> &Array2<Float> {
        self.correlations_.as_ref().unwrap()
    }

    /// Get the explained variance for each component
    pub fn explained_variance(&self) -> &Array1<Float> {
        self.explained_variance_.as_ref().unwrap()
    }

    /// Get the explained variance ratio for each component
    pub fn explained_variance_ratio(&self) -> &Array1<Float> {
        self.explained_variance_ratio_.as_ref().unwrap()
    }

    /// Get the number of iterations for convergence
    pub fn n_iter(&self) -> usize {
        self.n_iter_.unwrap()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use sklears_core::traits::Fit;

    #[test]
    fn test_multiview_cca_basic() {
        let view1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let view2 = array![[2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]];
        let view3 = array![[1.5, 2.5], [3.5, 4.5], [5.5, 6.5], [7.5, 8.5]];
        let views = vec![view1.clone(), view2, view3];

        let mv_cca = MultiViewCCA::new(1);
        let fitted = mv_cca.fit_views(&views).unwrap();

        // Check that weights were computed
        assert_eq!(fitted.weights().len(), 3);
        assert_eq!(fitted.weights()[0].shape(), &[2, 1]);
        assert_eq!(fitted.weights()[1].shape(), &[2, 1]);
        assert_eq!(fitted.weights()[2].shape(), &[2, 1]);

        // Check that scores were computed
        assert_eq!(fitted.scores().len(), 3);
        assert_eq!(fitted.scores()[0].shape(), &[4, 1]);

        // Check that correlations were computed
        assert_eq!(fitted.correlations().shape(), &[3, 1]);
    }

    #[test]
    fn test_multiview_cca_transform() {
        let view1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let view2 = array![[2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]];
        let views = vec![view1.clone(), view2.clone()];

        let mv_cca = MultiViewCCA::new(1);
        let fitted = mv_cca.fit_views(&views).unwrap();
        let transformed = fitted.transform(&views).unwrap();

        assert_eq!(transformed.len(), 2);
        assert_eq!(transformed[0].shape(), &[4, 1]);
        assert_eq!(transformed[1].shape(), &[4, 1]);
    }

    #[test]
    fn test_multiview_cca_error_cases() {
        // Test with insufficient views
        let view1 = array![[1.0, 2.0], [3.0, 4.0]];
        let views = vec![view1];
        let mv_cca = MultiViewCCA::new(1);
        assert!(mv_cca.fit_views(&views).is_err());

        // Test with mismatched sample sizes
        let view1 = array![[1.0, 2.0], [3.0, 4.0]];
        let view2 = array![[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]];
        let views = vec![view1, view2];
        let mv_cca = MultiViewCCA::new(1);
        assert!(mv_cca.fit_views(&views).is_err());
    }

    #[test]
    fn test_multiview_cca_regularization() {
        let view1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let view2 = array![[2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]];
        let view3 = array![[1.5, 2.5], [3.5, 4.5], [5.5, 6.5], [7.5, 8.5]];
        let views = vec![view1, view2, view3];

        let mv_cca = MultiViewCCA::new(1).regularization(0.1);
        let fitted = mv_cca.fit_views(&views).unwrap();

        // Should converge with regularization
        assert!(fitted.n_iter() > 0);
        assert!(fitted.explained_variance()[0] >= 0.0);
    }
}
