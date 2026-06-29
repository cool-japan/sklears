//! Principal Component Analysis and dimensionality reduction utilities
//!
//! This module provides comprehensive PCA implementations using truncated SVD
//! for exact principal components computation.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_linalg::svd;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};

/// Basic PCA configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PcaConfig {
    pub n_components: Option<usize>,
    pub whiten: bool,
    pub svd_solver: String,
    pub tol: Float,
    pub iterated_power: usize,
    pub random_state: Option<u64>,
}

impl Default for PcaConfig {
    fn default() -> Self {
        Self {
            n_components: None,
            whiten: false,
            svd_solver: "auto".to_string(),
            tol: 0.0,
            iterated_power: 5,
            random_state: None,
        }
    }
}

/// Basic PCA implementation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PCA<State = Untrained> {
    config: PcaConfig,
    state: std::marker::PhantomData<State>,
}

/// Trained PCA model
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PcaTrained {
    config: PcaConfig,
    /// Principal components (loadings)
    pub components: Array2<Float>,
    /// Explained variance
    pub explained_variance: Array1<Float>,
    /// Explained variance ratio
    pub explained_variance_ratio: Array1<Float>,
    /// Mean of training data
    pub mean: Array1<Float>,
    /// Number of components
    pub n_components: usize,
}

impl PCA<Untrained> {
    pub fn new(config: PcaConfig) -> Self {
        Self {
            config,
            state: std::marker::PhantomData,
        }
    }

    pub fn builder() -> PcaBuilder {
        PcaBuilder::default()
    }
}

impl Estimator for PCA<Untrained> {
    type Config = PcaConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, ()> for PCA<Untrained> {
    type Fitted = PcaTrained;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array must be non-empty".to_string(),
            ));
        }

        // Determine number of components
        let n_components = self
            .config
            .n_components
            .unwrap_or(n_features.min(n_samples));

        if n_components > n_features.min(n_samples) {
            return Err(SklearsError::InvalidInput(
                "n_components cannot be larger than min(n_samples, n_features)".to_string(),
            ));
        }

        // Compute mean
        let mean = x
            .mean_axis(Axis(0))
            .expect("array should have elements for mean computation");

        // Center data: X_c = X - mean
        let x_centered = x - &mean.clone().insert_axis(Axis(0));

        // Full SVD of the centered data matrix: X_c = U S Vt
        // Principal components are rows of Vt (right singular vectors).
        // Singular values s relate to explained variance: var_i = s_i^2 / (n_samples - 1).
        let (_u, singular_values, vt) = svd(&x_centered.view(), false, None)
            .map_err(|e| SklearsError::NumericalError(format!("SVD failed in PCA.fit: {}", e)))?;

        // Take the top n_components principal components (rows of Vt)
        let components = vt
            .slice(scirs2_core::ndarray::s![..n_components, ..])
            .to_owned();

        // Explained variance: s_i^2 / (n - 1), where s_i are singular values
        let denom = if n_samples > 1 {
            (n_samples - 1) as Float
        } else {
            1.0
        };
        let explained_variance: Array1<Float> = singular_values
            .iter()
            .take(n_components)
            .map(|&s| s * s / denom)
            .collect();

        // Total variance = sum of all squared singular values / (n - 1)
        let total_variance: Float = singular_values.iter().map(|&s| s * s / denom).sum();

        let explained_variance_ratio = if total_variance > Float::EPSILON {
            explained_variance.mapv(|v| v / total_variance)
        } else {
            Array1::zeros(n_components)
        };

        Ok(PcaTrained {
            config: self.config,
            components,
            explained_variance,
            explained_variance_ratio,
            mean,
            n_components,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for PcaTrained {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        // Center data
        let x_centered = x - &self.mean.clone().insert_axis(Axis(0));

        // Project onto components
        let x_transformed = x_centered.dot(&self.components.t());

        Ok(x_transformed)
    }
}

/// PCA builder pattern
#[derive(Debug, Clone, Default)]
pub struct PcaBuilder {
    config: PcaConfig,
}

impl PcaBuilder {
    pub fn n_components(mut self, n_components: Option<usize>) -> Self {
        self.config.n_components = n_components;
        self
    }

    pub fn whiten(mut self, whiten: bool) -> Self {
        self.config.whiten = whiten;
        self
    }

    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }

    pub fn build(self) -> PCA<Untrained> {
        PCA::new(self.config)
    }
}

// Placeholder types for compatibility with existing code
pub type PcaProcessor = PCA<Untrained>;
pub type PcaValidator = PCA<Untrained>;
pub type PcaEstimator = PCA<Untrained>;
pub type PcaTransformer = PcaTrained;
pub type PcaAnalyzer = PcaTrained;
pub type DimensionalityReducer = PcaTrained;
pub type DecompositionEngine = PcaTrained;
pub type StandardPcaAnalyzer = PcaTrained;
pub type PrincipalComponentAnalysis = PcaTrained;
pub type VarianceExplained = Array1<Float>;
pub type ComponentExtractor = PcaTrained;
pub type DimensionalityReduction = PcaTrained;
pub type PcaProjection = Array2<Float>;
pub type PcaInverseTransform = Array2<Float>;

// Additional placeholder types that might be referenced elsewhere
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RobustPCA;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SparsePCA;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ProbabilisticPCA;

// SVD solver related types
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SvdSolver;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SvdSolverConfig;

// More placeholder types for comprehensive compatibility
pub type RobustPcaConfig = PcaConfig;
pub type SparsePcaConfig = PcaConfig;
pub type ProbabilisticPcaConfig = PcaConfig;
pub type IncrementalPcaConfig = PcaConfig;

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    /// Build a simple 2D dataset with clear principal directions.
    fn two_feature_data() -> Array2<Float> {
        array![
            [1.0, 0.5],
            [2.0, 1.0],
            [3.0, 1.5],
            [4.0, 2.0],
            [5.0, 2.5],
            [6.0, 3.0],
        ]
    }

    #[test]
    fn test_pca_output_shape() {
        let x = two_feature_data();
        let config = PcaConfig {
            n_components: Some(1),
            ..Default::default()
        };
        let trained = PCA::new(config).fit(&x, &()).expect("fit should succeed");
        let transformed = trained.transform(&x).expect("transform should succeed");
        assert_eq!(transformed.shape(), &[6, 1]);
    }

    #[test]
    fn test_pca_components_shape() {
        let x = two_feature_data();
        let config = PcaConfig {
            n_components: Some(2),
            ..Default::default()
        };
        let trained = PCA::new(config).fit(&x, &()).expect("fit should succeed");
        assert_eq!(trained.components.shape(), &[2, 2]);
    }

    #[test]
    fn test_pca_explained_variance_ratio_sums_to_one() {
        let x = two_feature_data();
        let config = PcaConfig {
            n_components: Some(2),
            ..Default::default()
        };
        let trained = PCA::new(config).fit(&x, &()).expect("fit should succeed");
        let ratio_sum: Float = trained.explained_variance_ratio.sum();
        assert!((ratio_sum - 1.0).abs() < 1e-10, "ratio sum = {}", ratio_sum);
    }

    #[test]
    fn test_pca_explained_variance_ratio_non_negative() {
        let x = two_feature_data();
        let config = PcaConfig {
            n_components: Some(2),
            ..Default::default()
        };
        let trained = PCA::new(config).fit(&x, &()).expect("fit should succeed");
        for &r in trained.explained_variance_ratio.iter() {
            assert!(r >= 0.0, "negative ratio {}", r);
        }
    }

    #[test]
    fn test_pca_explained_variance_descending() {
        // For a dataset with clear variance, PC1 should explain more than PC2.
        let x = two_feature_data();
        let config = PcaConfig {
            n_components: Some(2),
            ..Default::default()
        };
        let trained = PCA::new(config).fit(&x, &()).expect("fit should succeed");
        assert!(
            trained.explained_variance[0] >= trained.explained_variance[1],
            "PC1 variance {} < PC2 variance {}",
            trained.explained_variance[0],
            trained.explained_variance[1]
        );
    }

    #[test]
    fn test_pca_transform_finiteness() {
        let x = two_feature_data();
        let config = PcaConfig {
            n_components: Some(1),
            ..Default::default()
        };
        let trained = PCA::new(config).fit(&x, &()).expect("fit should succeed");
        let t = trained.transform(&x).expect("transform should succeed");
        for &v in t.iter() {
            assert!(v.is_finite(), "non-finite value {}", v);
        }
    }

    #[test]
    fn test_pca_n_components_validation() {
        let x = two_feature_data();
        // n_components > min(n_samples, n_features)
        let config = PcaConfig {
            n_components: Some(100),
            ..Default::default()
        };
        assert!(PCA::new(config).fit(&x, &()).is_err());
    }

    #[test]
    fn test_pca_empty_input() {
        let x: Array2<Float> = Array2::zeros((0, 3));
        let config = PcaConfig::default();
        assert!(PCA::new(config).fit(&x, &()).is_err());
    }

    #[test]
    fn test_pca_first_component_captures_most_variance() {
        // The dataset has a strong first principal component (y ≈ 0.5x)
        let x = two_feature_data();
        let config = PcaConfig {
            n_components: Some(1),
            ..Default::default()
        };
        let trained = PCA::new(config).fit(&x, &()).expect("fit should succeed");
        // With 1 component, variance ratio must be close to 1 (data is nearly 1-D)
        let ratio = trained.explained_variance_ratio[0];
        assert!(ratio > 0.99, "1-PC ratio = {}", ratio);
    }
}
