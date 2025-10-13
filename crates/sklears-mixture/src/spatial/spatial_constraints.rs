//! Spatial Constraint Types and Configuration
//!
//! This module defines the various types of spatial constraints that can be applied
//! to mixture models, along with configuration structures for spatial modeling parameters.

use crate::common::CovarianceType;

/// Spatial constraint types for spatial mixture models
///
/// Different types of spatial constraints can be applied to mixture models
/// to incorporate spatial relationships and ensure spatial coherence in clustering.
#[derive(Debug, Clone, PartialEq)]
pub enum SpatialConstraint {
    /// Adjacency-based spatial constraint using neighbor relationships
    ///
    /// Uses topological adjacency to define spatial relationships.
    /// Neighboring spatial units influence each other's cluster assignments.
    Adjacency,

    /// Distance-based constraint with specified radius
    ///
    /// Spatial units within a given distance radius influence each other.
    /// The radius parameter controls the spatial influence range.
    Distance { radius: f64 },

    /// Grid-based constraint for regular spatial patterns
    ///
    /// Assumes data is arranged in a regular grid pattern.
    /// Neighboring grid cells influence each other based on grid topology.
    Grid { rows: usize, cols: usize },

    /// Custom spatial constraint with user-defined relationships
    ///
    /// Allows for arbitrary spatial relationship definitions.
    /// Users can provide custom spatial weights matrices.
    Custom,
}

impl Default for SpatialConstraint {
    fn default() -> Self {
        Self::Distance { radius: 1.0 }
    }
}

impl SpatialConstraint {
    /// Check if the constraint requires coordinate information
    pub fn requires_coordinates(&self) -> bool {
        matches!(self, Self::Distance { .. })
    }

    /// Check if the constraint requires grid dimensions
    pub fn requires_grid_dimensions(&self) -> bool {
        matches!(self, Self::Grid { .. })
    }

    /// Get a descriptive name for the constraint type
    pub fn constraint_name(&self) -> &'static str {
        match self {
            Self::Adjacency => "adjacency",
            Self::Distance { .. } => "distance",
            Self::Grid { .. } => "grid",
            Self::Custom => "custom",
        }
    }
}

/// Configuration for spatially constrained mixture models
///
/// This structure contains all parameters needed to configure spatial mixture models,
/// including the number of components, covariance structure, spatial constraints,
/// and optimization parameters.
#[derive(Debug, Clone)]
pub struct SpatialMixtureConfig {
    /// Number of mixture components
    pub n_components: usize,

    /// Type of covariance matrix structure
    pub covariance_type: CovarianceType,

    /// Spatial constraint configuration
    pub spatial_constraint: SpatialConstraint,

    /// Weight for spatial constraint term (0.0 to 1.0)
    ///
    /// Controls the balance between data likelihood and spatial constraint.
    /// Higher values enforce stronger spatial coherence.
    pub spatial_weight: f64,

    /// Maximum number of EM iterations
    pub max_iter: usize,

    /// Convergence tolerance for EM algorithm
    pub tol: f64,

    /// Random seed for reproducibility
    pub random_state: Option<u64>,
}

impl Default for SpatialMixtureConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            covariance_type: CovarianceType::Full,
            spatial_constraint: SpatialConstraint::default(),
            spatial_weight: 0.1,
            max_iter: 100,
            tol: 1e-4,
            random_state: None,
        }
    }
}

impl SpatialMixtureConfig {
    /// Create a new configuration with specified number of components
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            ..Default::default()
        }
    }

    /// Validate the configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.n_components == 0 {
            return Err("Number of components must be greater than 0".to_string());
        }

        if !(0.0..=1.0).contains(&self.spatial_weight) {
            return Err("Spatial weight must be between 0.0 and 1.0".to_string());
        }

        if self.max_iter == 0 {
            return Err("Maximum iterations must be greater than 0".to_string());
        }

        if self.tol <= 0.0 {
            return Err("Tolerance must be positive".to_string());
        }

        // Validate spatial constraint specific parameters
        match &self.spatial_constraint {
            SpatialConstraint::Distance { radius } => {
                if *radius <= 0.0 {
                    return Err("Distance radius must be positive".to_string());
                }
            }
            SpatialConstraint::Grid { rows, cols } => {
                if *rows == 0 || *cols == 0 {
                    return Err("Grid dimensions must be greater than 0".to_string());
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Get the effective number of parameters for this configuration
    pub fn parameter_count(&self, n_features: usize) -> usize {
        let weight_params = self.n_components - 1; // n-1 independent weights
        let mean_params = self.n_components * n_features;

        let covariance_params = match self.covariance_type {
            CovarianceType::Full => self.n_components * n_features * (n_features + 1) / 2,
            CovarianceType::Diagonal => self.n_components * n_features,
            CovarianceType::Spherical => self.n_components,
            CovarianceType::Tied => n_features * (n_features + 1) / 2,
        };

        weight_params + mean_params + covariance_params
    }

    /// Check if the configuration is suitable for the given data size
    pub fn check_data_requirements(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(), String> {
        if n_samples < self.n_components {
            return Err(format!(
                "Number of samples ({}) must be at least the number of components ({})",
                n_samples, self.n_components
            ));
        }

        let min_samples = self.parameter_count(n_features) * 2;
        if n_samples < min_samples {
            return Err(format!(
                "Insufficient data: need at least {} samples for {} parameters",
                min_samples,
                self.parameter_count(n_features)
            ));
        }

        Ok(())
    }
}

/// Spatial regularization types for mixture models
#[derive(Debug, Clone, PartialEq)]
pub enum SpatialRegularization {
    /// No spatial regularization
    None,

    /// L1 spatial penalty (encourages sparse spatial effects)
    L1 { lambda: f64 },

    /// L2 spatial penalty (smooth spatial effects)
    L2 { lambda: f64 },

    /// Total variation penalty (piecewise constant spatial effects)
    TotalVariation { lambda: f64 },

    /// Elastic net combination of L1 and L2
    ElasticNet { l1_ratio: f64, lambda: f64 },
}

impl Default for SpatialRegularization {
    fn default() -> Self {
        Self::None
    }
}

impl SpatialRegularization {
    /// Get the regularization strength parameter
    pub fn lambda(&self) -> f64 {
        match self {
            Self::None => 0.0,
            Self::L1 { lambda } => *lambda,
            Self::L2 { lambda } => *lambda,
            Self::TotalVariation { lambda } => *lambda,
            Self::ElasticNet { lambda, .. } => *lambda,
        }
    }

    /// Check if regularization is active
    pub fn is_active(&self) -> bool {
        !matches!(self, Self::None) && self.lambda() > 0.0
    }
}

/// Spatial smoothing parameters for mixture models
#[derive(Debug, Clone)]
pub struct SpatialSmoothingConfig {
    /// Type of spatial regularization
    pub regularization: SpatialRegularization,

    /// Kernel bandwidth for spatial smoothing
    pub bandwidth: f64,

    /// Whether to use adaptive bandwidth based on local density
    pub adaptive_bandwidth: bool,

    /// Minimum bandwidth value (for adaptive bandwidth)
    pub min_bandwidth: f64,

    /// Maximum bandwidth value (for adaptive bandwidth)
    pub max_bandwidth: f64,
}

impl Default for SpatialSmoothingConfig {
    fn default() -> Self {
        Self {
            regularization: SpatialRegularization::default(),
            bandwidth: 1.0,
            adaptive_bandwidth: false,
            min_bandwidth: 0.1,
            max_bandwidth: 10.0,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_constraint_default() {
        let constraint = SpatialConstraint::default();
        assert!(matches!(constraint, SpatialConstraint::Distance { radius } if radius == 1.0));
    }

    #[test]
    fn test_spatial_constraint_properties() {
        let distance_constraint = SpatialConstraint::Distance { radius: 2.0 };
        assert!(distance_constraint.requires_coordinates());
        assert!(!distance_constraint.requires_grid_dimensions());
        assert_eq!(distance_constraint.constraint_name(), "distance");

        let grid_constraint = SpatialConstraint::Grid { rows: 10, cols: 10 };
        assert!(!grid_constraint.requires_coordinates());
        assert!(grid_constraint.requires_grid_dimensions());
        assert_eq!(grid_constraint.constraint_name(), "grid");
    }

    #[test]
    fn test_spatial_mixture_config_validation() {
        let mut config = SpatialMixtureConfig::new(3);
        assert!(config.validate().is_ok());

        config.n_components = 0;
        assert!(config.validate().is_err());

        config.n_components = 3;
        config.spatial_weight = 1.5;
        assert!(config.validate().is_err());

        config.spatial_weight = 0.5;
        config.max_iter = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_spatial_mixture_config_parameter_count() {
        let config = SpatialMixtureConfig {
            n_components: 3,
            covariance_type: CovarianceType::Full,
            ..Default::default()
        };

        let n_features = 2;
        let param_count = config.parameter_count(n_features);

        // 3 components: 2 weights + 6 means + 9 covariances = 17 parameters
        assert_eq!(param_count, 17);
    }

    #[test]
    fn test_spatial_regularization() {
        let l1_reg = SpatialRegularization::L1 { lambda: 0.1 };
        assert!(l1_reg.is_active());
        assert_eq!(l1_reg.lambda(), 0.1);

        let no_reg = SpatialRegularization::None;
        assert!(!no_reg.is_active());
        assert_eq!(no_reg.lambda(), 0.0);
    }
}
