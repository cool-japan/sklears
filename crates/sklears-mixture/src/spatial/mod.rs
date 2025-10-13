//! Spatial Mixture Models Module
//!
//! This module provides comprehensive spatial mixture modeling capabilities,
//! including spatially constrained Gaussian mixture models, Markov Random Field
//! mixtures, geographic mixture models, and spatial autocorrelation analysis.
//!
//! # Overview
//!
//! Spatial mixture models extend traditional mixture models by incorporating
//! spatial relationships and constraints. This is particularly useful for:
//!
//! - Geographic data analysis
//! - Image segmentation with spatial coherence
//! - Environmental modeling
//! - Urban planning and demography
//! - Spatial clustering with connectivity constraints
//!
//! # Key Features
//!
//! - **Spatially Constrained GMM**: Traditional GMM with spatial smoothness constraints
//! - **Markov Random Field Mixtures**: Mixture models with local neighborhood dependencies
//! - **Geographic Mixtures**: Specialized models for geographic data with elevation and landmarks
//! - **Spatial Statistics**: Comprehensive spatial autocorrelation analysis (Moran's I, Geary's C, LISA)
//! - **Multiple Constraint Types**: Distance-based, adjacency-based, grid-based, and custom constraints
//!
//! # Examples
//!
//! ## Spatially Constrained Gaussian Mixture Model
//!
//! ```rust
//! use sklears_mixture::spatial::{SpatiallyConstrainedGMMBuilder, SpatialConstraint};
//! use scirs2_core::ndarray::array;
//! use sklears_core::traits::Fit;
//!
//! let X = array![[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [6.0, 6.0]];
//! let coords = array![[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [6.0, 6.0]];
//!
//! let gmm = SpatiallyConstrainedGMMBuilder::new(2)
//!     .spatial_constraint(SpatialConstraint::Distance { radius: 2.0 })
//!     .spatial_weight(0.3)
//!     .build()
//!     .with_coordinates(coords);
//!
//! let fitted = gmm.fit(&X, &()).expect("Fitting failed");
//! ```
//!
//! ## Geographic Mixture Model
//!
//! ```rust
//! use sklears_mixture::spatial::GeographicMixtureBuilder;
//! use scirs2_core::ndarray::array;
//! use sklears_core::traits::Fit;
//!
//! let X = array![[0.0, 0.0, 100.0], [1.0, 1.0, 150.0]]; // lat, lon, elevation
//! let landmarks = array![[0.5, 0.5]]; // Point of interest
//!
//! let geo = GeographicMixtureBuilder::new(2)
//!     .use_elevation(true)
//!     .use_distance_features(true)
//!     .landmark_coordinates(landmarks)
//!     .build();
//!
//! let fitted = geo.fit(&X, &()).expect("Fitting failed");
//! ```
//!
//! ## Spatial Autocorrelation Analysis
//!
//! ```rust
//! use sklears_mixture::spatial::{SpatialAutocorrelationAnalyzer, SpatialConstraint};
//! use scirs2_core::ndarray::array;
//!
//! let coords = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
//! let values = array![1.0, 1.0, 0.0]; // Spatially autocorrelated values
//!
//! let analyzer = SpatialAutocorrelationAnalyzer::new(
//!     coords,
//!     SpatialConstraint::Distance { radius: 1.5 }
//! ).expect("Analyzer creation failed");
//!
//! let morans_i = analyzer.morans_i(&values).expect("Moran's I calculation failed");
//! println!("Moran's I: {:.3}, p-value: {:.3}", morans_i.statistic, morans_i.p_value);
//! ```

// Core modules
pub mod geographic_mixture;
pub mod markov_random_field;
pub mod spatial_constraints;
pub mod spatial_statistics;
pub mod spatial_utils;
pub mod spatially_constrained_gmm;

// Tests module
#[allow(non_snake_case)]
#[cfg(test)]
pub mod spatial_tests;

// Public re-exports for main API
pub use spatial_constraints::{
    SpatialConstraint, SpatialMixtureConfig, SpatialRegularization, SpatialSmoothingConfig,
};

pub use spatial_utils::{
    euclidean_distance, k_nearest_neighbors, pairwise_distances, spatial_lag,
    spatial_weights_matrix,
};

pub use spatially_constrained_gmm::{
    SpatiallyConstrainedGMM, SpatiallyConstrainedGMMBuilder, SpatiallyConstrainedGMMTrained,
};

pub use markov_random_field::{
    MRFConfig, MRFInteractionType, MarkovRandomFieldMixture, MarkovRandomFieldMixtureBuilder,
    MarkovRandomFieldMixtureTrained,
};

pub use geographic_mixture::{
    GeographicFeatureConfig, GeographicFeatureType, GeographicMixture, GeographicMixtureBuilder,
    GeographicMixtureTrained,
};

pub use spatial_statistics::{
    GearysC, LocalIndicators, MoransI, SpatialAutocorrelationAnalyzer, SpatialClusteringQuality,
};

/// Convenience module for common spatial operations
pub mod presets {
    use super::*;
    use crate::common::CovarianceType;

    /// Create a standard spatially constrained GMM for image segmentation
    pub fn image_segmentation_gmm(
        n_components: usize,
        image_dims: (usize, usize),
    ) -> SpatiallyConstrainedGMM {
        SpatiallyConstrainedGMMBuilder::new(n_components)
            .spatial_constraint(SpatialConstraint::Grid {
                rows: image_dims.0,
                cols: image_dims.1,
            })
            .spatial_weight(0.2)
            .covariance_type(CovarianceType::Diagonal)
            .max_iter(100)
            .tolerance(1e-4)
            .build()
    }

    /// Create a geographic mixture model for environmental data
    pub fn environmental_geographic_gmm(n_components: usize) -> GeographicMixture {
        GeographicMixtureBuilder::new(n_components)
            .use_elevation(true)
            .use_distance_features(false)
            .max_iter(150)
            .tolerance(1e-5)
            .build()
    }

    /// Create a Markov Random Field mixture for spatial clustering
    pub fn spatial_clustering_mrf(n_components: usize) -> MarkovRandomFieldMixture {
        MarkovRandomFieldMixtureBuilder::new(n_components)
            .interaction_strength(1.0)
            .neighborhood_size(8)
            .covariance_type(CovarianceType::Spherical)
            .max_iter(50)
            .tolerance(1e-3)
            .build()
    }

    /// Create a distance-based spatially constrained GMM for point clustering
    pub fn distance_constrained_gmm(n_components: usize, radius: f64) -> SpatiallyConstrainedGMM {
        SpatiallyConstrainedGMMBuilder::new(n_components)
            .spatial_constraint(SpatialConstraint::Distance { radius })
            .spatial_weight(0.15)
            .covariance_type(CovarianceType::Full)
            .max_iter(200)
            .tolerance(1e-6)
            .build()
    }
}

/// Utilities for spatial data preprocessing and validation
pub mod utils {
    use super::*;
    use scirs2_core::ndarray::Array2;
    use sklears_core::error::{Result as SklResult, SklearsError};

    /// Validate spatial coordinates for consistency
    pub fn validate_coordinates(coords: &Array2<f64>) -> SklResult<()> {
        if coords.ncols() < 2 {
            return Err(SklearsError::InvalidInput(
                "Coordinates must have at least 2 dimensions".to_string(),
            ));
        }

        if coords.nrows() == 0 {
            return Err(SklearsError::InvalidInput(
                "Coordinates cannot be empty".to_string(),
            ));
        }

        // Check for NaN or infinite values
        for coord in coords.iter() {
            if !coord.is_finite() {
                return Err(SklearsError::InvalidInput(
                    "Coordinates must contain finite values".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Normalize spatial coordinates to unit scale
    pub fn normalize_coordinates(coords: &Array2<f64>) -> SklResult<Array2<f64>> {
        validate_coordinates(coords)?;

        let mut normalized = coords.clone();

        for j in 0..coords.ncols() {
            let col = coords.column(j);
            let min_val = col.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_val = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            let range = max_val - min_val;
            if range > 1e-10 {
                for i in 0..coords.nrows() {
                    normalized[[i, j]] = (coords[[i, j]] - min_val) / range;
                }
            }
        }

        Ok(normalized)
    }

    /// Compute spatial extent (bounding box) of coordinates
    pub fn spatial_extent(coords: &Array2<f64>) -> SklResult<Vec<(f64, f64)>> {
        validate_coordinates(coords)?;

        let mut extents = Vec::new();

        for j in 0..coords.ncols() {
            let col = coords.column(j);
            let min_val = col.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_val = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            extents.push((min_val, max_val));
        }

        Ok(extents)
    }

    /// Check if spatial configuration is valid for the given constraint
    pub fn validate_spatial_config(
        coords: &Array2<f64>,
        constraint: &SpatialConstraint,
    ) -> SklResult<()> {
        validate_coordinates(coords)?;

        match constraint {
            SpatialConstraint::Grid { rows, cols } => {
                if coords.nrows() != rows * cols {
                    return Err(SklearsError::InvalidInput(format!(
                        "Grid constraint requires {} samples but {} provided",
                        rows * cols,
                        coords.nrows()
                    )));
                }
            }
            SpatialConstraint::Distance { radius } => {
                if *radius <= 0.0 {
                    return Err(SklearsError::InvalidInput(
                        "Distance radius must be positive".to_string(),
                    ));
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Estimate optimal spatial weight based on data characteristics
    pub fn estimate_spatial_weight(coords: &Array2<f64>, data: &Array2<f64>) -> SklResult<f64> {
        validate_coordinates(coords)?;

        if coords.nrows() != data.nrows() {
            return Err(SklearsError::InvalidInput(
                "Coordinates and data must have same number of samples".to_string(),
            ));
        }

        // Simple heuristic: balance between spatial and data variance
        let spatial_var = {
            let distances = pairwise_distances(coords);
            let n = distances.nrows();
            let mut sum = 0.0;
            let mut count = 0;
            for i in 0..n {
                for j in (i + 1)..n {
                    sum += distances[[i, j]].powi(2);
                    count += 1;
                }
            }
            if count > 0 {
                sum / count as f64
            } else {
                1.0
            }
        };

        let data_var = {
            let mean = data.mean().unwrap_or(0.0);
            data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64
        };

        // Weight should be inversely related to the ratio of variances
        let ratio = spatial_var / (data_var + 1e-8);
        let weight = 1.0 / (1.0 + ratio);

        // Clamp to reasonable range
        Ok(weight.max(0.01).min(0.5))
    }
}

/// Common error types for spatial mixture models
pub mod errors {
    use sklears_core::error::SklearsError;

    /// Create error for invalid spatial constraint
    pub fn invalid_spatial_constraint(msg: &str) -> SklearsError {
        SklearsError::InvalidInput(format!("Invalid spatial constraint: {}", msg))
    }

    /// Create error for spatial dimension mismatch
    pub fn spatial_dimension_mismatch(expected: usize, actual: usize) -> SklearsError {
        SklearsError::InvalidInput(format!(
            "Spatial dimension mismatch: expected {}, got {}",
            expected, actual
        ))
    }

    /// Create error for insufficient spatial data
    pub fn insufficient_spatial_data(min_required: usize, actual: usize) -> SklearsError {
        SklearsError::InvalidInput(format!(
            "Insufficient spatial data: need at least {} samples, got {}",
            min_required, actual
        ))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod integration_tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use sklears_core::traits::Fit;

    #[test]
    fn test_spatial_module_integration() {
        // Test that all main components work together
        let coords = array![[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [6.0, 6.0]];
        let X = array![[1.0, 2.0], [1.1, 2.1], [5.0, 6.0], [5.1, 6.1]];

        // Test spatially constrained GMM
        let gmm = presets::distance_constrained_gmm(2, 2.0).with_coordinates(coords.clone());
        assert!(gmm.fit(&X, &()).is_ok());

        // Test spatial autocorrelation analysis
        let analyzer = SpatialAutocorrelationAnalyzer::new(
            coords.clone(),
            SpatialConstraint::Distance { radius: 2.0 },
        )
        .unwrap();

        let values = array![1.0, 1.0, 0.0, 0.0];
        assert!(analyzer.morans_i(&values).is_ok());
        assert!(analyzer.gearys_c(&values).is_ok());

        // Test utility functions
        assert!(utils::validate_coordinates(&coords).is_ok());
        assert!(utils::normalize_coordinates(&coords).is_ok());
        assert!(utils::spatial_extent(&coords).is_ok());
    }

    #[test]
    fn test_preset_configurations() {
        // Test image segmentation preset
        let image_gmm = presets::image_segmentation_gmm(3, (10, 10));
        assert_eq!(image_gmm.get_config().n_components, 3);

        // Test environmental preset
        let env_gmm = presets::environmental_geographic_gmm(4);
        assert_eq!(env_gmm.n_components, 4);

        // Test spatial clustering preset
        let cluster_mrf = presets::spatial_clustering_mrf(5);
        assert_eq!(cluster_mrf.n_components, 5);
    }

    #[test]
    fn test_utility_functions() {
        let coords = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let data = array![[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]];

        // Test coordinate validation
        assert!(utils::validate_coordinates(&coords).is_ok());

        // Test normalization
        let normalized = utils::normalize_coordinates(&coords).unwrap();
        assert_eq!(normalized.shape(), coords.shape());

        // Test spatial extent
        let extent = utils::spatial_extent(&coords).unwrap();
        assert_eq!(extent.len(), 2); // 2D coordinates

        // Test spatial weight estimation
        let weight = utils::estimate_spatial_weight(&coords, &data).unwrap();
        assert!(weight > 0.0 && weight <= 1.0);
    }
}
