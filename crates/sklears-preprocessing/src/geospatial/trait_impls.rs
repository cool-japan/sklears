//! # CoordinateTransformer - Trait Implementations
//!
//! This module contains trait implementations for `CoordinateTransformer`.
//!
//! ## Implemented Traits
//!
//! - `Estimator`
//! - `Fit`
//! - `Transform`
//! - `Estimator`
//! - `Fit`
//! - `Default`
//! - `Transform`
//! - `Estimator`
//! - `Fit`
//! - `Transform`
//! - `Estimator`
//! - `Fit`
//! - `Default`
//! - `Transform`
//! - `Estimator`
//! - `Fit`
//! - `Default`
//! - `Transform`
//! - `Estimator`
//! - `Fit`
//! - `Default`
//! - `Transform`
//! - `Estimator`
//! - `Fit`
//! - `Transform`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::seeded_rng;
use sklears_core::prelude::*;
use std::collections::HashMap;

use super::functions::{
    calculate_distance, utm_to_wgs84, web_mercator_to_wgs84, wgs84_to_utm, wgs84_to_web_mercator,
};
use super::types::{
    Coordinate, CoordinateSystem, CoordinateTransformer, CoordinateTransformerConfig,
    CoordinateTransformerFitted, Geohash, GeohashEncoder, GeohashEncoderConfig,
    GeohashEncoderFitted, ProximityFeatures, ProximityFeaturesConfig, ProximityFeaturesFitted,
    SpatialAutocorrelation, SpatialAutocorrelationConfig, SpatialAutocorrelationFitted,
    SpatialBinning, SpatialBinningConfig, SpatialBinningFitted, SpatialClustering,
    SpatialClusteringConfig, SpatialClusteringFitted, SpatialClusteringMethod,
    SpatialDistanceFeatures, SpatialDistanceFeaturesConfig, SpatialDistanceFeaturesFitted,
    SpatialDistanceMetric,
};

impl Estimator for CoordinateTransformer {
    type Config = CoordinateTransformerConfig;
    type Error = SklearsError;
    type Float = f64;
    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<f64>, ()> for CoordinateTransformer {
    type Fitted = CoordinateTransformerFitted;
    #[allow(non_snake_case)]
    fn fit(self, _X: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        Ok(CoordinateTransformerFitted {
            config: self.config,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for CoordinateTransformerFitted {
    #[allow(non_snake_case)]
    fn transform(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        if X.ncols() != 2 {
            return Err(SklearsError::InvalidInput(
                "Input must have exactly 2 columns (lat, lon)".to_string(),
            ));
        }
        let nrows = X.nrows();
        let mut result = Array2::zeros((nrows, 2));
        for i in 0..nrows {
            let lat = X[[i, 0]];
            let lon = X[[i, 1]];
            let transformed = match (&self.config.from, &self.config.to) {
                (CoordinateSystem::WGS84, CoordinateSystem::WebMercator) => {
                    wgs84_to_web_mercator(lat, lon)?
                }
                (CoordinateSystem::WebMercator, CoordinateSystem::WGS84) => {
                    web_mercator_to_wgs84(lat, lon)?
                }
                (CoordinateSystem::WGS84, CoordinateSystem::UTM { zone, northern }) => {
                    wgs84_to_utm(lat, lon, *zone, *northern)?
                }
                (CoordinateSystem::UTM { zone, northern }, CoordinateSystem::WGS84) => {
                    utm_to_wgs84(lat, lon, *zone, *northern)?
                }
                _ => {
                    return Err(SklearsError::InvalidInput(format!(
                        "Unsupported coordinate transformation: {:?} to {:?}",
                        self.config.from, self.config.to
                    )));
                }
            };
            result[[i, 0]] = transformed.0;
            result[[i, 1]] = transformed.1;
        }
        Ok(result)
    }
}

impl Estimator for GeohashEncoder {
    type Config = GeohashEncoderConfig;
    type Error = SklearsError;
    type Float = f64;
    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<f64>, ()> for GeohashEncoder {
    type Fitted = GeohashEncoderFitted;
    #[allow(non_snake_case)]
    fn fit(self, X: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        if X.ncols() != 2 {
            return Err(SklearsError::InvalidInput(
                "Input must have exactly 2 columns (lat, lon)".to_string(),
            ));
        }
        let mut vocabulary = HashMap::new();
        for i in 0..X.nrows() {
            let coord = Coordinate::new(X[[i, 0]], X[[i, 1]])?;
            let geohash = Geohash::encode(&coord, self.config.precision)?;
            if !vocabulary.contains_key(&geohash) {
                let idx = vocabulary.len();
                vocabulary.insert(geohash.clone(), idx);
            }
            if self.config.include_neighbors {
                for neighbor in Geohash::neighbors(&geohash)? {
                    if !vocabulary.contains_key(&neighbor) {
                        let idx = vocabulary.len();
                        vocabulary.insert(neighbor, idx);
                    }
                }
            }
        }
        Ok(GeohashEncoderFitted {
            config: self.config,
            vocabulary,
        })
    }
}

impl Default for GeohashEncoderConfig {
    fn default() -> Self {
        Self {
            precision: 7,
            include_neighbors: false,
        }
    }
}

impl Transform<Array2<f64>, Array2<f64>> for GeohashEncoderFitted {
    #[allow(non_snake_case)]
    fn transform(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        if X.ncols() != 2 {
            return Err(SklearsError::InvalidInput(
                "Input must have exactly 2 columns (lat, lon)".to_string(),
            ));
        }
        let nrows = X.nrows();
        let ncols = if self.config.include_neighbors {
            self.vocabulary.len() * 9
        } else {
            self.vocabulary.len()
        };
        let mut result = Array2::zeros((nrows, ncols));
        for i in 0..nrows {
            let coord = Coordinate::new(X[[i, 0]], X[[i, 1]])?;
            let geohash = Geohash::encode(&coord, self.config.precision)?;
            if let Some(&idx) = self.vocabulary.get(&geohash) {
                result[[i, idx]] = 1.0;
            }
            if self.config.include_neighbors {
                for neighbor in Geohash::neighbors(&geohash)? {
                    if let Some(&idx) = self.vocabulary.get(&neighbor) {
                        result[[i, idx]] = 1.0;
                    }
                }
            }
        }
        Ok(result)
    }
}

impl Estimator for ProximityFeatures {
    type Config = ProximityFeaturesConfig;
    type Error = SklearsError;
    type Float = f64;
    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<f64>, ()> for ProximityFeatures {
    type Fitted = ProximityFeaturesFitted;
    #[allow(non_snake_case)]
    fn fit(self, X: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        if X.ncols() != 2 {
            return Err(SklearsError::InvalidInput(
                "Input must have exactly 2 columns (lat, lon)".to_string(),
            ));
        }
        Ok(ProximityFeaturesFitted {
            config: self.config,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for ProximityFeaturesFitted {
    #[allow(non_snake_case)]
    fn transform(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        if X.ncols() != 2 {
            return Err(SklearsError::InvalidInput(
                "Input must have exactly 2 columns (lat, lon)".to_string(),
            ));
        }
        let nrows = X.nrows();
        let n_pois = self.config.points_of_interest.len();
        let n_features = if self.config.include_indicators {
            n_pois * 2
        } else {
            n_pois
        };
        let mut result = Array2::zeros((nrows, n_features));
        for i in 0..nrows {
            let coord = Coordinate::new(X[[i, 0]], X[[i, 1]])?;
            for (j, (_name, poi)) in self.config.points_of_interest.iter().enumerate() {
                let distance = calculate_distance(&coord, poi, self.config.metric)?;
                result[[i, j]] = distance;
                if self.config.include_indicators {
                    let is_within = if let Some(max_dist) = self.config.max_distance {
                        if distance <= max_dist {
                            1.0
                        } else {
                            0.0
                        }
                    } else {
                        1.0
                    };
                    result[[i, n_pois + j]] = is_within;
                }
            }
        }
        Ok(result)
    }
}

impl Estimator for SpatialAutocorrelation {
    type Config = SpatialAutocorrelationConfig;
    type Error = SklearsError;
    type Float = f64;
    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<f64>, Array1<f64>> for SpatialAutocorrelation {
    type Fitted = SpatialAutocorrelationFitted;
    #[allow(non_snake_case)]
    fn fit(self, X: &Array2<f64>, _y: &Array1<f64>) -> Result<Self::Fitted> {
        if X.ncols() != 2 {
            return Err(SklearsError::InvalidInput(
                "Input must have exactly 2 columns (lat, lon)".to_string(),
            ));
        }
        let mut training_coords = Vec::with_capacity(X.nrows());
        for i in 0..X.nrows() {
            training_coords.push(Coordinate::new(X[[i, 0]], X[[i, 1]])?);
        }
        Ok(SpatialAutocorrelationFitted {
            config: self.config,
            training_coords,
        })
    }
}

impl Default for SpatialAutocorrelationConfig {
    fn default() -> Self {
        Self {
            k_neighbors: 5,
            metric: SpatialDistanceMetric::Haversine,
            include_morans_i: true,
            include_lisa: true,
        }
    }
}

impl Transform<Array2<f64>, Array2<f64>> for SpatialAutocorrelationFitted {
    #[allow(non_snake_case)]
    fn transform(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        if X.ncols() != 2 {
            return Err(SklearsError::InvalidInput(
                "Input must have exactly 2 columns (lat, lon)".to_string(),
            ));
        }
        let nrows = X.nrows();
        let mut n_features = 0;
        if self.config.include_lisa {
            n_features += 1;
        }
        let mut result = Array2::zeros((nrows, n_features.max(1)));
        for i in 0..nrows {
            let coord = Coordinate::new(X[[i, 0]], X[[i, 1]])?;
            let mut distances: Vec<(usize, f64)> = self
                .training_coords
                .iter()
                .enumerate()
                .map(|(j, other)| {
                    let dist = calculate_distance(&coord, other, self.config.metric)
                        .unwrap_or(f64::INFINITY);
                    (j, dist)
                })
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("operation should succeed"));
            if self.config.include_lisa {
                let nearest_neighbors =
                    &distances[1..=self.config.k_neighbors.min(distances.len() - 1)];
                let avg_neighbor_dist = nearest_neighbors.iter().map(|(_, d)| d).sum::<f64>()
                    / nearest_neighbors.len() as f64;
                result[[i, 0]] = 1.0 / (1.0 + avg_neighbor_dist);
            }
        }
        Ok(result)
    }
}

impl Estimator for SpatialBinning {
    type Config = SpatialBinningConfig;
    type Error = SklearsError;
    type Float = f64;
    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<f64>, ()> for SpatialBinning {
    type Fitted = SpatialBinningFitted;
    #[allow(non_snake_case)]
    fn fit(self, X: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        if X.ncols() != 2 {
            return Err(SklearsError::InvalidInput(
                "Input must have exactly 2 columns (lat, lon)".to_string(),
            ));
        }
        let lat_range = if let Some(range) = self.config.lat_range {
            range
        } else {
            let lats = X.column(0);
            let min_lat = lats.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_lat = lats.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (min_lat, max_lat)
        };
        let lon_range = if let Some(range) = self.config.lon_range {
            range
        } else {
            let lons = X.column(1);
            let min_lon = lons.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_lon = lons.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (min_lon, max_lon)
        };
        Ok(SpatialBinningFitted {
            config: self.config,
            lat_range,
            lon_range,
        })
    }
}

impl Default for SpatialBinningConfig {
    fn default() -> Self {
        Self {
            n_lat_bins: 10,
            n_lon_bins: 10,
            lat_range: None,
            lon_range: None,
            one_hot: false,
        }
    }
}

impl Transform<Array2<f64>, Array2<f64>> for SpatialBinningFitted {
    #[allow(non_snake_case)]
    fn transform(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        if X.ncols() != 2 {
            return Err(SklearsError::InvalidInput(
                "Input must have exactly 2 columns (lat, lon)".to_string(),
            ));
        }
        let nrows = X.nrows();
        let ncols = if self.config.one_hot {
            self.config.n_lat_bins * self.config.n_lon_bins
        } else {
            2
        };
        let mut result = Array2::zeros((nrows, ncols));
        let lat_bin_size = (self.lat_range.1 - self.lat_range.0) / self.config.n_lat_bins as f64;
        let lon_bin_size = (self.lon_range.1 - self.lon_range.0) / self.config.n_lon_bins as f64;
        for i in 0..nrows {
            let lat = X[[i, 0]];
            let lon = X[[i, 1]];
            let lat_bin = ((lat - self.lat_range.0) / lat_bin_size)
                .floor()
                .clamp(0.0, (self.config.n_lat_bins - 1) as f64) as usize;
            let lon_bin = ((lon - self.lon_range.0) / lon_bin_size)
                .floor()
                .clamp(0.0, (self.config.n_lon_bins - 1) as f64) as usize;
            if self.config.one_hot {
                let bin_idx = lat_bin * self.config.n_lon_bins + lon_bin;
                result[[i, bin_idx]] = 1.0;
            } else {
                result[[i, 0]] = lat_bin as f64;
                result[[i, 1]] = lon_bin as f64;
            }
        }
        Ok(result)
    }
}

impl Estimator for SpatialClustering {
    type Config = SpatialClusteringConfig;
    type Error = SklearsError;
    type Float = f64;
    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<f64>, ()> for SpatialClustering {
    type Fitted = SpatialClusteringFitted;
    #[allow(non_snake_case)]
    fn fit(self, X: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        if X.ncols() != 2 {
            return Err(SklearsError::InvalidInput(
                "Input must have exactly 2 columns (lat, lon)".to_string(),
            ));
        }
        let cluster_centers = match self.config.method {
            SpatialClusteringMethod::Grid => {
                let lats = X.column(0);
                let lons = X.column(1);
                let lat_min = lats.iter().cloned().fold(f64::INFINITY, f64::min);
                let lat_max = lats.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let lon_min = lons.iter().cloned().fold(f64::INFINITY, f64::min);
                let lon_max = lons.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let grid_size = (self.config.n_clusters as f64).sqrt().ceil() as usize;
                let lat_step = (lat_max - lat_min) / grid_size as f64;
                let lon_step = (lon_max - lon_min) / grid_size as f64;
                let mut centers = Vec::new();
                for i in 0..grid_size {
                    for j in 0..grid_size {
                        let lat = lat_min + (i as f64 + 0.5) * lat_step;
                        let lon = lon_min + (j as f64 + 0.5) * lon_step;
                        if let Ok(coord) = Coordinate::new(lat, lon) {
                            centers.push(coord);
                        }
                    }
                }
                centers
            }
            SpatialClusteringMethod::KMeans => {
                let mut rng = seeded_rng(42);
                let nrows = X.nrows();
                let mut centers = Vec::new();
                for _ in 0..self.config.n_clusters.min(nrows) {
                    let idx = rng.random_range(0..nrows);
                    if let Ok(coord) = Coordinate::new(X[[idx, 0]], X[[idx, 1]]) {
                        centers.push(coord);
                    }
                }
                for _ in 0..10 {
                    let mut cluster_sums = vec![(0.0, 0.0); centers.len()];
                    let mut cluster_counts = vec![0; centers.len()];
                    for i in 0..nrows {
                        let coord = Coordinate::new(X[[i, 0]], X[[i, 1]])?;
                        let mut min_dist = f64::INFINITY;
                        let mut min_idx = 0;
                        for (j, center) in centers.iter().enumerate() {
                            let dist = calculate_distance(&coord, center, self.config.metric)?;
                            if dist < min_dist {
                                min_dist = dist;
                                min_idx = j;
                            }
                        }
                        cluster_sums[min_idx].0 += coord.lat;
                        cluster_sums[min_idx].1 += coord.lon;
                        cluster_counts[min_idx] += 1;
                    }
                    for (j, center) in centers.iter_mut().enumerate() {
                        if cluster_counts[j] > 0 {
                            let new_lat = cluster_sums[j].0 / cluster_counts[j] as f64;
                            let new_lon = cluster_sums[j].1 / cluster_counts[j] as f64;
                            if let Ok(new_coord) = Coordinate::new(new_lat, new_lon) {
                                *center = new_coord;
                            }
                        }
                    }
                }
                centers
            }
            SpatialClusteringMethod::Density | SpatialClusteringMethod::Hierarchical => {
                let mut rng = seeded_rng(42);
                let nrows = X.nrows();
                let mut centers = Vec::new();
                let sample_size = self.config.n_clusters.min(nrows);
                for _ in 0..sample_size {
                    let idx = rng.random_range(0..nrows);
                    if let Ok(coord) = Coordinate::new(X[[idx, 0]], X[[idx, 1]]) {
                        centers.push(coord);
                    }
                }
                centers
            }
        };
        let grid_bounds = if matches!(self.config.method, SpatialClusteringMethod::Grid) {
            let lats = X.column(0);
            let lons = X.column(1);
            let lat_min = lats.iter().cloned().fold(f64::INFINITY, f64::min);
            let lat_max = lats.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lon_min = lons.iter().cloned().fold(f64::INFINITY, f64::min);
            let lon_max = lons.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            Some((lat_min, lon_min, lat_max, lon_max))
        } else {
            None
        };
        Ok(SpatialClusteringFitted {
            config: self.config,
            cluster_centers,
            grid_bounds,
        })
    }
}

impl Default for SpatialClusteringConfig {
    fn default() -> Self {
        Self {
            method: SpatialClusteringMethod::Grid,
            n_clusters: 10,
            epsilon: Some(5.0),
            min_points: Some(5),
            metric: SpatialDistanceMetric::Haversine,
        }
    }
}

impl Transform<Array2<f64>, Array2<f64>> for SpatialClusteringFitted {
    #[allow(non_snake_case)]
    fn transform(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        if X.ncols() != 2 {
            return Err(SklearsError::InvalidInput(
                "Input must have exactly 2 columns (lat, lon)".to_string(),
            ));
        }
        let nrows = X.nrows();
        match self.config.method {
            SpatialClusteringMethod::Grid | SpatialClusteringMethod::KMeans => {
                let mut result = Array2::zeros((nrows, 2));
                for i in 0..nrows {
                    let coord = Coordinate::new(X[[i, 0]], X[[i, 1]])?;
                    let mut min_dist = f64::INFINITY;
                    let mut min_idx = 0;
                    for (j, center) in self.cluster_centers.iter().enumerate() {
                        let dist = calculate_distance(&coord, center, self.config.metric)?;
                        if dist < min_dist {
                            min_dist = dist;
                            min_idx = j;
                        }
                    }
                    result[[i, 0]] = min_idx as f64;
                    result[[i, 1]] = min_dist;
                }
                Ok(result)
            }
            SpatialClusteringMethod::Density => {
                let epsilon = self.config.epsilon.unwrap_or(5.0);
                let mut result = Array2::zeros((nrows, 1));
                for i in 0..nrows {
                    let coord = Coordinate::new(X[[i, 0]], X[[i, 1]])?;
                    let mut density = 0.0;
                    for j in 0..nrows {
                        if i != j {
                            let other = Coordinate::new(X[[j, 0]], X[[j, 1]])?;
                            let dist = calculate_distance(&coord, &other, self.config.metric)?;
                            if dist <= epsilon {
                                density += 1.0;
                            }
                        }
                    }
                    result[[i, 0]] = density;
                }
                Ok(result)
            }
            SpatialClusteringMethod::Hierarchical => {
                let n_centers = self.cluster_centers.len();
                let mut result = Array2::zeros((nrows, n_centers));
                for i in 0..nrows {
                    let coord = Coordinate::new(X[[i, 0]], X[[i, 1]])?;
                    for (j, center) in self.cluster_centers.iter().enumerate() {
                        let dist = calculate_distance(&coord, center, self.config.metric)?;
                        result[[i, j]] = dist;
                    }
                }
                Ok(result)
            }
        }
    }
}

impl Estimator for SpatialDistanceFeatures {
    type Config = SpatialDistanceFeaturesConfig;
    type Error = SklearsError;
    type Float = f64;
    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<f64>, ()> for SpatialDistanceFeatures {
    type Fitted = SpatialDistanceFeaturesFitted;
    #[allow(non_snake_case)]
    fn fit(self, X: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        if X.ncols() != 2 {
            return Err(SklearsError::InvalidInput(
                "Input must have exactly 2 columns (lat, lon)".to_string(),
            ));
        }
        Ok(SpatialDistanceFeaturesFitted {
            config: self.config,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for SpatialDistanceFeaturesFitted {
    #[allow(non_snake_case)]
    fn transform(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        if X.ncols() != 2 {
            return Err(SklearsError::InvalidInput(
                "Input must have exactly 2 columns (lat, lon)".to_string(),
            ));
        }
        let nrows = X.nrows();
        let n_refs = self.config.reference_points.len();
        let mut n_features = n_refs;
        if self.config.include_inverse {
            n_features += n_refs;
        }
        if self.config.include_squared {
            n_features += n_refs;
        }
        let mut result = Array2::zeros((nrows, n_features));
        for i in 0..nrows {
            let coord = Coordinate::new(X[[i, 0]], X[[i, 1]])?;
            for (j, ref_point) in self.config.reference_points.iter().enumerate() {
                let distance = calculate_distance(&coord, ref_point, self.config.metric)?;
                result[[i, j]] = distance;
                if self.config.include_inverse {
                    result[[i, n_refs + j]] = if distance > 0.0 { 1.0 / distance } else { 0.0 };
                }
                if self.config.include_squared {
                    let offset = if self.config.include_inverse {
                        n_refs * 2
                    } else {
                        n_refs
                    };
                    result[[i, offset + j]] = distance * distance;
                }
            }
        }
        Ok(result)
    }
}
