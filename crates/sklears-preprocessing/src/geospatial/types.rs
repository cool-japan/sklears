//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use sklears_core::prelude::*;
use std::collections::HashMap;

use super::functions::BASE32_CHARS;

/// Spatial autocorrelation feature extractor
pub struct SpatialAutocorrelation {
    pub(super) config: SpatialAutocorrelationConfig,
}
impl SpatialAutocorrelation {
    /// Create a new spatial autocorrelation feature extractor
    pub fn new(config: SpatialAutocorrelationConfig) -> Self {
        Self { config }
    }
}
/// Geohash encoder/decoder for spatial indexing
pub struct Geohash;
impl Geohash {
    /// Encode a coordinate into a geohash string of specified precision
    ///
    /// # Arguments
    /// * `coord` - The coordinate to encode
    /// * `precision` - The number of characters in the geohash (typically 1-12)
    pub fn encode(coord: &Coordinate, precision: usize) -> Result<String> {
        if precision == 0 || precision > 12 {
            return Err(SklearsError::InvalidInput(
                "Geohash precision must be between 1 and 12".to_string(),
            ));
        }
        let mut lat_min = -90.0;
        let mut lat_max = 90.0;
        let mut lon_min = -180.0;
        let mut lon_max = 180.0;
        let mut geohash = String::with_capacity(precision);
        let mut bit = 0;
        let mut ch = 0u8;
        let mut is_lon = true;
        while geohash.len() < precision {
            if is_lon {
                let mid = (lon_min + lon_max) / 2.0;
                if coord.lon > mid {
                    ch |= 1 << (4 - bit);
                    lon_min = mid;
                } else {
                    lon_max = mid;
                }
            } else {
                let mid = (lat_min + lat_max) / 2.0;
                if coord.lat > mid {
                    ch |= 1 << (4 - bit);
                    lat_min = mid;
                } else {
                    lat_max = mid;
                }
            }
            is_lon = !is_lon;
            bit += 1;
            if bit == 5 {
                geohash.push(BASE32_CHARS[ch as usize] as char);
                bit = 0;
                ch = 0;
            }
        }
        Ok(geohash)
    }
    /// Decode a geohash string into a coordinate (center of the geohash box)
    pub fn decode(geohash: &str) -> Result<Coordinate> {
        if geohash.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Geohash string cannot be empty".to_string(),
            ));
        }
        let mut char_map = HashMap::new();
        for (i, &c) in BASE32_CHARS.iter().enumerate() {
            char_map.insert(c as char, i as u8);
        }
        let mut lat_min = -90.0;
        let mut lat_max = 90.0;
        let mut lon_min = -180.0;
        let mut lon_max = 180.0;
        let mut is_lon = true;
        for ch in geohash.chars() {
            let bits = char_map.get(&ch).ok_or_else(|| {
                SklearsError::InvalidInput(format!("Invalid geohash character: {}", ch))
            })?;
            for i in (0..5).rev() {
                if is_lon {
                    let mid = (lon_min + lon_max) / 2.0;
                    if (bits >> i) & 1 == 1 {
                        lon_min = mid;
                    } else {
                        lon_max = mid;
                    }
                } else {
                    let mid = (lat_min + lat_max) / 2.0;
                    if (bits >> i) & 1 == 1 {
                        lat_min = mid;
                    } else {
                        lat_max = mid;
                    }
                }
                is_lon = !is_lon;
            }
        }
        let lat = (lat_min + lat_max) / 2.0;
        let lon = (lon_min + lon_max) / 2.0;
        Coordinate::new(lat, lon)
    }
    /// Get the bounding box (lat_min, lon_min, lat_max, lon_max) for a geohash
    pub fn bounds(geohash: &str) -> Result<(f64, f64, f64, f64)> {
        if geohash.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Geohash string cannot be empty".to_string(),
            ));
        }
        let mut char_map = HashMap::new();
        for (i, &c) in BASE32_CHARS.iter().enumerate() {
            char_map.insert(c as char, i as u8);
        }
        let mut lat_min = -90.0;
        let mut lat_max = 90.0;
        let mut lon_min = -180.0;
        let mut lon_max = 180.0;
        let mut is_lon = true;
        for ch in geohash.chars() {
            let bits = char_map.get(&ch).ok_or_else(|| {
                SklearsError::InvalidInput(format!("Invalid geohash character: {}", ch))
            })?;
            for i in (0..5).rev() {
                if is_lon {
                    let mid = (lon_min + lon_max) / 2.0;
                    if (bits >> i) & 1 == 1 {
                        lon_min = mid;
                    } else {
                        lon_max = mid;
                    }
                } else {
                    let mid = (lat_min + lat_max) / 2.0;
                    if (bits >> i) & 1 == 1 {
                        lat_min = mid;
                    } else {
                        lat_max = mid;
                    }
                }
                is_lon = !is_lon;
            }
        }
        Ok((lat_min, lon_min, lat_max, lon_max))
    }
    /// Get all neighbors of a geohash
    pub fn neighbors(geohash: &str) -> Result<Vec<String>> {
        let coord = Self::decode(geohash)?;
        let (lat_min, lon_min, lat_max, lon_max) = Self::bounds(geohash)?;
        let lat_diff = lat_max - lat_min;
        let lon_diff = lon_max - lon_min;
        let precision = geohash.len();
        let mut neighbors = Vec::with_capacity(8);
        let offsets = [
            (-1.0, -1.0),
            (-1.0, 0.0),
            (-1.0, 1.0),
            (0.0, -1.0),
            (0.0, 1.0),
            (1.0, -1.0),
            (1.0, 0.0),
            (1.0, 1.0),
        ];
        for (dlat, dlon) in offsets.iter() {
            let new_lat = (coord.lat + dlat * lat_diff).clamp(-90.0, 90.0);
            let new_lon = coord.lon + dlon * lon_diff;
            let new_lon = if new_lon > 180.0 {
                new_lon - 360.0
            } else if new_lon < -180.0 {
                new_lon + 360.0
            } else {
                new_lon
            };
            if let Ok(neighbor_coord) = Coordinate::new(new_lat, new_lon) {
                if let Ok(neighbor_hash) = Self::encode(&neighbor_coord, precision) {
                    if neighbor_hash != geohash {
                        neighbors.push(neighbor_hash);
                    }
                }
            }
        }
        Ok(neighbors)
    }
}
/// Coordinate system transformer (Fitted state)
pub struct CoordinateTransformerFitted {
    pub(super) config: CoordinateTransformerConfig,
}
/// Configuration for geohash encoding
#[derive(Debug, Clone)]
pub struct GeohashEncoderConfig {
    /// Geohash precision (number of characters, 1-12)
    pub precision: usize,
    /// Whether to include neighbor geohashes as features
    pub include_neighbors: bool,
}
/// Fitted geohash encoder
pub struct GeohashEncoderFitted {
    pub(super) config: GeohashEncoderConfig,
    /// Vocabulary of unique geohashes
    pub(super) vocabulary: HashMap<String, usize>,
}
/// Spatial distance metric
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialDistanceMetric {
    /// Haversine formula - great circle distance (fastest, good approximation)
    Haversine,
    /// Vincenty formula - more accurate for ellipsoid (slower, higher precision)
    Vincenty,
    /// Euclidean distance in projected coordinates
    Euclidean,
    /// Manhattan distance in projected coordinates
    Manhattan,
}
/// Proximity feature extractor
pub struct ProximityFeatures {
    pub(super) config: ProximityFeaturesConfig,
}
impl ProximityFeatures {
    /// Create a new proximity feature extractor
    pub fn new(config: ProximityFeaturesConfig) -> Self {
        Self { config }
    }
}
/// Fitted spatial distance feature extractor
pub struct SpatialDistanceFeaturesFitted {
    pub(super) config: SpatialDistanceFeaturesConfig,
}
/// Fitted proximity feature extractor
pub struct ProximityFeaturesFitted {
    pub(super) config: ProximityFeaturesConfig,
}
/// Coordinate system transformer (Untrained state)
pub struct CoordinateTransformer {
    pub(super) config: CoordinateTransformerConfig,
}
impl CoordinateTransformer {
    /// Create a new coordinate transformer
    pub fn new(from: CoordinateSystem, to: CoordinateSystem) -> Self {
        Self {
            config: CoordinateTransformerConfig { from, to },
        }
    }
}
/// Spatial binning transformer
pub struct SpatialBinning {
    pub(super) config: SpatialBinningConfig,
}
impl SpatialBinning {
    /// Create a new spatial binning transformer
    pub fn new(config: SpatialBinningConfig) -> Self {
        Self { config }
    }
}
/// Fitted spatial binning transformer
pub struct SpatialBinningFitted {
    pub(super) config: SpatialBinningConfig,
    pub(super) lat_range: (f64, f64),
    pub(super) lon_range: (f64, f64),
}
/// Configuration for spatial clustering features
#[derive(Debug, Clone)]
pub struct SpatialClusteringConfig {
    /// Clustering method
    pub method: SpatialClusteringMethod,
    /// Number of clusters (for grid-based methods)
    pub n_clusters: usize,
    /// Epsilon for density-based methods (in km)
    pub epsilon: Option<f64>,
    /// Minimum points for density-based methods
    pub min_points: Option<usize>,
    /// Distance metric
    pub metric: SpatialDistanceMetric,
}
/// Fitted spatial autocorrelation feature extractor
pub struct SpatialAutocorrelationFitted {
    pub(super) config: SpatialAutocorrelationConfig,
    /// Training data coordinates for spatial weights
    pub(super) training_coords: Vec<Coordinate>,
}
/// Configuration for spatial binning
#[derive(Debug, Clone)]
pub struct SpatialBinningConfig {
    /// Number of latitude bins
    pub n_lat_bins: usize,
    /// Number of longitude bins
    pub n_lon_bins: usize,
    /// Latitude range (min, max)
    pub lat_range: Option<(f64, f64)>,
    /// Longitude range (min, max)
    pub lon_range: Option<(f64, f64)>,
    /// Whether to use one-hot encoding for bins
    pub one_hot: bool,
}
/// A geographic coordinate in WGS84 system
#[derive(Debug, Clone, Copy)]
pub struct Coordinate {
    /// Latitude in degrees (-90 to 90)
    pub lat: f64,
    /// Longitude in degrees (-180 to 180)
    pub lon: f64,
}
impl Coordinate {
    /// Create a new coordinate
    pub fn new(lat: f64, lon: f64) -> Result<Self> {
        if !(-90.0..=90.0).contains(&lat) {
            return Err(SklearsError::InvalidInput(format!(
                "Latitude must be between -90 and 90, got {}",
                lat
            )));
        }
        if !(-180.0..=180.0).contains(&lon) {
            return Err(SklearsError::InvalidInput(format!(
                "Longitude must be between -180 and 180, got {}",
                lon
            )));
        }
        Ok(Self { lat, lon })
    }
    /// Convert to radians
    pub fn to_radians(&self) -> (f64, f64) {
        (self.lat.to_radians(), self.lon.to_radians())
    }
}
/// Configuration for coordinate transformation
#[derive(Debug, Clone)]
pub struct CoordinateTransformerConfig {
    /// Source coordinate system
    pub from: CoordinateSystem,
    /// Target coordinate system
    pub to: CoordinateSystem,
}
/// Configuration for proximity features
#[derive(Debug, Clone)]
pub struct ProximityFeaturesConfig {
    /// Points of interest to calculate proximity to
    pub points_of_interest: Vec<(String, Coordinate)>,
    /// Distance metric
    pub metric: SpatialDistanceMetric,
    /// Maximum distance to consider (in km)
    pub max_distance: Option<f64>,
    /// Whether to include binary indicators (within max_distance)
    pub include_indicators: bool,
}
/// Configuration for spatial autocorrelation features
#[derive(Debug, Clone)]
pub struct SpatialAutocorrelationConfig {
    /// Number of nearest neighbors to consider
    pub k_neighbors: usize,
    /// Distance metric
    pub metric: SpatialDistanceMetric,
    /// Whether to include Moran's I statistic
    pub include_morans_i: bool,
    /// Whether to include local indicators of spatial association (LISA)
    pub include_lisa: bool,
}
/// Fitted spatial clustering feature extractor
pub struct SpatialClusteringFitted {
    pub(super) config: SpatialClusteringConfig,
    /// Cluster centers or representative points
    pub(super) cluster_centers: Vec<Coordinate>,
    /// Grid bounds for grid-based clustering; retained for future bounds-aware transform
    #[allow(dead_code)]
    pub(super) grid_bounds: Option<(f64, f64, f64, f64)>,
}
/// Configuration for spatial distance feature extraction
#[derive(Debug, Clone)]
pub struct SpatialDistanceFeaturesConfig {
    /// Reference points to calculate distances from
    pub reference_points: Vec<Coordinate>,
    /// Distance metric to use
    pub metric: SpatialDistanceMetric,
    /// Whether to include inverse distances (1/distance)
    pub include_inverse: bool,
    /// Whether to include squared distances
    pub include_squared: bool,
}
/// Spatial distance feature extractor
pub struct SpatialDistanceFeatures {
    pub(super) config: SpatialDistanceFeaturesConfig,
}
impl SpatialDistanceFeatures {
    /// Create a new spatial distance feature extractor
    pub fn new(config: SpatialDistanceFeaturesConfig) -> Self {
        Self { config }
    }
}
/// Geohash encoder for spatial indexing
pub struct GeohashEncoder {
    pub(super) config: GeohashEncoderConfig,
}
impl GeohashEncoder {
    /// Create a new geohash encoder
    pub fn new(config: GeohashEncoderConfig) -> Self {
        Self { config }
    }
}
/// Supported coordinate reference systems
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordinateSystem {
    /// WGS84 geographic coordinates (latitude, longitude in degrees)
    WGS84,
    /// Web Mercator projection (EPSG:3857) - used by web mapping services
    WebMercator,
    /// Universal Transverse Mercator projection
    UTM { zone: u8, northern: bool },
}
/// Spatial clustering method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialClusteringMethod {
    /// Grid-based clustering using spatial binning
    Grid,
    /// Density-based features (local density estimation)
    Density,
    /// K-means-style clustering on geographic coordinates
    KMeans,
    /// Hierarchical clustering features
    Hierarchical,
}
/// Spatial clustering feature extractor
pub struct SpatialClustering {
    pub(super) config: SpatialClusteringConfig,
}
impl SpatialClustering {
    /// Create a new spatial clustering feature extractor
    pub fn new(config: SpatialClusteringConfig) -> Self {
        Self { config }
    }
}
