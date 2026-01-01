//! Geospatial preprocessing module
//!
//! This module provides comprehensive geospatial data preprocessing including:
//! - Coordinate system transformations (WGS84, Web Mercator, UTM)
//! - Geohash encoding and decoding for spatial indexing
//! - Spatial distance calculations (Haversine, Vincenty, Euclidean)
//! - Proximity and neighborhood features
//! - Spatial clustering and binning
//! - Bounding box operations

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::seeded_rng;
use sklears_core::prelude::*;
use std::collections::HashMap;
use std::f64::consts::PI;

// ================================================================================================
// Constants and Type Definitions
// ================================================================================================

/// Earth radius in kilometers (WGS84 mean radius)
const EARTH_RADIUS_KM: f64 = 6371.0088;

/// Earth radius in meters
const EARTH_RADIUS_M: f64 = 6371008.8;

/// WGS84 semi-major axis (meters)
const WGS84_A: f64 = 6378137.0;

/// WGS84 semi-minor axis (meters)
const WGS84_B: f64 = 6356752.314245;

/// WGS84 flattening
const WGS84_F: f64 = 1.0 / 298.257223563;

/// Web Mercator maximum latitude (approximately 85.051129 degrees)
const WEB_MERCATOR_MAX_LAT: f64 = 85.051129;

/// Base32 characters used in geohash encoding
const BASE32_CHARS: &[u8; 32] = b"0123456789bcdefghjkmnpqrstuvwxyz";

// ================================================================================================
// Coordinate Systems
// ================================================================================================

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

// ================================================================================================
// Distance Metrics
// ================================================================================================

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

/// Calculate distance between two coordinates using specified metric
pub fn calculate_distance(
    coord1: &Coordinate,
    coord2: &Coordinate,
    metric: SpatialDistanceMetric,
) -> Result<f64> {
    match metric {
        SpatialDistanceMetric::Haversine => Ok(haversine_distance(coord1, coord2)),
        SpatialDistanceMetric::Vincenty => vincenty_distance(coord1, coord2),
        SpatialDistanceMetric::Euclidean => {
            Ok(((coord2.lat - coord1.lat).powi(2) + (coord2.lon - coord1.lon).powi(2)).sqrt())
        }
        SpatialDistanceMetric::Manhattan => {
            Ok((coord2.lat - coord1.lat).abs() + (coord2.lon - coord1.lon).abs())
        }
    }
}

/// Haversine distance between two coordinates in kilometers
pub fn haversine_distance(coord1: &Coordinate, coord2: &Coordinate) -> f64 {
    let (lat1, lon1) = coord1.to_radians();
    let (lat2, lon2) = coord2.to_radians();

    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;

    let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

    EARTH_RADIUS_KM * c
}

/// Vincenty distance between two coordinates in kilometers (more accurate)
pub fn vincenty_distance(coord1: &Coordinate, coord2: &Coordinate) -> Result<f64> {
    let (lat1, lon1) = coord1.to_radians();
    let (lat2, lon2) = coord2.to_radians();

    let l = lon2 - lon1;
    let u1 = ((1.0 - WGS84_F) * lat1.tan()).atan();
    let u2 = ((1.0 - WGS84_F) * lat2.tan()).atan();
    let sin_u1 = u1.sin();
    let cos_u1 = u1.cos();
    let sin_u2 = u2.sin();
    let cos_u2 = u2.cos();

    let mut lambda = l;
    let mut lambda_prev;
    let mut iterations = 0;
    const MAX_ITERATIONS: usize = 100;
    const CONVERGENCE_THRESHOLD: f64 = 1e-12;

    let (
        mut sin_sigma,
        mut cos_sigma,
        mut sigma,
        mut sin_alpha,
        mut cos_sq_alpha,
        mut cos_2sigma_m,
    );

    loop {
        let sin_lambda = lambda.sin();
        let cos_lambda = lambda.cos();
        sin_sigma = ((cos_u2 * sin_lambda).powi(2)
            + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda).powi(2))
        .sqrt();

        if sin_sigma == 0.0 {
            return Ok(0.0); // Coincident points
        }

        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda;
        sigma = sin_sigma.atan2(cos_sigma);
        sin_alpha = cos_u1 * cos_u2 * sin_lambda / sin_sigma;
        cos_sq_alpha = 1.0 - sin_alpha.powi(2);
        cos_2sigma_m = if cos_sq_alpha != 0.0 {
            cos_sigma - 2.0 * sin_u1 * sin_u2 / cos_sq_alpha
        } else {
            0.0
        };

        let c = WGS84_F / 16.0 * cos_sq_alpha * (4.0 + WGS84_F * (4.0 - 3.0 * cos_sq_alpha));
        lambda_prev = lambda;
        lambda = l
            + (1.0 - c)
                * WGS84_F
                * sin_alpha
                * (sigma
                    + c * sin_sigma
                        * (cos_2sigma_m + c * cos_sigma * (-1.0 + 2.0 * cos_2sigma_m.powi(2))));

        iterations += 1;
        if (lambda - lambda_prev).abs() < CONVERGENCE_THRESHOLD || iterations >= MAX_ITERATIONS {
            break;
        }
    }

    if iterations >= MAX_ITERATIONS {
        return Err(SklearsError::InvalidInput(
            "Vincenty formula failed to converge".to_string(),
        ));
    }

    let u_sq = cos_sq_alpha * (WGS84_A.powi(2) - WGS84_B.powi(2)) / WGS84_B.powi(2);
    let a = 1.0 + u_sq / 16384.0 * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)));
    let b = u_sq / 1024.0 * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)));
    let delta_sigma = b
        * sin_sigma
        * (cos_2sigma_m
            + b / 4.0
                * (cos_sigma * (-1.0 + 2.0 * cos_2sigma_m.powi(2))
                    - b / 6.0
                        * cos_2sigma_m
                        * (-3.0 + 4.0 * sin_sigma.powi(2))
                        * (-3.0 + 4.0 * cos_2sigma_m.powi(2))));

    let s = WGS84_B * a * (sigma - delta_sigma);

    Ok(s / 1000.0) // Convert to kilometers
}

// ================================================================================================
// Geohash Encoding
// ================================================================================================

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

        // Build reverse lookup table
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
        // Neighbor finding algorithm based on geohash properties
        // This is a simplified version - a full implementation would handle edge cases
        let coord = Self::decode(geohash)?;
        let (lat_min, lon_min, lat_max, lon_max) = Self::bounds(geohash)?;

        let lat_diff = lat_max - lat_min;
        let lon_diff = lon_max - lon_min;

        let precision = geohash.len();
        let mut neighbors = Vec::with_capacity(8);

        // 8 neighboring cells
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
            // Handle longitude wrapping
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

// ================================================================================================
// Coordinate Transformers
// ================================================================================================

/// Configuration for coordinate transformation
#[derive(Debug, Clone)]
pub struct CoordinateTransformerConfig {
    /// Source coordinate system
    pub from: CoordinateSystem,
    /// Target coordinate system
    pub to: CoordinateSystem,
}

/// Coordinate system transformer (Untrained state)
pub struct CoordinateTransformer {
    config: CoordinateTransformerConfig,
}

/// Coordinate system transformer (Fitted state)
pub struct CoordinateTransformerFitted {
    config: CoordinateTransformerConfig,
}

impl CoordinateTransformer {
    /// Create a new coordinate transformer
    pub fn new(from: CoordinateSystem, to: CoordinateSystem) -> Self {
        Self {
            config: CoordinateTransformerConfig { from, to },
        }
    }
}

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

    fn fit(self, _X: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        // No fitting required for coordinate transformation
        Ok(CoordinateTransformerFitted {
            config: self.config,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for CoordinateTransformerFitted {
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
                    )))
                }
            };

            result[[i, 0]] = transformed.0;
            result[[i, 1]] = transformed.1;
        }

        Ok(result)
    }
}

// ================================================================================================
// Coordinate Transformation Functions
// ================================================================================================

/// Convert WGS84 (lat, lon) to Web Mercator (x, y)
fn wgs84_to_web_mercator(lat: f64, lon: f64) -> Result<(f64, f64)> {
    if lat.abs() > WEB_MERCATOR_MAX_LAT {
        return Err(SklearsError::InvalidInput(format!(
            "Latitude exceeds Web Mercator bounds: {}",
            lat
        )));
    }

    let x = WGS84_A * lon.to_radians();
    let y = WGS84_A * ((PI / 4.0) + (lat.to_radians() / 2.0)).tan().ln();

    Ok((x, y))
}

/// Convert Web Mercator (x, y) to WGS84 (lat, lon)
fn web_mercator_to_wgs84(x: f64, y: f64) -> Result<(f64, f64)> {
    let lon = (x / WGS84_A).to_degrees();
    let lat = (2.0 * (y / WGS84_A).exp().atan() - PI / 2.0).to_degrees();

    Ok((lat, lon))
}

/// Convert WGS84 to UTM coordinates
fn wgs84_to_utm(lat: f64, lon: f64, zone: u8, northern: bool) -> Result<(f64, f64)> {
    // Simplified UTM transformation
    // Full implementation would use more precise formulas
    let lat_rad = lat.to_radians();
    let lon_rad = lon.to_radians();

    let central_meridian = ((zone as f64 - 1.0) * 6.0 - 180.0 + 3.0).to_radians();
    let lon_diff = lon_rad - central_meridian;

    // UTM scale factor
    let k0 = 0.9996;

    // Simplified formulas
    let n = WGS84_A / (1.0 - (WGS84_F * (2.0 - WGS84_F) * lat_rad.sin().powi(2))).sqrt();
    let t = lat_rad.tan();
    let c = (WGS84_F * (2.0 - WGS84_F)) / (1.0 - WGS84_F * (2.0 - WGS84_F)) * lat_rad.cos().powi(2);
    let a = lon_diff * lat_rad.cos();

    let m = WGS84_A
        * ((1.0
            - WGS84_F * (2.0 - WGS84_F) / 4.0
            - 3.0 * (WGS84_F * (2.0 - WGS84_F)).powi(2) / 64.0)
            * lat_rad
            - (3.0 * WGS84_F * (2.0 - WGS84_F) / 8.0
                + 3.0 * (WGS84_F * (2.0 - WGS84_F)).powi(2) / 32.0)
                * (2.0 * lat_rad).sin()
            + (15.0 * (WGS84_F * (2.0 - WGS84_F)).powi(2) / 256.0) * (4.0 * lat_rad).sin());

    let easting = 500000.0
        + k0 * n
            * (a + (1.0 - t.powi(2) + c) * a.powi(3) / 6.0
                + (5.0 - 18.0 * t.powi(2) + t.powi(4)) * a.powi(5) / 120.0);

    let northing_offset = if northern { 0.0 } else { 10000000.0 };
    let northing = northing_offset
        + k0 * (m + n
            * t
            * (a.powi(2) / 2.0
                + (5.0 - t.powi(2) + 9.0 * c) * a.powi(4) / 24.0
                + (61.0 - 58.0 * t.powi(2) + t.powi(4)) * a.powi(6) / 720.0));

    Ok((easting, northing))
}

/// Convert UTM to WGS84 coordinates
fn utm_to_wgs84(easting: f64, northing: f64, zone: u8, northern: bool) -> Result<(f64, f64)> {
    // Simplified inverse UTM transformation
    let k0 = 0.9996;
    let central_meridian = ((zone as f64 - 1.0) * 6.0 - 180.0 + 3.0).to_radians();

    let x = easting - 500000.0;
    let y = if northern {
        northing
    } else {
        northing - 10000000.0
    };

    let m = y / k0;
    let mu = m
        / (WGS84_A
            * (1.0
                - WGS84_F * (2.0 - WGS84_F) / 4.0
                - 3.0 * (WGS84_F * (2.0 - WGS84_F)).powi(2) / 64.0));

    // Footprint latitude
    let e1 = (1.0 - (1.0 - WGS84_F * (2.0 - WGS84_F)).sqrt())
        / (1.0 + (1.0 - WGS84_F * (2.0 - WGS84_F)).sqrt());
    let phi1 = mu
        + (3.0 * e1 / 2.0 - 27.0 * e1.powi(3) / 32.0) * (2.0 * mu).sin()
        + (21.0 * e1.powi(2) / 16.0 - 55.0 * e1.powi(4) / 32.0) * (4.0 * mu).sin()
        + (151.0 * e1.powi(3) / 96.0) * (6.0 * mu).sin();

    let n1 = WGS84_A / (1.0 - WGS84_F * (2.0 - WGS84_F) * phi1.sin().powi(2)).sqrt();
    let t1 = phi1.tan();
    let c1 = (WGS84_F * (2.0 - WGS84_F)) / (1.0 - WGS84_F * (2.0 - WGS84_F)) * phi1.cos().powi(2);
    let r1 = WGS84_A * (1.0 - WGS84_F * (2.0 - WGS84_F))
        / (1.0 - WGS84_F * (2.0 - WGS84_F) * phi1.sin().powi(2)).powf(1.5);
    let d = x / (n1 * k0);

    let lat = phi1
        - (n1 * t1 / r1)
            * (d.powi(2) / 2.0
                - (5.0 + 3.0 * t1.powi(2) + 10.0 * c1 - 4.0 * c1.powi(2)) * d.powi(4) / 24.0
                + (61.0 + 90.0 * t1.powi(2) + 298.0 * c1 + 45.0 * t1.powi(4)) * d.powi(6) / 720.0);

    let lon = central_meridian
        + (d - (1.0 + 2.0 * t1.powi(2) + c1) * d.powi(3) / 6.0
            + (5.0 - 2.0 * c1 + 28.0 * t1.powi(2)) * d.powi(5) / 120.0)
            / phi1.cos();

    Ok((lat.to_degrees(), lon.to_degrees()))
}

// ================================================================================================
// Geohash Feature Encoder
// ================================================================================================

/// Configuration for geohash encoding
#[derive(Debug, Clone)]
pub struct GeohashEncoderConfig {
    /// Geohash precision (number of characters, 1-12)
    pub precision: usize,
    /// Whether to include neighbor geohashes as features
    pub include_neighbors: bool,
}

impl Default for GeohashEncoderConfig {
    fn default() -> Self {
        Self {
            precision: 7,
            include_neighbors: false,
        }
    }
}

/// Geohash encoder for spatial indexing
pub struct GeohashEncoder {
    config: GeohashEncoderConfig,
}

/// Fitted geohash encoder
pub struct GeohashEncoderFitted {
    config: GeohashEncoderConfig,
    /// Vocabulary of unique geohashes
    vocabulary: HashMap<String, usize>,
}

impl GeohashEncoder {
    /// Create a new geohash encoder
    pub fn new(config: GeohashEncoderConfig) -> Self {
        Self { config }
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

impl Transform<Array2<f64>, Array2<f64>> for GeohashEncoderFitted {
    fn transform(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        if X.ncols() != 2 {
            return Err(SklearsError::InvalidInput(
                "Input must have exactly 2 columns (lat, lon)".to_string(),
            ));
        }

        let nrows = X.nrows();
        let ncols = if self.config.include_neighbors {
            self.vocabulary.len() * 9 // Each geohash + 8 neighbors
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

// ================================================================================================
// Spatial Distance Features
// ================================================================================================

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
    config: SpatialDistanceFeaturesConfig,
}

/// Fitted spatial distance feature extractor
pub struct SpatialDistanceFeaturesFitted {
    config: SpatialDistanceFeaturesConfig,
}

impl SpatialDistanceFeatures {
    /// Create a new spatial distance feature extractor
    pub fn new(config: SpatialDistanceFeaturesConfig) -> Self {
        Self { config }
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

// ================================================================================================
// Spatial Binning
// ================================================================================================

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

/// Spatial binning transformer
pub struct SpatialBinning {
    config: SpatialBinningConfig,
}

/// Fitted spatial binning transformer
pub struct SpatialBinningFitted {
    config: SpatialBinningConfig,
    lat_range: (f64, f64),
    lon_range: (f64, f64),
}

impl SpatialBinning {
    /// Create a new spatial binning transformer
    pub fn new(config: SpatialBinningConfig) -> Self {
        Self { config }
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

impl Transform<Array2<f64>, Array2<f64>> for SpatialBinningFitted {
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

// ================================================================================================
// Spatial Clustering Features
// ================================================================================================

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

impl Default for SpatialClusteringConfig {
    fn default() -> Self {
        Self {
            method: SpatialClusteringMethod::Grid,
            n_clusters: 10,
            epsilon: Some(5.0), // 5 km
            min_points: Some(5),
            metric: SpatialDistanceMetric::Haversine,
        }
    }
}

/// Spatial clustering feature extractor
pub struct SpatialClustering {
    config: SpatialClusteringConfig,
}

/// Fitted spatial clustering feature extractor
pub struct SpatialClusteringFitted {
    config: SpatialClusteringConfig,
    /// Cluster centers or representative points
    cluster_centers: Vec<Coordinate>,
    /// Grid bounds for grid-based clustering
    grid_bounds: Option<(f64, f64, f64, f64)>,
}

impl SpatialClustering {
    /// Create a new spatial clustering feature extractor
    pub fn new(config: SpatialClusteringConfig) -> Self {
        Self { config }
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

    fn fit(self, X: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        if X.ncols() != 2 {
            return Err(SklearsError::InvalidInput(
                "Input must have exactly 2 columns (lat, lon)".to_string(),
            ));
        }

        let cluster_centers = match self.config.method {
            SpatialClusteringMethod::Grid => {
                // Grid-based clustering: create evenly spaced cluster centers
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
                // Simple K-means initialization using random sampling
                let mut rng = seeded_rng(42);
                let nrows = X.nrows();
                let mut centers = Vec::new();

                // Initialize with random samples
                for _ in 0..self.config.n_clusters.min(nrows) {
                    let idx = rng.random_range(0..nrows);
                    if let Ok(coord) = Coordinate::new(X[[idx, 0]], X[[idx, 1]]) {
                        centers.push(coord);
                    }
                }

                // Perform a few iterations of k-means
                for _ in 0..10 {
                    let mut cluster_sums = vec![(0.0, 0.0); centers.len()];
                    let mut cluster_counts = vec![0; centers.len()];

                    // Assignment step
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

                    // Update step
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
                // For density-based methods, we'll sample representative points
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

impl Transform<Array2<f64>, Array2<f64>> for SpatialClusteringFitted {
    fn transform(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        if X.ncols() != 2 {
            return Err(SklearsError::InvalidInput(
                "Input must have exactly 2 columns (lat, lon)".to_string(),
            ));
        }

        let nrows = X.nrows();

        match self.config.method {
            SpatialClusteringMethod::Grid | SpatialClusteringMethod::KMeans => {
                // Output: cluster assignment + distance to nearest cluster center
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
                // Output: local density estimate (number of neighbors within epsilon)
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
                // Output: distances to all cluster centers (linkage-based features)
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

// ================================================================================================
// Spatial Autocorrelation Features
// ================================================================================================

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

/// Spatial autocorrelation feature extractor
pub struct SpatialAutocorrelation {
    config: SpatialAutocorrelationConfig,
}

/// Fitted spatial autocorrelation feature extractor
pub struct SpatialAutocorrelationFitted {
    config: SpatialAutocorrelationConfig,
    /// Training data coordinates for spatial weights
    training_coords: Vec<Coordinate>,
}

impl SpatialAutocorrelation {
    /// Create a new spatial autocorrelation feature extractor
    pub fn new(config: SpatialAutocorrelationConfig) -> Self {
        Self { config }
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

impl Transform<Array2<f64>, Array2<f64>> for SpatialAutocorrelationFitted {
    fn transform(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        if X.ncols() != 2 {
            return Err(SklearsError::InvalidInput(
                "Input must have exactly 2 columns (lat, lon)".to_string(),
            ));
        }

        let nrows = X.nrows();
        let mut n_features = 0;

        if self.config.include_lisa {
            n_features += 1; // Local Moran's I
        }

        let mut result = Array2::zeros((nrows, n_features.max(1)));

        for i in 0..nrows {
            let coord = Coordinate::new(X[[i, 0]], X[[i, 1]])?;

            // Find k nearest neighbors
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

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Calculate local spatial statistics
            if self.config.include_lisa {
                // Simplified local Moran's I calculation
                // In a real implementation, this would use the target variable
                let nearest_neighbors =
                    &distances[1..=self.config.k_neighbors.min(distances.len() - 1)];
                let avg_neighbor_dist = nearest_neighbors.iter().map(|(_, d)| d).sum::<f64>()
                    / nearest_neighbors.len() as f64;

                result[[i, 0]] = 1.0 / (1.0 + avg_neighbor_dist); // Inverse distance weighted
            }
        }

        Ok(result)
    }
}

// ================================================================================================
// Proximity Features
// ================================================================================================

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

/// Proximity feature extractor
pub struct ProximityFeatures {
    config: ProximityFeaturesConfig,
}

/// Fitted proximity feature extractor
pub struct ProximityFeaturesFitted {
    config: ProximityFeaturesConfig,
}

impl ProximityFeatures {
    /// Create a new proximity feature extractor
    pub fn new(config: ProximityFeaturesConfig) -> Self {
        Self { config }
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
    fn transform(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        if X.ncols() != 2 {
            return Err(SklearsError::InvalidInput(
                "Input must have exactly 2 columns (lat, lon)".to_string(),
            ));
        }

        let nrows = X.nrows();
        let n_pois = self.config.points_of_interest.len();
        let n_features = if self.config.include_indicators {
            n_pois * 2 // Distance + indicator
        } else {
            n_pois // Just distance
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

// ================================================================================================
// Tests
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_coordinate_creation() {
        let coord = Coordinate::new(40.7128, -74.0060).unwrap();
        assert_eq!(coord.lat, 40.7128);
        assert_eq!(coord.lon, -74.0060);

        // Test invalid coordinates
        assert!(Coordinate::new(91.0, 0.0).is_err());
        assert!(Coordinate::new(0.0, 181.0).is_err());
    }

    #[test]
    fn test_haversine_distance() {
        // New York to London
        let nyc = Coordinate::new(40.7128, -74.0060).unwrap();
        let london = Coordinate::new(51.5074, -0.1278).unwrap();

        let distance = haversine_distance(&nyc, &london);
        // Expected distance is approximately 5570 km
        assert!((distance - 5570.0).abs() < 50.0);
    }

    #[test]
    fn test_vincenty_distance() {
        let nyc = Coordinate::new(40.7128, -74.0060).unwrap();
        let london = Coordinate::new(51.5074, -0.1278).unwrap();

        let distance = vincenty_distance(&nyc, &london).unwrap();
        // Vincenty should give similar but slightly more accurate result
        assert!((distance - 5570.0).abs() < 50.0);
    }

    #[test]
    fn test_geohash_encode_decode() {
        let coord = Coordinate::new(40.7128, -74.0060).unwrap();
        let geohash = Geohash::encode(&coord, 7).unwrap();
        assert_eq!(geohash.len(), 7);

        let decoded = Geohash::decode(&geohash).unwrap();
        assert!((decoded.lat - coord.lat).abs() < 0.01);
        assert!((decoded.lon - coord.lon).abs() < 0.01);
    }

    #[test]
    fn test_geohash_bounds() {
        let geohash = "dr5ru7";
        let (lat_min, lon_min, lat_max, lon_max) = Geohash::bounds(geohash).unwrap();

        assert!(lat_min < lat_max);
        assert!(lon_min < lon_max);
        assert!(lat_min >= -90.0 && lat_max <= 90.0);
        assert!(lon_min >= -180.0 && lon_max <= 180.0);
    }

    #[test]
    fn test_geohash_neighbors() {
        let geohash = "dr5ru7";
        let neighbors = Geohash::neighbors(geohash).unwrap();

        // Should have up to 8 neighbors
        assert!(neighbors.len() <= 8);
        assert!(neighbors.len() > 0);

        // All neighbors should have the same precision
        for neighbor in neighbors.iter() {
            assert_eq!(neighbor.len(), geohash.len());
        }
    }

    #[test]
    fn test_wgs84_to_web_mercator() {
        let (x, y) = wgs84_to_web_mercator(40.7128, -74.0060).unwrap();

        // Verify conversion is reasonable
        assert!(x.abs() < 20037508.34); // Web Mercator max
        assert!(y.abs() < 20037508.34);

        // Round trip
        let (lat, lon) = web_mercator_to_wgs84(x, y).unwrap();
        assert_relative_eq!(lat, 40.7128, epsilon = 0.0001);
        assert_relative_eq!(lon, -74.0060, epsilon = 0.0001);
    }

    #[test]
    fn test_coordinate_transformer() {
        use scirs2_core::ndarray::array;

        let X = array![[40.7128, -74.0060], [51.5074, -0.1278]];

        let transformer =
            CoordinateTransformer::new(CoordinateSystem::WGS84, CoordinateSystem::WebMercator);
        let fitted = transformer.fit(&X, &()).unwrap();
        let result = fitted.transform(&X).unwrap();

        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 2);

        // Verify transformation produces reasonable values
        assert!(result[[0, 0]].abs() < 20037508.34);
        assert!(result[[0, 1]].abs() < 20037508.34);
    }

    #[test]
    fn test_geohash_encoder() {
        use scirs2_core::ndarray::array;

        let X = array![[40.7128, -74.0060], [40.7589, -73.9851], [51.5074, -0.1278]];

        let config = GeohashEncoderConfig {
            precision: 5,
            include_neighbors: false,
        };

        let encoder = GeohashEncoder::new(config);
        let fitted = encoder.fit(&X, &()).unwrap();
        let result = fitted.transform(&X).unwrap();

        assert_eq!(result.nrows(), 3);
        // Should have as many columns as unique geohashes
        assert!(result.ncols() > 0);
    }

    #[test]
    fn test_spatial_distance_features() {
        use scirs2_core::ndarray::array;

        let X = array![[40.7128, -74.0060], [40.7589, -73.9851]];

        let reference_points = vec![
            Coordinate::new(40.7128, -74.0060).unwrap(), // NYC
            Coordinate::new(51.5074, -0.1278).unwrap(),  // London
        ];

        let config = SpatialDistanceFeaturesConfig {
            reference_points,
            metric: SpatialDistanceMetric::Haversine,
            include_inverse: false,
            include_squared: false,
        };

        let transformer = SpatialDistanceFeatures::new(config);
        let fitted = transformer.fit(&X, &()).unwrap();
        let result = fitted.transform(&X).unwrap();

        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 2); // 2 reference points

        // First point should be close to first reference point
        assert!(result[[0, 0]] < 1.0);
    }

    #[test]
    fn test_spatial_binning() {
        use scirs2_core::ndarray::array;

        let X = array![[40.0, -74.0], [41.0, -73.0], [42.0, -72.0], [43.0, -71.0]];

        let config = SpatialBinningConfig {
            n_lat_bins: 2,
            n_lon_bins: 2,
            lat_range: Some((40.0, 44.0)),
            lon_range: Some((-75.0, -70.0)),
            one_hot: false,
        };

        let binning = SpatialBinning::new(config);
        let fitted = binning.fit(&X, &()).unwrap();
        let result = fitted.transform(&X).unwrap();

        assert_eq!(result.nrows(), 4);
        assert_eq!(result.ncols(), 2);

        // Verify binning produces valid bin indices
        for i in 0..result.nrows() {
            assert!(result[[i, 0]] >= 0.0 && result[[i, 0]] < 2.0);
            assert!(result[[i, 1]] >= 0.0 && result[[i, 1]] < 2.0);
        }
    }

    #[test]
    fn test_spatial_binning_one_hot() {
        use scirs2_core::ndarray::array;

        let X = array![[40.0, -74.0], [41.0, -73.0],];

        let config = SpatialBinningConfig {
            n_lat_bins: 2,
            n_lon_bins: 2,
            lat_range: Some((40.0, 42.0)),
            lon_range: Some((-75.0, -72.0)),
            one_hot: true,
        };

        let binning = SpatialBinning::new(config);
        let fitted = binning.fit(&X, &()).unwrap();
        let result = fitted.transform(&X).unwrap();

        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 4); // 2x2 bins

        // Verify one-hot encoding
        for i in 0..result.nrows() {
            let sum: f64 = result.row(i).sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_spatial_clustering_grid() {
        use scirs2_core::ndarray::array;

        let X = array![[40.0, -74.0], [40.5, -73.5], [41.0, -73.0], [41.5, -72.5]];

        let config = SpatialClusteringConfig {
            method: SpatialClusteringMethod::Grid,
            n_clusters: 4,
            epsilon: None,
            min_points: None,
            metric: SpatialDistanceMetric::Haversine,
        };

        let clustering = SpatialClustering::new(config);
        let fitted = clustering.fit(&X, &()).unwrap();
        let result = fitted.transform(&X).unwrap();

        assert_eq!(result.nrows(), 4);
        assert_eq!(result.ncols(), 2); // Cluster assignment + distance

        // Verify cluster assignments are valid
        for i in 0..result.nrows() {
            assert!(result[[i, 0]] >= 0.0);
            assert!(result[[i, 1]] >= 0.0); // Distance should be non-negative
        }
    }

    #[test]
    fn test_spatial_clustering_kmeans() {
        use scirs2_core::ndarray::array;

        let X = array![[40.0, -74.0], [40.1, -74.1], [42.0, -72.0], [42.1, -72.1]];

        let config = SpatialClusteringConfig {
            method: SpatialClusteringMethod::KMeans,
            n_clusters: 2,
            epsilon: None,
            min_points: None,
            metric: SpatialDistanceMetric::Haversine,
        };

        let clustering = SpatialClustering::new(config);
        let fitted = clustering.fit(&X, &()).unwrap();
        let result = fitted.transform(&X).unwrap();

        assert_eq!(result.nrows(), 4);
        assert_eq!(result.ncols(), 2);

        // Points that are close together should be in the same cluster
        assert_eq!(result[[0, 0]], result[[1, 0]]);
        assert_eq!(result[[2, 0]], result[[3, 0]]);
    }

    #[test]
    fn test_spatial_clustering_density() {
        use scirs2_core::ndarray::array;

        let X = array![
            [40.0, -74.0],
            [40.001, -74.001], // Very close to first point
            [42.0, -72.0]      // Far away
        ];

        let config = SpatialClusteringConfig {
            method: SpatialClusteringMethod::Density,
            n_clusters: 2,
            epsilon: Some(1.0), // 1 km radius
            min_points: Some(1),
            metric: SpatialDistanceMetric::Haversine,
        };

        let clustering = SpatialClustering::new(config);
        let fitted = clustering.fit(&X, &()).unwrap();
        let result = fitted.transform(&X).unwrap();

        assert_eq!(result.nrows(), 3);
        assert_eq!(result.ncols(), 1); // Just density

        // First two points should have higher density
        assert!(result[[0, 0]] > result[[2, 0]]);
        assert!(result[[1, 0]] > result[[2, 0]]);
    }

    #[test]
    fn test_proximity_features() {
        use scirs2_core::ndarray::array;

        let X = array![[40.7128, -74.0060], [51.5074, -0.1278]];

        let pois = vec![
            (
                "NYC".to_string(),
                Coordinate::new(40.7128, -74.0060).unwrap(),
            ),
            (
                "London".to_string(),
                Coordinate::new(51.5074, -0.1278).unwrap(),
            ),
        ];

        let config = ProximityFeaturesConfig {
            points_of_interest: pois,
            metric: SpatialDistanceMetric::Haversine,
            max_distance: Some(100.0),
            include_indicators: true,
        };

        let proximity = ProximityFeatures::new(config);
        let fitted = proximity.fit(&X, &()).unwrap();
        let result = fitted.transform(&X).unwrap();

        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 4); // 2 POIs * 2 (distance + indicator)

        // First point should be very close to NYC POI
        assert!(result[[0, 0]] < 1.0);
        assert_relative_eq!(result[[0, 2]], 1.0, epsilon = 1e-10); // Within 100km

        // Second point should be very close to London POI
        assert!(result[[1, 1]] < 1.0);
        assert_relative_eq!(result[[1, 3]], 1.0, epsilon = 1e-10); // Within 100km
    }

    #[test]
    fn test_spatial_autocorrelation() {
        use scirs2_core::ndarray::array;

        let X = array![[40.0, -74.0], [40.1, -74.0], [40.2, -74.0], [42.0, -72.0]];

        let config = SpatialAutocorrelationConfig {
            k_neighbors: 2,
            metric: SpatialDistanceMetric::Haversine,
            include_morans_i: true,
            include_lisa: true,
        };

        let autocorr = SpatialAutocorrelation::new(config);
        let y = array![1.0, 2.0, 3.0, 4.0]; // Dummy y for testing
        let fitted = autocorr.fit(&X, &y).unwrap();
        let result = fitted.transform(&X).unwrap();

        assert_eq!(result.nrows(), 4);
        assert!(result.ncols() >= 1);

        // Verify all values are finite and non-negative
        for i in 0..result.nrows() {
            for j in 0..result.ncols() {
                assert!(result[[i, j]].is_finite());
                assert!(result[[i, j]] >= 0.0);
            }
        }
    }

    #[test]
    fn test_utm_transformation() {
        let (easting, northing) = wgs84_to_utm(40.7128, -74.0060, 18, true).unwrap();

        // Verify transformation produces reasonable UTM coordinates
        assert!(easting > 0.0);
        assert!(northing > 0.0);

        // Round trip
        let (lat, lon) = utm_to_wgs84(easting, northing, 18, true).unwrap();
        assert_relative_eq!(lat, 40.7128, epsilon = 0.01);
        assert_relative_eq!(lon, -74.0060, epsilon = 0.01);
    }

    #[test]
    fn test_coordinate_systems_enum() {
        let wgs84 = CoordinateSystem::WGS84;
        let web_merc = CoordinateSystem::WebMercator;
        let utm = CoordinateSystem::UTM {
            zone: 18,
            northern: true,
        };

        assert_eq!(wgs84, CoordinateSystem::WGS84);
        assert_eq!(web_merc, CoordinateSystem::WebMercator);
        assert_ne!(wgs84, web_merc);

        match utm {
            CoordinateSystem::UTM { zone, northern } => {
                assert_eq!(zone, 18);
                assert!(northern);
            }
            _ => panic!("Expected UTM"),
        }
    }

    #[test]
    fn test_distance_metrics() {
        let coord1 = Coordinate::new(40.7128, -74.0060).unwrap();
        let coord2 = Coordinate::new(40.7589, -73.9851).unwrap();

        let haversine =
            calculate_distance(&coord1, &coord2, SpatialDistanceMetric::Haversine).unwrap();
        let vincenty =
            calculate_distance(&coord1, &coord2, SpatialDistanceMetric::Vincenty).unwrap();

        // Both should give similar results
        assert!((haversine - vincenty).abs() < 0.1);

        // Distance should be reasonable (a few km)
        assert!(haversine > 0.0 && haversine < 10.0);
    }
}
