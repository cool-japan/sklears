//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use sklears_core::prelude::*;
use std::f64::consts::PI;

use super::types::{Coordinate, SpatialDistanceMetric};

/// Earth radius in kilometers (WGS84 mean radius)
const EARTH_RADIUS_KM: f64 = 6371.0088;
/// WGS84 semi-major axis (meters)
const WGS84_A: f64 = 6378137.0;
/// WGS84 semi-minor axis (meters)
const WGS84_B: f64 = 6356752.314245;
/// WGS84 flattening
const WGS84_F: f64 = 1.0 / 298.257223563;
/// Web Mercator maximum latitude (approximately 85.051129 degrees)
const WEB_MERCATOR_MAX_LAT: f64 = 85.051129;
/// Base32 characters used in geohash encoding
pub(super) const BASE32_CHARS: &[u8; 32] = b"0123456789bcdefghjkmnpqrstuvwxyz";
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
            return Ok(0.0);
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
    Ok(s / 1000.0)
}
/// Convert WGS84 (lat, lon) to Web Mercator (x, y)
pub(super) fn wgs84_to_web_mercator(lat: f64, lon: f64) -> Result<(f64, f64)> {
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
pub(super) fn web_mercator_to_wgs84(x: f64, y: f64) -> Result<(f64, f64)> {
    let lon = (x / WGS84_A).to_degrees();
    let lat = (2.0 * (y / WGS84_A).exp().atan() - PI / 2.0).to_degrees();
    Ok((lat, lon))
}
/// Convert WGS84 to UTM coordinates
pub(super) fn wgs84_to_utm(lat: f64, lon: f64, zone: u8, northern: bool) -> Result<(f64, f64)> {
    let lat_rad = lat.to_radians();
    let lon_rad = lon.to_radians();
    let central_meridian = ((zone as f64 - 1.0) * 6.0 - 180.0 + 3.0).to_radians();
    let lon_diff = lon_rad - central_meridian;
    let k0 = 0.9996;
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
pub(super) fn utm_to_wgs84(
    easting: f64,
    northing: f64,
    zone: u8,
    northern: bool,
) -> Result<(f64, f64)> {
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
#[cfg(test)]
mod tests {
    use super::super::types::{
        Coordinate, CoordinateSystem, CoordinateTransformer, Geohash, GeohashEncoder,
        GeohashEncoderConfig, ProximityFeatures, ProximityFeaturesConfig, SpatialAutocorrelation,
        SpatialAutocorrelationConfig, SpatialBinning, SpatialBinningConfig, SpatialClustering,
        SpatialClusteringConfig, SpatialClusteringMethod, SpatialDistanceFeatures,
        SpatialDistanceFeaturesConfig, SpatialDistanceMetric,
    };
    use super::*;
    use approx::assert_relative_eq;
    #[test]
    fn test_coordinate_creation() {
        let coord = Coordinate::new(40.7128, -74.0060).expect("operation should succeed");
        assert_eq!(coord.lat, 40.7128);
        assert_eq!(coord.lon, -74.0060);
        assert!(Coordinate::new(91.0, 0.0).is_err());
        assert!(Coordinate::new(0.0, 181.0).is_err());
    }
    #[test]
    fn test_haversine_distance() {
        let nyc = Coordinate::new(40.7128, -74.0060).expect("operation should succeed");
        let london = Coordinate::new(51.5074, -0.1278).expect("operation should succeed");
        let distance = haversine_distance(&nyc, &london);
        assert!((distance - 5570.0).abs() < 50.0);
    }
    #[test]
    fn test_vincenty_distance() {
        let nyc = Coordinate::new(40.7128, -74.0060).expect("operation should succeed");
        let london = Coordinate::new(51.5074, -0.1278).expect("operation should succeed");
        let distance = vincenty_distance(&nyc, &london).expect("operation should succeed");
        assert!((distance - 5570.0).abs() < 50.0);
    }
    #[test]
    fn test_geohash_encode_decode() {
        let coord = Coordinate::new(40.7128, -74.0060).expect("operation should succeed");
        let geohash = Geohash::encode(&coord, 7).expect("serialization should succeed");
        assert_eq!(geohash.len(), 7);
        let decoded = Geohash::decode(&geohash).expect("deserialization should succeed");
        assert!((decoded.lat - coord.lat).abs() < 0.01);
        assert!((decoded.lon - coord.lon).abs() < 0.01);
    }
    #[test]
    fn test_geohash_bounds() {
        let geohash = "dr5ru7";
        let (lat_min, lon_min, lat_max, lon_max) =
            Geohash::bounds(geohash).expect("operation should succeed");
        assert!(lat_min < lat_max);
        assert!(lon_min < lon_max);
        assert!(lat_min >= -90.0 && lat_max <= 90.0);
        assert!(lon_min >= -180.0 && lon_max <= 180.0);
    }
    #[test]
    fn test_geohash_neighbors() {
        let geohash = "dr5ru7";
        let neighbors = Geohash::neighbors(geohash).expect("operation should succeed");
        assert!(neighbors.len() <= 8);
        assert!(!neighbors.is_empty());
        for neighbor in neighbors.iter() {
            assert_eq!(neighbor.len(), geohash.len());
        }
    }
    #[test]
    fn test_wgs84_to_web_mercator() {
        let (x, y) = wgs84_to_web_mercator(40.7128, -74.0060).expect("operation should succeed");
        assert!(x.abs() < 20037508.34);
        assert!(y.abs() < 20037508.34);
        let (lat, lon) = web_mercator_to_wgs84(x, y).expect("operation should succeed");
        assert_relative_eq!(lat, 40.7128, epsilon = 0.0001);
        assert_relative_eq!(lon, -74.0060, epsilon = 0.0001);
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_coordinate_transformer() {
        use scirs2_core::ndarray::array;
        let X = array![[40.7128, -74.0060], [51.5074, -0.1278]];
        let transformer =
            CoordinateTransformer::new(CoordinateSystem::WGS84, CoordinateSystem::WebMercator);
        let fitted = transformer
            .fit(&X, &())
            .expect("model fitting should succeed");
        let result = fitted.transform(&X).expect("transformation should succeed");
        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 2);
        assert!(result[[0, 0]].abs() < 20037508.34);
        assert!(result[[0, 1]].abs() < 20037508.34);
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_geohash_encoder() {
        use scirs2_core::ndarray::array;
        let X = array![[40.7128, -74.0060], [40.7589, -73.9851], [51.5074, -0.1278]];
        let config = GeohashEncoderConfig {
            precision: 5,
            include_neighbors: false,
        };
        let encoder = GeohashEncoder::new(config);
        let fitted = encoder.fit(&X, &()).expect("serialization should succeed");
        let result = fitted.transform(&X).expect("transformation should succeed");
        assert_eq!(result.nrows(), 3);
        assert!(result.ncols() > 0);
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_spatial_distance_features() {
        use scirs2_core::ndarray::array;
        let X = array![[40.7128, -74.0060], [40.7589, -73.9851]];
        let reference_points = vec![
            Coordinate::new(40.7128, -74.0060).expect("operation should succeed"),
            Coordinate::new(51.5074, -0.1278).expect("operation should succeed"),
        ];
        let config = SpatialDistanceFeaturesConfig {
            reference_points,
            metric: SpatialDistanceMetric::Haversine,
            include_inverse: false,
            include_squared: false,
        };
        let transformer = SpatialDistanceFeatures::new(config);
        let fitted = transformer
            .fit(&X, &())
            .expect("model fitting should succeed");
        let result = fitted.transform(&X).expect("transformation should succeed");
        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 2);
        assert!(result[[0, 0]] < 1.0);
    }
    #[test]
    #[allow(non_snake_case)]
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
        let fitted = binning.fit(&X, &()).expect("model fitting should succeed");
        let result = fitted.transform(&X).expect("transformation should succeed");
        assert_eq!(result.nrows(), 4);
        assert_eq!(result.ncols(), 2);
        for i in 0..result.nrows() {
            assert!(result[[i, 0]] >= 0.0 && result[[i, 0]] < 2.0);
            assert!(result[[i, 1]] >= 0.0 && result[[i, 1]] < 2.0);
        }
    }
    #[test]
    #[allow(non_snake_case)]
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
        let fitted = binning.fit(&X, &()).expect("model fitting should succeed");
        let result = fitted.transform(&X).expect("transformation should succeed");
        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 4);
        for i in 0..result.nrows() {
            let sum: f64 = result.row(i).sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        }
    }
    #[test]
    #[allow(non_snake_case)]
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
        let fitted = clustering
            .fit(&X, &())
            .expect("model fitting should succeed");
        let result = fitted.transform(&X).expect("transformation should succeed");
        assert_eq!(result.nrows(), 4);
        assert_eq!(result.ncols(), 2);
        for i in 0..result.nrows() {
            assert!(result[[i, 0]] >= 0.0);
            assert!(result[[i, 1]] >= 0.0);
        }
    }
    #[test]
    #[allow(non_snake_case)]
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
        let fitted = clustering
            .fit(&X, &())
            .expect("model fitting should succeed");
        let result = fitted.transform(&X).expect("transformation should succeed");
        assert_eq!(result.nrows(), 4);
        assert_eq!(result.ncols(), 2);
        assert_eq!(result[[0, 0]], result[[1, 0]]);
        assert_eq!(result[[2, 0]], result[[3, 0]]);
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_spatial_clustering_density() {
        use scirs2_core::ndarray::array;
        let X = array![[40.0, -74.0], [40.001, -74.001], [42.0, -72.0]];
        let config = SpatialClusteringConfig {
            method: SpatialClusteringMethod::Density,
            n_clusters: 2,
            epsilon: Some(1.0),
            min_points: Some(1),
            metric: SpatialDistanceMetric::Haversine,
        };
        let clustering = SpatialClustering::new(config);
        let fitted = clustering
            .fit(&X, &())
            .expect("model fitting should succeed");
        let result = fitted.transform(&X).expect("transformation should succeed");
        assert_eq!(result.nrows(), 3);
        assert_eq!(result.ncols(), 1);
        assert!(result[[0, 0]] > result[[2, 0]]);
        assert!(result[[1, 0]] > result[[2, 0]]);
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_proximity_features() {
        use scirs2_core::ndarray::array;
        let X = array![[40.7128, -74.0060], [51.5074, -0.1278]];
        let pois = vec![
            (
                "NYC".to_string(),
                Coordinate::new(40.7128, -74.0060).expect("operation should succeed"),
            ),
            (
                "London".to_string(),
                Coordinate::new(51.5074, -0.1278).expect("operation should succeed"),
            ),
        ];
        let config = ProximityFeaturesConfig {
            points_of_interest: pois,
            metric: SpatialDistanceMetric::Haversine,
            max_distance: Some(100.0),
            include_indicators: true,
        };
        let proximity = ProximityFeatures::new(config);
        let fitted = proximity
            .fit(&X, &())
            .expect("model fitting should succeed");
        let result = fitted.transform(&X).expect("transformation should succeed");
        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 4);
        assert!(result[[0, 0]] < 1.0);
        assert_relative_eq!(result[[0, 2]], 1.0, epsilon = 1e-10);
        assert!(result[[1, 1]] < 1.0);
        assert_relative_eq!(result[[1, 3]], 1.0, epsilon = 1e-10);
    }
    #[test]
    #[allow(non_snake_case)]
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
        let y = array![1.0, 2.0, 3.0, 4.0];
        let fitted = autocorr.fit(&X, &y).expect("model fitting should succeed");
        let result = fitted.transform(&X).expect("transformation should succeed");
        assert_eq!(result.nrows(), 4);
        assert!(result.ncols() >= 1);
        for i in 0..result.nrows() {
            for j in 0..result.ncols() {
                assert!(result[[i, j]].is_finite());
                assert!(result[[i, j]] >= 0.0);
            }
        }
    }
    #[test]
    fn test_utm_transformation() {
        let (easting, northing) =
            wgs84_to_utm(40.7128, -74.0060, 18, true).expect("operation should succeed");
        assert!(easting > 0.0);
        assert!(northing > 0.0);
        let (lat, lon) =
            utm_to_wgs84(easting, northing, 18, true).expect("operation should succeed");
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
        let coord1 = Coordinate::new(40.7128, -74.0060).expect("operation should succeed");
        let coord2 = Coordinate::new(40.7589, -73.9851).expect("operation should succeed");
        let haversine = calculate_distance(&coord1, &coord2, SpatialDistanceMetric::Haversine)
            .expect("operation should succeed");
        let vincenty = calculate_distance(&coord1, &coord2, SpatialDistanceMetric::Vincenty)
            .expect("operation should succeed");
        assert!((haversine - vincenty).abs() < 0.1);
        assert!(haversine > 0.0 && haversine < 10.0);
    }
}
