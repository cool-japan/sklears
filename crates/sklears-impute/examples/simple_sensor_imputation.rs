//! Simple Sensor Data Imputation Example
//!
//! This example demonstrates basic sensor data imputation using simple
//! statistical methods from the sklears-impute crate.
//!
//! ```bash
//! # Run this example
//! cargo run --example simple_sensor_imputation
//! ```

use scirs2_core::ndarray::Array2;
use scirs2_core::random::thread_rng;
use sklears_impute::core::ImputationError;
use std::time::SystemTime;

/// Simple sensor dataset structure
#[derive(Debug, Clone)]
pub struct SimpleSensorData {
    /// Sensor measurements with missing values
    pub measurements: Array2<f64>,
    /// Sensor IDs
    pub sensor_ids: Vec<String>,
    /// Measurement timestamps
    pub timestamps: Vec<SystemTime>,
}

/// Sensor types for the demonstration
#[derive(Debug, Clone)]
pub enum SensorType {
    Temperature,
    Humidity,
    Pressure,
}

/// Create synthetic sensor data with missing values
fn create_sample_sensor_data() -> SimpleSensorData {
    let n_sensors = 10;
    let n_timesteps = 200;
    let mut rng = thread_rng();

    // Generate synthetic sensor data
    let mut measurements = Array2::zeros((n_timesteps, n_sensors));

    for sensor in 0..n_sensors {
        let sensor_type = match sensor % 3 {
            0 => SensorType::Temperature,
            1 => SensorType::Humidity,
            _ => SensorType::Pressure,
        };

        for time in 0..n_timesteps {
            let value = match sensor_type {
                SensorType::Temperature => {
                    // Temperature with daily cycle
                    let base_temp = 20.0;
                    let daily_variation =
                        10.0 * (2.0 * std::f64::consts::PI * time as f64 / 24.0).sin();
                    let noise = rng.random_range(-2.0..2.0);
                    base_temp + daily_variation + noise
                }
                SensorType::Humidity => {
                    // Humidity (somewhat inverse to temperature)
                    let base_humidity = 60.0;
                    let variation = -5.0 * (2.0 * std::f64::consts::PI * time as f64 / 24.0).sin();
                    let noise = rng.random_range(-5.0..5.0);
                    (base_humidity + variation + noise).clamp(0.0, 100.0)
                }
                SensorType::Pressure => {
                    // Pressure (more stable)
                    let base_pressure = 1013.25;
                    let noise = rng.random_range(-5.0..5.0);
                    base_pressure + noise
                }
            };

            measurements[[time, sensor]] = value;
        }
    }

    // Introduce missing values
    // Random missing (2%)
    for time in 0..n_timesteps {
        for sensor in 0..n_sensors {
            if rng.random_range(0.0..1.0) < 0.02 {
                measurements[[time, sensor]] = f64::NAN;
            }
        }
    }

    // Sensor failure periods (simulate maintenance)
    for _ in 0..3 {
        let sensor = rng.random_range(0..n_sensors);
        let start_time = rng.random_range(0..n_timesteps - 10);
        let duration = rng.random_range(2..8);

        for t in start_time..(start_time + duration).min(n_timesteps) {
            measurements[[t, sensor]] = f64::NAN;
        }
    }

    // Create sensor IDs
    let sensor_ids: Vec<String> = (0..n_sensors).map(|i| format!("SENSOR_{:03}", i)).collect();

    // Create timestamps (hourly data)
    let base_time = SystemTime::now();
    let timestamps: Vec<SystemTime> = (0..n_timesteps)
        .map(|i| base_time - std::time::Duration::from_secs(i as u64 * 3600))
        .collect();

    SimpleSensorData {
        measurements,
        sensor_ids,
        timestamps,
    }
}

/// Simple mean imputation implementation
fn mean_imputation(data: &Array2<f64>) -> Result<Array2<f64>, ImputationError> {
    let mut imputed = data.clone();
    let (n_timesteps, n_sensors) = data.dim();

    for sensor in 0..n_sensors {
        let column = data.column(sensor);

        // Calculate mean of non-missing values
        let valid_values: Vec<f64> = column.iter().filter(|&&x| !x.is_nan()).copied().collect();

        if valid_values.is_empty() {
            return Err(ImputationError::ValidationError(format!(
                "Sensor {} has no valid values",
                sensor
            )));
        }

        let mean = valid_values.iter().sum::<f64>() / valid_values.len() as f64;

        // Impute missing values with mean
        for time in 0..n_timesteps {
            if data[[time, sensor]].is_nan() {
                imputed[[time, sensor]] = mean;
            }
        }
    }

    Ok(imputed)
}

/// Simple forward fill imputation implementation
fn forward_fill_imputation(data: &Array2<f64>) -> Result<Array2<f64>, ImputationError> {
    let mut imputed = data.clone();
    let (n_timesteps, n_sensors) = data.dim();

    for sensor in 0..n_sensors {
        let mut last_valid_value = None;

        for time in 0..n_timesteps {
            let value = data[[time, sensor]];

            if value.is_nan() {
                if let Some(last_value) = last_valid_value {
                    imputed[[time, sensor]] = last_value;
                }
                // If no previous valid value, leave as NaN
            } else {
                last_valid_value = Some(value);
            }
        }
    }

    Ok(imputed)
}

/// Analyze missing data patterns
fn analyze_missing_data(data: &SimpleSensorData) {
    let total_values = data.measurements.len();
    let missing_count = data.measurements.iter().filter(|&&x| x.is_nan()).count();
    let missing_percentage = missing_count as f64 / total_values as f64 * 100.0;

    println!("📊 Missing Data Analysis:");
    println!("  - Total measurements: {}", total_values);
    println!("  - Missing measurements: {}", missing_count);
    println!("  - Missing percentage: {:.2}%", missing_percentage);

    // Analyze by sensor
    println!("  - Per-sensor missing data:");
    for (i, sensor_id) in data.sensor_ids.iter().enumerate() {
        let column = data.measurements.column(i);
        let sensor_missing = column.iter().filter(|&&x| x.is_nan()).count();
        let sensor_percentage = sensor_missing as f64 / column.len() as f64 * 100.0;

        if sensor_percentage > 5.0 {
            println!("    • {}: {:.1}% missing", sensor_id, sensor_percentage);
        }
    }
}

/// Compare imputation methods
fn compare_imputation_results(
    original: &Array2<f64>,
    mean_imputed: &Array2<f64>,
    forward_fill_imputed: &Array2<f64>,
) {
    let missing_positions: Vec<(usize, usize)> = (0..original.nrows())
        .flat_map(|i| (0..original.ncols()).map(move |j| (i, j)))
        .filter(|&(i, j)| original[[i, j]].is_nan())
        .collect();

    println!("📋 Sample Imputation Comparison (first 5 missing values):");
    println!(
        "    {:>8} {:>12} {:>12}",
        "Position", "Mean", "Forward Fill"
    );

    for (_i, &(row, col)) in missing_positions.iter().take(5).enumerate() {
        let mean_val = mean_imputed[[row, col]];
        let ff_val = forward_fill_imputed[[row, col]];

        println!(
            "    ({:2},{:2})   {:>8.2}    {:>8.2}",
            row, col, mean_val, ff_val
        );
    }

    let total_imputed = missing_positions.len();
    let mean_successful = missing_positions
        .iter()
        .filter(|&&(i, j)| !mean_imputed[[i, j]].is_nan())
        .count();
    let ff_successful = missing_positions
        .iter()
        .filter(|&&(i, j)| !forward_fill_imputed[[i, j]].is_nan())
        .count();

    println!("\n📈 Imputation Success Rates:");
    println!(
        "  - Mean imputation: {}/{} ({:.1}%)",
        mean_successful,
        total_imputed,
        mean_successful as f64 / total_imputed as f64 * 100.0
    );
    println!(
        "  - Forward fill: {}/{} ({:.1}%)",
        ff_successful,
        total_imputed,
        ff_successful as f64 / total_imputed as f64 * 100.0
    );
}

/// Demonstrate sensor data imputation
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌐 Simple Sensor Data Imputation Example");
    println!("=========================================");

    // Create sample data
    println!("\n📊 Creating synthetic sensor dataset...");
    let sensor_data = create_sample_sensor_data();

    println!("  - Sensors: {}", sensor_data.sensor_ids.len());
    println!("  - Time periods: {}", sensor_data.timestamps.len());
    println!("  - Data shape: {:?}", sensor_data.measurements.dim());

    // Analyze missing patterns
    println!("\n🔍 Analyzing missing data patterns:");
    analyze_missing_data(&sensor_data);

    // Perform different imputation methods
    println!("\n⚙️  Performing imputation methods:");

    println!("  1. Mean imputation...");
    let start_time = std::time::Instant::now();
    let mean_imputed = mean_imputation(&sensor_data.measurements)?;
    let mean_duration = start_time.elapsed();
    println!("     ✅ Completed in {:.2}ms", mean_duration.as_millis());

    println!("  2. Forward fill imputation...");
    let start_time = std::time::Instant::now();
    let forward_fill_imputed = forward_fill_imputation(&sensor_data.measurements)?;
    let ff_duration = start_time.elapsed();
    println!("     ✅ Completed in {:.2}ms", ff_duration.as_millis());

    // Compare results
    println!("\n🔧 Comparing imputation methods:");
    compare_imputation_results(
        &sensor_data.measurements,
        &mean_imputed,
        &forward_fill_imputed,
    );

    // Performance summary
    println!("\n⚡ Performance Summary:");
    println!("  - Mean imputation: {:.2}ms", mean_duration.as_millis());
    println!("  - Forward fill: {:.2}ms", ff_duration.as_millis());

    println!("\n✅ Sensor imputation example completed successfully!");
    println!("   Both methods successfully handled missing sensor data.");

    Ok(())
}
