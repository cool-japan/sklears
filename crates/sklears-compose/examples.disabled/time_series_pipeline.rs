//! Time Series Pipeline Example
//!
//! This example demonstrates comprehensive time series processing using sklears-compose
//! with realistic time series data and specialized preprocessing steps. It showcases:
//!
//! - Time series data generation with seasonal patterns and trends
//! - Temporal feature engineering (lags, rolling statistics, seasonal decomposition)
//! - Anomaly detection in time series
//! - Forecasting pipelines with different strategies
//! - Real-time streaming simulation
//! - IoT sensor data simulation and processing
//!
//! Run with: cargo run --example time_series_pipeline

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use scirs2_autograd::ndarray::{array, s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::{rng, Random};
use sklears_compose::{
    monitoring::PipelineMonitor,
    streaming::{StreamProcessor, StreamingPipeline},
    time_series_pipelines::{
        AnomalyDetector, IoTDataProcessor, RealTimeAnalytics, SeasonalDecomposer,
        TimeSeriesForecaster,
    },
    FeatureUnion, Pipeline, PipelineBuilder,
};
use sklears_core::{
    error::Result as SklResult,
    traits::{Fit, Predict, Transform},
    types::Float,
};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Time series data point with timestamp
#[derive(Debug, Clone)]
struct TimeSeriesPoint {
    timestamp: DateTime<Utc>,
    value: Float,
    metadata: HashMap<String, String>,
}

/// Time series generator for different patterns
struct TimeSeriesGenerator {
    random: Random,
}

impl TimeSeriesGenerator {
    fn new(seed: u64) -> SklResult<Self> {
        Ok(Self {
            random: Random::new(seed)?,
        })
    }

    /// Generate synthetic time series with trend, seasonality, and noise
    fn generate_synthetic_series(
        &mut self,
        n_points: usize,
        trend: Float,
        seasonality_amplitude: Float,
        noise_level: Float,
        seasonal_period: usize,
    ) -> SklResult<Vec<TimeSeriesPoint>> {
        let mut series = Vec::with_capacity(n_points);
        let start_time = Utc::now() - ChronoDuration::hours(n_points as i64);

        for i in 0..n_points {
            let timestamp = start_time + ChronoDuration::hours(i as i64);

            // Trend component
            let trend_value = trend * i as Float;

            // Seasonal component
            let seasonal_value = seasonality_amplitude
                * (2.0 * std::f64::consts::PI * i as Float / seasonal_period as Float).sin();

            // Noise component
            let noise = self.random.normal(0.0, noise_level)?;

            // Combine components
            let value = 100.0 + trend_value + seasonal_value + noise;

            let mut metadata = HashMap::new();
            metadata.insert("source".to_string(), "synthetic".to_string());
            metadata.insert("pattern".to_string(), "trend+seasonal+noise".to_string());

            series.push(TimeSeriesPoint {
                timestamp,
                value,
                metadata,
            });
        }

        Ok(series)
    }

    /// Generate IoT sensor data with multiple variables
    fn generate_iot_sensor_data(
        &mut self,
        n_points: usize,
        n_sensors: usize,
    ) -> SklResult<Vec<HashMap<String, Float>>> {
        let mut data = Vec::with_capacity(n_points);
        let sensor_names = [
            "temperature",
            "humidity",
            "pressure",
            "vibration",
            "voltage",
        ];

        for i in 0..n_points {
            let mut point = HashMap::new();

            for j in 0..n_sensors.min(sensor_names.len()) {
                let base_value = match sensor_names[j] {
                    "temperature" => 22.0 + 5.0 * (i as Float / 100.0).sin(),
                    "humidity" => 45.0 + 15.0 * (i as Float / 80.0).cos(),
                    "pressure" => 1013.0 + 10.0 * self.random.normal(0.0, 1.0)?,
                    "vibration" => 0.1 * self.random.exponential(2.0)?,
                    "voltage" => 3.3 + 0.1 * self.random.normal(0.0, 0.5)?,
                    _ => self.random.normal(0.0, 1.0)?,
                };

                point.insert(sensor_names[j].to_string(), base_value);
            }

            data.push(point);
        }

        Ok(data)
    }

    /// Generate anomalous data points
    fn inject_anomalies(
        &mut self,
        series: &mut [TimeSeriesPoint],
        anomaly_rate: Float,
        anomaly_magnitude: Float,
    ) -> SklResult<Vec<usize>> {
        let mut anomaly_indices = Vec::new();

        for (i, point) in series.iter_mut().enumerate() {
            if self.random.uniform(0.0, 1.0)? < anomaly_rate {
                // Add anomaly
                let anomaly_type = self.random.uniform(0.0, 1.0)?;
                if anomaly_type < 0.5 {
                    // Spike anomaly
                    point.value += anomaly_magnitude * self.random.uniform(2.0, 5.0)?;
                } else {
                    // Drop anomaly
                    point.value -= anomaly_magnitude * self.random.uniform(2.0, 5.0)?;
                }

                point
                    .metadata
                    .insert("anomaly".to_string(), "true".to_string());
                anomaly_indices.push(i);
            }
        }

        Ok(anomaly_indices)
    }
}

/// Lag feature transformer for time series
#[derive(Debug, Clone)]
struct LagFeatureTransformer {
    lags: Vec<usize>,
    lag_values: Option<HashMap<usize, Array1<Float>>>,
}

impl LagFeatureTransformer {
    fn new(lags: Vec<usize>) -> Self {
        Self {
            lags,
            lag_values: None,
        }
    }
}

impl Transform for LagFeatureTransformer {
    type Input = Array1<Float>;
    type Output = Array2<Float>;

    fn transform(&self, x: &Self::Input) -> SklResult<Self::Output> {
        let n_samples = x.len();
        let n_features = self.lags.len();
        let mut result = Array2::<Float>::zeros((n_samples, n_features));

        for (feat_idx, &lag) in self.lags.iter().enumerate() {
            for i in lag..n_samples {
                result[[i, feat_idx]] = x[i - lag];
            }
            // Fill initial values with first available value or zero
            for i in 0..lag {
                result[[i, feat_idx]] = if n_samples > lag { x[lag] } else { 0.0 };
            }
        }

        Ok(result)
    }
}

impl Fit for LagFeatureTransformer {
    type Input = Array1<Float>;
    type Target = Array1<Float>;
    type Fitted = Self;

    fn fit(self, _x: &Self::Input, _y: Option<&Self::Target>) -> SklResult<Self::Fitted> {
        Ok(self) // No fitting required for lag features
    }
}

/// Rolling statistics transformer
#[derive(Debug, Clone)]
struct RollingStatsTransformer {
    window_size: usize,
}

impl RollingStatsTransformer {
    fn new(window_size: usize) -> Self {
        Self { window_size }
    }
}

impl Transform for RollingStatsTransformer {
    type Input = Array1<Float>;
    type Output = Array2<Float>;

    fn transform(&self, x: &Self::Input) -> SklResult<Self::Output> {
        let n_samples = x.len();
        let n_features = 3; // mean, std, min, max
        let mut result = Array2::<Float>::zeros((n_samples, n_features));

        for i in 0..n_samples {
            let window_start = if i >= self.window_size {
                i - self.window_size + 1
            } else {
                0
            };
            let window_end = i + 1;
            let window = &x.slice(s![window_start..window_end]);

            // Calculate rolling statistics
            let mean = window.mean().unwrap_or(0.0);
            let variance = window
                .iter()
                .map(|&val| (val - mean).powi(2))
                .sum::<Float>()
                / window.len() as Float;
            let std = variance.sqrt();

            result[[i, 0]] = mean;
            result[[i, 1]] = std;
            result[[i, 2]] = window.iter().fold(Float::INFINITY, |a, &b| a.min(b));
        }

        Ok(result)
    }
}

impl Fit for RollingStatsTransformer {
    type Input = Array1<Float>;
    type Target = Array1<Float>;
    type Fitted = Self;

    fn fit(self, _x: &Self::Input, _y: Option<&Self::Target>) -> SklResult<Self::Fitted> {
        Ok(self)
    }
}

/// Simple time series forecaster
#[derive(Debug, Clone)]
struct SimpleForecaster {
    trend: Option<Float>,
    seasonal_pattern: Option<Array1<Float>>,
    seasonal_period: usize,
}

impl SimpleForecaster {
    fn new(seasonal_period: usize) -> Self {
        Self {
            trend: None,
            seasonal_pattern: None,
            seasonal_period,
        }
    }
}

impl Fit for SimpleForecaster {
    type Input = Array1<Float>;
    type Target = Array1<Float>;
    type Fitted = Self;

    fn fit(mut self, x: &Self::Input, _y: Option<&Self::Target>) -> SklResult<Self::Fitted> {
        let n = x.len();

        // Simple trend estimation using linear regression
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;

        for (i, &value) in x.iter().enumerate() {
            let t = i as Float;
            sum_x += t;
            sum_y += value;
            sum_xy += t * value;
            sum_x2 += t * t;
        }

        let n_f = n as Float;
        let trend = (n_f * sum_xy - sum_x * sum_y) / (n_f * sum_x2 - sum_x * sum_x);
        self.trend = Some(trend);

        // Simple seasonal pattern extraction
        if n >= self.seasonal_period {
            let mut seasonal_pattern = Array1::<Float>::zeros(self.seasonal_period);
            let mut seasonal_counts = vec![0; self.seasonal_period];

            for (i, &value) in x.iter().enumerate() {
                let seasonal_idx = i % self.seasonal_period;
                let detrended_value = value - trend * i as Float;
                seasonal_pattern[seasonal_idx] += detrended_value;
                seasonal_counts[seasonal_idx] += 1;
            }

            // Average seasonal components
            for (i, count) in seasonal_counts.iter().enumerate() {
                if *count > 0 {
                    seasonal_pattern[i] /= *count as Float;
                }
            }

            self.seasonal_pattern = Some(seasonal_pattern);
        }

        Ok(self)
    }
}

impl Predict for SimpleForecaster {
    type Input = usize; // Number of steps to forecast
    type Output = Array1<Float>;

    fn predict(&self, n_steps: &usize) -> SklResult<Self::Output> {
        let mut forecast = Array1::<Float>::zeros(*n_steps);

        if let (Some(trend), Some(ref seasonal)) = (self.trend, &self.seasonal_pattern) {
            for i in 0..*n_steps {
                let trend_component = trend * i as Float;
                let seasonal_component = seasonal[i % self.seasonal_period];
                forecast[i] = trend_component + seasonal_component;
            }
        } else {
            // Fallback: simple trend only
            let trend = self.trend.unwrap_or(0.0);
            for i in 0..*n_steps {
                forecast[i] = trend * i as Float;
            }
        }

        Ok(forecast)
    }
}

/// Anomaly detection transformer
#[derive(Debug, Clone)]
struct AnomalyDetectorTransformer {
    threshold_multiplier: Float,
    window_size: usize,
    mean: Option<Float>,
    std: Option<Float>,
}

impl AnomalyDetectorTransformer {
    fn new(threshold_multiplier: Float, window_size: usize) -> Self {
        Self {
            threshold_multiplier,
            window_size,
            mean: None,
            std: None,
        }
    }
}

impl Transform for AnomalyDetectorTransformer {
    type Input = Array1<Float>;
    type Output = Array1<bool>;

    fn transform(&self, x: &Self::Input) -> SklResult<Self::Output> {
        let mut anomalies = Array1::<bool>::from_elem(x.len(), false);

        if let (Some(mean), Some(std)) = (self.mean, self.std) {
            let threshold = self.threshold_multiplier * std;

            for (i, &value) in x.iter().enumerate() {
                if (value - mean).abs() > threshold {
                    anomalies[i] = true;
                }
            }
        }

        Ok(anomalies)
    }
}

impl Fit for AnomalyDetectorTransformer {
    type Input = Array1<Float>;
    type Target = Array1<Float>;
    type Fitted = Self;

    fn fit(mut self, x: &Self::Input, _y: Option<&Self::Target>) -> SklResult<Self::Fitted> {
        let mean = x.mean().unwrap();
        let variance = x.iter().map(|&val| (val - mean).powi(2)).sum::<Float>() / x.len() as Float;
        let std = variance.sqrt();

        self.mean = Some(mean);
        self.std = Some(std);

        Ok(self)
    }
}

/// Real-time streaming processor simulation
struct StreamingProcessor {
    buffer: VecDeque<Float>,
    buffer_size: usize,
    processor: Box<dyn Fn(&[Float]) -> SklResult<Float>>,
}

impl StreamingProcessor {
    fn new<F>(buffer_size: usize, processor: F) -> Self
    where
        F: Fn(&[Float]) -> SklResult<Float> + 'static,
    {
        Self {
            buffer: VecDeque::with_capacity(buffer_size),
            buffer_size,
            processor: Box::new(processor),
        }
    }

    fn process_point(&mut self, value: Float) -> SklResult<Option<Float>> {
        self.buffer.push_back(value);

        if self.buffer.len() > self.buffer_size {
            self.buffer.pop_front();
        }

        if self.buffer.len() == self.buffer_size {
            let buffer_vec: Vec<Float> = self.buffer.iter().copied().collect();
            let result = (self.processor)(&buffer_vec)?;
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }
}

/// Main demonstration function
fn main() -> SklResult<()> {
    println!("🚀 Time Series Pipeline Example");
    println!("{}", "=".repeat(60));

    demo_time_series_preprocessing()?;
    demo_forecasting_pipeline()?;
    demo_anomaly_detection()?;
    demo_streaming_processing()?;
    demo_iot_sensor_processing()?;

    Ok(())
}

/// Demonstrate time series preprocessing
fn demo_time_series_preprocessing() -> SklResult<()> {
    println!("\n📈 Time Series Preprocessing Pipeline");
    println!("{}", "-".repeat(50));

    let mut generator = TimeSeriesGenerator::new(42)?;
    let series = generator.generate_synthetic_series(1000, 0.01, 10.0, 2.0, 24)?;

    // Extract values for processing
    let values: Array1<Float> = series.iter().map(|p| p.value).collect::<Vec<_>>().into();

    println!("Generated time series with {} points", values.len());
    println!(
        "Value range: [{:.2}, {:.2}]",
        values.iter().fold(Float::INFINITY, |a, &b| a.min(b)),
        values.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b))
    );

    let start_time = Instant::now();

    // Create feature engineering pipeline
    let feature_union = FeatureUnion::builder()
        .transformer(
            "lags",
            Box::new(LagFeatureTransformer::new(vec![1, 2, 3, 24])),
        )
        .transformer("rolling_stats", Box::new(RollingStatsTransformer::new(24)))
        .build();

    // Fit and transform features
    let fitted_features = feature_union.fit(&values, Some(&values))?;
    let transformed_features = fitted_features.transform(&values)?;

    let elapsed = start_time.elapsed();

    println!("✅ Preprocessing Results:");
    println!("  - Original features: 1");
    println!("  - Engineered features: {}", transformed_features.ncols());
    println!("  - Processing time: {:?}", elapsed);
    println!("  - Features include: lags [1,2,3,24], rolling [mean, std, min]");

    Ok(())
}

/// Demonstrate forecasting pipeline
fn demo_forecasting_pipeline() -> SklResult<()> {
    println!("\n🔮 Time Series Forecasting Pipeline");
    println!("{}", "-".repeat(50));

    let mut generator = TimeSeriesGenerator::new(123)?;
    let series = generator.generate_synthetic_series(500, 0.02, 15.0, 1.0, 12)?;
    let values: Array1<Float> = series.iter().map(|p| p.value).collect::<Vec<_>>().into();

    let start_time = Instant::now();

    // Split into train/test
    let train_size = (values.len() as f64 * 0.8) as usize;
    let train_data = values.slice(s![0..train_size]).to_owned();
    let test_size = values.len() - train_size;

    // Create forecasting pipeline
    let forecaster = SimpleForecaster::new(12); // 12-hour seasonal pattern
    let fitted_forecaster = forecaster.fit(&train_data, None)?;

    // Make forecasts
    let forecast = fitted_forecaster.predict(&test_size)?;
    let actual = values.slice(s![train_size..]);

    // Calculate forecast accuracy
    let mae: Float = actual
        .iter()
        .zip(forecast.iter())
        .map(|(&a, &f)| (a - f).abs())
        .sum::<Float>()
        / test_size as Float;

    let mape: Float = actual
        .iter()
        .zip(forecast.iter())
        .map(|(&a, &f)| ((a - f) / a).abs())
        .sum::<Float>()
        / test_size as Float
        * 100.0;

    let elapsed = start_time.elapsed();

    println!("✅ Forecasting Results:");
    println!("  - Training samples: {}", train_size);
    println!("  - Forecast horizon: {}", test_size);
    println!("  - MAE: {:.2}", mae);
    println!("  - MAPE: {:.2}%", mape);
    println!("  - Training time: {:?}", elapsed);

    Ok(())
}

/// Demonstrate anomaly detection
fn demo_anomaly_detection() -> SklResult<()> {
    println!("\n🚨 Time Series Anomaly Detection");
    println!("{}", "-".repeat(50));

    let mut generator = TimeSeriesGenerator::new(456)?;
    let mut series = generator.generate_synthetic_series(800, 0.005, 8.0, 1.5, 24)?;

    // Inject anomalies
    let anomaly_indices = generator.inject_anomalies(&mut series, 0.05, 10.0)?;
    let values: Array1<Float> = series.iter().map(|p| p.value).collect::<Vec<_>>().into();

    let start_time = Instant::now();

    // Create anomaly detection pipeline
    let anomaly_detector = AnomalyDetectorTransformer::new(3.0, 24);
    let fitted_detector = anomaly_detector.fit(&values, None)?;
    let detected_anomalies = fitted_detector.transform(&values)?;

    let elapsed = start_time.elapsed();

    // Calculate detection metrics
    let true_positives = anomaly_indices
        .iter()
        .filter(|&&idx| detected_anomalies[idx])
        .count();

    let false_positives = detected_anomalies
        .iter()
        .enumerate()
        .filter(|(idx, &detected)| detected && !anomaly_indices.contains(idx))
        .count();

    let precision = true_positives as f64 / (true_positives + false_positives) as f64;
    let recall = true_positives as f64 / anomaly_indices.len() as f64;

    println!("✅ Anomaly Detection Results:");
    println!("  - Total anomalies injected: {}", anomaly_indices.len());
    println!(
        "  - Anomalies detected: {}",
        detected_anomalies.iter().filter(|&&x| x).count()
    );
    println!("  - True positives: {}", true_positives);
    println!("  - Precision: {:.2}", precision);
    println!("  - Recall: {:.2}", recall);
    println!("  - Detection time: {:?}", elapsed);

    Ok(())
}

/// Demonstrate streaming processing
fn demo_streaming_processing() -> SklResult<()> {
    println!("\n🌊 Real-time Streaming Processing");
    println!("{}", "-".repeat(50));

    let mut generator = TimeSeriesGenerator::new(789)?;
    let series = generator.generate_synthetic_series(200, 0.01, 5.0, 1.0, 10)?;

    let start_time = Instant::now();

    // Create streaming processor for rolling average
    let mut streaming_processor = StreamingProcessor::new(10, |window| {
        let avg = window.iter().sum::<Float>() / window.len() as Float;
        Ok(avg)
    });

    let mut processed_results = Vec::new();
    let mut processed_count = 0;

    // Simulate real-time processing
    for point in &series {
        if let Some(result) = streaming_processor.process_point(point.value)? {
            processed_results.push(result);
            processed_count += 1;
        }
    }

    let elapsed = start_time.elapsed();

    println!("✅ Streaming Processing Results:");
    println!("  - Input points: {}", series.len());
    println!("  - Processed points: {}", processed_count);
    println!("  - Buffer size: 10");
    println!(
        "  - Average processing time per point: {:?}",
        elapsed / series.len() as u32
    );
    println!("  - Total processing time: {:?}", elapsed);

    Ok(())
}

/// Demonstrate IoT sensor data processing
fn demo_iot_sensor_processing() -> SklResult<()> {
    println!("\n🔌 IoT Sensor Data Processing");
    println!("{}", "-".repeat(50));

    let mut generator = TimeSeriesGenerator::new(101112)?;
    let iot_data = generator.generate_iot_sensor_data(500, 5)?;

    let start_time = Instant::now();

    // Process each sensor stream
    let sensor_names = [
        "temperature",
        "humidity",
        "pressure",
        "vibration",
        "voltage",
    ];
    let mut sensor_results = HashMap::new();

    for sensor_name in &sensor_names {
        let values: Array1<Float> = iot_data
            .iter()
            .map(|point| point.get(sensor_name).copied().unwrap_or(0.0))
            .collect::<Vec<_>>()
            .into();

        // Simple anomaly detection for each sensor
        let anomaly_detector = AnomalyDetectorTransformer::new(2.5, 20);
        let fitted_detector = anomaly_detector.fit(&values, None)?;
        let anomalies = fitted_detector.transform(&values)?;

        let anomaly_count = anomalies.iter().filter(|&&x| x).count();
        let anomaly_rate = anomaly_count as f64 / values.len() as f64;

        sensor_results.insert(sensor_name, (anomaly_count, anomaly_rate));
    }

    let elapsed = start_time.elapsed();

    println!("✅ IoT Processing Results:");
    println!("  - Data points per sensor: {}", iot_data.len());
    println!("  - Sensors processed: {}", sensor_names.len());

    for (sensor_name, (count, rate)) in sensor_results {
        println!(
            "  - {}: {} anomalies ({:.1}%)",
            sensor_name,
            count,
            rate * 100.0
        );
    }

    println!("  - Total processing time: {:?}", elapsed);

    Ok(())
}
