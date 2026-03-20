/// Sensor Network Dataset
#[derive(Debug, Clone)]
pub struct SensorDataset {
    /// Sensor measurements (sensor_id, time, value)
    pub measurements: Array3<f64>, // (n_sensors, n_timesteps, n_features)
    /// Sensor locations (latitude, longitude, elevation)
    pub locations: Array2<f64>, // (n_sensors, 3)
    /// Sensor types and specifications
    pub sensor_specs: Vec<SensorSpec>,
    /// Timestamps for measurements
    pub timestamps: Vec<SystemTime>,
    /// Environmental conditions
    pub environmental_data: Array2<f64>, // (n_timesteps, n_env_variables)
    /// Network connectivity
    pub connectivity: Array2<f64>, // (n_sensors, n_sensors)
    /// Quality indicators
    pub quality_flags: Array3<bool>, // (n_sensors, n_timesteps, n_features)
}

#[derive(Debug, Clone)]
pub struct SensorSpec {
    pub sensor_id: String,
    pub sensor_type: SensorType,
    pub manufacturer: String,
    pub model: String,
    pub accuracy: f64,
    pub range: (f64, f64),
    pub sampling_rate: f64,
    pub power_consumption: f64,
    pub installation_date: SystemTime,
    pub last_calibration: SystemTime,
}

/// Missing Data Analysis for Sensor Networks
#[derive(Debug, Clone)]
pub struct SensorMissingPattern {
    /// Missing data by sensor
    pub missing_by_sensor: HashMap<String, f64>,
    /// Missing data by sensor type
    pub missing_by_type: HashMap<SensorType, f64>,
    /// Missing data by time period
    pub missing_by_period: Vec<(SystemTime, f64)>,
    /// Spatial clustering of missing data
    pub spatial_clusters: Vec<SpatialCluster>,
    /// Temporal patterns of failures
    pub temporal_patterns: Vec<TemporalPattern>,
    /// Correlation between sensor failures
    pub failure_correlations: Array2<f64>,
}

#[derive(Debug, Clone)]
pub struct SpatialCluster {
    pub cluster_id: String,
    pub affected_sensors: Vec<String>,
    pub geographic_center: (f64, f64),
    pub radius: f64,
    pub missing_percentage: f64,
    pub likely_cause: String,
}

#[derive(Debug, Clone)]
pub struct TemporalPattern {
    pub pattern_id: String,
    pub time_range: (SystemTime, SystemTime),
    pub affected_sensors: Vec<String>,
    pub pattern_type: PatternType,
    pub recurrence: Option<Duration>,
}

#[derive(Debug, Clone)]
pub enum PatternType {
    Maintenance,
    WeatherEvent,
    PowerOutage,
    NetworkFailure,
    SensorDrift,
    Interference,
    Unknown,
}
