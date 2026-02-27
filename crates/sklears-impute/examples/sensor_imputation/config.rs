
#[derive(Debug, Clone)]
pub enum NetworkTopology {
    Star,
    Mesh,
    Tree,
    Ring,
    Hybrid,
    AdHoc,
}

#[derive(Debug, Clone)]
pub enum CommunicationProtocol {
    WiFi,
    Bluetooth,
    ZigBee,
    LoRaWAN,
    Cellular,
    Ethernet,
    CAN,
    Modbus,
}

#[derive(Debug, Clone)]
pub enum RedundancyLevel {
    None,
    Dual,
    Triple,
    Quorum(usize),
}

#[derive(Debug, Clone)]
pub enum NetworkPartitioning {
    Geographic,
    SensorType,
    Temporal,
    Hierarchical,
    Adaptive,
}

#[derive(Debug, Clone)]
pub struct CalibrationModel {
    pub drift_rate: f64, // units per day
    pub calibration_interval: Duration,
    pub reference_standards: Vec<f64>,
    pub correction_function: CorrectionFunction,
}

#[derive(Debug, Clone)]
pub enum CorrectionFunction {
    Linear,
    Polynomial(usize),
    Exponential,
    Custom,
}

#[derive(Debug, Clone)]
pub struct PowerConstraints {
    pub max_power_consumption: f64, // watts
    pub battery_capacity: f64,      // watt-hours
    pub sleep_mode_available: bool,
    pub duty_cycle_limit: f64, // fraction of time active
}

#[derive(Debug, Clone)]
pub struct TemperatureModel {
    pub coefficients: Vec<f64>,
    pub reference_temperature: f64,
    pub compensation_enabled: bool,
}

#[derive(Debug, Clone)]
pub struct HumidityModel {
    pub sensitivity: f64,
    pub saturation_effects: bool,
    pub hysteresis_correction: bool,
}

#[derive(Debug, Clone)]
pub struct PressureModel {
    pub altitude_correction: bool,
    pub barometric_effects: bool,
    pub pressure_coefficients: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EMIModel {
    pub interference_sources: Vec<String>,
    pub frequency_bands: Vec<(f64, f64)>,
    pub mitigation_strategies: Vec<EMIMitigation>,
}

#[derive(Debug, Clone)]
pub enum EMIMitigation {
    Filtering,
    Shielding,
    FrequencyHopping,
    ErrorCorrection,
}

#[derive(Debug, Clone)]
pub struct SeasonalModel {
    pub periods: Vec<Duration>,
    pub amplitudes: Vec<f64>,
    pub phase_shifts: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct WeatherModel {
    pub precipitation_effects: bool,
    pub wind_effects: bool,
    pub solar_radiation_effects: bool,
    pub weather_stations: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SeasonalityParams {
    pub detection_method: SeasonalityDetection,
    pub min_period: Duration,
    pub max_period: Duration,
    pub significance_threshold: f64,
}

#[derive(Debug, Clone)]
pub enum SeasonalityDetection {
    Autocorrelation,
    Periodogram,
    STL,
    Fourier,
}

#[derive(Debug, Clone)]
pub struct LagAnalysis {
    pub max_lag: Duration,
    pub correlation_threshold: f64,
    pub cross_correlation_analysis: bool,
}

#[derive(Debug, Clone)]
pub enum SpatialInterpolation {
    InverseDistanceWeighting,
    Kriging,
    Spline,
    NearestNeighbor,
    RadialBasisFunction,
}

#[derive(Debug, Clone)]
pub struct KrigingParams {
    pub variogram_model: VariogramModel,
    pub nugget: f64,
    pub sill: f64,
    pub range: f64,
}

#[derive(Debug, Clone)]
pub enum VariogramModel {
    Spherical,
    Exponential,
    Gaussian,
    Linear,
    Power,
}

#[derive(Debug, Clone)]
pub struct SpatialCorrelationModel {
    pub correlation_function: CorrelationFunction,
    pub decay_rate: f64,
    pub anisotropy: Option<AnisotropyParams>,
}

#[derive(Debug, Clone)]
pub enum CorrelationFunction {
    Exponential,
    Gaussian,
    Matern,
    PowerLaw,
}

#[derive(Debug, Clone)]
pub struct AnisotropyParams {
    pub major_axis_direction: f64, // degrees
    pub anisotropy_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct GeographicClustering {
    pub clustering_algorithm: ClusteringAlgorithm,
    pub max_cluster_size: usize,
    pub distance_threshold: f64,
}

#[derive(Debug, Clone)]
pub enum ClusteringAlgorithm {
    KMeans,
    DBSCAN,
    Hierarchical,
    SpatialKMeans,
}

#[derive(Debug, Clone)]
pub struct TopographyModel {
    pub elevation_effects: bool,
    pub slope_effects: bool,
    pub aspect_effects: bool,
    pub terrain_roughness: f64,
}

#[derive(Debug, Clone)]
pub struct ComputeResources {
    pub cpu_cores: usize,
    pub cpu_frequency: f64,      // GHz
    pub available_memory: usize, // MB
    pub storage_capacity: usize, // GB
}

#[derive(Debug, Clone)]
pub struct MemoryConstraints {
    pub max_buffer_size: usize, // MB
    pub streaming_window_size: usize,
    pub cache_size: usize, // MB
}

#[derive(Debug, Clone)]
pub struct BatteryConstraints {
    pub remaining_capacity: f64,  // percentage
    pub discharge_rate: f64,      // per hour
    pub low_power_threshold: f64, // percentage
}

#[derive(Debug, Clone)]
pub struct BandwidthConstraints {
    pub upload_bandwidth: f64,   // Mbps
    pub download_bandwidth: f64, // Mbps
    pub data_compression: bool,
    pub priority_queuing: bool,
}

#[derive(Debug, Clone)]
pub enum ProcessingStrategy {
    EdgeOnly,
    CloudOnly,
    Hybrid,
    Adaptive,
}

#[derive(Debug, Clone)]
pub struct AnomalyThresholds {
    pub statistical_threshold: f64, // standard deviations
    pub percentage_threshold: f64,  // percentage change
    pub absolute_threshold: HashMap<SensorType, f64>,
    pub duration_threshold: Duration,
}

#[derive(Debug, Clone)]
pub struct ValidationRules {
    pub range_checks: bool,
    pub rate_of_change_limits: HashMap<SensorType, f64>,
    pub cross_sensor_validation: bool,
    pub temporal_consistency_checks: bool,
}

#[derive(Debug, Clone)]
pub enum UncertaintyMethods {
    Bootstrap,
    Bayesian,
    Ensemble,
    Conformal,
}

#[derive(Debug, Clone)]
pub struct QualityScoring {
    pub accuracy_weight: f64,
    pub completeness_weight: f64,
    pub consistency_weight: f64,
    pub timeliness_weight: f64,
}

#[derive(Debug, Clone)]
pub struct DriftDetection {
    pub detection_method: DriftDetectionMethod,
    pub sensitivity: f64,
    pub adaptation_rate: f64,
}

#[derive(Debug, Clone)]
pub enum DriftDetectionMethod {
    CUSUM,
    PageHinkley,
    ADWIN,
    Statistical,
}

#[derive(Debug, Clone)]
pub enum BufferingStrategy {
    FIFO,
    LIFO,
    Priority,
    Adaptive,
}

#[derive(Debug, Clone)]
pub struct PriorityScheduling {
    pub sensor_priorities: HashMap<SensorType, usize>,
    pub emergency_override: bool,
    pub load_balancing: bool,
}

#[derive(Debug, Clone)]
pub enum DegradationStrategies {
    ReducedSampling,
    SimplifiedModels,
    CacheResults,
    SkipNonCritical,
}

