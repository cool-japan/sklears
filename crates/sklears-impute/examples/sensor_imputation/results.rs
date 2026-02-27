
/// Sensor Imputation Results
#[derive(Debug, Clone)]
pub struct SensorImputationResult {
    /// Imputed sensor dataset
    pub imputed_data: SensorDataset,
    /// Imputation metadata
    pub metadata: ImputationMetadata,
    /// Sensor-specific validation
    pub sensor_validation: SensorValidation,
    /// Spatial validation results
    pub spatial_validation: SpatialValidation,
    /// Temporal validation results
    pub temporal_validation: TemporalValidation,
    /// Real-time performance metrics
    pub realtime_metrics: RealtimeMetrics,
    /// Edge computing efficiency
    pub edge_efficiency: EdgeEfficiency,
}

#[derive(Debug, Clone)]
pub struct SensorValidation {
    /// Physical plausibility checks
    pub physical_plausibility: PhysicalValidation,
    /// Cross-sensor consistency
    pub cross_sensor_consistency: ConsistencyValidation,
    /// Sensor drift compensation
    pub drift_compensation: DriftValidation,
    /// Calibration validation
    pub calibration_validation: CalibrationValidation,
}

#[derive(Debug, Clone)]
pub struct PhysicalValidation {
    pub range_violations: usize,
    pub rate_violations: usize,
    pub continuity_score: f64,
    pub smoothness_score: f64,
}

#[derive(Debug, Clone)]
pub struct ConsistencyValidation {
    pub cross_correlation_preservation: f64,
    pub redundant_sensor_agreement: f64,
    pub network_consistency_score: f64,
}

#[derive(Debug, Clone)]
pub struct DriftValidation {
    pub drift_detection_accuracy: f64,
    pub compensation_effectiveness: f64,
    pub trend_preservation: f64,
}

#[derive(Debug, Clone)]
pub struct CalibrationValidation {
    pub calibration_stability: f64,
    pub reference_agreement: f64,
    pub systematic_error_reduction: f64,
}

#[derive(Debug, Clone)]
pub struct SpatialValidation {
    /// Spatial interpolation accuracy
    pub interpolation_accuracy: f64,
    /// Kriging validation scores
    pub kriging_scores: KrigingValidation,
    /// Geographic continuity
    pub geographic_continuity: f64,
    /// Spatial correlation preservation
    pub correlation_preservation: f64,
}

#[derive(Debug, Clone)]
pub struct KrigingValidation {
    pub cross_validation_score: f64,
    pub prediction_variance: f64,
    pub variogram_fitting: f64,
}

#[derive(Debug, Clone)]
pub struct TemporalValidation {
    /// Time series properties preservation
    pub trend_preservation: f64,
    pub seasonality_preservation: f64,
    pub autocorrelation_preservation: f64,
    /// Forecasting accuracy
    pub forecasting_accuracy: f64,
    /// Change point detection
    pub change_point_detection: f64,
}

#[derive(Debug, Clone)]
pub struct RealtimeMetrics {
    /// Processing latency
    pub latency_percentiles: HashMap<String, Duration>,
    /// Throughput measurements
    pub throughput: f64, // measurements per second
    /// Buffer utilization
    pub buffer_utilization: f64, // percentage
    /// Queue depths
    pub queue_depths: HashMap<String, usize>,
}

#[derive(Debug, Clone)]
pub struct EdgeEfficiency {
    /// Compute resource utilization
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    /// Energy efficiency
    pub power_consumption: f64,
    pub battery_impact: f64,
    /// Network efficiency
    pub bandwidth_utilization: f64,
    pub compression_ratio: f64,
}

