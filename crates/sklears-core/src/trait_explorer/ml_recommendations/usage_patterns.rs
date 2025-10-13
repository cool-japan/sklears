use std::collections::{HashMap, BTreeMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use serde::{Serialize, Deserialize};
use scirs2_core::ndarray::{Array1, Array2, array};
use scirs2_core::random::{Random, rng};
use scirs2_core::error::CoreError;

use super::data_types::*;
use super::neural_networks::NeuralEmbeddingModel;
use super::feature_extraction::TraitFeatureExtractor;

/// Configuration for usage pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsagePatternConfig {
    pub temporal_window_days: u32,
    pub min_usage_threshold: u32,
    pub decay_factor: f64,
    pub pattern_detection_sensitivity: f64,
    pub clustering_threshold: f64,
    pub trend_analysis_enabled: bool,
    pub seasonal_analysis_enabled: bool,
    pub anomaly_detection_enabled: bool,
    pub prediction_horizon_days: u32,
}

impl Default for UsagePatternConfig {
    fn default() -> Self {
        Self {
            temporal_window_days: 30,
            min_usage_threshold: 5,
            decay_factor: 0.95,
            pattern_detection_sensitivity: 0.7,
            clustering_threshold: 0.8,
            trend_analysis_enabled: true,
            seasonal_analysis_enabled: true,
            anomaly_detection_enabled: true,
            prediction_horizon_days: 7,
        }
    }
}

/// Represents a usage event with timestamp and context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageEvent {
    pub trait_name: String,
    pub timestamp: u64,
    pub context: TraitContext,
    pub user_session: Option<String>,
    pub success: bool,
    pub duration_ms: Option<u64>,
    pub metadata: HashMap<String, String>,
}

/// Temporal pattern representing usage trends over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPattern {
    pub trait_name: String,
    pub hourly_distribution: Array1<f64>,
    pub daily_distribution: Array1<f64>,
    pub weekly_distribution: Array1<f64>,
    pub monthly_distribution: Array1<f64>,
    pub trend_slope: f64,
    pub seasonality_strength: f64,
    pub volatility: f64,
}

/// User behavior pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserBehaviorPattern {
    pub user_id: String,
    pub trait_preferences: HashMap<String, f64>,
    pub usage_frequency: f64,
    pub session_duration_avg: f64,
    pub success_rate: f64,
    pub learning_curve_slope: f64,
    pub trait_transition_matrix: HashMap<(String, String), f64>,
    pub preferred_contexts: Vec<TraitContext>,
}

/// Seasonal pattern detection and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalPattern {
    pub pattern_type: SeasonalType,
    pub amplitude: f64,
    pub period_days: f64,
    pub phase_offset: f64,
    pub confidence: f64,
    pub detected_peaks: Vec<u64>,
    pub detected_troughs: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SeasonalType {
    Daily,
    Weekly,
    Monthly,
    Yearly,
    Custom(f64),
}

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageAnomaly {
    pub timestamp: u64,
    pub trait_name: String,
    pub anomaly_type: AnomalyType,
    pub severity: f64,
    pub expected_value: f64,
    pub actual_value: f64,
    pub confidence: f64,
    pub context: Option<TraitContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    UnusualSpike,
    UnusualDrop,
    TemporalShift,
    ContextualAnomaly,
    BehaviorDeviation,
}

/// Main usage pattern analyzer
pub struct UsagePatternAnalyzer {
    usage_history: VecDeque<UsageEvent>,
    temporal_patterns: HashMap<String, TemporalPattern>,
    user_patterns: HashMap<String, UserBehaviorPattern>,
    seasonal_patterns: HashMap<String, Vec<SeasonalPattern>>,
    detected_anomalies: Vec<UsageAnomaly>,
    config: UsagePatternConfig,
    feature_extractor: TraitFeatureExtractor,
    embedding_model: Option<NeuralEmbeddingModel>,
    pattern_cache: HashMap<String, Vec<f64>>,
}

impl UsagePatternAnalyzer {
    pub fn new(config: UsagePatternConfig, feature_extractor: TraitFeatureExtractor) -> Self {
        Self {
            usage_history: VecDeque::new(),
            temporal_patterns: HashMap::new(),
            user_patterns: HashMap::new(),
            seasonal_patterns: HashMap::new(),
            detected_anomalies: Vec::new(),
            config,
            feature_extractor,
            embedding_model: None,
            pattern_cache: HashMap::new(),
        }
    }

    /// Record a new usage event
    pub fn record_usage(&mut self, event: UsageEvent) -> Result<(), CoreError> {
        // Add to history with size limit
        self.usage_history.push_back(event.clone());

        // Keep only events within temporal window
        let cutoff_time = self.current_timestamp() - (self.config.temporal_window_days as u64 * 24 * 3600);
        while let Some(front) = self.usage_history.front() {
            if front.timestamp < cutoff_time {
                self.usage_history.pop_front();
            } else {
                break;
            }
        }

        // Update patterns incrementally
        self.update_temporal_patterns(&event)?;
        self.update_user_patterns(&event)?;

        // Detect anomalies for this event
        if self.config.anomaly_detection_enabled {
            if let Some(anomaly) = self.detect_anomaly(&event)? {
                self.detected_anomalies.push(anomaly);
            }
        }

        Ok(())
    }

    /// Analyze temporal patterns for a specific trait
    pub fn analyze_temporal_patterns(&mut self, trait_name: &str) -> Result<TemporalPattern, CoreError> {
        let events: Vec<_> = self.usage_history
            .iter()
            .filter(|e| e.trait_name == trait_name)
            .collect();

        if events.len() < self.config.min_usage_threshold as usize {
            return Err(CoreError::new("Insufficient usage data for pattern analysis"));
        }

        // Initialize distribution arrays
        let mut hourly_dist = Array1::zeros(24);
        let mut daily_dist = Array1::zeros(31);
        let mut weekly_dist = Array1::zeros(7);
        let mut monthly_dist = Array1::zeros(12);

        // Populate distributions
        for event in &events {
            let datetime = self.timestamp_to_datetime(event.timestamp);
            hourly_dist[datetime.hour as usize] += 1.0;
            daily_dist[(datetime.day - 1) as usize] += 1.0;
            weekly_dist[datetime.weekday as usize] += 1.0;
            monthly_dist[(datetime.month - 1) as usize] += 1.0;
        }

        // Normalize distributions
        let total_events = events.len() as f64;
        hourly_dist /= total_events;
        daily_dist /= total_events;
        weekly_dist /= total_events;
        monthly_dist /= total_events;

        // Calculate trend slope using linear regression
        let timestamps: Vec<f64> = events.iter().map(|e| e.timestamp as f64).collect();
        let values: Vec<f64> = (0..events.len()).map(|i| i as f64).collect();
        let trend_slope = self.calculate_trend_slope(&timestamps, &values);

        // Calculate seasonality strength
        let seasonality_strength = self.calculate_seasonality_strength(&hourly_dist, &weekly_dist);

        // Calculate volatility
        let volatility = self.calculate_volatility(&events);

        let pattern = TemporalPattern {
            trait_name: trait_name.to_string(),
            hourly_distribution: hourly_dist,
            daily_distribution: daily_dist,
            weekly_distribution: weekly_dist,
            monthly_distribution: monthly_dist,
            trend_slope,
            seasonality_strength,
            volatility,
        };

        self.temporal_patterns.insert(trait_name.to_string(), pattern.clone());
        Ok(pattern)
    }

    /// Analyze user behavior patterns
    pub fn analyze_user_behavior(&mut self, user_id: &str) -> Result<UserBehaviorPattern, CoreError> {
        let user_events: Vec<_> = self.usage_history
            .iter()
            .filter(|e| e.user_session.as_ref().map_or(false, |s| s == user_id))
            .collect();

        if user_events.is_empty() {
            return Err(CoreError::new("No usage data for specified user"));
        }

        // Calculate trait preferences
        let mut trait_counts: HashMap<String, u32> = HashMap::new();
        let mut total_usage = 0;

        for event in &user_events {
            *trait_counts.entry(event.trait_name.clone()).or_insert(0) += 1;
            total_usage += 1;
        }

        let trait_preferences: HashMap<String, f64> = trait_counts
            .into_iter()
            .map(|(trait_name, count)| (trait_name, count as f64 / total_usage as f64))
            .collect();

        // Calculate usage frequency (events per day)
        let time_span_days = if user_events.len() > 1 {
            let first = user_events.first().unwrap().timestamp;
            let last = user_events.last().unwrap().timestamp;
            ((last - first) as f64 / (24.0 * 3600.0)).max(1.0)
        } else {
            1.0
        };
        let usage_frequency = user_events.len() as f64 / time_span_days;

        // Calculate session duration average
        let session_durations: Vec<f64> = user_events
            .iter()
            .filter_map(|e| e.duration_ms.map(|d| d as f64))
            .collect();
        let session_duration_avg = if !session_durations.is_empty() {
            session_durations.iter().sum::<f64>() / session_durations.len() as f64
        } else {
            0.0
        };

        // Calculate success rate
        let successful_events = user_events.iter().filter(|e| e.success).count();
        let success_rate = successful_events as f64 / user_events.len() as f64;

        // Calculate learning curve slope
        let learning_curve_slope = self.calculate_learning_curve(&user_events);

        // Build trait transition matrix
        let trait_transition_matrix = self.build_transition_matrix(&user_events);

        // Extract preferred contexts
        let preferred_contexts = self.extract_preferred_contexts(&user_events);

        let pattern = UserBehaviorPattern {
            user_id: user_id.to_string(),
            trait_preferences,
            usage_frequency,
            session_duration_avg,
            success_rate,
            learning_curve_slope,
            trait_transition_matrix,
            preferred_contexts,
        };

        self.user_patterns.insert(user_id.to_string(), pattern.clone());
        Ok(pattern)
    }

    /// Detect seasonal patterns in usage
    pub fn detect_seasonal_patterns(&mut self, trait_name: &str) -> Result<Vec<SeasonalPattern>, CoreError> {
        if !self.config.seasonal_analysis_enabled {
            return Ok(Vec::new());
        }

        let events: Vec<_> = self.usage_history
            .iter()
            .filter(|e| e.trait_name == trait_name)
            .collect();

        if events.len() < 50 { // Need sufficient data for seasonal analysis
            return Ok(Vec::new());
        }

        let mut patterns = Vec::new();

        // Daily patterns (24-hour cycle)
        if let Some(daily_pattern) = self.detect_seasonal_pattern(&events, SeasonalType::Daily, 24.0 * 3600.0)? {
            patterns.push(daily_pattern);
        }

        // Weekly patterns (7-day cycle)
        if let Some(weekly_pattern) = self.detect_seasonal_pattern(&events, SeasonalType::Weekly, 7.0 * 24.0 * 3600.0)? {
            patterns.push(weekly_pattern);
        }

        // Monthly patterns (30-day cycle)
        if let Some(monthly_pattern) = self.detect_seasonal_pattern(&events, SeasonalType::Monthly, 30.0 * 24.0 * 3600.0)? {
            patterns.push(monthly_pattern);
        }

        self.seasonal_patterns.insert(trait_name.to_string(), patterns.clone());
        Ok(patterns)
    }

    /// Predict future usage patterns
    pub fn predict_usage(&self, trait_name: &str, prediction_days: u32) -> Result<Vec<f64>, CoreError> {
        let temporal_pattern = self.temporal_patterns.get(trait_name)
            .ok_or_else(|| CoreError::new("No temporal pattern available for trait"))?;

        let seasonal_patterns = self.seasonal_patterns.get(trait_name).unwrap_or(&Vec::new());

        let mut predictions = Vec::new();
        let current_time = self.current_timestamp();

        for day in 0..prediction_days {
            let future_timestamp = current_time + (day as u64 * 24 * 3600);
            let datetime = self.timestamp_to_datetime(future_timestamp);

            // Base prediction from trend
            let trend_component = temporal_pattern.trend_slope * day as f64;

            // Seasonal components
            let mut seasonal_component = 0.0;
            for pattern in seasonal_patterns {
                seasonal_component += self.calculate_seasonal_value(pattern, future_timestamp);
            }

            // Weekly and daily patterns
            let weekly_component = temporal_pattern.weekly_distribution[datetime.weekday as usize];
            let daily_component = temporal_pattern.hourly_distribution[12]; // Noon average

            let prediction = trend_component + seasonal_component + weekly_component + daily_component;
            predictions.push(prediction.max(0.0)); // Ensure non-negative
        }

        Ok(predictions)
    }

    /// Get usage insights and recommendations
    pub fn get_usage_insights(&self, trait_name: &str) -> Result<UsageInsights, CoreError> {
        let temporal_pattern = self.temporal_patterns.get(trait_name)
            .ok_or_else(|| CoreError::new("No pattern data available"))?;

        let seasonal_patterns = self.seasonal_patterns.get(trait_name).unwrap_or(&Vec::new());

        // Peak usage times
        let peak_hour = temporal_pattern.hourly_distribution
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let peak_day = temporal_pattern.weekly_distribution
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Usage trend
        let trend_direction = if temporal_pattern.trend_slope > 0.1 {
            TrendDirection::Increasing
        } else if temporal_pattern.trend_slope < -0.1 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        // Seasonality insights
        let has_strong_seasonality = temporal_pattern.seasonality_strength > 0.3;

        // Recent anomalies
        let recent_anomalies: Vec<_> = self.detected_anomalies
            .iter()
            .filter(|a| a.trait_name == trait_name)
            .filter(|a| {
                let cutoff = self.current_timestamp() - (7 * 24 * 3600); // Last 7 days
                a.timestamp > cutoff
            })
            .cloned()
            .collect();

        Ok(UsageInsights {
            trait_name: trait_name.to_string(),
            peak_hour,
            peak_day,
            trend_direction,
            has_strong_seasonality,
            volatility: temporal_pattern.volatility,
            recent_anomalies,
            prediction_confidence: self.calculate_prediction_confidence(temporal_pattern),
        })
    }

    // Helper methods

    fn update_temporal_patterns(&mut self, event: &UsageEvent) -> Result<(), CoreError> {
        // Incremental update of temporal patterns
        // This is a simplified version - in practice, you'd want more sophisticated updating
        Ok(())
    }

    fn update_user_patterns(&mut self, event: &UsageEvent) -> Result<(), CoreError> {
        // Incremental update of user behavior patterns
        Ok(())
    }

    fn detect_anomaly(&self, event: &UsageEvent) -> Result<Option<UsageAnomaly>, CoreError> {
        // Simple anomaly detection based on temporal patterns
        if let Some(pattern) = self.temporal_patterns.get(&event.trait_name) {
            let datetime = self.timestamp_to_datetime(event.timestamp);
            let expected_usage = pattern.hourly_distribution[datetime.hour as usize];

            // Calculate actual usage rate in the current hour
            let current_hour_events = self.usage_history
                .iter()
                .filter(|e| {
                    let e_datetime = self.timestamp_to_datetime(e.timestamp);
                    e.trait_name == event.trait_name &&
                    e_datetime.hour == datetime.hour &&
                    event.timestamp - e.timestamp < 3600 // Within same hour
                })
                .count();

            let actual_usage = current_hour_events as f64;
            let threshold = expected_usage * 3.0; // 3-sigma rule

            if actual_usage > threshold {
                return Ok(Some(UsageAnomaly {
                    timestamp: event.timestamp,
                    trait_name: event.trait_name.clone(),
                    anomaly_type: AnomalyType::UnusualSpike,
                    severity: (actual_usage - expected_usage) / expected_usage,
                    expected_value: expected_usage,
                    actual_value: actual_usage,
                    confidence: 0.8,
                    context: Some(event.context.clone()),
                }));
            }
        }

        Ok(None)
    }

    fn current_timestamp(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    fn timestamp_to_datetime(&self, timestamp: u64) -> DateTime {
        // Simplified datetime conversion
        let days_since_epoch = timestamp / (24 * 3600);
        let seconds_in_day = timestamp % (24 * 3600);

        DateTime {
            year: 1970 + (days_since_epoch / 365) as u32,
            month: ((days_since_epoch % 365) / 30 + 1) as u8,
            day: ((days_since_epoch % 365) % 30 + 1) as u8,
            hour: (seconds_in_day / 3600) as u8,
            minute: ((seconds_in_day % 3600) / 60) as u8,
            second: (seconds_in_day % 60) as u8,
            weekday: ((days_since_epoch + 4) % 7) as u8, // Jan 1, 1970 was Thursday
        }
    }

    fn calculate_trend_slope(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }

        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
        let sum_x2: f64 = x.iter().map(|xi| xi * xi).sum();

        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < 1e-10 {
            0.0
        } else {
            (n * sum_xy - sum_x * sum_y) / denominator
        }
    }

    fn calculate_seasonality_strength(&self, hourly: &Array1<f64>, weekly: &Array1<f64>) -> f64 {
        let hourly_var = hourly.var(0.0);
        let weekly_var = weekly.var(0.0);
        (hourly_var + weekly_var) / 2.0
    }

    fn calculate_volatility(&self, events: &[&UsageEvent]) -> f64 {
        if events.len() < 2 {
            return 0.0;
        }

        // Calculate daily usage counts
        let mut daily_counts: BTreeMap<u64, u32> = BTreeMap::new();
        for event in events {
            let day = event.timestamp / (24 * 3600);
            *daily_counts.entry(day).or_insert(0) += 1;
        }

        let counts: Vec<f64> = daily_counts.values().map(|&c| c as f64).collect();
        if counts.len() < 2 {
            return 0.0;
        }

        let mean = counts.iter().sum::<f64>() / counts.len() as f64;
        let variance = counts.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (counts.len() - 1) as f64;
        variance.sqrt() / mean.max(1.0) // Coefficient of variation
    }

    fn calculate_learning_curve(&self, events: &[&UsageEvent]) -> f64 {
        if events.len() < 5 {
            return 0.0;
        }

        // Calculate success rate over time windows
        let window_size = events.len() / 5;
        let mut success_rates = Vec::new();

        for i in 0..5 {
            let start = i * window_size;
            let end = if i == 4 { events.len() } else { (i + 1) * window_size };
            let window_events = &events[start..end];

            let successes = window_events.iter().filter(|e| e.success).count();
            let rate = successes as f64 / window_events.len() as f64;
            success_rates.push(rate);
        }

        // Calculate slope of success rate improvement
        let x: Vec<f64> = (0..5).map(|i| i as f64).collect();
        self.calculate_trend_slope(&x, &success_rates)
    }

    fn build_transition_matrix(&self, events: &[&UsageEvent]) -> HashMap<(String, String), f64> {
        let mut transitions: HashMap<(String, String), u32> = HashMap::new();
        let mut total_transitions = 0;

        for window in events.windows(2) {
            let from_trait = &window[0].trait_name;
            let to_trait = &window[1].trait_name;
            *transitions.entry((from_trait.clone(), to_trait.clone())).or_insert(0) += 1;
            total_transitions += 1;
        }

        transitions
            .into_iter()
            .map(|(k, v)| (k, v as f64 / total_transitions as f64))
            .collect()
    }

    fn extract_preferred_contexts(&self, events: &[&UsageEvent]) -> Vec<TraitContext> {
        // Group events by context and find most common ones
        let mut context_counts: HashMap<String, u32> = HashMap::new();

        for event in events {
            let context_key = format!("{:?}", event.context); // Simplified
            *context_counts.entry(context_key).or_insert(0) += 1;
        }

        // Return top 3 most common contexts
        let mut sorted_contexts: Vec<_> = context_counts.into_iter().collect();
        sorted_contexts.sort_by(|a, b| b.1.cmp(&a.1));

        sorted_contexts
            .into_iter()
            .take(3)
            .map(|(_, _)| TraitContext::default()) // Simplified - would need proper deserialization
            .collect()
    }

    fn detect_seasonal_pattern(&self, events: &[&UsageEvent], pattern_type: SeasonalType, period_seconds: f64) -> Result<Option<SeasonalPattern>, CoreError> {
        // FFT-based seasonal pattern detection (simplified)
        let timestamps: Vec<f64> = events.iter().map(|e| e.timestamp as f64).collect();

        if timestamps.len() < 20 {
            return Ok(None);
        }

        // Simple amplitude detection based on period
        let mut amplitudes = Vec::new();
        let window_size = (period_seconds as usize).max(1);

        for window in timestamps.chunks(window_size) {
            amplitudes.push(window.len() as f64);
        }

        if amplitudes.len() < 3 {
            return Ok(None);
        }

        let mean_amplitude = amplitudes.iter().sum::<f64>() / amplitudes.len() as f64;
        let max_amplitude = amplitudes.iter().cloned().fold(0.0, f64::max);
        let amplitude_ratio = max_amplitude / mean_amplitude.max(1.0);

        if amplitude_ratio > 1.5 { // Threshold for seasonal detection
            Ok(Some(SeasonalPattern {
                pattern_type,
                amplitude: max_amplitude - mean_amplitude,
                period_days: period_seconds / (24.0 * 3600.0),
                phase_offset: 0.0, // Simplified
                confidence: (amplitude_ratio - 1.0).min(1.0),
                detected_peaks: Vec::new(), // Would be calculated with proper peak detection
                detected_troughs: Vec::new(),
            }))
        } else {
            Ok(None)
        }
    }

    fn calculate_seasonal_value(&self, pattern: &SeasonalPattern, timestamp: u64) -> f64 {
        let time_in_period = (timestamp as f64) % (pattern.period_days * 24.0 * 3600.0);
        let phase = 2.0 * std::f64::consts::PI * time_in_period / (pattern.period_days * 24.0 * 3600.0);
        pattern.amplitude * (phase + pattern.phase_offset).sin()
    }

    fn calculate_prediction_confidence(&self, pattern: &TemporalPattern) -> f64 {
        // Higher confidence for lower volatility and stronger trends/seasonality
        let volatility_factor = (1.0 - pattern.volatility.min(1.0)).max(0.0);
        let trend_factor = pattern.trend_slope.abs().min(1.0);
        let seasonality_factor = pattern.seasonality_strength.min(1.0);

        (volatility_factor + trend_factor + seasonality_factor) / 3.0
    }
}

/// Helper struct for datetime representation
#[derive(Debug, Clone)]
struct DateTime {
    year: u32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: u8,
    weekday: u8,
}

/// Usage insights and recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageInsights {
    pub trait_name: String,
    pub peak_hour: usize,
    pub peak_day: usize,
    pub trend_direction: TrendDirection,
    pub has_strong_seasonality: bool,
    pub volatility: f64,
    pub recent_anomalies: Vec<UsageAnomaly>,
    pub prediction_confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

/// Pattern-based recommendation engine
pub struct PatternBasedRecommender {
    pattern_analyzer: UsagePatternAnalyzer,
    recommendation_cache: HashMap<String, Vec<TraitRecommendation>>,
    config: UsagePatternConfig,
}

impl PatternBasedRecommender {
    pub fn new(pattern_analyzer: UsagePatternAnalyzer, config: UsagePatternConfig) -> Self {
        Self {
            pattern_analyzer,
            recommendation_cache: HashMap::new(),
            config,
        }
    }

    /// Generate recommendations based on usage patterns
    pub fn recommend_based_on_patterns(&mut self, context: &TraitContext) -> Result<Vec<TraitRecommendation>, CoreError> {
        let cache_key = format!("{:?}", context);

        if let Some(cached) = self.recommendation_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        let mut recommendations = Vec::new();

        // Time-based recommendations
        let current_time = self.pattern_analyzer.current_timestamp();
        let datetime = self.pattern_analyzer.timestamp_to_datetime(current_time);

        for (trait_name, pattern) in &self.pattern_analyzer.temporal_patterns {
            let current_hour_score = pattern.hourly_distribution[datetime.hour as usize];
            let current_day_score = pattern.weekly_distribution[datetime.weekday as usize];

            let temporal_score = (current_hour_score + current_day_score) / 2.0;

            if temporal_score > 0.1 { // Threshold for recommendation
                recommendations.push(TraitRecommendation {
                    trait_name: trait_name.clone(),
                    confidence: temporal_score,
                    reasoning: format!("High usage probability at current time ({}%)", (temporal_score * 100.0) as u32),
                    context: context.clone(),
                    metadata: HashMap::new(),
                });
            }
        }

        // Trend-based recommendations
        for (trait_name, pattern) in &self.pattern_analyzer.temporal_patterns {
            if pattern.trend_slope > 0.05 { // Positive trend
                recommendations.push(TraitRecommendation {
                    trait_name: trait_name.clone(),
                    confidence: pattern.trend_slope.min(1.0),
                    reasoning: "Increasing usage trend detected".to_string(),
                    context: context.clone(),
                    metadata: HashMap::new(),
                });
            }
        }

        // Sort by confidence and take top recommendations
        recommendations.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        recommendations.truncate(10);

        self.recommendation_cache.insert(cache_key, recommendations.clone());
        Ok(recommendations)
    }

    /// Get seasonal recommendations
    pub fn get_seasonal_recommendations(&self, trait_name: &str) -> Result<Vec<String>, CoreError> {
        let seasonal_patterns = self.pattern_analyzer.seasonal_patterns
            .get(trait_name)
            .ok_or_else(|| CoreError::new("No seasonal patterns available"))?;

        let mut recommendations = Vec::new();
        let current_time = self.pattern_analyzer.current_timestamp();

        for pattern in seasonal_patterns {
            let seasonal_value = self.pattern_analyzer.calculate_seasonal_value(pattern, current_time);

            if seasonal_value > 0.0 {
                recommendations.push(format!(
                    "Seasonal peak expected for {} pattern (confidence: {:.1}%)",
                    match pattern.pattern_type {
                        SeasonalType::Daily => "daily",
                        SeasonalType::Weekly => "weekly",
                        SeasonalType::Monthly => "monthly",
                        SeasonalType::Yearly => "yearly",
                        SeasonalType::Custom(_) => "custom",
                    },
                    pattern.confidence * 100.0
                ));
            }
        }

        Ok(recommendations)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_pattern_analyzer_creation() {
        let config = UsagePatternConfig::default();
        let feature_extractor = TraitFeatureExtractor::new(Default::default());
        let analyzer = UsagePatternAnalyzer::new(config, feature_extractor);

        assert_eq!(analyzer.usage_history.len(), 0);
        assert_eq!(analyzer.temporal_patterns.len(), 0);
    }

    #[test]
    fn test_usage_event_recording() {
        let config = UsagePatternConfig::default();
        let feature_extractor = TraitFeatureExtractor::new(Default::default());
        let mut analyzer = UsagePatternAnalyzer::new(config, feature_extractor);

        let event = UsageEvent {
            trait_name: "TestTrait".to_string(),
            timestamp: 1234567890,
            context: TraitContext::default(),
            user_session: Some("user1".to_string()),
            success: true,
            duration_ms: Some(1000),
            metadata: HashMap::new(),
        };

        assert!(analyzer.record_usage(event).is_ok());
        assert_eq!(analyzer.usage_history.len(), 1);
    }

    #[test]
    fn test_temporal_pattern_analysis() {
        let config = UsagePatternConfig::default();
        let feature_extractor = TraitFeatureExtractor::new(Default::default());
        let mut analyzer = UsagePatternAnalyzer::new(config, feature_extractor);

        // Add multiple events for the same trait
        for i in 0..10 {
            let event = UsageEvent {
                trait_name: "TestTrait".to_string(),
                timestamp: 1234567890 + (i * 3600), // Different hours
                context: TraitContext::default(),
                user_session: Some("user1".to_string()),
                success: true,
                duration_ms: Some(1000),
                metadata: HashMap::new(),
            };
            analyzer.record_usage(event).unwrap();
        }

        let pattern = analyzer.analyze_temporal_patterns("TestTrait").unwrap();
        assert_eq!(pattern.trait_name, "TestTrait");
        assert_eq!(pattern.hourly_distribution.len(), 24);
    }

    #[test]
    fn test_trend_slope_calculation() {
        let config = UsagePatternConfig::default();
        let feature_extractor = TraitFeatureExtractor::new(Default::default());
        let analyzer = UsagePatternAnalyzer::new(config, feature_extractor);

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect linear relationship

        let slope = analyzer.calculate_trend_slope(&x, &y);
        assert!((slope - 2.0).abs() < 1e-10); // Should be exactly 2.0
    }

    #[test]
    fn test_anomaly_detection() {
        let config = UsagePatternConfig::default();
        let feature_extractor = TraitFeatureExtractor::new(Default::default());
        let analyzer = UsagePatternAnalyzer::new(config, feature_extractor);

        let event = UsageEvent {
            trait_name: "TestTrait".to_string(),
            timestamp: 1234567890,
            context: TraitContext::default(),
            user_session: Some("user1".to_string()),
            success: true,
            duration_ms: Some(1000),
            metadata: HashMap::new(),
        };

        // Should not detect anomaly without pattern history
        let anomaly = analyzer.detect_anomaly(&event).unwrap();
        assert!(anomaly.is_none());
    }

    #[test]
    fn test_datetime_conversion() {
        let config = UsagePatternConfig::default();
        let feature_extractor = TraitFeatureExtractor::new(Default::default());
        let analyzer = UsagePatternAnalyzer::new(config, feature_extractor);

        let timestamp = 1234567890; // Known timestamp
        let datetime = analyzer.timestamp_to_datetime(timestamp);

        // Basic validation - should have reasonable values
        assert!(datetime.hour < 24);
        assert!(datetime.minute < 60);
        assert!(datetime.second < 60);
        assert!(datetime.weekday < 7);
    }
}