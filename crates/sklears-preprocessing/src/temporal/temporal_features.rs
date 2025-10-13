//! Temporal feature extraction from datetime data
//!
//! This module provides comprehensive date/time feature extraction capabilities including:
//! - Date component extraction (year, month, day, hour, minute, second)
//! - Cyclical feature encoding (sin/cos transformations for periodic features)
//! - Holiday and business day indicators
//! - Time-based feature generation with timezone support

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

use super::datetime_utils::{DateComponents, DateTime};

/// Configuration for TemporalFeatureExtractor
#[derive(Debug, Clone)]
pub struct TemporalFeatureExtractorConfig {
    /// Whether to extract year
    pub extract_year: bool,
    /// Whether to extract month
    pub extract_month: bool,
    /// Whether to extract day of month
    pub extract_day: bool,
    /// Whether to extract day of week (0=Monday, 6=Sunday)
    pub extract_day_of_week: bool,
    /// Whether to extract hour
    pub extract_hour: bool,
    /// Whether to extract minute
    pub extract_minute: bool,
    /// Whether to extract second
    pub extract_second: bool,
    /// Whether to extract quarter
    pub extract_quarter: bool,
    /// Whether to extract day of year
    pub extract_day_of_year: bool,
    /// Whether to extract week of year
    pub extract_week_of_year: bool,
    /// Whether to use cyclical encoding for periodic features (sin/cos)
    pub cyclical_encoding: bool,
    /// Whether to include holiday indicators
    pub include_holidays: bool,
    /// Whether to include business day indicators
    pub include_business_days: bool,
    /// Time zone offset in hours from UTC (for timezone-aware processing)
    pub timezone_offset: Option<Float>,
}

impl Default for TemporalFeatureExtractorConfig {
    fn default() -> Self {
        Self {
            extract_year: true,
            extract_month: true,
            extract_day: true,
            extract_day_of_week: true,
            extract_hour: false,
            extract_minute: false,
            extract_second: false,
            extract_quarter: true,
            extract_day_of_year: false,
            extract_week_of_year: false,
            cyclical_encoding: true,
            include_holidays: false,
            include_business_days: false,
            timezone_offset: None,
        }
    }
}

/// TemporalFeatureExtractor for extracting features from datetime data
#[derive(Debug, Clone)]
pub struct TemporalFeatureExtractor<S> {
    config: TemporalFeatureExtractorConfig,
    feature_names_: Option<Vec<String>>,
    n_features_out_: Option<usize>,
    _phantom: PhantomData<S>,
}

impl TemporalFeatureExtractor<Untrained> {
    /// Create a new TemporalFeatureExtractor
    pub fn new() -> Self {
        Self {
            config: TemporalFeatureExtractorConfig::default(),
            feature_names_: None,
            n_features_out_: None,
            _phantom: PhantomData,
        }
    }

    /// Set whether to extract year
    pub fn extract_year(mut self, extract_year: bool) -> Self {
        self.config.extract_year = extract_year;
        self
    }

    /// Set whether to extract month
    pub fn extract_month(mut self, extract_month: bool) -> Self {
        self.config.extract_month = extract_month;
        self
    }

    /// Set whether to extract day
    pub fn extract_day(mut self, extract_day: bool) -> Self {
        self.config.extract_day = extract_day;
        self
    }

    /// Set whether to extract day of week
    pub fn extract_day_of_week(mut self, extract_day_of_week: bool) -> Self {
        self.config.extract_day_of_week = extract_day_of_week;
        self
    }

    /// Set whether to extract hour
    pub fn extract_hour(mut self, extract_hour: bool) -> Self {
        self.config.extract_hour = extract_hour;
        self
    }

    /// Set whether to extract minute
    pub fn extract_minute(mut self, extract_minute: bool) -> Self {
        self.config.extract_minute = extract_minute;
        self
    }

    /// Set whether to extract second
    pub fn extract_second(mut self, extract_second: bool) -> Self {
        self.config.extract_second = extract_second;
        self
    }

    /// Set whether to extract quarter
    pub fn extract_quarter(mut self, extract_quarter: bool) -> Self {
        self.config.extract_quarter = extract_quarter;
        self
    }

    /// Set whether to extract day of year
    pub fn extract_day_of_year(mut self, extract_day_of_year: bool) -> Self {
        self.config.extract_day_of_year = extract_day_of_year;
        self
    }

    /// Set whether to extract week of year
    pub fn extract_week_of_year(mut self, extract_week_of_year: bool) -> Self {
        self.config.extract_week_of_year = extract_week_of_year;
        self
    }

    /// Set whether to use cyclical encoding
    pub fn cyclical_encoding(mut self, cyclical_encoding: bool) -> Self {
        self.config.cyclical_encoding = cyclical_encoding;
        self
    }

    /// Set whether to include holiday indicators
    pub fn include_holidays(mut self, include_holidays: bool) -> Self {
        self.config.include_holidays = include_holidays;
        self
    }

    /// Set whether to include business day indicators
    pub fn include_business_days(mut self, include_business_days: bool) -> Self {
        self.config.include_business_days = include_business_days;
        self
    }

    /// Set timezone offset in hours
    pub fn timezone_offset(mut self, timezone_offset: Float) -> Self {
        self.config.timezone_offset = Some(timezone_offset);
        self
    }

    /// Calculate the number of output features based on configuration
    fn calculate_n_features_out(&self) -> usize {
        let mut count = 0;

        if self.config.extract_year {
            count += 1;
        }

        if self.config.extract_month {
            count += if self.config.cyclical_encoding { 2 } else { 1 };
        }

        if self.config.extract_day {
            count += if self.config.cyclical_encoding { 2 } else { 1 };
        }

        if self.config.extract_day_of_week {
            count += if self.config.cyclical_encoding { 2 } else { 1 };
        }

        if self.config.extract_hour {
            count += if self.config.cyclical_encoding { 2 } else { 1 };
        }

        if self.config.extract_minute {
            count += if self.config.cyclical_encoding { 2 } else { 1 };
        }

        if self.config.extract_second {
            count += if self.config.cyclical_encoding { 2 } else { 1 };
        }

        if self.config.extract_quarter {
            count += if self.config.cyclical_encoding { 2 } else { 1 };
        }

        if self.config.extract_day_of_year {
            count += if self.config.cyclical_encoding { 2 } else { 1 };
        }

        if self.config.extract_week_of_year {
            count += if self.config.cyclical_encoding { 2 } else { 1 };
        }

        if self.config.include_holidays {
            count += 1;
        }

        if self.config.include_business_days {
            count += 1;
        }

        count
    }

    /// Generate feature names based on configuration
    fn generate_feature_names(&self) -> Vec<String> {
        let mut names = Vec::new();

        if self.config.extract_year {
            names.push("year".to_string());
        }

        if self.config.extract_month {
            if self.config.cyclical_encoding {
                names.push("month_sin".to_string());
                names.push("month_cos".to_string());
            } else {
                names.push("month".to_string());
            }
        }

        if self.config.extract_day {
            if self.config.cyclical_encoding {
                names.push("day_sin".to_string());
                names.push("day_cos".to_string());
            } else {
                names.push("day".to_string());
            }
        }

        if self.config.extract_day_of_week {
            if self.config.cyclical_encoding {
                names.push("day_of_week_sin".to_string());
                names.push("day_of_week_cos".to_string());
            } else {
                names.push("day_of_week".to_string());
            }
        }

        if self.config.extract_hour {
            if self.config.cyclical_encoding {
                names.push("hour_sin".to_string());
                names.push("hour_cos".to_string());
            } else {
                names.push("hour".to_string());
            }
        }

        if self.config.extract_minute {
            if self.config.cyclical_encoding {
                names.push("minute_sin".to_string());
                names.push("minute_cos".to_string());
            } else {
                names.push("minute".to_string());
            }
        }

        if self.config.extract_second {
            if self.config.cyclical_encoding {
                names.push("second_sin".to_string());
                names.push("second_cos".to_string());
            } else {
                names.push("second".to_string());
            }
        }

        if self.config.extract_quarter {
            if self.config.cyclical_encoding {
                names.push("quarter_sin".to_string());
                names.push("quarter_cos".to_string());
            } else {
                names.push("quarter".to_string());
            }
        }

        if self.config.extract_day_of_year {
            if self.config.cyclical_encoding {
                names.push("day_of_year_sin".to_string());
                names.push("day_of_year_cos".to_string());
            } else {
                names.push("day_of_year".to_string());
            }
        }

        if self.config.extract_week_of_year {
            if self.config.cyclical_encoding {
                names.push("week_of_year_sin".to_string());
                names.push("week_of_year_cos".to_string());
            } else {
                names.push("week_of_year".to_string());
            }
        }

        if self.config.include_holidays {
            names.push("is_holiday".to_string());
        }

        if self.config.include_business_days {
            names.push("is_business_day".to_string());
        }

        names
    }
}

impl TemporalFeatureExtractor<Trained> {
    /// Get the feature names
    pub fn feature_names(&self) -> &[String] {
        self.feature_names_
            .as_ref()
            .expect("Extractor should be fitted")
    }

    /// Get the number of output features
    pub fn n_features_out(&self) -> usize {
        self.n_features_out_.expect("Extractor should be fitted")
    }

    /// Check if a date is a holiday (simplified implementation)
    fn is_holiday(&self, components: &DateComponents) -> bool {
        // Simplified holiday detection - only major US holidays
        match (components.month, components.day) {
            (1, 1) => true,   // New Year's Day
            (7, 4) => true,   // Independence Day
            (12, 25) => true, // Christmas
            _ => false,
        }
    }

    /// Check if a date is a business day (Monday-Friday, not holiday)
    fn is_business_day(&self, components: &DateComponents) -> bool {
        let is_weekday = components.day_of_week < 5; // Monday (0) to Friday (4)
        let is_not_holiday = if self.config.include_holidays {
            !self.is_holiday(components)
        } else {
            true
        };
        is_weekday && is_not_holiday
    }

    /// Convert periodic value to cyclical encoding (sin/cos)
    fn to_cyclical(&self, value: Float, period: Float) -> (Float, Float) {
        let angle = 2.0 * std::f64::consts::PI * (value / period);
        (angle.sin(), angle.cos())
    }

    /// Extract features from a single timestamp
    fn extract_features_from_timestamp(&self, timestamp: Float) -> Array1<Float> {
        let datetime = DateTime::from_timestamp(timestamp as i64);
        let components = datetime.to_components(self.config.timezone_offset);

        let mut features = Vec::new();

        if self.config.extract_year {
            features.push(components.year as Float);
        }

        if self.config.extract_month {
            if self.config.cyclical_encoding {
                let (sin, cos) = self.to_cyclical(components.month as Float, 12.0);
                features.push(sin);
                features.push(cos);
            } else {
                features.push(components.month as Float);
            }
        }

        if self.config.extract_day {
            if self.config.cyclical_encoding {
                let (sin, cos) = self.to_cyclical(components.day as Float, 31.0);
                features.push(sin);
                features.push(cos);
            } else {
                features.push(components.day as Float);
            }
        }

        if self.config.extract_day_of_week {
            if self.config.cyclical_encoding {
                let (sin, cos) = self.to_cyclical(components.day_of_week as Float, 7.0);
                features.push(sin);
                features.push(cos);
            } else {
                features.push(components.day_of_week as Float);
            }
        }

        if self.config.extract_hour {
            if self.config.cyclical_encoding {
                let (sin, cos) = self.to_cyclical(components.hour as Float, 24.0);
                features.push(sin);
                features.push(cos);
            } else {
                features.push(components.hour as Float);
            }
        }

        if self.config.extract_minute {
            if self.config.cyclical_encoding {
                let (sin, cos) = self.to_cyclical(components.minute as Float, 60.0);
                features.push(sin);
                features.push(cos);
            } else {
                features.push(components.minute as Float);
            }
        }

        if self.config.extract_second {
            if self.config.cyclical_encoding {
                let (sin, cos) = self.to_cyclical(components.second as Float, 60.0);
                features.push(sin);
                features.push(cos);
            } else {
                features.push(components.second as Float);
            }
        }

        if self.config.extract_quarter {
            if self.config.cyclical_encoding {
                let (sin, cos) = self.to_cyclical(components.quarter as Float, 4.0);
                features.push(sin);
                features.push(cos);
            } else {
                features.push(components.quarter as Float);
            }
        }

        if self.config.extract_day_of_year {
            if self.config.cyclical_encoding {
                let (sin, cos) = self.to_cyclical(components.day_of_year as Float, 366.0);
                features.push(sin);
                features.push(cos);
            } else {
                features.push(components.day_of_year as Float);
            }
        }

        if self.config.extract_week_of_year {
            if self.config.cyclical_encoding {
                let (sin, cos) = self.to_cyclical(components.week_of_year as Float, 53.0);
                features.push(sin);
                features.push(cos);
            } else {
                features.push(components.week_of_year as Float);
            }
        }

        if self.config.include_holidays {
            features.push(if self.is_holiday(&components) {
                1.0
            } else {
                0.0
            });
        }

        if self.config.include_business_days {
            features.push(if self.is_business_day(&components) {
                1.0
            } else {
                0.0
            });
        }

        Array1::from_vec(features)
    }
}

impl Default for TemporalFeatureExtractor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<Array1<Float>, ()> for TemporalFeatureExtractor<Untrained> {
    type Fitted = TemporalFeatureExtractor<Trained>;

    fn fit(self, _x: &Array1<Float>, _y: &()) -> Result<Self::Fitted> {
        let n_features_out = self.calculate_n_features_out();
        let feature_names = self.generate_feature_names();

        if n_features_out == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "feature_extraction".to_string(),
                reason: "No features selected for extraction".to_string(),
            });
        }

        Ok(TemporalFeatureExtractor {
            config: self.config,
            feature_names_: Some(feature_names),
            n_features_out_: Some(n_features_out),
            _phantom: PhantomData,
        })
    }
}

impl Transform<Array1<Float>, Array2<Float>> for TemporalFeatureExtractor<Trained> {
    fn transform(&self, x: &Array1<Float>) -> Result<Array2<Float>> {
        let n_samples = x.len();
        let n_features_out = self.n_features_out();

        let mut result = Array2::<Float>::zeros((n_samples, n_features_out));

        for (i, &timestamp) in x.iter().enumerate() {
            let features = self.extract_features_from_timestamp(timestamp);
            for (j, &feature_value) in features.iter().enumerate() {
                result[[i, j]] = feature_value;
            }
        }

        Ok(result)
    }
}
