use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, Duration};

use super::config_types::*;
use super::core_data_processing::{TransformationData, DataRecord, DataValue, ProcessingError};

/// Comprehensive data cleaning engine providing advanced data preprocessing,
/// missing value handling, outlier detection, and duplicate removal capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCleaningEngine {
    /// Missing value handlers for different data types
    missing_value_handlers: HashMap<String, MissingValueHandler>,
    /// Outlier detection algorithms
    outlier_detectors: HashMap<String, OutlierDetector>,
    /// Duplicate detection configurations
    duplicate_detectors: HashMap<String, DuplicateDetector>,
    /// Data cleaning strategies
    cleaning_strategies: HashMap<String, CleaningStrategy>,
    /// Quality thresholds for cleaning decisions
    quality_thresholds: CleaningQualityThresholds,
    /// Performance monitoring
    performance_monitor: Arc<RwLock<CleaningPerformanceMonitor>>,
}

impl DataCleaningEngine {
    /// Create a new data cleaning engine
    pub fn new() -> Self {
        Self {
            missing_value_handlers: HashMap::new(),
            outlier_detectors: HashMap::new(),
            duplicate_detectors: HashMap::new(),
            cleaning_strategies: HashMap::new(),
            quality_thresholds: CleaningQualityThresholds::default(),
            performance_monitor: Arc::new(RwLock::new(CleaningPerformanceMonitor::new())),
        }
    }

    /// Clean transformation data using registered strategies
    pub async fn clean_data(&self, data: &TransformationData, strategy_id: &str) -> Result<TransformationData, ProcessingError> {
        let strategy = self.cleaning_strategies.get(strategy_id)
            .ok_or_else(|| ProcessingError::ConfigurationError(format!("Cleaning strategy not found: {}", strategy_id)))?;

        let start_time = Utc::now();
        let mut cleaned_data = data.clone();

        // Apply cleaning operations in order
        for operation in &strategy.operations {
            cleaned_data = self.apply_cleaning_operation(&cleaned_data, operation).await?;
        }

        // Update performance metrics
        {
            let mut monitor = self.performance_monitor.write().unwrap();
            monitor.record_cleaning_operation(
                strategy_id.to_string(),
                Utc::now().signed_duration_since(start_time),
                data.records.len(),
                cleaned_data.records.len(),
            );
        }

        Ok(cleaned_data)
    }

    /// Apply a single cleaning operation
    async fn apply_cleaning_operation(&self, data: &TransformationData, operation: &CleaningOperation) -> Result<TransformationData, ProcessingError> {
        match &operation.operation_type {
            CleaningOperationType::MissingValueHandling(config) => {
                self.handle_missing_values(data, config).await
            },
            CleaningOperationType::OutlierDetection(config) => {
                self.detect_and_handle_outliers(data, config).await
            },
            CleaningOperationType::DuplicateRemoval(config) => {
                self.remove_duplicates(data, config).await
            },
            CleaningOperationType::DataNormalization(config) => {
                self.normalize_data(data, config).await
            },
            CleaningOperationType::DataStandardization(config) => {
                self.standardize_data(data, config).await
            },
            CleaningOperationType::DataValidation(config) => {
                self.validate_and_clean(data, config).await
            },
            CleaningOperationType::Custom(config) => {
                self.apply_custom_cleaning(data, config).await
            },
        }
    }

    /// Handle missing values in the data
    async fn handle_missing_values(&self, data: &TransformationData, config: &MissingValueConfiguration) -> Result<TransformationData, ProcessingError> {
        let handler = self.missing_value_handlers.get(&config.handler_id)
            .ok_or_else(|| ProcessingError::ConfigurationError(format!("Missing value handler not found: {}", config.handler_id)))?;

        let mut cleaned_records = Vec::new();

        for record in &data.records {
            let cleaned_record = handler.handle_record(record, config)?;
            if let Some(record) = cleaned_record {
                cleaned_records.push(record);
            }
        }

        Ok(TransformationData {
            records: cleaned_records,
            metadata: data.metadata.clone(),
            schema: data.schema.clone(),
            quality_metrics: None, // Will be recalculated
        })
    }

    /// Detect and handle outliers
    async fn detect_and_handle_outliers(&self, data: &TransformationData, config: &OutlierDetectionConfiguration) -> Result<TransformationData, ProcessingError> {
        let detector = self.outlier_detectors.get(&config.detector_id)
            .ok_or_else(|| ProcessingError::ConfigurationError(format!("Outlier detector not found: {}", config.detector_id)))?;

        let outlier_indices = detector.detect_outliers(data, config)?;
        let mut cleaned_records = Vec::new();

        for (index, record) in data.records.iter().enumerate() {
            if !outlier_indices.contains(&index) {
                cleaned_records.push(record.clone());
            } else {
                // Handle outlier based on strategy
                match config.handling_strategy {
                    OutlierHandlingStrategy::Remove => {
                        // Skip this record
                        continue;
                    },
                    OutlierHandlingStrategy::Cap(ref cap_config) => {
                        let capped_record = self.cap_outlier_values(record, cap_config)?;
                        cleaned_records.push(capped_record);
                    },
                    OutlierHandlingStrategy::Transform(ref transform_config) => {
                        let transformed_record = self.transform_outlier_values(record, transform_config)?;
                        cleaned_records.push(transformed_record);
                    },
                    OutlierHandlingStrategy::Flag => {
                        let mut flagged_record = record.clone();
                        flagged_record.metadata.insert("outlier_flag".to_string(), "true".to_string());
                        cleaned_records.push(flagged_record);
                    },
                }
            }
        }

        Ok(TransformationData {
            records: cleaned_records,
            metadata: data.metadata.clone(),
            schema: data.schema.clone(),
            quality_metrics: None,
        })
    }

    /// Remove duplicate records
    async fn remove_duplicates(&self, data: &TransformationData, config: &DuplicateDetectionConfiguration) -> Result<TransformationData, ProcessingError> {
        let detector = self.duplicate_detectors.get(&config.detector_id)
            .ok_or_else(|| ProcessingError::ConfigurationError(format!("Duplicate detector not found: {}", config.detector_id)))?;

        let duplicate_groups = detector.find_duplicates(data, config)?;
        let mut cleaned_records = Vec::new();
        let mut processed_indices = std::collections::HashSet::new();

        for record in &data.records {
            let record_index = data.records.iter().position(|r| r.id == record.id).unwrap();

            if processed_indices.contains(&record_index) {
                continue;
            }

            // Check if this record is part of a duplicate group
            let is_duplicate = duplicate_groups.iter().any(|group| group.record_indices.contains(&record_index));

            if is_duplicate {
                // Handle duplicate based on strategy
                match config.resolution_strategy {
                    DuplicateResolutionStrategy::KeepFirst => {
                        if let Some(group) = duplicate_groups.iter().find(|g| g.record_indices.contains(&record_index)) {
                            let first_index = *group.record_indices.iter().min().unwrap();
                            if record_index == first_index {
                                cleaned_records.push(record.clone());
                            }
                            // Mark all indices in this group as processed
                            for &idx in &group.record_indices {
                                processed_indices.insert(idx);
                            }
                        }
                    },
                    DuplicateResolutionStrategy::KeepLast => {
                        if let Some(group) = duplicate_groups.iter().find(|g| g.record_indices.contains(&record_index)) {
                            let last_index = *group.record_indices.iter().max().unwrap();
                            if record_index == last_index {
                                cleaned_records.push(record.clone());
                            }
                            // Mark all indices in this group as processed
                            for &idx in &group.record_indices {
                                processed_indices.insert(idx);
                            }
                        }
                    },
                    DuplicateResolutionStrategy::Merge => {
                        if let Some(group) = duplicate_groups.iter().find(|g| g.record_indices.contains(&record_index)) {
                            let first_index = *group.record_indices.iter().min().unwrap();
                            if record_index == first_index {
                                let merged_record = self.merge_duplicate_records(data, &group.record_indices)?;
                                cleaned_records.push(merged_record);
                            }
                            // Mark all indices in this group as processed
                            for &idx in &group.record_indices {
                                processed_indices.insert(idx);
                            }
                        }
                    },
                    DuplicateResolutionStrategy::Remove => {
                        // Don't add any records from duplicate groups
                        for group in &duplicate_groups {
                            if group.record_indices.contains(&record_index) {
                                for &idx in &group.record_indices {
                                    processed_indices.insert(idx);
                                }
                                break;
                            }
                        }
                    },
                }
            } else {
                cleaned_records.push(record.clone());
                processed_indices.insert(record_index);
            }
        }

        Ok(TransformationData {
            records: cleaned_records,
            metadata: data.metadata.clone(),
            schema: data.schema.clone(),
            quality_metrics: None,
        })
    }

    /// Normalize data values
    async fn normalize_data(&self, data: &TransformationData, config: &NormalizationConfiguration) -> Result<TransformationData, ProcessingError> {
        let mut normalized_records = Vec::new();

        for record in &data.records {
            let mut normalized_record = record.clone();

            for field_name in &config.fields_to_normalize {
                if let Some(value) = record.fields.get(field_name) {
                    let normalized_value = self.normalize_value(value, &config.normalization_method)?;
                    normalized_record.fields.insert(field_name.clone(), normalized_value);
                }
            }

            normalized_records.push(normalized_record);
        }

        Ok(TransformationData {
            records: normalized_records,
            metadata: data.metadata.clone(),
            schema: data.schema.clone(),
            quality_metrics: None,
        })
    }

    /// Standardize data values
    async fn standardize_data(&self, data: &TransformationData, config: &StandardizationConfiguration) -> Result<TransformationData, ProcessingError> {
        let mut standardized_records = Vec::new();

        // Calculate field statistics for standardization
        let field_stats = self.calculate_field_statistics(data, &config.fields_to_standardize)?;

        for record in &data.records {
            let mut standardized_record = record.clone();

            for field_name in &config.fields_to_standardize {
                if let Some(value) = record.fields.get(field_name) {
                    if let Some(stats) = field_stats.get(field_name) {
                        let standardized_value = self.standardize_value(value, stats, &config.standardization_method)?;
                        standardized_record.fields.insert(field_name.clone(), standardized_value);
                    }
                }
            }

            standardized_records.push(standardized_record);
        }

        Ok(TransformationData {
            records: standardized_records,
            metadata: data.metadata.clone(),
            schema: data.schema.clone(),
            quality_metrics: None,
        })
    }

    /// Validate and clean data
    async fn validate_and_clean(&self, data: &TransformationData, config: &ValidationCleaningConfiguration) -> Result<TransformationData, ProcessingError> {
        let mut cleaned_records = Vec::new();

        for record in &data.records {
            let mut is_valid = true;
            let mut cleaned_record = record.clone();

            // Apply validation rules
            for rule in &config.validation_rules {
                if !self.validate_record_against_rule(record, rule)? {
                    match config.invalid_handling_strategy {
                        InvalidDataHandlingStrategy::Remove => {
                            is_valid = false;
                            break;
                        },
                        InvalidDataHandlingStrategy::Fix => {
                            cleaned_record = self.fix_record_violation(&cleaned_record, rule)?;
                        },
                        InvalidDataHandlingStrategy::Flag => {
                            cleaned_record.metadata.insert(
                                format!("validation_violation_{}", rule.rule_name),
                                "true".to_string()
                            );
                        },
                    }
                }
            }

            if is_valid {
                cleaned_records.push(cleaned_record);
            }
        }

        Ok(TransformationData {
            records: cleaned_records,
            metadata: data.metadata.clone(),
            schema: data.schema.clone(),
            quality_metrics: None,
        })
    }

    /// Apply custom cleaning logic
    async fn apply_custom_cleaning(&self, data: &TransformationData, config: &CustomCleaningConfiguration) -> Result<TransformationData, ProcessingError> {
        // Custom cleaning implementation would be plugin-based
        // For now, return data unchanged
        Ok(data.clone())
    }

    /// Cap outlier values to specified limits
    fn cap_outlier_values(&self, record: &DataRecord, config: &OutlierCappingConfiguration) -> Result<DataRecord, ProcessingError> {
        let mut capped_record = record.clone();

        for (field_name, limits) in &config.field_limits {
            if let Some(value) = record.fields.get(field_name) {
                let capped_value = match value {
                    DataValue::Integer(i) => {
                        let min = limits.min_value.as_ref()
                            .and_then(|v| if let DataValue::Integer(min_i) = v { Some(*min_i) } else { None })
                            .unwrap_or(i64::MIN);
                        let max = limits.max_value.as_ref()
                            .and_then(|v| if let DataValue::Integer(max_i) = v { Some(*max_i) } else { None })
                            .unwrap_or(i64::MAX);
                        DataValue::Integer((*i).clamp(min, max))
                    },
                    DataValue::Float(f) => {
                        let min = limits.min_value.as_ref()
                            .and_then(|v| if let DataValue::Float(min_f) = v { Some(*min_f) } else { None })
                            .unwrap_or(f64::NEG_INFINITY);
                        let max = limits.max_value.as_ref()
                            .and_then(|v| if let DataValue::Float(max_f) = v { Some(*max_f) } else { None })
                            .unwrap_or(f64::INFINITY);
                        DataValue::Float(f.clamp(min, max))
                    },
                    _ => value.clone(),
                };
                capped_record.fields.insert(field_name.clone(), capped_value);
            }
        }

        Ok(capped_record)
    }

    /// Transform outlier values using specified transformation
    fn transform_outlier_values(&self, record: &DataRecord, config: &OutlierTransformConfiguration) -> Result<DataRecord, ProcessingError> {
        let mut transformed_record = record.clone();

        for (field_name, transformation) in &config.field_transformations {
            if let Some(value) = record.fields.get(field_name) {
                let transformed_value = self.apply_transformation(value, transformation)?;
                transformed_record.fields.insert(field_name.clone(), transformed_value);
            }
        }

        Ok(transformed_record)
    }

    /// Apply data transformation
    fn apply_transformation(&self, value: &DataValue, transformation: &DataTransformation) -> Result<DataValue, ProcessingError> {
        match transformation {
            DataTransformation::Log => {
                match value {
                    DataValue::Float(f) => Ok(DataValue::Float(f.ln())),
                    DataValue::Integer(i) => Ok(DataValue::Float((*i as f64).ln())),
                    _ => Ok(value.clone()),
                }
            },
            DataTransformation::Sqrt => {
                match value {
                    DataValue::Float(f) => Ok(DataValue::Float(f.sqrt())),
                    DataValue::Integer(i) => Ok(DataValue::Float((*i as f64).sqrt())),
                    _ => Ok(value.clone()),
                }
            },
            DataTransformation::Square => {
                match value {
                    DataValue::Float(f) => Ok(DataValue::Float(f * f)),
                    DataValue::Integer(i) => Ok(DataValue::Integer(i * i)),
                    _ => Ok(value.clone()),
                }
            },
            DataTransformation::Reciprocal => {
                match value {
                    DataValue::Float(f) => Ok(DataValue::Float(1.0 / f)),
                    DataValue::Integer(i) => Ok(DataValue::Float(1.0 / (*i as f64))),
                    _ => Ok(value.clone()),
                }
            },
            DataTransformation::Custom(formula) => {
                // Custom transformation would evaluate formula
                Ok(value.clone()) // Simplified
            },
        }
    }

    /// Merge duplicate records into a single record
    fn merge_duplicate_records(&self, data: &TransformationData, indices: &[usize]) -> Result<DataRecord, ProcessingError> {
        if indices.is_empty() {
            return Err(ProcessingError::ConfigurationError("No indices provided for merging".to_string()));
        }

        let first_record = &data.records[indices[0]];
        let mut merged_record = first_record.clone();

        // Merge fields from all duplicate records
        for &index in indices.iter().skip(1) {
            let record = &data.records[index];
            for (field_name, field_value) in &record.fields {
                if !merged_record.fields.contains_key(field_name) || matches!(merged_record.fields.get(field_name), Some(DataValue::Null)) {
                    merged_record.fields.insert(field_name.clone(), field_value.clone());
                }
            }

            // Merge metadata
            for (key, value) in &record.metadata {
                if !merged_record.metadata.contains_key(key) {
                    merged_record.metadata.insert(key.clone(), value.clone());
                }
            }
        }

        // Add merge information to metadata
        merged_record.metadata.insert("merged_from".to_string(),
            indices.iter().map(|i| data.records[*i].id.clone()).collect::<Vec<_>>().join(","));
        merged_record.metadata.insert("merge_count".to_string(), indices.len().to_string());

        Ok(merged_record)
    }

    /// Normalize a single value
    fn normalize_value(&self, value: &DataValue, method: &NormalizationMethod) -> Result<DataValue, ProcessingError> {
        match method {
            NormalizationMethod::MinMax { min, max } => {
                match value {
                    DataValue::Float(f) => {
                        let normalized = (f - min) / (max - min);
                        Ok(DataValue::Float(normalized))
                    },
                    DataValue::Integer(i) => {
                        let normalized = (*i as f64 - min) / (max - min);
                        Ok(DataValue::Float(normalized))
                    },
                    _ => Ok(value.clone()),
                }
            },
            NormalizationMethod::ZScore { mean, std_dev } => {
                match value {
                    DataValue::Float(f) => {
                        let normalized = (f - mean) / std_dev;
                        Ok(DataValue::Float(normalized))
                    },
                    DataValue::Integer(i) => {
                        let normalized = (*i as f64 - mean) / std_dev;
                        Ok(DataValue::Float(normalized))
                    },
                    _ => Ok(value.clone()),
                }
            },
            NormalizationMethod::UnitVector => {
                // Unit vector normalization would require vector context
                Ok(value.clone()) // Simplified
            },
        }
    }

    /// Standardize a single value
    fn standardize_value(&self, value: &DataValue, stats: &FieldStatistics, method: &StandardizationMethod) -> Result<DataValue, ProcessingError> {
        match method {
            StandardizationMethod::ZScore => {
                match value {
                    DataValue::Float(f) => {
                        let standardized = (f - stats.mean) / stats.std_dev;
                        Ok(DataValue::Float(standardized))
                    },
                    DataValue::Integer(i) => {
                        let standardized = (*i as f64 - stats.mean) / stats.std_dev;
                        Ok(DataValue::Float(standardized))
                    },
                    _ => Ok(value.clone()),
                }
            },
            StandardizationMethod::RobustScaling => {
                match value {
                    DataValue::Float(f) => {
                        let standardized = (f - stats.median) / stats.iqr;
                        Ok(DataValue::Float(standardized))
                    },
                    DataValue::Integer(i) => {
                        let standardized = (*i as f64 - stats.median) / stats.iqr;
                        Ok(DataValue::Float(standardized))
                    },
                    _ => Ok(value.clone()),
                }
            },
        }
    }

    /// Calculate field statistics for standardization
    fn calculate_field_statistics(&self, data: &TransformationData, fields: &[String]) -> Result<HashMap<String, FieldStatistics>, ProcessingError> {
        let mut stats = HashMap::new();

        for field_name in fields {
            let mut values = Vec::new();

            for record in &data.records {
                if let Some(value) = record.fields.get(field_name) {
                    match value {
                        DataValue::Float(f) => values.push(*f),
                        DataValue::Integer(i) => values.push(*i as f64),
                        _ => continue,
                    }
                }
            }

            if !values.is_empty() {
                let field_stats = self.compute_statistics(&values)?;
                stats.insert(field_name.clone(), field_stats);
            }
        }

        Ok(stats)
    }

    /// Compute statistical measures for a set of values
    fn compute_statistics(&self, values: &[f64]) -> Result<FieldStatistics, ProcessingError> {
        if values.is_empty() {
            return Err(ProcessingError::ConfigurationError("Cannot compute statistics for empty values".to_string()));
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = if sorted_values.len() % 2 == 0 {
            (sorted_values[sorted_values.len() / 2 - 1] + sorted_values[sorted_values.len() / 2]) / 2.0
        } else {
            sorted_values[sorted_values.len() / 2]
        };

        let q1_index = sorted_values.len() / 4;
        let q3_index = 3 * sorted_values.len() / 4;
        let q1 = sorted_values[q1_index];
        let q3 = sorted_values[q3_index.min(sorted_values.len() - 1)];
        let iqr = q3 - q1;

        Ok(FieldStatistics {
            mean,
            median,
            std_dev,
            variance,
            min: sorted_values[0],
            max: sorted_values[sorted_values.len() - 1],
            q1,
            q3,
            iqr,
            count: values.len(),
        })
    }

    /// Validate record against a cleaning rule
    fn validate_record_against_rule(&self, record: &DataRecord, rule: &ValidationCleaningRule) -> Result<bool, ProcessingError> {
        match &rule.rule_type {
            ValidationCleaningRuleType::FieldPresence(field_name) => {
                Ok(record.fields.contains_key(field_name) &&
                   !matches!(record.fields.get(field_name), Some(DataValue::Null)))
            },
            ValidationCleaningRuleType::FieldFormat { field_name, pattern } => {
                if let Some(value) = record.fields.get(field_name) {
                    Ok(self.matches_pattern(value, pattern))
                } else {
                    Ok(false)
                }
            },
            ValidationCleaningRuleType::FieldRange { field_name, min, max } => {
                if let Some(value) = record.fields.get(field_name) {
                    Ok(self.is_in_range(value, min, max))
                } else {
                    Ok(false)
                }
            },
            ValidationCleaningRuleType::Custom(expression) => {
                // Custom validation would evaluate expression
                Ok(true) // Simplified
            },
        }
    }

    /// Fix record violation based on rule
    fn fix_record_violation(&self, record: &DataRecord, rule: &ValidationCleaningRule) -> Result<DataRecord, ProcessingError> {
        let mut fixed_record = record.clone();

        match &rule.rule_type {
            ValidationCleaningRuleType::FieldPresence(field_name) => {
                if !fixed_record.fields.contains_key(field_name) {
                    // Add default value
                    fixed_record.fields.insert(field_name.clone(), DataValue::String("".to_string()));
                }
            },
            ValidationCleaningRuleType::FieldFormat { field_name, pattern: _ } => {
                // Format correction would be rule-specific
                // For now, flag the issue
                fixed_record.metadata.insert(format!("format_fixed_{}", field_name), "true".to_string());
            },
            ValidationCleaningRuleType::FieldRange { field_name, min, max } => {
                if let Some(value) = fixed_record.fields.get(field_name) {
                    let capped_value = self.cap_value_to_range(value, min, max)?;
                    fixed_record.fields.insert(field_name.clone(), capped_value);
                }
            },
            ValidationCleaningRuleType::Custom(_) => {
                // Custom fix logic
            },
        }

        Ok(fixed_record)
    }

    /// Check if value matches pattern
    fn matches_pattern(&self, value: &DataValue, pattern: &str) -> bool {
        match value {
            DataValue::String(s) => {
                // Simplified pattern matching - would use regex in practice
                s.contains(pattern)
            },
            _ => false,
        }
    }

    /// Check if value is in range
    fn is_in_range(&self, value: &DataValue, min: &Option<DataValue>, max: &Option<DataValue>) -> bool {
        match value {
            DataValue::Integer(i) => {
                let min_check = min.as_ref()
                    .map(|m| if let DataValue::Integer(min_i) = m { i >= min_i } else { false })
                    .unwrap_or(true);
                let max_check = max.as_ref()
                    .map(|m| if let DataValue::Integer(max_i) = m { i <= max_i } else { false })
                    .unwrap_or(true);
                min_check && max_check
            },
            DataValue::Float(f) => {
                let min_check = min.as_ref()
                    .map(|m| if let DataValue::Float(min_f) = m { f >= min_f } else { false })
                    .unwrap_or(true);
                let max_check = max.as_ref()
                    .map(|m| if let DataValue::Float(max_f) = m { f <= max_f } else { false })
                    .unwrap_or(true);
                min_check && max_check
            },
            _ => true,
        }
    }

    /// Cap value to specified range
    fn cap_value_to_range(&self, value: &DataValue, min: &Option<DataValue>, max: &Option<DataValue>) -> Result<DataValue, ProcessingError> {
        match value {
            DataValue::Integer(i) => {
                let min_val = min.as_ref()
                    .and_then(|m| if let DataValue::Integer(min_i) = m { Some(*min_i) } else { None })
                    .unwrap_or(i64::MIN);
                let max_val = max.as_ref()
                    .and_then(|m| if let DataValue::Integer(max_i) = m { Some(*max_i) } else { None })
                    .unwrap_or(i64::MAX);
                Ok(DataValue::Integer((*i).clamp(min_val, max_val)))
            },
            DataValue::Float(f) => {
                let min_val = min.as_ref()
                    .and_then(|m| if let DataValue::Float(min_f) = m { Some(*min_f) } else { None })
                    .unwrap_or(f64::NEG_INFINITY);
                let max_val = max.as_ref()
                    .and_then(|m| if let DataValue::Float(max_f) = m { Some(*max_f) } else { None })
                    .unwrap_or(f64::INFINITY);
                Ok(DataValue::Float(f.clamp(min_val, max_val)))
            },
            _ => Ok(value.clone()),
        }
    }

    /// Register a missing value handler
    pub fn register_missing_value_handler(&mut self, handler_id: String, handler: MissingValueHandler) {
        self.missing_value_handlers.insert(handler_id, handler);
    }

    /// Register an outlier detector
    pub fn register_outlier_detector(&mut self, detector_id: String, detector: OutlierDetector) {
        self.outlier_detectors.insert(detector_id, detector);
    }

    /// Register a duplicate detector
    pub fn register_duplicate_detector(&mut self, detector_id: String, detector: DuplicateDetector) {
        self.duplicate_detectors.insert(detector_id, detector);
    }

    /// Register a cleaning strategy
    pub fn register_cleaning_strategy(&mut self, strategy_id: String, strategy: CleaningStrategy) {
        self.cleaning_strategies.insert(strategy_id, strategy);
    }
}

/// Missing value handler for different data imputation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingValueHandler {
    /// Handler identifier
    pub handler_id: String,
    /// Handler name
    pub handler_name: String,
    /// Handling strategies for different field types
    pub strategies: HashMap<String, MissingValueStrategy>,
    /// Quality checks for imputed values
    pub quality_checks: MissingValueQualityChecks,
}

impl MissingValueHandler {
    /// Handle missing values in a record
    pub fn handle_record(&self, record: &DataRecord, config: &MissingValueConfiguration) -> Result<Option<DataRecord>, ProcessingError> {
        let mut handled_record = record.clone();
        let mut has_critical_missing = false;

        for field_name in &config.fields_to_handle {
            if let Some(strategy) = self.strategies.get(field_name) {
                if !record.fields.contains_key(field_name) || matches!(record.fields.get(field_name), Some(DataValue::Null)) {
                    match config.handling_approach {
                        MissingValueHandlingApproach::Impute => {
                            let imputed_value = self.impute_value(record, field_name, strategy)?;
                            handled_record.fields.insert(field_name.clone(), imputed_value);
                        },
                        MissingValueHandlingApproach::Remove => {
                            has_critical_missing = true;
                            break;
                        },
                        MissingValueHandlingApproach::Flag => {
                            handled_record.metadata.insert(
                                format!("missing_{}", field_name),
                                "true".to_string()
                            );
                        },
                    }
                }
            }
        }

        if has_critical_missing {
            Ok(None) // Record should be removed
        } else {
            Ok(Some(handled_record))
        }
    }

    /// Impute a missing value
    fn impute_value(&self, record: &DataRecord, field_name: &str, strategy: &MissingValueStrategy) -> Result<DataValue, ProcessingError> {
        match strategy {
            MissingValueStrategy::Mean => {
                // Would calculate mean from dataset
                Ok(DataValue::Float(0.0)) // Simplified
            },
            MissingValueStrategy::Median => {
                // Would calculate median from dataset
                Ok(DataValue::Float(0.0)) // Simplified
            },
            MissingValueStrategy::Mode => {
                // Would calculate mode from dataset
                Ok(DataValue::String("default".to_string())) // Simplified
            },
            MissingValueStrategy::Forward => {
                // Forward fill from previous record
                Ok(DataValue::String("forward_filled".to_string())) // Simplified
            },
            MissingValueStrategy::Backward => {
                // Backward fill from next record
                Ok(DataValue::String("backward_filled".to_string())) // Simplified
            },
            MissingValueStrategy::Interpolate => {
                // Linear interpolation
                Ok(DataValue::Float(0.0)) // Simplified
            },
            MissingValueStrategy::Default(value) => {
                Ok(value.clone())
            },
            MissingValueStrategy::Predictive(model_id) => {
                // Use predictive model for imputation
                Ok(DataValue::String("predicted".to_string())) // Simplified
            },
        }
    }
}

/// Missing value handling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MissingValueStrategy {
    /// Use mean value for numeric fields
    Mean,
    /// Use median value for numeric fields
    Median,
    /// Use mode value for categorical fields
    Mode,
    /// Forward fill from previous record
    Forward,
    /// Backward fill from next record
    Backward,
    /// Linear interpolation
    Interpolate,
    /// Use default value
    Default(DataValue),
    /// Use predictive model
    Predictive(String),
}

/// Missing value configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingValueConfiguration {
    /// Handler identifier
    pub handler_id: String,
    /// Fields to handle
    pub fields_to_handle: Vec<String>,
    /// Handling approach
    pub handling_approach: MissingValueHandlingApproach,
    /// Quality threshold for acceptance
    pub quality_threshold: f64,
}

/// Missing value handling approach
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MissingValueHandlingApproach {
    /// Impute missing values
    Impute,
    /// Remove records with missing values
    Remove,
    /// Flag records with missing values
    Flag,
}

/// Quality checks for missing value handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingValueQualityChecks {
    /// Check imputation accuracy
    pub check_accuracy: bool,
    /// Maximum allowed missing percentage
    pub max_missing_percentage: f64,
    /// Validation rules for imputed values
    pub validation_rules: Vec<ImputationValidationRule>,
}

/// Imputation validation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImputationValidationRule {
    /// Rule name
    pub rule_name: String,
    /// Field name
    pub field_name: String,
    /// Validation condition
    pub condition: ImputationCondition,
}

/// Imputation validation condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImputationCondition {
    /// Value within expected range
    WithinRange { min: f64, max: f64 },
    /// Value matches pattern
    MatchesPattern(String),
    /// Value is one of allowed values
    AllowedValues(Vec<DataValue>),
}

/// Outlier detector for identifying anomalous data points
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierDetector {
    /// Detector identifier
    pub detector_id: String,
    /// Detection algorithm
    pub algorithm: OutlierDetectionAlgorithm,
    /// Algorithm parameters
    pub parameters: HashMap<String, f64>,
    /// Validation settings
    pub validation_settings: OutlierValidationSettings,
}

impl OutlierDetector {
    /// Detect outliers in the data
    pub fn detect_outliers(&self, data: &TransformationData, config: &OutlierDetectionConfiguration) -> Result<Vec<usize>, ProcessingError> {
        match &self.algorithm {
            OutlierDetectionAlgorithm::ZScore => {
                self.detect_zscore_outliers(data, config)
            },
            OutlierDetectionAlgorithm::IQR => {
                self.detect_iqr_outliers(data, config)
            },
            OutlierDetectionAlgorithm::IsolationForest => {
                self.detect_isolation_forest_outliers(data, config)
            },
            OutlierDetectionAlgorithm::LOF => {
                self.detect_lof_outliers(data, config)
            },
            OutlierDetectionAlgorithm::DBSCAN => {
                self.detect_dbscan_outliers(data, config)
            },
            OutlierDetectionAlgorithm::Custom(algorithm_id) => {
                self.detect_custom_outliers(data, config, algorithm_id)
            },
        }
    }

    /// Detect outliers using Z-score method
    fn detect_zscore_outliers(&self, data: &TransformationData, config: &OutlierDetectionConfiguration) -> Result<Vec<usize>, ProcessingError> {
        let mut outlier_indices = Vec::new();
        let threshold = self.parameters.get("threshold").unwrap_or(&3.0);

        for field_name in &config.fields_to_analyze {
            let values: Vec<f64> = data.records.iter()
                .filter_map(|r| r.fields.get(field_name))
                .filter_map(|v| match v {
                    DataValue::Float(f) => Some(*f),
                    DataValue::Integer(i) => Some(*i as f64),
                    _ => None,
                })
                .collect();

            if values.len() < 3 {
                continue; // Not enough data for statistical analysis
            }

            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
            let std_dev = variance.sqrt();

            for (record_index, record) in data.records.iter().enumerate() {
                if let Some(value) = record.fields.get(field_name) {
                    let numeric_value = match value {
                        DataValue::Float(f) => *f,
                        DataValue::Integer(i) => *i as f64,
                        _ => continue,
                    };

                    let z_score = (numeric_value - mean) / std_dev;
                    if z_score.abs() > *threshold {
                        outlier_indices.push(record_index);
                    }
                }
            }
        }

        // Remove duplicates
        outlier_indices.sort();
        outlier_indices.dedup();
        Ok(outlier_indices)
    }

    /// Detect outliers using IQR method
    fn detect_iqr_outliers(&self, data: &TransformationData, config: &OutlierDetectionConfiguration) -> Result<Vec<usize>, ProcessingError> {
        let mut outlier_indices = Vec::new();
        let multiplier = self.parameters.get("multiplier").unwrap_or(&1.5);

        for field_name in &config.fields_to_analyze {
            let mut values: Vec<(usize, f64)> = data.records.iter().enumerate()
                .filter_map(|(i, r)| r.fields.get(field_name).map(|v| (i, v)))
                .filter_map(|(i, v)| match v {
                    DataValue::Float(f) => Some((i, *f)),
                    DataValue::Integer(val) => Some((i, *val as f64)),
                    _ => None,
                })
                .collect();

            if values.len() < 4 {
                continue; // Not enough data for quartile analysis
            }

            values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let q1_index = values.len() / 4;
            let q3_index = 3 * values.len() / 4;
            let q1 = values[q1_index].1;
            let q3 = values[q3_index.min(values.len() - 1)].1;
            let iqr = q3 - q1;

            let lower_bound = q1 - multiplier * iqr;
            let upper_bound = q3 + multiplier * iqr;

            for (record_index, value) in &values {
                if *value < lower_bound || *value > upper_bound {
                    outlier_indices.push(*record_index);
                }
            }
        }

        // Remove duplicates
        outlier_indices.sort();
        outlier_indices.dedup();
        Ok(outlier_indices)
    }

    /// Detect outliers using Isolation Forest (simplified implementation)
    fn detect_isolation_forest_outliers(&self, data: &TransformationData, config: &OutlierDetectionConfiguration) -> Result<Vec<usize>, ProcessingError> {
        // Simplified implementation - would use actual Isolation Forest algorithm
        Ok(Vec::new())
    }

    /// Detect outliers using Local Outlier Factor (simplified implementation)
    fn detect_lof_outliers(&self, data: &TransformationData, config: &OutlierDetectionConfiguration) -> Result<Vec<usize>, ProcessingError> {
        // Simplified implementation - would use actual LOF algorithm
        Ok(Vec::new())
    }

    /// Detect outliers using DBSCAN (simplified implementation)
    fn detect_dbscan_outliers(&self, data: &TransformationData, config: &OutlierDetectionConfiguration) -> Result<Vec<usize>, ProcessingError> {
        // Simplified implementation - would use actual DBSCAN algorithm
        Ok(Vec::new())
    }

    /// Detect outliers using custom algorithm
    fn detect_custom_outliers(&self, data: &TransformationData, config: &OutlierDetectionConfiguration, algorithm_id: &str) -> Result<Vec<usize>, ProcessingError> {
        // Custom algorithm implementation would be plugin-based
        Ok(Vec::new())
    }
}

/// Outlier detection algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutlierDetectionAlgorithm {
    /// Z-score based detection
    ZScore,
    /// Interquartile Range based detection
    IQR,
    /// Isolation Forest algorithm
    IsolationForest,
    /// Local Outlier Factor
    LOF,
    /// DBSCAN clustering
    DBSCAN,
    /// Custom algorithm
    Custom(String),
}

/// Outlier detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierDetectionConfiguration {
    /// Detector identifier
    pub detector_id: String,
    /// Fields to analyze for outliers
    pub fields_to_analyze: Vec<String>,
    /// Outlier handling strategy
    pub handling_strategy: OutlierHandlingStrategy,
    /// Detection sensitivity
    pub sensitivity: f64,
}

/// Outlier handling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutlierHandlingStrategy {
    /// Remove outlier records
    Remove,
    /// Cap outlier values
    Cap(OutlierCappingConfiguration),
    /// Transform outlier values
    Transform(OutlierTransformConfiguration),
    /// Flag outlier records
    Flag,
}

/// Outlier capping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierCappingConfiguration {
    /// Field-specific capping limits
    pub field_limits: HashMap<String, CappingLimits>,
}

/// Capping limits for outlier values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CappingLimits {
    /// Minimum allowed value
    pub min_value: Option<DataValue>,
    /// Maximum allowed value
    pub max_value: Option<DataValue>,
}

/// Outlier transformation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierTransformConfiguration {
    /// Field-specific transformations
    pub field_transformations: HashMap<String, DataTransformation>,
}

/// Data transformation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataTransformation {
    /// Natural logarithm
    Log,
    /// Square root
    Sqrt,
    /// Square
    Square,
    /// Reciprocal
    Reciprocal,
    /// Custom transformation formula
    Custom(String),
}

/// Outlier validation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierValidationSettings {
    /// Validate detection accuracy
    pub validate_accuracy: bool,
    /// Maximum allowed outlier percentage
    pub max_outlier_percentage: f64,
    /// Review threshold for manual inspection
    pub review_threshold: f64,
}

/// Duplicate detector for identifying duplicate records
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateDetector {
    /// Detector identifier
    pub detector_id: String,
    /// Detection method
    pub detection_method: DuplicateDetectionMethod,
    /// Matching criteria
    pub matching_criteria: DuplicateMatchingCriteria,
    /// Performance settings
    pub performance_settings: DuplicateDetectionPerformanceSettings,
}

impl DuplicateDetector {
    /// Find duplicate records
    pub fn find_duplicates(&self, data: &TransformationData, config: &DuplicateDetectionConfiguration) -> Result<Vec<DuplicateGroup>, ProcessingError> {
        match &self.detection_method {
            DuplicateDetectionMethod::ExactMatch => {
                self.find_exact_duplicates(data, config)
            },
            DuplicateDetectionMethod::FuzzyMatch => {
                self.find_fuzzy_duplicates(data, config)
            },
            DuplicateDetectionMethod::Similarity => {
                self.find_similarity_duplicates(data, config)
            },
            DuplicateDetectionMethod::Custom(method_id) => {
                self.find_custom_duplicates(data, config, method_id)
            },
        }
    }

    /// Find exact duplicate matches
    fn find_exact_duplicates(&self, data: &TransformationData, config: &DuplicateDetectionConfiguration) -> Result<Vec<DuplicateGroup>, ProcessingError> {
        let mut groups = Vec::new();
        let mut processed_indices = std::collections::HashSet::new();

        for (i, record) in data.records.iter().enumerate() {
            if processed_indices.contains(&i) {
                continue;
            }

            let mut duplicate_indices = vec![i];

            for (j, other_record) in data.records.iter().enumerate().skip(i + 1) {
                if processed_indices.contains(&j) {
                    continue;
                }

                if self.records_match_exactly(record, other_record, &config.comparison_fields) {
                    duplicate_indices.push(j);
                }
            }

            if duplicate_indices.len() > 1 {
                for &idx in &duplicate_indices {
                    processed_indices.insert(idx);
                }

                groups.push(DuplicateGroup {
                    group_id: uuid::Uuid::new_v4().to_string(),
                    record_indices: duplicate_indices,
                    similarity_score: 1.0, // Exact match
                    matching_fields: config.comparison_fields.clone(),
                });
            }
        }

        Ok(groups)
    }

    /// Find fuzzy duplicate matches
    fn find_fuzzy_duplicates(&self, data: &TransformationData, config: &DuplicateDetectionConfiguration) -> Result<Vec<DuplicateGroup>, ProcessingError> {
        // Simplified fuzzy matching implementation
        let threshold = config.similarity_threshold.unwrap_or(0.8);
        let mut groups = Vec::new();
        let mut processed_indices = std::collections::HashSet::new();

        for (i, record) in data.records.iter().enumerate() {
            if processed_indices.contains(&i) {
                continue;
            }

            let mut duplicate_indices = vec![i];

            for (j, other_record) in data.records.iter().enumerate().skip(i + 1) {
                if processed_indices.contains(&j) {
                    continue;
                }

                let similarity = self.calculate_record_similarity(record, other_record, &config.comparison_fields)?;
                if similarity >= threshold {
                    duplicate_indices.push(j);
                }
            }

            if duplicate_indices.len() > 1 {
                for &idx in &duplicate_indices {
                    processed_indices.insert(idx);
                }

                groups.push(DuplicateGroup {
                    group_id: uuid::Uuid::new_v4().to_string(),
                    record_indices: duplicate_indices,
                    similarity_score: threshold,
                    matching_fields: config.comparison_fields.clone(),
                });
            }
        }

        Ok(groups)
    }

    /// Find similarity-based duplicates
    fn find_similarity_duplicates(&self, data: &TransformationData, config: &DuplicateDetectionConfiguration) -> Result<Vec<DuplicateGroup>, ProcessingError> {
        // Would implement more sophisticated similarity algorithms
        self.find_fuzzy_duplicates(data, config)
    }

    /// Find duplicates using custom method
    fn find_custom_duplicates(&self, data: &TransformationData, config: &DuplicateDetectionConfiguration, method_id: &str) -> Result<Vec<DuplicateGroup>, ProcessingError> {
        // Custom duplicate detection would be plugin-based
        Ok(Vec::new())
    }

    /// Check if two records match exactly
    fn records_match_exactly(&self, record1: &DataRecord, record2: &DataRecord, fields: &[String]) -> bool {
        for field_name in fields {
            let value1 = record1.fields.get(field_name);
            let value2 = record2.fields.get(field_name);

            match (value1, value2) {
                (Some(v1), Some(v2)) => {
                    if !self.values_equal(v1, v2) {
                        return false;
                    }
                },
                (None, None) => continue,
                _ => return false,
            }
        }
        true
    }

    /// Calculate similarity between two records
    fn calculate_record_similarity(&self, record1: &DataRecord, record2: &DataRecord, fields: &[String]) -> Result<f64, ProcessingError> {
        let mut total_similarity = 0.0;
        let mut field_count = 0;

        for field_name in fields {
            if let (Some(value1), Some(value2)) = (record1.fields.get(field_name), record2.fields.get(field_name)) {
                let field_similarity = self.calculate_value_similarity(value1, value2)?;
                total_similarity += field_similarity;
                field_count += 1;
            }
        }

        if field_count == 0 {
            Ok(0.0)
        } else {
            Ok(total_similarity / field_count as f64)
        }
    }

    /// Check if two values are equal
    fn values_equal(&self, value1: &DataValue, value2: &DataValue) -> bool {
        match (value1, value2) {
            (DataValue::String(s1), DataValue::String(s2)) => s1 == s2,
            (DataValue::Integer(i1), DataValue::Integer(i2)) => i1 == i2,
            (DataValue::Float(f1), DataValue::Float(f2)) => (f1 - f2).abs() < f64::EPSILON,
            (DataValue::Boolean(b1), DataValue::Boolean(b2)) => b1 == b2,
            (DataValue::Date(d1), DataValue::Date(d2)) => d1 == d2,
            (DataValue::Null, DataValue::Null) => true,
            _ => false,
        }
    }

    /// Calculate similarity between two values
    fn calculate_value_similarity(&self, value1: &DataValue, value2: &DataValue) -> Result<f64, ProcessingError> {
        match (value1, value2) {
            (DataValue::String(s1), DataValue::String(s2)) => {
                Ok(self.string_similarity(s1, s2))
            },
            (DataValue::Integer(i1), DataValue::Integer(i2)) => {
                let diff = (*i1 - *i2).abs() as f64;
                let max_val = (*i1).abs().max((*i2).abs()) as f64;
                if max_val == 0.0 {
                    Ok(1.0)
                } else {
                    Ok((max_val - diff) / max_val)
                }
            },
            (DataValue::Float(f1), DataValue::Float(f2)) => {
                let diff = (f1 - f2).abs();
                let max_val = f1.abs().max(f2.abs());
                if max_val == 0.0 {
                    Ok(1.0)
                } else {
                    Ok((max_val - diff) / max_val)
                }
            },
            (DataValue::Boolean(b1), DataValue::Boolean(b2)) => {
                Ok(if b1 == b2 { 1.0 } else { 0.0 })
            },
            _ => Ok(0.0), // Different types or null values
        }
    }

    /// Calculate string similarity (simplified Levenshtein distance)
    fn string_similarity(&self, s1: &str, s2: &str) -> f64 {
        let len1 = s1.len();
        let len2 = s2.len();

        if len1 == 0 && len2 == 0 {
            return 1.0;
        }

        let max_len = len1.max(len2);
        if max_len == 0 {
            return 1.0;
        }

        // Simplified similarity calculation
        let common_chars = s1.chars().zip(s2.chars()).filter(|(c1, c2)| c1 == c2).count();
        common_chars as f64 / max_len as f64
    }
}

/// Duplicate detection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DuplicateDetectionMethod {
    /// Exact field matching
    ExactMatch,
    /// Fuzzy string matching
    FuzzyMatch,
    /// Similarity-based matching
    Similarity,
    /// Custom detection method
    Custom(String),
}

/// Duplicate detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateDetectionConfiguration {
    /// Detector identifier
    pub detector_id: String,
    /// Fields to compare for duplicates
    pub comparison_fields: Vec<String>,
    /// Similarity threshold for fuzzy matching
    pub similarity_threshold: Option<f64>,
    /// Resolution strategy for duplicates
    pub resolution_strategy: DuplicateResolutionStrategy,
}

/// Duplicate resolution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DuplicateResolutionStrategy {
    /// Keep first occurrence
    KeepFirst,
    /// Keep last occurrence
    KeepLast,
    /// Merge duplicate records
    Merge,
    /// Remove all duplicates
    Remove,
}

/// Duplicate matching criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateMatchingCriteria {
    /// Fields that must match exactly
    pub exact_match_fields: Vec<String>,
    /// Fields that can have fuzzy matches
    pub fuzzy_match_fields: Vec<String>,
    /// Fuzzy matching parameters
    pub fuzzy_parameters: FuzzyMatchingParameters,
}

/// Fuzzy matching parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyMatchingParameters {
    /// String similarity threshold
    pub string_similarity_threshold: f64,
    /// Numeric tolerance for matching
    pub numeric_tolerance: f64,
    /// Date tolerance for matching
    pub date_tolerance: Duration,
}

/// Duplicate detection performance settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateDetectionPerformanceSettings {
    /// Enable parallel processing
    pub parallel_processing: bool,
    /// Batch size for processing
    pub batch_size: usize,
    /// Memory limit for processing
    pub memory_limit_mb: usize,
}

/// Duplicate group representing a set of duplicate records
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateGroup {
    /// Group identifier
    pub group_id: String,
    /// Indices of duplicate records
    pub record_indices: Vec<usize>,
    /// Similarity score for the group
    pub similarity_score: f64,
    /// Fields that matched
    pub matching_fields: Vec<String>,
}

/// Cleaning strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleaningStrategy {
    /// Strategy identifier
    pub strategy_id: String,
    /// Strategy name
    pub strategy_name: String,
    /// Cleaning operations in order
    pub operations: Vec<CleaningOperation>,
    /// Strategy configuration
    pub configuration: CleaningStrategyConfiguration,
}

/// Individual cleaning operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleaningOperation {
    /// Operation identifier
    pub operation_id: String,
    /// Operation name
    pub operation_name: String,
    /// Type of cleaning operation
    pub operation_type: CleaningOperationType,
    /// Operation parameters
    pub parameters: HashMap<String, DataValue>,
    /// Operation order/priority
    pub order: u32,
}

/// Cleaning operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CleaningOperationType {
    /// Missing value handling
    MissingValueHandling(MissingValueConfiguration),
    /// Outlier detection and handling
    OutlierDetection(OutlierDetectionConfiguration),
    /// Duplicate removal
    DuplicateRemoval(DuplicateDetectionConfiguration),
    /// Data normalization
    DataNormalization(NormalizationConfiguration),
    /// Data standardization
    DataStandardization(StandardizationConfiguration),
    /// Data validation and cleaning
    DataValidation(ValidationCleaningConfiguration),
    /// Custom cleaning operation
    Custom(CustomCleaningConfiguration),
}

/// Normalization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationConfiguration {
    /// Fields to normalize
    pub fields_to_normalize: Vec<String>,
    /// Normalization method
    pub normalization_method: NormalizationMethod,
}

/// Normalization methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NormalizationMethod {
    /// Min-max normalization
    MinMax { min: f64, max: f64 },
    /// Z-score normalization
    ZScore { mean: f64, std_dev: f64 },
    /// Unit vector normalization
    UnitVector,
}

/// Standardization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardizationConfiguration {
    /// Fields to standardize
    pub fields_to_standardize: Vec<String>,
    /// Standardization method
    pub standardization_method: StandardizationMethod,
}

/// Standardization methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StandardizationMethod {
    /// Z-score standardization
    ZScore,
    /// Robust scaling using median and IQR
    RobustScaling,
}

/// Field statistics for standardization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldStatistics {
    /// Mean value
    pub mean: f64,
    /// Median value
    pub median: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Variance
    pub variance: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// First quartile
    pub q1: f64,
    /// Third quartile
    pub q3: f64,
    /// Interquartile range
    pub iqr: f64,
    /// Number of values
    pub count: usize,
}

/// Validation cleaning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCleaningConfiguration {
    /// Validation rules to apply
    pub validation_rules: Vec<ValidationCleaningRule>,
    /// Strategy for handling invalid data
    pub invalid_handling_strategy: InvalidDataHandlingStrategy,
}

/// Validation cleaning rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCleaningRule {
    /// Rule name
    pub rule_name: String,
    /// Rule type
    pub rule_type: ValidationCleaningRuleType,
}

/// Validation cleaning rule types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationCleaningRuleType {
    /// Field must be present
    FieldPresence(String),
    /// Field must match format
    FieldFormat { field_name: String, pattern: String },
    /// Field must be in range
    FieldRange { field_name: String, min: Option<DataValue>, max: Option<DataValue> },
    /// Custom validation expression
    Custom(String),
}

/// Invalid data handling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InvalidDataHandlingStrategy {
    /// Remove invalid records
    Remove,
    /// Fix invalid data
    Fix,
    /// Flag invalid data
    Flag,
}

/// Custom cleaning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomCleaningConfiguration {
    /// Custom cleaning algorithm identifier
    pub algorithm_id: String,
    /// Algorithm parameters
    pub parameters: HashMap<String, DataValue>,
}

/// Cleaning strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleaningStrategyConfiguration {
    /// Enable parallel processing
    pub parallel_processing: bool,
    /// Maximum processing time
    pub max_processing_time: Duration,
    /// Quality requirements
    pub quality_requirements: QualityRequirements,
}

/// Quality requirements for cleaning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityRequirements {
    /// Minimum data completeness
    pub min_completeness: f64,
    /// Maximum allowed outlier percentage
    pub max_outlier_percentage: f64,
    /// Maximum allowed duplicate percentage
    pub max_duplicate_percentage: f64,
}

/// Cleaning quality thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleaningQualityThresholds {
    /// Completeness threshold
    pub completeness_threshold: f64,
    /// Accuracy threshold
    pub accuracy_threshold: f64,
    /// Consistency threshold
    pub consistency_threshold: f64,
    /// Validity threshold
    pub validity_threshold: f64,
}

impl Default for CleaningQualityThresholds {
    fn default() -> Self {
        Self {
            completeness_threshold: 0.9,
            accuracy_threshold: 0.95,
            consistency_threshold: 0.9,
            validity_threshold: 0.85,
        }
    }
}

/// Cleaning performance monitor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleaningPerformanceMonitor {
    /// Operation metrics
    operation_metrics: Vec<CleaningOperationMetric>,
    /// Performance statistics
    performance_stats: CleaningPerformanceStats,
}

impl CleaningPerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self {
            operation_metrics: Vec::new(),
            performance_stats: CleaningPerformanceStats::default(),
        }
    }

    /// Record a cleaning operation
    pub fn record_cleaning_operation(&mut self, strategy_id: String, duration: Duration, input_records: usize, output_records: usize) {
        let metric = CleaningOperationMetric {
            strategy_id,
            start_time: Utc::now() - duration,
            duration,
            input_records,
            output_records,
            records_removed: input_records - output_records,
        };

        self.operation_metrics.push(metric);
        self.update_performance_stats();

        // Limit metrics history
        if self.operation_metrics.len() > 1000 {
            self.operation_metrics.drain(0..500);
        }
    }

    /// Update performance statistics
    fn update_performance_stats(&mut self) {
        if self.operation_metrics.is_empty() {
            return;
        }

        let total_operations = self.operation_metrics.len();
        let total_duration: Duration = self.operation_metrics.iter().map(|m| m.duration).sum();
        let total_input_records: usize = self.operation_metrics.iter().map(|m| m.input_records).sum();
        let total_output_records: usize = self.operation_metrics.iter().map(|m| m.output_records).sum();

        self.performance_stats = CleaningPerformanceStats {
            total_operations,
            average_duration: total_duration / total_operations as u32,
            total_records_processed: total_input_records,
            total_records_cleaned: total_output_records,
            overall_cleaning_rate: if total_input_records > 0 {
                (total_input_records - total_output_records) as f64 / total_input_records as f64
            } else {
                0.0
            },
        };
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> &CleaningPerformanceStats {
        &self.performance_stats
    }
}

/// Cleaning operation metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleaningOperationMetric {
    /// Strategy identifier
    pub strategy_id: String,
    /// Operation start time
    pub start_time: DateTime<Utc>,
    /// Operation duration
    pub duration: Duration,
    /// Number of input records
    pub input_records: usize,
    /// Number of output records
    pub output_records: usize,
    /// Number of records removed
    pub records_removed: usize,
}

/// Cleaning performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleaningPerformanceStats {
    /// Total cleaning operations
    pub total_operations: usize,
    /// Average operation duration
    pub average_duration: Duration,
    /// Total records processed
    pub total_records_processed: usize,
    /// Total records after cleaning
    pub total_records_cleaned: usize,
    /// Overall cleaning rate (percentage of records removed)
    pub overall_cleaning_rate: f64,
}

impl Default for CleaningPerformanceStats {
    fn default() -> Self {
        Self {
            total_operations: 0,
            average_duration: Duration::from_millis(0),
            total_records_processed: 0,
            total_records_cleaned: 0,
            overall_cleaning_rate: 0.0,
        }
    }
}

/// Data cleaning error types for comprehensive error handling
#[derive(Debug, thiserror::Error)]
pub enum CleaningError {
    #[error("Missing value handling failed: {0}")]
    MissingValueHandlingFailed(String),

    #[error("Outlier detection failed: {0}")]
    OutlierDetectionFailed(String),

    #[error("Duplicate detection failed: {0}")]
    DuplicateDetectionFailed(String),

    #[error("Data normalization failed: {0}")]
    DataNormalizationFailed(String),

    #[error("Data standardization failed: {0}")]
    DataStandardizationFailed(String),

    #[error("Validation cleaning failed: {0}")]
    ValidationCleaningFailed(String),

    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    #[error("Insufficient data: {0}")]
    InsufficientData(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Type alias for cleaning results
pub type CleaningResult<T> = Result<T, CleaningError>;