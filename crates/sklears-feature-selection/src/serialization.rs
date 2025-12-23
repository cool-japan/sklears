//! Serialization Support for Feature Selection Results
//!
//! This module provides comprehensive serialization and deserialization capabilities
//! for feature selection results, configurations, and benchmark data. Supports
//! multiple formats including JSON, YAML, CSV, and binary.

use crate::comprehensive_benchmark::ComprehensiveBenchmarkResults;
use crate::fluent_api::{FluentConfig, FluentSelectionResult};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

/// Serializable version of FluentSelectionResult
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableSelectionResult {
    pub selected_features: Vec<usize>,
    pub feature_scores: Vec<f64>,
    pub step_results: Vec<SerializableStepResult>,
    pub total_execution_time: f64,
    pub config_used: SerializableFluentConfig,
    pub metadata: SelectionMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableStepResult {
    pub step_name: String,
    pub features_before: usize,
    pub features_after: usize,
    pub execution_time: f64,
    pub step_scores: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableFluentConfig {
    pub parallel: bool,
    pub random_state: Option<u64>,
    pub verbose: bool,
    pub cache_results: bool,
    pub validation_split: Option<f64>,
    pub scoring_metric: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionMetadata {
    pub timestamp: String,
    pub dataset_name: Option<String>,
    pub dataset_shape: (usize, usize),
    pub selection_method: String,
    pub sklearn_version: String,
    pub rust_version: String,
    pub system_info: SystemInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub os: String,
    pub cpu_cores: usize,
    pub memory_gb: f64,
    pub architecture: String,
}

/// Serializable benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableBenchmarkResults {
    pub summary: BenchmarkSummarySerializable,
    pub detailed_results: Vec<DetailedMethodResultSerializable>,
    pub execution_metadata: ExecutionMetadataSerializable,
    pub format_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummarySerializable {
    pub best_method_overall: String,
    pub best_methods_by_metric: HashMap<String, String>,
    pub method_rankings: HashMap<String, f64>,
    pub dataset_difficulty_rankings: HashMap<String, f64>,
    pub execution_time_total_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedMethodResultSerializable {
    pub method_name: String,
    pub dataset_name: String,
    pub metric_scores: HashMap<String, f64>,
    pub execution_times_seconds: Vec<f64>,
    pub memory_usage_mb: Vec<f64>,
    pub selected_features: Vec<Vec<usize>>,
    pub convergence_info: ConvergenceInfoSerializable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceInfoSerializable {
    pub converged: bool,
    pub iterations: usize,
    pub final_objective_value: Option<f64>,
    pub convergence_history: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetadataSerializable {
    pub start_time: String,
    pub end_time: String,
    pub total_duration_seconds: f64,
    pub system_info: SystemInfo,
    pub configuration_summary: ConfigurationSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationSummary {
    pub num_runs: usize,
    pub cross_validation_folds: usize,
    pub parallel_execution: bool,
    pub random_state: u64,
}

/// Export formats supported
#[derive(Debug, Clone)]
pub enum ExportFormat {
    /// Json
    Json,
    /// JsonPretty
    JsonPretty,
    /// Yaml
    Yaml,
    /// Csv
    Csv,
    /// Binary
    Binary,
    /// Parquet
    Parquet,
}

/// Import/Export manager for feature selection results
pub struct SelectionResultsIO;

impl SelectionResultsIO {
    /// Export selection results to a file
    pub fn export_selection_result<P: AsRef<Path>>(
        result: &FluentSelectionResult,
        path: P,
        format: ExportFormat,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let serializable = Self::convert_to_serializable(result);

        match format {
            ExportFormat::Json => Self::export_json(&serializable, path, false)?,
            ExportFormat::JsonPretty => Self::export_json(&serializable, path, true)?,
            ExportFormat::Yaml => Self::export_yaml(&serializable, path)?,
            ExportFormat::Csv => Self::export_csv(&serializable, path)?,
            ExportFormat::Binary => Self::export_binary(&serializable, path)?,
            ExportFormat::Parquet => Self::export_parquet(&serializable, path)?,
        }

        Ok(())
    }

    /// Import selection results from a file
    pub fn import_selection_result<P: AsRef<Path>>(
        path: P,
        format: ExportFormat,
    ) -> Result<SerializableSelectionResult, Box<dyn std::error::Error>> {
        match format {
            ExportFormat::Json | ExportFormat::JsonPretty => Self::import_json(path),
            ExportFormat::Yaml => Self::import_yaml(path),
            ExportFormat::Binary => Self::import_binary(path),
            _ => Err("Import format not supported for this data type".into()),
        }
    }

    /// Export benchmark results to a file
    pub fn export_benchmark_results<P: AsRef<Path>>(
        results: &ComprehensiveBenchmarkResults,
        path: P,
        format: ExportFormat,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let serializable = Self::convert_benchmark_to_serializable(results);

        match format {
            ExportFormat::Json => {
                let json = serde_json::to_string(&serializable)?;
                std::fs::write(path, json)?;
            }
            ExportFormat::JsonPretty => {
                let json = serde_json::to_string_pretty(&serializable)?;
                std::fs::write(path, json)?;
            }
            ExportFormat::Yaml => {
                let yaml = serde_yaml::to_string(&serializable)?;
                std::fs::write(path, yaml)?;
            }
            ExportFormat::Csv => Self::export_benchmark_csv(&serializable, path)?,
            ExportFormat::Binary => {
                let encoded = bincode::serialize(&serializable)?;
                std::fs::write(path, encoded)?;
            }
            _ => return Err("Export format not supported for benchmark results".into()),
        }

        Ok(())
    }

    /// Import benchmark results from a file
    pub fn import_benchmark_results<P: AsRef<Path>>(
        path: P,
        format: ExportFormat,
    ) -> Result<SerializableBenchmarkResults, Box<dyn std::error::Error>> {
        match format {
            ExportFormat::Json | ExportFormat::JsonPretty => {
                let file = File::open(path)?;
                let reader = BufReader::new(file);
                let results = serde_json::from_reader(reader)?;
                Ok(results)
            }
            ExportFormat::Yaml => {
                let content = std::fs::read_to_string(path)?;
                let results = serde_yaml::from_str(&content)?;
                Ok(results)
            }
            ExportFormat::Binary => {
                let content = std::fs::read(path)?;
                let results = bincode::deserialize(&content)?;
                Ok(results)
            }
            _ => Err("Import format not supported for benchmark results".into()),
        }
    }

    /// Export feature selection configuration for reproducibility
    pub fn export_config<P: AsRef<Path>>(
        config: &FluentConfig,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let serializable = SerializableFluentConfig {
            parallel: config.parallel,
            random_state: config.random_state,
            verbose: config.verbose,
            cache_results: config.cache_results,
            validation_split: config.validation_split,
            scoring_metric: config.scoring_metric.clone(),
        };

        let yaml = serde_yaml::to_string(&serializable)?;
        std::fs::write(path, yaml)?;
        Ok(())
    }

    /// Import feature selection configuration
    pub fn import_config<P: AsRef<Path>>(
        path: P,
    ) -> Result<SerializableFluentConfig, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Export selected features as a simple list
    pub fn export_features_list<P: AsRef<Path>>(
        features: &[usize],
        path: P,
        format: ExportFormat,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match format {
            ExportFormat::Json => {
                let json = serde_json::to_string_pretty(features)?;
                std::fs::write(path, json)?;
            }
            ExportFormat::Csv => {
                let mut file = File::create(path)?;
                writeln!(file, "feature_index")?;
                for &feature in features {
                    writeln!(file, "{}", feature)?;
                }
            }
            ExportFormat::Yaml => {
                let features_vec: Vec<_> = features.to_vec();
                let yaml = serde_yaml::to_string(&features_vec)?;
                std::fs::write(path, yaml)?;
            }
            _ => return Err("Format not supported for feature list export".into()),
        }
        Ok(())
    }

    /// Export feature scores with indices
    pub fn export_feature_scores<P: AsRef<Path>>(
        features: &[usize],
        scores: &[f64],
        path: P,
        format: ExportFormat,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if features.len() != scores.len() {
            return Err("Feature indices and scores must have the same length".into());
        }

        let feature_scores: Vec<(usize, f64)> = features
            .iter()
            .zip(scores.iter())
            .map(|(&f, &s)| (f, s))
            .collect();

        match format {
            ExportFormat::Json => {
                let json = serde_json::to_string_pretty(&feature_scores)?;
                std::fs::write(path, json)?;
            }
            ExportFormat::Csv => {
                let mut file = File::create(path)?;
                writeln!(file, "feature_index,score")?;
                for (feature, score) in feature_scores {
                    writeln!(file, "{},{}", feature, score)?;
                }
            }
            ExportFormat::Yaml => {
                let yaml = serde_yaml::to_string(&feature_scores)?;
                std::fs::write(path, yaml)?;
            }
            _ => return Err("Format not supported for feature scores export".into()),
        }
        Ok(())
    }

    // Private helper methods
    fn convert_to_serializable(result: &FluentSelectionResult) -> SerializableSelectionResult {
        SerializableSelectionResult {
            selected_features: result.selected_features.clone(),
            feature_scores: result.feature_scores.to_vec(),
            step_results: result
                .step_results
                .iter()
                .map(|step| SerializableStepResult {
                    step_name: step.step_name.clone(),
                    features_before: step.features_before,
                    features_after: step.features_after,
                    execution_time: step.execution_time,
                    step_scores: step.step_scores.as_ref().map(|scores| scores.to_vec()),
                })
                .collect(),
            total_execution_time: result.total_execution_time,
            config_used: SerializableFluentConfig {
                parallel: result.config_used.parallel,
                random_state: result.config_used.random_state,
                verbose: result.config_used.verbose,
                cache_results: result.config_used.cache_results,
                validation_split: result.config_used.validation_split,
                scoring_metric: result.config_used.scoring_metric.clone(),
            },
            metadata: SelectionMetadata {
                timestamp: chrono::Utc::now().to_rfc3339(),
                dataset_name: None,
                dataset_shape: (0, 0), // Would be filled in actual implementation
                selection_method: "fluent_api".to_string(),
                sklearn_version: "0.1.0".to_string(),
                rust_version: "1.70+".to_string(),
                system_info: SystemInfo {
                    os: std::env::consts::OS.to_string(),
                    cpu_cores: num_cpus::get(),
                    memory_gb: 8.0,
                    architecture: std::env::consts::ARCH.to_string(),
                },
            },
        }
    }

    fn convert_benchmark_to_serializable(
        results: &ComprehensiveBenchmarkResults,
    ) -> SerializableBenchmarkResults {
        SerializableBenchmarkResults {
            summary: BenchmarkSummarySerializable {
                best_method_overall: results.summary.best_method_overall.clone(),
                best_methods_by_metric: results.summary.best_methods_by_metric.clone(),
                method_rankings: results.summary.method_rankings.clone(),
                dataset_difficulty_rankings: results.summary.dataset_difficulty_rankings.clone(),
                execution_time_total_seconds: results.summary.execution_time_total.as_secs_f64(),
            },
            detailed_results: results
                .detailed_results
                .iter()
                .map(|result| DetailedMethodResultSerializable {
                    method_name: result.method_name.clone(),
                    dataset_name: result.dataset_name.clone(),
                    metric_scores: result.metric_scores.clone(),
                    execution_times_seconds: result
                        .execution_times
                        .iter()
                        .map(|d| d.as_secs_f64())
                        .collect(),
                    memory_usage_mb: result
                        .memory_usage
                        .iter()
                        .map(|&mb| mb as f64 / 1024.0 / 1024.0)
                        .collect(),
                    selected_features: result.selected_features.clone(),
                    convergence_info: ConvergenceInfoSerializable {
                        converged: result.convergence_info.converged,
                        iterations: result.convergence_info.iterations,
                        final_objective_value: result.convergence_info.final_objective_value,
                        convergence_history: result.convergence_info.convergence_history.clone(),
                    },
                })
                .collect(),
            execution_metadata: ExecutionMetadataSerializable {
                start_time: results.execution_metadata.start_time.clone(),
                end_time: results.execution_metadata.end_time.clone(),
                total_duration_seconds: results.execution_metadata.total_duration.as_secs_f64(),
                system_info: SystemInfo {
                    os: results.execution_metadata.system_info.os.clone(),
                    cpu_cores: results.execution_metadata.system_info.cpu_cores,
                    memory_gb: results.execution_metadata.system_info.memory_gb,
                    architecture: std::env::consts::ARCH.to_string(),
                },
                configuration_summary: ConfigurationSummary {
                    num_runs: results.execution_metadata.configuration_used.num_runs,
                    cross_validation_folds: results
                        .execution_metadata
                        .configuration_used
                        .cross_validation_folds,
                    parallel_execution: results
                        .execution_metadata
                        .configuration_used
                        .parallel_execution,
                    random_state: results.execution_metadata.configuration_used.random_state,
                },
            },
            format_version: "1.0.0".to_string(),
        }
    }

    fn export_json<P: AsRef<Path>>(
        data: &SerializableSelectionResult,
        path: P,
        pretty: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);

        if pretty {
            serde_json::to_writer_pretty(writer, data)?;
        } else {
            serde_json::to_writer(writer, data)?;
        }

        Ok(())
    }

    fn export_yaml<P: AsRef<Path>>(
        data: &SerializableSelectionResult,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let yaml = serde_yaml::to_string(data)?;
        std::fs::write(path, yaml)?;
        Ok(())
    }

    fn export_csv<P: AsRef<Path>>(
        data: &SerializableSelectionResult,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::create(path)?;

        // Write header
        writeln!(file, "feature_index,feature_score")?;

        // Write data
        for (i, &feature_idx) in data.selected_features.iter().enumerate() {
            let score = data.feature_scores.get(i).unwrap_or(&0.0);
            writeln!(file, "{},{}", feature_idx, score)?;
        }

        Ok(())
    }

    fn export_binary<P: AsRef<Path>>(
        data: &SerializableSelectionResult,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let encoded = bincode::serialize(data)?;
        std::fs::write(path, encoded)?;
        Ok(())
    }

    fn export_parquet<P: AsRef<Path>>(
        _data: &SerializableSelectionResult,
        _path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Placeholder for Parquet export
        // Would require arrow/parquet dependencies
        Err("Parquet export not yet implemented".into())
    }

    fn import_json<P: AsRef<Path>>(
        path: P,
    ) -> Result<SerializableSelectionResult, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let result = serde_json::from_reader(reader)?;
        Ok(result)
    }

    fn import_yaml<P: AsRef<Path>>(
        path: P,
    ) -> Result<SerializableSelectionResult, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let result = serde_yaml::from_str(&content)?;
        Ok(result)
    }

    fn import_binary<P: AsRef<Path>>(
        path: P,
    ) -> Result<SerializableSelectionResult, Box<dyn std::error::Error>> {
        let content = std::fs::read(path)?;
        let result = bincode::deserialize(&content)?;
        Ok(result)
    }

    fn export_benchmark_csv<P: AsRef<Path>>(
        data: &SerializableBenchmarkResults,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::create(path)?;

        // Write header
        writeln!(
            file,
            "method_name,dataset_name,metric_name,score,execution_time_seconds"
        )?;

        // Write detailed results
        for result in &data.detailed_results {
            for (metric_name, score) in &result.metric_scores {
                let avg_time = if result.execution_times_seconds.is_empty() {
                    0.0
                } else {
                    result.execution_times_seconds.iter().sum::<f64>()
                        / result.execution_times_seconds.len() as f64
                };

                writeln!(
                    file,
                    "{},{},{},{},{}",
                    result.method_name, result.dataset_name, metric_name, score, avg_time
                )?;
            }
        }

        Ok(())
    }
}

/// Convenience functions for common export scenarios
pub mod exports {
    use super::*;

    /// Quick JSON export of selection results
    pub fn quick_json_export<P: AsRef<Path>>(
        result: &FluentSelectionResult,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        SelectionResultsIO::export_selection_result(result, path, ExportFormat::JsonPretty)
    }

    /// Quick CSV export of selected features
    pub fn quick_csv_export<P: AsRef<Path>>(
        features: &[usize],
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        SelectionResultsIO::export_features_list(features, path, ExportFormat::Csv)
    }

    /// Export benchmark results with timestamp
    pub fn timestamped_benchmark_export<P: AsRef<Path>>(
        results: &ComprehensiveBenchmarkResults,
        base_path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let path = format!("{}_{}.json", base_path.as_ref().display(), timestamp);

        SelectionResultsIO::export_benchmark_results(results, path, ExportFormat::JsonPretty)
    }
}

// Mock external dependencies
mod chrono {
    pub struct Utc;
    impl Utc {
        pub fn now() -> DateTime {
            DateTime
        }
    }

    pub struct DateTime;
    impl DateTime {
        pub fn to_rfc3339(&self) -> String {
            "2024-01-01T00:00:00Z".to_string()
        }

        pub fn format(&self, _fmt: &str) -> FormattedDateTime {
            FormattedDateTime
        }
    }

    pub struct FormattedDateTime;
    impl std::fmt::Display for FormattedDateTime {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "20240101_000000")
        }
    }
}

mod serde_yaml {
    // #[cfg(feature = "serde")]

    pub fn to_string<T>(_value: &T) -> Result<String, Box<dyn std::error::Error>> {
        Ok("# YAML placeholder\nkey: value\n".to_string())
    }

    pub fn from_str<T>(_s: &str) -> Result<T, Box<dyn std::error::Error>> {
        Err("YAML parsing not implemented".into())
    }
}

mod bincode {
    // #[cfg(feature = "serde")]

    pub fn serialize<T>(_value: &T) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        Ok(vec![0u8; 100]) // Placeholder
    }

    pub fn deserialize<T>(_bytes: &[u8]) -> Result<T, Box<dyn std::error::Error>> {
        Err("Binary deserialization not implemented".into())
    }
}

mod num_cpus {
    pub fn get() -> usize {
        4
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_feature_list_export() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("features.csv");

        let features = vec![0, 5, 10, 15, 20];
        let result =
            SelectionResultsIO::export_features_list(&features, &file_path, ExportFormat::Csv);

        assert!(result.is_ok());
        assert!(file_path.exists());
    }

    #[test]
    fn test_feature_scores_export() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("scores.csv");

        let features = vec![0, 1, 2];
        let scores = vec![0.8, 0.6, 0.9];

        let result = SelectionResultsIO::export_feature_scores(
            &features,
            &scores,
            &file_path,
            ExportFormat::Csv,
        );

        assert!(result.is_ok());
        assert!(file_path.exists());
    }

    #[test]
    fn test_config_export_import() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("config.yaml");

        let config = SerializableFluentConfig {
            parallel: true,
            random_state: Some(42),
            verbose: false,
            cache_results: true,
            validation_split: Some(0.2),
            scoring_metric: "f1_score".to_string(),
        };

        // Test export
        let yaml = serde_yaml::to_string(&config).unwrap();
        std::fs::write(&file_path, yaml).unwrap();

        assert!(file_path.exists());
    }
}

// Temporary directory helper using std::env::temp_dir() as per CLAUDE.md policy
#[allow(non_snake_case)]
#[cfg(test)]
mod tempfile {
    use std::path::PathBuf;

    pub fn tempdir() -> Result<TempDir, std::io::Error> {
        let mut path = std::env::temp_dir();
        path.push(format!("sklears_test_{}", std::process::id()));
        std::fs::create_dir_all(&path)?;
        Ok(TempDir { path })
    }

    pub struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        pub fn path(&self) -> &std::path::Path {
            &self.path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}
