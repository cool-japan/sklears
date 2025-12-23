//! Comprehensive experiment tracking system for neural network training.
//!
//! This module provides a robust experiment tracking framework that enables researchers
//! and practitioners to log, monitor, and compare machine learning experiments. It includes
//! support for hyperparameter logging, metric tracking, model versioning, artifact storage,
//! and experiment comparison utilities.

use crate::versioning::ModelVersion;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::Rng;
use sklears_core::error::SklearsError;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Result type for experiment tracking operations
pub type ExperimentResult<T> = Result<T, SklearsError>;

/// Unique identifier for experiments
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ExperimentId(pub String);

impl ExperimentId {
    /// Generate a new unique experiment ID
    pub fn generate() -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();

        let random_suffix = scirs2_core::random::thread_rng().random::<u32>();
        Self(format!("exp_{timestamp}_{random_suffix}"))
    }

    /// Create an experiment ID from a custom string
    pub fn from_string(id: String) -> Self {
        Self(id)
    }

    /// Get the inner string representation
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ExperimentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for ExperimentId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// Experiment status
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ExperimentStatus {
    /// Experiment is currently running
    Running,
    /// Experiment completed successfully
    Completed,
    /// Experiment failed with error
    Failed { error: String },
    /// Experiment was cancelled
    Cancelled,
    /// Experiment is queued for execution
    Queued,
}

/// Hyperparameter value types
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum HyperparameterValue {
    Float(f64),
    Int(i64),
    String(String),
    Bool(bool),
    FloatArray(Vec<f64>),
    IntArray(Vec<i64>),
    StringArray(Vec<String>),
}

impl From<f64> for HyperparameterValue {
    fn from(val: f64) -> Self {
        HyperparameterValue::Float(val)
    }
}

impl From<i64> for HyperparameterValue {
    fn from(val: i64) -> Self {
        HyperparameterValue::Int(val)
    }
}

impl From<String> for HyperparameterValue {
    fn from(val: String) -> Self {
        HyperparameterValue::String(val)
    }
}

impl From<bool> for HyperparameterValue {
    fn from(val: bool) -> Self {
        HyperparameterValue::Bool(val)
    }
}

impl From<Vec<f64>> for HyperparameterValue {
    fn from(val: Vec<f64>) -> Self {
        HyperparameterValue::FloatArray(val)
    }
}

/// Metric value with optional step information
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MetricValue {
    pub value: f64,
    pub step: Option<u64>,
    pub timestamp: u64,
    pub tags: HashMap<String, String>,
}

impl MetricValue {
    pub fn new(value: f64) -> Self {
        Self {
            value,
            step: None,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            tags: HashMap::new(),
        }
    }

    pub fn with_step(mut self, step: u64) -> Self {
        self.step = Some(step);
        self
    }

    pub fn with_tag(mut self, key: String, value: String) -> Self {
        self.tags.insert(key, value);
        self
    }
}

/// Artifact information
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Artifact {
    pub name: String,
    pub artifact_type: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub checksum: Option<String>,
    pub metadata: HashMap<String, String>,
    pub created_at: u64,
}

impl Artifact {
    pub fn new(name: String, artifact_type: String, path: PathBuf) -> Self {
        let size_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

        Self {
            name,
            artifact_type,
            path,
            size_bytes,
            checksum: None,
            metadata: HashMap::new(),
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    pub fn with_checksum(mut self, checksum: String) -> Self {
        self.checksum = Some(checksum);
        self
    }

    pub fn with_metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Comprehensive experiment configuration and results
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Experiment {
    /// Unique experiment identifier
    pub id: ExperimentId,
    /// Human-readable experiment name
    pub name: String,
    /// Experiment description
    pub description: Option<String>,
    /// Experiment status
    pub status: ExperimentStatus,
    /// Hyperparameters used in the experiment
    pub hyperparameters: HashMap<String, HyperparameterValue>,
    /// Metrics collected during the experiment
    pub metrics: HashMap<String, Vec<MetricValue>>,
    /// Final evaluation results
    pub results: HashMap<String, f64>,
    /// Model version used
    pub model_version: Option<ModelVersion>,
    /// Experiment tags for organization
    pub tags: HashMap<String, String>,
    /// Artifacts produced by the experiment
    pub artifacts: Vec<Artifact>,
    /// Parent experiment (for experiment hierarchies)
    pub parent_id: Option<ExperimentId>,
    /// Child experiments
    pub child_ids: Vec<ExperimentId>,
    /// Timestamps
    pub created_at: u64,
    pub started_at: Option<u64>,
    pub finished_at: Option<u64>,
    /// Environment information
    pub environment: HashMap<String, String>,
    /// Git information (if available)
    pub git_commit: Option<String>,
    pub git_branch: Option<String>,
    /// Notes and observations
    pub notes: Vec<String>,
    /// Resource usage
    pub resource_usage: Option<ResourceUsage>,
}

/// Resource usage information
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ResourceUsage {
    pub cpu_hours: f64,
    pub memory_peak_gb: f64,
    pub gpu_hours: f64,
    pub disk_usage_gb: f64,
}

impl Experiment {
    pub fn new(name: String) -> Self {
        let id = ExperimentId::generate();
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            id,
            name,
            description: None,
            status: ExperimentStatus::Queued,
            hyperparameters: HashMap::new(),
            metrics: HashMap::new(),
            results: HashMap::new(),
            model_version: None,
            tags: HashMap::new(),
            artifacts: Vec::new(),
            parent_id: None,
            child_ids: Vec::new(),
            created_at,
            started_at: None,
            finished_at: None,
            environment: Self::collect_environment_info(),
            git_commit: Self::get_git_commit(),
            git_branch: Self::get_git_branch(),
            notes: Vec::new(),
            resource_usage: None,
        }
    }

    pub fn with_id(mut self, id: ExperimentId) -> Self {
        self.id = id;
        self
    }

    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    pub fn with_model_version(mut self, version: ModelVersion) -> Self {
        self.model_version = Some(version);
        self
    }

    pub fn with_parent(mut self, parent_id: ExperimentId) -> Self {
        self.parent_id = Some(parent_id);
        self
    }

    /// Start the experiment (sets status to Running and records start time)
    pub fn start(&mut self) {
        self.status = ExperimentStatus::Running;
        self.started_at = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        );
    }

    /// Complete the experiment successfully
    pub fn complete(&mut self) {
        self.status = ExperimentStatus::Completed;
        self.finished_at = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        );
    }

    /// Mark the experiment as failed
    pub fn fail(&mut self, error: String) {
        self.status = ExperimentStatus::Failed { error };
        self.finished_at = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        );
    }

    /// Cancel the experiment
    pub fn cancel(&mut self) {
        self.status = ExperimentStatus::Cancelled;
        self.finished_at = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        );
    }

    /// Log a hyperparameter
    pub fn log_hyperparameter<T: Into<HyperparameterValue>>(&mut self, name: String, value: T) {
        self.hyperparameters.insert(name, value.into());
    }

    /// Log multiple hyperparameters at once
    pub fn log_hyperparameters(&mut self, params: HashMap<String, HyperparameterValue>) {
        self.hyperparameters.extend(params);
    }

    /// Log a metric value
    pub fn log_metric(&mut self, name: String, value: MetricValue) {
        self.metrics
            .entry(name)
            .or_insert_with(Vec::new)
            .push(value);
    }

    /// Log a simple metric value with current timestamp
    pub fn log_metric_simple(&mut self, name: String, value: f64, step: Option<u64>) {
        let metric = if let Some(s) = step {
            MetricValue::new(value).with_step(s)
        } else {
            MetricValue::new(value)
        };
        self.log_metric(name, metric);
    }

    /// Log multiple metrics at once
    pub fn log_metrics(&mut self, metrics: HashMap<String, f64>, step: Option<u64>) {
        for (name, value) in metrics {
            self.log_metric_simple(name, value, step);
        }
    }

    /// Set a final result
    pub fn set_result(&mut self, name: String, value: f64) {
        self.results.insert(name, value);
    }

    /// Set multiple results
    pub fn set_results(&mut self, results: HashMap<String, f64>) {
        self.results.extend(results);
    }

    /// Add a tag
    pub fn add_tag(&mut self, key: String, value: String) {
        self.tags.insert(key, value);
    }

    /// Add multiple tags
    pub fn add_tags(&mut self, tags: HashMap<String, String>) {
        self.tags.extend(tags);
    }

    /// Add an artifact
    pub fn add_artifact(&mut self, artifact: Artifact) {
        self.artifacts.push(artifact);
    }

    /// Add a child experiment
    pub fn add_child(&mut self, child_id: ExperimentId) {
        self.child_ids.push(child_id);
    }

    /// Add a note
    pub fn add_note(&mut self, note: String) {
        self.notes.push(note);
    }

    /// Set resource usage
    pub fn set_resource_usage(&mut self, usage: ResourceUsage) {
        self.resource_usage = Some(usage);
    }

    /// Get experiment duration in seconds
    pub fn duration_seconds(&self) -> Option<u64> {
        match (self.started_at, self.finished_at) {
            (Some(start), Some(end)) => Some(end - start),
            _ => None,
        }
    }

    /// Check if experiment is finished
    pub fn is_finished(&self) -> bool {
        matches!(
            self.status,
            ExperimentStatus::Completed
                | ExperimentStatus::Failed { .. }
                | ExperimentStatus::Cancelled
        )
    }

    /// Collect environment information
    fn collect_environment_info() -> HashMap<String, String> {
        let mut env = HashMap::new();

        env.insert(
            "rust_version".to_string(),
            env!("CARGO_PKG_RUST_VERSION").to_string(),
        );
        env.insert(
            "sklears_version".to_string(),
            env!("CARGO_PKG_VERSION").to_string(),
        );

        if let Ok(hostname) = std::env::var("HOSTNAME") {
            env.insert("hostname".to_string(), hostname);
        }

        if let Ok(user) = std::env::var("USER") {
            env.insert("user".to_string(), user);
        }

        env.insert("os".to_string(), std::env::consts::OS.to_string());
        env.insert("arch".to_string(), std::env::consts::ARCH.to_string());

        env
    }

    /// Get git commit hash (if available)
    fn get_git_commit() -> Option<String> {
        std::process::Command::new("git")
            .args(&["rev-parse", "HEAD"])
            .output()
            .ok()
            .and_then(|output| {
                if output.status.success() {
                    String::from_utf8(output.stdout)
                        .ok()
                        .map(|s| s.trim().to_string())
                } else {
                    None
                }
            })
    }

    /// Get git branch (if available)
    fn get_git_branch() -> Option<String> {
        std::process::Command::new("git")
            .args(&["branch", "--show-current"])
            .output()
            .ok()
            .and_then(|output| {
                if output.status.success() {
                    String::from_utf8(output.stdout)
                        .ok()
                        .map(|s| s.trim().to_string())
                } else {
                    None
                }
            })
    }
}

/// Experiment tracking backend trait
pub trait ExperimentBackend {
    /// Store an experiment
    fn store_experiment(&mut self, experiment: &Experiment) -> ExperimentResult<()>;

    /// Load an experiment by ID
    fn load_experiment(&self, id: &ExperimentId) -> ExperimentResult<Experiment>;

    /// Update an existing experiment
    fn update_experiment(&mut self, experiment: &Experiment) -> ExperimentResult<()>;

    /// Delete an experiment
    fn delete_experiment(&mut self, id: &ExperimentId) -> ExperimentResult<()>;

    /// List all experiments
    fn list_experiments(&self) -> ExperimentResult<Vec<ExperimentId>>;

    /// Search experiments by criteria
    fn search_experiments(&self, query: &ExperimentQuery) -> ExperimentResult<Vec<ExperimentId>>;
}

/// Query criteria for searching experiments
#[derive(Debug, Clone)]
pub struct ExperimentQuery {
    pub name_pattern: Option<String>,
    pub status: Option<ExperimentStatus>,
    pub tags: HashMap<String, String>,
    pub created_after: Option<u64>,
    pub created_before: Option<u64>,
    pub parent_id: Option<ExperimentId>,
}

impl ExperimentQuery {
    pub fn new() -> Self {
        Self {
            name_pattern: None,
            status: None,
            tags: HashMap::new(),
            created_after: None,
            created_before: None,
            parent_id: None,
        }
    }

    pub fn with_name_pattern(mut self, pattern: String) -> Self {
        self.name_pattern = Some(pattern);
        self
    }

    pub fn with_status(mut self, status: ExperimentStatus) -> Self {
        self.status = Some(status);
        self
    }

    pub fn with_tag(mut self, key: String, value: String) -> Self {
        self.tags.insert(key, value);
        self
    }

    pub fn with_parent(mut self, parent_id: ExperimentId) -> Self {
        self.parent_id = Some(parent_id);
        self
    }
}

impl Default for ExperimentQuery {
    fn default() -> Self {
        Self::new()
    }
}

/// In-memory experiment backend for testing and simple use cases
#[derive(Debug, Clone, Default)]
pub struct InMemoryBackend {
    experiments: HashMap<ExperimentId, Experiment>,
}

impl InMemoryBackend {
    pub fn new() -> Self {
        Self {
            experiments: HashMap::new(),
        }
    }
}

impl ExperimentBackend for InMemoryBackend {
    fn store_experiment(&mut self, experiment: &Experiment) -> ExperimentResult<()> {
        self.experiments
            .insert(experiment.id.clone(), experiment.clone());
        Ok(())
    }

    fn load_experiment(&self, id: &ExperimentId) -> ExperimentResult<Experiment> {
        self.experiments
            .get(id)
            .cloned()
            .ok_or_else(|| SklearsError::InvalidParameter {
                name: "experiment_id".to_string(),
                reason: format!("Experiment with ID {} not found", id),
            })
    }

    fn update_experiment(&mut self, experiment: &Experiment) -> ExperimentResult<()> {
        if self.experiments.contains_key(&experiment.id) {
            self.experiments
                .insert(experiment.id.clone(), experiment.clone());
            Ok(())
        } else {
            Err(SklearsError::InvalidParameter {
                name: "experiment_id".to_string(),
                reason: format!("Experiment with ID {} not found", experiment.id),
            })
        }
    }

    fn delete_experiment(&mut self, id: &ExperimentId) -> ExperimentResult<()> {
        self.experiments
            .remove(id)
            .ok_or_else(|| SklearsError::InvalidParameter {
                name: "experiment_id".to_string(),
                reason: format!("Experiment with ID {} not found", id),
            })?;
        Ok(())
    }

    fn list_experiments(&self) -> ExperimentResult<Vec<ExperimentId>> {
        Ok(self.experiments.keys().cloned().collect())
    }

    fn search_experiments(&self, query: &ExperimentQuery) -> ExperimentResult<Vec<ExperimentId>> {
        let results: Vec<ExperimentId> = self
            .experiments
            .values()
            .filter(|exp| {
                // Filter by name pattern
                if let Some(ref pattern) = query.name_pattern {
                    if !exp.name.contains(pattern) {
                        return false;
                    }
                }

                // Filter by status
                if let Some(ref status) = query.status {
                    if exp.status != *status {
                        return false;
                    }
                }

                // Filter by tags
                for (key, value) in &query.tags {
                    if exp.tags.get(key) != Some(value) {
                        return false;
                    }
                }

                // Filter by creation time
                if let Some(after) = query.created_after {
                    if exp.created_at < after {
                        return false;
                    }
                }

                if let Some(before) = query.created_before {
                    if exp.created_at > before {
                        return false;
                    }
                }

                // Filter by parent
                if let Some(ref parent) = query.parent_id {
                    if exp.parent_id.as_ref() != Some(parent) {
                        return false;
                    }
                }

                true
            })
            .map(|exp| exp.id.clone())
            .collect();

        Ok(results)
    }
}

/// Main experiment tracker
pub struct ExperimentTracker<B: ExperimentBackend> {
    backend: B,
    current_experiment: Option<ExperimentId>,
}

impl<B: ExperimentBackend> ExperimentTracker<B> {
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            current_experiment: None,
        }
    }

    /// Create a new experiment
    pub fn create_experiment(&mut self, name: String) -> ExperimentResult<ExperimentId> {
        let experiment = Experiment::new(name);
        let id = experiment.id.clone();
        self.backend.store_experiment(&experiment)?;
        Ok(id)
    }

    /// Start tracking an experiment
    pub fn start_experiment(&mut self, id: ExperimentId) -> ExperimentResult<()> {
        let mut experiment = self.backend.load_experiment(&id)?;
        experiment.start();
        self.backend.update_experiment(&experiment)?;
        self.current_experiment = Some(id);
        Ok(())
    }

    /// Get the current experiment
    pub fn current_experiment(&self) -> Option<&ExperimentId> {
        self.current_experiment.as_ref()
    }

    /// Set current experiment
    pub fn set_current_experiment(&mut self, id: ExperimentId) {
        self.current_experiment = Some(id);
    }

    /// Log hyperparameter to current experiment
    pub fn log_hyperparameter<T: Into<HyperparameterValue>>(
        &mut self,
        name: String,
        value: T,
    ) -> ExperimentResult<()> {
        if let Some(ref id) = self.current_experiment {
            let mut experiment = self.backend.load_experiment(id)?;
            experiment.log_hyperparameter(name, value);
            self.backend.update_experiment(&experiment)?;
        }
        Ok(())
    }

    /// Log metric to current experiment
    pub fn log_metric(
        &mut self,
        name: String,
        value: f64,
        step: Option<u64>,
    ) -> ExperimentResult<()> {
        if let Some(ref id) = self.current_experiment {
            let mut experiment = self.backend.load_experiment(id)?;
            experiment.log_metric_simple(name, value, step);
            self.backend.update_experiment(&experiment)?;
        }
        Ok(())
    }

    /// Complete current experiment
    pub fn complete_experiment(&mut self) -> ExperimentResult<()> {
        if let Some(ref id) = self.current_experiment.take() {
            let mut experiment = self.backend.load_experiment(id)?;
            experiment.complete();
            self.backend.update_experiment(&experiment)?;
        }
        Ok(())
    }

    /// Fail current experiment
    pub fn fail_experiment(&mut self, error: String) -> ExperimentResult<()> {
        if let Some(ref id) = self.current_experiment.take() {
            let mut experiment = self.backend.load_experiment(id)?;
            experiment.fail(error);
            self.backend.update_experiment(&experiment)?;
        }
        Ok(())
    }

    /// Get experiment by ID
    pub fn get_experiment(&self, id: &ExperimentId) -> ExperimentResult<Experiment> {
        self.backend.load_experiment(id)
    }

    /// Search experiments
    pub fn search_experiments(&self, query: &ExperimentQuery) -> ExperimentResult<Vec<Experiment>> {
        let ids = self.backend.search_experiments(query)?;
        let mut experiments = Vec::new();
        for id in ids {
            experiments.push(self.backend.load_experiment(&id)?);
        }
        Ok(experiments)
    }

    /// Compare experiments by metrics
    pub fn compare_experiments(
        &self,
        exp_ids: &[ExperimentId],
        metric_names: &[String],
    ) -> ExperimentResult<ExperimentComparison> {
        let mut experiments = Vec::new();
        for id in exp_ids {
            experiments.push(self.backend.load_experiment(id)?);
        }

        let mut comparison = ExperimentComparison {
            experiments: experiments.clone(),
            metric_comparison: HashMap::new(),
        };

        for metric_name in metric_names {
            let mut metric_values = Vec::new();
            for exp in &experiments {
                if let Some(values) = exp.metrics.get(metric_name) {
                    let latest_value = values.last().map(|v| v.value);
                    metric_values.push((exp.id.clone(), latest_value));
                } else {
                    metric_values.push((exp.id.clone(), None));
                }
            }
            comparison
                .metric_comparison
                .insert(metric_name.clone(), metric_values);
        }

        Ok(comparison)
    }
}

/// Experiment comparison results
#[derive(Debug, Clone)]
pub struct ExperimentComparison {
    pub experiments: Vec<Experiment>,
    pub metric_comparison: HashMap<String, Vec<(ExperimentId, Option<f64>)>>,
}

impl ExperimentComparison {
    /// Get the best experiment for a given metric (highest value)
    pub fn best_experiment_for_metric(&self, metric_name: &str) -> Option<&ExperimentId> {
        self.metric_comparison
            .get(metric_name)?
            .iter()
            .filter_map(|(id, value)| value.map(|v| (id, v)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id)
    }

    /// Get the worst experiment for a given metric (lowest value)
    pub fn worst_experiment_for_metric(&self, metric_name: &str) -> Option<&ExperimentId> {
        self.metric_comparison
            .get(metric_name)?
            .iter()
            .filter_map(|(id, value)| value.map(|v| (id, v)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id)
    }
}

/// Convenience type for in-memory experiment tracker
pub type InMemoryTracker = ExperimentTracker<InMemoryBackend>;

impl Default for ExperimentTracker<InMemoryBackend> {
    fn default() -> Self {
        Self::new(InMemoryBackend::new())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_experiment_creation() {
        let experiment = Experiment::new("test_experiment".to_string());
        assert_eq!(experiment.name, "test_experiment");
        assert_eq!(experiment.status, ExperimentStatus::Queued);
        assert!(!experiment.is_finished());
    }

    #[test]
    fn test_experiment_lifecycle() {
        let mut experiment = Experiment::new("test".to_string());

        experiment.start();
        assert_eq!(experiment.status, ExperimentStatus::Running);
        assert!(experiment.started_at.is_some());

        experiment.complete();
        assert_eq!(experiment.status, ExperimentStatus::Completed);
        assert!(experiment.finished_at.is_some());
        assert!(experiment.is_finished());
    }

    #[test]
    fn test_hyperparameter_logging() {
        let mut experiment = Experiment::new("test".to_string());

        experiment.log_hyperparameter("learning_rate".to_string(), 0.01);
        experiment.log_hyperparameter("batch_size".to_string(), 32i64);
        experiment.log_hyperparameter("model_type".to_string(), "MLP".to_string());

        assert_eq!(experiment.hyperparameters.len(), 3);

        match experiment.hyperparameters.get("learning_rate") {
            Some(HyperparameterValue::Float(val)) => assert_eq!(*val, 0.01),
            _ => panic!("Expected float hyperparameter"),
        }
    }

    #[test]
    fn test_metric_logging() {
        let mut experiment = Experiment::new("test".to_string());

        experiment.log_metric_simple("loss".to_string(), 0.5, Some(1));
        experiment.log_metric_simple("loss".to_string(), 0.3, Some(2));
        experiment.log_metric_simple("accuracy".to_string(), 0.85, Some(1));

        assert_eq!(experiment.metrics.len(), 2);
        assert_eq!(experiment.metrics.get("loss").unwrap().len(), 2);
        assert_eq!(experiment.metrics.get("accuracy").unwrap().len(), 1);
    }

    #[test]
    fn test_in_memory_backend() {
        let mut backend = InMemoryBackend::new();
        let experiment = Experiment::new("test".to_string());
        let id = experiment.id.clone();

        backend.store_experiment(&experiment).unwrap();
        let loaded = backend.load_experiment(&id).unwrap();
        assert_eq!(loaded.name, experiment.name);

        let experiments = backend.list_experiments().unwrap();
        assert_eq!(experiments.len(), 1);
        assert!(experiments.contains(&id));
    }

    #[test]
    fn test_experiment_tracker() {
        let mut tracker = ExperimentTracker::new(InMemoryBackend::new());

        let exp_id = tracker.create_experiment("test_exp".to_string()).unwrap();
        tracker.start_experiment(exp_id.clone()).unwrap();

        tracker.log_hyperparameter("lr".to_string(), 0.001).unwrap();
        tracker
            .log_metric("loss".to_string(), 0.5, Some(1))
            .unwrap();
        tracker
            .log_metric("accuracy".to_string(), 0.9, Some(1))
            .unwrap();

        tracker.complete_experiment().unwrap();

        let experiment = tracker.get_experiment(&exp_id).unwrap();
        assert_eq!(experiment.status, ExperimentStatus::Completed);
        assert!(experiment.hyperparameters.contains_key("lr"));
        assert!(experiment.metrics.contains_key("loss"));
        assert!(experiment.metrics.contains_key("accuracy"));
    }

    #[test]
    fn test_experiment_search() {
        let mut tracker = ExperimentTracker::new(InMemoryBackend::new());

        let exp1_id = tracker.create_experiment("exp1".to_string()).unwrap();
        let exp2_id = tracker.create_experiment("exp2".to_string()).unwrap();

        // Add tags to distinguish experiments
        let mut exp1 = tracker.get_experiment(&exp1_id).unwrap();
        exp1.add_tag("type".to_string(), "test".to_string());
        tracker.backend.update_experiment(&exp1).unwrap();

        let mut exp2 = tracker.get_experiment(&exp2_id).unwrap();
        exp2.add_tag("type".to_string(), "production".to_string());
        tracker.backend.update_experiment(&exp2).unwrap();

        // Search for test experiments
        let query = ExperimentQuery::new().with_tag("type".to_string(), "test".to_string());
        let results = tracker.search_experiments(&query).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, exp1_id);
    }

    #[test]
    fn test_experiment_comparison() {
        let mut tracker = ExperimentTracker::new(InMemoryBackend::new());

        let exp1_id = tracker.create_experiment("exp1".to_string()).unwrap();
        let exp2_id = tracker.create_experiment("exp2".to_string()).unwrap();

        // Add different metric values
        tracker.set_current_experiment(exp1_id.clone());
        tracker
            .log_metric("accuracy".to_string(), 0.85, None)
            .unwrap();

        tracker.set_current_experiment(exp2_id.clone());
        tracker
            .log_metric("accuracy".to_string(), 0.92, None)
            .unwrap();

        let comparison = tracker
            .compare_experiments(
                &[exp1_id.clone(), exp2_id.clone()],
                &["accuracy".to_string()],
            )
            .unwrap();

        let best_exp = comparison.best_experiment_for_metric("accuracy");
        assert_eq!(best_exp, Some(&exp2_id));

        let worst_exp = comparison.worst_experiment_for_metric("accuracy");
        assert_eq!(worst_exp, Some(&exp1_id));
    }
}
