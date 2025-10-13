//! Configuration Management Utilities
//!
//! This module provides comprehensive configuration management utilities for ML applications,
//! including file parsing, environment variables, command-line arguments, validation, and hot-reloading.

use crate::UtilsError;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};

/// Configuration value types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ConfigValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Array(Vec<ConfigValue>),
    Object(HashMap<String, ConfigValue>),
    Null,
}

impl ConfigValue {
    /// Convert to string
    pub fn as_string(&self) -> Option<String> {
        match self {
            ConfigValue::String(s) => Some(s.clone()),
            ConfigValue::Integer(i) => Some(i.to_string()),
            ConfigValue::Float(f) => Some(f.to_string()),
            ConfigValue::Boolean(b) => Some(b.to_string()),
            _ => None,
        }
    }

    /// Convert to integer
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            ConfigValue::Integer(i) => Some(*i),
            ConfigValue::Float(f) => Some(*f as i64),
            ConfigValue::String(s) => s.parse().ok(),
            _ => None,
        }
    }

    /// Convert to float
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ConfigValue::Float(f) => Some(*f),
            ConfigValue::Integer(i) => Some(*i as f64),
            ConfigValue::String(s) => s.parse().ok(),
            _ => None,
        }
    }

    /// Convert to boolean
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ConfigValue::Boolean(b) => Some(*b),
            ConfigValue::String(s) => match s.to_lowercase().as_str() {
                "true" | "yes" | "on" | "1" => Some(true),
                "false" | "no" | "off" | "0" => Some(false),
                _ => None,
            },
            ConfigValue::Integer(i) => Some(*i != 0),
            _ => None,
        }
    }

    /// Convert to array
    pub fn as_array(&self) -> Option<&Vec<ConfigValue>> {
        match self {
            ConfigValue::Array(arr) => Some(arr),
            _ => None,
        }
    }

    /// Convert to object
    pub fn as_object(&self) -> Option<&HashMap<String, ConfigValue>> {
        match self {
            ConfigValue::Object(obj) => Some(obj),
            _ => None,
        }
    }

    /// Check if value is null
    pub fn is_null(&self) -> bool {
        matches!(self, ConfigValue::Null)
    }
}

impl From<String> for ConfigValue {
    fn from(s: String) -> Self {
        ConfigValue::String(s)
    }
}

impl From<&str> for ConfigValue {
    fn from(s: &str) -> Self {
        ConfigValue::String(s.to_string())
    }
}

impl From<i64> for ConfigValue {
    fn from(i: i64) -> Self {
        ConfigValue::Integer(i)
    }
}

impl From<f64> for ConfigValue {
    fn from(f: f64) -> Self {
        ConfigValue::Float(f)
    }
}

impl From<bool> for ConfigValue {
    fn from(b: bool) -> Self {
        ConfigValue::Boolean(b)
    }
}

/// Configuration manager with hierarchical support
#[derive(Debug, Clone)]
pub struct Config {
    data: HashMap<String, ConfigValue>,
    sources: Vec<ConfigSource>,
    metadata: ConfigMetadata,
}

#[derive(Debug, Clone)]
pub struct ConfigMetadata {
    pub loaded_at: SystemTime,
    pub source_files: Vec<PathBuf>,
    pub env_vars_used: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum ConfigSource {
    File(PathBuf),
    Environment,
    CommandLine,
    Default,
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

impl Config {
    /// Create a new empty configuration
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            sources: Vec::new(),
            metadata: ConfigMetadata {
                loaded_at: SystemTime::now(),
                source_files: Vec::new(),
                env_vars_used: Vec::new(),
            },
        }
    }

    /// Load configuration from a file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, UtilsError> {
        let path = path.as_ref();
        let content = fs::read_to_string(path).map_err(|e| {
            UtilsError::InvalidParameter(format!("Failed to read config file: {e}"))
        })?;

        let mut config = Self::new();
        config.sources.push(ConfigSource::File(path.to_path_buf()));
        config.metadata.source_files.push(path.to_path_buf());

        match path.extension().and_then(|ext| ext.to_str()) {
            Some("json") => config.load_json(&content)?,
            Some("toml") => config.load_toml(&content)?,
            Some("yaml") | Some("yml") => config.load_yaml(&content)?,
            _ => {
                return Err(UtilsError::InvalidParameter(
                    "Unsupported config file format".to_string(),
                ))
            }
        }

        Ok(config)
    }

    /// Load JSON configuration
    fn load_json(&mut self, content: &str) -> Result<(), UtilsError> {
        let json_value: serde_json::Value = serde_json::from_str(content)
            .map_err(|e| UtilsError::InvalidParameter(format!("Invalid JSON: {e}")))?;

        self.data = Self::json_to_config_map(json_value);
        Ok(())
    }

    /// Load TOML configuration (simplified - would need toml crate in practice)
    fn load_toml(&mut self, _content: &str) -> Result<(), UtilsError> {
        // This is a placeholder - in practice you'd use the toml crate
        Err(UtilsError::InvalidParameter(
            "TOML support not implemented".to_string(),
        ))
    }

    /// Load YAML configuration (simplified - would need yaml crate in practice)
    fn load_yaml(&mut self, _content: &str) -> Result<(), UtilsError> {
        // This is a placeholder - in practice you'd use the yaml crate
        Err(UtilsError::InvalidParameter(
            "YAML support not implemented".to_string(),
        ))
    }

    /// Convert JSON value to config map
    fn json_to_config_map(json: serde_json::Value) -> HashMap<String, ConfigValue> {
        match json {
            serde_json::Value::Object(map) => map
                .into_iter()
                .map(|(k, v)| (k, Self::json_value_to_config_value(v)))
                .collect(),
            _ => HashMap::new(),
        }
    }

    /// Convert JSON value to config value
    fn json_value_to_config_value(json: serde_json::Value) -> ConfigValue {
        match json {
            serde_json::Value::String(s) => ConfigValue::String(s),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    ConfigValue::Integer(i)
                } else if let Some(f) = n.as_f64() {
                    ConfigValue::Float(f)
                } else {
                    ConfigValue::Null
                }
            }
            serde_json::Value::Bool(b) => ConfigValue::Boolean(b),
            serde_json::Value::Array(arr) => ConfigValue::Array(
                arr.into_iter()
                    .map(Self::json_value_to_config_value)
                    .collect(),
            ),
            serde_json::Value::Object(obj) => ConfigValue::Object(
                obj.into_iter()
                    .map(|(k, v)| (k, Self::json_value_to_config_value(v)))
                    .collect(),
            ),
            serde_json::Value::Null => ConfigValue::Null,
        }
    }

    /// Load environment variables with prefix
    pub fn load_environment(&mut self, prefix: &str) -> Result<(), UtilsError> {
        self.sources.push(ConfigSource::Environment);

        for (key, value) in env::vars() {
            if let Some(stripped) = key.strip_prefix(prefix) {
                let config_key = stripped.to_lowercase().replace('_', ".");
                self.data
                    .insert(config_key, ConfigValue::String(value.clone()));
                self.metadata.env_vars_used.push(key);
            }
        }

        Ok(())
    }

    /// Get configuration value by key (supports dot notation)
    pub fn get(&self, key: &str) -> Option<&ConfigValue> {
        if key.contains('.') {
            self.get_nested(key)
        } else {
            self.data.get(key)
        }
    }

    /// Get nested configuration value
    fn get_nested(&self, key: &str) -> Option<&ConfigValue> {
        let parts: Vec<&str> = key.split('.').collect();
        let mut current = self.data.get(parts[0])?;

        for part in &parts[1..] {
            if let ConfigValue::Object(obj) = current {
                current = obj.get(*part)?;
            } else {
                return None;
            }
        }

        Some(current)
    }

    /// Set configuration value
    pub fn set(&mut self, key: &str, value: ConfigValue) {
        if key.contains('.') {
            self.set_nested(key, value);
        } else {
            self.data.insert(key.to_string(), value);
        }
    }

    /// Set nested configuration value
    fn set_nested(&mut self, key: &str, value: ConfigValue) {
        let parts: Vec<&str> = key.split('.').collect();
        if parts.is_empty() {
            return;
        }

        if parts.len() == 1 {
            self.data.insert(parts[0].to_string(), value);
            return;
        }

        // Use a static recursive approach to avoid borrow checker issues
        Self::set_nested_recursive(&mut self.data, &parts, 0, value);
    }

    /// Recursive helper for setting nested values
    fn set_nested_recursive(
        current: &mut HashMap<String, ConfigValue>,
        parts: &[&str],
        index: usize,
        value: ConfigValue,
    ) {
        if index == parts.len() - 1 {
            // Insert the final value
            current.insert(parts[index].to_string(), value);
            return;
        }

        let part = parts[index].to_string();

        // Ensure the entry exists and is an object
        let entry = current
            .entry(part)
            .or_insert_with(|| ConfigValue::Object(HashMap::new()));

        match entry {
            ConfigValue::Object(ref mut obj) => {
                Self::set_nested_recursive(obj, parts, index + 1, value);
            }
            _ => {
                // Convert non-object to object
                *entry = ConfigValue::Object(HashMap::new());
                if let ConfigValue::Object(ref mut obj) = entry {
                    Self::set_nested_recursive(obj, parts, index + 1, value);
                }
            }
        }
    }

    /// Get string value with default
    pub fn get_string(&self, key: &str, default: &str) -> String {
        self.get(key)
            .and_then(|v| v.as_string())
            .unwrap_or_else(|| default.to_string())
    }

    /// Get integer value with default
    pub fn get_i64(&self, key: &str, default: i64) -> i64 {
        self.get(key).and_then(|v| v.as_i64()).unwrap_or(default)
    }

    /// Get float value with default
    pub fn get_f64(&self, key: &str, default: f64) -> f64 {
        self.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
    }

    /// Get boolean value with default
    pub fn get_bool(&self, key: &str, default: bool) -> bool {
        self.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
    }

    /// Merge another configuration
    pub fn merge(&mut self, other: &Config) {
        for (key, value) in &other.data {
            self.data.insert(key.clone(), value.clone());
        }
        self.sources.extend(other.sources.clone());
    }

    /// Get all keys
    pub fn keys(&self) -> Vec<String> {
        let mut keys = Vec::new();
        self.collect_keys("", &self.data, &mut keys);
        keys
    }

    /// Recursively collect all keys
    #[allow(clippy::only_used_in_recursion)]
    fn collect_keys(
        &self,
        prefix: &str,
        data: &HashMap<String, ConfigValue>,
        keys: &mut Vec<String>,
    ) {
        for (key, value) in data {
            let full_key = if prefix.is_empty() {
                key.clone()
            } else {
                format!("{prefix}.{key}")
            };

            keys.push(full_key.clone());

            if let ConfigValue::Object(obj) = value {
                self.collect_keys(&full_key, obj, keys);
            }
        }
    }

    /// Serialize configuration to JSON
    pub fn to_json(&self) -> Result<String, UtilsError> {
        let json_value = self.config_map_to_json(&self.data);
        serde_json::to_string_pretty(&json_value)
            .map_err(|e| UtilsError::InvalidParameter(format!("Failed to serialize config: {e}")))
    }

    /// Convert config map to JSON value
    fn config_map_to_json(&self, data: &HashMap<String, ConfigValue>) -> serde_json::Value {
        let mut map = serde_json::Map::new();

        for (key, value) in data {
            map.insert(key.clone(), self.config_value_to_json(value));
        }

        serde_json::Value::Object(map)
    }

    /// Convert config value to JSON value
    #[allow(clippy::only_used_in_recursion)]
    fn config_value_to_json(&self, value: &ConfigValue) -> serde_json::Value {
        match value {
            ConfigValue::String(s) => serde_json::Value::String(s.clone()),
            ConfigValue::Integer(i) => serde_json::Value::Number((*i).into()),
            ConfigValue::Float(f) => serde_json::Value::Number(
                serde_json::Number::from_f64(*f).unwrap_or_else(|| 0.into()),
            ),
            ConfigValue::Boolean(b) => serde_json::Value::Bool(*b),
            ConfigValue::Array(arr) => {
                serde_json::Value::Array(arr.iter().map(|v| self.config_value_to_json(v)).collect())
            }
            ConfigValue::Object(obj) => {
                let mut map = serde_json::Map::new();
                for (k, v) in obj {
                    map.insert(k.clone(), self.config_value_to_json(v));
                }
                serde_json::Value::Object(map)
            }
            ConfigValue::Null => serde_json::Value::Null,
        }
    }
}

/// Configuration validation utilities
pub struct ConfigValidator;

impl ConfigValidator {
    /// Validate required keys exist
    pub fn validate_required_keys(
        config: &Config,
        required_keys: &[&str],
    ) -> Result<(), UtilsError> {
        for key in required_keys {
            if config.get(key).is_none() {
                return Err(UtilsError::InvalidParameter(format!(
                    "Required configuration key '{key}' is missing"
                )));
            }
        }
        Ok(())
    }

    /// Validate value types
    pub fn validate_types(
        config: &Config,
        type_specs: &HashMap<&str, &str>,
    ) -> Result<(), UtilsError> {
        for (key, expected_type) in type_specs {
            if let Some(value) = config.get(key) {
                let valid = match *expected_type {
                    "string" => matches!(value, ConfigValue::String(_)),
                    "integer" => matches!(value, ConfigValue::Integer(_)),
                    "float" => {
                        matches!(value, ConfigValue::Float(_))
                            || matches!(value, ConfigValue::Integer(_))
                    }
                    "boolean" => matches!(value, ConfigValue::Boolean(_)),
                    "array" => matches!(value, ConfigValue::Array(_)),
                    "object" => matches!(value, ConfigValue::Object(_)),
                    _ => false,
                };

                if !valid {
                    return Err(UtilsError::InvalidParameter(format!(
                        "Configuration key '{key}' has wrong type, expected {expected_type}"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Validate value ranges for numeric types
    pub fn validate_ranges(
        config: &Config,
        range_specs: &HashMap<&str, (f64, f64)>,
    ) -> Result<(), UtilsError> {
        for (key, (min_val, max_val)) in range_specs {
            if let Some(value) = config.get(key) {
                let numeric_value = match value {
                    ConfigValue::Integer(i) => Some(*i as f64),
                    ConfigValue::Float(f) => Some(*f),
                    _ => None,
                };

                if let Some(val) = numeric_value {
                    if val < *min_val || val > *max_val {
                        return Err(UtilsError::InvalidParameter(format!(
                            "Configuration key '{key}' value {val} is outside valid range [{min_val}, {max_val}]"
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Validate configuration using a custom validator function
    pub fn validate_custom<F>(config: &Config, validator: F) -> Result<(), UtilsError>
    where
        F: Fn(&Config) -> Result<(), String>,
    {
        validator(config).map_err(UtilsError::InvalidParameter)
    }
}

/// Hot-reloading configuration manager
pub struct HotReloadConfig {
    config: Arc<RwLock<Config>>,
    file_path: PathBuf,
    last_modified: SystemTime,
    check_interval: Duration,
}

impl HotReloadConfig {
    /// Create a new hot-reload configuration manager
    pub fn new<P: AsRef<Path>>(file_path: P, check_interval: Duration) -> Result<Self, UtilsError> {
        let file_path = file_path.as_ref().to_path_buf();
        let config = Config::from_file(&file_path)?;

        let last_modified = fs::metadata(&file_path)
            .and_then(|m| m.modified())
            .map_err(|e| {
                UtilsError::InvalidParameter(format!("Failed to get file metadata: {e}"))
            })?;

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            file_path,
            last_modified,
            check_interval,
        })
    }

    /// Get the current configuration
    pub fn get_config(&self) -> Arc<RwLock<Config>> {
        self.config.clone()
    }

    /// Check for file updates and reload if necessary
    pub fn check_and_reload(&mut self) -> Result<bool, UtilsError> {
        let current_modified = fs::metadata(&self.file_path)
            .and_then(|m| m.modified())
            .map_err(|e| {
                UtilsError::InvalidParameter(format!("Failed to get file metadata: {e}"))
            })?;

        if current_modified > self.last_modified {
            let new_config = Config::from_file(&self.file_path)?;

            {
                let mut config = self.config.write().unwrap();
                *config = new_config;
            }

            self.last_modified = current_modified;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Start automatic reloading in a background thread
    pub fn start_auto_reload(mut self) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || loop {
            std::thread::sleep(self.check_interval);

            if let Err(e) = self.check_and_reload() {
                eprintln!("Error reloading config: {e}");
            }
        })
    }
}

/// Command-line argument parser
pub struct ArgParser {
    args: Vec<String>,
    config: Config,
}

impl Default for ArgParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ArgParser {
    /// Create a new argument parser
    pub fn new() -> Self {
        let args: Vec<String> = env::args().collect();
        Self {
            args,
            config: Config::new(),
        }
    }

    /// Parse command-line arguments into configuration
    pub fn parse(&mut self) -> Result<(), UtilsError> {
        let mut i = 1; // Skip program name

        while i < self.args.len() {
            let arg = &self.args[i];

            if let Some(stripped) = arg.strip_prefix("--") {
                // Long option
                let key = stripped;

                if let Some(eq_pos) = key.find('=') {
                    // --key=value format
                    let (k, v) = key.split_at(eq_pos);
                    let value = &v[1..]; // Skip the '='
                    self.config.set(k, self.parse_value(value));
                } else if i + 1 < self.args.len() && !self.args[i + 1].starts_with('-') {
                    // --key value format
                    i += 1;
                    let value = &self.args[i];
                    self.config.set(key, self.parse_value(value));
                } else {
                    // Boolean flag
                    self.config.set(key, ConfigValue::Boolean(true));
                }
            } else if arg.starts_with('-') && arg.len() == 2 {
                // Short option
                let key = &arg[1..];

                if i + 1 < self.args.len() && !self.args[i + 1].starts_with('-') {
                    i += 1;
                    let value = &self.args[i];
                    self.config.set(key, self.parse_value(value));
                } else {
                    // Boolean flag
                    self.config.set(key, ConfigValue::Boolean(true));
                }
            }

            i += 1;
        }

        Ok(())
    }

    /// Parse string value to appropriate type
    #[allow(clippy::only_used_in_recursion)]
    fn parse_value(&self, value: &str) -> ConfigValue {
        // Try to parse as different types
        if let Ok(b) = value.parse::<bool>() {
            ConfigValue::Boolean(b)
        } else if let Ok(i) = value.parse::<i64>() {
            ConfigValue::Integer(i)
        } else if let Ok(f) = value.parse::<f64>() {
            ConfigValue::Float(f)
        } else if value.starts_with('[') && value.ends_with(']') {
            // Simple array parsing
            let inner = &value[1..value.len() - 1];
            let items: Vec<ConfigValue> = inner
                .split(',')
                .map(|s| self.parse_value(s.trim()))
                .collect();
            ConfigValue::Array(items)
        } else {
            ConfigValue::String(value.to_string())
        }
    }

    /// Get the parsed configuration
    pub fn get_config(self) -> Config {
        self.config
    }
}

/// Configuration builder for fluent API
pub struct ConfigBuilder {
    config: Config,
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigBuilder {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self {
            config: Config::new(),
        }
    }

    /// Add configuration from file
    pub fn add_file<P: AsRef<Path>>(mut self, path: P) -> Result<Self, UtilsError> {
        let file_config = Config::from_file(path)?;
        self.config.merge(&file_config);
        Ok(self)
    }

    /// Add environment variables with prefix
    pub fn add_env(mut self, prefix: &str) -> Result<Self, UtilsError> {
        self.config.load_environment(prefix)?;
        Ok(self)
    }

    /// Add command-line arguments
    pub fn add_args(mut self) -> Result<Self, UtilsError> {
        let mut parser = ArgParser::new();
        parser.parse()?;
        self.config.merge(&parser.get_config());
        Ok(self)
    }

    /// Set a default value
    pub fn set_default(mut self, key: &str, value: ConfigValue) -> Self {
        if self.config.get(key).is_none() {
            self.config.set(key, value);
        }
        self
    }

    /// Validate the configuration
    pub fn validate<F>(self, validator: F) -> Result<Self, UtilsError>
    where
        F: Fn(&Config) -> Result<(), String>,
    {
        ConfigValidator::validate_custom(&self.config, validator)?;
        Ok(self)
    }

    /// Build the final configuration
    pub fn build(self) -> Config {
        self.config
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_config_value_conversions() {
        let string_val = ConfigValue::String("test".to_string());
        assert_eq!(string_val.as_string(), Some("test".to_string()));

        let int_val = ConfigValue::Integer(42);
        assert_eq!(int_val.as_i64(), Some(42));
        assert_eq!(int_val.as_f64(), Some(42.0));

        let bool_val = ConfigValue::Boolean(true);
        assert_eq!(bool_val.as_bool(), Some(true));

        let float_val = ConfigValue::Float(std::f64::consts::PI);
        assert_eq!(float_val.as_f64(), Some(std::f64::consts::PI));
    }

    #[test]
    fn test_config_get_set() {
        let mut config = Config::new();

        config.set("test.key", ConfigValue::String("value".to_string()));
        assert_eq!(config.get_string("test.key", "default"), "value");

        config.set("number", ConfigValue::Integer(42));
        assert_eq!(config.get_i64("number", 0), 42);

        config.set("flag", ConfigValue::Boolean(true));
        assert!(config.get_bool("flag", false));
    }

    #[test]
    fn test_json_config_loading() {
        let json_content = r#"
        {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "test_db"
            },
            "debug": true,
            "timeout": 30.5
        }
        "#;

        let mut temp_file = NamedTempFile::with_suffix(".json").unwrap();
        temp_file.write_all(json_content.as_bytes()).unwrap();

        let config = Config::from_file(temp_file.path()).unwrap();

        assert_eq!(config.get_string("database.host", ""), "localhost");
        assert_eq!(config.get_i64("database.port", 0), 5432);
        assert_eq!(config.get_string("database.name", ""), "test_db");
        assert!(config.get_bool("debug", false));
        assert_eq!(config.get_f64("timeout", 0.0), 30.5);
    }

    #[test]
    fn test_config_validation() {
        let mut config = Config::new();
        config.set("required_key", ConfigValue::String("value".to_string()));
        config.set("number_key", ConfigValue::Integer(50));

        // Test required keys validation
        let required_keys = vec!["required_key"];
        assert!(ConfigValidator::validate_required_keys(&config, &required_keys).is_ok());

        let missing_keys = vec!["missing_key"];
        assert!(ConfigValidator::validate_required_keys(&config, &missing_keys).is_err());

        // Test type validation
        let mut type_specs = HashMap::new();
        type_specs.insert("required_key", "string");
        type_specs.insert("number_key", "integer");
        assert!(ConfigValidator::validate_types(&config, &type_specs).is_ok());

        type_specs.insert("number_key", "string");
        assert!(ConfigValidator::validate_types(&config, &type_specs).is_err());

        // Test range validation
        let mut range_specs = HashMap::new();
        range_specs.insert("number_key", (0.0, 100.0));
        assert!(ConfigValidator::validate_ranges(&config, &range_specs).is_ok());

        range_specs.insert("number_key", (60.0, 100.0));
        assert!(ConfigValidator::validate_ranges(&config, &range_specs).is_err());
    }

    #[test]
    fn test_arg_parser() {
        // Mock command line arguments
        let args = vec![
            "program".to_string(),
            "--host=localhost".to_string(),
            "--port".to_string(),
            "8080".to_string(),
            "--debug".to_string(),
            "-v".to_string(),
        ];

        let mut parser = ArgParser {
            args,
            config: Config::new(),
        };
        parser.parse().unwrap();

        let config = parser.get_config();
        assert_eq!(config.get_string("host", ""), "localhost");
        assert_eq!(config.get_i64("port", 0), 8080);
        assert!(config.get_bool("debug", false));
        assert!(config.get_bool("v", false));
    }

    #[test]
    fn test_config_builder() {
        let config = ConfigBuilder::new()
            .set_default("host", ConfigValue::String("localhost".to_string()))
            .set_default("port", ConfigValue::Integer(8080))
            .set_default("debug", ConfigValue::Boolean(false))
            .build();

        assert_eq!(config.get_string("host", ""), "localhost");
        assert_eq!(config.get_i64("port", 0), 8080);
        assert!(!config.get_bool("debug", true));
    }

    #[test]
    fn test_config_merge() {
        let mut config1 = Config::new();
        config1.set("key1", ConfigValue::String("value1".to_string()));
        config1.set("key2", ConfigValue::Integer(42));

        let mut config2 = Config::new();
        config2.set("key2", ConfigValue::Integer(100)); // Override
        config2.set("key3", ConfigValue::Boolean(true)); // New key

        config1.merge(&config2);

        assert_eq!(config1.get_string("key1", ""), "value1");
        assert_eq!(config1.get_i64("key2", 0), 100); // Overridden
        assert!(config1.get_bool("key3", false)); // New key
    }
}
