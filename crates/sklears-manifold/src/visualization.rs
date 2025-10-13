//! Visualization integration utilities for manifold learning
//!
//! This module provides utilities for exporting and preparing manifold learning
//! embeddings for visualization with common plotting libraries and tools.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use std::fs::File;
use std::io::Write;

/// Visualization backend type
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::Estimator,
};
use std::collections::HashMap;
use std::path::Path;
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VisualizationBackend {
    /// matplotlib/seaborn compatible CSV format
    Matplotlib,
    /// plotly compatible JSON format
    Plotly,
    /// D3.js compatible JSON format
    D3,
    /// Generic CSV format
    CSV,
    /// Generic JSON format
    JSON,
}

/// Visualization export configuration
#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    /// Backend to export for
    pub backend: VisualizationBackend,
    /// Include metadata in the export
    pub include_metadata: bool,
    /// Include original data if available
    pub include_original: bool,
    /// Additional metadata to include
    pub metadata: HashMap<String, String>,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            backend: VisualizationBackend::CSV,
            include_metadata: true,
            include_original: false,
            metadata: HashMap::new(),
        }
    }
}

/// Visualization data structure
#[derive(Debug, Clone)]
pub struct VisualizationData {
    /// Embedded coordinates
    pub embedding: Array2<f64>,
    /// Optional labels for points
    pub labels: Option<Array1<String>>,
    /// Optional colors for points
    pub colors: Option<Array1<f64>>,
    /// Optional sizes for points  
    pub sizes: Option<Array1<f64>>,
    /// Original high-dimensional data
    pub original: Option<Array2<f64>>,
    /// Metadata about the embedding
    pub metadata: HashMap<String, String>,
}

impl VisualizationData {
    /// Create new visualization data from embedding
    pub fn new(embedding: Array2<f64>) -> Self {
        Self {
            embedding,
            labels: None,
            colors: None,
            sizes: None,
            original: None,
            metadata: HashMap::new(),
        }
    }

    /// Add labels to the visualization data
    pub fn with_labels(mut self, labels: Array1<String>) -> SklResult<Self> {
        if labels.len() != self.embedding.shape()[0] {
            return Err(SklearsError::InvalidInput(
                "Number of labels must match number of points".to_string(),
            ));
        }
        self.labels = Some(labels);
        Ok(self)
    }

    /// Add colors to the visualization data
    pub fn with_colors(mut self, colors: Array1<f64>) -> SklResult<Self> {
        if colors.len() != self.embedding.shape()[0] {
            return Err(SklearsError::InvalidInput(
                "Number of colors must match number of points".to_string(),
            ));
        }
        self.colors = Some(colors);
        Ok(self)
    }

    /// Add sizes to the visualization data
    pub fn with_sizes(mut self, sizes: Array1<f64>) -> SklResult<Self> {
        if sizes.len() != self.embedding.shape()[0] {
            return Err(SklearsError::InvalidInput(
                "Number of sizes must match number of points".to_string(),
            ));
        }
        self.sizes = Some(sizes);
        Ok(self)
    }

    /// Add original high-dimensional data
    pub fn with_original(mut self, original: Array2<f64>) -> SklResult<Self> {
        if original.shape()[0] != self.embedding.shape()[0] {
            return Err(SklearsError::InvalidInput(
                "Number of original points must match number of embedded points".to_string(),
            ));
        }
        self.original = Some(original);
        Ok(self)
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Export to file
    pub fn export(&self, path: impl AsRef<Path>, config: &VisualizationConfig) -> SklResult<()> {
        match config.backend {
            VisualizationBackend::CSV | VisualizationBackend::Matplotlib => {
                self.export_csv(path, config)
            }
            VisualizationBackend::JSON
            | VisualizationBackend::Plotly
            | VisualizationBackend::D3 => self.export_json(path, config),
        }
    }

    /// Export to CSV format
    fn export_csv(&self, path: impl AsRef<Path>, config: &VisualizationConfig) -> SklResult<()> {
        let mut file = File::create(path)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to create file: {}", e)))?;

        let n_points = self.embedding.shape()[0];
        let n_dims = self.embedding.shape()[1];

        // Write header
        let mut header = Vec::new();
        for i in 0..n_dims {
            header.push(format!("dim_{}", i));
        }

        if self.labels.is_some() {
            header.push("label".to_string());
        }
        if self.colors.is_some() {
            header.push("color".to_string());
        }
        if self.sizes.is_some() {
            header.push("size".to_string());
        }

        writeln!(file, "{}", header.join(","))
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to write header: {}", e)))?;

        // Write data
        for i in 0..n_points {
            let mut row = Vec::new();

            // Add embedding coordinates
            for j in 0..n_dims {
                row.push(self.embedding[[i, j]].to_string());
            }

            // Add optional data
            if let Some(labels) = &self.labels {
                row.push(labels[i].clone());
            }
            if let Some(colors) = &self.colors {
                row.push(colors[i].to_string());
            }
            if let Some(sizes) = &self.sizes {
                row.push(sizes[i].to_string());
            }

            writeln!(file, "{}", row.join(",")).map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to write row {}: {}", i, e))
            })?;
        }

        // Write metadata as comments if requested
        if config.include_metadata {
            writeln!(file, "# Metadata:").map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to write metadata header: {}", e))
            })?;

            for (key, value) in &self.metadata {
                writeln!(file, "# {}: {}", key, value).map_err(|e| {
                    SklearsError::InvalidInput(format!("Failed to write metadata: {}", e))
                })?;
            }
        }

        Ok(())
    }

    /// Export to JSON format
    fn export_json(&self, path: impl AsRef<Path>, config: &VisualizationConfig) -> SklResult<()> {
        let mut file = File::create(path)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to create file: {}", e)))?;

        let n_points = self.embedding.shape()[0];
        let n_dims = self.embedding.shape()[1];

        // Create JSON structure
        let mut json = String::new();
        json.push_str("{\n");
        json.push_str("  \"data\": [\n");

        for i in 0..n_points {
            json.push_str("    {\n");

            // Add embedding coordinates
            json.push_str("      \"coordinates\": [");
            for j in 0..n_dims {
                if j > 0 {
                    json.push_str(", ");
                }
                json.push_str(&self.embedding[[i, j]].to_string());
            }
            json.push_str("],\n");

            // Add optional data
            if let Some(labels) = &self.labels {
                json.push_str(&format!("      \"label\": \"{}\",\n", labels[i]));
            }
            if let Some(colors) = &self.colors {
                json.push_str(&format!("      \"color\": {},\n", colors[i]));
            }
            if let Some(sizes) = &self.sizes {
                json.push_str(&format!("      \"size\": {},\n", sizes[i]));
            }

            // Remove trailing comma
            if json.ends_with(",\n") {
                json.truncate(json.len() - 2);
                json.push('\n');
            }

            json.push_str("    }");
            if i < n_points - 1 {
                json.push(',');
            }
            json.push('\n');
        }

        json.push_str("  ]");

        // Add metadata if requested
        if config.include_metadata && !self.metadata.is_empty() {
            json.push_str(",\n  \"metadata\": {\n");
            let mut first = true;
            for (key, value) in &self.metadata {
                if !first {
                    json.push_str(",\n");
                }
                json.push_str(&format!("    \"{}\": \"{}\"", key, value));
                first = false;
            }
            json.push_str("\n  }");
        }

        json.push_str("\n}");

        file.write_all(json.as_bytes())
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to write JSON: {}", e)))?;

        Ok(())
    }
}

/// Quick visualization export for common use cases
pub struct QuickVisualization;

impl QuickVisualization {
    /// Export 2D embedding to CSV for matplotlib
    pub fn to_matplotlib_csv(
        embedding: ArrayView2<f64>,
        path: impl AsRef<Path>,
        labels: Option<Array1<String>>,
    ) -> SklResult<()> {
        let viz_data = VisualizationData::new(embedding.to_owned());
        let viz_data = if let Some(labels) = labels {
            viz_data.with_labels(labels)?
        } else {
            viz_data
        };

        let config = VisualizationConfig {
            backend: VisualizationBackend::Matplotlib,
            include_metadata: true,
            include_original: false,
            metadata: HashMap::new(),
        };

        viz_data.export(path, &config)
    }

    /// Export embedding to JSON for plotly
    pub fn to_plotly_json(
        embedding: ArrayView2<f64>,
        path: impl AsRef<Path>,
        labels: Option<Array1<String>>,
        colors: Option<Array1<f64>>,
    ) -> SklResult<()> {
        let mut viz_data = VisualizationData::new(embedding.to_owned());

        if let Some(labels) = labels {
            viz_data = viz_data.with_labels(labels)?;
        }
        if let Some(colors) = colors {
            viz_data = viz_data.with_colors(colors)?;
        }

        let config = VisualizationConfig {
            backend: VisualizationBackend::Plotly,
            include_metadata: true,
            include_original: false,
            metadata: HashMap::new(),
        };

        viz_data.export(path, &config)
    }

    /// Export embedding to JSON for D3.js
    pub fn to_d3_json(
        embedding: ArrayView2<f64>,
        path: impl AsRef<Path>,
        labels: Option<Array1<String>>,
    ) -> SklResult<()> {
        let viz_data = VisualizationData::new(embedding.to_owned());
        let viz_data = if let Some(labels) = labels {
            viz_data.with_labels(labels)?
        } else {
            viz_data
        };

        let config = VisualizationConfig {
            backend: VisualizationBackend::D3,
            include_metadata: false,
            include_original: false,
            metadata: HashMap::new(),
        };

        viz_data.export(path, &config)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use tempfile::NamedTempFile;

    #[test]
    fn test_visualization_data_creation() {
        let embedding = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let viz_data = VisualizationData::new(embedding.clone());

        assert_eq!(viz_data.embedding, embedding);
        assert!(viz_data.labels.is_none());
        assert!(viz_data.colors.is_none());
        assert!(viz_data.sizes.is_none());
    }

    #[test]
    fn test_visualization_data_with_labels() {
        let embedding = array![[1.0, 2.0], [3.0, 4.0]];
        let labels = array!["A".to_string(), "B".to_string()];

        let viz_data = VisualizationData::new(embedding)
            .with_labels(labels.clone())
            .unwrap();

        assert_eq!(viz_data.labels.unwrap(), labels);
    }

    #[test]
    fn test_csv_export() {
        let embedding = array![[1.0, 2.0], [3.0, 4.0]];
        let labels = array!["A".to_string(), "B".to_string()];

        let viz_data = VisualizationData::new(embedding)
            .with_labels(labels)
            .unwrap();

        let config = VisualizationConfig::default();
        let temp_file = NamedTempFile::new().unwrap();

        viz_data.export(temp_file.path(), &config).unwrap();

        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("dim_0,dim_1,label"));
        assert!(content.contains("1,2,A"));
        assert!(content.contains("3,4,B"));
    }

    #[test]
    fn test_json_export() {
        let embedding = array![[1.0, 2.0], [3.0, 4.0]];

        let viz_data = VisualizationData::new(embedding);

        let config = VisualizationConfig {
            backend: VisualizationBackend::JSON,
            include_metadata: false,
            include_original: false,
            metadata: HashMap::new(),
        };

        let temp_file = NamedTempFile::new().unwrap();
        viz_data.export(temp_file.path(), &config).unwrap();

        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("\"coordinates\""));
        assert!(content.contains("[1, 2]"));
        assert!(content.contains("[3, 4]"));
    }

    #[test]
    fn test_quick_matplotlib_export() {
        let embedding = array![[1.0, 2.0], [3.0, 4.0]];
        let labels = array!["A".to_string(), "B".to_string()];

        let temp_file = NamedTempFile::new().unwrap();

        QuickVisualization::to_matplotlib_csv(embedding.view(), temp_file.path(), Some(labels))
            .unwrap();

        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("dim_0,dim_1,label"));
    }
}
