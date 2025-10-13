//! Output generation and export functionality
//!
//! This module provides capabilities for generating HTML output,
//! exporting visualizations, and handling different output formats.

use crate::SklResult;
use serde::Serialize;

/// Generate HTML visualization output
///
/// # Arguments
///
/// * `plots` - Array of plot data to serialize
/// * `title` - Title for the HTML page
/// * `output_path` - Optional path to save the HTML file
///
/// # Returns
///
/// Result containing the generated HTML content as a string
pub fn generate_html_output<T: Serialize>(
    plots: &[T],
    title: &str,
    output_path: Option<&str>,
) -> SklResult<String> {
    let html_template = include_str!("../../templates/visualization.html");

    // Serialize plots to JSON
    let plots_json = serde_json::to_string_pretty(plots).map_err(|e| {
        crate::SklearsError::InvalidInput(format!("Failed to serialize plots: {}", e))
    })?;

    let html_content = html_template
        .replace("{{TITLE}}", title)
        .replace("{{PLOTS_DATA}}", &plots_json);

    if let Some(path) = output_path {
        std::fs::write(path, &html_content).map_err(|e| {
            crate::SklearsError::InvalidInput(format!("Failed to write HTML file: {}", e))
        })?;
    }

    Ok(html_content)
}

/// Generate meshgrid for 3D surface plots
///
/// # Arguments
///
/// * `x_range` - Range of x values (start, end, num_points)
/// * `y_range` - Range of y values (start, end, num_points)
///
/// # Returns
///
/// Result containing (X_grid, Y_grid) arrays
pub fn generate_meshgrid(
    x_range: (f64, f64, usize),
    y_range: (f64, f64, usize),
) -> SklResult<(
    scirs2_core::ndarray::Array2<f64>,
    scirs2_core::ndarray::Array2<f64>,
)> {
    let (x_start, x_end, x_points) = x_range;
    let (y_start, y_end, y_points) = y_range;

    if x_points == 0 || y_points == 0 {
        return Err(crate::SklearsError::InvalidInput(
            "Number of points must be greater than 0".to_string(),
        ));
    }

    if x_start >= x_end || y_start >= y_end {
        return Err(crate::SklearsError::InvalidInput(
            "Range end must be greater than start".to_string(),
        ));
    }

    let x_step = if x_points == 1 {
        0.0
    } else {
        (x_end - x_start) / (x_points - 1) as f64
    };
    let y_step = if y_points == 1 {
        0.0
    } else {
        (y_end - y_start) / (y_points - 1) as f64
    };

    let mut x_grid = scirs2_core::ndarray::Array2::zeros((y_points, x_points));
    let mut y_grid = scirs2_core::ndarray::Array2::zeros((y_points, x_points));

    for i in 0..y_points {
        for j in 0..x_points {
            x_grid[(i, j)] = x_start + j as f64 * x_step;
            y_grid[(i, j)] = y_start + i as f64 * y_step;
        }
    }

    Ok((x_grid, y_grid))
}

/// Export visualization to different formats
#[derive(Debug, Clone)]
pub enum ExportFormat {
    /// HTML
    HTML,
    /// JSON
    JSON,
    /// CSV
    CSV,
    /// PNG
    PNG,
    /// SVG
    SVG,
}

/// Export configuration
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Export format
    pub format: ExportFormat,
    /// Output file path
    pub output_path: String,
    /// Additional options
    pub options: std::collections::HashMap<String, String>,
}

impl ExportConfig {
    /// Create new export configuration
    pub fn new(format: ExportFormat, output_path: String) -> Self {
        Self {
            format,
            output_path,
            options: std::collections::HashMap::new(),
        }
    }

    /// Add export option
    pub fn with_option(mut self, key: String, value: String) -> Self {
        self.options.insert(key, value);
        self
    }
}

/// Export visualization data
pub fn export_visualization<T: Serialize>(data: &T, config: &ExportConfig) -> SklResult<()> {
    match config.format {
        ExportFormat::HTML => {
            let html_content = generate_html_output(&[data], "Exported Visualization", None)?;
            std::fs::write(&config.output_path, html_content).map_err(|e| {
                crate::SklearsError::InvalidInput(format!("Failed to write HTML file: {}", e))
            })?;
        }
        ExportFormat::JSON => {
            let json_content = serde_json::to_string_pretty(data).map_err(|e| {
                crate::SklearsError::InvalidInput(format!("Failed to serialize to JSON: {}", e))
            })?;
            std::fs::write(&config.output_path, json_content).map_err(|e| {
                crate::SklearsError::InvalidInput(format!("Failed to write JSON file: {}", e))
            })?;
        }
        _ => {
            return Err(crate::SklearsError::InvalidInput(format!(
                "Export format {:?} not implemented yet",
                config.format
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_generate_meshgrid() {
        let (x_grid, y_grid) = generate_meshgrid((0.0, 2.0, 3), (0.0, 1.0, 2)).unwrap();

        assert_eq!(x_grid.dim(), (2, 3));
        assert_eq!(y_grid.dim(), (2, 3));

        // Check some values
        assert_eq!(x_grid[(0, 0)], 0.0);
        assert_eq!(x_grid[(0, 2)], 2.0);
        assert_eq!(y_grid[(0, 0)], 0.0);
        assert_eq!(y_grid[(1, 0)], 1.0);
    }

    #[test]
    fn test_generate_meshgrid_invalid_range() {
        let result = generate_meshgrid((2.0, 1.0, 3), (0.0, 1.0, 2));
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_meshgrid_zero_points() {
        let result = generate_meshgrid((0.0, 1.0, 0), (0.0, 1.0, 2));
        assert!(result.is_err());
    }

    #[test]
    fn test_export_config_creation() {
        let config = ExportConfig::new(ExportFormat::JSON, "test.json".to_string())
            .with_option("pretty".to_string(), "true".to_string());

        assert!(matches!(config.format, ExportFormat::JSON));
        assert_eq!(config.output_path, "test.json");
        assert!(config.options.contains_key("pretty"));
    }

    #[test]
    fn test_generate_html_output() {
        let test_data = vec![json!({"test": "data"})];
        let result = generate_html_output(&test_data, "Test Title", None);

        assert!(result.is_ok());
        let html = result.unwrap();
        assert!(html.contains("Test Title"));
        assert!(html.contains("test"));
        assert!(html.contains("data"));
    }
}
