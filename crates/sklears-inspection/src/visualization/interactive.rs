//! Interactive visualization features
//!
//! This module provides real-time plot updates, interactive visualizations,
//! and user interaction handling capabilities.

use crate::{Float, SklResult};
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::core::PlotConfig;

/// Interactive visualizer with real-time capabilities
pub struct InteractiveVisualizer {
    /// Plot configuration
    config: PlotConfig,
    /// Update callback function
    update_callback: Option<Box<dyn Fn(&ArrayView2<Float>) -> SklResult<()> + Send + Sync>>,
}

impl std::fmt::Debug for InteractiveVisualizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InteractiveVisualizer")
            .field("config", &self.config)
            .field("update_callback", &self.update_callback.is_some())
            .finish()
    }
}

impl InteractiveVisualizer {
    /// Create new interactive visualizer
    pub fn new(config: PlotConfig) -> Self {
        Self {
            config,
            update_callback: None,
        }
    }

    /// Set update callback function
    pub fn with_update_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(&ArrayView2<Float>) -> SklResult<()> + Send + Sync + 'static,
    {
        self.update_callback = Some(Box::new(callback));
        self
    }

    /// Update visualization with new data
    pub fn update_data(&self, data: &ArrayView2<Float>) -> SklResult<()> {
        if let Some(ref callback) = self.update_callback {
            callback(data)?;
        }
        Ok(())
    }

    /// Enable real-time updates
    pub fn enable_real_time(&mut self, enabled: bool) {
        self.config.interactive = enabled;
    }

    /// Get current configuration
    pub fn config(&self) -> &PlotConfig {
        &self.config
    }
}

/// Real-time plot updater
pub struct RealTimePlotUpdater {
    /// Current data buffer
    data_buffer: HashMap<String, Array2<Float>>,
    /// Update frequency in Hz
    update_frequency: Float,
    /// Maximum buffer size
    max_buffer_size: usize,
    /// Is actively updating
    is_active: bool,
    /// Plot configuration
    config: PlotConfig,
}

impl RealTimePlotUpdater {
    /// Create new real-time updater
    pub fn new(config: PlotConfig) -> Self {
        Self {
            data_buffer: HashMap::new(),
            update_frequency: 30.0, // 30 Hz default
            max_buffer_size: 1000,
            is_active: false,
            config,
        }
    }

    /// Set update frequency in Hz
    pub fn with_frequency(mut self, frequency: Float) -> Self {
        self.update_frequency = frequency;
        self
    }

    /// Set maximum buffer size
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.max_buffer_size = size;
        self
    }

    /// Add data to buffer
    pub fn add_data(&mut self, series_name: String, data: Array2<Float>) -> SklResult<()> {
        if data.nrows() > self.max_buffer_size {
            return Err(crate::SklearsError::InvalidInput(
                "Data size exceeds maximum buffer size".to_string(),
            ));
        }

        self.data_buffer.insert(series_name, data);
        Ok(())
    }

    /// Start real-time updates
    pub fn start(&mut self) {
        self.is_active = true;
    }

    /// Stop real-time updates
    pub fn stop(&mut self) {
        self.is_active = false;
    }

    /// Check if actively updating
    pub fn is_active(&self) -> bool {
        self.is_active
    }

    /// Get current buffer data
    pub fn get_buffer_data(&self) -> &HashMap<String, Array2<Float>> {
        &self.data_buffer
    }

    /// Clear buffer
    pub fn clear_buffer(&mut self) {
        self.data_buffer.clear();
    }

    /// Get update frequency
    pub fn update_frequency(&self) -> Float {
        self.update_frequency
    }
}

/// Plot interaction handler for user interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotInteractionHandler {
    /// Zoom enabled
    pub zoom_enabled: bool,
    /// Pan enabled
    pub pan_enabled: bool,
    /// Selection enabled
    pub selection_enabled: bool,
    /// Hover tooltips enabled
    pub hover_enabled: bool,
    /// Click callbacks enabled
    pub click_enabled: bool,
    /// Brush selection enabled
    pub brush_enabled: bool,
}

impl Default for PlotInteractionHandler {
    fn default() -> Self {
        Self {
            zoom_enabled: true,
            pan_enabled: true,
            selection_enabled: true,
            hover_enabled: true,
            click_enabled: true,
            brush_enabled: false,
        }
    }
}

impl PlotInteractionHandler {
    /// Create new interaction handler
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable/disable zoom functionality
    pub fn with_zoom(mut self, enabled: bool) -> Self {
        self.zoom_enabled = enabled;
        self
    }

    /// Enable/disable pan functionality
    pub fn with_pan(mut self, enabled: bool) -> Self {
        self.pan_enabled = enabled;
        self
    }

    /// Enable/disable selection
    pub fn with_selection(mut self, enabled: bool) -> Self {
        self.selection_enabled = enabled;
        self
    }

    /// Enable/disable hover tooltips
    pub fn with_hover(mut self, enabled: bool) -> Self {
        self.hover_enabled = enabled;
        self
    }

    /// Enable/disable click callbacks
    pub fn with_click(mut self, enabled: bool) -> Self {
        self.click_enabled = enabled;
        self
    }

    /// Enable/disable brush selection
    pub fn with_brush(mut self, enabled: bool) -> Self {
        self.brush_enabled = enabled;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;

    #[test]
    fn test_interactive_visualizer_creation() {
        let config = PlotConfig::default();
        let visualizer = InteractiveVisualizer::new(config);

        assert!(!visualizer.update_callback.is_some());
        assert!(visualizer.config().interactive);
    }

    #[test]
    fn test_real_time_plot_updater() {
        let config = PlotConfig::default();
        let mut updater = RealTimePlotUpdater::new(config);

        assert!(!updater.is_active());
        assert_eq!(updater.update_frequency(), 30.0);

        updater.start();
        assert!(updater.is_active());

        updater.stop();
        assert!(!updater.is_active());
    }

    #[test]
    fn test_plot_interaction_handler() {
        let handler = PlotInteractionHandler::new()
            .with_zoom(false)
            .with_pan(true)
            .with_selection(false);

        assert!(!handler.zoom_enabled);
        assert!(handler.pan_enabled);
        assert!(!handler.selection_enabled);
        assert!(handler.hover_enabled); // Default remains true
    }

    #[test]
    fn test_real_time_updater_data_buffer() {
        let config = PlotConfig::default();
        let mut updater = RealTimePlotUpdater::new(config);

        let data = array![[1.0, 2.0], [3.0, 4.0]];
        updater.add_data("test_series".to_string(), data).unwrap();

        assert_eq!(updater.get_buffer_data().len(), 1);
        assert!(updater.get_buffer_data().contains_key("test_series"));

        updater.clear_buffer();
        assert_eq!(updater.get_buffer_data().len(), 0);
    }
}
