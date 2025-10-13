//! Interactive Visualization Framework
//!
//! This module provides the core interactive visualization framework that enables
//! real-time updates, callback handling, and dynamic plot management.
//!
//! ## Key Features
//!
//! - **Real-time Updates**: Live data streaming and plot updates
//! - **Callback System**: Flexible event handling and custom interactions
//! - **Multi-plot Management**: Centralized configuration and control
//! - **Thread-safe Operations**: Safe concurrent access and updates

use super::config_types::PlotConfig;
use crate::Float;
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;

/// Interactive visualization framework
pub struct InteractiveVisualizer {
    /// Current plot configurations
    pub plot_configs: HashMap<String, PlotConfig>,
    /// Update callbacks
    pub update_callbacks: Vec<Box<dyn Fn(&str, &Array2<Float>) + Send + Sync>>,
    /// Real-time update enabled
    pub real_time_updates: bool,
}

impl std::fmt::Debug for InteractiveVisualizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InteractiveVisualizer")
            .field("plot_configs", &self.plot_configs)
            .field("callback_count", &self.update_callbacks.len())
            .field("real_time_updates", &self.real_time_updates)
            .finish()
    }
}

impl InteractiveVisualizer {
    /// Create a new interactive visualizer
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_inspection::visualization::InteractiveVisualizer;
    ///
    /// let visualizer = InteractiveVisualizer::new();
    /// assert!(!visualizer.real_time_updates);
    /// ```
    pub fn new() -> Self {
        Self {
            plot_configs: HashMap::new(),
            update_callbacks: Vec::new(),
            real_time_updates: false,
        }
    }

    /// Enable real-time updates
    pub fn enable_real_time_updates(&mut self) {
        self.real_time_updates = true;
    }

    /// Disable real-time updates
    pub fn disable_real_time_updates(&mut self) {
        self.real_time_updates = false;
    }

    /// Check if real-time updates are enabled
    pub fn is_real_time_enabled(&self) -> bool {
        self.real_time_updates
    }

    /// Add an update callback
    pub fn add_update_callback<F>(&mut self, callback: F)
    where
        F: Fn(&str, &Array2<Float>) + Send + Sync + 'static,
    {
        self.update_callbacks.push(Box::new(callback));
    }

    /// Clear all update callbacks
    pub fn clear_callbacks(&mut self) {
        self.update_callbacks.clear();
    }

    /// Get number of registered callbacks
    pub fn callback_count(&self) -> usize {
        self.update_callbacks.len()
    }

    /// Update plot data
    pub fn update_plot_data(&self, plot_id: &str, data: &Array2<Float>) {
        if self.real_time_updates {
            for callback in &self.update_callbacks {
                callback(plot_id, data);
            }
        }
    }

    /// Add or update a plot configuration
    pub fn set_plot_config(&mut self, plot_id: String, config: PlotConfig) {
        self.plot_configs.insert(plot_id, config);
    }

    /// Get plot configuration by ID
    pub fn get_plot_config(&self, plot_id: &str) -> Option<&PlotConfig> {
        self.plot_configs.get(plot_id)
    }

    /// Remove a plot configuration
    pub fn remove_plot(&mut self, plot_id: &str) -> Option<PlotConfig> {
        self.plot_configs.remove(plot_id)
    }

    /// Get all plot IDs
    pub fn plot_ids(&self) -> Vec<&String> {
        self.plot_configs.keys().collect()
    }

    /// Get number of configured plots
    pub fn plot_count(&self) -> usize {
        self.plot_configs.len()
    }

    /// Clear all plot configurations
    pub fn clear_plots(&mut self) {
        self.plot_configs.clear();
    }
}

impl Default for InteractiveVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::config_types::{ColorScheme, PlotConfig};
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_interactive_visualizer_creation() {
        let visualizer = InteractiveVisualizer::new();
        assert!(!visualizer.real_time_updates);
        assert_eq!(visualizer.callback_count(), 0);
        assert_eq!(visualizer.plot_count(), 0);
    }

    #[test]
    fn test_real_time_updates() {
        let mut visualizer = InteractiveVisualizer::new();
        assert!(!visualizer.is_real_time_enabled());

        visualizer.enable_real_time_updates();
        assert!(visualizer.is_real_time_enabled());

        visualizer.disable_real_time_updates();
        assert!(!visualizer.is_real_time_enabled());
    }

    #[test]
    fn test_callbacks() {
        let mut visualizer = InteractiveVisualizer::new();

        // Add callback
        visualizer.add_update_callback(|_id, _data| {
            // Test callback
        });
        assert_eq!(visualizer.callback_count(), 1);

        // Clear callbacks
        visualizer.clear_callbacks();
        assert_eq!(visualizer.callback_count(), 0);
    }

    #[test]
    fn test_plot_management() {
        let mut visualizer = InteractiveVisualizer::new();
        let config = PlotConfig::default();
        let plot_id = "test_plot".to_string();

        // Add plot
        visualizer.set_plot_config(plot_id.clone(), config.clone());
        assert_eq!(visualizer.plot_count(), 1);
        assert!(visualizer.get_plot_config("test_plot").is_some());

        // Update plot
        let mut updated_config = config.clone();
        updated_config.title = "Updated Title".to_string();
        visualizer.set_plot_config(plot_id.clone(), updated_config);
        assert_eq!(visualizer.plot_count(), 1);
        assert_eq!(
            visualizer.get_plot_config("test_plot").unwrap().title,
            "Updated Title"
        );

        // Remove plot
        let removed = visualizer.remove_plot("test_plot");
        assert!(removed.is_some());
        assert_eq!(visualizer.plot_count(), 0);
    }

    #[test]
    fn test_plot_data_update() {
        let mut visualizer = InteractiveVisualizer::new();
        let data = array![[1.0, 2.0], [3.0, 4.0]];

        // Should not trigger callbacks when real-time is disabled
        visualizer.update_plot_data("test", &data);

        // Enable real-time updates
        visualizer.enable_real_time_updates();

        // Add a callback that sets a flag
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let callback_triggered = Arc::new(AtomicBool::new(false));
        let callback_triggered_clone = callback_triggered.clone();

        visualizer.add_update_callback(move |_id, _data| {
            callback_triggered_clone.store(true, Ordering::Relaxed);
        });

        // Update should trigger callback
        visualizer.update_plot_data("test", &data);
        assert!(callback_triggered.load(Ordering::Relaxed));
    }

    #[test]
    fn test_default_implementation() {
        let visualizer = InteractiveVisualizer::default();
        assert!(!visualizer.real_time_updates);
        assert_eq!(visualizer.plot_count(), 0);
    }
}
