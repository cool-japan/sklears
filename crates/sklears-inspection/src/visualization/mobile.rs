//! Mobile optimization and responsive design
//!
//! This module provides mobile-specific optimizations for visualizations,
//! responsive design utilities, and device-specific configurations.

use crate::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::core::{DeviceType, MobileConfig, PlotConfig};

/// Layout configuration for different device types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsiveLayout {
    /// Desktop layout settings
    pub desktop: LayoutSettings,
    /// Tablet layout settings
    pub tablet: LayoutSettings,
    /// Mobile layout settings
    pub mobile: LayoutSettings,
}

/// Layout settings for a specific device type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutSettings {
    /// Plot width (can be percentage or pixels)
    pub width: String,
    /// Plot height in pixels
    pub height: usize,
    /// Margin settings
    pub margins: Margins,
    /// Font size scaling factor
    pub font_scale: Float,
    /// Whether to stack plots vertically
    pub stack_vertically: bool,
    /// Simplification level (0.0 = no simplification, 1.0 = maximum)
    pub simplification: Float,
}

/// Margin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Margins {
    /// Top margin
    pub top: usize,
    /// Right margin
    pub right: usize,
    /// Bottom margin
    pub bottom: usize,
    /// Left margin
    pub left: usize,
}

impl Default for ResponsiveLayout {
    fn default() -> Self {
        Self {
            desktop: LayoutSettings {
                width: "800px".to_string(),
                height: 600,
                margins: Margins {
                    top: 20,
                    right: 20,
                    bottom: 40,
                    left: 60,
                },
                font_scale: 1.0,
                stack_vertically: false,
                simplification: 0.0,
            },
            tablet: LayoutSettings {
                width: "100%".to_string(),
                height: 400,
                margins: Margins {
                    top: 15,
                    right: 15,
                    bottom: 35,
                    left: 50,
                },
                font_scale: 0.9,
                stack_vertically: true,
                simplification: 0.2,
            },
            mobile: LayoutSettings {
                width: "100%".to_string(),
                height: 300,
                margins: Margins {
                    top: 10,
                    right: 10,
                    bottom: 30,
                    left: 40,
                },
                font_scale: 0.8,
                stack_vertically: true,
                simplification: 0.5,
            },
        }
    }
}

/// Mobile plot optimizer
pub struct MobilePlotOptimizer {
    /// Configuration for mobile optimization
    config: MobileConfig,
    /// Cache for optimized plots
    cache: HashMap<String, OptimizedPlotData>,
    /// Maximum cache size
    max_cache_size: usize,
}

/// Optimized plot data for mobile devices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedPlotData {
    /// Simplified data points
    pub data_points: usize,
    /// Reduced complexity level
    pub complexity_level: Float,
    /// Touch-optimized controls
    pub touch_controls: bool,
    /// Cached at timestamp
    pub cached_at: std::time::SystemTime,
}

impl MobilePlotOptimizer {
    /// Create new mobile plot optimizer
    pub fn new(config: MobileConfig) -> Self {
        Self {
            config,
            cache: HashMap::new(),
            max_cache_size: 100,
        }
    }

    /// Optimize plot for mobile device
    pub fn optimize_for_mobile(
        &mut self,
        plot_id: String,
        original_data_points: usize,
    ) -> OptimizedPlotData {
        // Check if we have a cached version
        if let Some(cached) = self.cache.get(&plot_id) {
            return cached.clone();
        }

        // Create optimized version
        let optimized = OptimizedPlotData {
            data_points: self.reduce_data_points(original_data_points),
            complexity_level: self.calculate_complexity_reduction(),
            touch_controls: self.config.touch_friendly,
            cached_at: std::time::SystemTime::now(),
        };

        // Add to cache
        if self.cache.len() >= self.max_cache_size {
            self.clear_old_cache_entries();
        }
        self.cache.insert(plot_id, optimized.clone());

        optimized
    }

    /// Reduce number of data points for mobile
    fn reduce_data_points(&self, original: usize) -> usize {
        if !self.config.simplify_on_mobile {
            return original;
        }

        // Reduce to maximum 500 points for mobile
        std::cmp::min(original, 500)
    }

    /// Calculate complexity reduction factor
    fn calculate_complexity_reduction(&self) -> Float {
        if self.config.simplify_on_mobile {
            0.6 // Reduce complexity by 40%
        } else {
            1.0
        }
    }

    /// Clear old cache entries
    fn clear_old_cache_entries(&mut self) {
        let cutoff = std::time::SystemTime::now() - std::time::Duration::from_secs(3600); // 1 hour
        self.cache.retain(|_, data| data.cached_at > cutoff);
    }

    /// Clear entire cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get cache size
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }
}

/// Generate responsive CSS for visualizations
pub fn generate_responsive_css(layout: &ResponsiveLayout) -> String {
    format!(
        r#"
        .visualization-container {{
            width: {};
            height: {}px;
            margin: {}px {}px {}px {}px;
            font-size: {}em;
        }}

        @media (max-width: 1024px) {{
            .visualization-container {{
                width: {};
                height: {}px;
                margin: {}px {}px {}px {}px;
                font-size: {}em;
            }}
            .plot-stack {{
                flex-direction: {};
            }}
        }}

        @media (max-width: 768px) {{
            .visualization-container {{
                width: {};
                height: {}px;
                margin: {}px {}px {}px {}px;
                font-size: {}em;
            }}
            .plot-stack {{
                flex-direction: column;
            }}
            .touch-control {{
                min-height: 44px;
                min-width: 44px;
            }}
        }}
        "#,
        layout.desktop.width,
        layout.desktop.height,
        layout.desktop.margins.top,
        layout.desktop.margins.right,
        layout.desktop.margins.bottom,
        layout.desktop.margins.left,
        layout.desktop.font_scale,
        layout.tablet.width,
        layout.tablet.height,
        layout.tablet.margins.top,
        layout.tablet.margins.right,
        layout.tablet.margins.bottom,
        layout.tablet.margins.left,
        layout.tablet.font_scale,
        if layout.tablet.stack_vertically {
            "column"
        } else {
            "row"
        },
        layout.mobile.width,
        layout.mobile.height,
        layout.mobile.margins.top,
        layout.mobile.margins.right,
        layout.mobile.margins.bottom,
        layout.mobile.margins.left,
        layout.mobile.font_scale,
    )
}

/// Generate responsive JavaScript for mobile interactions
pub fn generate_responsive_javascript() -> String {
    r#"
    function initMobileOptimizations() {
        // Detect device type
        const isMobile = window.innerWidth <= 768;
        const isTablet = window.innerWidth <= 1024 && window.innerWidth > 768;

        // Optimize for mobile
        if (isMobile) {
            // Reduce animation duration for performance
            const animations = document.querySelectorAll('[data-animation]');
            animations.forEach(el => {
                el.style.animationDuration = '0.5s';
            });

            // Enable touch-friendly interactions
            enableTouchInteractions();

            // Simplify complex visualizations
            simplifyVisualizationsForMobile();
        }

        // Handle orientation changes
        window.addEventListener('orientationchange', function() {
            setTimeout(function() {
                resizeVisualizations();
            }, 100);
        });
    }

    function enableTouchInteractions() {
        const plots = document.querySelectorAll('.plot-container');
        plots.forEach(plot => {
            plot.style.touchAction = 'pan-x pan-y';
            plot.addEventListener('touchstart', handleTouchStart, {passive: true});
            plot.addEventListener('touchmove', handleTouchMove, {passive: true});
        });
    }

    function simplifyVisualizationsForMobile() {
        // Reduce data point density
        const dataPoints = document.querySelectorAll('.data-point');
        dataPoints.forEach((point, index) => {
            if (index % 2 === 1) {
                point.style.display = 'none';
            }
        });
    }

    function resizeVisualizations() {
        // Trigger resize event for all plots
        const event = new Event('resize');
        window.dispatchEvent(event);
    }

    function handleTouchStart(e) {
        // Handle touch start for mobile interactions
        const touch = e.touches[0];
        this.touchStartX = touch.clientX;
        this.touchStartY = touch.clientY;
    }

    function handleTouchMove(e) {
        if (!this.touchStartX || !this.touchStartY) {
            return;
        }

        const touch = e.touches[0];
        const deltaX = this.touchStartX - touch.clientX;
        const deltaY = this.touchStartY - touch.clientY;

        // Handle pan gestures
        if (Math.abs(deltaX) > Math.abs(deltaY)) {
            // Horizontal pan
            this.dispatchEvent(new CustomEvent('plotPan', {
                detail: { direction: deltaX > 0 ? 'left' : 'right', delta: Math.abs(deltaX) }
            }));
        } else {
            // Vertical pan
            this.dispatchEvent(new CustomEvent('plotPan', {
                detail: { direction: deltaY > 0 ? 'up' : 'down', delta: Math.abs(deltaY) }
            }));
        }
    }

    // Initialize on DOM load
    document.addEventListener('DOMContentLoaded', initMobileOptimizations);
    "#
    .to_string()
}

/// Detect device type based on screen dimensions
pub fn detect_device_type(width: usize, height: usize) -> DeviceType {
    let min_dimension = std::cmp::min(width, height);
    let max_dimension = std::cmp::max(width, height);

    if max_dimension <= 768 {
        DeviceType::Mobile
    } else if max_dimension <= 1024 {
        DeviceType::Tablet
    } else {
        DeviceType::Desktop
    }
}

/// Optimize plot configuration for device type
pub fn optimize_plot_for_device(mut config: PlotConfig, device_type: DeviceType) -> PlotConfig {
    match device_type {
        DeviceType::Mobile => {
            config.width = 350;
            config.height = 250;
            config.mobile_config.enabled = true;
            config.mobile_config.touch_friendly = true;
            config.mobile_config.simplify_on_mobile = true;
        }
        DeviceType::Tablet => {
            config.width = 600;
            config.height = 400;
            config.mobile_config.enabled = true;
            config.mobile_config.simplify_on_mobile = true;
        }
        DeviceType::Desktop => {
            // Keep default settings for desktop
        }
    }

    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mobile_config_default() {
        let config = MobileConfig::default();
        assert!(config.enabled);
        assert_eq!(config.mobile_breakpoint, 768);
        assert_eq!(config.tablet_breakpoint, 1024);
        assert!(config.touch_friendly);
    }

    #[test]
    fn test_mobile_plot_optimizer_creation() {
        let config = MobileConfig::default();
        let optimizer = MobilePlotOptimizer::new(config);
        assert_eq!(optimizer.cache_size(), 0);
    }

    #[test]
    fn test_mobile_optimization() {
        let config = MobileConfig::default();
        let mut optimizer = MobilePlotOptimizer::new(config);

        let optimized = optimizer.optimize_for_mobile("test_plot".to_string(), 1000);
        assert_eq!(optimized.data_points, 500); // Should be reduced for mobile
        assert!(optimized.touch_controls);
        assert_eq!(optimizer.cache_size(), 1);
    }

    #[test]
    fn test_device_type_detection() {
        assert_eq!(detect_device_type(400, 600), DeviceType::Mobile);
        assert_eq!(detect_device_type(768, 1024), DeviceType::Tablet);
        assert_eq!(detect_device_type(1200, 800), DeviceType::Desktop);
    }

    #[test]
    fn test_plot_optimization_for_device() {
        let config = PlotConfig::default();

        let mobile_config = optimize_plot_for_device(config.clone(), DeviceType::Mobile);
        assert_eq!(mobile_config.width, 350);
        assert_eq!(mobile_config.height, 250);

        let tablet_config = optimize_plot_for_device(config.clone(), DeviceType::Tablet);
        assert_eq!(tablet_config.width, 600);
        assert_eq!(tablet_config.height, 400);
    }

    #[test]
    fn test_responsive_css_generation() {
        let layout = ResponsiveLayout::default();
        let css = generate_responsive_css(&layout);

        assert!(css.contains("@media (max-width: 1024px)"));
        assert!(css.contains("@media (max-width: 768px)"));
        assert!(css.contains(".visualization-container"));
        assert!(css.contains(".touch-control"));
    }

    #[test]
    fn test_cache_management() {
        let config = MobileConfig::default();
        let mut optimizer = MobilePlotOptimizer::new(config);

        // Add some data to cache
        optimizer.optimize_for_mobile("plot1".to_string(), 1000);
        optimizer.optimize_for_mobile("plot2".to_string(), 1500);

        assert_eq!(optimizer.cache_size(), 2);

        optimizer.clear_cache();
        assert_eq!(optimizer.cache_size(), 0);
    }
}
