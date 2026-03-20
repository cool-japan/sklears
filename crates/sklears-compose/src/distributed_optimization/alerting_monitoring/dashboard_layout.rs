use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::dashboard_visualization::{
    PaddingConfiguration, ShadowConfiguration, BorderConfiguration,
    GradientConfiguration, AnimationConfiguration
};
use super::dashboard_widgets::{WidgetSize, WidgetPosition, Widget};

/// Dashboard layout
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardLayout {
    pub layout_type: LayoutType,
    pub grid_config: GridConfiguration,
    pub responsive_config: ResponsiveConfiguration,
    pub container_config: ContainerConfiguration,
}

/// Layout types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayoutType {
    Grid,
    Flex,
    Absolute,
    Flow,
    Masonry,
    Custom(String),
}

/// Grid configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridConfiguration {
    pub columns: u32,
    pub rows: Option<u32>,
    pub column_gap: f64,
    pub row_gap: f64,
    pub auto_rows: bool,
    pub auto_columns: bool,
    pub grid_template: Option<String>,
}

/// Responsive configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsiveConfiguration {
    pub enabled: bool,
    pub breakpoints: Vec<Breakpoint>,
    pub adaptation_strategy: AdaptationStrategy,
    pub reflow_enabled: bool,
}

/// Breakpoints for responsive design
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Breakpoint {
    pub name: String,
    pub width: f64,
    pub height: Option<f64>,
    pub layout_overrides: LayoutOverrides,
}

/// Layout overrides for breakpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutOverrides {
    pub grid_columns: Option<u32>,
    pub widget_sizes: HashMap<String, WidgetSize>,
    pub widget_positions: HashMap<String, WidgetPosition>,
    pub hidden_widgets: Vec<String>,
}

/// Adaptation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationStrategy {
    Scale,
    Reflow,
    Hide,
    Stack,
    Carousel,
    Custom(String),
}

/// Container configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerConfiguration {
    pub max_width: Option<f64>,
    pub max_height: Option<f64>,
    pub padding: PaddingConfiguration,
    pub margin: PaddingConfiguration,
    pub overflow: OverflowBehavior,
    pub scrolling: ScrollingConfiguration,
}

/// Overflow behaviors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OverflowBehavior {
    Visible,
    Hidden,
    Scroll,
    Auto,
    Clip,
}

/// Scrolling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrollingConfiguration {
    pub horizontal_scrolling: bool,
    pub vertical_scrolling: bool,
    pub smooth_scrolling: bool,
    pub scroll_bars: ScrollBarConfiguration,
}

/// Scroll bar configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrollBarConfiguration {
    pub visible: bool,
    pub width: f64,
    pub color: String,
    pub background_color: String,
    pub style: ScrollBarStyle,
}

/// Scroll bar styles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScrollBarStyle {
    Default,
    Minimal,
    Custom(ScrollBarCustomStyle),
}

/// Custom scroll bar style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrollBarCustomStyle {
    pub thumb_color: String,
    pub track_color: String,
    pub thumb_radius: f64,
    pub track_radius: f64,
}

/// Dashboard theme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardTheme {
    pub theme_id: String,
    pub name: String,
    pub description: Option<String>,
    pub base_theme: Option<String>,
    pub color_palette: ColorPalette,
    pub typography: Typography,
    pub spacing: SpacingTheme,
    pub effects: EffectsTheme,
    pub components: ComponentThemes,
}

/// Color palette for themes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorPalette {
    pub primary: String,
    pub secondary: String,
    pub accent: String,
    pub background: String,
    pub surface: String,
    pub text: String,
    pub text_secondary: String,
    pub border: String,
    pub error: String,
    pub warning: String,
    pub success: String,
    pub info: String,
    pub custom_colors: HashMap<String, String>,
}

/// Typography theme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Typography {
    pub font_families: FontFamilies,
    pub font_sizes: FontSizes,
    pub font_weights: FontWeights,
    pub line_heights: LineHeights,
    pub letter_spacing: LetterSpacing,
}

/// Font families
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontFamilies {
    pub primary: String,
    pub secondary: String,
    pub monospace: String,
    pub custom_fonts: HashMap<String, String>,
}

/// Font sizes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontSizes {
    pub xs: f64,
    pub sm: f64,
    pub md: f64,
    pub lg: f64,
    pub xl: f64,
    pub xxl: f64,
    pub custom_sizes: HashMap<String, f64>,
}

/// Font weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontWeights {
    pub light: u32,
    pub normal: u32,
    pub medium: u32,
    pub bold: u32,
    pub extra_bold: u32,
    pub custom_weights: HashMap<String, u32>,
}

/// Line heights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineHeights {
    pub tight: f64,
    pub normal: f64,
    pub relaxed: f64,
    pub loose: f64,
    pub custom_heights: HashMap<String, f64>,
}

/// Letter spacing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LetterSpacing {
    pub tight: f64,
    pub normal: f64,
    pub wide: f64,
    pub wider: f64,
    pub custom_spacing: HashMap<String, f64>,
}

/// Spacing theme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpacingTheme {
    pub base_unit: f64,
    pub scale_factor: f64,
    pub sizes: HashMap<String, f64>,
}

/// Effects theme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectsTheme {
    pub shadows: HashMap<String, ShadowConfiguration>,
    pub borders: HashMap<String, BorderConfiguration>,
    pub gradients: HashMap<String, GradientConfiguration>,
    pub animations: HashMap<String, AnimationConfiguration>,
}

/// Component themes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentThemes {
    pub button: ComponentTheme,
    pub input: ComponentTheme,
    pub card: ComponentTheme,
    pub chart: ComponentTheme,
    pub table: ComponentTheme,
    pub custom_components: HashMap<String, ComponentTheme>,
}

/// Individual component theme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentTheme {
    pub base_styles: HashMap<String, String>,
    pub variants: HashMap<String, HashMap<String, String>>,
    pub states: HashMap<String, HashMap<String, String>>,
}

/// Layout manager for handling complex layout operations
#[derive(Debug, Clone)]
pub struct LayoutManager {
    pub layouts: HashMap<String, DashboardLayout>,
    pub themes: HashMap<String, DashboardTheme>,
    pub active_breakpoint: Option<String>,
}

impl LayoutManager {
    pub fn new() -> Self {
        Self {
            layouts: HashMap::new(),
            themes: HashMap::new(),
            active_breakpoint: None,
        }
    }

    pub fn add_layout(&mut self, layout_id: String, layout: DashboardLayout) {
        self.layouts.insert(layout_id, layout);
    }

    pub fn remove_layout(&mut self, layout_id: &str) -> Option<DashboardLayout> {
        self.layouts.remove(layout_id)
    }

    pub fn get_layout(&self, layout_id: &str) -> Option<&DashboardLayout> {
        self.layouts.get(layout_id)
    }

    pub fn get_layout_mut(&mut self, layout_id: &str) -> Option<&mut DashboardLayout> {
        self.layouts.get_mut(layout_id)
    }

    pub fn add_theme(&mut self, theme_id: String, theme: DashboardTheme) {
        self.themes.insert(theme_id, theme);
    }

    pub fn remove_theme(&mut self, theme_id: &str) -> Option<DashboardTheme> {
        self.themes.remove(theme_id)
    }

    pub fn get_theme(&self, theme_id: &str) -> Option<&DashboardTheme> {
        self.themes.get(theme_id)
    }

    pub fn get_theme_mut(&mut self, theme_id: &str) -> Option<&mut DashboardTheme> {
        self.themes.get_mut(theme_id)
    }

    pub fn apply_responsive_layout(&mut self, layout_id: &str, viewport_width: f64, viewport_height: f64) -> Result<(), LayoutError> {
        let layout = self.get_layout_mut(layout_id)
            .ok_or_else(|| LayoutError::LayoutNotFound(layout_id.to_string()))?;

        if !layout.responsive_config.enabled {
            return Ok(());
        }

        // Find the appropriate breakpoint
        let mut matching_breakpoint: Option<&Breakpoint> = None;
        for breakpoint in &layout.responsive_config.breakpoints {
            if viewport_width >= breakpoint.width {
                if let Some(height) = breakpoint.height {
                    if viewport_height >= height {
                        matching_breakpoint = Some(breakpoint);
                    }
                } else {
                    matching_breakpoint = Some(breakpoint);
                }
            }
        }

        if let Some(breakpoint) = matching_breakpoint {
            self.active_breakpoint = Some(breakpoint.name.clone());
            // Apply layout overrides
            if let Some(grid_columns) = breakpoint.layout_overrides.grid_columns {
                layout.grid_config.columns = grid_columns;
            }
        }

        Ok(())
    }

    pub fn calculate_widget_positions(&self, layout_id: &str, widgets: &[Widget]) -> Result<HashMap<String, WidgetPosition>, LayoutError> {
        let layout = self.get_layout(layout_id)
            .ok_or_else(|| LayoutError::LayoutNotFound(layout_id.to_string()))?;

        let mut positions = HashMap::new();

        match layout.layout_type {
            LayoutType::Grid => {
                let column_width = 100.0 / layout.grid_config.columns as f64;
                let mut current_row = 0;
                let mut current_col = 0;

                for widget in widgets {
                    let x = current_col as f64 * column_width;
                    let y = current_row as f64 * 200.0; // Assuming 200px row height

                    positions.insert(widget.widget_id.clone(), WidgetPosition {
                        x,
                        y,
                        z_index: 0,
                    });

                    current_col += 1;
                    if current_col >= layout.grid_config.columns {
                        current_col = 0;
                        current_row += 1;
                    }
                }
            }
            LayoutType::Flex => {
                // Simple flex layout implementation
                let mut y_offset = 0.0;
                for widget in widgets {
                    positions.insert(widget.widget_id.clone(), WidgetPosition {
                        x: 0.0,
                        y: y_offset,
                        z_index: 0,
                    });
                    y_offset += widget.size.height + 10.0; // 10px gap
                }
            }
            LayoutType::Absolute => {
                // Keep existing positions
                for widget in widgets {
                    positions.insert(widget.widget_id.clone(), widget.position.clone());
                }
            }
            _ => {
                // Default to grid layout
                return self.calculate_widget_positions(layout_id, widgets);
            }
        }

        Ok(positions)
    }

    pub fn get_active_breakpoint(&self) -> Option<&String> {
        self.active_breakpoint.as_ref()
    }

    pub fn list_layouts(&self) -> Vec<&String> {
        self.layouts.keys().collect()
    }

    pub fn list_themes(&self) -> Vec<&String> {
        self.themes.keys().collect()
    }
}

impl Default for LayoutManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Theme builder for creating custom themes
#[derive(Debug, Clone)]
pub struct ThemeBuilder {
    theme: DashboardTheme,
}

impl ThemeBuilder {
    pub fn new(theme_id: String, name: String) -> Self {
        Self {
            theme: DashboardTheme {
                theme_id,
                name,
                description: None,
                base_theme: None,
                color_palette: ColorPalette::default(),
                typography: Typography::default(),
                spacing: SpacingTheme::default(),
                effects: EffectsTheme::default(),
                components: ComponentThemes::default(),
            },
        }
    }

    pub fn description(mut self, description: String) -> Self {
        self.theme.description = Some(description);
        self
    }

    pub fn base_theme(mut self, base_theme: String) -> Self {
        self.theme.base_theme = Some(base_theme);
        self
    }

    pub fn color_palette(mut self, color_palette: ColorPalette) -> Self {
        self.theme.color_palette = color_palette;
        self
    }

    pub fn typography(mut self, typography: Typography) -> Self {
        self.theme.typography = typography;
        self
    }

    pub fn spacing(mut self, spacing: SpacingTheme) -> Self {
        self.theme.spacing = spacing;
        self
    }

    pub fn effects(mut self, effects: EffectsTheme) -> Self {
        self.theme.effects = effects;
        self
    }

    pub fn components(mut self, components: ComponentThemes) -> Self {
        self.theme.components = components;
        self
    }

    pub fn build(self) -> DashboardTheme {
        self.theme
    }
}

/// Layout validation and error handling
#[derive(Debug, Clone)]
pub enum LayoutError {
    LayoutNotFound(String),
    ThemeNotFound(String),
    InvalidBreakpoint(String),
    InvalidGridConfiguration,
    InvalidResponsiveConfiguration,
}

impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LayoutNotFound(id) => write!(f, "Layout not found: {}", id),
            Self::ThemeNotFound(id) => write!(f, "Theme not found: {}", id),
            Self::InvalidBreakpoint(name) => write!(f, "Invalid breakpoint: {}", name),
            Self::InvalidGridConfiguration => write!(f, "Invalid grid configuration"),
            Self::InvalidResponsiveConfiguration => write!(f, "Invalid responsive configuration"),
        }
    }
}

impl std::error::Error for LayoutError {}

/// Default implementations for common structures
impl Default for ColorPalette {
    fn default() -> Self {
        Self {
            primary: "#007bff".to_string(),
            secondary: "#6c757d".to_string(),
            accent: "#fd7e14".to_string(),
            background: "#ffffff".to_string(),
            surface: "#f8f9fa".to_string(),
            text: "#212529".to_string(),
            text_secondary: "#6c757d".to_string(),
            border: "#dee2e6".to_string(),
            error: "#dc3545".to_string(),
            warning: "#ffc107".to_string(),
            success: "#28a745".to_string(),
            info: "#17a2b8".to_string(),
            custom_colors: HashMap::new(),
        }
    }
}

impl Default for Typography {
    fn default() -> Self {
        Self {
            font_families: FontFamilies::default(),
            font_sizes: FontSizes::default(),
            font_weights: FontWeights::default(),
            line_heights: LineHeights::default(),
            letter_spacing: LetterSpacing::default(),
        }
    }
}

impl Default for FontFamilies {
    fn default() -> Self {
        Self {
            primary: "Inter, system-ui, sans-serif".to_string(),
            secondary: "Georgia, serif".to_string(),
            monospace: "JetBrains Mono, Monaco, monospace".to_string(),
            custom_fonts: HashMap::new(),
        }
    }
}

impl Default for FontSizes {
    fn default() -> Self {
        Self {
            xs: 12.0,
            sm: 14.0,
            md: 16.0,
            lg: 18.0,
            xl: 20.0,
            xxl: 24.0,
            custom_sizes: HashMap::new(),
        }
    }
}

impl Default for FontWeights {
    fn default() -> Self {
        Self {
            light: 300,
            normal: 400,
            medium: 500,
            bold: 700,
            extra_bold: 800,
            custom_weights: HashMap::new(),
        }
    }
}

impl Default for LineHeights {
    fn default() -> Self {
        Self {
            tight: 1.2,
            normal: 1.5,
            relaxed: 1.7,
            loose: 2.0,
            custom_heights: HashMap::new(),
        }
    }
}

impl Default for LetterSpacing {
    fn default() -> Self {
        Self {
            tight: -0.025,
            normal: 0.0,
            wide: 0.025,
            wider: 0.05,
            custom_spacing: HashMap::new(),
        }
    }
}

impl Default for SpacingTheme {
    fn default() -> Self {
        let mut sizes = HashMap::new();
        sizes.insert("xs".to_string(), 4.0);
        sizes.insert("sm".to_string(), 8.0);
        sizes.insert("md".to_string(), 16.0);
        sizes.insert("lg".to_string(), 24.0);
        sizes.insert("xl".to_string(), 32.0);

        Self {
            base_unit: 4.0,
            scale_factor: 1.5,
            sizes,
        }
    }
}

impl Default for EffectsTheme {
    fn default() -> Self {
        Self {
            shadows: HashMap::new(),
            borders: HashMap::new(),
            gradients: HashMap::new(),
            animations: HashMap::new(),
        }
    }
}

impl Default for ComponentThemes {
    fn default() -> Self {
        Self {
            button: ComponentTheme::default(),
            input: ComponentTheme::default(),
            card: ComponentTheme::default(),
            chart: ComponentTheme::default(),
            table: ComponentTheme::default(),
            custom_components: HashMap::new(),
        }
    }
}

impl Default for ComponentTheme {
    fn default() -> Self {
        Self {
            base_styles: HashMap::new(),
            variants: HashMap::new(),
            states: HashMap::new(),
        }
    }
}