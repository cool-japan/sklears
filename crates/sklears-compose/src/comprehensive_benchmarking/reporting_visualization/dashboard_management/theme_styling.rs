use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Dashboard theme management system
/// Handles theme creation, inheritance, customization, and application
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardThemeManager {
    /// Available themes
    pub themes: HashMap<String, DashboardTheme>,
    /// Theme inheritance configuration
    pub inheritance: ThemeInheritance,
    /// Theme customization settings
    pub customization: ThemeCustomization,
}

/// Complete dashboard theme definition
/// Contains all styling information for dashboard components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardTheme {
    /// Theme identifier
    pub theme_id: String,
    /// Human-readable theme name
    pub theme_name: String,
    /// Theme description
    pub description: Option<String>,
    /// Theme colors configuration
    pub colors: ThemeColors,
    /// Typography settings
    pub typography: ThemeTypography,
    /// Spacing and layout settings
    pub spacing: ThemeSpacing,
    /// Component-specific themes
    pub components: ThemeComponents,
    /// Theme metadata
    pub metadata: ThemeMetadata,
}

/// Comprehensive color scheme for dashboard themes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeColors {
    /// Primary color scheme
    pub primary: ColorScheme,
    /// Secondary color scheme
    pub secondary: ColorScheme,
    /// Success color scheme
    pub success: ColorScheme,
    /// Warning color scheme
    pub warning: ColorScheme,
    /// Error color scheme
    pub error: ColorScheme,
    /// Info color scheme
    pub info: ColorScheme,
    /// Background colors
    pub background: BackgroundColors,
    /// Text colors
    pub text: TextColors,
    /// Border colors
    pub borders: BorderColors,
    /// Shadow colors
    pub shadows: ShadowColors,
}

/// Color scheme with variants and contrast
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorScheme {
    /// Main color
    pub main: String,
    /// Light variant
    pub light: String,
    /// Dark variant
    pub dark: String,
    /// Contrast text color
    pub contrast_text: String,
    /// Hover state color
    pub hover: Option<String>,
    /// Active state color
    pub active: Option<String>,
    /// Focus state color
    pub focus: Option<String>,
}

/// Background color configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackgroundColors {
    /// Default background
    pub default: String,
    /// Paper/card background
    pub paper: String,
    /// Surface background
    pub surface: String,
    /// Elevated surface background
    pub elevated: String,
    /// Overlay background
    pub overlay: String,
    /// Modal background
    pub modal: String,
    /// Tooltip background
    pub tooltip: String,
}

/// Text color configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextColors {
    /// Primary text color
    pub primary: String,
    /// Secondary text color
    pub secondary: String,
    /// Disabled text color
    pub disabled: String,
    /// Hint text color
    pub hint: String,
    /// Link text color
    pub link: String,
    /// Visited link color
    pub link_visited: String,
    /// Inverse text color (for dark backgrounds)
    pub inverse: String,
}

/// Border color configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BorderColors {
    /// Default border color
    pub default: String,
    /// Light border color
    pub light: String,
    /// Medium border color
    pub medium: String,
    /// Dark border color
    pub dark: String,
    /// Focus border color
    pub focus: String,
    /// Error border color
    pub error: String,
}

/// Shadow color configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowColors {
    /// Light shadow
    pub light: String,
    /// Medium shadow
    pub medium: String,
    /// Dark shadow
    pub dark: String,
    /// Colored shadow
    pub colored: String,
}

/// Typography configuration for themes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeTypography {
    /// Primary font family
    pub font_family: String,
    /// Secondary font family
    pub font_family_secondary: Option<String>,
    /// Monospace font family
    pub font_family_mono: String,
    /// Font sizes for different text types
    pub font_sizes: HashMap<String, f64>,
    /// Font weights mapping
    pub font_weights: HashMap<String, u16>,
    /// Line heights for different text types
    pub line_heights: HashMap<String, f64>,
    /// Letter spacing configuration
    pub letter_spacing: HashMap<String, f64>,
    /// Text transform options
    pub text_transforms: HashMap<String, String>,
}

/// Spacing and layout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeSpacing {
    /// Base spacing unit (typically 8px)
    pub base_unit: f64,
    /// Spacing scale multipliers
    pub scale: Vec<f64>,
    /// Component-specific spacing
    pub component_spacing: HashMap<String, f64>,
    /// Grid spacing
    pub grid_spacing: GridSpacing,
    /// Container spacing
    pub container_spacing: ContainerSpacing,
    /// Widget spacing
    pub widget_spacing: WidgetSpacing,
}

/// Grid-specific spacing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridSpacing {
    /// Column gap
    pub column_gap: f64,
    /// Row gap
    pub row_gap: f64,
    /// Container padding
    pub container_padding: f64,
    /// Item padding
    pub item_padding: f64,
}

/// Container spacing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerSpacing {
    /// Internal padding
    pub padding: f64,
    /// External margins
    pub margin: f64,
    /// Border radius
    pub border_radius: f64,
    /// Border width
    pub border_width: f64,
}

/// Widget-specific spacing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetSpacing {
    /// Widget margins
    pub margin: f64,
    /// Widget padding
    pub padding: f64,
    /// Widget header spacing
    pub header_spacing: f64,
    /// Widget content spacing
    pub content_spacing: f64,
    /// Widget footer spacing
    pub footer_spacing: f64,
}

/// Component-specific theme configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeComponents {
    /// Widget themes
    pub widgets: HashMap<String, WidgetTheme>,
    /// Layout themes
    pub layouts: HashMap<String, LayoutTheme>,
    /// Control themes (buttons, inputs, etc.)
    pub controls: HashMap<String, ControlTheme>,
    /// Chart themes
    pub charts: HashMap<String, ChartTheme>,
    /// Navigation themes
    pub navigation: HashMap<String, NavigationTheme>,
}

/// Widget-specific theme configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetTheme {
    /// Base widget styling
    pub styling: HashMap<String, String>,
    /// Widget state styles (hover, active, focus, disabled)
    pub states: HashMap<String, HashMap<String, String>>,
    /// Widget variants (outlined, filled, text, etc.)
    pub variants: HashMap<String, HashMap<String, String>>,
    /// Widget size variations
    pub sizes: HashMap<String, HashMap<String, String>>,
    /// Widget animations
    pub animations: WidgetAnimations,
}

/// Widget animation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetAnimations {
    /// Transition duration
    pub transition_duration: f64,
    /// Transition timing function
    pub transition_timing: String,
    /// Hover animations
    pub hover_animations: HashMap<String, String>,
    /// Loading animations
    pub loading_animations: HashMap<String, String>,
    /// State change animations
    pub state_animations: HashMap<String, String>,
}

/// Layout-specific theme configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutTheme {
    /// Layout styling properties
    pub styling: HashMap<String, String>,
    /// Responsive breakpoint styles
    pub breakpoints: HashMap<String, HashMap<String, String>>,
    /// Grid layout styles
    pub grid_styles: HashMap<String, String>,
    /// Flexbox layout styles
    pub flex_styles: HashMap<String, String>,
}

/// Control element theme configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlTheme {
    /// Base control styling
    pub styling: HashMap<String, String>,
    /// Interactive states (hover, focus, active, disabled)
    pub interactions: HashMap<String, HashMap<String, String>>,
    /// Control variants
    pub variants: HashMap<String, HashMap<String, String>>,
    /// Control sizes
    pub sizes: HashMap<String, HashMap<String, String>>,
}

/// Chart and visualization theme configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartTheme {
    /// Chart color palette
    pub color_palette: Vec<String>,
    /// Chart styling
    pub styling: HashMap<String, String>,
    /// Axis styling
    pub axis_styles: HashMap<String, String>,
    /// Legend styling
    pub legend_styles: HashMap<String, String>,
    /// Grid line styling
    pub grid_styles: HashMap<String, String>,
}

/// Navigation theme configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavigationTheme {
    /// Navigation styling
    pub styling: HashMap<String, String>,
    /// Menu item styles
    pub menu_items: HashMap<String, String>,
    /// Breadcrumb styles
    pub breadcrumbs: HashMap<String, String>,
    /// Tab styles
    pub tabs: HashMap<String, String>,
}

/// Theme inheritance system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeInheritance {
    /// Enable theme inheritance
    pub enabled: bool,
    /// Inheritance hierarchy (parent themes)
    pub hierarchy: Vec<String>,
    /// Property override rules
    pub override_rules: Vec<OverrideRule>,
    /// Merge strategy for conflicting properties
    pub merge_strategy: MergeStrategy,
}

/// Theme property override rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverrideRule {
    /// Target theme identifier
    pub target_theme: String,
    /// Properties to override
    pub properties: HashMap<String, String>,
    /// Conditions for applying override
    pub conditions: Vec<String>,
    /// Priority level
    pub priority: u32,
}

/// Strategy for merging inherited theme properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MergeStrategy {
    /// Child properties override parent
    ChildOverrides,
    /// Parent properties override child
    ParentOverrides,
    /// Merge properties where possible
    Merge,
    /// Custom merge strategy
    Custom(String),
}

/// Theme customization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeCustomization {
    /// Allow theme customization
    pub enabled: bool,
    /// Properties that can be customized
    pub customizable_properties: Vec<String>,
    /// Customization constraints
    pub constraints: Vec<CustomizationConstraint>,
    /// Custom property validators
    pub validators: Vec<PropertyValidator>,
    /// Default customization values
    pub defaults: HashMap<String, String>,
}

/// Constraint for theme customization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomizationConstraint {
    /// Property name
    pub property: String,
    /// Constraint type
    pub constraint_type: ConstraintType,
    /// Constraint value
    pub value: String,
    /// Error message for constraint violation
    pub error_message: Option<String>,
}

/// Property validator for custom validation logic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyValidator {
    /// Property name
    pub property: String,
    /// Validation function name
    pub validator_function: String,
    /// Validation parameters
    pub parameters: HashMap<String, String>,
}

/// Type of customization constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    /// Minimum value constraint
    Min,
    /// Maximum value constraint
    Max,
    /// Allowed values enumeration
    AllowedValues,
    /// Regular expression pattern
    Pattern,
    /// Color format validation
    ColorFormat,
    /// CSS unit validation
    CssUnit,
    /// Custom constraint function
    Custom(String),
}

/// Theme metadata for management and organization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeMetadata {
    /// Theme creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last modified timestamp
    pub modified_at: chrono::DateTime<chrono::Utc>,
    /// Theme author
    pub author: String,
    /// Theme version
    pub version: String,
    /// Theme category
    pub category: Option<String>,
    /// Theme tags
    pub tags: Vec<String>,
    /// Theme description
    pub description: Option<String>,
    /// Theme screenshot/preview URL
    pub preview_url: Option<String>,
    /// Theme popularity rating
    pub rating: Option<f64>,
    /// Usage statistics
    pub usage_count: u64,
}

/// Implementation of DashboardThemeManager
impl DashboardThemeManager {
    /// Create a new theme manager
    pub fn new() -> Self {
        Self {
            themes: HashMap::new(),
            inheritance: ThemeInheritance::default(),
            customization: ThemeCustomization::default(),
        }
    }

    /// Add a new theme
    pub fn add_theme(&mut self, theme: DashboardTheme) -> Result<(), ThemeError> {
        // Validate theme
        self.validate_theme(&theme)?;

        // Check for duplicate theme ID
        if self.themes.contains_key(&theme.theme_id) {
            return Err(ThemeError::DuplicateTheme(theme.theme_id));
        }

        self.themes.insert(theme.theme_id.clone(), theme);
        Ok(())
    }

    /// Get theme by ID
    pub fn get_theme(&self, theme_id: &str) -> Option<&DashboardTheme> {
        self.themes.get(theme_id)
    }

    /// Remove theme
    pub fn remove_theme(&mut self, theme_id: &str) -> Result<DashboardTheme, ThemeError> {
        self.themes.remove(theme_id)
            .ok_or_else(|| ThemeError::ThemeNotFound(theme_id.to_string()))
    }

    /// List all theme IDs
    pub fn list_themes(&self) -> Vec<String> {
        self.themes.keys().cloned().collect()
    }

    /// Get theme with inheritance applied
    pub fn get_resolved_theme(&self, theme_id: &str) -> Result<DashboardTheme, ThemeError> {
        let base_theme = self.get_theme(theme_id)
            .ok_or_else(|| ThemeError::ThemeNotFound(theme_id.to_string()))?;

        if !self.inheritance.enabled {
            return Ok(base_theme.clone());
        }

        // Apply inheritance
        let mut resolved_theme = base_theme.clone();
        for parent_id in &self.inheritance.hierarchy {
            if let Some(parent_theme) = self.get_theme(parent_id) {
                resolved_theme = self.merge_themes(&resolved_theme, parent_theme)?;
            }
        }

        // Apply override rules
        for rule in &self.inheritance.override_rules {
            if rule.target_theme == theme_id {
                resolved_theme = self.apply_override_rule(&resolved_theme, rule)?;
            }
        }

        Ok(resolved_theme)
    }

    /// Validate theme configuration
    fn validate_theme(&self, theme: &DashboardTheme) -> Result<(), ThemeError> {
        // Validate theme ID
        if theme.theme_id.is_empty() {
            return Err(ThemeError::InvalidTheme("Theme ID cannot be empty".to_string()));
        }

        // Validate color formats
        self.validate_color_scheme(&theme.colors.primary)?;
        self.validate_color_scheme(&theme.colors.secondary)?;

        // Validate typography
        if theme.typography.font_family.is_empty() {
            return Err(ThemeError::InvalidTheme("Font family cannot be empty".to_string()));
        }

        Ok(())
    }

    /// Validate color scheme
    fn validate_color_scheme(&self, colors: &ColorScheme) -> Result<(), ThemeError> {
        // Simple color validation (could be more sophisticated)
        let color_fields = vec![&colors.main, &colors.light, &colors.dark, &colors.contrast_text];

        for color in color_fields {
            if !self.is_valid_color(color) {
                return Err(ThemeError::InvalidColor(color.clone()));
            }
        }

        Ok(())
    }

    /// Basic color format validation
    fn is_valid_color(&self, color: &str) -> bool {
        // Simple validation for hex colors, rgb, rgba, hsl, hsla
        color.starts_with('#') ||
        color.starts_with("rgb") ||
        color.starts_with("hsl") ||
        color.starts_with("var(") ||
        ["transparent", "inherit", "initial", "unset"].contains(&color)
    }

    /// Merge two themes according to merge strategy
    fn merge_themes(&self, child: &DashboardTheme, parent: &DashboardTheme) -> Result<DashboardTheme, ThemeError> {
        let mut merged = child.clone();

        match self.inheritance.merge_strategy {
            MergeStrategy::ChildOverrides => {
                // Child properties take precedence, only fill missing values from parent
                // Implementation would merge missing properties
            },
            MergeStrategy::ParentOverrides => {
                // Parent properties override child
                merged = parent.clone();
                merged.theme_id = child.theme_id.clone();
                merged.theme_name = child.theme_name.clone();
            },
            MergeStrategy::Merge => {
                // Intelligent merging of properties
                // Implementation would merge properties intelligently
            },
            MergeStrategy::Custom(_) => {
                // Custom merge logic
                return Err(ThemeError::UnsupportedOperation("Custom merge strategy not implemented".to_string()));
            }
        }

        Ok(merged)
    }

    /// Apply override rule to theme
    fn apply_override_rule(&self, theme: &DashboardTheme, rule: &OverrideRule) -> Result<DashboardTheme, ThemeError> {
        let mut modified_theme = theme.clone();

        // Apply property overrides
        for (property, value) in &rule.properties {
            // This would need a proper property setter system
            // For now, just validate the override is possible
            if self.is_valid_property_override(property, value) {
                // Apply the override (implementation depends on property system)
            }
        }

        Ok(modified_theme)
    }

    /// Check if property override is valid
    fn is_valid_property_override(&self, property: &str, value: &str) -> bool {
        // Validate that the property can be overridden with the given value
        // Implementation would check against schema
        !property.is_empty() && !value.is_empty()
    }

    /// Customize theme with user preferences
    pub fn customize_theme(&mut self, theme_id: &str, customizations: HashMap<String, String>) -> Result<(), ThemeError> {
        if !self.customization.enabled {
            return Err(ThemeError::CustomizationDisabled);
        }

        let theme = self.themes.get_mut(theme_id)
            .ok_or_else(|| ThemeError::ThemeNotFound(theme_id.to_string()))?;

        // Validate customizations
        for (property, value) in &customizations {
            self.validate_customization(property, value)?;
        }

        // Apply customizations (implementation would modify theme properties)
        // For now, this is a placeholder
        Ok(())
    }

    /// Validate customization against constraints
    fn validate_customization(&self, property: &str, value: &str) -> Result<(), ThemeError> {
        // Check if property is customizable
        if !self.customization.customizable_properties.contains(&property.to_string()) {
            return Err(ThemeError::PropertyNotCustomizable(property.to_string()));
        }

        // Check constraints
        for constraint in &self.customization.constraints {
            if constraint.property == property {
                if !self.satisfies_constraint(value, &constraint.constraint_type, &constraint.value) {
                    return Err(ThemeError::ConstraintViolation(
                        constraint.error_message.clone()
                            .unwrap_or_else(|| format!("Value '{}' violates constraint for property '{}'", value, property))
                    ));
                }
            }
        }

        Ok(())
    }

    /// Check if value satisfies constraint
    fn satisfies_constraint(&self, value: &str, constraint_type: &ConstraintType, constraint_value: &str) -> bool {
        match constraint_type {
            ConstraintType::Pattern => {
                // Regex pattern matching (would use proper regex)
                true // Placeholder
            },
            ConstraintType::ColorFormat => {
                self.is_valid_color(value)
            },
            ConstraintType::AllowedValues => {
                let allowed: Vec<&str> = constraint_value.split(',').collect();
                allowed.contains(&value)
            },
            _ => true, // Placeholder for other constraint types
        }
    }
}

/// Theme-related error types
#[derive(Debug, thiserror::Error)]
pub enum ThemeError {
    #[error("Theme not found: {0}")]
    ThemeNotFound(String),
    #[error("Duplicate theme: {0}")]
    DuplicateTheme(String),
    #[error("Invalid theme: {0}")]
    InvalidTheme(String),
    #[error("Invalid color: {0}")]
    InvalidColor(String),
    #[error("Property not customizable: {0}")]
    PropertyNotCustomizable(String),
    #[error("Customization disabled")]
    CustomizationDisabled,
    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
}

// Default implementations

impl Default for DashboardThemeManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ThemeInheritance {
    fn default() -> Self {
        Self {
            enabled: true,
            hierarchy: Vec::new(),
            override_rules: Vec::new(),
            merge_strategy: MergeStrategy::ChildOverrides,
        }
    }
}

impl Default for ThemeCustomization {
    fn default() -> Self {
        Self {
            enabled: true,
            customizable_properties: vec![
                "colors.primary.main".to_string(),
                "colors.secondary.main".to_string(),
                "typography.font_family".to_string(),
                "spacing.base_unit".to_string(),
            ],
            constraints: Vec::new(),
            validators: Vec::new(),
            defaults: HashMap::new(),
        }
    }
}

impl Default for WidgetAnimations {
    fn default() -> Self {
        Self {
            transition_duration: 0.2,
            transition_timing: "ease-in-out".to_string(),
            hover_animations: HashMap::new(),
            loading_animations: HashMap::new(),
            state_animations: HashMap::new(),
        }
    }
}

impl Default for ThemeMetadata {
    fn default() -> Self {
        Self {
            created_at: chrono::Utc::now(),
            modified_at: chrono::Utc::now(),
            author: "system".to_string(),
            version: "1.0.0".to_string(),
            category: None,
            tags: Vec::new(),
            description: None,
            preview_url: None,
            rating: None,
            usage_count: 0,
        }
    }
}