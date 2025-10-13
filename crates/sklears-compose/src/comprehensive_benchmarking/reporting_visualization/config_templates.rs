//! Template configuration management
//!
//! This module handles export template definitions, metadata, usage statistics,
//! performance metrics, and compatibility information for template-based exports.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, Duration};

use super::format_definitions::ExportFormat;

/// Export template definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportTemplate {
    /// Unique template identifier
    pub template_id: String,
    /// Human-readable template name
    pub template_name: String,
    /// Template description
    pub template_description: String,
    /// Format-specific settings
    pub format_settings: HashMap<String, String>,
    /// Quality configuration settings
    pub quality_settings: HashMap<String, String>,
    /// Template metadata
    pub metadata: TemplateMetadata,
    /// Template category
    pub category: TemplateCategory,
    /// Template tags for searching/filtering
    pub tags: Vec<String>,
    /// Default export format for this template
    pub default_format: Option<ExportFormat>,
    /// Template dependencies
    pub dependencies: Vec<String>,
    /// Compatibility information
    pub compatibility: CompatibilityInfo,
}

/// Template metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateMetadata {
    /// Template author
    pub author: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last modified timestamp
    pub updated_at: DateTime<Utc>,
    /// Template version
    pub version: String,
    /// License information
    pub license: Option<String>,
    /// Documentation URL
    pub documentation_url: Option<String>,
    /// Usage statistics
    pub usage_stats: TemplateUsageStats,
}

impl Default for TemplateMetadata {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            author: "System".to_string(),
            created_at: now,
            updated_at: now,
            version: "1.0.0".to_string(),
            license: Some("MIT".to_string()),
            documentation_url: None,
            usage_stats: TemplateUsageStats::default(),
        }
    }
}

/// Template usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateUsageStats {
    /// Number of times used
    pub usage_count: u64,
    /// Last used timestamp
    pub last_used: Option<DateTime<Utc>>,
    /// Average rating (0.0-5.0)
    pub average_rating: f64,
    /// Number of ratings
    pub rating_count: u32,
    /// Performance metrics
    pub performance_metrics: TemplatePerformanceMetrics,
}

impl Default for TemplateUsageStats {
    fn default() -> Self {
        Self {
            usage_count: 0,
            last_used: None,
            average_rating: 0.0,
            rating_count: 0,
            performance_metrics: TemplatePerformanceMetrics::default(),
        }
    }
}

/// Template performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplatePerformanceMetrics {
    /// Average processing time
    pub average_processing_time: Option<Duration>,
    /// Success rate (0.0-1.0)
    pub success_rate: f64,
    /// Average output quality score
    pub average_quality_score: f64,
    /// Resource efficiency score
    pub resource_efficiency: f64,
}

impl Default for TemplatePerformanceMetrics {
    fn default() -> Self {
        Self {
            average_processing_time: None,
            success_rate: 1.0,
            average_quality_score: 90.0,
            resource_efficiency: 0.8,
        }
    }
}

/// Template categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemplateCategory {
    /// Web publishing templates
    Web,
    /// Print publishing templates
    Print,
    /// Social media templates
    SocialMedia,
    /// Professional/business templates
    Professional,
    /// Archive/backup templates
    Archive,
    /// Development/testing templates
    Development,
    /// High-quality/art templates
    HighQuality,
    /// Performance/speed optimized templates
    Performance,
    /// Custom category
    Custom(String),
}

/// Compatibility information for templates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityInfo {
    /// Supported platforms
    pub platforms: Vec<Platform>,
    /// Minimum system requirements
    pub min_requirements: SystemRequirements,
    /// Supported export formats
    pub supported_formats: Vec<ExportFormat>,
    /// Browser compatibility (for web templates)
    pub browser_compatibility: Option<BrowserCompatibility>,
    /// Software compatibility
    pub software_compatibility: SoftwareCompatibility,
}

impl Default for CompatibilityInfo {
    fn default() -> Self {
        Self {
            platforms: vec![Platform::Windows, Platform::MacOS, Platform::Linux],
            min_requirements: SystemRequirements::default(),
            supported_formats: vec![],
            browser_compatibility: None,
            software_compatibility: SoftwareCompatibility::default(),
        }
    }
}

/// Supported platforms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Platform {
    /// Windows platform
    Windows,
    /// macOS platform
    MacOS,
    /// Linux platform
    Linux,
    /// iOS platform
    iOS,
    /// Android platform
    Android,
    /// Web platform
    Web,
    /// Custom platform
    Custom(String),
}

/// System requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemRequirements {
    /// Minimum RAM (MB)
    pub min_ram_mb: usize,
    /// Minimum storage (MB)
    pub min_storage_mb: usize,
    /// Required CPU features
    pub cpu_features: Vec<String>,
    /// GPU requirements
    pub gpu_requirements: Option<GpuRequirements>,
    /// Network requirements
    pub network_requirements: Option<NetworkRequirements>,
}

impl Default for SystemRequirements {
    fn default() -> Self {
        Self {
            min_ram_mb: 512,
            min_storage_mb: 100,
            cpu_features: vec!["sse2".to_string()],
            gpu_requirements: None,
            network_requirements: None,
        }
    }
}

/// GPU requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuRequirements {
    /// Minimum GPU memory (MB)
    pub min_gpu_memory_mb: usize,
    /// Required GPU features
    pub gpu_features: Vec<String>,
    /// Supported GPU vendors
    pub supported_vendors: Vec<String>,
}

/// Network requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkRequirements {
    /// Minimum bandwidth (Mbps)
    pub min_bandwidth_mbps: f64,
    /// Latency tolerance (milliseconds)
    pub max_latency_ms: u64,
    /// Offline capability
    pub offline_capable: bool,
}

/// Browser compatibility information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserCompatibility {
    /// Supported browsers with minimum versions
    pub supported_browsers: HashMap<String, String>,
    /// Required browser features
    pub required_features: Vec<String>,
    /// Performance recommendations
    pub performance_recommendations: Vec<String>,
}

/// Software compatibility information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftwareCompatibility {
    /// Compatible with software versions
    pub compatible_versions: HashMap<String, String>,
    /// Required plugins/extensions
    pub required_plugins: Vec<String>,
    /// Optional enhancements
    pub optional_enhancements: Vec<String>,
}

impl Default for SoftwareCompatibility {
    fn default() -> Self {
        Self {
            compatible_versions: HashMap::new(),
            required_plugins: vec![],
            optional_enhancements: vec![],
        }
    }
}

/// Template management operations
impl ExportTemplate {
    /// Creates a new export template
    pub fn new(
        template_id: String,
        template_name: String,
        template_description: String,
        category: TemplateCategory,
    ) -> Self {
        Self {
            template_id,
            template_name,
            template_description,
            format_settings: HashMap::new(),
            quality_settings: HashMap::new(),
            metadata: TemplateMetadata::default(),
            category,
            tags: vec![],
            default_format: None,
            dependencies: vec![],
            compatibility: CompatibilityInfo::default(),
        }
    }

    /// Sets format-specific settings
    pub fn with_format_settings(mut self, settings: HashMap<String, String>) -> Self {
        self.format_settings = settings;
        self
    }

    /// Sets quality configuration settings
    pub fn with_quality_settings(mut self, settings: HashMap<String, String>) -> Self {
        self.quality_settings = settings;
        self
    }

    /// Sets template tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Sets default export format
    pub fn with_default_format(mut self, format: ExportFormat) -> Self {
        self.default_format = Some(format);
        self
    }

    /// Sets template dependencies
    pub fn with_dependencies(mut self, dependencies: Vec<String>) -> Self {
        self.dependencies = dependencies;
        self
    }

    /// Sets compatibility information
    pub fn with_compatibility(mut self, compatibility: CompatibilityInfo) -> Self {
        self.compatibility = compatibility;
        self
    }

    /// Updates usage statistics
    pub fn update_usage_stats(&mut self) {
        self.metadata.usage_stats.usage_count += 1;
        self.metadata.usage_stats.last_used = Some(Utc::now());
        self.metadata.updated_at = Utc::now();
    }

    /// Updates template rating
    pub fn update_rating(&mut self, rating: f64) {
        let stats = &mut self.metadata.usage_stats;
        let total_score = stats.average_rating * stats.rating_count as f64;
        stats.rating_count += 1;
        stats.average_rating = (total_score + rating) / stats.rating_count as f64;
    }

    /// Checks if template is compatible with given platform
    pub fn is_compatible_with_platform(&self, platform: &Platform) -> bool {
        self.compatibility.platforms.iter().any(|p| {
            match (p, platform) {
                (Platform::Custom(_), Platform::Custom(_)) => true,
                _ => std::mem::discriminant(p) == std::mem::discriminant(platform),
            }
        })
    }

    /// Checks if template supports given format
    pub fn supports_format(&self, format: &ExportFormat) -> bool {
        self.compatibility.supported_formats.iter().any(|f| {
            std::mem::discriminant(f) == std::mem::discriminant(format)
        })
    }

    /// Gets template efficiency score based on usage and performance
    pub fn get_efficiency_score(&self) -> f64 {
        let usage_score = if self.metadata.usage_stats.usage_count > 0 {
            (self.metadata.usage_stats.usage_count as f64).ln() / 10.0
        } else {
            0.0
        };

        let rating_score = self.metadata.usage_stats.average_rating / 5.0;
        let performance_score = self.metadata.usage_stats.performance_metrics.resource_efficiency;

        (usage_score + rating_score + performance_score) / 3.0
    }
}

impl TemplateMetadata {
    /// Updates the template metadata with new version
    pub fn update_version(&mut self, version: String) {
        self.version = version;
        self.updated_at = Utc::now();
    }

    /// Sets the author information
    pub fn set_author(&mut self, author: String) {
        self.author = author;
        self.updated_at = Utc::now();
    }

    /// Sets the license information
    pub fn set_license(&mut self, license: String) {
        self.license = Some(license);
        self.updated_at = Utc::now();
    }

    /// Sets the documentation URL
    pub fn set_documentation_url(&mut self, url: String) {
        self.documentation_url = Some(url);
        self.updated_at = Utc::now();
    }
}

impl CompatibilityInfo {
    /// Creates compatibility info for web templates
    pub fn web_compatible() -> Self {
        Self {
            platforms: vec![Platform::Web, Platform::Windows, Platform::MacOS, Platform::Linux],
            min_requirements: SystemRequirements {
                min_ram_mb: 256,
                min_storage_mb: 50,
                cpu_features: vec![],
                gpu_requirements: None,
                network_requirements: Some(NetworkRequirements {
                    min_bandwidth_mbps: 1.0,
                    max_latency_ms: 1000,
                    offline_capable: false,
                }),
            },
            supported_formats: vec![],
            browser_compatibility: Some(BrowserCompatibility {
                supported_browsers: {
                    let mut browsers = HashMap::new();
                    browsers.insert("Chrome".to_string(), "80+".to_string());
                    browsers.insert("Firefox".to_string(), "75+".to_string());
                    browsers.insert("Safari".to_string(), "13+".to_string());
                    browsers.insert("Edge".to_string(), "80+".to_string());
                    browsers
                },
                required_features: vec!["canvas".to_string(), "webgl".to_string()],
                performance_recommendations: vec![
                    "Enable hardware acceleration".to_string(),
                    "Use modern browser".to_string(),
                ],
            }),
            software_compatibility: SoftwareCompatibility::default(),
        }
    }

    /// Creates compatibility info for high-performance templates
    pub fn high_performance() -> Self {
        Self {
            platforms: vec![Platform::Windows, Platform::MacOS, Platform::Linux],
            min_requirements: SystemRequirements {
                min_ram_mb: 4096,
                min_storage_mb: 1024,
                cpu_features: vec!["avx2".to_string(), "sse4".to_string()],
                gpu_requirements: Some(GpuRequirements {
                    min_gpu_memory_mb: 2048,
                    gpu_features: vec!["cuda".to_string(), "opencl".to_string()],
                    supported_vendors: vec!["NVIDIA".to_string(), "AMD".to_string()],
                }),
                network_requirements: None,
            },
            supported_formats: vec![],
            browser_compatibility: None,
            software_compatibility: SoftwareCompatibility::default(),
        }
    }

    /// Creates compatibility info for mobile templates
    pub fn mobile_compatible() -> Self {
        Self {
            platforms: vec![Platform::iOS, Platform::Android],
            min_requirements: SystemRequirements {
                min_ram_mb: 1024,
                min_storage_mb: 200,
                cpu_features: vec!["neon".to_string()],
                gpu_requirements: None,
                network_requirements: Some(NetworkRequirements {
                    min_bandwidth_mbps: 0.5,
                    max_latency_ms: 2000,
                    offline_capable: true,
                }),
            },
            supported_formats: vec![],
            browser_compatibility: None,
            software_compatibility: SoftwareCompatibility::default(),
        }
    }
}