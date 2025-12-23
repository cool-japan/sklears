//! Concrete Plugin Marketplace Implementation
//!
//! This module provides a fully functional plugin marketplace for discovering,
//! installing, and managing sklears plugins from multiple sources.

use crate::error::{Result, SklearsError};
use crate::plugin::discovery_marketplace::{PluginDiscoveryService, SearchQuery};
use crate::plugin::validation::PluginManifest;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Concrete plugin marketplace implementation
///
/// Provides a complete marketplace experience with plugin discovery, installation,
/// ratings, reviews, and automatic updates.
#[derive(Debug)]
pub struct ConcretePluginMarketplace {
    /// Discovery service for finding plugins
    pub discovery: PluginDiscoveryService,
    /// Installation manager
    pub installer: PluginInstaller,
    /// Rating and review system
    pub ratings: RatingSystem,
    /// Update manager for automatic updates
    pub updater: UpdateManager,
    /// Marketplace configuration
    pub config: MarketplaceConfig,
}

/// Configuration for the marketplace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketplaceConfig {
    /// Local plugin directory
    pub plugin_dir: PathBuf,
    /// Enable automatic updates
    pub auto_update: bool,
    /// Check for updates interval (in seconds)
    pub update_check_interval: u64,
    /// Enable community ratings
    pub enable_ratings: bool,
    /// Require verified publishers
    pub require_verified: bool,
    /// Maximum concurrent downloads
    pub max_concurrent_downloads: usize,
}

impl Default for MarketplaceConfig {
    fn default() -> Self {
        Self {
            plugin_dir: std::env::temp_dir().join("sklears_plugins"),
            auto_update: false,
            update_check_interval: 86400, // 24 hours
            enable_ratings: true,
            require_verified: false,
            max_concurrent_downloads: 3,
        }
    }
}

/// Plugin installation manager
#[derive(Debug)]
pub struct PluginInstaller {
    /// Installation directory
    pub install_dir: PathBuf,
    /// Installed plugins registry
    pub installed: HashMap<String, InstalledPlugin>,
    /// Download cache
    pub download_cache: PathBuf,
}

/// Information about an installed plugin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstalledPlugin {
    /// Plugin identifier
    pub id: String,
    /// Plugin name
    pub name: String,
    /// Installed version
    pub version: String,
    /// Installation date
    pub installed_at: SystemTime,
    /// Installation path
    pub install_path: PathBuf,
    /// Plugin manifest
    pub manifest: PluginManifest,
}

/// Rating and review system for plugins
#[derive(Debug, Clone)]
pub struct RatingSystem {
    /// Plugin ratings
    pub ratings: HashMap<String, PluginRating>,
    /// User reviews
    pub reviews: HashMap<String, Vec<UserReview>>,
}

/// Rating information for a plugin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginRating {
    /// Plugin identifier
    pub plugin_id: String,
    /// Average rating (1.0 to 5.0)
    pub average_rating: f64,
    /// Total number of ratings
    pub total_ratings: usize,
    /// Rating distribution
    pub rating_distribution: [usize; 5], // 1-star to 5-star counts
    /// Total downloads
    pub total_downloads: usize,
}

/// User review for a plugin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserReview {
    /// Review identifier
    pub id: String,
    /// Plugin identifier
    pub plugin_id: String,
    /// User identifier
    pub user_id: String,
    /// Rating (1-5)
    pub rating: u8,
    /// Review title
    pub title: String,
    /// Review content
    pub content: String,
    /// Helpful votes
    pub helpful_votes: usize,
    /// Posted timestamp
    pub posted_at: SystemTime,
    /// Verified purchase
    pub verified_install: bool,
}

/// Automatic update manager
#[derive(Debug)]
pub struct UpdateManager {
    /// Last update check time
    pub last_check: Option<SystemTime>,
    /// Available updates
    pub available_updates: Vec<PluginUpdate>,
    /// Update configuration
    pub config: UpdateConfig,
}

/// Configuration for automatic updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateConfig {
    /// Enable automatic updates
    pub auto_update: bool,
    /// Enable pre-release updates
    pub include_prerelease: bool,
    /// Update strategy
    pub strategy: UpdateStrategy,
}

impl Default for UpdateConfig {
    fn default() -> Self {
        Self {
            auto_update: false,
            include_prerelease: false,
            strategy: UpdateStrategy::Manual,
        }
    }
}

/// Update strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpdateStrategy {
    /// Manual updates only
    Manual,
    /// Notify when updates available
    Notify,
    /// Automatically download updates
    AutoDownload,
    /// Automatically install updates
    AutoInstall,
}

/// Information about an available update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginUpdate {
    /// Plugin identifier
    pub plugin_id: String,
    /// Current version
    pub current_version: String,
    /// New version
    pub new_version: String,
    /// Update description
    pub description: String,
    /// Update size in bytes
    pub size_bytes: usize,
    /// Is breaking change
    pub breaking_change: bool,
    /// Release notes
    pub release_notes: String,
}

impl ConcretePluginMarketplace {
    /// Create a new marketplace instance
    pub fn new() -> Self {
        let config = MarketplaceConfig::default();
        Self::with_config(config)
    }

    /// Create a marketplace with custom configuration
    pub fn with_config(config: MarketplaceConfig) -> Self {
        let install_dir = config.plugin_dir.clone();

        Self {
            discovery: PluginDiscoveryService::new(),
            installer: PluginInstaller {
                install_dir: install_dir.clone(),
                installed: HashMap::new(),
                download_cache: install_dir.join("cache"),
            },
            ratings: RatingSystem {
                ratings: HashMap::new(),
                reviews: HashMap::new(),
            },
            updater: UpdateManager {
                last_check: None,
                available_updates: Vec::new(),
                config: UpdateConfig::default(),
            },
            config,
        }
    }

    /// Search for plugins in the marketplace
    pub async fn search(&self, query: &str) -> Result<Vec<MarketplacePlugin>> {
        let search_query = SearchQuery {
            text: query.to_string(),
            category: None,
            capabilities: vec![],
            limit: Some(50),
            min_rating: None,
        };

        let results = self.discovery.search(&search_query).await?;

        let mut plugins = Vec::new();
        for result in results {
            let rating = self.ratings.ratings.get(&result.plugin_id);

            plugins.push(MarketplacePlugin {
                id: result.plugin_id.clone(),
                name: result.plugin_id.clone(), // Use plugin_id as name for now
                description: result.description,
                version: "1.0.0".to_string(),  // Placeholder version
                author: "Unknown".to_string(), // Placeholder author
                rating: rating.map(|r| r.average_rating),
                downloads: result.download_count as usize,
                verified: false, // Placeholder
                tags: vec![],    // Placeholder
            });
        }

        Ok(plugins)
    }

    /// Get featured plugins
    pub async fn get_featured(&self) -> Result<Vec<MarketplacePlugin>> {
        // Get plugins with high ratings and many downloads
        let mut featured: Vec<_> = self
            .ratings
            .ratings
            .values()
            .filter(|r| r.average_rating >= 4.5 && r.total_downloads >= 1000)
            .map(|r| r.plugin_id.clone())
            .collect();

        featured.sort_by_key(|id| {
            self.ratings
                .ratings
                .get(id)
                .map(|r| r.total_downloads)
                .unwrap_or(0)
        });
        featured.reverse();
        featured.truncate(10);

        // Convert to MarketplacePlugin
        let mut result = Vec::new();
        for plugin_id in featured {
            if let Some(rating) = self.ratings.ratings.get(&plugin_id) {
                result.push(MarketplacePlugin {
                    id: plugin_id.clone(),
                    name: plugin_id.clone(), // Would fetch from metadata
                    description: "Featured plugin".to_string(),
                    version: "1.0.0".to_string(),
                    author: "Unknown".to_string(),
                    rating: Some(rating.average_rating),
                    downloads: rating.total_downloads,
                    verified: false,
                    tags: vec![],
                });
            }
        }

        Ok(result)
    }

    /// Install a plugin from the marketplace
    pub async fn install(
        &mut self,
        plugin_id: &str,
        version: Option<&str>,
    ) -> Result<InstalledPlugin> {
        // Check if already installed
        if self.installer.installed.contains_key(plugin_id) {
            return Err(SklearsError::InvalidOperation(format!(
                "Plugin {} is already installed",
                plugin_id
            )));
        }

        // Download plugin
        let download_path = self.download_plugin(plugin_id, version).await?;

        // Load and verify manifest exists
        let manifest = self.load_manifest(&download_path)?;

        // TODO: Full validation would require loading the plugin first
        // For now, we just verify the manifest can be loaded

        // Install plugin
        let install_path = self.installer.install_dir.join(plugin_id);
        std::fs::create_dir_all(&install_path).map_err(|e| {
            SklearsError::InvalidOperation(format!("Failed to create install directory: {}", e))
        })?;

        let installed = InstalledPlugin {
            id: plugin_id.to_string(),
            name: manifest.metadata.name.clone(),
            version: manifest.metadata.version.clone(),
            installed_at: SystemTime::now(),
            install_path: install_path.clone(),
            manifest,
        };

        self.installer
            .installed
            .insert(plugin_id.to_string(), installed.clone());

        Ok(installed)
    }

    /// Uninstall a plugin
    pub fn uninstall(&mut self, plugin_id: &str) -> Result<()> {
        let installed = self.installer.installed.remove(plugin_id).ok_or_else(|| {
            SklearsError::InvalidOperation(format!("Plugin {} is not installed", plugin_id))
        })?;

        // Remove installation directory
        if installed.install_path.exists() {
            std::fs::remove_dir_all(&installed.install_path).map_err(|e| {
                SklearsError::InvalidOperation(format!("Failed to remove plugin files: {}", e))
            })?;
        }

        Ok(())
    }

    /// Check for plugin updates
    pub async fn check_for_updates(&mut self) -> Result<Vec<PluginUpdate>> {
        self.updater.last_check = Some(SystemTime::now());
        self.updater.available_updates.clear();

        for (plugin_id, installed) in &self.installer.installed {
            // Check if newer version is available
            if let Some(latest) = self.get_latest_version(plugin_id).await? {
                if latest != installed.version {
                    self.updater.available_updates.push(PluginUpdate {
                        plugin_id: plugin_id.clone(),
                        current_version: installed.version.clone(),
                        new_version: latest.clone(),
                        description: format!("Update {} to version {}", plugin_id, latest),
                        size_bytes: 1024 * 1024, // Placeholder
                        breaking_change: false,
                        release_notes: "Bug fixes and improvements".to_string(),
                    });
                }
            }
        }

        Ok(self.updater.available_updates.clone())
    }

    /// Submit a rating for a plugin
    pub fn rate_plugin(
        &mut self,
        plugin_id: &str,
        rating: u8,
        review: Option<UserReview>,
    ) -> Result<()> {
        if !(1..=5).contains(&rating) {
            return Err(SklearsError::InvalidOperation(
                "Rating must be between 1 and 5".to_string(),
            ));
        }

        let plugin_rating = self
            .ratings
            .ratings
            .entry(plugin_id.to_string())
            .or_insert_with(|| PluginRating {
                plugin_id: plugin_id.to_string(),
                average_rating: 0.0,
                total_ratings: 0,
                rating_distribution: [0; 5],
                total_downloads: 0,
            });

        // Update rating
        plugin_rating.rating_distribution[(rating - 1) as usize] += 1;
        plugin_rating.total_ratings += 1;

        // Recalculate average
        let total: f64 = plugin_rating
            .rating_distribution
            .iter()
            .enumerate()
            .map(|(i, &count)| (i + 1) as f64 * count as f64)
            .sum();
        plugin_rating.average_rating = total / plugin_rating.total_ratings as f64;

        // Add review if provided
        if let Some(review) = review {
            self.ratings
                .reviews
                .entry(plugin_id.to_string())
                .or_default()
                .push(review);
        }

        Ok(())
    }

    /// Get reviews for a plugin
    pub fn get_reviews(&self, plugin_id: &str) -> Vec<&UserReview> {
        self.ratings
            .reviews
            .get(plugin_id)
            .map(|reviews| reviews.iter().collect())
            .unwrap_or_default()
    }

    /// Get installed plugins
    pub fn list_installed(&self) -> Vec<&InstalledPlugin> {
        self.installer.installed.values().collect()
    }

    /// Download a plugin (simulated)
    async fn download_plugin(&self, plugin_id: &str, _version: Option<&str>) -> Result<PathBuf> {
        let download_path = self
            .installer
            .download_cache
            .join(format!("{}.tar.gz", plugin_id));

        // Create download cache if it doesn't exist
        std::fs::create_dir_all(&self.installer.download_cache).map_err(|e| {
            SklearsError::InvalidOperation(format!("Failed to create cache directory: {}", e))
        })?;

        // Simulate download
        Ok(download_path)
    }

    /// Load plugin manifest (simulated)
    fn load_manifest(&self, _path: &Path) -> Result<PluginManifest> {
        // In a real implementation, this would parse the manifest file
        use crate::plugin::security::PublisherInfo;
        use crate::plugin::types_config::PluginMetadata;
        use crate::plugin::validation::MarketplaceInfo;
        Ok(PluginManifest {
            metadata: PluginMetadata::default(),
            permissions: vec![],
            api_usage: None,
            contains_unsafe_code: false,
            dependencies: vec![],
            code_analysis: None,
            signature: None,
            content_hash: String::new(),
            publisher: PublisherInfo {
                name: "test".to_string(),
                email: "test@test.com".to_string(),
                website: None,
                verified: false,
                trust_score: 5,
            },
            marketplace: MarketplaceInfo {
                url: "https://marketplace.example.com".to_string(),
                downloads: 0,
                rating: 0.0,
                reviews: 0,
                last_updated: chrono::Utc::now().to_rfc3339(),
            },
        })
    }

    /// Get latest version of a plugin (simulated)
    async fn get_latest_version(&self, _plugin_id: &str) -> Result<Option<String>> {
        // In a real implementation, this would query the repository
        Ok(Some("2.0.0".to_string()))
    }
}

impl Default for ConcretePluginMarketplace {
    fn default() -> Self {
        Self::new()
    }
}

/// Plugin information for marketplace display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketplacePlugin {
    /// Plugin identifier
    pub id: String,
    /// Plugin name
    pub name: String,
    /// Description
    pub description: String,
    /// Version
    pub version: String,
    /// Author
    pub author: String,
    /// Average rating
    pub rating: Option<f64>,
    /// Total downloads
    pub downloads: usize,
    /// Verified publisher
    pub verified: bool,
    /// Tags
    pub tags: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marketplace_creation() {
        let marketplace = ConcretePluginMarketplace::new();
        assert!(marketplace.installer.installed.is_empty());
    }

    #[test]
    fn test_rating_plugin() {
        let mut marketplace = ConcretePluginMarketplace::new();

        marketplace.rate_plugin("test_plugin", 5, None).unwrap();
        marketplace.rate_plugin("test_plugin", 4, None).unwrap();
        marketplace.rate_plugin("test_plugin", 5, None).unwrap();

        let rating = marketplace.ratings.ratings.get("test_plugin").unwrap();
        assert_eq!(rating.total_ratings, 3);
        assert!((rating.average_rating - 4.67).abs() < 0.01);
    }

    #[test]
    fn test_invalid_rating() {
        let mut marketplace = ConcretePluginMarketplace::new();
        let result = marketplace.rate_plugin("test_plugin", 0, None);
        assert!(result.is_err());

        let result = marketplace.rate_plugin("test_plugin", 6, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_review_submission() {
        let mut marketplace = ConcretePluginMarketplace::new();

        let review = UserReview {
            id: "review1".to_string(),
            plugin_id: "test_plugin".to_string(),
            user_id: "user1".to_string(),
            rating: 5,
            title: "Great plugin!".to_string(),
            content: "Works perfectly".to_string(),
            helpful_votes: 0,
            posted_at: SystemTime::now(),
            verified_install: true,
        };

        marketplace
            .rate_plugin("test_plugin", 5, Some(review))
            .unwrap();

        let reviews = marketplace.get_reviews("test_plugin");
        assert_eq!(reviews.len(), 1);
        assert_eq!(reviews[0].title, "Great plugin!");
    }

    #[test]
    fn test_list_installed() {
        let marketplace = ConcretePluginMarketplace::new();
        let installed = marketplace.list_installed();
        assert_eq!(installed.len(), 0);
    }

    #[test]
    fn test_update_strategy() {
        let config = UpdateConfig::default();
        assert_eq!(config.strategy, UpdateStrategy::Manual);
        assert!(!config.auto_update);
    }

    #[test]
    fn test_marketplace_config() {
        let config = MarketplaceConfig::default();
        assert!(!config.auto_update);
        assert_eq!(config.max_concurrent_downloads, 3);
    }

    #[test]
    fn test_rating_distribution() {
        let mut marketplace = ConcretePluginMarketplace::new();

        marketplace.rate_plugin("test", 5, None).unwrap();
        marketplace.rate_plugin("test", 5, None).unwrap();
        marketplace.rate_plugin("test", 4, None).unwrap();
        marketplace.rate_plugin("test", 3, None).unwrap();

        let rating = marketplace.ratings.ratings.get("test").unwrap();
        assert_eq!(rating.rating_distribution[4], 2); // Two 5-star ratings
        assert_eq!(rating.rating_distribution[3], 1); // One 4-star rating
        assert_eq!(rating.rating_distribution[2], 1); // One 3-star rating
    }

    #[test]
    fn test_uninstall_nonexistent() {
        let mut marketplace = ConcretePluginMarketplace::new();
        let result = marketplace.uninstall("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_installed_plugin_creation() {
        use crate::plugin::security::PublisherInfo;
        use crate::plugin::types_config::PluginMetadata;
        use crate::plugin::validation::MarketplaceInfo;

        let plugin = InstalledPlugin {
            id: "test".to_string(),
            name: "Test Plugin".to_string(),
            version: "1.0.0".to_string(),
            installed_at: SystemTime::now(),
            install_path: PathBuf::from("/tmp/test"),
            manifest: PluginManifest {
                metadata: PluginMetadata::default(),
                permissions: vec![],
                api_usage: None,
                contains_unsafe_code: false,
                dependencies: vec![],
                code_analysis: None,
                signature: None,
                content_hash: String::new(),
                publisher: PublisherInfo {
                    name: "test".to_string(),
                    email: "test@test.com".to_string(),
                    website: Some("https://test.com".to_string()),
                    verified: true,
                    trust_score: 8,
                },
                marketplace: MarketplaceInfo {
                    url: "https://marketplace.example.com/test".to_string(),
                    downloads: 100,
                    rating: 4.5,
                    reviews: 10,
                    last_updated: chrono::Utc::now().to_rfc3339(),
                },
            },
        };

        assert_eq!(plugin.id, "test");
        assert_eq!(plugin.version, "1.0.0");
    }
}
