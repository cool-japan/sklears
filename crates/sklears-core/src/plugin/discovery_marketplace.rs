//! Plugin Discovery and Marketplace
//!
//! This module provides comprehensive plugin discovery and marketplace functionality
//! for the sklears plugin system. It enables finding, installing, and managing
//! plugins from remote repositories and community marketplaces.

use super::core_traits::Plugin;
use super::types_config::{PluginCapability, PluginCategory, PluginMetadata};
use super::validation::{PluginManifest, PluginValidator, ValidationReport};
use crate::error::{Result, SklearsError};
use std::cmp::Ordering;
use std::collections::HashMap;

/// Plugin discovery service for remote repositories
///
/// The PluginDiscoveryService enables automatic discovery and installation of plugins
/// from configured remote repositories. It provides caching, search functionality,
/// and network-based plugin management.
///
/// # Features
///
/// - Multi-repository plugin discovery
/// - Intelligent caching and index management
/// - Advanced search capabilities with relevance scoring
/// - Automatic plugin validation and installation
/// - Network resilience and error handling
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_core::plugin::PluginDiscoveryService;
///
/// async fn discover_plugins() -> Result<(), Box<dyn std::error::Error>> {
///     let discovery = PluginDiscoveryService::new();
///
///     // Discover all available plugins
///     let plugins = discovery.discover_all().await?;
///     println!("Found {} plugins", plugins.len());
///
///     // Search for specific plugins
///     let query = SearchQuery {
///         text: "linear regression".to_string(),
///         category: Some(PluginCategory::Algorithm),
///         limit: Some(10),
///     };
///     let results = discovery.search(&query).await?;
///
///     // Install a plugin
///     if let Some(result) = results.first() {
///         let install_result = discovery.install_plugin(&result.plugin_id, None).await?;
///         println!("Installed plugin at: {}", install_result.install_path);
///     }
///
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct PluginDiscoveryService {
    /// Remote repositories for plugin discovery
    repositories: Vec<PluginRepository>,
    /// Local cache for repository data
    cache: PluginCache,
    /// Network client for remote operations
    client: NetworkClient,
    /// Search index for fast plugin lookups
    search_index: SearchIndex,
}

impl PluginDiscoveryService {
    /// Create a new discovery service
    ///
    /// Initializes the service with default official and community repositories.
    /// Additional repositories can be added after creation.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sklears_core::plugin::PluginDiscoveryService;
    ///
    /// let discovery = PluginDiscoveryService::new();
    /// ```
    pub fn new() -> Self {
        Self {
            repositories: vec![PluginRepository::official(), PluginRepository::community()],
            cache: PluginCache::new(),
            client: NetworkClient::new(),
            search_index: SearchIndex::new(),
        }
    }

    /// Add a custom repository
    ///
    /// Adds a new repository to the discovery service for plugin lookup.
    ///
    /// # Arguments
    ///
    /// * `repository` - The repository to add
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sklears_core::plugin::{PluginDiscoveryService, PluginRepository};
    ///
    /// let mut discovery = PluginDiscoveryService::new();
    /// let custom_repo = PluginRepository {
    ///     name: "Company Internal".to_string(),
    ///     url: "https://internal.company.com/plugins".to_string(),
    ///     verified: true,
    ///     priority: 5,
    /// };
    /// discovery.add_repository(custom_repo);
    /// ```
    pub fn add_repository(&mut self, repository: PluginRepository) {
        self.repositories.push(repository);
        // Sort by priority (higher priority first)
        self.repositories
            .sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Discover plugins from all repositories
    ///
    /// Scans all configured repositories for available plugins and returns
    /// their manifests. Results are cached for improved performance.
    ///
    /// # Returns
    ///
    /// A vector of plugin manifests from all repositories, or an error if
    /// no repositories are accessible.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sklears_core::plugin::PluginDiscoveryService;
    ///
    /// async fn discover_all() -> Result<(), Box<dyn std::error::Error>> {
    ///     let discovery = PluginDiscoveryService::new();
    ///     let all_plugins = discovery.discover_all().await?;
    ///
    ///     for manifest in &all_plugins {
    ///         println!("Found plugin: {} v{}",
    ///                  manifest.metadata.name,
    ///                  manifest.metadata.version);
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub async fn discover_all(&self) -> Result<Vec<PluginManifest>> {
        let mut all_plugins = Vec::new();
        let mut discovered_from_any = false;

        for repository in &self.repositories {
            match self.discover_from_repository(repository).await {
                Ok(mut plugins) => {
                    all_plugins.append(&mut plugins);
                    discovered_from_any = true;
                }
                Err(e) => {
                    eprintln!(
                        "Failed to discover from repository {}: {}",
                        repository.name, e
                    );
                }
            }
        }

        if !discovered_from_any && !self.repositories.is_empty() {
            return Err(SklearsError::InvalidOperation(
                "Failed to discover plugins from any repository".to_string(),
            ));
        }

        // Remove duplicates based on plugin name and version
        all_plugins.sort_by(|a, b| {
            a.metadata
                .name
                .cmp(&b.metadata.name)
                .then_with(|| a.metadata.version.cmp(&b.metadata.version))
        });
        all_plugins.dedup_by(|a, b| {
            a.metadata.name == b.metadata.name && a.metadata.version == b.metadata.version
        });

        Ok(all_plugins)
    }

    /// Discover plugins from a specific repository
    ///
    /// Fetches plugins from a single repository, utilizing caching to avoid
    /// unnecessary network requests.
    ///
    /// # Arguments
    ///
    /// * `repository` - The repository to query
    ///
    /// # Returns
    ///
    /// A vector of plugin manifests from the repository, or an error if
    /// the repository is inaccessible.
    pub async fn discover_from_repository(
        &self,
        repository: &PluginRepository,
    ) -> Result<Vec<PluginManifest>> {
        // Check cache first
        if let Some(cached) = self.cache.get_repository_plugins(&repository.url) {
            if !cached.is_expired() {
                return Ok(cached.plugins);
            }
        }

        // Fetch from remote
        let plugins = self
            .client
            .fetch_repository_plugins(repository)
            .await
            .map_err(|e| {
                SklearsError::InvalidOperation(format!(
                    "Failed to fetch plugins from {}: {}",
                    repository.name, e
                ))
            })?;

        // Update cache
        self.cache
            .store_repository_plugins(&repository.url, &plugins);

        // Update search index
        self.search_index.index_plugins(&plugins);

        Ok(plugins)
    }

    /// Search plugins by query
    ///
    /// Performs intelligent search across all known plugins using text matching,
    /// category filtering, and relevance scoring. Combines local index results
    /// with live repository searches for comprehensive results.
    ///
    /// # Arguments
    ///
    /// * `query` - The search query with filters and options
    ///
    /// # Returns
    ///
    /// A vector of search results sorted by relevance and popularity.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sklears_core::plugin::{PluginDiscoveryService, SearchQuery, PluginCategory};
    ///
    /// async fn search_plugins() -> Result<(), Box<dyn std::error::Error>> {
    ///     let discovery = PluginDiscoveryService::new();
    ///
    ///     let query = SearchQuery {
    ///         text: "classification".to_string(),
    ///         category: Some(PluginCategory::Algorithm),
    ///         capabilities: vec![PluginCapability::Parallel],
    ///         limit: Some(20),
    ///         min_rating: Some(4.0),
    ///     };
    ///
    ///     let results = discovery.search(&query).await?;
    ///     for result in results {
    ///         println!("{}: {} (score: {:.2})",
    ///                  result.plugin_id,
    ///                  result.description,
    ///                  result.relevance_score);
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub async fn search(&self, query: &SearchQuery) -> Result<Vec<PluginSearchResult>> {
        // First search local index
        let mut results = self.search_index.search(query)?;

        // If not enough results, search repositories
        let target_count = query.limit.unwrap_or(10);
        if results.len() < target_count {
            for repository in &self.repositories {
                if let Ok(remote_results) = self.client.search_repository(repository, query).await {
                    results.extend(remote_results);
                    if results.len() >= target_count * 2 {
                        break; // Enough results to sort and filter
                    }
                }
            }
        }

        // Apply filtering
        if let Some(category) = &query.category {
            results.retain(|r| r.category == *category);
        }

        if !query.capabilities.is_empty() {
            results.retain(|r| {
                query
                    .capabilities
                    .iter()
                    .all(|cap| r.capabilities.contains(cap))
            });
        }

        if let Some(min_rating) = query.min_rating {
            results.retain(|r| r.rating >= min_rating);
        }

        // Sort by relevance and popularity
        results.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| {
                    b.popularity_score
                        .partial_cmp(&a.popularity_score)
                        .unwrap_or(Ordering::Equal)
                })
        });

        Ok(results.into_iter().take(target_count).collect())
    }

    /// Download and install plugin
    ///
    /// Downloads a plugin from repositories, validates it comprehensively,
    /// and installs it locally for use.
    ///
    /// # Arguments
    ///
    /// * `plugin_id` - The unique identifier of the plugin to install
    /// * `version` - Optional specific version to install (latest if None)
    ///
    /// # Returns
    ///
    /// Installation result with manifest, path, and validation report.
    ///
    /// # Errors
    ///
    /// Returns an error if the plugin is not found, validation fails,
    /// or installation encounters issues.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sklears_core::plugin::PluginDiscoveryService;
    ///
    /// async fn install_plugin() -> Result<(), Box<dyn std::error::Error>> {
    ///     let discovery = PluginDiscoveryService::new();
    ///
    ///     let result = discovery.install_plugin("linear_regression", Some("2.1.0")).await?;
    ///
    ///     println!("Plugin installed at: {}", result.install_path);
    ///     if !result.validation_report.warnings.is_empty() {
    ///         println!("Warnings: {:?}", result.validation_report.warnings);
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub async fn install_plugin(
        &self,
        plugin_id: &str,
        version: Option<&str>,
    ) -> Result<PluginInstallResult> {
        // Find plugin in repositories
        let manifest = self.find_plugin_manifest(plugin_id, version).await?;

        // Validate plugin comprehensively
        let validator = PluginValidator::new();
        let dummy_plugin = DummyPlugin::new();
        let validation_report = validator.validate_comprehensive(&*dummy_plugin, &manifest)?;

        if validation_report.has_errors() {
            return Err(SklearsError::InvalidOperation(format!(
                "Plugin validation failed: {} errors found",
                validation_report.errors.len()
            )));
        }

        // Download plugin
        let plugin_data = self.client.download_plugin(&manifest).await.map_err(|e| {
            SklearsError::InvalidOperation(format!("Failed to download plugin: {}", e))
        })?;

        // Verify download integrity
        let computed_hash = self.compute_content_hash(&plugin_data);
        if computed_hash != manifest.content_hash {
            return Err(SklearsError::InvalidOperation(
                "Plugin content hash verification failed".to_string(),
            ));
        }

        // Install locally
        let install_path = self.install_plugin_locally(&manifest, &plugin_data)?;

        Ok(PluginInstallResult {
            manifest,
            install_path,
            validation_report,
        })
    }

    /// Find plugin manifest in repositories
    ///
    /// Searches through all configured repositories to find a plugin manifest
    /// matching the specified ID and optional version.
    async fn find_plugin_manifest(
        &self,
        plugin_id: &str,
        version: Option<&str>,
    ) -> Result<PluginManifest> {
        let mut last_error = None;

        for repository in &self.repositories {
            match self
                .client
                .get_plugin_manifest(repository, plugin_id, version)
                .await
            {
                Ok(manifest) => return Ok(manifest),
                Err(e) => last_error = Some(e),
            }
        }

        Err(SklearsError::InvalidOperation(format!(
            "Plugin '{}' not found in any repository. Last error: {:?}",
            plugin_id, last_error
        )))
    }

    /// Install plugin locally
    ///
    /// Handles the local installation of a downloaded plugin, including
    /// file extraction, permission setup, and registration preparation.
    fn install_plugin_locally(&self, manifest: &PluginManifest, data: &[u8]) -> Result<String> {
        // Create plugin directory
        let plugin_dir = format!("/tmp/plugins/{}", manifest.metadata.name);
        std::fs::create_dir_all(&plugin_dir).map_err(|e| {
            SklearsError::InvalidOperation(format!("Failed to create plugin directory: {}", e))
        })?;

        // Write plugin data
        let plugin_file = format!("{}/plugin.so", plugin_dir);
        std::fs::write(&plugin_file, data).map_err(|e| {
            SklearsError::InvalidOperation(format!("Failed to write plugin file: {}", e))
        })?;

        // Set appropriate permissions (readable and executable)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&plugin_file, std::fs::Permissions::from_mode(0o755))
                .map_err(|e| {
                    SklearsError::InvalidOperation(format!("Failed to set permissions: {}", e))
                })?;
        }

        // Write manifest for future reference
        let manifest_file = format!("{}/manifest.json", plugin_dir);
        let manifest_json = serde_json::to_string_pretty(manifest).map_err(|e| {
            SklearsError::InvalidOperation(format!("Failed to serialize manifest: {}", e))
        })?;
        std::fs::write(manifest_file, manifest_json).map_err(|e| {
            SklearsError::InvalidOperation(format!("Failed to write manifest: {}", e))
        })?;

        Ok(plugin_file)
    }

    /// Compute content hash for verification
    fn compute_content_hash(&self, data: &[u8]) -> String {
        // Simple hash implementation - in production, use SHA-256 or similar
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Get repository statistics
    ///
    /// Returns statistics about configured repositories and their status.
    pub async fn get_repository_stats(&self) -> Vec<RepositoryStats> {
        let mut stats = Vec::new();

        for repository in &self.repositories {
            let plugin_count = match self.discover_from_repository(repository).await {
                Ok(plugins) => plugins.len(),
                Err(_) => 0,
            };

            stats.push(RepositoryStats {
                name: repository.name.clone(),
                url: repository.url.clone(),
                plugin_count,
                verified: repository.verified,
                last_updated: std::time::SystemTime::now(), // Placeholder
            });
        }

        stats
    }
}

impl Default for PluginDiscoveryService {
    fn default() -> Self {
        Self::new()
    }
}

/// Community plugin marketplace
///
/// The PluginMarketplace provides a comprehensive platform for plugin discovery,
/// rating, reviewing, and analytics. It combines the discovery service with
/// community features to create a full marketplace experience.
///
/// # Features
///
/// - Featured plugin recommendations
/// - Community ratings and reviews
/// - Download tracking and analytics
/// - Plugin popularity scoring
/// - Trend analysis and reporting
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_core::plugin::PluginMarketplace;
///
/// async fn marketplace_demo() -> Result<(), Box<dyn std::error::Error>> {
///     let marketplace = PluginMarketplace::new();
///
///     // Get featured plugins
///     let featured = marketplace.get_featured_plugins().await?;
///     println!("Featured plugins: {}", featured.len());
///
///     // Rate a plugin
///     marketplace.rate_plugin("linear_regression", "user123", 4.5).await?;
///
///     // Get plugin statistics
///     let stats = marketplace.get_plugin_stats("linear_regression").await?;
///     println!("Average rating: {:.1}", stats.average_rating);
///
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct PluginMarketplace {
    /// Discovery service for plugin management
    discovery: PluginDiscoveryService,
    /// Rating system for community feedback
    rating_system: RatingSystem,
    /// Review system for detailed feedback
    review_system: ReviewSystem,
    /// Download tracking for popularity metrics
    download_tracker: DownloadTracker,
    /// Analytics engine for trend analysis
    analytics: PluginAnalytics,
}

impl PluginMarketplace {
    /// Create a new marketplace
    ///
    /// Initializes all marketplace components including discovery, ratings,
    /// reviews, and analytics systems.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_core::plugin::PluginMarketplace;
    ///
    /// let marketplace = PluginMarketplace::new();
    /// ```
    pub fn new() -> Self {
        Self {
            discovery: PluginDiscoveryService::new(),
            rating_system: RatingSystem::new(),
            review_system: ReviewSystem::new(),
            download_tracker: DownloadTracker::new(),
            analytics: PluginAnalytics::new(),
        }
    }

    /// Get featured plugins
    ///
    /// Returns a curated list of high-quality plugins based on ratings,
    /// download counts, and community engagement metrics.
    ///
    /// # Returns
    ///
    /// A vector of featured plugins sorted by feature score.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sklears_core::plugin::PluginMarketplace;
    ///
    /// async fn show_featured() -> Result<(), Box<dyn std::error::Error>> {
    ///     let marketplace = PluginMarketplace::new();
    ///     let featured = marketplace.get_featured_plugins().await?;
    ///
    ///     for plugin in featured {
    ///         println!("{}: {:.1} stars, {} downloads",
    ///                  plugin.manifest.metadata.name,
    ///                  plugin.rating,
    ///                  plugin.download_count);
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub async fn get_featured_plugins(&self) -> Result<Vec<FeaturedPlugin>> {
        let plugins = self.discovery.discover_all().await?;

        let mut featured = Vec::new();
        for plugin_manifest in plugins {
            let plugin_id = plugin_manifest.metadata.name.clone();

            // Get community metrics
            let rating = self.rating_system.get_average_rating(&plugin_id).await?;
            let download_count = self.download_tracker.get_download_count(&plugin_id).await?;
            let review_count = self.review_system.get_review_count(&plugin_id).await?;

            // Calculate feature score
            let feature_score = self.calculate_feature_score(rating, download_count, review_count);

            // Only include high-quality plugins
            if feature_score > 7.0 {
                featured.push(FeaturedPlugin {
                    manifest: plugin_manifest,
                    rating,
                    download_count,
                    review_count,
                    feature_score,
                    trend_direction: self
                        .analytics
                        .get_trend_direction(&plugin_id)
                        .await
                        .unwrap_or(TrendDirection::Stable),
                });
            }
        }

        // Sort by feature score (descending)
        featured.sort_by(|a, b| {
            b.feature_score
                .partial_cmp(&a.feature_score)
                .unwrap_or(Ordering::Equal)
        });

        Ok(featured.into_iter().take(10).collect())
    }

    /// Submit plugin rating
    ///
    /// Allows users to rate plugins on a 1-5 scale. Ratings are used for
    /// featured plugin selection and recommendation algorithms.
    ///
    /// # Arguments
    ///
    /// * `plugin_id` - The plugin to rate
    /// * `user_id` - The user submitting the rating
    /// * `rating` - Rating value (1.0 to 5.0)
    ///
    /// # Errors
    ///
    /// Returns an error if the rating is outside the valid range.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sklears_core::plugin::PluginMarketplace;
    ///
    /// async fn rate_plugin() -> Result<(), Box<dyn std::error::Error>> {
    ///     let marketplace = PluginMarketplace::new();
    ///     marketplace.rate_plugin("awesome_classifier", "user123", 4.8).await?;
    ///     println!("Rating submitted successfully");
    ///     Ok(())
    /// }
    /// ```
    pub async fn rate_plugin(&self, plugin_id: &str, user_id: &str, rating: f32) -> Result<()> {
        if !(1.0..=5.0).contains(&rating) {
            return Err(SklearsError::InvalidOperation(
                "Rating must be between 1.0 and 5.0".to_string(),
            ));
        }

        self.rating_system
            .submit_rating(plugin_id, user_id, rating)
            .await?;
        self.analytics.track_rating_event(plugin_id, rating).await?;

        Ok(())
    }

    /// Submit plugin review
    ///
    /// Allows users to submit detailed reviews for plugins, providing
    /// valuable feedback to other users and plugin developers.
    ///
    /// # Arguments
    ///
    /// * `plugin_id` - The plugin to review
    /// * `review` - The review content and metadata
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sklears_core::plugin::{PluginMarketplace, PluginReview};
    ///
    /// async fn submit_review() -> Result<(), Box<dyn std::error::Error>> {
    ///     let marketplace = PluginMarketplace::new();
    ///
    ///     let review = PluginReview {
    ///         user_id: "reviewer123".to_string(),
    ///         rating: 4.5,
    ///         title: "Excellent performance".to_string(),
    ///         content: "This plugin significantly improved our model accuracy.".to_string(),
    ///         verified_download: true,
    ///     };
    ///
    ///     marketplace.submit_review("awesome_classifier", review).await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn submit_review(&self, plugin_id: &str, review: PluginReview) -> Result<()> {
        self.review_system.submit_review(plugin_id, review).await?;
        self.analytics.track_review_event(plugin_id).await?;

        Ok(())
    }

    /// Get comprehensive plugin statistics
    ///
    /// Returns detailed statistics about a plugin including ratings,
    /// downloads, reviews, and trend data.
    ///
    /// # Arguments
    ///
    /// * `plugin_id` - The plugin to get statistics for
    ///
    /// # Returns
    ///
    /// Comprehensive plugin statistics and metrics.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sklears_core::plugin::PluginMarketplace;
    ///
    /// async fn show_stats() -> Result<(), Box<dyn std::error::Error>> {
    ///     let marketplace = PluginMarketplace::new();
    ///     let stats = marketplace.get_plugin_stats("popular_plugin").await?;
    ///
    ///     println!("Rating: {:.1}/5.0", stats.average_rating);
    ///     println!("Downloads: {}", stats.total_downloads);
    ///     println!("Reviews: {}", stats.recent_reviews.len());
    ///     Ok(())
    /// }
    /// ```
    pub async fn get_plugin_stats(&self, plugin_id: &str) -> Result<PluginStats> {
        let rating = self.rating_system.get_average_rating(plugin_id).await?;
        let downloads = self.download_tracker.get_download_count(plugin_id).await?;
        let reviews = self.review_system.get_reviews(plugin_id, 0, 5).await?;
        let trend = self.analytics.get_trend_data(plugin_id).await?;
        let rating_distribution = self
            .rating_system
            .get_rating_distribution(plugin_id)
            .await?;

        Ok(PluginStats {
            average_rating: rating,
            total_downloads: downloads,
            recent_reviews: reviews,
            trend_data: trend,
            rating_distribution,
            monthly_downloads: self
                .download_tracker
                .get_monthly_downloads(plugin_id)
                .await?,
            last_updated: self.analytics.get_last_update_time(plugin_id).await?,
        })
    }

    /// Get trending plugins
    ///
    /// Returns plugins that are currently trending based on recent
    /// download activity, ratings, and community engagement.
    pub async fn get_trending_plugins(&self, limit: usize) -> Result<Vec<TrendingPlugin>> {
        let all_plugins = self.discovery.discover_all().await?;
        let mut trending = Vec::new();

        for manifest in all_plugins {
            let plugin_id = manifest.metadata.name.clone();
            let trend_score = self.analytics.calculate_trend_score(&plugin_id).await?;

            if trend_score > 0.5 {
                // Threshold for trending
                trending.push(TrendingPlugin {
                    manifest,
                    trend_score,
                    recent_downloads: self
                        .download_tracker
                        .get_recent_downloads(&plugin_id, 7)
                        .await?,
                    velocity: self.analytics.get_download_velocity(&plugin_id).await?,
                });
            }
        }

        trending.sort_by(|a, b| {
            b.trend_score
                .partial_cmp(&a.trend_score)
                .unwrap_or(Ordering::Equal)
        });
        Ok(trending.into_iter().take(limit).collect())
    }

    /// Calculate feature score for plugin ranking
    ///
    /// Computes a composite score based on rating, downloads, and reviews
    /// to determine plugin quality and popularity for featured listings.
    pub fn calculate_feature_score(&self, rating: f32, downloads: u64, reviews: u64) -> f32 {
        let rating_weight = 0.4;
        let download_weight = 0.4;
        let review_weight = 0.2;

        // Normalize values to 0-10 scale
        let normalized_rating = rating * 2.0; // 5-star scale to 10-point scale

        // Log scale for downloads (handles zero values)
        let normalized_downloads = if downloads == 0 {
            0.0
        } else {
            (downloads as f32).log10().min(6.0) / 6.0 * 10.0 // Max at 1M downloads
        };

        // Log scale for reviews (handles zero values)
        let normalized_reviews = if reviews == 0 {
            0.0
        } else {
            (reviews as f32).log10().min(3.0) / 3.0 * 10.0 // Max at 1K reviews
        };

        normalized_rating * rating_weight
            + normalized_downloads * download_weight
            + normalized_reviews * review_weight
    }

    /// Get marketplace analytics summary
    ///
    /// Returns overall marketplace statistics and trends.
    pub async fn get_marketplace_summary(&self) -> Result<MarketplaceSummary> {
        let total_plugins = self.discovery.discover_all().await?.len();
        let total_downloads = self.download_tracker.get_total_downloads().await?;
        let active_users = self.rating_system.get_active_user_count().await?;
        let trending_categories = self.analytics.get_trending_categories().await?;

        Ok(MarketplaceSummary {
            total_plugins,
            total_downloads,
            active_users,
            trending_categories,
            last_updated: std::time::SystemTime::now(),
        })
    }
}

impl Default for PluginMarketplace {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Supporting Types
// =============================================================================

/// Plugin repository configuration
#[derive(Debug, Clone)]
pub struct PluginRepository {
    /// Repository name
    pub name: String,
    /// Repository URL
    pub url: String,
    /// Whether this repository is verified/trusted
    pub verified: bool,
    /// Repository priority (higher = checked first)
    pub priority: u8,
}

impl PluginRepository {
    /// Create the official SKLears plugin repository
    pub fn official() -> Self {
        Self {
            name: "Official".to_string(),
            url: "https://plugins.sklears.rs".to_string(),
            verified: true,
            priority: 10,
        }
    }

    /// Create the community plugin repository
    pub fn community() -> Self {
        Self {
            name: "Community".to_string(),
            url: "https://community.sklears.rs".to_string(),
            verified: true,
            priority: 5,
        }
    }
}

/// Search query for plugin discovery
#[derive(Debug, Clone)]
pub struct SearchQuery {
    /// Text to search for in plugin names and descriptions
    pub text: String,
    /// Filter by plugin category
    pub category: Option<PluginCategory>,
    /// Required capabilities
    pub capabilities: Vec<PluginCapability>,
    /// Maximum number of results
    pub limit: Option<usize>,
    /// Minimum rating filter
    pub min_rating: Option<f32>,
}

/// Plugin search result
#[derive(Debug, Clone)]
pub struct PluginSearchResult {
    /// Plugin identifier
    pub plugin_id: String,
    /// Plugin description
    pub description: String,
    /// Search relevance score
    pub relevance_score: f32,
    /// Popularity score
    pub popularity_score: f32,
    /// Plugin category
    pub category: PluginCategory,
    /// Plugin capabilities
    pub capabilities: Vec<PluginCapability>,
    /// Average rating
    pub rating: f32,
    /// Download count
    pub download_count: u64,
}

/// Plugin installation result
#[derive(Debug, Clone)]
pub struct PluginInstallResult {
    /// Plugin manifest
    pub manifest: PluginManifest,
    /// Installation path
    pub install_path: String,
    /// Validation report
    pub validation_report: ValidationReport,
}

/// Featured plugin information
#[derive(Debug, Clone)]
pub struct FeaturedPlugin {
    /// Plugin manifest
    pub manifest: PluginManifest,
    /// Average rating
    pub rating: f32,
    /// Total download count
    pub download_count: u64,
    /// Number of reviews
    pub review_count: u64,
    /// Feature score (0-10)
    pub feature_score: f32,
    /// Trend direction
    pub trend_direction: TrendDirection,
}

/// Plugin review
#[derive(Debug, Clone)]
pub struct PluginReview {
    /// User who submitted the review
    pub user_id: String,
    /// Rating given (1-5)
    pub rating: f32,
    /// Review title
    pub title: String,
    /// Review content
    pub content: String,
    /// Whether user has verified download
    pub verified_download: bool,
}

/// Comprehensive plugin statistics
#[derive(Debug, Clone)]
pub struct PluginStats {
    /// Average rating
    pub average_rating: f32,
    /// Total downloads
    pub total_downloads: u64,
    /// Recent reviews
    pub recent_reviews: Vec<PluginReview>,
    /// Trend data points
    pub trend_data: Vec<f32>,
    /// Rating distribution (1-5 stars)
    pub rating_distribution: HashMap<u8, u64>,
    /// Monthly download counts
    pub monthly_downloads: Vec<u64>,
    /// Last update time
    pub last_updated: std::time::SystemTime,
}

/// Trending plugin information
#[derive(Debug, Clone)]
pub struct TrendingPlugin {
    /// Plugin manifest
    pub manifest: PluginManifest,
    /// Trend score
    pub trend_score: f32,
    /// Recent downloads
    pub recent_downloads: u64,
    /// Download velocity (downloads per day)
    pub velocity: f32,
}

/// Repository statistics
#[derive(Debug, Clone)]
pub struct RepositoryStats {
    /// Repository name
    pub name: String,
    /// Repository URL
    pub url: String,
    /// Number of plugins
    pub plugin_count: usize,
    /// Whether repository is verified
    pub verified: bool,
    /// Last update time
    pub last_updated: std::time::SystemTime,
}

/// Marketplace summary statistics
#[derive(Debug, Clone)]
pub struct MarketplaceSummary {
    /// Total number of plugins
    pub total_plugins: usize,
    /// Total download count across all plugins
    pub total_downloads: u64,
    /// Number of active users
    pub active_users: u64,
    /// Trending categories
    pub trending_categories: Vec<PluginCategory>,
    /// Last update time
    pub last_updated: std::time::SystemTime,
}

/// Trend direction enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    Rising,
    Stable,
    Declining,
}

/// Marketplace specific metadata
#[derive(Debug, Clone)]
pub struct MarketplaceInfo {
    pub tags: Vec<String>,
    pub license: String,
    pub repository_url: Option<String>,
    pub documentation_url: Option<String>,
    pub pricing: PricingInfo,
    pub support_level: SupportLevel,
}

/// Plugin pricing information
#[derive(Debug, Clone)]
pub enum PricingInfo {
    Free,
    OneTime(f64),
    Subscription(f64, SubscriptionPeriod),
    UsageBased(f64, UsageUnit),
}

/// Subscription period
#[derive(Debug, Clone)]
pub enum SubscriptionPeriod {
    Monthly,
    Yearly,
}

/// Usage unit for pricing
#[derive(Debug, Clone)]
pub enum UsageUnit {
    PerPrediction,
    PerDataPoint,
    PerHour,
}

/// Support level offered
#[derive(Debug, Clone)]
pub enum SupportLevel {
    Community,
    Basic,
    Professional,
    Enterprise,
}

// =============================================================================
// Placeholder Implementations for Supporting Services
// =============================================================================

/// Plugin cache for local storage
#[derive(Debug, Clone)]
pub struct PluginCache;

impl Default for PluginCache {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginCache {
    pub fn new() -> Self {
        Self
    }

    pub fn get_repository_plugins(&self, _url: &str) -> Option<CachedPlugins> {
        None
    }

    pub fn store_repository_plugins(&self, _url: &str, _plugins: &[PluginManifest]) {}
}

/// Search index for fast plugin lookups
#[derive(Debug, Clone)]
pub struct SearchIndex;

impl Default for SearchIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchIndex {
    pub fn new() -> Self {
        Self
    }

    pub fn index_plugins(&self, _plugins: &[PluginManifest]) {}

    pub fn search(&self, _query: &SearchQuery) -> Result<Vec<PluginSearchResult>> {
        Ok(Vec::new())
    }
}

/// Network client for remote operations
#[derive(Debug, Clone)]
pub struct NetworkClient;

impl Default for NetworkClient {
    fn default() -> Self {
        Self::new()
    }
}

impl NetworkClient {
    pub fn new() -> Self {
        Self
    }

    pub async fn fetch_repository_plugins(
        &self,
        _repo: &PluginRepository,
    ) -> Result<Vec<PluginManifest>> {
        Ok(Vec::new())
    }

    pub async fn search_repository(
        &self,
        _repo: &PluginRepository,
        _query: &SearchQuery,
    ) -> Result<Vec<PluginSearchResult>> {
        Ok(Vec::new())
    }

    pub async fn download_plugin(&self, _manifest: &PluginManifest) -> Result<Vec<u8>> {
        Ok(Vec::new())
    }

    pub async fn get_plugin_manifest(
        &self,
        _repo: &PluginRepository,
        _id: &str,
        _version: Option<&str>,
    ) -> Result<PluginManifest> {
        Err(SklearsError::InvalidOperation("Not found".to_string()))
    }
}

/// Rating system for community feedback
#[derive(Debug, Clone)]
pub struct RatingSystem;

impl Default for RatingSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl RatingSystem {
    pub fn new() -> Self {
        Self
    }

    pub async fn get_average_rating(&self, _plugin_id: &str) -> Result<f32> {
        Ok(4.5)
    }

    pub async fn submit_rating(
        &self,
        _plugin_id: &str,
        _user_id: &str,
        _rating: f32,
    ) -> Result<()> {
        Ok(())
    }

    pub async fn get_rating_distribution(&self, _plugin_id: &str) -> Result<HashMap<u8, u64>> {
        Ok(HashMap::new())
    }

    pub async fn get_active_user_count(&self) -> Result<u64> {
        Ok(1000)
    }
}

/// Review system for detailed feedback
#[derive(Debug, Clone)]
pub struct ReviewSystem;

impl Default for ReviewSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl ReviewSystem {
    pub fn new() -> Self {
        Self
    }

    pub async fn get_review_count(&self, _plugin_id: &str) -> Result<u64> {
        Ok(100)
    }

    pub async fn submit_review(&self, _plugin_id: &str, _review: PluginReview) -> Result<()> {
        Ok(())
    }

    pub async fn get_reviews(
        &self,
        _plugin_id: &str,
        _offset: usize,
        _limit: usize,
    ) -> Result<Vec<PluginReview>> {
        Ok(Vec::new())
    }
}

/// Download tracking for popularity metrics
#[derive(Debug, Clone)]
pub struct DownloadTracker;

impl Default for DownloadTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl DownloadTracker {
    pub fn new() -> Self {
        Self
    }

    pub async fn get_download_count(&self, _plugin_id: &str) -> Result<u64> {
        Ok(1000)
    }

    pub async fn get_monthly_downloads(&self, _plugin_id: &str) -> Result<Vec<u64>> {
        Ok(vec![100, 120, 150, 180])
    }

    pub async fn get_recent_downloads(&self, _plugin_id: &str, _days: u32) -> Result<u64> {
        Ok(50)
    }

    pub async fn get_total_downloads(&self) -> Result<u64> {
        Ok(1000000)
    }
}

/// Analytics engine for trend analysis
#[derive(Debug, Clone)]
pub struct PluginAnalytics;

impl Default for PluginAnalytics {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginAnalytics {
    pub fn new() -> Self {
        Self
    }

    pub async fn track_rating_event(&self, _plugin_id: &str, _rating: f32) -> Result<()> {
        Ok(())
    }

    pub async fn track_review_event(&self, _plugin_id: &str) -> Result<()> {
        Ok(())
    }

    pub async fn get_trend_data(&self, _plugin_id: &str) -> Result<Vec<f32>> {
        Ok(vec![1.0, 1.2, 1.5, 1.8])
    }

    pub async fn get_trend_direction(&self, _plugin_id: &str) -> Result<TrendDirection> {
        Ok(TrendDirection::Stable)
    }

    pub async fn calculate_trend_score(&self, _plugin_id: &str) -> Result<f32> {
        Ok(0.7)
    }

    pub async fn get_download_velocity(&self, _plugin_id: &str) -> Result<f32> {
        Ok(5.0)
    }

    pub async fn get_last_update_time(&self, _plugin_id: &str) -> Result<std::time::SystemTime> {
        Ok(std::time::SystemTime::now())
    }

    pub async fn get_trending_categories(&self) -> Result<Vec<PluginCategory>> {
        Ok(vec![PluginCategory::Algorithm, PluginCategory::Transformer])
    }
}

/// Cached plugins with expiration
#[derive(Debug, Clone)]
pub struct CachedPlugins {
    pub plugins: Vec<PluginManifest>,
}

impl CachedPlugins {
    pub fn is_expired(&self) -> bool {
        false // Placeholder implementation
    }
}

/// Dummy plugin for validation testing
#[derive(Debug)]
pub struct DummyPlugin;

impl DummyPlugin {
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> Box<dyn Plugin> {
        Box::new(Self)
    }
}

impl Plugin for DummyPlugin {
    fn id(&self) -> &str {
        "dummy"
    }

    fn metadata(&self) -> PluginMetadata {
        PluginMetadata::default()
    }

    fn initialize(&mut self, _config: &super::types_config::PluginConfig) -> Result<()> {
        Ok(())
    }

    fn is_compatible(&self, _input_type: std::any::TypeId) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn validate_config(&self, _config: &super::types_config::PluginConfig) -> Result<()> {
        Ok(())
    }

    fn cleanup(&mut self) -> Result<()> {
        Ok(())
    }
}
