use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};

/// Pending notification in the queue
#[derive(Debug, Clone)]
pub struct PendingNotification {
    /// Notification ID
    pub notification_id: String,
    /// Target channel ID
    pub channel_id: String,
    /// Message content
    pub message: NotificationMessage,
    /// Priority level
    pub priority: NotificationPriority,
    /// Created timestamp
    pub created_at: SystemTime,
    /// Scheduled delivery time
    pub scheduled_delivery: SystemTime,
    /// Maximum delivery attempts
    pub max_attempts: u32,
    /// Current attempt count
    pub current_attempts: u32,
    /// Delivery context
    pub context: DeliveryContext,
    /// Dependencies
    pub dependencies: Vec<String>,
}

/// Notification message structure
#[derive(Debug, Clone)]
pub struct NotificationMessage {
    /// Message subject
    pub subject: String,
    /// Message body
    pub body: String,
    /// Message type
    pub message_type: MessageType,
    /// Attachments
    pub attachments: Vec<MessageAttachment>,
    /// Metadata
    pub metadata: MessageMetadata,
    /// Rich content
    pub rich_content: Option<RichContent>,
    /// Localization data
    pub localization_data: HashMap<String, String>,
}

/// Message types
#[derive(Debug, Clone)]
pub enum MessageType {
    Alert,
    Notification,
    Reminder,
    Update,
    Report,
    Custom(String),
}

/// Notification priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum NotificationPriority {
    Low,
    Normal,
    High,
    Critical,
    Emergency,
    Custom(u8),
}

/// Message attachment
#[derive(Debug, Clone)]
pub struct MessageAttachment {
    /// Attachment ID
    pub attachment_id: String,
    /// File name
    pub filename: String,
    /// Content type
    pub content_type: String,
    /// File size
    pub size: usize,
    /// Attachment data
    pub data: AttachmentData,
    /// Attachment metadata
    pub metadata: AttachmentMetadata,
}

/// Attachment data storage
#[derive(Debug, Clone)]
pub enum AttachmentData {
    /// In-memory data
    InMemory(Vec<u8>),
    /// File path reference
    FilePath(String),
    /// URL reference
    Url(String),
    /// External storage reference
    ExternalRef(String),
}

/// Attachment metadata
#[derive(Debug, Clone)]
pub struct AttachmentMetadata {
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Description
    pub description: Option<String>,
    /// Tags
    pub tags: Vec<String>,
    /// Encryption status
    pub encrypted: bool,
    /// Checksum
    pub checksum: Option<String>,
}

/// Message metadata
#[derive(Debug, Clone)]
pub struct MessageMetadata {
    /// Source system
    pub source: String,
    /// Correlation ID
    pub correlation_id: Option<String>,
    /// Thread ID
    pub thread_id: Option<String>,
    /// Reply to message ID
    pub reply_to: Option<String>,
    /// Message tags
    pub tags: Vec<String>,
    /// Custom fields
    pub custom_fields: HashMap<String, String>,
}

/// Rich content for messages
#[derive(Debug, Clone)]
pub struct RichContent {
    /// Content type
    pub content_type: RichContentType,
    /// Content data
    pub content_data: RichContentData,
    /// Rendering options
    pub rendering_options: RenderingOptions,
}

/// Rich content types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RichContentType {
    Images,
    Videos,
    Attachments,
    Embeds,
    Cards,
    Buttons,
    Custom(String),
}

/// Rich content data
#[derive(Debug, Clone)]
pub enum RichContentData {
    /// Card content
    Card(CardContent),
    /// Embed content
    Embed(EmbedContent),
    /// Button content
    Buttons(Vec<ButtonContent>),
    /// Custom content
    Custom(HashMap<String, String>),
}

/// Card content
#[derive(Debug, Clone)]
pub struct CardContent {
    /// Card title
    pub title: String,
    /// Card description
    pub description: String,
    /// Card image URL
    pub image_url: Option<String>,
    /// Card actions
    pub actions: Vec<CardAction>,
    /// Card color
    pub color: Option<String>,
}

/// Card action
#[derive(Debug, Clone)]
pub struct CardAction {
    /// Action type
    pub action_type: ActionType,
    /// Action label
    pub label: String,
    /// Action URL or command
    pub action: String,
    /// Action style
    pub style: ActionStyle,
}

/// Action types
#[derive(Debug, Clone)]
pub enum ActionType {
    Button,
    Link,
    Command,
    Custom(String),
}

/// Action styles
#[derive(Debug, Clone)]
pub enum ActionStyle {
    Primary,
    Secondary,
    Success,
    Warning,
    Danger,
    Custom(String),
}

/// Embed content
#[derive(Debug, Clone)]
pub struct EmbedContent {
    /// Embed URL
    pub url: String,
    /// Embed title
    pub title: Option<String>,
    /// Embed description
    pub description: Option<String>,
    /// Embed thumbnail
    pub thumbnail: Option<String>,
    /// Embed fields
    pub fields: Vec<EmbedField>,
}

/// Embed field
#[derive(Debug, Clone)]
pub struct EmbedField {
    /// Field name
    pub name: String,
    /// Field value
    pub value: String,
    /// Inline display
    pub inline: bool,
}

/// Button content
#[derive(Debug, Clone)]
pub struct ButtonContent {
    /// Button ID
    pub button_id: String,
    /// Button text
    pub text: String,
    /// Button action
    pub action: String,
    /// Button style
    pub style: ActionStyle,
    /// Button metadata
    pub metadata: HashMap<String, String>,
}

/// Rendering options for rich content
#[derive(Debug, Clone)]
pub struct RenderingOptions {
    /// Theme
    pub theme: Option<String>,
    /// Custom CSS
    pub custom_css: Option<String>,
    /// Layout options
    pub layout_options: HashMap<String, String>,
}

/// Delivery context for notifications
#[derive(Debug, Clone)]
pub struct DeliveryContext {
    /// Original alert ID
    pub alert_id: String,
    /// Escalation level
    pub escalation_level: u32,
    /// Retry context
    pub retry_context: RetryContext,
    /// Delivery preferences
    pub delivery_preferences: DeliveryPreferences,
    /// Tracking information
    pub tracking_info: TrackingInfo,
}

/// Retry context for delivery attempts
#[derive(Debug, Clone)]
pub struct RetryContext {
    /// Previous attempts
    pub previous_attempts: Vec<DeliveryAttempt>,
    /// Next retry time
    pub next_retry_time: Option<SystemTime>,
    /// Retry strategy
    pub retry_strategy: RetryStrategy,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
}

/// Delivery attempt record
#[derive(Debug, Clone)]
pub struct DeliveryAttempt {
    /// Attempt number
    pub attempt_number: u32,
    /// Attempt timestamp
    pub timestamp: SystemTime,
    /// Success status
    pub success: bool,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Response time
    pub response_time: Duration,
    /// HTTP status code if applicable
    pub status_code: Option<u16>,
}

/// Retry strategies
#[derive(Debug, Clone)]
pub enum RetryStrategy {
    Fixed,
    Linear,
    Exponential,
    Fibonacci,
    Custom(String),
}

/// Delivery preferences
#[derive(Debug, Clone)]
pub struct DeliveryPreferences {
    /// Preferred delivery time windows
    pub time_windows: Vec<TimeWindow>,
    /// Blocked time windows
    pub blocked_windows: Vec<TimeWindow>,
    /// Preferred channels in order
    pub preferred_channels: Vec<String>,
    /// Delivery consolidation
    pub consolidation_rules: Vec<ConsolidationRule>,
}

/// Time window for delivery preferences
#[derive(Debug, Clone)]
pub struct TimeWindow {
    /// Start time (hour of day)
    pub start_hour: u8,
    /// End time (hour of day)
    pub end_hour: u8,
    /// Days of week (0=Sunday)
    pub days_of_week: Vec<u8>,
    /// Timezone
    pub timezone: String,
}

/// Consolidation rules for message grouping
#[derive(Debug, Clone)]
pub struct ConsolidationRule {
    /// Rule ID
    pub rule_id: String,
    /// Consolidation type
    pub consolidation_type: ConsolidationType,
    /// Time window for consolidation
    pub time_window: Duration,
    /// Maximum messages to consolidate
    pub max_messages: usize,
    /// Consolidation template
    pub template: String,
}

/// Consolidation types
#[derive(Debug, Clone)]
pub enum ConsolidationType {
    /// Consolidate by alert type
    ByAlertType,
    /// Consolidate by severity
    BySeverity,
    /// Consolidate by source
    BySource,
    /// Consolidate all
    All,
    /// Custom consolidation logic
    Custom(String),
}

/// Tracking information for notifications
#[derive(Debug, Clone)]
pub struct TrackingInfo {
    /// Tracking ID
    pub tracking_id: String,
    /// External tracking ID
    pub external_tracking_id: Option<String>,
    /// Delivery path
    pub delivery_path: Vec<String>,
    /// Delivery events
    pub delivery_events: Vec<DeliveryEvent>,
    /// Performance metrics
    pub performance_metrics: DeliveryPerformanceMetrics,
}

/// Delivery event
#[derive(Debug, Clone)]
pub struct DeliveryEvent {
    /// Event type
    pub event_type: DeliveryEventType,
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Event description
    pub description: String,
    /// Event metadata
    pub metadata: HashMap<String, String>,
}

/// Delivery event types
#[derive(Debug, Clone)]
pub enum DeliveryEventType {
    Queued,
    Processing,
    Sent,
    Delivered,
    Failed,
    Acknowledged,
    Clicked,
    Custom(String),
}

/// Performance metrics for delivery
#[derive(Debug, Clone)]
pub struct DeliveryPerformanceMetrics {
    /// Queue time
    pub queue_time: Duration,
    /// Processing time
    pub processing_time: Duration,
    /// Network time
    pub network_time: Duration,
    /// Total delivery time
    pub total_delivery_time: Duration,
    /// Retry count
    pub retry_count: u32,
}

/// Message formatting configuration with rich features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageFormat {
    /// Message template
    pub template: String,
    /// Template engine
    pub template_engine: TemplateEngine,
    /// Template variables
    pub variables: HashMap<String, String>,
    /// Formatting options
    pub formatting: FormattingOptions,
    /// Localization settings
    pub localization: LocalizationConfig,
    /// Rich content support
    pub rich_content: RichContentConfig,
}

/// Template engines
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemplateEngine {
    Handlebars,
    Jinja2,
    Mustache,
    Liquid,
    Simple,
    Custom(String),
}

/// Formatting options for messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormattingOptions {
    /// Include timestamps
    pub include_timestamp: bool,
    /// Timestamp format
    pub timestamp_format: String,
    /// Include severity in subject
    pub include_severity: bool,
    /// Include node information
    pub include_node_info: bool,
    /// Include metric values
    pub include_metrics: bool,
    /// Custom formatting rules
    pub custom_rules: Vec<FormattingRule>,
    /// Message length limits
    pub length_limits: MessageLengthLimits,
    /// Content encoding
    pub content_encoding: ContentEncoding,
}

/// Custom formatting rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormattingRule {
    /// Condition for applying rule
    pub condition: String,
    /// Formatting template
    pub template: String,
    /// Priority order
    pub priority: u32,
    /// Rule scope
    pub scope: FormattingScope,
}

/// Formatting rule scope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FormattingScope {
    Subject,
    Body,
    Both,
    Custom(String),
}

/// Message length limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageLengthLimits {
    /// Maximum subject length
    pub max_subject_length: usize,
    /// Maximum body length
    pub max_body_length: usize,
    /// Truncation strategy
    pub truncation_strategy: TruncationStrategy,
    /// Truncation indicator
    pub truncation_indicator: String,
}

/// Truncation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TruncationStrategy {
    End,
    Middle,
    Beginning,
    Smart,
    None,
}

/// Content encoding options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentEncoding {
    PlainText,
    HTML,
    Markdown,
    JSON,
    XML,
    Custom(String),
}

/// Localization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalizationConfig {
    /// Enable localization
    pub enabled: bool,
    /// Default locale
    pub default_locale: String,
    /// Supported locales
    pub supported_locales: Vec<String>,
    /// Locale detection method
    pub locale_detection: LocaleDetection,
    /// Translation provider
    pub translation_provider: TranslationProvider,
}

/// Locale detection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LocaleDetection {
    UserPreference,
    ChannelDefault,
    SystemDefault,
    GeoLocation,
    Custom(String),
}

/// Translation providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TranslationProvider {
    Internal,
    GoogleTranslate,
    AzureTranslator,
    AWSTranslate,
    Custom(String),
}

/// Rich content configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RichContentConfig {
    /// Enable rich content
    pub enabled: bool,
    /// Supported content types
    pub supported_types: Vec<RichContentType>,
    /// Attachment configuration
    pub attachment_config: AttachmentConfig,
    /// Embed configuration
    pub embed_config: EmbedConfig,
}

/// Attachment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttachmentConfig {
    /// Maximum attachment size
    pub max_size: usize,
    /// Allowed file types
    pub allowed_types: Vec<String>,
    /// Enable virus scanning
    pub virus_scanning: bool,
    /// Storage configuration
    pub storage_config: AttachmentStorageConfig,
}

/// Attachment storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttachmentStorageConfig {
    /// Storage type
    pub storage_type: AttachmentStorageType,
    /// Storage path or URL
    pub storage_path: String,
    /// Access credentials
    pub credentials: Option<String>,
    /// Encryption settings
    pub encryption: Option<AttachmentEncryption>,
}

/// Attachment storage types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttachmentStorageType {
    LocalFilesystem,
    S3,
    AzureBlob,
    GoogleCloud,
    Custom(String),
}

/// Attachment encryption settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttachmentEncryption {
    /// Encryption algorithm
    pub algorithm: String,
    /// Key management
    pub key_management: String,
    /// Encryption at rest
    pub at_rest: bool,
    /// Encryption in transit
    pub in_transit: bool,
}

/// Embed configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedConfig {
    /// Enable embeds
    pub enabled: bool,
    /// Maximum embed size
    pub max_size: usize,
    /// Allowed domains
    pub allowed_domains: Vec<String>,
    /// Timeout for embed fetching
    pub fetch_timeout: Duration,
}

/// Message formatter for notification content
#[derive(Debug)]
pub struct MessageFormatter {
    /// Template engine
    pub template_engine: TemplateEngineInstance,
    /// Formatter configuration
    pub config: MessageFormatterConfig,
    /// Template cache
    pub template_cache: Arc<RwLock<TemplateCache>>,
    /// Formatting statistics
    pub statistics: Arc<RwLock<FormattingStatistics>>,
}

/// Template engine instance
#[derive(Debug)]
pub enum TemplateEngineInstance {
    Handlebars(HandlebarsEngine),
    Simple(SimpleEngine),
    Custom(Box<dyn CustomTemplateEngine>),
}

/// Handlebars template engine
#[derive(Debug)]
pub struct HandlebarsEngine {
    /// Handlebars registry
    pub registry: handlebars::Handlebars<'static>,
    /// Helper functions
    pub helpers: HashMap<String, Box<dyn handlebars::HelperDef>>,
}

/// Simple template engine
#[derive(Debug)]
pub struct SimpleEngine {
    /// Variable pattern
    pub variable_pattern: String,
    /// Escape characters
    pub escape_chars: bool,
}

/// Custom template engine trait
pub trait CustomTemplateEngine: std::fmt::Debug + Send + Sync {
    fn render(&self, template: &str, context: &HashMap<String, String>) -> Result<String, String>;
    fn validate_template(&self, template: &str) -> Result<(), String>;
}

/// Message formatter configuration
#[derive(Debug, Clone)]
pub struct MessageFormatterConfig {
    /// Default template engine
    pub default_engine: TemplateEngine,
    /// Template validation
    pub template_validation: bool,
    /// Output sanitization
    pub output_sanitization: bool,
    /// Maximum output length
    pub max_output_length: usize,
    /// Error handling strategy
    pub error_handling: FormatterErrorHandling,
}

/// Formatter error handling strategies
#[derive(Debug, Clone)]
pub enum FormatterErrorHandling {
    Strict,
    Fallback,
    BestEffort,
    Custom(String),
}

/// Template cache for performance
#[derive(Debug)]
pub struct TemplateCache {
    /// Cached templates
    pub templates: HashMap<String, CachedTemplate>,
    /// Cache configuration
    pub config: TemplateCacheConfig,
    /// Cache statistics
    pub statistics: TemplateCacheStatistics,
}

/// Cached template
#[derive(Debug, Clone)]
pub struct CachedTemplate {
    /// Template content
    pub content: String,
    /// Compiled template
    pub compiled: Option<String>,
    /// Cache timestamp
    pub cached_at: SystemTime,
    /// Access count
    pub access_count: u64,
    /// Last accessed
    pub last_accessed: SystemTime,
}

/// Template cache configuration
#[derive(Debug, Clone)]
pub struct TemplateCacheConfig {
    /// Cache size limit
    pub max_cache_size: usize,
    /// Cache TTL
    pub cache_ttl: Duration,
    /// Enable LRU eviction
    pub enable_lru_eviction: bool,
    /// Cache statistics tracking
    pub track_statistics: bool,
}

/// Template cache statistics
#[derive(Debug, Clone)]
pub struct TemplateCacheStatistics {
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Cache evictions
    pub cache_evictions: u64,
    /// Cache size
    pub current_cache_size: usize,
    /// Hit ratio
    pub hit_ratio: f64,
}

/// Formatting statistics
#[derive(Debug, Clone)]
pub struct FormattingStatistics {
    /// Total formatting operations
    pub total_operations: u64,
    /// Successful operations
    pub successful_operations: u64,
    /// Failed operations
    pub failed_operations: u64,
    /// Average formatting time
    pub average_formatting_time: Duration,
    /// Template usage statistics
    pub template_usage: HashMap<String, TemplateUsageStats>,
}

/// Template usage statistics
#[derive(Debug, Clone)]
pub struct TemplateUsageStats {
    /// Usage count
    pub usage_count: u64,
    /// Success rate
    pub success_rate: f64,
    /// Average render time
    pub average_render_time: Duration,
    /// Error messages
    pub error_messages: Vec<String>,
}

/// Content filtering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentFilterConfig {
    /// Enable content filtering
    pub enabled: bool,
    /// Blocked keywords
    pub blocked_keywords: Vec<String>,
    /// Required keywords
    pub required_keywords: Vec<String>,
    /// Maximum message length
    pub max_message_length: usize,
    /// Content validation rules
    pub validation_rules: Vec<ContentValidationRule>,
}

/// Content validation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentValidationRule {
    /// Rule name
    pub name: String,
    /// Rule pattern (regex)
    pub pattern: String,
    /// Action when rule fails
    pub action: ValidationAction,
    /// Rule severity
    pub severity: ValidationSeverity,
}

/// Validation actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationAction {
    Allow,
    Block,
    Modify,
    Warn,
    Escalate,
}

/// Validation severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl PendingNotification {
    /// Create a new pending notification
    pub fn new(
        notification_id: String,
        channel_id: String,
        message: NotificationMessage,
        priority: NotificationPriority,
    ) -> Self {
        let now = SystemTime::now();
        Self {
            notification_id,
            channel_id,
            message,
            priority,
            created_at: now,
            scheduled_delivery: now,
            max_attempts: 3,
            current_attempts: 0,
            context: DeliveryContext::default(),
            dependencies: Vec::new(),
        }
    }

    /// Check if notification has expired
    pub fn is_expired(&self, max_age: Duration) -> bool {
        if let Ok(age) = SystemTime::now().duration_since(self.created_at) {
            age > max_age
        } else {
            false
        }
    }

    /// Get retry delay based on attempt count
    pub fn get_retry_delay(&self) -> Duration {
        let base_delay = Duration::from_secs(30);
        let multiplier = 2_u32.pow(self.current_attempts);
        base_delay * multiplier.min(60) // Cap at 30 minutes
    }
}

impl Default for DeliveryContext {
    fn default() -> Self {
        Self {
            alert_id: String::new(),
            escalation_level: 0,
            retry_context: RetryContext::default(),
            delivery_preferences: DeliveryPreferences::default(),
            tracking_info: TrackingInfo::default(),
        }
    }
}

impl Default for RetryContext {
    fn default() -> Self {
        Self {
            previous_attempts: Vec::new(),
            next_retry_time: None,
            retry_strategy: RetryStrategy::Exponential,
            backoff_multiplier: 2.0,
        }
    }
}

impl Default for DeliveryPreferences {
    fn default() -> Self {
        Self {
            time_windows: Vec::new(),
            blocked_windows: Vec::new(),
            preferred_channels: Vec::new(),
            consolidation_rules: Vec::new(),
        }
    }
}

impl Default for TrackingInfo {
    fn default() -> Self {
        Self {
            tracking_id: uuid::Uuid::new_v4().to_string(),
            external_tracking_id: None,
            delivery_path: Vec::new(),
            delivery_events: Vec::new(),
            performance_metrics: DeliveryPerformanceMetrics::default(),
        }
    }
}

impl Default for DeliveryPerformanceMetrics {
    fn default() -> Self {
        Self {
            queue_time: Duration::from_secs(0),
            processing_time: Duration::from_secs(0),
            network_time: Duration::from_secs(0),
            total_delivery_time: Duration::from_secs(0),
            retry_count: 0,
        }
    }
}

impl MessageFormatter {
    /// Create a new message formatter
    pub fn new(
        template_engine: TemplateEngineInstance,
        config: MessageFormatterConfig,
    ) -> Self {
        Self {
            template_engine,
            config,
            template_cache: Arc::new(RwLock::new(TemplateCache::new())),
            statistics: Arc::new(RwLock::new(FormattingStatistics::default())),
        }
    }

    /// Format a message using the template engine
    pub fn format_message(
        &self,
        template: &str,
        context: &HashMap<String, String>,
    ) -> Result<String, String> {
        let start_time = SystemTime::now();
        let result = match &self.template_engine {
            TemplateEngineInstance::Handlebars(engine) => {
                engine.registry.render_template(template, context)
                    .map_err(|e| e.to_string())
            },
            TemplateEngineInstance::Simple(engine) => {
                self.simple_template_render(template, context, engine)
            },
            TemplateEngineInstance::Custom(engine) => {
                engine.render(template, context)
            },
        };

        // Update statistics
        if let Ok(mut stats) = self.statistics.write() {
            stats.total_operations += 1;
            if result.is_ok() {
                stats.successful_operations += 1;
            } else {
                stats.failed_operations += 1;
            }

            if let Ok(duration) = SystemTime::now().duration_since(start_time) {
                let total_time = stats.average_formatting_time.as_nanos() * stats.total_operations as u128;
                stats.average_formatting_time = Duration::from_nanos(
                    ((total_time + duration.as_nanos()) / (stats.total_operations as u128 + 1)) as u64
                );
            }
        }

        result
    }

    fn simple_template_render(
        &self,
        template: &str,
        context: &HashMap<String, String>,
        engine: &SimpleEngine,
    ) -> Result<String, String> {
        let mut result = template.to_string();

        for (key, value) in context {
            let pattern = format!("{{{}}}", key);
            result = result.replace(&pattern, value);
        }

        if engine.escape_chars {
            result = html_escape::encode_text(&result).to_string();
        }

        Ok(result)
    }
}

impl TemplateCache {
    /// Create a new template cache
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
            config: TemplateCacheConfig::default(),
            statistics: TemplateCacheStatistics::default(),
        }
    }

    /// Get template from cache
    pub fn get(&mut self, template_id: &str) -> Option<&CachedTemplate> {
        if let Some(template) = self.templates.get_mut(template_id) {
            template.access_count += 1;
            template.last_accessed = SystemTime::now();
            self.statistics.cache_hits += 1;
            Some(template)
        } else {
            self.statistics.cache_misses += 1;
            None
        }
    }

    /// Put template in cache
    pub fn put(&mut self, template_id: String, content: String) {
        let cached_template = CachedTemplate {
            content,
            compiled: None,
            cached_at: SystemTime::now(),
            access_count: 0,
            last_accessed: SystemTime::now(),
        };

        self.templates.insert(template_id, cached_template);
        self.statistics.current_cache_size = self.templates.len();

        // Evict if necessary
        if self.templates.len() > self.config.max_cache_size {
            self.evict_lru();
        }
    }

    fn evict_lru(&mut self) {
        if let Some((oldest_key, _)) = self.templates
            .iter()
            .min_by_key(|(_, template)| template.last_accessed)
            .map(|(k, v)| (k.clone(), v.clone()))
        {
            self.templates.remove(&oldest_key);
            self.statistics.cache_evictions += 1;
            self.statistics.current_cache_size = self.templates.len();
        }
    }
}

impl Default for TemplateCacheConfig {
    fn default() -> Self {
        Self {
            max_cache_size: 1000,
            cache_ttl: Duration::from_secs(3600),
            enable_lru_eviction: true,
            track_statistics: true,
        }
    }
}

impl Default for TemplateCacheStatistics {
    fn default() -> Self {
        Self {
            cache_hits: 0,
            cache_misses: 0,
            cache_evictions: 0,
            current_cache_size: 0,
            hit_ratio: 0.0,
        }
    }
}

impl Default for FormattingStatistics {
    fn default() -> Self {
        Self {
            total_operations: 0,
            successful_operations: 0,
            failed_operations: 0,
            average_formatting_time: Duration::from_secs(0),
            template_usage: HashMap::new(),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pending_notification_creation() {
        let message = NotificationMessage {
            subject: "Test Alert".to_string(),
            body: "This is a test message".to_string(),
            message_type: MessageType::Alert,
            attachments: Vec::new(),
            metadata: MessageMetadata {
                source: "test_system".to_string(),
                correlation_id: None,
                thread_id: None,
                reply_to: None,
                tags: Vec::new(),
                custom_fields: HashMap::new(),
            },
            rich_content: None,
            localization_data: HashMap::new(),
        };

        let notification = PendingNotification::new(
            "test-123".to_string(),
            "email-channel".to_string(),
            message,
            NotificationPriority::High,
        );

        assert_eq!(notification.notification_id, "test-123");
        assert_eq!(notification.channel_id, "email-channel");
        assert_eq!(notification.priority, NotificationPriority::High);
        assert_eq!(notification.current_attempts, 0);
        assert_eq!(notification.max_attempts, 3);
    }

    #[test]
    fn test_notification_priority_ordering() {
        assert!(NotificationPriority::Emergency > NotificationPriority::Critical);
        assert!(NotificationPriority::Critical > NotificationPriority::High);
        assert!(NotificationPriority::High > NotificationPriority::Normal);
        assert!(NotificationPriority::Normal > NotificationPriority::Low);
    }

    #[test]
    fn test_retry_delay_calculation() {
        let mut notification = PendingNotification::new(
            "test".to_string(),
            "channel".to_string(),
            NotificationMessage {
                subject: "Test".to_string(),
                body: "Test".to_string(),
                message_type: MessageType::Alert,
                attachments: Vec::new(),
                metadata: MessageMetadata {
                    source: "test".to_string(),
                    correlation_id: None,
                    thread_id: None,
                    reply_to: None,
                    tags: Vec::new(),
                    custom_fields: HashMap::new(),
                },
                rich_content: None,
                localization_data: HashMap::new(),
            },
            NotificationPriority::Normal,
        );

        assert_eq!(notification.get_retry_delay(), Duration::from_secs(30));

        notification.current_attempts = 1;
        assert_eq!(notification.get_retry_delay(), Duration::from_secs(60));

        notification.current_attempts = 2;
        assert_eq!(notification.get_retry_delay(), Duration::from_secs(120));
    }

    #[test]
    fn test_message_formatter_simple_template() {
        let engine = SimpleEngine {
            variable_pattern: "{}".to_string(),
            escape_chars: false,
        };

        let config = MessageFormatterConfig {
            default_engine: TemplateEngine::Simple,
            template_validation: false,
            output_sanitization: false,
            max_output_length: 1000,
            error_handling: FormatterErrorHandling::BestEffort,
        };

        let formatter = MessageFormatter::new(
            TemplateEngineInstance::Simple(engine),
            config,
        );

        let mut context = HashMap::new();
        context.insert("name".to_string(), "John".to_string());
        context.insert("action".to_string(), "logged in".to_string());

        let result = formatter.format_message(
            "User {name} has {action}",
            &context,
        ).unwrap();

        assert_eq!(result, "User John has logged in");
    }

    #[test]
    fn test_template_cache_operations() {
        let mut cache = TemplateCache::new();

        // Test cache miss
        assert!(cache.get("template1").is_none());
        assert_eq!(cache.statistics.cache_misses, 1);

        // Add template to cache
        cache.put("template1".to_string(), "Hello {name}".to_string());
        assert_eq!(cache.statistics.current_cache_size, 1);

        // Test cache hit
        let template = cache.get("template1");
        assert!(template.is_some());
        assert_eq!(cache.statistics.cache_hits, 1);
        assert_eq!(template.unwrap().access_count, 1);
    }
}