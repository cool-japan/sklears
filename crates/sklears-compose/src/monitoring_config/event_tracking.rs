//! Event Tracking Configuration
//!
//! This module contains all configuration structures related to event tracking,
//! enrichment, collection, and processing. Events provide detailed operational
//! insights and audit trails for the monitoring system.

use std::collections::HashMap;
use std::time::Duration;

/// Event tracking configuration
///
/// Controls all aspects of event tracking including what events to capture,
/// how to enrich them with contextual information, and how to process and store
/// the collected event data.
///
/// # Architecture
///
/// The event tracking system follows a pipeline architecture:
///
/// ```text
/// Event Source → Collection → Enrichment → Filtering → Storage
///      ↓              ↓           ↓           ↓         ↓
///   Raw Events   Collection   Enriched    Filtered   Stored
///                 Config      Events      Events     Events
/// ```
///
/// # Event Types
///
/// Events are categorized by type to enable selective tracking and processing:
/// - System events: Infrastructure and platform events
/// - Application events: Business logic and user actions
/// - Security events: Authentication, authorization, and audit events
/// - Performance events: Timing, resource usage, and optimization events
///
/// # Usage Examples
///
/// ## Basic Event Tracking
/// ```rust
/// use sklears_compose::monitoring_config::EventTrackingConfig;
///
/// let config = EventTrackingConfig::default();
/// ```
///
/// ## Security-Focused Configuration
/// ```rust
/// let config = EventTrackingConfig::security_focused();
/// ```
///
/// ## High-Volume Configuration
/// ```rust
/// let config = EventTrackingConfig::high_volume();
/// ```
#[derive(Debug, Clone)]
pub struct EventTrackingConfig {
    /// Enable event tracking
    ///
    /// Global switch to enable or disable all event tracking.
    /// When disabled, no events will be captured, providing a clean
    /// way to turn off event tracking for performance or privacy reasons.
    pub enabled: bool,

    /// Types of events to track
    ///
    /// Defines which categories of events should be captured.
    /// This allows selective tracking to focus on relevant events
    /// and reduce storage and processing overhead.
    pub event_types: Vec<EventType>,

    /// Event collection configuration
    ///
    /// Controls how events are collected from various sources,
    /// including collection methods, buffering, and reliability settings.
    pub collection: EventCollectionConfig,

    /// Event enrichment configuration
    ///
    /// Defines how events are enhanced with additional contextual information
    /// to make them more useful for analysis and debugging.
    pub enrichment: EventEnrichmentConfig,

    /// Event filtering configuration
    ///
    /// Controls which events are kept, discarded, or transformed
    /// based on content, frequency, or other criteria.
    pub filtering: EventFilteringConfig,

    /// Sampling configuration for high-volume events
    ///
    /// Controls sampling strategies to manage event volume
    /// in high-throughput scenarios while maintaining representativeness.
    pub sampling: EventSamplingConfig,
}

/// Types of events to track
///
/// Provides a comprehensive categorization of events that can occur
/// in the system. Each event type has specific characteristics and
/// use cases for monitoring and analysis.
#[derive(Debug, Clone, PartialEq)]
pub enum EventType {
    System,

    Application,

    User,

    Security,

    Performance,

    Error,

    DataProcessing,

    Network,

    Resource,

    Audit,

    Custom {
        name: String,
        description: String,
        critical: bool,
    },
}

/// Event collection configuration
///
/// Controls how events are gathered from various sources including
/// buffering strategies, reliability settings, and performance tuning.
#[derive(Debug, Clone)]
pub struct EventCollectionConfig {
    /// Event collection method
    ///
    /// Defines how events are collected from the application.
    pub method: CollectionMethod,

    /// Buffer size for event collection
    ///
    /// Number of events to buffer before processing.
    /// Larger buffers improve efficiency but increase memory usage.
    pub buffer_size: usize,

    /// Maximum buffer size before forced flush
    ///
    /// Prevents unlimited memory growth by forcing event processing
    /// when the buffer reaches this size.
    pub max_buffer_size: usize,

    /// Flush interval for buffered events
    ///
    /// How often to process buffered events even if buffer isn't full.
    /// More frequent flushes reduce latency but increase overhead.
    pub flush_interval: Duration,

    /// Enable reliable event delivery
    ///
    /// When enabled, uses acknowledgments and retries to ensure
    /// events are not lost during collection and processing.
    pub reliable_delivery: bool,

    /// Maximum retries for failed event delivery
    ///
    /// Number of times to retry delivering events before giving up.
    pub max_retries: u32,

    /// Timeout for event collection operations
    ///
    /// Maximum time to wait for event collection operations.
    pub timeout: Duration,
}

/// Event collection methods
///
/// Defines different strategies for collecting events from applications
/// and systems, each with different performance and reliability characteristics.
#[derive(Debug, Clone)]
pub enum CollectionMethod {
    /// Push-based collection
    ///
    /// Events are actively sent to the collection system.
    /// Provides low latency but requires active integration.
    Push {
        /// Endpoint for receiving events
        endpoint: String,
        /// Maximum concurrent connections
        max_connections: usize,
    },

    /// Pull-based collection
    ///
    /// Collection system actively retrieves events from sources.
    /// Provides better control but may have higher latency.
    Pull {
        /// Sources to poll for events
        sources: Vec<String>,
        /// Polling interval
        poll_interval: Duration,
    },

    /// File-based collection
    ///
    /// Events are written to files and collected by monitoring.
    /// Provides durability but may have higher latency.
    File {
        /// Directory to monitor for event files
        watch_directory: String,
        /// File pattern to match
        file_pattern: String,
    },

    /// Message queue collection
    ///
    /// Events are sent through a message queue system.
    /// Provides reliability and decoupling.
    MessageQueue {
        /// Queue connection string
        connection: String,
        /// Queue or topic name
        queue_name: String,
    },
}

/// Event enrichment configuration
///
/// Controls how events are enhanced with additional contextual information
/// to make them more valuable for analysis, debugging, and correlation.
#[derive(Debug, Clone)]
pub struct EventEnrichmentConfig {
    /// Enable event enrichment
    ///
    /// Global switch for event enrichment functionality.
    pub enabled: bool,

    /// Enrichment rules to apply
    ///
    /// List of rules that define how to enrich different types of events
    /// with additional contextual information.
    pub rules: Vec<EnrichmentRule>,

    /// Context sources for enrichment
    ///
    /// External sources of contextual information that can be used
    /// to enrich events with additional data.
    pub context_sources: Vec<ContextSource>,

    /// Maximum enrichment processing time
    ///
    /// Maximum time to spend enriching a single event.
    /// Prevents enrichment from causing unacceptable delays.
    pub max_processing_time: Duration,

    /// Enable caching of enrichment data
    ///
    /// Caches enrichment data to improve performance for frequently
    /// accessed contextual information.
    pub enable_caching: bool,

    /// Cache TTL for enrichment data
    ///
    /// How long to cache enrichment data before refreshing.
    pub cache_ttl: Duration,
}

/// Enrichment rules for events
///
/// Defines how specific types of events should be enhanced with
/// additional contextual information to make them more useful.
#[derive(Debug, Clone)]
pub struct EnrichmentRule {
    /// Rule name for identification
    pub name: String,

    /// Event types this rule applies to
    ///
    /// List of event types that should be processed by this rule.
    pub target_events: Vec<EventType>,

    /// Enrichment source to use
    ///
    /// Where to get the additional information for enrichment.
    pub source: EnrichmentSource,

    /// Field mappings for enrichment
    ///
    /// Maps fields from the enrichment source to fields in the event.
    /// Key is the target field name, value is the source field name.
    pub field_mappings: HashMap<String, String>,

    /// Conditions for applying this rule
    ///
    /// Optional conditions that must be met for this rule to be applied.
    pub conditions: Vec<EnrichmentCondition>,

    /// Rule priority for ordering
    ///
    /// When multiple rules apply, they are processed in priority order.
    /// Higher numbers indicate higher priority.
    pub priority: u32,
}

/// Sources of enrichment data
///
/// Defines where additional contextual information can be obtained
/// to enhance events with more useful data.
#[derive(Debug, Clone)]
pub enum EnrichmentSource {
    Static {
        mappings: HashMap<String, String>
    },

    /// Database lookup
    ///
    /// Query a database for enrichment information.
    Database {
        /// Database connection string
        connection: String,
        /// SQL query template
        query: String,
    },

    /// API call for enrichment
    ///
    /// Call an external API to get enrichment data.
    Api {
        /// API endpoint URL
        url: String,
        /// HTTP method to use
        method: String,
        /// Request headers
        headers: HashMap<String, String>,
    },

    /// File-based lookup
    ///
    /// Load enrichment data from a file.
    File {
        path: String,
        format: String,
    },

    /// Context from the system environment
    ///
    /// Use system information like hostname, environment variables, etc.
    SystemContext,
}

/// Context sources for event enrichment
///
/// External sources that provide contextual information for enriching events.
#[derive(Debug, Clone)]
pub struct ContextSource {
    /// Source name for identification
    pub name: String,

    /// Source type and configuration
    pub source_type: EnrichmentSource,

    /// Refresh interval for dynamic sources
    pub refresh_interval: Option<Duration>,

    /// Whether this source is required for enrichment
    pub required: bool,
}

/// Conditions for enrichment rule application
///
/// Defines criteria that must be met for an enrichment rule to be applied.
#[derive(Debug, Clone)]
pub struct EnrichmentCondition {
    /// Field name to check
    pub field: String,

    /// Comparison operator
    pub operator: ConditionOperator,

    /// Value to compare against
    pub value: String,
}

/// Operators for enrichment conditions
#[derive(Debug, Clone)]
pub enum ConditionOperator {
    /// Exact equality
    Equals,
    /// Field contains the value
    Contains,
    /// Field starts with the value
    StartsWith,
    /// Field ends with the value
    EndsWith,
    /// Regular expression match
    Regex,
    /// Field exists (non-empty)
    Exists,
    /// Field does not exist or is empty
    NotExists,
}

/// Event filtering configuration
///
/// Controls which events are kept, discarded, or transformed based on
/// content, frequency, or other criteria to manage event volume and relevance.
#[derive(Debug, Clone)]
pub struct EventFilteringConfig {
    /// Enable event filtering
    pub enabled: bool,

    /// Filtering rules to apply
    pub rules: Vec<FilterRule>,

    /// Default action when no rules match
    pub default_action: FilterAction,

    /// Enable rate limiting for high-frequency events
    pub rate_limiting: bool,

    /// Rate limit configuration
    pub rate_limits: HashMap<String, RateLimit>,
}

/// Event filtering rules
#[derive(Debug, Clone)]
pub struct FilterRule {
    /// Rule name
    pub name: String,

    /// Conditions that must be met
    pub conditions: Vec<FilterCondition>,

    /// Action to take when conditions match
    pub action: FilterAction,

    /// Rule priority
    pub priority: u32,
}

/// Conditions for event filtering
#[derive(Debug, Clone)]
pub struct FilterCondition {
    /// Field to evaluate
    pub field: String,

    /// Comparison operator
    pub operator: ConditionOperator,

    /// Value to compare against
    pub value: String,
}

/// Actions for event filtering
#[derive(Debug, Clone)]
pub enum FilterAction {
    /// Keep the event
    Keep,
    /// Discard the event
    Discard,
    /// Transform the event
    Transform {
        /// Transformation rules
        transformations: HashMap<String, String>
    },
    /// Sample the event (keep with probability)
    Sample {
        /// Sampling rate (0.0 to 1.0)
        rate: f64
    },
}

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimit {
    /// Maximum events per time window
    pub max_events: u32,

    /// Time window for rate limiting
    pub window: Duration,

    /// Action when rate limit is exceeded
    pub action: FilterAction,
}

/// Event sampling configuration
///
/// Controls sampling strategies for managing event volume while maintaining
/// representativeness of the data.
#[derive(Debug, Clone)]
pub struct EventSamplingConfig {
    /// Enable event sampling
    pub enabled: bool,

    /// Default sampling rate for all events
    pub default_rate: f64,

    /// Per-event-type sampling rates
    pub type_specific_rates: HashMap<String, f64>,

    /// Adaptive sampling configuration
    pub adaptive: AdaptiveEventSamplingConfig,
}

/// Adaptive event sampling configuration
#[derive(Debug, Clone)]
pub struct AdaptiveEventSamplingConfig {
    /// Enable adaptive sampling
    pub enabled: bool,

    /// Target event rate (events per second)
    pub target_rate: f64,

    /// Adaptation algorithm
    pub algorithm: String,

    /// Adaptation frequency
    pub adaptation_interval: Duration,
}

impl EventTrackingConfig {
    /// Create configuration optimized for security monitoring
    ///
    /// Security-focused configuration includes:
    /// - All security-related event types enabled
    /// - Reliable delivery with retries
    /// - Comprehensive enrichment for audit trails
    /// - No sampling to ensure complete security records
    /// - Extended retention for compliance
    pub fn security_focused() -> Self {
        Self {
            enabled: true,
            event_types: vec![
                EventType::Security,
                EventType::Audit,
                EventType::User,
                EventType::Error,
                EventType::System,
            ],
            collection: EventCollectionConfig {
                method: CollectionMethod::Push {
                    endpoint: "https://security-events.internal".to_string(),
                    max_connections: 100,
                },
                buffer_size: 100,
                max_buffer_size: 1000,
                flush_interval: Duration::from_secs(1),
                reliable_delivery: true,
                max_retries: 5,
                timeout: Duration::from_secs(30),
            },
            enrichment: EventEnrichmentConfig {
                enabled: true,
                rules: Vec::new(),
                context_sources: Vec::new(),
                max_processing_time: Duration::from_millis(100),
                enable_caching: true,
                cache_ttl: Duration::from_secs(300),
            },
            filtering: EventFilteringConfig {
                enabled: false, // No filtering for security events
                rules: Vec::new(),
                default_action: FilterAction::Keep,
                rate_limiting: false,
                rate_limits: HashMap::new(),
            },
            sampling: EventSamplingConfig {
                enabled: false, // No sampling for security events
                default_rate: 1.0,
                type_specific_rates: HashMap::new(),
                adaptive: AdaptiveEventSamplingConfig {
                    enabled: false,
                    target_rate: 1000.0,
                    algorithm: "none".to_string(),
                    adaptation_interval: Duration::from_secs(60),
                },
            },
        }
    }

    /// Create configuration optimized for high-volume scenarios
    ///
    /// High-volume configuration includes:
    /// - Selective event types to reduce volume
    /// - Aggressive sampling strategies
    /// - Large buffers for efficiency
    /// - Rate limiting to prevent overload
    /// - Minimal enrichment to reduce latency
    pub fn high_volume() -> Self {
        Self {
            enabled: true,
            event_types: vec![
                EventType::Error,
                EventType::Performance,
                EventType::System,
            ],
            collection: EventCollectionConfig {
                method: CollectionMethod::MessageQueue {
                    connection: "amqp://localhost:5672".to_string(),
                    queue_name: "events".to_string(),
                },
                buffer_size: 10000,
                max_buffer_size: 50000,
                flush_interval: Duration::from_secs(10),
                reliable_delivery: false,
                max_retries: 1,
                timeout: Duration::from_secs(5),
            },
            enrichment: EventEnrichmentConfig {
                enabled: false, // Disabled for performance
                rules: Vec::new(),
                context_sources: Vec::new(),
                max_processing_time: Duration::from_millis(1),
                enable_caching: false,
                cache_ttl: Duration::from_secs(0),
            },
            filtering: EventFilteringConfig {
                enabled: true,
                rules: Vec::new(),
                default_action: FilterAction::Sample { rate: 0.01 }, // 1% sampling
                rate_limiting: true,
                rate_limits: HashMap::new(),
            },
            sampling: EventSamplingConfig {
                enabled: true,
                default_rate: 0.01, // 1% sampling
                type_specific_rates: HashMap::new(),
                adaptive: AdaptiveEventSamplingConfig {
                    enabled: true,
                    target_rate: 1000.0,
                    algorithm: "load_based".to_string(),
                    adaptation_interval: Duration::from_secs(10),
                },
            },
        }
    }

    /// Validate the event tracking configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate buffer sizes
        if self.collection.buffer_size == 0 {
            return Err("Buffer size must be positive".to_string());
        }

        if self.collection.max_buffer_size < self.collection.buffer_size {
            return Err("Max buffer size must be at least as large as buffer size".to_string());
        }

        // Validate sampling rates
        if self.sampling.default_rate < 0.0 || self.sampling.default_rate > 1.0 {
            return Err("Default sampling rate must be between 0.0 and 1.0".to_string());
        }

        for (event_type, rate) in &self.sampling.type_specific_rates {
            if *rate < 0.0 || *rate > 1.0 {
                return Err(format!("Sampling rate for {} must be between 0.0 and 1.0", event_type));
            }
        }

        Ok(())
    }
}

impl Default for EventTrackingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            event_types: vec![
                EventType::System,
                EventType::Application,
                EventType::Error,
                EventType::Performance,
            ],
            collection: EventCollectionConfig {
                method: CollectionMethod::Push {
                    endpoint: "http://localhost:8080/events".to_string(),
                    max_connections: 10,
                },
                buffer_size: 1000,
                max_buffer_size: 10000,
                flush_interval: Duration::from_secs(10),
                reliable_delivery: false,
                max_retries: 3,
                timeout: Duration::from_secs(10),
            },
            enrichment: EventEnrichmentConfig {
                enabled: true,
                rules: Vec::new(),
                context_sources: Vec::new(),
                max_processing_time: Duration::from_millis(50),
                enable_caching: true,
                cache_ttl: Duration::from_secs(300),
            },
            filtering: EventFilteringConfig {
                enabled: false,
                rules: Vec::new(),
                default_action: FilterAction::Keep,
                rate_limiting: false,
                rate_limits: HashMap::new(),
            },
            sampling: EventSamplingConfig {
                enabled: false,
                default_rate: 1.0,
                type_specific_rates: HashMap::new(),
                adaptive: AdaptiveEventSamplingConfig {
                    enabled: false,
                    target_rate: 1000.0,
                    algorithm: "none".to_string(),
                    adaptation_interval: Duration::from_secs(60),
                },
            },
        }
    }
}