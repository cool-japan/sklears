use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

use super::notification_message_types::{PendingNotification, NotificationPriority, RetryableNotification, DeadLetterNotification};

/// Rate limit manager for controlling notification frequency
#[derive(Debug, Clone)]
pub struct RateLimitManager {
    /// Rate limiters per channel
    pub channel_limiters: HashMap<String, ChannelRateLimiter>,
    /// Global rate limiter
    pub global_limiter: GlobalRateLimiter,
    /// Rate limit configuration
    pub config: RateLimitManagerConfig,
    /// Rate limit statistics
    pub statistics: RateLimitStatistics,
}

/// Rate limiter for individual channels
#[derive(Debug, Clone)]
pub struct ChannelRateLimiter {
    /// Channel ID
    pub channel_id: String,
    /// Token bucket state
    pub token_bucket: TokenBucket,
    /// Rate limit configuration
    pub config: RateLimit,
    /// Usage statistics
    pub usage_stats: RateLimitUsageStats,
    /// Overflow queue
    pub overflow_queue: VecDeque<PendingNotification>,
}

/// Token bucket for rate limiting
#[derive(Debug, Clone)]
pub struct TokenBucket {
    /// Current token count
    pub current_tokens: f64,
    /// Maximum tokens
    pub max_tokens: f64,
    /// Refill rate (tokens per second)
    pub refill_rate: f64,
    /// Last refill time
    pub last_refill: SystemTime,
    /// Bucket capacity
    pub capacity: f64,
}

/// Channel-specific rate limit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    /// Maximum messages per time window
    pub max_messages: u32,
    /// Time window duration
    pub time_window: Duration,
    /// Burst allowance
    pub burst_allowance: u32,
    /// Action when rate limit exceeded
    pub overflow_action: OverflowAction,
    /// Rate limiting algorithm
    pub algorithm: RateLimitAlgorithm,
    /// Sliding window configuration
    pub sliding_window: SlidingWindowConfig,
}

/// Actions when rate limit is exceeded
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OverflowAction {
    Drop,
    Queue,
    Batch,
    Escalate,
    Throttle,
    Custom(String),
}

/// Rate limiting algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RateLimitAlgorithm {
    TokenBucket,
    LeakyBucket,
    FixedWindow,
    SlidingWindow,
    SlidingLog,
    Custom(String),
}

/// Sliding window configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlidingWindowConfig {
    /// Window size
    pub window_size: Duration,
    /// Number of sub-windows
    pub sub_windows: usize,
    /// Smoothing factor
    pub smoothing_factor: f64,
}

/// Global rate limit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalRateLimit {
    /// Enable global rate limiting
    pub enabled: bool,
    /// Global message limit per time window
    pub global_limit: u64,
    /// Time window for global limit
    pub time_window: Duration,
    /// Burst allowance
    pub burst_allowance: u64,
    /// Action when global limit exceeded
    pub overflow_action: GlobalOverflowAction,
}

/// Actions when global rate limit is exceeded
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GlobalOverflowAction {
    /// Drop lowest priority messages
    DropLowPriority,
    /// Queue messages for later
    Queue,
    /// Reduce message frequency
    Throttle,
    /// Emergency escalation
    Escalate,
    /// Custom action
    Custom(String),
}

/// Rate limit usage statistics
#[derive(Debug, Clone)]
pub struct RateLimitUsageStats {
    /// Total requests
    pub total_requests: u64,
    /// Allowed requests
    pub allowed_requests: u64,
    /// Rejected requests
    pub rejected_requests: u64,
    /// Current request rate
    pub current_rate: f64,
    /// Peak request rate
    pub peak_rate: f64,
    /// Last request time
    pub last_request: Option<SystemTime>,
}

/// Global rate limiter
#[derive(Debug, Clone)]
pub struct GlobalRateLimiter {
    /// Global token bucket
    pub token_bucket: TokenBucket,
    /// Global configuration
    pub config: GlobalRateLimit,
    /// Global statistics
    pub statistics: GlobalRateLimitStats,
    /// Priority queues
    pub priority_queues: HashMap<NotificationPriority, VecDeque<PendingNotification>>,
}

/// Global rate limit statistics
#[derive(Debug, Clone)]
pub struct GlobalRateLimitStats {
    /// Total system requests
    pub total_requests: u64,
    /// Allowed requests
    pub allowed_requests: u64,
    /// Rejected requests
    pub rejected_requests: u64,
    /// Current system load
    pub current_load: f64,
    /// Peak system load
    pub peak_load: f64,
    /// Load by priority
    pub load_by_priority: HashMap<NotificationPriority, f64>,
}

/// Rate limit manager configuration
#[derive(Debug, Clone)]
pub struct RateLimitManagerConfig {
    /// Enable distributed rate limiting
    pub enable_distributed_limiting: bool,
    /// Rate limit sharing configuration
    pub sharing_config: RateLimitSharingConfig,
    /// Burst handling strategy
    pub burst_handling: BurstHandlingStrategy,
    /// Monitoring configuration
    pub monitoring_config: RateLimitMonitoringConfig,
}

/// Rate limit sharing configuration
#[derive(Debug, Clone)]
pub struct RateLimitSharingConfig {
    /// Sharing mechanism
    pub sharing_mechanism: SharingMechanism,
    /// Sharing interval
    pub sharing_interval: Duration,
    /// Coordination endpoint
    pub coordination_endpoint: Option<String>,
    /// Conflict resolution strategy
    pub conflict_resolution: ConflictResolutionStrategy,
}

/// Sharing mechanisms for distributed rate limiting
#[derive(Debug, Clone)]
pub enum SharingMechanism {
    Redis,
    Database,
    MessageQueue,
    CustomAPI,
    None,
}

/// Conflict resolution strategies
#[derive(Debug, Clone)]
pub enum ConflictResolutionStrategy {
    MostRestrictive,
    LeastRestrictive,
    Weighted,
    Priority,
    Custom(String),
}

/// Burst handling strategies
#[derive(Debug, Clone)]
pub enum BurstHandlingStrategy {
    Allow,
    Queue,
    Reject,
    Throttle,
    Adaptive,
}

/// Rate limit monitoring configuration
#[derive(Debug, Clone)]
pub struct RateLimitMonitoringConfig {
    /// Enable monitoring
    pub enabled: bool,
    /// Monitoring interval
    pub interval: Duration,
    /// Alert thresholds
    pub alert_thresholds: RateLimitAlertThresholds,
    /// Metrics collection
    pub metrics_collection: bool,
}

/// Alert thresholds for rate limiting
#[derive(Debug, Clone)]
pub struct RateLimitAlertThresholds {
    /// Rejection rate threshold
    pub rejection_rate_threshold: f64,
    /// Queue length threshold
    pub queue_length_threshold: usize,
    /// Response time threshold
    pub response_time_threshold: Duration,
}

/// Rate limit statistics
#[derive(Debug, Clone)]
pub struct RateLimitStatistics {
    /// Overall statistics
    pub overall_stats: OverallRateLimitStats,
    /// Channel statistics
    pub channel_stats: HashMap<String, ChannelRateLimitStats>,
    /// Time-based statistics
    pub time_based_stats: TimeBasedRateLimitStats,
}

/// Overall rate limit statistics
#[derive(Debug, Clone)]
pub struct OverallRateLimitStats {
    /// Total requests processed
    pub total_requests: u64,
    /// Allowed requests
    pub allowed_requests: u64,
    /// Rejected requests
    pub rejected_requests: u64,
    /// Queued requests
    pub queued_requests: u64,
    /// Current rejection rate
    pub current_rejection_rate: f64,
    /// Average processing time
    pub avg_processing_time: Duration,
}

/// Channel-specific rate limit statistics
#[derive(Debug, Clone)]
pub struct ChannelRateLimitStats {
    /// Channel ID
    pub channel_id: String,
    /// Request count
    pub request_count: u64,
    /// Allowed count
    pub allowed_count: u64,
    /// Rejected count
    pub rejected_count: u64,
    /// Current rate
    pub current_rate: f64,
    /// Token utilization
    pub token_utilization: f64,
}

/// Time-based rate limit statistics
#[derive(Debug, Clone)]
pub struct TimeBasedRateLimitStats {
    /// Hourly statistics
    pub hourly_stats: Vec<HourlyRateLimitStats>,
    /// Daily statistics
    pub daily_stats: Vec<DailyRateLimitStats>,
    /// Peak usage times
    pub peak_usage_times: Vec<PeakUsageTime>,
}

/// Hourly rate limit statistics
#[derive(Debug, Clone)]
pub struct HourlyRateLimitStats {
    /// Hour timestamp
    pub hour: SystemTime,
    /// Request count
    pub request_count: u64,
    /// Rejection count
    pub rejection_count: u64,
    /// Average rate
    pub average_rate: f64,
}

/// Daily rate limit statistics
#[derive(Debug, Clone)]
pub struct DailyRateLimitStats {
    /// Day timestamp
    pub day: SystemTime,
    /// Total requests
    pub total_requests: u64,
    /// Total rejections
    pub total_rejections: u64,
    /// Peak hourly rate
    pub peak_hourly_rate: f64,
}

/// Peak usage time
#[derive(Debug, Clone)]
pub struct PeakUsageTime {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Request rate
    pub rate: f64,
    /// Duration
    pub duration: Duration,
}

/// Retry manager for handling failed notifications
#[derive(Debug, Clone)]
pub struct RetryManager {
    /// Retry queues by priority
    pub retry_queues: HashMap<NotificationPriority, VecDeque<RetryableNotification>>,
    /// Retry configuration
    pub config: RetryManagerConfig,
    /// Retry statistics
    pub statistics: RetryStatistics,
    /// Dead letter queue
    pub dead_letter_queue: VecDeque<DeadLetterNotification>,
}

/// Retry reasons
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum RetryReason {
    NetworkError,
    Timeout,
    RateLimit,
    TemporaryFailure,
    AuthenticationFailure,
    Custom(String),
}

/// Backoff state for retry logic
#[derive(Debug, Clone)]
pub struct BackoffState {
    /// Current delay
    pub current_delay: Duration,
    /// Multiplier
    pub multiplier: f64,
    /// Jitter amount
    pub jitter: Duration,
    /// Maximum delay
    pub max_delay: Duration,
}

/// Retry manager configuration
#[derive(Debug, Clone)]
pub struct RetryManagerConfig {
    /// Maximum retry attempts per notification
    pub max_retry_attempts: u32,
    /// Default retry delay
    pub default_retry_delay: Duration,
    /// Maximum retry delay
    pub max_retry_delay: Duration,
    /// Retry queue size limit
    pub retry_queue_size_limit: usize,
    /// Dead letter queue size limit
    pub dead_letter_queue_size_limit: usize,
    /// Retry processing interval
    pub retry_processing_interval: Duration,
    /// Enable priority-based retries
    pub enable_priority_retries: bool,
}

/// Retry statistics
#[derive(Debug, Clone)]
pub struct RetryStatistics {
    /// Total retry attempts
    pub total_retries: u64,
    /// Successful retries
    pub successful_retries: u64,
    /// Failed retries
    pub failed_retries: u64,
    /// Dead lettered notifications
    pub dead_lettered: u64,
    /// Average retry count
    pub average_retry_count: f64,
    /// Retry success rate
    pub retry_success_rate: f64,
    /// Retry statistics by reason
    pub retry_by_reason: HashMap<RetryReason, RetryReasonStats>,
}

/// Retry statistics by reason
#[derive(Debug, Clone)]
pub struct RetryReasonStats {
    /// Total count for this reason
    pub total_count: u64,
    /// Success count for this reason
    pub success_count: u64,
    /// Average attempts for this reason
    pub average_attempts: f64,
    /// Success rate for this reason
    pub success_rate: f64,
}

impl TokenBucket {
    /// Create a new token bucket
    pub fn new(capacity: f64, refill_rate: f64) -> Self {
        Self {
            current_tokens: capacity,
            max_tokens: capacity,
            refill_rate,
            last_refill: SystemTime::now(),
            capacity,
        }
    }

    /// Try to consume tokens from the bucket
    pub fn try_consume(&mut self, tokens: f64) -> bool {
        self.refill();

        if self.current_tokens >= tokens {
            self.current_tokens -= tokens;
            true
        } else {
            false
        }
    }

    /// Refill the token bucket based on elapsed time
    pub fn refill(&mut self) {
        let now = SystemTime::now();
        if let Ok(elapsed) = now.duration_since(self.last_refill) {
            let tokens_to_add = elapsed.as_secs_f64() * self.refill_rate;
            self.current_tokens = (self.current_tokens + tokens_to_add).min(self.max_tokens);
            self.last_refill = now;
        }
    }

    /// Get current token count
    pub fn current_tokens(&mut self) -> f64 {
        self.refill();
        self.current_tokens
    }

    /// Get token utilization percentage
    pub fn utilization(&mut self) -> f64 {
        self.refill();
        (self.max_tokens - self.current_tokens) / self.max_tokens * 100.0
    }
}

impl ChannelRateLimiter {
    /// Create a new channel rate limiter
    pub fn new(channel_id: String, config: RateLimit) -> Self {
        let token_bucket = TokenBucket::new(
            config.max_messages as f64,
            config.max_messages as f64 / config.time_window.as_secs_f64(),
        );

        Self {
            channel_id,
            token_bucket,
            config,
            usage_stats: RateLimitUsageStats::default(),
            overflow_queue: VecDeque::new(),
        }
    }

    /// Check if a notification can be sent
    pub fn can_send(&mut self, notification: &PendingNotification) -> bool {
        self.usage_stats.total_requests += 1;
        self.usage_stats.last_request = Some(SystemTime::now());

        let tokens_needed = match notification.priority {
            NotificationPriority::Emergency => 0.5, // Emergency uses fewer tokens
            NotificationPriority::Critical => 0.7,
            NotificationPriority::High => 1.0,
            NotificationPriority::Normal => 1.0,
            NotificationPriority::Low => 1.5, // Low priority uses more tokens
            NotificationPriority::Custom(weight) => weight as f64 / 10.0,
        };

        if self.token_bucket.try_consume(tokens_needed) {
            self.usage_stats.allowed_requests += 1;
            self.update_rate_stats(true);
            true
        } else {
            self.usage_stats.rejected_requests += 1;
            self.update_rate_stats(false);
            self.handle_overflow(notification.clone());
            false
        }
    }

    fn update_rate_stats(&mut self, allowed: bool) {
        let now = SystemTime::now();
        if let Some(last_request) = self.usage_stats.last_request {
            if let Ok(elapsed) = now.duration_since(last_request) {
                if elapsed > Duration::from_secs(0) {
                    self.usage_stats.current_rate = 1.0 / elapsed.as_secs_f64();
                    if self.usage_stats.current_rate > self.usage_stats.peak_rate {
                        self.usage_stats.peak_rate = self.usage_stats.current_rate;
                    }
                }
            }
        }
    }

    fn handle_overflow(&mut self, notification: PendingNotification) {
        match &self.config.overflow_action {
            OverflowAction::Queue => {
                self.overflow_queue.push_back(notification);
            }
            OverflowAction::Drop => {
                // Notification is dropped
            }
            OverflowAction::Throttle => {
                // Implement throttling logic
                self.overflow_queue.push_back(notification);
            }
            _ => {
                // Default to queuing
                self.overflow_queue.push_back(notification);
            }
        }
    }

    /// Process overflow queue
    pub fn process_overflow_queue(&mut self) -> Vec<PendingNotification> {
        let mut processed = Vec::new();

        while let Some(notification) = self.overflow_queue.front() {
            if self.can_send(notification) {
                if let Some(notification) = self.overflow_queue.pop_front() {
                    processed.push(notification);
                }
            } else {
                break;
            }
        }

        processed
    }
}

impl GlobalRateLimiter {
    /// Create a new global rate limiter
    pub fn new(config: GlobalRateLimit) -> Self {
        let token_bucket = TokenBucket::new(
            config.global_limit as f64,
            config.global_limit as f64 / config.time_window.as_secs_f64(),
        );

        Self {
            token_bucket,
            config,
            statistics: GlobalRateLimitStats::default(),
            priority_queues: HashMap::new(),
        }
    }

    /// Check if a notification can be sent globally
    pub fn can_send_globally(&mut self, notification: &PendingNotification) -> bool {
        if !self.config.enabled {
            return true;
        }

        self.statistics.total_requests += 1;

        let tokens_needed = match notification.priority {
            NotificationPriority::Emergency => 0.3, // Emergency notifications use fewer global tokens
            NotificationPriority::Critical => 0.5,
            NotificationPriority::High => 0.8,
            NotificationPriority::Normal => 1.0,
            NotificationPriority::Low => 1.5,
            NotificationPriority::Custom(weight) => weight as f64 / 15.0,
        };

        if self.token_bucket.try_consume(tokens_needed) {
            self.statistics.allowed_requests += 1;
            self.update_load_stats(notification.priority.clone());
            true
        } else {
            self.statistics.rejected_requests += 1;
            self.handle_global_overflow(notification.clone());
            false
        }
    }

    fn update_load_stats(&mut self, priority: NotificationPriority) {
        let utilization = self.token_bucket.utilization();
        self.statistics.current_load = utilization;

        if utilization > self.statistics.peak_load {
            self.statistics.peak_load = utilization;
        }

        *self.statistics.load_by_priority.entry(priority).or_insert(0.0) += 1.0;
    }

    fn handle_global_overflow(&mut self, notification: PendingNotification) {
        match &self.config.overflow_action {
            GlobalOverflowAction::Queue => {
                let priority = notification.priority.clone();
                self.priority_queues
                    .entry(priority)
                    .or_insert_with(VecDeque::new)
                    .push_back(notification);
            }
            GlobalOverflowAction::DropLowPriority => {
                // Only queue high priority notifications
                if matches!(notification.priority,
                    NotificationPriority::Emergency |
                    NotificationPriority::Critical |
                    NotificationPriority::High) {
                    let priority = notification.priority.clone();
                    self.priority_queues
                        .entry(priority)
                        .or_insert_with(VecDeque::new)
                        .push_back(notification);
                }
            }
            _ => {
                // Default behavior
                let priority = notification.priority.clone();
                self.priority_queues
                    .entry(priority)
                    .or_insert_with(VecDeque::new)
                    .push_back(notification);
            }
        }
    }

    /// Process priority queues
    pub fn process_priority_queues(&mut self) -> Vec<PendingNotification> {
        let mut processed = Vec::new();

        // Process in priority order
        let priorities = vec![
            NotificationPriority::Emergency,
            NotificationPriority::Critical,
            NotificationPriority::High,
            NotificationPriority::Normal,
            NotificationPriority::Low,
        ];

        for priority in priorities {
            if let Some(queue) = self.priority_queues.get_mut(&priority) {
                while let Some(notification) = queue.front() {
                    if self.can_send_globally(notification) {
                        if let Some(notification) = queue.pop_front() {
                            processed.push(notification);
                        }
                    } else {
                        break;
                    }
                }
            }
        }

        processed
    }
}

impl RateLimitManager {
    /// Create a new rate limit manager
    pub fn new(config: RateLimitManagerConfig) -> Self {
        Self {
            channel_limiters: HashMap::new(),
            global_limiter: GlobalRateLimiter::new(GlobalRateLimit::default()),
            config,
            statistics: RateLimitStatistics::default(),
        }
    }

    /// Add a channel rate limiter
    pub fn add_channel_limiter(&mut self, channel_id: String, rate_limit: RateLimit) {
        let limiter = ChannelRateLimiter::new(channel_id.clone(), rate_limit);
        self.channel_limiters.insert(channel_id, limiter);
    }

    /// Check if notification can be sent through channel
    pub fn can_send(&mut self, channel_id: &str, notification: &PendingNotification) -> bool {
        // Check global rate limit first
        if !self.global_limiter.can_send_globally(notification) {
            return false;
        }

        // Check channel-specific rate limit
        if let Some(limiter) = self.channel_limiters.get_mut(channel_id) {
            limiter.can_send(notification)
        } else {
            true // No rate limit configured for this channel
        }
    }

    /// Process all overflow queues
    pub fn process_overflow_queues(&mut self) -> HashMap<String, Vec<PendingNotification>> {
        let mut processed = HashMap::new();

        // Process global overflow first
        let global_processed = self.global_limiter.process_priority_queues();
        if !global_processed.is_empty() {
            processed.insert("global".to_string(), global_processed);
        }

        // Process channel overflows
        for (channel_id, limiter) in &mut self.channel_limiters {
            let channel_processed = limiter.process_overflow_queue();
            if !channel_processed.is_empty() {
                processed.insert(channel_id.clone(), channel_processed);
            }
        }

        processed
    }

    /// Get rate limit statistics
    pub fn get_statistics(&mut self) -> RateLimitStatistics {
        let mut stats = self.statistics.clone();

        // Update overall stats
        stats.overall_stats.total_requests = self.global_limiter.statistics.total_requests;
        stats.overall_stats.allowed_requests = self.global_limiter.statistics.allowed_requests;
        stats.overall_stats.rejected_requests = self.global_limiter.statistics.rejected_requests;

        // Update channel stats
        for (channel_id, limiter) in &mut self.channel_limiters {
            let channel_stats = ChannelRateLimitStats {
                channel_id: channel_id.clone(),
                request_count: limiter.usage_stats.total_requests,
                allowed_count: limiter.usage_stats.allowed_requests,
                rejected_count: limiter.usage_stats.rejected_requests,
                current_rate: limiter.usage_stats.current_rate,
                token_utilization: limiter.token_bucket.utilization(),
            };
            stats.channel_stats.insert(channel_id.clone(), channel_stats);
        }

        self.statistics = stats.clone();
        stats
    }
}

impl Default for RateLimitUsageStats {
    fn default() -> Self {
        Self {
            total_requests: 0,
            allowed_requests: 0,
            rejected_requests: 0,
            current_rate: 0.0,
            peak_rate: 0.0,
            last_request: None,
        }
    }
}

impl Default for GlobalRateLimitStats {
    fn default() -> Self {
        Self {
            total_requests: 0,
            allowed_requests: 0,
            rejected_requests: 0,
            current_load: 0.0,
            peak_load: 0.0,
            load_by_priority: HashMap::new(),
        }
    }
}

impl Default for GlobalRateLimit {
    fn default() -> Self {
        Self {
            enabled: true,
            global_limit: 1000,
            time_window: Duration::from_secs(60),
            burst_allowance: 100,
            overflow_action: GlobalOverflowAction::Queue,
        }
    }
}

impl Default for RateLimit {
    fn default() -> Self {
        Self {
            max_messages: 100,
            time_window: Duration::from_secs(60),
            burst_allowance: 10,
            overflow_action: OverflowAction::Queue,
            algorithm: RateLimitAlgorithm::TokenBucket,
            sliding_window: SlidingWindowConfig::default(),
        }
    }
}

impl Default for SlidingWindowConfig {
    fn default() -> Self {
        Self {
            window_size: Duration::from_secs(60),
            sub_windows: 10,
            smoothing_factor: 0.1,
        }
    }
}

impl Default for RateLimitStatistics {
    fn default() -> Self {
        Self {
            overall_stats: OverallRateLimitStats::default(),
            channel_stats: HashMap::new(),
            time_based_stats: TimeBasedRateLimitStats::default(),
        }
    }
}

impl Default for OverallRateLimitStats {
    fn default() -> Self {
        Self {
            total_requests: 0,
            allowed_requests: 0,
            rejected_requests: 0,
            queued_requests: 0,
            current_rejection_rate: 0.0,
            avg_processing_time: Duration::from_secs(0),
        }
    }
}

impl Default for TimeBasedRateLimitStats {
    fn default() -> Self {
        Self {
            hourly_stats: Vec::new(),
            daily_stats: Vec::new(),
            peak_usage_times: Vec::new(),
        }
    }
}

impl Default for RetryManagerConfig {
    fn default() -> Self {
        Self {
            max_retry_attempts: 3,
            default_retry_delay: Duration::from_secs(30),
            max_retry_delay: Duration::from_secs(3600),
            retry_queue_size_limit: 10000,
            dead_letter_queue_size_limit: 1000,
            retry_processing_interval: Duration::from_secs(10),
            enable_priority_retries: true,
        }
    }
}

impl Default for RetryStatistics {
    fn default() -> Self {
        Self {
            total_retries: 0,
            successful_retries: 0,
            failed_retries: 0,
            dead_lettered: 0,
            average_retry_count: 0.0,
            retry_success_rate: 0.0,
            retry_by_reason: HashMap::new(),
        }
    }
}

impl RetryManager {
    /// Create a new retry manager
    pub fn new(config: RetryManagerConfig) -> Self {
        Self {
            retry_queues: HashMap::new(),
            config,
            statistics: RetryStatistics::default(),
            dead_letter_queue: VecDeque::new(),
        }
    }

    /// Add notification to retry queue
    pub fn add_for_retry(&mut self, retryable_notification: RetryableNotification) {
        let priority = retryable_notification.notification.priority.clone();

        self.retry_queues
            .entry(priority)
            .or_insert_with(VecDeque::new)
            .push_back(retryable_notification);

        self.statistics.total_retries += 1;
    }

    /// Get notifications ready for retry
    pub fn get_ready_for_retry(&mut self) -> Vec<RetryableNotification> {
        let mut ready = Vec::new();
        let now = SystemTime::now();

        for queue in self.retry_queues.values_mut() {
            let mut i = 0;
            while i < queue.len() {
                if let Some(retryable) = queue.get(i) {
                    if retryable.next_retry_time <= now {
                        if let Some(retryable) = queue.remove(i) {
                            ready.push(retryable);
                        } else {
                            i += 1;
                        }
                    } else {
                        i += 1;
                    }
                } else {
                    i += 1;
                }
            }
        }

        ready
    }

    /// Move notification to dead letter queue
    pub fn move_to_dead_letter(&mut self, dead_letter: DeadLetterNotification) {
        if self.dead_letter_queue.len() >= self.config.dead_letter_queue_size_limit {
            self.dead_letter_queue.pop_front(); // Remove oldest
        }

        self.dead_letter_queue.push_back(dead_letter);
        self.statistics.dead_lettered += 1;
    }

    /// Get retry statistics
    pub fn get_statistics(&self) -> &RetryStatistics {
        &self.statistics
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_bucket_creation() {
        let mut bucket = TokenBucket::new(10.0, 2.0);
        assert_eq!(bucket.current_tokens(), 10.0);
        assert_eq!(bucket.capacity, 10.0);
        assert_eq!(bucket.refill_rate, 2.0);
    }

    #[test]
    fn test_token_bucket_consume() {
        let mut bucket = TokenBucket::new(10.0, 2.0);

        // Should be able to consume tokens
        assert!(bucket.try_consume(5.0));
        assert_eq!(bucket.current_tokens(), 5.0);

        // Should not be able to consume more than available
        assert!(!bucket.try_consume(10.0));
        assert_eq!(bucket.current_tokens(), 5.0);
    }

    #[test]
    fn test_channel_rate_limiter() {
        let config = RateLimit {
            max_messages: 5,
            time_window: Duration::from_secs(60),
            burst_allowance: 2,
            overflow_action: OverflowAction::Queue,
            algorithm: RateLimitAlgorithm::TokenBucket,
            sliding_window: SlidingWindowConfig::default(),
        };

        let mut limiter = ChannelRateLimiter::new("test-channel".to_string(), config);

        // Create test notification
        let notification = PendingNotification::new(
            "test-123".to_string(),
            "test-channel".to_string(),
            super::super::notification_message_types::NotificationMessage {
                subject: "Test".to_string(),
                body: "Test".to_string(),
                message_type: super::super::notification_message_types::MessageType::Alert,
                attachments: Vec::new(),
                metadata: super::super::notification_message_types::MessageMetadata {
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

        // Should allow notifications initially
        assert!(limiter.can_send(&notification));
        assert!(limiter.can_send(&notification));

        // Check statistics
        assert_eq!(limiter.usage_stats.total_requests, 2);
        assert_eq!(limiter.usage_stats.allowed_requests, 2);
    }

    #[test]
    fn test_global_rate_limiter() {
        let config = GlobalRateLimit {
            enabled: true,
            global_limit: 5,
            time_window: Duration::from_secs(60),
            burst_allowance: 1,
            overflow_action: GlobalOverflowAction::Queue,
        };

        let mut limiter = GlobalRateLimiter::new(config);

        let notification = PendingNotification::new(
            "test-123".to_string(),
            "test-channel".to_string(),
            super::super::notification_message_types::NotificationMessage {
                subject: "Test".to_string(),
                body: "Test".to_string(),
                message_type: super::super::notification_message_types::MessageType::Alert,
                attachments: Vec::new(),
                metadata: super::super::notification_message_types::MessageMetadata {
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

        // Should allow notifications initially
        assert!(limiter.can_send_globally(&notification));
        assert_eq!(limiter.statistics.total_requests, 1);
        assert_eq!(limiter.statistics.allowed_requests, 1);
    }

    #[test]
    fn test_retry_manager() {
        let config = RetryManagerConfig::default();
        let mut retry_manager = RetryManager::new(config);

        let notification = PendingNotification::new(
            "test-123".to_string(),
            "test-channel".to_string(),
            super::super::notification_message_types::NotificationMessage {
                subject: "Test".to_string(),
                body: "Test".to_string(),
                message_type: super::super::notification_message_types::MessageType::Alert,
                attachments: Vec::new(),
                metadata: super::super::notification_message_types::MessageMetadata {
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

        let retryable = RetryableNotification {
            notification,
            retry_count: 1,
            next_retry_time: SystemTime::now() - Duration::from_secs(1), // Past time, ready for retry
            retry_reason: RetryReason::NetworkError,
            backoff_state: BackoffState {
                current_delay: Duration::from_secs(30),
                multiplier: 2.0,
                jitter: Duration::from_secs(5),
                max_delay: Duration::from_secs(300),
            },
        };

        retry_manager.add_for_retry(retryable);
        assert_eq!(retry_manager.statistics.total_retries, 1);

        let ready = retry_manager.get_ready_for_retry();
        assert_eq!(ready.len(), 1);
    }

    #[test]
    fn test_rate_limit_manager_integration() {
        let config = RateLimitManagerConfig {
            enable_distributed_limiting: false,
            sharing_config: RateLimitSharingConfig {
                sharing_mechanism: SharingMechanism::None,
                sharing_interval: Duration::from_secs(60),
                coordination_endpoint: None,
                conflict_resolution: ConflictResolutionStrategy::MostRestrictive,
            },
            burst_handling: BurstHandlingStrategy::Queue,
            monitoring_config: RateLimitMonitoringConfig {
                enabled: true,
                interval: Duration::from_secs(10),
                alert_thresholds: RateLimitAlertThresholds {
                    rejection_rate_threshold: 0.1,
                    queue_length_threshold: 100,
                    response_time_threshold: Duration::from_secs(1),
                },
                metrics_collection: true,
            },
        };

        let mut manager = RateLimitManager::new(config);

        let rate_limit = RateLimit {
            max_messages: 10,
            time_window: Duration::from_secs(60),
            burst_allowance: 5,
            overflow_action: OverflowAction::Queue,
            algorithm: RateLimitAlgorithm::TokenBucket,
            sliding_window: SlidingWindowConfig::default(),
        };

        manager.add_channel_limiter("test-channel".to_string(), rate_limit);

        let notification = PendingNotification::new(
            "test-123".to_string(),
            "test-channel".to_string(),
            super::super::notification_message_types::NotificationMessage {
                subject: "Test".to_string(),
                body: "Test".to_string(),
                message_type: super::super::notification_message_types::MessageType::Alert,
                attachments: Vec::new(),
                metadata: super::super::notification_message_types::MessageMetadata {
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

        // Test that notifications can be sent
        assert!(manager.can_send("test-channel", &notification));

        // Get and verify statistics
        let stats = manager.get_statistics();
        assert!(stats.overall_stats.total_requests > 0);
    }
}