//! Retry Strategy Implementations
//!
//! This module provides concrete implementations of retry strategies including
//! exponential backoff, linear backoff, adaptive strategies with machine learning,
//! and circuit breaker patterns with SIMD acceleration support.

use super::core::*;
use sklears_core::error::Result as SklResult;
use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex, RwLock},
    time::{Duration, SystemTime},
};

/// Exponential backoff retry strategy
#[derive(Debug)]
pub struct ExponentialBackoffStrategy {
    config: StrategyConfiguration,
    state: Arc<Mutex<ExponentialBackoffState>>,
}

/// State for exponential backoff strategy
#[derive(Debug)]
struct ExponentialBackoffState {
    current_multiplier: f64,
    consecutive_failures: u32,
    last_success_time: Option<SystemTime>,
}

impl ExponentialBackoffStrategy {
    /// Create new exponential backoff strategy
    pub fn new(config: StrategyConfiguration) -> Self {
        Self {
            config,
            state: Arc::new(Mutex::new(ExponentialBackoffState {
                current_multiplier: 1.0,
                consecutive_failures: 0,
                last_success_time: None,
            })),
        }
    }

    /// Configure with custom parameters
    pub fn with_multiplier(mut self, multiplier: f64) -> Self {
        self.config.parameters.insert("multiplier".to_string(), multiplier.to_string());
        self
    }

    /// Configure maximum delay
    pub fn with_max_delay(mut self, max_delay: Duration) -> Self {
        self.config.parameters.insert("max_delay_ms".to_string(), max_delay.as_millis().to_string());
        self
    }
}

impl RetryStrategy for ExponentialBackoffStrategy {
    fn should_retry(&self, context: &RetryContext) -> bool {
        if context.current_attempt >= self.config.max_attempts {
            return false;
        }

        let total_elapsed = SystemTime::now().duration_since(context.created_at).unwrap_or(Duration::ZERO);
        if total_elapsed >= self.config.max_duration {
            return false;
        }

        // Don't retry on certain error types
        if let Some(last_error) = context.errors.last() {
            match last_error {
                RetryError::Auth { .. } | RetryError::Configuration { .. } => return false,
                RetryError::CircuitOpen { .. } => return false,
                _ => {}
            }
        }

        true
    }

    fn calculate_delay(&self, attempt: u32, context: &RetryContext) -> Duration {
        let multiplier = self.config.parameters
            .get("multiplier")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(2.0);

        let max_delay_ms = self.config.parameters
            .get("max_delay_ms")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(300_000); // 5 minutes

        let base_ms = self.config.base_delay.as_millis() as f64;
        let exponential_delay = base_ms * multiplier.powi(attempt as i32);
        let capped_delay = exponential_delay.min(max_delay_ms as f64);

        Duration::from_millis(capped_delay as u64)
    }

    fn update_state(&mut self, result: &RetryResult, context: &RetryContext) {
        let mut state = self.state.lock().unwrap();

        match result {
            Ok(()) => {
                state.consecutive_failures = 0;
                state.last_success_time = Some(SystemTime::now());
                state.current_multiplier = 1.0;
            }
            Err(_) => {
                state.consecutive_failures += 1;
            }
        }
    }

    fn configuration(&self) -> StrategyConfiguration {
        self.config.clone()
    }

    fn name(&self) -> &str {
        "exponential_backoff"
    }

    fn capabilities(&self) -> StrategyCapabilities {
        StrategyCapabilities {
            adaptive: false,
            circuit_breaking: false,
            rate_limiting: false,
            simd_acceleration: false,
            performance: PerformanceCharacteristics {
                cpu_usage: PerformanceLevel::Low,
                memory_usage: PerformanceLevel::Low,
                latency: PerformanceLevel::Low,
                throughput: PerformanceLevel::High,
            },
        }
    }
}

/// Linear backoff retry strategy
#[derive(Debug)]
pub struct LinearBackoffStrategy {
    config: StrategyConfiguration,
    increment: Duration,
}

impl LinearBackoffStrategy {
    /// Create new linear backoff strategy
    pub fn new(config: StrategyConfiguration) -> Self {
        let increment = Duration::from_millis(
            config.parameters
                .get("increment_ms")
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(1000)
        );

        Self { config, increment }
    }

    /// Configure increment
    pub fn with_increment(mut self, increment: Duration) -> Self {
        self.increment = increment;
        self.config.parameters.insert("increment_ms".to_string(), increment.as_millis().to_string());
        self
    }
}

impl RetryStrategy for LinearBackoffStrategy {
    fn should_retry(&self, context: &RetryContext) -> bool {
        context.current_attempt < self.config.max_attempts
    }

    fn calculate_delay(&self, attempt: u32, _context: &RetryContext) -> Duration {
        let linear_delay = self.config.base_delay + (self.increment * attempt);
        linear_delay.min(Duration::from_secs(300)) // Cap at 5 minutes
    }

    fn update_state(&mut self, _result: &RetryResult, _context: &RetryContext) {
        // Linear strategy doesn't need state updates
    }

    fn configuration(&self) -> StrategyConfiguration {
        self.config.clone()
    }

    fn name(&self) -> &str {
        "linear_backoff"
    }

    fn capabilities(&self) -> StrategyCapabilities {
        StrategyCapabilities {
            adaptive: false,
            circuit_breaking: false,
            rate_limiting: false,
            simd_acceleration: false,
            performance: PerformanceCharacteristics {
                cpu_usage: PerformanceLevel::Low,
                memory_usage: PerformanceLevel::Low,
                latency: PerformanceLevel::Low,
                throughput: PerformanceLevel::High,
            },
        }
    }
}

/// Fixed delay retry strategy
#[derive(Debug)]
pub struct FixedDelayStrategy {
    config: StrategyConfiguration,
    fixed_delay: Duration,
}

impl FixedDelayStrategy {
    /// Create new fixed delay strategy
    pub fn new(config: StrategyConfiguration, fixed_delay: Duration) -> Self {
        Self { config, fixed_delay }
    }
}

impl RetryStrategy for FixedDelayStrategy {
    fn should_retry(&self, context: &RetryContext) -> bool {
        context.current_attempt < self.config.max_attempts
    }

    fn calculate_delay(&self, _attempt: u32, _context: &RetryContext) -> Duration {
        self.fixed_delay
    }

    fn update_state(&mut self, _result: &RetryResult, _context: &RetryContext) {
        // Fixed delay strategy doesn't need state updates
    }

    fn configuration(&self) -> StrategyConfiguration {
        self.config.clone()
    }

    fn name(&self) -> &str {
        "fixed_delay"
    }

    fn capabilities(&self) -> StrategyCapabilities {
        StrategyCapabilities {
            adaptive: false,
            circuit_breaking: false,
            rate_limiting: false,
            simd_acceleration: false,
            performance: PerformanceCharacteristics {
                cpu_usage: PerformanceLevel::Low,
                memory_usage: PerformanceLevel::Low,
                latency: PerformanceLevel::Low,
                throughput: PerformanceLevel::High,
            },
        }
    }
}

/// Adaptive retry strategy with machine learning
#[derive(Debug)]
pub struct AdaptiveStrategy {
    config: StrategyConfiguration,
    state: Arc<Mutex<AdaptiveState>>,
    learning_enabled: bool,
}

/// State for adaptive strategy
#[derive(Debug)]
struct AdaptiveState {
    success_rate_history: VecDeque<f64>,
    average_delay_history: VecDeque<Duration>,
    error_pattern_history: VecDeque<String>,
    current_strategy: String,
    performance_score: f64,
    learning_rate: f64,
}

impl AdaptiveStrategy {
    /// Create new adaptive strategy
    pub fn new(config: StrategyConfiguration) -> Self {
        Self {
            config,
            state: Arc::new(Mutex::new(AdaptiveState {
                success_rate_history: VecDeque::with_capacity(100),
                average_delay_history: VecDeque::with_capacity(100),
                error_pattern_history: VecDeque::with_capacity(100),
                current_strategy: "exponential".to_string(),
                performance_score: 0.5,
                learning_rate: 0.1,
            })),
            learning_enabled: true,
        }
    }

    /// Enable/disable learning
    pub fn set_learning_enabled(mut self, enabled: bool) -> Self {
        self.learning_enabled = enabled;
        self
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, rate: f64) -> Self {
        if let Ok(mut state) = self.state.lock() {
            state.learning_rate = rate.clamp(0.001, 1.0);
        }
        self
    }

    /// Calculate adaptive delay based on context
    fn calculate_adaptive_delay(&self, attempt: u32, context: &RetryContext) -> Duration {
        let state = self.state.lock().unwrap();

        // Base calculation using current strategy
        let base_delay = match state.current_strategy.as_str() {
            "exponential" => {
                let multiplier = 2.0_f64.powi(attempt as i32);
                Duration::from_millis((self.config.base_delay.as_millis() as f64 * multiplier) as u64)
            }
            "linear" => {
                Duration::from_millis(self.config.base_delay.as_millis() + (attempt as u64 * 1000))
            }
            _ => self.config.base_delay,
        };

        // Adjust based on recent performance
        let performance_multiplier = if state.performance_score > 0.8 {
            0.8 // Reduce delay if performing well
        } else if state.performance_score < 0.3 {
            1.5 // Increase delay if performing poorly
        } else {
            1.0
        };

        Duration::from_millis((base_delay.as_millis() as f64 * performance_multiplier) as u64)
    }

    /// Update performance based on attempt results
    fn update_performance(&self, result: &RetryResult, context: &RetryContext) {
        if !self.learning_enabled {
            return;
        }

        let mut state = self.state.lock().unwrap();

        // Calculate current success rate
        let success = result.is_ok();
        let current_success_rate = if success { 1.0 } else { 0.0 };

        // Update success rate history
        state.success_rate_history.push_back(current_success_rate);
        if state.success_rate_history.len() > 100 {
            state.success_rate_history.pop_front();
        }

        // Calculate average success rate
        let avg_success_rate: f64 = state.success_rate_history.iter().sum::<f64>()
            / state.success_rate_history.len() as f64;

        // Update performance score with learning rate
        state.performance_score = state.performance_score * (1.0 - state.learning_rate)
            + avg_success_rate * state.learning_rate;

        // Adapt strategy based on performance
        if state.performance_score < 0.3 && state.current_strategy != "linear" {
            state.current_strategy = "linear".to_string();
        } else if state.performance_score > 0.7 && state.current_strategy != "exponential" {
            state.current_strategy = "exponential".to_string();
        }
    }
}

impl RetryStrategy for AdaptiveStrategy {
    fn should_retry(&self, context: &RetryContext) -> bool {
        if context.current_attempt >= self.config.max_attempts {
            return false;
        }

        // Adaptive decision based on recent performance
        let state = self.state.lock().unwrap();
        if state.performance_score < 0.1 && context.current_attempt > 1 {
            return false; // Stop early if performing very poorly
        }

        true
    }

    fn calculate_delay(&self, attempt: u32, context: &RetryContext) -> Duration {
        self.calculate_adaptive_delay(attempt, context)
    }

    fn update_state(&mut self, result: &RetryResult, context: &RetryContext) {
        self.update_performance(result, context);
    }

    fn configuration(&self) -> StrategyConfiguration {
        self.config.clone()
    }

    fn name(&self) -> &str {
        "adaptive"
    }

    fn capabilities(&self) -> StrategyCapabilities {
        StrategyCapabilities {
            adaptive: true,
            circuit_breaking: false,
            rate_limiting: false,
            simd_acceleration: false,
            performance: PerformanceCharacteristics {
                cpu_usage: PerformanceLevel::Medium,
                memory_usage: PerformanceLevel::Medium,
                latency: PerformanceLevel::Low,
                throughput: PerformanceLevel::High,
            },
        }
    }
}

/// Circuit breaker retry strategy
#[derive(Debug)]
pub struct CircuitBreakerStrategy {
    config: StrategyConfiguration,
    circuit_config: CircuitBreakerConfig,
    state: Arc<Mutex<CircuitBreakerState>>,
}

/// Circuit breaker state
#[derive(Debug)]
enum CircuitBreakerState {
    Closed {
        failure_count: u32,
    },
    Open {
        opened_at: SystemTime,
    },
    HalfOpen {
        success_count: u32,
    },
}

impl CircuitBreakerStrategy {
    /// Create new circuit breaker strategy
    pub fn new(config: StrategyConfiguration, circuit_config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            circuit_config,
            state: Arc::new(Mutex::new(CircuitBreakerState::Closed { failure_count: 0 })),
        }
    }

    /// Check if circuit is open
    fn is_circuit_open(&self) -> bool {
        let state = self.state.lock().unwrap();
        matches!(*state, CircuitBreakerState::Open { .. })
    }

    /// Check if recovery timeout has passed
    fn should_attempt_reset(&self) -> bool {
        let state = self.state.lock().unwrap();
        if let CircuitBreakerState::Open { opened_at } = *state {
            SystemTime::now().duration_since(opened_at).unwrap_or(Duration::ZERO)
                >= self.circuit_config.recovery_timeout
        } else {
            false
        }
    }
}

impl RetryStrategy for CircuitBreakerStrategy {
    fn should_retry(&self, context: &RetryContext) -> bool {
        if self.is_circuit_open() && !self.should_attempt_reset() {
            return false;
        }

        context.current_attempt < self.config.max_attempts
    }

    fn calculate_delay(&self, attempt: u32, _context: &RetryContext) -> Duration {
        // Use exponential backoff as base
        let multiplier = 2.0_f64.powi(attempt as i32);
        Duration::from_millis((self.config.base_delay.as_millis() as f64 * multiplier) as u64)
    }

    fn update_state(&mut self, result: &RetryResult, _context: &RetryContext) {
        let mut state = self.state.lock().unwrap();

        match &mut *state {
            CircuitBreakerState::Closed { failure_count } => {
                match result {
                    Ok(()) => {
                        *failure_count = 0;
                    }
                    Err(_) => {
                        *failure_count += 1;
                        if *failure_count >= self.circuit_config.failure_threshold {
                            *state = CircuitBreakerState::Open {
                                opened_at: SystemTime::now(),
                            };
                        }
                    }
                }
            }
            CircuitBreakerState::Open { opened_at } => {
                if SystemTime::now().duration_since(*opened_at).unwrap_or(Duration::ZERO)
                    >= self.circuit_config.recovery_timeout {
                    *state = CircuitBreakerState::HalfOpen { success_count: 0 };
                }
            }
            CircuitBreakerState::HalfOpen { success_count } => {
                match result {
                    Ok(()) => {
                        *success_count += 1;
                        if *success_count >= self.circuit_config.success_threshold {
                            *state = CircuitBreakerState::Closed { failure_count: 0 };
                        }
                    }
                    Err(_) => {
                        *state = CircuitBreakerState::Open {
                            opened_at: SystemTime::now(),
                        };
                    }
                }
            }
        }
    }

    fn configuration(&self) -> StrategyConfiguration {
        self.config.clone()
    }

    fn name(&self) -> &str {
        "circuit_breaker"
    }

    fn capabilities(&self) -> StrategyCapabilities {
        StrategyCapabilities {
            adaptive: false,
            circuit_breaking: true,
            rate_limiting: false,
            simd_acceleration: false,
            performance: PerformanceCharacteristics {
                cpu_usage: PerformanceLevel::Low,
                memory_usage: PerformanceLevel::Low,
                latency: PerformanceLevel::Low,
                throughput: PerformanceLevel::High,
            },
        }
    }
}

/// Factory for creating retry strategies
pub struct StrategyFactory;

impl StrategyFactory {
    /// Create strategy by name
    pub fn create_strategy(
        name: &str,
        config: StrategyConfiguration
    ) -> SklResult<Box<dyn RetryStrategy + Send + Sync>> {
        match name {
            "exponential" => Ok(Box::new(ExponentialBackoffStrategy::new(config))),
            "linear" => Ok(Box::new(LinearBackoffStrategy::new(config))),
            "fixed" => {
                let delay = config.base_delay;
                Ok(Box::new(FixedDelayStrategy::new(config, delay)))
            }
            "adaptive" => Ok(Box::new(AdaptiveStrategy::new(config))),
            "circuit_breaker" => {
                let circuit_config = CircuitBreakerConfig {
                    failure_threshold: 5,
                    recovery_timeout: Duration::from_secs(30),
                    success_threshold: 3,
                };
                Ok(Box::new(CircuitBreakerStrategy::new(config, circuit_config)))
            }
            _ => Err(RetryError::Configuration {
                parameter: "strategy".to_string(),
                message: format!("Unknown strategy: {}", name),
            }.into()),
        }
    }

    /// List available strategies
    pub fn available_strategies() -> Vec<&'static str> {
        vec!["exponential", "linear", "fixed", "adaptive", "circuit_breaker"]
    }
}