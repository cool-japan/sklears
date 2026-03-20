//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use serde::{Deserialize, Serialize};
use super::types::*;

/// Notification provider trait
pub trait NotificationProvider: Send + Sync {
    /// Send notification
    fn send_notification(
        &self,
        channel: &NotificationChannel,
        message: &NotificationMessage,
    ) -> Result<NotificationResult, NotificationError>;
    /// Validate channel configuration
    fn validate_config(&self, config: &ChannelConfig) -> Result<(), Vec<String>>;
    /// Check channel health
    fn check_health(
        &self,
        channel: &NotificationChannel,
    ) -> Result<HealthStatus, String>;
    /// Get provider capabilities
    fn get_capabilities(&self) -> ProviderCapabilities;
    /// Initialize provider
    fn initialize(&mut self, config: &ProviderConfiguration) -> Result<(), String>;
    /// Shutdown provider
    fn shutdown(&mut self) -> Result<(), String>;
}
