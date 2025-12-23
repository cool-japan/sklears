//! # AuditLogging - Trait Implementations
//!
//! This module contains trait implementations for `AuditLogging`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for AuditLogging {
    fn default() -> Self {
        Self {
            enabled: true,
            logged_events: vec![
                AuditEvent::FileAccess, AuditEvent::PermissionDenied,
                AuditEvent::PolicyViolation, AuditEvent::AdminAction,
            ],
            storage: AuditLogStorage::File("/var/log/sklears/audit.log".to_string()),
            retention: Duration::days(90),
        }
    }
}

