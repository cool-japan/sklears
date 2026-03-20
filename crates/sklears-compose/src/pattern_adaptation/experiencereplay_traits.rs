//! # ExperienceReplay - Trait Implementations
//!
//! This module contains trait implementations for `ExperienceReplay`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque, HashSet};
use std::time::{Duration, Instant, SystemTime};

use super::types::{ExperienceAnalyzer, ExperienceReplay, PrioritizedSampling, ReplayMetrics, ReplayScheduler};

impl Default for ExperienceReplay {
    fn default() -> Self {
        Self {
            replay_id: format!(
                "replay_{}", SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default().as_millis()
            ),
            experience_buffer: VecDeque::new(),
            prioritized_sampling: PrioritizedSampling::default(),
            experience_analyzer: ExperienceAnalyzer::default(),
            replay_scheduler: ReplayScheduler::default(),
            buffer_capacity: 100000,
            replay_metrics: ReplayMetrics::default(),
        }
    }
}

