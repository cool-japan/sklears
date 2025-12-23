//! # UCBBandit - Trait Implementations
//!
//! This module contains trait implementations for `UCBBandit`.
//!
//! ## Implemented Traits
//!
//! - `BanditAlgorithm`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque, HashSet};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::fmt;
use std::cmp::Ordering as CmpOrdering;
use std::thread;
use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis, Ix1, Ix2, array};
use scirs2_core::ndarray;
use scirs2_core::random::{Random, rng, DistributionExt};
use scirs2_core::random::Rng;
use scirs2_core::ndarray_ext::{stats, manipulation, matrix};
use scirs2_core::simd::{SimdArray, SimdOps, auto_vectorize};
use scirs2_core::parallel::{ParallelExecutor, ChunkStrategy, LoadBalancer};
use scirs2_core::memory_efficient::{MemoryMappedArray, LazyArray, ChunkedArray};
use sklears_core::error::SklearsError;
use super::types::*;
use super::functions::*;

use super::types::UCBBandit;

impl BanditAlgorithm for UCBBandit {
    fn select_arm(&mut self) -> SklResult<usize> {
        for (i, arm_stat) in self.arm_statistics.iter().enumerate() {
            if arm_stat.selection_count == 0 {
                return Ok(i);
            }
        }
        self.compute_ucb_values()?;
        let selected_arm = self
            .arm_statistics
            .iter()
            .enumerate()
            .max_by(|a, b| {
                a.1.ucb_value.partial_cmp(&b.1.ucb_value).unwrap_or(CmpOrdering::Equal)
            })
            .map(|(i, _)| i)
            .ok_or_else(|| SklearsError::OptimizationError(
                "No arms available for selection".to_string(),
            ))?;
        Ok(selected_arm)
    }
    fn update_arm(&mut self, arm: usize, reward: f64) -> SklResult<()> {
        if arm >= self.num_arms {
            return Err(
                SklearsError::OptimizationError(format!("Invalid arm index: {}", arm)),
            );
        }
        self.total_rounds += 1;
        let arm_stat = &mut self.arm_statistics[arm];
        arm_stat.selection_count += 1;
        arm_stat.total_reward += reward;
        arm_stat.average_reward = arm_stat.total_reward
            / arm_stat.selection_count as f64;
        arm_stat.last_selection = SystemTime::now();
        arm_stat.reward_history.push_back(reward);
        if arm_stat.reward_history.len() > 1000 {
            arm_stat.reward_history.pop_front();
        }
        self.update_variance(arm, reward)?;
        self.regret_tracker.update_regret(reward, arm)?;
        Ok(())
    }
    fn get_arm_statistics(&self) -> Vec<ArmStatistics> {
        self.arm_statistics.clone()
    }
    fn get_regret(&self) -> f64 {
        self.regret_tracker.cumulative_regret
    }
    fn reset_bandit(&mut self) -> SklResult<()> {
        for arm_stat in &mut self.arm_statistics {
            arm_stat.selection_count = 0;
            arm_stat.total_reward = 0.0;
            arm_stat.average_reward = 0.0;
            arm_stat.reward_variance = 0.0;
            arm_stat.reward_history.clear();
            arm_stat.ucb_value = f64::INFINITY;
        }
        self.total_rounds = 0;
        self.regret_tracker = RegretTracker::default();
        Ok(())
    }
    fn get_regret_bound(&self, time_horizon: u64) -> f64 {
        let k = self.num_arms as f64;
        let t = time_horizon as f64;
        self.regret_tracker.regret_bound_multiplier * (k * t * t.ln()).sqrt()
    }
    fn get_confidence_bounds(&self) -> Vec<(f64, f64)> {
        self.arm_statistics.iter().map(|arm| arm.confidence_interval).collect()
    }
}

