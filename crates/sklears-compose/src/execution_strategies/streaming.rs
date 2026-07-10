//! Streaming execution strategy for real-time, low-latency processing.

use crate::task_definitions::ExecutionTask;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use super::core::{StrategyConfig, StrategyMetrics, StrategyState};


/// Backpressure handling strategies
#[derive(Debug, Clone)]
pub enum BackpressureStrategy {
    /// Block until buffer space available
    Block,
    /// Drop oldest items in buffer
    DropOldest,
    /// Drop newest items
    DropNewest,
    /// Spill to disk
    SpillToDisk,
    /// Scale out resources
    ScaleOut,
}
/// Stream processing context
#[derive(Debug, Clone)]
pub struct Stream {
    /// Stream identifier
    pub id: String,
    /// Stream buffer
    pub buffer: VecDeque<ExecutionTask>,
    /// Stream metrics
    pub metrics: StreamMetrics,
    /// Stream state
    pub state: StreamState,
}
/// Stream-specific metrics
#[derive(Debug, Clone)]
pub struct StreamMetrics {
    /// Items processed
    pub items_processed: u64,
    /// Current buffer size
    pub buffer_size: usize,
    /// Average processing latency
    pub avg_latency: Duration,
    /// Throughput
    pub throughput: f64,
}
/// Stream processing state
#[derive(Debug, Clone, PartialEq)]
pub enum StreamState {
    /// Active
    Active,
    /// Paused
    Paused,
    /// Draining
    Draining,
    /// Stopped
    Stopped,
}
/// Streaming execution strategy for real-time processing
#[derive(Debug)]
#[allow(dead_code)]
pub struct StreamingExecutionStrategy {
    /// Strategy configuration
    pub(super) config: StrategyConfig,
    /// Stream buffer size
    pub(super) buffer_size: usize,
    /// Maximum acceptable latency
    pub(super) max_latency: Duration,
    /// Backpressure handling strategy
    pub(super) backpressure_strategy: BackpressureStrategy,
    /// Flow control enabled
    pub(super) flow_control: bool,
    /// Watermark interval for event time processing
    pub(super) watermark_interval: Duration,
    /// Active streams
    pub(super) active_streams: Arc<Mutex<HashMap<String, Stream>>>,
    /// Execution metrics
    pub(super) metrics: Arc<Mutex<StrategyMetrics>>,
    /// Strategy state
    pub(super) state: Arc<RwLock<StrategyState>>,
}
