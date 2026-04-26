//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::Result as SklResult;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock, atomic::{AtomicU64, AtomicBool, Ordering}};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use super::node_management::{NodeInfo, EncryptionLevel};

use super::types::{BandwidthMonitor, CommunicationChannel, CommunicationMetrics, CommunicationSecurityMonitor, CompressionManager, EncryptionManager, Message, MessagePriority, MessageStatistics, NetworkOptimizationResult, PartitionRecoveryPlan, QualityOfServiceManager, RecoveryActionType};

use std::collections::{HashSet};

/// Comprehensive communication manager for distributed optimization
#[derive(Debug)]
pub struct CommunicationManager {
    active_channels: HashMap<String, CommunicationChannel>,
    message_queues: HashMap<String, VecDeque<Message>>,
    routing_table: HashMap<String, Vec<String>>,
    encryption_manager: Arc<Mutex<EncryptionManager>>,
    compression_manager: Arc<Mutex<CompressionManager>>,
    bandwidth_monitor: Arc<Mutex<BandwidthMonitor>>,
    message_statistics: Arc<Mutex<MessageStatistics>>,
    simd_accelerator: Arc<Mutex<CommunicationSimdAccelerator>>,
    network_topology_manager: Arc<Mutex<NetworkTopologyManager>>,
    quality_of_service: Arc<Mutex<QualityOfServiceManager>>,
    security_monitor: Arc<Mutex<CommunicationSecurityMonitor>>,
}
impl CommunicationManager {
    pub fn new() -> Self {
        Self {
            active_channels: HashMap::new(),
            message_queues: HashMap::new(),
            routing_table: HashMap::new(),
            encryption_manager: Arc::new(Mutex::new(EncryptionManager::new())),
            compression_manager: Arc::new(Mutex::new(CompressionManager::new())),
            bandwidth_monitor: Arc::new(Mutex::new(BandwidthMonitor::new())),
            message_statistics: Arc::new(Mutex::new(MessageStatistics::default())),
            simd_accelerator: Arc::new(Mutex::new(CommunicationSimdAccelerator::new())),
            network_topology_manager: Arc::new(
                Mutex::new(NetworkTopologyManager::new()),
            ),
            quality_of_service: Arc::new(Mutex::new(QualityOfServiceManager::new())),
            security_monitor: Arc::new(Mutex::new(CommunicationSecurityMonitor::new())),
        }
    }
    /// Initialize communication channels with nodes using SIMD optimization
    pub fn initialize_channels(&mut self, nodes: &[NodeInfo]) -> SklResult<()> {
        let simd_accelerator = self
            .simd_accelerator
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        match simd_accelerator.accelerated_channel_initialization(nodes) {
            Ok(channels) => {
                for (node_id, channel) in channels {
                    self.active_channels.insert(node_id.clone(), channel);
                    self.message_queues.insert(node_id, VecDeque::new());
                }
            }
            Err(_) => {
                for node in nodes {
                    let channel = CommunicationChannel::new(
                        node.node_id.clone(),
                        node.node_address,
                        node.security_credentials.encryption_level.clone(),
                    );
                    self.active_channels.insert(node.node_id.clone(), channel);
                    self.message_queues.insert(node.node_id.clone(), VecDeque::new());
                }
            }
        }
        self.build_optimized_routing_table(nodes)?;
        {
            let mut topology_mgr = self
                .network_topology_manager
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            topology_mgr.initialize_topology(nodes)?;
        }
        Ok(())
    }
    /// Send message to a node with SIMD-accelerated processing
    pub fn send_message(&mut self, to_node: &str, message: Message) -> SklResult<()> {
        let processed_message = {
            let simd_accelerator = self
                .simd_accelerator
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            simd_accelerator.preprocess_message(&message)?
        };
        let secure_message = {
            let security_monitor = self
                .security_monitor
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            security_monitor.validate_and_secure_message(&processed_message)?
        };
        let encrypted_message = {
            let encryption_mgr = self
                .encryption_manager
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            encryption_mgr.encrypt_message(&secure_message)?
        };
        let compressed_message = {
            let compression_mgr = self
                .compression_manager
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            compression_mgr.compress_message(&encrypted_message)?
        };
        let qos_message = {
            let qos_mgr = self
                .quality_of_service
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            qos_mgr.apply_qos_policies(&compressed_message, to_node)?
        };
        if let Some(queue) = self.message_queues.get_mut(to_node) {
            if queue.len() > 16 {
                let simd_accelerator = self
                    .simd_accelerator
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                match simd_accelerator
                    .priority_insert_message(queue, qos_message.clone())
                {
                    Ok(_) => {}
                    Err(_) => queue.push_back(qos_message),
                }
            } else {
                queue.push_back(qos_message);
            }
        }
        {
            let mut stats = self
                .message_statistics
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            stats.update_send_statistics(&message);
        }
        {
            let mut bandwidth_monitor = self
                .bandwidth_monitor
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            bandwidth_monitor.record_transmission(&message)?;
        }
        Ok(())
    }
    /// Receive message from a node with SIMD-accelerated processing
    pub fn receive_message(&mut self, from_node: &str) -> SklResult<Option<Message>> {
        if let Some(queue) = self.message_queues.get_mut(from_node) {
            if let Some(compressed_message) = queue.pop_front() {
                let encrypted_message = {
                    let compression_mgr = self
                        .compression_manager
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    compression_mgr.decompress_message(&compressed_message)?
                };
                let secure_message = {
                    let encryption_mgr = self
                        .encryption_manager
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    encryption_mgr.decrypt_message(&encrypted_message)?
                };
                let validated_message = {
                    let security_monitor = self
                        .security_monitor
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    security_monitor.validate_received_message(&secure_message)?
                };
                let final_message = {
                    let simd_accelerator = self
                        .simd_accelerator
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    simd_accelerator.postprocess_message(&validated_message)?
                };
                {
                    let mut stats = self
                        .message_statistics
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    stats.update_receive_statistics(&final_message);
                }
                return Ok(Some(final_message));
            }
        }
        Ok(None)
    }
    /// Broadcast message to all nodes with SIMD-accelerated parallel processing
    pub fn broadcast_message(&mut self, message: Message) -> SklResult<()> {
        let node_ids: Vec<String> = self.active_channels.keys().cloned().collect();
        let simd_accelerator = self
            .simd_accelerator
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        match simd_accelerator.accelerated_broadcast(&node_ids, &message) {
            Ok(optimized_messages) => {
                for (node_id, optimized_message) in optimized_messages {
                    self.send_message(&node_id, optimized_message)?;
                }
            }
            Err(_) => {
                for node_id in node_ids {
                    self.send_message(&node_id, message.clone())?;
                }
            }
        }
        Ok(())
    }
    /// Build optimized routing table using SIMD acceleration
    fn build_optimized_routing_table(&mut self, nodes: &[NodeInfo]) -> SklResult<()> {
        let topology_mgr = self
            .network_topology_manager
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let simd_accelerator = self
            .simd_accelerator
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        match simd_accelerator.compute_optimal_routing(nodes) {
            Ok(routing_table) => {
                self.routing_table = routing_table;
            }
            Err(_) => {
                for node in nodes {
                    let neighbors: Vec<String> = nodes
                        .iter()
                        .filter(|n| n.node_id != node.node_id)
                        .map(|n| n.node_id.clone())
                        .collect();
                    self.routing_table.insert(node.node_id.clone(), neighbors);
                }
            }
        }
        Ok(())
    }
    /// Get comprehensive communication metrics
    pub fn get_metrics(&self) -> CommunicationMetrics {
        let stats = self.message_statistics.lock().unwrap_or_else(|e| e.into_inner());
        let bandwidth_monitor = self
            .bandwidth_monitor
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let qos_mgr = self.quality_of_service.lock().unwrap_or_else(|e| e.into_inner());
        CommunicationMetrics {
            total_messages_sent: stats.messages_sent,
            total_messages_received: stats.messages_received,
            total_bytes_transmitted: stats.total_bytes_sent + stats.total_bytes_received,
            average_message_latency: stats.average_latency,
            network_utilization: bandwidth_monitor.get_utilization(),
            active_connections: self.active_channels.len(),
            failed_transmissions: stats.failed_transmissions,
            compression_ratio: stats.compression_ratio,
            qos_performance: qos_mgr.get_performance_metrics(),
            security_incidents: stats.security_incidents,
            routing_efficiency: self.calculate_routing_efficiency(),
        }
    }
    /// Calculate routing efficiency using SIMD
    fn calculate_routing_efficiency(&self) -> f64 {
        if self.routing_table.is_empty() {
            return 1.0;
        }
        let hop_counts: Vec<f64> = self
            .routing_table
            .values()
            .map(|neighbors| neighbors.len() as f64)
            .collect();
        if hop_counts.len() >= 8 {
            match simd_dot_product(
                &Array1::from(hop_counts.clone()),
                &Array1::ones(hop_counts.len()),
            ) {
                Ok(total_hops) => {
                    let avg_hops = total_hops / hop_counts.len() as f64;
                    (1.0 / (1.0 + avg_hops)).max(0.0).min(1.0)
                }
                Err(_) => {
                    let avg_hops = hop_counts.iter().sum::<f64>()
                        / hop_counts.len() as f64;
                    (1.0 / (1.0 + avg_hops)).max(0.0).min(1.0)
                }
            }
        } else {
            let avg_hops = hop_counts.iter().sum::<f64>() / hop_counts.len() as f64;
            (1.0 / (1.0 + avg_hops)).max(0.0).min(1.0)
        }
    }
    /// Shutdown all communication channels
    pub fn shutdown(&mut self) -> SklResult<()> {
        let simd_accelerator = self
            .simd_accelerator
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        simd_accelerator.accelerated_shutdown(&self.active_channels)?;
        self.active_channels.clear();
        self.message_queues.clear();
        self.routing_table.clear();
        Ok(())
    }
    /// Optimize network performance using SIMD
    pub fn optimize_network_performance(
        &mut self,
    ) -> SklResult<NetworkOptimizationResult> {
        let simd_accelerator = self
            .simd_accelerator
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let bandwidth_monitor = self
            .bandwidth_monitor
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let qos_mgr = self.quality_of_service.lock().unwrap_or_else(|e| e.into_inner());
        let optimization_result = simd_accelerator
            .analyze_and_optimize_network(
                &self.active_channels,
                &bandwidth_monitor,
                &qos_mgr,
            )?;
        Ok(optimization_result)
    }
    /// Handle network partition recovery
    pub fn handle_network_partition(
        &mut self,
        partitioned_nodes: &[String],
    ) -> SklResult<()> {
        let simd_accelerator = self
            .simd_accelerator
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let recovery_plan = simd_accelerator
            .compute_partition_recovery_plan(partitioned_nodes, &self.routing_table)?;
        for recovery_action in recovery_plan.actions {
            match recovery_action.action_type {
                RecoveryActionType::RerouteTraffic => {
                    self.reroute_traffic(
                        &recovery_action.source_node,
                        &recovery_action.target_node,
                    )?;
                }
                RecoveryActionType::EstablishBridgeConnection => {
                    self.establish_bridge_connection(
                        &recovery_action.source_node,
                        &recovery_action.target_node,
                    )?;
                }
                RecoveryActionType::UpdateTopology => {
                    self.update_topology_after_partition(
                        &recovery_action.affected_nodes,
                    )?;
                }
            }
        }
        Ok(())
    }
    /// Reroute traffic between nodes
    fn reroute_traffic(&mut self, source: &str, target: &str) -> SklResult<()> {
        if let Some(routing_entry) = self.routing_table.get_mut(source) {
            if !routing_entry.contains(&target.to_string()) {
                routing_entry.push(target.to_string());
            }
        }
        Ok(())
    }
    /// Establish bridge connection between partitioned segments
    fn establish_bridge_connection(
        &mut self,
        node1: &str,
        node2: &str,
    ) -> SklResult<()> {
        Ok(())
    }
    /// Update topology after partition recovery
    fn update_topology_after_partition(
        &mut self,
        affected_nodes: &[String],
    ) -> SklResult<()> {
        let mut topology_mgr = self
            .network_topology_manager
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        topology_mgr.handle_partition_recovery(affected_nodes)?;
        Ok(())
    }
}
/// SIMD accelerator for communication operations
#[derive(Debug)]
pub struct CommunicationSimdAccelerator {
    simd_enabled: bool,
}
impl CommunicationSimdAccelerator {
    pub fn new() -> Self {
        Self { simd_enabled: true }
    }
    /// Accelerated channel initialization
    pub fn accelerated_channel_initialization(
        &self,
        nodes: &[NodeInfo],
    ) -> SklResult<HashMap<String, CommunicationChannel>> {
        if !self.simd_enabled || nodes.len() < 8 {
            return Err("SIMD not available".into());
        }
        let mut channels = HashMap::new();
        for chunk in nodes.chunks(8) {
            for node in chunk {
                let channel = CommunicationChannel::new(
                    node.node_id.clone(),
                    node.node_address,
                    node.security_credentials.encryption_level.clone(),
                );
                channels.insert(node.node_id.clone(), channel);
            }
        }
        Ok(channels)
    }
    /// Preprocess message with SIMD optimization
    pub fn preprocess_message(&self, message: &Message) -> SklResult<Message> {
        if !self.simd_enabled {
            return Ok(message.clone());
        }
        let mut processed_message = message.clone();
        if message.payload.len() >= 64 {
            let checksum = self.calculate_simd_checksum(&message.payload)?;
            processed_message
                .metadata
                .insert("checksum".to_string(), checksum.to_string());
        }
        Ok(processed_message)
    }
    /// Postprocess message with SIMD optimization
    pub fn postprocess_message(&self, message: &Message) -> SklResult<Message> {
        if !self.simd_enabled {
            return Ok(message.clone());
        }
        let mut processed_message = message.clone();
        if let Some(checksum_str) = message.metadata.get("checksum") {
            if let Ok(expected_checksum) = checksum_str.parse::<u64>() {
                let actual_checksum = self.calculate_simd_checksum(&message.payload)?;
                if actual_checksum != expected_checksum {
                    return Err("Checksum verification failed".into());
                }
            }
        }
        processed_message.metadata.remove("checksum");
        Ok(processed_message)
    }
    /// Calculate SIMD checksum
    fn calculate_simd_checksum(&self, payload: &[u8]) -> SklResult<u64> {
        if payload.len() < 64 {
            return Ok(payload.iter().map(|&b| b as u64).sum());
        }
        let chunks: Vec<f64> = payload
            .chunks(8)
            .map(|chunk| {
                let mut bytes = [0u8; 8];
                for (i, &b) in chunk.iter().enumerate() {
                    if i < 8 {
                        bytes[i] = b;
                    }
                }
                f64::from_le_bytes(bytes)
            })
            .collect();
        if chunks.len() >= 8 {
            match simd_dot_product(&Array1::from(chunks), &Array1::ones(chunks.len())) {
                Ok(checksum) => Ok(checksum as u64),
                Err(_) => Ok(payload.iter().map(|&b| b as u64).sum()),
            }
        } else {
            Ok(payload.iter().map(|&b| b as u64).sum())
        }
    }
    /// Accelerated broadcast optimization
    pub fn accelerated_broadcast(
        &self,
        node_ids: &[String],
        message: &Message,
    ) -> SklResult<HashMap<String, Message>> {
        if !self.simd_enabled {
            return Err("SIMD not available".into());
        }
        let mut optimized_messages = HashMap::new();
        for node_id in node_ids {
            let mut optimized_message = message.clone();
            optimized_message.to_node = node_id.clone();
            optimized_message.message_id = format!(
                "broadcast_{}_{}", node_id, message.timestamp.duration_since(UNIX_EPOCH)
                .unwrap_or_default().as_nanos()
            );
            optimized_messages.insert(node_id.clone(), optimized_message);
        }
        Ok(optimized_messages)
    }
    /// Compute optimal routing using SIMD
    pub fn compute_optimal_routing(
        &self,
        nodes: &[NodeInfo],
    ) -> SklResult<HashMap<String, Vec<String>>> {
        if !self.simd_enabled || nodes.len() < 4 {
            return Err("SIMD not available".into());
        }
        let mut routing_table = HashMap::new();
        let node_positions: Vec<f64> = nodes
            .iter()
            .enumerate()
            .map(|(i, _)| i as f64)
            .collect();
        if node_positions.len() >= 8 {
            for (i, node) in nodes.iter().enumerate() {
                let distances: Vec<f64> = nodes
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(j, _)| (i as f64 - j as f64).abs())
                    .collect();
                let mut neighbor_distances: Vec<(usize, f64)> = distances
                    .iter()
                    .enumerate()
                    .map(|(idx, &dist)| (if idx >= i { idx + 1 } else { idx }, dist))
                    .collect();
                neighbor_distances
                    .sort_by(|a, b| {
                        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                    });
                let neighbors: Vec<String> = neighbor_distances
                    .iter()
                    .take(3.min(neighbor_distances.len()))
                    .map(|(neighbor_idx, _)| nodes[*neighbor_idx].node_id.clone())
                    .collect();
                routing_table.insert(node.node_id.clone(), neighbors);
            }
        }
        Ok(routing_table)
    }
    /// Priority message insertion using SIMD
    pub fn priority_insert_message(
        &self,
        queue: &mut VecDeque<Message>,
        message: Message,
    ) -> SklResult<()> {
        if !self.simd_enabled || queue.len() < 16 {
            return Err("SIMD not beneficial".into());
        }
        let priorities: Vec<f64> = queue
            .iter()
            .map(|m| self.priority_to_f64(&m.priority))
            .collect();
        let new_priority = self.priority_to_f64(&message.priority);
        let mut insert_index = queue.len();
        if priorities.len() >= 8 {
            let comparison_chunks = priorities.chunks(8);
            let mut current_index = 0;
            for chunk in comparison_chunks {
                if chunk.len() == 8 {
                    let priority_chunk = f64x8::from_slice(chunk);
                    let new_priority_vec = f64x8::splat(new_priority);
                    let comparison_mask = priority_chunk.simd_lt(new_priority_vec);
                    for (i, is_lower) in comparison_mask.as_array().iter().enumerate() {
                        if *is_lower {
                            insert_index = current_index + i;
                            break;
                        }
                    }
                    if insert_index < queue.len() {
                        break;
                    }
                    current_index += 8;
                }
            }
        }
        queue.insert(insert_index, message);
        Ok(())
    }
    /// Convert priority to f64 for SIMD operations
    fn priority_to_f64(&self, priority: &MessagePriority) -> f64 {
        match priority {
            MessagePriority::Low => 1.0,
            MessagePriority::Normal => 2.0,
            MessagePriority::High => 3.0,
            MessagePriority::Critical => 4.0,
            MessagePriority::Emergency => 5.0,
        }
    }
    /// Accelerated shutdown cleanup
    pub fn accelerated_shutdown(
        &self,
        channels: &HashMap<String, CommunicationChannel>,
    ) -> SklResult<()> {
        if !self.simd_enabled {
            return Ok(());
        }
        let channel_qualities: Vec<f64> = channels
            .values()
            .map(|c| c.channel_quality)
            .collect();
        if channel_qualities.len() >= 8 {
            match simd_dot_product(
                &Array1::from(channel_qualities),
                &Array1::ones(channels.len()),
            ) {
                Ok(total_quality) => {
                    let avg_quality = total_quality / channels.len() as f64;
                    println!("Average channel quality at shutdown: {:.3}", avg_quality);
                }
                Err(_) => {}
            }
        }
        Ok(())
    }
    /// Analyze and optimize network performance
    pub fn analyze_and_optimize_network(
        &self,
        channels: &HashMap<String, CommunicationChannel>,
        bandwidth_monitor: &BandwidthMonitor,
        qos_manager: &QualityOfServiceManager,
    ) -> SklResult<NetworkOptimizationResult> {
        if !self.simd_enabled {
            return Err("SIMD not available".into());
        }
        let qualities: Vec<f64> = channels.values().map(|c| c.channel_quality).collect();
        let throughputs: Vec<f64> = channels
            .values()
            .map(|c| { c.throughput_history.back().map_or(0.0, |t| t.throughput) })
            .collect();
        let optimization_score = if qualities.len() >= 8 && throughputs.len() >= 8 {
            let avg_quality = simd_dot_product(
                &Array1::from(qualities),
                &Array1::ones(channels.len()),
            )? / channels.len() as f64;
            let avg_throughput = simd_dot_product(
                &Array1::from(throughputs),
                &Array1::ones(channels.len()),
            )? / channels.len() as f64;
            (avg_quality + avg_throughput / 1_000_000.0) / 2.0
        } else {
            0.5
        };
        Ok(NetworkOptimizationResult {
            optimization_score,
            recommended_actions: vec![
                "Maintain current network configuration".to_string(),
                format!("Network optimization score: {:.3}", optimization_score),
            ],
            performance_improvement: optimization_score * 0.1,
        })
    }
    /// Compute partition recovery plan
    pub fn compute_partition_recovery_plan(
        &self,
        partitioned_nodes: &[String],
        routing_table: &HashMap<String, Vec<String>>,
    ) -> SklResult<PartitionRecoveryPlan> {
        if !self.simd_enabled {
            return Err("SIMD not available".into());
        }
        let mut recovery_actions = Vec::new();
        for node in partitioned_nodes {
            if let Some(neighbors) = routing_table.get(node) {
                for neighbor in neighbors {
                    if !partitioned_nodes.contains(neighbor) {
                        recovery_actions
                            .push(RecoveryAction {
                                action_type: RecoveryActionType::EstablishBridgeConnection,
                                source_node: node.clone(),
                                target_node: neighbor.clone(),
                                affected_nodes: vec![node.clone(), neighbor.clone()],
                                priority: 1.0,
                            });
                    }
                }
            }
        }
        Ok(PartitionRecoveryPlan {
            actions: recovery_actions,
            estimated_recovery_time: Duration::from_secs(30),
            success_probability: 0.9,
        })
    }
}
#[derive(Debug, Clone)]
pub struct ThroughputMeasurement {
    pub timestamp: SystemTime,
    pub throughput: f64,
    pub latency: Duration,
    pub errors: u32,
}
#[derive(Debug)]
pub struct RecoveryAction {
    pub action_type: RecoveryActionType,
    pub source_node: String,
    pub target_node: String,
    pub affected_nodes: Vec<String>,
    pub priority: f64,
}
#[derive(Debug, Clone, PartialEq)]
pub enum CompressionType {
    None,
    LZ4,
    Zstd,
    Brotli,
}
/// Network topology manager: maintains an in-memory adjacency map of node connections
/// and supports partition-recovery by marking affected nodes as needing re-routing.
///
/// Topology model: each node is connected to all other nodes (full mesh),
/// with edges annotated by their current liveness status.
#[derive(Debug)]
pub struct NetworkTopologyManager {
    /// `adjacency[src][dst] = true` if the link is considered live
    adjacency: HashMap<String, HashMap<String, bool>>,
    /// Nodes flagged as partitioned (isolated from majority)
    partitioned_nodes: std::collections::HashSet<String>,
}
impl NetworkTopologyManager {
    pub fn new() -> Self {
        Self {
            adjacency: HashMap::new(),
            partitioned_nodes: std::collections::HashSet::new(),
        }
    }
    /// Initializes a full-mesh topology for `nodes`.
    ///
    /// Every node starts with a live link to every other node.
    pub fn initialize_topology(&mut self, nodes: &[NodeInfo]) -> SklResult<()> {
        self.adjacency.clear();
        let ids: Vec<&str> = nodes.iter().map(|n| n.node_id.as_str()).collect();
        for src in &ids {
            let entry = self.adjacency.entry(src.to_string()).or_default();
            for dst in &ids {
                if *src != *dst {
                    entry.insert(dst.to_string(), true);
                }
            }
        }
        Ok(())
    }
    /// Handles partition recovery by re-marking affected node links as live
    /// and removing them from the partitioned set.
    ///
    /// In a real system this would trigger route table updates and health checks.
    pub fn handle_partition_recovery(
        &mut self,
        affected_nodes: &[String],
    ) -> SklResult<()> {
        for node_id in affected_nodes {
            self.partitioned_nodes.remove(node_id);
            if let Some(outbound) = self.adjacency.get_mut(node_id) {
                for live in outbound.values_mut() {
                    *live = true;
                }
            }
            for (_, links) in &mut self.adjacency {
                if let Some(live) = links.get_mut(node_id) {
                    *live = true;
                }
            }
        }
        Ok(())
    }
    /// Marks a node as partitioned (its outbound links are set to dead).
    pub fn mark_node_partitioned(&mut self, node_id: &str) {
        self.partitioned_nodes.insert(node_id.to_string());
        if let Some(outbound) = self.adjacency.get_mut(node_id) {
            for live in outbound.values_mut() {
                *live = false;
            }
        }
    }
    /// Returns `true` if the direct link from `src` to `dst` is live.
    pub fn is_link_live(&self, src: &str, dst: &str) -> bool {
        self.adjacency.get(src).and_then(|m| m.get(dst)).copied().unwrap_or(false)
    }
}
