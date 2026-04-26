//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use super::node_management::{NodeInfo, EncryptionLevel};

use super::types_2::{CompressionType, NetworkTopologyManager};
use super::types::{AdaptiveQoSController, BandwidthMeasurement, BandwidthSimdAnalyzer, CommunicationSecurityMonitor, Message, MessagePriority, MessageType, SimdCompressionEngine};

#[cfg(test)]
mod comm_layer_tests {
    use super::*;
    fn make_measurement(throughput: f64, bytes: u64) -> BandwidthMeasurement {
        BandwidthMeasurement {
            timestamp: SystemTime::now(),
            bytes_transmitted: bytes,
            latency: Duration::from_millis(10),
            throughput,
        }
    }
    #[test]
    fn test_bandwidth_analyzer_empty_history() {
        let analyzer = BandwidthSimdAnalyzer::new();
        let history = VecDeque::new();
        let analysis = analyzer
            .analyze_bandwidth_patterns(&history)
            .expect("analyze failed");
        assert_eq!(analysis.current_utilization, 0.0);
        assert_eq!(analysis.average_bandwidth, 0.0);
    }
    #[test]
    fn test_bandwidth_analyzer_computes_average() {
        let analyzer = BandwidthSimdAnalyzer::new();
        let mut history = VecDeque::new();
        for &tp in &[100.0, 200.0, 300.0] {
            history.push_back(make_measurement(tp, tp as u64));
        }
        let analysis = analyzer
            .analyze_bandwidth_patterns(&history)
            .expect("analyze failed");
        assert!(
            (analysis.average_bandwidth - 200.0).abs() < 1.0, "expected ~200.0 got {}",
            analysis.average_bandwidth
        );
    }
    #[test]
    fn test_bandwidth_prediction_trending_up() {
        let analyzer = BandwidthSimdAnalyzer::new();
        let mut history = VecDeque::new();
        for i in 0..10u64 {
            history
                .push_back(
                    make_measurement(1_000.0 * (i as f64 + 1.0), 10_000 * (i + 1)),
                );
        }
        let pred = analyzer
            .predict_bandwidth_trends(&history, Duration::from_secs(300))
            .expect("predict failed");
        assert!(pred.confidence > 0.0 && pred.confidence <= 1.0);
        assert!(pred.predicted_utilization >= 0.0);
    }
    #[test]
    fn test_adaptive_qos_congestion_event_counting() {
        let mut ctrl = AdaptiveQoSController::new();
        ctrl.handle_congestion(0.5).expect("congestion failed");
        assert_eq!(ctrl.get_congestion_event_count(), 0);
        for _ in 0..5 {
            ctrl.handle_congestion(0.95).expect("congestion failed");
        }
        assert!(ctrl.get_congestion_event_count() > 0, "expected congestion events");
    }
    #[test]
    fn test_network_topology_full_mesh_initialization() {
        let mut manager = NetworkTopologyManager::new();
        let nodes: Vec<NodeInfo> = (0..3)
            .map(|i| NodeInfo {
                node_id: format!("node{i}"),
                node_address: "127.0.0.1:0".parse().expect("parse addr"),
                node_type: super::super::node_management::NodeType::Worker,
                status: super::super::node_management::NodeStatus::Active,
                performance_metrics: super::super::node_management::NodePerformanceMetrics {
                    cpu_utilization: 0.1,
                    memory_utilization: 0.2,
                    network_utilization: 0.1,
                    error_rate: 0.0,
                    average_response_time: Duration::from_millis(10),
                    throughput: 1000.0,
                },
                security_credentials: super::super::node_management::SecurityCredentials {
                    node_certificate: Vec::new(),
                    encryption_level: EncryptionLevel::None,
                    authentication_token: Vec::new(),
                },
                metadata: HashMap::new(),
                last_heartbeat: SystemTime::now(),
                failure_count: 0,
            })
            .collect();
        manager.initialize_topology(&nodes).expect("init failed");
        assert!(manager.is_link_live("node0", "node1"));
        assert!(manager.is_link_live("node1", "node2"));
        assert!(! manager.is_link_live("node0", "node0"), "no self-loops");
    }
    #[test]
    fn test_network_topology_partition_and_recovery() {
        let mut manager = NetworkTopologyManager::new();
        let nodes: Vec<NodeInfo> = (0..2)
            .map(|i| NodeInfo {
                node_id: format!("n{i}"),
                node_address: "127.0.0.1:0".parse().expect("parse addr"),
                node_type: super::super::node_management::NodeType::Worker,
                status: super::super::node_management::NodeStatus::Active,
                performance_metrics: super::super::node_management::NodePerformanceMetrics {
                    cpu_utilization: 0.1,
                    memory_utilization: 0.2,
                    network_utilization: 0.1,
                    error_rate: 0.0,
                    average_response_time: Duration::from_millis(10),
                    throughput: 1000.0,
                },
                security_credentials: super::super::node_management::SecurityCredentials {
                    node_certificate: Vec::new(),
                    encryption_level: EncryptionLevel::None,
                    authentication_token: Vec::new(),
                },
                metadata: HashMap::new(),
                last_heartbeat: SystemTime::now(),
                failure_count: 0,
            })
            .collect();
        manager.initialize_topology(&nodes).expect("init failed");
        manager.mark_node_partitioned("n0");
        assert!(! manager.is_link_live("n0", "n1"), "partitioned link should be dead");
        manager.handle_partition_recovery(&["n0".to_string()]).expect("recovery failed");
        assert!(manager.is_link_live("n0", "n1"), "link should be live after recovery");
    }
    #[test]
    fn test_simd_compression_engine_passthrough_for_none() {
        let engine = SimdCompressionEngine::new();
        let data = b"hello world".to_vec();
        let compressed = engine
            .simd_compress(&data, &CompressionType::None)
            .expect("compress failed");
        let decompressed = engine
            .simd_decompress(&compressed, &CompressionType::None)
            .expect("decompress failed");
        assert_eq!(data, decompressed, "pass-through roundtrip failed");
    }
    #[test]
    fn test_security_monitor_rejects_oversized_message() {
        let monitor = CommunicationSecurityMonitor::new();
        let oversized_payload = vec![0u8; 65 * 1024 * 1024];
        let msg = Message {
            message_id: "test-msg".to_string(),
            source_node: "src".to_string(),
            destination_node: "dst".to_string(),
            message_type: MessageType::Data,
            payload: oversized_payload,
            priority: MessagePriority::Normal,
            timestamp: SystemTime::now(),
            sequence_number: 0,
            compression_type: CompressionType::None,
            requires_acknowledgment: false,
        };
        let result = monitor.validate_and_secure_message(&msg);
        assert!(result.is_err(), "oversized message should be rejected");
        assert_eq!(monitor.get_incident_count(), 1);
    }
}
