//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::types_3::{EnergyAwareOptimizer, LoadBalancingPerformanceMonitor, LoadImbalanceMetrics, ScalingAction, ScalingMetrics, UtilizationMetrics};
use super::types::{AdaptiveLoadController, AutoScalingManager, LoadBalancingStrategy, LoadMeasurement, LoadPredictionEngine, ScalingRecommendation, TaskPriority, WorkerLoad, WorkloadDistributionRequest, WorkloadType};

#[cfg(test)]
mod advanced_lb_tests {
    use super::*;
    fn make_worker(id: &str, load_pct: f64, max: f64) -> (String, WorkerLoad) {
        let mut w = WorkerLoad::new(id);
        w.max_capacity = max;
        w.current_load = load_pct * max;
        w.cpu_utilization = load_pct;
        (id.to_string(), w)
    }
    #[test]
    fn test_load_prediction_empty_history() {
        let engine = LoadPredictionEngine::new();
        let workers: HashMap<_, _> = [make_worker("w1", 0.5, 100.0)]
            .into_iter()
            .collect();
        let preds = engine
            .predict_future_loads(&workers, Duration::from_secs(300))
            .expect("predict failed");
        let pred = preds["w1"];
        assert!(pred >= 0.0 && pred <= 100.0, "prediction out of range: {pred}");
    }
    #[test]
    fn test_load_prediction_trending_up() {
        let mut engine = LoadPredictionEngine::new();
        let mut w = WorkerLoad::new("w1");
        w.max_capacity = 100.0;
        w.current_load = 60.0;
        for i in 0..10usize {
            w.load_history
                .push_back(LoadMeasurement {
                    timestamp: SystemTime::now(),
                    load_value: 30.0 + i as f64 * 4.0,
                    utilization_metrics: UtilizationMetrics {
                        cpu: 0.3,
                        memory: 0.3,
                        network: 0.2,
                    },
                });
        }
        let mut workers = HashMap::new();
        workers.insert("w1".to_string(), w);
        let preds = engine
            .predict_future_loads(&workers, Duration::from_secs(600))
            .expect("predict failed");
        assert!(
            preds["w1"] >= 55.0, "expected trending-up prediction, got {}", preds["w1"]
        );
    }
    #[test]
    fn test_adaptive_controller_selects_capacity_aware_when_overloaded() {
        let ctrl = AdaptiveLoadController::new();
        let workers: HashMap<_, _> = [
            make_worker("w1", 0.95, 100.0),
            make_worker("w2", 0.50, 100.0),
        ]
            .into_iter()
            .collect();
        let request = WorkloadDistributionRequest {
            request_id: "r1".to_string(),
            workload_type: WorkloadType::MachineLearning,
            total_computational_load: 10.0,
            memory_requirements: 0.0,
            network_requirements: 0.0,
            priority: TaskPriority::Normal,
            deadline: None,
            parallelizable: false,
            resource_constraints: Vec::new(),
            affinity_preferences: Vec::new(),
        };
        let strategy = ctrl
            .select_optimal_strategy(&workers, &request)
            .expect("select failed");
        assert_eq!(strategy, LoadBalancingStrategy::CapacityAware);
    }
    #[test]
    fn test_adaptive_controller_selects_energy_aware_when_underutilized() {
        let ctrl = AdaptiveLoadController::new();
        let workers: HashMap<_, _> = [
            make_worker("w1", 0.10, 100.0),
            make_worker("w2", 0.15, 100.0),
        ]
            .into_iter()
            .collect();
        let request = WorkloadDistributionRequest {
            request_id: "r2".to_string(),
            workload_type: WorkloadType::MachineLearning,
            total_computational_load: 5.0,
            memory_requirements: 0.0,
            network_requirements: 0.0,
            priority: TaskPriority::Normal,
            deadline: None,
            parallelizable: false,
            resource_constraints: Vec::new(),
            affinity_preferences: Vec::new(),
        };
        let strategy = ctrl
            .select_optimal_strategy(&workers, &request)
            .expect("select failed");
        assert_eq!(strategy, LoadBalancingStrategy::EnergyAware);
    }
    #[test]
    fn test_auto_scaler_scale_up_decision() {
        let mut scaler = AutoScalingManager::new();
        scaler.current_worker_count = 4;
        let metrics = ScalingMetrics {
            average_utilization: 0.85,
            peak_utilization: 0.95,
            scaling_recommendation: ScalingRecommendation::ScaleUp,
        };
        let decision = scaler.make_scaling_decision(metrics).expect("decision failed");
        assert!(matches!(decision.action, ScalingAction::ScaleUp));
        assert!(decision.target_count > 4, "should scale up beyond 4 workers");
    }
    #[test]
    fn test_auto_scaler_no_action_at_moderate_utilization() {
        let mut scaler = AutoScalingManager::new();
        scaler.current_worker_count = 4;
        let metrics = ScalingMetrics {
            average_utilization: 0.50,
            peak_utilization: 0.70,
            scaling_recommendation: ScalingRecommendation::NoAction,
        };
        let decision = scaler.make_scaling_decision(metrics).expect("decision failed");
        assert!(matches!(decision.action, ScalingAction::NoAction));
    }
    #[test]
    fn test_energy_optimizer_costs() {
        let optimizer = EnergyAwareOptimizer::new();
        let workers: HashMap<_, _> = [
            make_worker("light", 0.10, 100.0),
            make_worker("heavy", 0.90, 100.0),
        ]
            .into_iter()
            .collect();
        let costs = optimizer.calculate_energy_costs(&workers).expect("costs failed");
        assert!(
            costs["light"] < costs["heavy"], "light worker should cost less: {} vs {}",
            costs["light"], costs["heavy"]
        );
    }
    #[test]
    fn test_performance_monitor_update() {
        let mut monitor = LoadBalancingPerformanceMonitor::new();
        let workers: HashMap<_, _> = [
            make_worker("w1", 0.60, 100.0),
            make_worker("w2", 0.40, 100.0),
        ]
            .into_iter()
            .collect();
        let imbalance = LoadImbalanceMetrics {
            max_imbalance: 0.2,
            average_imbalance: 0.1,
            coefficient_of_variation: 0.15,
            gini_coefficient: 0.1,
        };
        monitor.update_performance_metrics(&workers, &imbalance).expect("update failed");
        let metrics = monitor.get_current_performance_metrics();
        assert!(metrics.error_rate >= 0.0 && metrics.error_rate <= 1.0);
        assert!(metrics.throughput >= 0.0);
    }
    #[test]
    fn test_ols_slope_positive_for_increasing_series() {
        let engine = LoadPredictionEngine::new();
        let samples = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let slope = engine.compute_slope(&samples);
        assert!(slope > 0.0, "slope should be positive, got {slope}");
    }
}
