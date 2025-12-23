//! Energy efficiency benchmarking for SIMD operations
//!
//! This module provides tools to measure power consumption and energy efficiency
//! of SIMD operations, essential for mobile and edge deployment optimization.

#[cfg(feature = "no-std")]
extern crate alloc;

#[cfg(feature = "no-std")]
use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

// Conditional imports for std/no-std compatibility
#[cfg(feature = "no-std")]
use alloc::collections::BTreeMap as HashMap;
#[cfg(not(feature = "no-std"))]
use std::{collections::HashMap, string::ToString, time::Duration};

#[cfg(feature = "no-std")]
use core::time::Duration;
#[cfg(not(feature = "no-std"))]
use std::time::Instant;

// No-std compatible time implementation
#[cfg(feature = "no-std")]
#[derive(Debug, Clone, Copy)]
pub struct Instant {
    // Mock timestamp for no-std compatibility
    mock_time: u64,
}

#[cfg(feature = "no-std")]
impl Instant {
    pub fn now() -> Self {
        // In no-std, we can't get actual time, so we use a mock
        // This could be replaced with platform-specific time sources
        static mut MOCK_TIME: u64 = 0;
        unsafe {
            MOCK_TIME += 1;
            Self {
                mock_time: MOCK_TIME,
            }
        }
    }

    pub fn elapsed(&self) -> Duration {
        // Return a minimal duration for no-std compatibility
        Duration::from_nanos(1000)
    }
}

// Thread sleep functionality
#[cfg(not(feature = "no-std"))]
use std::thread;

#[cfg(feature = "no-std")]
mod thread {
    use core::time::Duration;

    pub fn sleep(_duration: Duration) {
        // No-op in no-std environment
        // Could be replaced with platform-specific delay functions
    }
}

/// Energy measurement result
#[derive(Debug, Clone)]
pub struct EnergyMeasurement {
    pub duration: Duration,
    pub estimated_power_watts: f64,
    pub energy_joules: f64,
    pub operations_per_joule: f64,
    pub gflops_per_watt: f64,
}

/// Energy efficiency metrics for different operations
#[derive(Debug, Clone)]
pub struct EnergyEfficiencyMetrics {
    pub operation_name: String,
    pub scalar_energy: EnergyMeasurement,
    pub simd_energy: EnergyMeasurement,
    pub energy_efficiency_ratio: f64,
    pub performance_per_watt_ratio: f64,
}

/// Thermal state monitoring
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThermalState {
    Cool,
    Warm,
    Hot,
    Throttling,
}

/// Energy profiler for SIMD operations
pub struct EnergyProfiler {
    baseline_power: f64,
    cpu_tdp: f64,
    measurement_duration: Duration,
    thermal_monitoring: bool,
}

impl EnergyProfiler {
    /// Create a new energy profiler
    pub fn new(cpu_tdp: f64) -> Self {
        Self {
            baseline_power: 0.0,
            cpu_tdp,
            measurement_duration: Duration::from_millis(100),
            thermal_monitoring: true,
        }
    }

    /// Set the measurement duration for energy tests
    pub fn set_measurement_duration(&mut self, duration: Duration) {
        self.measurement_duration = duration;
    }

    /// Enable or disable thermal monitoring
    pub fn set_thermal_monitoring(&mut self, enabled: bool) {
        self.thermal_monitoring = enabled;
    }

    /// Calibrate baseline power consumption
    pub fn calibrate_baseline(&mut self) -> Result<(), String> {
        let start = Instant::now();

        // Idle for calibration period
        thread::sleep(self.measurement_duration);

        let _elapsed = start.elapsed();

        // Estimate baseline power (simplified model)
        // In a real implementation, this would use platform-specific APIs
        self.baseline_power = self.estimate_idle_power();

        Ok(())
    }

    /// Estimate idle power consumption
    fn estimate_idle_power(&self) -> f64 {
        // Simplified model: baseline is typically 20-30% of TDP
        self.cpu_tdp * 0.25
    }

    /// Estimate current CPU power consumption
    fn estimate_cpu_power(&self, cpu_utilization: f64) -> f64 {
        // Simplified power model: P = P_idle + (P_max - P_idle) * utilization^1.5
        let p_idle = self.baseline_power;
        let p_max = self.cpu_tdp;
        p_idle + (p_max - p_idle) * cpu_utilization.powf(1.5)
    }

    /// Get current thermal state
    fn get_thermal_state(&self) -> ThermalState {
        // In a real implementation, this would check:
        // - CPU temperature sensors
        // - Frequency scaling state
        // - Thermal throttling indicators

        // Simplified simulation based on system load
        let load = self.estimate_system_load();

        if load < 0.3 {
            ThermalState::Cool
        } else if load < 0.6 {
            ThermalState::Warm
        } else if load < 0.9 {
            ThermalState::Hot
        } else {
            ThermalState::Throttling
        }
    }

    /// Estimate system load
    fn estimate_system_load(&self) -> f64 {
        // Simplified load estimation
        // In a real implementation, this would use system APIs
        0.5 // Assume moderate load
    }

    /// Measure energy consumption of a function
    pub fn measure_energy<F>(
        &self,
        _operation_name: &str,
        operation_count: u64,
        func: F,
    ) -> EnergyMeasurement
    where
        F: FnOnce(),
    {
        let _initial_thermal = if self.thermal_monitoring {
            self.get_thermal_state()
        } else {
            ThermalState::Cool
        };

        let start_time = Instant::now();

        // Estimate CPU utilization during operation
        let estimated_utilization = 0.8; // Assume high utilization during SIMD ops

        // Execute the operation
        func();

        let duration = start_time.elapsed();

        // Estimate power consumption
        let estimated_power = self.estimate_cpu_power(estimated_utilization);

        // Calculate energy
        let energy_joules = estimated_power * duration.as_secs_f64();

        // Calculate efficiency metrics
        let operations_per_joule = operation_count as f64 / energy_joules;
        let gflops_per_watt =
            (operation_count as f64 / 1e9) / (duration.as_secs_f64() * estimated_power);

        EnergyMeasurement {
            duration,
            estimated_power_watts: estimated_power,
            energy_joules,
            operations_per_joule,
            gflops_per_watt,
        }
    }

    /// Compare energy efficiency of scalar vs SIMD operations
    pub fn compare_energy_efficiency<F1, F2>(
        &self,
        operation_name: &str,
        operation_count: u64,
        scalar_func: F1,
        simd_func: F2,
    ) -> EnergyEfficiencyMetrics
    where
        F1: FnOnce(),
        F2: FnOnce(),
    {
        // Measure scalar implementation
        let scalar_energy = self.measure_energy(
            &format!("{}_scalar", operation_name),
            operation_count,
            scalar_func,
        );

        // Allow system to cool down between measurements
        thread::sleep(Duration::from_millis(50));

        // Measure SIMD implementation
        let simd_energy = self.measure_energy(
            &format!("{}_simd", operation_name),
            operation_count,
            simd_func,
        );

        // Calculate efficiency ratios
        let energy_efficiency_ratio = scalar_energy.energy_joules / simd_energy.energy_joules;
        let performance_per_watt_ratio =
            simd_energy.gflops_per_watt / scalar_energy.gflops_per_watt;

        EnergyEfficiencyMetrics {
            operation_name: operation_name.to_string(),
            scalar_energy,
            simd_energy,
            energy_efficiency_ratio,
            performance_per_watt_ratio,
        }
    }

    /// Benchmark energy efficiency across different vector sizes
    pub fn benchmark_vector_sizes<F>(
        &self,
        operation_name: &str,
        sizes: &[usize],
        operation_factory: F,
    ) -> Vec<(usize, EnergyMeasurement)>
    where
        F: Fn(usize) -> Box<dyn FnOnce()>,
    {
        let mut results = Vec::new();

        for &size in sizes {
            let operation = operation_factory(size);
            let measurement =
                self.measure_energy(&format!("{}_{}", operation_name, size), size as u64, || {
                    operation()
                });
            results.push((size, measurement));

            // Cool down between measurements
            thread::sleep(Duration::from_millis(25));
        }

        results
    }

    /// Generate energy efficiency report
    pub fn generate_report(&self, metrics: &[EnergyEfficiencyMetrics]) -> String {
        let mut report = String::new();
        report.push_str("Energy Efficiency Report\n");
        report.push_str("========================\n\n");

        for metric in metrics {
            report.push_str(&format!("Operation: {}\n", metric.operation_name));
            report.push_str(&format!(
                "  Scalar Energy: {:.3} J\n",
                metric.scalar_energy.energy_joules
            ));
            report.push_str(&format!(
                "  SIMD Energy: {:.3} J\n",
                metric.simd_energy.energy_joules
            ));
            report.push_str(&format!(
                "  Energy Efficiency Ratio: {:.2}x\n",
                metric.energy_efficiency_ratio
            ));
            report.push_str(&format!(
                "  Scalar GFLOPS/W: {:.2}\n",
                metric.scalar_energy.gflops_per_watt
            ));
            report.push_str(&format!(
                "  SIMD GFLOPS/W: {:.2}\n",
                metric.simd_energy.gflops_per_watt
            ));
            report.push_str(&format!(
                "  Performance/Watt Ratio: {:.2}x\n",
                metric.performance_per_watt_ratio
            ));
            report.push('\n');
        }

        // Summary statistics
        let avg_energy_ratio: f64 = metrics
            .iter()
            .map(|m| m.energy_efficiency_ratio)
            .sum::<f64>()
            / metrics.len() as f64;
        let avg_perf_per_watt_ratio: f64 = metrics
            .iter()
            .map(|m| m.performance_per_watt_ratio)
            .sum::<f64>()
            / metrics.len() as f64;

        report.push_str("Summary:\n");
        report.push_str(&format!(
            "  Average Energy Efficiency: {:.2}x\n",
            avg_energy_ratio
        ));
        report.push_str(&format!(
            "  Average Performance per Watt: {:.2}x\n",
            avg_perf_per_watt_ratio
        ));

        report
    }
}

/// Power efficiency optimizer
pub struct PowerEfficiencyOptimizer {
    thermal_threshold: f64,
    performance_scaling: f64,
    energy_budget: f64,
}

impl PowerEfficiencyOptimizer {
    /// Create a new power efficiency optimizer
    pub fn new(thermal_threshold: f64, energy_budget: f64) -> Self {
        Self {
            thermal_threshold,
            performance_scaling: 1.0,
            energy_budget,
        }
    }

    /// Adjust SIMD operation parameters based on thermal state
    pub fn optimize_for_thermal_state(&mut self, thermal_state: ThermalState) -> f64 {
        match thermal_state {
            ThermalState::Cool => {
                self.performance_scaling = 1.0; // Full performance
            }
            ThermalState::Warm => {
                self.performance_scaling = 0.9; // Slight reduction
            }
            ThermalState::Hot => {
                self.performance_scaling = 0.7; // Significant reduction
            }
            ThermalState::Throttling => {
                self.performance_scaling = 0.5; // Emergency throttling
            }
        }
        self.performance_scaling
    }

    /// Recommend optimal SIMD width based on energy constraints
    pub fn recommend_simd_width(
        &self,
        available_widths: &[usize],
        energy_per_width: &[f64],
    ) -> usize {
        assert_eq!(available_widths.len(), energy_per_width.len());

        let mut best_width = available_widths[0];
        let mut best_efficiency = 0.0;

        for (i, &width) in available_widths.iter().enumerate() {
            let energy = energy_per_width[i];
            if energy <= self.energy_budget {
                let efficiency = width as f64 / energy;
                if efficiency > best_efficiency {
                    best_efficiency = efficiency;
                    best_width = width;
                }
            }
        }

        best_width
    }

    /// Generate power optimization recommendations
    pub fn generate_recommendations(&self, measurements: &[EnergyMeasurement]) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Analyze average power consumption
        let avg_power: f64 = measurements
            .iter()
            .map(|m| m.estimated_power_watts)
            .sum::<f64>()
            / measurements.len() as f64;

        if avg_power > self.thermal_threshold {
            recommendations.push("Consider reducing SIMD width to manage thermal load".to_string());
            recommendations.push("Implement adaptive performance scaling".to_string());
        }

        // Analyze energy efficiency
        let avg_efficiency: f64 = measurements
            .iter()
            .map(|m| m.operations_per_joule)
            .sum::<f64>()
            / measurements.len() as f64;

        if avg_efficiency < 1e6 {
            recommendations.push("Consider optimizing memory access patterns".to_string());
            recommendations
                .push("Evaluate use of lower precision arithmetic (FP16/BF16)".to_string());
        }

        // Check for performance consistency
        let power_variance: f64 = {
            let powers: Vec<f64> = measurements
                .iter()
                .map(|m| m.estimated_power_watts)
                .collect();
            let mean = avg_power;
            let variance =
                powers.iter().map(|&p| (p - mean).powi(2)).sum::<f64>() / powers.len() as f64;
            variance.sqrt()
        };

        if power_variance > avg_power * 0.2 {
            recommendations
                .push("High power variance detected - consider workload balancing".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("Energy efficiency is optimal for current workload".to_string());
        }

        recommendations
    }
}

/// Predefined energy benchmarks for common operations
pub mod standard_benchmarks {
    use super::*;
    use crate::vector;

    /// Run standard energy benchmarks for vector operations
    pub fn run_vector_benchmarks(profiler: &EnergyProfiler) -> Vec<EnergyEfficiencyMetrics> {
        let mut results = Vec::new();
        let size = 1024;
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();

        // Vector addition benchmark
        let add_metrics = profiler.compare_energy_efficiency(
            "vector_add",
            size as u64,
            || {
                // Scalar addition
                let mut result = vec![0.0f32; size];
                for i in 0..size {
                    result[i] = data[i] + data[i];
                }
            },
            || {
                // SIMD addition
                let mut result = vec![0.0; data.len()];
                vector::add_simd(&data, &data, &mut result);
            },
        );
        results.push(add_metrics);

        // Vector dot product benchmark
        let dot_metrics = profiler.compare_energy_efficiency(
            "vector_dot",
            size as u64,
            || {
                // Scalar dot product
                let mut _sum = 0.0f32;
                for i in 0..size {
                    _sum += data[i] * data[i];
                }
            },
            || {
                // SIMD dot product
                let _result = vector::dot_product(&data, &data);
            },
        );
        results.push(dot_metrics);

        results
    }

    /// Run energy benchmarks for different vector sizes
    pub fn run_scaling_benchmarks(
        profiler: &EnergyProfiler,
    ) -> HashMap<String, Vec<(usize, EnergyMeasurement)>> {
        let mut results = HashMap::new();
        let sizes = vec![64, 128, 256, 512, 1024, 2048, 4096];

        // Vector addition scaling
        let add_results = profiler.benchmark_vector_sizes("vector_add", &sizes, |size| {
            let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
            Box::new(move || {
                let mut result = vec![0.0; data.len()];
                vector::add_simd(&data, &data, &mut result);
            })
        });
        results.insert("vector_add".to_string(), add_results);

        // Vector multiplication scaling
        let mul_results = profiler.benchmark_vector_sizes("vector_mul", &sizes, |size| {
            let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
            Box::new(move || {
                // Use FMA with zero as element-wise multiplication: a * b + 0
                let mut result = data.clone();
                let zeros = vec![0.0f32; size];
                vector::fma_simd(&mut result, &data, &zeros);
            })
        });
        results.insert("vector_mul".to_string(), mul_results);

        results
    }
}

#[allow(non_snake_case)]
#[cfg(all(test, not(feature = "no-std")))]
mod tests {
    use super::*;

    #[cfg(feature = "no-std")]
    use alloc::{string::ToString, vec, vec::Vec};

    #[test]
    fn test_energy_profiler_creation() {
        let profiler = EnergyProfiler::new(65.0); // 65W TDP
        assert_eq!(profiler.cpu_tdp, 65.0);
        assert_eq!(profiler.measurement_duration, Duration::from_millis(100));
    }

    #[test]
    fn test_energy_measurement() {
        let profiler = EnergyProfiler::new(65.0);

        let measurement = profiler.measure_energy("test_op", 1000, || {
            // Simulate some work
            let mut _sum = 0.0f32;
            for i in 0..1000 {
                _sum += (i as f32).sin();
            }
        });

        assert!(measurement.duration > Duration::from_nanos(0));
        assert!(measurement.estimated_power_watts > 0.0);
        assert!(measurement.energy_joules > 0.0);
        assert!(measurement.operations_per_joule > 0.0);
        assert!(measurement.gflops_per_watt >= 0.0);
    }

    #[test]
    fn test_energy_comparison() {
        let profiler = EnergyProfiler::new(65.0);

        let metrics = profiler.compare_energy_efficiency(
            "test_comparison",
            1000,
            || {
                // Scalar operation (slower)
                let mut _sum = 0.0f32;
                for i in 0..1000 {
                    _sum += (i as f32) * 2.0;
                }
            },
            || {
                // "SIMD" operation (faster simulation)
                let data: Vec<f32> = (0..1000).map(|i| i as f32 * 2.0).collect();
                let _sum: f32 = data.iter().sum();
            },
        );

        assert_eq!(metrics.operation_name, "test_comparison");
        assert!(metrics.energy_efficiency_ratio > 0.0);
        assert!(metrics.performance_per_watt_ratio > 0.0);
    }

    #[test]
    fn test_power_optimizer() {
        let mut optimizer = PowerEfficiencyOptimizer::new(50.0, 10.0);

        // Test thermal scaling
        let cool_scaling = optimizer.optimize_for_thermal_state(ThermalState::Cool);
        assert_eq!(cool_scaling, 1.0);

        let hot_scaling = optimizer.optimize_for_thermal_state(ThermalState::Hot);
        assert!(hot_scaling < 1.0);

        let throttling_scaling = optimizer.optimize_for_thermal_state(ThermalState::Throttling);
        assert!(throttling_scaling < hot_scaling);
    }

    #[test]
    fn test_simd_width_recommendation() {
        let optimizer = PowerEfficiencyOptimizer::new(50.0, 10.0);

        let widths = vec![4, 8, 16, 32];
        let energies = vec![5.0, 8.0, 12.0, 20.0]; // Energy increases with width

        let recommended = optimizer.recommend_simd_width(&widths, &energies);
        assert!(widths.contains(&recommended));

        // Should pick a width that fits within energy budget
        let energy_index = widths.iter().position(|&w| w == recommended).unwrap();
        assert!(energies[energy_index] <= 10.0);
    }

    #[test]
    fn test_thermal_state_enum() {
        let states = vec![
            ThermalState::Cool,
            ThermalState::Warm,
            ThermalState::Hot,
            ThermalState::Throttling,
        ];

        for state in states {
            // Test that enum values can be used
            match state {
                ThermalState::Cool => assert!(true),
                ThermalState::Warm => assert!(true),
                ThermalState::Hot => assert!(true),
                ThermalState::Throttling => assert!(true),
            }
        }
    }

    #[test]
    fn test_report_generation() {
        let profiler = EnergyProfiler::new(65.0);

        let scalar_measurement = EnergyMeasurement {
            duration: Duration::from_millis(10),
            estimated_power_watts: 30.0,
            energy_joules: 0.3,
            operations_per_joule: 1000.0,
            gflops_per_watt: 1.0,
        };

        let simd_measurement = EnergyMeasurement {
            duration: Duration::from_millis(5),
            estimated_power_watts: 35.0,
            energy_joules: 0.175,
            operations_per_joule: 2000.0,
            gflops_per_watt: 2.0,
        };

        let metrics = EnergyEfficiencyMetrics {
            operation_name: "test_op".to_string(),
            scalar_energy: scalar_measurement,
            simd_energy: simd_measurement,
            energy_efficiency_ratio: 1.7,
            performance_per_watt_ratio: 2.0,
        };

        let report = profiler.generate_report(&vec![metrics]);
        assert!(report.contains("Energy Efficiency Report"));
        assert!(report.contains("test_op"));
        assert!(report.contains("1.70x"));
        assert!(report.contains("2.00x"));
    }

    #[test]
    fn test_benchmark_vector_sizes() {
        let profiler = EnergyProfiler::new(65.0);
        let sizes = vec![64, 128, 256];

        let results = profiler.benchmark_vector_sizes("test_scaling", &sizes, |size| {
            Box::new(move || {
                // Simulate work proportional to size
                let mut _sum = 0.0f32;
                for i in 0..size {
                    _sum += (i as f32).sqrt();
                }
            })
        });

        assert_eq!(results.len(), sizes.len());
        for (i, (size, measurement)) in results.iter().enumerate() {
            assert_eq!(*size, sizes[i]);
            assert!(measurement.duration > Duration::from_nanos(0));
            assert!(measurement.energy_joules > 0.0);
        }
    }

    #[test]
    fn test_recommendations_generation() {
        let optimizer = PowerEfficiencyOptimizer::new(40.0, 5.0);

        // High power consumption measurements
        let high_power_measurements = vec![EnergyMeasurement {
            duration: Duration::from_millis(10),
            estimated_power_watts: 50.0, // Above threshold
            energy_joules: 0.5,
            operations_per_joule: 500.0, // Low efficiency
            gflops_per_watt: 0.5,
        }];

        let recommendations = optimizer.generate_recommendations(&high_power_measurements);
        assert!(!recommendations.is_empty());
        assert!(recommendations
            .iter()
            .any(|r| r.contains("thermal") || r.contains("SIMD")));
    }
}
