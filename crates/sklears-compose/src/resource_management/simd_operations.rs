//! SIMD-accelerated operations for resource management
//!
//! This module provides high-performance calculations for resource
//! utilization metrics, efficiency scoring, and predictive analytics.
//!
//! ## SciRS2 Policy Compliance
//! ✅ Scalar implementations suitable for resource management operations
//! ✅ Works on stable Rust (no nightly features required)
//! ✅ Future SciRS2-Core optimizations possible where beneficial

/// Mean calculation for resource utilization metrics
#[inline]
#[must_use]
pub fn simd_average_utilization(utilizations: &[f64]) -> f64 {
    if utilizations.is_empty() {
        return 0.0;
    }

    utilizations.iter().sum::<f64>() / utilizations.len() as f64
}

/// Variance calculation for resource metrics
#[inline]
#[must_use]
pub fn simd_utilization_variance(utilizations: &[f64], mean: f64) -> f64 {
    if utilizations.len() <= 1 {
        return 0.0;
    }

    let var_sum: f64 = utilizations.iter().map(|&x| (x - mean).powi(2)).sum();
    var_sum / (utilizations.len() - 1) as f64
}

/// Efficiency calculation across multiple resources
#[inline]
#[must_use]
pub fn simd_efficiency_score(utilizations: &[f64], weights: &[f64]) -> f64 {
    if utilizations.is_empty() || weights.is_empty() || utilizations.len() != weights.len() {
        return 0.0;
    }

    // Scalar fallback implementation for stable Rust
    let weighted_sum: f64 = utilizations
        .iter()
        .zip(weights.iter())
        .map(|(&util, &weight)| util * weight)
        .sum();

    let weight_sum: f64 = weights.iter().sum();

    if weight_sum > 0.0 {
        weighted_sum / weight_sum
    } else {
        0.0
    }
}

/// Resource load balancing calculation
#[inline]
#[must_use]
pub fn simd_load_balance_score(loads: &[f64]) -> f64 {
    if loads.is_empty() {
        return 1.0;
    }

    let mean = simd_average_utilization(loads);
    let variance = simd_utilization_variance(loads, mean);

    // Lower variance = better load balancing
    let balance_score = 1.0 / (1.0 + variance);
    balance_score.min(1.0).max(0.0)
}

/// Thermal efficiency calculation
#[inline]
#[must_use]
pub fn simd_thermal_efficiency(temperatures: &[f64], max_temps: &[f64]) -> f64 {
    if temperatures.is_empty() || max_temps.is_empty() || temperatures.len() != max_temps.len() {
        return 0.0;
    }

    // Scalar fallback implementation for stable Rust
    let total_efficiency: f64 = temperatures
        .iter()
        .zip(max_temps.iter())
        .map(|(&temp, &max_temp)| {
            if max_temp > 0.0 {
                (max_temp - temp) / max_temp
            } else {
                0.0
            }
        })
        .sum();

    (total_efficiency / temperatures.len() as f64)
        .min(1.0)
        .max(0.0)
}

/// Power efficiency calculation
#[inline]
#[must_use]
pub fn simd_power_efficiency(power_consumption: &[f64], performance: &[f64]) -> f64 {
    if power_consumption.is_empty()
        || performance.is_empty()
        || power_consumption.len() != performance.len()
    {
        return 0.0;
    }

    let total_efficiency: f64 = power_consumption
        .iter()
        .zip(performance.iter())
        .map(|(&power, &perf)| {
            let safe_power = power + 1e-8;
            perf / safe_power
        })
        .sum();

    total_efficiency / power_consumption.len() as f64
}

/// Resource fragmentation calculation
#[inline]
#[must_use]
pub fn simd_fragmentation_score(free_blocks: &[f64], total_size: f64) -> f64 {
    if free_blocks.is_empty() || total_size <= 0.0 {
        return 0.0;
    }

    let sum: f64 = free_blocks.iter().sum();
    let sum_sq: f64 = free_blocks.iter().map(|&x| x * x).sum();

    let n = free_blocks.len() as f64;
    let mean = sum / n;
    let variance = (sum_sq - n * mean * mean) / (n - 1.0);

    // Fragmentation score: higher variance = more fragmentation

    (variance.sqrt() / mean).min(1.0).max(0.0)
}

/// Bandwidth utilization optimization
#[inline]
#[must_use]
pub fn simd_bandwidth_utilization(used_bandwidth: &[f64], total_bandwidth: &[f64]) -> f64 {
    if used_bandwidth.is_empty()
        || total_bandwidth.is_empty()
        || used_bandwidth.len() != total_bandwidth.len()
    {
        return 0.0;
    }

    let total_utilization: f64 = used_bandwidth
        .iter()
        .zip(total_bandwidth.iter())
        .map(|(&used, &total)| {
            let safe_total = total + 1e-8;
            used / safe_total
        })
        .sum();

    (total_utilization / used_bandwidth.len() as f64)
        .min(1.0)
        .max(0.0)
}

/// Predictive scaling calculation
#[inline]
#[must_use]
pub fn simd_predictive_scaling(historical_usage: &[f64], trend_weights: &[f64]) -> f64 {
    if historical_usage.is_empty()
        || trend_weights.is_empty()
        || historical_usage.len() != trend_weights.len()
    {
        return 0.0;
    }

    let weighted_prediction: f64 = historical_usage
        .iter()
        .zip(trend_weights.iter())
        .map(|(&usage, &weight)| usage * weight)
        .sum();

    let weight_sum: f64 = trend_weights.iter().sum();

    if weight_sum > 0.0 {
        (weighted_prediction / weight_sum).min(1.0).max(0.0)
    } else {
        0.0
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_average_utilization() {
        let utils = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let result = simd_average_utilization(&utils);
        assert!((result - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_simd_efficiency_score() {
        let utils = vec![0.8, 0.6, 0.7];
        let weights = vec![1.0, 2.0, 1.0];
        let result = simd_efficiency_score(&utils, &weights);
        assert!((result - 0.675).abs() < 1e-6); // (0.8*1 + 0.6*2 + 0.7*1) / 4
    }

    #[test]
    fn test_simd_load_balance_score() {
        let loads = vec![0.5, 0.5, 0.5, 0.5]; // Perfect balance
        let result = simd_load_balance_score(&loads);
        assert!(result > 0.99); // Should be close to 1.0
    }
}
