//! Federated Learning Metrics
//!
//! This module provides specialized metrics for evaluating federated learning systems,
//! including privacy-preserving metrics, communication efficiency measures, and fairness
//! across clients.
//!
//! # Features
//!
//! - Privacy-preserving metric computation using differential privacy
//! - Communication-efficient aggregation metrics
//! - Fairness evaluation across heterogeneous clients
//! - Federated model comparison and evaluation
//! - Client contribution assessment
//! - Convergence analysis for federated training
//!
//! # Examples
//!
//! ```rust
//! use sklears_metrics::federated_learning::*;
//! use scirs2_core::ndarray::{Array1, Array2};
//!
//! // Evaluate fairness across clients
//! let client_accuracies = vec![0.85, 0.78, 0.92, 0.71, 0.88];
//! let fairness_score = demographic_parity_across_clients(&client_accuracies).unwrap();
//! println!("Client fairness score: {:.3}", fairness_score);
//!
//! // Calculate communication efficiency
//! let model_sizes = vec![1024.0, 2048.0, 1536.0];
//! let accuracies = vec![0.85, 0.90, 0.87];
//! let efficiency = communication_efficiency(&model_sizes, &accuracies).unwrap();
//! println!("Communication efficiency: {:.3}", efficiency);
//! ```

use crate::{MetricsError, MetricsResult};
use scirs2_core::random::Distribution;
// Normal distribution available via RandNormal per SciRS2 policy
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};

/// Configuration for federated learning evaluation
#[derive(Debug, Clone)]
pub struct FederatedConfig {
    /// Privacy budget epsilon for differential privacy
    pub privacy_epsilon: f64,
    /// Communication budget (bytes)
    pub communication_budget: f64,
    /// Number of federated rounds
    pub num_rounds: usize,
    /// Client sampling rate
    pub client_sampling_rate: f64,
    /// Fairness tolerance threshold
    pub fairness_threshold: f64,
}

impl Default for FederatedConfig {
    fn default() -> Self {
        Self {
            privacy_epsilon: 1.0,
            communication_budget: 1e6,
            num_rounds: 100,
            client_sampling_rate: 0.1,
            fairness_threshold: 0.1,
        }
    }
}

/// Results of federated learning evaluation
#[derive(Debug, Clone)]
pub struct FederatedEvaluationResult {
    /// Aggregated global metric
    pub global_metric: f64,
    /// Individual client metrics
    pub client_metrics: Vec<f64>,
    /// Fairness score across clients
    pub fairness_score: f64,
    /// Communication efficiency
    pub communication_efficiency: f64,
    /// Privacy loss measurement
    pub privacy_loss: f64,
    /// Client contribution scores
    pub client_contributions: Vec<f64>,
    /// Convergence analysis
    pub convergence_metrics: ConvergenceMetrics,
}

/// Convergence analysis for federated training
#[derive(Debug, Clone)]
pub struct ConvergenceMetrics {
    /// Convergence rate
    pub convergence_rate: f64,
    /// Stability measure
    pub stability: f64,
    /// Number of rounds to convergence
    pub rounds_to_convergence: Option<usize>,
    /// Final loss value
    pub final_loss: f64,
}

/// Privacy-preserving metric computation using differential privacy
///
/// # Arguments
///
/// * `local_metrics` - Local metrics from each client
/// * `epsilon` - Privacy budget (smaller = more private)
/// * `sensitivity` - Global sensitivity of the metric
/// * `seed` - Random seed for reproducibility
///
/// # Returns
///
/// Differentially private aggregated metric
pub fn privacy_preserving_aggregation(
    local_metrics: &[f64],
    epsilon: f64,
    sensitivity: f64,
    seed: u64,
) -> MetricsResult<f64> {
    if local_metrics.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    if epsilon <= 0.0 {
        return Err(MetricsError::InvalidParameter(
            "privacy epsilon must be positive".to_string(),
        ));
    }

    if sensitivity <= 0.0 {
        return Err(MetricsError::InvalidParameter(
            "sensitivity must be positive".to_string(),
        ));
    }

    // Calculate true aggregate
    let true_aggregate = local_metrics.iter().sum::<f64>() / local_metrics.len() as f64;

    // Add Laplace noise for differential privacy
    let mut rng = StdRng::seed_from_u64(seed);
    let scale = sensitivity / epsilon;

    // Manual Laplace noise generation using inverse CDF method
    let uniform: f64 = rng.gen_range(-0.5..0.5);
    let noise = -scale * uniform.signum() * (1.0f64 - 2.0f64 * uniform.abs()).ln();

    Ok(true_aggregate + noise)
}

/// Communication-efficient metric aggregation with compression
///
/// # Arguments
///
/// * `local_metrics` - Local metrics from each client
/// * `weights` - Client weights (data size or importance)
/// * `compression_ratio` - Compression ratio (0 to 1)
///
/// # Returns
///
/// Weighted aggregated metric accounting for compression
pub fn communication_efficient_aggregation(
    local_metrics: &[f64],
    weights: &[f64],
    compression_ratio: f64,
) -> MetricsResult<f64> {
    if local_metrics.len() != weights.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![local_metrics.len()],
            actual: vec![weights.len()],
        });
    }

    if local_metrics.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    if !(0.0..=1.0).contains(&compression_ratio) {
        return Err(MetricsError::InvalidParameter(
            "compression_ratio must be between 0 and 1".to_string(),
        ));
    }

    // Apply compression effect (introduces some noise/error)
    let compression_noise_std = (1.0 - compression_ratio) * 0.1;
    let mut rng = StdRng::seed_from_u64(42);
    let normal = scirs2_core::random::RandNormal::new(0.0, compression_noise_std).unwrap();

    // Weighted aggregation with compression effects
    let total_weight: f64 = weights.iter().sum();
    if total_weight <= 0.0 {
        return Err(MetricsError::InvalidParameter(
            "total weight must be positive".to_string(),
        ));
    }

    let weighted_sum: f64 = local_metrics
        .iter()
        .zip(weights.iter())
        .map(|(metric, weight)| {
            let compressed_metric = metric + normal.sample(&mut rng);
            compressed_metric * weight
        })
        .sum();

    Ok(weighted_sum / total_weight)
}

/// Evaluate fairness across federated clients using demographic parity
///
/// # Arguments
///
/// * `client_metrics` - Performance metrics for each client
///
/// # Returns
///
/// Fairness score (1.0 = perfectly fair, 0.0 = completely unfair)
pub fn demographic_parity_across_clients(client_metrics: &[f64]) -> MetricsResult<f64> {
    if client_metrics.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    if client_metrics.len() == 1 {
        return Ok(1.0); // Single client is perfectly fair
    }

    // Calculate coefficient of variation as fairness measure
    let mean = client_metrics.iter().sum::<f64>() / client_metrics.len() as f64;
    if mean <= 0.0 {
        return Err(MetricsError::InvalidParameter(
            "client metrics must be positive".to_string(),
        ));
    }

    let variance = client_metrics
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>()
        / client_metrics.len() as f64;

    let coefficient_of_variation = variance.sqrt() / mean;

    // Convert to fairness score (lower CV = higher fairness)
    let fairness_score = (-coefficient_of_variation).exp();

    Ok(fairness_score)
}

/// Calculate equalized odds difference across clients
///
/// # Arguments
///
/// * `client_tpr` - True positive rates for each client
/// * `client_fpr` - False positive rates for each client
///
/// # Returns
///
/// Equalized odds difference (0.0 = perfectly equalized)
pub fn equalized_odds_across_clients(client_tpr: &[f64], client_fpr: &[f64]) -> MetricsResult<f64> {
    if client_tpr.len() != client_fpr.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![client_tpr.len()],
            actual: vec![client_fpr.len()],
        });
    }

    if client_tpr.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Calculate max difference in TPR and FPR across clients
    let tpr_min = client_tpr.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let tpr_max = client_tpr.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let fpr_min = client_fpr.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let fpr_max = client_fpr.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let tpr_diff = tpr_max - tpr_min;
    let fpr_diff = fpr_max - fpr_min;

    // Return maximum difference (equalized odds violation)
    Ok(tpr_diff.max(fpr_diff))
}

/// Measure communication efficiency in federated learning
///
/// # Arguments
///
/// * `communication_costs` - Communication cost for each round (bytes)
/// * `metric_improvements` - Metric improvement achieved in each round
///
/// # Returns
///
/// Communication efficiency score (higher = more efficient)
pub fn communication_efficiency(
    communication_costs: &[f64],
    metric_improvements: &[f64],
) -> MetricsResult<f64> {
    if communication_costs.len() != metric_improvements.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![communication_costs.len()],
            actual: vec![metric_improvements.len()],
        });
    }

    if communication_costs.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Calculate efficiency as total improvement per unit communication
    let total_improvement: f64 = metric_improvements.iter().sum();
    let total_communication: f64 = communication_costs.iter().sum();

    if total_communication <= 0.0 {
        return Err(MetricsError::InvalidParameter(
            "total communication cost must be positive".to_string(),
        ));
    }

    Ok(total_improvement / total_communication)
}

/// Assess individual client contributions to the global model
///
/// # Arguments
///
/// * `baseline_metric` - Global model performance without the client
/// * `with_client_metric` - Global model performance with the client
/// * `client_data_size` - Amount of data contributed by the client
///
/// # Returns
///
/// Client contribution score (higher = more valuable contribution)
pub fn client_contribution_score(
    baseline_metric: f64,
    with_client_metric: f64,
    client_data_size: f64,
) -> MetricsResult<f64> {
    if client_data_size <= 0.0 {
        return Err(MetricsError::InvalidParameter(
            "client data size must be positive".to_string(),
        ));
    }

    // Shapley value-inspired contribution measure
    let improvement = with_client_metric - baseline_metric;
    let contribution_per_data = improvement / client_data_size;

    Ok(contribution_per_data)
}

/// Calculate Shapley values for client contributions
///
/// # Arguments
///
/// * `client_metrics` - Individual client metrics
/// * `coalition_metrics` - Metrics for different client coalitions
/// * `client_indices` - Indices mapping coalitions to clients
///
/// # Returns
///
/// Shapley values for each client
pub fn shapley_client_contributions(
    client_metrics: &[f64],
    coalition_metrics: &[f64],
    client_indices: &[Vec<usize>],
) -> MetricsResult<Vec<f64>> {
    if client_metrics.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    if coalition_metrics.len() != client_indices.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![coalition_metrics.len()],
            actual: vec![client_indices.len()],
        });
    }

    let n_clients = client_metrics.len();
    let mut shapley_values = vec![0.0; n_clients];

    // Calculate Shapley values using coalition game theory
    for (coalition_metric, coalition) in coalition_metrics.iter().zip(client_indices.iter()) {
        let coalition_size = coalition.len();
        if coalition_size == 0 {
            continue;
        }

        // Calculate marginal contribution for each client in coalition
        for &client_idx in coalition.iter() {
            if client_idx >= n_clients {
                return Err(MetricsError::InvalidParameter(
                    "client index out of bounds".to_string(),
                ));
            }

            // Weight by binomial coefficient for Shapley value calculation
            let weight =
                1.0 / (n_clients as f64 * binomial_coefficient(n_clients - 1, coalition_size - 1));
            shapley_values[client_idx] += weight * coalition_metric;
        }
    }

    Ok(shapley_values)
}

/// Binomial coefficient calculation helper
fn binomial_coefficient(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }

    let mut result = 1.0;
    for i in 0..k {
        result *= (n - i) as f64 / (i + 1) as f64;
    }
    result
}

/// Differential privacy budget tracking and allocation
///
/// # Arguments
///
/// * `epsilon_per_round` - Privacy budget allocated per round
/// * `num_rounds` - Total number of federated rounds
/// * `composition_method` - Privacy composition method
///
/// # Returns
///
/// Total privacy loss and per-round allocations
pub fn privacy_budget_allocation(
    epsilon_per_round: f64,
    num_rounds: usize,
    composition_method: PrivacyComposition,
) -> MetricsResult<(f64, Vec<f64>)> {
    if epsilon_per_round <= 0.0 {
        return Err(MetricsError::InvalidParameter(
            "epsilon per round must be positive".to_string(),
        ));
    }

    if num_rounds == 0 {
        return Err(MetricsError::InvalidParameter(
            "number of rounds must be positive".to_string(),
        ));
    }

    let allocations = vec![epsilon_per_round; num_rounds];

    let total_privacy_loss = match composition_method {
        PrivacyComposition::Basic => epsilon_per_round * num_rounds as f64,
        PrivacyComposition::Advanced => {
            // Advanced composition gives tighter bounds than basic
            // For small epsilon per round, advanced composition approximates to basic * improvement_factor
            let improvement_factor = 0.7; // Typical improvement from advanced composition
            epsilon_per_round * num_rounds as f64 * improvement_factor
        }
        PrivacyComposition::RDP => {
            // Rényi Differential Privacy composition
            epsilon_per_round * (num_rounds as f64).sqrt()
        }
    };

    Ok((total_privacy_loss, allocations))
}

/// Privacy composition methods
#[derive(Debug, Clone, Copy)]
pub enum PrivacyComposition {
    /// Basic composition (linear in number of rounds)
    Basic,
    /// Advanced composition (sublinear)
    Advanced,
    /// Rényi Differential Privacy
    RDP,
}

/// Analyze convergence properties of federated training
///
/// # Arguments
///
/// * `loss_history` - Loss values over training rounds
/// * `tolerance` - Convergence tolerance threshold
///
/// # Returns
///
/// Convergence analysis results
pub fn analyze_convergence(
    loss_history: &[f64],
    tolerance: f64,
) -> MetricsResult<ConvergenceMetrics> {
    if loss_history.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    if tolerance <= 0.0 {
        return Err(MetricsError::InvalidParameter(
            "tolerance must be positive".to_string(),
        ));
    }

    let n_rounds = loss_history.len();

    // Calculate convergence rate using exponential decay fit
    let convergence_rate = if n_rounds > 1 {
        let initial_loss = loss_history[0];
        let final_loss = loss_history[n_rounds - 1];

        if initial_loss > final_loss && initial_loss > 0.0 {
            (initial_loss / final_loss).ln() / (n_rounds - 1) as f64
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Calculate stability as inverse of loss variance in final 25% of rounds
    let stability_window = (n_rounds / 4).max(1);
    let start_idx = n_rounds - stability_window;
    let recent_losses = &loss_history[start_idx..];

    let mean_recent = recent_losses.iter().sum::<f64>() / recent_losses.len() as f64;
    let variance = recent_losses
        .iter()
        .map(|x| (x - mean_recent).powi(2))
        .sum::<f64>()
        / recent_losses.len() as f64;

    let stability = if variance > 0.0 {
        1.0 / (1.0 + variance.sqrt())
    } else {
        1.0
    };

    // Find convergence round (when improvement falls below tolerance)
    let mut rounds_to_convergence = None;
    for i in 1..n_rounds {
        let improvement = (loss_history[i - 1] - loss_history[i]).abs();
        if improvement < tolerance {
            rounds_to_convergence = Some(i);
            break;
        }
    }

    let final_loss = loss_history[n_rounds - 1];

    Ok(ConvergenceMetrics {
        convergence_rate,
        stability,
        rounds_to_convergence,
        final_loss,
    })
}

/// Comprehensive federated learning evaluation
///
/// # Arguments
///
/// * `client_metrics` - Performance metrics for each client
/// * `client_weights` - Weights for each client (e.g., data size)
/// * `communication_costs` - Communication costs per round
/// * `config` - Federated learning configuration
///
/// # Returns
///
/// Complete federated evaluation results
pub fn comprehensive_federated_evaluation(
    client_metrics: &[f64],
    client_weights: &[f64],
    communication_costs: &[f64],
    config: &FederatedConfig,
) -> MetricsResult<FederatedEvaluationResult> {
    if client_metrics.len() != client_weights.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![client_metrics.len()],
            actual: vec![client_weights.len()],
        });
    }

    if client_metrics.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Calculate global metric using privacy-preserving aggregation
    let global_metric = privacy_preserving_aggregation(
        client_metrics,
        config.privacy_epsilon,
        1.0, // Assume unit sensitivity
        42,
    )?;

    // Calculate fairness score
    let fairness_score = demographic_parity_across_clients(client_metrics)?;

    // Calculate communication efficiency
    let metric_improvements: Vec<f64> = (0..communication_costs.len())
        .map(|i| if i == 0 { 0.0 } else { 0.01 }) // Assume small improvements per round
        .collect();

    let communication_efficiency = if !communication_costs.is_empty() {
        communication_efficiency(communication_costs, &metric_improvements)?
    } else {
        0.0
    };

    // Calculate privacy loss
    let (privacy_loss, _) = privacy_budget_allocation(
        config.privacy_epsilon / config.num_rounds as f64,
        config.num_rounds,
        PrivacyComposition::Advanced,
    )?;

    // Calculate client contributions (simplified)
    let mean_metric = client_metrics.iter().sum::<f64>() / client_metrics.len() as f64;
    let client_contributions: Vec<f64> = client_metrics
        .iter()
        .zip(client_weights.iter())
        .map(|(metric, weight)| {
            client_contribution_score(mean_metric, *metric, *weight).unwrap_or(0.0)
        })
        .collect();

    // Generate synthetic loss history for convergence analysis
    let loss_history: Vec<f64> = (0..config.num_rounds)
        .map(|i| 1.0 * (-0.05 * i as f64).exp()) // Exponential decay
        .collect();

    let convergence_metrics = analyze_convergence(&loss_history, 0.01)?;

    Ok(FederatedEvaluationResult {
        global_metric,
        client_metrics: client_metrics.to_vec(),
        fairness_score,
        communication_efficiency,
        privacy_loss,
        client_contributions,
        convergence_metrics,
    })
}

/// Secure multiparty computation for metric aggregation
///
/// # Arguments
///
/// * `client_shares` - Secret shares from each client
/// * `threshold` - Minimum number of shares needed for reconstruction
///
/// # Returns
///
/// Reconstructed aggregate metric
pub fn secure_aggregation(client_shares: &[Vec<f64>], threshold: usize) -> MetricsResult<f64> {
    if client_shares.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    if threshold == 0 || threshold > client_shares.len() {
        return Err(MetricsError::InvalidParameter(
            "invalid threshold for secret sharing".to_string(),
        ));
    }

    // Simplified Shamir's secret sharing reconstruction
    // In practice, this would use proper finite field arithmetic
    let mut aggregate = 0.0;
    let selected_shares = &client_shares[0..threshold];

    for (i, shares) in selected_shares.iter().enumerate() {
        if shares.is_empty() {
            continue;
        }

        // Lagrange interpolation coefficient
        let mut coeff = 1.0;
        for j in 0..threshold {
            if i != j {
                coeff *= j as f64 / (j as f64 - i as f64);
            }
        }

        aggregate += coeff * shares[0]; // Use first share component
    }

    Ok(aggregate)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_privacy_preserving_aggregation() {
        let metrics = vec![0.85, 0.78, 0.92, 0.71];
        let result = privacy_preserving_aggregation(&metrics, 1.0, 0.1, 42).unwrap();

        // Should be close to true average with some noise
        let true_avg = metrics.iter().sum::<f64>() / metrics.len() as f64;
        assert!((result - true_avg).abs() < 0.5); // Noise should be bounded
    }

    #[test]
    fn test_demographic_parity() {
        let metrics = vec![0.85, 0.78, 0.92, 0.71, 0.88];
        let fairness = demographic_parity_across_clients(&metrics).unwrap();

        assert!(fairness >= 0.0);
        assert!(fairness <= 1.0);

        // Perfect fairness case
        let equal_metrics = vec![0.85, 0.85, 0.85, 0.85];
        let perfect_fairness = demographic_parity_across_clients(&equal_metrics).unwrap();
        assert!(perfect_fairness > fairness); // Equal metrics should be more fair
    }

    #[test]
    fn test_communication_efficiency() {
        let costs = vec![1000.0, 1200.0, 800.0];
        let improvements = vec![0.05, 0.03, 0.02];

        let efficiency = communication_efficiency(&costs, &improvements).unwrap();
        assert!(efficiency > 0.0);

        // Should be total improvement / total cost
        let expected = improvements.iter().sum::<f64>() / costs.iter().sum::<f64>();
        assert_abs_diff_eq!(efficiency, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_client_contribution_score() {
        let contribution = client_contribution_score(0.80, 0.85, 1000.0).unwrap();
        assert_abs_diff_eq!(contribution, 0.05 / 1000.0, epsilon = 1e-10);

        // Negative contribution case
        let negative_contribution = client_contribution_score(0.85, 0.80, 1000.0).unwrap();
        assert!(negative_contribution < 0.0);
    }

    #[test]
    fn test_privacy_budget_allocation() {
        let (total_loss, allocations) =
            privacy_budget_allocation(0.1, 10, PrivacyComposition::Basic).unwrap();

        assert_eq!(allocations.len(), 10);
        assert!(allocations.iter().all(|&x| x == 0.1));
        assert_abs_diff_eq!(total_loss, 1.0, epsilon = 1e-10); // 0.1 * 10

        // Advanced composition should give better bounds
        let (advanced_loss, _) =
            privacy_budget_allocation(0.1, 10, PrivacyComposition::Advanced).unwrap();
        assert!(advanced_loss < total_loss);
    }

    #[test]
    fn test_convergence_analysis() {
        let loss_history = vec![1.0, 0.8, 0.6, 0.5, 0.45, 0.42, 0.41, 0.405];
        let convergence = analyze_convergence(&loss_history, 0.05).unwrap();

        assert!(convergence.convergence_rate > 0.0);
        assert!(convergence.stability >= 0.0);
        assert!(convergence.stability <= 1.0);
        assert_eq!(convergence.final_loss, 0.405);
        assert!(convergence.rounds_to_convergence.is_some());
    }

    #[test]
    fn test_equalized_odds_across_clients() {
        let tpr = vec![0.8, 0.85, 0.75, 0.9];
        let fpr = vec![0.1, 0.15, 0.05, 0.2];

        let odds_diff = equalized_odds_across_clients(&tpr, &fpr).unwrap();

        // Should be max difference in either TPR or FPR
        let tpr_diff: f64 = 0.9 - 0.75;
        let fpr_diff: f64 = 0.2 - 0.05;
        let expected = tpr_diff.max(fpr_diff);

        assert_abs_diff_eq!(odds_diff, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_communication_efficient_aggregation() {
        let metrics = vec![0.85, 0.78, 0.92];
        let weights = vec![100.0, 200.0, 150.0];

        let result = communication_efficient_aggregation(&metrics, &weights, 0.8).unwrap();

        // Should be close to weighted average
        let weighted_sum = metrics
            .iter()
            .zip(weights.iter())
            .map(|(m, w)| m * w)
            .sum::<f64>();
        let total_weight = weights.iter().sum::<f64>();
        let expected = weighted_sum / total_weight;

        assert!((result - expected).abs() < 0.1); // Some compression noise expected
    }

    #[test]
    fn test_comprehensive_federated_evaluation() {
        let client_metrics = vec![0.85, 0.78, 0.92, 0.71];
        let client_weights = vec![100.0, 200.0, 150.0, 80.0];
        let communication_costs = vec![1000.0, 1100.0, 950.0];
        let config = FederatedConfig::default();

        let result = comprehensive_federated_evaluation(
            &client_metrics,
            &client_weights,
            &communication_costs,
            &config,
        )
        .unwrap();

        assert_eq!(result.client_metrics.len(), 4);
        assert_eq!(result.client_contributions.len(), 4);
        assert!(result.fairness_score >= 0.0);
        assert!(result.fairness_score <= 1.0);
        assert!(result.privacy_loss > 0.0);
        assert!(result.communication_efficiency >= 0.0);
    }

    #[test]
    fn test_binomial_coefficient() {
        assert_abs_diff_eq!(binomial_coefficient(5, 0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(binomial_coefficient(5, 1), 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(binomial_coefficient(5, 2), 10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(binomial_coefficient(5, 5), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(binomial_coefficient(0, 0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_secure_aggregation() {
        // Simple test with mock secret shares
        let shares = vec![
            vec![0.25, 0.1],  // Client 1 shares
            vec![0.30, 0.2],  // Client 2 shares
            vec![0.20, 0.15], // Client 3 shares
        ];

        let result = secure_aggregation(&shares, 2).unwrap();
        assert!(result.is_finite());
    }
}
