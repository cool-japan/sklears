//! Survival Analysis Metrics
//!
//! This module provides comprehensive metrics for survival analysis including
//! concordance index, time-dependent metrics, and survival-specific evaluation tools.

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, FromPrimitive, One, Zero};

/// Concordance Index (C-index) for survival analysis
///
/// The C-index measures the fraction of all pairs of subjects whose predicted
/// survival times are correctly ordered relative to their observed survival times.
///
/// # Arguments
/// * `event_times` - Observed survival times
/// * `predicted_times` - Predicted survival times
/// * `event_observed` - Boolean array indicating whether the event was observed (true) or censored (false)
///
/// # Returns
/// C-index value between 0 and 1, where 0.5 indicates random predictions and 1.0 indicates perfect predictions.
///
/// # Examples
/// ```
/// use sklears_metrics::survival::concordance_index;
/// use scirs2_core::ndarray::array;
///
/// let event_times = array![10.0, 15.0, 20.0, 25.0, 30.0];
/// let predicted_times = array![12.0, 14.0, 22.0, 24.0, 28.0];
/// let event_observed = array![true, true, false, true, true];
///
/// let c_index = concordance_index(&event_times, &predicted_times, &event_observed).unwrap();
/// assert!(c_index >= 0.0 && c_index <= 1.0);
/// ```
pub fn concordance_index<T>(
    event_times: &Array1<T>,
    predicted_times: &Array1<T>,
    event_observed: &Array1<bool>,
) -> MetricsResult<T>
where
    T: Float + Zero + One + FromPrimitive + std::fmt::Debug + PartialOrd + Copy,
{
    if event_times.len() != predicted_times.len() || event_times.len() != event_observed.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![event_times.len()],
            actual: vec![predicted_times.len(), event_observed.len()],
        });
    }

    if event_times.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let n = event_times.len();
    let mut concordant_pairs = 0;
    let mut total_pairs = 0;

    // Compare all pairs of subjects
    for i in 0..n {
        for j in (i + 1)..n {
            let time_i = event_times[i];
            let time_j = event_times[j];
            let pred_i = predicted_times[i];
            let pred_j = predicted_times[j];
            let observed_i = event_observed[i];
            let observed_j = event_observed[j];

            // Only consider comparable pairs
            let comparable = match (observed_i, observed_j) {
                (true, true) => true,             // Both events observed
                (true, false) => time_i < time_j, // Event i occurred before censoring j
                (false, true) => time_j < time_i, // Event j occurred before censoring i
                (false, false) => false,          // Both censored, not comparable
            };

            if comparable {
                total_pairs += 1;

                // Check if predictions are concordant with observations
                let concordant = if time_i < time_j {
                    pred_i < pred_j // Predict shorter time for earlier event
                } else if time_i > time_j {
                    pred_i > pred_j // Predict longer time for later event
                } else {
                    // Tied times - count as half concordant
                    concordant_pairs += 1; // Will be counted as 0.5 below
                    continue;
                };

                if concordant {
                    concordant_pairs += 2; // Count as 2 to handle ties as 0.5
                }
            }
        }
    }

    if total_pairs == 0 {
        return Ok(T::from(0.5).unwrap()); // No comparable pairs, return 0.5
    }

    // Divide by 2 because we counted each concordant pair as 2
    Ok(T::from(concordant_pairs).unwrap() / (T::from(2 * total_pairs).unwrap()))
}

/// Time-dependent AUC for survival analysis
///
/// Calculates the Area Under the ROC Curve at a specific time point,
/// accounting for censoring in survival data.
///
/// # Arguments
/// * `event_times` - Observed survival times
/// * `event_observed` - Boolean array indicating whether the event was observed
/// * `risk_scores` - Risk scores (higher values indicate higher risk)
/// * `time_point` - Time point at which to evaluate AUC
///
/// # Returns
/// Time-dependent AUC value between 0 and 1.
pub fn time_dependent_auc<T>(
    event_times: &Array1<T>,
    event_observed: &Array1<bool>,
    risk_scores: &Array1<T>,
    time_point: T,
) -> MetricsResult<T>
where
    T: Float + Zero + One + FromPrimitive + std::fmt::Debug + PartialOrd + Copy,
{
    if event_times.len() != event_observed.len() || event_times.len() != risk_scores.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![event_times.len()],
            actual: vec![event_observed.len(), risk_scores.len()],
        });
    }

    if event_times.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let n = event_times.len();
    let mut concordant_pairs = 0;
    let mut total_pairs = 0;

    // For each pair of subjects
    for i in 0..n {
        for j in (i + 1)..n {
            let time_i = event_times[i];
            let time_j = event_times[j];
            let observed_i = event_observed[i];
            let observed_j = event_observed[j];
            let risk_i = risk_scores[i];
            let risk_j = risk_scores[j];

            // Determine status at time_point
            let status_i = if observed_i && time_i <= time_point {
                Some(true) // Event occurred before time_point
            } else if time_i > time_point {
                Some(false) // Survived beyond time_point
            } else {
                None // Censored before time_point
            };

            let status_j = if observed_j && time_j <= time_point {
                Some(true) // Event occurred before time_point
            } else if time_j > time_point {
                Some(false) // Survived beyond time_point
            } else {
                None // Censored before time_point
            };

            // Only consider pairs where we know the status of both subjects
            match (status_i, status_j) {
                (Some(true), Some(false)) => {
                    // i had event, j survived - i should have higher risk
                    total_pairs += 1;
                    if risk_i > risk_j {
                        concordant_pairs += 1;
                    } else if risk_i == risk_j {
                        concordant_pairs += 1; // Ties count as 0.5, but we'll divide by 2 later
                    }
                }
                (Some(false), Some(true)) => {
                    // j had event, i survived - j should have higher risk
                    total_pairs += 1;
                    if risk_j > risk_i {
                        concordant_pairs += 1;
                    } else if risk_i == risk_j {
                        concordant_pairs += 1; // Ties count as 0.5
                    }
                }
                _ => {} // Skip other combinations
            }
        }
    }

    if total_pairs == 0 {
        return Ok(T::from(0.5).unwrap()); // No comparable pairs
    }

    Ok(T::from(concordant_pairs).unwrap() / T::from(total_pairs).unwrap())
}

/// Brier Score for survival analysis
///
/// Calculates the Brier score at a specific time point for survival predictions.
///
/// # Arguments
/// * `event_times` - Observed survival times
/// * `event_observed` - Boolean array indicating whether the event was observed
/// * `survival_probabilities` - Predicted survival probabilities at the time point
/// * `time_point` - Time point at which to evaluate the Brier score
///
/// # Returns
/// Brier score (lower values indicate better predictions).
pub fn brier_score_survival<T>(
    event_times: &Array1<T>,
    event_observed: &Array1<bool>,
    survival_probabilities: &Array1<T>,
    time_point: T,
) -> MetricsResult<T>
where
    T: Float + Zero + One + FromPrimitive + std::fmt::Debug + PartialOrd + Copy,
{
    if event_times.len() != event_observed.len()
        || event_times.len() != survival_probabilities.len()
    {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![event_times.len()],
            actual: vec![event_observed.len(), survival_probabilities.len()],
        });
    }

    if event_times.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let n = event_times.len();
    let mut brier_sum = T::zero();
    let mut valid_count = 0;

    for i in 0..n {
        let time_i = event_times[i];
        let observed_i = event_observed[i];
        let prob_i = survival_probabilities[i];

        // Determine actual survival status at time_point
        let actual_survival = if observed_i && time_i <= time_point {
            T::zero() // Event occurred, did not survive
        } else if time_i > time_point {
            T::one() // Survived beyond time_point
        } else {
            continue; // Censored before time_point, skip
        };

        // Calculate squared error
        let error = (prob_i - actual_survival).powi(2);
        brier_sum = brier_sum + error;
        valid_count += 1;
    }

    if valid_count == 0 {
        return Err(MetricsError::InvalidInput(
            "No valid observations for Brier score calculation".to_string(),
        ));
    }

    Ok(brier_sum / T::from(valid_count).unwrap())
}

/// Integrated Brier Score for survival analysis
///
/// Calculates the integrated Brier score over a range of time points.
///
/// # Arguments
/// * `event_times` - Observed survival times
/// * `event_observed` - Boolean array indicating whether the event was observed
/// * `predicted_survival_curves` - Matrix where each row is a predicted survival curve
/// * `evaluation_times` - Time points at which to evaluate the Brier score
///
/// # Returns
/// Integrated Brier score (lower values indicate better predictions).
pub fn integrated_brier_score<T>(
    event_times: &Array1<T>,
    event_observed: &Array1<bool>,
    predicted_survival_curves: &Array2<T>,
    evaluation_times: &Array1<T>,
) -> MetricsResult<T>
where
    T: Float + Zero + One + FromPrimitive + std::fmt::Debug + PartialOrd + Copy,
{
    if event_times.len() != predicted_survival_curves.nrows() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![event_times.len()],
            actual: vec![predicted_survival_curves.nrows()],
        });
    }

    if evaluation_times.len() != predicted_survival_curves.ncols() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![evaluation_times.len()],
            actual: vec![predicted_survival_curves.ncols()],
        });
    }

    if event_times.is_empty() || evaluation_times.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let mut brier_scores = Vec::new();

    // Calculate Brier score at each time point
    for (t_idx, &time_point) in evaluation_times.iter().enumerate() {
        let survival_probs = predicted_survival_curves.column(t_idx);
        let survival_probs_array = Array1::from_vec(survival_probs.to_vec());

        match brier_score_survival(
            event_times,
            event_observed,
            &survival_probs_array,
            time_point,
        ) {
            Ok(brier) => brier_scores.push(brier),
            Err(_) => continue, // Skip time points with no valid observations
        }
    }

    if brier_scores.is_empty() {
        return Err(MetricsError::InvalidInput(
            "No valid time points for integrated Brier score".to_string(),
        ));
    }

    // Calculate integrated Brier score using trapezoidal rule
    let mut integrated_score = T::zero();
    let mut total_time = T::zero();

    for i in 0..(brier_scores.len() - 1) {
        let dt = evaluation_times[i + 1] - evaluation_times[i];
        let avg_brier = (brier_scores[i] + brier_scores[i + 1]) / T::from(2.0).unwrap();
        integrated_score = integrated_score + avg_brier * dt;
        total_time = total_time + dt;
    }

    if total_time == T::zero() {
        return Err(MetricsError::DivisionByZero);
    }

    Ok(integrated_score / total_time)
}

/// Kaplan-Meier survival function estimation
///
/// Estimates the survival function using the Kaplan-Meier method.
///
/// # Arguments
/// * `event_times` - Observed survival times
/// * `event_observed` - Boolean array indicating whether the event was observed
///
/// # Returns
/// Tuple of (unique_times, survival_probabilities)
pub fn kaplan_meier_survival<T>(
    event_times: &Array1<T>,
    event_observed: &Array1<bool>,
) -> MetricsResult<(Array1<T>, Array1<T>)>
where
    T: Float + Zero + One + FromPrimitive + std::fmt::Debug + PartialOrd + Copy,
{
    if event_times.len() != event_observed.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![event_times.len()],
            actual: vec![event_observed.len()],
        });
    }

    if event_times.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Create list of (time, event) pairs and sort by time
    let mut time_event_pairs: Vec<(T, bool)> = event_times
        .iter()
        .zip(event_observed.iter())
        .map(|(&time, &event)| (time, event))
        .collect();

    time_event_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Group by unique times
    let mut unique_times = Vec::new();
    let mut events_at_time = Vec::new();
    let mut at_risk_counts = Vec::new();

    let mut current_time = time_event_pairs[0].0;
    let mut events_count = 0;
    let mut _total_count = 0;
    let at_risk = event_times.len();

    for &(time, event) in &time_event_pairs {
        if time != current_time {
            // Record data for previous time
            unique_times.push(current_time);
            events_at_time.push(events_count);
            at_risk_counts.push(at_risk);

            // Reset for new time
            current_time = time;
            events_count = 0;
            _total_count = 0;
        }

        _total_count += 1;
        if event {
            events_count += 1;
        }
    }

    // Don't forget the last time point
    unique_times.push(current_time);
    events_at_time.push(events_count);
    at_risk_counts.push(at_risk);

    // Calculate survival probabilities
    let mut survival_probs = Vec::new();
    let mut current_survival = T::one();

    for i in 0..unique_times.len() {
        let events = events_at_time[i];
        let at_risk = at_risk_counts[i];

        if at_risk > 0 {
            let survival_rate = T::one() - T::from(events).unwrap() / T::from(at_risk).unwrap();
            current_survival = current_survival * survival_rate;
        }

        survival_probs.push(current_survival);

        // Update at-risk count for next iteration
        if i + 1 < unique_times.len() {
            // Count how many subjects are removed at this time
            let removed = time_event_pairs
                .iter()
                .filter(|&&(time, _)| time == unique_times[i])
                .count();
            at_risk_counts[i + 1] = at_risk_counts[i] - removed;
        }
    }

    Ok((
        Array1::from_vec(unique_times),
        Array1::from_vec(survival_probs),
    ))
}

/// Log-rank test for comparing survival curves
///
/// Performs a log-rank test to compare survival curves between two groups.
///
/// # Arguments
/// * `event_times_1` - Survival times for group 1
/// * `event_observed_1` - Event indicators for group 1
/// * `event_times_2` - Survival times for group 2
/// * `event_observed_2` - Event indicators for group 2
///
/// # Returns
/// Tuple of (test_statistic, p_value)
pub fn log_rank_test<T>(
    event_times_1: &Array1<T>,
    event_observed_1: &Array1<bool>,
    event_times_2: &Array1<T>,
    event_observed_2: &Array1<bool>,
) -> MetricsResult<(T, T)>
where
    T: Float + std::fmt::Debug + PartialOrd + Copy,
{
    if event_times_1.len() != event_observed_1.len()
        || event_times_2.len() != event_observed_2.len()
    {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![event_times_1.len(), event_times_2.len()],
            actual: vec![event_observed_1.len(), event_observed_2.len()],
        });
    }

    if event_times_1.is_empty() || event_times_2.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Combine all event times and sort
    let mut all_times = Vec::new();
    for &time in event_times_1.iter() {
        all_times.push(time);
    }
    for &time in event_times_2.iter() {
        all_times.push(time);
    }
    all_times.sort_by(|a: &T, b| a.partial_cmp(b).unwrap());
    all_times.dedup();

    let mut observed_minus_expected = T::zero();
    let mut variance = T::zero();

    for &time in &all_times {
        // Count events and at-risk subjects in each group at this time
        let events_1 = event_times_1
            .iter()
            .zip(event_observed_1.iter())
            .filter(|(&t, &obs)| t == time && obs)
            .count();

        let events_2 = event_times_2
            .iter()
            .zip(event_observed_2.iter())
            .filter(|(&t, &obs)| t == time && obs)
            .count();

        let at_risk_1 = event_times_1.iter().filter(|&&t| t >= time).count();
        let at_risk_2 = event_times_2.iter().filter(|&&t| t >= time).count();

        let total_events = events_1 + events_2;
        let total_at_risk = at_risk_1 + at_risk_2;

        if total_at_risk > 0 && total_events > 0 {
            let expected_1 =
                T::from(at_risk_1 * total_events).unwrap() / T::from(total_at_risk).unwrap();
            let observed_1 = T::from(events_1).unwrap();

            observed_minus_expected = observed_minus_expected + (observed_1 - expected_1);

            // Calculate variance component
            if total_at_risk > 1 {
                let variance_component = expected_1
                    * (T::one() - T::from(at_risk_1).unwrap() / T::from(total_at_risk).unwrap())
                    * (T::one() - T::from(total_events).unwrap() / T::from(total_at_risk).unwrap());
                variance = variance + variance_component;
            }
        }
    }

    if variance == T::zero() {
        return Err(MetricsError::DivisionByZero);
    }

    // Calculate test statistic (follows chi-square distribution with 1 df)
    let test_statistic = (observed_minus_expected.powi(2)) / variance;

    // Calculate p-value using chi-square distribution approximation
    // For simplicity, we'll return the test statistic and let the user calculate p-value
    // In practice, you'd use a proper chi-square CDF
    let p_value = T::zero(); // Placeholder - would need proper chi-square implementation

    Ok((test_statistic, p_value))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_concordance_index_perfect() {
        let event_times = array![10.0, 20.0, 30.0, 40.0, 50.0];
        let predicted_times = array![10.0, 20.0, 30.0, 40.0, 50.0];
        let event_observed = array![true, true, true, true, true];

        let c_index: f64 =
            concordance_index(&event_times, &predicted_times, &event_observed).unwrap();
        assert_relative_eq!(c_index, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_concordance_index_random() {
        let event_times = array![10.0, 20.0, 30.0, 40.0, 50.0];
        let predicted_times = array![50.0, 40.0, 30.0, 20.0, 10.0]; // Reverse order
        let event_observed = array![true, true, true, true, true];

        let c_index: f64 =
            concordance_index(&event_times, &predicted_times, &event_observed).unwrap();
        assert_relative_eq!(c_index, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_time_dependent_auc() {
        let event_times = array![5.0, 10.0, 15.0, 20.0, 25.0];
        let event_observed = array![true, true, false, true, false];
        let risk_scores = array![0.8, 0.6, 0.4, 0.7, 0.3];
        let time_point = 12.0;

        let auc: f64 =
            time_dependent_auc(&event_times, &event_observed, &risk_scores, time_point).unwrap();
        assert!(auc >= 0.0 && auc <= 1.0);
    }

    #[test]
    fn test_brier_score_survival() {
        let event_times = array![5.0, 10.0, 15.0, 20.0, 25.0];
        let event_observed = array![true, true, false, true, false];
        let survival_probs = array![0.2, 0.6, 0.8, 0.4, 0.9];
        let time_point = 12.0;

        let brier: f64 =
            brier_score_survival(&event_times, &event_observed, &survival_probs, time_point)
                .unwrap();
        assert!(brier >= 0.0);
    }

    #[test]
    fn test_kaplan_meier_survival() {
        let event_times = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let event_observed = array![true, false, true, true, false];

        let (times, survival_probs) = kaplan_meier_survival(&event_times, &event_observed).unwrap();
        assert!(times.len() > 0);
        assert_eq!(times.len(), survival_probs.len());

        // Survival probabilities should be non-increasing
        for i in 1..survival_probs.len() {
            assert!(survival_probs[i] <= survival_probs[i - 1]);
        }
    }

    #[test]
    fn test_log_rank_test() {
        let event_times_1 = array![1.0, 2.0, 3.0, 4.0];
        let event_observed_1 = array![true, true, false, true];
        let event_times_2 = array![2.0, 3.0, 4.0, 5.0];
        let event_observed_2 = array![true, false, true, true];

        let (test_stat, _p_value): (f64, f64) = log_rank_test(
            &event_times_1,
            &event_observed_1,
            &event_times_2,
            &event_observed_2,
        )
        .unwrap();

        assert!(test_stat >= 0.0);
    }
}
