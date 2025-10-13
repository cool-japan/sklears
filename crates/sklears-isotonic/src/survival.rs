//! Survival analysis isotonic regression methods
//!
//! This module implements isotonic regression for survival analysis, including
//! censored data handling, hazard estimation, and competing risks models.

use crate::core::{isotonic_regression, LossFunction};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{prelude::SklearsError, types::Float};
use std::collections::HashMap;

/// Event type for survival analysis
#[derive(Debug, Clone, PartialEq)]
/// SurvivalEvent
pub enum SurvivalEvent {
    /// No event occurred (censored)
    Censored,
    /// Death or event of interest
    Death,
    /// Competing risk event
    CompetingRisk(u32),
}

/// Survival data point
#[derive(Debug, Clone)]
/// SurvivalDataPoint
pub struct SurvivalDataPoint {
    /// Time to event or censoring
    pub time: Float,
    /// Event status
    pub event: SurvivalEvent,
    /// Covariates (optional)
    pub covariates: Option<Array1<Float>>,
}

/// Isotonic survival regression
///
/// This struct implements isotonic regression for survival analysis with proper
/// handling of censored observations using the Kaplan-Meier estimator framework.
#[derive(Debug, Clone)]
/// IsotonicSurvivalRegression
pub struct IsotonicSurvivalRegression {
    /// Whether to enforce increasing or decreasing monotonicity
    increasing: bool,
    /// The fitted survival function values
    fitted_times: Option<Array1<Float>>,
    /// The fitted survival probabilities
    fitted_survival: Option<Array1<Float>>,
    /// The fitted hazard rates
    fitted_hazard: Option<Array1<Float>>,
    /// Number of at-risk individuals at each time point
    at_risk: Option<Array1<Float>>,
    /// Number of events at each time point
    events: Option<Array1<Float>>,
}

impl IsotonicSurvivalRegression {
    /// Create a new isotonic survival regression model
    pub fn new() -> Self {
        Self {
            increasing: false, // Survival functions are typically decreasing
            fitted_times: None,
            fitted_survival: None,
            fitted_hazard: None,
            at_risk: None,
            events: None,
        }
    }

    /// Set whether the function should be increasing or decreasing
    pub fn increasing(mut self, increasing: bool) -> Self {
        self.increasing = increasing;
        self
    }

    /// Fit the isotonic survival regression model
    pub fn fit(&mut self, data: &[SurvivalDataPoint]) -> Result<(), SklearsError> {
        if data.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Empty survival data".to_string(),
            ));
        }

        // Extract times and events, handling only death events
        let mut time_event_pairs: Vec<(Float, bool)> = data
            .iter()
            .map(|point| {
                let is_event = matches!(point.event, SurvivalEvent::Death);
                (point.time, is_event)
            })
            .collect();

        // Sort by time
        time_event_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Compute Kaplan-Meier estimates
        let (times, survival_probs, at_risk_counts, event_counts) =
            self.kaplan_meier_estimator(&time_event_pairs)?;

        // Apply isotonic regression to survival probabilities
        let x_values = Array1::from_vec((0..times.len()).map(|i| i as Float).collect());
        let y_values = Array1::from_vec(survival_probs.clone());

        let isotonic_survival =
            isotonic_regression(&x_values, &y_values, Some(self.increasing), None, None)?;

        // Compute hazard rates from survival probabilities
        let hazard_rates = self.compute_hazard_rates(&times, &isotonic_survival.to_vec())?;

        // Store the fitted model
        self.fitted_times = Some(Array1::from_vec(times));
        self.fitted_survival = Some(isotonic_survival);
        self.fitted_hazard = Some(Array1::from_vec(hazard_rates));
        self.at_risk = Some(Array1::from_vec(at_risk_counts));
        self.events = Some(Array1::from_vec(event_counts));

        Ok(())
    }

    /// Predict survival probability at given times
    pub fn predict_survival(&self, times: &Array1<Float>) -> Result<Array1<Float>, SklearsError> {
        let fitted_times = self
            .fitted_times
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict_survival".to_string(),
            })?;

        let fitted_survival =
            self.fitted_survival
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict_survival".to_string(),
                })?;

        let mut predictions = Array1::zeros(times.len());

        for (i, &time) in times.iter().enumerate() {
            predictions[i] = self.interpolate_survival(time, fitted_times, fitted_survival)?;
        }

        Ok(predictions)
    }

    /// Predict hazard rate at given times
    pub fn predict_hazard(&self, times: &Array1<Float>) -> Result<Array1<Float>, SklearsError> {
        let fitted_times = self
            .fitted_times
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict_hazard".to_string(),
            })?;

        let fitted_hazard = self
            .fitted_hazard
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict_hazard".to_string(),
            })?;

        let mut predictions = Array1::zeros(times.len());

        for (i, &time) in times.iter().enumerate() {
            predictions[i] = self.interpolate_hazard(time, fitted_times, fitted_hazard)?;
        }

        Ok(predictions)
    }

    /// Get the fitted survival function
    pub fn survival_function(&self) -> Option<(&Array1<Float>, &Array1<Float>)> {
        match (&self.fitted_times, &self.fitted_survival) {
            (Some(times), Some(survival)) => Some((times, survival)),
            _ => None,
        }
    }

    /// Get the fitted hazard function
    pub fn hazard_function(&self) -> Option<(&Array1<Float>, &Array1<Float>)> {
        match (&self.fitted_times, &self.fitted_hazard) {
            (Some(times), Some(hazard)) => Some((times, hazard)),
            _ => None,
        }
    }

    /// Kaplan-Meier estimator
    fn kaplan_meier_estimator(
        &self,
        time_event_pairs: &[(Float, bool)],
    ) -> Result<(Vec<Float>, Vec<Float>, Vec<Float>, Vec<Float>), SklearsError> {
        let mut unique_times = Vec::new();
        let mut survival_probs = Vec::new();
        let mut at_risk_counts = Vec::new();
        let mut event_counts = Vec::new();

        // Group by time and count events and censoring
        let mut time_map: HashMap<String, (usize, usize)> = HashMap::new(); // (events, total)

        for &(time, is_event) in time_event_pairs {
            let time_key = format!("{:.10}", time); // Use string key for float comparison
            let entry = time_map.entry(time_key).or_insert((0, 0));
            if is_event {
                entry.0 += 1;
            }
            entry.1 += 1;
        }

        // Sort times
        let mut sorted_times: Vec<(Float, usize, usize)> = time_map
            .iter()
            .map(|(time_str, &(events, total))| {
                let time: Float = time_str.parse().unwrap_or(0.0);
                (time, events, total)
            })
            .collect();
        sorted_times.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut current_survival = 1.0;
        let mut n_at_risk = time_event_pairs.len() as Float;

        for (time, n_events, n_total) in sorted_times {
            if n_events > 0 {
                // Update survival probability
                let hazard_increment = n_events as Float / n_at_risk;
                current_survival *= 1.0 - hazard_increment;

                unique_times.push(time);
                survival_probs.push(current_survival);
                at_risk_counts.push(n_at_risk);
                event_counts.push(n_events as Float);
            }

            // Update at-risk count
            n_at_risk -= n_total as Float;
        }

        Ok((unique_times, survival_probs, at_risk_counts, event_counts))
    }

    /// Compute hazard rates from survival probabilities
    fn compute_hazard_rates(
        &self,
        times: &[Float],
        survival_probs: &[Float],
    ) -> Result<Vec<Float>, SklearsError> {
        if times.len() != survival_probs.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{}", times.len()),
                actual: format!("{}", survival_probs.len()),
            });
        }

        let mut hazard_rates = Vec::with_capacity(times.len());

        for i in 0..times.len() {
            let hazard = if i == 0 {
                // Initial hazard rate
                if survival_probs[0] < 1.0 {
                    -(survival_probs[0].ln()) / times[0]
                } else {
                    0.0
                }
            } else {
                // Compute instantaneous hazard rate
                let dt = times[i] - times[i - 1];
                if dt > 0.0 && survival_probs[i] > 0.0 && survival_probs[i - 1] > 0.0 {
                    -(survival_probs[i] / survival_probs[i - 1]).ln() / dt
                } else {
                    0.0
                }
            };

            hazard_rates.push(hazard.max(0.0)); // Ensure non-negative hazard
        }

        Ok(hazard_rates)
    }

    /// Interpolate survival probability at a given time
    fn interpolate_survival(
        &self,
        time: Float,
        fitted_times: &Array1<Float>,
        fitted_survival: &Array1<Float>,
    ) -> Result<Float, SklearsError> {
        if fitted_times.is_empty() {
            return Ok(1.0); // Default survival probability
        }

        // Find the appropriate interval
        if time <= fitted_times[0] {
            return Ok(fitted_survival[0]);
        }

        if time >= fitted_times[fitted_times.len() - 1] {
            return Ok(fitted_survival[fitted_survival.len() - 1]);
        }

        // Linear interpolation
        for i in 0..fitted_times.len() - 1 {
            if time >= fitted_times[i] && time <= fitted_times[i + 1] {
                let t0 = fitted_times[i];
                let t1 = fitted_times[i + 1];
                let s0 = fitted_survival[i];
                let s1 = fitted_survival[i + 1];

                let weight = (time - t0) / (t1 - t0);
                return Ok(s0 + weight * (s1 - s0));
            }
        }

        Ok(fitted_survival[fitted_survival.len() - 1])
    }

    /// Interpolate hazard rate at a given time
    fn interpolate_hazard(
        &self,
        time: Float,
        fitted_times: &Array1<Float>,
        fitted_hazard: &Array1<Float>,
    ) -> Result<Float, SklearsError> {
        if fitted_times.is_empty() {
            return Ok(0.0); // Default hazard rate
        }

        // Find the appropriate interval
        if time <= fitted_times[0] {
            return Ok(fitted_hazard[0]);
        }

        if time >= fitted_times[fitted_times.len() - 1] {
            return Ok(fitted_hazard[fitted_hazard.len() - 1]);
        }

        // Linear interpolation
        for i in 0..fitted_times.len() - 1 {
            if time >= fitted_times[i] && time <= fitted_times[i + 1] {
                let t0 = fitted_times[i];
                let t1 = fitted_times[i + 1];
                let h0 = fitted_hazard[i];
                let h1 = fitted_hazard[i + 1];

                let weight = (time - t0) / (t1 - t0);
                return Ok(h0 + weight * (h1 - h0));
            }
        }

        Ok(fitted_hazard[fitted_hazard.len() - 1])
    }
}

impl Default for IsotonicSurvivalRegression {
    fn default() -> Self {
        Self::new()
    }
}

/// Competing risks isotonic regression
///
/// This struct handles survival analysis with multiple competing events.
#[derive(Debug, Clone)]
/// CompetingRisksIsotonicRegression
pub struct CompetingRisksIsotonicRegression {
    /// Whether to enforce increasing or decreasing monotonicity
    increasing: bool,
    /// The number of competing risks
    num_risks: u32,
    /// Fitted cumulative incidence functions for each risk
    cumulative_incidence: Option<HashMap<u32, (Array1<Float>, Array1<Float>)>>,
    /// Fitted cause-specific hazard functions
    cause_specific_hazards: Option<HashMap<u32, (Array1<Float>, Array1<Float>)>>,
    /// Overall survival function
    overall_survival: Option<(Array1<Float>, Array1<Float>)>,
}

impl CompetingRisksIsotonicRegression {
    /// Create a new competing risks isotonic regression model
    pub fn new(num_risks: u32) -> Self {
        Self {
            increasing: false,
            num_risks,
            cumulative_incidence: None,
            cause_specific_hazards: None,
            overall_survival: None,
        }
    }

    /// Set whether the function should be increasing or decreasing
    pub fn increasing(mut self, increasing: bool) -> Self {
        self.increasing = increasing;
        self
    }

    /// Fit the competing risks model
    pub fn fit(&mut self, data: &[SurvivalDataPoint]) -> Result<(), SklearsError> {
        if data.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Empty survival data".to_string(),
            ));
        }

        // Separate data by event type
        let mut risk_data: HashMap<u32, Vec<(Float, bool)>> = HashMap::new();
        let mut all_times: Vec<Float> = Vec::new();

        for point in data {
            all_times.push(point.time);

            match &point.event {
                SurvivalEvent::Death => {
                    let entry = risk_data.entry(0).or_insert_with(Vec::new);
                    entry.push((point.time, true));
                }
                SurvivalEvent::CompetingRisk(risk_id) => {
                    let entry = risk_data.entry(*risk_id).or_insert_with(Vec::new);
                    entry.push((point.time, true));
                }
                SurvivalEvent::Censored => {
                    // Add to all risk categories as censored
                    for risk_id in 0..=self.num_risks {
                        let entry = risk_data.entry(risk_id).or_insert_with(Vec::new);
                        entry.push((point.time, false));
                    }
                }
            }
        }

        // Compute cause-specific hazards for each risk
        let mut cause_specific_hazards = HashMap::new();
        let mut cumulative_incidences = HashMap::new();

        for (&risk_id, time_event_pairs) in &risk_data {
            if !time_event_pairs.is_empty() {
                // Compute Kaplan-Meier for this specific cause
                let mut survival_model =
                    IsotonicSurvivalRegression::new().increasing(self.increasing);
                let survival_data: Vec<SurvivalDataPoint> = time_event_pairs
                    .iter()
                    .map(|&(time, is_event)| SurvivalDataPoint {
                        time,
                        event: if is_event {
                            SurvivalEvent::Death
                        } else {
                            SurvivalEvent::Censored
                        },
                        covariates: None,
                    })
                    .collect();

                survival_model.fit(&survival_data)?;

                if let Some((times, hazards)) = survival_model.hazard_function() {
                    cause_specific_hazards.insert(risk_id, (times.clone(), hazards.clone()));
                }

                // Compute cumulative incidence (1 - survival for this cause)
                if let Some((times, survival)) = survival_model.survival_function() {
                    let cumulative_inc = survival.mapv(|s| 1.0 - s);
                    cumulative_incidences.insert(risk_id, (times.clone(), cumulative_inc));
                }
            }
        }

        // Compute overall survival (considering all competing events)
        let overall_data: Vec<SurvivalDataPoint> = data
            .iter()
            .map(|point| SurvivalDataPoint {
                time: point.time,
                event: match point.event {
                    SurvivalEvent::Censored => SurvivalEvent::Censored,
                    _ => SurvivalEvent::Death, // Any event is considered as "death" for overall survival
                },
                covariates: point.covariates.clone(),
            })
            .collect();

        let mut overall_model = IsotonicSurvivalRegression::new().increasing(self.increasing);
        overall_model.fit(&overall_data)?;

        if let Some((times, survival)) = overall_model.survival_function() {
            self.overall_survival = Some((times.clone(), survival.clone()));
        }

        self.cause_specific_hazards = Some(cause_specific_hazards);
        self.cumulative_incidence = Some(cumulative_incidences);

        Ok(())
    }

    /// Predict cumulative incidence for a specific risk at given times
    pub fn predict_cumulative_incidence(
        &self,
        risk_id: u32,
        times: &Array1<Float>,
    ) -> Result<Array1<Float>, SklearsError> {
        let cumulative_incidence =
            self.cumulative_incidence
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict_cumulative_incidence".to_string(),
                })?;

        let (fitted_times, fitted_incidence) = cumulative_incidence
            .get(&risk_id)
            .ok_or_else(|| SklearsError::InvalidInput(format!("Risk {} not found", risk_id)))?;

        let mut predictions = Array1::zeros(times.len());

        for (i, &time) in times.iter().enumerate() {
            predictions[i] = self.interpolate_value(time, fitted_times, fitted_incidence)?;
        }

        Ok(predictions)
    }

    /// Predict cause-specific hazard for a specific risk at given times
    pub fn predict_cause_specific_hazard(
        &self,
        risk_id: u32,
        times: &Array1<Float>,
    ) -> Result<Array1<Float>, SklearsError> {
        let hazards =
            self.cause_specific_hazards
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict_cause_specific_hazard".to_string(),
                })?;

        let (fitted_times, fitted_hazard) = hazards
            .get(&risk_id)
            .ok_or_else(|| SklearsError::InvalidInput(format!("Risk {} not found", risk_id)))?;

        let mut predictions = Array1::zeros(times.len());

        for (i, &time) in times.iter().enumerate() {
            predictions[i] = self.interpolate_value(time, fitted_times, fitted_hazard)?;
        }

        Ok(predictions)
    }

    /// Predict overall survival probability at given times
    pub fn predict_overall_survival(
        &self,
        times: &Array1<Float>,
    ) -> Result<Array1<Float>, SklearsError> {
        let (fitted_times, fitted_survival) =
            self.overall_survival
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict_overall_survival".to_string(),
                })?;

        let mut predictions = Array1::zeros(times.len());

        for (i, &time) in times.iter().enumerate() {
            predictions[i] = self.interpolate_value(time, fitted_times, fitted_survival)?;
        }

        Ok(predictions)
    }

    /// Get available risk IDs
    pub fn risk_ids(&self) -> Vec<u32> {
        if let Some(ref hazards) = self.cause_specific_hazards {
            hazards.keys().cloned().collect()
        } else {
            Vec::new()
        }
    }

    /// Interpolate value at a given time
    fn interpolate_value(
        &self,
        time: Float,
        fitted_times: &Array1<Float>,
        fitted_values: &Array1<Float>,
    ) -> Result<Float, SklearsError> {
        if fitted_times.is_empty() {
            return Ok(0.0);
        }

        if time <= fitted_times[0] {
            return Ok(fitted_values[0]);
        }

        if time >= fitted_times[fitted_times.len() - 1] {
            return Ok(fitted_values[fitted_values.len() - 1]);
        }

        // Linear interpolation
        for i in 0..fitted_times.len() - 1 {
            if time >= fitted_times[i] && time <= fitted_times[i + 1] {
                let t0 = fitted_times[i];
                let t1 = fitted_times[i + 1];
                let v0 = fitted_values[i];
                let v1 = fitted_values[i + 1];

                let weight = (time - t0) / (t1 - t0);
                return Ok(v0 + weight * (v1 - v0));
            }
        }

        Ok(fitted_values[fitted_values.len() - 1])
    }
}

/// Recurrent event isotonic regression
///
/// This struct handles recurrent events with isotonic constraints on the intensity function.
#[derive(Debug, Clone)]
/// RecurrentEventIsotonicRegression
pub struct RecurrentEventIsotonicRegression {
    /// Whether to enforce increasing or decreasing intensity
    increasing: bool,
    /// Fitted intensity function
    fitted_intensity: Option<(Array1<Float>, Array1<Float>)>,
    /// Fitted cumulative intensity function
    fitted_cumulative_intensity: Option<(Array1<Float>, Array1<Float>)>,
}

impl RecurrentEventIsotonicRegression {
    /// Create a new recurrent event isotonic regression model
    pub fn new() -> Self {
        Self {
            increasing: true, // Cumulative intensity typically increases
            fitted_intensity: None,
            fitted_cumulative_intensity: None,
        }
    }

    /// Set whether the function should be increasing or decreasing
    pub fn increasing(mut self, increasing: bool) -> Self {
        self.increasing = increasing;
        self
    }

    /// Fit the recurrent event model
    pub fn fit(&mut self, event_times: &[Vec<Float>]) -> Result<(), SklearsError> {
        if event_times.is_empty() {
            return Err(SklearsError::InvalidInput("Empty event times".to_string()));
        }

        // Collect all event times and compute Nelson-Aalen estimator
        let mut all_times: Vec<Float> = Vec::new();
        for individual_times in event_times {
            all_times.extend_from_slice(individual_times);
        }

        if all_times.is_empty() {
            return Err(SklearsError::InvalidInput("No events found".to_string()));
        }

        all_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Compute empirical intensity
        let (times, intensities, cumulative_intensities) =
            self.nelson_aalen_estimator(&all_times, event_times.len())?;

        // Apply isotonic regression to cumulative intensity
        let x_values = Array1::from_vec(times.clone());
        let y_values = Array1::from_vec(cumulative_intensities.clone());

        let isotonic_cumulative =
            isotonic_regression(&x_values, &y_values, Some(self.increasing), None, None)?;

        // Compute intensity from cumulative intensity
        let mut isotonic_intensity = Vec::with_capacity(times.len());
        for i in 0..times.len() {
            let intensity = if i == 0 {
                isotonic_cumulative[0] / times[0]
            } else {
                let dt = times[i] - times[i - 1];
                if dt > 0.0 {
                    (isotonic_cumulative[i] - isotonic_cumulative[i - 1]) / dt
                } else {
                    0.0
                }
            };
            isotonic_intensity.push(intensity.max(0.0));
        }

        // Store the fitted model
        self.fitted_intensity = Some((
            Array1::from_vec(times.clone()),
            Array1::from_vec(isotonic_intensity),
        ));
        self.fitted_cumulative_intensity = Some((Array1::from_vec(times), isotonic_cumulative));

        Ok(())
    }

    /// Predict intensity at given times
    pub fn predict_intensity(&self, times: &Array1<Float>) -> Result<Array1<Float>, SklearsError> {
        let (fitted_times, fitted_intensity) =
            self.fitted_intensity
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict_intensity".to_string(),
                })?;

        let mut predictions = Array1::zeros(times.len());

        for (i, &time) in times.iter().enumerate() {
            predictions[i] = self.interpolate_value(time, fitted_times, fitted_intensity)?;
        }

        Ok(predictions)
    }

    /// Predict cumulative intensity at given times
    pub fn predict_cumulative_intensity(
        &self,
        times: &Array1<Float>,
    ) -> Result<Array1<Float>, SklearsError> {
        let (fitted_times, fitted_cumulative) = self
            .fitted_cumulative_intensity
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict_cumulative_intensity".to_string(),
            })?;

        let mut predictions = Array1::zeros(times.len());

        for (i, &time) in times.iter().enumerate() {
            predictions[i] = self.interpolate_value(time, fitted_times, fitted_cumulative)?;
        }

        Ok(predictions)
    }

    /// Get the fitted intensity function
    pub fn intensity_function(&self) -> Option<(&Array1<Float>, &Array1<Float>)> {
        match &self.fitted_intensity {
            Some((times, intensity)) => Some((times, intensity)),
            None => None,
        }
    }

    /// Get the fitted cumulative intensity function
    pub fn cumulative_intensity_function(&self) -> Option<(&Array1<Float>, &Array1<Float>)> {
        match &self.fitted_cumulative_intensity {
            Some((times, cumulative)) => Some((times, cumulative)),
            None => None,
        }
    }

    /// Nelson-Aalen estimator for recurrent events
    fn nelson_aalen_estimator(
        &self,
        all_times: &[Float],
        num_individuals: usize,
    ) -> Result<(Vec<Float>, Vec<Float>, Vec<Float>), SklearsError> {
        let mut unique_times = Vec::new();
        let mut intensities = Vec::new();
        let mut cumulative_intensities = Vec::new();

        // Count events at each time point
        let mut time_counts: HashMap<String, usize> = HashMap::new();
        for &time in all_times {
            let time_key = format!("{:.10}", time);
            *time_counts.entry(time_key).or_insert(0) += 1;
        }

        // Sort times and compute Nelson-Aalen estimates
        let mut sorted_times: Vec<(Float, usize)> = time_counts
            .iter()
            .map(|(time_str, &count)| {
                let time: Float = time_str.parse().unwrap_or(0.0);
                (time, count)
            })
            .collect();
        sorted_times.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut cumulative = 0.0;

        for (time, count) in sorted_times {
            let intensity = count as Float / num_individuals as Float;
            cumulative += intensity;

            unique_times.push(time);
            intensities.push(intensity);
            cumulative_intensities.push(cumulative);
        }

        Ok((unique_times, intensities, cumulative_intensities))
    }

    /// Interpolate value at a given time
    fn interpolate_value(
        &self,
        time: Float,
        fitted_times: &Array1<Float>,
        fitted_values: &Array1<Float>,
    ) -> Result<Float, SklearsError> {
        if fitted_times.is_empty() {
            return Ok(0.0);
        }

        if time <= fitted_times[0] {
            return Ok(fitted_values[0]);
        }

        if time >= fitted_times[fitted_times.len() - 1] {
            return Ok(fitted_values[fitted_values.len() - 1]);
        }

        // Linear interpolation
        for i in 0..fitted_times.len() - 1 {
            if time >= fitted_times[i] && time <= fitted_times[i + 1] {
                let t0 = fitted_times[i];
                let t1 = fitted_times[i + 1];
                let v0 = fitted_values[i];
                let v1 = fitted_values[i + 1];

                let weight = (time - t0) / (t1 - t0);
                return Ok(v0 + weight * (v1 - v0));
            }
        }

        Ok(fitted_values[fitted_values.len() - 1])
    }
}

impl Default for RecurrentEventIsotonicRegression {
    fn default() -> Self {
        Self::new()
    }
}

// Function APIs for survival analysis

/// Perform isotonic survival regression
pub fn isotonic_survival_regression(
    data: &[SurvivalDataPoint],
    increasing: bool,
) -> Result<(Array1<Float>, Array1<Float>), SklearsError> {
    let mut model = IsotonicSurvivalRegression::new().increasing(increasing);
    model.fit(data)?;

    if let Some((times, survival)) = model.survival_function() {
        Ok((times.clone(), survival.clone()))
    } else {
        Err(SklearsError::InvalidInput(
            "Failed to fit survival model".to_string(),
        ))
    }
}

/// Perform isotonic hazard estimation
pub fn isotonic_hazard_estimation(
    data: &[SurvivalDataPoint],
    increasing: bool,
) -> Result<(Array1<Float>, Array1<Float>), SklearsError> {
    let mut model = IsotonicSurvivalRegression::new().increasing(increasing);
    model.fit(data)?;

    if let Some((times, hazard)) = model.hazard_function() {
        Ok((times.clone(), hazard.clone()))
    } else {
        Err(SklearsError::InvalidInput(
            "Failed to fit hazard model".to_string(),
        ))
    }
}

/// Perform competing risks isotonic regression
pub fn competing_risks_isotonic_regression(
    data: &[SurvivalDataPoint],
    num_risks: u32,
    increasing: bool,
) -> Result<CompetingRisksIsotonicRegression, SklearsError> {
    let mut model = CompetingRisksIsotonicRegression::new(num_risks).increasing(increasing);
    model.fit(data)?;
    Ok(model)
}

/// Perform recurrent event isotonic regression
pub fn recurrent_event_isotonic_regression(
    event_times: &[Vec<Float>],
    increasing: bool,
) -> Result<(Array1<Float>, Array1<Float>), SklearsError> {
    let mut model = RecurrentEventIsotonicRegression::new().increasing(increasing);
    model.fit(event_times)?;

    if let Some((times, cumulative)) = model.cumulative_intensity_function() {
        Ok((times.clone(), cumulative.clone()))
    } else {
        Err(SklearsError::InvalidInput(
            "Failed to fit recurrent event model".to_string(),
        ))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_isotonic_survival_regression() {
        let data = vec![
            SurvivalDataPoint {
                time: 1.0,
                event: SurvivalEvent::Death,
                covariates: None,
            },
            SurvivalDataPoint {
                time: 2.0,
                event: SurvivalEvent::Censored,
                covariates: None,
            },
            SurvivalDataPoint {
                time: 3.0,
                event: SurvivalEvent::Death,
                covariates: None,
            },
            SurvivalDataPoint {
                time: 4.0,
                event: SurvivalEvent::Censored,
                covariates: None,
            },
            SurvivalDataPoint {
                time: 5.0,
                event: SurvivalEvent::Death,
                covariates: None,
            },
        ];

        let mut model = IsotonicSurvivalRegression::new();
        assert!(model.fit(&data).is_ok());

        let test_times = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let survival_predictions = model.predict_survival(&test_times);
        assert!(survival_predictions.is_ok());

        let hazard_predictions = model.predict_hazard(&test_times);
        assert!(hazard_predictions.is_ok());

        // Check that survival function is decreasing
        let survival = survival_predictions.unwrap();
        for i in 0..survival.len() - 1 {
            assert!(survival[i] >= survival[i + 1]);
        }
    }

    #[test]
    fn test_competing_risks_isotonic_regression() {
        let data = vec![
            SurvivalDataPoint {
                time: 1.0,
                event: SurvivalEvent::Death,
                covariates: None,
            },
            SurvivalDataPoint {
                time: 2.0,
                event: SurvivalEvent::CompetingRisk(1),
                covariates: None,
            },
            SurvivalDataPoint {
                time: 3.0,
                event: SurvivalEvent::Censored,
                covariates: None,
            },
            SurvivalDataPoint {
                time: 4.0,
                event: SurvivalEvent::Death,
                covariates: None,
            },
            SurvivalDataPoint {
                time: 5.0,
                event: SurvivalEvent::CompetingRisk(2),
                covariates: None,
            },
        ];

        let mut model = CompetingRisksIsotonicRegression::new(3);
        assert!(model.fit(&data).is_ok());

        let test_times = array![1.0, 2.0, 3.0, 4.0, 5.0];

        // Test overall survival
        let overall_survival = model.predict_overall_survival(&test_times);
        assert!(overall_survival.is_ok());

        // Test cumulative incidence for death (risk 0)
        let cumulative_incidence = model.predict_cumulative_incidence(0, &test_times);
        assert!(cumulative_incidence.is_ok());

        // Test cause-specific hazard
        let cause_hazard = model.predict_cause_specific_hazard(0, &test_times);
        assert!(cause_hazard.is_ok());

        let risk_ids = model.risk_ids();
        assert!(!risk_ids.is_empty());
    }

    #[test]
    fn test_recurrent_event_isotonic_regression() {
        let event_times = vec![
            vec![1.0, 3.0, 5.0], // Individual 1
            vec![2.0, 4.0],      // Individual 2
            vec![1.5, 2.5, 4.5], // Individual 3
        ];

        let mut model = RecurrentEventIsotonicRegression::new();
        assert!(model.fit(&event_times).is_ok());

        let test_times = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let intensity_predictions = model.predict_intensity(&test_times);
        assert!(intensity_predictions.is_ok());

        let cumulative_predictions = model.predict_cumulative_intensity(&test_times);
        assert!(cumulative_predictions.is_ok());

        // Check that cumulative intensity is increasing
        let cumulative = cumulative_predictions.unwrap();
        for i in 0..cumulative.len() - 1 {
            assert!(cumulative[i] <= cumulative[i + 1]);
        }
    }

    #[test]
    fn test_survival_function_apis() {
        let data = vec![
            SurvivalDataPoint {
                time: 1.0,
                event: SurvivalEvent::Death,
                covariates: None,
            },
            SurvivalDataPoint {
                time: 2.0,
                event: SurvivalEvent::Censored,
                covariates: None,
            },
            SurvivalDataPoint {
                time: 3.0,
                event: SurvivalEvent::Death,
                covariates: None,
            },
        ];

        // Test survival regression API
        let result = isotonic_survival_regression(&data, false);
        assert!(result.is_ok());
        let (times, survival) = result.unwrap();
        assert_eq!(times.len(), survival.len());

        // Test hazard estimation API
        let hazard_result = isotonic_hazard_estimation(&data, true);
        assert!(hazard_result.is_ok());

        // Test competing risks API
        let competing_result = competing_risks_isotonic_regression(&data, 2, false);
        assert!(competing_result.is_ok());

        // Test recurrent event API
        let event_times = vec![vec![1.0, 2.0], vec![1.5, 3.0]];
        let recurrent_result = recurrent_event_isotonic_regression(&event_times, true);
        assert!(recurrent_result.is_ok());
    }

    #[test]
    fn test_empty_data_handling() {
        let empty_data: Vec<SurvivalDataPoint> = vec![];
        let mut model = IsotonicSurvivalRegression::new();
        assert!(model.fit(&empty_data).is_err());

        let mut competing_model = CompetingRisksIsotonicRegression::new(2);
        assert!(competing_model.fit(&empty_data).is_err());

        let empty_events: Vec<Vec<Float>> = vec![];
        let mut recurrent_model = RecurrentEventIsotonicRegression::new();
        assert!(recurrent_model.fit(&empty_events).is_err());
    }
}
