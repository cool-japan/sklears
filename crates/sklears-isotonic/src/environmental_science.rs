/// Environmental Science Module for Isotonic Regression
///
/// This module implements isotonic regression methods for environmental science applications,
/// including environmental dose-response modeling, ecological threshold estimation,
/// climate trend analysis, pollution dispersion modeling, and ecosystem modeling.
use scirs2_core::ndarray::{s, Array1};
use sklears_core::prelude::SklearsError;

/// Types of environmental dose-response models
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EnvironmentalDoseResponseModel {
    /// Linear dose-response model
    Linear,
    /// Threshold model (no effect below threshold)
    Threshold,
    /// Hormesis model (beneficial at low doses, harmful at high doses)
    Hormesis,
    /// Logistic model
    Logistic,
    /// Probit model
    Probit,
    /// Hockey stick model
    HockeyStick,
}

/// Environmental Dose-Response Modeling
///
/// Models the relationship between environmental pollutant doses and biological responses
/// using isotonic regression with appropriate monotonicity constraints.
#[derive(Debug, Clone)]
pub struct EnvironmentalDoseResponseRegression {
    /// Dose-response model type
    model_type: EnvironmentalDoseResponseModel,
    /// Threshold dose (if applicable)
    threshold: Option<f64>,
    /// Maximum iterations
    max_iterations: usize,
    /// Tolerance
    tolerance: f64,
    /// Fitted doses
    fitted_dose: Option<Array1<f64>>,
    /// Fitted responses
    fitted_response: Option<Array1<f64>>,
}

impl Default for EnvironmentalDoseResponseRegression {
    fn default() -> Self {
        Self {
            model_type: EnvironmentalDoseResponseModel::Linear,
            threshold: None,
            max_iterations: 1000,
            tolerance: 1e-6,
            fitted_dose: None,
            fitted_response: None,
        }
    }
}

impl EnvironmentalDoseResponseRegression {
    /// Create a new environmental dose-response model
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the model type
    pub fn model_type(mut self, model: EnvironmentalDoseResponseModel) -> Self {
        self.model_type = model;
        self
    }

    /// Set the threshold dose
    pub fn threshold(mut self, t: f64) -> Self {
        self.threshold = Some(t);
        self
    }

    /// Fit the dose-response curve
    pub fn fit(&mut self, dose: &Array1<f64>, response: &Array1<f64>) -> Result<(), SklearsError> {
        if dose.len() != response.len() {
            return Err(SklearsError::InvalidInput(
                "Dose and response arrays must have the same length".to_string(),
            ));
        }

        // Sort by dose
        let mut indices: Vec<usize> = (0..dose.len()).collect();
        indices.sort_by(|&i, &j| dose[i].partial_cmp(&dose[j]).unwrap());

        let mut sorted_dose = Array1::zeros(dose.len());
        let mut sorted_response = Array1::zeros(response.len());

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            sorted_dose[new_idx] = dose[old_idx];
            sorted_response[new_idx] = response[old_idx];
        }

        // Apply isotonic regression based on model type
        let fitted_response = match self.model_type {
            EnvironmentalDoseResponseModel::Hormesis => {
                // For hormesis, allow non-monotonic behavior
                self.fit_hormesis(&sorted_response)?
            }
            _ => {
                // For other models, response increases with dose
                self.fit_monotonic_increasing(&sorted_response)?
            }
        };

        self.fitted_dose = Some(sorted_dose);
        self.fitted_response = Some(fitted_response);

        Ok(())
    }

    /// Fit monotonic increasing model
    fn fit_monotonic_increasing(
        &self,
        response: &Array1<f64>,
    ) -> Result<Array1<f64>, SklearsError> {
        let mut fitted = response.clone();

        for _ in 0..self.max_iterations {
            let old_fitted = fitted.clone();

            // Pool Adjacent Violators
            for i in 1..fitted.len() {
                if fitted[i] < fitted[i - 1] {
                    fitted[i] = fitted[i - 1];
                }
            }

            // Check convergence
            let diff: f64 = fitted
                .iter()
                .zip(old_fitted.iter())
                .map(|(new, old)| (new - old).powi(2))
                .sum();

            if diff.sqrt() < self.tolerance {
                break;
            }
        }

        Ok(fitted)
    }

    /// Fit hormesis model (allows non-monotonic behavior)
    fn fit_hormesis(&self, response: &Array1<f64>) -> Result<Array1<f64>, SklearsError> {
        // For hormesis, we smooth the response but allow a peak
        let mut fitted = response.clone();

        // Simple smoothing
        for _ in 0..10 {
            let old_fitted = fitted.clone();
            for i in 1..fitted.len() - 1 {
                fitted[i] =
                    0.25 * old_fitted[i - 1] + 0.5 * old_fitted[i] + 0.25 * old_fitted[i + 1];
            }
        }

        Ok(fitted)
    }

    /// Predict response for given doses
    pub fn predict(&self, dose: &Array1<f64>) -> Result<Array1<f64>, SklearsError> {
        let fitted_dose = self
            .fitted_dose
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;
        let fitted_response =
            self.fitted_response
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict".to_string(),
                })?;

        let mut predictions = Array1::zeros(dose.len());

        for (i, &d) in dose.iter().enumerate() {
            // Apply threshold if specified
            if let Some(threshold) = self.threshold {
                if d < threshold {
                    predictions[i] = 0.0;
                    continue;
                }
            }

            // Linear interpolation
            if d <= fitted_dose[0] {
                predictions[i] = fitted_response[0];
            } else if d >= fitted_dose[fitted_dose.len() - 1] {
                predictions[i] = fitted_response[fitted_response.len() - 1];
            } else {
                for j in 1..fitted_dose.len() {
                    if d <= fitted_dose[j] {
                        let t = (d - fitted_dose[j - 1]) / (fitted_dose[j] - fitted_dose[j - 1]);
                        predictions[i] =
                            (1.0 - t) * fitted_response[j - 1] + t * fitted_response[j];
                        break;
                    }
                }
            }
        }

        Ok(predictions)
    }

    /// Estimate no-observed-adverse-effect level (NOAEL)
    pub fn estimate_noael(&self, adverse_effect_threshold: f64) -> Result<f64, SklearsError> {
        let fitted_dose = self
            .fitted_dose
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "estimate_noael".to_string(),
            })?;
        let fitted_response =
            self.fitted_response
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "estimate_noael".to_string(),
                })?;

        // Find the highest dose where response is below threshold
        for i in (0..fitted_response.len()).rev() {
            if fitted_response[i] < adverse_effect_threshold {
                return Ok(fitted_dose[i]);
            }
        }

        Ok(0.0) // If all doses exceed threshold, return 0
    }
}

/// Ecological Threshold Estimation
///
/// Estimates ecological thresholds (tipping points) where ecosystems
/// undergo abrupt transitions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThresholdDetectionMethod {
    /// Change point detection
    ChangePoint,
    /// Regime shift detection
    RegimeShift,
    /// Tipping point analysis
    TippingPoint,
    /// Early warning signals
    EarlyWarning,
}

#[derive(Debug, Clone)]
pub struct EcologicalThresholdEstimation {
    /// Detection method
    method: ThresholdDetectionMethod,
    /// Sensitivity parameter
    sensitivity: f64,
    /// Maximum iterations
    max_iterations: usize,
    /// Tolerance
    tolerance: f64,
}

impl Default for EcologicalThresholdEstimation {
    fn default() -> Self {
        Self {
            method: ThresholdDetectionMethod::ChangePoint,
            sensitivity: 0.1,
            max_iterations: 1000,
            tolerance: 1e-6,
        }
    }
}

impl EcologicalThresholdEstimation {
    /// Create a new threshold estimation model
    pub fn new() -> Self {
        Self::default()
    }

    /// Set detection method
    pub fn method(mut self, method: ThresholdDetectionMethod) -> Self {
        self.method = method;
        self
    }

    /// Set sensitivity
    pub fn sensitivity(mut self, s: f64) -> Self {
        self.sensitivity = s;
        self
    }

    /// Detect thresholds in ecological data
    pub fn detect_threshold(
        &self,
        pressure: &Array1<f64>,
        state: &Array1<f64>,
    ) -> Result<Vec<f64>, SklearsError> {
        if pressure.len() != state.len() {
            return Err(SklearsError::InvalidInput(
                "Pressure and state arrays must have the same length".to_string(),
            ));
        }

        match self.method {
            ThresholdDetectionMethod::ChangePoint => self.detect_change_point(pressure, state),
            ThresholdDetectionMethod::RegimeShift => self.detect_regime_shift(pressure, state),
            ThresholdDetectionMethod::TippingPoint => self.detect_tipping_point(pressure, state),
            ThresholdDetectionMethod::EarlyWarning => {
                self.detect_early_warning_signals(pressure, state)
            }
        }
    }

    /// Detect change points using isotonic regression
    fn detect_change_point(
        &self,
        pressure: &Array1<f64>,
        state: &Array1<f64>,
    ) -> Result<Vec<f64>, SklearsError> {
        let n = pressure.len();
        let mut change_points = Vec::new();

        // Fit piecewise isotonic regression and look for abrupt changes
        let mut fitted = state.clone();

        // Simple PAV
        for i in 1..n {
            if fitted[i] < fitted[i - 1] {
                fitted[i] = fitted[i - 1];
            }
        }

        // Detect points where slope changes significantly
        for i in 2..n - 1 {
            let slope_before =
                (fitted[i] - fitted[i - 1]) / (pressure[i] - pressure[i - 1] + 1e-10);
            let slope_after = (fitted[i + 1] - fitted[i]) / (pressure[i + 1] - pressure[i] + 1e-10);

            let slope_change = (slope_after - slope_before).abs();
            let avg_slope = 0.5 * (slope_before.abs() + slope_after.abs());

            if slope_change > self.sensitivity * avg_slope {
                change_points.push(pressure[i]);
            }
        }

        Ok(change_points)
    }

    /// Detect regime shifts
    fn detect_regime_shift(
        &self,
        pressure: &Array1<f64>,
        state: &Array1<f64>,
    ) -> Result<Vec<f64>, SklearsError> {
        // Use variance change detection
        let n = pressure.len();
        let window_size = (n / 5).max(3);
        let mut regime_shifts = Vec::new();

        for i in window_size..n - window_size {
            let before_mean: f64 = state.slice(s![i - window_size..i]).mean().unwrap_or(0.0);
            let after_mean: f64 = state.slice(s![i..i + window_size]).mean().unwrap_or(0.0);

            let before_var: f64 = state
                .slice(s![i - window_size..i])
                .iter()
                .map(|&x| (x - before_mean).powi(2))
                .sum::<f64>()
                / window_size as f64;

            let after_var: f64 = state
                .slice(s![i..i + window_size])
                .iter()
                .map(|&x| (x - after_mean).powi(2))
                .sum::<f64>()
                / window_size as f64;

            let mean_change = (after_mean - before_mean).abs();
            let var_change = (after_var - before_var).abs();

            if mean_change > self.sensitivity || var_change > self.sensitivity {
                regime_shifts.push(pressure[i]);
            }
        }

        Ok(regime_shifts)
    }

    /// Detect tipping points
    fn detect_tipping_point(
        &self,
        pressure: &Array1<f64>,
        state: &Array1<f64>,
    ) -> Result<Vec<f64>, SklearsError> {
        // Look for points where small pressure changes cause large state changes
        let mut tipping_points = Vec::new();

        for i in 1..pressure.len() - 1 {
            let dp = pressure[i + 1] - pressure[i];
            let ds = state[i + 1] - state[i];

            if dp.abs() > 1e-10 {
                let sensitivity_local = (ds / dp).abs();

                // High sensitivity indicates tipping point
                if sensitivity_local > 1.0 / self.sensitivity {
                    tipping_points.push(pressure[i]);
                }
            }
        }

        Ok(tipping_points)
    }

    /// Detect early warning signals
    fn detect_early_warning_signals(
        &self,
        pressure: &Array1<f64>,
        state: &Array1<f64>,
    ) -> Result<Vec<f64>, SklearsError> {
        // Look for increasing variance and autocorrelation as early warnings
        let n = pressure.len();
        let window_size = (n / 10).max(5);
        let mut warning_points = Vec::new();

        for i in window_size..n - window_size {
            // Calculate rolling variance
            let window_mean: f64 = state.slice(s![i - window_size..i]).mean().unwrap_or(0.0);
            let variance: f64 = state
                .slice(s![i - window_size..i])
                .iter()
                .map(|&x| (x - window_mean).powi(2))
                .sum::<f64>()
                / window_size as f64;

            // Calculate lag-1 autocorrelation
            let mut autocorr = 0.0;
            for j in i - window_size + 1..i {
                autocorr += (state[j] - window_mean) * (state[j - 1] - window_mean);
            }
            autocorr /= (window_size - 1) as f64 * variance.max(1e-10);

            // High variance and autocorrelation are warning signals
            if variance > self.sensitivity || autocorr > 0.7 {
                warning_points.push(pressure[i]);
            }
        }

        Ok(warning_points)
    }
}

/// Climate Trend Analysis
///
/// Analyzes climate trends using isotonic regression to detect monotonic
/// changes in temperature, precipitation, sea level, etc.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClimateTrendType {
    /// Temperature trend
    Temperature,
    /// Precipitation trend
    Precipitation,
    /// Sea level trend
    SeaLevel,
    /// Ice extent trend
    IceExtent,
    /// CO2 concentration trend
    CO2Concentration,
}

#[derive(Debug, Clone)]
pub struct ClimateTrendAnalysis {
    /// Trend type
    trend_type: ClimateTrendType,
    /// Detrending method
    detrend: bool,
    /// Maximum iterations
    max_iterations: usize,
    /// Tolerance
    tolerance: f64,
    /// Fitted time points
    fitted_time: Option<Array1<f64>>,
    /// Fitted trend
    fitted_trend: Option<Array1<f64>>,
}

impl Default for ClimateTrendAnalysis {
    fn default() -> Self {
        Self {
            trend_type: ClimateTrendType::Temperature,
            detrend: false,
            max_iterations: 1000,
            tolerance: 1e-6,
            fitted_time: None,
            fitted_trend: None,
        }
    }
}

impl ClimateTrendAnalysis {
    /// Create a new climate trend analysis model
    pub fn new() -> Self {
        Self::default()
    }

    /// Set trend type
    pub fn trend_type(mut self, trend: ClimateTrendType) -> Self {
        self.trend_type = trend;
        self
    }

    /// Set detrending
    pub fn detrend(mut self, d: bool) -> Self {
        self.detrend = d;
        self
    }

    /// Fit the climate trend
    pub fn fit(&mut self, time: &Array1<f64>, values: &Array1<f64>) -> Result<(), SklearsError> {
        if time.len() != values.len() {
            return Err(SklearsError::InvalidInput(
                "Time and values arrays must have the same length".to_string(),
            ));
        }

        // Sort by time
        let mut indices: Vec<usize> = (0..time.len()).collect();
        indices.sort_by(|&i, &j| time[i].partial_cmp(&time[j]).unwrap());

        let mut sorted_time = Array1::zeros(time.len());
        let mut sorted_values = Array1::zeros(values.len());

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            sorted_time[new_idx] = time[old_idx];
            sorted_values[new_idx] = values[old_idx];
        }

        // Determine monotonicity based on trend type
        let increasing = matches!(
            self.trend_type,
            ClimateTrendType::Temperature
                | ClimateTrendType::SeaLevel
                | ClimateTrendType::CO2Concentration
        );

        // Apply isotonic regression
        let mut fitted_trend = sorted_values.clone();

        for _ in 0..self.max_iterations {
            let old_trend = fitted_trend.clone();

            if increasing {
                // Increasing trend
                for i in 1..fitted_trend.len() {
                    if fitted_trend[i] < fitted_trend[i - 1] {
                        fitted_trend[i] = fitted_trend[i - 1];
                    }
                }
            } else {
                // Decreasing trend (for ice extent)
                for i in 1..fitted_trend.len() {
                    if fitted_trend[i] > fitted_trend[i - 1] {
                        fitted_trend[i] = fitted_trend[i - 1];
                    }
                }
            }

            // Check convergence
            let diff: f64 = fitted_trend
                .iter()
                .zip(old_trend.iter())
                .map(|(new, old)| (new - old).powi(2))
                .sum();

            if diff.sqrt() < self.tolerance {
                break;
            }
        }

        self.fitted_time = Some(sorted_time);
        self.fitted_trend = Some(fitted_trend);

        Ok(())
    }

    /// Predict trend at given time points
    pub fn predict(&self, time: &Array1<f64>) -> Result<Array1<f64>, SklearsError> {
        let fitted_time = self
            .fitted_time
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;
        let fitted_trend = self
            .fitted_trend
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let mut predictions = Array1::zeros(time.len());

        for (i, &t) in time.iter().enumerate() {
            if t <= fitted_time[0] {
                predictions[i] = fitted_trend[0];
            } else if t >= fitted_time[fitted_time.len() - 1] {
                predictions[i] = fitted_trend[fitted_trend.len() - 1];
            } else {
                for j in 1..fitted_time.len() {
                    if t <= fitted_time[j] {
                        let tau = (t - fitted_time[j - 1]) / (fitted_time[j] - fitted_time[j - 1]);
                        predictions[i] = (1.0 - tau) * fitted_trend[j - 1] + tau * fitted_trend[j];
                        break;
                    }
                }
            }
        }

        Ok(predictions)
    }

    /// Estimate rate of change
    pub fn estimate_rate_of_change(&self) -> Result<f64, SklearsError> {
        let fitted_time = self
            .fitted_time
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "estimate_rate_of_change".to_string(),
            })?;
        let fitted_trend = self
            .fitted_trend
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "estimate_rate_of_change".to_string(),
            })?;

        let n = fitted_time.len();
        if n < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 points to estimate rate".to_string(),
            ));
        }

        // Simple linear regression on the fitted trend
        let time_mean: f64 = fitted_time.iter().sum::<f64>() / n as f64;
        let trend_mean: f64 = fitted_trend.iter().sum::<f64>() / n as f64;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..n {
            numerator += (fitted_time[i] - time_mean) * (fitted_trend[i] - trend_mean);
            denominator += (fitted_time[i] - time_mean).powi(2);
        }

        if denominator < 1e-10 {
            return Ok(0.0);
        }

        Ok(numerator / denominator)
    }
}

/// Pollution Dispersion Modeling
///
/// Models the dispersion of pollutants in the environment using isotonic
/// regression for concentration profiles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PollutionDispersionModel {
    /// Gaussian plume model
    GaussianPlume,
    /// Atmospheric dispersion
    Atmospheric,
    /// Aquatic dispersion
    Aquatic,
    /// Soil contamination
    Soil,
}

#[derive(Debug, Clone)]
pub struct PollutionDispersionRegression {
    /// Dispersion model type
    model_type: PollutionDispersionModel,
    /// Maximum iterations
    max_iterations: usize,
    /// Tolerance
    tolerance: f64,
    /// Fitted distances
    fitted_distance: Option<Array1<f64>>,
    /// Fitted concentrations
    fitted_concentration: Option<Array1<f64>>,
}

impl Default for PollutionDispersionRegression {
    fn default() -> Self {
        Self {
            model_type: PollutionDispersionModel::GaussianPlume,
            max_iterations: 1000,
            tolerance: 1e-6,
            fitted_distance: None,
            fitted_concentration: None,
        }
    }
}

impl PollutionDispersionRegression {
    /// Create a new pollution dispersion model
    pub fn new() -> Self {
        Self::default()
    }

    /// Set model type
    pub fn model_type(mut self, model: PollutionDispersionModel) -> Self {
        self.model_type = model;
        self
    }

    /// Fit the dispersion curve (concentration decreases with distance)
    pub fn fit(
        &mut self,
        distance: &Array1<f64>,
        concentration: &Array1<f64>,
    ) -> Result<(), SklearsError> {
        if distance.len() != concentration.len() {
            return Err(SklearsError::InvalidInput(
                "Distance and concentration arrays must have the same length".to_string(),
            ));
        }

        // Sort by distance
        let mut indices: Vec<usize> = (0..distance.len()).collect();
        indices.sort_by(|&i, &j| distance[i].partial_cmp(&distance[j]).unwrap());

        let mut sorted_distance = Array1::zeros(distance.len());
        let mut sorted_concentration = Array1::zeros(concentration.len());

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            sorted_distance[new_idx] = distance[old_idx];
            sorted_concentration[new_idx] = concentration[old_idx];
        }

        // Apply isotonic regression (concentration decreases with distance)
        let mut fitted_concentration = sorted_concentration.clone();

        for _ in 0..self.max_iterations {
            let old_concentration = fitted_concentration.clone();

            // Decreasing constraint
            for i in 1..fitted_concentration.len() {
                if fitted_concentration[i] > fitted_concentration[i - 1] {
                    fitted_concentration[i] = fitted_concentration[i - 1];
                }
            }

            // Ensure non-negative concentrations
            fitted_concentration.mapv_inplace(|v| v.max(0.0));

            // Check convergence
            let diff: f64 = fitted_concentration
                .iter()
                .zip(old_concentration.iter())
                .map(|(new, old)| (new - old).powi(2))
                .sum();

            if diff.sqrt() < self.tolerance {
                break;
            }
        }

        self.fitted_distance = Some(sorted_distance);
        self.fitted_concentration = Some(fitted_concentration);

        Ok(())
    }

    /// Predict concentration at given distances
    pub fn predict(&self, distance: &Array1<f64>) -> Result<Array1<f64>, SklearsError> {
        let fitted_distance =
            self.fitted_distance
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict".to_string(),
                })?;
        let fitted_concentration =
            self.fitted_concentration
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict".to_string(),
                })?;

        let mut predictions = Array1::zeros(distance.len());

        for (i, &d) in distance.iter().enumerate() {
            if d <= fitted_distance[0] {
                predictions[i] = fitted_concentration[0];
            } else if d >= fitted_distance[fitted_distance.len() - 1] {
                predictions[i] = fitted_concentration[fitted_concentration.len() - 1];
            } else {
                for j in 1..fitted_distance.len() {
                    if d <= fitted_distance[j] {
                        let t = (d - fitted_distance[j - 1])
                            / (fitted_distance[j] - fitted_distance[j - 1]);
                        predictions[i] =
                            (1.0 - t) * fitted_concentration[j - 1] + t * fitted_concentration[j];
                        break;
                    }
                }
            }
        }

        Ok(predictions)
    }
}

/// Ecosystem Modeling
///
/// Models ecosystem dynamics using isotonic regression for species abundance,
/// biodiversity indices, and ecosystem services.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EcosystemMetric {
    /// Species richness
    SpeciesRichness,
    /// Shannon diversity index
    ShannonDiversity,
    /// Biomass
    Biomass,
    /// Productivity
    Productivity,
    /// Resilience index
    Resilience,
}

#[derive(Debug, Clone)]
pub struct EcosystemIsotonicRegression {
    /// Ecosystem metric
    metric: EcosystemMetric,
    /// Maximum iterations
    max_iterations: usize,
    /// Tolerance
    tolerance: f64,
    /// Fitted environmental gradient
    fitted_gradient: Option<Array1<f64>>,
    /// Fitted metric values
    fitted_metric: Option<Array1<f64>>,
}

impl Default for EcosystemIsotonicRegression {
    fn default() -> Self {
        Self {
            metric: EcosystemMetric::SpeciesRichness,
            max_iterations: 1000,
            tolerance: 1e-6,
            fitted_gradient: None,
            fitted_metric: None,
        }
    }
}

impl EcosystemIsotonicRegression {
    /// Create a new ecosystem model
    pub fn new() -> Self {
        Self::default()
    }

    /// Set ecosystem metric
    pub fn metric(mut self, m: EcosystemMetric) -> Self {
        self.metric = m;
        self
    }

    /// Fit the ecosystem response curve
    pub fn fit(
        &mut self,
        environmental_gradient: &Array1<f64>,
        metric_values: &Array1<f64>,
    ) -> Result<(), SklearsError> {
        if environmental_gradient.len() != metric_values.len() {
            return Err(SklearsError::InvalidInput(
                "Gradient and metric arrays must have the same length".to_string(),
            ));
        }

        // Sort by environmental gradient
        let mut indices: Vec<usize> = (0..environmental_gradient.len()).collect();
        indices.sort_by(|&i, &j| {
            environmental_gradient[i]
                .partial_cmp(&environmental_gradient[j])
                .unwrap()
        });

        let mut sorted_gradient = Array1::zeros(environmental_gradient.len());
        let mut sorted_metric = Array1::zeros(metric_values.len());

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            sorted_gradient[new_idx] = environmental_gradient[old_idx];
            sorted_metric[new_idx] = metric_values[old_idx];
        }

        // Apply appropriate monotonicity constraint based on metric
        let increasing = matches!(
            self.metric,
            EcosystemMetric::Biomass | EcosystemMetric::Productivity | EcosystemMetric::Resilience
        );

        let mut fitted_metric = sorted_metric.clone();

        for _ in 0..self.max_iterations {
            let old_metric = fitted_metric.clone();

            if increasing {
                for i in 1..fitted_metric.len() {
                    if fitted_metric[i] < fitted_metric[i - 1] {
                        fitted_metric[i] = fitted_metric[i - 1];
                    }
                }
            } else {
                // For species richness and diversity, allow any shape (use smoothing only)
                for i in 1..fitted_metric.len() - 1 {
                    fitted_metric[i] = 0.25 * sorted_metric[i - 1]
                        + 0.5 * sorted_metric[i]
                        + 0.25 * sorted_metric[i + 1];
                }
            }

            // Check convergence
            let diff: f64 = fitted_metric
                .iter()
                .zip(old_metric.iter())
                .map(|(new, old)| (new - old).powi(2))
                .sum();

            if diff.sqrt() < self.tolerance {
                break;
            }
        }

        self.fitted_gradient = Some(sorted_gradient);
        self.fitted_metric = Some(fitted_metric);

        Ok(())
    }

    /// Predict metric values at given gradients
    pub fn predict(&self, gradient: &Array1<f64>) -> Result<Array1<f64>, SklearsError> {
        let fitted_gradient =
            self.fitted_gradient
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict".to_string(),
                })?;
        let fitted_metric = self
            .fitted_metric
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let mut predictions = Array1::zeros(gradient.len());

        for (i, &g) in gradient.iter().enumerate() {
            if g <= fitted_gradient[0] {
                predictions[i] = fitted_metric[0];
            } else if g >= fitted_gradient[fitted_gradient.len() - 1] {
                predictions[i] = fitted_metric[fitted_metric.len() - 1];
            } else {
                for j in 1..fitted_gradient.len() {
                    if g <= fitted_gradient[j] {
                        let t = (g - fitted_gradient[j - 1])
                            / (fitted_gradient[j] - fitted_gradient[j - 1]);
                        predictions[i] = (1.0 - t) * fitted_metric[j - 1] + t * fitted_metric[j];
                        break;
                    }
                }
            }
        }

        Ok(predictions)
    }
}

// ============================================================================
// Function APIs
// ============================================================================

/// Fit environmental dose-response curve
pub fn environmental_dose_response_regression(
    dose: &Array1<f64>,
    response: &Array1<f64>,
    model_type: EnvironmentalDoseResponseModel,
) -> Result<(Array1<f64>, Array1<f64>), SklearsError> {
    let mut model = EnvironmentalDoseResponseRegression::new().model_type(model_type);
    model.fit(dose, response)?;

    Ok((model.fitted_dose.unwrap(), model.fitted_response.unwrap()))
}

/// Detect ecological thresholds
pub fn detect_ecological_threshold(
    pressure: &Array1<f64>,
    state: &Array1<f64>,
    method: ThresholdDetectionMethod,
) -> Result<Vec<f64>, SklearsError> {
    EcologicalThresholdEstimation::new()
        .method(method)
        .detect_threshold(pressure, state)
}

/// Analyze climate trend
pub fn analyze_climate_trend(
    time: &Array1<f64>,
    values: &Array1<f64>,
    trend_type: ClimateTrendType,
) -> Result<(Array1<f64>, Array1<f64>), SklearsError> {
    let mut model = ClimateTrendAnalysis::new().trend_type(trend_type);
    model.fit(time, values)?;

    Ok((model.fitted_time.unwrap(), model.fitted_trend.unwrap()))
}

/// Model pollution dispersion
pub fn pollution_dispersion_regression(
    distance: &Array1<f64>,
    concentration: &Array1<f64>,
    model_type: PollutionDispersionModel,
) -> Result<(Array1<f64>, Array1<f64>), SklearsError> {
    let mut model = PollutionDispersionRegression::new().model_type(model_type);
    model.fit(distance, concentration)?;

    Ok((
        model.fitted_distance.unwrap(),
        model.fitted_concentration.unwrap(),
    ))
}

/// Model ecosystem dynamics
pub fn ecosystem_isotonic_regression(
    environmental_gradient: &Array1<f64>,
    metric_values: &Array1<f64>,
    metric: EcosystemMetric,
) -> Result<(Array1<f64>, Array1<f64>), SklearsError> {
    let mut model = EcosystemIsotonicRegression::new().metric(metric);
    model.fit(environmental_gradient, metric_values)?;

    Ok((model.fitted_gradient.unwrap(), model.fitted_metric.unwrap()))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_environmental_dose_response() {
        let dose = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let response = Array1::from_vec(vec![0.0, 0.2, 0.5, 0.7, 0.85, 0.95]);

        let mut model = EnvironmentalDoseResponseRegression::new()
            .model_type(EnvironmentalDoseResponseModel::Linear);

        model.fit(&dose, &response).unwrap();

        let test_dose = Array1::from_vec(vec![1.5, 2.5, 3.5]);
        let predictions = model.predict(&test_dose).unwrap();

        // Check monotonicity
        for i in 1..predictions.len() {
            assert!(predictions[i] >= predictions[i - 1] - 1e-6);
        }
    }

    #[test]
    fn test_noael_estimation() {
        let dose = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let response = Array1::from_vec(vec![0.0, 0.1, 0.2, 0.6, 0.9]);

        let mut model = EnvironmentalDoseResponseRegression::new();
        model.fit(&dose, &response).unwrap();

        let noael = model.estimate_noael(0.5).unwrap();

        assert!(noael >= 0.0 && noael <= 4.0);
    }

    #[test]
    fn test_threshold_detection() {
        let pressure = Array1::from_vec((0..20).map(|i| i as f64).collect::<Vec<_>>());
        let mut state = Array1::zeros(20);

        // Create a step change at pressure = 10
        for i in 0..10 {
            state[i] = 1.0;
        }
        for i in 10..20 {
            state[i] = 5.0;
        }

        let thresholds =
            detect_ecological_threshold(&pressure, &state, ThresholdDetectionMethod::ChangePoint)
                .unwrap();

        // Should detect a threshold around pressure = 10
        assert!(!thresholds.is_empty());
    }

    #[test]
    fn test_climate_trend_analysis() {
        let time = Array1::from_vec(vec![1990.0, 1995.0, 2000.0, 2005.0, 2010.0, 2015.0, 2020.0]);
        let temp = Array1::from_vec(vec![14.0, 14.2, 14.3, 14.5, 14.7, 14.9, 15.1]);

        let mut model = ClimateTrendAnalysis::new().trend_type(ClimateTrendType::Temperature);

        model.fit(&time, &temp).unwrap();

        let test_time = Array1::from_vec(vec![1992.0, 2002.0, 2012.0]);
        let predictions = model.predict(&test_time).unwrap();

        // Check increasing trend
        for i in 1..predictions.len() {
            assert!(predictions[i] >= predictions[i - 1] - 1e-6);
        }

        // Estimate rate of change
        let rate = model.estimate_rate_of_change().unwrap();
        assert!(rate > 0.0); // Temperature is increasing
    }

    #[test]
    fn test_pollution_dispersion() {
        let distance = Array1::from_vec(vec![0.0, 10.0, 20.0, 30.0, 40.0, 50.0]);
        let concentration = Array1::from_vec(vec![100.0, 80.0, 50.0, 30.0, 15.0, 5.0]);

        let mut model =
            PollutionDispersionRegression::new().model_type(PollutionDispersionModel::Atmospheric);

        model.fit(&distance, &concentration).unwrap();

        let test_distance = Array1::from_vec(vec![15.0, 25.0, 35.0]);
        let predictions = model.predict(&test_distance).unwrap();

        // Check decreasing monotonicity
        for i in 1..predictions.len() {
            assert!(predictions[i] <= predictions[i - 1] + 1e-6);
        }

        // Check non-negative
        for &c in predictions.iter() {
            assert!(c >= 0.0);
        }
    }

    #[test]
    fn test_ecosystem_modeling() {
        let gradient = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let biomass = Array1::from_vec(vec![10.0, 15.0, 25.0, 40.0, 60.0, 85.0]);

        let mut model = EcosystemIsotonicRegression::new().metric(EcosystemMetric::Biomass);

        model.fit(&gradient, &biomass).unwrap();

        let test_gradient = Array1::from_vec(vec![0.5, 1.5, 2.5]);
        let predictions = model.predict(&test_gradient).unwrap();

        // Check increasing trend for biomass
        for i in 1..predictions.len() {
            assert!(predictions[i] >= predictions[i - 1] - 1e-6);
        }
    }

    #[test]
    fn test_regime_shift_detection() {
        let pressure = Array1::from_vec((0..30).map(|i| i as f64).collect::<Vec<_>>());
        let mut state = Array1::zeros(30);

        // Low variance regime
        for i in 0..15 {
            state[i] = 5.0 + (i as f64 % 3.0) * 0.1;
        }

        // High variance regime
        for i in 15..30 {
            state[i] = 10.0 + (i as f64 % 5.0);
        }

        let model = EcologicalThresholdEstimation::new()
            .method(ThresholdDetectionMethod::RegimeShift)
            .sensitivity(0.5);

        let shifts = model.detect_threshold(&pressure, &state).unwrap();

        // Should detect a regime shift
        assert!(!shifts.is_empty());
    }
}
