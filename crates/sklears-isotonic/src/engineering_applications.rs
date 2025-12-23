/// Engineering Applications Module for Isotonic Regression
///
/// This module implements isotonic regression methods for various engineering applications,
/// including stress-strain curve modeling, fatigue life estimation, reliability analysis,
/// control system constraints, and signal processing.
use scirs2_core::ndarray::Array1;
use sklears_core::prelude::SklearsError;

/// Types of stress-strain models
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StressStrainModel {
    /// Linear elastic model (Hooke's law)
    LinearElastic,
    /// Elastic-plastic model with yield point
    ElasticPlastic,
    /// Power law hardening model
    PowerLawHardening,
    /// Ramberg-Osgood model
    RambergOsgood,
    /// Ludwik model
    Ludwik,
    /// Swift model
    Swift,
}

/// Stress-Strain Curve Modeling
///
/// Models material behavior under loading using isotonic regression
/// to ensure physically meaningful monotonic stress-strain relationships.
#[derive(Debug, Clone)]
pub struct StressStrainIsotonicRegression {
    /// Stress-strain model type
    model_type: StressStrainModel,
    /// Elastic modulus (Young's modulus)
    elastic_modulus: f64,
    /// Yield stress
    yield_stress: Option<f64>,
    /// Maximum iterations for fitting
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: f64,
    /// Fitted stress-strain curve
    fitted_strain: Option<Array1<f64>>,
    fitted_stress: Option<Array1<f64>>,
}

impl Default for StressStrainIsotonicRegression {
    fn default() -> Self {
        Self {
            model_type: StressStrainModel::LinearElastic,
            elastic_modulus: 200e9, // Steel: ~200 GPa
            yield_stress: None,
            max_iterations: 1000,
            tolerance: 1e-6,
            fitted_strain: None,
            fitted_stress: None,
        }
    }
}

impl StressStrainIsotonicRegression {
    /// Create a new stress-strain model
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the stress-strain model type
    pub fn model_type(mut self, model: StressStrainModel) -> Self {
        self.model_type = model;
        self
    }

    /// Set elastic modulus
    pub fn elastic_modulus(mut self, e: f64) -> Self {
        self.elastic_modulus = e;
        self
    }

    /// Set yield stress
    pub fn yield_stress(mut self, sigma_y: f64) -> Self {
        self.yield_stress = Some(sigma_y);
        self
    }

    /// Set maximum iterations
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Set tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Fit the stress-strain curve with isotonic constraint
    pub fn fit(&mut self, strain: &Array1<f64>, stress: &Array1<f64>) -> Result<(), SklearsError> {
        if strain.len() != stress.len() {
            return Err(SklearsError::InvalidInput(
                "Strain and stress arrays must have the same length".to_string(),
            ));
        }

        // Sort by strain
        let mut indices: Vec<usize> = (0..strain.len()).collect();
        indices.sort_by(|&i, &j| strain[i].partial_cmp(&strain[j]).unwrap());

        let mut sorted_strain = Array1::zeros(strain.len());
        let mut sorted_stress = Array1::zeros(stress.len());

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            sorted_strain[new_idx] = strain[old_idx];
            sorted_stress[new_idx] = stress[old_idx];
        }

        // Apply isotonic regression (stress must be non-decreasing with strain)
        let mut fitted_stress = sorted_stress.clone();

        for _ in 0..self.max_iterations {
            let old_stress = fitted_stress.clone();

            // Pool Adjacent Violators to ensure monotonicity
            fitted_stress = self.pool_adjacent_violators(&fitted_stress)?;

            // Check convergence
            let diff: f64 = fitted_stress
                .iter()
                .zip(old_stress.iter())
                .map(|(new, old)| (new - old).powi(2))
                .sum();

            if diff.sqrt() < self.tolerance {
                break;
            }
        }

        self.fitted_strain = Some(sorted_strain);
        self.fitted_stress = Some(fitted_stress);

        Ok(())
    }

    /// Pool Adjacent Violators algorithm
    fn pool_adjacent_violators(&self, stress: &Array1<f64>) -> Result<Array1<f64>, SklearsError> {
        let mut result = stress.clone();

        // Simple PAV algorithm
        for i in 1..result.len() {
            if result[i] < result[i - 1] {
                result[i] = result[i - 1];
            }
        }

        Ok(result)
    }

    /// Predict stress for given strain values
    pub fn predict(&self, strain: &Array1<f64>) -> Result<Array1<f64>, SklearsError> {
        let fitted_strain = self
            .fitted_strain
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;
        let fitted_stress = self
            .fitted_stress
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let mut predictions = Array1::zeros(strain.len());

        for (i, &eps) in strain.iter().enumerate() {
            // Linear interpolation
            if eps <= fitted_strain[0] {
                predictions[i] = fitted_stress[0];
            } else if eps >= fitted_strain[fitted_strain.len() - 1] {
                predictions[i] = fitted_stress[fitted_stress.len() - 1];
            } else {
                // Find bracketing points
                for j in 1..fitted_strain.len() {
                    if eps <= fitted_strain[j] {
                        let t = (eps - fitted_strain[j - 1])
                            / (fitted_strain[j] - fitted_strain[j - 1]);
                        predictions[i] = (1.0 - t) * fitted_stress[j - 1] + t * fitted_stress[j];
                        break;
                    }
                }
            }
        }

        Ok(predictions)
    }

    /// Estimate elastic modulus from fitted curve
    pub fn estimate_elastic_modulus(&self) -> Result<f64, SklearsError> {
        let fitted_strain = self
            .fitted_strain
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "estimate_elastic_modulus".to_string(),
            })?;
        let fitted_stress = self
            .fitted_stress
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "estimate_elastic_modulus".to_string(),
            })?;

        // Estimate from the initial linear region (first 10% of data)
        let n_linear = (fitted_strain.len() / 10).max(2);
        let mut slope = 0.0;

        for i in 1..n_linear {
            let ds = fitted_stress[i] - fitted_stress[i - 1];
            let de = fitted_strain[i] - fitted_strain[i - 1];
            if de > 1e-10 {
                slope += ds / de;
            }
        }

        Ok(slope / (n_linear - 1) as f64)
    }
}

/// Fatigue Life Estimation
///
/// Estimates fatigue life of materials under cyclic loading using
/// isotonic regression with S-N curves (stress vs. number of cycles).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FatigueModel {
    /// Basquin model: log(S) = log(A) - b*log(N)
    Basquin,
    /// Coffin-Manson model (for low-cycle fatigue)
    CoffinManson,
    /// Paris law (for crack growth)
    ParisLaw,
    /// Modified Goodman model
    ModifiedGoodman,
}

#[derive(Debug, Clone)]
pub struct FatigueLifeIsotonicRegression {
    /// Fatigue model type
    model_type: FatigueModel,
    /// Maximum iterations
    max_iterations: usize,
    /// Tolerance
    tolerance: f64,
    /// Fitted stress amplitudes
    fitted_stress: Option<Array1<f64>>,
    /// Fitted cycles to failure
    fitted_cycles: Option<Array1<f64>>,
}

impl Default for FatigueLifeIsotonicRegression {
    fn default() -> Self {
        Self {
            model_type: FatigueModel::Basquin,
            max_iterations: 1000,
            tolerance: 1e-6,
            fitted_stress: None,
            fitted_cycles: None,
        }
    }
}

impl FatigueLifeIsotonicRegression {
    /// Create a new fatigue life model
    pub fn new() -> Self {
        Self::default()
    }

    /// Set fatigue model type
    pub fn model_type(mut self, model: FatigueModel) -> Self {
        self.model_type = model;
        self
    }

    /// Fit the S-N curve with isotonic constraint (stress decreases with cycles)
    pub fn fit(
        &mut self,
        cycles: &Array1<f64>,
        stress_amplitude: &Array1<f64>,
    ) -> Result<(), SklearsError> {
        if cycles.len() != stress_amplitude.len() {
            return Err(SklearsError::InvalidInput(
                "Cycles and stress arrays must have the same length".to_string(),
            ));
        }

        // Sort by number of cycles
        let mut indices: Vec<usize> = (0..cycles.len()).collect();
        indices.sort_by(|&i, &j| cycles[i].partial_cmp(&cycles[j]).unwrap());

        let mut sorted_cycles = Array1::zeros(cycles.len());
        let mut sorted_stress = Array1::zeros(stress_amplitude.len());

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            sorted_cycles[new_idx] = cycles[old_idx];
            sorted_stress[new_idx] = stress_amplitude[old_idx];
        }

        // Apply isotonic regression (stress must be non-increasing with cycles)
        let mut fitted_stress = sorted_stress.clone();

        for _ in 0..self.max_iterations {
            let old_stress = fitted_stress.clone();

            // Pool Adjacent Violators for decreasing constraint
            for i in 1..fitted_stress.len() {
                if fitted_stress[i] > fitted_stress[i - 1] {
                    fitted_stress[i] = fitted_stress[i - 1];
                }
            }

            // Check convergence
            let diff: f64 = fitted_stress
                .iter()
                .zip(old_stress.iter())
                .map(|(new, old)| (new - old).powi(2))
                .sum();

            if diff.sqrt() < self.tolerance {
                break;
            }
        }

        self.fitted_cycles = Some(sorted_cycles);
        self.fitted_stress = Some(fitted_stress);

        Ok(())
    }

    /// Predict stress amplitude for given number of cycles
    pub fn predict_stress(&self, cycles: &Array1<f64>) -> Result<Array1<f64>, SklearsError> {
        let fitted_cycles = self
            .fitted_cycles
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict_stress".to_string(),
            })?;
        let fitted_stress = self
            .fitted_stress
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict_stress".to_string(),
            })?;

        let mut predictions = Array1::zeros(cycles.len());

        for (i, &n) in cycles.iter().enumerate() {
            if n <= fitted_cycles[0] {
                predictions[i] = fitted_stress[0];
            } else if n >= fitted_cycles[fitted_cycles.len() - 1] {
                predictions[i] = fitted_stress[fitted_stress.len() - 1];
            } else {
                for j in 1..fitted_cycles.len() {
                    if n <= fitted_cycles[j] {
                        let t =
                            (n - fitted_cycles[j - 1]) / (fitted_cycles[j] - fitted_cycles[j - 1]);
                        predictions[i] = (1.0 - t) * fitted_stress[j - 1] + t * fitted_stress[j];
                        break;
                    }
                }
            }
        }

        Ok(predictions)
    }

    /// Predict number of cycles to failure for given stress amplitude
    pub fn predict_cycles(&self, stress: &Array1<f64>) -> Result<Array1<f64>, SklearsError> {
        let fitted_cycles = self
            .fitted_cycles
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict_cycles".to_string(),
            })?;
        let fitted_stress = self
            .fitted_stress
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict_cycles".to_string(),
            })?;

        let mut predictions = Array1::zeros(stress.len());

        for (i, &s) in stress.iter().enumerate() {
            // Find where this stress level appears in the fitted curve
            if s >= fitted_stress[0] {
                predictions[i] = fitted_cycles[0];
            } else if s <= fitted_stress[fitted_stress.len() - 1] {
                predictions[i] = fitted_cycles[fitted_cycles.len() - 1];
            } else {
                for j in 1..fitted_stress.len() {
                    if s >= fitted_stress[j] {
                        let t = (s - fitted_stress[j])
                            / (fitted_stress[j - 1] - fitted_stress[j] + 1e-10);
                        predictions[i] = (1.0 - t) * fitted_cycles[j] + t * fitted_cycles[j - 1];
                        break;
                    }
                }
            }
        }

        Ok(predictions)
    }
}

/// Reliability Modeling
///
/// Models reliability and failure rates using isotonic regression
/// with Weibull, exponential, or lognormal distributions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReliabilityModel {
    /// Weibull distribution
    Weibull,
    /// Exponential distribution
    Exponential,
    /// Lognormal distribution
    Lognormal,
    /// Bathtub curve model
    Bathtub,
}

#[derive(Debug, Clone)]
pub struct ReliabilityIsotonicRegression {
    /// Reliability model type
    model_type: ReliabilityModel,
    /// Maximum iterations
    max_iterations: usize,
    /// Tolerance
    tolerance: f64,
    /// Fitted time points
    fitted_time: Option<Array1<f64>>,
    /// Fitted reliability values
    fitted_reliability: Option<Array1<f64>>,
}

impl Default for ReliabilityIsotonicRegression {
    fn default() -> Self {
        Self {
            model_type: ReliabilityModel::Weibull,
            max_iterations: 1000,
            tolerance: 1e-6,
            fitted_time: None,
            fitted_reliability: None,
        }
    }
}

impl ReliabilityIsotonicRegression {
    /// Create a new reliability model
    pub fn new() -> Self {
        Self::default()
    }

    /// Set reliability model type
    pub fn model_type(mut self, model: ReliabilityModel) -> Self {
        self.model_type = model;
        self
    }

    /// Fit the reliability curve (reliability decreases with time)
    pub fn fit(
        &mut self,
        time: &Array1<f64>,
        reliability: &Array1<f64>,
    ) -> Result<(), SklearsError> {
        if time.len() != reliability.len() {
            return Err(SklearsError::InvalidInput(
                "Time and reliability arrays must have the same length".to_string(),
            ));
        }

        // Validate reliability values are between 0 and 1
        for &r in reliability.iter() {
            if !(0.0..=1.0).contains(&r) {
                return Err(SklearsError::InvalidInput(
                    "Reliability values must be between 0 and 1".to_string(),
                ));
            }
        }

        // Sort by time
        let mut indices: Vec<usize> = (0..time.len()).collect();
        indices.sort_by(|&i, &j| time[i].partial_cmp(&time[j]).unwrap());

        let mut sorted_time = Array1::zeros(time.len());
        let mut sorted_reliability = Array1::zeros(reliability.len());

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            sorted_time[new_idx] = time[old_idx];
            sorted_reliability[new_idx] = reliability[old_idx];
        }

        // Apply isotonic regression (reliability must be non-increasing with time)
        let mut fitted_reliability = sorted_reliability.clone();

        for _ in 0..self.max_iterations {
            let old_reliability = fitted_reliability.clone();

            // Pool Adjacent Violators for decreasing constraint
            for i in 1..fitted_reliability.len() {
                if fitted_reliability[i] > fitted_reliability[i - 1] {
                    fitted_reliability[i] = fitted_reliability[i - 1];
                }
            }

            // Ensure values stay in [0, 1]
            fitted_reliability.mapv_inplace(|v| v.clamp(0.0, 1.0));

            // Check convergence
            let diff: f64 = fitted_reliability
                .iter()
                .zip(old_reliability.iter())
                .map(|(new, old)| (new - old).powi(2))
                .sum();

            if diff.sqrt() < self.tolerance {
                break;
            }
        }

        self.fitted_time = Some(sorted_time);
        self.fitted_reliability = Some(fitted_reliability);

        Ok(())
    }

    /// Predict reliability at given time points
    pub fn predict(&self, time: &Array1<f64>) -> Result<Array1<f64>, SklearsError> {
        let fitted_time = self
            .fitted_time
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;
        let fitted_reliability =
            self.fitted_reliability
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict".to_string(),
                })?;

        let mut predictions = Array1::zeros(time.len());

        for (i, &t) in time.iter().enumerate() {
            if t <= fitted_time[0] {
                predictions[i] = fitted_reliability[0];
            } else if t >= fitted_time[fitted_time.len() - 1] {
                predictions[i] = fitted_reliability[fitted_reliability.len() - 1];
            } else {
                for j in 1..fitted_time.len() {
                    if t <= fitted_time[j] {
                        let tau = (t - fitted_time[j - 1]) / (fitted_time[j] - fitted_time[j - 1]);
                        predictions[i] =
                            (1.0 - tau) * fitted_reliability[j - 1] + tau * fitted_reliability[j];
                        break;
                    }
                }
            }
        }

        Ok(predictions)
    }

    /// Compute hazard rate (failure rate)
    pub fn hazard_rate(&self, time: &Array1<f64>) -> Result<Array1<f64>, SklearsError> {
        let reliability = self.predict(time)?;
        let fitted_time = self.fitted_time.as_ref().unwrap();
        let fitted_reliability = self.fitted_reliability.as_ref().unwrap();

        let mut hazard = Array1::zeros(time.len());

        for (i, &t) in time.iter().enumerate() {
            // h(t) = -d/dt[ln(R(t))] ≈ -[ln(R(t+dt)) - ln(R(t))]/dt
            let dt = 1e-6;

            let r_t = reliability[i];

            // Find reliability at t + dt
            let t_plus_dt = t + dt;
            let mut r_t_plus_dt = r_t;

            if t_plus_dt <= fitted_time[fitted_time.len() - 1] {
                for j in 1..fitted_time.len() {
                    if t_plus_dt <= fitted_time[j] {
                        let tau = (t_plus_dt - fitted_time[j - 1])
                            / (fitted_time[j] - fitted_time[j - 1]);
                        r_t_plus_dt =
                            (1.0 - tau) * fitted_reliability[j - 1] + tau * fitted_reliability[j];
                        break;
                    }
                }
            }

            if r_t > 1e-10 && r_t_plus_dt > 1e-10 {
                hazard[i] = -(r_t_plus_dt.ln() - r_t.ln()) / dt;
                hazard[i] = hazard[i].max(0.0); // Hazard rate must be non-negative
            }
        }

        Ok(hazard)
    }

    /// Compute mean time to failure (MTTF)
    pub fn mean_time_to_failure(&self) -> Result<f64, SklearsError> {
        let fitted_time = self
            .fitted_time
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "mean_time_to_failure".to_string(),
            })?;
        let fitted_reliability =
            self.fitted_reliability
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "mean_time_to_failure".to_string(),
                })?;

        // MTTF = ∫₀^∞ R(t) dt ≈ Σ R(t) Δt
        let mut mttf = 0.0;

        for i in 1..fitted_time.len() {
            let dt = fitted_time[i] - fitted_time[i - 1];
            let avg_reliability = 0.5 * (fitted_reliability[i] + fitted_reliability[i - 1]);
            mttf += avg_reliability * dt;
        }

        Ok(mttf)
    }
}

/// Control System Constraints
///
/// Applies isotonic constraints to control system outputs to ensure
/// stability, monotonicity, and physical feasibility.
#[derive(Debug, Clone)]
pub struct ControlSystemIsotonic {
    /// Maximum control input
    max_input: f64,
    /// Minimum control input
    min_input: f64,
    /// Maximum rate of change
    max_rate: Option<f64>,
    /// Whether output must be monotonic
    monotonic: bool,
    /// Maximum iterations
    max_iterations: usize,
    /// Tolerance
    tolerance: f64,
}

impl Default for ControlSystemIsotonic {
    fn default() -> Self {
        Self {
            max_input: 1.0,
            min_input: -1.0,
            max_rate: None,
            monotonic: false,
            max_iterations: 1000,
            tolerance: 1e-6,
        }
    }
}

impl ControlSystemIsotonic {
    /// Create a new control system isotonic regressor
    pub fn new() -> Self {
        Self::default()
    }

    /// Set input bounds
    pub fn input_bounds(mut self, min: f64, max: f64) -> Self {
        self.min_input = min;
        self.max_input = max;
        self
    }

    /// Set maximum rate of change
    pub fn max_rate(mut self, rate: f64) -> Self {
        self.max_rate = Some(rate);
        self
    }

    /// Set monotonicity constraint
    pub fn monotonic(mut self, mono: bool) -> Self {
        self.monotonic = mono;
        self
    }

    /// Apply constraints to control input sequence
    pub fn apply_constraints(
        &self,
        control_input: &Array1<f64>,
    ) -> Result<Array1<f64>, SklearsError> {
        let mut constrained = control_input.clone();

        // Apply input bounds
        constrained.mapv_inplace(|v| v.max(self.min_input).min(self.max_input));

        // Apply monotonicity constraint if required
        if self.monotonic {
            for i in 1..constrained.len() {
                if constrained[i] < constrained[i - 1] {
                    constrained[i] = constrained[i - 1];
                }
            }
        }

        // Apply rate constraints if specified
        if let Some(max_rate) = self.max_rate {
            for i in 1..constrained.len() {
                let rate = constrained[i] - constrained[i - 1];
                if rate.abs() > max_rate {
                    constrained[i] = constrained[i - 1] + rate.signum() * max_rate;
                }
            }
        }

        Ok(constrained)
    }

    /// Design monotonic controller gain schedule
    pub fn monotonic_gain_schedule(
        &self,
        operating_points: &Array1<f64>,
        desired_gains: &Array1<f64>,
    ) -> Result<Array1<f64>, SklearsError> {
        if operating_points.len() != desired_gains.len() {
            return Err(SklearsError::InvalidInput(
                "Operating points and gains must have same length".to_string(),
            ));
        }

        // Sort by operating points
        let mut indices: Vec<usize> = (0..operating_points.len()).collect();
        indices.sort_by(|&i, &j| {
            operating_points[i]
                .partial_cmp(&operating_points[j])
                .unwrap()
        });

        let mut sorted_gains = Array1::zeros(desired_gains.len());
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            sorted_gains[new_idx] = desired_gains[old_idx];
        }

        // Apply isotonic regression to ensure monotonicity
        let mut monotonic_gains = sorted_gains.clone();

        for i in 1..monotonic_gains.len() {
            if monotonic_gains[i] < monotonic_gains[i - 1] {
                monotonic_gains[i] = monotonic_gains[i - 1];
            }
        }

        Ok(monotonic_gains)
    }
}

/// Signal Processing Applications
///
/// Applies isotonic regression for signal smoothing, trend extraction,
/// and envelope detection with monotonicity constraints.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignalProcessingMode {
    /// Smoothing with monotonicity
    MonotonicSmoothing,
    /// Trend extraction
    TrendExtraction,
    /// Upper envelope detection
    UpperEnvelope,
    /// Lower envelope detection
    LowerEnvelope,
    /// Baseline correction
    BaselineCorrection,
}

#[derive(Debug, Clone)]
pub struct SignalProcessingIsotonic {
    /// Signal processing mode
    mode: SignalProcessingMode,
    /// Smoothing parameter
    smoothing: f64,
    /// Maximum iterations
    max_iterations: usize,
    /// Tolerance
    tolerance: f64,
}

impl Default for SignalProcessingIsotonic {
    fn default() -> Self {
        Self {
            mode: SignalProcessingMode::MonotonicSmoothing,
            smoothing: 0.1,
            max_iterations: 1000,
            tolerance: 1e-6,
        }
    }
}

impl SignalProcessingIsotonic {
    /// Create a new signal processing isotonic regressor
    pub fn new() -> Self {
        Self::default()
    }

    /// Set processing mode
    pub fn mode(mut self, mode: SignalProcessingMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set smoothing parameter
    pub fn smoothing(mut self, s: f64) -> Self {
        self.smoothing = s;
        self
    }

    /// Process signal with isotonic constraints
    pub fn process(&self, signal: &Array1<f64>) -> Result<Array1<f64>, SklearsError> {
        match self.mode {
            SignalProcessingMode::MonotonicSmoothing => self.monotonic_smoothing(signal),
            SignalProcessingMode::TrendExtraction => self.extract_trend(signal),
            SignalProcessingMode::UpperEnvelope => self.upper_envelope(signal),
            SignalProcessingMode::LowerEnvelope => self.lower_envelope(signal),
            SignalProcessingMode::BaselineCorrection => self.baseline_correction(signal),
        }
    }

    /// Monotonic smoothing
    fn monotonic_smoothing(&self, signal: &Array1<f64>) -> Result<Array1<f64>, SklearsError> {
        let mut smoothed = signal.clone();

        // Apply simple moving average
        let window = (self.smoothing * signal.len() as f64).max(3.0) as usize;
        let half_window = window / 2;

        for i in 0..signal.len() {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(signal.len());

            let mut sum = 0.0;
            for j in start..end {
                sum += signal[j];
            }
            smoothed[i] = sum / (end - start) as f64;
        }

        // Apply isotonic constraint
        for i in 1..smoothed.len() {
            if smoothed[i] < smoothed[i - 1] {
                smoothed[i] = smoothed[i - 1];
            }
        }

        Ok(smoothed)
    }

    /// Extract monotonic trend
    fn extract_trend(&self, signal: &Array1<f64>) -> Result<Array1<f64>, SklearsError> {
        // Use simple linear regression then apply isotonic constraint
        let n = signal.len() as f64;
        let x_mean = (signal.len() - 1) as f64 / 2.0;
        let y_mean: f64 = signal.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &y) in signal.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        let slope = if denominator > 1e-10 {
            numerator / denominator
        } else {
            0.0
        };
        let intercept = y_mean - slope * x_mean;

        let mut trend = Array1::zeros(signal.len());
        for i in 0..signal.len() {
            trend[i] = slope * i as f64 + intercept;
        }

        // Apply isotonic constraint
        for i in 1..trend.len() {
            if trend[i] < trend[i - 1] {
                trend[i] = trend[i - 1];
            }
        }

        Ok(trend)
    }

    /// Detect upper envelope
    fn upper_envelope(&self, signal: &Array1<f64>) -> Result<Array1<f64>, SklearsError> {
        let mut envelope = signal.clone();

        // Smooth the signal first
        let window = (self.smoothing * signal.len() as f64).max(3.0) as usize;
        let half_window = window / 2;

        for i in 0..signal.len() {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(signal.len());

            let mut max_val = f64::NEG_INFINITY;
            for j in start..end {
                max_val = max_val.max(signal[j]);
            }
            envelope[i] = max_val;
        }

        // Apply isotonic constraint (upper envelope should be non-decreasing for increasing signals)
        for i in 1..envelope.len() {
            if envelope[i] < envelope[i - 1] {
                envelope[i] = envelope[i - 1];
            }
        }

        Ok(envelope)
    }

    /// Detect lower envelope
    fn lower_envelope(&self, signal: &Array1<f64>) -> Result<Array1<f64>, SklearsError> {
        let mut envelope = signal.clone();

        // Smooth the signal first
        let window = (self.smoothing * signal.len() as f64).max(3.0) as usize;
        let half_window = window / 2;

        for i in 0..signal.len() {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(signal.len());

            let mut min_val = f64::INFINITY;
            for j in start..end {
                min_val = min_val.min(signal[j]);
            }
            envelope[i] = min_val;
        }

        // Apply isotonic constraint (lower envelope should be non-decreasing for increasing signals)
        for i in 1..envelope.len() {
            if envelope[i] < envelope[i - 1] {
                envelope[i] = envelope[i - 1];
            }
        }

        Ok(envelope)
    }

    /// Baseline correction
    fn baseline_correction(&self, signal: &Array1<f64>) -> Result<Array1<f64>, SklearsError> {
        // Extract lower envelope as baseline
        let baseline = self.lower_envelope(signal)?;

        // Subtract baseline
        let mut corrected = Array1::zeros(signal.len());
        for i in 0..signal.len() {
            corrected[i] = signal[i] - baseline[i];
        }

        Ok(corrected)
    }
}

// ============================================================================
// Function APIs
// ============================================================================

/// Fit stress-strain curve with isotonic regression
pub fn stress_strain_isotonic_regression(
    strain: &Array1<f64>,
    stress: &Array1<f64>,
    model_type: StressStrainModel,
) -> Result<(Array1<f64>, Array1<f64>), SklearsError> {
    let mut model = StressStrainIsotonicRegression::new().model_type(model_type);
    model.fit(strain, stress)?;

    Ok((model.fitted_strain.unwrap(), model.fitted_stress.unwrap()))
}

/// Fit fatigue life S-N curve with isotonic regression
pub fn fatigue_life_isotonic_regression(
    cycles: &Array1<f64>,
    stress_amplitude: &Array1<f64>,
    model_type: FatigueModel,
) -> Result<(Array1<f64>, Array1<f64>), SklearsError> {
    let mut model = FatigueLifeIsotonicRegression::new().model_type(model_type);
    model.fit(cycles, stress_amplitude)?;

    Ok((model.fitted_cycles.unwrap(), model.fitted_stress.unwrap()))
}

/// Fit reliability curve with isotonic regression
pub fn reliability_isotonic_regression(
    time: &Array1<f64>,
    reliability: &Array1<f64>,
    model_type: ReliabilityModel,
) -> Result<(Array1<f64>, Array1<f64>), SklearsError> {
    let mut model = ReliabilityIsotonicRegression::new().model_type(model_type);
    model.fit(time, reliability)?;

    Ok((
        model.fitted_time.unwrap(),
        model.fitted_reliability.unwrap(),
    ))
}

/// Apply control system constraints with isotonic regression
pub fn control_system_isotonic_constraints(
    control_input: &Array1<f64>,
    min_input: f64,
    max_input: f64,
    monotonic: bool,
) -> Result<Array1<f64>, SklearsError> {
    ControlSystemIsotonic::new()
        .input_bounds(min_input, max_input)
        .monotonic(monotonic)
        .apply_constraints(control_input)
}

/// Process signal with isotonic constraints
pub fn signal_processing_isotonic(
    signal: &Array1<f64>,
    mode: SignalProcessingMode,
    smoothing: f64,
) -> Result<Array1<f64>, SklearsError> {
    SignalProcessingIsotonic::new()
        .mode(mode)
        .smoothing(smoothing)
        .process(signal)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stress_strain_model() {
        // Generate synthetic stress-strain data
        let strain = Array1::from_vec(vec![0.0, 0.001, 0.002, 0.003, 0.004, 0.005]);
        let stress = Array1::from_vec(vec![0.0, 200.0, 400.0, 550.0, 650.0, 700.0]);

        let mut model =
            StressStrainIsotonicRegression::new().model_type(StressStrainModel::LinearElastic);

        model.fit(&strain, &stress).unwrap();

        // Predict at new strain values
        let test_strain = Array1::from_vec(vec![0.0015, 0.0025, 0.0035]);
        let predictions = model.predict(&test_strain).unwrap();

        assert_eq!(predictions.len(), test_strain.len());

        // Check monotonicity
        for i in 1..predictions.len() {
            assert!(predictions[i] >= predictions[i - 1]);
        }
    }

    #[test]
    fn test_fatigue_life_model() {
        // Generate synthetic S-N curve data
        let cycles = Array1::from_vec(vec![1e3, 1e4, 1e5, 1e6, 1e7]);
        let stress = Array1::from_vec(vec![500.0, 400.0, 300.0, 200.0, 150.0]);

        let mut model = FatigueLifeIsotonicRegression::new().model_type(FatigueModel::Basquin);

        model.fit(&cycles, &stress).unwrap();

        // Predict stress at intermediate cycles
        let test_cycles = Array1::from_vec(vec![5e3, 5e4, 5e5]);
        let predictions = model.predict_stress(&test_cycles).unwrap();

        assert_eq!(predictions.len(), test_cycles.len());

        // Check decreasing monotonicity
        for i in 1..predictions.len() {
            assert!(predictions[i] <= predictions[i - 1]);
        }
    }

    #[test]
    fn test_reliability_model() {
        // Generate synthetic reliability data
        let time = Array1::from_vec(vec![0.0, 100.0, 200.0, 300.0, 400.0, 500.0]);
        let reliability = Array1::from_vec(vec![1.0, 0.95, 0.85, 0.70, 0.50, 0.30]);

        let mut model = ReliabilityIsotonicRegression::new().model_type(ReliabilityModel::Weibull);

        model.fit(&time, &reliability).unwrap();

        // Predict reliability at intermediate times
        let test_time = Array1::from_vec(vec![50.0, 150.0, 250.0]);
        let predictions = model.predict(&test_time).unwrap();

        assert_eq!(predictions.len(), test_time.len());

        // Check values are in [0, 1]
        for &r in predictions.iter() {
            assert!(r >= 0.0 && r <= 1.0);
        }

        // Check decreasing monotonicity
        for i in 1..predictions.len() {
            assert!(predictions[i] <= predictions[i - 1] + 1e-6);
        }

        // Compute MTTF
        let mttf = model.mean_time_to_failure().unwrap();
        assert!(mttf > 0.0);
    }

    #[test]
    fn test_control_system_constraints() {
        let control_input = Array1::from_vec(vec![0.5, 1.2, -0.3, 0.8, 1.5]);

        let constrained =
            control_system_isotonic_constraints(&control_input, -1.0, 1.0, true).unwrap();

        // Check bounds
        for &c in constrained.iter() {
            assert!(c >= -1.0 && c <= 1.0);
        }

        // Check monotonicity
        for i in 1..constrained.len() {
            assert!(constrained[i] >= constrained[i - 1] - 1e-6);
        }
    }

    #[test]
    fn test_signal_processing() {
        // Generate noisy increasing signal
        let signal = Array1::from_vec(vec![1.0, 1.5, 1.3, 2.0, 2.5, 2.3, 3.0, 3.5]);

        let smoothed =
            signal_processing_isotonic(&signal, SignalProcessingMode::MonotonicSmoothing, 0.2)
                .unwrap();

        // Check monotonicity
        for i in 1..smoothed.len() {
            assert!(
                smoothed[i] >= smoothed[i - 1] - 1e-6,
                "Not monotonic at index {}",
                i
            );
        }

        // Test trend extraction
        let trend = signal_processing_isotonic(&signal, SignalProcessingMode::TrendExtraction, 0.1)
            .unwrap();

        assert_eq!(trend.len(), signal.len());
    }

    #[test]
    fn test_monotonic_gain_schedule() {
        let operating_points = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let desired_gains = Array1::from_vec(vec![1.0, 1.5, 1.3, 2.0, 2.5]);

        let controller = ControlSystemIsotonic::new().monotonic(true);

        let monotonic_gains = controller
            .monotonic_gain_schedule(&operating_points, &desired_gains)
            .unwrap();

        // Check monotonicity
        for i in 1..monotonic_gains.len() {
            assert!(monotonic_gains[i] >= monotonic_gains[i - 1]);
        }
    }

    #[test]
    fn test_envelope_detection() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0]);

        let upper_env =
            signal_processing_isotonic(&signal, SignalProcessingMode::UpperEnvelope, 0.3).unwrap();

        // Upper envelope should be above or equal to signal
        for i in 0..signal.len() {
            assert!(upper_env[i] >= signal[i] - 1e-6);
        }

        // Test lower envelope
        let lower_env =
            signal_processing_isotonic(&signal, SignalProcessingMode::LowerEnvelope, 0.3).unwrap();

        // Lower envelope should be below or equal to signal
        for i in 0..signal.len() {
            assert!(lower_env[i] <= signal[i] + 1e-6);
        }
    }

    #[test]
    fn test_elastic_modulus_estimation() {
        let strain = Array1::from_vec(vec![0.0, 0.001, 0.002, 0.003, 0.004]);
        let stress = Array1::from_vec(vec![0.0, 200.0, 400.0, 600.0, 800.0]);

        let mut model = StressStrainIsotonicRegression::new();
        model.fit(&strain, &stress).unwrap();

        let estimated_modulus = model.estimate_elastic_modulus().unwrap();

        // Should be close to 200 GPa / 1e9 = 200,000
        assert!(estimated_modulus > 150_000.0 && estimated_modulus < 250_000.0);
    }
}
