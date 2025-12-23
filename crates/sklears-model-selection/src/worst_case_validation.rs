//! Worst-Case Validation Scenarios
//!
//! This module provides worst-case scenario validation for robust model evaluation.
//! It generates challenging validation scenarios to test model robustness and reliability
//! under adverse conditions, including adversarial examples, distribution shifts, and
//! extreme data conditions.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::Rng;
use scirs2_core::random::SeedableRng;
use scirs2_core::SliceRandomExt;
use sklears_core::types::Float;
use std::collections::HashMap;

/// Worst-case scenario types
#[derive(Debug, Clone)]
pub enum WorstCaseScenario {
    /// Adversarial examples with maximum perturbation
    AdversarialExamples {
        epsilon: Float,

        attack_method: AdversarialAttackMethod,

        targeted: bool,
    },
    /// Distribution shift scenarios
    DistributionShift {
        shift_type: DistributionShiftType,

        severity: Float,
    },
    /// Extreme outliers and anomalies
    ExtremeOutliers {
        outlier_fraction: Float,
        outlier_magnitude: Float,
    },
    /// Class imbalance scenarios
    ClassImbalance {
        minority_fraction: Float,
        imbalance_ratio: Float,
    },
    /// Feature corruption scenarios
    FeatureCorruption {
        corruption_rate: Float,
        corruption_type: CorruptionType,
    },
    /// Temporal drift for time series
    TemporalDrift {
        drift_rate: Float,
        drift_pattern: DriftPattern,
    },
    /// Label noise scenarios
    LabelNoise {
        noise_rate: Float,
        noise_pattern: NoisePattern,
    },
    /// Missing data scenarios
    MissingData {
        missing_rate: Float,
        missing_pattern: MissingPattern,
    },
}

/// Adversarial attack methods
#[derive(Debug, Clone)]
pub enum AdversarialAttackMethod {
    /// Fast Gradient Sign Method
    FGSM,
    /// Projected Gradient Descent
    PGD { iterations: usize },
    /// Basic Iterative Method
    BIM { iterations: usize },
    /// Carlini & Wagner attack
    CW { confidence: Float },
    /// Boundary Attack
    BoundaryAttack { iterations: usize },
    /// Random noise attack
    RandomNoise,
}

/// Distribution shift types
#[derive(Debug, Clone)]
pub enum DistributionShiftType {
    /// Covariate shift (input distribution changes)
    CovariateShift,
    /// Prior probability shift (class distribution changes)
    PriorShift,
    /// Concept drift (relationship between input and output changes)
    ConceptDrift,
    /// Domain shift (complete domain change)
    DomainShift,
}

/// Corruption types for features
#[derive(Debug, Clone)]
pub enum CorruptionType {
    /// Gaussian noise
    GaussianNoise { std: Float },
    /// Salt and pepper noise
    SaltPepperNoise { ratio: Float },
    /// Multiplicative noise
    MultiplicativeNoise { factor: Float },
    /// Feature masking
    FeatureMasking,
    /// Value quantization
    Quantization { levels: usize },
}

/// Drift patterns for temporal data
#[derive(Debug, Clone)]
pub enum DriftPattern {
    /// Gradual linear drift
    Linear,
    /// Sudden step change
    Sudden,
    /// Exponential drift
    Exponential,
    /// Seasonal drift
    Seasonal { period: usize },
    /// Random walk drift
    RandomWalk,
}

/// Label noise patterns
#[derive(Debug, Clone)]
pub enum NoisePattern {
    /// Uniform random label flipping
    Uniform,
    /// Class-conditional noise (some classes more affected)
    ClassConditional { class_weights: Vec<Float> },
    /// Systematic bias towards specific classes
    SystematicBias { target_class: usize },
}

/// Missing data patterns
#[derive(Debug, Clone)]
pub enum MissingPattern {
    /// Missing completely at random
    MCAR,
    /// Missing at random (depends on observed variables)
    MAR,
    /// Missing not at random (depends on unobserved variables)
    MNAR,
    /// Block missing (consecutive features)
    BlockMissing { block_size: usize },
}

/// Worst-case validation configuration
#[derive(Debug, Clone)]
pub struct WorstCaseValidationConfig {
    pub scenarios: Vec<WorstCaseScenario>,
    pub n_worst_case_samples: usize,
    pub evaluation_metric: String,
    pub confidence_level: Float,
    pub random_state: Option<u64>,
    pub severity_levels: Vec<Float>,
}

/// Worst-case validation result
#[derive(Debug, Clone)]
pub struct WorstCaseValidationResult {
    pub scenario_results: HashMap<String, ScenarioResult>,
    pub overall_worst_case_score: Float,
    pub robustness_score: Float,
    pub failure_rate: Float,
    pub performance_degradation: Float,
    pub confidence_intervals: HashMap<String, (Float, Float)>,
}

/// Individual scenario result
#[derive(Debug, Clone)]
pub struct ScenarioResult {
    pub scenario_name: String,
    pub worst_case_score: Float,
    pub baseline_score: Float,
    pub performance_drop: Float,
    pub failure_examples: Vec<usize>,
    pub robustness_metrics: RobustnessMetrics,
}

/// Robustness metrics for validation
#[derive(Debug, Clone)]
pub struct RobustnessMetrics {
    pub stability_score: Float,
    pub consistency_score: Float,
    pub resilience_score: Float,
    pub recovery_score: Float,
    pub breakdown_point: Float,
}

/// Worst-case scenario generator
#[derive(Debug, Clone)]
pub struct WorstCaseScenarioGenerator {
    config: WorstCaseValidationConfig,
    rng: StdRng,
}

/// Worst-case validator
#[derive(Debug)]
pub struct WorstCaseValidator {
    generator: WorstCaseScenarioGenerator,
}

impl Default for WorstCaseValidationConfig {
    fn default() -> Self {
        Self {
            scenarios: vec![
                WorstCaseScenario::AdversarialExamples {
                    epsilon: 0.1,
                    attack_method: AdversarialAttackMethod::FGSM,
                    targeted: false,
                },
                WorstCaseScenario::DistributionShift {
                    shift_type: DistributionShiftType::CovariateShift,
                    severity: 1.0,
                },
                WorstCaseScenario::ExtremeOutliers {
                    outlier_fraction: 0.1,
                    outlier_magnitude: 3.0,
                },
            ],
            n_worst_case_samples: 1000,
            evaluation_metric: "accuracy".to_string(),
            confidence_level: 0.95,
            random_state: None,
            severity_levels: vec![0.5, 1.0, 1.5, 2.0],
        }
    }
}

impl WorstCaseScenarioGenerator {
    /// Create a new worst-case scenario generator
    pub fn new(config: WorstCaseValidationConfig) -> Self {
        let rng = match config.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                use scirs2_core::random::thread_rng;
                StdRng::from_rng(&mut thread_rng())
            }
        };

        Self { config, rng }
    }

    /// Generate worst-case scenarios for given data
    pub fn generate_scenarios(
        &mut self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<Vec<(Array2<Float>, Array1<Float>, String)>, Box<dyn std::error::Error>> {
        let mut scenarios = Vec::new();

        let scenarios_clone = self.config.scenarios.clone();
        let severity_levels_clone = self.config.severity_levels.clone();

        for scenario in &scenarios_clone {
            for &severity in &severity_levels_clone {
                let (worst_x, worst_y, name) =
                    self.generate_single_scenario(x, y, scenario, severity)?;
                scenarios.push((worst_x, worst_y, name));
            }
        }

        Ok(scenarios)
    }

    /// Generate a single worst-case scenario
    fn generate_single_scenario(
        &mut self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        scenario: &WorstCaseScenario,
        severity: Float,
    ) -> Result<(Array2<Float>, Array1<Float>, String), Box<dyn std::error::Error>> {
        match scenario {
            WorstCaseScenario::AdversarialExamples {
                epsilon,
                attack_method,
                targeted,
            } => {
                let (adv_x, adv_y) = self.generate_adversarial_examples(
                    x,
                    y,
                    *epsilon * severity,
                    attack_method,
                    *targeted,
                )?;
                let name = format!(
                    "Adversarial_{:?}_eps_{:.3}",
                    attack_method,
                    epsilon * severity
                );
                Ok((adv_x, adv_y, name))
            }
            WorstCaseScenario::DistributionShift {
                shift_type,
                severity: base_severity,
            } => {
                let (shift_x, shift_y) =
                    self.generate_distribution_shift(x, y, shift_type, base_severity * severity)?;
                let name = format!(
                    "DistShift_{:?}_sev_{:.2}",
                    shift_type,
                    base_severity * severity
                );
                Ok((shift_x, shift_y, name))
            }
            WorstCaseScenario::ExtremeOutliers {
                outlier_fraction,
                outlier_magnitude,
            } => {
                let (outlier_x, outlier_y) = self.generate_extreme_outliers(
                    x,
                    y,
                    *outlier_fraction,
                    outlier_magnitude * severity,
                )?;
                let name = format!(
                    "Outliers_frac_{:.2}_mag_{:.2}",
                    outlier_fraction,
                    outlier_magnitude * severity
                );
                Ok((outlier_x, outlier_y, name))
            }
            WorstCaseScenario::ClassImbalance {
                minority_fraction,
                imbalance_ratio,
            } => {
                let (imbal_x, imbal_y) = self.generate_class_imbalance(
                    x,
                    y,
                    *minority_fraction,
                    imbalance_ratio * severity,
                )?;
                let name = format!(
                    "ClassImbalance_frac_{:.2}_ratio_{:.2}",
                    minority_fraction,
                    imbalance_ratio * severity
                );
                Ok((imbal_x, imbal_y, name))
            }
            WorstCaseScenario::FeatureCorruption {
                corruption_rate,
                corruption_type,
            } => {
                let (corr_x, corr_y) = self.generate_feature_corruption(
                    x,
                    y,
                    corruption_rate * severity,
                    corruption_type,
                )?;
                let name = format!(
                    "Corruption_{:?}_rate_{:.2}",
                    corruption_type,
                    corruption_rate * severity
                );
                Ok((corr_x, corr_y, name))
            }
            WorstCaseScenario::TemporalDrift {
                drift_rate,
                drift_pattern,
            } => {
                let (drift_x, drift_y) =
                    self.generate_temporal_drift(x, y, drift_rate * severity, drift_pattern)?;
                let name = format!(
                    "TemporalDrift_{:?}_rate_{:.2}",
                    drift_pattern,
                    drift_rate * severity
                );
                Ok((drift_x, drift_y, name))
            }
            WorstCaseScenario::LabelNoise {
                noise_rate,
                noise_pattern,
            } => {
                let (noise_x, noise_y) =
                    self.generate_label_noise(x, y, noise_rate * severity, noise_pattern)?;
                let name = format!(
                    "LabelNoise_{:?}_rate_{:.2}",
                    noise_pattern,
                    noise_rate * severity
                );
                Ok((noise_x, noise_y, name))
            }
            WorstCaseScenario::MissingData {
                missing_rate,
                missing_pattern,
            } => {
                let (missing_x, missing_y) =
                    self.generate_missing_data(x, y, missing_rate * severity, missing_pattern)?;
                let name = format!(
                    "MissingData_{:?}_rate_{:.2}",
                    missing_pattern,
                    missing_rate * severity
                );
                Ok((missing_x, missing_y, name))
            }
        }
    }

    /// Generate adversarial examples
    fn generate_adversarial_examples(
        &mut self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        epsilon: Float,
        attack_method: &AdversarialAttackMethod,
        _targeted: bool,
    ) -> Result<(Array2<Float>, Array1<Float>), Box<dyn std::error::Error>> {
        let mut adv_x = x.clone();

        match attack_method {
            AdversarialAttackMethod::FGSM => {
                // Fast Gradient Sign Method
                for mut row in adv_x.axis_iter_mut(Axis(0)) {
                    for val in row.iter_mut() {
                        let perturbation = if self.rng.gen_bool(0.5) {
                            epsilon
                        } else {
                            -epsilon
                        };
                        *val += perturbation;
                    }
                }
            }
            AdversarialAttackMethod::PGD { iterations } => {
                // Projected Gradient Descent
                for _ in 0..*iterations {
                    for mut row in adv_x.axis_iter_mut(Axis(0)) {
                        for val in row.iter_mut() {
                            let step_size = epsilon / (*iterations as Float);
                            let perturbation = if self.rng.gen_bool(0.5) {
                                step_size
                            } else {
                                -step_size
                            };
                            *val += perturbation;
                            // Project back to epsilon ball
                            *val = val.max(-epsilon).min(epsilon);
                        }
                    }
                }
            }
            AdversarialAttackMethod::BIM { iterations } => {
                // Basic Iterative Method
                let alpha = epsilon / (*iterations as Float);
                for _ in 0..*iterations {
                    for mut row in adv_x.axis_iter_mut(Axis(0)) {
                        for val in row.iter_mut() {
                            let perturbation = if self.rng.gen_bool(0.5) {
                                alpha
                            } else {
                                -alpha
                            };
                            *val += perturbation;
                        }
                    }
                }
            }
            AdversarialAttackMethod::RandomNoise => {
                // Random noise attack
                for mut row in adv_x.axis_iter_mut(Axis(0)) {
                    for val in row.iter_mut() {
                        let noise = self.rng.gen_range(-epsilon..epsilon + 1.0);
                        *val += noise;
                    }
                }
            }
            AdversarialAttackMethod::CW { .. } | AdversarialAttackMethod::BoundaryAttack { .. } => {
                // Simplified implementation for C&W and Boundary Attack
                for mut row in adv_x.axis_iter_mut(Axis(0)) {
                    for val in row.iter_mut() {
                        let perturbation = self.rng.gen_range(-epsilon..epsilon + 1.0);
                        *val += perturbation;
                    }
                }
            }
        }

        Ok((adv_x, y.clone()))
    }

    /// Generate distribution shift scenarios
    fn generate_distribution_shift(
        &mut self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        shift_type: &DistributionShiftType,
        severity: Float,
    ) -> Result<(Array2<Float>, Array1<Float>), Box<dyn std::error::Error>> {
        let mut shift_x = x.clone();
        let mut shift_y = y.clone();

        match shift_type {
            DistributionShiftType::CovariateShift => {
                // Add systematic bias to features
                for mut row in shift_x.axis_iter_mut(Axis(0)) {
                    for (i, val) in row.iter_mut().enumerate() {
                        let shift = severity * (i as Float * 0.1).sin();
                        *val += shift;
                    }
                }
            }
            DistributionShiftType::PriorShift => {
                // Change class distribution by removing samples from certain classes
                let mut unique_classes: Vec<Float> = y.iter().cloned().collect();
                unique_classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
                unique_classes.dedup();
                if unique_classes.len() > 1 {
                    let target_class = unique_classes[0];
                    let removal_prob = severity * 0.5;

                    let mut keep_indices = Vec::new();
                    for (i, &class) in y.iter().enumerate() {
                        if class != target_class || self.rng.random::<Float>() > removal_prob {
                            keep_indices.push(i);
                        }
                    }

                    // Create new arrays with selected indices
                    let mut new_x_data = Vec::new();
                    for &i in keep_indices.iter() {
                        new_x_data.extend(x.row(i).iter().cloned());
                    }
                    let new_x =
                        Array2::from_shape_vec((keep_indices.len(), x.ncols()), new_x_data)?;
                    let new_y = Array1::from_vec(keep_indices.iter().map(|&i| y[i]).collect());

                    return Ok((new_x, new_y));
                }
            }
            DistributionShiftType::ConceptDrift => {
                // Change the relationship between features and labels
                for label in shift_y.iter_mut() {
                    if self.rng.random::<Float>() < severity * 0.2 {
                        // Flip some labels to simulate concept drift
                        *label = 1.0 - *label;
                    }
                }
            }
            DistributionShiftType::DomainShift => {
                // Apply domain transformation
                for mut row in shift_x.axis_iter_mut(Axis(0)) {
                    for val in row.iter_mut() {
                        // Apply non-linear transformation
                        *val = val.tanh() * severity;
                    }
                }
            }
        }

        Ok((shift_x, shift_y))
    }

    /// Generate extreme outliers
    fn generate_extreme_outliers(
        &mut self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        outlier_fraction: Float,
        outlier_magnitude: Float,
    ) -> Result<(Array2<Float>, Array1<Float>), Box<dyn std::error::Error>> {
        let mut outlier_x = x.clone();
        let n_outliers = (x.nrows() as Float * outlier_fraction) as usize;

        let mut outlier_indices: Vec<usize> = (0..x.nrows()).collect();
        outlier_indices.shuffle(&mut self.rng);
        outlier_indices.truncate(n_outliers);

        for &idx in &outlier_indices {
            for val in outlier_x.row_mut(idx) {
                let outlier_value = self
                    .rng
                    .gen_range(-outlier_magnitude..outlier_magnitude + 1.0);
                *val += outlier_value;
            }
        }

        Ok((outlier_x, y.clone()))
    }

    /// Generate class imbalance scenarios
    fn generate_class_imbalance(
        &mut self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        minority_fraction: Float,
        _imbalance_ratio: Float,
    ) -> Result<(Array2<Float>, Array1<Float>), Box<dyn std::error::Error>> {
        let mut unique_classes: Vec<Float> = y.iter().cloned().collect();
        unique_classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique_classes.dedup();
        if unique_classes.len() < 2 {
            return Ok((x.clone(), y.clone()));
        }

        let minority_class = unique_classes[0];
        let target_minority_count = (x.nrows() as Float * minority_fraction) as usize;

        let mut keep_indices = Vec::new();
        let mut minority_count = 0;

        for (i, &class) in y.iter().enumerate() {
            if class == minority_class {
                if minority_count < target_minority_count {
                    keep_indices.push(i);
                    minority_count += 1;
                }
            } else {
                keep_indices.push(i);
            }
        }

        let mut new_x_data = Vec::new();
        for &i in keep_indices.iter() {
            new_x_data.extend(x.row(i).iter().cloned());
        }
        let new_x = Array2::from_shape_vec((keep_indices.len(), x.ncols()), new_x_data)?;
        let new_y = Array1::from_vec(keep_indices.iter().map(|&i| y[i]).collect());

        Ok((new_x, new_y))
    }

    /// Generate feature corruption
    fn generate_feature_corruption(
        &mut self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        corruption_rate: Float,
        corruption_type: &CorruptionType,
    ) -> Result<(Array2<Float>, Array1<Float>), Box<dyn std::error::Error>> {
        let mut corrupted_x = x.clone();

        match corruption_type {
            CorruptionType::GaussianNoise { std } => {
                for val in corrupted_x.iter_mut() {
                    if self.rng.random::<Float>() < corruption_rate {
                        let noise = self.rng.random::<Float>() * std;
                        *val += noise;
                    }
                }
            }
            CorruptionType::SaltPepperNoise { ratio } => {
                for val in corrupted_x.iter_mut() {
                    if self.rng.random::<Float>() < corruption_rate {
                        *val = if self.rng.random::<Float>() < *ratio {
                            1.0
                        } else {
                            0.0
                        };
                    }
                }
            }
            CorruptionType::MultiplicativeNoise { factor } => {
                for val in corrupted_x.iter_mut() {
                    if self.rng.random::<Float>() < corruption_rate {
                        let noise = 1.0 + (self.rng.random::<Float>() - 0.5) * factor;
                        *val *= noise;
                    }
                }
            }
            CorruptionType::FeatureMasking => {
                for val in corrupted_x.iter_mut() {
                    if self.rng.random::<Float>() < corruption_rate {
                        *val = 0.0;
                    }
                }
            }
            CorruptionType::Quantization { levels } => {
                let step_size = 2.0 / (*levels as Float);
                for val in corrupted_x.iter_mut() {
                    if self.rng.random::<Float>() < corruption_rate {
                        *val = ((*val / step_size).round() * step_size).clamp(-1.0, 1.0);
                    }
                }
            }
        }

        Ok((corrupted_x, y.clone()))
    }

    /// Generate temporal drift
    fn generate_temporal_drift(
        &mut self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        drift_rate: Float,
        drift_pattern: &DriftPattern,
    ) -> Result<(Array2<Float>, Array1<Float>), Box<dyn std::error::Error>> {
        let mut drift_x = x.clone();
        let n_samples = x.nrows();

        for (t, row) in drift_x.axis_iter_mut(Axis(0)).enumerate() {
            let time_factor = t as Float / n_samples as Float;

            let drift_magnitude = match drift_pattern {
                DriftPattern::Linear => drift_rate * time_factor,
                DriftPattern::Sudden => {
                    if time_factor > 0.5 {
                        drift_rate
                    } else {
                        0.0
                    }
                }
                DriftPattern::Exponential => drift_rate * time_factor.exp(),
                DriftPattern::Seasonal { period } => {
                    drift_rate
                        * (2.0 * std::f64::consts::PI * t as Float / *period as Float).sin()
                            as Float
                }
                DriftPattern::RandomWalk => drift_rate * self.rng.random::<Float>(),
            };

            for val in row {
                *val += drift_magnitude;
            }
        }

        Ok((drift_x, y.clone()))
    }

    /// Generate label noise
    fn generate_label_noise(
        &mut self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        noise_rate: Float,
        noise_pattern: &NoisePattern,
    ) -> Result<(Array2<Float>, Array1<Float>), Box<dyn std::error::Error>> {
        let mut noisy_y = y.clone();
        let mut unique_classes: Vec<Float> = y.iter().cloned().collect();
        unique_classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique_classes.dedup();

        if unique_classes.len() < 2 {
            return Ok((x.clone(), noisy_y));
        }

        match noise_pattern {
            NoisePattern::Uniform => {
                for label in noisy_y.iter_mut() {
                    if self.rng.random::<Float>() < noise_rate {
                        // Flip to random other class
                        let other_classes: Vec<Float> = unique_classes
                            .iter()
                            .filter(|&&c| c != *label)
                            .cloned()
                            .collect();
                        if !other_classes.is_empty() {
                            *label = other_classes[self.rng.gen_range(0..other_classes.len())];
                        }
                    }
                }
            }
            NoisePattern::ClassConditional { class_weights } => {
                for label in noisy_y.iter_mut() {
                    let class_idx = unique_classes
                        .iter()
                        .position(|&c| c == *label)
                        .unwrap_or(0);
                    let class_noise_rate = if class_idx < class_weights.len() {
                        noise_rate * class_weights[class_idx]
                    } else {
                        noise_rate
                    };

                    if self.rng.random::<Float>() < class_noise_rate {
                        let other_classes: Vec<Float> = unique_classes
                            .iter()
                            .filter(|&&c| c != *label)
                            .cloned()
                            .collect();
                        if !other_classes.is_empty() {
                            *label = other_classes[self.rng.gen_range(0..other_classes.len())];
                        }
                    }
                }
            }
            NoisePattern::SystematicBias { target_class } => {
                let target_class_value = if *target_class < unique_classes.len() {
                    unique_classes[*target_class]
                } else {
                    unique_classes[0]
                };

                for label in noisy_y.iter_mut() {
                    if self.rng.random::<Float>() < noise_rate {
                        *label = target_class_value;
                    }
                }
            }
        }

        Ok((x.clone(), noisy_y))
    }

    /// Generate missing data scenarios
    fn generate_missing_data(
        &mut self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        missing_rate: Float,
        missing_pattern: &MissingPattern,
    ) -> Result<(Array2<Float>, Array1<Float>), Box<dyn std::error::Error>> {
        let mut missing_x = x.clone();

        match missing_pattern {
            MissingPattern::MCAR => {
                // Missing completely at random
                for val in missing_x.iter_mut() {
                    if self.rng.random::<Float>() < missing_rate {
                        *val = Float::NAN;
                    }
                }
            }
            MissingPattern::MAR => {
                // Missing at random (depends on other features)
                for row in missing_x.axis_iter_mut(Axis(0)) {
                    let row_mean =
                        row.iter().filter(|v| v.is_finite()).sum::<Float>() / row.len() as Float;
                    let missing_prob = if row_mean > 0.0 {
                        missing_rate * 1.5
                    } else {
                        missing_rate * 0.5
                    };

                    for val in row {
                        if self.rng.random::<Float>() < missing_prob {
                            *val = Float::NAN;
                        }
                    }
                }
            }
            MissingPattern::MNAR => {
                // Missing not at random (depends on the value itself)
                for val in missing_x.iter_mut() {
                    let missing_prob = if *val > 0.5 {
                        missing_rate * 2.0
                    } else {
                        missing_rate * 0.5
                    };
                    if self.rng.random::<Float>() < missing_prob {
                        *val = Float::NAN;
                    }
                }
            }
            MissingPattern::BlockMissing { block_size } => {
                // Block missing (consecutive features)
                let n_cols = missing_x.ncols();
                let n_blocks = (missing_rate * n_cols as Float) as usize / block_size;

                for _ in 0..n_blocks {
                    let start_col = self.rng.gen_range(0..n_cols.saturating_sub(*block_size));
                    let end_col = (start_col + block_size).min(n_cols);

                    for mut row in missing_x.axis_iter_mut(Axis(0)) {
                        for j in start_col..end_col {
                            row[j] = Float::NAN;
                        }
                    }
                }
            }
        }

        Ok((missing_x, y.clone()))
    }
}

impl WorstCaseValidator {
    /// Create a new worst-case validator
    pub fn new(config: WorstCaseValidationConfig) -> Self {
        let generator = WorstCaseScenarioGenerator::new(config);
        Self { generator }
    }

    /// Validate model robustness under worst-case scenarios
    pub fn validate<F>(
        &mut self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        model_fn: F,
    ) -> Result<WorstCaseValidationResult, Box<dyn std::error::Error>>
    where
        F: Fn(&Array2<Float>, &Array1<Float>) -> Result<Float, Box<dyn std::error::Error>>,
    {
        // Baseline performance
        let baseline_score = model_fn(x, y)?;

        // Generate worst-case scenarios
        let scenarios = self.generator.generate_scenarios(x, y)?;

        let mut scenario_results = HashMap::new();
        let mut all_scores = Vec::new();
        let mut failure_count = 0;

        for (scenario_x, scenario_y, scenario_name) in scenarios {
            let scenario_score = model_fn(&scenario_x, &scenario_y).unwrap_or(0.0);
            all_scores.push(scenario_score);

            let performance_drop = (baseline_score - scenario_score) / baseline_score;

            // Check for failure (significant performance drop)
            if performance_drop > 0.5 {
                failure_count += 1;
            }

            let robustness_metrics =
                self.calculate_robustness_metrics(baseline_score, scenario_score, &scenario_x, x);

            let result = ScenarioResult {
                scenario_name: scenario_name.clone(),
                worst_case_score: scenario_score,
                baseline_score,
                performance_drop,
                failure_examples: vec![], // Could be populated with specific failure indices
                robustness_metrics,
            };

            scenario_results.insert(scenario_name, result);
        }

        let overall_worst_case_score = all_scores.iter().fold(Float::INFINITY, |a, &b| a.min(b));

        let performance_degradation = (baseline_score - overall_worst_case_score) / baseline_score;
        let failure_rate = failure_count as Float / all_scores.len() as Float;
        let robustness_score = 1.0 - performance_degradation;

        // Calculate confidence intervals (simplified)
        let mut confidence_intervals = HashMap::new();
        for (scenario_name, result) in &scenario_results {
            let ci_lower = result.worst_case_score * 0.9;
            let ci_upper = result.worst_case_score * 1.1;
            confidence_intervals.insert(scenario_name.clone(), (ci_lower, ci_upper));
        }

        Ok(WorstCaseValidationResult {
            scenario_results,
            overall_worst_case_score,
            robustness_score,
            failure_rate,
            performance_degradation,
            confidence_intervals,
        })
    }

    /// Calculate robustness metrics
    fn calculate_robustness_metrics(
        &self,
        baseline_score: Float,
        scenario_score: Float,
        scenario_x: &Array2<Float>,
        original_x: &Array2<Float>,
    ) -> RobustnessMetrics {
        let stability_score = (scenario_score / baseline_score).min(1.0);

        // Calculate data similarity for consistency score
        let data_similarity = self.calculate_data_similarity(scenario_x, original_x);
        let consistency_score = stability_score * data_similarity;

        let resilience_score = if scenario_score > baseline_score * 0.7 {
            1.0
        } else {
            0.0
        };
        let recovery_score = stability_score; // Simplified
        let breakdown_point = 1.0 - stability_score;

        RobustnessMetrics {
            stability_score,
            consistency_score,
            resilience_score,
            recovery_score,
            breakdown_point,
        }
    }

    /// Calculate similarity between datasets
    fn calculate_data_similarity(&self, x1: &Array2<Float>, x2: &Array2<Float>) -> Float {
        if x1.dim() != x2.dim() {
            return 0.0;
        }

        let mut similarity_sum = 0.0;
        let mut count = 0;

        for (row1, row2) in x1.axis_iter(Axis(0)).zip(x2.axis_iter(Axis(0))) {
            let mut row_similarity = 0.0;
            let mut valid_features = 0;

            for (&val1, &val2) in row1.iter().zip(row2.iter()) {
                if val1.is_finite() && val2.is_finite() {
                    row_similarity += 1.0 - (val1 - val2).abs();
                    valid_features += 1;
                }
            }

            if valid_features > 0 {
                similarity_sum += row_similarity / valid_features as Float;
                count += 1;
            }
        }

        if count > 0 {
            similarity_sum / count as Float
        } else {
            0.0
        }
    }
}

/// Convenience function for worst-case validation
pub fn worst_case_validate<F>(
    x: &Array2<Float>,
    y: &Array1<Float>,
    model_fn: F,
    config: Option<WorstCaseValidationConfig>,
) -> Result<WorstCaseValidationResult, Box<dyn std::error::Error>>
where
    F: Fn(&Array2<Float>, &Array1<Float>) -> Result<Float, Box<dyn std::error::Error>>,
{
    let config = config.unwrap_or_default();
    let mut validator = WorstCaseValidator::new(config);
    validator.validate(x, y, model_fn)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worst_case_scenario_generator() {
        let config = WorstCaseValidationConfig::default();
        let mut generator = WorstCaseScenarioGenerator::new(config);

        let x = Array2::from_shape_vec((10, 3), (0..30).map(|i| i as Float).collect()).unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);

        let scenarios = generator.generate_scenarios(&x, &y).unwrap();
        assert!(!scenarios.is_empty());
    }

    #[test]
    fn test_adversarial_example_generation() {
        let config = WorstCaseValidationConfig::default();
        let mut generator = WorstCaseScenarioGenerator::new(config);

        let x = Array2::from_shape_vec((5, 3), (0..15).map(|i| i as Float).collect()).unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0]);

        let (adv_x, adv_y) = generator
            .generate_adversarial_examples(&x, &y, 0.1, &AdversarialAttackMethod::FGSM, false)
            .unwrap();

        assert_eq!(adv_x.dim(), x.dim());
        assert_eq!(adv_y.len(), y.len());
    }

    #[test]
    fn test_worst_case_validation() {
        let config = WorstCaseValidationConfig {
            scenarios: vec![WorstCaseScenario::ExtremeOutliers {
                outlier_fraction: 0.1,
                outlier_magnitude: 2.0,
            }],
            n_worst_case_samples: 100,
            severity_levels: vec![1.0],
            ..Default::default()
        };

        let x =
            Array2::from_shape_vec((10, 3), (0..30).map(|i| i as Float * 0.1).collect()).unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);

        let model_fn =
            |_x: &Array2<Float>, _y: &Array1<Float>| -> Result<Float, Box<dyn std::error::Error>> {
                Ok(0.8) // Mock accuracy
            };

        let result = worst_case_validate(&x, &y, model_fn, Some(config)).unwrap();

        assert!(result.robustness_score >= 0.0);
        assert!(result.robustness_score <= 1.0);
        assert!(!result.scenario_results.is_empty());
    }

    #[test]
    fn test_label_noise_generation() {
        let config = WorstCaseValidationConfig::default();
        let mut generator = WorstCaseScenarioGenerator::new(config);

        let x = Array2::from_shape_vec((10, 3), (0..30).map(|i| i as Float).collect()).unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);

        let (noisy_x, noisy_y) = generator
            .generate_label_noise(&x, &y, 0.2, &NoisePattern::Uniform)
            .unwrap();

        assert_eq!(noisy_x.dim(), x.dim());
        assert_eq!(noisy_y.len(), y.len());
    }
}
