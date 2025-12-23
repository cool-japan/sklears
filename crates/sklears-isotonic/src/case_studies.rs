//! Real-world case studies and practical examples for isotonic regression
//!
//! This module provides comprehensive examples demonstrating the application
//! of isotonic regression to various real-world problems across different domains.

use crate::fluent_api::FluentIsotonicRegression;
use scirs2_core::ndarray::Array1;
use sklears_core::{
    error::Result,
    traits::{Fit, Predict},
    types::Float,
};

// ============================================================================
// Medical Case Studies
// ============================================================================

/// Case Study: Drug Dose-Response Analysis
///
/// In pharmaceutical research, dose-response curves are critical for determining
/// the effective and safe dosage ranges for medications. Isotonic regression
/// ensures that the response is monotonic with dose, which is a biological requirement.
pub mod medical_dose_response {
    use super::*;

    /// Drug dose-response data
    pub struct DrugDoseResponseData {
        /// Doses (in mg)
        pub doses: Array1<Float>,
        /// Response rates (0-1)
        pub responses: Array1<Float>,
    }

    impl DrugDoseResponseData {
        /// Create sample data for a hypothetical drug
        pub fn sample_data() -> Self {
            // Sample doses from a clinical trial
            let doses = Array1::from_vec(vec![
                0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0,
            ]);

            // Observed response rates (with some noise)
            let responses = Array1::from_vec(vec![
                0.05, 0.12, 0.22, 0.35, 0.48, 0.58, 0.68, 0.75, 0.81, 0.85, 0.88,
            ]);

            Self { doses, responses }
        }

        /// Analyze dose-response relationship
        pub fn analyze(&self) -> Result<DoseResponseAnalysis> {
            // Fit isotonic regression with bounds (response must be in [0, 1])
            let model = FluentIsotonicRegression::new()
                .increasing()
                .probability_bounds()
                .huber_loss(1.0);

            let fitted = model.fit(&self.doses, &self.responses)?;
            let fitted_responses = fitted.predict(&self.doses)?;

            // Calculate effective dose (ED50 - dose at 50% response)
            let ed50 = Self::calculate_ed50(&self.doses, &fitted_responses);

            // Calculate therapeutic index (ratio of toxic to effective dose)
            let therapeutic_index =
                Self::calculate_therapeutic_index(&self.doses, &fitted_responses);

            Ok(DoseResponseAnalysis {
                fitted_responses,
                ed50,
                therapeutic_index,
            })
        }

        fn calculate_ed50(doses: &Array1<Float>, responses: &Array1<Float>) -> Option<Float> {
            for i in 0..responses.len() - 1 {
                if responses[i] <= 0.5 && responses[i + 1] >= 0.5 {
                    // Linear interpolation
                    let t = (0.5 - responses[i]) / (responses[i + 1] - responses[i]);
                    return Some(doses[i] + t * (doses[i + 1] - doses[i]));
                }
            }
            None
        }

        fn calculate_therapeutic_index(doses: &Array1<Float>, responses: &Array1<Float>) -> Float {
            // Therapeutic index = TD50 / ED50 (simplified)
            // TD50 would typically be from a separate toxicity study
            // Here we use ED75 / ED25 as a proxy for safety margin
            let ed25 = Self::find_dose_at_response(doses, responses, 0.25).unwrap_or(0.0);
            let ed75 = Self::find_dose_at_response(doses, responses, 0.75).unwrap_or(100.0);

            if ed25 > 0.0 {
                ed75 / ed25
            } else {
                0.0
            }
        }

        fn find_dose_at_response(
            doses: &Array1<Float>,
            responses: &Array1<Float>,
            target: Float,
        ) -> Option<Float> {
            for i in 0..responses.len() - 1 {
                if responses[i] <= target && responses[i + 1] >= target {
                    let t = (target - responses[i]) / (responses[i + 1] - responses[i]);
                    return Some(doses[i] + t * (doses[i + 1] - doses[i]));
                }
            }
            None
        }
    }

    pub struct DoseResponseAnalysis {
        pub fitted_responses: Array1<Float>,
        pub ed50: Option<Float>,
        pub therapeutic_index: Float,
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_drug_dose_response_analysis() {
            let data = DrugDoseResponseData::sample_data();
            let analysis = data.analyze().unwrap();

            // Check that responses are monotonic
            for i in 0..analysis.fitted_responses.len() - 1 {
                assert!(analysis.fitted_responses[i] <= analysis.fitted_responses[i + 1]);
            }

            // Check ED50 exists
            assert!(analysis.ed50.is_some());

            // Check therapeutic index is positive
            assert!(analysis.therapeutic_index > 0.0);
        }
    }
}

// ============================================================================
// Financial Case Studies
// ============================================================================

/// Case Study: Credit Scoring and Default Risk
///
/// Financial institutions use monotonic relationships between credit scores
/// and default probabilities to make lending decisions.
pub mod credit_scoring {
    use super::*;

    /// Credit score data
    pub struct CreditScoreData {
        /// Credit scores (300-850)
        pub scores: Array1<Float>,
        /// Default probabilities (0-1)
        pub default_probs: Array1<Float>,
    }

    impl CreditScoreData {
        /// Create sample credit scoring data
        pub fn sample_data() -> Self {
            // Sample credit scores
            let scores = Array1::from_vec(vec![
                350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 800.0,
            ]);

            // Observed default probabilities (decreasing with score)
            let default_probs = Array1::from_vec(vec![
                0.45, 0.38, 0.30, 0.22, 0.16, 0.11, 0.07, 0.04, 0.02, 0.01,
            ]);

            Self {
                scores,
                default_probs,
            }
        }

        /// Analyze credit risk
        pub fn analyze(&self) -> Result<CreditRiskAnalysis> {
            // Fit isotonic regression (decreasing - higher score = lower risk)
            let model = FluentIsotonicRegression::new()
                .decreasing()
                .probability_bounds()
                .robust(); // Robust to outliers

            let fitted = model.fit(&self.scores, &self.default_probs)?;
            let fitted_probs = fitted.predict(&self.scores)?;

            // Calculate risk tiers
            let risk_tiers = Self::calculate_risk_tiers(&fitted_probs);

            // Calculate acceptance threshold for target default rate
            let target_default_rate = 0.10;
            let acceptance_threshold =
                Self::find_acceptance_threshold(&self.scores, &fitted_probs, target_default_rate);

            Ok(CreditRiskAnalysis {
                fitted_probs,
                risk_tiers,
                acceptance_threshold,
            })
        }

        fn calculate_risk_tiers(probs: &Array1<Float>) -> Vec<RiskTier> {
            let mut tiers = Vec::new();
            let n = probs.len();

            for i in 0..n {
                let tier = if probs[i] > 0.30 {
                    RiskTier::High
                } else if probs[i] > 0.15 {
                    RiskTier::Medium
                } else if probs[i] > 0.05 {
                    RiskTier::Low
                } else {
                    RiskTier::VeryLow
                };
                tiers.push(tier);
            }

            tiers
        }

        fn find_acceptance_threshold(
            scores: &Array1<Float>,
            probs: &Array1<Float>,
            target_rate: Float,
        ) -> Option<Float> {
            for i in 0..probs.len() {
                if probs[i] <= target_rate {
                    return Some(scores[i]);
                }
            }
            None
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum RiskTier {
        VeryLow,
        Low,
        Medium,
        High,
    }

    pub struct CreditRiskAnalysis {
        pub fitted_probs: Array1<Float>,
        pub risk_tiers: Vec<RiskTier>,
        pub acceptance_threshold: Option<Float>,
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_credit_scoring_analysis() {
            let data = CreditScoreData::sample_data();
            let analysis = data.analyze().unwrap();

            // Check that probabilities are monotonically decreasing
            for i in 0..analysis.fitted_probs.len() - 1 {
                assert!(analysis.fitted_probs[i] >= analysis.fitted_probs[i + 1] - 1e-6);
            }

            // Check risk tiers are assigned
            assert_eq!(analysis.risk_tiers.len(), data.scores.len());
        }
    }
}

// ============================================================================
// Environmental Case Studies
// ============================================================================

/// Case Study: Pollution Level Monitoring
///
/// Environmental agencies monitor pollution levels that typically increase
/// monotonically with proximity to pollution sources or time.
pub mod pollution_monitoring {
    use super::*;

    /// Pollution monitoring data
    pub struct PollutionData {
        /// Distance from source (km)
        pub distances: Array1<Float>,
        /// Pollution concentration (ppm)
        pub concentrations: Array1<Float>,
    }

    impl PollutionData {
        /// Create sample pollution data
        pub fn sample_data() -> Self {
            // Distances from pollution source
            let distances = Array1::from_vec(vec![0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]);

            // Observed pollution concentrations (decreasing with distance)
            let concentrations =
                Array1::from_vec(vec![85.0, 65.0, 48.0, 38.0, 25.0, 18.0, 12.0, 7.0, 4.0]);

            Self {
                distances,
                concentrations,
            }
        }

        /// Analyze pollution dispersion
        pub fn analyze(&self) -> Result<PollutionAnalysis> {
            // Fit isotonic regression (decreasing with distance)
            let model = FluentIsotonicRegression::new()
                .decreasing()
                .non_negative()
                .robust(); // Robust to measurement errors

            let fitted = model.fit(&self.distances, &self.concentrations)?;
            let fitted_concentrations = fitted.predict(&self.distances)?;

            // Find safe distance (concentration below threshold)
            let safe_threshold = 10.0; // ppm
            let safe_distance =
                Self::find_safe_distance(&self.distances, &fitted_concentrations, safe_threshold);

            // Calculate dispersion coefficient
            let dispersion_coeff =
                Self::calculate_dispersion_coefficient(&self.distances, &fitted_concentrations);

            Ok(PollutionAnalysis {
                fitted_concentrations,
                safe_distance,
                dispersion_coefficient: dispersion_coeff,
            })
        }

        fn find_safe_distance(
            distances: &Array1<Float>,
            concentrations: &Array1<Float>,
            threshold: Float,
        ) -> Option<Float> {
            for i in 0..concentrations.len() {
                if concentrations[i] <= threshold {
                    return Some(distances[i]);
                }
            }
            None
        }

        fn calculate_dispersion_coefficient(
            distances: &Array1<Float>,
            concentrations: &Array1<Float>,
        ) -> Float {
            // Simplified dispersion coefficient (rate of concentration decrease)
            if concentrations.len() < 2 {
                return 0.0;
            }

            let initial_conc = concentrations[0];
            let final_conc = concentrations[concentrations.len() - 1];
            let distance_range = distances[distances.len() - 1] - distances[0];

            if distance_range > 0.0 {
                (initial_conc - final_conc) / distance_range
            } else {
                0.0
            }
        }
    }

    pub struct PollutionAnalysis {
        pub fitted_concentrations: Array1<Float>,
        pub safe_distance: Option<Float>,
        pub dispersion_coefficient: Float,
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_pollution_monitoring_analysis() {
            let data = PollutionData::sample_data();
            let analysis = data.analyze().unwrap();

            // Check that concentrations are monotonically decreasing
            for i in 0..analysis.fitted_concentrations.len() - 1 {
                assert!(
                    analysis.fitted_concentrations[i]
                        >= analysis.fitted_concentrations[i + 1] - 1e-6
                );
            }

            // Check non-negativity
            for &conc in analysis.fitted_concentrations.iter() {
                assert!(conc >= 0.0);
            }

            // Check dispersion coefficient is positive
            assert!(analysis.dispersion_coefficient > 0.0);
        }
    }
}

// ============================================================================
// Machine Learning Case Studies
// ============================================================================

/// Case Study: Classifier Calibration
///
/// Machine learning classifiers often produce uncalibrated probabilities.
/// Isotonic regression can be used to calibrate these probabilities to
/// match true class frequencies.
pub mod classifier_calibration {
    use super::*;

    /// Classifier calibration data
    pub struct CalibrationData {
        /// Predicted probabilities from classifier
        pub predicted_probs: Array1<Float>,
        /// True labels (0 or 1)
        pub true_labels: Array1<Float>,
    }

    impl CalibrationData {
        /// Create sample calibration data
        pub fn sample_data() -> Self {
            // Predicted probabilities (uncalibrated)
            let predicted_probs = Array1::from_vec(vec![
                0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95,
            ]);

            // Observed frequencies (true calibrated probabilities)
            let true_labels = Array1::from_vec(vec![
                0.02, 0.08, 0.18, 0.30, 0.42, 0.58, 0.68, 0.78, 0.88, 0.96,
            ]);

            Self {
                predicted_probs,
                true_labels,
            }
        }

        /// Calibrate classifier
        pub fn calibrate(&self) -> Result<CalibrationAnalysis> {
            // Fit isotonic regression for calibration
            let model = FluentIsotonicRegression::new()
                .increasing()
                .probability_bounds();

            let fitted = model.fit(&self.predicted_probs, &self.true_labels)?;
            let calibrated_probs = fitted.predict(&self.predicted_probs)?;

            // Calculate calibration metrics
            let calibration_error = Self::calculate_calibration_error(
                &self.predicted_probs,
                &self.true_labels,
                &calibrated_probs,
            );

            let brier_score_before =
                Self::calculate_brier_score(&self.predicted_probs, &self.true_labels);
            let brier_score_after =
                Self::calculate_brier_score(&calibrated_probs, &self.true_labels);

            Ok(CalibrationAnalysis {
                calibrated_probs,
                calibration_error,
                brier_score_before,
                brier_score_after,
                improvement: brier_score_before - brier_score_after,
            })
        }

        fn calculate_calibration_error(
            predicted: &Array1<Float>,
            true_labels: &Array1<Float>,
            calibrated: &Array1<Float>,
        ) -> Float {
            let error_before: Float = predicted
                .iter()
                .zip(true_labels.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum();

            let error_after: Float = calibrated
                .iter()
                .zip(true_labels.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum();

            (error_before - error_after) / predicted.len() as Float
        }

        fn calculate_brier_score(predicted: &Array1<Float>, true_labels: &Array1<Float>) -> Float {
            let n = predicted.len() as Float;
            predicted
                .iter()
                .zip(true_labels.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<Float>()
                / n
        }
    }

    pub struct CalibrationAnalysis {
        pub calibrated_probs: Array1<Float>,
        pub calibration_error: Float,
        pub brier_score_before: Float,
        pub brier_score_after: Float,
        pub improvement: Float,
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_classifier_calibration() {
            let data = CalibrationData::sample_data();
            let analysis = data.calibrate().unwrap();

            // Check that calibrated probabilities are monotonic
            for i in 0..analysis.calibrated_probs.len() - 1 {
                assert!(analysis.calibrated_probs[i] <= analysis.calibrated_probs[i + 1] + 1e-6);
            }

            // Check that calibrated probabilities are in [0, 1]
            for &prob in analysis.calibrated_probs.iter() {
                assert!(prob >= 0.0 && prob <= 1.0);
            }

            // Check that calibration improves (or at least doesn't hurt)
            assert!(analysis.brier_score_after <= analysis.brier_score_before + 1e-6);
        }
    }
}

// ============================================================================
// Economics Case Studies
// ============================================================================

/// Case Study: Demand Curve Estimation
///
/// In economics, demand curves show the relationship between price and quantity
/// demanded. They are typically monotonically decreasing.
pub mod demand_curve {
    use super::*;

    /// Demand curve data
    pub struct DemandData {
        /// Prices
        pub prices: Array1<Float>,
        /// Quantities demanded
        pub quantities: Array1<Float>,
    }

    impl DemandData {
        /// Create sample demand data
        pub fn sample_data() -> Self {
            // Sample prices
            let prices =
                Array1::from_vec(vec![10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]);

            // Observed quantities (decreasing with price)
            let quantities = Array1::from_vec(vec![
                950.0, 820.0, 710.0, 620.0, 540.0, 470.0, 410.0, 360.0, 320.0,
            ]);

            Self { prices, quantities }
        }

        /// Analyze demand curve
        pub fn analyze(&self) -> Result<DemandAnalysis> {
            // Fit isotonic regression (decreasing demand with price)
            let model = FluentIsotonicRegression::new()
                .decreasing()
                .non_negative()
                .huber_loss(10.0);

            let fitted = model.fit(&self.prices, &self.quantities)?;
            let fitted_quantities = fitted.predict(&self.prices)?;

            // Calculate price elasticity at midpoint
            let midpoint_idx = self.prices.len() / 2;
            let elasticity =
                Self::calculate_price_elasticity(&self.prices, &fitted_quantities, midpoint_idx);

            // Calculate consumer surplus
            let consumer_surplus =
                Self::calculate_consumer_surplus(&self.prices, &fitted_quantities);

            // Find revenue-maximizing price
            let optimal_price = Self::find_optimal_price(&self.prices, &fitted_quantities);

            Ok(DemandAnalysis {
                fitted_quantities,
                price_elasticity: elasticity,
                consumer_surplus,
                optimal_price,
            })
        }

        fn calculate_price_elasticity(
            prices: &Array1<Float>,
            quantities: &Array1<Float>,
            idx: usize,
        ) -> Float {
            if idx == 0 || idx >= prices.len() - 1 {
                return 0.0;
            }

            let delta_q = quantities[idx + 1] - quantities[idx - 1];
            let delta_p = prices[idx + 1] - prices[idx - 1];

            if delta_p.abs() < 1e-10 {
                return 0.0;
            }

            (delta_q / delta_p) * (prices[idx] / quantities[idx])
        }

        fn calculate_consumer_surplus(prices: &Array1<Float>, quantities: &Array1<Float>) -> Float {
            // Simplified consumer surplus (area under demand curve)
            let mut surplus = 0.0;
            for i in 0..prices.len() - 1 {
                let avg_quantity = (quantities[i] + quantities[i + 1]) / 2.0;
                let price_diff = prices[i + 1] - prices[i];
                surplus += avg_quantity * price_diff;
            }
            surplus
        }

        fn find_optimal_price(prices: &Array1<Float>, quantities: &Array1<Float>) -> Float {
            // Find price that maximizes revenue (price * quantity)
            let mut max_revenue = 0.0;
            let mut optimal_price = prices[0];

            for i in 0..prices.len() {
                let revenue = prices[i] * quantities[i];
                if revenue > max_revenue {
                    max_revenue = revenue;
                    optimal_price = prices[i];
                }
            }

            optimal_price
        }
    }

    pub struct DemandAnalysis {
        pub fitted_quantities: Array1<Float>,
        pub price_elasticity: Float,
        pub consumer_surplus: Float,
        pub optimal_price: Float,
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_demand_curve_analysis() {
            let data = DemandData::sample_data();
            let analysis = data.analyze().unwrap();

            // Check that quantities are monotonically decreasing with price
            for i in 0..analysis.fitted_quantities.len() - 1 {
                assert!(analysis.fitted_quantities[i] >= analysis.fitted_quantities[i + 1] - 1e-6);
            }

            // Check non-negativity
            for &qty in analysis.fitted_quantities.iter() {
                assert!(qty >= 0.0);
            }

            // Check that optimal price is in range
            let min_price = data.prices[0];
            let max_price = data.prices[data.prices.len() - 1];
            assert!(analysis.optimal_price >= min_price);
            assert!(analysis.optimal_price <= max_price);
        }
    }
}

// ============================================================================
// Summary Module
// ============================================================================

/// Case study summary and recommendations
pub mod summary {

    /// Get all available case studies
    pub fn available_case_studies() -> Vec<CaseStudyInfo> {
        vec![
            CaseStudyInfo {
                name: "Drug Dose-Response Analysis".to_string(),
                domain: Domain::Medical,
                description: "Analyze drug efficacy and safety using dose-response curves"
                    .to_string(),
                key_features: vec![
                    "Monotonic increasing response".to_string(),
                    "Probability bounds".to_string(),
                    "ED50 calculation".to_string(),
                ],
            },
            CaseStudyInfo {
                name: "Credit Scoring".to_string(),
                domain: Domain::Finance,
                description: "Estimate default risk and optimize lending decisions".to_string(),
                key_features: vec![
                    "Monotonic decreasing risk".to_string(),
                    "Risk tiering".to_string(),
                    "Robust to outliers".to_string(),
                ],
            },
            CaseStudyInfo {
                name: "Pollution Monitoring".to_string(),
                domain: Domain::Environmental,
                description: "Model pollution dispersion from sources".to_string(),
                key_features: vec![
                    "Monotonic decreasing concentration".to_string(),
                    "Non-negativity constraint".to_string(),
                    "Safe distance calculation".to_string(),
                ],
            },
            CaseStudyInfo {
                name: "Classifier Calibration".to_string(),
                domain: Domain::MachineLearning,
                description: "Calibrate ML classifier probabilities".to_string(),
                key_features: vec![
                    "Monotonic probability mapping".to_string(),
                    "Brier score improvement".to_string(),
                    "Probability bounds".to_string(),
                ],
            },
            CaseStudyInfo {
                name: "Demand Curve Estimation".to_string(),
                domain: Domain::Economics,
                description: "Estimate price-quantity relationships".to_string(),
                key_features: vec![
                    "Monotonic decreasing demand".to_string(),
                    "Price elasticity".to_string(),
                    "Revenue optimization".to_string(),
                ],
            },
        ]
    }

    #[derive(Debug, Clone)]
    pub struct CaseStudyInfo {
        pub name: String,
        pub domain: Domain,
        pub description: String,
        pub key_features: Vec<String>,
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum Domain {
        Medical,
        Finance,
        Environmental,
        MachineLearning,
        Economics,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        classifier_calibration, credit_scoring, demand_curve, medical_dose_response,
        pollution_monitoring, summary,
    };

    #[test]
    fn test_all_case_studies() {
        // Test medical case study
        let medical_data = medical_dose_response::DrugDoseResponseData::sample_data();
        assert!(medical_data.analyze().is_ok());

        // Test financial case study
        let credit_data = credit_scoring::CreditScoreData::sample_data();
        assert!(credit_data.analyze().is_ok());

        // Test environmental case study
        let pollution_data = pollution_monitoring::PollutionData::sample_data();
        assert!(pollution_data.analyze().is_ok());

        // Test ML case study
        let calibration_data = classifier_calibration::CalibrationData::sample_data();
        assert!(calibration_data.calibrate().is_ok());

        // Test economics case study
        let demand_data = demand_curve::DemandData::sample_data();
        assert!(demand_data.analyze().is_ok());
    }

    #[test]
    fn test_case_study_summary() {
        let studies = summary::available_case_studies();
        assert_eq!(studies.len(), 5);

        // Check that all domains are represented
        let domains: Vec<_> = studies.iter().map(|s| s.domain).collect();
        assert!(domains.contains(&summary::Domain::Medical));
        assert!(domains.contains(&summary::Domain::Finance));
        assert!(domains.contains(&summary::Domain::Environmental));
        assert!(domains.contains(&summary::Domain::MachineLearning));
        assert!(domains.contains(&summary::Domain::Economics));
    }
}
