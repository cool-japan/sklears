//! Economics and Finance Applications
//!
//! This module provides specialized isotonic regression applications for economics and finance,
//! including utility function estimation, demand curve modeling, and risk preference analysis.

use crate::core::LossFunction;
use scirs2_core::ndarray::Array1;
use sklears_core::error::SklearsError;
use sklears_core::traits::{Fit, Predict};
use sklears_core::types::Float;

/// Economics and Finance isotonic regression applications
#[derive(Debug, Clone)]
/// EconomicsFinanceIsotonicRegression
pub struct EconomicsFinanceIsotonicRegression {
    /// Application type
    application_type: EconomicsApplicationType,
    /// Loss function to use
    loss: LossFunction,
    /// Bounds for the output
    bounds: Option<(Float, Float)>,
}

/// Type of economics/finance application
#[derive(Debug, Clone, PartialEq)]
/// EconomicsApplicationType
pub enum EconomicsApplicationType {
    UtilityFunction,
    DemandCurve,
    ProductionFunction,
    RiskPreference,
    PortfolioOptimization,
}

impl Default for EconomicsFinanceIsotonicRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl EconomicsFinanceIsotonicRegression {
    /// Create a new economics/finance isotonic regression
    pub fn new() -> Self {
        Self {
            application_type: EconomicsApplicationType::UtilityFunction,
            loss: LossFunction::SquaredLoss,
            bounds: None,
        }
    }

    /// Set the application type
    pub fn application_type(mut self, app_type: EconomicsApplicationType) -> Self {
        self.application_type = app_type;
        self
    }

    /// Set the loss function
    pub fn loss(mut self, loss: LossFunction) -> Self {
        self.loss = loss;
        self
    }

    /// Set bounds for the output
    pub fn bounds(mut self, min: Option<Float>, max: Option<Float>) -> Self {
        match (min, max) {
            (Some(min_val), Some(max_val)) => self.bounds = Some((min_val, max_val)),
            _ => self.bounds = None,
        }
        self
    }

    /// Fit the model to data
    pub fn fit(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<EconomicsFinanceIsotonicRegressionFitted, SklearsError> {
        // Configure isotonic regression based on application type
        let increasing = match self.application_type {
            EconomicsApplicationType::UtilityFunction => true, // More income -> more utility
            EconomicsApplicationType::DemandCurve => false,    // Higher price -> lower demand
            EconomicsApplicationType::ProductionFunction => true, // More input -> more output
            EconomicsApplicationType::RiskPreference => true,  // More wealth -> more utility
            EconomicsApplicationType::PortfolioOptimization => true, // Configurable
        };

        // Use appropriate isotonic regression
        let mut isotonic = crate::core::IsotonicRegression::new()
            .increasing(increasing)
            .loss(self.loss.clone());

        // Apply bounds if specified
        if let Some((min_val, max_val)) = self.bounds {
            isotonic = isotonic.y_min(min_val).y_max(max_val);
        }

        let fitted_model = isotonic.fit(x, y)?;

        Ok(EconomicsFinanceIsotonicRegressionFitted {
            application_type: self.application_type.clone(),
            inner_model: fitted_model,
            bounds: self.bounds,
        })
    }
}

/// Fitted economics/finance isotonic regression model
#[derive(Debug)]
/// EconomicsFinanceIsotonicRegressionFitted
pub struct EconomicsFinanceIsotonicRegressionFitted {
    /// Application type
    application_type: EconomicsApplicationType,
    /// Inner fitted isotonic model
    inner_model: crate::core::IsotonicRegression<sklears_core::traits::Trained>,
    /// Output bounds
    bounds: Option<(Float, Float)>,
}

impl EconomicsFinanceIsotonicRegressionFitted {
    /// Predict utility, demand, or production for new data points
    pub fn predict(&self, x: &Array1<Float>) -> Result<Array1<Float>, SklearsError> {
        self.inner_model.predict(x)
    }

    /// Get the application-specific interpretation of predictions
    pub fn interpret_predictions(&self, x: &Array1<Float>) -> Result<String, SklearsError> {
        let predictions = self.predict(x)?;

        match self.application_type {
            EconomicsApplicationType::UtilityFunction => {
                Ok(format!(
                    "Utility Function: For income levels {:?}, predicted utilities are {:?}. \
                    Higher income generally leads to higher utility (diminishing marginal utility).",
                    x.to_vec(), predictions.to_vec()
                ))
            }
            EconomicsApplicationType::DemandCurve => {
                Ok(format!(
                    "Demand Curve: For prices {:?}, predicted demand quantities are {:?}. \
                    Higher prices generally lead to lower demand (law of demand).",
                    x.to_vec(), predictions.to_vec()
                ))
            }
            EconomicsApplicationType::ProductionFunction => {
                Ok(format!(
                    "Production Function: For input levels {:?}, predicted outputs are {:?}. \
                    More inputs generally lead to higher output (with potential diminishing returns).",
                    x.to_vec(), predictions.to_vec()
                ))
            }
            EconomicsApplicationType::RiskPreference => {
                Ok(format!(
                    "Risk Preference: For wealth levels {:?}, predicted utility values are {:?}. \
                    Shape indicates risk aversion (concave), risk neutrality (linear), or risk seeking (convex).",
                    x.to_vec(), predictions.to_vec()
                ))
            }
            EconomicsApplicationType::PortfolioOptimization => {
                Ok(format!(
                    "Portfolio Optimization: For risk levels {:?}, predicted returns are {:?}. \
                    Higher risk typically associated with higher expected returns.",
                    x.to_vec(), predictions.to_vec()
                ))
            }
        }
    }

    /// Analyze economic properties of the fitted function
    pub fn analyze_economic_properties(
        &self,
        x: &Array1<Float>,
    ) -> Result<EconomicAnalysis, SklearsError> {
        let predictions = self.predict(x)?;

        // Calculate marginal effects (discrete approximation)
        let mut marginal_effects = Vec::new();
        for i in 1..predictions.len() {
            let dx = x[i] - x[i - 1];
            let dy = predictions[i] - predictions[i - 1];
            marginal_effects.push(if dx != 0.0 { dy / dx } else { 0.0 });
        }

        // Check for diminishing marginal effects
        let mut diminishing_marginal = true;
        for i in 1..marginal_effects.len() {
            if marginal_effects[i] > marginal_effects[i - 1] {
                diminishing_marginal = false;
                break;
            }
        }

        Ok(EconomicAnalysis {
            application_type: self.application_type.clone(),
            marginal_effects,
            diminishing_marginal,
            elasticity: self.calculate_elasticity(x, &predictions)?,
        })
    }

    /// Calculate elasticity (percentage change in y / percentage change in x)
    fn calculate_elasticity(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<Vec<Float>, SklearsError> {
        let mut elasticities = Vec::new();

        for i in 1..x.len() {
            let percent_change_x = ((x[i] - x[i - 1]) / x[i - 1]) * 100.0;
            let percent_change_y = ((y[i] - y[i - 1]) / y[i - 1]) * 100.0;

            let elasticity = if percent_change_x != 0.0 {
                percent_change_y / percent_change_x
            } else {
                0.0
            };

            elasticities.push(elasticity);
        }

        Ok(elasticities)
    }
}

/// Economic analysis results
#[derive(Debug)]
/// EconomicAnalysis
pub struct EconomicAnalysis {
    /// Type of application
    pub application_type: EconomicsApplicationType,
    /// Marginal effects (first differences)
    pub marginal_effects: Vec<Float>,
    /// Whether marginal effects are diminishing
    pub diminishing_marginal: bool,
    /// Elasticity values
    pub elasticity: Vec<Float>,
}

/// Convenience function for utility function estimation
pub fn utility_function_estimation(
    income: &Array1<Float>,
    utility: &Array1<Float>,
) -> Result<EconomicsFinanceIsotonicRegressionFitted, SklearsError> {
    EconomicsFinanceIsotonicRegression::new()
        .application_type(EconomicsApplicationType::UtilityFunction)
        .bounds(Some(0.0), None) // Utility should be non-negative
        .fit(income, utility)
}

/// Convenience function for demand curve modeling
pub fn demand_curve_modeling(
    prices: &Array1<Float>,
    quantities: &Array1<Float>,
) -> Result<EconomicsFinanceIsotonicRegressionFitted, SklearsError> {
    EconomicsFinanceIsotonicRegression::new()
        .application_type(EconomicsApplicationType::DemandCurve)
        .bounds(Some(0.0), None) // Demand quantities should be non-negative
        .fit(prices, quantities)
}

/// Convenience function for production function estimation
pub fn production_function_estimation(
    inputs: &Array1<Float>,
    outputs: &Array1<Float>,
) -> Result<EconomicsFinanceIsotonicRegressionFitted, SklearsError> {
    EconomicsFinanceIsotonicRegression::new()
        .application_type(EconomicsApplicationType::ProductionFunction)
        .bounds(Some(0.0), None) // Production output should be non-negative
        .fit(inputs, outputs)
}

/// Convenience function for risk preference modeling
pub fn risk_preference_modeling(
    wealth: &Array1<Float>,
    utility: &Array1<Float>,
) -> Result<EconomicsFinanceIsotonicRegressionFitted, SklearsError> {
    EconomicsFinanceIsotonicRegression::new()
        .application_type(EconomicsApplicationType::RiskPreference)
        .bounds(Some(0.0), None) // Utility should be non-negative
        .fit(wealth, utility)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_utility_function_estimation() {
        // Typical utility function: logarithmic (diminishing marginal utility)
        let income = array![1000.0, 2000.0, 3000.0, 4000.0, 5000.0];
        let utility = array![10.0, 15.0, 18.0, 20.0, 21.5]; // Concave function

        let model = utility_function_estimation(&income, &utility).unwrap();
        let predictions = model.predict(&income).unwrap();

        // Check monotonicity (more income should give more utility)
        for i in 1..predictions.len() {
            assert!(predictions[i] >= predictions[i - 1]);
        }

        // Test interpretation
        let interpretation = model.interpret_predictions(&income).unwrap();
        assert!(interpretation.contains("Utility Function"));
        assert!(interpretation.contains("utility"));
    }

    #[test]
    fn test_demand_curve_modeling() {
        // Typical demand curve: decreasing
        let prices = array![10.0, 20.0, 30.0, 40.0, 50.0];
        let quantities = array![100.0, 80.0, 60.0, 40.0, 20.0]; // Decreasing

        let model = demand_curve_modeling(&prices, &quantities).unwrap();
        let predictions = model.predict(&prices).unwrap();

        // Check monotonicity (higher price should give lower demand)
        for i in 1..predictions.len() {
            assert!(predictions[i] <= predictions[i - 1]);
        }

        // Test interpretation
        let interpretation = model.interpret_predictions(&prices).unwrap();
        assert!(interpretation.contains("Demand Curve"));
        assert!(interpretation.contains("demand"));
    }

    #[test]
    fn test_production_function_estimation() {
        // Typical production function: increasing with diminishing returns
        let inputs = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let outputs = array![10.0, 18.0, 24.0, 28.0, 30.0]; // Concave

        let model = production_function_estimation(&inputs, &outputs).unwrap();
        let predictions = model.predict(&inputs).unwrap();

        // Check monotonicity (more input should give more output)
        for i in 1..predictions.len() {
            assert!(predictions[i] >= predictions[i - 1]);
        }

        // Test economic analysis
        let analysis = model.analyze_economic_properties(&inputs).unwrap();
        assert_eq!(
            analysis.application_type,
            EconomicsApplicationType::ProductionFunction
        );
        assert!(!analysis.marginal_effects.is_empty());
    }

    #[test]
    fn test_risk_preference_modeling() {
        // Risk-averse utility function
        let wealth = array![1000.0, 2000.0, 3000.0, 4000.0, 5000.0];
        let utility = array![31.6, 44.7, 54.8, 63.2, 70.7]; // sqrt function

        let model = risk_preference_modeling(&wealth, &utility).unwrap();
        let analysis = model.analyze_economic_properties(&wealth).unwrap();

        assert_eq!(
            analysis.application_type,
            EconomicsApplicationType::RiskPreference
        );
        assert!(analysis.diminishing_marginal); // Risk aversion implies diminishing marginal utility
    }

    #[test]
    fn test_economics_finance_isotonic_regression_basic() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0];

        let model = EconomicsFinanceIsotonicRegression::new()
            .application_type(EconomicsApplicationType::UtilityFunction)
            .loss(LossFunction::SquaredLoss)
            .bounds(Some(0.0), Some(10.0))
            .fit(&x, &y)
            .unwrap();

        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.len(), x.len());

        // Check bounds are respected
        for &pred in predictions.iter() {
            assert!(pred >= 0.0 && pred <= 10.0);
        }
    }
}
