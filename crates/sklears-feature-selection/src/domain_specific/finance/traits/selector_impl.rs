//! # FinanceFeatureSelector - Trait Implementations
//!
//! This module contains trait implementations for `FinanceFeatureSelector`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Estimator`
//! - `Fit`
//! - `Transform`
//! - `SelectorMixin`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::base::SelectorMixin;
use crate::domain_specific::finance::algorithms::*;
use crate::domain_specific::finance::types::{FinanceFeatureSelector, Trained, Untrained};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::error::{Result as SklResult, SklearsError};
use sklears_core::traits::{Estimator, Fit, Transform};
use sklears_core::types::Float;
use std::marker::PhantomData;

type Result<T> = SklResult<T>;

impl Default for FinanceFeatureSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for FinanceFeatureSelector<Untrained> {
    type Config = ();
    type Error = sklears_core::error::SklearsError;
    type Float = Float;
    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Estimator for FinanceFeatureSelector<Trained> {
    type Config = ();
    type Error = sklears_core::error::SklearsError;
    type Float = Float;
    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for FinanceFeatureSelector<Untrained> {
    type Fitted = FinanceFeatureSelector<Trained>;
    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();
        if y.len() != n_samples {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }
        let (
            feature_scores,
            technical_scores,
            risk_metrics,
            regime_probs,
            correlation_matrix,
            sharpe_ratios,
            info_ratios,
        ) = match self.feature_type.as_str() {
            "technical_indicators" => self.analyze_technical_indicators(x, y)?,
            "risk_metrics" => self.analyze_risk_metrics(x, y)?,
            "microstructure" => self.analyze_microstructure(x, y)?,
            "cross_asset" => self.analyze_cross_asset(x, y)?,
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown feature type: {}",
                    self.feature_type
                )));
            }
        };
        let selected_features = if self.k > 0 {
            select_top_k_features(&feature_scores, self.k)
        } else {
            select_features_by_threshold(&feature_scores, self.score_threshold)
        };
        let trained_state = Trained {
            selected_features,
            feature_scores,
            technical_indicator_scores: technical_scores,
            risk_metrics,
            regime_probabilities: regime_probs,
            correlation_matrix,
            sharpe_ratios,
            information_ratios: info_ratios,
            n_features,
            feature_type: self.feature_type.clone(),
        };
        Ok(FinanceFeatureSelector {
            feature_type: self.feature_type,
            include_momentum_indicators: self.include_momentum_indicators,
            include_volatility_indicators: self.include_volatility_indicators,
            include_volume_indicators: self.include_volume_indicators,
            include_var_metrics: self.include_var_metrics,
            include_drawdown_metrics: self.include_drawdown_metrics,
            include_order_flow: self.include_order_flow,
            include_bid_ask_analysis: self.include_bid_ask_analysis,
            include_market_impact: self.include_market_impact,
            include_regime_detection: self.include_regime_detection,
            lookback_periods: self.lookback_periods,
            var_confidence_level: self.var_confidence_level,
            drawdown_threshold: self.drawdown_threshold,
            correlation_threshold: self.correlation_threshold,
            tick_frequency: self.tick_frequency,
            liquidity_threshold: self.liquidity_threshold,
            regime_detection_method: self.regime_detection_method,
            cross_asset_correlation: self.cross_asset_correlation,
            spillover_analysis: self.spillover_analysis,
            economic_indicators_weight: self.economic_indicators_weight,
            risk_adjusted_scoring: self.risk_adjusted_scoring,
            transaction_cost_weight: self.transaction_cost_weight,
            market_neutrality_weight: self.market_neutrality_weight,
            k: self.k,
            score_threshold: self.score_threshold,
            strategy: self.strategy,
            state: PhantomData,
            trained_state: Some(trained_state),
        })
    }
}

impl Transform<Array2<Float>> for FinanceFeatureSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let trained = self.trained_state.as_ref().ok_or_else(|| {
            SklearsError::InvalidState("Selector must be fitted before transforming".to_string())
        })?;
        let (_n_samples, n_features) = x.dim();
        if n_features != trained.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                trained.n_features, n_features
            )));
        }
        if trained.selected_features.is_empty() {
            return Err(SklearsError::InvalidState(
                "No features were selected".to_string(),
            ));
        }
        let selected_data = x.select(Axis(1), &trained.selected_features);
        Ok(selected_data)
    }
}

impl SelectorMixin for FinanceFeatureSelector<Trained> {
    fn get_support(&self) -> Result<Array1<bool>> {
        let trained = self.trained_state.as_ref().ok_or_else(|| {
            SklearsError::InvalidState("Selector must be fitted before getting support".to_string())
        })?;
        let mut support = Array1::from_elem(trained.n_features, false);
        for &idx in &trained.selected_features {
            support[idx] = true;
        }
        Ok(support)
    }
    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let trained = self.trained_state.as_ref().ok_or_else(|| {
            SklearsError::InvalidState(
                "Selector must be fitted before transforming features".to_string(),
            )
        })?;
        let selected: Vec<usize> = indices
            .iter()
            .filter(|&&idx| trained.selected_features.contains(&idx))
            .cloned()
            .collect();
        Ok(selected)
    }
}
