//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result as SklResult, SklearsError};
use sklears_core::types::Float;
use std::collections::HashMap;
use std::marker::PhantomData;

// Import algorithm functions
use super::algorithms::*;

type Result<T> = SklResult<T>;

/// Strategy for finance feature selection
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinanceStrategy {
    /// Momentum
    Momentum,
    /// Volatility
    Volatility,
    /// Volume
    Volume,
    /// RiskAdjusted
    RiskAdjusted,
    /// TechnicalIndicators
    TechnicalIndicators,
    /// MarketMicrostructure
    MarketMicrostructure,
    /// Fama-French factor selection
    FamaFrench,
    /// Risk-based selection with VaR and CVaR
    RiskBased,
    /// Regime-aware selection with market state detection
    RegimeAware,
    /// Macroeconomic factor selection
    MacroeconomicFactors,
    /// Portfolio optimization integration
    PortfolioOptimization,
}
#[derive(Debug, Clone)]
pub struct Trained {
    pub(crate) selected_features: Vec<usize>,
    pub(crate) feature_scores: Array1<Float>,
    pub(crate) technical_indicator_scores: Option<HashMap<String, Array1<Float>>>,
    pub(crate) risk_metrics: Option<HashMap<String, Float>>,
    pub(crate) regime_probabilities: Option<Array2<Float>>,
    pub(crate) correlation_matrix: Option<Array2<Float>>,
    pub(crate) sharpe_ratios: Option<Array1<Float>>,
    pub(crate) information_ratios: Option<Array1<Float>>,
    pub(crate) n_features: usize,
    pub(crate) feature_type: String,
}
#[derive(Debug, Clone)]
pub struct Untrained;
/// Finance-specific feature selector for trading and market data.
///
/// This selector provides specialized methods for financial data analysis, including
/// technical indicator evaluation, risk metric calculation, market microstructure analysis,
/// and cross-asset correlation studies. It incorporates financial domain knowledge and
/// risk-adjusted performance metrics for feature selection.
#[derive(Debug, Clone)]
pub struct FinanceFeatureSelector<State = Untrained> {
    pub(crate) feature_type: String,
    pub(crate) include_momentum_indicators: bool,
    pub(crate) include_volatility_indicators: bool,
    pub(crate) include_volume_indicators: bool,
    pub(crate) include_var_metrics: bool,
    pub(crate) include_drawdown_metrics: bool,
    pub(crate) include_order_flow: bool,
    pub(crate) include_bid_ask_analysis: bool,
    pub(crate) include_market_impact: bool,
    pub(crate) include_regime_detection: bool,
    pub(crate) lookback_periods: Vec<usize>,
    pub(crate) var_confidence_level: Float,
    pub(crate) drawdown_threshold: Float,
    pub(crate) correlation_threshold: Float,
    pub(crate) tick_frequency: String,
    pub(crate) liquidity_threshold: Float,
    pub(crate) regime_detection_method: String,
    pub(crate) cross_asset_correlation: bool,
    pub(crate) spillover_analysis: bool,
    pub(crate) economic_indicators_weight: Float,
    pub(crate) risk_adjusted_scoring: bool,
    pub(crate) transaction_cost_weight: Float,
    pub(crate) market_neutrality_weight: Float,
    pub k: usize,
    pub(crate) score_threshold: Float,
    pub(crate) strategy: FinanceStrategy,
    pub(crate) state: PhantomData<State>,
    pub(crate) trained_state: Option<Trained>,
}
impl FinanceFeatureSelector<Untrained> {
    /// Creates a new FinanceFeatureSelector with default parameters.
    pub fn new() -> Self {
        Self {
            feature_type: "technical_indicators".to_string(),
            include_momentum_indicators: true,
            include_volatility_indicators: true,
            include_volume_indicators: true,
            include_var_metrics: false,
            include_drawdown_metrics: false,
            include_order_flow: false,
            include_bid_ask_analysis: false,
            include_market_impact: false,
            include_regime_detection: false,
            lookback_periods: vec![5, 10, 20],
            var_confidence_level: 0.95,
            drawdown_threshold: 0.05,
            correlation_threshold: 0.7,
            tick_frequency: "1day".to_string(),
            liquidity_threshold: 0.1,
            regime_detection_method: "markov_switching".to_string(),
            cross_asset_correlation: false,
            spillover_analysis: false,
            economic_indicators_weight: 0.2,
            risk_adjusted_scoring: true,
            transaction_cost_weight: 0.1,
            market_neutrality_weight: 0.1,
            k: 10,
            score_threshold: 0.1,
            strategy: FinanceStrategy::Momentum,
            state: PhantomData,
            trained_state: None,
        }
    }
    /// Set the number of features to select
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }
    /// Set the finance strategy
    pub fn strategy(mut self, strategy: FinanceStrategy) -> Self {
        self.strategy = strategy;
        self
    }
    /// Set lookback window (just adds to lookback_periods)
    pub fn lookback_window(mut self, window: usize) -> Self {
        if !self.lookback_periods.contains(&window) {
            self.lookback_periods.push(window);
        }
        self
    }
    /// Creates a builder for configuring the FinanceFeatureSelector.
    pub fn builder() -> FinanceFeatureSelectorBuilder {
        FinanceFeatureSelectorBuilder::new()
    }
}
impl FinanceFeatureSelector<Trained> {
    /// Get the selected feature indices
    pub fn selected_features(&self) -> Result<&Vec<usize>> {
        let trained = self.trained_state.as_ref().ok_or_else(|| {
            SklearsError::InvalidState(
                "Selector must be fitted before getting selected features".to_string(),
            )
        })?;
        Ok(&trained.selected_features)
    }
}
impl FinanceFeatureSelector<Untrained> {
    pub(crate) fn analyze_technical_indicators(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Array1<Float>,
        Option<HashMap<String, Array1<Float>>>,
        Option<HashMap<String, Float>>,
        Option<Array2<Float>>,
        Option<Array2<Float>>,
        Option<Array1<Float>>,
        Option<Array1<Float>>,
    )> {
        let (_, n_features) = x.dim();
        let mut feature_scores = Array1::zeros(n_features);
        let mut technical_scores = HashMap::new();
        for j in 0..n_features {
            let feature = x.column(j);
            let mut indicator_score = 0.0;
            let mut score_count = 0;
            if self.include_momentum_indicators {
                for &period in &self.lookback_periods {
                    let rsi_score = compute_rsi_predictive_power(&feature, y, period)?;
                    let macd_score = compute_macd_predictive_power(&feature, y, period)?;
                    indicator_score += rsi_score + macd_score;
                    score_count += 2;
                }
            }
            if self.include_volatility_indicators {
                for &period in &self.lookback_periods {
                    let bb_score = compute_bollinger_bands_predictive_power(&feature, y, period)?;
                    let atr_score = compute_atr_predictive_power(&feature, y, period)?;
                    indicator_score += bb_score + atr_score;
                    score_count += 2;
                }
            }
            if self.include_volume_indicators && j >= n_features.saturating_sub(5) {
                for &period in &self.lookback_periods {
                    let obv_score = compute_obv_predictive_power(&feature, y, period)?;
                    let vwap_score = compute_vwap_predictive_power(&feature, y, period)?;
                    indicator_score += obv_score + vwap_score;
                    score_count += 2;
                }
            }
            feature_scores[j] = if score_count > 0 {
                indicator_score / score_count as Float
            } else {
                0.0
            };
        }
        technical_scores.insert("momentum".to_string(), feature_scores.clone());
        technical_scores.insert("volatility".to_string(), feature_scores.clone());
        technical_scores.insert("volume".to_string(), feature_scores.clone());
        let (sharpe_ratios, info_ratios) = if self.risk_adjusted_scoring {
            let sharpe = compute_feature_sharpe_ratios(x, y)?;
            let info = compute_information_ratios(x, y)?;
            for j in 0..n_features {
                feature_scores[j] *= (1.0 + sharpe[j]).max(0.1);
            }
            (Some(sharpe), Some(info))
        } else {
            (None, None)
        };
        Ok((
            feature_scores,
            Some(technical_scores),
            None,
            None,
            None,
            sharpe_ratios,
            info_ratios,
        ))
    }
    pub(crate) fn analyze_risk_metrics(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Array1<Float>,
        Option<HashMap<String, Array1<Float>>>,
        Option<HashMap<String, Float>>,
        Option<Array2<Float>>,
        Option<Array2<Float>>,
        Option<Array1<Float>>,
        Option<Array1<Float>>,
    )> {
        let (_, n_features) = x.dim();
        let mut feature_scores = Array1::zeros(n_features);
        let mut risk_metrics = HashMap::new();
        if self.include_var_metrics {
            let var_scores = compute_var_based_scores(x, y, self.var_confidence_level)?;
            feature_scores = &feature_scores + &var_scores;
            risk_metrics.insert(
                "var_95".to_string(),
                compute_portfolio_var(x, self.var_confidence_level)?,
            );
        }
        if self.include_drawdown_metrics {
            let drawdown_scores = compute_drawdown_based_scores(x, y)?;
            feature_scores = &feature_scores + &drawdown_scores;
            risk_metrics.insert("max_drawdown".to_string(), compute_max_drawdown(x)?);
        }
        let correlation_matrix = compute_correlation_matrix(x)?;
        let correlation_scores = compute_correlation_based_scores(x, y)?;
        feature_scores = &feature_scores + &correlation_scores;
        let sharpe_ratios = compute_feature_sharpe_ratios(x, y)?;
        let info_ratios = compute_information_ratios(x, y)?;
        Ok((
            feature_scores,
            None,
            Some(risk_metrics),
            None,
            Some(correlation_matrix),
            Some(sharpe_ratios),
            Some(info_ratios),
        ))
    }
    pub(crate) fn analyze_microstructure(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Array1<Float>,
        Option<HashMap<String, Array1<Float>>>,
        Option<HashMap<String, Float>>,
        Option<Array2<Float>>,
        Option<Array2<Float>>,
        Option<Array1<Float>>,
        Option<Array1<Float>>,
    )> {
        let (_, n_features) = x.dim();
        let mut feature_scores = Array1::zeros(n_features);
        let mut microstructure_scores = HashMap::new();
        if self.include_order_flow {
            let order_flow_scores = compute_order_flow_scores(x, y)?;
            feature_scores = &feature_scores + &order_flow_scores;
            microstructure_scores.insert("order_flow".to_string(), order_flow_scores);
        }
        if self.include_bid_ask_analysis {
            let bid_ask_scores = compute_bid_ask_scores(x, y, self.liquidity_threshold)?;
            feature_scores = &feature_scores + &bid_ask_scores;
            microstructure_scores.insert("bid_ask".to_string(), bid_ask_scores);
        }
        if self.include_market_impact {
            let impact_scores = compute_market_impact_scores(x, y)?;
            feature_scores = &feature_scores + &impact_scores;
            microstructure_scores.insert("market_impact".to_string(), impact_scores);
        }
        if self.transaction_cost_weight > 0.0 {
            let tc_scores = compute_transaction_cost_scores(x, y)?;
            for j in 0..n_features {
                feature_scores[j] *= (1.0 - self.transaction_cost_weight * tc_scores[j]).max(0.1);
            }
        }
        Ok((
            feature_scores,
            Some(microstructure_scores),
            None,
            None,
            None,
            None,
            None,
        ))
    }
    pub(crate) fn analyze_cross_asset(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Array1<Float>,
        Option<HashMap<String, Array1<Float>>>,
        Option<HashMap<String, Float>>,
        Option<Array2<Float>>,
        Option<Array2<Float>>,
        Option<Array1<Float>>,
        Option<Array1<Float>>,
    )> {
        let (_, n_features) = x.dim();
        let mut feature_scores = Array1::zeros(n_features);
        let correlation_matrix = if self.cross_asset_correlation {
            Some(compute_correlation_matrix(x)?)
        } else {
            None
        };
        if let Some(ref corr_matrix) = correlation_matrix {
            let cross_asset_scores = compute_cross_asset_scores(corr_matrix, y)?;
            feature_scores = &feature_scores + &cross_asset_scores;
        }
        if self.spillover_analysis {
            let spillover_scores = compute_spillover_scores(x, y)?;
            feature_scores = &feature_scores + &spillover_scores;
        }
        let regime_probabilities = if self.include_regime_detection {
            Some(detect_market_regimes(x, &self.regime_detection_method)?)
        } else {
            None
        };
        if let Some(ref _regimes) = regime_probabilities {
            let regime_scores = compute_regime_based_scores(x, y, &self.regime_detection_method)?;
            feature_scores = &feature_scores + &regime_scores;
        }
        if self.economic_indicators_weight > 0.0 {
            let econ_scores = compute_economic_indicator_scores(x, y)?;
            for j in 0..n_features {
                feature_scores[j] = feature_scores[j] * (1.0 - self.economic_indicators_weight)
                    + econ_scores[j] * self.economic_indicators_weight;
            }
        }
        Ok((
            feature_scores,
            None,
            None,
            regime_probabilities,
            correlation_matrix,
            None,
            None,
        ))
    }
    /// Fama-French Factor Selection
    ///
    /// Implements factor-based selection using Fama-French three-factor and five-factor models
    /// (Market, SMB, HML, RMW, CMA) to identify features with systematic risk premia.
    pub(crate) fn analyze_fama_french(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Array1<Float>,
        Option<HashMap<String, Array1<Float>>>,
        Option<HashMap<String, Float>>,
        Option<Array2<Float>>,
        Option<Array2<Float>>,
        Option<Array1<Float>>,
        Option<Array1<Float>>,
    )> {
        let (_, n_features) = x.dim();
        let mut feature_scores = Array1::zeros(n_features);
        let mut factor_scores = HashMap::new();
        let market_factor = compute_market_factor(x, y)?;
        let smb_factor = compute_smb_factor(x)?;
        let hml_factor = compute_hml_factor(x)?;
        let rmw_factor = compute_rmw_factor(x)?;
        let cma_factor = compute_cma_factor(x)?;
        for j in 0..n_features {
            let feature = x.column(j);
            let market_beta = compute_factor_loading(&feature, &market_factor);
            let smb_beta = compute_factor_loading(&feature, &smb_factor);
            let hml_beta = compute_factor_loading(&feature, &hml_factor);
            let rmw_beta = compute_factor_loading(&feature, &rmw_factor);
            let cma_beta = compute_factor_loading(&feature, &cma_factor);
            let predicted_return = market_beta * market_factor.mean().unwrap_or(0.0)
                + smb_beta * smb_factor.mean().unwrap_or(0.0)
                + hml_beta * hml_factor.mean().unwrap_or(0.0)
                + rmw_beta * rmw_factor.mean().unwrap_or(0.0)
                + cma_beta * cma_factor.mean().unwrap_or(0.0);
            let actual_return = feature.mean().unwrap_or(0.0);
            let alpha = actual_return - predicted_return;
            let factors = vec![
                market_factor.clone(),
                smb_factor.clone(),
                hml_factor.clone(),
                rmw_factor.clone(),
                cma_factor.clone(),
            ];
            let r_squared = compute_factor_model_r_squared(&feature, &factors);
            feature_scores[j] = alpha.abs() * r_squared;
        }
        factor_scores.insert("market_beta".to_string(), feature_scores.clone());
        factor_scores.insert("size_loading".to_string(), feature_scores.clone());
        factor_scores.insert("value_loading".to_string(), feature_scores.clone());
        Ok((
            feature_scores,
            Some(factor_scores),
            None,
            None,
            None,
            None,
            None,
        ))
    }
    /// Risk-Based Feature Selection
    ///
    /// Advanced risk metrics including Value-at-Risk (VaR), Conditional VaR (CVaR),
    /// tail risk measures, and downside deviation for comprehensive risk assessment.
    pub(crate) fn analyze_risk_based(
        &self,
        x: &Array2<Float>,
        _y: &Array1<Float>,
    ) -> Result<(
        Array1<Float>,
        Option<HashMap<String, Array1<Float>>>,
        Option<HashMap<String, Float>>,
        Option<Array2<Float>>,
        Option<Array2<Float>>,
        Option<Array1<Float>>,
        Option<Array1<Float>>,
    )> {
        let (_, n_features) = x.dim();
        let mut feature_scores = Array1::zeros(n_features);
        let mut risk_metrics = HashMap::new();
        for j in 0..n_features {
            let feature = x.column(j);
            let _var_95 = compute_var(&feature, 0.95);
            let _var_99 = compute_var(&feature, 0.99);
            let cvar_95 = compute_cvar(&feature, 0.95);
            let _cvar_99 = compute_cvar(&feature, 0.99);
            let tail_risk = compute_tail_risk(&feature);
            let _extreme_value_index = compute_extreme_value_index(&feature);
            let downside_deviation = compute_downside_deviation(&feature, 0.0);
            let _max_drawdown = compute_feature_max_drawdown(&feature);
            let mean_return = feature.mean().unwrap_or(0.0);
            let sortino_ratio = if downside_deviation > 1e-10 {
                mean_return / downside_deviation
            } else {
                0.0
            };
            let omega_ratio = compute_omega_ratio(&feature, 0.0);
            let risk_score = sortino_ratio.abs() * omega_ratio.abs()
                / ((1.0 + cvar_95.abs()) * (1.0 + tail_risk));
            feature_scores[j] = risk_score;
        }
        risk_metrics.insert(
            "portfolio_var_95".to_string(),
            compute_portfolio_var(x, 0.95)?,
        );
        risk_metrics.insert(
            "portfolio_var_99".to_string(),
            compute_portfolio_var(x, 0.99)?,
        );
        risk_metrics.insert(
            "portfolio_cvar_95".to_string(),
            compute_portfolio_cvar(x, 0.95)?,
        );
        risk_metrics.insert(
            "portfolio_max_drawdown".to_string(),
            compute_max_drawdown(x)?,
        );
        Ok((
            feature_scores,
            None,
            Some(risk_metrics),
            None,
            None,
            None,
            None,
        ))
    }
    /// Regime-Aware Feature Selection
    ///
    /// Identifies market regimes (bull, bear, high volatility, low volatility) using
    /// Hidden Markov Models and Markov Switching models, then selects features that
    /// perform well across different market states.
    pub(crate) fn analyze_regime_aware(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Array1<Float>,
        Option<HashMap<String, Array1<Float>>>,
        Option<HashMap<String, Float>>,
        Option<Array2<Float>>,
        Option<Array2<Float>>,
        Option<Array1<Float>>,
        Option<Array1<Float>>,
    )> {
        let (n_samples, n_features) = x.dim();
        let mut feature_scores = Array1::zeros(n_features);
        let regime_probs = detect_market_regimes_hmm(x, 3)?;
        for j in 0..n_features {
            let _feature = x.column(j);
            let mut regime_performances = Vec::new();
            for regime in 0..regime_probs.ncols() {
                let regime_prob = regime_probs.column(regime);
                let mut weighted_return = 0.0;
                let mut total_weight = 0.0;
                for i in 0..n_samples {
                    if i > 0 && regime_prob[i] > 0.3 {
                        let return_val = if i < y.len() { y[i] } else { 0.0 };
                        weighted_return += return_val * regime_prob[i];
                        total_weight += regime_prob[i];
                    }
                }
                let regime_perf = if total_weight > 0.0 {
                    weighted_return / total_weight
                } else {
                    0.0
                };
                regime_performances.push(regime_perf);
            }
            let mean_perf =
                regime_performances.iter().sum::<Float>() / regime_performances.len() as Float;
            let variance = regime_performances
                .iter()
                .map(|&p| (p - mean_perf).powi(2))
                .sum::<Float>()
                / regime_performances.len() as Float;
            feature_scores[j] = mean_perf.abs() / (1.0 + variance.sqrt());
        }
        let transition_matrix = compute_regime_transitions(&regime_probs)?;
        let mut regime_specific_scores = HashMap::new();
        for regime in 0..regime_probs.ncols() {
            let regime_name = format!("regime_{}", regime);
            regime_specific_scores.insert(regime_name, feature_scores.clone());
        }
        Ok((
            feature_scores,
            Some(regime_specific_scores),
            None,
            Some(regime_probs),
            Some(transition_matrix),
            None,
            None,
        ))
    }
    /// Macroeconomic Factor Selection
    ///
    /// Selects features based on their relationship with macroeconomic indicators
    /// (GDP growth, inflation, interest rates, unemployment) to identify systematic factors.
    pub(crate) fn analyze_macroeconomic_factors(
        &self,
        x: &Array2<Float>,
        _y: &Array1<Float>,
    ) -> Result<(
        Array1<Float>,
        Option<HashMap<String, Array1<Float>>>,
        Option<HashMap<String, Float>>,
        Option<Array2<Float>>,
        Option<Array2<Float>>,
        Option<Array1<Float>>,
        Option<Array1<Float>>,
    )> {
        let (_, n_features) = x.dim();
        let mut feature_scores = Array1::zeros(n_features);
        let mut macro_scores = HashMap::new();
        let gdp_growth = simulate_gdp_growth(x)?;
        let inflation = simulate_inflation(x)?;
        let interest_rate = simulate_interest_rate(x)?;
        let unemployment = simulate_unemployment(x)?;
        for j in 0..n_features {
            let feature = x.column(j);
            let gdp_beta = compute_factor_loading(&feature, &gdp_growth);
            let inflation_beta = compute_factor_loading(&feature, &inflation);
            let rate_beta = compute_factor_loading(&feature, &interest_rate);
            let unemployment_beta = compute_factor_loading(&feature, &unemployment);
            let macro_factors = vec![
                gdp_growth.clone(),
                inflation.clone(),
                interest_rate.clone(),
                unemployment.clone(),
            ];
            let macro_r_squared = compute_macro_model_r_squared(&feature, &macro_factors);
            feature_scores[j] = macro_r_squared
                * (gdp_beta.abs()
                    + inflation_beta.abs()
                    + rate_beta.abs()
                    + unemployment_beta.abs())
                / 4.0;
        }
        macro_scores.insert("gdp_sensitivity".to_string(), feature_scores.clone());
        macro_scores.insert("inflation_sensitivity".to_string(), feature_scores.clone());
        macro_scores.insert("rate_sensitivity".to_string(), feature_scores.clone());
        Ok((
            feature_scores,
            Some(macro_scores),
            None,
            None,
            None,
            None,
            None,
        ))
    }
    /// Portfolio Optimization Integration
    ///
    /// Selects features based on mean-variance optimization, minimum variance portfolios,
    /// maximum Sharpe ratio, and risk parity principles for portfolio construction.
    pub(crate) fn analyze_portfolio_optimization(
        &self,
        x: &Array2<Float>,
        _y: &Array1<Float>,
    ) -> Result<(
        Array1<Float>,
        Option<HashMap<String, Array1<Float>>>,
        Option<HashMap<String, Float>>,
        Option<Array2<Float>>,
        Option<Array2<Float>>,
        Option<Array1<Float>>,
        Option<Array1<Float>>,
    )> {
        let (_, n_features) = x.dim();
        let mut feature_scores = Array1::zeros(n_features);
        let mut portfolio_metrics = HashMap::new();
        let cov_matrix = compute_covariance_matrix(x)?;
        let expected_returns = compute_expected_returns(x)?;
        let mv_weights = mean_variance_optimization(&expected_returns, &cov_matrix)?;
        let min_var_weights = minimum_variance_portfolio(&cov_matrix)?;
        let max_sharpe_weights = maximum_sharpe_ratio_portfolio(
            &expected_returns,
            &cov_matrix,
            0.02, // risk_free_rate
        )?;
        let risk_parity_weights = risk_parity_portfolio(&cov_matrix)?;
        for j in 0..n_features {
            let mv_score = mv_weights[j].abs();
            let min_var_score = min_var_weights[j].abs();
            let sharpe_score = max_sharpe_weights[j].abs();
            let parity_score = risk_parity_weights[j].abs();
            feature_scores[j] = (mv_score + min_var_score + sharpe_score + parity_score) / 4.0;
        }
        let portfolio_return = compute_portfolio_return(&mv_weights, &expected_returns);
        let portfolio_variance = compute_portfolio_variance(&mv_weights, &cov_matrix);
        let portfolio_sharpe = portfolio_return / portfolio_variance.sqrt();
        portfolio_metrics.insert("portfolio_return".to_string(), portfolio_return);
        portfolio_metrics.insert("portfolio_variance".to_string(), portfolio_variance);
        portfolio_metrics.insert("portfolio_sharpe".to_string(), portfolio_sharpe);
        portfolio_metrics.insert(
            "diversification_ratio".to_string(),
            compute_diversification_ratio(&mv_weights, &cov_matrix),
        );
        Ok((
            feature_scores,
            None,
            Some(portfolio_metrics),
            None,
            Some(cov_matrix),
            None,
            None,
        ))
    }
}
/// Builder for FinanceFeatureSelector configuration.
#[derive(Debug)]
pub struct FinanceFeatureSelectorBuilder {
    pub(crate) feature_type: String,
    pub(crate) include_momentum_indicators: bool,
    pub(crate) include_volatility_indicators: bool,
    pub(crate) include_volume_indicators: bool,
    pub(crate) include_var_metrics: bool,
    pub(crate) include_drawdown_metrics: bool,
    pub(crate) include_order_flow: bool,
    pub(crate) include_bid_ask_analysis: bool,
    pub(crate) include_market_impact: bool,
    pub(crate) include_regime_detection: bool,
    pub(crate) lookback_periods: Vec<usize>,
    pub(crate) var_confidence_level: Float,
    pub(crate) drawdown_threshold: Float,
    pub(crate) correlation_threshold: Float,
    pub(crate) tick_frequency: String,
    pub(crate) liquidity_threshold: Float,
    pub(crate) regime_detection_method: String,
    pub(crate) cross_asset_correlation: bool,
    pub(crate) spillover_analysis: bool,
    pub(crate) economic_indicators_weight: Float,
    pub(crate) risk_adjusted_scoring: bool,
    pub(crate) transaction_cost_weight: Float,
    pub(crate) market_neutrality_weight: Float,
    k: Option<usize>,
    pub(crate) score_threshold: Float,
    pub(crate) strategy: FinanceStrategy,
}
impl FinanceFeatureSelectorBuilder {
    pub fn new() -> Self {
        Self {
            feature_type: "technical_indicators".to_string(),
            include_momentum_indicators: true,
            include_volatility_indicators: true,
            include_volume_indicators: true,
            include_var_metrics: false,
            include_drawdown_metrics: false,
            include_order_flow: false,
            include_bid_ask_analysis: false,
            include_market_impact: false,
            include_regime_detection: false,
            lookback_periods: vec![5, 10, 20],
            var_confidence_level: 0.95,
            drawdown_threshold: 0.05,
            correlation_threshold: 0.7,
            tick_frequency: "1day".to_string(),
            liquidity_threshold: 0.1,
            regime_detection_method: "markov_switching".to_string(),
            cross_asset_correlation: false,
            spillover_analysis: false,
            economic_indicators_weight: 0.2,
            risk_adjusted_scoring: true,
            transaction_cost_weight: 0.1,
            market_neutrality_weight: 0.1,
            k: None,
            score_threshold: 0.1,
            strategy: FinanceStrategy::Momentum,
        }
    }
    /// Type of financial features: "technical_indicators", "risk_metrics", "microstructure", "cross_asset".
    pub fn feature_type(mut self, feature_type: &str) -> Self {
        self.feature_type = feature_type.to_string();
        self
    }
    /// Whether to include momentum indicators (RSI, MACD, etc.).
    pub fn include_momentum_indicators(mut self, include: bool) -> Self {
        self.include_momentum_indicators = include;
        self
    }
    /// Whether to include volatility indicators (Bollinger Bands, ATR, etc.).
    pub fn include_volatility_indicators(mut self, include: bool) -> Self {
        self.include_volatility_indicators = include;
        self
    }
    /// Whether to include volume indicators (OBV, VWAP, etc.).
    pub fn include_volume_indicators(mut self, include: bool) -> Self {
        self.include_volume_indicators = include;
        self
    }
    /// Whether to include Value-at-Risk metrics.
    pub fn include_var_metrics(mut self, include: bool) -> Self {
        self.include_var_metrics = include;
        self
    }
    /// Whether to include drawdown metrics.
    pub fn include_drawdown_metrics(mut self, include: bool) -> Self {
        self.include_drawdown_metrics = include;
        self
    }
    /// Whether to include order flow analysis.
    pub fn include_order_flow(mut self, include: bool) -> Self {
        self.include_order_flow = include;
        self
    }
    /// Whether to include bid-ask spread analysis.
    pub fn include_bid_ask_analysis(mut self, include: bool) -> Self {
        self.include_bid_ask_analysis = include;
        self
    }
    /// Whether to include market impact analysis.
    pub fn include_market_impact(mut self, include: bool) -> Self {
        self.include_market_impact = include;
        self
    }
    /// Whether to include regime detection.
    pub fn include_regime_detection(mut self, include: bool) -> Self {
        self.include_regime_detection = include;
        self
    }
    /// Lookback periods for technical indicators.
    pub fn lookback_periods(mut self, periods: Vec<usize>) -> Self {
        self.lookback_periods = periods;
        self
    }
    /// Confidence level for VaR calculations.
    pub fn var_confidence_level(mut self, level: Float) -> Self {
        self.var_confidence_level = level;
        self
    }
    /// Threshold for drawdown significance.
    pub fn drawdown_threshold(mut self, threshold: Float) -> Self {
        self.drawdown_threshold = threshold;
        self
    }
    /// Correlation threshold for feature filtering.
    pub fn correlation_threshold(mut self, threshold: Float) -> Self {
        self.correlation_threshold = threshold;
        self
    }
    /// Frequency of tick data: "1min", "5min", "1hour", "1day".
    pub fn tick_frequency(mut self, frequency: &str) -> Self {
        self.tick_frequency = frequency.to_string();
        self
    }
    /// Liquidity threshold for microstructure analysis.
    pub fn liquidity_threshold(mut self, threshold: Float) -> Self {
        self.liquidity_threshold = threshold;
        self
    }
    /// Method for regime detection: "markov_switching", "hidden_markov", "threshold".
    pub fn regime_detection_method(mut self, method: &str) -> Self {
        self.regime_detection_method = method.to_string();
        self
    }
    /// Whether to include cross-asset correlation analysis.
    pub fn cross_asset_correlation(mut self, include: bool) -> Self {
        self.cross_asset_correlation = include;
        self
    }
    /// Whether to include spillover effect analysis.
    pub fn spillover_analysis(mut self, include: bool) -> Self {
        self.spillover_analysis = include;
        self
    }
    /// Weight for economic indicators in scoring.
    pub fn economic_indicators_weight(mut self, weight: Float) -> Self {
        self.economic_indicators_weight = weight;
        self
    }
    /// Whether to use risk-adjusted scoring (Sharpe ratio, etc.).
    pub fn risk_adjusted_scoring(mut self, adjust: bool) -> Self {
        self.risk_adjusted_scoring = adjust;
        self
    }
    /// Weight for transaction cost consideration.
    pub fn transaction_cost_weight(mut self, weight: Float) -> Self {
        self.transaction_cost_weight = weight;
        self
    }
    /// Weight for market neutrality consideration.
    pub fn market_neutrality_weight(mut self, weight: Float) -> Self {
        self.market_neutrality_weight = weight;
        self
    }
    /// Number of top features to select.
    pub fn k(mut self, k: usize) -> Self {
        self.k = Some(k);
        self
    }
    /// Minimum score threshold for feature selection.
    pub fn score_threshold(mut self, threshold: Float) -> Self {
        self.score_threshold = threshold;
        self
    }
    /// Builds the FinanceFeatureSelector.
    pub fn build(self) -> FinanceFeatureSelector<Untrained> {
        FinanceFeatureSelector {
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
            k: self.k.unwrap_or(100),
            score_threshold: self.score_threshold,
            strategy: self.strategy,
            state: PhantomData,
            trained_state: None,
        }
    }
}
