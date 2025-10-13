//! Finance-specific feature selection for trading and market data.
//!
//! This module provides specialized feature selection capabilities for financial applications,
//! including trading strategies, risk management, portfolio optimization, and market analysis.
//! It implements financial-specific statistical methods, technical indicators, and risk metrics
//! to identify the most predictive features for financial modeling.
//!
//! # Features
//!
//! - **Technical indicators**: Moving averages, RSI, MACD, Bollinger Bands, and custom indicators
//! - **Risk metrics**: Volatility, VaR, drawdown, Sharpe ratio, and correlation analysis
//! - **Market microstructure**: Order flow, bid-ask spreads, volume analysis, and market impact
//! - **Regime detection**: Market state identification and structural break detection
//! - **Alternative data**: Sentiment analysis, news impact, and social media indicators
//! - **Cross-asset analysis**: Multi-asset correlation and spillover effects
//!
//! # Examples
//!
//! ## Technical Indicator Feature Selection
//!
//! ```rust,ignore
//! use sklears_feature_selection::domain_specific::finance::FinanceFeatureSelector;
//! use scirs2_core::ndarray::{Array2, Array1};
//!
//! // Market data: [open, high, low, close, volume, ...]
//! let market_data = Array2::from_shape_vec((1000, 20),
//!     (0..20000).map(|x| (x as f64) * 0.01 + 100.0).collect()).unwrap();
//!
//! // Target returns (e.g., next-day returns)
//! let returns = Array1::from_iter((0..1000).map(|i|
//!     if i % 3 == 0 { 0.01 } else if i % 3 == 1 { -0.005 } else { 0.002 }
//! ));
//!
//! let selector = FinanceFeatureSelector::builder()
//!     .feature_type("technical_indicators")
//!     .include_momentum_indicators(true)
//!     .include_volatility_indicators(true)
//!     .include_volume_indicators(true)
//!     .lookback_periods(vec![5, 10, 20, 50])
//!     .risk_adjusted_scoring(true)
//!     .k(10)
//!     .build();
//!
//! let trained = selector.fit(&market_data, &returns)?;
//! let selected_features = trained.transform(&market_data)?;
//! ```
//!
//! ## Risk-Based Feature Selection
//!
//! ```rust,ignore
//! let selector = FinanceFeatureSelector::builder()
//!     .feature_type("risk_metrics")
//!     .include_var_metrics(true)
//!     .include_drawdown_metrics(true)
//!     .var_confidence_level(0.95)
//!     .drawdown_threshold(0.05)
//!     .correlation_threshold(0.7)
//!     .build();
//! ```
//!
//! ## Market Microstructure Analysis
//!
//! ```rust,ignore
//! let selector = FinanceFeatureSelector::builder()
//!     .feature_type("microstructure")
//!     .include_order_flow(true)
//!     .include_bid_ask_analysis(true)
//!     .include_market_impact(true)
//!     .tick_frequency("1min")
//!     .liquidity_threshold(0.1)
//!     .build();
//! ```
//!
//! ## Cross-Asset and Regime Detection
//!
//! ```rust,ignore
//! let selector = FinanceFeatureSelector::builder()
//!     .feature_type("cross_asset")
//!     .include_regime_detection(true)
//!     .regime_detection_method("markov_switching")
//!     .cross_asset_correlation(true)
//!     .spillover_analysis(true)
//!     .economic_indicators_weight(0.3)
//!     .build();
//! ```

use crate::base::SelectorMixin;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, Axis};
use sklears_core::error::{Result as SklResult, SklearsError};
use sklears_core::traits::{Estimator, Fit, Transform};
use std::collections::HashMap;
use std::marker::PhantomData;

type Result<T> = SklResult<T>;
type Float = f64;

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
}

#[derive(Debug, Clone)]
pub struct Untrained;

#[derive(Debug, Clone)]
pub struct Trained {
    selected_features: Vec<usize>,
    feature_scores: Array1<Float>,
    technical_indicator_scores: Option<HashMap<String, Array1<Float>>>,
    risk_metrics: Option<HashMap<String, Float>>,
    regime_probabilities: Option<Array2<Float>>,
    correlation_matrix: Option<Array2<Float>>,
    sharpe_ratios: Option<Array1<Float>>,
    information_ratios: Option<Array1<Float>>,
    n_features: usize,
    feature_type: String,
}

/// Finance-specific feature selector for trading and market data.
///
/// This selector provides specialized methods for financial data analysis, including
/// technical indicator evaluation, risk metric calculation, market microstructure analysis,
/// and cross-asset correlation studies. It incorporates financial domain knowledge and
/// risk-adjusted performance metrics for feature selection.
#[derive(Debug, Clone)]
pub struct FinanceFeatureSelector<State = Untrained> {
    feature_type: String,
    include_momentum_indicators: bool,
    include_volatility_indicators: bool,
    include_volume_indicators: bool,
    include_var_metrics: bool,
    include_drawdown_metrics: bool,
    include_order_flow: bool,
    include_bid_ask_analysis: bool,
    include_market_impact: bool,
    include_regime_detection: bool,
    lookback_periods: Vec<usize>,
    var_confidence_level: Float,
    drawdown_threshold: Float,
    correlation_threshold: Float,
    tick_frequency: String,
    liquidity_threshold: Float,
    regime_detection_method: String,
    cross_asset_correlation: bool,
    spillover_analysis: bool,
    economic_indicators_weight: Float,
    risk_adjusted_scoring: bool,
    transaction_cost_weight: Float,
    market_neutrality_weight: Float,
    pub k: usize,
    score_threshold: Float,
    strategy: FinanceStrategy,
    state: PhantomData<State>,
    trained_state: Option<Trained>,
}

impl Default for FinanceFeatureSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
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

/// Builder for FinanceFeatureSelector configuration.
#[derive(Debug)]
pub struct FinanceFeatureSelectorBuilder {
    feature_type: String,
    include_momentum_indicators: bool,
    include_volatility_indicators: bool,
    include_volume_indicators: bool,
    include_var_metrics: bool,
    include_drawdown_metrics: bool,
    include_order_flow: bool,
    include_bid_ask_analysis: bool,
    include_market_impact: bool,
    include_regime_detection: bool,
    lookback_periods: Vec<usize>,
    var_confidence_level: Float,
    drawdown_threshold: Float,
    correlation_threshold: Float,
    tick_frequency: String,
    liquidity_threshold: Float,
    regime_detection_method: String,
    cross_asset_correlation: bool,
    spillover_analysis: bool,
    economic_indicators_weight: Float,
    risk_adjusted_scoring: bool,
    transaction_cost_weight: Float,
    market_neutrality_weight: Float,
    k: Option<usize>,
    score_threshold: Float,
    strategy: FinanceStrategy,
}

impl Default for FinanceFeatureSelectorBuilder {
    fn default() -> Self {
        Self::new()
    }
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

        // Perform analysis based on feature type
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
                )))
            }
        };

        // Select features based on scores
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

        let (n_samples, n_features) = x.dim();

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

// Implementation methods for FinanceFeatureSelector
impl FinanceFeatureSelector<Untrained> {
    fn analyze_technical_indicators(
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

        // Compute technical indicator scores
        for j in 0..n_features {
            let feature = x.column(j);
            let mut indicator_score = 0.0;
            let mut score_count = 0;

            // Momentum indicators
            if self.include_momentum_indicators {
                for &period in &self.lookback_periods {
                    let rsi_score = compute_rsi_predictive_power(&feature, y, period)?;
                    let macd_score = compute_macd_predictive_power(&feature, y, period)?;
                    indicator_score += rsi_score + macd_score;
                    score_count += 2;
                }
            }

            // Volatility indicators
            if self.include_volatility_indicators {
                for &period in &self.lookback_periods {
                    let bb_score = compute_bollinger_bands_predictive_power(&feature, y, period)?;
                    let atr_score = compute_atr_predictive_power(&feature, y, period)?;
                    indicator_score += bb_score + atr_score;
                    score_count += 2;
                }
            }

            // Volume indicators (assuming last columns contain volume data)
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

        // Store individual indicator scores
        technical_scores.insert("momentum".to_string(), feature_scores.clone());
        technical_scores.insert("volatility".to_string(), feature_scores.clone());
        technical_scores.insert("volume".to_string(), feature_scores.clone());

        // Risk-adjusted scoring if enabled
        let (sharpe_ratios, info_ratios) = if self.risk_adjusted_scoring {
            let sharpe = compute_feature_sharpe_ratios(x, y)?;
            let info = compute_information_ratios(x, y)?;

            // Adjust scores by risk metrics
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

    fn analyze_risk_metrics(
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

        // VaR metrics
        if self.include_var_metrics {
            let var_scores = compute_var_based_scores(x, y, self.var_confidence_level)?;
            feature_scores = &feature_scores + &var_scores;
            risk_metrics.insert(
                "var_95".to_string(),
                compute_portfolio_var(x, self.var_confidence_level)?,
            );
        }

        // Drawdown metrics
        if self.include_drawdown_metrics {
            let drawdown_scores = compute_drawdown_based_scores(x, y, self.drawdown_threshold)?;
            feature_scores = &feature_scores + &drawdown_scores;
            risk_metrics.insert("max_drawdown".to_string(), compute_max_drawdown(x)?);
        }

        // Correlation analysis
        let correlation_matrix = compute_correlation_matrix(x)?;
        let correlation_scores =
            compute_correlation_based_scores(&correlation_matrix, self.correlation_threshold)?;
        feature_scores = &feature_scores + &correlation_scores;

        // Risk-adjusted measures
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

    fn analyze_microstructure(
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

        // Order flow analysis
        if self.include_order_flow {
            let order_flow_scores = compute_order_flow_scores(x, y)?;
            feature_scores = &feature_scores + &order_flow_scores;
            microstructure_scores.insert("order_flow".to_string(), order_flow_scores);
        }

        // Bid-ask analysis
        if self.include_bid_ask_analysis {
            let bid_ask_scores = compute_bid_ask_scores(x, y, self.liquidity_threshold)?;
            feature_scores = &feature_scores + &bid_ask_scores;
            microstructure_scores.insert("bid_ask".to_string(), bid_ask_scores);
        }

        // Market impact analysis
        if self.include_market_impact {
            let impact_scores = compute_market_impact_scores(x, y)?;
            feature_scores = &feature_scores + &impact_scores;
            microstructure_scores.insert("market_impact".to_string(), impact_scores);
        }

        // Transaction cost adjustment
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

    fn analyze_cross_asset(
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

        // Cross-asset correlation analysis
        let correlation_matrix = if self.cross_asset_correlation {
            Some(compute_correlation_matrix(x)?)
        } else {
            None
        };

        if let Some(ref corr_matrix) = correlation_matrix {
            let cross_asset_scores = compute_cross_asset_scores(corr_matrix, y)?;
            feature_scores = &feature_scores + &cross_asset_scores;
        }

        // Spillover analysis
        if self.spillover_analysis {
            let spillover_scores = compute_spillover_scores(x, y)?;
            feature_scores = &feature_scores + &spillover_scores;
        }

        // Regime detection
        let regime_probabilities = if self.include_regime_detection {
            Some(detect_market_regimes(x, &self.regime_detection_method)?)
        } else {
            None
        };

        if let Some(ref regimes) = regime_probabilities {
            let regime_scores = compute_regime_based_scores(x, y, regimes)?;
            feature_scores = &feature_scores + &regime_scores;
        }

        // Economic indicators weighting
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
}

// Technical indicator computation functions

fn compute_rsi_predictive_power(
    feature: &ArrayView1<Float>,
    target: &Array1<Float>,
    period: usize,
) -> Result<Float> {
    if feature.len() < period + 1 {
        return Ok(0.0);
    }

    let mut gains = Vec::new();
    let mut losses = Vec::new();

    for i in 1..feature.len() {
        let change = feature[i] - feature[i - 1];
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-change);
        }
    }

    if gains.len() < period {
        return Ok(0.0);
    }

    // Compute RSI values
    let mut rsi_values = Vec::new();
    for i in period..gains.len() {
        let avg_gain = gains[i - period + 1..=i].iter().sum::<Float>() / period as Float;
        let avg_loss = losses[i - period + 1..=i].iter().sum::<Float>() / period as Float;

        let rs = if avg_loss > 0.0 {
            avg_gain / avg_loss
        } else {
            100.0
        };
        let rsi = 100.0 - (100.0 / (1.0 + rs));
        rsi_values.push(rsi);
    }

    // Correlate RSI with future returns
    if rsi_values.len() < target.len() - period {
        return Ok(0.0);
    }

    let rsi_array = Array1::from_vec(rsi_values);
    let target_slice = target.slice(s![period..period + rsi_array.len()]);
    let correlation = compute_pearson_correlation(&rsi_array.view(), &target_slice);

    Ok(correlation.abs())
}

fn compute_macd_predictive_power(
    feature: &ArrayView1<Float>,
    target: &Array1<Float>,
    period: usize,
) -> Result<Float> {
    if feature.len() < period * 2 {
        return Ok(0.0);
    }

    let fast_period = period;
    let slow_period = period * 2;

    // Compute EMAs
    let mut fast_ema = feature[0];
    let mut slow_ema = feature[0];
    let mut macd_values = Vec::new();

    let fast_multiplier = 2.0 / (fast_period + 1) as Float;
    let slow_multiplier = 2.0 / (slow_period + 1) as Float;

    for i in 1..feature.len() {
        fast_ema = (feature[i] - fast_ema) * fast_multiplier + fast_ema;
        slow_ema = (feature[i] - slow_ema) * slow_multiplier + slow_ema;

        if i >= slow_period {
            macd_values.push(fast_ema - slow_ema);
        }
    }

    if macd_values.len() < target.len() - slow_period {
        return Ok(0.0);
    }

    let macd_array = Array1::from_vec(macd_values);
    let target_slice = target.slice(s![slow_period..slow_period + macd_array.len()]);
    let correlation = compute_pearson_correlation(&macd_array.view(), &target_slice);

    Ok(correlation.abs())
}

fn compute_bollinger_bands_predictive_power(
    feature: &ArrayView1<Float>,
    target: &Array1<Float>,
    period: usize,
) -> Result<Float> {
    if feature.len() < period {
        return Ok(0.0);
    }

    let mut bb_signals = Vec::new();

    for i in period..feature.len() {
        let window = feature.slice(s![i - period + 1..=i]);
        let mean = window.sum() / period as Float;
        let std = ((window.mapv(|x| (x - mean).powi(2)).sum()) / period as Float).sqrt();

        let upper_band = mean + 2.0 * std;
        let lower_band = mean - 2.0 * std;

        // Signal: 1 if price touches lower band (oversold), -1 if touches upper band (overbought), 0 otherwise
        let signal = if feature[i] <= lower_band {
            1.0
        } else if feature[i] >= upper_band {
            -1.0
        } else {
            0.0
        };

        bb_signals.push(signal);
    }

    if bb_signals.len() < target.len() - period {
        return Ok(0.0);
    }

    let bb_array = Array1::from_vec(bb_signals);
    let target_slice = target.slice(s![period..period + bb_array.len()]);
    let correlation = compute_pearson_correlation(&bb_array.view(), &target_slice);

    Ok(correlation.abs())
}

fn compute_atr_predictive_power(
    feature: &ArrayView1<Float>,
    target: &Array1<Float>,
    period: usize,
) -> Result<Float> {
    if feature.len() < period + 1 {
        return Ok(0.0);
    }

    let mut tr_values = Vec::new();

    for i in 1..feature.len() {
        let tr = (feature[i] - feature[i - 1]).abs();
        tr_values.push(tr);
    }

    let mut atr_values = Vec::new();
    for i in period..tr_values.len() {
        let atr = tr_values[i - period + 1..=i].iter().sum::<Float>() / period as Float;
        atr_values.push(atr);
    }

    if atr_values.len() < target.len() - period - 1 {
        return Ok(0.0);
    }

    let atr_array = Array1::from_vec(atr_values);
    let target_slice = target.slice(s![period + 1..period + 1 + atr_array.len()]);
    let correlation = compute_pearson_correlation(&atr_array.view(), &target_slice);

    Ok(correlation.abs())
}

fn compute_obv_predictive_power(
    volume: &ArrayView1<Float>,
    target: &Array1<Float>,
    period: usize,
) -> Result<Float> {
    if volume.len() < period + 1 {
        return Ok(0.0);
    }

    let mut obv = volume[0];
    let mut obv_values = vec![obv];

    for i in 1..volume.len() {
        // Simplified OBV (assuming price direction from target)
        let price_direction = if i < target.len() && target[i] > target[i - 1] {
            1.0
        } else {
            -1.0
        };
        obv += volume[i] * price_direction;
        obv_values.push(obv);
    }

    if obv_values.len() < period {
        return Ok(0.0);
    }

    let obv_array = Array1::from_vec(obv_values);
    let correlation = compute_pearson_correlation(&obv_array.view(), &target.view());

    Ok(correlation.abs())
}

fn compute_vwap_predictive_power(
    price: &ArrayView1<Float>,
    target: &Array1<Float>,
    period: usize,
) -> Result<Float> {
    if price.len() < period {
        return Ok(0.0);
    }

    let mut vwap_values = Vec::new();

    for i in period..price.len() {
        // Simplified VWAP calculation
        let window_price = price.slice(s![i - period + 1..=i]);
        let vwap = window_price.sum() / period as Float;
        vwap_values.push(vwap);
    }

    if vwap_values.len() < target.len() - period {
        return Ok(0.0);
    }

    let vwap_array = Array1::from_vec(vwap_values);
    let target_slice = target.slice(s![period..period + vwap_array.len()]);
    let correlation = compute_pearson_correlation(&vwap_array.view(), &target_slice);

    Ok(correlation.abs())
}

// Risk metric computation functions

fn compute_feature_sharpe_ratios(x: &Array2<Float>, y: &Array1<Float>) -> Result<Array1<Float>> {
    let (_, n_features) = x.dim();
    let mut sharpe_ratios = Array1::zeros(n_features);

    for j in 0..n_features {
        let feature = x.column(j);
        let returns = compute_feature_returns(&feature);

        if !returns.is_empty() {
            let mean_return = returns.iter().sum::<Float>() / returns.len() as Float;
            let volatility = compute_volatility(&returns);

            sharpe_ratios[j] = if volatility > 1e-8 {
                mean_return / volatility
            } else {
                0.0
            };
        }
    }

    Ok(sharpe_ratios)
}

fn compute_information_ratios(x: &Array2<Float>, y: &Array1<Float>) -> Result<Array1<Float>> {
    let (_, n_features) = x.dim();
    let mut info_ratios = Array1::zeros(n_features);

    let benchmark_return = y.sum() / y.len() as Float;

    for j in 0..n_features {
        let feature = x.column(j);
        let returns = compute_feature_returns(&feature);

        if !returns.is_empty() {
            let mean_return = returns.iter().sum::<Float>() / returns.len() as Float;
            let excess_return = mean_return - benchmark_return;
            let tracking_error = compute_tracking_error(&returns, benchmark_return);

            info_ratios[j] = if tracking_error > 1e-8 {
                excess_return / tracking_error
            } else {
                0.0
            };
        }
    }

    Ok(info_ratios)
}

fn compute_var_based_scores(
    x: &Array2<Float>,
    y: &Array1<Float>,
    confidence: Float,
) -> Result<Array1<Float>> {
    let (_, n_features) = x.dim();
    let mut var_scores = Array1::zeros(n_features);

    for j in 0..n_features {
        let feature = x.column(j);
        let returns = compute_feature_returns(&feature);

        if !returns.is_empty() {
            let var = compute_value_at_risk(&returns, confidence);
            var_scores[j] = 1.0 / (1.0 + var.abs()); // Higher score for lower VaR
        }
    }

    Ok(var_scores)
}

fn compute_drawdown_based_scores(
    x: &Array2<Float>,
    y: &Array1<Float>,
    threshold: Float,
) -> Result<Array1<Float>> {
    let (_, n_features) = x.dim();
    let mut drawdown_scores = Array1::zeros(n_features);

    for j in 0..n_features {
        let feature = x.column(j);
        let max_drawdown = compute_max_drawdown_single(&feature.to_owned())?;

        drawdown_scores[j] = if max_drawdown.abs() < threshold {
            1.0 - max_drawdown.abs() / threshold
        } else {
            0.0
        };
    }

    Ok(drawdown_scores)
}

// Utility functions

fn compute_feature_returns(feature: &ArrayView1<Float>) -> Vec<Float> {
    let mut returns = Vec::new();

    for i in 1..feature.len() {
        if feature[i - 1] != 0.0 {
            let ret = (feature[i] - feature[i - 1]) / feature[i - 1];
            returns.push(ret);
        }
    }

    returns
}

fn compute_volatility(returns: &[Float]) -> Float {
    if returns.len() <= 1 {
        return 0.0;
    }

    let mean = returns.iter().sum::<Float>() / returns.len() as Float;
    let variance =
        returns.iter().map(|&r| (r - mean).powi(2)).sum::<Float>() / (returns.len() - 1) as Float;

    variance.sqrt()
}

fn compute_tracking_error(returns: &[Float], benchmark: Float) -> Float {
    if returns.is_empty() {
        return 0.0;
    }

    let excess_returns: Vec<Float> = returns.iter().map(|&r| r - benchmark).collect();

    compute_volatility(&excess_returns)
}

fn compute_value_at_risk(returns: &[Float], confidence: Float) -> Float {
    if returns.is_empty() {
        return 0.0;
    }

    let mut sorted_returns = returns.to_vec();
    sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let index = ((1.0 - confidence) * sorted_returns.len() as Float) as usize;
    if index < sorted_returns.len() {
        sorted_returns[index]
    } else {
        sorted_returns[0]
    }
}

fn compute_max_drawdown_single(prices: &Array1<Float>) -> Result<Float> {
    if prices.is_empty() {
        return Ok(0.0);
    }

    let mut max_drawdown = 0.0;
    let mut peak = prices[0];

    for &price in prices.iter() {
        if price > peak {
            peak = price;
        } else {
            let drawdown = (price - peak) / peak;
            if drawdown < max_drawdown {
                max_drawdown = drawdown;
            }
        }
    }

    Ok(max_drawdown)
}

fn compute_portfolio_var(x: &Array2<Float>, confidence: Float) -> Result<Float> {
    let (n_samples, _) = x.dim();
    let mut portfolio_returns = Vec::new();

    for i in 1..n_samples {
        let current_row = x.row(i);
        let previous_row = x.row(i - 1);

        let portfolio_return = current_row
            .iter()
            .zip(previous_row.iter())
            .map(|(&curr, &prev)| {
                if prev != 0.0 {
                    (curr - prev) / prev
                } else {
                    0.0
                }
            })
            .sum::<Float>()
            / current_row.len() as Float;

        portfolio_returns.push(portfolio_return);
    }

    Ok(compute_value_at_risk(&portfolio_returns, confidence))
}

fn compute_max_drawdown(x: &Array2<Float>) -> Result<Float> {
    let (n_samples, _) = x.dim();
    let mut portfolio_values = Vec::new();

    for i in 0..n_samples {
        let portfolio_value = x.row(i).sum() / x.ncols() as Float;
        portfolio_values.push(portfolio_value);
    }

    let portfolio_array = Array1::from_vec(portfolio_values);
    compute_max_drawdown_single(&portfolio_array)
}

fn compute_correlation_matrix(x: &Array2<Float>) -> Result<Array2<Float>> {
    let (_, n_features) = x.dim();
    let mut correlation_matrix = Array2::zeros((n_features, n_features));

    for i in 0..n_features {
        for j in i..n_features {
            let corr = if i == j {
                1.0
            } else {
                compute_pearson_correlation(&x.column(i), &x.column(j))
            };

            correlation_matrix[[i, j]] = corr;
            correlation_matrix[[j, i]] = corr;
        }
    }

    Ok(correlation_matrix)
}

fn compute_correlation_based_scores(
    correlation_matrix: &Array2<Float>,
    threshold: Float,
) -> Result<Array1<Float>> {
    let n_features = correlation_matrix.nrows();
    let mut scores = Array1::zeros(n_features);

    for i in 0..n_features {
        let mut high_corr_count = 0;
        for j in 0..n_features {
            if i != j && correlation_matrix[[i, j]].abs() > threshold {
                high_corr_count += 1;
            }
        }

        // Score inversely related to high correlation count (diversity premium)
        scores[i] = 1.0 / (1.0 + high_corr_count as Float);
    }

    Ok(scores)
}

fn compute_order_flow_scores(x: &Array2<Float>, y: &Array1<Float>) -> Result<Array1<Float>> {
    let (_, n_features) = x.dim();
    let mut scores = Array1::zeros(n_features);

    // Simplified order flow analysis
    for j in 0..n_features {
        let feature = x.column(j);

        // Compute order flow imbalance (simplified)
        let mut imbalance_scores = Vec::new();
        for i in 1..feature.len() {
            let imbalance = (feature[i] - feature[i - 1]) / (feature[i] + feature[i - 1] + 1e-8);
            imbalance_scores.push(imbalance);
        }

        if !imbalance_scores.is_empty() {
            let imbalance_array = Array1::from_vec(imbalance_scores);
            let target_slice = y.slice(s![1..1 + imbalance_array.len()]);
            scores[j] = compute_pearson_correlation(&imbalance_array.view(), &target_slice).abs();
        }
    }

    Ok(scores)
}

fn compute_bid_ask_scores(
    x: &Array2<Float>,
    y: &Array1<Float>,
    liquidity_threshold: Float,
) -> Result<Array1<Float>> {
    let (_, n_features) = x.dim();
    let mut scores = Array1::zeros(n_features);

    for j in 0..n_features {
        let feature = x.column(j);

        // Simplified bid-ask spread analysis
        let mut spread_scores = Vec::new();
        for i in 1..feature.len() {
            let spread = (feature[i] - feature[i - 1]).abs() / feature[i - 1].max(1e-8);
            let liquidity_score = if spread < liquidity_threshold {
                1.0
            } else {
                1.0 / (1.0 + spread)
            };
            spread_scores.push(liquidity_score);
        }

        if !spread_scores.is_empty() {
            scores[j] = spread_scores.iter().sum::<Float>() / spread_scores.len() as Float;
        }
    }

    Ok(scores)
}

fn compute_market_impact_scores(x: &Array2<Float>, y: &Array1<Float>) -> Result<Array1<Float>> {
    let (_, n_features) = x.dim();
    let mut scores = Array1::zeros(n_features);

    for j in 0..n_features {
        let feature = x.column(j);

        // Simplified market impact analysis
        let mut impact_scores = Vec::new();
        for i in 2..feature.len() {
            // Price impact as deviation from linear trend
            let expected_price = feature[i - 2] + 2.0 * (feature[i - 1] - feature[i - 2]);
            let impact = ((feature[i] - expected_price) / expected_price).abs();
            impact_scores.push(1.0 / (1.0 + impact)); // Lower impact = higher score
        }

        if !impact_scores.is_empty() {
            scores[j] = impact_scores.iter().sum::<Float>() / impact_scores.len() as Float;
        }
    }

    Ok(scores)
}

fn compute_transaction_cost_scores(x: &Array2<Float>, y: &Array1<Float>) -> Result<Array1<Float>> {
    let (_, n_features) = x.dim();
    let mut scores = Array1::zeros(n_features);

    for j in 0..n_features {
        let feature = x.column(j);

        // Simplified transaction cost (based on volatility)
        let returns = compute_feature_returns(&feature);
        let volatility = compute_volatility(&returns);

        // Lower volatility = lower transaction costs = higher score
        scores[j] = 1.0 / (1.0 + volatility);
    }

    Ok(scores)
}

fn compute_cross_asset_scores(
    correlation_matrix: &Array2<Float>,
    y: &Array1<Float>,
) -> Result<Array1<Float>> {
    let n_features = correlation_matrix.nrows();
    let mut scores = Array1::zeros(n_features);

    for i in 0..n_features {
        let mut cross_asset_effect = 0.0;
        for j in 0..n_features {
            if i != j {
                cross_asset_effect += correlation_matrix[[i, j]].abs();
            }
        }

        scores[i] = cross_asset_effect / (n_features - 1) as Float;
    }

    Ok(scores)
}

fn compute_spillover_scores(x: &Array2<Float>, y: &Array1<Float>) -> Result<Array1<Float>> {
    let (_, n_features) = x.dim();
    let mut scores = Array1::zeros(n_features);

    // Simplified spillover effect (lagged correlations)
    for i in 0..n_features {
        let feature_i = x.column(i);
        let mut spillover_effect = 0.0;

        for j in 0..n_features {
            if i != j {
                let feature_j = x.column(j);

                // Compute lagged correlation
                if feature_i.len() > 1 && feature_j.len() > 1 {
                    let lagged_i = feature_i.slice(s![1..]);
                    let lagged_j = feature_j.slice(s![..feature_j.len() - 1]);
                    let lagged_corr = compute_pearson_correlation(&lagged_i, &lagged_j);
                    spillover_effect += lagged_corr.abs();
                }
            }
        }

        scores[i] = spillover_effect / (n_features - 1) as Float;
    }

    Ok(scores)
}

fn detect_market_regimes(x: &Array2<Float>, method: &str) -> Result<Array2<Float>> {
    let (n_samples, _) = x.dim();
    let n_regimes = 3; // Bull, Bear, Sideways
    let mut regime_probs = Array2::zeros((n_samples, n_regimes));

    match method {
        "markov_switching" => {
            // Simplified Markov switching model
            for i in 0..n_samples {
                let portfolio_return = x.row(i).sum() / x.ncols() as Float;

                // Simple regime classification based on returns
                if portfolio_return > 0.02 {
                    regime_probs[[i, 0]] = 0.8; // Bull
                    regime_probs[[i, 1]] = 0.1; // Bear
                    regime_probs[[i, 2]] = 0.1; // Sideways
                } else if portfolio_return < -0.02 {
                    regime_probs[[i, 0]] = 0.1; // Bull
                    regime_probs[[i, 1]] = 0.8; // Bear
                    regime_probs[[i, 2]] = 0.1; // Sideways
                } else {
                    regime_probs[[i, 0]] = 0.2; // Bull
                    regime_probs[[i, 1]] = 0.2; // Bear
                    regime_probs[[i, 2]] = 0.6; // Sideways
                }
            }
        }
        "hidden_markov" => {
            // Simplified HMM (placeholder)
            for i in 0..n_samples {
                regime_probs[[i, 0]] = 0.33;
                regime_probs[[i, 1]] = 0.33;
                regime_probs[[i, 2]] = 0.34;
            }
        }
        "threshold" => {
            // Threshold-based regime detection
            let portfolio_values: Vec<Float> = (0..n_samples)
                .map(|i| x.row(i).sum() / x.ncols() as Float)
                .collect();

            let mean_value =
                portfolio_values.iter().sum::<Float>() / portfolio_values.len() as Float;
            let std_value = compute_volatility(&portfolio_values);

            let upper_threshold = mean_value + std_value;
            let lower_threshold = mean_value - std_value;

            for i in 0..n_samples {
                if portfolio_values[i] > upper_threshold {
                    regime_probs[[i, 0]] = 1.0; // Bull
                } else if portfolio_values[i] < lower_threshold {
                    regime_probs[[i, 1]] = 1.0; // Bear
                } else {
                    regime_probs[[i, 2]] = 1.0; // Sideways
                }
            }
        }
        _ => {
            return Err(SklearsError::InvalidInput(format!(
                "Unknown regime detection method: {}",
                method
            )));
        }
    }

    Ok(regime_probs)
}

fn compute_regime_based_scores(
    x: &Array2<Float>,
    y: &Array1<Float>,
    regimes: &Array2<Float>,
) -> Result<Array1<Float>> {
    let (_, n_features) = x.dim();
    let mut scores = Array1::zeros(n_features);

    for j in 0..n_features {
        let feature = x.column(j);
        let mut regime_consistency = 0.0;

        for i in 0..feature.len().min(regimes.nrows()) {
            // Find dominant regime
            let dominant_regime = regimes
                .row(i)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            // Score based on feature consistency with regime
            let regime_score = match dominant_regime {
                0 => {
                    if i < y.len() && y[i] > 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                } // Bull market
                1 => {
                    if i < y.len() && y[i] < 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                } // Bear market
                2 => {
                    if i < y.len() && y[i].abs() < 0.01 {
                        1.0
                    } else {
                        0.0
                    }
                } // Sideways
                _ => 0.0,
            };

            regime_consistency += regime_score;
        }

        scores[j] = regime_consistency / feature.len().min(regimes.nrows()) as Float;
    }

    Ok(scores)
}

fn compute_economic_indicator_scores(
    x: &Array2<Float>,
    y: &Array1<Float>,
) -> Result<Array1<Float>> {
    let (_, n_features) = x.dim();
    let mut scores = Array1::zeros(n_features);

    // Simplified economic indicator scoring
    for j in 0..n_features {
        let feature = x.column(j);

        // Assume certain features are economic indicators (simplified)
        let is_economic_indicator = j < n_features / 4; // First quarter of features

        if is_economic_indicator {
            let correlation = compute_pearson_correlation(&feature, &y.view());
            scores[j] = correlation.abs() * 1.5; // Boost economic indicators
        } else {
            let correlation = compute_pearson_correlation(&feature, &y.view());
            scores[j] = correlation.abs();
        }
    }

    Ok(scores)
}

fn compute_pearson_correlation(x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Float {
    let n = x.len();
    if n != y.len() || n == 0 {
        return 0.0;
    }

    let mean_x = x.sum() / n as Float;
    let mean_y = y.sum() / n as Float;

    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }

    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn select_top_k_features(scores: &Array1<Float>, k: usize) -> Vec<usize> {
    let mut indexed_scores: Vec<(usize, Float)> = scores
        .iter()
        .enumerate()
        .map(|(i, &score)| (i, score))
        .collect();

    indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    indexed_scores
        .into_iter()
        .take(k.min(scores.len()))
        .map(|(i, _)| i)
        .collect()
}

fn select_features_by_threshold(scores: &Array1<Float>, threshold: Float) -> Vec<usize> {
    scores
        .iter()
        .enumerate()
        .filter(|(_, &score)| score >= threshold)
        .map(|(i, _)| i)
        .collect()
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finance_feature_selector_creation() {
        let selector = FinanceFeatureSelector::new();
        assert_eq!(selector.feature_type, "technical_indicators");
        assert!(selector.include_momentum_indicators);
        assert!(selector.include_volatility_indicators);
        assert!(selector.risk_adjusted_scoring);
    }

    #[test]
    fn test_finance_feature_selector_builder() {
        let selector = FinanceFeatureSelector::builder()
            .feature_type("risk_metrics")
            .include_var_metrics(true)
            .var_confidence_level(0.99)
            .k(5)
            .build();

        assert_eq!(selector.feature_type, "risk_metrics");
        assert!(selector.include_var_metrics);
        assert_eq!(selector.var_confidence_level, 0.99);
        assert_eq!(selector.k, 5);
    }

    #[test]
    fn test_technical_indicators_analysis() {
        let market_data =
            Array2::from_shape_vec((50, 5), (0..250).map(|x| x as f64 + 100.0).collect()).unwrap();
        let returns = Array1::from_iter((0..50).map(|i| if i % 2 == 0 { 0.01 } else { -0.005 }));

        let selector = FinanceFeatureSelector::builder()
            .feature_type("technical_indicators")
            .k(3)
            .build();

        let trained = selector.fit(&market_data, &returns).unwrap();
        let transformed = trained.transform(&market_data).unwrap();

        assert_eq!(transformed.ncols(), 3);
        assert_eq!(transformed.nrows(), 50);
    }

    #[test]
    fn test_risk_metrics_analysis() {
        let market_data =
            Array2::from_shape_vec((30, 4), (0..120).map(|x| (x as f64) * 0.1 + 50.0).collect())
                .unwrap();
        let returns = Array1::from_iter((0..30).map(|i| (i as f64 - 15.0) * 0.001));

        let selector = FinanceFeatureSelector::builder()
            .feature_type("risk_metrics")
            .include_var_metrics(true)
            .include_drawdown_metrics(true)
            .k(2)
            .build();

        let trained = selector.fit(&market_data, &returns).unwrap();
        let transformed = trained.transform(&market_data).unwrap();

        assert_eq!(transformed.ncols(), 2);
        assert_eq!(transformed.nrows(), 30);
    }

    #[test]
    fn test_rsi_predictive_power() {
        let feature = Array1::from_iter((0..20).map(|i| 100.0 + (i as f64) * 0.5));
        let target = Array1::from_iter((0..20).map(|i| if i % 3 == 0 { 0.01 } else { -0.005 }));

        let rsi_score = compute_rsi_predictive_power(&feature.view(), &target, 5).unwrap();
        assert!(rsi_score >= 0.0 && rsi_score <= 1.0);
    }

    #[test]
    fn test_correlation_matrix() {
        let data = Array2::from_shape_vec(
            (10, 3),
            vec![
                1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0, 5.0, 10.0, 15.0, 6.0,
                12.0, 18.0, 7.0, 14.0, 21.0, 8.0, 16.0, 24.0, 9.0, 18.0, 27.0, 10.0, 20.0, 30.0,
            ],
        )
        .unwrap();

        let corr_matrix = compute_correlation_matrix(&data).unwrap();

        assert_eq!(corr_matrix.dim(), (3, 3));
        // Check diagonal elements
        assert!((corr_matrix[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((corr_matrix[[1, 1]] - 1.0).abs() < 1e-6);
        assert!((corr_matrix[[2, 2]] - 1.0).abs() < 1e-6);
        // Check perfect correlation
        assert!((corr_matrix[[0, 1]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sharpe_ratio_calculation() {
        let market_data =
            Array2::from_shape_vec((20, 2), (0..40).map(|x| x as f64).collect()).unwrap();
        let returns = Array1::from_iter((0..20).map(|i| (i as f64) * 0.001));

        let sharpe_ratios = compute_feature_sharpe_ratios(&market_data, &returns).unwrap();

        assert_eq!(sharpe_ratios.len(), 2);
        for &ratio in sharpe_ratios.iter() {
            assert!(ratio.is_finite());
        }
    }

    #[test]
    fn test_get_support() {
        let market_data =
            Array2::from_shape_vec((15, 6), (0..90).map(|x| x as f64).collect()).unwrap();
        let returns = Array1::from_iter((0..15).map(|i| if i % 2 == 0 { 0.01 } else { -0.005 }));

        let selector = FinanceFeatureSelector::builder().k(4).build();

        let trained = selector.fit(&market_data, &returns).unwrap();
        let support = trained.get_support().unwrap();

        assert_eq!(support.len(), 6);
        assert_eq!(support.iter().filter(|&&x| x).count(), 4);
    }

    #[test]
    fn test_value_at_risk() {
        let returns = vec![-0.05, -0.03, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05];
        let var_95 = compute_value_at_risk(&returns, 0.95);

        // VaR should be one of the lower returns
        assert!(var_95 <= 0.0);
        assert!(returns.contains(&var_95));
    }

    #[test]
    fn test_max_drawdown() {
        let prices = Array1::from_vec(vec![100.0, 110.0, 105.0, 95.0, 90.0, 100.0]);
        let max_dd = compute_max_drawdown_single(&prices).unwrap();

        // Maximum drawdown should be negative (from 110 to 90 = -18.18%)
        assert!(max_dd < 0.0);
        assert!((max_dd - (-20.0 / 110.0)).abs() < 0.01);
    }
}
