//! Finance and Economics Kernel Methods
//!
//! This module implements kernel methods for financial and economic applications,
//! including financial time series analysis, volatility modeling, econometric methods,
//! portfolio optimization, and risk analysis.
//!
//! # References
//! - Engle (1982): "Autoregressive Conditional Heteroskedasticity"
//! - Bollerslev (1986): "Generalized Autoregressive Conditional Heteroskedasticity"
//! - Tsay (2010): "Analysis of Financial Time Series"
//! - McNeil et al. (2015): "Quantitative Risk Management"

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{thread_rng, Distribution};
use sklears_core::{
    error::{Result, SklearsError},
    prelude::{Fit, Transform},
    traits::{Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

// ============================================================================
// Financial Kernel
// ============================================================================

/// Kernel method for financial time series analysis
///
/// This kernel extracts features from financial time series data including
/// returns, volatility, technical indicators, and market microstructure features.
///
/// # References
/// - Murphy (1999): "Technical Analysis of the Financial Markets"
pub struct FinancialKernel<State = Untrained> {
    /// Number of random features
    n_components: usize,
    /// Window size for technical indicators
    window_size: usize,
    /// Whether to include volatility features
    include_volatility: bool,
    /// Random projection matrix
    projection: Option<Array2<Float>>,
    /// Feature weights (returns, volatility, indicators)
    feature_weights: Option<Array1<Float>>,
    /// State marker
    _state: PhantomData<State>,
}

impl FinancialKernel<Untrained> {
    /// Create a new financial kernel
    pub fn new(n_components: usize, window_size: usize) -> Self {
        Self {
            n_components,
            window_size,
            include_volatility: true,
            projection: None,
            feature_weights: None,
            _state: PhantomData,
        }
    }

    /// Set whether to include volatility features
    pub fn include_volatility(mut self, include_volatility: bool) -> Self {
        self.include_volatility = include_volatility;
        self
    }
}

impl Default for FinancialKernel<Untrained> {
    fn default() -> Self {
        Self::new(100, 20)
    }
}

impl Fit<Array2<Float>, ()> for FinancialKernel<Untrained> {
    type Fitted = FinancialKernel<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array cannot be empty".to_string(),
            ));
        }

        // Feature dimension: returns + volatility + technical indicators
        let base_features = 10; // Returns features
        let volatility_features = if self.include_volatility { 5 } else { 0 };
        let technical_features = 8; // Moving averages, RSI, etc.
        let feature_dim = base_features + volatility_features + technical_features;

        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0 / (feature_dim as Float).sqrt()).unwrap();

        // Generate random projection
        let mut projection = Array2::zeros((feature_dim, self.n_components));
        for i in 0..feature_dim {
            for j in 0..self.n_components {
                projection[[i, j]] = normal.sample(&mut rng);
            }
        }

        // Initialize feature weights (returns > volatility > indicators)
        let mut feature_weights = Array1::zeros(3);
        feature_weights[0] = 1.0; // Returns
        feature_weights[1] = 0.7; // Volatility
        feature_weights[2] = 0.5; // Technical indicators

        Ok(FinancialKernel {
            n_components: self.n_components,
            window_size: self.window_size,
            include_volatility: self.include_volatility,
            projection: Some(projection),
            feature_weights: Some(feature_weights),
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for FinancialKernel<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array cannot be empty".to_string(),
            ));
        }

        let projection = self.projection.as_ref().unwrap();
        let feature_dim = projection.nrows();
        let feature_weights = self.feature_weights.as_ref().unwrap();

        // Extract financial features
        let mut financial_features = Array2::zeros((n_samples, feature_dim));

        for i in 0..n_samples {
            let mut feat_idx = 0;

            // Returns features (log returns approximation)
            for j in 0..10.min(n_features) {
                if j + 1 < n_features {
                    let ret = (x[[i, j + 1]] / (x[[i, j]] + 1e-8)).ln();
                    financial_features[[i, feat_idx]] = ret * feature_weights[0];
                    feat_idx += 1;
                }
            }

            // Volatility features (rolling std approximation)
            if self.include_volatility {
                for j in 0..5.min(n_features / 2) {
                    let mut sum_sq = 0.0;
                    let window = self.window_size.min(n_features - j);
                    for k in 0..window {
                        if j + k < n_features {
                            sum_sq += (x[[i, j + k]] - x[[i, j]]).powi(2);
                        }
                    }
                    let volatility = (sum_sq / window as Float).sqrt();
                    if feat_idx < feature_dim {
                        financial_features[[i, feat_idx]] = volatility * feature_weights[1];
                        feat_idx += 1;
                    }
                }
            }

            // Technical indicators (simple moving average approximation)
            for j in 0..8.min(n_features / 3) {
                let window = self.window_size.min(n_features - j);
                let mut sma = 0.0;
                for k in 0..window {
                    if j + k < n_features {
                        sma += x[[i, j + k]];
                    }
                }
                sma /= window as Float;
                if feat_idx < feature_dim {
                    financial_features[[i, feat_idx]] = sma * feature_weights[2];
                    feat_idx += 1;
                }
            }
        }

        // Apply random projection
        let features = financial_features.dot(projection);

        Ok(features)
    }
}

// ============================================================================
// Volatility Kernel
// ============================================================================

/// Volatility modeling method
#[derive(Debug, Clone, Copy)]
pub enum VolatilityModel {
    /// Simple historical volatility
    Historical,
    /// EWMA volatility
    EWMA,
    /// GARCH-like volatility
    GARCH,
    /// Realized volatility
    Realized,
}

/// Kernel method for volatility modeling and forecasting
///
/// This kernel focuses on volatility features using various volatility models.
///
/// # References
/// - Bollerslev (1986): "Generalized Autoregressive Conditional Heteroskedasticity"
pub struct VolatilityKernel<State = Untrained> {
    /// Number of random features
    n_components: usize,
    /// Volatility model to use
    model: VolatilityModel,
    /// Decay factor for EWMA (typically 0.94)
    decay_factor: Float,
    /// Random projection matrix
    projection: Option<Array2<Float>>,
    /// State marker
    _state: PhantomData<State>,
}

impl VolatilityKernel<Untrained> {
    /// Create a new volatility kernel
    pub fn new(n_components: usize, model: VolatilityModel) -> Self {
        Self {
            n_components,
            model,
            decay_factor: 0.94,
            projection: None,
            _state: PhantomData,
        }
    }

    /// Set the decay factor for EWMA
    pub fn decay_factor(mut self, decay_factor: Float) -> Self {
        self.decay_factor = decay_factor;
        self
    }
}

impl Default for VolatilityKernel<Untrained> {
    fn default() -> Self {
        Self::new(100, VolatilityModel::EWMA)
    }
}

impl Fit<Array2<Float>, ()> for VolatilityKernel<Untrained> {
    type Fitted = VolatilityKernel<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array cannot be empty".to_string(),
            ));
        }

        // Volatility features dimension
        let feature_dim = 20;

        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0 / (feature_dim as Float).sqrt()).unwrap();

        // Generate random projection
        let mut projection = Array2::zeros((feature_dim, self.n_components));
        for i in 0..feature_dim {
            for j in 0..self.n_components {
                projection[[i, j]] = normal.sample(&mut rng);
            }
        }

        Ok(VolatilityKernel {
            n_components: self.n_components,
            model: self.model,
            decay_factor: self.decay_factor,
            projection: Some(projection),
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for VolatilityKernel<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array cannot be empty".to_string(),
            ));
        }

        let projection = self.projection.as_ref().unwrap();
        let feature_dim = projection.nrows();

        // Extract volatility features
        let mut volatility_features = Array2::zeros((n_samples, feature_dim));

        for i in 0..n_samples {
            match self.model {
                VolatilityModel::Historical => {
                    // Simple historical volatility at different horizons
                    for j in 0..feature_dim.min(n_features) {
                        let window = 5 + j;
                        let mut sum_sq = 0.0;
                        let count = window.min(n_features);
                        for k in 0..count {
                            sum_sq += x[[i, k]].powi(2);
                        }
                        volatility_features[[i, j]] = (sum_sq / count as Float).sqrt();
                    }
                }
                VolatilityModel::EWMA => {
                    // Exponentially weighted moving average volatility
                    for j in 0..feature_dim.min(n_features) {
                        let mut ewma_var = x[[i, 0]].powi(2);
                        for k in 1..n_features {
                            ewma_var = self.decay_factor * ewma_var
                                + (1.0 - self.decay_factor) * x[[i, k]].powi(2);
                        }
                        volatility_features[[i, j]] = ewma_var.sqrt();
                    }
                }
                VolatilityModel::GARCH => {
                    // Simplified GARCH(1,1) features
                    for j in 0..feature_dim.min(n_features) {
                        let alpha = 0.1;
                        let beta = 0.85;
                        let omega = 0.05;
                        let mut variance = x[[i, 0]].powi(2);
                        for k in 1..n_features {
                            variance = omega + alpha * x[[i, k - 1]].powi(2) + beta * variance;
                        }
                        volatility_features[[i, j]] = variance.sqrt();
                    }
                }
                VolatilityModel::Realized => {
                    // Realized volatility (sum of squared returns)
                    for j in 0..feature_dim.min(n_features) {
                        let mut realized_var = 0.0;
                        for k in 0..n_features {
                            realized_var += x[[i, k]].powi(2);
                        }
                        volatility_features[[i, j]] = (realized_var / n_features as Float).sqrt();
                    }
                }
            }
        }

        // Apply random projection
        let features = volatility_features.dot(projection);

        Ok(features)
    }
}

// ============================================================================
// Econometric Kernel
// ============================================================================

/// Kernel method for econometric time series analysis
///
/// This kernel implements features based on econometric models including
/// autoregressive features, cointegration, and structural breaks.
///
/// # References
/// - Hamilton (1994): "Time Series Analysis"
pub struct EconometricKernel<State = Untrained> {
    /// Number of random features
    n_components: usize,
    /// AR model order
    ar_order: usize,
    /// Whether to include difference features
    include_differences: bool,
    /// Random projection matrix
    projection: Option<Array2<Float>>,
    /// State marker
    _state: PhantomData<State>,
}

impl EconometricKernel<Untrained> {
    /// Create a new econometric kernel
    pub fn new(n_components: usize, ar_order: usize) -> Self {
        Self {
            n_components,
            ar_order,
            include_differences: true,
            projection: None,
            _state: PhantomData,
        }
    }

    /// Set whether to include difference features
    pub fn include_differences(mut self, include_differences: bool) -> Self {
        self.include_differences = include_differences;
        self
    }
}

impl Default for EconometricKernel<Untrained> {
    fn default() -> Self {
        Self::new(100, 4)
    }
}

impl Fit<Array2<Float>, ()> for EconometricKernel<Untrained> {
    type Fitted = EconometricKernel<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array cannot be empty".to_string(),
            ));
        }

        // Feature dimension based on AR order and differences
        let base_dim = self.ar_order * 2;
        let diff_dim = if self.include_differences { 10 } else { 0 };
        let feature_dim = base_dim + diff_dim;

        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0 / (feature_dim as Float).sqrt()).unwrap();

        // Generate random projection
        let mut projection = Array2::zeros((feature_dim, self.n_components));
        for i in 0..feature_dim {
            for j in 0..self.n_components {
                projection[[i, j]] = normal.sample(&mut rng);
            }
        }

        Ok(EconometricKernel {
            n_components: self.n_components,
            ar_order: self.ar_order,
            include_differences: self.include_differences,
            projection: Some(projection),
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for EconometricKernel<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array cannot be empty".to_string(),
            ));
        }

        let projection = self.projection.as_ref().unwrap();
        let feature_dim = projection.nrows();

        // Extract econometric features
        let mut econ_features = Array2::zeros((n_samples, feature_dim));

        for i in 0..n_samples {
            let mut feat_idx = 0;

            // AR features (lagged values)
            for lag in 1..=self.ar_order {
                if lag < n_features && feat_idx < feature_dim {
                    econ_features[[i, feat_idx]] = x[[i, n_features - lag - 1]];
                    feat_idx += 1;
                }
            }

            // Autocorrelation approximation
            for lag in 1..=self.ar_order {
                if lag < n_features && feat_idx < feature_dim {
                    let mut autocorr = 0.0;
                    let count = n_features - lag;
                    for k in 0..count {
                        autocorr += x[[i, k]] * x[[i, k + lag]];
                    }
                    econ_features[[i, feat_idx]] = autocorr / count as Float;
                    feat_idx += 1;
                }
            }

            // First and second differences
            if self.include_differences {
                for j in 0..10.min(n_features - 1) {
                    if feat_idx < feature_dim {
                        let diff1 = x[[i, j + 1]] - x[[i, j]];
                        econ_features[[i, feat_idx]] = diff1;
                        feat_idx += 1;
                    }
                }
            }
        }

        // Apply random projection
        let features = econ_features.dot(projection);

        Ok(features)
    }
}

// ============================================================================
// Portfolio Kernel
// ============================================================================

/// Kernel method for portfolio optimization and analysis
///
/// This kernel extracts portfolio-relevant features including risk-return
/// characteristics, diversification measures, and factor exposures.
///
/// # References
/// - Markowitz (1952): "Portfolio Selection"
pub struct PortfolioKernel<State = Untrained> {
    /// Number of random features
    n_components: usize,
    /// Number of assets in portfolio
    n_assets: usize,
    /// Risk-free rate for Sharpe ratio
    risk_free_rate: Float,
    /// Random projection matrix
    projection: Option<Array2<Float>>,
    /// State marker
    _state: PhantomData<State>,
}

impl PortfolioKernel<Untrained> {
    /// Create a new portfolio kernel
    pub fn new(n_components: usize, n_assets: usize) -> Self {
        Self {
            n_components,
            n_assets,
            risk_free_rate: 0.02,
            projection: None,
            _state: PhantomData,
        }
    }

    /// Set the risk-free rate
    pub fn risk_free_rate(mut self, risk_free_rate: Float) -> Self {
        self.risk_free_rate = risk_free_rate;
        self
    }
}

impl Default for PortfolioKernel<Untrained> {
    fn default() -> Self {
        Self::new(100, 10)
    }
}

impl Fit<Array2<Float>, ()> for PortfolioKernel<Untrained> {
    type Fitted = PortfolioKernel<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array cannot be empty".to_string(),
            ));
        }

        // Portfolio features: returns, risk, correlations, diversification
        let feature_dim = 30;

        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0 / (feature_dim as Float).sqrt()).unwrap();

        // Generate random projection
        let mut projection = Array2::zeros((feature_dim, self.n_components));
        for i in 0..feature_dim {
            for j in 0..self.n_components {
                projection[[i, j]] = normal.sample(&mut rng);
            }
        }

        Ok(PortfolioKernel {
            n_components: self.n_components,
            n_assets: self.n_assets,
            risk_free_rate: self.risk_free_rate,
            projection: Some(projection),
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for PortfolioKernel<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array cannot be empty".to_string(),
            ));
        }

        let projection = self.projection.as_ref().unwrap();
        let feature_dim = projection.nrows();

        // Extract portfolio features
        let mut portfolio_features = Array2::zeros((n_samples, feature_dim));

        for i in 0..n_samples {
            let mut feat_idx = 0;

            // Expected return approximation
            let mut mean_return = 0.0;
            for j in 0..n_features {
                mean_return += x[[i, j]];
            }
            mean_return /= n_features as Float;
            if feat_idx < feature_dim {
                portfolio_features[[i, feat_idx]] = mean_return;
                feat_idx += 1;
            }

            // Portfolio volatility approximation
            let mut variance = 0.0;
            for j in 0..n_features {
                variance += (x[[i, j]] - mean_return).powi(2);
            }
            let volatility = (variance / n_features as Float).sqrt();
            if feat_idx < feature_dim {
                portfolio_features[[i, feat_idx]] = volatility;
                feat_idx += 1;
            }

            // Sharpe ratio approximation
            if volatility > 1e-8 && feat_idx < feature_dim {
                let sharpe = (mean_return - self.risk_free_rate) / volatility;
                portfolio_features[[i, feat_idx]] = sharpe;
                feat_idx += 1;
            }

            // Diversification measure (inverse of concentration)
            let mut herfindahl = 0.0;
            let weight = 1.0 / n_features as Float;
            for _ in 0..n_features {
                herfindahl += weight.powi(2);
            }
            let diversification = 1.0 - herfindahl;
            if feat_idx < feature_dim {
                portfolio_features[[i, feat_idx]] = diversification;
                feat_idx += 1;
            }

            // Correlation-based features (simplified)
            for j in 0..10.min(n_features) {
                if j + 1 < n_features && feat_idx < feature_dim {
                    let corr_approx = x[[i, j]] * x[[i, j + 1]];
                    portfolio_features[[i, feat_idx]] = corr_approx;
                    feat_idx += 1;
                }
            }
        }

        // Apply random projection
        let features = portfolio_features.dot(projection);

        Ok(features)
    }
}

// ============================================================================
// Risk Kernel
// ============================================================================

/// Kernel method for financial risk analysis
///
/// This kernel computes risk-based features including Value-at-Risk (VaR),
/// Conditional VaR (CVaR), and risk contribution measures.
///
/// # References
/// - Jorion (2006): "Value at Risk: The New Benchmark for Managing Financial Risk"
pub struct RiskKernel<State = Untrained> {
    /// Number of random features
    n_components: usize,
    /// Confidence level for VaR (e.g., 0.95 for 95% VaR)
    confidence_level: Float,
    /// Whether to include tail risk measures
    include_tail_risk: bool,
    /// Random projection matrix
    projection: Option<Array2<Float>>,
    /// State marker
    _state: PhantomData<State>,
}

impl RiskKernel<Untrained> {
    /// Create a new risk kernel
    pub fn new(n_components: usize, confidence_level: Float) -> Self {
        Self {
            n_components,
            confidence_level,
            include_tail_risk: true,
            projection: None,
            _state: PhantomData,
        }
    }

    /// Set whether to include tail risk measures
    pub fn include_tail_risk(mut self, include_tail_risk: bool) -> Self {
        self.include_tail_risk = include_tail_risk;
        self
    }
}

impl Default for RiskKernel<Untrained> {
    fn default() -> Self {
        Self::new(100, 0.95)
    }
}

impl Fit<Array2<Float>, ()> for RiskKernel<Untrained> {
    type Fitted = RiskKernel<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array cannot be empty".to_string(),
            ));
        }

        // Risk features dimension
        let base_dim = 15;
        let tail_dim = if self.include_tail_risk { 10 } else { 0 };
        let feature_dim = base_dim + tail_dim;

        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0 / (feature_dim as Float).sqrt()).unwrap();

        // Generate random projection
        let mut projection = Array2::zeros((feature_dim, self.n_components));
        for i in 0..feature_dim {
            for j in 0..self.n_components {
                projection[[i, j]] = normal.sample(&mut rng);
            }
        }

        Ok(RiskKernel {
            n_components: self.n_components,
            confidence_level: self.confidence_level,
            include_tail_risk: self.include_tail_risk,
            projection: Some(projection),
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for RiskKernel<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array cannot be empty".to_string(),
            ));
        }

        let projection = self.projection.as_ref().unwrap();
        let feature_dim = projection.nrows();

        // Extract risk features
        let mut risk_features = Array2::zeros((n_samples, feature_dim));

        for i in 0..n_samples {
            let mut feat_idx = 0;

            // VaR approximation using parametric method
            let mut mean = 0.0;
            let mut variance = 0.0;
            for j in 0..n_features {
                mean += x[[i, j]];
            }
            mean /= n_features as Float;

            for j in 0..n_features {
                variance += (x[[i, j]] - mean).powi(2);
            }
            variance /= n_features as Float;
            let std_dev = variance.sqrt();

            // Normal VaR (parametric)
            let z_score = 1.645; // Approximate z-score for 95% confidence
            let var = mean - z_score * std_dev;
            if feat_idx < feature_dim {
                risk_features[[i, feat_idx]] = var;
                feat_idx += 1;
            }

            // CVaR approximation (expected shortfall)
            let cvar = mean - (2.5 * std_dev); // Simplified CVaR
            if feat_idx < feature_dim {
                risk_features[[i, feat_idx]] = cvar;
                feat_idx += 1;
            }

            // Downside deviation
            let mut downside_var = 0.0;
            let mut downside_count = 0;
            for j in 0..n_features {
                if x[[i, j]] < mean {
                    downside_var += (x[[i, j]] - mean).powi(2);
                    downside_count += 1;
                }
            }
            if downside_count > 0 {
                let downside_dev = (downside_var / downside_count as Float).sqrt();
                if feat_idx < feature_dim {
                    risk_features[[i, feat_idx]] = downside_dev;
                    feat_idx += 1;
                }
            }

            // Maximum drawdown approximation
            let mut max_dd = 0.0;
            let mut peak = x[[i, 0]];
            for j in 0..n_features {
                if x[[i, j]] > peak {
                    peak = x[[i, j]];
                }
                let dd = (peak - x[[i, j]]) / (peak.abs() + 1e-8);
                if dd > max_dd {
                    max_dd = dd;
                }
            }
            if feat_idx < feature_dim {
                risk_features[[i, feat_idx]] = max_dd;
                feat_idx += 1;
            }

            // Tail risk measures
            if self.include_tail_risk {
                // Skewness approximation
                let mut skew = 0.0;
                for j in 0..n_features {
                    skew += ((x[[i, j]] - mean) / (std_dev + 1e-8)).powi(3);
                }
                skew /= n_features as Float;
                if feat_idx < feature_dim {
                    risk_features[[i, feat_idx]] = skew;
                    feat_idx += 1;
                }

                // Kurtosis approximation
                let mut kurt = 0.0;
                for j in 0..n_features {
                    kurt += ((x[[i, j]] - mean) / (std_dev + 1e-8)).powi(4);
                }
                kurt /= n_features as Float;
                if feat_idx < feature_dim {
                    risk_features[[i, feat_idx]] = kurt - 3.0; // Excess kurtosis
                }
            }
        }

        // Apply random projection
        let features = risk_features.dot(projection);

        Ok(features)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_financial_kernel_basic() {
        let x = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];

        let kernel = FinancialKernel::new(50, 10);
        let fitted = kernel.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[2, 50]);
    }

    #[test]
    fn test_financial_kernel_volatility() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let kernel = FinancialKernel::new(40, 5).include_volatility(true);
        let fitted = kernel.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[2, 40]);
    }

    #[test]
    fn test_volatility_kernel_ewma() {
        let x = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];

        let kernel = VolatilityKernel::new(50, VolatilityModel::EWMA).decay_factor(0.94);
        let fitted = kernel.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[2, 50]);
    }

    #[test]
    fn test_volatility_models() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let models = vec![
            VolatilityModel::Historical,
            VolatilityModel::EWMA,
            VolatilityModel::GARCH,
            VolatilityModel::Realized,
        ];

        for model in models {
            let kernel = VolatilityKernel::new(30, model);
            let fitted = kernel.fit(&x, &()).unwrap();
            let features = fitted.transform(&x).unwrap();
            assert_eq!(features.shape(), &[2, 30]);
        }
    }

    #[test]
    fn test_econometric_kernel_basic() {
        let x = array![[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]];

        let kernel = EconometricKernel::new(50, 3);
        let fitted = kernel.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[2, 50]);
    }

    #[test]
    fn test_econometric_kernel_differences() {
        let x = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];

        let kernel = EconometricKernel::new(40, 2).include_differences(true);
        let fitted = kernel.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[2, 40]);
    }

    #[test]
    fn test_portfolio_kernel_basic() {
        let x = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];

        let kernel = PortfolioKernel::new(50, 4);
        let fitted = kernel.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[2, 50]);
    }

    #[test]
    fn test_portfolio_kernel_risk_free_rate() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let kernel = PortfolioKernel::new(40, 3).risk_free_rate(0.03);
        let fitted = kernel.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[2, 40]);
    }

    #[test]
    fn test_risk_kernel_basic() {
        let x = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];

        let kernel = RiskKernel::new(50, 0.95);
        let fitted = kernel.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[2, 50]);
    }

    #[test]
    fn test_risk_kernel_tail_risk() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let kernel = RiskKernel::new(40, 0.99).include_tail_risk(true);
        let fitted = kernel.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[2, 40]);
    }

    #[test]
    fn test_empty_input_error() {
        let x_empty: Array2<Float> = Array2::zeros((0, 3));

        let kernel = FinancialKernel::new(50, 10);
        assert!(kernel.fit(&x_empty, &()).is_err());

        let kernel2 = VolatilityKernel::new(50, VolatilityModel::EWMA);
        assert!(kernel2.fit(&x_empty, &()).is_err());
    }
}
