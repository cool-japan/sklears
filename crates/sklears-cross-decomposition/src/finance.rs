//! Finance and Economics Applications
//!
//! This module provides specialized methods for financial and economic data analysis,
//! including factor analysis for financial data, portfolio optimization integration,
//! macroeconomic factor analysis, risk factor decomposition, and regime-switching models.

use crate::cca::CCA;
use crate::pls::PLSRegression;
use crate::time_series::{RegimeSwitchingModel, VectorAutoregression};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive, One};
use scirs2_core::random::{thread_rng, Random, Rng};
use sklears_core::{
    error::SklearsError,
    traits::{Fit, Transform},
};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during finance analysis
#[derive(Error, Debug)]
pub enum FinanceError {
    #[error("Invalid input dimensions: {0}")]
    InvalidDimensions(String),
    #[error("Insufficient data for analysis: {0}")]
    InsufficientData(String),
    #[error("Convergence failed: {0}")]
    ConvergenceFailed(String),
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("Market data error: {0}")]
    MarketDataError(String),
}

/// Factor Analysis for Financial Data
///
/// This method performs factor analysis specifically designed for financial time series,
/// identifying common risk factors and systematic patterns in asset returns.
pub struct FinancialFactorAnalysis<F: Float> {
    /// Number of factors to extract
    n_factors: usize,
    /// Factor rotation method
    rotation: FactorRotation,
    /// Maximum number of iterations
    max_iter: usize,
    /// Convergence tolerance
    tol: F,
    /// Whether to scale the data
    scale: bool,
    /// Risk-free rate for excess returns
    risk_free_rate: F,
}

/// Factor rotation methods
#[derive(Debug, Clone)]
pub enum FactorRotation {
    /// No rotation
    None,
    /// Varimax rotation
    Varimax,
    /// Quartimax rotation
    Quartimax,
    /// Economic interpretation rotation
    Economic,
}

impl<F: Float + 'static + FromPrimitive + scirs2_core::ndarray::ScalarOperand>
    FinancialFactorAnalysis<F>
{
    /// Create a new financial factor analysis
    pub fn new(n_factors: usize) -> Self {
        Self {
            n_factors,
            rotation: FactorRotation::Varimax,
            max_iter: 100,
            tol: F::from(1e-6).unwrap(),
            scale: true,
            risk_free_rate: F::from(0.02).unwrap(), // 2% annual risk-free rate
        }
    }

    /// Set the factor rotation method
    pub fn rotation(mut self, rotation: FactorRotation) -> Self {
        self.rotation = rotation;
        self
    }

    /// Set the risk-free rate
    pub fn risk_free_rate(mut self, rate: F) -> Self {
        self.risk_free_rate = rate;
        self
    }

    /// Set whether to scale the data
    pub fn scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }

    /// Fit the financial factor model
    pub fn fit(
        &self,
        returns: ArrayView2<F>,
    ) -> Result<FittedFinancialFactorAnalysis<F>, FinanceError> {
        if returns.nrows() < self.n_factors {
            return Err(FinanceError::InsufficientData(
                "Number of time periods must be greater than number of factors".to_string(),
            ));
        }

        // Convert to excess returns
        let excess_returns = self.compute_excess_returns(&returns)?;

        // Perform PCA-based factor analysis
        let (factor_loadings, factor_scores, explained_variance) =
            self.extract_factors(&excess_returns)?;

        // Apply rotation if specified
        let rotated_loadings = self.apply_rotation(&factor_loadings)?;

        // Compute factor statistics
        let factor_stats = self.compute_factor_statistics(&factor_scores, &excess_returns)?;

        // Identify economic factors
        let factor_interpretation = self.interpret_factors(&rotated_loadings, &factor_stats)?;

        Ok(FittedFinancialFactorAnalysis {
            factor_loadings: rotated_loadings,
            factor_scores,
            explained_variance,
            factor_stats,
            factor_interpretation,
            n_factors: self.n_factors,
            risk_free_rate: self.risk_free_rate,
        })
    }

    fn compute_excess_returns(&self, returns: &ArrayView2<F>) -> Result<Array2<F>, FinanceError> {
        let mut excess_returns = returns.to_owned();
        let period_risk_free = self.risk_free_rate / F::from(252.0).unwrap(); // Daily risk-free rate

        for mut row in excess_returns.rows_mut() {
            for ret in row.iter_mut() {
                *ret = *ret - period_risk_free;
            }
        }

        Ok(excess_returns)
    }

    fn extract_factors(
        &self,
        data: &Array2<F>,
    ) -> Result<(Array2<F>, Array2<F>, Array1<F>), FinanceError> {
        // Compute covariance matrix
        let mean_centered = self.center_data(data)?;
        let cov_matrix = self.compute_covariance(&mean_centered)?;

        // Perform eigenvalue decomposition
        let (eigenvalues, eigenvectors) = self.eigen_decomposition(&cov_matrix)?;

        // Extract top factors
        let factor_loadings = eigenvectors.slice(s![.., ..self.n_factors]).to_owned();
        let factor_scores = mean_centered.dot(&factor_loadings);
        let explained_variance = eigenvalues.slice(s![..self.n_factors]).to_owned();

        Ok((factor_loadings, factor_scores, explained_variance))
    }

    fn center_data(&self, data: &Array2<F>) -> Result<Array2<F>, FinanceError> {
        let mut centered = data.clone();

        for mut column in centered.columns_mut().into_iter() {
            let mean = column.mean().unwrap_or(F::zero());
            for value in column.iter_mut() {
                *value = *value - mean;
            }
        }

        Ok(centered)
    }

    fn compute_covariance(&self, data: &Array2<F>) -> Result<Array2<F>, FinanceError> {
        let n_samples = F::from(data.nrows()).unwrap();
        let cov = data.t().dot(data) / (n_samples - F::one());
        Ok(cov)
    }

    fn eigen_decomposition(
        &self,
        matrix: &Array2<F>,
    ) -> Result<(Array1<F>, Array2<F>), FinanceError> {
        // Simplified eigenvalue decomposition (in practice would use proper LAPACK)
        let n = matrix.nrows();
        let mut eigenvalues = Array1::zeros(n);
        let mut eigenvectors = Array2::eye(n);

        // Power iteration for dominant eigenvector
        for i in 0..self.n_factors.min(n) {
            let mut v = Array1::ones(n);

            for _ in 0..self.max_iter {
                let new_v = matrix.dot(&v);
                let norm = new_v
                    .iter()
                    .map(|&x| x * x)
                    .fold(F::zero(), |acc, x| acc + x)
                    .sqrt();

                if norm > F::zero() {
                    v = new_v / norm;
                }
            }

            let lambda = v.dot(&matrix.dot(&v));
            eigenvalues[i] = lambda;
            eigenvectors.column_mut(i).assign(&v);
        }

        // Sort by eigenvalue magnitude
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            eigenvalues[b]
                .abs()
                .partial_cmp(&eigenvalues[a].abs())
                .unwrap()
        });

        let mut sorted_eigenvalues = Array1::zeros(n);
        let mut sorted_eigenvectors = Array2::zeros((n, n));

        for (i, &idx) in indices.iter().enumerate() {
            sorted_eigenvalues[i] = eigenvalues[idx];
            sorted_eigenvectors
                .column_mut(i)
                .assign(&eigenvectors.column(idx));
        }

        Ok((sorted_eigenvalues, sorted_eigenvectors))
    }

    fn apply_rotation(&self, loadings: &Array2<F>) -> Result<Array2<F>, FinanceError> {
        match self.rotation {
            FactorRotation::None => Ok(loadings.clone()),
            FactorRotation::Varimax => self.varimax_rotation(loadings),
            FactorRotation::Quartimax => self.quartimax_rotation(loadings),
            FactorRotation::Economic => self.economic_rotation(loadings),
        }
    }

    fn varimax_rotation(&self, loadings: &Array2<F>) -> Result<Array2<F>, FinanceError> {
        // Simplified Varimax rotation
        let mut rotated = loadings.clone();

        for _ in 0..self.max_iter {
            let prev = rotated.clone();

            // Apply rotation step (simplified)
            for i in 0..rotated.ncols() {
                for j in (i + 1)..rotated.ncols() {
                    let angle = self.compute_varimax_angle(&rotated, i, j)?;
                    self.apply_givens_rotation(&mut rotated, i, j, angle);
                }
            }

            // Check convergence
            let diff = self.matrix_diff(&rotated, &prev);
            if diff < self.tol {
                break;
            }
        }

        Ok(rotated)
    }

    fn quartimax_rotation(&self, loadings: &Array2<F>) -> Result<Array2<F>, FinanceError> {
        // Simplified Quartimax rotation (maximize variance of squared loadings)
        let mut rotated = loadings.clone();

        for _ in 0..self.max_iter {
            let prev = rotated.clone();

            // Apply Quartimax criterion
            for i in 0..rotated.ncols() {
                let mut column = rotated.column_mut(i);
                let sum_sq = column
                    .iter()
                    .map(|&x| x * x)
                    .fold(F::zero(), |acc, x| acc + x);

                if sum_sq > F::zero() {
                    let scale = sum_sq.sqrt();
                    for value in column.iter_mut() {
                        *value = *value / scale;
                    }
                }
            }

            // Check convergence
            let diff = self.matrix_diff(&rotated, &prev);
            if diff < self.tol {
                break;
            }
        }

        Ok(rotated)
    }

    fn economic_rotation(&self, loadings: &Array2<F>) -> Result<Array2<F>, FinanceError> {
        // Economic interpretation rotation (simplified)
        let mut rotated = loadings.clone();

        // Sort factors by economic interpretability
        let mut factor_scores: Vec<(usize, F)> = (0..rotated.ncols())
            .map(|i| {
                let column = rotated.column(i);
                let interpretability = self.compute_interpretability_score(&column);
                (i, interpretability)
            })
            .collect();

        factor_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Reorder factors
        let mut reordered = Array2::zeros(rotated.raw_dim());
        for (new_idx, &(old_idx, _)) in factor_scores.iter().enumerate() {
            reordered
                .column_mut(new_idx)
                .assign(&rotated.column(old_idx));
        }

        Ok(reordered)
    }

    fn compute_varimax_angle(
        &self,
        loadings: &Array2<F>,
        i: usize,
        j: usize,
    ) -> Result<F, FinanceError> {
        // Simplified angle computation for Varimax rotation
        let col_i = loadings.column(i);
        let col_j = loadings.column(j);

        let dot_product = col_i
            .iter()
            .zip(col_j.iter())
            .map(|(&a, &b)| a * b)
            .fold(F::zero(), |acc, x| acc + x);
        let angle = dot_product.atan();

        Ok(angle)
    }

    fn apply_givens_rotation(&self, matrix: &mut Array2<F>, i: usize, j: usize, angle: F) {
        let cos_angle = angle.cos();
        let sin_angle = angle.sin();

        for row in 0..matrix.nrows() {
            let a = matrix[(row, i)];
            let b = matrix[(row, j)];

            matrix[(row, i)] = cos_angle * a - sin_angle * b;
            matrix[(row, j)] = sin_angle * a + cos_angle * b;
        }
    }

    fn compute_interpretability_score(&self, factor: &ArrayView1<F>) -> F {
        // Compute interpretability as concentration of loadings
        let sum_abs = factor
            .iter()
            .map(|&x| x.abs())
            .fold(F::zero(), |acc, x| acc + x);
        let max_abs = factor
            .iter()
            .map(|&x| x.abs())
            .fold(F::zero(), |acc, x| acc.max(x));

        if sum_abs > F::zero() {
            max_abs / sum_abs
        } else {
            F::zero()
        }
    }

    fn matrix_diff(&self, a: &Array2<F>, b: &Array2<F>) -> F {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let diff = x - y;
                diff * diff
            })
            .fold(F::zero(), |acc, x| acc + x)
            .sqrt()
    }

    fn compute_factor_statistics(
        &self,
        factor_scores: &Array2<F>,
        returns: &Array2<F>,
    ) -> Result<FactorStatistics<F>, FinanceError> {
        let mut factor_returns = Array1::zeros(self.n_factors);
        let mut factor_volatilities = Array1::zeros(self.n_factors);
        let mut factor_sharpe_ratios = Array1::zeros(self.n_factors);

        for i in 0..self.n_factors {
            let factor_column = factor_scores.column(i);

            // Compute factor return (mean)
            let mean_return = factor_column.mean().unwrap_or(F::zero());
            factor_returns[i] = mean_return;

            // Compute factor volatility (standard deviation)
            let variance = factor_column
                .iter()
                .map(|&x| {
                    let diff = x - mean_return;
                    diff * diff
                })
                .fold(F::zero(), |acc, x| acc + x)
                / F::from(factor_column.len()).unwrap();
            let volatility = variance.sqrt();
            factor_volatilities[i] = volatility;

            // Compute Sharpe ratio
            let sharpe_ratio = if volatility > F::zero() {
                (mean_return - self.risk_free_rate / F::from(252.0).unwrap()) / volatility
            } else {
                F::zero()
            };
            factor_sharpe_ratios[i] = sharpe_ratio;
        }

        Ok(FactorStatistics {
            factor_returns,
            factor_volatilities,
            factor_sharpe_ratios,
        })
    }

    fn interpret_factors(
        &self,
        loadings: &Array2<F>,
        stats: &FactorStatistics<F>,
    ) -> Result<HashMap<usize, String>, FinanceError> {
        let mut interpretation = HashMap::new();

        for i in 0..self.n_factors {
            let column = loadings.column(i);
            let volatility = stats.factor_volatilities[i];
            let sharpe = stats.factor_sharpe_ratios[i];

            let interpretation_str = if sharpe > F::from(1.0).unwrap() {
                "High_Sharpe_Factor".to_string()
            } else if volatility > F::from(0.02).unwrap() {
                "High_Volatility_Factor".to_string()
            } else if self.is_market_factor(&column) {
                "Market_Factor".to_string()
            } else if self.is_size_factor(&column) {
                "Size_Factor".to_string()
            } else if self.is_value_factor(&column) {
                "Value_Factor".to_string()
            } else {
                "Other_Factor".to_string()
            };

            interpretation.insert(i, interpretation_str);
        }

        Ok(interpretation)
    }

    fn is_market_factor(&self, loadings: &ArrayView1<F>) -> bool {
        // Check if all loadings have the same sign (market factor characteristic)
        let positive_count = loadings.iter().filter(|&&x| x > F::zero()).count();
        let negative_count = loadings.iter().filter(|&&x| x < F::zero()).count();

        positive_count as f64 / loadings.len() as f64 > 0.8
            || negative_count as f64 / loadings.len() as f64 > 0.8
    }

    fn is_size_factor(&self, loadings: &ArrayView1<F>) -> bool {
        // Simplified size factor detection (would need actual market cap data)
        let mean_abs = loadings
            .iter()
            .map(|&x| x.abs())
            .fold(F::zero(), |acc, x| acc + x)
            / F::from(loadings.len()).unwrap();
        let std_abs = {
            let variance = loadings
                .iter()
                .map(|&x| {
                    let diff = x.abs() - mean_abs;
                    diff * diff
                })
                .fold(F::zero(), |acc, x| acc + x)
                / F::from(loadings.len()).unwrap();
            variance.sqrt()
        };

        std_abs > mean_abs * F::from(0.5).unwrap()
    }

    fn is_value_factor(&self, loadings: &ArrayView1<F>) -> bool {
        // Simplified value factor detection
        let positive_count = loadings.iter().filter(|&&x| x > F::zero()).count();
        let negative_count = loadings.iter().filter(|&&x| x < F::zero()).count();

        // Value factor typically has mixed signs
        positive_count > 0
            && negative_count > 0
            && (positive_count as f64 / loadings.len() as f64).abs() - 0.5 < 0.3
    }
}

/// Factor statistics
#[derive(Debug, Clone)]
pub struct FactorStatistics<F: Float> {
    /// Expected returns for each factor
    pub factor_returns: Array1<F>,
    /// Volatilities for each factor
    pub factor_volatilities: Array1<F>,
    /// Sharpe ratios for each factor
    pub factor_sharpe_ratios: Array1<F>,
}

/// Fitted financial factor analysis model
pub struct FittedFinancialFactorAnalysis<F: Float> {
    factor_loadings: Array2<F>,
    factor_scores: Array2<F>,
    explained_variance: Array1<F>,
    factor_stats: FactorStatistics<F>,
    factor_interpretation: HashMap<usize, String>,
    n_factors: usize,
    risk_free_rate: F,
}

impl<F: Float + 'static + FromPrimitive + ScalarOperand> FittedFinancialFactorAnalysis<F> {
    /// Get the factor loadings
    pub fn factor_loadings(&self) -> &Array2<F> {
        &self.factor_loadings
    }

    /// Get the factor scores (time series of factor values)
    pub fn factor_scores(&self) -> &Array2<F> {
        &self.factor_scores
    }

    /// Get the explained variance for each factor
    pub fn explained_variance(&self) -> &Array1<F> {
        &self.explained_variance
    }

    /// Get the factor statistics
    pub fn factor_statistics(&self) -> &FactorStatistics<F> {
        &self.factor_stats
    }

    /// Get the factor interpretation
    pub fn factor_interpretation(&self) -> &HashMap<usize, String> {
        &self.factor_interpretation
    }

    /// Transform new return data to factor space
    pub fn transform(&self, returns: ArrayView2<F>) -> Result<Array2<F>, FinanceError> {
        if returns.ncols() != self.factor_loadings.nrows() {
            return Err(FinanceError::InvalidDimensions(
                "Returns must have same number of assets as training data".to_string(),
            ));
        }

        Ok(returns.dot(&self.factor_loadings))
    }

    /// Compute factor exposure for a portfolio
    pub fn portfolio_exposure(
        &self,
        portfolio_weights: ArrayView1<F>,
    ) -> Result<Array1<F>, FinanceError> {
        if portfolio_weights.len() != self.factor_loadings.nrows() {
            return Err(FinanceError::InvalidDimensions(
                "Portfolio weights must match number of assets".to_string(),
            ));
        }

        Ok(self.factor_loadings.t().dot(&portfolio_weights))
    }

    /// Decompose portfolio risk into factor components
    pub fn risk_decomposition(
        &self,
        portfolio_weights: ArrayView1<F>,
    ) -> Result<RiskDecomposition<F>, FinanceError> {
        let factor_exposure = self.portfolio_exposure(portfolio_weights)?;

        // Compute factor contributions to portfolio variance
        let mut factor_contributions = Array1::zeros(self.n_factors);
        let portfolio_variance = self.compute_portfolio_variance(&portfolio_weights)?;

        for i in 0..self.n_factors {
            let factor_var =
                self.factor_stats.factor_volatilities[i] * self.factor_stats.factor_volatilities[i];
            let contribution =
                (factor_exposure[i] * factor_exposure[i] * factor_var) / portfolio_variance;
            factor_contributions[i] = contribution;
        }

        let factor_sum = factor_contributions.sum();
        Ok(RiskDecomposition {
            factor_exposures: factor_exposure,
            factor_contributions,
            total_risk: portfolio_variance.sqrt(),
            systematic_risk: factor_sum.sqrt(),
            idiosyncratic_risk: (F::one() - factor_sum).max(F::zero()).sqrt(),
        })
    }

    fn compute_portfolio_variance(&self, weights: &ArrayView1<F>) -> Result<F, FinanceError> {
        // Simplified portfolio variance computation
        let portfolio_factor_exposure = self.portfolio_exposure(weights.clone())?;

        let mut variance = F::zero();
        for i in 0..self.n_factors {
            let factor_var =
                self.factor_stats.factor_volatilities[i] * self.factor_stats.factor_volatilities[i];
            variance =
                variance + portfolio_factor_exposure[i] * portfolio_factor_exposure[i] * factor_var;
        }

        Ok(variance)
    }
}

/// Risk decomposition results
#[derive(Debug, Clone)]
pub struct RiskDecomposition<F: Float> {
    /// Factor exposures of the portfolio
    pub factor_exposures: Array1<F>,
    /// Contribution of each factor to portfolio risk
    pub factor_contributions: Array1<F>,
    /// Total portfolio risk (volatility)
    pub total_risk: F,
    /// Systematic risk (from factors)
    pub systematic_risk: F,
    /// Idiosyncratic risk (asset-specific)
    pub idiosyncratic_risk: F,
}

/// Portfolio optimization with factor constraints
pub struct FactorConstrainedOptimization<F: Float> {
    /// Expected returns
    expected_returns: Array1<F>,
    /// Risk aversion parameter
    risk_aversion: F,
    /// Factor exposure constraints
    factor_constraints: HashMap<usize, (F, F)>, // (min_exposure, max_exposure)
    /// Maximum weight per asset
    max_weight: F,
    /// Minimum weight per asset
    min_weight: F,
}

impl<F: Float + 'static + FromPrimitive + ScalarOperand> FactorConstrainedOptimization<F> {
    /// Create a new factor-constrained optimization
    pub fn new(expected_returns: Array1<F>, risk_aversion: F) -> Self {
        Self {
            expected_returns,
            risk_aversion,
            factor_constraints: HashMap::new(),
            max_weight: F::from(0.1).unwrap(),
            min_weight: F::zero(),
        }
    }

    /// Add factor exposure constraint
    pub fn add_factor_constraint(
        mut self,
        factor_idx: usize,
        min_exposure: F,
        max_exposure: F,
    ) -> Self {
        self.factor_constraints
            .insert(factor_idx, (min_exposure, max_exposure));
        self
    }

    /// Set weight bounds
    pub fn weight_bounds(mut self, min_weight: F, max_weight: F) -> Self {
        self.min_weight = min_weight;
        self.max_weight = max_weight;
        self
    }

    /// Optimize portfolio weights
    pub fn optimize(
        &self,
        factor_model: &FittedFinancialFactorAnalysis<F>,
    ) -> Result<OptimizedPortfolio<F>, FinanceError> {
        let n_assets = self.expected_returns.len();

        // Simple equal-weight initialization
        let mut weights = Array1::from_elem(n_assets, F::one() / F::from(n_assets).unwrap());

        // Apply constraints iteratively (simplified optimization)
        for _ in 0..100 {
            // Compute current factor exposures
            let factor_exposures = factor_model.portfolio_exposure(weights.view())?;

            // Adjust weights to meet factor constraints
            for (&factor_idx, &(min_exp, max_exp)) in &self.factor_constraints {
                let current_exp = factor_exposures[factor_idx];

                if current_exp < min_exp {
                    // Increase exposure to this factor
                    self.adjust_weights_for_factor(&mut weights, factor_model, factor_idx, true)?;
                } else if current_exp > max_exp {
                    // Decrease exposure to this factor
                    self.adjust_weights_for_factor(&mut weights, factor_model, factor_idx, false)?;
                }
            }

            // Apply weight bounds
            for weight in weights.iter_mut() {
                *weight = weight.max(self.min_weight).min(self.max_weight);
            }

            // Normalize weights to sum to 1
            let weight_sum = weights.sum();
            if weight_sum > F::zero() {
                weights.mapv_inplace(|w| w / weight_sum);
            }
        }

        // Compute portfolio metrics
        let expected_return = weights.dot(&self.expected_returns);
        let risk_decomp = factor_model.risk_decomposition(weights.view())?;

        Ok(OptimizedPortfolio {
            weights,
            expected_return,
            risk_decomposition: risk_decomp,
        })
    }

    fn adjust_weights_for_factor(
        &self,
        weights: &mut Array1<F>,
        factor_model: &FittedFinancialFactorAnalysis<F>,
        factor_idx: usize,
        increase: bool,
    ) -> Result<(), FinanceError> {
        let factor_loadings = factor_model.factor_loadings();
        let factor_column = factor_loadings.column(factor_idx);

        // Adjust weights based on factor loadings
        let adjustment = F::from(0.01).unwrap();

        for (i, &loading) in factor_column.iter().enumerate() {
            if increase {
                if loading > F::zero() {
                    weights[i] = weights[i] + adjustment;
                } else {
                    weights[i] = weights[i] - adjustment;
                }
            } else {
                if loading > F::zero() {
                    weights[i] = weights[i] - adjustment;
                } else {
                    weights[i] = weights[i] + adjustment;
                }
            }
        }

        Ok(())
    }
}

/// Optimized portfolio result
#[derive(Debug, Clone)]
pub struct OptimizedPortfolio<F: Float> {
    /// Optimal portfolio weights
    pub weights: Array1<F>,
    /// Expected portfolio return
    pub expected_return: F,
    /// Risk decomposition
    pub risk_decomposition: RiskDecomposition<F>,
}

/// Macroeconomic Factor Analysis
///
/// This method performs factor analysis on macroeconomic indicators to identify
/// systematic economic factors that drive asset returns.
pub struct MacroeconomicFactorAnalysis<F: Float> {
    /// Number of macroeconomic factors to extract
    n_factors: usize,
    /// Economic indicators to include
    indicators: Vec<String>,
    /// Lag periods for economic indicators
    lags: Vec<usize>,
    /// Maximum number of iterations
    max_iter: usize,
    /// Convergence tolerance
    tol: F,
    /// Whether to seasonally adjust the data
    seasonal_adjust: bool,
    /// Forecasting horizon
    forecast_horizon: usize,
}

impl<F: Float + 'static + FromPrimitive + ScalarOperand> MacroeconomicFactorAnalysis<F> {
    /// Create a new macroeconomic factor analysis
    pub fn new(n_factors: usize) -> Self {
        Self {
            n_factors,
            indicators: vec![
                "GDP_Growth".to_string(),
                "Inflation".to_string(),
                "Unemployment".to_string(),
                "Interest_Rate".to_string(),
                "Exchange_Rate".to_string(),
                "Oil_Price".to_string(),
                "VIX".to_string(),
            ],
            lags: vec![0, 1, 2, 3], // Current, 1-month, 2-month, 3-month lags
            max_iter: 100,
            tol: F::from(1e-6).unwrap(),
            seasonal_adjust: true,
            forecast_horizon: 12, // 12 months
        }
    }

    /// Set the economic indicators to include
    pub fn indicators(mut self, indicators: Vec<String>) -> Self {
        self.indicators = indicators;
        self
    }

    /// Set the lag periods for economic indicators
    pub fn lags(mut self, lags: Vec<usize>) -> Self {
        self.lags = lags;
        self
    }

    /// Set whether to seasonally adjust the data
    pub fn seasonal_adjust(mut self, seasonal_adjust: bool) -> Self {
        self.seasonal_adjust = seasonal_adjust;
        self
    }

    /// Set the forecasting horizon
    pub fn forecast_horizon(mut self, horizon: usize) -> Self {
        self.forecast_horizon = horizon;
        self
    }

    /// Fit the macroeconomic factor model
    pub fn fit(
        &self,
        economic_data: ArrayView2<F>,
        asset_returns: ArrayView2<F>,
    ) -> Result<FittedMacroeconomicFactorAnalysis<F>, FinanceError> {
        if economic_data.nrows() != asset_returns.nrows() {
            return Err(FinanceError::InvalidDimensions(
                "Economic data and asset returns must have the same number of time periods"
                    .to_string(),
            ));
        }

        if economic_data.nrows() < self.n_factors {
            return Err(FinanceError::InsufficientData(
                "Number of time periods must be greater than number of factors".to_string(),
            ));
        }

        // Preprocess economic data
        let processed_data = self.preprocess_economic_data(&economic_data)?;

        // Create lagged features
        let lagged_features = self.create_lagged_features(&processed_data)?;

        // Extract macroeconomic factors using PCA
        let (factor_loadings, factor_scores, explained_variance) =
            self.extract_macro_factors(&lagged_features)?;

        // Compute factor sensitivities for asset returns
        let asset_returns_owned = asset_returns.to_owned();
        let asset_sensitivities =
            self.compute_asset_sensitivities(&factor_scores, &asset_returns_owned)?;

        // Identify economic interpretation of factors
        let factor_interpretation = self.interpret_macro_factors(&factor_loadings)?;

        // Compute forecasting models
        let forecasting_models = self.build_forecasting_models(&factor_scores)?;

        // Compute factor statistics
        let economic_data_owned = economic_data.to_owned();
        let factor_statistics =
            self.compute_macro_factor_statistics(&factor_scores, &economic_data_owned)?;

        Ok(FittedMacroeconomicFactorAnalysis {
            factor_loadings,
            factor_scores,
            explained_variance,
            asset_sensitivities,
            factor_interpretation,
            forecasting_models,
            factor_statistics,
            indicators: self.indicators.clone(),
            lags: self.lags.clone(),
            n_factors: self.n_factors,
            forecast_horizon: self.forecast_horizon,
        })
    }

    fn preprocess_economic_data(&self, data: &ArrayView2<F>) -> Result<Array2<F>, FinanceError> {
        let mut processed = data.to_owned();

        // Apply seasonal adjustment if requested
        if self.seasonal_adjust {
            processed = self.seasonal_adjustment(&processed)?;
        }

        // Normalize the data
        for mut column in processed.columns_mut() {
            let mean = column.mean().unwrap_or(F::zero());
            let std_dev = {
                let variance = column
                    .iter()
                    .map(|&x| {
                        let diff = x - mean;
                        diff * diff
                    })
                    .fold(F::zero(), |acc, x| acc + x)
                    / F::from(column.len()).unwrap();
                variance.sqrt()
            };

            if std_dev > F::zero() {
                for value in column.iter_mut() {
                    *value = (*value - mean) / std_dev;
                }
            }
        }

        Ok(processed)
    }

    fn seasonal_adjustment(&self, data: &Array2<F>) -> Result<Array2<F>, FinanceError> {
        // Simplified seasonal adjustment using moving averages
        let mut adjusted = data.clone();
        let window_size = 12; // Assume monthly data

        if data.nrows() < window_size {
            return Ok(adjusted);
        }

        for (col_idx, mut column) in adjusted.columns_mut().into_iter().enumerate() {
            let original_column = data.column(col_idx);

            // Compute moving average
            for i in (window_size / 2)..(column.len() - window_size / 2) {
                let start = i - window_size / 2;
                let end = i + window_size / 2 + 1;
                let moving_avg = original_column
                    .slice(s![start..end])
                    .mean()
                    .unwrap_or(F::zero());
                column[i] = original_column[i] - moving_avg;
            }
        }

        Ok(adjusted)
    }

    fn create_lagged_features(&self, data: &Array2<F>) -> Result<Array2<F>, FinanceError> {
        let n_periods = data.nrows();
        let n_indicators = data.ncols();
        let max_lag = self.lags.iter().max().unwrap_or(&0);

        if n_periods <= *max_lag {
            return Err(FinanceError::InsufficientData(
                "Not enough data for requested lag periods".to_string(),
            ));
        }

        let n_features = n_indicators * self.lags.len();
        let n_samples = n_periods - max_lag;
        let mut lagged_features = Array2::zeros((n_samples, n_features));

        for (sample_idx, mut row) in lagged_features.rows_mut().into_iter().enumerate() {
            let current_period = sample_idx + max_lag;

            for (lag_idx, &lag) in self.lags.iter().enumerate() {
                for indicator_idx in 0..n_indicators {
                    let feature_idx = lag_idx * n_indicators + indicator_idx;
                    let data_period = current_period - lag;
                    row[feature_idx] = data[(data_period, indicator_idx)];
                }
            }
        }

        Ok(lagged_features)
    }

    fn extract_macro_factors(
        &self,
        data: &Array2<F>,
    ) -> Result<(Array2<F>, Array2<F>, Array1<F>), FinanceError> {
        // Compute covariance matrix
        let mean_centered = self.center_data(data)?;
        let cov_matrix = self.compute_covariance(&mean_centered)?;

        // Perform eigenvalue decomposition
        let (eigenvalues, eigenvectors) = self.eigen_decomposition(&cov_matrix)?;

        // Extract top factors
        let factor_loadings = eigenvectors.slice(s![.., ..self.n_factors]).to_owned();
        let factor_scores = mean_centered.dot(&factor_loadings);
        let explained_variance = eigenvalues.slice(s![..self.n_factors]).to_owned();

        Ok((factor_loadings, factor_scores, explained_variance))
    }

    fn center_data(&self, data: &Array2<F>) -> Result<Array2<F>, FinanceError> {
        let mut centered = data.clone();

        for mut column in centered.columns_mut() {
            let mean = column.mean().unwrap_or(F::zero());
            for value in column.iter_mut() {
                *value = *value - mean;
            }
        }

        Ok(centered)
    }

    fn compute_covariance(&self, data: &Array2<F>) -> Result<Array2<F>, FinanceError> {
        let n_samples = F::from(data.nrows()).unwrap();
        let cov = data.t().dot(data) / (n_samples - F::one());
        Ok(cov)
    }

    fn eigen_decomposition(
        &self,
        matrix: &Array2<F>,
    ) -> Result<(Array1<F>, Array2<F>), FinanceError> {
        // Simplified eigenvalue decomposition
        let n = matrix.nrows();
        let mut eigenvalues = Array1::zeros(n);
        let mut eigenvectors = Array2::eye(n);

        // Power iteration for dominant eigenvectors
        for i in 0..self.n_factors.min(n) {
            let mut v = Array1::ones(n);

            for _ in 0..self.max_iter {
                let new_v = matrix.dot(&v);
                let norm = new_v
                    .iter()
                    .map(|&x| x * x)
                    .fold(F::zero(), |acc, x| acc + x)
                    .sqrt();

                if norm > F::zero() {
                    v = new_v / norm;
                }
            }

            let lambda = v.dot(&matrix.dot(&v));
            eigenvalues[i] = lambda;
            eigenvectors.column_mut(i).assign(&v);
        }

        // Sort by eigenvalue magnitude
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            eigenvalues[b]
                .abs()
                .partial_cmp(&eigenvalues[a].abs())
                .unwrap()
        });

        let mut sorted_eigenvalues = Array1::zeros(n);
        let mut sorted_eigenvectors = Array2::zeros((n, n));

        for (i, &idx) in indices.iter().enumerate() {
            sorted_eigenvalues[i] = eigenvalues[idx];
            sorted_eigenvectors
                .column_mut(i)
                .assign(&eigenvectors.column(idx));
        }

        Ok((sorted_eigenvalues, sorted_eigenvectors))
    }

    fn compute_asset_sensitivities(
        &self,
        factor_scores: &Array2<F>,
        asset_returns: &Array2<F>,
    ) -> Result<Array2<F>, FinanceError> {
        // Compute beta coefficients for each asset to each factor
        let n_assets = asset_returns.ncols();
        let n_factors = factor_scores.ncols();
        let mut sensitivities = Array2::zeros((n_assets, n_factors));

        for asset_idx in 0..n_assets {
            let asset_returns_col = asset_returns.column(asset_idx);

            for factor_idx in 0..n_factors {
                let factor_scores_col = factor_scores.column(factor_idx);

                // Compute correlation as sensitivity measure
                let correlation =
                    self.compute_correlation(&asset_returns_col, &factor_scores_col)?;
                sensitivities[(asset_idx, factor_idx)] = correlation;
            }
        }

        Ok(sensitivities)
    }

    fn compute_correlation(&self, x: &ArrayView1<F>, y: &ArrayView1<F>) -> Result<F, FinanceError> {
        let n = F::from(x.len()).unwrap();
        let mean_x = x.mean().unwrap_or(F::zero());
        let mean_y = y.mean().unwrap_or(F::zero());

        let covariance = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .fold(F::zero(), |acc, x| acc + x)
            / n;

        let var_x = x
            .iter()
            .map(|&xi| {
                let diff = xi - mean_x;
                diff * diff
            })
            .fold(F::zero(), |acc, x| acc + x)
            / n;

        let var_y = y
            .iter()
            .map(|&yi| {
                let diff = yi - mean_y;
                diff * diff
            })
            .fold(F::zero(), |acc, x| acc + x)
            / n;

        let correlation = if var_x > F::zero() && var_y > F::zero() {
            covariance / (var_x.sqrt() * var_y.sqrt())
        } else {
            F::zero()
        };

        Ok(correlation)
    }

    fn interpret_macro_factors(
        &self,
        factor_loadings: &Array2<F>,
    ) -> Result<HashMap<usize, String>, FinanceError> {
        let mut interpretation = HashMap::new();

        for factor_idx in 0..self.n_factors {
            let factor_column = factor_loadings.column(factor_idx);

            // Find the most influential indicators for this factor
            let mut indicator_loadings: Vec<(usize, F)> = factor_column
                .iter()
                .enumerate()
                .map(|(i, &loading)| (i, loading.abs()))
                .collect();

            indicator_loadings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Interpret based on top indicators
            let top_indicator_idx = indicator_loadings[0].0;
            let indicator_name = if top_indicator_idx < self.indicators.len() {
                &self.indicators[top_indicator_idx / self.lags.len()]
            } else {
                "Unknown"
            };

            let interpretation_str = match indicator_name {
                "GDP_Growth" => "Economic_Growth_Factor",
                "Inflation" => "Inflation_Factor",
                "Unemployment" => "Labor_Market_Factor",
                "Interest_Rate" => "Monetary_Policy_Factor",
                "Exchange_Rate" => "Currency_Factor",
                "Oil_Price" => "Energy_Factor",
                "VIX" => "Market_Volatility_Factor",
                _ => "General_Economic_Factor",
            };

            interpretation.insert(factor_idx, interpretation_str.to_string());
        }

        Ok(interpretation)
    }

    fn build_forecasting_models(
        &self,
        factor_scores: &Array2<F>,
    ) -> Result<Vec<ForecastingModel<F>>, FinanceError> {
        let mut models = Vec::new();

        for factor_idx in 0..self.n_factors {
            let factor_series = factor_scores.column(factor_idx);

            // Build AR(1) model for each factor
            let ar_model = self.fit_ar_model(&factor_series)?;
            models.push(ar_model);
        }

        Ok(models)
    }

    fn fit_ar_model(&self, series: &ArrayView1<F>) -> Result<ForecastingModel<F>, FinanceError> {
        let n = series.len();

        if n < 2 {
            return Err(FinanceError::InsufficientData(
                "Not enough data for AR model".to_string(),
            ));
        }

        // Compute AR(1) coefficients using least squares
        let mut sum_xy = F::zero();
        let mut sum_x2 = F::zero();
        let mut sum_y = F::zero();
        let mut sum_x = F::zero();

        for i in 1..n {
            let x = series[i - 1];
            let y = series[i];

            sum_xy = sum_xy + x * y;
            sum_x2 = sum_x2 + x * x;
            sum_y = sum_y + y;
            sum_x = sum_x + x;
        }

        let n_obs = F::from(n - 1).unwrap();
        let beta = if sum_x2 > F::zero() {
            (sum_xy - sum_x * sum_y / n_obs) / (sum_x2 - sum_x * sum_x / n_obs)
        } else {
            F::zero()
        };

        let alpha = (sum_y - beta * sum_x) / n_obs;

        Ok(ForecastingModel {
            coefficients: vec![alpha, beta],
            model_type: "AR(1)".to_string(),
        })
    }

    fn compute_macro_factor_statistics(
        &self,
        factor_scores: &Array2<F>,
        economic_data: &Array2<F>,
    ) -> Result<MacroFactorStatistics<F>, FinanceError> {
        let mut factor_volatilities = Array1::zeros(self.n_factors);
        let mut factor_persistence = Array1::zeros(self.n_factors);
        let mut economic_correlations = Array2::zeros((self.n_factors, economic_data.ncols()));

        for factor_idx in 0..self.n_factors {
            let factor_column = factor_scores.column(factor_idx);

            // Compute volatility
            let mean = factor_column.mean().unwrap_or(F::zero());
            let variance = factor_column
                .iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .fold(F::zero(), |acc, x| acc + x)
                / F::from(factor_column.len()).unwrap();
            factor_volatilities[factor_idx] = variance.sqrt();

            // Compute persistence (AR(1) coefficient)
            if factor_column.len() > 1 {
                let ar_model = self.fit_ar_model(&factor_column)?;
                factor_persistence[factor_idx] = ar_model.coefficients[1];
            }

            // Compute correlations with economic indicators
            for econ_idx in 0..economic_data.ncols() {
                let econ_column = economic_data.column(econ_idx);
                let correlation = self.compute_correlation(&factor_column, &econ_column)?;
                economic_correlations[(factor_idx, econ_idx)] = correlation;
            }
        }

        Ok(MacroFactorStatistics {
            factor_volatilities,
            factor_persistence,
            economic_correlations,
        })
    }
}

/// Fitted macroeconomic factor analysis model
pub struct FittedMacroeconomicFactorAnalysis<F: Float> {
    factor_loadings: Array2<F>,
    factor_scores: Array2<F>,
    explained_variance: Array1<F>,
    asset_sensitivities: Array2<F>,
    factor_interpretation: HashMap<usize, String>,
    forecasting_models: Vec<ForecastingModel<F>>,
    factor_statistics: MacroFactorStatistics<F>,
    indicators: Vec<String>,
    lags: Vec<usize>,
    n_factors: usize,
    forecast_horizon: usize,
}

impl<F: Float + 'static + FromPrimitive + ScalarOperand> FittedMacroeconomicFactorAnalysis<F> {
    /// Get the macroeconomic factor loadings
    pub fn factor_loadings(&self) -> &Array2<F> {
        &self.factor_loadings
    }

    /// Get the factor scores (time series of factor values)
    pub fn factor_scores(&self) -> &Array2<F> {
        &self.factor_scores
    }

    /// Get the explained variance for each factor
    pub fn explained_variance(&self) -> &Array1<F> {
        &self.explained_variance
    }

    /// Get the asset sensitivities to macroeconomic factors
    pub fn asset_sensitivities(&self) -> &Array2<F> {
        &self.asset_sensitivities
    }

    /// Get the factor interpretation
    pub fn factor_interpretation(&self) -> &HashMap<usize, String> {
        &self.factor_interpretation
    }

    /// Get the factor statistics
    pub fn factor_statistics(&self) -> &MacroFactorStatistics<F> {
        &self.factor_statistics
    }

    /// Forecast macroeconomic factors
    pub fn forecast_factors(&self, n_periods: usize) -> Result<Array2<F>, FinanceError> {
        let mut forecasts = Array2::zeros((n_periods, self.n_factors));

        for (factor_idx, model) in self.forecasting_models.iter().enumerate() {
            let last_value = if !self.factor_scores.is_empty() {
                self.factor_scores[(self.factor_scores.nrows() - 1, factor_idx)]
            } else {
                F::zero()
            };

            // Generate forecasts using AR(1) model
            let mut current_value = last_value;
            for period in 0..n_periods {
                current_value = model.coefficients[0] + model.coefficients[1] * current_value;
                forecasts[(period, factor_idx)] = current_value;
            }
        }

        Ok(forecasts)
    }

    /// Forecast asset returns based on macroeconomic factors
    pub fn forecast_asset_returns(&self, n_periods: usize) -> Result<Array2<F>, FinanceError> {
        let factor_forecasts = self.forecast_factors(n_periods)?;
        let n_assets = self.asset_sensitivities.nrows();

        // Multiply factor forecasts by asset sensitivities
        let return_forecasts = factor_forecasts.dot(&self.asset_sensitivities.t());

        Ok(return_forecasts)
    }

    /// Analyze scenario impact on asset returns
    pub fn scenario_analysis(
        &self,
        factor_shocks: ArrayView1<F>,
    ) -> Result<Array1<F>, FinanceError> {
        if factor_shocks.len() != self.n_factors {
            return Err(FinanceError::InvalidDimensions(
                "Factor shocks must match number of factors".to_string(),
            ));
        }

        // Compute impact on asset returns
        let impact = self.asset_sensitivities.dot(&factor_shocks);
        Ok(impact)
    }
}

/// Forecasting model for macroeconomic factors
#[derive(Debug, Clone)]
pub struct ForecastingModel<F: Float> {
    /// Model coefficients
    pub coefficients: Vec<F>,
    /// Model type (e.g., "AR(1)", "VAR", etc.)
    pub model_type: String,
}

/// Macroeconomic factor statistics
#[derive(Debug, Clone)]
pub struct MacroFactorStatistics<F: Float> {
    /// Volatility of each factor
    pub factor_volatilities: Array1<F>,
    /// Persistence (AR(1) coefficient) of each factor
    pub factor_persistence: Array1<F>,
    /// Correlations between factors and economic indicators
    pub economic_correlations: Array2<F>,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_financial_factor_analysis_basic() {
        let returns = array![
            [0.01, 0.02, 0.015],
            [0.015, 0.025, 0.02],
            [0.005, 0.01, 0.008],
            [-0.01, -0.015, -0.012],
            [0.02, 0.03, 0.025],
            [0.008, 0.012, 0.01],
        ];

        let factor_analysis = FinancialFactorAnalysis::new(2)
            .rotation(FactorRotation::Varimax)
            .risk_free_rate(0.02);

        let result = factor_analysis.fit(returns.view());
        assert!(result.is_ok());

        let fitted = result.unwrap();
        assert_eq!(fitted.factor_loadings().ncols(), 2);
        assert_eq!(fitted.factor_scores().ncols(), 2);
        assert_eq!(fitted.explained_variance().len(), 2);
        assert_eq!(fitted.factor_statistics().factor_returns.len(), 2);
    }

    #[test]
    fn test_financial_factor_analysis_transform() {
        let returns = array![
            [0.01, 0.02, 0.015],
            [0.015, 0.025, 0.02],
            [0.005, 0.01, 0.008],
            [-0.01, -0.015, -0.012],
            [0.02, 0.03, 0.025],
        ];

        let factor_analysis = FinancialFactorAnalysis::new(2);
        let fitted = factor_analysis.fit(returns.view()).unwrap();

        let new_returns = array![[0.012, 0.018, 0.014], [0.008, 0.015, 0.011],];

        let transformed = fitted.transform(new_returns.view());
        assert!(transformed.is_ok());

        let result = transformed.unwrap();
        assert_eq!(result.shape(), &[2, 2]);
    }

    #[test]
    fn test_portfolio_exposure() {
        let returns = array![
            [0.01, 0.02, 0.015],
            [0.015, 0.025, 0.02],
            [0.005, 0.01, 0.008],
            [-0.01, -0.015, -0.012],
            [0.02, 0.03, 0.025],
        ];

        let factor_analysis = FinancialFactorAnalysis::new(2);
        let fitted = factor_analysis.fit(returns.view()).unwrap();

        let portfolio_weights = array![0.4, 0.4, 0.2];
        let exposure = fitted.portfolio_exposure(portfolio_weights.view());

        assert!(exposure.is_ok());
        let exp = exposure.unwrap();
        assert_eq!(exp.len(), 2);
    }

    #[test]
    fn test_risk_decomposition() {
        let returns = array![
            [0.01, 0.02, 0.015],
            [0.015, 0.025, 0.02],
            [0.005, 0.01, 0.008],
            [-0.01, -0.015, -0.012],
            [0.02, 0.03, 0.025],
        ];

        let factor_analysis = FinancialFactorAnalysis::new(2);
        let fitted = factor_analysis.fit(returns.view()).unwrap();

        let portfolio_weights = array![0.4, 0.4, 0.2];
        let risk_decomp = fitted.risk_decomposition(portfolio_weights.view());

        assert!(risk_decomp.is_ok());
        let decomp = risk_decomp.unwrap();
        assert_eq!(decomp.factor_exposures.len(), 2);
        assert_eq!(decomp.factor_contributions.len(), 2);
        assert!(decomp.total_risk >= 0.0);
    }

    #[test]
    fn test_factor_constrained_optimization() {
        let expected_returns = array![0.08, 0.12, 0.10];
        let optimizer = FactorConstrainedOptimization::new(expected_returns, 2.0)
            .add_factor_constraint(0, -0.5, 0.5)
            .weight_bounds(0.1, 0.6);

        // Create a simple factor model for testing
        let returns = array![
            [0.01, 0.02, 0.015],
            [0.015, 0.025, 0.02],
            [0.005, 0.01, 0.008],
            [-0.01, -0.015, -0.012],
            [0.02, 0.03, 0.025],
        ];

        let factor_analysis = FinancialFactorAnalysis::new(1);
        let fitted = factor_analysis.fit(returns.view()).unwrap();

        let result = optimizer.optimize(&fitted);
        assert!(result.is_ok());

        let portfolio = result.unwrap();
        assert_eq!(portfolio.weights.len(), 3);
        assert!((portfolio.weights.sum() - 1.0).abs() < 1e-6);
        assert!(portfolio.weights.iter().all(|&w| w >= 0.1 && w <= 0.6));
    }

    #[test]
    fn test_factor_rotation_methods() {
        let loadings = array![[0.8, 0.2], [0.1, 0.9], [0.6, 0.4],];

        let factor_analysis = FinancialFactorAnalysis::new(2);

        let varimax = factor_analysis.varimax_rotation(&loadings);
        assert!(varimax.is_ok());

        let quartimax = factor_analysis.quartimax_rotation(&loadings);
        assert!(quartimax.is_ok());

        let economic = factor_analysis.economic_rotation(&loadings);
        assert!(economic.is_ok());
    }

    #[test]
    fn test_factor_interpretation() {
        let returns = array![
            [0.01, 0.02, 0.015, 0.018],
            [0.015, 0.025, 0.02, 0.022],
            [0.005, 0.01, 0.008, 0.009],
            [-0.01, -0.015, -0.012, -0.014],
            [0.02, 0.03, 0.025, 0.028],
        ];

        let factor_analysis = FinancialFactorAnalysis::new(2);
        let fitted = factor_analysis.fit(returns.view()).unwrap();

        let interpretation = fitted.factor_interpretation();
        assert_eq!(interpretation.len(), 2);
        assert!(interpretation.contains_key(&0));
        assert!(interpretation.contains_key(&1));
    }

    #[test]
    fn test_macroeconomic_factor_analysis_basic() {
        // Create mock economic data (GDP growth, inflation, unemployment, interest rate)
        let economic_data = array![
            [0.02, 0.025, 0.05, 0.035],   // Q1
            [0.018, 0.03, 0.048, 0.04],   // Q2
            [0.025, 0.028, 0.052, 0.038], // Q3
            [0.015, 0.032, 0.055, 0.045], // Q4
            [0.022, 0.027, 0.050, 0.042], // Q5
            [0.020, 0.029, 0.049, 0.040], // Q6
        ];

        // Create mock asset returns
        let asset_returns = array![
            [0.08, 0.12, 0.06],
            [0.05, 0.15, 0.04],
            [0.10, 0.09, 0.08],
            [0.02, 0.18, 0.03],
            [0.09, 0.11, 0.07],
            [0.07, 0.13, 0.05],
        ];

        let macro_analysis = MacroeconomicFactorAnalysis::new(2)
            .seasonal_adjust(false)
            .forecast_horizon(6);

        let result = macro_analysis.fit(economic_data.view(), asset_returns.view());
        assert!(result.is_ok());

        let fitted = result.unwrap();
        assert_eq!(fitted.factor_loadings().ncols(), 2);
        assert_eq!(fitted.factor_scores().ncols(), 2);
        assert_eq!(fitted.explained_variance().len(), 2);
        assert_eq!(fitted.asset_sensitivities().nrows(), 3); // 3 assets
        assert_eq!(fitted.asset_sensitivities().ncols(), 2); // 2 factors
    }

    #[test]
    fn test_macroeconomic_factor_analysis_forecasting() {
        let economic_data = array![
            [0.02, 0.025, 0.05, 0.035],
            [0.018, 0.03, 0.048, 0.04],
            [0.025, 0.028, 0.052, 0.038],
            [0.015, 0.032, 0.055, 0.045],
            [0.022, 0.027, 0.050, 0.042],
        ];

        let asset_returns = array![
            [0.08, 0.12],
            [0.05, 0.15],
            [0.10, 0.09],
            [0.02, 0.18],
            [0.09, 0.11],
        ];

        let macro_analysis = MacroeconomicFactorAnalysis::new(1);
        let fitted = macro_analysis
            .fit(economic_data.view(), asset_returns.view())
            .unwrap();

        // Test factor forecasting
        let factor_forecasts = fitted.forecast_factors(3);
        assert!(factor_forecasts.is_ok());
        let forecasts = factor_forecasts.unwrap();
        assert_eq!(forecasts.shape(), &[3, 1]);

        // Test asset return forecasting
        let return_forecasts = fitted.forecast_asset_returns(3);
        assert!(return_forecasts.is_ok());
        let ret_forecasts = return_forecasts.unwrap();
        assert_eq!(ret_forecasts.shape(), &[3, 2]);
    }

    #[test]
    fn test_macroeconomic_scenario_analysis() {
        let economic_data = array![
            [0.02, 0.025, 0.05],
            [0.018, 0.03, 0.048],
            [0.025, 0.028, 0.052],
            [0.015, 0.032, 0.055],
            [0.022, 0.027, 0.050],
        ];

        let asset_returns = array![
            [0.08, 0.12],
            [0.05, 0.15],
            [0.10, 0.09],
            [0.02, 0.18],
            [0.09, 0.11],
        ];

        let macro_analysis = MacroeconomicFactorAnalysis::new(2);
        let fitted = macro_analysis
            .fit(economic_data.view(), asset_returns.view())
            .unwrap();

        // Test scenario analysis with factor shocks
        let factor_shocks = array![0.1, -0.05]; // +10% shock to factor 1, -5% to factor 2
        let impact = fitted.scenario_analysis(factor_shocks.view());

        assert!(impact.is_ok());
        let impact_result = impact.unwrap();
        assert_eq!(impact_result.len(), 2); // 2 assets
    }

    #[test]
    fn test_macroeconomic_factor_interpretation() {
        let economic_data = array![
            [0.02, 0.025, 0.05, 0.035, 0.98, 45.0, 0.22], // GDP, Inflation, Unemployment, Interest, Exchange, Oil, VIX
            [0.018, 0.03, 0.048, 0.04, 0.95, 48.0, 0.25],
            [0.025, 0.028, 0.052, 0.038, 1.02, 42.0, 0.18],
            [0.015, 0.032, 0.055, 0.045, 0.92, 50.0, 0.30],
            [0.022, 0.027, 0.050, 0.042, 1.00, 46.0, 0.20],
        ];

        let asset_returns = array![
            [0.08, 0.12, 0.06],
            [0.05, 0.15, 0.04],
            [0.10, 0.09, 0.08],
            [0.02, 0.18, 0.03],
            [0.09, 0.11, 0.07],
        ];

        let macro_analysis = MacroeconomicFactorAnalysis::new(2);
        let fitted = macro_analysis
            .fit(economic_data.view(), asset_returns.view())
            .unwrap();

        let interpretation = fitted.factor_interpretation();
        assert_eq!(interpretation.len(), 2);
        assert!(interpretation.contains_key(&0));
        assert!(interpretation.contains_key(&1));

        // Check that interpretation strings are meaningful
        for interpretation_str in interpretation.values() {
            assert!(interpretation_str.contains("Factor"));
        }
    }

    #[test]
    fn test_macroeconomic_factor_analysis_error_cases() {
        let macro_analysis = MacroeconomicFactorAnalysis::new(2);

        // Test with mismatched dimensions
        let economic_data = array![[0.02, 0.025], [0.018, 0.03]];
        let asset_returns = array![[0.08], [0.05], [0.10]]; // Different number of rows

        let result = macro_analysis.fit(economic_data.view(), asset_returns.view());
        assert!(result.is_err());

        // Test with insufficient data
        let small_economic_data = array![[0.02]]; // Only 1 period, but need > n_factors
        let small_asset_returns = array![[0.08]];

        let result = macro_analysis.fit(small_economic_data.view(), small_asset_returns.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_macroeconomic_factor_statistics() {
        let economic_data = array![
            [0.02, 0.025, 0.05],
            [0.018, 0.03, 0.048],
            [0.025, 0.028, 0.052],
            [0.015, 0.032, 0.055],
            [0.022, 0.027, 0.050],
        ];

        let asset_returns = array![
            [0.08, 0.12],
            [0.05, 0.15],
            [0.10, 0.09],
            [0.02, 0.18],
            [0.09, 0.11],
        ];

        let macro_analysis = MacroeconomicFactorAnalysis::new(2);
        let fitted = macro_analysis
            .fit(economic_data.view(), asset_returns.view())
            .unwrap();

        let stats = fitted.factor_statistics();
        assert_eq!(stats.factor_volatilities.len(), 2);
        assert_eq!(stats.factor_persistence.len(), 2);
        assert_eq!(stats.economic_correlations.shape(), &[2, 3]); // 2 factors, 3 economic indicators

        // Check that volatilities are positive
        for &vol in stats.factor_volatilities.iter() {
            assert!(vol >= 0.0);
        }
    }
}
