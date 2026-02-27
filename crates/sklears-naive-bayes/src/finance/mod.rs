//! Finance and Economics Naive Bayes Implementations
//!
//! This module provides specialized Naive Bayes implementations for finance and economics applications,
//! including financial time series classification, risk assessment, portfolio classification, credit scoring,
//! and fraud detection.

use scirs2_core::numeric::Float;
// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2};

// Type aliases for compatibility with DMatrix/DVector usage
pub(crate) type DMatrix<T> = Array2<T>;
pub(crate) type DVector<T> = Array1<T>;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum FinanceError {
    #[error("Invalid time series data: {0}")]
    InvalidTimeSeries(String),
    #[error("Risk assessment error: {0}")]
    RiskAssessment(String),
    #[error("Portfolio classification error: {0}")]
    PortfolioClassification(String),
    #[error("Credit scoring error: {0}")]
    CreditScoring(String),
    #[error("Fraud detection error: {0}")]
    FraudDetection(String),
    #[error("Mathematical computation error: {0}")]
    MathError(String),
}

// Module declarations
mod timeseries;
mod risk;
mod portfolio;
mod credit;
mod fraud;

// Re-exports
pub use timeseries::{
    FinancialTimeSeriesNB, FeatureStatistics, PriceStatistics, VolumeStatistics,
    VolatilityStatistics, TechnicalStatistics, TechnicalIndicators, VolatilityModels,
    GarchParams, FinancialFeatures,
};

pub use risk::{RiskAssessmentNB, RiskLevel, RiskFeatureStats, RiskAssessmentParams};

pub use portfolio::{
    PortfolioClassificationNB, PortfolioCategory, PortfolioFeatureStats,
    PortfolioClassificationParams, PortfolioData,
};

pub use credit::{CreditScoringNB, CreditRisk, CreditFeatureStats, CreditScoringParams, CreditData};

pub use fraud::{
    FraudDetectionNB, FraudLabel, FraudFeatureStats, FraudDetectionParams, TransactionData,
};
