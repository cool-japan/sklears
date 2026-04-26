//! Finance and Economics Naive Bayes Implementations
//!
//! This module provides specialized Naive Bayes implementations for finance and economics applications,
//! including financial time series classification, risk assessment, portfolio classification, credit scoring,
//! and fraud detection.

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
mod credit;
mod fraud;
mod portfolio;
mod risk;
mod timeseries;

// Re-exports
pub use timeseries::{
    FeatureStatistics, FinancialFeatures, FinancialTimeSeriesNB, GarchParams, PriceStatistics,
    TechnicalIndicators, TechnicalStatistics, VolatilityModels, VolatilityStatistics,
    VolumeStatistics,
};

pub use risk::{RiskAssessmentNB, RiskAssessmentParams, RiskFeatureStats, RiskLevel};

pub use portfolio::{
    PortfolioCategory, PortfolioClassificationNB, PortfolioClassificationParams, PortfolioData,
    PortfolioFeatureStats,
};

pub use credit::{
    CreditData, CreditFeatureStats, CreditRisk, CreditScoringNB, CreditScoringParams,
};

pub use fraud::{
    FraudDetectionNB, FraudDetectionParams, FraudFeatureStats, FraudLabel, TransactionData,
};
