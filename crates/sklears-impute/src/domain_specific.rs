//! Domain-specific imputation methods
//!
//! This module provides specialized imputation methods for specific domains
//! such as bioinformatics, finance, and social sciences.

pub mod bioinformatics;
pub mod finance;
pub mod social_science;

// Re-export commonly used types
pub use bioinformatics::{
    GenomicImputer, MetabolomicsImputer, PhylogeneticImputer, ProteinExpressionImputer,
    SingleCellRNASeqImputer,
};
pub use finance::{
    CreditScoringImputer, EconomicIndicatorImputer, FinancialTimeSeriesImputer,
    PortfolioDataImputer, RiskFactorImputer,
};
pub use social_science::{
    DemographicDataImputer, LongitudinalStudyImputer, MissingResponseHandler, SocialNetworkImputer,
    SurveyDataImputer,
};
