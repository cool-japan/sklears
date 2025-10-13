//! Domain-specific feature selection modules.
//!
//! This module provides specialized feature selection capabilities for various domains,
//! each optimized for specific data types and application areas. The modules are organized
//! by domain expertise and include sophisticated algorithms tailored to their respective
//! data characteristics and analytical requirements.
//!
//! # Available Domains
//!
//! ## Time Series Analysis
//! - **Module**: [`time_series`]
//! - **Selector**: [`TimeSeriesSelector`](time_series::TimeSeriesSelector)
//! - **Focus**: Temporal patterns, autocorrelation, seasonality, trend analysis
//! - **Applications**: Financial time series, sensor data, economic indicators
//!
//! ## Text and Document Analysis
//! - **Module**: [`text_features`]
//! - **Selector**: [`TextFeatureSelector`](text_features::TextFeatureSelector)
//! - **Focus**: TF-IDF, document frequency, chi-squared analysis, linguistic features
//! - **Applications**: Document classification, sentiment analysis, information retrieval
//!
//! ## Image and Visual Data
//! - **Module**: [`image_features`]
//! - **Selector**: [`ImageFeatureSelector`](image_features::ImageFeatureSelector)
//! - **Focus**: Spatial correlation, frequency domain, texture analysis, computer vision
//! - **Applications**: Medical imaging, pattern recognition, visual classification
//!
//! ## Graph and Network Data
//! - **Module**: [`graph_features`]
//! - **Selector**: [`GraphFeatureSelector`](graph_features::GraphFeatureSelector)
//! - **Focus**: Centrality measures, community detection, structural properties
//! - **Applications**: Social networks, biological networks, knowledge graphs
//!
//! ## Multi-Modal Data
//! - **Module**: [`multi_modal`]
//! - **Selector**: [`MultiModalFeatureSelector`](multi_modal::MultiModalFeatureSelector)
//! - **Focus**: Cross-modal analysis, fusion strategies, heterogeneous data integration
//! - **Applications**: Multimedia analysis, sensor fusion, multimodal machine learning
//!
//! ## Bioinformatics and Genomics
//! - **Module**: [`bioinformatics`]
//! - **Selector**: [`BioinformaticsFeatureSelector`](bioinformatics::BioinformaticsFeatureSelector)
//! - **Focus**: Gene expression, SNP analysis, pathway enrichment, multiple testing correction
//! - **Applications**: Genomics, proteomics, drug discovery, personalized medicine
//!
//! ## Finance and Trading
//! - **Module**: [`finance`]
//! - **Selector**: [`FinanceFeatureSelector`](finance::FinanceFeatureSelector)
//! - **Focus**: Technical indicators, risk metrics, market microstructure, regime detection
//! - **Applications**: Algorithmic trading, risk management, portfolio optimization
//!
//! ## Advanced NLP
//! - **Module**: [`advanced_nlp`]
//! - **Selector**: [`AdvancedNLPFeatureSelector`](advanced_nlp::AdvancedNLPFeatureSelector)
//! - **Focus**: Syntactic parsing, semantic analysis, discourse features, transformer models
//! - **Applications**: Language understanding, text generation, machine translation
//!
//! # Usage Patterns
//!
//! ## Domain-Specific Selection
//!
//! Each domain provides specialized feature selection methods optimized for its data type:
//!
//! ```rust,ignore
//! use sklears_feature_selection::domain_specific::time_series::TimeSeriesSelector;
//! use sklears_feature_selection::domain_specific::finance::FinanceFeatureSelector;
//! use sklears_feature_selection::domain_specific::bioinformatics::BioinformaticsFeatureSelector;
//!
//! // Time series feature selection
//! let ts_selector = TimeSeriesSelector::builder()
//!     .max_lag(20)
//!     .include_seasonal(true)
//!     .build();
//!
//! // Financial feature selection
//! let finance_selector = FinanceFeatureSelector::builder()
//!     .feature_type("technical_indicators")
//!     .risk_adjusted_scoring(true)
//!     .build();
//!
//! // Bioinformatics feature selection
//! let bio_selector = BioinformaticsFeatureSelector::builder()
//!     .data_type("gene_expression")
//!     .multiple_testing_correction("fdr")
//!     .build();
//! ```
//!
//! ## Cross-Domain Integration
//!
//! The multi-modal selector can integrate features from multiple domains:
//!
//! ```rust,ignore
//! use sklears_feature_selection::domain_specific::multi_modal::MultiModalFeatureSelector;
//! use std::collections::HashMap;
//! use scirs2_core::ndarray::Array2;
//!
//! let mut modalities = HashMap::new();
//! modalities.insert("text".to_string(), text_features);
//! modalities.insert("image".to_string(), image_features);
//! modalities.insert("time_series".to_string(), ts_features);
//!
//! let selector = MultiModalFeatureSelector::builder()
//!     .fusion_strategy("hybrid")
//!     .cross_modal_analysis(true)
//!     .build();
//! ```
//!
//! # Design Principles
//!
//! ## Domain Expertise
//! Each module incorporates domain-specific knowledge and best practices:
//! - Statistical methods appropriate for the data type
//! - Domain-specific feature engineering techniques
//! - Specialized evaluation metrics and validation approaches
//! - Integration with domain standards and conventions
//!
//! ## Algorithmic Sophistication
//! The selectors implement advanced algorithms beyond basic statistical tests:
//! - Multi-scale analysis for temporal and spatial data
//! - Graph-theoretic methods for network data
//! - Information-theoretic approaches for complex dependencies
//! - Machine learning-based feature importance estimation
//!
//! ## Flexibility and Extensibility
//! The design supports various usage patterns:
//! - Builder pattern for complex configuration
//! - Multiple feature selection strategies per domain
//! - Configurable parameters for different use cases
//! - Extensible architecture for custom domain adaptations
//!
//! ## Performance and Scalability
//! Implementations are optimized for practical use:
//! - Efficient algorithms for large-scale data
//! - Parallel processing where applicable
//! - Memory-efficient operations for high-dimensional data
//! - Incremental processing capabilities
//!
//! # Integration with Core Framework
//!
//! All domain-specific selectors implement the core traits:
//! - [`Estimator`](crate::core::Estimator) for consistent configuration
//! - [`Fit`](crate::core::Fit) and [`Transform`](crate::core::Transform) for sklearn-like API
//! - [`SelectorMixin`](crate::core::SelectorMixin) for feature selection functionality
//!
//! This ensures compatibility with the broader sklears ecosystem while providing
//! specialized capabilities for domain-specific applications.

// Domain-specific feature selection modules
pub mod advanced_nlp;
pub mod bioinformatics;
pub mod finance;
pub mod graph_features;
pub mod image_features;
pub mod multi_modal;
pub mod text_features;
pub mod time_series;

// Re-export main selectors for convenience
pub use advanced_nlp::AdvancedNLPFeatureSelector;
pub use bioinformatics::BioinformaticsFeatureSelector;
pub use finance::FinanceFeatureSelector;
pub use graph_features::GraphFeatureSelector;
pub use image_features::ImageFeatureSelector;
pub use multi_modal::MultiModalFeatureSelector;
pub use text_features::TextFeatureSelector;
pub use time_series::TimeSeriesSelector;

/// Domain-specific feature selection capabilities.
///
/// This enum provides a unified interface for accessing different domain-specific
/// feature selectors. It enables runtime selection of the appropriate domain
/// expert based on data characteristics or user requirements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Domain {
    /// Time series and temporal data analysis
    TimeSeries,
    /// Text and document analysis
    Text,
    /// Image and visual data processing
    Image,
    /// Graph and network data analysis
    Graph,
    /// Multi-modal and heterogeneous data integration
    MultiModal,
    /// Bioinformatics and genomic data analysis
    Bioinformatics,
    /// Finance and trading data analysis
    Finance,
    /// Advanced natural language processing
    AdvancedNLP,
}

impl Domain {
    /// Returns a human-readable description of the domain.
    pub fn description(&self) -> &'static str {
        match self {
            Domain::TimeSeries => "Time series and temporal pattern analysis",
            Domain::Text => "Text mining and document analysis",
            Domain::Image => "Image processing and computer vision",
            Domain::Graph => "Graph theory and network analysis",
            Domain::MultiModal => "Multi-modal data fusion and integration",
            Domain::Bioinformatics => "Bioinformatics and genomic data analysis",
            Domain::Finance => "Financial modeling and trading analytics",
            Domain::AdvancedNLP => "Advanced natural language processing",
        }
    }

    /// Returns the recommended use cases for this domain.
    pub fn use_cases(&self) -> Vec<&'static str> {
        match self {
            Domain::TimeSeries => vec![
                "Financial time series forecasting",
                "Sensor data analysis",
                "Economic indicator modeling",
                "Climate and weather data",
                "IoT and monitoring systems",
            ],
            Domain::Text => vec![
                "Document classification",
                "Sentiment analysis",
                "Information retrieval",
                "Content recommendation",
                "Text summarization",
            ],
            Domain::Image => vec![
                "Medical image analysis",
                "Object recognition",
                "Quality control inspection",
                "Satellite imagery processing",
                "Facial recognition systems",
            ],
            Domain::Graph => vec![
                "Social network analysis",
                "Biological network modeling",
                "Knowledge graph construction",
                "Citation analysis",
                "Infrastructure network optimization",
            ],
            Domain::MultiModal => vec![
                "Multimedia content analysis",
                "Sensor fusion applications",
                "Cross-modal retrieval",
                "Multimodal machine learning",
                "Human-computer interaction",
            ],
            Domain::Bioinformatics => vec![
                "Gene expression analysis",
                "Genome-wide association studies",
                "Drug discovery",
                "Personalized medicine",
                "Protein function prediction",
            ],
            Domain::Finance => vec![
                "Algorithmic trading",
                "Risk management",
                "Portfolio optimization",
                "Fraud detection",
                "Credit scoring",
            ],
            Domain::AdvancedNLP => vec![
                "Machine translation",
                "Question answering systems",
                "Text generation",
                "Language understanding",
                "Conversational AI",
            ],
        }
    }

    /// Returns the key techniques used in this domain.
    pub fn key_techniques(&self) -> Vec<&'static str> {
        match self {
            Domain::TimeSeries => vec![
                "Autocorrelation analysis",
                "Seasonal decomposition",
                "Cross-correlation",
                "Spectral analysis",
                "Stationarity testing",
            ],
            Domain::Text => vec![
                "TF-IDF weighting",
                "Chi-squared testing",
                "Document frequency analysis",
                "N-gram extraction",
                "Feature hashing",
            ],
            Domain::Image => vec![
                "Spatial correlation analysis",
                "Frequency domain transforms",
                "Texture analysis",
                "Edge detection",
                "Local feature descriptors",
            ],
            Domain::Graph => vec![
                "Centrality measures",
                "Community detection",
                "PageRank algorithm",
                "Graph clustering",
                "Structural analysis",
            ],
            Domain::MultiModal => vec![
                "Early fusion strategies",
                "Late fusion approaches",
                "Cross-modal correlation",
                "Attention mechanisms",
                "Feature alignment",
            ],
            Domain::Bioinformatics => vec![
                "Multiple testing correction",
                "Pathway enrichment analysis",
                "Population stratification",
                "Gene set analysis",
                "Linkage disequilibrium",
            ],
            Domain::Finance => vec![
                "Technical indicators",
                "Risk-adjusted returns",
                "Regime detection",
                "Market microstructure",
                "Value-at-Risk modeling",
            ],
            Domain::AdvancedNLP => vec![
                "Dependency parsing",
                "Semantic role labeling",
                "Attention mechanisms",
                "Transformer architectures",
                "Discourse analysis",
            ],
        }
    }

    /// Returns all available domains.
    pub fn all() -> Vec<Domain> {
        vec![
            Domain::TimeSeries,
            Domain::Text,
            Domain::Image,
            Domain::Graph,
            Domain::MultiModal,
            Domain::Bioinformatics,
            Domain::Finance,
            Domain::AdvancedNLP,
        ]
    }
}

impl std::fmt::Display for Domain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Domain::TimeSeries => "Time Series",
            Domain::Text => "Text",
            Domain::Image => "Image",
            Domain::Graph => "Graph",
            Domain::MultiModal => "Multi-Modal",
            Domain::Bioinformatics => "Bioinformatics",
            Domain::Finance => "Finance",
            Domain::AdvancedNLP => "Advanced NLP",
        };
        write!(f, "{}", name)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_descriptions() {
        for domain in Domain::all() {
            assert!(!domain.description().is_empty());
            assert!(!domain.use_cases().is_empty());
            assert!(!domain.key_techniques().is_empty());
        }
    }

    #[test]
    fn test_domain_display() {
        assert_eq!(Domain::TimeSeries.to_string(), "Time Series");
        assert_eq!(Domain::Text.to_string(), "Text");
        assert_eq!(Domain::Image.to_string(), "Image");
        assert_eq!(Domain::Graph.to_string(), "Graph");
        assert_eq!(Domain::MultiModal.to_string(), "Multi-Modal");
        assert_eq!(Domain::Bioinformatics.to_string(), "Bioinformatics");
        assert_eq!(Domain::Finance.to_string(), "Finance");
        assert_eq!(Domain::AdvancedNLP.to_string(), "Advanced NLP");
    }

    #[test]
    fn test_domain_enum_completeness() {
        let all_domains = Domain::all();
        assert_eq!(all_domains.len(), 8);

        // Verify each domain has reasonable metadata
        for domain in all_domains {
            assert!(domain.use_cases().len() >= 3);
            assert!(domain.key_techniques().len() >= 3);
        }
    }
}
