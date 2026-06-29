use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::indexing::QueryCache;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryEngine {
    query_processors: Vec<QueryProcessor>,
    query_cache: QueryCache,
    result_formatters: Vec<ResultFormatter>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryProcessor {
    processor_id: String,
    supported_query_types: Vec<QueryType>,
    optimization_level: OptimizationLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryType {
    Select,
    Aggregate,
    Join,
    TimeRange,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationLevel {
    None,
    Basic,
    Advanced,
    Aggressive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultFormatter {
    formatter_id: String,
    output_format: OutputFormat,
    formatting_options: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    JSON,
    CSV,
    XML,
    Parquet,
    Custom(String),
}

impl Default for QueryEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryEngine {
    pub fn new() -> Self {
        Self {
            query_processors: vec![],
            query_cache: QueryCache::new(),
            result_formatters: vec![],
        }
    }
}
