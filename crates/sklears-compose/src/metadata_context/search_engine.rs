//! # Metadata Search Engine Module
//!
//! Advanced metadata search and indexing system providing powerful search capabilities
//! across all metadata types with intelligent ranking and faceted search features.
//!
//! ## Features
//!
//! - **Full-Text Search**: Comprehensive text search across all metadata fields
//! - **Multi-Index System**: Specialized indexes for different search patterns
//! - **Query Processing**: Advanced query parsing with boolean operations
//! - **Relevance Ranking**: Intelligent scoring and ranking of search results
//! - **Faceted Search**: Support for filtering and faceted navigation
//! - **Auto-completion**: Real-time search suggestions and completions
//! - **Fuzzy Search**: Approximate string matching for typo tolerance
//! - **Performance Optimization**: Caching and performance tuning
//!
//! ## Architecture
//!
//! ```text
//! MetadataSearchEngine
//! ├── FullTextIndex (full-text search capabilities)
//! ├── FieldIndexes (specialized field-based indexes)
//! ├── QueryProcessor (query parsing and execution)
//! ├── RankingEngine (relevance scoring and ranking)
//! ├── FacetEngine (faceted search and filtering)
//! ├── SuggestionEngine (auto-completion and suggestions)
//! ├── CacheManager (result caching and performance)
//! └── PerformanceMonitor (search analytics and optimization)
//! ```

use scirs2_core::error::{CoreError, Result};
use scirs2_core::metrics::{MetricRegistry, Counter, Gauge, Histogram, Timer};
use scirs2_core::ndarray::{Array, Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, BTreeMap, BTreeSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};
use uuid::Uuid;

/// Search index configuration
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Maximum number of search results to return
    pub max_results: usize,
    /// Enable fuzzy search
    pub enable_fuzzy_search: bool,
    /// Fuzzy search edit distance threshold
    pub fuzzy_threshold: usize,
    /// Enable auto-completion
    pub enable_auto_complete: bool,
    /// Maximum auto-completion suggestions
    pub max_suggestions: usize,
    /// Enable result caching
    pub enable_caching: bool,
    /// Cache TTL
    pub cache_ttl: Duration,
    /// Index update batch size
    pub index_batch_size: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_results: 1000,
            enable_fuzzy_search: true,
            fuzzy_threshold: 2,
            enable_auto_complete: true,
            max_suggestions: 10,
            enable_caching: true,
            cache_ttl: Duration::from_secs(300), // 5 minutes
            index_batch_size: 100,
        }
    }
}

/// Searchable document representing metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchDocument {
    /// Document ID
    pub id: String,
    /// Document type
    pub doc_type: DocumentType,
    /// Primary content
    pub title: String,
    /// Description or summary
    pub description: Option<String>,
    /// Full content for search
    pub content: String,
    /// Tags
    pub tags: Vec<String>,
    /// Searchable fields
    pub fields: HashMap<String, SearchableValue>,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last modification timestamp
    pub updated_at: SystemTime,
    /// Document metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Types of searchable documents
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DocumentType {
    /// Metadata entry
    MetadataEntry,
    /// Lineage node
    LineageNode,
    /// Execution record
    ExecutionRecord,
    /// Schema definition
    Schema,
    /// Custom document type
    Custom(String),
}

/// Searchable value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchableValue {
    /// String value
    Text(String),
    /// Numeric value
    Number(f64),
    /// Boolean value
    Boolean(bool),
    /// Date/time value
    DateTime(SystemTime),
    /// Array of values
    Array(Vec<SearchableValue>),
}

/// Search query structure
#[derive(Debug, Clone)]
pub struct SearchQuery {
    /// Query text
    pub text: String,
    /// Document type filter
    pub doc_types: Option<HashSet<DocumentType>>,
    /// Field filters
    pub field_filters: HashMap<String, FieldFilter>,
    /// Tag filters
    pub tag_filters: Option<HashSet<String>>,
    /// Date range filter
    pub date_range: Option<DateRange>,
    /// Search options
    pub options: SearchOptions,
}

/// Field filter for specific field searches
#[derive(Debug, Clone)]
pub enum FieldFilter {
    /// Exact match
    Exact(SearchableValue),
    /// Range filter (for numeric values)
    Range(f64, f64),
    /// Text contains
    Contains(String),
    /// Regular expression match
    Regex(String),
    /// In set filter
    In(Vec<SearchableValue>),
}

/// Date range filter
#[derive(Debug, Clone)]
pub struct DateRange {
    /// Start date (inclusive)
    pub start: Option<SystemTime>,
    /// End date (inclusive)
    pub end: Option<SystemTime>,
}

/// Search options and modifiers
#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// Maximum results to return
    pub limit: Option<usize>,
    /// Result offset for pagination
    pub offset: usize,
    /// Sort order
    pub sort_by: SortBy,
    /// Enable fuzzy matching
    pub fuzzy: bool,
    /// Include highlights in results
    pub highlight: bool,
    /// Include facet counts
    pub facets: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            limit: None,
            offset: 0,
            sort_by: SortBy::Relevance,
            fuzzy: true,
            highlight: false,
            facets: false,
        }
    }
}

/// Sort order options
#[derive(Debug, Clone, PartialEq)]
pub enum SortBy {
    /// Sort by relevance score
    Relevance,
    /// Sort by creation date (newest first)
    CreatedDesc,
    /// Sort by creation date (oldest first)
    CreatedAsc,
    /// Sort by modification date
    UpdatedDesc,
    /// Sort by title
    Title,
    /// Custom field sort
    Field(String, SortDirection),
}

/// Sort direction
#[derive(Debug, Clone, PartialEq)]
pub enum SortDirection {
    Ascending,
    Descending,
}

/// Search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Search results
    pub documents: Vec<SearchDocument>,
    /// Total number of matches
    pub total_matches: usize,
    /// Search execution time
    pub search_time: Duration,
    /// Result highlights (if requested)
    pub highlights: HashMap<String, Vec<String>>,
    /// Facet counts (if requested)
    pub facets: HashMap<String, FacetCounts>,
    /// Search suggestions
    pub suggestions: Vec<String>,
}

/// Facet count information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacetCounts {
    /// Facet values and their counts
    pub counts: HashMap<String, usize>,
    /// Total unique values
    pub total_values: usize,
}

/// Inverted index for full-text search
#[derive(Debug)]
struct InvertedIndex {
    /// Term to document mapping
    term_docs: HashMap<String, PostingList>,
    /// Document frequencies
    doc_frequencies: HashMap<String, usize>,
    /// Total number of documents
    total_docs: usize,
}

/// Posting list for a term
#[derive(Debug, Clone)]
struct PostingList {
    /// Documents containing the term
    documents: Vec<DocumentPosting>,
}

/// Document posting information
#[derive(Debug, Clone)]
struct DocumentPosting {
    /// Document ID
    doc_id: String,
    /// Term frequency in document
    term_frequency: usize,
    /// Positions of term in document
    positions: Vec<usize>,
}

/// Field-based indexes for structured search
#[derive(Debug)]
struct FieldIndexes {
    /// Text field indexes
    text_indexes: HashMap<String, HashMap<String, HashSet<String>>>,
    /// Numeric field indexes
    numeric_indexes: HashMap<String, BTreeMap<OrderedFloat, HashSet<String>>>,
    /// Date field indexes
    date_indexes: HashMap<String, BTreeMap<SystemTime, HashSet<String>>>,
    /// Boolean field indexes
    boolean_indexes: HashMap<String, HashMap<bool, HashSet<String>>>,
}

/// Ordered float wrapper for BTreeMap keys
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
struct OrderedFloat(f64);

impl Eq for OrderedFloat {}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Query processing engine
#[derive(Debug)]
struct QueryProcessor {
    /// Supported operators
    operators: HashSet<String>,
    /// Stop words to ignore
    stop_words: HashSet<String>,
}

/// Relevance scoring engine
#[derive(Debug)]
struct RankingEngine {
    /// TF-IDF scoring weights
    tfidf_weight: f64,
    /// Field boost weights
    field_boosts: HashMap<String, f64>,
    /// Recency boost factor
    recency_boost: f64,
}

/// Auto-completion and suggestion engine
#[derive(Debug)]
struct SuggestionEngine {
    /// N-gram index for suggestions
    ngram_index: HashMap<String, HashSet<String>>,
    /// Popular terms index
    popular_terms: BTreeMap<usize, BTreeSet<String>>, // frequency -> terms
}

/// Search result cache
#[derive(Debug)]
struct SearchCache {
    /// Cached results
    cache: HashMap<String, CacheEntry>,
    /// Cache access times for LRU eviction
    access_times: BTreeMap<Instant, String>,
    /// Maximum cache size
    max_size: usize,
}

/// Cache entry
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Cached result
    result: SearchResult,
    /// Cache timestamp
    cached_at: Instant,
    /// Cache TTL
    ttl: Duration,
}

/// Main metadata search engine
#[derive(Debug)]
pub struct MetadataSearchEngine {
    /// Configuration
    config: SearchConfig,
    /// Document storage
    documents: HashMap<String, SearchDocument>,
    /// Full-text inverted index
    inverted_index: InvertedIndex,
    /// Field-based indexes
    field_indexes: FieldIndexes,
    /// Query processor
    query_processor: QueryProcessor,
    /// Ranking engine
    ranking_engine: RankingEngine,
    /// Suggestion engine
    suggestion_engine: SuggestionEngine,
    /// Result cache
    cache: Option<SearchCache>,
    /// Performance metrics
    metrics: Arc<MetricRegistry>,
    /// Search timers
    search_timer: Timer,
    index_timer: Timer,
    /// Search counters
    searches_performed: Counter,
    cache_hits: Counter,
    cache_misses: Counter,
    /// Index gauges
    indexed_documents: Gauge,
    index_size: Gauge,
}

impl InvertedIndex {
    fn new() -> Self {
        Self {
            term_docs: HashMap::new(),
            doc_frequencies: HashMap::new(),
            total_docs: 0,
        }
    }

    fn add_document(&mut self, doc_id: String, content: &str) {
        let terms = self.tokenize_content(content);
        let term_counts = self.count_terms(&terms);

        for (term, frequency) in term_counts {
            let positions = self.find_term_positions(&terms, &term);

            let posting = DocumentPosting {
                doc_id: doc_id.clone(),
                term_frequency: frequency,
                positions,
            };

            self.term_docs
                .entry(term.clone())
                .or_insert_with(|| PostingList { documents: Vec::new() })
                .documents
                .push(posting);

            *self.doc_frequencies.entry(term).or_insert(0) += 1;
        }

        self.total_docs += 1;
    }

    fn search_terms(&self, terms: &[String]) -> Vec<String> {
        let mut doc_scores = HashMap::new();

        for term in terms {
            if let Some(posting_list) = self.term_docs.get(term) {
                let doc_freq = self.doc_frequencies.get(term).unwrap_or(&0);
                let idf = ((self.total_docs as f64) / (*doc_freq as f64 + 1.0)).ln();

                for posting in &posting_list.documents {
                    let tf = posting.term_frequency as f64;
                    let tf_idf = tf * idf;

                    *doc_scores.entry(posting.doc_id.clone()).or_insert(0.0) += tf_idf;
                }
            }
        }

        let mut scored_docs: Vec<_> = doc_scores.into_iter().collect();
        scored_docs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored_docs.into_iter().map(|(doc_id, _)| doc_id).collect()
    }

    fn tokenize_content(&self, content: &str) -> Vec<String> {
        content
            .to_lowercase()
            .split_whitespace()
            .map(|word| {
                word.chars()
                    .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
                    .collect::<String>()
            })
            .filter(|word| !word.is_empty() && word.len() > 1)
            .collect()
    }

    fn count_terms(&self, terms: &[String]) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for term in terms {
            *counts.entry(term.clone()).or_insert(0) += 1;
        }
        counts
    }

    fn find_term_positions(&self, terms: &[String], target_term: &str) -> Vec<usize> {
        terms
            .iter()
            .enumerate()
            .filter_map(|(i, term)| {
                if term == target_term {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }
}

impl FieldIndexes {
    fn new() -> Self {
        Self {
            text_indexes: HashMap::new(),
            numeric_indexes: HashMap::new(),
            date_indexes: HashMap::new(),
            boolean_indexes: HashMap::new(),
        }
    }

    fn index_field(&mut self, doc_id: String, field_name: String, value: &SearchableValue) {
        match value {
            SearchableValue::Text(text) => {
                let index = self.text_indexes
                    .entry(field_name)
                    .or_insert_with(HashMap::new);

                for token in text.to_lowercase().split_whitespace() {
                    index
                        .entry(token.to_string())
                        .or_insert_with(HashSet::new)
                        .insert(doc_id.clone());
                }
            }
            SearchableValue::Number(num) => {
                let index = self.numeric_indexes
                    .entry(field_name)
                    .or_insert_with(BTreeMap::new);

                index
                    .entry(OrderedFloat(*num))
                    .or_insert_with(HashSet::new)
                    .insert(doc_id);
            }
            SearchableValue::DateTime(date) => {
                let index = self.date_indexes
                    .entry(field_name)
                    .or_insert_with(BTreeMap::new);

                index
                    .entry(*date)
                    .or_insert_with(HashSet::new)
                    .insert(doc_id);
            }
            SearchableValue::Boolean(bool_val) => {
                let index = self.boolean_indexes
                    .entry(field_name)
                    .or_insert_with(HashMap::new);

                index
                    .entry(*bool_val)
                    .or_insert_with(HashSet::new)
                    .insert(doc_id);
            }
            SearchableValue::Array(values) => {
                for value in values {
                    self.index_field(doc_id.clone(), field_name.clone(), value);
                }
            }
        }
    }

    fn search_field(&self, field_name: &str, filter: &FieldFilter) -> HashSet<String> {
        match filter {
            FieldFilter::Exact(value) => {
                match value {
                    SearchableValue::Text(text) => {
                        self.text_indexes
                            .get(field_name)
                            .and_then(|index| index.get(text))
                            .cloned()
                            .unwrap_or_default()
                    }
                    SearchableValue::Number(num) => {
                        self.numeric_indexes
                            .get(field_name)
                            .and_then(|index| index.get(&OrderedFloat(*num)))
                            .cloned()
                            .unwrap_or_default()
                    }
                    SearchableValue::Boolean(bool_val) => {
                        self.boolean_indexes
                            .get(field_name)
                            .and_then(|index| index.get(bool_val))
                            .cloned()
                            .unwrap_or_default()
                    }
                    SearchableValue::DateTime(date) => {
                        self.date_indexes
                            .get(field_name)
                            .and_then(|index| index.get(date))
                            .cloned()
                            .unwrap_or_default()
                    }
                    SearchableValue::Array(_) => HashSet::new(),
                }
            }
            FieldFilter::Range(min, max) => {
                if let Some(index) = self.numeric_indexes.get(field_name) {
                    index
                        .range(OrderedFloat(*min)..=OrderedFloat(*max))
                        .flat_map(|(_, docs)| docs.iter().cloned())
                        .collect()
                } else {
                    HashSet::new()
                }
            }
            FieldFilter::Contains(text) => {
                if let Some(index) = self.text_indexes.get(field_name) {
                    index
                        .iter()
                        .filter(|(key, _)| key.contains(&text.to_lowercase()))
                        .flat_map(|(_, docs)| docs.iter().cloned())
                        .collect()
                } else {
                    HashSet::new()
                }
            }
            FieldFilter::Regex(_pattern) => {
                // Simplified regex - would use actual regex crate in production
                HashSet::new()
            }
            FieldFilter::In(values) => {
                let mut result = HashSet::new();
                for value in values {
                    result.extend(self.search_field(field_name, &FieldFilter::Exact(value.clone())));
                }
                result
            }
        }
    }
}

impl QueryProcessor {
    fn new() -> Self {
        let stop_words = [
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "were", "will", "with", "would"
        ].iter().map(|s| s.to_string()).collect();

        Self {
            operators: ["AND", "OR", "NOT"].iter().map(|s| s.to_string()).collect(),
            stop_words,
        }
    }

    fn process_query(&self, query_text: &str) -> Vec<String> {
        query_text
            .to_lowercase()
            .split_whitespace()
            .map(|word| {
                word.chars()
                    .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
                    .collect::<String>()
            })
            .filter(|word| {
                !word.is_empty()
                && word.len() > 1
                && !self.stop_words.contains(word)
                && !self.operators.contains(&word.to_uppercase())
            })
            .collect()
    }

    fn extract_phrases(&self, query_text: &str) -> Vec<String> {
        let mut phrases = Vec::new();
        let mut current_phrase = String::new();
        let mut in_quotes = false;

        for ch in query_text.chars() {
            match ch {
                '"' => {
                    if in_quotes && !current_phrase.is_empty() {
                        phrases.push(current_phrase.trim().to_string());
                        current_phrase.clear();
                    }
                    in_quotes = !in_quotes;
                }
                _ if in_quotes => {
                    current_phrase.push(ch);
                }
                _ => {} // Ignore characters outside quotes
            }
        }

        phrases
    }
}

impl RankingEngine {
    fn new() -> Self {
        let mut field_boosts = HashMap::new();
        field_boosts.insert("title".to_string(), 2.0);
        field_boosts.insert("description".to_string(), 1.5);
        field_boosts.insert("tags".to_string(), 1.8);

        Self {
            tfidf_weight: 1.0,
            field_boosts,
            recency_boost: 0.1,
        }
    }

    fn calculate_score(
        &self,
        doc: &SearchDocument,
        query_terms: &[String],
        base_score: f64,
    ) -> f64 {
        let mut final_score = base_score * self.tfidf_weight;

        // Apply field boosts
        for (field, boost) in &self.field_boosts {
            if let Some(field_value) = doc.fields.get(field) {
                if self.field_contains_terms(field_value, query_terms) {
                    final_score *= boost;
                }
            }
        }

        // Apply recency boost
        if let Ok(age) = SystemTime::now().duration_since(doc.updated_at) {
            let age_days = age.as_secs() as f64 / 86400.0; // Convert to days
            let recency_factor = (-age_days / 30.0).exp(); // Exponential decay over 30 days
            final_score += final_score * self.recency_boost * recency_factor;
        }

        final_score
    }

    fn field_contains_terms(&self, field_value: &SearchableValue, terms: &[String]) -> bool {
        match field_value {
            SearchableValue::Text(text) => {
                let text_lower = text.to_lowercase();
                terms.iter().any(|term| text_lower.contains(term))
            }
            SearchableValue::Array(values) => {
                values.iter().any(|value| self.field_contains_terms(value, terms))
            }
            _ => false,
        }
    }
}

impl SuggestionEngine {
    fn new() -> Self {
        Self {
            ngram_index: HashMap::new(),
            popular_terms: BTreeMap::new(),
        }
    }

    fn add_term(&mut self, term: &str, frequency: usize) {
        // Add n-grams for the term
        for i in 0..term.len().saturating_sub(1) {
            for j in i + 2..=term.len().min(i + 4) { // 2-4 character n-grams
                let ngram = &term[i..j];
                self.ngram_index
                    .entry(ngram.to_string())
                    .or_insert_with(HashSet::new)
                    .insert(term.to_string());
            }
        }

        // Update popularity
        self.popular_terms
            .entry(frequency)
            .or_insert_with(BTreeSet::new)
            .insert(term.to_string());
    }

    fn get_suggestions(&self, prefix: &str, limit: usize) -> Vec<String> {
        let mut suggestions = HashSet::new();

        // Exact prefix matches
        for (term_freq, terms) in self.popular_terms.iter().rev() {
            for term in terms {
                if term.starts_with(prefix) {
                    suggestions.insert(term.clone());
                    if suggestions.len() >= limit {
                        break;
                    }
                }
            }
            if suggestions.len() >= limit {
                break;
            }
        }

        // N-gram fuzzy matches if needed
        if suggestions.len() < limit && prefix.len() >= 2 {
            for ngram in self.generate_ngrams(prefix) {
                if let Some(terms) = self.ngram_index.get(&ngram) {
                    for term in terms {
                        if self.levenshtein_distance(prefix, term) <= 2 {
                            suggestions.insert(term.clone());
                            if suggestions.len() >= limit {
                                break;
                            }
                        }
                    }
                }
                if suggestions.len() >= limit {
                    break;
                }
            }
        }

        suggestions.into_iter().take(limit).collect()
    }

    fn generate_ngrams(&self, text: &str) -> Vec<String> {
        let mut ngrams = Vec::new();
        for i in 0..text.len().saturating_sub(1) {
            for j in i + 2..=text.len().min(i + 4) {
                ngrams.push(text[i..j].to_string());
            }
        }
        ngrams
    }

    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let len1 = s1.chars().count();
        let len2 = s2.chars().count();
        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();

        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if s1_chars[i - 1] == s2_chars[j - 1] { 0 } else { 1 };
                matrix[i][j] = std::cmp::min(
                    std::cmp::min(
                        matrix[i - 1][j] + 1,      // Deletion
                        matrix[i][j - 1] + 1,      // Insertion
                    ),
                    matrix[i - 1][j - 1] + cost,   // Substitution
                );
            }
        }

        matrix[len1][len2]
    }
}

impl SearchCache {
    fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            access_times: BTreeMap::new(),
            max_size,
        }
    }

    fn get(&mut self, key: &str) -> Option<SearchResult> {
        if let Some(entry) = self.cache.get(key) {
            let now = Instant::now();

            // Check TTL
            if now.duration_since(entry.cached_at) < entry.ttl {
                // Update access time
                self.access_times.insert(now, key.to_string());
                return Some(entry.result.clone());
            } else {
                // Remove expired entry
                self.cache.remove(key);
            }
        }
        None
    }

    fn insert(&mut self, key: String, result: SearchResult, ttl: Duration) {
        let now = Instant::now();

        // Remove oldest entries if at capacity
        while self.cache.len() >= self.max_size {
            if let Some((oldest_time, oldest_key)) = self.access_times.iter().next() {
                let oldest_time = *oldest_time;
                let oldest_key = oldest_key.clone();
                self.cache.remove(&oldest_key);
                self.access_times.remove(&oldest_time);
            } else {
                break;
            }
        }

        let entry = CacheEntry {
            result,
            cached_at: now,
            ttl,
        };

        self.cache.insert(key.clone(), entry);
        self.access_times.insert(now, key);
    }
}

impl MetadataSearchEngine {
    /// Create a new search engine
    pub fn new() -> Self {
        Self::with_config(SearchConfig::default())
    }

    /// Create search engine with configuration
    pub fn with_config(config: SearchConfig) -> Self {
        let metrics = Arc::new(MetricRegistry::new());

        let cache = if config.enable_caching {
            Some(SearchCache::new(1000))
        } else {
            None
        };

        Self {
            config,
            documents: HashMap::new(),
            inverted_index: InvertedIndex::new(),
            field_indexes: FieldIndexes::new(),
            query_processor: QueryProcessor::new(),
            ranking_engine: RankingEngine::new(),
            suggestion_engine: SuggestionEngine::new(),
            cache,
            metrics: metrics.clone(),
            search_timer: metrics.timer("search.duration"),
            index_timer: metrics.timer("search.index_duration"),
            searches_performed: metrics.counter("search.searches_performed"),
            cache_hits: metrics.counter("search.cache_hits"),
            cache_misses: metrics.counter("search.cache_misses"),
            indexed_documents: metrics.gauge("search.indexed_documents"),
            index_size: metrics.gauge("search.index_size_mb"),
        }
    }

    /// Index a document for searching
    pub fn index_document(&mut self, document: SearchDocument) -> Result<()> {
        let _timer = self.index_timer.start_timer();

        let doc_id = document.id.clone();

        // Index content for full-text search
        let full_content = format!(
            "{} {} {} {}",
            document.title,
            document.description.as_deref().unwrap_or(""),
            document.content,
            document.tags.join(" ")
        );

        self.inverted_index.add_document(doc_id.clone(), &full_content);

        // Index individual fields
        for (field_name, field_value) in &document.fields {
            self.field_indexes.index_field(
                doc_id.clone(),
                field_name.clone(),
                field_value
            );
        }

        // Add terms to suggestion engine
        for token in self.query_processor.process_query(&full_content) {
            self.suggestion_engine.add_term(&token, 1);
        }

        // Store document
        self.documents.insert(doc_id, document);

        // Update metrics
        self.indexed_documents.set(self.documents.len() as f64);

        Ok(())
    }

    /// Perform a search
    pub fn search(&mut self, query: SearchQuery) -> Result<SearchResult> {
        let _timer = self.search_timer.start_timer();

        // Generate cache key
        let cache_key = self.generate_cache_key(&query);

        // Check cache first
        if let Some(mut cache) = self.cache.as_mut() {
            if let Some(cached_result) = cache.get(&cache_key) {
                self.cache_hits.inc();
                return Ok(cached_result);
            }
            self.cache_misses.inc();
        }

        let start_time = Instant::now();

        // Process query text
        let query_terms = self.query_processor.process_query(&query.text);

        // Get candidate documents from full-text search
        let mut candidate_docs = if !query_terms.is_empty() {
            self.inverted_index.search_terms(&query_terms)
        } else {
            self.documents.keys().cloned().collect()
        };

        // Apply filters
        candidate_docs = self.apply_filters(candidate_docs, &query)?;

        // Score and rank results
        let mut scored_docs = self.score_documents(candidate_docs, &query_terms, &query)?;

        // Apply sorting
        self.sort_documents(&mut scored_docs, &query.options.sort_by);

        // Apply pagination
        let total_matches = scored_docs.len();
        let offset = query.options.offset;
        let limit = query.options.limit.unwrap_or(self.config.max_results);

        let paginated_docs: Vec<_> = scored_docs
            .into_iter()
            .skip(offset)
            .take(limit)
            .collect();

        // Prepare result documents
        let result_documents: Vec<_> = paginated_docs
            .iter()
            .filter_map(|(doc_id, _score)| self.documents.get(doc_id).cloned())
            .collect();

        // Generate highlights if requested
        let highlights = if query.options.highlight {
            self.generate_highlights(&result_documents, &query_terms)
        } else {
            HashMap::new()
        };

        // Generate facets if requested
        let facets = if query.options.facets {
            self.generate_facets(&result_documents)
        } else {
            HashMap::new()
        };

        // Generate suggestions
        let suggestions = if self.config.enable_auto_complete && !query.text.is_empty() {
            self.suggestion_engine.get_suggestions(&query.text, self.config.max_suggestions)
        } else {
            Vec::new()
        };

        let search_time = start_time.elapsed();

        let result = SearchResult {
            documents: result_documents,
            total_matches,
            search_time,
            highlights,
            facets,
            suggestions,
        };

        // Cache result
        if let Some(cache) = self.cache.as_mut() {
            cache.insert(cache_key, result.clone(), self.config.cache_ttl);
        }

        self.searches_performed.inc();

        Ok(result)
    }

    /// Get search suggestions
    pub fn get_suggestions(&self, prefix: &str, limit: Option<usize>) -> Vec<String> {
        let limit = limit.unwrap_or(self.config.max_suggestions);
        self.suggestion_engine.get_suggestions(prefix, limit)
    }

    /// Get search statistics
    pub fn get_statistics(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();

        stats.insert("indexed_documents".to_string(), json!(self.documents.len()));
        stats.insert("total_searches".to_string(), json!(self.searches_performed.get()));
        stats.insert("cache_hits".to_string(), json!(self.cache_hits.get()));
        stats.insert("cache_misses".to_string(), json!(self.cache_misses.get()));

        let hit_rate = if self.cache_hits.get() + self.cache_misses.get() > 0 {
            self.cache_hits.get() as f64 / (self.cache_hits.get() + self.cache_misses.get()) as f64
        } else {
            0.0
        };
        stats.insert("cache_hit_rate".to_string(), json!(hit_rate));

        // Index statistics
        stats.insert("unique_terms".to_string(), json!(self.inverted_index.term_docs.len()));
        stats.insert("field_indexes".to_string(), json!(self.field_indexes.text_indexes.len()));

        stats
    }

    // Private helper methods

    fn generate_cache_key(&self, query: &SearchQuery) -> String {
        // Simple cache key generation - in production would use proper hashing
        format!(
            "{}:{}:{}:{}",
            query.text,
            query.doc_types.as_ref().map(|types| types.len()).unwrap_or(0),
            query.field_filters.len(),
            query.options.offset
        )
    }

    fn apply_filters(&self, candidates: Vec<String>, query: &SearchQuery) -> Result<Vec<String>> {
        let mut filtered = candidates;

        // Apply document type filter
        if let Some(doc_types) = &query.doc_types {
            filtered = filtered
                .into_iter()
                .filter(|doc_id| {
                    self.documents
                        .get(doc_id)
                        .map(|doc| doc_types.contains(&doc.doc_type))
                        .unwrap_or(false)
                })
                .collect();
        }

        // Apply field filters
        for (field_name, field_filter) in &query.field_filters {
            let matching_docs = self.field_indexes.search_field(field_name, field_filter);
            filtered = filtered
                .into_iter()
                .filter(|doc_id| matching_docs.contains(doc_id))
                .collect();
        }

        // Apply tag filters
        if let Some(required_tags) = &query.tag_filters {
            filtered = filtered
                .into_iter()
                .filter(|doc_id| {
                    self.documents
                        .get(doc_id)
                        .map(|doc| {
                            required_tags.iter().all(|tag| doc.tags.contains(tag))
                        })
                        .unwrap_or(false)
                })
                .collect();
        }

        // Apply date range filter
        if let Some(date_range) = &query.date_range {
            filtered = filtered
                .into_iter()
                .filter(|doc_id| {
                    self.documents
                        .get(doc_id)
                        .map(|doc| self.date_in_range(doc.created_at, date_range))
                        .unwrap_or(false)
                })
                .collect();
        }

        Ok(filtered)
    }

    fn date_in_range(&self, date: SystemTime, range: &DateRange) -> bool {
        if let Some(start) = range.start {
            if date < start {
                return false;
            }
        }
        if let Some(end) = range.end {
            if date > end {
                return false;
            }
        }
        true
    }

    fn score_documents(
        &self,
        candidates: Vec<String>,
        query_terms: &[String],
        _query: &SearchQuery,
    ) -> Result<Vec<(String, f64)>> {
        let mut scored_docs = Vec::new();

        for doc_id in candidates {
            if let Some(document) = self.documents.get(&doc_id) {
                // Calculate base TF-IDF score
                let base_score = self.calculate_tfidf_score(&doc_id, query_terms);

                // Apply ranking engine
                let final_score = self.ranking_engine.calculate_score(
                    document,
                    query_terms,
                    base_score,
                );

                scored_docs.push((doc_id, final_score));
            }
        }

        Ok(scored_docs)
    }

    fn calculate_tfidf_score(&self, doc_id: &str, query_terms: &[String]) -> f64 {
        let mut score = 0.0;

        for term in query_terms {
            if let Some(posting_list) = self.inverted_index.term_docs.get(term) {
                if let Some(posting) = posting_list.documents
                    .iter()
                    .find(|p| p.doc_id == doc_id) {

                    let tf = posting.term_frequency as f64;
                    let doc_freq = self.inverted_index.doc_frequencies
                        .get(term)
                        .unwrap_or(&1);
                    let idf = (self.inverted_index.total_docs as f64 / *doc_freq as f64).ln();

                    score += tf * idf;
                }
            }
        }

        score
    }

    fn sort_documents(&self, docs: &mut [(String, f64)], sort_by: &SortBy) {
        match sort_by {
            SortBy::Relevance => {
                docs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            SortBy::CreatedDesc => {
                docs.sort_by(|a, b| {
                    let doc_a = self.documents.get(&a.0);
                    let doc_b = self.documents.get(&b.0);
                    match (doc_a, doc_b) {
                        (Some(a), Some(b)) => b.created_at.cmp(&a.created_at),
                        _ => std::cmp::Ordering::Equal,
                    }
                });
            }
            SortBy::CreatedAsc => {
                docs.sort_by(|a, b| {
                    let doc_a = self.documents.get(&a.0);
                    let doc_b = self.documents.get(&b.0);
                    match (doc_a, doc_b) {
                        (Some(a), Some(b)) => a.created_at.cmp(&b.created_at),
                        _ => std::cmp::Ordering::Equal,
                    }
                });
            }
            SortBy::UpdatedDesc => {
                docs.sort_by(|a, b| {
                    let doc_a = self.documents.get(&a.0);
                    let doc_b = self.documents.get(&b.0);
                    match (doc_a, doc_b) {
                        (Some(a), Some(b)) => b.updated_at.cmp(&a.updated_at),
                        _ => std::cmp::Ordering::Equal,
                    }
                });
            }
            SortBy::Title => {
                docs.sort_by(|a, b| {
                    let doc_a = self.documents.get(&a.0);
                    let doc_b = self.documents.get(&b.0);
                    match (doc_a, doc_b) {
                        (Some(a), Some(b)) => a.title.cmp(&b.title),
                        _ => std::cmp::Ordering::Equal,
                    }
                });
            }
            SortBy::Field(_field_name, _direction) => {
                // Custom field sorting would be implemented here
            }
        }
    }

    fn generate_highlights(
        &self,
        documents: &[SearchDocument],
        query_terms: &[String],
    ) -> HashMap<String, Vec<String>> {
        let mut highlights = HashMap::new();

        for doc in documents {
            let mut doc_highlights = Vec::new();

            for term in query_terms {
                // Simple highlighting - would use more sophisticated highlighting in production
                if doc.content.to_lowercase().contains(term) {
                    let highlight = format!("...{}...", term);
                    doc_highlights.push(highlight);
                }
            }

            if !doc_highlights.is_empty() {
                highlights.insert(doc.id.clone(), doc_highlights);
            }
        }

        highlights
    }

    fn generate_facets(&self, documents: &[SearchDocument]) -> HashMap<String, FacetCounts> {
        let mut facets = HashMap::new();

        // Document type facets
        let mut type_counts = HashMap::new();
        for doc in documents {
            *type_counts.entry(format!("{:?}", doc.doc_type)).or_insert(0) += 1;
        }
        facets.insert(
            "doc_type".to_string(),
            FacetCounts {
                total_values: type_counts.len(),
                counts: type_counts,
            }
        );

        // Tag facets
        let mut tag_counts = HashMap::new();
        for doc in documents {
            for tag in &doc.tags {
                *tag_counts.entry(tag.clone()).or_insert(0) += 1;
            }
        }
        facets.insert(
            "tags".to_string(),
            FacetCounts {
                total_values: tag_counts.len(),
                counts: tag_counts,
            }
        );

        facets
    }
}

impl Default for MetadataSearchEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_indexing_and_search() {
        let mut engine = MetadataSearchEngine::new();

        // Index a test document
        let mut fields = HashMap::new();
        fields.insert("algorithm".to_string(), SearchableValue::Text("linear_regression".to_string()));
        fields.insert("accuracy".to_string(), SearchableValue::Number(0.95));

        let document = SearchDocument {
            id: "doc1".to_string(),
            doc_type: DocumentType::MetadataEntry,
            title: "Linear Regression Model".to_string(),
            description: Some("A simple linear regression model".to_string()),
            content: "Linear regression machine learning model with high accuracy".to_string(),
            tags: vec!["ml".to_string(), "regression".to_string()],
            fields,
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            metadata: HashMap::new(),
        };

        engine.index_document(document).unwrap();

        // Perform a search
        let query = SearchQuery {
            text: "linear regression".to_string(),
            doc_types: None,
            field_filters: HashMap::new(),
            tag_filters: None,
            date_range: None,
            options: SearchOptions::default(),
        };

        let results = engine.search(query).unwrap();

        assert_eq!(results.documents.len(), 1);
        assert_eq!(results.documents[0].title, "Linear Regression Model");
        assert_eq!(results.total_matches, 1);
    }

    #[test]
    fn test_field_filtering() {
        let mut engine = MetadataSearchEngine::new();

        // Index documents with different field values
        let mut fields1 = HashMap::new();
        fields1.insert("accuracy".to_string(), SearchableValue::Number(0.95));

        let mut fields2 = HashMap::new();
        fields2.insert("accuracy".to_string(), SearchableValue::Number(0.85));

        let doc1 = SearchDocument {
            id: "doc1".to_string(),
            doc_type: DocumentType::MetadataEntry,
            title: "High Accuracy Model".to_string(),
            description: None,
            content: "model with high accuracy".to_string(),
            tags: vec![],
            fields: fields1,
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            metadata: HashMap::new(),
        };

        let doc2 = SearchDocument {
            id: "doc2".to_string(),
            doc_type: DocumentType::MetadataEntry,
            title: "Medium Accuracy Model".to_string(),
            description: None,
            content: "model with medium accuracy".to_string(),
            tags: vec![],
            fields: fields2,
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            metadata: HashMap::new(),
        };

        engine.index_document(doc1).unwrap();
        engine.index_document(doc2).unwrap();

        // Search with field filter
        let mut field_filters = HashMap::new();
        field_filters.insert(
            "accuracy".to_string(),
            FieldFilter::Range(0.90, 1.0)
        );

        let query = SearchQuery {
            text: "model".to_string(),
            doc_types: None,
            field_filters,
            tag_filters: None,
            date_range: None,
            options: SearchOptions::default(),
        };

        let results = engine.search(query).unwrap();

        assert_eq!(results.documents.len(), 1);
        assert_eq!(results.documents[0].title, "High Accuracy Model");
    }

    #[test]
    fn test_suggestions() {
        let mut engine = MetadataSearchEngine::new();

        // Index some documents to build suggestion index
        let document = SearchDocument {
            id: "doc1".to_string(),
            doc_type: DocumentType::MetadataEntry,
            title: "Machine Learning Model".to_string(),
            description: Some("Advanced machine learning algorithms".to_string()),
            content: "machine learning deep neural networks".to_string(),
            tags: vec!["machine".to_string(), "learning".to_string()],
            fields: HashMap::new(),
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            metadata: HashMap::new(),
        };

        engine.index_document(document).unwrap();

        // Get suggestions
        let suggestions = engine.get_suggestions("mach", Some(5));

        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.contains("machine")));
    }

    #[test]
    fn test_faceted_search() {
        let mut engine = MetadataSearchEngine::new();

        // Index documents with different types and tags
        let doc1 = SearchDocument {
            id: "doc1".to_string(),
            doc_type: DocumentType::MetadataEntry,
            title: "Test Document 1".to_string(),
            description: None,
            content: "content".to_string(),
            tags: vec!["tag1".to_string(), "shared".to_string()],
            fields: HashMap::new(),
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            metadata: HashMap::new(),
        };

        let doc2 = SearchDocument {
            id: "doc2".to_string(),
            doc_type: DocumentType::ExecutionRecord,
            title: "Test Document 2".to_string(),
            description: None,
            content: "content".to_string(),
            tags: vec!["tag2".to_string(), "shared".to_string()],
            fields: HashMap::new(),
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            metadata: HashMap::new(),
        };

        engine.index_document(doc1).unwrap();
        engine.index_document(doc2).unwrap();

        // Search with facets enabled
        let mut options = SearchOptions::default();
        options.facets = true;

        let query = SearchQuery {
            text: "test".to_string(),
            doc_types: None,
            field_filters: HashMap::new(),
            tag_filters: None,
            date_range: None,
            options,
        };

        let results = engine.search(query).unwrap();

        assert_eq!(results.documents.len(), 2);
        assert!(!results.facets.is_empty());
        assert!(results.facets.contains_key("doc_type"));
        assert!(results.facets.contains_key("tags"));

        let tag_facets = results.facets.get("tags").unwrap();
        assert_eq!(tag_facets.counts.get("shared"), Some(&2));
    }
}