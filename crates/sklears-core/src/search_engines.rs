use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::api_data_structures::{CodeExample, TraitInfo, TypeInfo};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchEngineConfig {
    pub semantic_search_enabled: bool,
    pub type_search_enabled: bool,
    pub fuzzy_matching_enabled: bool,
    pub autocomplete_enabled: bool,
    pub max_results: usize,
    pub similarity_threshold: f64,
    pub indexing_batch_size: usize,
    pub cache_size: usize,
}

impl Default for SearchEngineConfig {
    fn default() -> Self {
        Self {
            semantic_search_enabled: true,
            type_search_enabled: true,
            fuzzy_matching_enabled: true,
            autocomplete_enabled: true,
            max_results: 50,
            similarity_threshold: 0.3,
            indexing_batch_size: 1000,
            cache_size: 10000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    pub text: String,
    pub query_type: SearchQueryType,
    pub filters: SearchFilters,
    pub options: SearchOptions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchQueryType {
    General,
    Semantic,
    TypeSignature,
    Usage,
    Documentation,
    Examples,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchFilters {
    pub categories: Vec<ItemCategory>,
    pub visibility: Vec<Visibility>,
    pub stability: Vec<Stability>,
    pub crates: Vec<String>,
    pub modules: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ItemCategory {
    Trait,
    Struct,
    Enum,
    Function,
    Method,
    Constant,
    Type,
    Module,
    Macro,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Visibility {
    Public,
    Private,
    Crate,
    Super,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Stability {
    Stable,
    Unstable,
    Deprecated,
    Experimental,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchOptions {
    pub case_sensitive: bool,
    pub whole_words_only: bool,
    pub use_stemming: bool,
    pub include_examples: bool,
    pub include_tests: bool,
    pub rank_by_usage: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            case_sensitive: false,
            whole_words_only: false,
            use_stemming: true,
            include_examples: true,
            include_tests: false,
            rank_by_usage: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub title: String,
    pub description: String,
    pub category: ItemCategory,
    pub url: String,
    pub score: f64,
    pub snippet: Option<String>,
    pub metadata: SearchResultMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResultMetadata {
    pub crate_name: String,
    pub module_path: String,
    pub line_number: Option<usize>,
    pub visibility: Visibility,
    pub stability: Stability,
    pub since_version: Option<String>,
    pub deprecated_since: Option<String>,
    pub related_items: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchIndex {
    pub items: HashMap<String, IndexedItem>,
    pub word_index: HashMap<String, HashSet<String>>,
    pub type_index: HashMap<String, HashSet<String>>,
    pub usage_index: HashMap<String, UsageInfo>,
    pub semantic_index: SemanticIndex,
    pub autocomplete_trie: AutocompleteTrie,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedItem {
    pub id: String,
    pub content: String,
    pub category: ItemCategory,
    pub keywords: Vec<String>,
    pub type_signature: Option<String>,
    pub documentation: String,
    pub examples: Vec<String>,
    pub metadata: SearchResultMetadata,
    pub usage_count: usize,
    pub popularity_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageInfo {
    pub frequency: usize,
    pub contexts: Vec<UsageContext>,
    pub common_patterns: Vec<String>,
    pub related_functions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageContext {
    pub location: String,
    pub snippet: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticIndex {
    pub embeddings: HashMap<String, Vec<f32>>,
    pub clusters: Vec<SemanticCluster>,
    pub similarity_matrix: HashMap<String, HashMap<String, f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCluster {
    pub id: String,
    pub center: Vec<f32>,
    pub items: Vec<String>,
    pub coherence_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutocompleteTrie {
    pub root: TrieNode,
    pub suggestions_cache: HashMap<String, Vec<AutocompleteSuggestion>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrieNode {
    pub value: Option<char>,
    pub children: HashMap<char, TrieNode>,
    pub is_end_of_word: bool,
    pub completions: Vec<AutocompleteSuggestion>,
    pub frequency: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutocompleteSuggestion {
    pub text: String,
    pub category: ItemCategory,
    pub description: String,
    pub frequency: usize,
    pub relevance_score: f64,
}

pub struct SemanticSearchEngine {
    config: SearchEngineConfig,
    index: SearchIndex,
    query_cache: HashMap<String, Vec<SearchResult>>,
    performance_metrics: SearchMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMetrics {
    pub total_queries: usize,
    pub cache_hits: usize,
    pub average_response_time: f64,
    pub index_size: usize,
    pub last_reindex_time: chrono::DateTime<chrono::Utc>,
}

impl SemanticSearchEngine {
    pub fn new(config: SearchEngineConfig) -> Self {
        Self {
            config,
            index: SearchIndex::new(),
            query_cache: HashMap::new(),
            performance_metrics: SearchMetrics::default(),
        }
    }

    pub fn build_index(
        &mut self,
        traits: &[TraitInfo],
        types: &[TypeInfo],
        examples: &[CodeExample],
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.index_traits(traits)?;
        self.index_types(types)?;
        self.index_examples(examples)?;
        self.build_semantic_embeddings()?;
        self.build_autocomplete_trie()?;
        self.update_usage_statistics()?;
        Ok(())
    }

    fn index_traits(&mut self, traits: &[TraitInfo]) -> Result<(), Box<dyn std::error::Error>> {
        for trait_info in traits {
            let item = IndexedItem {
                id: format!("trait_{}", trait_info.name),
                content: format!("{} {}", trait_info.name, trait_info.description),
                category: ItemCategory::Trait,
                keywords: self.extract_keywords(&trait_info.description),
                type_signature: Some(self.build_trait_signature(trait_info)),
                documentation: trait_info.description.clone(),
                examples: vec![], // Examples not directly available in TraitInfo
                metadata: SearchResultMetadata {
                    crate_name: "api".to_string(), // Default crate name
                    module_path: trait_info.path.clone(),
                    line_number: None,
                    visibility: Visibility::Public,
                    stability: Stability::Stable,
                    since_version: None, // Not available in TraitInfo
                    deprecated_since: None,
                    related_items: trait_info.implementations.clone(),
                },
                usage_count: 0,
                popularity_score: 0.0,
            };

            let item_id = item.id.clone();
            self.index.items.insert(item_id.clone(), item);
            self.index_words(&trait_info.name, &item_id);
        }
        Ok(())
    }

    fn index_types(&mut self, types: &[TypeInfo]) -> Result<(), Box<dyn std::error::Error>> {
        for type_info in types {
            let item = IndexedItem {
                id: format!("type_{}", type_info.name),
                content: format!("{} {}", type_info.name, type_info.description),
                category: self.determine_type_category(type_info),
                keywords: self.extract_keywords(&type_info.description),
                type_signature: Some(format!("{:?}", type_info.kind)), // Convert enum to string
                documentation: type_info.description.clone(),
                examples: vec![], // Examples not directly available in TypeInfo
                metadata: SearchResultMetadata {
                    crate_name: "api".to_string(), // Default crate name
                    module_path: type_info.path.clone(),
                    line_number: None,
                    visibility: Visibility::Public,
                    stability: Stability::Stable,
                    since_version: None, // Not available in TypeInfo
                    deprecated_since: None,
                    related_items: type_info.trait_impls.clone(),
                },
                usage_count: 0,
                popularity_score: 0.0,
            };

            let item_id = item.id.clone();
            self.index.items.insert(item_id.clone(), item);
            self.index_words(&type_info.name, &item_id);
            self.index_type_signature(&format!("{:?}", type_info.kind), &item_id);
        }
        Ok(())
    }

    fn index_examples(
        &mut self,
        examples: &[CodeExample],
    ) -> Result<(), Box<dyn std::error::Error>> {
        for (idx, example) in examples.iter().enumerate() {
            let item = IndexedItem {
                id: format!("example_{}", idx),
                content: format!("{} {}", example.title, example.code),
                category: ItemCategory::Function,
                keywords: self.extract_keywords(&example.description),
                type_signature: None,
                documentation: example.description.clone(),
                examples: vec![example.code.clone()],
                metadata: SearchResultMetadata {
                    crate_name: "examples".to_string(),
                    module_path: "examples".to_string(),
                    line_number: None,
                    visibility: Visibility::Public,
                    stability: Stability::Stable,
                    since_version: None,
                    deprecated_since: None,
                    related_items: vec![],
                },
                usage_count: 0,
                popularity_score: 0.0,
            };

            let item_id = item.id.clone();
            self.index.items.insert(item_id.clone(), item);
            self.index_words(&example.title, &item_id);
            self.index_words(&example.description, &item_id);
        }
        Ok(())
    }

    fn build_trait_signature(&self, trait_info: &TraitInfo) -> String {
        format!(
            "trait {}{}",
            trait_info.name,
            if trait_info.generics.is_empty() {
                String::new()
            } else {
                format!("<{}>", trait_info.generics.join(", "))
            }
        )
    }

    fn determine_type_category(&self, type_info: &TypeInfo) -> ItemCategory {
        use crate::api_data_structures::TypeKind;
        match type_info.kind {
            TypeKind::Struct => ItemCategory::Struct,
            TypeKind::Enum => ItemCategory::Enum,
            TypeKind::Union => ItemCategory::Type,
            TypeKind::TypeAlias => ItemCategory::Type,
            TypeKind::Trait => ItemCategory::Trait,
        }
    }

    fn extract_keywords(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|word| word.to_lowercase())
            .filter(|word| word.len() > 2)
            .collect()
    }

    fn index_words(&mut self, text: &str, item_id: &str) {
        for word in text.split_whitespace() {
            let word = word.to_lowercase();
            self.index
                .word_index
                .entry(word)
                .or_default()
                .insert(item_id.to_string());
        }
    }

    fn index_type_signature(&mut self, type_sig: &str, item_id: &str) {
        self.index
            .type_index
            .entry(type_sig.to_string())
            .or_default()
            .insert(item_id.to_string());
    }

    fn build_semantic_embeddings(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        for (item_id, item) in &self.index.items {
            let embedding = self.compute_embedding(&item.content);
            self.index
                .semantic_index
                .embeddings
                .insert(item_id.clone(), embedding);
        }
        self.build_semantic_clusters()?;
        Ok(())
    }

    fn compute_embedding(&self, text: &str) -> Vec<f32> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut embedding = vec![0.0; 300];

        for (i, word) in words.iter().enumerate().take(300) {
            embedding[i] = word.len() as f32;
        }

        embedding
    }

    fn build_semantic_clusters(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let embeddings: Vec<(String, Vec<f32>)> = self
            .index
            .semantic_index
            .embeddings
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let clusters = self.k_means_clustering(&embeddings, 10)?;
        self.index.semantic_index.clusters = clusters;
        Ok(())
    }

    fn k_means_clustering(
        &self,
        embeddings: &[(String, Vec<f32>)],
        k: usize,
    ) -> Result<Vec<SemanticCluster>, Box<dyn std::error::Error>> {
        let mut clusters = Vec::new();
        let embedding_dim = embeddings.first().map(|(_, e)| e.len()).unwrap_or(300);

        for i in 0..k {
            clusters.push(SemanticCluster {
                id: format!("cluster_{}", i),
                center: vec![0.0; embedding_dim],
                items: Vec::new(),
                coherence_score: 0.0,
            });
        }

        for (item_id, embedding) in embeddings {
            let closest_cluster = self.find_closest_cluster(&clusters, embedding);
            clusters[closest_cluster].items.push(item_id.clone());
        }

        Ok(clusters)
    }

    fn find_closest_cluster(&self, clusters: &[SemanticCluster], embedding: &[f32]) -> usize {
        clusters
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let dist_a = self.cosine_distance(&a.center, embedding);
                let dist_b = self.cosine_distance(&b.center, embedding);
                dist_a
                    .partial_cmp(&dist_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn cosine_distance(&self, a: &[f32], b: &[f32]) -> f64 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            1.0
        } else {
            1.0 - (dot_product / (norm_a * norm_b)) as f64
        }
    }

    fn build_autocomplete_trie(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut trie = AutocompleteTrie::new();

        for (item_id, item) in &self.index.items {
            let suggestion = AutocompleteSuggestion {
                text: item_id.clone(),
                category: item.category.clone(),
                description: item.documentation.clone(),
                frequency: item.usage_count,
                relevance_score: item.popularity_score,
            };
            trie.insert(item_id, suggestion);
        }

        self.index.autocomplete_trie = trie;
        Ok(())
    }

    fn update_usage_statistics(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let scores: Vec<(String, f64)> = self
            .index
            .items
            .iter()
            .map(|(id, item)| (id.clone(), self.calculate_popularity_score(item)))
            .collect();

        for (item_id, score) in scores {
            if let Some(item) = self.index.items.get_mut(&item_id) {
                item.popularity_score = score;
            }
        }
        Ok(())
    }

    fn calculate_popularity_score(&self, item: &IndexedItem) -> f64 {
        let base_score = item.usage_count as f64;
        let documentation_score = if item.documentation.len() > 100 {
            1.5
        } else {
            1.0
        };
        let examples_score = if !item.examples.is_empty() { 1.3 } else { 1.0 };

        base_score * documentation_score * examples_score
    }

    pub fn search(
        &mut self,
        query: &SearchQuery,
    ) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
        let cache_key = self.build_cache_key(query);

        if let Some(cached_results) = self.query_cache.get(&cache_key) {
            self.performance_metrics.cache_hits += 1;
            return Ok(cached_results.clone());
        }

        let results = match query.query_type {
            SearchQueryType::Semantic => self.semantic_search(query)?,
            SearchQueryType::TypeSignature => self.type_search(query)?,
            SearchQueryType::Usage => self.usage_search(query)?,
            _ => self.general_search(query)?,
        };

        let filtered_results = self.apply_filters(&results, &query.filters);
        let ranked_results = self.rank_results(filtered_results, query);

        let final_results: Vec<SearchResult> = ranked_results
            .into_iter()
            .take(self.config.max_results)
            .collect();

        self.query_cache.insert(cache_key, final_results.clone());
        self.performance_metrics.total_queries += 1;

        Ok(final_results)
    }

    fn build_cache_key(&self, query: &SearchQuery) -> String {
        format!("{:?}", query)
    }

    fn semantic_search(
        &self,
        query: &SearchQuery,
    ) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
        let query_embedding = self.compute_embedding(&query.text);
        let mut results = Vec::new();

        for (item_id, item_embedding) in &self.index.semantic_index.embeddings {
            let similarity = 1.0 - self.cosine_distance(&query_embedding, item_embedding);

            if similarity >= self.config.similarity_threshold {
                if let Some(item) = self.index.items.get(item_id) {
                    results.push(SearchResult {
                        id: item_id.clone(),
                        title: item_id.clone(),
                        description: item.documentation.clone(),
                        category: item.category.clone(),
                        url: format!("/docs/{}", item_id),
                        score: similarity,
                        snippet: self.generate_snippet(&item.content, &query.text),
                        metadata: item.metadata.clone(),
                    });
                }
            }
        }

        Ok(results)
    }

    fn type_search(
        &self,
        query: &SearchQuery,
    ) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();

        for (type_sig, item_ids) in &self.index.type_index {
            if type_sig.contains(&query.text) {
                for item_id in item_ids {
                    if let Some(item) = self.index.items.get(item_id) {
                        results.push(SearchResult {
                            id: item_id.clone(),
                            title: item_id.clone(),
                            description: item.documentation.clone(),
                            category: item.category.clone(),
                            url: format!("/docs/{}", item_id),
                            score: self.calculate_type_match_score(type_sig, &query.text),
                            snippet: item.type_signature.clone(),
                            metadata: item.metadata.clone(),
                        });
                    }
                }
            }
        }

        Ok(results)
    }

    fn usage_search(
        &self,
        query: &SearchQuery,
    ) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();

        for (pattern, usage_info) in &self.index.usage_index {
            if pattern.contains(&query.text)
                || usage_info
                    .common_patterns
                    .iter()
                    .any(|p| p.contains(&query.text))
            {
                for context in &usage_info.contexts {
                    results.push(SearchResult {
                        id: format!("usage_{}", pattern),
                        title: format!("Usage: {}", pattern),
                        description: context.description.clone(),
                        category: ItemCategory::Function,
                        url: context.location.clone(),
                        score: usage_info.frequency as f64,
                        snippet: Some(context.snippet.clone()),
                        metadata: SearchResultMetadata {
                            crate_name: "usage".to_string(),
                            module_path: pattern.clone(),
                            line_number: None,
                            visibility: Visibility::Public,
                            stability: Stability::Stable,
                            since_version: None,
                            deprecated_since: None,
                            related_items: usage_info.related_functions.clone(),
                        },
                    });
                }
            }
        }

        Ok(results)
    }

    fn general_search(
        &self,
        query: &SearchQuery,
    ) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        let query_words: Vec<String> = query
            .text
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();

        for word in &query_words {
            if let Some(item_ids) = self.index.word_index.get(word) {
                for item_id in item_ids {
                    if let Some(item) = self.index.items.get(item_id) {
                        let score = self.calculate_text_match_score(&item.content, &query.text);

                        results.push(SearchResult {
                            id: item_id.clone(),
                            title: item_id.clone(),
                            description: item.documentation.clone(),
                            category: item.category.clone(),
                            url: format!("/docs/{}", item_id),
                            score,
                            snippet: self.generate_snippet(&item.content, &query.text),
                            metadata: item.metadata.clone(),
                        });
                    }
                }
            }
        }

        Ok(results)
    }

    fn calculate_type_match_score(&self, type_sig: &str, query: &str) -> f64 {
        if type_sig == query {
            1.0
        } else if type_sig.contains(query) {
            0.8
        } else {
            0.3
        }
    }

    fn calculate_text_match_score(&self, content: &str, query: &str) -> f64 {
        let content_lower = content.to_lowercase();
        let query_lower = query.to_lowercase();

        if content_lower.contains(&query_lower) {
            let exact_matches = content_lower.matches(&query_lower).count();
            let word_count = content.split_whitespace().count();
            (exact_matches as f64) / (word_count as f64).max(1.0)
        } else {
            0.1
        }
    }

    fn generate_snippet(&self, content: &str, query: &str) -> Option<String> {
        let query_lower = query.to_lowercase();
        let content_lower = content.to_lowercase();

        if let Some(pos) = content_lower.find(&query_lower) {
            let start = pos.saturating_sub(50);
            let end = (pos + query.len() + 50).min(content.len());
            Some(content[start..end].to_string())
        } else {
            Some(content.chars().take(100).collect())
        }
    }

    fn apply_filters(
        &self,
        results: &[SearchResult],
        filters: &SearchFilters,
    ) -> Vec<SearchResult> {
        results
            .iter()
            .filter(|result| {
                if !filters.categories.is_empty() && !filters.categories.contains(&result.category)
                {
                    return false;
                }
                if !filters.crates.is_empty()
                    && !filters.crates.contains(&result.metadata.crate_name)
                {
                    return false;
                }
                true
            })
            .cloned()
            .collect()
    }

    fn rank_results(
        &self,
        mut results: Vec<SearchResult>,
        _query: &SearchQuery,
    ) -> Vec<SearchResult> {
        results.sort_by(|a, b| {
            let score_cmp = b
                .score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal);
            if score_cmp != std::cmp::Ordering::Equal {
                return score_cmp;
            }

            a.title.cmp(&b.title)
        });

        results
    }

    pub fn get_autocomplete_suggestions(
        &self,
        prefix: &str,
        limit: usize,
    ) -> Vec<AutocompleteSuggestion> {
        self.index.autocomplete_trie.get_suggestions(prefix, limit)
    }

    pub fn get_search_metrics(&self) -> &SearchMetrics {
        &self.performance_metrics
    }
}

impl Default for SearchIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchIndex {
    pub fn new() -> Self {
        Self {
            items: HashMap::new(),
            word_index: HashMap::new(),
            type_index: HashMap::new(),
            usage_index: HashMap::new(),
            semantic_index: SemanticIndex::new(),
            autocomplete_trie: AutocompleteTrie::new(),
            last_updated: chrono::Utc::now(),
        }
    }
}

impl Default for SemanticIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticIndex {
    pub fn new() -> Self {
        Self {
            embeddings: HashMap::new(),
            clusters: Vec::new(),
            similarity_matrix: HashMap::new(),
        }
    }
}

impl Default for AutocompleteTrie {
    fn default() -> Self {
        Self::new()
    }
}

impl AutocompleteTrie {
    pub fn new() -> Self {
        Self {
            root: TrieNode::new(),
            suggestions_cache: HashMap::new(),
        }
    }

    pub fn insert(&mut self, word: &str, suggestion: AutocompleteSuggestion) {
        let mut current = &mut self.root;

        for ch in word.chars() {
            current = current.children.entry(ch).or_default();
        }

        current.is_end_of_word = true;
        current.completions.push(suggestion);
        current.frequency += 1;
    }

    pub fn get_suggestions(&self, prefix: &str, limit: usize) -> Vec<AutocompleteSuggestion> {
        if let Some(cached) = self.suggestions_cache.get(prefix) {
            return cached.iter().take(limit).cloned().collect();
        }

        let mut current = &self.root;

        for ch in prefix.chars() {
            if let Some(child) = current.children.get(&ch) {
                current = child;
            } else {
                return Vec::new();
            }
        }

        let mut suggestions = Vec::new();
        self.collect_suggestions(current, &mut suggestions);

        suggestions.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.frequency.cmp(&a.frequency))
        });

        suggestions.into_iter().take(limit).collect()
    }

    #[allow(clippy::only_used_in_recursion)]
    fn collect_suggestions(&self, node: &TrieNode, suggestions: &mut Vec<AutocompleteSuggestion>) {
        if node.is_end_of_word {
            suggestions.extend(node.completions.iter().cloned());
        }

        for child in node.children.values() {
            self.collect_suggestions(child, suggestions);
        }
    }
}

impl Default for TrieNode {
    fn default() -> Self {
        Self::new()
    }
}

impl TrieNode {
    pub fn new() -> Self {
        Self {
            value: None,
            children: HashMap::new(),
            is_end_of_word: false,
            completions: Vec::new(),
            frequency: 0,
        }
    }
}

impl Default for SearchMetrics {
    fn default() -> Self {
        Self {
            total_queries: 0,
            cache_hits: 0,
            average_response_time: 0.0,
            index_size: 0,
            last_reindex_time: chrono::Utc::now(),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_engine_creation() {
        let config = SearchEngineConfig::default();
        let engine = SemanticSearchEngine::new(config);
        assert_eq!(engine.config.max_results, 50);
    }

    #[test]
    fn test_autocomplete_trie() {
        let mut trie = AutocompleteTrie::new();
        let suggestion = AutocompleteSuggestion {
            text: "test".to_string(),
            category: ItemCategory::Function,
            description: "Test function".to_string(),
            frequency: 1,
            relevance_score: 1.0,
        };

        trie.insert("test", suggestion);
        let suggestions = trie.get_suggestions("te", 10);
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0].text, "test");
    }

    #[test]
    fn test_search_query_creation() {
        let query = SearchQuery {
            text: "linear regression".to_string(),
            query_type: SearchQueryType::Semantic,
            filters: SearchFilters::default(),
            options: SearchOptions::default(),
        };

        assert_eq!(query.text, "linear regression");
        assert!(matches!(query.query_type, SearchQueryType::Semantic));
    }

    #[test]
    fn test_cosine_distance() {
        let engine = SemanticSearchEngine::new(SearchEngineConfig::default());
        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![0.0, 1.0, 0.0];

        let distance = engine.cosine_distance(&vec1, &vec2);
        assert!((distance - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_computation() {
        let engine = SemanticSearchEngine::new(SearchEngineConfig::default());
        let embedding = engine.compute_embedding("test string");
        assert_eq!(embedding.len(), 300);
    }
}
