//! Natural Language Processing Interpretability Methods
//!
//! This module provides specialized interpretability methods for text and NLP models,
//! including text-specific LIME, attention visualization, word importance analysis,
//! syntactic explanations, and semantic similarity-based explanations.

use crate::{types::Float, SklResult, SklearsError};
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::Rng;
use std::collections::HashMap;

/// Configuration for NLP interpretability methods
#[derive(Debug, Clone)]
pub struct NLPConfig {
    /// Number of perturbations to generate for LIME
    pub n_perturbations: usize,
    /// Vocabulary size for tokenization
    pub vocab_size: Option<usize>,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Whether to use attention weights if available
    pub use_attention: bool,
    /// Minimum token frequency for inclusion
    pub min_token_frequency: usize,
    /// Whether to include syntactic features
    pub include_syntax: bool,
    /// Semantic similarity threshold
    pub similarity_threshold: Float,
}

impl Default for NLPConfig {
    fn default() -> Self {
        Self {
            n_perturbations: 1000,
            vocab_size: None,
            max_sequence_length: 512,
            use_attention: true,
            min_token_frequency: 2,
            include_syntax: false,
            similarity_threshold: 0.7,
        }
    }
}

/// Token with metadata for NLP explanations
#[derive(Debug, Clone)]
pub struct Token {
    /// The text content of the token
    pub text: String,
    /// Position in the original sequence
    pub position: usize,
    /// Part-of-speech tag if available
    pub pos_tag: Option<String>,
    /// Named entity tag if available
    pub ner_tag: Option<String>,
    /// Syntactic dependency relation
    pub dependency: Option<String>,
}

/// Word importance result for individual tokens
#[derive(Debug, Clone)]
pub struct WordImportanceResult {
    /// Tokens in the input
    pub tokens: Vec<Token>,
    /// Importance scores for each token
    pub importance_scores: Array1<Float>,
    /// Confidence intervals for importance scores
    pub confidence_intervals: Option<Array2<Float>>,
    /// Attention weights if available
    pub attention_weights: Option<Array1<Float>>,
}

/// Attention analysis result
#[derive(Debug, Clone)]
pub struct AttentionExplanation {
    /// Attention weights matrix (layer x head x sequence x sequence)
    pub attention_weights: Array2<Float>,
    /// Token-to-token attention scores
    pub token_attention: HashMap<(usize, usize), Float>,
    /// Head-level attention patterns
    pub head_patterns: Vec<AttentionPattern>,
    /// Layer-wise attention summary
    pub layer_summary: Vec<Float>,
}

/// Attention pattern in a specific head
#[derive(Debug, Clone)]
pub struct AttentionPattern {
    /// Head identifier
    pub head_id: usize,
    /// Layer identifier  
    pub layer_id: usize,
    /// Pattern type (e.g., positional, syntactic, semantic)
    pub pattern_type: AttentionPatternType,
    /// Strength of the pattern
    pub strength: Float,
    /// Representative token pairs
    pub representative_pairs: Vec<(usize, usize, Float)>,
}

/// Types of attention patterns
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttentionPatternType {
    /// Positional attention (attending to nearby tokens)
    Positional,
    /// Syntactic attention (attending to syntactically related tokens)
    Syntactic,
    /// Semantic attention (attending to semantically similar tokens)
    Semantic,
    /// Long-range dependencies
    LongRange,
    /// Unknown or mixed pattern
    Unknown,
}

/// Syntactic explanation result
#[derive(Debug, Clone)]
pub struct SyntacticExplanation {
    /// Syntactic tree structure
    pub parse_tree: SyntacticTree,
    /// Importance of syntactic relations
    pub relation_importance: HashMap<String, Float>,
    /// Contribution of syntactic features to prediction
    pub syntactic_contributions: Array1<Float>,
    /// Most important syntactic patterns
    pub key_patterns: Vec<SyntacticPattern>,
}

/// Syntactic tree structure
#[derive(Debug, Clone)]
pub struct SyntacticTree {
    /// Tree nodes representing tokens and their relations
    pub nodes: Vec<SyntacticNode>,
    /// Edges representing dependencies
    pub edges: Vec<(usize, usize, String)>,
}

/// Node in syntactic tree
#[derive(Debug, Clone)]
pub struct SyntacticNode {
    /// Token index
    pub token_id: usize,
    /// Token text
    pub text: String,
    /// Part-of-speech tag
    pub pos: String,
    /// Head token index (parent in dependency tree)
    pub head: Option<usize>,
    /// Dependency relation to head
    pub relation: Option<String>,
}

/// Syntactic pattern
#[derive(Debug, Clone)]
pub struct SyntacticPattern {
    /// Pattern description
    pub description: String,
    /// Tokens involved in the pattern
    pub tokens: Vec<usize>,
    /// Importance score
    pub importance: Float,
    /// Pattern frequency
    pub frequency: usize,
}

/// Semantic similarity explanation
#[derive(Debug, Clone)]
pub struct SemanticExplanation {
    /// Word embeddings for tokens
    pub embeddings: Array2<Float>,
    /// Semantic similarity matrix between tokens
    pub similarity_matrix: Array2<Float>,
    /// Semantic clusters
    pub clusters: Vec<SemanticCluster>,
    /// Most influential semantic relations
    pub key_relations: Vec<SemanticRelation>,
}

/// Semantic cluster of related tokens
#[derive(Debug, Clone)]
pub struct SemanticCluster {
    /// Cluster identifier
    pub cluster_id: usize,
    /// Tokens in the cluster
    pub tokens: Vec<usize>,
    /// Cluster centroid
    pub centroid: Array1<Float>,
    /// Cluster coherence score
    pub coherence: Float,
}

/// Semantic relation between tokens
#[derive(Debug, Clone)]
pub struct SemanticRelation {
    /// First token
    pub token1: usize,
    /// Second token
    pub token2: usize,
    /// Similarity score
    pub similarity: Float,
    /// Relation type (e.g., synonym, antonym, hypernym)
    pub relation_type: String,
}

/// Text-specific LIME explanation
pub fn explain_text_with_lime<F>(
    text: &str,
    model_fn: F,
    config: &NLPConfig,
) -> SklResult<WordImportanceResult>
where
    F: Fn(&[String]) -> SklResult<Array1<Float>>,
{
    // Tokenize the input text
    let tokens = tokenize_text(text)?;

    if tokens.is_empty() {
        return Err(SklearsError::InvalidInput("Empty text input".to_string()));
    }

    // Generate perturbations by masking different tokens
    let mut perturbations = Vec::new();
    let mut labels = Vec::new();

    // Original prediction
    let original_tokens: Vec<String> = tokens.iter().map(|t| t.text.clone()).collect();
    let original_pred = model_fn(&original_tokens)?;

    // Generate perturbations
    for _ in 0..config.n_perturbations {
        let perturbed_tokens = generate_text_perturbation(&tokens, 0.3)?;
        let token_strings: Vec<String> = perturbed_tokens.iter().map(|t| t.text.clone()).collect();
        let pred = model_fn(&token_strings)?;

        perturbations.push(perturbed_tokens);
        labels.push(pred);
    }

    // Compute importance scores using linear model
    let importance_scores =
        compute_text_lime_weights(&tokens, &perturbations, &labels, &original_pred)?;

    Ok(WordImportanceResult {
        tokens,
        importance_scores,
        confidence_intervals: None,
        attention_weights: None,
    })
}

/// Analyze attention patterns in text
pub fn analyze_text_attention(
    attention_weights: &ArrayView2<Float>,
    tokens: &[Token],
    config: &NLPConfig,
) -> SklResult<AttentionExplanation> {
    if attention_weights.is_empty() {
        return Err(SklearsError::InvalidInput(
            "Empty attention weights".to_string(),
        ));
    }

    // Analyze token-to-token attention
    let mut token_attention = HashMap::new();
    for i in 0..attention_weights.nrows() {
        for j in 0..attention_weights.ncols() {
            if i < tokens.len() && j < tokens.len() {
                token_attention.insert((i, j), attention_weights[[i, j]]);
            }
        }
    }

    // Identify attention patterns
    let head_patterns = identify_attention_patterns(attention_weights, tokens)?;

    // Compute layer-wise summary
    let layer_summary = vec![attention_weights.mean().unwrap_or(0.0)];

    Ok(AttentionExplanation {
        attention_weights: attention_weights.to_owned(),
        token_attention,
        head_patterns,
        layer_summary,
    })
}

/// Visualize word importance
pub fn visualize_word_importance(
    result: &WordImportanceResult,
    config: &NLPConfig,
) -> SklResult<String> {
    let mut html = String::new();
    html.push_str("<div class=\"word-importance-visualization\">\n");

    for (i, token) in result.tokens.iter().enumerate() {
        let importance = result.importance_scores[i];
        let color_intensity = (importance.abs() * 255.0) as u8;
        let color = if importance >= 0.0 {
            format!("rgba(0, 255, 0, {})", importance.abs())
        } else {
            format!("rgba(255, 0, 0, {})", importance.abs())
        };

        html.push_str(&format!(
            "<span style=\"background-color: {}; padding: 2px; margin: 1px;\" title=\"Importance: {:.3}\">{}</span>\n",
            color, importance, token.text
        ));
    }

    html.push_str("</div>");
    Ok(html)
}

/// Generate syntactic explanations
pub fn explain_syntax(
    tokens: &[Token],
    predictions: &ArrayView1<Float>,
    config: &NLPConfig,
) -> SklResult<SyntacticExplanation> {
    if !config.include_syntax {
        return Err(SklearsError::InvalidInput(
            "Syntax analysis not enabled".to_string(),
        ));
    }

    // Build syntactic tree
    let parse_tree = build_syntactic_tree(tokens)?;

    // Compute relation importance
    let mut relation_importance = HashMap::new();
    for edge in &parse_tree.edges {
        let relation = &edge.2;
        let contribution = predictions.mean().unwrap_or(0.0);
        relation_importance.insert(relation.clone(), contribution);
    }

    // Compute syntactic contributions
    let syntactic_contributions = Array1::zeros(tokens.len());

    // Identify key patterns
    let key_patterns = identify_syntactic_patterns(&parse_tree, predictions)?;

    Ok(SyntacticExplanation {
        parse_tree,
        relation_importance,
        syntactic_contributions,
        key_patterns,
    })
}

/// Generate semantic similarity explanations
pub fn explain_semantic_similarity(
    tokens: &[Token],
    embeddings: &ArrayView2<Float>,
    config: &NLPConfig,
) -> SklResult<SemanticExplanation> {
    if embeddings.nrows() != tokens.len() {
        return Err(SklearsError::InvalidInput(
            "Embeddings and tokens length mismatch".to_string(),
        ));
    }

    // Compute similarity matrix
    let similarity_matrix = compute_similarity_matrix(embeddings)?;

    // Find semantic clusters
    let clusters = find_semantic_clusters(embeddings, &similarity_matrix, config)?;

    // Identify key semantic relations
    let key_relations = identify_semantic_relations(&similarity_matrix, config)?;

    Ok(SemanticExplanation {
        embeddings: embeddings.to_owned(),
        similarity_matrix,
        clusters,
        key_relations,
    })
}

// Helper functions

/// Tokenize text into tokens with metadata
fn tokenize_text(text: &str) -> SklResult<Vec<Token>> {
    let mut tokens = Vec::new();

    // Simple whitespace tokenization (in practice, would use a proper tokenizer)
    for (i, word) in text.split_whitespace().enumerate() {
        tokens.push(Token {
            text: word.to_string(),
            position: i,
            pos_tag: None,
            ner_tag: None,
            dependency: None,
        });
    }

    Ok(tokens)
}

/// Generate text perturbation by randomly masking tokens
fn generate_text_perturbation(tokens: &[Token], mask_probability: Float) -> SklResult<Vec<Token>> {
    let mut perturbed = tokens.to_vec();

    for token in &mut perturbed {
        if scirs2_core::random::thread_rng().gen::<Float>() < mask_probability {
            token.text = "[MASK]".to_string();
        }
    }

    Ok(perturbed)
}

/// Compute LIME weights for text explanation
fn compute_text_lime_weights(
    original_tokens: &[Token],
    perturbations: &[Vec<Token>],
    predictions: &[Array1<Float>],
    original_pred: &Array1<Float>,
) -> SklResult<Array1<Float>> {
    let n_tokens = original_tokens.len();

    if n_tokens == 0 {
        return Err(SklearsError::InvalidInput("No tokens provided".to_string()));
    }

    // Create feature matrix indicating which tokens are present
    let mut feature_matrix = Array2::zeros((perturbations.len(), n_tokens));

    for (i, perturbed) in perturbations.iter().enumerate() {
        for (j, original_token) in original_tokens.iter().enumerate() {
            if j < perturbed.len() && perturbed[j].text == original_token.text {
                feature_matrix[[i, j]] = 1.0;
            }
        }
    }

    // Solve linear regression to find token importance
    let weights = solve_lime_regression(&feature_matrix, predictions, original_pred)?;

    Ok(weights)
}

/// Solve linear regression for LIME weights
fn solve_lime_regression(
    X: &Array2<Float>,
    y: &[Array1<Float>],
    baseline: &Array1<Float>,
) -> SklResult<Array1<Float>> {
    let n_features = X.ncols();

    if n_features == 0 || y.is_empty() {
        return Ok(Array1::zeros(n_features));
    }

    // Simple approximation: compute correlation between features and target change
    let mut weights = Array1::zeros(n_features);

    for j in 0..n_features {
        let mut correlation = 0.0;
        let mut count = 0;

        for (i, pred) in y.iter().enumerate() {
            if i < X.nrows() {
                let feature_value = X[[i, j]];
                let target_change = (pred[0] - baseline[0]).abs();
                correlation += feature_value * target_change;
                count += 1;
            }
        }

        if count > 0 {
            weights[j] = correlation / count as Float;
        }
    }

    Ok(weights)
}

/// Identify attention patterns
fn identify_attention_patterns(
    attention_weights: &ArrayView2<Float>,
    tokens: &[Token],
) -> SklResult<Vec<AttentionPattern>> {
    let mut patterns = Vec::new();

    // Simple pattern detection based on attention distribution
    let mean_attention = attention_weights.mean().unwrap_or(0.0);
    let std_attention = attention_weights.var(0.0).sqrt();

    // Look for positional patterns (attention to nearby tokens)
    let mut positional_strength = 0.0;
    let mut positional_pairs = Vec::new();

    for i in 0..attention_weights.nrows().min(tokens.len()) {
        for j in 0..attention_weights.ncols().min(tokens.len()) {
            let attention = attention_weights[[i, j]];
            let distance = (i as isize - j as isize).abs() as Float;

            if distance <= 3.0 && attention > mean_attention + std_attention {
                positional_strength += attention;
                positional_pairs.push((i, j, attention));
            }
        }
    }

    if positional_strength > 0.0 {
        patterns.push(AttentionPattern {
            head_id: 0,
            layer_id: 0,
            pattern_type: AttentionPatternType::Positional,
            strength: positional_strength / positional_pairs.len() as Float,
            representative_pairs: positional_pairs,
        });
    }

    Ok(patterns)
}

/// Build syntactic tree from tokens
fn build_syntactic_tree(tokens: &[Token]) -> SklResult<SyntacticTree> {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    // Create nodes
    for (i, token) in tokens.iter().enumerate() {
        nodes.push(SyntacticNode {
            token_id: i,
            text: token.text.clone(),
            pos: token.pos_tag.clone().unwrap_or("UNK".to_string()),
            head: None,
            relation: token.dependency.clone(),
        });
    }

    // Create simple linear dependencies (in practice, would use a parser)
    for i in 1..tokens.len() {
        edges.push((i - 1, i, "next".to_string()));
    }

    Ok(SyntacticTree { nodes, edges })
}

/// Identify syntactic patterns
fn identify_syntactic_patterns(
    tree: &SyntacticTree,
    predictions: &ArrayView1<Float>,
) -> SklResult<Vec<SyntacticPattern>> {
    let mut patterns = Vec::new();

    // Simple pattern: sequences of specific POS tags
    let mut pos_sequence = Vec::new();
    for node in &tree.nodes {
        pos_sequence.push(node.pos.clone());
    }

    // Find common patterns
    if pos_sequence.len() >= 2 {
        let importance = predictions.mean().unwrap_or(0.0);
        patterns.push(SyntacticPattern {
            description: format!("POS sequence: {:?}", pos_sequence),
            tokens: (0..tree.nodes.len()).collect(),
            importance,
            frequency: 1,
        });
    }

    Ok(patterns)
}

/// Compute semantic similarity matrix
fn compute_similarity_matrix(embeddings: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
    let n_tokens = embeddings.nrows();
    let mut similarity = Array2::zeros((n_tokens, n_tokens));

    for i in 0..n_tokens {
        for j in 0..n_tokens {
            if i == j {
                similarity[[i, j]] = 1.0;
            } else {
                // Cosine similarity
                let embedding_i = embeddings.row(i);
                let embedding_j = embeddings.row(j);

                let dot_product = embedding_i.dot(&embedding_j);
                let norm_i = embedding_i.dot(&embedding_i).sqrt();
                let norm_j = embedding_j.dot(&embedding_j).sqrt();

                if norm_i > 0.0 && norm_j > 0.0 {
                    similarity[[i, j]] = dot_product / (norm_i * norm_j);
                }
            }
        }
    }

    Ok(similarity)
}

/// Find semantic clusters
fn find_semantic_clusters(
    embeddings: &ArrayView2<Float>,
    similarity: &Array2<Float>,
    config: &NLPConfig,
) -> SklResult<Vec<SemanticCluster>> {
    let n_tokens = embeddings.nrows();
    let mut clusters = Vec::new();
    let mut assigned = vec![false; n_tokens];

    for i in 0..n_tokens {
        if assigned[i] {
            continue;
        }

        let mut cluster_tokens = vec![i];
        assigned[i] = true;

        // Find similar tokens
        for j in (i + 1)..n_tokens {
            if !assigned[j] && similarity[[i, j]] > config.similarity_threshold {
                cluster_tokens.push(j);
                assigned[j] = true;
            }
        }

        // Compute centroid
        let mut centroid = Array1::zeros(embeddings.ncols());
        for &token_id in &cluster_tokens {
            centroid += &embeddings.row(token_id);
        }
        centroid /= cluster_tokens.len() as Float;

        // Compute coherence (average intra-cluster similarity)
        let mut coherence = 0.0;
        let mut count = 0;
        for &token1 in &cluster_tokens {
            for &token2 in &cluster_tokens {
                if token1 != token2 {
                    coherence += similarity[[token1, token2]];
                    count += 1;
                }
            }
        }
        if count > 0 {
            coherence /= count as Float;
        }

        clusters.push(SemanticCluster {
            cluster_id: clusters.len(),
            tokens: cluster_tokens,
            centroid,
            coherence,
        });
    }

    Ok(clusters)
}

/// Identify key semantic relations
fn identify_semantic_relations(
    similarity: &Array2<Float>,
    config: &NLPConfig,
) -> SklResult<Vec<SemanticRelation>> {
    let mut relations = Vec::new();

    for i in 0..similarity.nrows() {
        for j in (i + 1)..similarity.ncols() {
            let sim = similarity[[i, j]];
            if sim > config.similarity_threshold {
                let relation_type = if sim > 0.9 {
                    "synonym".to_string()
                } else if sim > 0.8 {
                    "related".to_string()
                } else {
                    "similar".to_string()
                };

                relations.push(SemanticRelation {
                    token1: i,
                    token2: j,
                    similarity: sim,
                    relation_type,
                });
            }
        }
    }

    // Sort by similarity score
    relations.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());

    Ok(relations)
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;

    #[test]
    fn test_tokenize_text() {
        let text = "Hello world test";
        let tokens = tokenize_text(text).unwrap();

        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "Hello");
        assert_eq!(tokens[1].text, "world");
        assert_eq!(tokens[2].text, "test");
        assert_eq!(tokens[0].position, 0);
        assert_eq!(tokens[2].position, 2);
    }

    #[test]
    fn test_text_perturbation() {
        let tokens = vec![
            Token {
                text: "Hello".to_string(),
                position: 0,
                pos_tag: None,
                ner_tag: None,
                dependency: None,
            },
            Token {
                text: "world".to_string(),
                position: 1,
                pos_tag: None,
                ner_tag: None,
                dependency: None,
            },
        ];

        let perturbed = generate_text_perturbation(&tokens, 0.5).unwrap();
        assert_eq!(perturbed.len(), 2);
        // Some tokens might be masked as [MASK]
    }

    #[test]
    fn test_attention_analysis() {
        let attention = array![[0.8, 0.2], [0.3, 0.7]];
        let tokens = vec![
            Token {
                text: "Hello".to_string(),
                position: 0,
                pos_tag: None,
                ner_tag: None,
                dependency: None,
            },
            Token {
                text: "world".to_string(),
                position: 1,
                pos_tag: None,
                ner_tag: None,
                dependency: None,
            },
        ];

        let config = NLPConfig::default();
        let result = analyze_text_attention(&attention.view(), &tokens, &config).unwrap();

        assert_eq!(result.token_attention.len(), 4); // 2x2 matrix
        assert_eq!(result.layer_summary.len(), 1);
    }

    #[test]
    fn test_word_importance_visualization() {
        let tokens = vec![
            Token {
                text: "Hello".to_string(),
                position: 0,
                pos_tag: None,
                ner_tag: None,
                dependency: None,
            },
            Token {
                text: "world".to_string(),
                position: 1,
                pos_tag: None,
                ner_tag: None,
                dependency: None,
            },
        ];
        let importance_scores = array![0.8, -0.3];

        let result = WordImportanceResult {
            tokens,
            importance_scores,
            confidence_intervals: None,
            attention_weights: None,
        };

        let config = NLPConfig::default();
        let html = visualize_word_importance(&result, &config).unwrap();

        assert!(html.contains("word-importance-visualization"));
        assert!(html.contains("Hello"));
        assert!(html.contains("world"));
    }

    #[test]
    fn test_similarity_matrix_computation() {
        let embeddings = array![[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]];
        let similarity = compute_similarity_matrix(&embeddings.view()).unwrap();

        assert_eq!(similarity.shape(), &[3, 3]);
        // First and third vectors are identical
        assert!((similarity[[0, 2]] - 1.0).abs() < 1e-6);
        // First and second vectors are orthogonal
        assert!(similarity[[0, 1]].abs() < 1e-6);
    }

    #[test]
    fn test_semantic_clustering() {
        let embeddings = array![[1.0, 0.0], [1.1, 0.1], [0.0, 1.0]];
        let similarity = compute_similarity_matrix(&embeddings.view()).unwrap();

        let config = NLPConfig {
            similarity_threshold: 0.8,
            ..Default::default()
        };
        let clusters = find_semantic_clusters(&embeddings.view(), &similarity, &config).unwrap();

        assert!(!clusters.is_empty());
        // First two embeddings should be clustered together
    }

    #[test]
    fn test_syntactic_tree_building() {
        let tokens = vec![
            Token {
                text: "The".to_string(),
                position: 0,
                pos_tag: Some("DT".to_string()),
                ner_tag: None,
                dependency: None,
            },
            Token {
                text: "cat".to_string(),
                position: 1,
                pos_tag: Some("NN".to_string()),
                ner_tag: None,
                dependency: None,
            },
        ];

        let tree = build_syntactic_tree(&tokens).unwrap();

        assert_eq!(tree.nodes.len(), 2);
        assert_eq!(tree.edges.len(), 1);
        assert_eq!(tree.nodes[0].pos, "DT");
        assert_eq!(tree.nodes[1].pos, "NN");
    }

    #[test]
    fn test_nlp_config_default() {
        let config = NLPConfig::default();

        assert_eq!(config.n_perturbations, 1000);
        assert_eq!(config.max_sequence_length, 512);
        assert!(config.use_attention);
        assert_eq!(config.min_token_frequency, 2);
        assert!(!config.include_syntax);
        assert_eq!(config.similarity_threshold, 0.7);
    }

    #[test]
    fn test_text_lime_explanation() {
        let text = "Hello world";
        let config = NLPConfig {
            n_perturbations: 10,
            ..Default::default()
        };

        // Mock model function
        let model_fn = |tokens: &[String]| -> SklResult<Array1<Float>> {
            // Simple model: count tokens
            Ok(array![tokens.len() as Float])
        };

        let result = explain_text_with_lime(text, model_fn, &config).unwrap();

        assert_eq!(result.tokens.len(), 2);
        assert_eq!(result.importance_scores.len(), 2);
    }

    #[test]
    fn test_semantic_relations_identification() {
        let similarity = array![[1.0, 0.9, 0.2], [0.9, 1.0, 0.3], [0.2, 0.3, 1.0]];
        let config = NLPConfig {
            similarity_threshold: 0.8,
            ..Default::default()
        };

        let relations = identify_semantic_relations(&similarity, &config).unwrap();

        assert!(!relations.is_empty());
        // Should find the high similarity between tokens 0 and 1
        assert!(relations
            .iter()
            .any(|r| r.token1 == 0 && r.token2 == 1 && r.similarity == 0.9));
    }

    #[test]
    fn test_attention_pattern_identification() {
        let attention = array![[0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.1, 0.1, 0.8]];
        let tokens = vec![
            Token {
                text: "Hello".to_string(),
                position: 0,
                pos_tag: None,
                ner_tag: None,
                dependency: None,
            },
            Token {
                text: "world".to_string(),
                position: 1,
                pos_tag: None,
                ner_tag: None,
                dependency: None,
            },
            Token {
                text: "test".to_string(),
                position: 2,
                pos_tag: None,
                ner_tag: None,
                dependency: None,
            },
        ];

        let patterns = identify_attention_patterns(&attention.view(), &tokens).unwrap();

        assert!(!patterns.is_empty());
        // Should identify positional pattern
        assert!(patterns
            .iter()
            .any(|p| p.pattern_type == AttentionPatternType::Positional));
    }
}
