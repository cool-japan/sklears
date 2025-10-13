//! Natural Language Processing Metrics
//!
//! This module provides comprehensive metrics for natural language processing tasks including
//! machine translation evaluation (BLEU), summarization evaluation (ROUGE), language model
//! evaluation (perplexity), and semantic similarity metrics.

use crate::{MetricsError, MetricsResult};
use scirs2_core::numeric::Float;
use std::collections::{HashMap, HashSet};

/// BLEU Score for Machine Translation Evaluation
///
/// Bilingual Evaluation Understudy (BLEU) measures the correspondence between
/// a machine translation and human reference translations.
///
/// # Arguments
/// * `candidate` - The candidate (machine) translation as a vector of tokens
/// * `references` - Vector of reference translations, each as a vector of tokens
/// * `weights` - Weights for n-gram precisions (default: [0.25, 0.25, 0.25, 0.25])
/// * `smoothing_function` - Smoothing method for zero n-gram matches
///
/// # Returns
/// BLEU score between 0 and 1, where 1 indicates perfect match.
///
/// # Examples
/// ```
/// use sklears_metrics::nlp::{bleu_score, SmoothingFunction};
///
/// let candidate = vec!["the", "cat", "is", "on", "the", "mat"];
/// let references = vec![
///     vec!["the", "cat", "is", "on", "the", "mat"],
///     vec!["there", "is", "a", "cat", "on", "the", "mat"]
/// ];
/// let weights = [0.25, 0.25, 0.25, 0.25];
/// let score = bleu_score(&candidate, &references, &weights, SmoothingFunction::Method1).unwrap();
/// assert!(score > 0.0);
/// ```
pub fn bleu_score<T>(
    candidate: &[&str],
    references: &[Vec<&str>],
    weights: &[T; 4],
    smoothing_function: SmoothingFunction,
) -> MetricsResult<T>
where
    T: Float + std::fmt::Debug,
{
    if references.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    if candidate.is_empty() {
        return Ok(T::zero());
    }

    // Calculate n-gram precisions for n=1,2,3,4
    let mut precisions = Vec::new();
    let mut _total_candidate_ngrams = 0;

    for n in 1..=4 {
        let candidate_ngrams = extract_ngrams(candidate, n);
        let mut reference_ngrams_counts = HashMap::new();

        // Collect all reference n-grams with maximum counts
        for reference in references {
            let ref_ngrams = extract_ngrams(reference, n);
            for (ngram, count) in ref_ngrams {
                let current_max = reference_ngrams_counts.get(&ngram).unwrap_or(&0);
                reference_ngrams_counts.insert(ngram, (*current_max).max(count));
            }
        }

        // Count matches
        let mut matches = 0;
        let candidate_ngram_count: usize = candidate_ngrams.values().sum();

        for (ngram, candidate_count) in candidate_ngrams {
            if let Some(&ref_count) = reference_ngrams_counts.get(&ngram) {
                matches += candidate_count.min(ref_count);
            }
        }

        if n == 1 {
            _total_candidate_ngrams = candidate_ngram_count;
        }

        // Apply smoothing if needed
        let precision = if candidate_ngram_count == 0 {
            T::zero()
        } else if matches == 0 {
            match smoothing_function {
                SmoothingFunction::None => T::zero(),
                SmoothingFunction::Method1 => {
                    // Add 1 to numerator and denominator
                    T::one() / T::from(candidate_ngram_count + 1).unwrap()
                }
                SmoothingFunction::Method2 => {
                    // Use 1/2^n for zero matches
                    T::one() / T::from(2_usize.pow(n as u32)).unwrap()
                }
            }
        } else {
            T::from(matches).unwrap() / T::from(candidate_ngram_count).unwrap()
        };

        precisions.push(precision);
    }

    // Calculate geometric mean of precisions
    let log_precision_sum = precisions
        .iter()
        .zip(weights.iter())
        .map(|(&p, &w)| {
            if p > T::zero() {
                w * p.ln()
            } else {
                T::from(-f64::INFINITY).unwrap()
            }
        })
        .fold(T::zero(), |acc, x| acc + x);

    if log_precision_sum.is_infinite() && log_precision_sum < T::zero() {
        return Ok(T::zero());
    }

    let geometric_mean = log_precision_sum.exp();

    // Calculate brevity penalty
    let candidate_length = T::from(candidate.len()).unwrap();
    let closest_reference_length = find_closest_reference_length(references, candidate.len());
    let reference_length = T::from(closest_reference_length).unwrap();

    let brevity_penalty = if candidate_length >= reference_length {
        T::one()
    } else if candidate_length == T::zero() {
        T::zero()
    } else {
        (T::one() - reference_length / candidate_length).exp()
    };

    Ok(brevity_penalty * geometric_mean)
}

/// Smoothing functions for BLEU score calculation
#[derive(Debug, Clone, Copy)]
pub enum SmoothingFunction {
    None,
    /// Method1
    Method1, // Add 1 smoothing
    /// Method2
    Method2, // Exponential decay smoothing
}

/// Extract n-grams from a sequence of tokens
fn extract_ngrams(tokens: &[&str], n: usize) -> HashMap<Vec<String>, usize> {
    let mut ngrams = HashMap::new();

    if tokens.len() < n {
        return ngrams;
    }

    for i in 0..=tokens.len() - n {
        let ngram: Vec<String> = tokens[i..i + n].iter().map(|s| s.to_string()).collect();
        *ngrams.entry(ngram).or_insert(0) += 1;
    }

    ngrams
}

/// Find the reference length closest to candidate length
fn find_closest_reference_length(references: &[Vec<&str>], candidate_length: usize) -> usize {
    references
        .iter()
        .map(|r| r.len())
        .min_by_key(|&len| (len as i32 - candidate_length as i32).abs())
        .unwrap_or(0)
}

/// ROUGE-N Score for Summarization Evaluation
///
/// Recall-Oriented Understudy for Gisting Evaluation (ROUGE) measures
/// the overlap between system summaries and reference summaries.
///
/// # Arguments
/// * `system_summary` - System-generated summary as tokens
/// * `reference_summaries` - Reference summaries as vectors of tokens
/// * `n` - N-gram size (1 for ROUGE-1, 2 for ROUGE-2, etc.)
///
/// # Returns
/// ROUGE-N score between 0 and 1.
pub fn rouge_n_score<T>(
    system_summary: &[&str],
    reference_summaries: &[Vec<&str>],
    n: usize,
) -> MetricsResult<T>
where
    T: Float + std::fmt::Debug,
{
    if reference_summaries.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    if n == 0 {
        return Err(MetricsError::InvalidParameter(
            "N-gram size must be greater than 0".to_string(),
        ));
    }

    let system_ngrams = extract_ngrams(system_summary, n);

    let mut total_reference_ngrams = 0;
    let mut total_overlapping_ngrams = 0;

    for reference in reference_summaries {
        let reference_ngrams = extract_ngrams(reference, n);
        let reference_ngram_count: usize = reference_ngrams.values().sum();
        total_reference_ngrams += reference_ngram_count;

        // Count overlapping n-grams
        for (ngram, &system_count) in &system_ngrams {
            if let Some(&ref_count) = reference_ngrams.get(ngram) {
                total_overlapping_ngrams += system_count.min(ref_count);
            }
        }
    }

    if total_reference_ngrams == 0 {
        return Ok(T::zero());
    }

    Ok(T::from(total_overlapping_ngrams).unwrap() / T::from(total_reference_ngrams).unwrap())
}

/// ROUGE-L Score using Longest Common Subsequence
///
/// ROUGE-L measures the longest common subsequence between system and reference summaries.
///
/// # Arguments
/// * `system_summary` - System-generated summary as tokens
/// * `reference_summary` - Reference summary as tokens
///
/// # Returns
/// ROUGE-L F1 score between 0 and 1.
pub fn rouge_l_score<T>(system_summary: &[&str], reference_summary: &[&str]) -> MetricsResult<T>
where
    T: Float + std::fmt::Debug,
{
    if system_summary.is_empty() || reference_summary.is_empty() {
        return Ok(T::zero());
    }

    let lcs_length = longest_common_subsequence_length(system_summary, reference_summary);

    if lcs_length == 0 {
        return Ok(T::zero());
    }

    let lcs_f = T::from(lcs_length).unwrap();
    let system_len = T::from(system_summary.len()).unwrap();
    let reference_len = T::from(reference_summary.len()).unwrap();

    let precision = lcs_f / system_len;
    let recall = lcs_f / reference_len;

    if precision + recall == T::zero() {
        return Ok(T::zero());
    }

    // F1 score
    Ok((T::from(2.0).unwrap() * precision * recall) / (precision + recall))
}

/// Calculate longest common subsequence length
fn longest_common_subsequence_length(seq1: &[&str], seq2: &[&str]) -> usize {
    let m = seq1.len();
    let n = seq2.len();

    let mut dp = vec![vec![0; n + 1]; m + 1];

    for i in 1..=m {
        for j in 1..=n {
            if seq1[i - 1] == seq2[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    dp[m][n]
}

/// Perplexity calculation for language models
///
/// Perplexity measures how well a probability model predicts a sample.
/// Lower perplexity indicates better model performance.
///
/// # Arguments
/// * `log_probabilities` - Log probabilities assigned by the model to each token
///
/// # Returns
/// Perplexity value. Lower values indicate better model performance.
///
/// # Examples
/// ```
/// use sklears_metrics::nlp::perplexity;
///
/// let log_probs = vec![-1.2, -0.8, -1.5, -0.9, -1.1];
/// let perp = perplexity(&log_probs).unwrap();
/// assert!(perp > 0.0);
/// ```
pub fn perplexity<T>(log_probabilities: &[T]) -> MetricsResult<T>
where
    T: Float + std::fmt::Debug,
{
    if log_probabilities.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Calculate average log probability
    let sum_log_prob = log_probabilities.iter().fold(T::zero(), |acc, &x| acc + x);
    let avg_log_prob = sum_log_prob / T::from(log_probabilities.len()).unwrap();

    // Perplexity = exp(-avg_log_prob)
    Ok((-avg_log_prob).exp())
}

/// Jaccard similarity coefficient for text
///
/// Measures similarity between two texts based on the intersection and union of their token sets.
///
/// # Arguments
/// * `text1` - First text as tokens
/// * `text2` - Second text as tokens
///
/// # Returns
/// Jaccard similarity between 0 and 1.
pub fn jaccard_similarity<T>(text1: &[&str], text2: &[&str]) -> MetricsResult<T>
where
    T: Float + std::fmt::Debug,
{
    let set1: HashSet<&str> = text1.iter().cloned().collect();
    let set2: HashSet<&str> = text2.iter().cloned().collect();

    let intersection = set1.intersection(&set2).count();
    let union = set1.union(&set2).count();

    if union == 0 {
        return Ok(T::one()); // Both sets are empty, perfect similarity
    }

    Ok(T::from(intersection).unwrap() / T::from(union).unwrap())
}

/// Cosine similarity for text using TF-IDF vectors
///
/// Calculates cosine similarity between two texts using Term Frequency-Inverse Document Frequency.
///
/// # Arguments
/// * `text1` - First text as tokens
/// * `text2` - Second text as tokens
/// * `corpus` - Collection of documents for IDF calculation
///
/// # Returns
/// Cosine similarity between -1 and 1.
pub fn cosine_similarity_tfidf<T>(
    text1: &[&str],
    text2: &[&str],
    corpus: &[Vec<&str>],
) -> MetricsResult<T>
where
    T: Float + std::fmt::Debug,
{
    if corpus.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Build vocabulary from corpus
    let mut vocabulary: HashSet<&str> = HashSet::new();
    for doc in corpus {
        for &token in doc {
            vocabulary.insert(token);
        }
    }

    let vocab_vec: Vec<&str> = vocabulary.into_iter().collect();
    let _vocab_size = vocab_vec.len();

    // Calculate IDF for each term
    let mut idf_map = HashMap::new();
    let corpus_size = T::from(corpus.len()).unwrap();

    for &term in &vocab_vec {
        let doc_freq = corpus.iter().filter(|doc| doc.contains(&term)).count();
        let idf = if doc_freq > 0 {
            (corpus_size / T::from(doc_freq).unwrap()).ln()
        } else {
            T::zero()
        };
        idf_map.insert(term, idf);
    }

    // Calculate TF-IDF vectors
    let tfidf1 = calculate_tfidf_vector(text1, &vocab_vec, &idf_map);
    let tfidf2 = calculate_tfidf_vector(text2, &vocab_vec, &idf_map);

    // Calculate cosine similarity
    let dot_product = tfidf1
        .iter()
        .zip(tfidf2.iter())
        .map(|(&a, &b)| a * b)
        .fold(T::zero(), |acc, x| acc + x);
    let norm1 = tfidf1
        .iter()
        .map(|&x| x * x)
        .fold(T::zero(), |acc, x| acc + x)
        .sqrt();
    let norm2 = tfidf2
        .iter()
        .map(|&x| x * x)
        .fold(T::zero(), |acc, x| acc + x)
        .sqrt();

    if norm1 == T::zero() || norm2 == T::zero() {
        return Ok(T::zero());
    }

    Ok(dot_product / (norm1 * norm2))
}

/// Calculate TF-IDF vector for a text
fn calculate_tfidf_vector<T>(
    text: &[&str],
    vocabulary: &[&str],
    idf_map: &HashMap<&str, T>,
) -> Vec<T>
where
    T: Float + std::fmt::Debug,
{
    let text_len = T::from(text.len()).unwrap();
    let mut tfidf_vector = Vec::new();

    for &term in vocabulary {
        let tf_count = text.iter().filter(|&&token| token == term).count();
        let tf = if tf_count > 0 {
            T::from(tf_count).unwrap() / text_len
        } else {
            T::zero()
        };

        let idf = idf_map.get(term).map_or(T::zero(), |&v| v);
        tfidf_vector.push(tf * idf);
    }

    tfidf_vector
}

/// Edit Distance (Levenshtein Distance) between two strings
///
/// Calculates the minimum number of single-character edits required
/// to change one string into another.
///
/// # Arguments
/// * `s1` - First string
/// * `s2` - Second string
///
/// # Returns
/// Edit distance as an integer.
pub fn edit_distance(s1: &str, s2: &str) -> usize {
    let chars1: Vec<char> = s1.chars().collect();
    let chars2: Vec<char> = s2.chars().collect();
    let m = chars1.len();
    let n = chars2.len();

    let mut dp = vec![vec![0; n + 1]; m + 1];

    // Initialize base cases
    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }

    // Fill DP table
    for i in 1..=m {
        for j in 1..=n {
            if chars1[i - 1] == chars2[j - 1] {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = 1 + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
            }
        }
    }

    dp[m][n]
}

/// Normalized Edit Distance between two strings
///
/// Edit distance normalized by the length of the longer string.
///
/// # Arguments
/// * `s1` - First string
/// * `s2` - Second string
///
/// # Returns
/// Normalized edit distance between 0 and 1.
pub fn normalized_edit_distance<T>(s1: &str, s2: &str) -> T
where
    T: Float + std::fmt::Debug,
{
    let edit_dist = edit_distance(s1, s2);
    let max_len = s1.len().max(s2.len());

    if max_len == 0 {
        return T::zero();
    }

    T::from(edit_dist).unwrap() / T::from(max_len).unwrap()
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bleu_score_perfect_match() {
        let candidate = vec!["the", "cat", "is", "on", "the", "mat"];
        let references = vec![vec!["the", "cat", "is", "on", "the", "mat"]];
        let weights = [0.25, 0.25, 0.25, 0.25];

        let score: f64 =
            bleu_score(&candidate, &references, &weights, SmoothingFunction::None).unwrap();
        assert_relative_eq!(score, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_bleu_score_no_match() {
        let candidate = vec!["hello", "world"];
        let references = vec![vec!["the", "cat", "is", "on", "the", "mat"]];
        let weights = [0.25, 0.25, 0.25, 0.25];

        let score: f64 =
            bleu_score(&candidate, &references, &weights, SmoothingFunction::None).unwrap();
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_rouge_n_score() {
        let system = vec!["the", "cat", "was", "found", "under", "the", "bed"];
        let references = vec![vec!["the", "cat", "was", "under", "the", "bed"]];

        let score: f64 = rouge_n_score(&system, &references, 1).unwrap();
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_rouge_l_score() {
        let system = vec!["the", "cat", "was", "found", "under", "the", "bed"];
        let reference = vec!["the", "cat", "was", "under", "the", "bed"];

        let score: f64 = rouge_l_score(&system, &reference).unwrap();
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_perplexity() {
        let log_probs = vec![-1.2, -0.8, -1.5, -0.9, -1.1];
        let perp: f64 = perplexity(&log_probs).unwrap();
        assert!(perp > 0.0);
    }

    #[test]
    fn test_jaccard_similarity() {
        let text1 = vec!["the", "cat", "is", "happy"];
        let text2 = vec!["the", "dog", "is", "happy"];

        let similarity: f64 = jaccard_similarity(&text1, &text2).unwrap();
        assert!(similarity > 0.0 && similarity <= 1.0);
        // Should be 3/5 = 0.6 (intersection: "the", "is", "happy"; union: "the", "cat", "dog", "is", "happy")
        assert_relative_eq!(similarity, 0.6, epsilon = 1e-6);
    }

    #[test]
    fn test_edit_distance() {
        let distance = edit_distance("kitten", "sitting");
        assert_eq!(distance, 3); // k->s, e->i, insert g
    }

    #[test]
    fn test_normalized_edit_distance() {
        let distance: f64 = normalized_edit_distance("kitten", "sitting");
        assert_relative_eq!(distance, 3.0 / 7.0, epsilon = 1e-6); // 3 edits / 7 chars (max length)
    }
}
