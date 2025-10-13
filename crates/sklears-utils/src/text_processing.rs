//! Text processing utilities for machine learning workflows
//!
//! This module provides utilities for text parsing, string similarity measures,
//! regular expression helpers, unicode handling, and text normalization.

use crate::UtilsResult;
use std::cmp;
use std::collections::{HashMap, HashSet};

// ===== TEXT PARSING UTILITIES =====

/// Text parser for extracting structured data from text
pub struct TextParser;

impl TextParser {
    /// Split text into tokens using various delimiters
    pub fn tokenize(text: &str, delimiters: &[char]) -> Vec<String> {
        if delimiters.is_empty() {
            return vec![text.to_string()];
        }

        let mut tokens = Vec::new();
        let mut current_token = String::new();

        for ch in text.chars() {
            if delimiters.contains(&ch) {
                if !current_token.is_empty() {
                    tokens.push(current_token.trim().to_string());
                    current_token.clear();
                }
            } else {
                current_token.push(ch);
            }
        }

        if !current_token.is_empty() {
            tokens.push(current_token.trim().to_string());
        }

        tokens
    }

    /// Extract numbers from text
    pub fn extract_numbers(text: &str) -> Vec<f64> {
        let mut numbers = Vec::new();
        let mut current_number = String::new();

        for ch in text.chars() {
            if ch.is_ascii_digit() || ch == '.' || ch == '-' || ch == '+' {
                current_number.push(ch);
            } else if !current_number.is_empty() {
                if let Ok(num) = current_number.parse::<f64>() {
                    numbers.push(num);
                }
                current_number.clear();
            }
        }

        if !current_number.is_empty() {
            if let Ok(num) = current_number.parse::<f64>() {
                numbers.push(num);
            }
        }

        numbers
    }

    /// Extract key-value pairs from text
    pub fn extract_key_value_pairs(
        text: &str,
        pair_delimiter: char,
        kv_delimiter: char,
    ) -> HashMap<String, String> {
        let mut pairs = HashMap::new();

        for pair in text.split(pair_delimiter) {
            if let Some(kv_pos) = pair.find(kv_delimiter) {
                let key = pair[..kv_pos].trim().to_string();
                let value = pair[kv_pos + 1..].trim().to_string();
                pairs.insert(key, value);
            }
        }

        pairs
    }

    /// Parse structured text lines (e.g., log files)
    pub fn parse_structured_lines<F, T>(lines: &[String], parser: F) -> UtilsResult<Vec<T>>
    where
        F: Fn(&str) -> Option<T>,
    {
        let mut results = Vec::new();

        for line in lines {
            if let Some(parsed) = parser(line) {
                results.push(parsed);
            }
        }

        Ok(results)
    }

    /// Extract words and their frequencies
    pub fn word_frequency(text: &str) -> HashMap<String, usize> {
        let mut frequencies = HashMap::new();

        let words = Self::tokenize(text, &[' ', '\t', '\n', '\r', '.', ',', '!', '?', ';', ':']);

        for word in words {
            let word_lower = word.to_lowercase();
            if !word_lower.is_empty() {
                *frequencies.entry(word_lower).or_insert(0) += 1;
            }
        }

        frequencies
    }
}

// ===== STRING SIMILARITY MEASURES =====

/// String similarity utilities
pub struct StringSimilarity;

impl StringSimilarity {
    /// Calculate Levenshtein distance between two strings
    pub fn levenshtein_distance(s1: &str, s2: &str) -> usize {
        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();
        let m = s1_chars.len();
        let n = s2_chars.len();

        if m == 0 {
            return n;
        }
        if n == 0 {
            return m;
        }

        let mut dp = vec![vec![0; n + 1]; m + 1];

        // Initialize first row and column
        for (i, row) in dp.iter_mut().enumerate().take(m + 1) {
            row[0] = i;
        }
        for j in 0..=n {
            dp[0][j] = j;
        }

        // Fill the DP table
        for i in 1..=m {
            for j in 1..=n {
                let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                    0
                } else {
                    1
                };
                dp[i][j] = cmp::min(
                    dp[i - 1][j] + 1, // deletion
                    cmp::min(
                        dp[i][j - 1] + 1,        // insertion
                        dp[i - 1][j - 1] + cost, // substitution
                    ),
                );
            }
        }

        dp[m][n]
    }

    /// Calculate normalized Levenshtein similarity (0.0 to 1.0)
    pub fn levenshtein_similarity(s1: &str, s2: &str) -> f64 {
        let max_len = cmp::max(s1.len(), s2.len());
        if max_len == 0 {
            return 1.0;
        }

        let distance = Self::levenshtein_distance(s1, s2);
        1.0 - (distance as f64 / max_len as f64)
    }

    /// Calculate Jaccard similarity between two strings (based on character n-grams)
    pub fn jaccard_similarity(s1: &str, s2: &str, n: usize) -> f64 {
        if n == 0 {
            return 0.0;
        }

        let ngrams1 = Self::character_ngrams(s1, n);
        let ngrams2 = Self::character_ngrams(s2, n);

        if ngrams1.is_empty() && ngrams2.is_empty() {
            return 1.0;
        }

        let intersection: HashSet<_> = ngrams1.intersection(&ngrams2).collect();
        let union: HashSet<_> = ngrams1.union(&ngrams2).collect();

        intersection.len() as f64 / union.len() as f64
    }

    /// Calculate cosine similarity between two strings (based on word frequencies)
    pub fn cosine_similarity(s1: &str, s2: &str) -> f64 {
        let freq1 = TextParser::word_frequency(s1);
        let freq2 = TextParser::word_frequency(s2);

        if freq1.is_empty() || freq2.is_empty() {
            return 0.0;
        }

        let mut dot_product = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;

        let all_words: HashSet<_> = freq1.keys().chain(freq2.keys()).collect();

        for word in all_words {
            let f1 = *freq1.get(word).unwrap_or(&0) as f64;
            let f2 = *freq2.get(word).unwrap_or(&0) as f64;

            dot_product += f1 * f2;
            norm1 += f1 * f1;
            norm2 += f2 * f2;
        }

        if norm1 == 0.0 || norm2 == 0.0 {
            return 0.0;
        }

        dot_product / (norm1.sqrt() * norm2.sqrt())
    }

    /// Generate character n-grams from a string
    fn character_ngrams(s: &str, n: usize) -> HashSet<String> {
        let chars: Vec<char> = s.chars().collect();
        let mut ngrams = HashSet::new();

        if chars.len() < n {
            return ngrams;
        }

        for i in 0..=chars.len() - n {
            let ngram: String = chars[i..i + n].iter().collect();
            ngrams.insert(ngram);
        }

        ngrams
    }

    /// Find the best matching string from a list
    pub fn find_best_match(
        target: &str,
        candidates: &[String],
        similarity_fn: fn(&str, &str) -> f64,
        threshold: f64,
    ) -> Option<(String, f64)> {
        let mut best_match = None;
        let mut best_score = threshold;

        for candidate in candidates {
            let score = similarity_fn(target, candidate);
            if score > best_score {
                best_score = score;
                best_match = Some((candidate.clone(), score));
            }
        }

        best_match
    }
}

// ===== REGULAR EXPRESSION HELPERS =====

/// Regular expression utilities for common patterns
pub struct RegexUtils;

impl RegexUtils {
    /// Check if string is a valid email address (simple pattern)
    pub fn is_email(text: &str) -> bool {
        let email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$";
        Self::matches_pattern(text, email_pattern)
    }

    /// Check if string is a valid URL (simple pattern)
    pub fn is_url(text: &str) -> bool {
        let url_pattern = r"^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$";
        Self::matches_pattern(text, url_pattern)
    }

    /// Check if string contains only digits
    pub fn is_numeric(text: &str) -> bool {
        !text.is_empty() && text.chars().all(|c| c.is_ascii_digit())
    }

    /// Check if string is alphanumeric
    pub fn is_alphanumeric(text: &str) -> bool {
        !text.is_empty() && text.chars().all(|c| c.is_alphanumeric())
    }

    /// Extract all words from text
    pub fn extract_words(text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|word| {
                word.chars()
                    .filter(|c| c.is_alphabetic())
                    .collect::<String>()
            })
            .filter(|word| !word.is_empty())
            .collect()
    }

    /// Simple pattern matching (without regex crate dependency)
    fn matches_pattern(text: &str, pattern: &str) -> bool {
        // This is a simplified implementation for common patterns
        // In a real implementation, you'd use the regex crate
        match pattern {
            r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$" => {
                text.contains('@')
                    && text.contains('.')
                    && !text.starts_with('@')
                    && !text.ends_with('@')
            }
            r"^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$" => {
                text.starts_with("http://") || text.starts_with("https://")
            }
            _ => false,
        }
    }

    /// Find all occurrences of a substring
    pub fn find_all_occurrences(text: &str, pattern: &str) -> Vec<usize> {
        let mut positions = Vec::new();
        let mut start = 0;

        while let Some(pos) = text[start..].find(pattern) {
            let absolute_pos = start + pos;
            positions.push(absolute_pos);
            start = absolute_pos + 1;
        }

        positions
    }
}

// ===== UNICODE HANDLING =====

/// Unicode handling utilities
pub struct UnicodeUtils;

impl UnicodeUtils {
    /// Normalize unicode text (simple normalization)
    pub fn simple_normalize(text: &str) -> String {
        text.chars()
            .map(|c| match c {
                'À'..='Ä' | 'à'..='ä' => 'a',
                'È'..='Ë' | 'è'..='ë' => 'e',
                'Ì'..='Ï' | 'ì'..='ï' => 'i',
                'Ò'..='Ö' | 'ò'..='ö' => 'o',
                'Ù'..='Ü' | 'ù'..='ü' => 'u',
                'Ñ' | 'ñ' => 'n',
                'Ç' | 'ç' => 'c',
                _ => c,
            })
            .collect()
    }

    /// Remove diacritics (accents) from text
    pub fn remove_diacritics(text: &str) -> String {
        Self::simple_normalize(text)
    }

    /// Check if text contains non-ASCII characters
    pub fn has_non_ascii(text: &str) -> bool {
        !text.is_ascii()
    }

    /// Count unicode characters (not bytes)
    pub fn char_count(text: &str) -> usize {
        text.chars().count()
    }

    /// Get unicode character categories
    pub fn analyze_text(text: &str) -> TextAnalysis {
        let mut analysis = TextAnalysis::default();

        for ch in text.chars() {
            analysis.total_chars += 1;

            if ch.is_alphabetic() {
                analysis.alphabetic += 1;
            }
            if ch.is_numeric() {
                analysis.numeric += 1;
            }
            if ch.is_whitespace() {
                analysis.whitespace += 1;
            }
            if ch.is_ascii_punctuation() {
                analysis.punctuation += 1;
            }
            if !ch.is_ascii() {
                analysis.non_ascii += 1;
            }
        }

        analysis
    }
}

/// Text analysis results
#[derive(Debug, Default, Clone)]
pub struct TextAnalysis {
    pub total_chars: usize,
    pub alphabetic: usize,
    pub numeric: usize,
    pub whitespace: usize,
    pub punctuation: usize,
    pub non_ascii: usize,
}

// ===== TEXT NORMALIZATION =====

/// Text normalization utilities
pub struct TextNormalizer;

impl TextNormalizer {
    /// Normalize text for machine learning (lowercase, trim, etc.)
    pub fn normalize_for_ml(text: &str) -> String {
        text.to_lowercase()
            .trim()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Remove extra whitespace
    pub fn normalize_whitespace(text: &str) -> String {
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// Convert to title case
    pub fn to_title_case(text: &str) -> String {
        text.split_whitespace()
            .map(|word| {
                let mut chars = word.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => {
                        first.to_uppercase().collect::<String>()
                            + chars.as_str().to_lowercase().as_str()
                    }
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Remove HTML tags (simple implementation)
    pub fn remove_html_tags(text: &str) -> String {
        let mut result = String::new();
        let mut in_tag = false;

        for ch in text.chars() {
            match ch {
                '<' => in_tag = true,
                '>' => in_tag = false,
                _ if !in_tag => result.push(ch),
                _ => {}
            }
        }

        result
    }

    /// Clean text for analysis (remove punctuation, normalize case)
    pub fn clean_for_analysis(text: &str) -> String {
        let cleaned = Self::remove_html_tags(text);
        let cleaned = UnicodeUtils::remove_diacritics(&cleaned);
        Self::normalize_for_ml(&cleaned)
    }

    /// Truncate text to specified length with ellipsis
    pub fn truncate(text: &str, max_length: usize, add_ellipsis: bool) -> String {
        if text.len() <= max_length {
            return text.to_string();
        }

        let truncated = &text[..max_length.saturating_sub(if add_ellipsis { 3 } else { 0 })];
        if add_ellipsis {
            format!("{truncated}...")
        } else {
            truncated.to_string()
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_parser() {
        let text = "Hello, world! How are you?";
        let tokens = TextParser::tokenize(text, &[' ', ',', '!', '?']);
        assert_eq!(tokens, vec!["Hello", "world", "How", "are", "you"]);

        let numbers = TextParser::extract_numbers("Price: $12.99, Quantity: 5, Discount: -2.5");
        assert_eq!(numbers, vec![12.99, 5.0, -2.5]);

        let freq = TextParser::word_frequency("hello world hello");
        assert_eq!(*freq.get("hello").unwrap(), 2);
        assert_eq!(*freq.get("world").unwrap(), 1);
    }

    #[test]
    fn test_string_similarity() {
        assert_eq!(StringSimilarity::levenshtein_distance("cat", "bat"), 1);
        assert_eq!(
            StringSimilarity::levenshtein_distance("kitten", "sitting"),
            3
        );

        let similarity = StringSimilarity::levenshtein_similarity("hello", "hallo");
        assert!(similarity > 0.5);

        let jaccard = StringSimilarity::jaccard_similarity("hello", "hallo", 2);
        assert!(jaccard > 0.0);

        let cosine = StringSimilarity::cosine_similarity("hello world", "hello earth");
        assert!(cosine > 0.0);
    }

    #[test]
    fn test_regex_utils() {
        assert!(RegexUtils::is_email("test@example.com"));
        assert!(!RegexUtils::is_email("invalid.email"));

        assert!(RegexUtils::is_url("https://example.com"));
        assert!(!RegexUtils::is_url("not-a-url"));

        assert!(RegexUtils::is_numeric("12345"));
        assert!(!RegexUtils::is_numeric("123a45"));

        let words = RegexUtils::extract_words("Hello, world! 123");
        assert_eq!(words, vec!["Hello", "world"]);

        let positions = RegexUtils::find_all_occurrences("hello hello world", "hello");
        assert_eq!(positions, vec![0, 6]);
    }

    #[test]
    fn test_unicode_utils() {
        let normalized = UnicodeUtils::simple_normalize("café");
        assert_eq!(normalized, "cafe");

        assert!(UnicodeUtils::has_non_ascii("café"));
        assert!(!UnicodeUtils::has_non_ascii("cafe"));

        assert_eq!(UnicodeUtils::char_count("café"), 4);

        let analysis = UnicodeUtils::analyze_text("Hello, 世界!");
        assert!(analysis.total_chars > 0);
        assert!(analysis.alphabetic > 0);
        assert!(analysis.non_ascii > 0);
    }

    #[test]
    fn test_text_normalizer() {
        let normalized = TextNormalizer::normalize_for_ml("  Hello, WORLD!  ");
        assert_eq!(normalized, "hello world");

        let whitespace = TextNormalizer::normalize_whitespace("  hello   world  ");
        assert_eq!(whitespace, "hello world");

        let title = TextNormalizer::to_title_case("hello world");
        assert_eq!(title, "Hello World");

        let no_html = TextNormalizer::remove_html_tags("<p>Hello <b>world</b>!</p>");
        assert_eq!(no_html, "Hello world!");

        let truncated = TextNormalizer::truncate("Hello, world!", 8, true);
        assert_eq!(truncated, "Hello...");
    }

    #[test]
    fn test_text_analysis() {
        let analysis = UnicodeUtils::analyze_text("Hello123! ");
        assert_eq!(analysis.total_chars, 10);
        assert_eq!(analysis.alphabetic, 5);
        assert_eq!(analysis.numeric, 3);
        assert_eq!(analysis.whitespace, 1);
        assert_eq!(analysis.punctuation, 1);
    }
}
