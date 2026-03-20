//! Text preprocessing utilities
//!
//! Provides basic text preprocessing functionality including tokenization,
//! stop word removal, and stemming.

use std::collections::HashSet;

/// Simple tokenizer that splits text into words
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::text::preprocessing::SimpleTokenizer;
///
/// let tokenizer = SimpleTokenizer::new();
/// let tokens = tokenizer.tokenize("Hello, world! This is a test.");
/// assert_eq!(tokens, vec!["hello", "world", "this", "is", "a", "test"]);
/// ```
#[derive(Debug, Clone)]
pub struct SimpleTokenizer {
    lowercase: bool,
    remove_punctuation: bool,
}

impl Default for SimpleTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleTokenizer {
    /// Create a new simple tokenizer
    pub fn new() -> Self {
        Self {
            lowercase: true,
            remove_punctuation: true,
        }
    }

    /// Set whether to convert to lowercase
    pub fn lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }

    /// Set whether to remove punctuation
    pub fn remove_punctuation(mut self, remove_punctuation: bool) -> Self {
        self.remove_punctuation = remove_punctuation;
        self
    }

    /// Tokenize text into words
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let mut text = text.to_string();

        if self.lowercase {
            text = text.to_lowercase();
        }

        if self.remove_punctuation {
            text = text
                .chars()
                .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                .collect();
        }

        text.split_whitespace()
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    }
}

/// Stop word remover
///
/// Removes common stop words from text.
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::text::preprocessing::StopWordRemover;
///
/// let remover = StopWordRemover::english();
/// let tokens = vec!["the", "quick", "brown", "fox"];
/// let filtered = remover.remove_stop_words(&tokens);
/// assert_eq!(filtered, vec!["quick", "brown", "fox"]);
/// ```
#[derive(Debug, Clone)]
pub struct StopWordRemover {
    stop_words: HashSet<String>,
}

impl StopWordRemover {
    /// Create a new stop word remover with custom stop words
    pub fn new(stop_words: HashSet<String>) -> Self {
        Self { stop_words }
    }

    /// Create a stop word remover with common English stop words
    pub fn english() -> Self {
        let stop_words = [
            "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from", "has", "he",
            "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "will", "with", "the",
            "this", "but", "they", "have", "had", "what", "said", "each", "which", "she", "do",
            "how", "their", "if", "up", "out", "many", "then", "them", "these", "so", "some",
            "her", "would", "make", "like", "into", "him", "time", "two", "more", "very", "when",
            "come", "can", "could", "say", "get", "use", "your", "may", "way", "work", "just",
            "now", "think", "also", "after", "back", "other", "good", "new", "first", "well",
            "year", "see", "own", "want", "over",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        Self { stop_words }
    }

    /// Remove stop words from a list of tokens
    pub fn remove_stop_words(&self, tokens: &[impl AsRef<str>]) -> Vec<String> {
        tokens
            .iter()
            .filter_map(|token| {
                let token_str = token.as_ref().to_lowercase();
                if self.stop_words.contains(&token_str) {
                    None
                } else {
                    Some(token.as_ref().to_string())
                }
            })
            .collect()
    }

    /// Add custom stop words
    pub fn add_stop_words<I>(&mut self, words: I)
    where
        I: IntoIterator<Item = String>,
    {
        for word in words {
            self.stop_words.insert(word.to_lowercase());
        }
    }
}

/// Simple Porter stemmer implementation
///
/// Implements a simplified version of the Porter stemming algorithm
/// for reducing words to their root form.
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::text::preprocessing::PorterStemmer;
///
/// let stemmer = PorterStemmer::new();
/// assert_eq!(stemmer.stem("running"), "run");
/// assert_eq!(stemmer.stem("flies"), "fli");
/// ```
#[derive(Debug, Clone)]
pub struct PorterStemmer;

impl Default for PorterStemmer {
    fn default() -> Self {
        Self::new()
    }
}

impl PorterStemmer {
    /// Create a new Porter stemmer
    pub fn new() -> Self {
        Self
    }

    /// Stem a word to its root form
    pub fn stem(&self, word: &str) -> String {
        let word = word.to_lowercase();
        let mut result = word.clone();

        // Step 1a: Remove plurals
        if result.ends_with("sses") {
            result = result[..result.len() - 2].to_string();
        } else if result.ends_with("ies") {
            result = result[..result.len() - 2].to_string();
        } else if result.ends_with("ss") {
            // Keep as is
        } else if result.ends_with("s") && result.len() > 1 {
            result = result[..result.len() - 1].to_string();
        }

        // Step 1b: Handle -ed and -ing
        if result.ends_with("eed") && self.get_measure(&result[..result.len() - 3]) > 0 {
            result = result[..result.len() - 1].to_string();
        } else if result.ends_with("ed") && self.contains_vowel(&result[..result.len() - 2]) {
            result = result[..result.len() - 2].to_string();
            result = self.post_process_1b(result);
        } else if result.ends_with("ing") && self.contains_vowel(&result[..result.len() - 3]) {
            result = result[..result.len() - 3].to_string();
            result = self.post_process_1b(result);
        }

        // Step 2: Handle common suffix transformations
        result = self.step_2(result);

        // Step 3: Handle more suffix transformations
        result = self.step_3(result);

        // Step 4: Remove common suffixes
        result = self.step_4(result);

        // Step 5: Final cleanup
        result = self.step_5(result);

        result
    }

    fn is_vowel(&self, c: char) -> bool {
        matches!(c, 'a' | 'e' | 'i' | 'o' | 'u')
    }

    fn contains_vowel(&self, word: &str) -> bool {
        word.chars().any(|c| self.is_vowel(c))
    }

    fn get_measure(&self, word: &str) -> usize {
        let mut measure = 0;
        let mut prev_was_vowel = false;

        for c in word.chars() {
            let is_vowel = self.is_vowel(c);
            if !is_vowel && prev_was_vowel {
                measure += 1;
            }
            prev_was_vowel = is_vowel;
        }

        measure
    }

    fn post_process_1b(&self, mut word: String) -> String {
        if word.ends_with("at") || word.ends_with("bl") || word.ends_with("iz") {
            word.push('e');
        } else if self.ends_with_double_consonant(&word)
            && !word.ends_with("l")
            && !word.ends_with("s")
            && !word.ends_with("z")
        {
            word.pop();
        } else if self.get_measure(&word) == 1 && self.ends_with_cvc(&word) {
            word.push('e');
        }
        word
    }

    fn ends_with_double_consonant(&self, word: &str) -> bool {
        if word.len() < 2 {
            return false;
        }
        let chars: Vec<char> = word.chars().collect();
        let last = chars[chars.len() - 1];
        let second_last = chars[chars.len() - 2];
        !self.is_vowel(last) && last == second_last
    }

    fn ends_with_cvc(&self, word: &str) -> bool {
        if word.len() < 3 {
            return false;
        }
        let chars: Vec<char> = word.chars().collect();
        let len = chars.len();
        !self.is_vowel(chars[len - 3])
            && self.is_vowel(chars[len - 2])
            && !self.is_vowel(chars[len - 1])
            && !matches!(chars[len - 1], 'w' | 'x' | 'y')
    }

    fn step_2(&self, mut word: String) -> String {
        let suffixes = [
            ("ational", "ate"),
            ("tional", "tion"),
            ("enci", "ence"),
            ("anci", "ance"),
            ("izer", "ize"),
            ("abli", "able"),
            ("alli", "al"),
            ("entli", "ent"),
            ("eli", "e"),
            ("ousli", "ous"),
            ("ization", "ize"),
            ("ation", "ate"),
            ("ator", "ate"),
            ("alism", "al"),
            ("iveness", "ive"),
            ("fulness", "ful"),
            ("ousness", "ous"),
            ("aliti", "al"),
            ("iviti", "ive"),
            ("biliti", "ble"),
        ];

        for (suffix, replacement) in &suffixes {
            if word.ends_with(suffix) {
                let stem = &word[..word.len() - suffix.len()];
                if self.get_measure(stem) > 0 {
                    word = format!("{}{}", stem, replacement);
                    break;
                }
            }
        }
        word
    }

    fn step_3(&self, mut word: String) -> String {
        let suffixes = [
            ("icate", "ic"),
            ("ative", ""),
            ("alize", "al"),
            ("iciti", "ic"),
            ("ical", "ic"),
            ("ful", ""),
            ("ness", ""),
        ];

        for (suffix, replacement) in &suffixes {
            if word.ends_with(suffix) {
                let stem = &word[..word.len() - suffix.len()];
                if self.get_measure(stem) > 0 {
                    word = format!("{}{}", stem, replacement);
                    break;
                }
            }
        }
        word
    }

    fn step_4(&self, mut word: String) -> String {
        let suffixes = [
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ion",
            "ou", "ism", "ate", "iti", "ous", "ive", "ize",
        ];

        for suffix in &suffixes {
            if word.ends_with(suffix) {
                let stem = &word[..word.len() - suffix.len()];
                if self.get_measure(stem) > 1 {
                    if *suffix == "ion" && stem.ends_with("s") || stem.ends_with("t") {
                        word = stem.to_string();
                    } else if *suffix != "ion" {
                        word = stem.to_string();
                    }
                    break;
                }
            }
        }
        word
    }

    fn step_5(&self, mut word: String) -> String {
        if word.ends_with("e") {
            let stem = &word[..word.len() - 1];
            let measure = self.get_measure(stem);
            if measure > 1 || (measure == 1 && !self.ends_with_cvc(stem)) {
                word = stem.to_string();
            }
        }

        if word.ends_with("ll") && self.get_measure(&word) > 1 {
            word.pop();
        }

        word
    }
}

/// Text preprocessor that combines tokenization, stop word removal, and stemming
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::text::preprocessing::TextPreprocessor;
///
/// let preprocessor = TextPreprocessor::new()
///     .enable_stemming(true)
///     .enable_stop_word_removal(true);
///
/// let processed = preprocessor.preprocess("The running dogs are playing");
/// // Result will be stemmed and stop words removed
/// ```
#[derive(Debug, Clone)]
pub struct TextPreprocessor {
    tokenizer: SimpleTokenizer,
    stop_word_remover: Option<StopWordRemover>,
    stemmer: Option<PorterStemmer>,
    enable_stemming: bool,
    enable_stop_word_removal: bool,
}

impl Default for TextPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

impl TextPreprocessor {
    /// Create a new text preprocessor
    pub fn new() -> Self {
        Self {
            tokenizer: SimpleTokenizer::new(),
            stop_word_remover: Some(StopWordRemover::english()),
            stemmer: Some(PorterStemmer::new()),
            enable_stemming: false,
            enable_stop_word_removal: false,
        }
    }

    /// Enable or disable stemming
    pub fn enable_stemming(mut self, enable: bool) -> Self {
        self.enable_stemming = enable;
        self
    }

    /// Enable or disable stop word removal
    pub fn enable_stop_word_removal(mut self, enable: bool) -> Self {
        self.enable_stop_word_removal = enable;
        self
    }

    /// Set custom stop words
    pub fn stop_words(mut self, stop_words: HashSet<String>) -> Self {
        self.stop_word_remover = Some(StopWordRemover::new(stop_words));
        self
    }

    /// Configure the tokenizer
    pub fn tokenizer(mut self, tokenizer: SimpleTokenizer) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// Preprocess text through the configured pipeline
    pub fn preprocess(&self, text: &str) -> Vec<String> {
        // Tokenize
        let mut tokens = self.tokenizer.tokenize(text);

        // Remove stop words
        if self.enable_stop_word_removal {
            if let Some(ref remover) = self.stop_word_remover {
                tokens = remover.remove_stop_words(&tokens);
            }
        }

        // Stem words
        if self.enable_stemming {
            if let Some(ref stemmer) = self.stemmer {
                tokens = tokens.iter().map(|token| stemmer.stem(token)).collect();
            }
        }

        tokens
    }
}
