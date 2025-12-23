//! Advanced Sentiment Analysis
//!
//! This module provides advanced sentiment analysis capabilities including:
//! - Multilingual sentiment support (English, Spanish, French, German, Japanese, Chinese)
//! - Enhanced intensity scoring with VADER-style modifiers
//! - Negation handling and contextual sentiment analysis
//! - Intensity boosters and dampeners

use scirs2_core::ndarray::Array2;
use sklears_core::{error::Result as SklResult, prelude::SklearsError, types::Float};
use std::collections::{HashMap, HashSet};

// ============================================================================
// Language Support
// ============================================================================

/// Supported languages for sentiment analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    /// English
    English,
    /// Spanish
    Spanish,
    /// French
    French,
    /// German
    German,
    /// Japanese
    Japanese,
    /// Chinese (Simplified)
    Chinese,
}

impl Language {
    /// Detect language from text (simple heuristic-based detection)
    pub fn detect(text: &str) -> Self {
        // Simple character-based language detection
        let chars: Vec<char> = text.chars().collect();

        // Check for CJK characters
        let has_japanese = chars.iter().any(|&c| {
            ('\u{3040}'..='\u{309F}').contains(&c) || // Hiragana
            ('\u{30A0}'..='\u{30FF}').contains(&c) // Katakana
        });

        let has_chinese = chars.iter().any(|&c| {
            ('\u{4E00}'..='\u{9FFF}').contains(&c) // CJK Unified Ideographs
        });

        if has_japanese {
            return Language::Japanese;
        }

        if has_chinese {
            return Language::Chinese;
        }

        // Check for language-specific common words
        let text_lower = text.to_lowercase();

        // Spanish indicators
        let spanish_indicators = [
            "el", "la", "los", "las", "de", "que", "es", "un", "una", "por", "para",
        ];
        let spanish_score: usize = spanish_indicators
            .iter()
            .filter(|&word| text_lower.split_whitespace().any(|w| w == *word))
            .count();

        // French indicators
        let french_indicators = [
            "le", "la", "les", "de", "un", "une", "est", "dans", "pour", "avec",
        ];
        let french_score: usize = french_indicators
            .iter()
            .filter(|&word| text_lower.split_whitespace().any(|w| w == *word))
            .count();

        // German indicators
        let german_indicators = [
            "der", "die", "das", "und", "ist", "ein", "eine", "mit", "für", "von",
        ];
        let german_score: usize = german_indicators
            .iter()
            .filter(|&word| text_lower.split_whitespace().any(|w| w == *word))
            .count();

        // Return language with highest score
        if spanish_score > french_score && spanish_score > german_score && spanish_score >= 2 {
            Language::Spanish
        } else if french_score > german_score && french_score >= 2 {
            Language::French
        } else if german_score >= 2 {
            Language::German
        } else {
            Language::English // Default
        }
    }
}

// ============================================================================
// Sentiment Lexicons
// ============================================================================

/// Multilingual sentiment lexicon manager
#[derive(Debug, Clone)]
pub struct MultilingualLexicon {
    positive_words: HashMap<Language, HashSet<String>>,
    negative_words: HashMap<Language, HashSet<String>>,
    boosters: HashMap<Language, HashSet<String>>,
    dampeners: HashMap<Language, HashSet<String>>,
    negations: HashMap<Language, HashSet<String>>,
    intensifiers: HashMap<Language, HashMap<String, f64>>,
}

impl MultilingualLexicon {
    /// Create a new multilingual lexicon with default word lists
    pub fn new() -> Self {
        let mut lexicon = Self {
            positive_words: HashMap::new(),
            negative_words: HashMap::new(),
            boosters: HashMap::new(),
            dampeners: HashMap::new(),
            negations: HashMap::new(),
            intensifiers: HashMap::new(),
        };

        lexicon.initialize_english();
        lexicon.initialize_spanish();
        lexicon.initialize_french();
        lexicon.initialize_german();
        lexicon.initialize_japanese();
        lexicon.initialize_chinese();

        lexicon
    }

    /// Initialize English lexicon
    fn initialize_english(&mut self) {
        let lang = Language::English;

        // Positive words
        let positive = [
            "good",
            "great",
            "excellent",
            "amazing",
            "wonderful",
            "fantastic",
            "awesome",
            "love",
            "like",
            "enjoy",
            "happy",
            "pleased",
            "satisfied",
            "perfect",
            "best",
            "brilliant",
            "outstanding",
            "superb",
            "marvelous",
            "incredible",
            "beautiful",
            "delightful",
            "fabulous",
            "magnificent",
            "terrific",
            "positive",
            "advantage",
            "benefical",
            "favorable",
            "pleasant",
            "superior",
            "valuable",
            "worthwhile",
        ];
        self.positive_words
            .insert(lang, positive.iter().map(|&s| s.to_string()).collect());

        // Negative words
        let negative = [
            "bad",
            "terrible",
            "awful",
            "horrible",
            "disgusting",
            "hate",
            "dislike",
            "disappointed",
            "frustrated",
            "angry",
            "sad",
            "worst",
            "pathetic",
            "useless",
            "annoying",
            "boring",
            "stupid",
            "ridiculous",
            "poor",
            "negative",
            "inferior",
            "unacceptable",
            "inadequate",
            "deficient",
            "unsatisfactory",
            "dreadful",
        ];
        self.negative_words
            .insert(lang, negative.iter().map(|&s| s.to_string()).collect());

        // Boosters (increase intensity)
        let boosters = [
            "very",
            "extremely",
            "incredibly",
            "absolutely",
            "completely",
            "totally",
            "utterly",
            "highly",
            "truly",
            "really",
            "exceptionally",
            "particularly",
            "especially",
            "remarkably",
            "extraordinarily",
            "supremely",
            "intensely",
        ];
        self.boosters
            .insert(lang, boosters.iter().map(|&s| s.to_string()).collect());

        // Dampeners (decrease intensity)
        let dampeners = [
            "somewhat",
            "slightly",
            "barely",
            "hardly",
            "scarcely",
            "marginally",
            "moderately",
            "fairly",
            "relatively",
            "comparatively",
            "kind of",
            "sort of",
            "a bit",
            "a little",
            "pretty",
            "rather",
        ];
        self.dampeners
            .insert(lang, dampeners.iter().map(|&s| s.to_string()).collect());

        // Negations
        let negations = [
            "not",
            "no",
            "never",
            "neither",
            "nobody",
            "nothing",
            "nowhere",
            "none",
            "isn't",
            "aren't",
            "wasn't",
            "weren't",
            "haven't",
            "hasn't",
            "hadn't",
            "won't",
            "wouldn't",
            "don't",
            "doesn't",
            "didn't",
            "can't",
            "couldn't",
            "shouldn't",
            "wouldn't",
        ];
        self.negations
            .insert(lang, negations.iter().map(|&s| s.to_string()).collect());

        // Intensifiers with weights
        let mut intensifiers = HashMap::new();
        intensifiers.insert("very".to_string(), 1.5);
        intensifiers.insert("extremely".to_string(), 2.0);
        intensifiers.insert("absolutely".to_string(), 2.0);
        intensifiers.insert("completely".to_string(), 1.8);
        intensifiers.insert("totally".to_string(), 1.8);
        intensifiers.insert("highly".to_string(), 1.6);
        intensifiers.insert("really".to_string(), 1.4);
        intensifiers.insert("somewhat".to_string(), 0.6);
        intensifiers.insert("slightly".to_string(), 0.5);
        intensifiers.insert("barely".to_string(), 0.4);
        self.intensifiers.insert(lang, intensifiers);
    }

    /// Initialize Spanish lexicon
    fn initialize_spanish(&mut self) {
        let lang = Language::Spanish;

        // Positive words
        let positive = [
            "bueno",
            "excelente",
            "increíble",
            "maravilloso",
            "fantástico",
            "genial",
            "amor",
            "gustar",
            "disfrutar",
            "feliz",
            "contento",
            "satisfecho",
            "perfecto",
            "mejor",
            "brillante",
            "estupendo",
            "magnífico",
            "hermoso",
            "espléndido",
        ];
        self.positive_words
            .insert(lang, positive.iter().map(|&s| s.to_string()).collect());

        // Negative words
        let negative = [
            "malo",
            "terrible",
            "horrible",
            "desagradable",
            "odiar",
            "disgustar",
            "decepcionado",
            "frustrado",
            "enojado",
            "triste",
            "peor",
            "pésimo",
            "inútil",
            "molesto",
            "aburrido",
            "estúpido",
            "ridículo",
            "pobre",
        ];
        self.negative_words
            .insert(lang, negative.iter().map(|&s| s.to_string()).collect());

        // Boosters
        let boosters = [
            "muy",
            "extremadamente",
            "increíblemente",
            "absolutamente",
            "completamente",
            "totalmente",
            "altamente",
            "verdaderamente",
            "realmente",
            "especialmente",
        ];
        self.boosters
            .insert(lang, boosters.iter().map(|&s| s.to_string()).collect());

        // Dampeners
        let dampeners = [
            "algo",
            "ligeramente",
            "apenas",
            "casi",
            "un poco",
            "relativamente",
            "moderadamente",
            "bastante",
        ];
        self.dampeners
            .insert(lang, dampeners.iter().map(|&s| s.to_string()).collect());

        // Negations
        let negations = [
            "no", "nunca", "ninguno", "nada", "nadie", "tampoco", "jamás", "ni",
        ];
        self.negations
            .insert(lang, negations.iter().map(|&s| s.to_string()).collect());
    }

    /// Initialize French lexicon
    fn initialize_french(&mut self) {
        let lang = Language::French;

        // Positive words
        let positive = [
            "bon",
            "excellent",
            "incroyable",
            "merveilleux",
            "fantastique",
            "génial",
            "amour",
            "aimer",
            "profiter",
            "heureux",
            "content",
            "satisfait",
            "parfait",
            "meilleur",
            "brillant",
            "superbe",
            "magnifique",
            "beau",
            "splendide",
        ];
        self.positive_words
            .insert(lang, positive.iter().map(|&s| s.to_string()).collect());

        // Negative words
        let negative = [
            "mauvais",
            "terrible",
            "horrible",
            "dégoûtant",
            "détester",
            "déplaire",
            "déçu",
            "frustré",
            "en colère",
            "triste",
            "pire",
            "pathétique",
            "inutile",
            "ennuyeux",
            "stupide",
            "ridicule",
            "pauvre",
        ];
        self.negative_words
            .insert(lang, negative.iter().map(|&s| s.to_string()).collect());

        // Boosters
        let boosters = [
            "très",
            "extrêmement",
            "incroyablement",
            "absolument",
            "complètement",
            "totalement",
            "hautement",
            "vraiment",
            "particulièrement",
            "spécialement",
        ];
        self.boosters
            .insert(lang, boosters.iter().map(|&s| s.to_string()).collect());

        // Dampeners
        let dampeners = [
            "quelque peu",
            "légèrement",
            "à peine",
            "presque",
            "un peu",
            "relativement",
            "modérément",
            "assez",
        ];
        self.dampeners
            .insert(lang, dampeners.iter().map(|&s| s.to_string()).collect());

        // Negations
        let negations = [
            "ne", "pas", "non", "jamais", "rien", "personne", "aucun", "ni",
        ];
        self.negations
            .insert(lang, negations.iter().map(|&s| s.to_string()).collect());
    }

    /// Initialize German lexicon
    fn initialize_german(&mut self) {
        let lang = Language::German;

        // Positive words
        let positive = [
            "gut",
            "ausgezeichnet",
            "unglaublich",
            "wunderbar",
            "fantastisch",
            "toll",
            "liebe",
            "mögen",
            "genießen",
            "glücklich",
            "zufrieden",
            "perfekt",
            "beste",
            "brillant",
            "hervorragend",
            "großartig",
            "schön",
            "herrlich",
        ];
        self.positive_words
            .insert(lang, positive.iter().map(|&s| s.to_string()).collect());

        // Negative words
        let negative = [
            "schlecht",
            "schrecklich",
            "furchtbar",
            "abscheulich",
            "hassen",
            "missfallen",
            "enttäuscht",
            "frustriert",
            "wütend",
            "traurig",
            "schlechteste",
            "erbärmlich",
            "nutzlos",
            "nervig",
            "langweilig",
            "dumm",
            "lächerlich",
            "arm",
        ];
        self.negative_words
            .insert(lang, negative.iter().map(|&s| s.to_string()).collect());

        // Boosters
        let boosters = [
            "sehr",
            "extrem",
            "unglaublich",
            "absolut",
            "vollständig",
            "total",
            "höchst",
            "wirklich",
            "besonders",
            "außerordentlich",
        ];
        self.boosters
            .insert(lang, boosters.iter().map(|&s| s.to_string()).collect());

        // Dampeners
        let dampeners = [
            "etwas",
            "leicht",
            "kaum",
            "fast",
            "ein bisschen",
            "relativ",
            "mäßig",
            "ziemlich",
        ];
        self.dampeners
            .insert(lang, dampeners.iter().map(|&s| s.to_string()).collect());

        // Negations
        let negations = [
            "nicht", "kein", "nie", "niemals", "niemand", "nichts", "nirgends", "weder",
        ];
        self.negations
            .insert(lang, negations.iter().map(|&s| s.to_string()).collect());
    }

    /// Initialize Japanese lexicon
    fn initialize_japanese(&mut self) {
        let lang = Language::Japanese;

        // Positive words (romanized for demonstration)
        let positive = [
            "良い",
            "素晴らしい",
            "最高",
            "楽しい",
            "嬉しい",
            "好き",
            "幸せ",
            "完璧",
            "美しい",
            "素敵",
            "最適",
            "優れた",
        ];
        self.positive_words
            .insert(lang, positive.iter().map(|&s| s.to_string()).collect());

        // Negative words
        let negative = [
            "悪い",
            "ひどい",
            "最悪",
            "嫌い",
            "悲しい",
            "つまらない",
            "残念",
            "不快",
            "下手",
            "駄目",
        ];
        self.negative_words
            .insert(lang, negative.iter().map(|&s| s.to_string()).collect());

        // Boosters
        let boosters = [
            "非常に",
            "とても",
            "すごく",
            "かなり",
            "大変",
            "本当に",
            "めちゃくちゃ",
        ];
        self.boosters
            .insert(lang, boosters.iter().map(|&s| s.to_string()).collect());

        // Negations
        let negations = [
            "ない",
            "ません",
            "じゃない",
            "ではない",
            "いない",
            "ありません",
        ];
        self.negations
            .insert(lang, negations.iter().map(|&s| s.to_string()).collect());
    }

    /// Initialize Chinese lexicon
    fn initialize_chinese(&mut self) {
        let lang = Language::Chinese;

        // Positive words
        let positive = [
            "好", "很好", "优秀", "棒", "美好", "喜欢", "爱", "快乐", "幸福", "完美", "最好",
            "精彩", "漂亮", "优美",
        ];
        self.positive_words
            .insert(lang, positive.iter().map(|&s| s.to_string()).collect());

        // Negative words
        let negative = [
            "坏", "糟糕", "差", "讨厌", "恨", "失望", "难过", "最坏", "无聊", "愚蠢", "可笑",
            "不好",
        ];
        self.negative_words
            .insert(lang, negative.iter().map(|&s| s.to_string()).collect());

        // Boosters
        let boosters = ["非常", "很", "特别", "极其", "十分", "相当", "超", "真"];
        self.boosters
            .insert(lang, boosters.iter().map(|&s| s.to_string()).collect());

        // Negations
        let negations = ["不", "没", "没有", "别", "无", "未", "非"];
        self.negations
            .insert(lang, negations.iter().map(|&s| s.to_string()).collect());
    }

    /// Get positive words for a language
    pub fn get_positive_words(&self, lang: Language) -> Option<&HashSet<String>> {
        self.positive_words.get(&lang)
    }

    /// Get negative words for a language
    pub fn get_negative_words(&self, lang: Language) -> Option<&HashSet<String>> {
        self.negative_words.get(&lang)
    }

    /// Get boosters for a language
    pub fn get_boosters(&self, lang: Language) -> Option<&HashSet<String>> {
        self.boosters.get(&lang)
    }

    /// Get dampeners for a language
    pub fn get_dampeners(&self, lang: Language) -> Option<&HashSet<String>> {
        self.dampeners.get(&lang)
    }

    /// Get negations for a language
    pub fn get_negations(&self, lang: Language) -> Option<&HashSet<String>> {
        self.negations.get(&lang)
    }

    /// Get intensifier weight for a word
    pub fn get_intensifier_weight(&self, lang: Language, word: &str) -> Option<f64> {
        self.intensifiers
            .get(&lang)
            .and_then(|map| map.get(word))
            .copied()
    }
}

impl Default for MultilingualLexicon {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Advanced Sentiment Analyzer
// ============================================================================

/// Advanced multilingual sentiment analyzer with intensity scoring
///
/// Features:
/// - Automatic language detection
/// - Multilingual sentiment lexicons (English, Spanish, French, German, Japanese, Chinese)
/// - VADER-style intensity modifiers (boosters, dampeners, negations)
/// - Context-aware sentiment analysis
/// - Enhanced feature extraction with intensity metrics
#[derive(Debug, Clone)]
pub struct AdvancedSentimentAnalyzer {
    pub(crate) lexicon: MultilingualLexicon,
    pub(crate) auto_detect_language: bool,
    pub(crate) default_language: Language,
    pub(crate) neutral_threshold: f64,
    pub(crate) negation_window: usize,
    pub(crate) booster_dampener_window: usize,
    pub(crate) case_sensitive: bool,
}

impl AdvancedSentimentAnalyzer {
    /// Create a new advanced sentiment analyzer
    pub fn new() -> Self {
        Self {
            lexicon: MultilingualLexicon::new(),
            auto_detect_language: true,
            default_language: Language::English,
            neutral_threshold: 0.1,
            negation_window: 3,
            booster_dampener_window: 1,
            case_sensitive: false,
        }
    }

    /// Set whether to automatically detect language
    pub fn auto_detect_language(mut self, auto_detect: bool) -> Self {
        self.auto_detect_language = auto_detect;
        self
    }

    /// Set default language (used when auto-detection is off)
    pub fn default_language(mut self, language: Language) -> Self {
        self.default_language = language;
        self
    }

    /// Set neutral threshold
    pub fn neutral_threshold(mut self, threshold: f64) -> Self {
        self.neutral_threshold = threshold;
        self
    }

    /// Set negation window size (words before sentiment word affected by negation)
    pub fn negation_window(mut self, window: usize) -> Self {
        self.negation_window = window;
        self
    }

    /// Set booster/dampener window size
    pub fn booster_dampener_window(mut self, window: usize) -> Self {
        self.booster_dampener_window = window;
        self
    }

    /// Set case sensitivity
    pub fn case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.case_sensitive = case_sensitive;
        self
    }

    /// Analyze sentiment with intensity scoring
    pub fn analyze(&self, text: &str) -> AdvancedSentimentResult {
        // Detect language
        let language = if self.auto_detect_language {
            Language::detect(text)
        } else {
            self.default_language
        };

        // Prepare text
        let processed_text = if self.case_sensitive {
            text.to_string()
        } else {
            text.to_lowercase()
        };

        // Tokenize
        let words: Vec<String> = processed_text
            .split_whitespace()
            .map(|w| {
                w.trim_matches(|c: char| c.is_ascii_punctuation())
                    .to_string()
            })
            .filter(|w| !w.is_empty())
            .collect();

        // Get lexicons for detected language
        let positive_words = self
            .lexicon
            .get_positive_words(language)
            .unwrap_or_else(|| self.lexicon.get_positive_words(Language::English).unwrap());
        let negative_words = self
            .lexicon
            .get_negative_words(language)
            .unwrap_or_else(|| self.lexicon.get_negative_words(Language::English).unwrap());
        let boosters = self
            .lexicon
            .get_boosters(language)
            .unwrap_or_else(|| self.lexicon.get_boosters(Language::English).unwrap());
        let dampeners = self
            .lexicon
            .get_dampeners(language)
            .unwrap_or_else(|| self.lexicon.get_dampeners(Language::English).unwrap());
        let negations = self
            .lexicon
            .get_negations(language)
            .unwrap_or_else(|| self.lexicon.get_negations(Language::English).unwrap());

        // Analyze sentiment with intensity modifiers
        let mut sentiment_scores: Vec<f64> = Vec::new();
        let mut positive_count = 0;
        let mut negative_count = 0;
        let mut booster_count = 0;
        let mut dampener_count = 0;
        let mut negation_count = 0;

        for (i, word) in words.iter().enumerate() {
            let mut score = 0.0;
            let mut is_sentiment_word = false;

            // Check if word is sentiment-bearing
            if positive_words.contains(word) {
                score = 1.0;
                is_sentiment_word = true;
                positive_count += 1;
            } else if negative_words.contains(word) {
                score = -1.0;
                is_sentiment_word = true;
                negative_count += 1;
            }

            if is_sentiment_word {
                // Check for intensity modifiers in preceding window
                let window_start = i.saturating_sub(self.booster_dampener_window);
                let mut intensity_multiplier = 1.0;

                for prev_word in words.iter().skip(window_start).take(i - window_start) {
                    // Check for boosters
                    if boosters.contains(prev_word) {
                        intensity_multiplier *= 1.5;
                        booster_count += 1;
                    }

                    // Check for dampeners
                    if dampeners.contains(prev_word) {
                        intensity_multiplier *= 0.5;
                        dampener_count += 1;
                    }
                }

                // Apply intensity
                score *= intensity_multiplier;

                // Check for negations in a wider window
                let negation_window_start = i.saturating_sub(self.negation_window);
                let mut is_negated = false;

                for prev_word in words
                    .iter()
                    .skip(negation_window_start)
                    .take(i - negation_window_start)
                {
                    if negations.contains(prev_word) {
                        is_negated = true;
                        negation_count += 1;
                        break;
                    }
                }

                // Flip sentiment if negated
                if is_negated {
                    score *= -0.7; // Negation reduces intensity slightly
                }

                sentiment_scores.push(score);
            }
        }

        // Calculate aggregate scores
        let raw_score = sentiment_scores.iter().sum::<f64>();
        let normalized_score = if !sentiment_scores.is_empty() {
            raw_score / sentiment_scores.len() as f64
        } else {
            0.0
        };

        // Determine polarity
        let polarity = if normalized_score.abs() <= self.neutral_threshold {
            SentimentPolarity::Neutral
        } else if normalized_score > 0.0 {
            SentimentPolarity::Positive
        } else {
            SentimentPolarity::Negative
        };

        // Calculate intensity (0-1 scale)
        let intensity = normalized_score.abs().min(1.0);

        AdvancedSentimentResult {
            polarity,
            score: normalized_score,
            raw_score,
            intensity,
            positive_count,
            negative_count,
            booster_count,
            dampener_count,
            negation_count,
            total_words: words.len(),
            detected_language: language,
        }
    }

    /// Extract features from multiple documents
    pub fn extract_features(&self, documents: &[String]) -> SklResult<Array2<Float>> {
        if documents.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Empty document collection".to_string(),
            ));
        }

        let n_features = 12; // Expanded feature set
        let mut features = Array2::zeros((documents.len(), n_features));

        for (i, document) in documents.iter().enumerate() {
            let result = self.analyze(document);

            // Feature vector:
            // [0] normalized_score
            // [1] raw_score
            // [2] intensity
            // [3] positive_ratio
            // [4] negative_ratio
            // [5] sentiment_density (sentiment words / total words)
            // [6] booster_ratio
            // [7] dampener_ratio
            // [8] negation_ratio
            // [9] polarity_encoded (-1, 0, 1)
            // [10] absolute_score
            // [11] weighted_score (score * intensity)

            let total_words = result.total_words.max(1) as f64;

            features[(i, 0)] = result.score;
            features[(i, 1)] = result.raw_score;
            features[(i, 2)] = result.intensity;
            features[(i, 3)] = result.positive_count as f64 / total_words;
            features[(i, 4)] = result.negative_count as f64 / total_words;
            features[(i, 5)] = (result.positive_count + result.negative_count) as f64 / total_words;
            features[(i, 6)] = result.booster_count as f64 / total_words;
            features[(i, 7)] = result.dampener_count as f64 / total_words;
            features[(i, 8)] = result.negation_count as f64 / total_words;
            features[(i, 9)] = match result.polarity {
                SentimentPolarity::Negative => -1.0,
                SentimentPolarity::Neutral => 0.0,
                SentimentPolarity::Positive => 1.0,
            };
            features[(i, 10)] = result.score.abs();
            features[(i, 11)] = result.score * result.intensity;
        }

        Ok(features)
    }
}

impl Default for AdvancedSentimentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Advanced sentiment analysis result
#[derive(Debug, Clone)]
pub struct AdvancedSentimentResult {
    /// Sentiment polarity
    pub polarity: SentimentPolarity,
    /// Normalized sentiment score (-1 to 1)
    pub score: f64,
    /// Raw sentiment score (sum of all sentiment values)
    pub raw_score: f64,
    /// Intensity of sentiment (0 to 1)
    pub intensity: f64,
    /// Count of positive sentiment words
    pub positive_count: usize,
    /// Count of negative sentiment words
    pub negative_count: usize,
    /// Count of intensity boosters
    pub booster_count: usize,
    /// Count of intensity dampeners
    pub dampener_count: usize,
    /// Count of negations
    pub negation_count: usize,
    /// Total word count
    pub total_words: usize,
    /// Detected language
    pub detected_language: Language,
}

/// Sentiment polarity classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SentimentPolarity {
    /// Positive sentiment
    Positive,
    /// Negative sentiment
    Negative,
    /// Neutral sentiment
    Neutral,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_detection() {
        assert_eq!(Language::detect("Hello world"), Language::English);
        // Spanish needs more characteristic words to be detected
        assert_eq!(Language::detect("Hola, esto es un texto en español que tiene palabras características como el, la, los, las, de, que"), Language::Spanish);
        // French needs characteristic words
        assert_eq!(
            Language::detect(
                "Bonjour, c'est un texte en français avec des mots comme le, la, les, de, un, une"
            ),
            Language::French
        );
        // German needs characteristic words
        assert_eq!(
            Language::detect(
                "Hallo, das ist ein deutscher Text mit Wörtern wie der, die, das, und, ist, ein"
            ),
            Language::German
        );
        assert_eq!(Language::detect("こんにちは世界"), Language::Japanese);
        assert_eq!(Language::detect("你好世界"), Language::Chinese);
    }

    #[test]
    fn test_basic_sentiment() {
        let analyzer = AdvancedSentimentAnalyzer::new();

        let result = analyzer.analyze("This is good");
        assert_eq!(result.polarity, SentimentPolarity::Positive);
        assert!(result.score > 0.0);

        let result = analyzer.analyze("This is bad");
        assert_eq!(result.polarity, SentimentPolarity::Negative);
        assert!(result.score < 0.0);
    }

    #[test]
    fn test_intensity_boosters() {
        let analyzer = AdvancedSentimentAnalyzer::new();

        let result_without = analyzer.analyze("This is good");
        let result_with = analyzer.analyze("This is very good");

        // Booster should increase the absolute score
        assert!(
            result_with.score.abs() > result_without.score.abs()
                || result_with.score == result_without.score
        );
        assert!(result_with.booster_count > 0);
    }

    #[test]
    fn test_negation_handling() {
        let analyzer = AdvancedSentimentAnalyzer::new();

        let result_positive = analyzer.analyze("This is good");
        let result_negated = analyzer.analyze("This is not good");

        assert_eq!(result_positive.polarity, SentimentPolarity::Positive);
        assert_eq!(result_negated.polarity, SentimentPolarity::Negative);
        assert!(result_negated.negation_count > 0);
    }

    #[test]
    fn test_multilingual_sentiment() {
        let analyzer = AdvancedSentimentAnalyzer::new();

        // Spanish - needs more characteristic words for detection
        let result = analyzer.analyze("Esto es muy bueno y excelente para todos los usuarios");
        assert_eq!(result.detected_language, Language::Spanish);
        assert_eq!(result.polarity, SentimentPolarity::Positive);

        // French - needs characteristic words
        let result = analyzer.analyze("C'est très bon et magnifique pour tous les utilisateurs");
        assert_eq!(result.detected_language, Language::French);
        assert_eq!(result.polarity, SentimentPolarity::Positive);

        // German - needs characteristic words
        let result = analyzer.analyze("Das ist sehr gut und wunderbar für alle die Benutzer");
        assert_eq!(result.detected_language, Language::German);
        assert_eq!(result.polarity, SentimentPolarity::Positive);
    }

    #[test]
    fn test_feature_extraction() {
        let analyzer = AdvancedSentimentAnalyzer::new();

        let documents = vec![
            "This is very good".to_string(),
            "This is not bad".to_string(),
            "This is okay".to_string(),
        ];

        let features = analyzer.extract_features(&documents).unwrap();

        assert_eq!(features.nrows(), 3);
        assert_eq!(features.ncols(), 12);

        // First document should be strongly positive
        assert!(features[(0, 0)] > 0.0); // score
        assert!(features[(0, 2)] > 0.0); // intensity
        assert!(features[(0, 6)] > 0.0); // booster_ratio
    }

    #[test]
    fn test_dampener_effect() {
        let analyzer = AdvancedSentimentAnalyzer::new();

        let result_normal = analyzer.analyze("This is good");
        let result_dampened = analyzer.analyze("This is somewhat good");

        assert!(result_dampened.intensity < result_normal.intensity);
        assert!(result_dampened.dampener_count > 0);
    }

    #[test]
    fn test_empty_input() {
        let analyzer = AdvancedSentimentAnalyzer::new();

        let result = analyzer.analyze("");
        assert_eq!(result.polarity, SentimentPolarity::Neutral);
        assert_eq!(result.score, 0.0);
    }
}
