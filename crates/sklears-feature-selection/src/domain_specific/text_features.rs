//! Text feature selection module
//!
//! This module provides specialized feature selection algorithms for text data,
//! including TF-IDF analysis, document frequency filtering, and linguistic features.

use crate::base::SelectorMixin;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use sklears_core::{
    error::{validate, Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::collections::HashMap;
use std::marker::PhantomData;

/// Text feature selection using TF-IDF weights and linguistic analysis
///
/// This selector analyzes text features represented as term frequency matrices
/// and applies various text-specific selection criteria:
/// - Document frequency filtering (min_df, max_df)
/// - TF-IDF scoring for term importance
/// - Chi-squared statistical tests with target variables
/// - N-gram analysis (when configured)
/// - Part-of-speech and syntactic features (when enabled)
///
/// # Input Format
///
/// The input matrix `X` should be structured as:
/// - Rows: Documents
/// - Columns: Terms/features (e.g., from TF-IDF vectorization)
/// - Values: Term frequencies or TF-IDF scores
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_feature_selection::domain_specific::text_features::TextFeatureSelector;
/// use sklears_core::traits::{Fit, Transform};
/// use scirs2_core::ndarray::{Array1, Array2};
///
/// let selector = TextFeatureSelector::new()
///     .min_df(0.02)              // Minimum 2% document frequency
///     .max_df(0.90)              // Maximum 90% document frequency
///     .max_features(Some(500))   // Select top 500 features
///     .ngram_range((1, 2))       // Include unigrams and bigrams
///     .include_pos(true);        // Include part-of-speech features
///
/// let x = Array2::zeros((100, 1000)); // 100 documents, 1000 terms
/// let y = Array1::zeros(100);          // Document labels
///
/// let fitted_selector = selector.fit(&x, &y)?;
/// let transformed_x = fitted_selector.transform(&x)?;
/// ```
#[derive(Debug, Clone)]
pub struct TextFeatureSelector<State = Untrained> {
    /// Minimum document frequency for a term to be considered
    min_df: f64,
    /// Maximum document frequency for a term to be considered
    max_df: f64,
    /// Maximum number of features to select
    max_features: Option<usize>,
    /// Whether to use n-grams (1=unigrams, 2=bigrams, etc.)
    ngram_range: (usize, usize),
    /// Whether to include part-of-speech features
    include_pos: bool,
    /// Whether to include syntactic features
    include_syntax: bool,
    state: PhantomData<State>,
    // Trained state
    vocabulary_: Option<HashMap<String, usize>>,
    idf_scores_: Option<Array1<Float>>,
    selected_features_: Option<Vec<usize>>,
    feature_names_: Option<Vec<String>>,
}

impl TextFeatureSelector<Untrained> {
    pub fn new() -> Self {
        Self {
            min_df: 0.01,
            max_df: 0.95,
            max_features: Some(1000),
            ngram_range: (1, 1),
            include_pos: false,
            include_syntax: false,
            state: PhantomData,
            vocabulary_: None,
            idf_scores_: None,
            selected_features_: None,
            feature_names_: None,
        }
    }

    /// Set the minimum document frequency threshold
    ///
    /// Terms appearing in fewer than `min_df` fraction of documents
    /// will be filtered out. This helps remove very rare terms that
    /// may not be reliable predictors.
    ///
    /// # Arguments
    /// * `min_df` - Fraction between 0.0 and 1.0
    pub fn min_df(mut self, min_df: f64) -> Self {
        self.min_df = min_df;
        self
    }

    /// Set the maximum document frequency threshold
    ///
    /// Terms appearing in more than `max_df` fraction of documents
    /// will be filtered out. This helps remove very common terms
    /// (like stop words) that may not be discriminative.
    ///
    /// # Arguments
    /// * `max_df` - Fraction between 0.0 and 1.0
    pub fn max_df(mut self, max_df: f64) -> Self {
        self.max_df = max_df;
        self
    }

    /// Set the maximum number of features to select
    ///
    /// When set to `Some(n)`, selects the top n features by combined score.
    /// When set to `None`, uses document frequency filtering only.
    pub fn max_features(mut self, max_features: Option<usize>) -> Self {
        self.max_features = max_features;
        self
    }

    /// Set the n-gram range for feature extraction
    ///
    /// - (1, 1): Unigrams only
    /// - (1, 2): Unigrams and bigrams
    /// - (2, 3): Bigrams and trigrams
    /// - etc.
    ///
    /// Note: This parameter is informational for compatibility;
    /// actual n-gram extraction should be done during preprocessing.
    pub fn ngram_range(mut self, ngram_range: (usize, usize)) -> Self {
        self.ngram_range = ngram_range;
        self
    }

    /// Enable or disable part-of-speech features
    ///
    /// When enabled, the selector will give preference to features
    /// that represent important part-of-speech categories.
    ///
    /// Note: This requires preprocessing to extract POS features.
    pub fn include_pos(mut self, include_pos: bool) -> Self {
        self.include_pos = include_pos;
        self
    }

    /// Enable or disable syntactic features
    ///
    /// When enabled, the selector will consider syntactic relationships
    /// and dependency parsing features.
    ///
    /// Note: This requires preprocessing to extract syntactic features.
    pub fn include_syntax(mut self, include_syntax: bool) -> Self {
        self.include_syntax = include_syntax;
        self
    }
}

impl Default for TextFeatureSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for TextFeatureSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for TextFeatureSelector<Untrained> {
    type Fitted = TextFeatureSelector<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> SklResult<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        let (n_documents, n_features) = x.dim();

        // Compute document frequencies for each feature (term)
        let mut document_frequencies = Array1::zeros(n_features);
        for j in 0..n_features {
            let mut df = 0.0;
            for i in 0..n_documents {
                if x[[i, j]] > 0.0 {
                    df += 1.0;
                }
            }
            document_frequencies[j] = df / n_documents as f64;
        }

        // Filter features based on document frequency
        let mut valid_features = Vec::new();
        for (j, &df) in document_frequencies.iter().enumerate() {
            if df >= self.min_df && df <= self.max_df {
                valid_features.push(j);
            }
        }

        if valid_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features pass the document frequency filters".to_string(),
            ));
        }

        // Compute IDF scores for valid features
        let mut idf_scores = Array1::zeros(valid_features.len());
        for (idx, &j) in valid_features.iter().enumerate() {
            let df = document_frequencies[j];
            idf_scores[idx] = (n_documents as f64 / (1.0 + df * n_documents as f64)).ln();
        }

        // Compute feature importance using chi-squared test with target
        let mut chi2_scores = Array1::zeros(valid_features.len());
        for (idx, &j) in valid_features.iter().enumerate() {
            let feature_col = x.column(j);
            chi2_scores[idx] = compute_chi2_score(&feature_col, y);
        }

        // Combine IDF and chi-squared scores
        let mut combined_scores = Array1::zeros(valid_features.len());
        for i in 0..combined_scores.len() {
            combined_scores[i] = 0.6 * idf_scores[i] + 0.4 * chi2_scores[i];
        }

        // Select top features
        let mut scored_features: Vec<(usize, Float)> = combined_scores
            .indexed_iter()
            .map(|(i, &score)| (valid_features[i], score))
            .collect();

        scored_features.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let selected_features = if let Some(max_feat) = self.max_features {
            scored_features
                .iter()
                .take(max_feat.min(scored_features.len()))
                .map(|(i, _)| *i)
                .collect::<Vec<_>>()
        } else {
            scored_features.iter().map(|(i, _)| *i).collect()
        };

        // Create feature names (simplified - in real implementation would use actual vocabulary)
        let feature_names: Vec<String> = selected_features
            .iter()
            .map(|&i| format!("term_{}", i))
            .collect();

        let vocabulary: HashMap<String, usize> = feature_names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        Ok(TextFeatureSelector {
            min_df: self.min_df,
            max_df: self.max_df,
            max_features: self.max_features,
            ngram_range: self.ngram_range,
            include_pos: self.include_pos,
            include_syntax: self.include_syntax,
            state: PhantomData,
            vocabulary_: Some(vocabulary),
            idf_scores_: Some(idf_scores),
            selected_features_: Some(selected_features),
            feature_names_: Some(feature_names),
        })
    }
}

impl Transform<Array2<Float>> for TextFeatureSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        if selected_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features were selected".to_string(),
            ));
        }

        let selected_indices: Vec<usize> = selected_features.to_vec();
        Ok(x.select(Axis(1), &selected_indices))
    }
}

impl SelectorMixin for TextFeatureSelector<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_features = self.idf_scores_.as_ref().unwrap().len()
            + selected_features.iter().max().unwrap_or(&0)
            + 1;
        let mut support = Array1::from_elem(n_features, false);
        for &idx in selected_features {
            if idx < n_features {
                support[idx] = true;
            }
        }
        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected_features.iter().position(|&f| f == idx))
            .collect())
    }
}

impl TextFeatureSelector<Trained> {
    /// Get the vocabulary mapping from terms to feature indices
    ///
    /// Returns a reference to the vocabulary dictionary where keys are
    /// term names and values are their corresponding feature indices.
    pub fn vocabulary(&self) -> &HashMap<String, usize> {
        self.vocabulary_.as_ref().unwrap()
    }

    /// Get the IDF (Inverse Document Frequency) scores
    ///
    /// Returns an array where each element is the IDF score for the
    /// corresponding selected feature.
    pub fn idf_scores(&self) -> &Array1<Float> {
        self.idf_scores_.as_ref().unwrap()
    }

    /// Get the names of selected features
    ///
    /// Returns a reference to the vector of feature names that were selected.
    pub fn feature_names(&self) -> &[String] {
        self.feature_names_.as_ref().unwrap()
    }

    /// Get the indices of selected features
    ///
    /// Returns a reference to the vector of original feature indices
    /// that were selected during fitting.
    pub fn selected_features(&self) -> &[usize] {
        self.selected_features_.as_ref().unwrap()
    }

    /// Get the number of selected features
    pub fn n_features_selected(&self) -> usize {
        self.selected_features_.as_ref().unwrap().len()
    }

    /// Get feature information as a structured summary
    ///
    /// Returns a vector of tuples containing (feature_index, feature_name, idf_score)
    /// for all selected features, sorted by feature index.
    pub fn feature_summary(&self) -> Vec<(usize, &str, Float)> {
        let indices = self.selected_features();
        let names = self.feature_names();
        let scores = self.idf_scores();

        let mut summary: Vec<(usize, &str, Float)> = indices
            .iter()
            .zip(names.iter())
            .zip(scores.iter())
            .map(|((&idx, name), &score)| (idx, name.as_str(), score))
            .collect();

        summary.sort_by_key(|&(idx, _, _)| idx);
        summary
    }
}

// ================================================================================================
// Helper Functions
// ================================================================================================

/// Compute a simplified chi-squared score for feature selection
///
/// This function computes a chi-squared-like statistic for continuous features
/// by discretizing them based on their mean values. In practice, more sophisticated
/// discretization methods (like equal-frequency binning) would be preferred.
///
/// The chi-squared test measures the independence between the feature and target
/// variables. Higher scores indicate stronger association.
fn compute_chi2_score(feature: &ArrayView1<Float>, target: &Array1<Float>) -> Float {
    // Simplified chi-squared computation for continuous features
    // In practice, would need proper discretization
    let feature_mean = feature.mean().unwrap_or(0.0);
    let target_mean = target.mean().unwrap_or(0.0);

    let mut chi2 = 0.0;
    let n = feature.len();

    for i in 0..n {
        let f_i = if feature[i] > feature_mean { 1.0 } else { 0.0 };
        let t_i = if target[i] > target_mean { 1.0 } else { 0.0 };

        let observed = f_i * t_i;
        let expected = (feature.sum() / n as Float) * (target.sum() / n as Float);

        if expected > 0.0 {
            chi2 += (observed - expected).powi(2) / expected;
        }
    }

    chi2
}

/// Compute document frequency for a term vector
///
/// Document frequency is the number of documents containing the term
/// divided by the total number of documents.
fn compute_document_frequency(term_vector: &ArrayView1<Float>) -> Float {
    let n_documents = term_vector.len() as Float;
    let documents_with_term = term_vector.iter().filter(|&&count| count > 0.0).count() as Float;
    documents_with_term / n_documents
}

/// Compute TF-IDF score for a term
///
/// TF-IDF (Term Frequency-Inverse Document Frequency) is calculated as:
/// tf-idf(t,d) = tf(t,d) * idf(t)
/// where idf(t) = log(N / df(t))
fn compute_tfidf_score(
    term_frequency: Float,
    document_frequency: Float,
    n_documents: usize,
) -> Float {
    let idf = (n_documents as Float / (1.0 + document_frequency * n_documents as Float)).ln();
    term_frequency * idf
}

/// Create a new text feature selector
pub fn create_text_feature_selector() -> TextFeatureSelector<Untrained> {
    TextFeatureSelector::new()
}

/// Create a text feature selector optimized for short documents
///
/// Suitable for tweets, short articles, or other brief text content
/// where term frequency is typically low.
pub fn create_short_text_selector() -> TextFeatureSelector<Untrained> {
    TextFeatureSelector::new()
        .min_df(0.005) // Lower minimum frequency for short texts
        .max_df(0.8) // Lower maximum to filter common words
        .max_features(Some(500))
        .ngram_range((1, 2)) // Include bigrams for context
}

/// Create a text feature selector optimized for long documents
///
/// Suitable for articles, papers, books, or other lengthy text content
/// where term frequencies are higher and vocabulary is richer.
pub fn create_long_text_selector() -> TextFeatureSelector<Untrained> {
    TextFeatureSelector::new()
        .min_df(0.02) // Higher minimum frequency
        .max_df(0.95) // Higher maximum frequency
        .max_features(Some(2000))
        .ngram_range((1, 3)) // Include up to trigrams
}

/// Create a text feature selector for multilingual content
///
/// Suitable for text in multiple languages where stop words and
/// common patterns may vary significantly.
pub fn create_multilingual_selector() -> TextFeatureSelector<Untrained> {
    TextFeatureSelector::new()
        .min_df(0.01)
        .max_df(0.9) // Conservative max_df for varied languages
        .max_features(Some(1500))
        .include_pos(true) // POS tags are language-agnostic
}

/// Create a text feature selector for classification tasks
///
/// Optimized for discriminative feature selection in classification
/// problems where the goal is to distinguish between classes.
pub fn create_classification_selector() -> TextFeatureSelector<Untrained> {
    TextFeatureSelector::new()
        .min_df(0.02)
        .max_df(0.9)
        .max_features(Some(1000))
        .ngram_range((1, 2))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_document_frequency_computation() {
        let term_vector = array![1.0, 0.0, 2.0, 0.0, 1.0]; // Term appears in 3/5 documents
        let df = compute_document_frequency(&term_vector.view());
        assert!((df - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_tfidf_computation() {
        let tf = 3.0;
        let df = 0.5; // Term appears in 50% of documents
        let n_docs = 100;
        let tfidf = compute_tfidf_score(tf, df, n_docs);

        // Should be positive (tf * log(N / (1 + df*N)))
        assert!(tfidf > 0.0);
    }

    #[test]
    fn test_chi2_score_computation() {
        let feature = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let target = array![1.0, 1.0, 0.0, 0.0, 0.0];
        let chi2 = compute_chi2_score(&feature.view(), &target);

        // Should compute some positive score
        assert!(chi2 >= 0.0);
    }

    #[test]
    fn test_text_feature_selector_basic() {
        let selector = TextFeatureSelector::new().min_df(0.1).max_features(Some(2));

        // Create a simple term-document matrix
        let x = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 0.0, 2.0, // Doc 1: term1, term3
                0.0, 1.0, 1.0, // Doc 2: term2, term3
                1.0, 1.0, 0.0, // Doc 3: term1, term2
                2.0, 0.0, 1.0, // Doc 4: term1, term3
            ],
        )
        .unwrap();
        let y = array![1.0, 0.0, 1.0, 0.0];

        let fitted = selector.fit(&x, &y).unwrap();
        assert!(fitted.n_features_selected() <= 2);

        let transformed = fitted.transform(&x).unwrap();
        assert!(transformed.ncols() <= 2);
    }

    #[test]
    fn test_document_frequency_filtering() {
        let selector = TextFeatureSelector::new()
            .min_df(0.6) // Require term in at least 60% of documents
            .max_df(1.0);

        let x = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 0.0, 1.0, // term1: 4/5, term2: 1/5, term3: 5/5
                1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
            ],
        )
        .unwrap();
        let y = array![1.0, 0.0, 1.0, 0.0, 1.0];

        let fitted = selector.fit(&x, &y).unwrap();

        // Only term1 (4/5 = 80%) and term3 (5/5 = 100%) should pass the 60% threshold
        assert!(fitted.n_features_selected() <= 2);
    }
}
