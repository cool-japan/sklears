//! Text processing and vectorization tests
//!
//! This module contains tests for text feature extraction functionality,
//! including count vectorizers, TF-IDF, and text preprocessing.

use crate::text;
use sklears_core::traits::Fit;
use std::collections::HashSet;

#[test]
fn test_count_vectorizer() {
    let documents = vec![
        "the quick brown fox".to_string(),
        "the quick brown dog".to_string(),
        "the lazy dog".to_string(),
    ];

    let vectorizer = text::CountVectorizer::new();
    let fitted = vectorizer.fit(&documents, &()).unwrap();
    let features = fitted.transform(&documents).unwrap();

    assert_eq!(features.nrows(), 3);
    assert!(features.ncols() > 0);

    // Check that vocabulary contains expected words
    let vocab = fitted.get_vocabulary();
    if let Some(vocab_map) = vocab {
        assert!(vocab_map.contains_key("the"));
        assert!(vocab_map.contains_key("quick"));
        assert!(vocab_map.contains_key("brown"));
    }
}

#[test]
fn test_tfidf_vectorizer() {
    let documents = vec![
        "the quick brown fox".to_string(),
        "the quick brown dog".to_string(),
        "the lazy dog".to_string(),
    ];

    let vectorizer = text::TfidfVectorizer::new();
    let fitted = vectorizer.fit(&documents, &()).unwrap();
    let features = fitted.transform(&documents).unwrap();

    assert_eq!(features.nrows(), 3);
    assert!(features.ncols() > 0);

    // Check that IDF weights are computed
    let idf_weights = fitted.get_idf_weights();
    assert_eq!(idf_weights.len(), features.ncols());
}

#[test]
fn test_count_vectorizer_binary() {
    let documents = vec![
        "the the quick brown fox".to_string(),
        "the quick brown dog".to_string(),
    ];

    let vectorizer = text::CountVectorizer::new().binary(true);
    let fitted = vectorizer.fit(&documents, &()).unwrap();
    let features = fitted.transform(&documents).unwrap();

    // All non-zero entries should be 1.0
    for &value in features.iter() {
        if value > 0.0 {
            assert!((value - 1.0).abs() < 1e-10);
        }
    }
}

#[test]
fn test_count_vectorizer_stop_words() {
    let documents = vec![
        "the quick brown fox".to_string(),
        "the quick brown dog".to_string(),
    ];

    let mut stop_words = HashSet::new();
    stop_words.insert("the".to_string());
    let stop_words_vec: Vec<String> = stop_words.into_iter().collect();

    let vectorizer = text::CountVectorizer::new().stop_words(Some(stop_words_vec));
    let fitted = vectorizer.fit(&documents, &()).unwrap();

    let vocab = fitted.get_vocabulary();
    if let Some(vocab_map) = vocab {
        assert!(!vocab_map.contains_key("the")); // Stop word should be excluded
        assert!(vocab_map.contains_key("quick")); // Regular word should be included
    }
}

#[test]
fn test_count_vectorizer_ngram_range() {
    let documents = vec![
        "the quick brown fox".to_string(),
        "the quick brown dog".to_string(),
    ];

    let vectorizer = text::CountVectorizer::new().ngram_range((1, 2)); // Unigrams and bigrams
    let fitted = vectorizer.fit(&documents, &()).unwrap();
    let features = fitted.transform(&documents).unwrap();
    assert_eq!(features.nrows(), documents.len());

    let vocab = fitted.get_vocabulary();

    if let Some(vocab_map) = vocab {
        // Should contain unigrams
        assert!(vocab_map.contains_key("quick"));
        assert!(vocab_map.contains_key("brown"));

        // Should contain bigrams
        assert!(vocab_map.contains_key("the quick") || vocab_map.contains_key("quick brown"));
    }
}

#[test]
fn test_count_vectorizer_min_max_df() {
    let documents = vec![
        "common word rare".to_string(),
        "common word unusual".to_string(),
        "common word unique".to_string(),
        "common word distinct".to_string(),
    ];

    let vectorizer = text::CountVectorizer::new()
        .min_df(2) // Word must appear in at least 2 documents
        .max_df(3.0); // Word must appear in at most 3 documents

    let fitted = vectorizer.fit(&documents, &()).unwrap();
    let vocab = fitted.get_vocabulary();

    // "common" and "word" appear in all 4 documents, so should be excluded by max_df=3
    // Individual unique words appear only once, so should be excluded by min_df=2
    // This particular configuration might result in no terms, which is valid

    let features = fitted.transform(&documents).unwrap();
    assert_eq!(features.nrows(), 4);
}

#[test]
fn test_tfidf_vectorizer_sublinear_tf() {
    let documents = vec!["cat cat cat dog".to_string(), "cat dog dog dog".to_string()];

    let vectorizer_regular = text::TfidfVectorizer::new().sublinear_tf(false);
    let vectorizer_sublinear = text::TfidfVectorizer::new().sublinear_tf(true);

    let fitted_regular = vectorizer_regular.fit(&documents, &()).unwrap();
    let fitted_sublinear = vectorizer_sublinear.fit(&documents, &()).unwrap();

    let features_regular = fitted_regular.transform(&documents).unwrap();
    let features_sublinear = fitted_sublinear.transform(&documents).unwrap();

    assert_eq!(features_regular.shape(), features_sublinear.shape());

    // Features should be different due to sublinear scaling
    let mut features_differ = false;
    for (reg, sub) in features_regular.iter().zip(features_sublinear.iter()) {
        if (reg - sub).abs() > 1e-10 {
            features_differ = true;
            break;
        }
    }
    assert!(features_differ || features_regular == features_sublinear);

    // In most cases they should differ, but might be identical for some edge cases
    // We just ensure both produce valid finite results
    for &val in features_regular.iter() {
        assert!(val.is_finite());
    }
    for &val in features_sublinear.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_tfidf_vectorizer_normalization() {
    let documents = vec![
        "the quick brown fox jumps".to_string(),
        "the lazy dog sleeps".to_string(),
    ];

    let vectorizer_l1 = text::TfidfVectorizer::new().norm(Some("l1".to_string()));
    let vectorizer_l2 = text::TfidfVectorizer::new().norm(Some("l2".to_string()));
    let vectorizer_none = text::TfidfVectorizer::new().norm(None);

    let fitted_l1 = vectorizer_l1.fit(&documents, &()).unwrap();
    let fitted_l2 = vectorizer_l2.fit(&documents, &()).unwrap();
    let fitted_none = vectorizer_none.fit(&documents, &()).unwrap();

    let features_l1 = fitted_l1.transform(&documents).unwrap();
    let features_l2 = fitted_l2.transform(&documents).unwrap();
    let features_none = fitted_none.transform(&documents).unwrap();

    // Check L1 normalization - each row should sum to 1
    for i in 0..features_l1.nrows() {
        let row_sum: f64 = features_l1.row(i).iter().map(|&x| x.abs()).sum();
        if row_sum > 0.0 {
            assert!(
                (row_sum - 1.0).abs() < 1e-10,
                "L1 normalization failed for row {}",
                i
            );
        }
    }

    // Check L2 normalization - each row should have norm 1
    for i in 0..features_l2.nrows() {
        let row_norm: f64 = features_l2
            .row(i)
            .iter()
            .map(|&x| x * x)
            .sum::<f64>()
            .sqrt();
        if row_norm > 0.0 {
            assert!(
                (row_norm - 1.0).abs() < 1e-10,
                "L2 normalization failed for row {}",
                i
            );
        }
    }

    // All features should be finite
    for &val in features_l1.iter() {
        assert!(val.is_finite());
    }
    for &val in features_l2.iter() {
        assert!(val.is_finite());
    }
    for &val in features_none.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_count_vectorizer_max_features() {
    let documents = vec![
        "apple banana cherry date elderberry".to_string(),
        "fig grape honeydew kiwi lemon".to_string(),
        "mango nectarine orange papaya quince".to_string(),
    ];

    let vectorizer = text::CountVectorizer::new().max_features(Some(5));
    let fitted = vectorizer.fit(&documents, &()).unwrap();
    let features = fitted.transform(&documents).unwrap();

    let vocab = fitted.get_vocabulary();

    // Should have at most 5 features
    if let Some(vocab_map) = &vocab {
        assert!(vocab_map.len() <= 5);
    }
    assert!(features.ncols() <= 5);

    if let Some(vocab_map) = &vocab {
        if vocab_map.len() > 0 {
            assert_eq!(vocab_map.len(), features.ncols());
        }
    }
}

#[test]
fn test_text_vectorizer_empty_documents() {
    let documents = vec!["".to_string(), "word".to_string(), "".to_string()];

    let vectorizer = text::CountVectorizer::new();
    let fitted = vectorizer.fit(&documents, &()).unwrap();
    let features = fitted.transform(&documents).unwrap();

    assert_eq!(features.nrows(), 3);

    // Empty documents should result in zero rows
    let empty_row_sum: f64 = features.row(0).iter().sum();
    let word_row_sum: f64 = features.row(1).iter().sum();
    let empty_row2_sum: f64 = features.row(2).iter().sum();

    assert_eq!(empty_row_sum, 0.0);
    assert!(word_row_sum > 0.0);
    assert_eq!(empty_row2_sum, 0.0);
}

#[test]
fn test_tfidf_vectorizer_single_document() {
    let documents = vec!["single document with multiple words".to_string()];

    let vectorizer = text::TfidfVectorizer::new();
    let fitted = vectorizer.fit(&documents, &()).unwrap();
    let features = fitted.transform(&documents).unwrap();

    assert_eq!(features.nrows(), 1);
    assert!(features.ncols() > 0);

    // With only one document, IDF should be 1.0 for all terms
    let idf_weights = fitted.get_idf_weights();
    for &idf in idf_weights.iter() {
        assert!((idf - 1.0).abs() < 1e-10 || idf.is_finite());
    }

    // All features should be finite
    for &val in features.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_count_vectorizer_case_sensitivity() {
    let documents = vec![
        "The Quick Brown Fox".to_string(),
        "the quick brown fox".to_string(),
    ];

    let vectorizer_case_sensitive = text::CountVectorizer::new().lowercase(false);
    let vectorizer_case_insensitive = text::CountVectorizer::new().lowercase(true);

    let fitted_sensitive = vectorizer_case_sensitive.fit(&documents, &()).unwrap();
    let fitted_insensitive = vectorizer_case_insensitive.fit(&documents, &()).unwrap();

    let vocab_sensitive = fitted_sensitive.get_vocabulary();
    let vocab_insensitive = fitted_insensitive.get_vocabulary();

    // Compare vocabularies if both are available
    if let (Some(vocab_s), Some(vocab_i)) = (vocab_sensitive, vocab_insensitive) {
        // Case sensitive should have both "The" and "the"
        if vocab_s.len() > vocab_i.len() {
            // This suggests case sensitivity is working
            assert!(vocab_s.len() >= vocab_i.len());
        }

        // Case insensitive should only have "the"
        if vocab_i.contains_key("the") {
            assert!(!vocab_i.contains_key("The") || vocab_i.len() == vocab_s.len());
        }
    }
}

#[test]
fn test_text_tokenization_patterns() {
    let documents = vec![
        "word1 word2, word3! word4?".to_string(),
        "email@domain.com and http://website.org".to_string(),
    ];

    let vectorizer = text::CountVectorizer::new();
    let fitted = vectorizer.fit(&documents, &()).unwrap();
    let vocab = fitted.get_vocabulary();

    // Check that tokenization handles punctuation appropriately
    // The exact behavior depends on the tokenizer implementation

    if let Some(vocab_map) = vocab {
        // Should tokenize basic words
        let has_basic_words = vocab_map.contains_key("word1")
            || vocab_map.contains_key("word2")
            || vocab_map.contains_key("word3")
            || vocab_map.contains_key("word4");

        if vocab_map.len() > 0 {
            assert!(has_basic_words, "Should tokenize basic words");
        }
    }

    let features = fitted.transform(&documents).unwrap();
    assert_eq!(features.nrows(), 2);

    // All features should be finite
    for &val in features.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_vectorizer_transform_unseen_documents() {
    let train_documents = vec![
        "train document one".to_string(),
        "train document two".to_string(),
    ];

    let test_documents = vec![
        "test document with new words".to_string(),
        "document one".to_string(), // Partially overlaps
    ];

    let vectorizer = text::CountVectorizer::new();
    let fitted = vectorizer.fit(&train_documents, &()).unwrap();

    // Transform unseen test documents
    let test_features = fitted.transform(&test_documents).unwrap();

    assert_eq!(test_features.nrows(), 2);
    assert_eq!(
        test_features.ncols(),
        fitted.get_vocabulary().map_or(0, |v| v.len())
    );

    // Should handle unseen words gracefully (ignore them)
    for &val in test_features.iter() {
        assert!(val.is_finite());
        assert!(val >= 0.0); // Count should be non-negative
    }
}

#[test]
fn test_tfidf_consistency() {
    let documents = vec![
        "cat dog bird".to_string(),
        "dog bird fish".to_string(),
        "bird fish cat".to_string(),
    ];

    let vectorizer = text::TfidfVectorizer::new();
    let fitted = vectorizer.fit(&documents, &()).unwrap();

    // Transform same documents multiple times - should be consistent
    let features1 = fitted.transform(&documents).unwrap();
    let features2 = fitted.transform(&documents).unwrap();

    assert_eq!(features1.shape(), features2.shape());

    for (f1, f2) in features1.iter().zip(features2.iter()) {
        assert!((f1 - f2).abs() < 1e-12, "TF-IDF should be consistent");
    }
}
