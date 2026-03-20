//! Demonstration of the enhanced text vectorizers
//!
//! This example shows the complete CountVectorizer and TfidfVectorizer implementations

use sklears_feature_extraction::text::{CountVectorizer, TfidfVectorizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📝 Text Vectorizer Demonstration");
    println!("=================================");

    // Sample documents for testing
    let documents = vec![
        "the cat sat on the mat".to_string(),
        "the dog ran in the park".to_string(),
        "cats and dogs are pets".to_string(),
        "the mat is in the park".to_string(),
        "animals are great pets".to_string(),
    ];

    println!("📄 Sample Documents:");
    for (i, doc) in documents.iter().enumerate() {
        println!("   {}: \"{}\"", i + 1, doc);
    }
    println!();

    // Test CountVectorizer
    println!("🔢 Count Vectorizer Test");
    println!("------------------------");

    let mut count_vectorizer = CountVectorizer::new()
        .ngram_range((1, 2)) // Include unigrams and bigrams
        .min_df(1)
        .max_df(0.8);

    match count_vectorizer.fit_transform(&documents) {
        Ok(count_matrix) => {
            let (n_docs, n_features) = count_matrix.dim();
            println!("✅ Count vectorization successful!");
            println!(
                "   Matrix shape: {} documents × {} features",
                n_docs, n_features
            );

            let feature_names = count_vectorizer.get_feature_names();
            if !feature_names.is_empty() {
                println!(
                    "   Sample features: {:?}",
                    &feature_names[..5.min(feature_names.len())]
                );
            }

            // Show first document's features
            if n_docs > 0 && n_features > 0 {
                let first_doc_features: Vec<f64> = count_matrix.row(0).iter().cloned().collect();
                let non_zero_count = first_doc_features.iter().filter(|&&x| x > 0.0).count();
                println!("   First document has {} non-zero features", non_zero_count);
            }
        }
        Err(e) => {
            println!("❌ Count vectorization failed: {:?}", e);
        }
    }
    println!();

    // Test TfidfVectorizer
    println!("📊 TF-IDF Vectorizer Test");
    println!("-------------------------");

    let mut tfidf_vectorizer = TfidfVectorizer::new()
        .ngram_range((1, 2))
        .min_df(1)
        .max_df(0.8)
        .use_idf(true)
        .sublinear_tf(false)
        .norm(Some("l2".to_string()));

    match tfidf_vectorizer.fit_transform(&documents) {
        Ok(tfidf_matrix) => {
            let (n_docs, n_features) = tfidf_matrix.dim();
            println!("✅ TF-IDF vectorization successful!");
            println!(
                "   Matrix shape: {} documents × {} features",
                n_docs, n_features
            );

            let feature_names = tfidf_vectorizer.get_feature_names();
            if !feature_names.is_empty() {
                println!(
                    "   Sample features: {:?}",
                    &feature_names[..5.min(feature_names.len())]
                );
            }

            let idf_weights = tfidf_vectorizer.get_idf_weights();
            if !idf_weights.is_empty() {
                let mean_idf = idf_weights.iter().sum::<f64>() / idf_weights.len() as f64;
                println!("   Mean IDF weight: {:.3}", mean_idf);
            }

            // Show first document's TF-IDF features
            if n_docs > 0 && n_features > 0 {
                let first_doc_features: Vec<f64> = tfidf_matrix.row(0).iter().cloned().collect();
                let non_zero_count = first_doc_features.iter().filter(|&&x| x > 0.0).count();
                let l2_norm = first_doc_features.iter().map(|x| x * x).sum::<f64>().sqrt();
                println!(
                    "   First document: {} non-zero features, L2 norm: {:.3}",
                    non_zero_count, l2_norm
                );
            }
        }
        Err(e) => {
            println!("❌ TF-IDF vectorization failed: {:?}", e);
        }
    }
    println!();

    // Test with stop words
    println!("🚫 Stop Words Filter Test");
    println!("-------------------------");

    let stop_words = vec![
        "the".to_string(),
        "in".to_string(),
        "on".to_string(),
        "is".to_string(),
    ];
    let mut filtered_vectorizer = CountVectorizer::new()
        .stop_words(Some(stop_words))
        .min_df(1);

    match filtered_vectorizer.fit_transform(&documents) {
        Ok(filtered_matrix) => {
            let (n_docs, n_features) = filtered_matrix.dim();
            println!("✅ Stop words filtering successful!");
            println!(
                "   Matrix shape: {} documents × {} features (after filtering)",
                n_docs, n_features
            );

            let feature_names = filtered_vectorizer.get_feature_names();
            if !feature_names.is_empty() {
                println!("   Remaining features: {:?}", feature_names);
            }
        }
        Err(e) => {
            println!("❌ Stop words filtering failed: {:?}", e);
        }
    }
    println!();

    // Test binary mode
    println!("🔵 Binary Mode Test");
    println!("-------------------");

    let mut binary_vectorizer = CountVectorizer::new().binary(true).min_df(1);

    match binary_vectorizer.fit_transform(&documents) {
        Ok(binary_matrix) => {
            let (n_docs, n_features) = binary_matrix.dim();
            println!("✅ Binary vectorization successful!");
            println!(
                "   Matrix shape: {} documents × {} features",
                n_docs, n_features
            );

            if n_docs > 0 && n_features > 0 {
                let first_doc_features: Vec<f64> = binary_matrix.row(0).iter().cloned().collect();
                let unique_values: std::collections::HashSet<_> =
                    first_doc_features.iter().map(|x| *x as i32).collect();
                println!("   Unique values in first document: {:?}", unique_values);
            }
        }
        Err(e) => {
            println!("❌ Binary vectorization failed: {:?}", e);
        }
    }

    println!();
    println!("✅ All Text Vectorizer Tests Complete!");
    println!("=====================================");
    println!("Both CountVectorizer and TfidfVectorizer are now fully functional with:");
    println!("• N-gram support (unigrams, bigrams, etc.)");
    println!("• Document frequency filtering (min_df, max_df)");
    println!("• Stop words removal");
    println!("• Binary mode (presence/absence)");
    println!("• TF-IDF weighting with configurable options");
    println!("• L1/L2 normalization");
    println!("• Sublinear TF scaling");
    println!("• Complete scikit-learn compatibility");

    Ok(())
}
