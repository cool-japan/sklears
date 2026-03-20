use scirs2_core::ndarray::array;
use sklears_metrics::ranking::*;

fn main() {
    println!("=== Ranking Metrics Demo ===\n");

    // Binary classification scores
    let y_true = array![0, 0, 1, 1, 0, 1, 0, 1];
    let y_score = array![0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.25, 0.85];

    // ROC AUC Score
    match roc_auc_score(&y_true, &y_score) {
        Ok(score) => println!("ROC AUC Score: {:.3}", score),
        Err(e) => println!("Error computing ROC AUC: {}", e),
    }

    // Average Precision Score
    match average_precision_score(&y_true, &y_score) {
        Ok(score) => println!("Average Precision Score: {:.3}", score),
        Err(e) => println!("Error computing Average Precision: {}", e),
    }

    // ROC Curve
    match roc_curve(&y_true, &y_score) {
        Ok((fpr, tpr, thresholds)) => {
            println!("\nROC Curve Points:");
            println!("FPR: {:?}", fpr.to_vec());
            println!("TPR: {:?}", tpr.to_vec());
            println!("Thresholds: {:?}", thresholds.to_vec());
        }
        Err(e) => println!("Error computing ROC curve: {}", e),
    }

    // Precision-Recall Curve
    match precision_recall_curve(&y_true, &y_score) {
        Ok((precision, recall, thresholds)) => {
            println!("\nPrecision-Recall Curve Points:");
            println!("Precision: {:?}", precision.to_vec());
            println!("Recall: {:?}", recall.to_vec());
            println!("Thresholds: {:?}", thresholds.to_vec());
        }
        Err(e) => println!("Error computing PR curve: {}", e),
    }

    // Ranking metrics (DCG/NDCG)
    println!("\n=== Ranking Metrics (DCG/NDCG) ===");
    let relevance_scores = array![3.0, 2.0, 3.0, 0.0, 1.0, 2.0];
    let predicted_scores = array![0.9, 0.7, 0.8, 0.2, 0.5, 0.6];

    match dcg_score(&relevance_scores, &predicted_scores, None) {
        Ok(score) => println!("DCG@all: {:.3}", score),
        Err(e) => println!("Error computing DCG: {}", e),
    }

    match dcg_score(&relevance_scores, &predicted_scores, Some(3)) {
        Ok(score) => println!("DCG@3: {:.3}", score),
        Err(e) => println!("Error computing DCG@3: {}", e),
    }

    match ndcg_score(&relevance_scores, &predicted_scores, None) {
        Ok(score) => println!("NDCG@all: {:.3}", score),
        Err(e) => println!("Error computing NDCG: {}", e),
    }

    match ndcg_score(&relevance_scores, &predicted_scores, Some(3)) {
        Ok(score) => println!("NDCG@3: {:.3}", score),
        Err(e) => println!("Error computing NDCG@3: {}", e),
    }

    // Multi-label ranking (Coverage Error)
    println!("\n=== Multi-label Ranking ===");
    let y_true_ml = array![[1, 0, 0], [0, 1, 1], [0, 0, 1], [1, 1, 0]];
    let y_score_ml = array![
        [0.9, 0.1, 0.2],
        [0.1, 0.9, 0.8],
        [0.2, 0.3, 0.9],
        [0.8, 0.7, 0.1]
    ];

    match coverage_error(&y_true_ml, &y_score_ml) {
        Ok(error) => println!("Coverage Error: {:.3}", error),
        Err(e) => println!("Error computing Coverage Error: {}", e),
    }

    // AUC computation example
    println!("\n=== AUC Computation ===");
    let x = array![0.0, 0.25, 0.5, 0.75, 1.0];
    let y = array![0.0, 0.4, 0.7, 0.9, 1.0];

    match auc(&x, &y) {
        Ok(area) => println!("Area Under Curve: {:.3}", area),
        Err(e) => println!("Error computing AUC: {}", e),
    }
}
