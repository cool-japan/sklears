use super::*;
use crate::mock::{MockPredictor, MockTransformer};
use crate::PipelinePredictor;
use scirs2_core::ndarray::Array2;
use sklears_core::traits::Fit;

/// A genuine 1-D threshold classifier used to exercise real fit/predict/score
/// in cross-validation tests.  When `learn` is true it fits the decision
/// boundary to the midpoint between the per-class means of feature 0; when
/// false it is a degenerate predictor that always returns class 0.
#[derive(Debug, Clone)]
struct ThresholdClassifier {
    threshold: f64,
    learn: bool,
}

impl PipelinePredictor for ThresholdClassifier {
    fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<f64>> {
        if !self.learn {
            return Ok(Array1::zeros(x.nrows()));
        }
        let predictions: Vec<f64> = x
            .rows()
            .into_iter()
            .map(|row| if row[0] > self.threshold { 1.0 } else { 0.0 })
            .collect();
        Ok(Array1::from_vec(predictions))
    }

    fn fit(&mut self, x: &ArrayView2<'_, Float>, y: &ArrayView1<'_, Float>) -> SklResult<()> {
        if self.learn {
            let mut sum_class0 = 0.0;
            let mut count_class0 = 0.0;
            let mut sum_class1 = 0.0;
            let mut count_class1 = 0.0;
            for (row, &label) in x.rows().into_iter().zip(y.iter()) {
                if label < 0.5 {
                    sum_class0 += row[0];
                    count_class0 += 1.0;
                } else {
                    sum_class1 += row[0];
                    count_class1 += 1.0;
                }
            }
            let mean0 = if count_class0 > 0.0 {
                sum_class0 / count_class0
            } else {
                0.0
            };
            let mean1 = if count_class1 > 0.0 {
                sum_class1 / count_class1
            } else {
                0.0
            };
            self.threshold = (mean0 + mean1) / 2.0;
        }
        Ok(())
    }

    fn clone_predictor(&self) -> Box<dyn PipelinePredictor> {
        Box::new(self.clone())
    }
}

/// Build a small trained pipeline (one transformer + one predictor) and
/// verify that `execute_trained` returns one prediction per input row.
#[test]
fn test_execute_trained_returns_one_prediction_per_row() {
    // 4 samples, 2 features
    let x = Array2::from_shape_vec((4, 2), vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .expect("valid shape");
    let y = scirs2_core::ndarray::Array1::from_vec(vec![0.0_f64, 1.0, 0.0, 1.0]);

    // Construct and train pipeline
    let mut pipeline = Pipeline::new();
    pipeline.add_step("scaler".to_string(), Box::new(MockTransformer::new()));
    pipeline.set_estimator(Box::new(MockPredictor::new()));

    let trained = pipeline
        .fit(&x.view(), &Some(&y.view()))
        .expect("fit must succeed");

    let executor = RobustPipelineExecutor::new();
    let preds = executor
        .execute_trained(&trained, &x.view(), None)
        .expect("execute_trained must succeed");

    assert_eq!(
        preds.len(),
        x.nrows(),
        "one prediction per input row expected"
    );
    for p in preds.iter() {
        assert!(p.is_finite(), "prediction must be finite");
    }
}

/// Empty input must return an error, not panic.
#[test]
fn test_execute_trained_empty_input_errors() {
    let x_train =
        Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0, 3.0, 4.0]).expect("valid shape");
    let y_train = scirs2_core::ndarray::Array1::from_vec(vec![0.0_f64, 1.0]);

    let mut pipeline = Pipeline::new();
    pipeline.set_estimator(Box::new(MockPredictor::new()));
    let trained = pipeline
        .fit(&x_train.view(), &Some(&y_train.view()))
        .expect("fit must succeed");

    let x_empty: Array2<f64> = Array2::zeros((0, 2));
    let executor = RobustPipelineExecutor::new();
    let result = executor.execute_trained(&trained, &x_empty.view(), None);
    assert!(result.is_err(), "empty input should return an error");
}

/// K-fold partition must cover every sample exactly once across validation
/// folds, keep train/validation disjoint, and produce near-equal fold sizes
/// following the sklearn convention (first `n % k` folds get one extra).
#[test]
fn test_k_fold_indices_partition_is_exact_and_near_equal() {
    // n = 10, k = 3 -> validation sizes [4, 3, 3] (first 10 % 3 = 1 fold larger).
    let folds = k_fold_indices(10, 3);
    assert_eq!(folds.len(), 3, "expected exactly k folds");

    let validation_sizes: Vec<usize> = folds.iter().map(|(_, val)| val.len()).collect();
    assert_eq!(validation_sizes, vec![4, 3, 3]);

    let max_size = *validation_sizes.iter().max().expect("non-empty");
    let min_size = *validation_sizes.iter().min().expect("non-empty");
    assert!(
        max_size - min_size <= 1,
        "fold sizes must differ by at most 1"
    );

    let mut coverage = vec![0usize; 10];
    for (train, validation) in &folds {
        // Train and validation jointly partition all samples.
        assert_eq!(train.len() + validation.len(), 10);
        // Disjoint: no validation index appears in the train set.
        for &index in validation {
            assert!(!train.contains(&index), "train/val overlap at {index}");
            coverage[index] += 1;
        }
    }
    assert!(
        coverage.iter().all(|&count| count == 1),
        "each sample must appear in exactly one validation fold: {coverage:?}"
    );

    // Evenly divisible case: n = 8, k = 4 -> four folds of size 2.
    let even = k_fold_indices(8, 4);
    assert_eq!(even.len(), 4);
    assert!(even.iter().all(|(_, val)| val.len() == 2));
}

/// A regression CV must return a finite, negative (= negated MSE) score, and
/// a clearly worse-fitting configuration must score strictly lower — proving
/// the score genuinely depends on the pipeline configuration.
#[test]
fn test_regression_cv_is_finite_negative_and_config_dependent() {
    // y = 2 * x on a single feature.
    let x = Array2::from_shape_vec((8, 1), vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .expect("valid shape");
    let y = Array1::from_vec(vec![2.0_f64, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);

    let build = |scale: f64| {
        let mut pipeline = Pipeline::new();
        pipeline.add_step(
            "scaler".to_string(),
            Box::new(MockTransformer::with_scale(scale)),
        );
        pipeline.set_estimator(Box::new(MockPredictor::new()));
        pipeline
    };

    let optimizer = PipelineOptimizer::new()
        .cv_folds(4)
        .scoring(ScoringMetric::MeanSquaredError);

    let score_good = optimizer
        .cross_validate_pipeline(&build(1.0), &x.view(), &y.view())
        .expect("CV of the good configuration must succeed");
    let score_bad = optimizer
        .cross_validate_pipeline(&build(10.0), &x.view(), &y.view())
        .expect("CV of the bad configuration must succeed");

    assert!(score_good.is_finite(), "good score must be finite");
    assert!(score_bad.is_finite(), "bad score must be finite");
    assert!(
        score_good <= 0.0,
        "negated MSE must be non-positive, got {score_good}"
    );
    assert!(
        score_good > score_bad,
        "better-fitting config must score higher: good={score_good}, bad={score_bad}"
    );
}

/// Accuracy on a linearly separable toy problem should be ~1.0 for a real
/// learning classifier, while a degenerate always-class-0 predictor scores
/// strictly lower.
#[test]
fn test_accuracy_high_on_separable_and_degenerate_lower() {
    // Classes interleaved so every contiguous (no-shuffle) fold sees both
    // classes; class 0 has small feature values, class 1 has large ones.
    let x = Array2::from_shape_vec((8, 1), vec![0.0_f64, 5.0, 0.5, 5.5, 0.2, 5.2, 0.8, 5.8])
        .expect("valid shape");
    let y = Array1::from_vec(vec![0.0_f64, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);

    let build = |learn: bool| {
        let mut pipeline = Pipeline::new();
        pipeline.add_step(
            "scaler".to_string(),
            Box::new(MockTransformer::with_scale(1.0)),
        );
        pipeline.set_estimator(Box::new(ThresholdClassifier {
            threshold: 0.0,
            learn,
        }));
        pipeline
    };

    let optimizer = PipelineOptimizer::new()
        .cv_folds(4)
        .scoring(ScoringMetric::Accuracy);

    let accuracy_good = optimizer
        .cross_validate_pipeline(&build(true), &x.view(), &y.view())
        .expect("CV of the learning classifier must succeed");
    let accuracy_degenerate = optimizer
        .cross_validate_pipeline(&build(false), &x.view(), &y.view())
        .expect("CV of the degenerate classifier must succeed");

    assert!(
        accuracy_good > 0.99,
        "separable accuracy should be ~1.0, got {accuracy_good}"
    );
    assert!(
        (0.0..=1.0).contains(&accuracy_degenerate),
        "accuracy must lie in [0, 1], got {accuracy_degenerate}"
    );
    assert!(
        accuracy_degenerate < accuracy_good,
        "degenerate predictor must score lower: good={accuracy_good}, degenerate={accuracy_degenerate}"
    );
}

/// F1 macro scoring must produce a perfect 1.0 when the learning classifier
/// separates the toy data, demonstrating the F1 branch is wired to real
/// predictions.
#[test]
fn test_f1_macro_perfect_on_separable() {
    let x = Array2::from_shape_vec((8, 1), vec![0.0_f64, 5.0, 0.5, 5.5, 0.2, 5.2, 0.8, 5.8])
        .expect("valid shape");
    let y = Array1::from_vec(vec![0.0_f64, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);

    let mut pipeline = Pipeline::new();
    pipeline.set_estimator(Box::new(ThresholdClassifier {
        threshold: 0.0,
        learn: true,
    }));

    let optimizer = PipelineOptimizer::new()
        .cv_folds(4)
        .scoring(ScoringMetric::F1Score);

    let f1 = optimizer
        .cross_validate_pipeline(&pipeline, &x.view(), &y.view())
        .expect("F1 CV must succeed");

    assert!(
        f1 > 0.99,
        "macro F1 should be ~1.0 on separable data, got {f1}"
    );
}

/// Custom and MultiObjective metrics cannot be reduced to a scalar CV score;
/// they must return honest, distinct errors rather than fabricated numbers.
#[test]
fn test_custom_and_multiobjective_return_honest_errors() {
    let x = Array2::from_shape_vec((4, 1), vec![1.0_f64, 2.0, 3.0, 4.0]).expect("valid shape");
    let y = Array1::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0]);

    let mut pipeline = Pipeline::new();
    pipeline.set_estimator(Box::new(MockPredictor::new()));

    let custom = PipelineOptimizer::new()
        .cv_folds(2)
        .scoring(ScoringMetric::Custom {
            name: "made_up_metric".to_string(),
        });
    let custom_result = custom.cross_validate_pipeline(&pipeline, &x.view(), &y.view());
    assert!(
        matches!(custom_result, Err(SklearsError::NotImplemented(_))),
        "custom metric must return NotImplemented, got {custom_result:?}"
    );

    let multi = PipelineOptimizer::new()
        .cv_folds(2)
        .scoring(ScoringMetric::MultiObjective {
            metrics: vec![ScoringMetric::MeanSquaredError, ScoringMetric::Accuracy],
        });
    let multi_result = multi.cross_validate_pipeline(&pipeline, &x.view(), &y.view());
    assert!(
        matches!(multi_result, Err(SklearsError::InvalidInput(_))),
        "multi-objective metric must return InvalidInput, got {multi_result:?}"
    );
}

/// Fewer than two samples or two folds is invalid and must error rather than
/// silently degenerate.
#[test]
fn test_invalid_dimensions_return_invalid_input() {
    let mut pipeline = Pipeline::new();
    pipeline.set_estimator(Box::new(MockPredictor::new()));

    // n < 2.
    let x_single = Array2::from_shape_vec((1, 1), vec![1.0_f64]).expect("valid shape");
    let y_single = Array1::from_vec(vec![1.0_f64]);
    let too_few_samples = PipelineOptimizer::new()
        .cv_folds(3)
        .scoring(ScoringMetric::MeanSquaredError);
    let result_n =
        too_few_samples.cross_validate_pipeline(&pipeline, &x_single.view(), &y_single.view());
    assert!(
        matches!(result_n, Err(SklearsError::InvalidInput(_))),
        "n < 2 must return InvalidInput, got {result_n:?}"
    );

    // k < 2.
    let x_four = Array2::from_shape_vec((4, 1), vec![1.0_f64, 2.0, 3.0, 4.0]).expect("valid shape");
    let y_four = Array1::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0]);
    let too_few_folds = PipelineOptimizer::new()
        .cv_folds(1)
        .scoring(ScoringMetric::MeanSquaredError);
    let result_k = too_few_folds.cross_validate_pipeline(&pipeline, &x_four.view(), &y_four.view());
    assert!(
        matches!(result_k, Err(SklearsError::InvalidInput(_))),
        "k < 2 must return InvalidInput, got {result_k:?}"
    );
}
