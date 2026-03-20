//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result, SklearsError};

use super::types::{AnchorExplainer, AnchorExplanation, AnchorPredicate, ComparisonOperator, DecisionCondition, DecisionPath, DistanceMetric, LimeExplainer, LimeExplanation, PerturbationStrategy, PredicateOperator, PredicateValue, TreePathExtractor, TreeStructure};

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    #[test]
    fn test_decision_condition() {
        let condition = DecisionCondition {
            feature_idx: 0,
            feature_name: Some("age".to_string()),
            threshold: 30.0,
            operator: ComparisonOperator::LessThanOrEqual,
        };
        assert_eq!(condition.to_string(), "age <= 30");
    }
    #[test]
    fn test_decision_path_matching() {
        let conditions = vec![
            DecisionCondition { feature_idx : 0, feature_name : None, threshold : 0.5,
            operator : ComparisonOperator::LessThanOrEqual, }, DecisionCondition {
            feature_idx : 1, feature_name : None, threshold : 1.0, operator :
            ComparisonOperator::GreaterThan, },
        ];
        let path = DecisionPath::new(
            conditions,
            1.0,
            Some(0.9),
            Some(10),
            Some(0.1),
            None,
        );
        let sample1 = Array1::from_vec(vec![0.3, 1.5]);
        assert!(path.matches_sample(& sample1));
        let sample2 = Array1::from_vec(vec![0.7, 1.5]);
        assert!(! path.matches_sample(& sample2));
    }
    #[test]
    fn test_tree_path_extraction() {
        let extractor = TreePathExtractor::new(
            Some(vec!["feature_0".to_string()]),
            0.01,
            0.8,
            None,
        );
        let tree = TreeStructure::create_simple_tree();
        let sample = Array1::from_vec(vec![0.3]);
        let path = extractor.extract_sample_path(&sample, &tree).expect("sampling should succeed");
        assert_eq!(path.depth(), 1);
        assert_eq!(path.prediction, 0.0);
        assert!(path.conditions[0].operator == ComparisonOperator::LessThanOrEqual);
    }
    #[test]
    fn test_extract_all_paths() {
        let extractor = TreePathExtractor::new(None, 0.01, 0.8, None);
        let tree = TreeStructure::create_simple_tree();
        let paths = extractor.extract_all_paths(&tree).expect("operation should succeed");
        assert_eq!(paths.len(), 2);
        let predictions: Vec<f64> = paths.iter().map(|p| p.prediction).collect();
        assert!(predictions.contains(& 0.0));
        assert!(predictions.contains(& 1.0));
    }
    #[test]
    fn test_rule_extraction() {
        let extractor = TreePathExtractor::new(None, 0.1, 0.8, None);
        let tree = TreeStructure::create_simple_tree();
        let paths = extractor.extract_all_paths(&tree).expect("operation should succeed");
        let rules = extractor.extract_rules_from_paths(&paths, 100);
        assert!(! rules.is_empty());
        for rule in &rules {
            assert!(rule.support >= 0.1);
            assert!(rule.confidence >= 0.8);
        }
    }
    #[test]
    fn test_tree_summary() {
        let extractor = TreePathExtractor::new(
            Some(vec!["age".to_string()]),
            0.01,
            0.8,
            None,
        );
        let tree = TreeStructure::create_simple_tree();
        let paths = extractor.extract_all_paths(&tree).expect("operation should succeed");
        let summary = extractor.generate_tree_summary(&paths);
        assert!(summary.contains("Tree Summary"));
        assert!(summary.contains("Total paths: 2"));
        assert!(summary.contains("Feature Usage"));
    }
    #[test]
    fn test_lime_explainer_creation() {
        let explainer = LimeExplainer::default();
        assert_eq!(explainer.n_samples, 5000);
        assert_eq!(explainer.sigma, 0.25);
        let custom_explainer = LimeExplainer::new(
            1000,
            0.1,
            DistanceMetric::Manhattan,
            PerturbationStrategy::RandomSampling,
        );
        assert_eq!(custom_explainer.n_samples, 1000);
        assert_eq!(custom_explainer.sigma, 0.1);
    }
    #[test]
    fn test_lime_perturbation_generation() {
        let explainer = LimeExplainer::default().with_random_seed(42);
        let instance = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let perturbations = explainer.generate_perturbations(&instance).expect("operation should succeed");
        assert_eq!(perturbations.nrows(), 5000);
        assert_eq!(perturbations.ncols(), 3);
        for i in 0..perturbations.nrows() {
            for j in 0..perturbations.ncols() {
                let diff = (perturbations[(i, j)] - instance[j]).abs();
                assert!(diff < 3.0);
            }
        }
    }
    #[test]
    fn test_lime_distance_calculation() {
        let explainer = LimeExplainer::default();
        let instance = Array1::from_vec(vec![0.0, 0.0]);
        let samples = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 3.0, 4.0,])
            .expect("operation should succeed");
        let distances = explainer.calculate_distances(&instance, &samples).expect("sampling should succeed");
        assert_abs_diff_eq!(distances[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(distances[1], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(distances[2], 5.0, epsilon = 1e-10);
    }
    #[test]
    fn test_lime_explanation() {
        let explainer = LimeExplainer::new(
                100,
                0.1,
                DistanceMetric::Euclidean,
                PerturbationStrategy::Gaussian,
            )
            .with_random_seed(42);
        let instance = Array1::from_vec(vec![1.0, 2.0]);
        let predict_fn = |samples: &Array2<f64>| -> Result<Array1<f64>> {
            let predictions = samples
                .axis_iter(scirs2_core::ndarray::Axis(0))
                .map(|row| 2.0 * row[0] + 3.0 * row[1] + 1.0)
                .collect::<Array1<f64>>();
            Ok(predictions)
        };
        let explanation = explainer
            .explain_instance(
                &instance,
                predict_fn,
                Some(vec!["feature_1".to_string(), "feature_2".to_string()]),
            )
            .expect("operation should succeed");
        assert_eq!(explanation.instance, instance);
        assert_eq!(explanation.feature_importances.len(), 2);
        assert!(explanation.local_score >= 0.0);
        assert!(explanation.local_score <= 1.0);
        assert_eq!(explanation.n_samples_used, 100);
        let top_features = explanation.top_features(2);
        assert_eq!(top_features.len(), 2);
        assert!(
            top_features.iter().any(| (_, _, name) | name.as_ref() == Some(& "feature_1"
            .to_string()))
        );
        assert!(
            top_features.iter().any(| (_, _, name) | name.as_ref() == Some(& "feature_2"
            .to_string()))
        );
    }
    #[test]
    fn test_lime_explanation_display() {
        let instance = Array1::from_vec(vec![1.0, 2.0]);
        let feature_importances = Array1::from_vec(vec![0.5, - 0.3]);
        let explanation = LimeExplanation {
            instance,
            prediction: 0.8,
            feature_importances,
            feature_names: Some(vec!["height".to_string(), "weight".to_string()]),
            local_score: 0.95,
            n_samples_used: 1000,
        };
        let display = explanation.explain(Some(2));
        assert!(display.contains("LIME Explanation"));
        assert!(display.contains("0.8000"));
        assert!(display.contains("0.9500"));
        assert!(display.contains("1000"));
        assert!(display.contains("height"));
        assert!(display.contains("weight"));
        assert!(display.contains("supports"));
        assert!(display.contains("opposes"));
    }
    #[test]
    fn test_lime_top_features() {
        let instance = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let feature_importances = Array1::from_vec(vec![0.1, - 0.5, 0.3]);
        let explanation = LimeExplanation {
            instance,
            prediction: 0.8,
            feature_importances,
            feature_names: None,
            local_score: 0.9,
            n_samples_used: 1000,
        };
        let top_features = explanation.top_features(2);
        assert_eq!(top_features.len(), 2);
        assert_eq!(top_features[0].0, 1);
        assert_eq!(top_features[0].1, - 0.5);
        assert_eq!(top_features[1].0, 2);
        assert_eq!(top_features[1].1, 0.3);
    }
    #[test]
    fn test_anchor_explainer_creation() {
        let explainer = AnchorExplainer::default();
        assert_eq!(explainer.precision_threshold, 0.95);
        assert_eq!(explainer.max_anchor_size, 5);
        assert_eq!(explainer.coverage_samples, 10000);
        let custom_explainer = AnchorExplainer::new(0.9, 3, 1000, 5, 0.1);
        assert_eq!(custom_explainer.precision_threshold, 0.9);
        assert_eq!(custom_explainer.max_anchor_size, 3);
        assert_eq!(custom_explainer.coverage_samples, 1000);
    }
    #[test]
    fn test_anchor_predicate_generation() {
        let explainer = AnchorExplainer::default();
        let instance = Array1::from_vec(vec![1.0, 2.0]);
        let feature_names = Some(vec!["feature_1".to_string(), "feature_2".to_string()]);
        let predicates = explainer
            .generate_candidate_predicates(&instance, &feature_names)
            .expect("operation should succeed");
        assert!(predicates.len() >= 4);
        assert!(predicates.iter().any(| p | p.feature_idx == 0));
        assert!(predicates.iter().any(| p | p.feature_idx == 1));
        assert!(
            predicates.iter().any(| p | p.feature_name.as_ref() == Some(& "feature_1"
            .to_string()))
        );
        assert!(
            predicates.iter().any(| p | p.feature_name.as_ref() == Some(& "feature_2"
            .to_string()))
        );
    }
    #[test]
    fn test_anchor_predicate_to_string() {
        let feature_names = Some(vec!["age".to_string(), "income".to_string()]);
        let predicate1 = AnchorPredicate {
            feature_idx: 0,
            feature_name: Some("age".to_string()),
            operator: PredicateOperator::LessEqualThan,
            value: PredicateValue::Threshold(30.0),
        };
        assert_eq!(predicate1.to_string(& feature_names), "age <= 30.000");
        let predicate2 = AnchorPredicate {
            feature_idx: 1,
            feature_name: Some("income".to_string()),
            operator: PredicateOperator::InRange,
            value: PredicateValue::Range(40000.0, 60000.0),
        };
        assert_eq!(
            predicate2.to_string(& feature_names), "40000.000 <= income <= 60000.000"
        );
    }
    #[test]
    fn test_anchor_sample_satisfaction() {
        let explainer = AnchorExplainer::default();
        let predicates = vec![
            AnchorPredicate { feature_idx : 0, feature_name : None, operator :
            PredicateOperator::LessEqualThan, value : PredicateValue::Threshold(5.0), },
            AnchorPredicate { feature_idx : 1, feature_name : None, operator :
            PredicateOperator::GreaterThan, value : PredicateValue::Threshold(2.0), },
        ];
        let sample1 = Array1::from_vec(vec![3.0, 4.0]);
        assert!(explainer.sample_satisfies_anchor(& sample1, & predicates));
        let sample2 = Array1::from_vec(vec![7.0, 4.0]);
        assert!(! explainer.sample_satisfies_anchor(& sample2, & predicates));
        let sample3 = Array1::from_vec(vec![3.0, 1.0]);
        assert!(! explainer.sample_satisfies_anchor(& sample3, & predicates));
    }
    #[test]
    fn test_anchor_explanation() {
        let explainer = AnchorExplainer::new(0.8, 2, 100, 5, 0.01).with_random_seed(42);
        let instance = Array1::from_vec(vec![1.0, 2.0]);
        let predict_fn = |samples: &Array2<f64>| -> Result<Array1<f64>> {
            let predictions = samples
                .axis_iter(scirs2_core::ndarray::Axis(0))
                .map(|row| {
                    if row.len() >= 2 && row[0] <= 2.0 && row[1] > 1.0 {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect::<Array1<f64>>();
            Ok(predictions)
        };
        let explanation = explainer
            .explain_instance(
                &instance,
                predict_fn,
                Some(vec!["feature_1".to_string(), "feature_2".to_string()]),
            )
            .expect("operation should succeed");
        assert_eq!(explanation.instance, instance);
        assert_eq!(explanation.prediction, 1.0);
        assert!(explanation.precision >= 0.0 && explanation.precision <= 1.0);
        assert!(explanation.coverage >= 0.0 && explanation.coverage <= 1.0);
        assert_eq!(explanation.n_samples_evaluated, 100);
    }
    #[test]
    fn test_anchor_explanation_display() {
        let instance = Array1::from_vec(vec![1.0, 2.0]);
        let anchor = vec![
            AnchorPredicate { feature_idx : 0, feature_name : Some("age".to_string()),
            operator : PredicateOperator::LessEqualThan, value :
            PredicateValue::Threshold(30.0), }, AnchorPredicate { feature_idx : 1,
            feature_name : Some("income".to_string()), operator :
            PredicateOperator::GreaterThan, value : PredicateValue::Threshold(50000.0),
            },
        ];
        let explanation = AnchorExplanation {
            instance,
            prediction: 1.0,
            anchor,
            precision: 0.95,
            coverage: 0.3,
            n_samples_evaluated: 1000,
            feature_names: Some(vec!["age".to_string(), "income".to_string()]),
        };
        let display = explanation.explain();
        assert!(display.contains("Anchor Explanation"));
        assert!(display.contains("1.0000"));
        assert!(display.contains("0.9500"));
        assert!(display.contains("0.3000"));
        assert!(display.contains("1000"));
        assert!(display.contains("IF"));
        assert!(display.contains("AND"));
        assert!(display.contains("THEN"));
        assert!(display.contains("age"));
        assert!(display.contains("income"));
        assert!(display.contains("95.0%"));
    }
    #[test]
    fn test_anchor_explanation_empty() {
        let instance = Array1::from_vec(vec![1.0, 2.0]);
        let explanation = AnchorExplanation {
            instance,
            prediction: 0.5,
            anchor: vec![],
            precision: 0.0,
            coverage: 1.0,
            n_samples_evaluated: 100,
            feature_names: None,
        };
        let display = explanation.explain();
        assert!(display.contains("No reliable anchor found"));
        assert_eq!(explanation.anchor_size(), 0);
        assert!(! explanation.is_sufficient(0.9));
    }
    #[test]
    fn test_anchor_explanation_utilities() {
        let instance = Array1::from_vec(vec![1.0, 2.0]);
        let anchor = vec![
            AnchorPredicate { feature_idx : 0, feature_name : None, operator :
            PredicateOperator::LessEqualThan, value : PredicateValue::Threshold(5.0), }
        ];
        let explanation = AnchorExplanation {
            instance,
            prediction: 1.0,
            anchor,
            precision: 0.9,
            coverage: 0.3,
            n_samples_evaluated: 1000,
            feature_names: None,
        };
        assert_eq!(explanation.anchor_size(), 1);
        assert!(explanation.is_sufficient(0.8));
        assert!(! explanation.is_sufficient(0.95));
    }
}
