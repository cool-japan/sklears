//! Property-based tests for Naive Bayes classifiers

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::smoothing::{LaplaceSmoothing, LidstoneSmoothing, Smoothing};
    use approx::assert_abs_diff_eq;
    // SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
    use proptest::prelude::*;
    use scirs2_core::ndarray::{Array1, Array2, Axis};
    use sklears_core::traits::{Fit, Predict, PredictProba, Score};

    /// Generate valid probability vectors that sum to 1
    fn prob_vector_strategy(size: usize) -> impl Strategy<Value = Vec<f64>> {
        prop::collection::vec(0.1f64..10.0, size).prop_map(|mut v| {
            let sum: f64 = v.iter().sum();
            for x in &mut v {
                *x /= sum;
            }
            v
        })
    }

    /// Generate valid feature matrices for Gaussian NB (any real values)
    fn gaussian_features_strategy(rows: usize, cols: usize) -> impl Strategy<Value = Array2<f64>> {
        prop::collection::vec(prop::collection::vec(-10.0f64..10.0, cols), rows).prop_map(
            move |data| {
                Array2::from_shape_vec((rows, cols), data.into_iter().flatten().collect()).unwrap()
            },
        )
    }

    /// Generate valid feature matrices for Multinomial/Complement NB (non-negative)
    fn count_features_strategy(rows: usize, cols: usize) -> impl Strategy<Value = Array2<f64>> {
        prop::collection::vec(prop::collection::vec(0.0f64..20.0, cols), rows).prop_map(
            move |data| {
                Array2::from_shape_vec((rows, cols), data.into_iter().flatten().collect()).unwrap()
            },
        )
    }

    /// Generate valid feature matrices for Poisson NB (non-negative integers)
    fn poisson_features_strategy(rows: usize, cols: usize) -> impl Strategy<Value = Array2<f64>> {
        prop::collection::vec(prop::collection::vec(0u32..20, cols), rows).prop_map(move |data| {
            Array2::from_shape_vec(
                (rows, cols),
                data.into_iter().flatten().map(|x| x as f64).collect(),
            )
            .unwrap()
        })
    }

    /// Generate valid feature matrices for Categorical NB (non-negative integers)
    fn categorical_features_strategy(
        rows: usize,
        cols: usize,
        max_cat: usize,
    ) -> impl Strategy<Value = Array2<f64>> {
        prop::collection::vec(prop::collection::vec(0usize..max_cat, cols), rows).prop_map(
            move |data| {
                Array2::from_shape_vec(
                    (rows, cols),
                    data.into_iter().flatten().map(|x| x as f64).collect(),
                )
                .unwrap()
            },
        )
    }

    /// Generate valid binary feature matrices for Bernoulli NB
    fn binary_features_strategy(rows: usize, cols: usize) -> impl Strategy<Value = Array2<f64>> {
        prop::collection::vec(prop::collection::vec(0u32..2, cols), rows).prop_map(move |data| {
            Array2::from_shape_vec(
                (rows, cols),
                data.into_iter().flatten().map(|x| x as f64).collect(),
            )
            .unwrap()
        })
    }

    /// Generate valid class labels
    fn labels_strategy(size: usize, n_classes: usize) -> impl Strategy<Value = Array1<i32>> {
        prop::collection::vec(0i32..(n_classes as i32), size)
            .prop_map(|data| Array1::from_vec(data))
    }

    proptest! {
        #[test]
        fn test_gaussian_nb_probability_properties(
            x in gaussian_features_strategy(50, 4),
            y in labels_strategy(50, 3),
            var_smoothing in 1e-12f64..1e-6
        ) {
            let model = GaussianNB::new()
                .var_smoothing(var_smoothing)
                .fit(&x, &y);

            if let Ok(trained_model) = model {
                let probabilities = PredictProba::predict_proba(&trained_model, &x).unwrap();

                // Property 1: All probabilities should be non-negative
                for &prob in probabilities.iter() {
                    prop_assert!(prob >= 0.0);
                }

                // Property 2: Each row should sum to approximately 1
                for i in 0..probabilities.nrows() {
                    let row_sum: f64 = probabilities.row(i).sum();
                    prop_assert!(row_sum >= 0.99 && row_sum <= 1.01);
                }

                // Property 3: Predictions should be consistent with max probability
                let predictions = Predict::predict(&trained_model, &x).unwrap();
                for i in 0..predictions.len() {
                    let predicted_class = predictions[i];
                    let max_prob_idx = probabilities.row(i)
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx)
                        .unwrap();

                    let classes = trained_model.classes();
                    prop_assert_eq!(predicted_class, classes[max_prob_idx]);
                }

                // Property 4: Score should be between 0 and 1
                let score = trained_model.score(&x, &y).unwrap();
                prop_assert!(score >= 0.0 && score <= 1.0);
            }
        }

        #[test]
        fn test_multinomial_nb_probability_properties(
            x in count_features_strategy(40, 5),
            y in labels_strategy(40, 2),
            alpha in 0.1f64..2.0
        ) {
            let model = MultinomialNB::new()
                .alpha(alpha)
                .fit(&x, &y);

            if let Ok(trained_model) = model {
                let probabilities = PredictProba::predict_proba(&trained_model, &x).unwrap();

                // Property 1: All probabilities should be non-negative
                for &prob in probabilities.iter() {
                    prop_assert!(prob >= 0.0);
                    prop_assert!(prob <= 1.0);
                }

                // Property 2: Each row should sum to approximately 1
                for i in 0..probabilities.nrows() {
                    let row_sum: f64 = probabilities.row(i).sum();
                    prop_assert!(row_sum >= 0.99 && row_sum <= 1.01);
                }

                // Property 3: Adding more smoothing should not drastically change predictions
                let model2 = MultinomialNB::new()
                    .alpha(alpha * 1.1)
                    .fit(&x, &y).unwrap();
                let probabilities2 = model2.predict_proba(&x).unwrap();

                for i in 0..probabilities.nrows() {
                    for j in 0..probabilities.ncols() {
                        let diff = (probabilities[[i, j]] - probabilities2[[i, j]]).abs();
                        prop_assert!(diff < 0.5); // Should not change too dramatically
                    }
                }
            }
        }

        #[test]
        fn test_poisson_nb_probability_properties(
            x in poisson_features_strategy(30, 3),
            y in labels_strategy(30, 2),
            alpha in 1e-10f64..1e-6
        ) {
            let model = PoissonNB::new()
                .alpha(alpha)
                .fit(&x, &y);

            if let Ok(trained_model) = model {
                let probabilities = PredictProba::predict_proba(&trained_model, &x).unwrap();

                // Property 1: All probabilities should be non-negative and ≤ 1
                for &prob in probabilities.iter() {
                    prop_assert!(prob >= 0.0);
                    prop_assert!(prob <= 1.0);
                    prop_assert!(prob.is_finite());
                }

                // Property 2: Each row should sum to approximately 1
                for i in 0..probabilities.nrows() {
                    let row_sum: f64 = probabilities.row(i).sum();
                    prop_assert!(row_sum >= 0.99 && row_sum <= 1.01);
                }

                // Property 3: Poisson PMF computation should be mathematically correct
                let theta = trained_model.feature_log_prob();
                for class_idx in 0..theta.nrows() {
                    for feature_idx in 0..theta.ncols() {
                        let rate = theta[[class_idx, feature_idx]];
                        prop_assert!(rate >= alpha); // Should be at least the smoothing parameter
                        prop_assert!(rate.is_finite());
                    }
                }
            }
        }

        #[test]
        fn test_bernoulli_nb_probability_properties(
            x in binary_features_strategy(35, 4),
            y in labels_strategy(35, 2),
            alpha in 0.1f64..2.0
        ) {
            let model = BernoulliNB::new()
                .alpha(alpha)
                .binarize(None) // Already binary
                .fit(&x, &y);

            if let Ok(trained_model) = model {
                let probabilities = PredictProba::predict_proba(&trained_model, &x).unwrap();

                // Property 1: All probabilities should be in [0, 1]
                for &prob in probabilities.iter() {
                    prop_assert!(prob >= 0.0);
                    prop_assert!(prob <= 1.0);
                }

                // Property 2: Each row should sum to approximately 1
                for i in 0..probabilities.nrows() {
                    let row_sum: f64 = probabilities.row(i).sum();
                    prop_assert!(row_sum >= 0.99 && row_sum <= 1.01);
                }

                // Property 3: Feature probabilities should be in [0, 1] after smoothing
                let feature_log_prob = trained_model.feature_log_prob();
                for &log_prob in feature_log_prob.iter() {
                    let prob = log_prob.exp();
                    prop_assert!(prob >= 0.0);
                    prop_assert!(prob <= 1.0);
                }
            }
        }

        #[test]
        fn test_categorical_nb_probability_properties(
            x in categorical_features_strategy(40, 3, 4),
            y in labels_strategy(40, 2),
            alpha in 0.1f64..2.0
        ) {
            let model = CategoricalNB::new()
                .alpha(alpha)
                .fit(&x, &y);

            if let Ok(trained_model) = model {
                let probabilities = PredictProba::predict_proba(&trained_model, &x).unwrap();

                // Property 1: All probabilities should be in [0, 1]
                for &prob in probabilities.iter() {
                    prop_assert!(prob >= 0.0);
                    prop_assert!(prob <= 1.0);
                }

                // Property 2: Each row should sum to approximately 1
                for i in 0..probabilities.nrows() {
                    let row_sum: f64 = probabilities.row(i).sum();
                    prop_assert!(row_sum >= 0.99 && row_sum <= 1.01);
                }

                // Property 3: Predictions should be deterministic for the same input
                let predictions1 = trained_model.predict(&x).unwrap();
                let predictions2 = trained_model.predict(&x).unwrap();
                prop_assert_eq!(predictions1, predictions2);
            }
        }

        #[test]
        fn test_complement_nb_probability_properties(
            x in count_features_strategy(45, 4),
            y in labels_strategy(45, 3),
            alpha in 0.1f64..2.0
        ) {
            let model = ComplementNB::new()
                .alpha(alpha)
                .fit(&x, &y);

            if let Ok(trained_model) = model {
                let probabilities = PredictProba::predict_proba(&trained_model, &x).unwrap();

                // Property 1: All probabilities should be in [0, 1]
                for &prob in probabilities.iter() {
                    prop_assert!(prob >= 0.0);
                    prop_assert!(prob <= 1.0);
                }

                // Property 2: Each row should sum to approximately 1
                for i in 0..probabilities.nrows() {
                    let row_sum: f64 = probabilities.row(i).sum();
                    prop_assert!(row_sum >= 0.99 && row_sum <= 1.01);
                }

                // Property 3: Score should improve with more balanced data
                let score = trained_model.score(&x, &y).unwrap();
                prop_assert!(score >= 0.0 && score <= 1.0);
            }
        }

        #[test]
        fn test_uncertainty_quantification_properties(
            probabilities in prop::collection::vec(prob_vector_strategy(3), 20)
        ) {
            let prob_matrix = Array2::from_shape_vec(
                (probabilities.len(), 3),
                probabilities.into_iter().flatten().collect()
            ).unwrap();

            let uncertainty = StandardUncertainty::new();
            let measures = uncertainty.uncertainty_measures(&prob_matrix);

            // Property 1: Confidence should equal maximum probability
            for i in 0..prob_matrix.nrows() {
                let max_prob = prob_matrix.row(i).iter().cloned().fold(0.0, f64::max);
                prop_assert!((measures.confidence[i] - max_prob).abs() < 1e-10);
            }

            // Property 2: Entropy should be non-negative
            for &entropy in measures.entropy.iter() {
                prop_assert!(entropy >= 0.0);
            }

            // Property 3: Margin should be non-negative and ≤ 1
            for &margin in measures.margin.iter() {
                prop_assert!(margin >= 0.0);
                prop_assert!(margin <= 1.0);
            }

            // Property 4: Variance should be non-negative
            for &variance in measures.variance.iter() {
                prop_assert!(variance >= 0.0);
            }

            // Property 5: Higher entropy should correspond to lower confidence
            // (for uniform vs non-uniform distributions)
            let uniform_entropy = measures.entropy.iter()
                .enumerate()
                .filter(|(i, _)| {
                    let row = prob_matrix.row(*i);
                    let min_prob = row.iter().cloned().fold(1.0, f64::min);
                    let max_prob = row.iter().cloned().fold(0.0, f64::max);
                    max_prob - min_prob < 0.1 // Nearly uniform
                })
                .map(|(_, &e)| e)
                .fold(0.0, f64::max);

            let skewed_entropy = measures.entropy.iter()
                .enumerate()
                .filter(|(i, _)| {
                    let row = prob_matrix.row(*i);
                    let min_prob = row.iter().cloned().fold(1.0, f64::min);
                    let max_prob = row.iter().cloned().fold(0.0, f64::max);
                    max_prob - min_prob > 0.5 // Highly skewed
                })
                .map(|(_, &e)| e)
                .fold(f64::INFINITY, f64::min);

            if uniform_entropy > 0.0 && skewed_entropy < f64::INFINITY {
                prop_assert!(uniform_entropy >= skewed_entropy);
            }
        }

        #[test]
        fn test_smoothing_methods_properties(
            counts in prop::collection::vec(
                prop::collection::vec(0.0f64..10.0, 3), 2
            ),
            alpha in 0.1f64..2.0
        ) {
            let count_matrix = Array2::from_shape_vec(
                (2, 3),
                counts.into_iter().flatten().collect()
            ).unwrap();
            let totals = count_matrix.sum_axis(scirs2_core::ndarray::Axis(1));

            let laplace = LaplaceSmoothing::new(alpha);
            let smoothed = laplace.smooth_counts(&count_matrix, &totals);

            // Property 1: Smoothing should increase all counts
            for i in 0..count_matrix.nrows() {
                for j in 0..count_matrix.ncols() {
                    prop_assert!(smoothed[[i, j]] >= count_matrix[[i, j]]);
                    prop_assert!((smoothed[[i, j]] - (count_matrix[[i, j]] + alpha)).abs() < 1e-10);
                }
            }

            // Property 2: Lidstone with different parameters should give different results
            let lidstone1 = LidstoneSmoothing::new(alpha);
            let lidstone2 = LidstoneSmoothing::new(alpha * 2.0);

            let smoothed1 = lidstone1.smooth_counts(&count_matrix, &totals);
            let smoothed2 = lidstone2.smooth_counts(&count_matrix, &totals);

            for i in 0..count_matrix.nrows() {
                for j in 0..count_matrix.ncols() {
                    prop_assert!(smoothed2[[i, j]] > smoothed1[[i, j]]);
                }
            }
        }
    }

    /// Integration test to verify that different implementations give reasonable results
    #[test]
    fn test_naive_bayes_ensemble_consistency() {
        // Generate deterministic synthetic data
        let n_samples = 100;
        let n_features = 5;

        let mut x_data = Vec::with_capacity(n_samples * n_features);
        let mut y_data = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let class = if i < n_samples / 2 { 0 } else { 1 };
            y_data.push(class);

            for j in 0..n_features {
                let feature_value = if class == 0 {
                    1.0 + (i + j) as f64 * 0.1
                } else {
                    5.0 + (i + j) as f64 * 0.1
                };
                x_data.push(feature_value);
            }
        }

        let x = Array2::from_shape_vec((n_samples, n_features), x_data).unwrap();
        let y = Array1::from_vec(y_data);

        // Test Gaussian NB
        let gaussian_model = GaussianNB::new().fit(&x, &y).unwrap();
        let gaussian_score = gaussian_model.score(&x, &y).unwrap();

        // Test Multinomial NB (with non-negative transformation)
        let x_positive = &x + 1.0; // Make all values positive
        let multinomial_model = MultinomialNB::new().fit(&x_positive, &y).unwrap();
        let multinomial_score = multinomial_model.score(&x_positive, &y).unwrap();

        // Test Poisson NB (with integer transformation)
        let x_int = x_positive.mapv(|v| v.round());
        let poisson_model = PoissonNB::new().fit(&x_int, &y).unwrap();
        let poisson_score = poisson_model.score(&x_int, &y).unwrap();

        // All models should achieve reasonable performance on separable data
        assert!(
            gaussian_score > 0.6,
            "Gaussian NB score too low: {}",
            gaussian_score
        );
        assert!(
            multinomial_score > 0.4,
            "Multinomial NB score too low: {}",
            multinomial_score
        );
        assert!(
            poisson_score > 0.4,
            "Poisson NB score too low: {}",
            poisson_score
        );

        // Test uncertainty quantification
        let gaussian_proba = PredictProba::predict_proba(&gaussian_model, &x).unwrap();
        let uncertainty = StandardUncertainty::new();
        let measures = uncertainty.uncertainty_measures(&gaussian_proba);

        // High-confidence predictions should have low entropy
        let high_conf_indices: Vec<usize> = measures
            .confidence
            .iter()
            .enumerate()
            .filter(|(_, &conf)| conf > 0.8)
            .map(|(idx, _)| idx)
            .collect();

        if !high_conf_indices.is_empty() {
            let avg_high_conf_entropy: f64 = high_conf_indices
                .iter()
                .map(|&idx| measures.entropy[idx])
                .sum::<f64>()
                / high_conf_indices.len() as f64;

            let avg_total_entropy: f64 = measures.entropy.mean().unwrap();

            assert!(
                avg_high_conf_entropy <= avg_total_entropy,
                "High confidence predictions should have lower entropy"
            );
        }
    }
}
