use super::*;
use scirs2_core::ndarray::array;

#[test]
fn test_cnn_activation_apply() {
    let relu = CNNActivation::ReLU;
    assert_eq!(relu.apply(5.0), 5.0);
    assert_eq!(relu.apply(-3.0), 0.0);

    let tanh = CNNActivation::Tanh;
    assert!((tanh.apply(0.0) - 0.0).abs() < 1e-10);
    assert!(tanh.apply(1.0) > 0.0);

    let sigmoid = CNNActivation::Sigmoid;
    assert!((sigmoid.apply(0.0) - 0.5).abs() < 1e-10);
    assert!(sigmoid.apply(100.0) > 0.9);

    let leaky_relu = CNNActivation::LeakyReLU(0.1);
    assert_eq!(leaky_relu.apply(5.0), 5.0);
    assert_eq!(leaky_relu.apply(-10.0), -1.0);
}

#[test]
fn test_cnn_activation_derivative() {
    let relu = CNNActivation::ReLU;
    assert_eq!(relu.derivative(5.0), 1.0);
    assert_eq!(relu.derivative(-3.0), 0.0);

    let sigmoid = CNNActivation::Sigmoid;
    let sigmoid_deriv = sigmoid.derivative(0.0);
    assert!((sigmoid_deriv - 0.25).abs() < 1e-10);
}

#[test]
fn test_softmax() {
    let x = array![1.0, 2.0, 3.0];
    let result = softmax(&x);

    let sum: f64 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10);

    assert!(result[2] > result[1]);
    assert!(result[1] > result[0]);
}

#[test]
fn test_layer_norm() {
    let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let normalized = layer_norm(&x);

    for i in 0..x.nrows() {
        let row_mean = normalized.row(i).mean().unwrap();
        assert!(row_mean.abs() < 1e-10);
    }
}

#[test]
fn test_autoencoder_feature_extractor_creation() {
    let extractor = AutoencoderFeatureExtractor::new()
        .encoding_dim(16)
        .learning_rate(0.001)
        .n_epochs(10);

    assert_eq!(extractor.encoding_dim, 16);
    assert_eq!(extractor.learning_rate, 0.001);
    assert_eq!(extractor.n_epochs, 10);
}

#[test]
fn test_autoencoder_fit_and_transform() {
    let extractor = AutoencoderFeatureExtractor::new()
        .encoding_dim(2)
        .n_epochs(1);

    let x = Array2::from_shape_vec((3, 4), vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
    ]).unwrap();

    let fitted = extractor.fit(&x, &()).unwrap();
    let transformed = fitted.transform(&x).unwrap();

    assert_eq!(transformed.dim(), (3, 2));
}

#[test]
fn test_neural_embedding_extractor_creation() {
    let extractor = NeuralEmbeddingExtractor::new()
        .embedding_dim(64)
        .vocab_size(1000)
        .learning_rate(0.01);

    assert_eq!(extractor.embedding_dim, 64);
    assert_eq!(extractor.vocab_size, 1000);
    assert_eq!(extractor.learning_rate, 0.01);
}

#[test]
fn test_neural_embedding_fit_and_transform() {
    let extractor = NeuralEmbeddingExtractor::new()
        .embedding_dim(4)
        .vocab_size(10)
        .n_epochs(1);

    let x = array![0, 1, 2, 3, 4];

    let fitted = extractor.fit(&x, &()).unwrap();
    let transformed = fitted.transform(&x).unwrap();

    assert_eq!(transformed.dim(), (5, 4));
}

#[test]
fn test_cnn_feature_extractor_creation() {
    let extractor = CNNFeatureExtractor::new()
        .filters(vec![16, 32])
        .kernel_sizes(vec![3, 3])
        .activation(CNNActivation::ReLU);

    assert_eq!(extractor.filters, vec![16, 32]);
    assert_eq!(extractor.kernel_sizes, vec![3, 3]);
}

#[test]
fn test_cnn_feature_extractor_fit() {
    let extractor = CNNFeatureExtractor::new()
        .filters(vec![8])
        .kernel_sizes(vec![3])
        .n_epochs(1);

    let x = Array3::zeros((10, 10, 3));

    let result = extractor.fit(&x, &());
    assert!(result.is_ok());
}

#[test]
fn test_attention_feature_extractor_creation() {
    let extractor = AttentionFeatureExtractor::new()
        .d_model(64)
        .n_heads(4)
        .attention_type(AttentionType::MultiHead);

    assert_eq!(extractor.d_model, 64);
    assert_eq!(extractor.n_heads, 4);
}

#[test]
fn test_attention_feature_extractor_fit() {
    let extractor = AttentionFeatureExtractor::new()
        .d_model(8)
        .n_heads(2)
        .n_epochs(1);

    let x = Array2::zeros((5, 8));

    let result = extractor.fit(&x, &());
    assert!(result.is_ok());
}

#[test]
fn test_transformer_feature_extractor_creation() {
    let extractor = TransformerFeatureExtractor::new()
        .d_model(128)
        .n_heads(8)
        .n_layers(2)
        .d_ff(256);

    assert_eq!(extractor.d_model, 128);
    assert_eq!(extractor.n_heads, 8);
    assert_eq!(extractor.n_layers, 2);
    assert_eq!(extractor.d_ff, 256);
}

#[test]
fn test_transformer_feature_extractor_fit() {
    let extractor = TransformerFeatureExtractor::new()
        .d_model(8)
        .n_heads(2)
        .n_layers(1)
        .d_ff(16)
        .n_epochs(1);

    let x = Array2::zeros((5, 8));

    let result = extractor.fit(&x, &());
    assert!(result.is_ok());
}

#[test]
fn test_dense_layer_creation() {
    let layer = DenseLayer::new(10, 5, Some(CNNActivation::ReLU));

    assert_eq!(layer.input_dim, 10);
    assert_eq!(layer.output_dim, 5);
    assert!(layer.activation.is_some());
}

#[test]
fn test_dense_layer_forward() {
    let layer = DenseLayer::new(3, 2, None);
    let input = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let output = layer.forward(&input);
    assert_eq!(output.dim(), (2, 2));
}

#[test]
fn test_conv2d_layer_creation() {
    let layer = Conv2DLayer::new(3, 16, 3, 1, 1, Some(CNNActivation::ReLU));

    assert_eq!(layer.in_channels, 3);
    assert_eq!(layer.out_channels, 16);
    assert_eq!(layer.kernel_size, 3);
}

#[test]
fn test_max_pool2d_layer() {
    let layer = MaxPool2DLayer::new(2, 2);
    let input = Array3::ones((4, 4, 2));

    let output = layer.forward(&input);
    assert_eq!(output.dim(), (2, 2, 2));
}

#[test]
fn test_avg_pool2d_layer() {
    let layer = AvgPool2DLayer::new(2, 2);
    let input = Array3::ones((4, 4, 2));

    let output = layer.forward(&input);
    assert_eq!(output.dim(), (2, 2, 2));
    assert!((output[(0, 0, 0)] - 1.0).abs() < 1e-10);
}

#[test]
fn test_dropout_layer() {
    let mut layer = DropoutLayer::new(0.5);
    layer.set_training(false);

    let input = Array2::ones((3, 4));
    let output = layer.forward_2d(&input);

    assert_eq!(output.dim(), input.dim());
    assert!((output[(0, 0)] - 1.0).abs() < 1e-10);
}

#[test]
fn test_batch_norm_layer() {
    let mut layer = BatchNormLayer::new(3, 1e-5, 0.1, true);
    let input = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let output = layer.forward_2d(&input);
    assert_eq!(output.dim(), (2, 3));
}

#[test]
fn test_xavier_normal_init() {
    let weights = xavier_normal_init(10, 5, Some(42));
    assert_eq!(weights.dim(), (10, 5));

    let mean = weights.iter().sum::<f64>() / weights.len() as f64;
    assert!(mean.abs() < 0.1);
}

#[test]
fn test_he_normal_init() {
    let weights = he_normal_init(10, 5, Some(42));
    assert_eq!(weights.dim(), (10, 5));
}

#[test]
fn test_cosine_similarity() {
    let a = array![1.0, 2.0, 3.0];
    let b = array![1.0, 2.0, 3.0];
    let similarity = cosine_similarity(&a, &b);

    assert!((similarity - 1.0).abs() < 1e-10);

    let c = array![-1.0, -2.0, -3.0];
    let similarity2 = cosine_similarity(&a, &c);
    assert!((similarity2 + 1.0).abs() < 1e-10);
}

#[test]
fn test_euclidean_distance() {
    let a = array![0.0, 0.0];
    let b = array![3.0, 4.0];
    let distance = euclidean_distance(&a, &b);

    assert!((distance - 5.0).abs() < 1e-10);
}

#[test]
fn test_manhattan_distance() {
    let a = array![0.0, 0.0];
    let b = array![3.0, 4.0];
    let distance = manhattan_distance(&a, &b);

    assert!((distance - 7.0).abs() < 1e-10);
}

#[test]
fn test_l2_regularization_loss() {
    let weights = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let loss = l2_regularization_loss(&weights, 0.01);

    let expected = 0.01 * (1.0 + 4.0 + 9.0 + 16.0);
    assert!((loss - expected).abs() < 1e-10);
}

#[test]
fn test_standard_scaler() {
    let data = Array2::from_shape_vec((3, 2), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
    let (means, stds) = standard_scaler_fit(&data);

    assert!((means[0] - 2.0).abs() < 1e-10);
    assert!((means[1] - 5.0).abs() < 1e-10);

    let scaled = standard_scaler_transform(&data, &means, &stds);
    let scaled_mean_0 = scaled.column(0).mean().unwrap();
    assert!(scaled_mean_0.abs() < 1e-10);
}

#[test]
fn test_min_max_scaler() {
    let data = Array2::from_shape_vec((3, 2), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
    let (mins, maxs) = min_max_scaler_fit(&data);

    assert!((mins[0] - 1.0).abs() < 1e-10);
    assert!((maxs[0] - 3.0).abs() < 1e-10);

    let scaled = min_max_scaler_transform(&data, &mins, &maxs);
    assert!((scaled[(0, 0)] - 0.0).abs() < 1e-10);
    assert!((scaled[(2, 0)] - 1.0).abs() < 1e-10);
}

#[test]
fn test_create_batches() {
    let data = Array2::zeros((10, 5));
    let batches = create_batches(&data, 3);

    assert_eq!(batches.len(), 4);
    assert_eq!(batches[0].dim(), (3, 5));
    assert_eq!(batches[1].dim(), (3, 5));
    assert_eq!(batches[2].dim(), (3, 5));
    assert_eq!(batches[3].dim(), (1, 5));
}

#[test]
fn test_train_test_split() {
    let data = Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();
    let (train, test) = train_test_split(&data, 0.3, Some(42));

    assert_eq!(train.nrows(), 7);
    assert_eq!(test.nrows(), 3);
    assert_eq!(train.ncols(), 2);
    assert_eq!(test.ncols(), 2);
}

#[test]
fn test_sinusoidal_positional_encoding() {
    let encoding = sinusoidal_positional_encoding(10, 8);
    assert_eq!(encoding.dim(), (10, 8));

    assert!((encoding[(0, 0)] - 0.0).abs() < 1e-10);
    assert!((encoding[(0, 1)] - 1.0).abs() < 1e-10);
}

#[test]
fn test_rotary_positional_encoding() {
    let encoding = rotary_positional_encoding(5, 4);
    assert_eq!(encoding.dim(), (5, 4));
}

#[test]
fn test_positional_embedding() {
    let embedding = PositionalEmbedding::new(
        PositionalEncodingType::Sinusoidal,
        10,
        8,
        Some(42),
    );

    let encoding = embedding.get_encoding(5);
    assert_eq!(encoding.dim(), (5, 8));

    let x = Array2::zeros((5, 8));
    let applied = embedding.apply_encoding(&x);
    assert_eq!(applied.dim(), (5, 8));
}

#[test]
fn test_sgd_optimizer() {
    let mut optimizer = SGD::new(0.01).with_momentum(0.9);
    let mut params = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let gradients = Array2::ones((2, 2));

    let initial_param = params[(0, 0)];
    optimizer.step(&mut params, &gradients);

    assert!(params[(0, 0)] < initial_param);
}

#[test]
fn test_adam_optimizer() {
    let mut optimizer = Adam::new(0.01);
    let mut params = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let gradients = Array2::ones((2, 2));

    let initial_param = params[(0, 0)];
    optimizer.step(&mut params, &gradients);

    assert!(params[(0, 0)] < initial_param);
}

#[test]
fn test_learning_rate_scheduler() {
    let mut scheduler = LearningRateScheduler::new(0.1, SchedulerType::StepLR)
        .with_step_size(10)
        .with_gamma(0.1);

    let mut optimizer = SGD::new(0.1);

    assert_eq!(scheduler.get_lr(), 0.1);

    for _ in 0..10 {
        scheduler.step(&mut optimizer);
    }

    assert!((scheduler.get_lr() - 0.01).abs() < 1e-10);
}

#[test]
fn test_create_optimizer() {
    let optimizer = create_optimizer("adam", 0.01);
    assert!((optimizer.get_learning_rate() - 0.01).abs() < 1e-10);

    let optimizer2 = create_optimizer("sgd", 0.1);
    assert!((optimizer2.get_learning_rate() - 0.1).abs() < 1e-10);
}

#[test]
fn test_scaled_dot_product_attention() {
    let attention = ScaledDotProductAttention::new(8, 1.0, 0.0);
    let q = Array2::ones((4, 8));
    let k = Array2::ones((4, 8));
    let v = Array2::ones((4, 8));

    let (output, weights) = attention.forward(&q, &k, &v, None);
    assert_eq!(output.dim(), (4, 8));
    assert_eq!(weights.dim(), (4, 4));

    for i in 0..4 {
        let row_sum: f64 = weights.row(i).iter().sum();
        assert!((row_sum - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_multi_head_attention() {
    let attention = MultiHeadAttention::new(16, 4, 0.0, Some(42));
    let q = Array2::ones((5, 16));
    let k = Array2::ones((5, 16));
    let v = Array2::ones((5, 16));

    let (output, weights) = attention.forward(&q, &k, &v, None);
    assert_eq!(output.dim(), (5, 16));
    assert_eq!(weights.len(), 4);
}

#[test]
fn test_feed_forward_network() {
    let mut ffn = FeedForwardNetwork::new(8, 32, CNNActivation::ReLU, 0.0, Some(42));
    let input = Array2::ones((4, 8));

    let output = ffn.forward(&input);
    assert_eq!(output.dim(), (4, 8));
}

#[test]
fn test_layer_norm_layer() {
    let layer_norm = LayerNorm::new(8);
    let input = Array2::from_shape_vec((3, 8), vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
    ]).unwrap();

    let output = layer_norm.forward(&input);
    assert_eq!(output.dim(), (3, 8));

    for i in 0..3 {
        let row_mean = output.row(i).mean().unwrap();
        assert!(row_mean.abs() < 1e-10);
    }
}

#[test]
fn test_transformer_encoder_layer() {
    let mut layer = TransformerEncoderLayer::new(8, 2, 16, 0.0, CNNActivation::ReLU, Some(42));
    let input = Array2::ones((4, 8));

    let output = layer.forward(&input, None);
    assert_eq!(output.dim(), (4, 8));
}

#[test]
fn test_glu() {
    let glu = GLU::new(8, 4, Some(42));
    let input = Array2::ones((3, 8));

    let output = glu.forward(&input);
    assert_eq!(output.dim(), (3, 4));
}

#[test]
fn test_gelu() {
    let gelu = GELU::new(false);
    let input = Array2::zeros((2, 3));

    let output = gelu.forward(&input);
    assert_eq!(output.dim(), (2, 3));
    assert!((output[(0, 0)] - 0.0).abs() < 1e-10);
}

#[test]
fn test_attention_entropy() {
    let attention_weights = Array2::from_shape_vec((2, 3), vec![
        0.5, 0.3, 0.2,
        0.8, 0.1, 0.1,
    ]).unwrap();

    let entropy = attention_entropy(&attention_weights);
    assert_eq!(entropy.len(), 2);
    assert!(entropy[0] > entropy[1]);
}

#[test]
fn test_create_causal_mask() {
    let mask = create_causal_mask(4);
    assert_eq!(mask.dim(), (4, 4));

    assert!(mask[(0, 0)]);
    assert!(!mask[(0, 1)]);
    assert!(mask[(1, 0)]);
    assert!(mask[(1, 1)]);
    assert!(!mask[(1, 2)]);
}

#[test]
fn test_interpolate_positional_encoding() {
    let original = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let interpolated = interpolate_positional_encoding(&original, 5);

    assert_eq!(interpolated.dim(), (5, 2));
    assert!((interpolated[(0, 0)] - 1.0).abs() < 1e-10);
    assert!((interpolated[(4, 0)] - 5.0).abs() < 1e-10);
}

#[allow(non_snake_case)]
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_end_to_end_autoencoder() {
        let extractor = AutoencoderFeatureExtractor::new()
            .encoding_dim(4)
            .n_epochs(2)
            .learning_rate(0.1);

        let data = Array2::from_shape_vec((5, 8), (0..40).map(|x| x as f64).collect()).unwrap();

        let fitted = extractor.fit(&data, &()).unwrap();
        let features = fitted.transform(&data).unwrap();
        let reconstructed = fitted.reconstruct(&data).unwrap();

        assert_eq!(features.dim(), (5, 4));
        assert_eq!(reconstructed.dim(), (5, 8));

        let error = fitted.reconstruction_error(&data).unwrap();
        assert!(error >= 0.0);
    }

    #[test]
    fn test_end_to_end_transformer() {
        let extractor = TransformerFeatureExtractor::new()
            .d_model(8)
            .n_heads(2)
            .n_layers(1)
            .d_ff(16)
            .n_epochs(1);

        let data = Array2::ones((6, 8));

        let fitted = extractor.fit(&data, &()).unwrap();
        let features = fitted.transform(&data).unwrap();

        assert_eq!(features.dim(), (8,));

        let layer_outputs = fitted.get_layer_outputs(&data).unwrap();
        assert!(!layer_outputs.is_empty());
    }
}