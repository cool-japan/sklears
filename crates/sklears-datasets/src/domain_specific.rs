//! Domain-specific dataset generators
//!
//! This module provides specialized generators for various application domains including
//! bioinformatics, natural language processing, computer vision, privacy-preserving ML,
//! and other specialized use cases.

use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::random::{Random, rng};
use scirs2_core::random::distributions::{Normal, StandardNormal};
use sklears_core::error::{Result, SklearsError};

/// Generate synthetic gene expression dataset
///
/// Creates a dataset simulating gene expression data with specified characteristics
/// such as cell types, experimental conditions, and gene regulation patterns.
///
/// # Parameters
/// - `n_samples`: Number of samples (cells/experiments)
/// - `n_genes`: Number of genes to simulate
/// - `n_cell_types`: Number of distinct cell types
/// - `noise_level`: Level of technical noise to add
/// - `random_state`: Random seed for reproducibility
///
/// # Returns
/// Tuple of (expression_matrix, cell_type_labels)
pub fn make_gene_expression_dataset(
    n_samples: usize,
    n_genes: usize,
    n_cell_types: usize,
    noise_level: f64,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<i32>)> {
    if n_samples == 0 || n_genes == 0 || n_cell_types == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples, n_genes, and n_cell_types must be positive".to_string(),
        ));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));

    // Generate cell type assignments
    let mut labels = Array1::zeros(n_samples);
    for i in 0..n_samples {
        labels[i] = rng.gen_range(0..n_cell_types) as i32;
    }

    // Generate base expression patterns for each cell type
    let mut expression_patterns = Array2::zeros((n_cell_types, n_genes));
    for i in 0..n_cell_types {
        for j in 0..n_genes {
            // Log-normal distribution for expression levels
            let log_expr = rng.sample(Normal::new(5.0, 2.0).unwrap());
            expression_patterns[[i, j]] = log_expr.exp();
        }
    }

    // Generate expression matrix
    let mut expression = Array2::zeros((n_samples, n_genes));
    for i in 0..n_samples {
        let cell_type = labels[i] as usize;
        for j in 0..n_genes {
            let base_expression = expression_patterns[[cell_type, j]];
            let noise = rng.sample(Normal::new(0.0, noise_level).unwrap());
            expression[[i, j]] = base_expression + noise;
        }
    }

    Ok((expression, labels))
}

/// Generate synthetic DNA sequence dataset
///
/// Creates DNA sequences with specified properties including motifs, background
/// composition, and sequence classification labels.
///
/// # Parameters
/// - `n_sequences`: Number of sequences to generate
/// - `sequence_length`: Length of each sequence
/// - `n_classes`: Number of sequence classes
/// - `gc_content`: GC content ratio (0.0 to 1.0)
/// - `random_state`: Random seed for reproducibility
///
/// # Returns
/// Tuple of (encoded_sequences, sequence_labels)
pub fn make_dna_sequence_dataset(
    n_sequences: usize,
    sequence_length: usize,
    n_classes: usize,
    gc_content: f64,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<i32>)> {
    if n_sequences == 0 || sequence_length == 0 || n_classes == 0 {
        return Err(SklearsError::InvalidInput(
            "n_sequences, sequence_length, and n_classes must be positive".to_string(),
        ));
    }

    if gc_content < 0.0 || gc_content > 1.0 {
        return Err(SklearsError::InvalidInput(
            "gc_content must be between 0.0 and 1.0".to_string(),
        ));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));

    // One-hot encoding: A=1000, T=0100, G=0010, C=0001
    let mut sequences = Array2::zeros((n_sequences, sequence_length * 4));
    let mut labels = Array1::zeros(n_sequences);

    for i in 0..n_sequences {
        labels[i] = rng.gen_range(0..n_classes) as i32;

        for j in 0..sequence_length {
            let base_idx = if rng.gen() < gc_content / 2.0 {
                if rng.gen() < 0.5 { 2 } else { 3 } // G or C
            } else {
                if rng.gen() < 0.5 { 0 } else { 1 } // A or T
            };

            sequences[[i, j * 4 + base_idx]] = 1.0;
        }
    }

    Ok((sequences, labels))
}

/// Generate synthetic document clustering dataset
///
/// Creates a dataset simulating document vectors for text clustering tasks
/// with specified topic structure and vocabulary characteristics.
///
/// # Parameters
/// - `n_documents`: Number of documents to generate
/// - `n_features`: Vocabulary size (number of word features)
/// - `n_topics`: Number of distinct topics
/// - `sparsity`: Sparsity level of document vectors (0.0 to 1.0)
/// - `random_state`: Random seed for reproducibility
///
/// # Returns
/// Tuple of (document_term_matrix, topic_labels)
pub fn make_document_clustering_dataset(
    n_documents: usize,
    n_features: usize,
    n_topics: usize,
    sparsity: f64,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<i32>)> {
    if n_documents == 0 || n_features == 0 || n_topics == 0 {
        return Err(SklearsError::InvalidInput(
            "n_documents, n_features, and n_topics must be positive".to_string(),
        ));
    }

    if sparsity < 0.0 || sparsity > 1.0 {
        return Err(SklearsError::InvalidInput(
            "sparsity must be between 0.0 and 1.0".to_string(),
        ));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));

    // Generate topic-word distributions
    let mut topic_distributions = Array2::zeros((n_topics, n_features));
    for i in 0..n_topics {
        for j in 0..n_features {
            // Use gamma distribution for topic-word probabilities
            topic_distributions[[i, j]] = rng.sample(Normal::new(0.0, 1.0).unwrap()).abs();
        }

        // Normalize to get probability distribution
        let row_sum = topic_distributions.row(i).sum();
        for j in 0..n_features {
            topic_distributions[[i, j]] /= row_sum;
        }
    }

    // Generate documents
    let mut documents = Array2::zeros((n_documents, n_features));
    let mut labels = Array1::zeros(n_documents);

    for i in 0..n_documents {
        let topic = rng.gen_range(0..n_topics);
        labels[i] = topic as i32;

        for j in 0..n_features {
            if rng.gen() > sparsity {
                // Sample word count based on topic distribution
                let word_prob = topic_distributions[[topic, j]];
                let word_count = rng.sample(Normal::new(word_prob * 100.0, 10.0).unwrap()).max(0.0);
                documents[[i, j]] = word_count;
            }
        }
    }

    Ok((documents, labels))
}

/// Generate synthetic image classification dataset
///
/// Creates a dataset simulating image features for computer vision tasks
/// with configurable image properties and class structure.
///
/// # Parameters
/// - `n_images`: Number of images to generate
/// - `image_height`: Height of each image
/// - `image_width`: Width of each image
/// - `n_channels`: Number of color channels (1=grayscale, 3=RGB)
/// - `n_classes`: Number of image classes
/// - `noise_level`: Level of pixel noise to add
/// - `random_state`: Random seed for reproducibility
///
/// # Returns
/// Tuple of (flattened_images, image_labels)
pub fn make_synthetic_image_classification(
    n_images: usize,
    image_height: usize,
    image_width: usize,
    n_channels: usize,
    n_classes: usize,
    noise_level: f64,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<i32>)> {
    if n_images == 0 || image_height == 0 || image_width == 0 || n_channels == 0 || n_classes == 0 {
        return Err(SklearsError::InvalidInput(
            "All dimension parameters must be positive".to_string(),
        ));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));

    let n_pixels = image_height * image_width * n_channels;
    let mut images = Array2::zeros((n_images, n_pixels));
    let mut labels = Array1::zeros(n_images);

    // Generate class prototypes
    let mut prototypes = Array2::zeros((n_classes, n_pixels));
    for i in 0..n_classes {
        for j in 0..n_pixels {
            prototypes[[i, j]] = rng.random_range(0.0, 1.0);
        }
    }

    // Generate images based on prototypes
    for i in 0..n_images {
        let class = rng.gen_range(0..n_classes);
        labels[i] = class as i32;

        for j in 0..n_pixels {
            let base_value = prototypes[[class, j]];
            let noise = rng.sample(Normal::new(0.0, noise_level).unwrap());
            images[[i, j]] = (base_value + noise).clamp(0.0, 1.0);
        }
    }

    Ok((images, labels))
}

/// Generate privacy-preserving dataset
///
/// Creates a dataset with differential privacy guarantees by adding calibrated
/// noise to protect individual privacy while maintaining statistical utility.
///
/// # Parameters
/// - `data`: Original data matrix
/// - `epsilon`: Privacy budget (smaller = more private)
/// - `delta`: Privacy parameter for approximate differential privacy
/// - `sensitivity`: Sensitivity of the query/function
/// - `random_state`: Random seed for reproducibility
///
/// # Returns
/// Privacy-preserved version of the input data
pub fn make_privacy_preserving_dataset(
    data: &Array2<f64>,
    epsilon: f64,
    delta: f64,
    sensitivity: f64,
    random_state: Option<u64>,
) -> Result<Array2<f64>> {
    if epsilon <= 0.0 {
        return Err(SklearsError::InvalidInput(
            "epsilon must be positive".to_string(),
        ));
    }

    if delta < 0.0 || delta >= 1.0 {
        return Err(SklearsError::InvalidInput(
            "delta must be between 0.0 and 1.0".to_string(),
        ));
    }

    if sensitivity <= 0.0 {
        return Err(SklearsError::InvalidInput(
            "sensitivity must be positive".to_string(),
        ));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));

    let mut result = data.clone();
    let (n_rows, n_cols) = data.dim();

    // Calculate noise scale for Gaussian mechanism
    let noise_scale = (2.0 * (1.25 / delta).ln()).sqrt() * sensitivity / epsilon;

    // Add calibrated Gaussian noise
    for i in 0..n_rows {
        for j in 0..n_cols {
            let noise = rng.sample(Normal::new(0.0, noise_scale).unwrap());
            result[[i, j]] += noise;
        }
    }

    Ok(result)
}

/// Generate multi-agent environment simulation data
///
/// Creates datasets simulating multi-agent interactions for reinforcement learning
/// and game theory applications.
///
/// # Parameters
/// - `n_episodes`: Number of episodes to simulate
/// - `episode_length`: Length of each episode
/// - `n_agents`: Number of agents in the environment
/// - `n_actions`: Number of possible actions per agent
/// - `cooperation_probability`: Probability of cooperative behavior
/// - `random_state`: Random seed for reproducibility
///
/// # Returns
/// Tuple of (state_action_sequences, rewards)
pub fn make_multi_agent_environment(
    n_episodes: usize,
    episode_length: usize,
    n_agents: usize,
    n_actions: usize,
    cooperation_probability: f64,
    random_state: Option<u64>,
) -> Result<(Array3<f64>, Array2<f64>)> {
    if n_episodes == 0 || episode_length == 0 || n_agents == 0 || n_actions == 0 {
        return Err(SklearsError::InvalidInput(
            "All parameters must be positive".to_string(),
        ));
    }

    if cooperation_probability < 0.0 || cooperation_probability > 1.0 {
        return Err(SklearsError::InvalidInput(
            "cooperation_probability must be between 0.0 and 1.0".to_string(),
        ));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));

    // State-action sequences: (episode, timestep, agent_action_encoding)
    let mut sequences = Array3::zeros((n_episodes, episode_length, n_agents * n_actions));
    let mut rewards = Array2::zeros((n_episodes, episode_length));

    for episode in 0..n_episodes {
        for timestep in 0..episode_length {
            let mut episode_reward = 0.0;

            // Generate actions for each agent
            for agent in 0..n_agents {
                let action = if rng.gen() < cooperation_probability {
                    // Cooperative action (biased towards specific actions)
                    rng.gen_range(0..(n_actions / 2).max(1))
                } else {
                    // Random action
                    rng.gen_range(0..n_actions)
                };

                // One-hot encode the action
                sequences[[episode, timestep, agent * n_actions + action]] = 1.0;

                // Simple reward function based on cooperation
                if action < n_actions / 2 {
                    episode_reward += 1.0;
                } else {
                    episode_reward -= 0.5;
                }
            }

            rewards[[episode, timestep]] = episode_reward / n_agents as f64;
        }
    }

    Ok((sequences, rewards))
}

/// Generate A/B testing simulation data
///
/// Creates datasets for testing A/B testing methods and causal inference
/// with configurable treatment effects and confounding variables.
///
/// # Parameters
/// - `n_users`: Number of users in the experiment
/// - `n_features`: Number of user features
/// - `treatment_effect`: True average treatment effect
/// - `confounding_strength`: Strength of confounding bias
/// - `random_state`: Random seed for reproducibility
///
/// # Returns
/// Tuple of (user_features, treatment_assignments, outcomes)
pub fn make_ab_testing_simulation(
    n_users: usize,
    n_features: usize,
    treatment_effect: f64,
    confounding_strength: f64,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<i32>, Array1<f64>)> {
    if n_users == 0 || n_features == 0 {
        return Err(SklearsError::InvalidInput(
            "n_users and n_features must be positive".to_string(),
        ));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));

    // Generate user features
    let mut features = Array2::zeros((n_users, n_features));
    for i in 0..n_users {
        for j in 0..n_features {
            features[[i, j]] = rng.sample(StandardNormal);
        }
    }

    // Generate treatment assignments with confounding
    let mut treatments = Array1::zeros(n_users);
    for i in 0..n_users {
        let confounding_score = features.row(i).sum() * confounding_strength;
        let treatment_prob = 0.5 + 0.2 * confounding_score.tanh();
        treatments[i] = if rng.gen() < treatment_prob { 1 } else { 0 };
    }

    // Generate outcomes
    let mut outcomes = Array1::zeros(n_users);
    for i in 0..n_users {
        let baseline = features.row(i).sum() * 0.5; // Feature effect
        let treatment_contribution = treatments[i] as f64 * treatment_effect;
        let noise = rng.sample(Normal::new(0.0, 1.0).unwrap());

        outcomes[i] = baseline + treatment_contribution + noise;
    }

    Ok((features, treatments, outcomes))
}