//! Multi-modal and multi-agent environment generators
//!
//! This module contains generators for multi-modal datasets and multi-agent
//! reinforcement learning environments including vision-language, audio-visual,
//! and cooperative multi-agent scenarios.

use scirs2_core::ndarray::{s, Array1, Array2, Array3};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::Normal;
use sklears_core::error::{Result, SklearsError};
use std::f64::consts::PI;

/// Multi-agent environment configuration
#[derive(Debug, Clone)]
pub struct MultiAgentConfig {
    pub n_agents: usize,
    pub n_states: usize,
    pub n_actions: usize,
    pub cooperation_level: f64,
    pub communication_enabled: bool,
    pub reward_sharing: bool,
}

/// Generate multi-agent environment simulation data
pub fn make_multi_agent_environment(
    config: MultiAgentConfig,
    n_episodes: usize,
    episode_length: usize,
    random_state: Option<u64>,
) -> Result<(Array3<usize>, Array3<usize>, Array2<f64>, Array1<f64>)> {
    if config.n_agents == 0 || config.n_states == 0 || config.n_actions == 0 {
        return Err(SklearsError::InvalidInput(
            "n_agents, n_states, and n_actions must be positive".to_string(),
        ));
    }

    if config.cooperation_level < 0.0 || config.cooperation_level > 1.0 {
        return Err(SklearsError::InvalidInput(
            "cooperation_level must be in [0, 1]".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut rand::thread_rng())
    };

    // States: [episode, timestep, agent] -> state_id
    let mut states = Array3::zeros((n_episodes, episode_length, config.n_agents));
    // Actions: [episode, timestep, agent] -> action_id
    let mut actions = Array3::zeros((n_episodes, episode_length, config.n_agents));
    // Rewards: [episode, timestep] -> reward (shared or individual)
    let mut rewards = Array2::zeros((n_episodes, episode_length));
    // Global rewards per episode
    let mut episode_rewards = Array1::zeros(n_episodes);

    for episode in 0..n_episodes {
        let mut episode_reward = 0.0;

        for timestep in 0..episode_length {
            let mut timestep_reward = 0.0;

            // Generate states and actions for each agent
            for agent in 0..config.n_agents {
                // State depends on previous states if not first timestep
                let state = if timestep == 0 {
                    rng.gen_range(0..config.n_states)
                } else {
                    // State transition influenced by other agents if cooperation is enabled
                    if config.cooperation_level > 0.0 && rng.gen::<f64>() < config.cooperation_level
                    {
                        let other_agent = rng.gen_range(0..config.n_agents);
                        if other_agent != agent {
                            let other_state = states[[episode, timestep - 1, other_agent]];
                            (other_state + rng.gen_range(0..3)) % config.n_states
                        } else {
                            rng.gen_range(0..config.n_states)
                        }
                    } else {
                        rng.gen_range(0..config.n_states)
                    }
                };

                states[[episode, timestep, agent]] = state;

                // Action selection - can be influenced by communication
                let action = if config.communication_enabled && rng.gen::<f64>() < 0.3 {
                    // Agent considers other agents' previous actions
                    if timestep > 0 {
                        let other_agent = rng.gen_range(0..config.n_agents);
                        let other_action = actions[[episode, timestep - 1, other_agent]];
                        (other_action + rng.gen_range(0..2)) % config.n_actions
                    } else {
                        rng.gen_range(0..config.n_actions)
                    }
                } else {
                    rng.gen_range(0..config.n_actions)
                };

                actions[[episode, timestep, agent]] = action;

                // Calculate individual reward
                let individual_reward = if state == action % config.n_states {
                    1.0
                } else {
                    0.1 * rng.gen::<f64>()
                };

                if config.reward_sharing {
                    timestep_reward += individual_reward / config.n_agents as f64;
                } else {
                    timestep_reward += individual_reward;
                }
            }

            // Add cooperation bonus
            if config.cooperation_level > 0.0 {
                let mut coordination_bonus = 0.0;
                for agent1 in 0..config.n_agents {
                    for agent2 in (agent1 + 1)..config.n_agents {
                        let state_diff = (states[[episode, timestep, agent1]] as i32
                            - states[[episode, timestep, agent2]] as i32)
                            .abs();
                        if state_diff <= 1 {
                            coordination_bonus += 0.5 * config.cooperation_level;
                        }
                    }
                }
                timestep_reward += coordination_bonus;
            }

            rewards[[episode, timestep]] = timestep_reward;
            episode_reward += timestep_reward;
        }

        episode_rewards[episode] = episode_reward;
    }

    Ok((states, actions, rewards, episode_rewards))
}

/// Generate vision-language aligned dataset
pub fn make_vision_language_dataset(
    n_samples: usize,
    image_size: (usize, usize),
    vocab_size: usize,
    max_sequence_length: usize,
    alignment_strength: f64,
    random_state: Option<u64>,
) -> Result<(Array3<f64>, Array2<usize>, Array1<f64>)> {
    if n_samples == 0 || vocab_size == 0 || max_sequence_length == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples, vocab_size, and max_sequence_length must be positive".to_string(),
        ));
    }

    if alignment_strength < 0.0 || alignment_strength > 1.0 {
        return Err(SklearsError::InvalidInput(
            "alignment_strength must be in [0, 1]".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut rand::thread_rng())
    };

    let (height, width) = image_size;

    // Generate synthetic images
    let mut images = Array3::zeros((n_samples, height, width));
    let normal = Normal::new(0.5, 0.3).unwrap();

    for i in 0..n_samples {
        for h in 0..height {
            for w in 0..width {
                let sample: f64 = rng.sample(normal);
                images[[i, h, w]] = sample.max(0.0).min(1.0);
            }
        }
    }

    // Generate text sequences aligned with images
    let mut texts = Array2::zeros((n_samples, max_sequence_length));
    let mut alignment_scores = Array1::zeros(n_samples);

    for i in 0..n_samples {
        // Calculate image features (mean intensity in regions)
        let image_mean = images.slice(s![i, .., ..]).mean().unwrap_or(0.5);

        // Generate text based on image features
        for j in 0..max_sequence_length {
            let token = if rng.gen::<f64>() < alignment_strength {
                // Aligned token based on image features
                let feature_based_token =
                    ((image_mean * vocab_size as f64) as usize).min(vocab_size - 1);
                feature_based_token
            } else {
                // Random token
                rng.gen_range(0..vocab_size)
            };
            texts[[i, j]] = token;
        }

        // Calculate alignment score
        let text_diversity = texts
            .slice(s![i, ..])
            .iter()
            .map(|&t| t as f64 / vocab_size as f64)
            .collect::<Vec<_>>();
        let text_mean = text_diversity.iter().sum::<f64>() / max_sequence_length as f64;
        alignment_scores[i] = 1.0 - (image_mean - text_mean).abs();
    }

    Ok((images, texts, alignment_scores))
}

/// Generate audio-visual aligned dataset
pub fn make_audio_visual_dataset(
    n_samples: usize,
    audio_length: usize,
    video_frames: usize,
    frame_size: (usize, usize),
    sync_strength: f64,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>)> {
    if n_samples == 0 || audio_length == 0 || video_frames == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples, audio_length, and video_frames must be positive".to_string(),
        ));
    }

    if sync_strength < 0.0 || sync_strength > 1.0 {
        return Err(SklearsError::InvalidInput(
            "sync_strength must be in [0, 1]".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut rand::thread_rng())
    };

    let (height, width) = frame_size;

    // Generate audio signals
    let mut audio = Array2::zeros((n_samples, audio_length));
    let normal = Normal::new(0.0, 0.5).unwrap();

    for i in 0..n_samples {
        let base_frequency = rng.gen_range(0.1..0.5);
        for t in 0..audio_length {
            let time = t as f64 / audio_length as f64;
            let signal = (2.0 * PI * base_frequency * time).sin();
            let noise: f64 = rng.sample(normal);
            audio[[i, t]] = signal + 0.1 * noise;
        }
    }

    // Generate video frames synchronized with audio
    let mut video = Array2::zeros((n_samples, video_frames * height * width));
    let mut sync_scores = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let audio_energy = audio.slice(s![i, ..]).mapv(|x| x * x).mean().unwrap_or(0.0);

        for frame in 0..video_frames {
            let frame_start = frame * height * width;
            let audio_frame_idx = (frame * audio_length) / video_frames;
            let audio_frame_energy = if audio_frame_idx < audio_length {
                audio[[i, audio_frame_idx]].abs()
            } else {
                0.0
            };

            for pixel in 0..(height * width) {
                let idx = frame_start + pixel;
                if rng.gen::<f64>() < sync_strength {
                    // Synchronized pixel based on audio
                    video[[i, idx]] = audio_frame_energy + 0.2 * rng.gen::<f64>();
                } else {
                    // Random pixel
                    video[[i, idx]] = rng.gen::<f64>();
                }
            }
        }

        // Calculate synchronization score
        let video_energy = video.slice(s![i, ..]).mapv(|x| x * x).mean().unwrap_or(0.0);
        sync_scores[i] = 1.0 - (audio_energy.sqrt() - video_energy.sqrt()).abs();
    }

    Ok((audio, video, sync_scores))
}

/// Communication cost configuration for federated learning
#[derive(Debug, Clone)]
pub struct CommunicationCostConfig {
    pub n_clients: usize,
    pub network_topology: String, // "star", "ring", "full_mesh", "hierarchical"
    pub bandwidth_mbps: f64,
    pub latency_ms: f64,
    pub packet_loss_rate: f64,
    pub compression_ratio: f64,
}

/// Generate communication cost datasets for federated learning
pub fn make_communication_cost_datasets(
    config: CommunicationCostConfig,
    n_rounds: usize,
    model_size_mb: f64,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>)> {
    if config.n_clients == 0 || n_rounds == 0 {
        return Err(SklearsError::InvalidInput(
            "n_clients and n_rounds must be positive".to_string(),
        ));
    }

    if model_size_mb <= 0.0 || config.bandwidth_mbps <= 0.0 || config.latency_ms < 0.0 {
        return Err(SklearsError::InvalidInput(
            "model_size_mb, bandwidth_mbps must be positive, latency_ms must be non-negative"
                .to_string(),
        ));
    }

    if config.packet_loss_rate < 0.0 || config.packet_loss_rate > 1.0 {
        return Err(SklearsError::InvalidInput(
            "packet_loss_rate must be in [0, 1]".to_string(),
        ));
    }

    if config.compression_ratio <= 0.0 || config.compression_ratio > 1.0 {
        return Err(SklearsError::InvalidInput(
            "compression_ratio must be in (0, 1]".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut rand::thread_rng())
    };

    // Communication costs per round per client
    let mut upload_costs = Array2::zeros((n_rounds, config.n_clients));
    let mut download_costs = Array2::zeros((n_rounds, config.n_clients));
    let mut round_times = Array1::zeros(n_rounds);
    let mut total_bandwidth_usage = Array1::zeros(n_rounds);

    // Calculate topology-specific parameters
    let (n_connections, aggregation_factor) = match config.network_topology.as_str() {
        "star" => (1, 1.0), // Each client connects to server
        "ring" => (2, 0.5), // Each client connects to 2 neighbors
        "full_mesh" => (config.n_clients - 1, 1.0 / (config.n_clients - 1) as f64),
        "hierarchical" => ((config.n_clients as f64).sqrt() as usize, 0.3),
        _ => {
            return Err(SklearsError::InvalidInput(
                "Invalid network topology. Must be one of: star, ring, full_mesh, hierarchical"
                    .to_string(),
            ))
        }
    };

    let compressed_model_size = model_size_mb * config.compression_ratio;

    for round in 0..n_rounds {
        let mut max_round_time: f64 = 0.0;
        let mut total_bandwidth_round = 0.0;

        for client in 0..config.n_clients {
            // Add random variation to network conditions
            let bandwidth_variation = 1.0 + 0.3 * (rng.gen::<f64>() - 0.5); // ±15% variation
            let latency_variation = 1.0 + 0.5 * (rng.gen::<f64>() - 0.5); // ±25% variation
            let effective_bandwidth = config.bandwidth_mbps * bandwidth_variation;
            let effective_latency = config.latency_ms * latency_variation;

            // Calculate upload cost (client to server/aggregator)
            let upload_time = (compressed_model_size / effective_bandwidth) * 1000.0; // Convert to ms
            let upload_total_time = upload_time + effective_latency;

            // Factor in packet loss (retransmissions)
            let retransmission_factor = if config.packet_loss_rate > 0.0 {
                1.0 / (1.0 - config.packet_loss_rate).max(0.1) // Avoid division by zero
            } else {
                1.0
            };

            let upload_cost = upload_total_time * retransmission_factor;
            upload_costs[[round, client]] = upload_cost;

            // Calculate download cost (server/aggregator to client)
            let download_time = (compressed_model_size / effective_bandwidth) * 1000.0; // Convert to ms
            let download_total_time = download_time + effective_latency;
            let download_cost = download_total_time * retransmission_factor * aggregation_factor;
            download_costs[[round, client]] = download_cost;

            // Calculate maximum time for this round (bottleneck client)
            let client_round_time = upload_cost + download_cost;
            max_round_time = max_round_time.max(client_round_time);

            // Calculate bandwidth usage
            let bandwidth_used = (compressed_model_size * 2.0) * retransmission_factor; // Upload + download
            total_bandwidth_round += bandwidth_used;
        }

        round_times[round] = max_round_time;
        total_bandwidth_usage[round] = total_bandwidth_round;
    }

    Ok((
        upload_costs,
        download_costs,
        round_times,
        total_bandwidth_usage,
    ))
}

/// Generate sensor fusion datasets for multi-modal data
pub fn make_sensor_fusion_dataset(
    n_samples: usize,
    sensor_types: Vec<String>, // e.g., ["accelerometer", "gyroscope", "magnetometer", "camera", "lidar"]
    temporal_length: usize,
    sync_accuracy: f64, // Synchronization accuracy between sensors
    random_state: Option<u64>,
) -> Result<(Vec<Array2<f64>>, Array1<f64>, Array1<usize>)> {
    if n_samples == 0 || sensor_types.is_empty() || temporal_length == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples, sensor_types, and temporal_length must be positive".to_string(),
        ));
    }

    if sync_accuracy < 0.0 || sync_accuracy > 1.0 {
        return Err(SklearsError::InvalidInput(
            "sync_accuracy must be in [0, 1]".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut rand::thread_rng())
    };

    let n_sensors = sensor_types.len();
    let mut sensor_data = Vec::new();
    let mut fusion_quality = Array1::zeros(n_samples);
    let mut ground_truth_events = Array1::zeros(n_samples);

    // Define sensor characteristics
    let sensor_dimensions: Vec<usize> = sensor_types
        .iter()
        .map(|sensor_type| {
            match sensor_type.as_str() {
                "accelerometer" => 3, // x, y, z
                "gyroscope" => 3,     // roll, pitch, yaw
                "magnetometer" => 3,  // x, y, z
                "camera" => 16,       // Feature vector from image
                "lidar" => 8,         // Distance measurements
                "gps" => 2,           // lat, lon
                "microphone" => 1,    // Audio signal
                "pressure" => 1,      // Atmospheric pressure
                _ => 4,               // Default dimension
            }
        })
        .collect();

    // Initialize sensor data arrays
    for i in 0..n_sensors {
        sensor_data.push(Array2::zeros((
            n_samples,
            temporal_length * sensor_dimensions[i],
        )));
    }

    let normal = Normal::new(0.0, 1.0).unwrap();

    for sample in 0..n_samples {
        // Generate a ground truth event (e.g., motion pattern)
        let event_type = rng.gen_range(0..5); // 5 different event types
        ground_truth_events[sample] = event_type;

        // Generate base signal for this event
        let base_frequency = 0.1 + (event_type as f64) * 0.05;
        let base_amplitude = 0.5 + (event_type as f64) * 0.2;

        let mut sensor_correlations = Vec::new();

        for (sensor_idx, sensor_type) in sensor_types.iter().enumerate() {
            let dims = sensor_dimensions[sensor_idx];
            let mut sensor_correlation = 0.0;

            for t in 0..temporal_length {
                let time = t as f64 / temporal_length as f64;

                // Generate base signal
                let base_signal = base_amplitude * (2.0 * PI * base_frequency * time).sin();

                // Add sync error
                let sync_error = if rng.gen::<f64>() > sync_accuracy {
                    0.1 * rng.gen::<f64>() // Random desynchronization
                } else {
                    0.0
                };

                for dim in 0..dims {
                    let feature_idx = t * dims + dim;

                    // Generate sensor-specific signal
                    let sensor_signal = match sensor_type.as_str() {
                        "accelerometer" => {
                            let gravity_component = if dim == 2 { 9.81 } else { 0.0 };
                            base_signal * (1.0 + dim as f64 * 0.3) + gravity_component
                        }
                        "gyroscope" => {
                            base_signal * (2.0 + dim as f64 * 0.5) // Higher frequency for rotation
                        }
                        "magnetometer" => {
                            let magnetic_field = if dim == 0 { 25.0 } else { 5.0 };
                            base_signal * 0.5 + magnetic_field
                        }
                        "camera" => {
                            let pixel_intensity = 0.5 + 0.3 * base_signal;
                            pixel_intensity.max(0.0).min(1.0)
                        }
                        "lidar" => {
                            let distance = 5.0 + 2.0 * base_signal + 0.5 * (dim as f64);
                            distance.max(0.1)
                        }
                        "gps" => {
                            let coord_variation = 0.0001 * base_signal;
                            if dim == 0 {
                                37.7749 + coord_variation
                            } else {
                                -122.4194 + coord_variation
                            }
                        }
                        "microphone" => base_signal * 0.8,
                        "pressure" => {
                            1013.25 + 10.0 * base_signal // Standard atmospheric pressure with variation
                        }
                        _ => base_signal,
                    };

                    // Add noise and sync error
                    let noise: f64 = rng.sample(normal);
                    let final_signal = sensor_signal + 0.1 * noise + sync_error;

                    sensor_data[sensor_idx][[sample, feature_idx]] = final_signal;

                    // Calculate correlation with base signal for quality assessment
                    sensor_correlation += (final_signal - sensor_signal).abs();
                }
            }

            sensor_correlations.push(sensor_correlation / (temporal_length * dims) as f64);
        }

        // Calculate overall fusion quality
        let avg_correlation = sensor_correlations.iter().sum::<f64>() / n_sensors as f64;
        fusion_quality[sample] = (1.0 - avg_correlation.min(1.0)).max(0.0);
    }

    Ok((sensor_data, fusion_quality, ground_truth_events))
}

/// Generate multi-modal alignment datasets
pub fn make_multimodal_alignment_dataset(
    n_samples: usize,
    modality_types: Vec<String>, // e.g., ["text", "image", "audio", "video"]
    alignment_strength: f64,
    cross_modal_noise: f64,
    random_state: Option<u64>,
) -> Result<(Vec<Array2<f64>>, Array2<f64>, Array1<f64>)> {
    if n_samples == 0 || modality_types.is_empty() {
        return Err(SklearsError::InvalidInput(
            "n_samples and modality_types must be positive".to_string(),
        ));
    }

    if alignment_strength < 0.0 || alignment_strength > 1.0 {
        return Err(SklearsError::InvalidInput(
            "alignment_strength must be in [0, 1]".to_string(),
        ));
    }

    if cross_modal_noise < 0.0 || cross_modal_noise > 1.0 {
        return Err(SklearsError::InvalidInput(
            "cross_modal_noise must be in [0, 1]".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut rand::thread_rng())
    };

    let n_modalities = modality_types.len();
    let mut modality_data = Vec::new();
    let mut alignment_scores = Array1::zeros(n_samples);

    // Define modality dimensions
    let modality_dimensions: Vec<usize> = modality_types
        .iter()
        .map(|modality| {
            match modality.as_str() {
                "text" => 300,   // Text embeddings
                "image" => 2048, // Image features
                "audio" => 128,  // Audio features
                "video" => 1024, // Video features
                _ => 256,        // Default dimension
            }
        })
        .collect();

    // Initialize modality data arrays
    for i in 0..n_modalities {
        modality_data.push(Array2::zeros((n_samples, modality_dimensions[i])));
    }

    // Cross-modal alignment matrix
    let mut cross_modal_alignment = Array2::zeros((n_samples, n_modalities * n_modalities));

    let normal = Normal::new(0.0, 1.0).unwrap();

    for sample in 0..n_samples {
        // Generate shared semantic content
        let mut shared_content = Array1::zeros(64); // Shared semantic space
        for i in 0..64 {
            shared_content[i] = rng.sample(normal);
        }

        let mut modality_embeddings = Vec::new();

        for (mod_idx, modality) in modality_types.iter().enumerate() {
            let dims = modality_dimensions[mod_idx];
            let mut embedding = Array1::zeros(dims);

            // Generate modality-specific content based on shared content
            for i in 0..dims {
                let shared_influence = if i < 64 {
                    alignment_strength * shared_content[i]
                } else {
                    0.0
                };

                let modality_specific = match modality.as_str() {
                    "text" => {
                        // Text features: semantic similarity, word frequency, etc.
                        let word_freq = rng.gen::<f64>() * 0.1;
                        let semantic_sim = shared_influence * 0.8;
                        semantic_sim + word_freq
                    }
                    "image" => {
                        // Image features: visual patterns, colors, textures
                        let visual_pattern = (shared_influence * 2.0).sin() * 0.5;
                        let color_intensity = rng.gen::<f64>() * 0.3;
                        visual_pattern + color_intensity
                    }
                    "audio" => {
                        // Audio features: spectral content, rhythm, etc.
                        let spectral_content = shared_influence * 0.6;
                        let rhythm_component = ((i as f64 / 10.0) * PI).sin() * 0.2;
                        spectral_content + rhythm_component
                    }
                    "video" => {
                        // Video features: temporal patterns, motion, etc.
                        let temporal_pattern = shared_influence * 0.7;
                        let motion_component = rng.gen::<f64>() * 0.2;
                        temporal_pattern + motion_component
                    }
                    _ => shared_influence,
                };

                let noise: f64 = rng.sample(normal);
                embedding[i] = modality_specific + cross_modal_noise * noise;
            }

            modality_embeddings.push(embedding.clone());

            // Store in output array
            for i in 0..dims {
                modality_data[mod_idx][[sample, i]] = embedding[i];
            }
        }

        // Calculate cross-modal alignment scores
        let mut total_alignment = 0.0;
        let mut alignment_count = 0;

        for i in 0..n_modalities {
            for j in (i + 1)..n_modalities {
                // Calculate cross-modal similarity using shared content overlap
                // Since modalities have different dimensions, we use the shared content portion
                let shared_len = 64
                    .min(modality_embeddings[i].len())
                    .min(modality_embeddings[j].len());

                let mut dot_product = 0.0;
                let mut norm_i_sq = 0.0;
                let mut norm_j_sq = 0.0;

                for k in 0..shared_len {
                    let val_i = modality_embeddings[i][k];
                    let val_j = modality_embeddings[j][k];
                    dot_product += val_i * val_j;
                    norm_i_sq += val_i * val_i;
                    norm_j_sq += val_j * val_j;
                }

                let cosine_sim = if norm_i_sq > 0.0 && norm_j_sq > 0.0 {
                    dot_product / (norm_i_sq.sqrt() * norm_j_sq.sqrt())
                } else {
                    0.0
                };

                cross_modal_alignment[[sample, i * n_modalities + j]] = cosine_sim;
                total_alignment += cosine_sim.abs();
                alignment_count += 1;
            }
        }

        alignment_scores[sample] = total_alignment / alignment_count as f64;
    }

    Ok((modality_data, cross_modal_alignment, alignment_scores))
}

/// Generate cross-modal retrieval datasets
pub fn make_cross_modal_retrieval_dataset(
    n_samples: usize,
    source_modality: String,   // "text", "image", "audio"
    target_modality: String,   // "text", "image", "audio"
    n_distractors: usize,      // Number of negative examples per positive
    retrieval_difficulty: f64, // 0.0 (easy) to 1.0 (hard)
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array2<f64>, Array1<usize>, Array1<f64>)> {
    if n_samples == 0 || n_distractors == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples and n_distractors must be positive".to_string(),
        ));
    }

    if retrieval_difficulty < 0.0 || retrieval_difficulty > 1.0 {
        return Err(SklearsError::InvalidInput(
            "retrieval_difficulty must be in [0, 1]".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut rand::thread_rng())
    };

    // Define embedding dimensions for each modality
    let source_dim = match source_modality.as_str() {
        "text" => 300,
        "image" => 2048,
        "audio" => 128,
        _ => 256,
    };

    let target_dim = match target_modality.as_str() {
        "text" => 300,
        "image" => 2048,
        "audio" => 128,
        _ => 256,
    };

    let total_targets = n_samples * (1 + n_distractors);
    let mut source_embeddings = Array2::zeros((n_samples, source_dim));
    let mut target_embeddings = Array2::zeros((total_targets, target_dim));
    let mut ground_truth_indices = Array1::zeros(n_samples);
    let mut retrieval_scores = Array1::zeros(n_samples);

    let normal = Normal::new(0.0, 1.0).unwrap();

    for sample in 0..n_samples {
        // Generate shared semantic representation
        let mut shared_content = Array1::zeros(64);
        for i in 0..64 {
            shared_content[i] = rng.sample(normal);
        }

        // Generate source modality embedding
        for i in 0..source_dim {
            let shared_influence = if i < 64 { 0.8 * shared_content[i] } else { 0.0 };

            let modality_specific = match source_modality.as_str() {
                "text" => {
                    let semantic_weight = 0.6 + 0.4 * rng.gen::<f64>();
                    shared_influence * semantic_weight
                }
                "image" => {
                    let visual_pattern = (shared_influence * 1.5).sin() * 0.7;
                    visual_pattern
                }
                "audio" => {
                    let spectral_component = shared_influence * 0.8;
                    spectral_component
                }
                _ => shared_influence,
            };

            let noise: f64 = rng.sample(normal);
            source_embeddings[[sample, i]] = modality_specific + 0.1 * noise;
        }

        // Generate positive target (ground truth)
        let target_idx = sample * (1 + n_distractors);
        ground_truth_indices[sample] = target_idx;

        for i in 0..target_dim {
            let shared_influence = if i < 64 { 0.8 * shared_content[i] } else { 0.0 };

            let difficulty_noise = retrieval_difficulty * rng.sample(normal);

            let modality_specific = match target_modality.as_str() {
                "text" => {
                    let semantic_weight = 0.6 + 0.4 * rng.gen::<f64>();
                    shared_influence * semantic_weight
                }
                "image" => {
                    let visual_pattern = (shared_influence * 1.5).sin() * 0.7;
                    visual_pattern
                }
                "audio" => {
                    let spectral_component = shared_influence * 0.8;
                    spectral_component
                }
                _ => shared_influence,
            };

            let noise: f64 = rng.sample(normal);
            target_embeddings[[target_idx, i]] = modality_specific + 0.1 * noise + difficulty_noise;
        }

        // Generate negative targets (distractors)
        for distractor in 0..n_distractors {
            let distractor_idx = target_idx + 1 + distractor;

            // Generate different shared content for distractors
            let mut distractor_content = Array1::zeros(64);
            for i in 0..64 {
                distractor_content[i] = rng.sample(normal);
            }

            for i in 0..target_dim {
                let shared_influence = if i < 64 {
                    0.8 * distractor_content[i]
                } else {
                    0.0
                };

                let modality_specific = match target_modality.as_str() {
                    "text" => {
                        let semantic_weight = 0.6 + 0.4 * rng.gen::<f64>();
                        shared_influence * semantic_weight
                    }
                    "image" => {
                        let visual_pattern = (shared_influence * 1.5).sin() * 0.7;
                        visual_pattern
                    }
                    "audio" => {
                        let spectral_component = shared_influence * 0.8;
                        spectral_component
                    }
                    _ => shared_influence,
                };

                let noise: f64 = rng.sample(normal);
                target_embeddings[[distractor_idx, i]] = modality_specific + 0.1 * noise;
            }
        }

        // Calculate retrieval score (similarity between source and ground truth target)
        // Use shared content dimensions for cross-modal similarity
        let shared_len = 64.min(source_dim).min(target_dim);

        let mut dot_product = 0.0;
        let mut source_norm_sq = 0.0;
        let mut target_norm_sq = 0.0;

        for k in 0..shared_len {
            let source_val = source_embeddings[[sample, k]];
            let target_val = target_embeddings[[target_idx, k]];
            dot_product += source_val * target_val;
            source_norm_sq += source_val * source_val;
            target_norm_sq += target_val * target_val;
        }

        let cosine_sim = if source_norm_sq > 0.0 && target_norm_sq > 0.0 {
            dot_product / (source_norm_sq.sqrt() * target_norm_sq.sqrt())
        } else {
            0.0
        };

        retrieval_scores[sample] = cosine_sim;
    }

    Ok((
        source_embeddings,
        target_embeddings,
        ground_truth_indices,
        retrieval_scores,
    ))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_multi_agent_environment() {
        let config = MultiAgentConfig {
            n_agents: 3,
            n_states: 4,
            n_actions: 2,
            cooperation_level: 0.5,
            communication_enabled: true,
            reward_sharing: false,
        };

        let (states, actions, rewards, episode_rewards) =
            make_multi_agent_environment(config, 10, 20, Some(42)).unwrap();

        assert_eq!(states.shape(), &[10, 20, 3]);
        assert_eq!(actions.shape(), &[10, 20, 3]);
        assert_eq!(rewards.shape(), &[10, 20]);
        assert_eq!(episode_rewards.len(), 10);

        // Check state values are within bounds
        for &state in states.iter() {
            assert!(state < 4, "States should be within [0, n_states)");
        }

        // Check action values are within bounds
        for &action in actions.iter() {
            assert!(action < 2, "Actions should be within [0, n_actions)");
        }
    }

    #[test]
    fn test_make_vision_language_dataset() {
        let (images, texts, alignment_scores) =
            make_vision_language_dataset(50, (32, 32), 1000, 10, 0.7, Some(42)).unwrap();

        assert_eq!(images.shape(), &[50, 32, 32]);
        assert_eq!(texts.shape(), &[50, 10]);
        assert_eq!(alignment_scores.len(), 50);

        // Check image values are in [0, 1]
        for &pixel in images.iter() {
            assert!(
                pixel >= 0.0 && pixel <= 1.0,
                "Image pixels should be in [0, 1]"
            );
        }

        // Check text tokens are within vocabulary
        for &token in texts.iter() {
            assert!(token < 1000, "Text tokens should be within vocab_size");
        }

        // Check alignment scores are in [0, 1]
        for &score in alignment_scores.iter() {
            assert!(
                score >= 0.0 && score <= 1.0,
                "Alignment scores should be in [0, 1]"
            );
        }
    }

    #[test]
    fn test_make_audio_visual_dataset() {
        let (audio, video, sync_scores) =
            make_audio_visual_dataset(20, 100, 10, (8, 8), 0.8, Some(42)).unwrap();

        assert_eq!(audio.shape(), &[20, 100]);
        assert_eq!(video.shape(), &[20, 640]); // 10 frames * 8 * 8 pixels
        assert_eq!(sync_scores.len(), 20);

        // Check that audio has some variation
        let audio_var = audio.var(0.0);
        assert!(audio_var > 0.0, "Audio should have variation");

        // Check sync scores are reasonable
        for &score in sync_scores.iter() {
            assert!(
                score >= 0.0 && score <= 1.0,
                "Sync scores should be in [0, 1]"
            );
        }
    }

    #[test]
    fn test_make_communication_cost_datasets() {
        let config = CommunicationCostConfig {
            n_clients: 5,
            network_topology: "star".to_string(),
            bandwidth_mbps: 10.0,
            latency_ms: 50.0,
            packet_loss_rate: 0.05,
            compression_ratio: 0.8,
        };

        let (upload_costs, download_costs, round_times, bandwidth_usage) =
            make_communication_cost_datasets(config, 10, 2.5, Some(42)).unwrap();

        assert_eq!(upload_costs.shape(), &[10, 5]);
        assert_eq!(download_costs.shape(), &[10, 5]);
        assert_eq!(round_times.len(), 10);
        assert_eq!(bandwidth_usage.len(), 10);

        // Check that costs are positive
        for &cost in upload_costs.iter() {
            assert!(cost > 0.0, "Upload costs should be positive");
        }

        for &cost in download_costs.iter() {
            assert!(cost > 0.0, "Download costs should be positive");
        }

        for &time in round_times.iter() {
            assert!(time > 0.0, "Round times should be positive");
        }

        for &usage in bandwidth_usage.iter() {
            assert!(usage > 0.0, "Bandwidth usage should be positive");
        }
    }

    #[test]
    fn test_make_sensor_fusion_dataset() {
        let sensor_types = vec![
            "accelerometer".to_string(),
            "gyroscope".to_string(),
            "camera".to_string(),
        ];

        let (sensor_data, fusion_quality, ground_truth_events) =
            make_sensor_fusion_dataset(20, sensor_types, 10, 0.9, Some(42)).unwrap();

        assert_eq!(sensor_data.len(), 3); // 3 sensors
        assert_eq!(sensor_data[0].shape(), &[20, 30]); // accelerometer: 20 samples, 10 timesteps * 3 dims
        assert_eq!(sensor_data[1].shape(), &[20, 30]); // gyroscope: 20 samples, 10 timesteps * 3 dims
        assert_eq!(sensor_data[2].shape(), &[20, 160]); // camera: 20 samples, 10 timesteps * 16 dims
        assert_eq!(fusion_quality.len(), 20);
        assert_eq!(ground_truth_events.len(), 20);

        // Check fusion quality is in [0, 1]
        for &quality in fusion_quality.iter() {
            assert!(
                quality >= 0.0 && quality <= 1.0,
                "Fusion quality should be in [0, 1]"
            );
        }

        // Check ground truth events are within expected range
        for &event in ground_truth_events.iter() {
            assert!(event < 5, "Ground truth events should be within [0, 5)");
        }
    }

    #[test]
    fn test_make_multimodal_alignment_dataset() {
        let modality_types = vec!["text".to_string(), "image".to_string(), "audio".to_string()];

        let (modality_data, cross_modal_alignment, alignment_scores) =
            make_multimodal_alignment_dataset(15, modality_types, 0.8, 0.2, Some(42)).unwrap();

        assert_eq!(modality_data.len(), 3); // 3 modalities
        assert_eq!(modality_data[0].shape(), &[15, 300]); // text: 15 samples, 300 dims
        assert_eq!(modality_data[1].shape(), &[15, 2048]); // image: 15 samples, 2048 dims
        assert_eq!(modality_data[2].shape(), &[15, 128]); // audio: 15 samples, 128 dims
        assert_eq!(cross_modal_alignment.shape(), &[15, 9]); // 15 samples, 3*3 alignment matrix
        assert_eq!(alignment_scores.len(), 15);

        // Check alignment scores are reasonable
        for &score in alignment_scores.iter() {
            assert!(
                score >= 0.0 && score <= 1.0,
                "Alignment scores should be in [0, 1]"
            );
        }
    }

    #[test]
    fn test_make_cross_modal_retrieval_dataset() {
        let (source_embeddings, target_embeddings, ground_truth_indices, retrieval_scores) =
            make_cross_modal_retrieval_dataset(
                10,
                "text".to_string(),
                "image".to_string(),
                5,
                0.3,
                Some(42),
            )
            .unwrap();

        assert_eq!(source_embeddings.shape(), &[10, 300]); // 10 samples, 300 text dims
        assert_eq!(target_embeddings.shape(), &[60, 2048]); // 10 * (1 + 5) targets, 2048 image dims
        assert_eq!(ground_truth_indices.len(), 10);
        assert_eq!(retrieval_scores.len(), 10);

        // Check ground truth indices are within expected range
        for &idx in ground_truth_indices.iter() {
            assert!(
                idx < 60,
                "Ground truth indices should be within target range"
            );
        }

        // Check retrieval scores are reasonable
        for &score in retrieval_scores.iter() {
            assert!(
                score >= -1.0 && score <= 1.0,
                "Retrieval scores should be in [-1, 1] (cosine similarity)"
            );
        }
    }
}
