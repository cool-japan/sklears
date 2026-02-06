//! Concrete Tutorial Examples and Learning Paths
//!
//! This module provides complete, executable tutorial examples for learning
//! sklears concepts, from basic usage to advanced techniques.

use crate::tutorial_system::{
    Assessment, DifficultyLevel, LearningPath, SectionContent, Tutorial, TutorialCategory,
    TutorialMetadata, TutorialSection, TutorialSystem,
};
use std::collections::HashMap;

/// Generate a complete set of beginner tutorials
///
/// Creates a comprehensive tutorial series for users new to sklears,
/// covering basic concepts, data loading, model training, and evaluation.
pub fn generate_beginner_tutorials() -> Vec<Tutorial> {
    vec![
        create_getting_started_tutorial(),
        create_data_loading_tutorial(),
        create_basic_regression_tutorial(),
        create_basic_classification_tutorial(),
        create_preprocessing_tutorial(),
    ]
}

/// Generate intermediate level tutorials
///
/// Creates tutorials for users familiar with basics who want to learn
/// advanced features like pipelines, cross-validation, and hyperparameter tuning.
pub fn generate_intermediate_tutorials() -> Vec<Tutorial> {
    vec![
        create_pipeline_tutorial(),
        create_cross_validation_tutorial(),
        create_hyperparameter_tuning_tutorial(),
        create_ensemble_methods_tutorial(),
        create_feature_engineering_tutorial(),
    ]
}

/// Generate advanced level tutorials
///
/// Creates tutorials for advanced users covering optimization, distributed
/// computing, custom estimators, and performance tuning.
pub fn generate_advanced_tutorials() -> Vec<Tutorial> {
    vec![
        create_custom_estimator_tutorial(),
        create_distributed_computing_tutorial(),
        create_performance_optimization_tutorial(),
        create_gpu_acceleration_tutorial(),
        create_type_safety_tutorial(),
    ]
}

/// Create a comprehensive learning path for beginners
pub fn create_beginner_learning_path() -> LearningPath {
    LearningPath {
        id: "beginner_path".to_string(),
        title: "Beginner's Path to Machine Learning with sklears".to_string(),
        description: "Complete learning path for mastering basic machine learning with sklears"
            .to_string(),
        difficulty: DifficultyLevel::Beginner,
        estimated_hours: 12,
        tutorial_sequence: generate_beginner_tutorials()
            .iter()
            .map(|t| t.id.clone())
            .collect(),
        prerequisites: vec![],
        completion_rewards: vec![
            "Understand core sklears concepts and APIs".to_string(),
            "Load and preprocess data effectively".to_string(),
            "Train and evaluate basic ML models".to_string(),
            "Apply best practices for model development".to_string(),
        ],
    }
}

/// Create the "Getting Started" tutorial
fn create_getting_started_tutorial() -> Tutorial {
    Tutorial {
        id: "getting_started".to_string(),
        title: "Getting Started with sklears".to_string(),
        description: "Learn the basics of sklears and build your first machine learning model"
            .to_string(),
        difficulty: DifficultyLevel::Beginner,
        duration_minutes: 30,
        prerequisites: vec![],
        learning_objectives: vec![
            "Understand the sklears architecture".to_string(),
            "Install and configure sklears".to_string(),
            "Build and train your first model".to_string(),
            "Make predictions on new data".to_string(),
        ],
        sections: vec![
            TutorialSection {
                id: "intro".to_string(),
                title: "Introduction to sklears".to_string(),
                content: SectionContent::Text {
                    content: r#"
# Welcome to sklears!

sklears is a high-performance machine learning library for Rust, offering:

- **Pure Rust implementation** with ongoing performance optimization
- **Type safety** at compile time
- **Zero-cost abstractions** for ML algorithms
- **scikit-learn compatible** API

## Core Concepts

sklears is built around several key traits:

- `Estimator`: Base trait for all ML algorithms
- `Fit`: Training interface
- `Predict`: Prediction interface
- `Transform`: Data transformation interface

Let's see them in action!
"#
                    .to_string(),
                    format: crate::tutorial_system::ContentFormat::Markdown,
                },
                interactive_elements: vec![],
                estimated_duration: 5,
                completion_criteria: crate::tutorial_system::CompletionCriteria {
                    required_interactions: vec![],
                    minimum_score: None,
                    time_spent_minimum: Some(60), // At least 1 minute
                    code_execution_required: false,
                },
            },
            TutorialSection {
                id: "installation".to_string(),
                title: "Installation and Setup".to_string(),
                content: SectionContent::Text {
                    content: r#"
# Installation

Add sklears to your `Cargo.toml`:

```toml
[dependencies]
sklears = "0.1.0"
```

For specific features:

```toml
[dependencies]
sklears = { version = "0.1.0", features = ["full"] }
```

## Quick Test

Let's verify your installation:

```rust
use sklears::prelude::*;

fn main() {
    println!("sklears version: {}", sklears::VERSION);
}
```
"#
                    .to_string(),
                    format: crate::tutorial_system::ContentFormat::Markdown,
                },
                interactive_elements: vec![],
                estimated_duration: 10,
                completion_criteria: crate::tutorial_system::CompletionCriteria {
                    required_interactions: vec!["verify_install".to_string()],
                    minimum_score: None,
                    time_spent_minimum: Some(120),
                    code_execution_required: true,
                },
            },
            TutorialSection {
                id: "first_model".to_string(),
                title: "Your First Model".to_string(),
                content: SectionContent::Text {
                    content: r#"
# Building Your First Model

Let's create a simple linear regression model:

```rust
use sklears::prelude::*;
use sklears::linear::LinearRegression;

fn main() -> Result<()> {
    // Create training data
    let X = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

    // Create and train the model
    let model = LinearRegression::builder()
        .fit_intercept(true)
        .build()?;

    let trained = model.fit(&X, &y)?;

    // Make predictions
    let X_test = array![[6.0], [7.0]];
    let predictions = trained.predict(&X_test)?;

    println!("Predictions: {:?}", predictions);
    // Expected: [12.0, 14.0]

    Ok(())
}
```

## Key Points

1. **Builder Pattern**: Configure models using builders
2. **Type Safety**: Compile-time checks prevent errors
3. **Error Handling**: Use `Result<T>` for operations that can fail
4. **Immutability**: Training returns a new trained model
"#
                    .to_string(),
                    format: crate::tutorial_system::ContentFormat::Markdown,
                },
                interactive_elements: vec![],
                estimated_duration: 15,
                completion_criteria: crate::tutorial_system::CompletionCriteria {
                    required_interactions: vec!["first_model".to_string()],
                    minimum_score: None,
                    time_spent_minimum: Some(180),
                    code_execution_required: true,
                },
            },
        ],
        assessment: Some(Assessment {
            id: "getting_started_quiz".to_string(),
            title: "Getting Started Quiz".to_string(),
            description: "Test your understanding of basic sklears concepts".to_string(),
            questions: vec![],
            time_limit: Some(600), // 10 minutes in seconds
            passing_score: 0.7,
            max_attempts: Some(3),
            feedback_mode: crate::tutorial_system::FeedbackMode::Immediate,
        }),
        metadata: TutorialMetadata {
            author: "sklears Team".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: "1.0.0".to_string(),
            tags: vec!["getting-started".to_string(), "beginner".to_string()],
            category: TutorialCategory::GettingStarted,
            language: "en".to_string(),
            popularity_score: 5.0,
        },
    }
}

/// Create the data loading tutorial
fn create_data_loading_tutorial() -> Tutorial {
    Tutorial {
        id: "data_loading".to_string(),
        title: "Loading and Preparing Data".to_string(),
        description: "Learn how to load data from various sources and prepare it for training"
            .to_string(),
        difficulty: DifficultyLevel::Beginner,
        duration_minutes: 45,
        prerequisites: vec!["getting_started".to_string()],
        learning_objectives: vec![
            "Load data from CSV files".to_string(),
            "Use built-in datasets".to_string(),
            "Create synthetic data for testing".to_string(),
            "Handle missing values".to_string(),
        ],
        sections: vec![TutorialSection {
            id: "builtin_datasets".to_string(),
            title: "Built-in Datasets".to_string(),
            content: SectionContent::Text {
                content: r#"
# Built-in Datasets

sklears provides several built-in datasets for learning and testing:

```rust
use sklears::prelude::*;

fn main() -> Result<()> {
    // Load the classic iris dataset
    let iris = load_iris()?;
    println!("Features shape: {:?}", iris.features.shape());
    println!("Number of classes: {}", iris.target_names.len());

    // Create synthetic regression data
    let (X, y) = make_regression()
        .n_samples(100)
        .n_features(5)
        .noise(0.1)
        .generate()?;

    // Create synthetic classification data
    let (X_class, y_class) = make_blobs()
        .n_samples(200)
        .n_features(2)
        .centers(3)
        .generate()?;

    Ok(())
}
```

## Available Datasets

- `load_iris()`: Classic iris flower dataset
- `load_diabetes()`: Diabetes regression dataset
- `make_regression()`: Generate regression data
- `make_blobs()`: Generate clustered data
- `make_classification()`: Generate classification data
"#
                .to_string(),
                format: crate::tutorial_system::ContentFormat::Markdown,
            },
            interactive_elements: vec![],
            estimated_duration: 15,
            completion_criteria: crate::tutorial_system::CompletionCriteria {
                required_interactions: vec![],
                minimum_score: None,
                time_spent_minimum: Some(60), // At least 1 minute
                code_execution_required: false,
            },
        }],
        assessment: None,
        metadata: TutorialMetadata {
            author: "sklears Team".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: "1.0.0".to_string(),
            tags: vec!["data".to_string(), "preprocessing".to_string()],
            category: TutorialCategory::CoreConcepts,
            language: "en".to_string(),
            popularity_score: 4.5,
        },
    }
}

/// Create basic regression tutorial
fn create_basic_regression_tutorial() -> Tutorial {
    Tutorial {
        id: "basic_regression".to_string(),
        title: "Regression Models".to_string(),
        description: "Master regression techniques from linear models to advanced methods"
            .to_string(),
        difficulty: DifficultyLevel::Beginner,
        duration_minutes: 60,
        prerequisites: vec!["getting_started".to_string(), "data_loading".to_string()],
        learning_objectives: vec![
            "Understand regression fundamentals".to_string(),
            "Apply linear regression".to_string(),
            "Use regularization techniques".to_string(),
            "Evaluate regression models".to_string(),
        ],
        sections: vec![],
        assessment: None,
        metadata: TutorialMetadata {
            author: "sklears Team".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: "1.0.0".to_string(),
            tags: vec!["regression".to_string(), "supervised".to_string()],
            category: TutorialCategory::CoreConcepts,
            language: "en".to_string(),
            popularity_score: 4.8,
        },
    }
}

/// Create basic classification tutorial
fn create_basic_classification_tutorial() -> Tutorial {
    Tutorial {
        id: "basic_classification".to_string(),
        title: "Classification Models".to_string(),
        description: "Learn classification from logistic regression to advanced classifiers"
            .to_string(),
        difficulty: DifficultyLevel::Beginner,
        duration_minutes: 60,
        prerequisites: vec!["getting_started".to_string(), "data_loading".to_string()],
        learning_objectives: vec![
            "Understand classification fundamentals".to_string(),
            "Apply logistic regression".to_string(),
            "Use decision trees".to_string(),
            "Evaluate classifiers with metrics".to_string(),
        ],
        sections: vec![],
        assessment: None,
        metadata: TutorialMetadata {
            author: "sklears Team".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: "1.0.0".to_string(),
            tags: vec!["classification".to_string(), "supervised".to_string()],
            category: TutorialCategory::CoreConcepts,
            language: "en".to_string(),
            popularity_score: 4.9,
        },
    }
}

/// Create preprocessing tutorial
fn create_preprocessing_tutorial() -> Tutorial {
    Tutorial {
        id: "preprocessing".to_string(),
        title: "Data Preprocessing".to_string(),
        description: "Learn essential data preprocessing techniques".to_string(),
        difficulty: DifficultyLevel::Beginner,
        duration_minutes: 45,
        prerequisites: vec!["data_loading".to_string()],
        learning_objectives: vec![
            "Normalize and standardize data".to_string(),
            "Handle missing values".to_string(),
            "Encode categorical variables".to_string(),
            "Scale features appropriately".to_string(),
        ],
        sections: vec![],
        assessment: None,
        metadata: TutorialMetadata {
            author: "sklears Team".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: "1.0.0".to_string(),
            tags: vec!["preprocessing".to_string(), "data".to_string()],
            category: TutorialCategory::CoreConcepts,
            language: "en".to_string(),
            popularity_score: 4.6,
        },
    }
}

// Intermediate tutorials
fn create_pipeline_tutorial() -> Tutorial {
    Tutorial {
        id: "pipelines".to_string(),
        title: "Building ML Pipelines".to_string(),
        description: "Create efficient ML pipelines for complex workflows".to_string(),
        difficulty: DifficultyLevel::Intermediate,
        duration_minutes: 75,
        prerequisites: vec!["preprocessing".to_string()],
        learning_objectives: vec![
            "Understand pipeline architecture".to_string(),
            "Chain transformers and estimators".to_string(),
            "Build complex workflows".to_string(),
            "Optimize pipeline performance".to_string(),
        ],
        sections: vec![],
        assessment: None,
        metadata: TutorialMetadata {
            author: "sklears Team".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: "1.0.0".to_string(),
            tags: vec!["pipelines".to_string(), "workflows".to_string()],
            category: TutorialCategory::AdvancedFeatures,
            language: "en".to_string(),
            popularity_score: 4.7,
        },
    }
}

fn create_cross_validation_tutorial() -> Tutorial {
    Tutorial {
        id: "cross_validation".to_string(),
        title: "Model Validation Techniques".to_string(),
        description: "Master cross-validation and model evaluation".to_string(),
        difficulty: DifficultyLevel::Intermediate,
        duration_minutes: 60,
        prerequisites: vec!["basic_regression".to_string()],
        learning_objectives: vec![],
        sections: vec![],
        assessment: None,
        metadata: TutorialMetadata {
            author: "sklears Team".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: "1.0.0".to_string(),
            tags: vec!["validation".to_string(), "evaluation".to_string()],
            category: TutorialCategory::BestPractices,
            language: "en".to_string(),
            popularity_score: 4.5,
        },
    }
}

fn create_hyperparameter_tuning_tutorial() -> Tutorial {
    Tutorial {
        id: "hyperparameter_tuning".to_string(),
        title: "Hyperparameter Optimization".to_string(),
        description: "Optimize model performance with advanced tuning techniques".to_string(),
        difficulty: DifficultyLevel::Intermediate,
        duration_minutes: 90,
        prerequisites: vec!["cross_validation".to_string()],
        learning_objectives: vec![],
        sections: vec![],
        assessment: None,
        metadata: TutorialMetadata {
            author: "sklears Team".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: "1.0.0".to_string(),
            tags: vec!["optimization".to_string(), "tuning".to_string()],
            category: TutorialCategory::AdvancedFeatures,
            language: "en".to_string(),
            popularity_score: 4.8,
        },
    }
}

fn create_ensemble_methods_tutorial() -> Tutorial {
    Tutorial {
        id: "ensemble_methods".to_string(),
        title: "Ensemble Learning".to_string(),
        description: "Combine multiple models for better predictions".to_string(),
        difficulty: DifficultyLevel::Intermediate,
        duration_minutes: 80,
        prerequisites: vec!["basic_classification".to_string()],
        learning_objectives: vec![],
        sections: vec![],
        assessment: None,
        metadata: TutorialMetadata {
            author: "sklears Team".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: "1.0.0".to_string(),
            tags: vec!["ensemble".to_string(), "advanced".to_string()],
            category: TutorialCategory::AdvancedFeatures,
            language: "en".to_string(),
            popularity_score: 4.7,
        },
    }
}

fn create_feature_engineering_tutorial() -> Tutorial {
    Tutorial {
        id: "feature_engineering".to_string(),
        title: "Feature Engineering".to_string(),
        description: "Create powerful features for better model performance".to_string(),
        difficulty: DifficultyLevel::Intermediate,
        duration_minutes: 75,
        prerequisites: vec!["preprocessing".to_string()],
        learning_objectives: vec![],
        sections: vec![],
        assessment: None,
        metadata: TutorialMetadata {
            author: "sklears Team".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: "1.0.0".to_string(),
            tags: vec!["features".to_string(), "engineering".to_string()],
            category: TutorialCategory::BestPractices,
            language: "en".to_string(),
            popularity_score: 4.9,
        },
    }
}

// Advanced tutorials
fn create_custom_estimator_tutorial() -> Tutorial {
    Tutorial {
        id: "custom_estimator".to_string(),
        title: "Building Custom Estimators".to_string(),
        description: "Create your own estimators using sklears traits".to_string(),
        difficulty: DifficultyLevel::Advanced,
        duration_minutes: 120,
        prerequisites: vec!["pipelines".to_string()],
        learning_objectives: vec![],
        sections: vec![],
        assessment: None,
        metadata: TutorialMetadata {
            author: "sklears Team".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: "1.0.0".to_string(),
            tags: vec!["custom".to_string(), "advanced".to_string()],
            category: TutorialCategory::AdvancedFeatures,
            language: "en".to_string(),
            popularity_score: 4.4,
        },
    }
}

fn create_distributed_computing_tutorial() -> Tutorial {
    Tutorial {
        id: "distributed_computing".to_string(),
        title: "Distributed Machine Learning".to_string(),
        description: "Scale your ML workloads across multiple machines".to_string(),
        difficulty: DifficultyLevel::Advanced,
        duration_minutes: 150,
        prerequisites: vec!["pipelines".to_string()],
        learning_objectives: vec![],
        sections: vec![],
        assessment: None,
        metadata: TutorialMetadata {
            author: "sklears Team".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: "1.0.0".to_string(),
            tags: vec!["distributed".to_string(), "scaling".to_string()],
            category: TutorialCategory::Performance,
            language: "en".to_string(),
            popularity_score: 4.3,
        },
    }
}

fn create_performance_optimization_tutorial() -> Tutorial {
    Tutorial {
        id: "performance_optimization".to_string(),
        title: "Performance Optimization".to_string(),
        description: "Squeeze maximum performance from your ML code".to_string(),
        difficulty: DifficultyLevel::Advanced,
        duration_minutes: 100,
        prerequisites: vec!["pipelines".to_string()],
        learning_objectives: vec![],
        sections: vec![],
        assessment: None,
        metadata: TutorialMetadata {
            author: "sklears Team".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: "1.0.0".to_string(),
            tags: vec!["performance".to_string(), "optimization".to_string()],
            category: TutorialCategory::Performance,
            language: "en".to_string(),
            popularity_score: 4.6,
        },
    }
}

fn create_gpu_acceleration_tutorial() -> Tutorial {
    Tutorial {
        id: "gpu_acceleration".to_string(),
        title: "GPU Acceleration".to_string(),
        description: "Leverage GPU power for faster training".to_string(),
        difficulty: DifficultyLevel::Advanced,
        duration_minutes: 90,
        prerequisites: vec!["performance_optimization".to_string()],
        learning_objectives: vec![],
        sections: vec![],
        assessment: None,
        metadata: TutorialMetadata {
            author: "sklears Team".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: "1.0.0".to_string(),
            tags: vec!["gpu".to_string(), "acceleration".to_string()],
            category: TutorialCategory::Performance,
            language: "en".to_string(),
            popularity_score: 4.7,
        },
    }
}

fn create_type_safety_tutorial() -> Tutorial {
    Tutorial {
        id: "type_safety".to_string(),
        title: "Advanced Type Safety".to_string(),
        description: "Leverage Rust's type system for safer ML code".to_string(),
        difficulty: DifficultyLevel::Advanced,
        duration_minutes: 110,
        prerequisites: vec!["custom_estimator".to_string()],
        learning_objectives: vec![],
        sections: vec![],
        assessment: None,
        metadata: TutorialMetadata {
            author: "sklears Team".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: "1.0.0".to_string(),
            tags: vec!["types".to_string(), "safety".to_string()],
            category: TutorialCategory::BestPractices,
            language: "en".to_string(),
            popularity_score: 4.2,
        },
    }
}

/// Create a complete tutorial system with all tutorials and learning paths
pub fn create_complete_tutorial_system() -> TutorialSystem {
    let mut all_tutorials = Vec::new();
    all_tutorials.extend(generate_beginner_tutorials());
    all_tutorials.extend(generate_intermediate_tutorials());
    all_tutorials.extend(generate_advanced_tutorials());

    let learning_paths = vec![
        create_beginner_learning_path(),
        create_intermediate_learning_path(),
        create_advanced_learning_path(),
    ];

    TutorialSystem {
        tutorials: all_tutorials,
        learning_paths,
        progress_tracker: crate::tutorial_system::ProgressTracker {
            user_progress: HashMap::new(),
            global_statistics: crate::tutorial_system::GlobalStatistics {
                total_users: 0,
                tutorial_completion_rates: HashMap::new(),
                average_scores: HashMap::new(),
                popular_tutorials: vec![],
                common_challenges: vec![],
            },
        },
        assessment_engine: crate::tutorial_system::AssessmentEngine {
            assessments: HashMap::new(),
            question_bank: vec![],
            scoring_algorithms: HashMap::new(),
        },
        config: crate::tutorial_system::TutorialConfig::default(),
    }
}

/// Create intermediate learning path
fn create_intermediate_learning_path() -> LearningPath {
    LearningPath {
        id: "intermediate_path".to_string(),
        title: "Intermediate Machine Learning".to_string(),
        description: "Advanced techniques and best practices".to_string(),
        difficulty: DifficultyLevel::Intermediate,
        estimated_hours: 20,
        tutorial_sequence: generate_intermediate_tutorials()
            .iter()
            .map(|t| t.id.clone())
            .collect(),
        prerequisites: vec!["beginner_path".to_string()],
        completion_rewards: vec![
            "Build complex ML pipelines".to_string(),
            "Optimize model performance".to_string(),
            "Apply ensemble methods".to_string(),
        ],
    }
}

/// Create advanced learning path
fn create_advanced_learning_path() -> LearningPath {
    LearningPath {
        id: "advanced_path".to_string(),
        title: "Advanced sklears Mastery".to_string(),
        description: "Master advanced features and optimization".to_string(),
        difficulty: DifficultyLevel::Advanced,
        estimated_hours: 35,
        tutorial_sequence: generate_advanced_tutorials()
            .iter()
            .map(|t| t.id.clone())
            .collect(),
        prerequisites: vec!["intermediate_path".to_string()],
        completion_rewards: vec![
            "Create custom estimators".to_string(),
            "Optimize performance at scale".to_string(),
            "Leverage advanced type safety".to_string(),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beginner_tutorials() {
        let tutorials = generate_beginner_tutorials();
        assert_eq!(tutorials.len(), 5);
        assert_eq!(tutorials[0].id, "getting_started");
    }

    #[test]
    fn test_intermediate_tutorials() {
        let tutorials = generate_intermediate_tutorials();
        assert_eq!(tutorials.len(), 5);
        assert!(tutorials
            .iter()
            .all(|t| matches!(t.difficulty, DifficultyLevel::Intermediate)));
    }

    #[test]
    fn test_advanced_tutorials() {
        let tutorials = generate_advanced_tutorials();
        assert_eq!(tutorials.len(), 5);
        assert!(tutorials
            .iter()
            .all(|t| matches!(t.difficulty, DifficultyLevel::Advanced)));
    }

    #[test]
    fn test_complete_system() {
        let system = create_complete_tutorial_system();
        assert_eq!(system.tutorials.len(), 15);
        assert_eq!(system.learning_paths.len(), 3);
    }

    #[test]
    fn test_learning_path_structure() {
        let path = create_beginner_learning_path();
        assert_eq!(path.id, "beginner_path");
        assert_eq!(path.tutorial_sequence.len(), 5);
        assert!(path.prerequisites.is_empty());
    }

    #[test]
    fn test_tutorial_metadata() {
        let tutorial = create_getting_started_tutorial();
        assert_eq!(tutorial.metadata.author, "sklears Team");
        assert!(tutorial
            .metadata
            .tags
            .contains(&"getting-started".to_string()));
    }
}
