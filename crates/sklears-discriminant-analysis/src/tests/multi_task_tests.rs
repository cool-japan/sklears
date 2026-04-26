//! Multi-task discriminant learning tests

use super::super::*;
use approx::assert_abs_diff_eq;
use scirs2_core::ndarray::{array, Axis};
use sklears_core::traits::Fit;
use sklears_core::types::Float;

#[test]
fn test_task_creation() {
    let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
    let y = array![0, 0, 1, 1];

    let task = Task::new(0, x.clone(), y.clone()).expect("construction should succeed");

    assert_eq!(task.task_id, 0);
    assert_eq!(task.n_samples(), 4);
    assert_eq!(task.n_features(), 2);
    assert_eq!(task.n_classes(), 2);
    assert_eq!(task.classes, vec![0, 1]);
}

#[test]
fn test_multi_task_discriminant_learning_basic() {
    // Create two related tasks
    let task1_x = array![
        [1.0, 2.0],
        [1.1, 2.1],
        [1.2, 2.2], // Class 0
        [3.0, 4.0],
        [3.1, 4.1],
        [3.2, 4.2] // Class 1
    ];
    let task1_y = array![0, 0, 0, 1, 1, 1];

    let task2_x = array![
        [1.5, 2.5],
        [1.6, 2.6],
        [1.7, 2.7], // Class 0
        [3.5, 4.5],
        [3.6, 4.6],
        [3.7, 4.7] // Class 1
    ];
    let task2_y = array![0, 0, 0, 1, 1, 1];

    let task1 =
        Task::new(0, task1_x.clone(), task1_y.clone()).expect("construction should succeed");
    let task2 =
        Task::new(1, task2_x.clone(), task2_y.clone()).expect("construction should succeed");
    let tasks = vec![task1, task2];

    let mtdl = MultiTaskDiscriminantLearning::new();
    let fitted = mtdl.fit(&tasks, &()).expect("model fitting should succeed");

    // Test prediction for task 1
    let predictions = fitted
        .predict_task(&task1_x, 0)
        .expect("operation should succeed");
    assert_eq!(predictions.len(), 6);

    // Test prediction for task 2
    let predictions = fitted
        .predict_task(&task2_x, 1)
        .expect("operation should succeed");
    assert_eq!(predictions.len(), 6);
}

#[test]
fn test_multi_task_predict_proba() {
    let task1_x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
    let task1_y = array![0, 0, 1, 1];

    let task2_x = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]];
    let task2_y = array![0, 0, 1, 1];

    let task1 =
        Task::new(0, task1_x.clone(), task1_y.clone()).expect("construction should succeed");
    let task2 =
        Task::new(1, task2_x.clone(), task2_y.clone()).expect("construction should succeed");
    let tasks = vec![task1, task2];

    let mtdl = MultiTaskDiscriminantLearning::new();
    let fitted = mtdl.fit(&tasks, &()).expect("model fitting should succeed");

    let probas = fitted
        .predict_proba_task(&task1_x, 0)
        .expect("operation should succeed");
    assert_eq!(probas.dim(), (4, 2));

    for row in probas.axis_iter(Axis(0)) {
        let sum: Float = row.sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
    }
}

#[test]
fn test_multi_task_transform() {
    let task1_x = array![
        [1.0, 2.0, 0.5],
        [2.0, 1.0, 1.5],
        [3.0, 4.0, 2.0],
        [4.0, 3.0, 3.5],
        [5.0, 2.0, 4.0],
        [6.0, 1.0, 4.5]
    ];
    let task1_y = array![0, 0, 1, 1, 2, 2];

    let task2_x = array![
        [1.5, 2.5, 0.8],
        [2.5, 1.5, 1.8],
        [3.5, 4.5, 2.5],
        [4.5, 3.5, 3.8],
        [5.5, 2.5, 4.3],
        [6.5, 1.5, 4.8]
    ];
    let task2_y = array![0, 0, 1, 1, 2, 2];

    let task1 =
        Task::new(0, task1_x.clone(), task1_y.clone()).expect("construction should succeed");
    let task2 =
        Task::new(1, task2_x.clone(), task2_y.clone()).expect("construction should succeed");
    let tasks = vec![task1, task2];

    let mtdl = MultiTaskDiscriminantLearning::new()
        .n_shared_components(Some(2))
        .n_task_components(Some(1));
    let fitted = mtdl.fit(&tasks, &()).expect("model fitting should succeed");

    // Test shared transformation
    let shared_transformed = fitted
        .transform_shared(&task1_x)
        .expect("operation should succeed");
    assert!(shared_transformed.ncols() >= 1); // Ensure we get some components

    // Test task-specific transformation
    let task_transformed = fitted
        .transform_task(&task1_x, 0)
        .expect("operation should succeed");
    assert!(task_transformed.ncols() >= shared_transformed.ncols()); // Should include shared + task components
}

#[test]
fn test_multi_task_with_qda() {
    let task1_x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
    let task1_y = array![0, 0, 1, 1];

    let task2_x = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]];
    let task2_y = array![0, 0, 1, 1];

    let task1 =
        Task::new(0, task1_x.clone(), task1_y.clone()).expect("construction should succeed");
    let task2 =
        Task::new(1, task2_x.clone(), task2_y.clone()).expect("construction should succeed");
    let tasks = vec![task1, task2];

    let mtdl = MultiTaskDiscriminantLearning::new().base_discriminant("qda");
    let fitted = mtdl.fit(&tasks, &()).expect("model fitting should succeed");

    let predictions = fitted
        .predict_task(&task1_x, 0)
        .expect("operation should succeed");
    assert_eq!(predictions.len(), 4);
}

#[test]
fn test_task_weighting_strategies() {
    let task1_x = array![[1.0, 2.0], [2.0, 3.0]];
    let task1_y = array![0, 1];

    let task2_x = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]];
    let task2_y = array![0, 0, 1, 1];

    let task1 =
        Task::new(0, task1_x.clone(), task1_y.clone()).expect("construction should succeed");
    let task2 =
        Task::new(1, task2_x.clone(), task2_y.clone()).expect("construction should succeed");
    let tasks = vec![task1, task2];

    let strategies = ["uniform", "proportional", "inverse"];
    for strategy in &strategies {
        let mtdl = MultiTaskDiscriminantLearning::new().task_weighting(strategy);
        let fitted = mtdl.fit(&tasks, &()).expect("model fitting should succeed");

        assert_eq!(fitted.task_weights().len(), 2);
        assert!(fitted.task_weights().iter().all(|&w| w > 0.0));
    }
}

#[test]
fn test_add_new_task() {
    let task1_x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
    let task1_y = array![0, 0, 1, 1];

    let task2_x = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]];
    let task2_y = array![0, 0, 1, 1];

    let task1 =
        Task::new(0, task1_x.clone(), task1_y.clone()).expect("construction should succeed");
    let task2 =
        Task::new(1, task2_x.clone(), task2_y.clone()).expect("construction should succeed");
    let tasks = vec![task1, task2];

    let mtdl = MultiTaskDiscriminantLearning::new();
    let mut fitted = mtdl.fit(&tasks, &()).expect("model fitting should succeed");

    // Add a new task
    let task3_x = array![[2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]];
    let task3_y = array![0, 0, 1, 1];
    let task3 =
        Task::new(2, task3_x.clone(), task3_y.clone()).expect("construction should succeed");

    let new_task_id = fitted.add_task(task3).expect("operation should succeed");
    assert_eq!(new_task_id, 2);

    // Test prediction for new task
    let predictions = fitted
        .predict_task(&task3_x, new_task_id)
        .expect("operation should succeed");
    assert_eq!(predictions.len(), 4);
}
