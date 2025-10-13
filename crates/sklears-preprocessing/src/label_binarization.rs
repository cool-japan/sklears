//! Label Binarization transformers
//!
//! This module provides transformers for binarizing labels:
//! - LabelBinarizer: One-hot encoding for single-label classification
//! - MultiLabelBinarizer: Binary encoding for multi-label classification

use scirs2_core::ndarray::{Array1, Array2};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::marker::PhantomData;

use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};

/// Configuration for LabelBinarizer
#[derive(Debug, Clone)]
pub struct LabelBinarizerConfig {
    /// Value for negative class in binary classification
    pub neg_label: i32,
    /// Value for positive class in binary classification
    pub pos_label: i32,
    /// Whether to use sparse output (not implemented)
    pub sparse_output: bool,
}

impl Default for LabelBinarizerConfig {
    fn default() -> Self {
        Self {
            neg_label: 0,
            pos_label: 1,
            sparse_output: false,
        }
    }
}

/// LabelBinarizer transforms labels to binary form
pub struct LabelBinarizer<T: Eq + Hash + Clone = i32, State = Untrained> {
    config: LabelBinarizerConfig,
    state: PhantomData<State>,
    classes_: Option<Vec<T>>,
    class_to_index_: Option<HashMap<T, usize>>,
}

impl<T: Eq + Hash + Clone> LabelBinarizer<T, Untrained> {
    /// Create a new LabelBinarizer with default configuration
    pub fn new() -> Self {
        Self {
            config: LabelBinarizerConfig::default(),
            state: PhantomData,
            classes_: None,
            class_to_index_: None,
        }
    }

    /// Set the negative label value
    pub fn neg_label(mut self, neg_label: i32) -> Self {
        self.config.neg_label = neg_label;
        self
    }

    /// Set the positive label value
    pub fn pos_label(mut self, pos_label: i32) -> Self {
        self.config.pos_label = pos_label;
        self
    }
}

impl<T: Eq + Hash + Clone> Default for LabelBinarizer<T, Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Eq + Hash + Clone> Estimator for LabelBinarizer<T, Untrained> {
    type Config = LabelBinarizerConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl<T: Eq + Hash + Clone> Estimator for LabelBinarizer<T, Trained> {
    type Config = LabelBinarizerConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl<T: Eq + Hash + Clone + Ord + Send + Sync> Fit<Array1<T>, ()> for LabelBinarizer<T, Untrained> {
    type Fitted = LabelBinarizer<T, Trained>;

    fn fit(self, y: &Array1<T>, _x: &()) -> Result<Self::Fitted> {
        // Extract unique classes
        let mut classes = HashSet::new();
        for label in y.iter() {
            classes.insert(label.clone());
        }

        // Sort classes for consistency
        let mut sorted_classes: Vec<T> = classes.into_iter().collect();
        sorted_classes.sort();

        // Create class to index mapping
        let class_to_index: HashMap<T, usize> = sorted_classes
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), i))
            .collect();

        Ok(LabelBinarizer {
            config: self.config,
            state: PhantomData,
            classes_: Some(sorted_classes),
            class_to_index_: Some(class_to_index),
        })
    }
}

impl<T: Eq + Hash + Clone> Transform<Array1<T>, Array2<Float>> for LabelBinarizer<T, Trained> {
    fn transform(&self, y: &Array1<T>) -> Result<Array2<Float>> {
        let classes = self.classes_.as_ref().unwrap();
        let class_to_index = self.class_to_index_.as_ref().unwrap();
        let n_samples = y.len();
        let n_classes = classes.len();

        if n_classes == 0 {
            return Err(SklearsError::InvalidInput(
                "No classes found during fit".to_string(),
            ));
        }

        // Special case for binary classification
        if n_classes == 2 {
            let mut result = Array2::zeros((n_samples, 1));
            for (i, label) in y.iter().enumerate() {
                if let Some(&class_idx) = class_to_index.get(label) {
                    result[[i, 0]] = if class_idx == 1 {
                        self.config.pos_label as Float
                    } else {
                        self.config.neg_label as Float
                    };
                } else {
                    return Err(SklearsError::InvalidInput(
                        "Unknown label encountered during transform".to_string(),
                    ));
                }
            }
            Ok(result)
        } else {
            // Multi-class case: one-hot encoding
            let mut result =
                Array2::from_elem((n_samples, n_classes), self.config.neg_label as Float);
            for (i, label) in y.iter().enumerate() {
                if let Some(&class_idx) = class_to_index.get(label) {
                    result[[i, class_idx]] = self.config.pos_label as Float;
                } else {
                    return Err(SklearsError::InvalidInput(
                        "Unknown label encountered during transform".to_string(),
                    ));
                }
            }
            Ok(result)
        }
    }
}

impl<T: Eq + Hash + Clone> LabelBinarizer<T, Trained> {
    /// Get the classes
    pub fn classes(&self) -> &Vec<T> {
        self.classes_.as_ref().unwrap()
    }

    /// Transform binary matrix back to original labels
    pub fn inverse_transform(&self, y: &Array2<Float>) -> Result<Array1<T>> {
        let classes = self.classes_.as_ref().unwrap();
        let n_samples = y.nrows();
        let n_classes = classes.len();

        if n_classes == 2 && y.ncols() == 1 {
            // Binary case
            let mut result = Vec::with_capacity(n_samples);
            let threshold = (self.config.neg_label + self.config.pos_label) as Float / 2.0;

            for i in 0..n_samples {
                let class_idx = if y[[i, 0]] > threshold { 1 } else { 0 };
                result.push(classes[class_idx].clone());
            }
            Ok(Array1::from_vec(result))
        } else if y.ncols() == n_classes {
            // Multi-class case
            let mut result = Vec::with_capacity(n_samples);

            for i in 0..n_samples {
                // Find the column with the maximum value
                let row = y.row(i);
                let mut max_idx = 0;
                let mut max_val = row[0];

                for j in 1..n_classes {
                    if row[j] > max_val {
                        max_val = row[j];
                        max_idx = j;
                    }
                }

                result.push(classes[max_idx].clone());
            }
            Ok(Array1::from_vec(result))
        } else {
            Err(SklearsError::InvalidInput(format!(
                "Shape mismatch: y has {} columns but {} classes were expected",
                y.ncols(),
                n_classes
            )))
        }
    }
}

/// Configuration for MultiLabelBinarizer
#[derive(Debug, Clone, Default)]
pub struct MultiLabelBinarizerConfig {
    /// Classes to consider (None = infer from data)
    pub classes: Option<Vec<String>>,
    /// Whether to use sparse output (not implemented)
    pub sparse_output: bool,
}

/// MultiLabelBinarizer transforms between iterable of labels and binary matrix
pub struct MultiLabelBinarizer<State = Untrained> {
    config: MultiLabelBinarizerConfig,
    state: PhantomData<State>,
    classes_: Option<Vec<String>>,
    class_to_index_: Option<HashMap<String, usize>>,
}

impl MultiLabelBinarizer<Untrained> {
    /// Create a new MultiLabelBinarizer with default configuration
    pub fn new() -> Self {
        Self {
            config: MultiLabelBinarizerConfig::default(),
            state: PhantomData,
            classes_: None,
            class_to_index_: None,
        }
    }

    /// Set the classes to use
    pub fn classes(mut self, classes: Vec<String>) -> Self {
        self.config.classes = Some(classes);
        self
    }
}

impl Default for MultiLabelBinarizer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MultiLabelBinarizer<Untrained> {
    type Config = MultiLabelBinarizerConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for MultiLabelBinarizer<Trained> {
    type Config = MultiLabelBinarizerConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Vec<Vec<String>>, ()> for MultiLabelBinarizer<Untrained> {
    type Fitted = MultiLabelBinarizer<Trained>;

    fn fit(self, y: &Vec<Vec<String>>, _x: &()) -> Result<Self::Fitted> {
        let classes = if let Some(ref classes) = self.config.classes {
            classes.clone()
        } else {
            // Infer classes from data
            let mut unique_classes = HashSet::new();
            for labels in y.iter() {
                for label in labels.iter() {
                    unique_classes.insert(label.clone());
                }
            }

            let mut sorted_classes: Vec<String> = unique_classes.into_iter().collect();
            sorted_classes.sort();
            sorted_classes
        };

        // Create class to index mapping
        let class_to_index: HashMap<String, usize> = classes
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), i))
            .collect();

        Ok(MultiLabelBinarizer {
            config: self.config,
            state: PhantomData,
            classes_: Some(classes),
            class_to_index_: Some(class_to_index),
        })
    }
}

impl Transform<Vec<Vec<String>>, Array2<Float>> for MultiLabelBinarizer<Trained> {
    fn transform(&self, y: &Vec<Vec<String>>) -> Result<Array2<Float>> {
        let classes = self.classes_.as_ref().unwrap();
        let class_to_index = self.class_to_index_.as_ref().unwrap();
        let n_samples = y.len();
        let n_classes = classes.len();

        let mut result = Array2::zeros((n_samples, n_classes));

        for (i, labels) in y.iter().enumerate() {
            for label in labels.iter() {
                if let Some(&class_idx) = class_to_index.get(label) {
                    result[[i, class_idx]] = 1.0;
                }
                // Ignore unknown labels during transform
            }
        }

        Ok(result)
    }
}

impl MultiLabelBinarizer<Trained> {
    /// Get the classes
    pub fn classes(&self) -> &Vec<String> {
        self.classes_.as_ref().unwrap()
    }

    /// Transform binary matrix back to multi-label format
    pub fn inverse_transform(&self, y: &Array2<Float>) -> Result<Vec<Vec<String>>> {
        let classes = self.classes_.as_ref().unwrap();
        let n_samples = y.nrows();
        let n_classes = classes.len();

        if y.ncols() != n_classes {
            return Err(SklearsError::InvalidInput(format!(
                "Shape mismatch: y has {} columns but {} classes were expected",
                y.ncols(),
                n_classes
            )));
        }

        let mut result = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let mut labels = Vec::new();
            for j in 0..n_classes {
                if y[[i, j]] > 0.5 {
                    labels.push(classes[j].clone());
                }
            }
            result.push(labels);
        }

        Ok(result)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_label_binarizer_binary() {
        let y = array![1, 0, 1, 0, 1];

        let binarizer = LabelBinarizer::new().fit(&y, &()).unwrap();

        let y_bin = binarizer.transform(&y).unwrap();

        // Binary case: should have 1 column
        assert_eq!(y_bin.shape(), &[5, 1]);
        assert_eq!(y_bin[[0, 0]], 1.0);
        assert_eq!(y_bin[[1, 0]], 0.0);
        assert_eq!(y_bin[[2, 0]], 1.0);
    }

    #[test]
    fn test_label_binarizer_multiclass() {
        let y = array![0, 1, 2, 1, 0];

        let binarizer = LabelBinarizer::new().fit(&y, &()).unwrap();

        let y_bin = binarizer.transform(&y).unwrap();

        // Multiclass case: should have 3 columns
        assert_eq!(y_bin.shape(), &[5, 3]);
        // First sample is class 0
        assert_eq!(y_bin.row(0).to_vec(), vec![1.0, 0.0, 0.0]);
        // Second sample is class 1
        assert_eq!(y_bin.row(1).to_vec(), vec![0.0, 1.0, 0.0]);
        // Third sample is class 2
        assert_eq!(y_bin.row(2).to_vec(), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_label_binarizer_inverse_transform() {
        let y = array!["cat", "dog", "cat", "bird", "dog"];

        let binarizer = LabelBinarizer::new().fit(&y, &()).unwrap();

        let y_bin = binarizer.transform(&y).unwrap();
        let y_inv = binarizer.inverse_transform(&y_bin).unwrap();

        assert_eq!(y, y_inv);
    }

    #[test]
    fn test_label_binarizer_custom_labels() {
        let y = array![1, 0, 1, 0];

        let binarizer = LabelBinarizer::new()
            .neg_label(-1)
            .pos_label(1)
            .fit(&y, &())
            .unwrap();

        let y_bin = binarizer.transform(&y).unwrap();

        assert_eq!(y_bin[[0, 0]], 1.0); // positive class
        assert_eq!(y_bin[[1, 0]], -1.0); // negative class
    }

    #[test]
    fn test_multilabel_binarizer() {
        let y = vec![
            vec!["sci-fi".to_string(), "thriller".to_string()],
            vec!["comedy".to_string()],
            vec!["sci-fi".to_string(), "comedy".to_string()],
        ];

        let binarizer = MultiLabelBinarizer::new().fit(&y, &()).unwrap();

        let y_bin = binarizer.transform(&y).unwrap();

        // Should have 3 samples, 3 classes
        assert_eq!(y_bin.shape(), &[3, 3]);
        let classes = binarizer.classes();
        assert_eq!(classes.len(), 3);

        // First sample has sci-fi and thriller
        let row0_sum: Float = y_bin.row(0).sum();
        assert_eq!(row0_sum, 2.0);

        // Second sample has only comedy
        let row1_sum: Float = y_bin.row(1).sum();
        assert_eq!(row1_sum, 1.0);
    }

    #[test]
    fn test_multilabel_binarizer_inverse() {
        let y = vec![
            vec!["red".to_string(), "blue".to_string()],
            vec!["green".to_string()],
            vec!["red".to_string(), "green".to_string()],
        ];

        let binarizer = MultiLabelBinarizer::new().fit(&y, &()).unwrap();

        let y_bin = binarizer.transform(&y).unwrap();
        let y_inv = binarizer.inverse_transform(&y_bin).unwrap();

        // Check that we get back the same labels (order might differ)
        for (original, reconstructed) in y.iter().zip(y_inv.iter()) {
            let orig_set: HashSet<_> = original.iter().collect();
            let recon_set: HashSet<_> = reconstructed.iter().collect();
            assert_eq!(orig_set, recon_set);
        }
    }

    #[test]
    fn test_multilabel_binarizer_with_classes() {
        let y = vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["c".to_string()],
        ];

        let classes = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ];

        let binarizer = MultiLabelBinarizer::new()
            .classes(classes.clone())
            .fit(&y, &())
            .unwrap();

        let y_bin = binarizer.transform(&y).unwrap();

        // Should have 4 columns (including 'd' which wasn't in the data)
        assert_eq!(y_bin.shape(), &[2, 4]);
        assert_eq!(binarizer.classes(), &classes);
    }
}
