use crate::{UtilsError, UtilsResult};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MultiClassStrategy {
    OneVsRest,
    OneVsOne,
}

pub struct OneVsRestClassifier<C> {
    pub estimators: Vec<C>,
    pub classes: Vec<i32>,
    pub strategy: MultiClassStrategy,
}

impl<C> Default for OneVsRestClassifier<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C> OneVsRestClassifier<C> {
    pub fn new() -> Self {
        Self {
            estimators: Vec::new(),
            classes: Vec::new(),
            strategy: MultiClassStrategy::OneVsRest,
        }
    }
}

pub fn type_of_target(y: &Array1<i32>) -> UtilsResult<String> {
    if y.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    let unique_values: HashSet<i32> = y.iter().copied().collect();
    let n_unique = unique_values.len();

    if n_unique == 1 {
        Ok("unknown".to_string())
    } else if n_unique == 2 {
        Ok("binary".to_string())
    } else {
        Ok("multiclass".to_string())
    }
}

pub fn check_classification_targets(y: &Array1<i32>) -> UtilsResult<()> {
    let target_type = type_of_target(y)?;

    match target_type.as_str() {
        "binary" | "multiclass" => Ok(()),
        "unknown" => Err(UtilsError::InvalidParameter(
            "Unknown label type: all samples have the same label".to_string(),
        )),
        _ => Err(UtilsError::InvalidParameter(format!(
            "Unknown target type: {target_type}"
        ))),
    }
}

pub fn unique_labels_multiclass(y: &Array1<i32>) -> Vec<i32> {
    let mut unique: Vec<i32> = y.iter().copied().collect();
    unique.sort();
    unique.dedup();
    unique
}

pub fn class_distribution(y: &Array1<i32>) -> HashMap<i32, usize> {
    let mut counts = HashMap::new();
    for &label in y.iter() {
        *counts.entry(label).or_insert(0) += 1;
    }
    counts
}

pub fn check_multi_class(y: &Array1<i32>) -> UtilsResult<bool> {
    let unique_labels = unique_labels_multiclass(y);

    if unique_labels.len() < 2 {
        Err(UtilsError::InvalidParameter(
            "Need at least 2 classes for classification".to_string(),
        ))
    } else {
        Ok(unique_labels.len() > 2)
    }
}

pub fn one_vs_rest_transform(y: &Array1<i32>, positive_class: i32) -> Array1<i32> {
    y.mapv(|label| if label == positive_class { 1 } else { 0 })
}

pub fn one_vs_one_pairs(classes: &[i32]) -> Vec<(i32, i32)> {
    let mut pairs = Vec::new();

    for (i, &class_a) in classes.iter().enumerate() {
        for &class_b in classes.iter().skip(i + 1) {
            pairs.push((class_a, class_b));
        }
    }

    pairs
}

pub fn one_vs_one_transform(
    y: &Array1<i32>,
    class_a: i32,
    class_b: i32,
) -> (Array1<i32>, Vec<usize>) {
    let mut new_y = Vec::new();
    let mut indices = Vec::new();

    for (i, &label) in y.iter().enumerate() {
        if label == class_a || label == class_b {
            new_y.push(if label == class_a { 0 } else { 1 });
            indices.push(i);
        }
    }

    (Array1::from_vec(new_y), indices)
}

pub fn is_multilabel(y: &Array2<i32>) -> bool {
    y.ncols() > 1
}

pub fn multilabel_to_indicator(y: &Array1<i32>, classes: &[i32]) -> UtilsResult<Array2<i32>> {
    let n_samples = y.len();
    let n_classes = classes.len();
    let mut indicator = Array2::zeros((n_samples, n_classes));

    for (i, &label) in y.iter().enumerate() {
        if let Some(class_idx) = classes.iter().position(|&c| c == label) {
            indicator[[i, class_idx]] = 1;
        } else {
            return Err(UtilsError::InvalidParameter(format!(
                "Label {label} not found in classes"
            )));
        }
    }

    Ok(indicator)
}

pub fn indicator_to_multilabel(
    indicator: &Array2<i32>,
    classes: &[i32],
) -> UtilsResult<Vec<Vec<i32>>> {
    if indicator.ncols() != classes.len() {
        return Err(UtilsError::ShapeMismatch {
            expected: vec![indicator.nrows(), classes.len()],
            actual: vec![indicator.nrows(), indicator.ncols()],
        });
    }

    let mut result = Vec::new();

    for row in indicator.axis_iter(Axis(0)) {
        let mut labels = Vec::new();
        for (j, &value) in row.iter().enumerate() {
            if value == 1 {
                labels.push(classes[j]);
            }
        }
        result.push(labels);
    }

    Ok(result)
}

pub fn check_binary_indicators_multioutput(y: &Array2<i32>) -> UtilsResult<()> {
    for value in y.iter() {
        if *value != 0 && *value != 1 {
            return Err(UtilsError::InvalidParameter(
                "Binary indicators must contain only 0 and 1".to_string(),
            ));
        }
    }
    Ok(())
}

pub fn compute_class_weight_balanced(y: &Array1<i32>) -> HashMap<i32, f64> {
    let class_counts = class_distribution(y);
    let n_samples = y.len() as f64;
    let n_classes = class_counts.len() as f64;

    let mut weights = HashMap::new();
    for (&class, &count) in &class_counts {
        weights.insert(class, n_samples / (n_classes * count as f64));
    }

    weights
}

pub fn compute_sample_weight(y: &Array1<i32>, class_weight: &HashMap<i32, f64>) -> Array1<f64> {
    let mut sample_weights = Array1::zeros(y.len());

    for (i, &label) in y.iter().enumerate() {
        sample_weights[i] = class_weight.get(&label).copied().unwrap_or(1.0);
    }

    sample_weights
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_type_of_target() {
        let binary = array![0, 1, 0, 1];
        assert_eq!(type_of_target(&binary).unwrap(), "binary");

        let multiclass = array![0, 1, 2, 0, 1, 2];
        assert_eq!(type_of_target(&multiclass).unwrap(), "multiclass");

        let constant = array![1, 1, 1, 1];
        assert_eq!(type_of_target(&constant).unwrap(), "unknown");
    }

    #[test]
    fn test_check_multi_class() {
        let binary = array![0, 1, 0, 1];
        assert!(!check_multi_class(&binary).unwrap());

        let multiclass = array![0, 1, 2, 0, 1, 2];
        assert!(check_multi_class(&multiclass).unwrap());
    }

    #[test]
    fn test_one_vs_rest_transform() {
        let y = array![0, 1, 2, 0, 1, 2];
        let binary_y = one_vs_rest_transform(&y, 1);
        assert_eq!(binary_y, array![0, 1, 0, 0, 1, 0]);
    }

    #[test]
    fn test_one_vs_one_pairs() {
        let classes = vec![0, 1, 2];
        let pairs = one_vs_one_pairs(&classes);
        assert_eq!(pairs, vec![(0, 1), (0, 2), (1, 2)]);
    }

    #[test]
    fn test_one_vs_one_transform() {
        let y = array![0, 1, 2, 0, 1, 2];
        let (binary_y, indices) = one_vs_one_transform(&y, 0, 2);
        assert_eq!(binary_y, array![0, 1, 0, 1]);
        assert_eq!(indices, vec![0, 2, 3, 5]);
    }

    #[test]
    fn test_multilabel_to_indicator() {
        let y = array![0, 1, 2];
        let classes = vec![0, 1, 2];
        let indicator = multilabel_to_indicator(&y, &classes).unwrap();

        let expected = Array2::from_shape_vec((3, 3), vec![1, 0, 0, 0, 1, 0, 0, 0, 1]).unwrap();
        assert_eq!(indicator, expected);
    }

    #[test]
    fn test_compute_class_weight_balanced() {
        let y = array![0, 0, 1, 1, 1, 2]; // Class distribution: 0->2, 1->3, 2->1
        let weights = compute_class_weight_balanced(&y);

        // Expected: n_samples / (n_classes * class_count) = 6 / (3 * count)
        assert!((weights[&0] - 1.0).abs() < 1e-10); // 6 / (3 * 2) = 1.0
        assert!((weights[&1] - 2.0 / 3.0).abs() < 1e-10); // 6 / (3 * 3) = 2/3
        assert!((weights[&2] - 2.0).abs() < 1e-10); // 6 / (3 * 1) = 2.0
    }

    #[test]
    fn test_class_distribution() {
        let y = array![0, 1, 0, 2, 1, 1];
        let dist = class_distribution(&y);

        assert_eq!(dist[&0], 2);
        assert_eq!(dist[&1], 3);
        assert_eq!(dist[&2], 1);
    }
}
