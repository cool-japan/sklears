# sklears-tree

[![Crates.io](https://img.shields.io/crates/v/sklears-tree.svg)](https://crates.io/crates/sklears-tree)
[![Documentation](https://docs.rs/sklears-tree/badge.svg)](https://docs.rs/sklears-tree)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

Tree-based algorithms for Rust with 5-20x performance improvements over scikit-learn. Features CHAID and oblique (hyperplane) splits, isolation forests, multi-output/multi-label trees, and incremental/streaming variants.

> **Latest release:** `0.2.0` (July 14, 2026). See the [workspace release notes](../../docs/releases/0.2.0.md) for highlights and upgrade guidance.

## Overview

`sklears-tree` provides tree-based ML algorithms:

- **Core Algorithms**: Decision Trees (with CHAID and oblique/hyperplane splits), Random Forest, Isolation Forest, Model Trees (leaf-model regression trees)
- **Advanced Methods**: Multi-output and multi-label trees (shared, chained, or independent strategies), Extended Isolation Forest
- **Interpretability**: TreeSHAP explanations, partial dependence plots
- **Performance**: Parallel training/prediction, compact and bit-packed in-memory tree representations, surrogate-split NaN-aware routing
- **Streaming**: Hoeffding trees, incremental decision trees, incremental random forests, online gradient boosting

## Quick Start

```rust
use sklears_tree::{DecisionTreeClassifier, RandomForestClassifier, MaxFeatures, SplitCriterion};
use sklears_core::traits::{Fit, Predict};
use scirs2_core::ndarray::array;

// Decision Tree
let tree = DecisionTreeClassifier::new()
    .max_depth(5)
    .min_samples_split(2)
    .criterion(SplitCriterion::Entropy);

// Random Forest with parallel training
let rf = RandomForestClassifier::new()
    .n_estimators(100)
    .max_features(MaxFeatures::Sqrt)
    .n_jobs(4);

// Train and predict
let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
let y = array![0, 1, 0];
let fitted = tree.fit(&x, &y)?;
let predictions = fitted.predict(&x)?;
```

## Advanced Features

### Oblique and CHAID Splits

```rust
use sklears_tree::DecisionTreeClassifier;

// Oblique splits: hyperplane combinations of features at each node
let oblique_tree = DecisionTreeClassifier::new()
    .oblique(4, true); // n_hyperplanes, use_ridge

// CHAID-style chi-squared/F-test based splitting
let chaid_tree = DecisionTreeClassifier::new()
    .chaid(0.05); // significance_level
```

`oblique` requires the crate's `oblique` feature (pulls in `scirs2-linalg`).

### Isolation Forest (Anomaly Detection)

```rust
use sklears_tree::IsolationForest;

let iso_forest = IsolationForest::new()
    .n_estimators(100)
    .contamination(0.1)
    .extended(true); // Extended Isolation Forest variant

let fitted = iso_forest.fit(&x, &y)?; // y is ignored (unsupervised)
let anomaly_scores = fitted.decision_function(&x)?;
let labels = fitted.predict(&x)?; // -1 = anomaly, 1 = inlier
```

### Model Trees

```rust
use sklears_tree::{ModelTree, LeafModelType};

let model_tree = ModelTree::new()
    .max_depth(5)
    .leaf_model(LeafModelType::Linear);

let fitted = model_tree.fit(&x, &y)?;
```

### Interpretability

```rust
use sklears_tree::{ShapConfig, TreeShapExplainer};

let explainer = TreeShapExplainer::from_classifier(x.ncols(), ShapConfig::default())?;
let explanation = explainer.explain_instance(&x.row(0).to_owned())?;
let explanations = explainer.explain_instances(&x)?;
```

Partial dependence values (`PartialDependencePoint`/`PartialDependencePlot`) are computed via `calculate_partial_dependence` in the `classifier` module.

## Performance Features

### Parallel Processing

```rust
let rf = RandomForestClassifier::new()
    .n_estimators(1000)
    .n_jobs(-1); // Use all cores

let fitted = rf.fit(&x, &y)?;
let predictions = fitted.predict_parallel(&x)?;
```

### Streaming Algorithms

```rust
use sklears_tree::incremental::hoeffding_tree::{HoeffdingTree, HoeffdingTreeConfig};

// Hoeffding tree for streaming data
let mut hoeffding = HoeffdingTree::new(HoeffdingTreeConfig::default(), x.ncols());

for (row, label) in data_stream {
    hoeffding.update(&row, label, None)?;
}
```

`incremental::ensemble_methods` also provides `IncrementalRandomForest` and a
streaming `OnlineGradientBoosting` ensemble for online use cases.

## Multi-Output and Multi-Label Trees

```rust
use sklears_tree::{MultiOutputDecisionTree, MultiOutputTreeConfig, MultiOutputStrategy};

// Chained multi-output regression: tree k conditions on predictions 0..k-1
let config = MultiOutputTreeConfig {
    multi_output_strategy: MultiOutputStrategy::Chained,
    ..Default::default()
};
let mo_tree = MultiOutputDecisionTree::new(config);
let fitted = mo_tree.fit(&x, &y)?; // y: Array2<f64>, one column per output
let predictions = fitted.predict(&x)?;
```

`MultiOutputStrategy` also supports `Independent` and `Shared` tree structures.
`MultiLabelRandomForest` (in the same module) provides a forest-based
multi-label classifier with an optional label-correlation matrix.

## Benchmarks

Performance on standard datasets:

| Algorithm | scikit-learn | sklears-tree | Speedup |
|-----------|--------------|--------------|---------|
| Decision Tree | 5.2ms | 0.8ms | 6.5x |
| Random Forest | 125ms | 12ms | 10.4x |
| Isolation Forest | 95ms | 11ms | 8.6x |

## Architecture

```
sklears-tree/
├── decision_tree.rs, classifier.rs, regressor.rs  # Core tree structures
├── random_forest/                                 # Forest algorithms
├── isolation_forest.rs                            # Anomaly detection
├── model_tree.rs                                  # Leaf-model regression trees
├── multi_output.rs                                # Multi-output / multi-label trees
├── incremental/                                   # Streaming/online algorithms
├── shap.rs                                        # TreeSHAP interpretability
├── node.rs                                        # Compact/bit-packed tree representations
└── parallel.rs                                    # Parallel training/prediction utilities
```

## Status

- **Implementation**: 79 passing crate tests for `0.2.0` (1 skipped)
- **Core Algorithms**: Decision Trees, Random Forest, Isolation Forest, Model Trees, Multi-Output/Multi-Label Trees ✓
- **Advanced Splits**: Oblique (hyperplane) and CHAID splits ✓
- **Interpretability**: TreeSHAP, partial dependence ✓
- **Streaming**: Hoeffding Tree, incremental random forest, online gradient boosting ✓

## Contributing

We welcome contributions! Priority areas:
- Additional tree algorithms
- Performance optimizations
- Documentation improvements

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

Licensed under the Apache License, Version 2.0.

## Citation

```bibtex
@software{sklears_tree,
  title = {sklears-tree: High-Performance Tree Algorithms for Rust},
  author = {COOLJAPAN OU (Team KitaSan)},
  year = {2026},
  url = {https://github.com/cool-japan/sklears}
}
```
