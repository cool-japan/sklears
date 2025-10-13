# Sklears Python API Reference

Complete reference documentation for all Sklears Python classes, functions, and modules.

## Table of Contents

1. [Linear Models](#linear-models)
2. [Clustering](#clustering)
3. [Preprocessing](#preprocessing)
4. [Metrics](#metrics)
5. [Model Selection](#model-selection)
6. [Utilities](#utilities)

## Linear Models

### LinearRegression

Ordinary least squares linear regression.

```python
class LinearRegression(fit_intercept=True, copy_x=True)
```

**Parameters:**
- `fit_intercept` : bool, default=True
  Whether to calculate the intercept for this model.
- `copy_x` : bool, default=True
  If True, X will be copied; else, it may be overwritten.

**Attributes:**
- `coef_` : ndarray of shape (n_features,)
  Estimated coefficients for the linear regression problem.
- `intercept_` : float
  Independent term in the linear model.

**Methods:**

#### fit(X, y)
Fit linear model.

**Parameters:**
- `X` : array-like of shape (n_samples, n_features)
  Training data.
- `y` : array-like of shape (n_samples,)
  Target values.

**Returns:**
- `self` : object
  Fitted estimator.

#### predict(X)
Predict using the linear model.

**Parameters:**
- `X` : array-like of shape (n_samples, n_features)
  Samples.

**Returns:**
- `y_pred` : ndarray of shape (n_samples,)
  Returns predicted values.

#### score(X, y)
Return the coefficient of determination R² of the prediction.

**Parameters:**
- `X` : array-like of shape (n_samples, n_features)
  Test samples.
- `y` : array-like of shape (n_samples,)
  True values for X.

**Returns:**
- `score` : float
  R² of self.predict(X) wrt. y.

**Example:**
```python
import sklears_python as skl
import numpy as np

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

model = skl.LinearRegression()
model.fit(X, y)
print(model.score(X, y))  # 1.0
```

### Ridge

Linear least squares with L2 regularization.

```python
class Ridge(alpha=1.0, fit_intercept=True, copy_x=True)
```

**Parameters:**
- `alpha` : float, default=1.0
  Regularization strength; must be a positive float.
- `fit_intercept` : bool, default=True
  Whether to fit the intercept.
- `copy_x` : bool, default=True
  If True, X will be copied; else, it may be overwritten.

**Example:**
```python
import sklears_python as skl
import numpy as np

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

model = skl.Ridge(alpha=0.5)
model.fit(X, y)
predictions = model.predict(X)
```

### Lasso

Linear Model trained with L1 prior as regularizer.

```python
class Lasso(alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-4)
```

**Parameters:**
- `alpha` : float, default=1.0
  Constant that multiplies the L1 term.
- `fit_intercept` : bool, default=True
  Whether to fit the intercept.
- `max_iter` : int, default=1000
  The maximum number of iterations.
- `tol` : float, default=1e-4
  The tolerance for the optimization.

### LogisticRegression

Logistic Regression classifier.

```python
class LogisticRegression(penalty='l2', C=1.0, fit_intercept=True, max_iter=100)
```

**Parameters:**
- `penalty` : str, default='l2'
  Used to specify the norm used in the penalization. 'l1' or 'l2'.
- `C` : float, default=1.0
  Inverse of regularization strength.
- `fit_intercept` : bool, default=True
  Specifies if a constant should be added to the decision function.
- `max_iter` : int, default=100
  Maximum number of iterations of the solver.

**Methods:**

#### predict_proba(X)
Probability estimates.

**Parameters:**
- `X` : array-like of shape (n_samples, n_features)
  Vector to be scored.

**Returns:**
- `T` : ndarray of shape (n_samples, n_classes)
  Returns the probability of the sample for each class.

## Clustering

### KMeans

K-Means clustering.

```python
class KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=1e-4, random_state=None)
```

**Parameters:**
- `n_clusters` : int, default=8
  The number of clusters to form.
- `init` : str, default='k-means++'
  Method for initialization: 'k-means++' or 'random'.
- `n_init` : int, default=10
  Number of time the k-means algorithm will be run.
- `max_iter` : int, default=300
  Maximum number of iterations for a single run.
- `tol` : float, default=1e-4
  Relative tolerance for convergence.
- `random_state` : int, optional
  Random state for reproducible results.

**Attributes:**
- `cluster_centers_` : ndarray of shape (n_clusters, n_features)
  Coordinates of cluster centers.
- `labels_` : ndarray of shape (n_samples,)
  Labels of each point.
- `inertia_` : float
  Sum of squared distances of samples to their closest cluster center.

**Methods:**

#### fit_predict(X)
Compute cluster centers and predict cluster index for each sample.

**Parameters:**
- `X` : array-like of shape (n_samples, n_features)
  New data to transform.

**Returns:**
- `labels` : ndarray of shape (n_samples,)
  Index of the cluster each sample belongs to.

**Example:**
```python
import sklears_python as skl
import numpy as np

X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
kmeans = skl.KMeans(n_clusters=2, random_state=0)
labels = kmeans.fit_predict(X)
print(labels)  # [1 1 1 0 0 0]
```

### DBSCAN

Density-Based Spatial Clustering of Applications with Noise.

```python
class DBSCAN(eps=0.5, min_samples=5, metric='euclidean', algorithm='auto')
```

**Parameters:**
- `eps` : float, default=0.5
  The maximum distance between two samples for one to be considered as in the neighborhood of the other.
- `min_samples` : int, default=5
  The number of samples in a neighborhood for a point to be considered as a core point.
- `metric` : str, default='euclidean'
  The metric to use when calculating distance between instances.
- `algorithm` : str, default='auto'
  The algorithm to be used: 'auto', 'ball_tree', 'kd_tree', or 'brute'.

## Preprocessing

### StandardScaler

Standardize features by removing the mean and scaling to unit variance.

```python
class StandardScaler(copy=True, with_mean=True, with_std=True)
```

**Parameters:**
- `copy` : bool, default=True
  If False, try to avoid a copy and do inplace scaling instead.
- `with_mean` : bool, default=True
  If True, center the data before scaling.
- `with_std` : bool, default=True
  If True, scale the data to unit variance.

**Attributes:**
- `mean_` : ndarray of shape (n_features,)
  The mean value for each feature in the training set.
- `scale_` : ndarray of shape (n_features,)
  Per feature relative scaling of the data.

**Methods:**

#### fit_transform(X)
Fit to data, then transform it.

#### inverse_transform(X)
Scale back the data to the original representation.

**Example:**
```python
import sklears_python as skl
import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6]])
scaler = skl.StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled.mean(axis=0))  # [0. 0.]
print(X_scaled.std(axis=0))   # [1. 1.]
```

### MinMaxScaler

Transform features by scaling each feature to a given range.

```python
class MinMaxScaler(feature_range=(0, 1), copy=True, clip=False)
```

**Parameters:**
- `feature_range` : tuple (min, max), default=(0, 1)
  Desired range of transformed data.
- `copy` : bool, default=True
  Set to False to perform inplace scaling.
- `clip` : bool, default=False
  Set to True to clip transformed values to the provided feature range.

**Attributes:**
- `data_min_` : ndarray of shape (n_features,)
  Per feature minimum seen in the data.
- `data_max_` : ndarray of shape (n_features,)
  Per feature maximum seen in the data.

### LabelEncoder

Encode target labels with value between 0 and n_classes-1.

```python
class LabelEncoder()
```

**Attributes:**
- `classes_` : list
  Holds the label for each class.

**Methods:**

#### fit_transform(y)
Fit label encoder and return encoded labels.

#### inverse_transform(y)
Transform labels back to original encoding.

**Example:**
```python
import sklears_python as skl

le = skl.LabelEncoder()
labels = ['paris', 'paris', 'tokyo', 'amsterdam']
encoded = le.fit_transform(labels)
print(encoded)  # [1 1 2 0]
print(le.classes_)  # ['amsterdam', 'paris', 'tokyo']
```

## Metrics

### Classification Metrics

#### accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)

Classification accuracy score.

**Parameters:**
- `y_true` : array-like of shape (n_samples,)
  Ground truth (correct) labels.
- `y_pred` : array-like of shape (n_samples,)
  Predicted labels.
- `normalize` : bool, default=True
  If False, return the number of correctly classified samples.
- `sample_weight` : array-like of shape (n_samples,), optional
  Sample weights.

**Returns:**
- `score` : float
  If normalize == True, return the fraction of correctly classified samples, else returns the number of correctly classified samples.

#### precision_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')

Compute the precision.

#### recall_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')

Compute the recall.

#### f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')

Compute the F1 score.

#### confusion_matrix(y_true, y_pred, labels=None, sample_weight=None, normalize=None)

Compute confusion matrix to evaluate the accuracy of a classification.

### Regression Metrics

#### mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average')

Mean squared error regression loss.

**Parameters:**
- `y_true` : array-like of shape (n_samples,)
  Ground truth (correct) target values.
- `y_pred` : array-like of shape (n_samples,)
  Estimated target values.
- `sample_weight` : array-like of shape (n_samples,), optional
  Sample weights.
- `multioutput` : str
  Defines aggregating of multiple output values.

**Returns:**
- `loss` : float
  A non-negative floating point value (the best value is 0.0).

#### mean_absolute_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average')

Mean absolute error regression loss.

#### r2_score(y_true, y_pred, sample_weight=None, multioutput='uniform_average')

R² (coefficient of determination) regression score function.

**Example:**
```python
import sklears_python as skl
import numpy as np

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

mse = skl.mean_squared_error(y_true, y_pred)
mae = skl.mean_absolute_error(y_true, y_pred)
r2 = skl.r2_score(y_true, y_pred)

print(f"MSE: {mse:.3f}")  # MSE: 0.375
print(f"MAE: {mae:.3f}")  # MAE: 0.5
print(f"R²: {r2:.3f}")    # R²: 0.948
```

## Model Selection

### train_test_split(X, y=None, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)

Split arrays into random train and test subsets.

**Parameters:**
- `X` : array-like of shape (n_samples, n_features)
  Training data.
- `y` : array-like of shape (n_samples,), optional
  Target variable.
- `test_size` : float or int, optional
  Proportion of the dataset to include in the test split.
- `train_size` : float or int, optional
  Proportion of the dataset to include in the train split.
- `random_state` : int, optional
  Random state for reproducible output.
- `shuffle` : bool, default=True
  Whether to shuffle the data before splitting.
- `stratify` : array-like, optional
  Data is split in a stratified fashion using this as the class labels.

**Returns:**
- `splitting` : list
  List containing train-test split of inputs.

### KFold

K-Fold cross-validator.

```python
class KFold(n_splits=5, shuffle=False, random_state=None)
```

**Parameters:**
- `n_splits` : int, default=5
  Number of folds.
- `shuffle` : bool, default=False
  Whether to shuffle the data before splitting.
- `random_state` : int, optional
  Random state for reproducible output.

**Methods:**

#### split(X, y=None)
Generate indices to split data into training and test set.

#### get_n_splits(X=None, y=None)
Returns the number of splitting iterations in the cross-validator.

**Example:**
```python
import sklears_python as skl
import numpy as np

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])

kf = skl.KFold(n_splits=2)
for train_index, test_index in kf.split(X):
    print(f"TRAIN: {train_index} TEST: {test_index}")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```

## Utilities

### get_version()

Get the version of sklears.

**Returns:**
- `version` : str
  Version string.

### get_build_info()

Get build information about sklears.

**Returns:**
- `info` : dict
  Dictionary containing build information including version, features, and dependencies.

### get_hardware_info()

Get hardware acceleration capabilities.

**Returns:**
- `info` : dict
  Dictionary containing boolean values for various hardware features.

### benchmark_basic_operations()

Run basic performance benchmarks.

**Returns:**
- `results` : dict
  Dictionary containing benchmark results in milliseconds.

### set_config(option, value)

Set global configuration options.

**Parameters:**
- `option` : str
  Configuration option name.
- `value` : str
  Configuration value.

### get_config()

Get current configuration.

**Returns:**
- `config` : dict
  Dictionary containing current configuration.

### show_versions()

Print comprehensive system information.

**Returns:**
- `info` : str
  Formatted string containing version and system information.

**Example:**
```python
import sklears_python as skl

print(f"Version: {skl.get_version()}")

# Hardware capabilities
hw_info = skl.get_hardware_info()
print(f"SIMD support: {hw_info.get('avx2', False)}")

# Performance benchmarks
benchmarks = skl.benchmark_basic_operations()
print(f"Matrix multiplication: {benchmarks['matrix_multiplication_100x100_ms']:.2f} ms")

# System information
print(skl.show_versions())
```

## Error Handling

All functions and methods raise appropriate Python exceptions:

- `ValueError` - Invalid parameter values or data shape mismatches
- `RuntimeError` - Internal computation errors
- `TypeError` - Incorrect parameter types

**Example:**
```python
import sklears_python as skl
import numpy as np

try:
    model = skl.LinearRegression()
    # This will raise ValueError due to shape mismatch
    model.fit(np.array([[1, 2]]), np.array([1, 2]))
except ValueError as e:
    print(f"Error: {e}")
```

## Performance Notes

- All algorithms are optimized for performance and use SIMD instructions when available
- Memory usage is optimized with zero-copy operations where possible
- Parallel processing is used automatically when beneficial
- Large datasets are handled efficiently with streaming algorithms

For optimal performance:
1. Use contiguous NumPy arrays when possible
2. Consider data types (float64 vs float32)
3. Enable hardware acceleration features
4. Use appropriate batch sizes for large datasets