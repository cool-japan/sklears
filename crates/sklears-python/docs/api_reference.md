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
import sklears as skl
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
import sklears as skl
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
import sklears as skl
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

> **Coming Soon:** Preprocessing classes (`StandardScaler`, `MinMaxScaler`, `LabelEncoder`) are not yet exposed in this release. They will be available in a future version.

### StandardScaler *(Coming Soon)*

Standardize features by removing the mean and scaling to unit variance.

```python
# Not yet available - Coming Soon
# class StandardScaler(copy=True, with_mean=True, with_std=True)
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

### MinMaxScaler *(Coming Soon)*

Transform features by scaling each feature to a given range.

```python
# Not yet available - Coming Soon
# class MinMaxScaler(feature_range=(0, 1), copy=True, clip=False)
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

### LabelEncoder *(Coming Soon)*

Encode target labels with value between 0 and n_classes-1.

```python
# Not yet available - Coming Soon
# class LabelEncoder()
```

**Attributes:**
- `classes_` : list
  Holds the label for each class.

**Methods:**

#### fit_transform(y)
Fit label encoder and return encoded labels.

#### inverse_transform(y)
Transform labels back to original encoding.

## Metrics

> **Coming Soon:** All metrics functions are not yet exposed in this release. They will be available in a future version. Use `model.score()` for R² evaluation, or compute metrics manually with NumPy as an interim solution.

### Classification Metrics *(Coming Soon)*

#### accuracy_score *(Coming Soon)*

Classification accuracy score. Not yet available.

#### precision_score *(Coming Soon)*

Compute the precision. Not yet available.

#### recall_score *(Coming Soon)*

Compute the recall. Not yet available.

#### f1_score *(Coming Soon)*

Compute the F1 score. Not yet available.

#### confusion_matrix *(Coming Soon)*

Compute confusion matrix. Not yet available.

### Regression Metrics *(Coming Soon)*

#### mean_squared_error *(Coming Soon)*

Mean squared error regression loss. Not yet available.

#### mean_absolute_error *(Coming Soon)*

Mean absolute error regression loss. Not yet available.

#### r2_score *(Coming Soon)*

R² (coefficient of determination) regression score function. Not yet available.

**Interim NumPy-based example:**
```python
import numpy as np

y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

# Compute metrics manually until sklears exposes them
mse = float(np.mean((y_true - y_pred) ** 2))
mae = float(np.mean(np.abs(y_true - y_pred)))
ss_res = float(np.sum((y_true - y_pred) ** 2))
ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
r2 = 1.0 - ss_res / ss_tot

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
import sklears as skl
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

### get_hardware_info() *(Coming Soon)*

Get hardware acceleration capabilities. Not yet available in this release.

### benchmark_basic_operations() *(Coming Soon)*

Run basic performance benchmarks. Not yet available in this release.

### set_config(option, value) *(Coming Soon)*

Set global configuration options. Not yet available in this release.

### get_config() *(Coming Soon)*

Get current configuration. Not yet available in this release.

### show_versions() *(Coming Soon)*

Print comprehensive system information. Not yet available in this release.

**Available utilities example:**
```python
import sklears as skl

print(f"Version: {skl.get_version()}")

build_info = skl.get_build_info()
print(f"Build info: {build_info}")
```

## Error Handling

All functions and methods raise appropriate Python exceptions:

- `ValueError` - Invalid parameter values or data shape mismatches
- `RuntimeError` - Internal computation errors
- `TypeError` - Incorrect parameter types

**Example:**
```python
import sklears as skl
import numpy as np

try:
    model = skl.LinearRegression()
    # This will raise an error due to shape mismatch
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