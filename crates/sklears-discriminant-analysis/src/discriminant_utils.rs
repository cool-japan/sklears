//! Utility Functions for Discriminant Analysis
//!
//! This module provides shared utility functions, mathematical operations,
//! and helper functions used across all discriminant analysis algorithms.

use scirs2_core::ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use scirs2_core::random::{Random, rng};
use sklears_core::{
    error::{validate, Result as SklResult, SklearsError},
    types::Float,
};
use std::collections::{HashMap, HashSet};

/// Mathematical utility functions
pub mod math {
    use super::*;

    /// Compute the Moore-Penrose pseudoinverse of a matrix
    pub fn pseudoinverse(matrix: &Array2<Float>, tolerance: Float) -> SklResult<Array2<Float>> {
        let (m, n) = matrix.dim();

        // Use simplified approach for small matrices
        if m <= n {
            // Left pseudoinverse: (A^T A)^-1 A^T
            let at = matrix.t().to_owned();
            let ata = at.dot(matrix);
            let ata_inv = matrix_inverse(&ata, tolerance)?;
            Ok(ata_inv.dot(&at))
        } else {
            // Right pseudoinverse: A^T (A A^T)^-1
            let at = matrix.t().to_owned();
            let aat = matrix.dot(&at);
            let aat_inv = matrix_inverse(&aat, tolerance)?;
            Ok(at.dot(&aat_inv))
        }
    }

    /// Compute matrix inverse using LU decomposition with partial pivoting
    pub fn matrix_inverse(matrix: &Array2<Float>, tolerance: Float) -> SklResult<Array2<Float>> {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(SklearsError::InvalidInput("Matrix must be square for inversion".to_string()));
        }

        let mut a = matrix.clone();
        let mut b = Array2::eye(n);

        // Forward elimination with partial pivoting
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if a[[k, i]].abs() > a[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            // Swap rows if needed
            if max_row != i {
                for j in 0..n {
                    a.swap([i, j], [max_row, j]);
                    b.swap([i, j], [max_row, j]);
                }
            }

            // Check for singularity
            if a[[i, i]].abs() < tolerance {
                return Err(SklearsError::NumericalError(
                    "Matrix is singular or near-singular".to_string()
                ));
            }

            // Eliminate column
            for k in (i + 1)..n {
                let factor = a[[k, i]] / a[[i, i]];
                for j in i..n {
                    a[[k, j]] -= factor * a[[i, j]];
                }
                for j in 0..n {
                    b[[k, j]] -= factor * b[[i, j]];
                }
            }
        }

        // Back substitution
        for i in (0..n).rev() {
            for j in 0..n {
                for k in (i + 1)..n {
                    b[[i, j]] -= a[[i, k]] * b[[k, j]];
                }
                b[[i, j]] /= a[[i, i]];
            }
        }

        Ok(b)
    }

    /// Compute eigenvalues and eigenvectors using QR algorithm (simplified implementation)
    pub fn eigen_decomposition(matrix: &Array2<Float>, max_iter: usize, tolerance: Float) -> SklResult<(Array1<Float>, Array2<Float>)> {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(SklearsError::InvalidInput("Matrix must be square for eigendecomposition".to_string()));
        }

        let mut a = matrix.clone();
        let mut q_total = Array2::eye(n);

        // QR algorithm
        for _ in 0..max_iter {
            let (q, r) = qr_decomposition(&a)?;
            a = r.dot(&q);
            q_total = q_total.dot(&q);

            // Check convergence (simplified)
            let mut off_diagonal_norm = 0.0;
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        off_diagonal_norm += a[[i, j]].abs();
                    }
                }
            }

            if off_diagonal_norm < tolerance {
                break;
            }
        }

        // Extract eigenvalues from diagonal
        let eigenvalues = a.diag().to_owned();

        Ok((eigenvalues, q_total))
    }

    /// QR decomposition using Gram-Schmidt process
    pub fn qr_decomposition(matrix: &Array2<Float>) -> SklResult<(Array2<Float>, Array2<Float>)> {
        let (m, n) = matrix.dim();
        let mut q = Array2::zeros((m, n));
        let mut r = Array2::zeros((n, n));

        for j in 0..n {
            let mut col = matrix.column(j).to_owned();

            // Gram-Schmidt orthogonalization
            for i in 0..j {
                let q_col = q.column(i);
                let proj = col.dot(&q_col);
                r[[i, j]] = proj;

                for k in 0..m {
                    col[k] -= proj * q_col[k];
                }
            }

            // Normalize
            let norm = col.dot(&col).sqrt();
            r[[j, j]] = norm;

            if norm > 1e-12 {
                for k in 0..m {
                    q[[k, j]] = col[k] / norm;
                }
            }
        }

        Ok((q, r))
    }

    /// Compute singular value decomposition (simplified implementation)
    pub fn svd(matrix: &Array2<Float>, tolerance: Float) -> SklResult<(Array2<Float>, Array1<Float>, Array2<Float>)> {
        let (m, n) = matrix.dim();

        // Compute A^T A for V and singular values
        let at = matrix.t().to_owned();
        let ata = at.dot(matrix);
        let (eigenvals, v) = eigen_decomposition(&ata, 100, tolerance)?;

        // Singular values are square roots of eigenvalues
        let mut singular_values = Array1::zeros(n);
        for i in 0..n {
            singular_values[i] = eigenvals[i].max(0.0).sqrt();
        }

        // Sort singular values in descending order
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| singular_values[j].partial_cmp(&singular_values[i]).unwrap());

        let mut sorted_singular_values = Array1::zeros(n);
        let mut sorted_v = Array2::zeros((n, n));
        for (i, &idx) in indices.iter().enumerate() {
            sorted_singular_values[i] = singular_values[idx];
            sorted_v.column_mut(i).assign(&v.column(idx));
        }

        // Compute U = A V S^-1
        let mut u = Array2::zeros((m, n.min(m)));
        for i in 0..n.min(m) {
            if sorted_singular_values[i] > tolerance {
                let av = matrix.dot(&sorted_v.column(i));
                for j in 0..m {
                    u[[j, i]] = av[j] / sorted_singular_values[i];
                }
            }
        }

        Ok((u, sorted_singular_values, sorted_v))
    }

    /// Compute log determinant safely
    pub fn safe_log_determinant(matrix: &Array2<Float>, regularization: Float) -> Float {
        // Use product of diagonal elements as approximation
        matrix.diag().iter()
            .map(|&x| (x.abs() + regularization).ln())
            .sum()
    }

    /// Compute trace of a matrix
    pub fn trace(matrix: &Array2<Float>) -> Float {
        matrix.diag().sum()
    }

    /// Compute Frobenius norm of a matrix
    pub fn frobenius_norm(matrix: &Array2<Float>) -> Float {
        matrix.iter().map(|&x| x * x).sum::<Float>().sqrt()
    }

    /// Compute condition number approximation
    pub fn condition_number(matrix: &Array2<Float>) -> Float {
        let diag = matrix.diag();
        let max_val = diag.iter().fold(0.0, |a, &b| a.max(b.abs()));
        let min_val = diag.iter().fold(Float::INFINITY, |a, &b| a.min(b.abs()));

        if min_val > 1e-15 {
            max_val / min_val
        } else {
            Float::INFINITY
        }
    }

    /// Matrix power (for integer exponents)
    pub fn matrix_power(matrix: &Array2<Float>, power: i32) -> SklResult<Array2<Float>> {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(SklearsError::InvalidInput("Matrix must be square for matrix power".to_string()));
        }

        if power == 0 {
            return Ok(Array2::eye(n));
        }

        if power < 0 {
            let inv = matrix_inverse(matrix, 1e-12)?;
            return matrix_power(&inv, -power);
        }

        let mut result = Array2::eye(n);
        let mut base = matrix.clone();
        let mut exp = power as u32;

        while exp > 0 {
            if exp % 2 == 1 {
                result = result.dot(&base);
            }
            base = base.dot(&base);
            exp /= 2;
        }

        Ok(result)
    }
}

/// Statistical utility functions
pub mod stats {
    use super::*;

    /// Compute sample covariance matrix
    pub fn covariance_matrix(data: &Array2<Float>, ddof: usize) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = data.dim();

        if n_samples <= ddof {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be greater than degrees of freedom".to_string()
            ));
        }

        let mean = data.mean_axis(Axis(0)).unwrap();
        let mut cov = Array2::zeros((n_features, n_features));

        for i in 0..n_samples {
            let sample = data.row(i);
            let diff = &sample - &mean;
            for j in 0..n_features {
                for k in 0..n_features {
                    cov[[j, k]] += diff[j] * diff[k];
                }
            }
        }

        let denom = (n_samples - ddof) as Float;
        cov = cov / denom;

        Ok(cov)
    }

    /// Compute correlation matrix from covariance matrix
    pub fn correlation_from_covariance(cov: &Array2<Float>) -> SklResult<Array2<Float>> {
        let n = cov.nrows();
        if n != cov.ncols() {
            return Err(SklearsError::InvalidInput("Covariance matrix must be square".to_string()));
        }

        let mut corr = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                let std_i = cov[[i, i]].sqrt();
                let std_j = cov[[j, j]].sqrt();

                if std_i > 1e-15 && std_j > 1e-15 {
                    corr[[i, j]] = cov[[i, j]] / (std_i * std_j);
                } else {
                    corr[[i, j]] = if i == j { 1.0 } else { 0.0 };
                }
            }
        }

        Ok(corr)
    }

    /// Compute Mahalanobis distance
    pub fn mahalanobis_distance(
        x: &ArrayView1<Float>,
        mean: &ArrayView1<Float>,
        inv_cov: &Array2<Float>,
    ) -> SklResult<Float> {
        let diff = x - mean;
        let temp = inv_cov.dot(&diff);
        let distance_sq = diff.dot(&temp);

        if distance_sq < 0.0 {
            return Err(SklearsError::NumericalError(
                "Negative squared distance computed".to_string()
            ));
        }

        Ok(distance_sq.sqrt())
    }

    /// Robust covariance estimation using Minimum Covariance Determinant (MCD)
    pub fn robust_covariance_mcd(
        data: &Array2<Float>,
        contamination: Float,
        max_trials: usize,
        random_seed: Option<u64>,
    ) -> SklResult<(Array1<Float>, Array2<Float>, Array1<Float>)> {
        let (n_samples, n_features) = data.dim();
        let subset_size = ((n_samples as Float * (1.0 - contamination)).ceil() as usize)
            .max(n_features + 1)
            .min(n_samples);

        let mut best_mean = data.mean_axis(Axis(0)).unwrap();
        let mut best_cov = Array2::eye(n_features);
        let mut best_det = Float::INFINITY;
        let mut best_weights = Array1::ones(n_samples);

        let seed = random_seed.unwrap_or(42);

        for trial in 0..max_trials {
            // Generate subset indices
            let mut indices: Vec<usize> = (0..n_samples).collect();

            // Pseudo-random shuffle
            for i in 0..n_samples {
                let j = (seed + trial as u64 * 17 + i as u64 * 31) as usize % n_samples;
                indices.swap(i, j);
            }
            indices.truncate(subset_size);

            // Compute subset statistics
            let subset_data = Array2::from_shape_fn((subset_size, n_features), |(i, j)| {
                data[[indices[i], j]]
            });

            let subset_mean = subset_data.mean_axis(Axis(0)).unwrap();
            let subset_cov = covariance_matrix(&subset_data, 1)?;

            // Compute determinant approximation
            let det = subset_cov.diag().iter().product::<Float>();

            if det > 0.0 && det < best_det {
                best_det = det;
                best_mean = subset_mean;
                best_cov = subset_cov;

                // Compute robust weights
                if let Ok(inv_cov) = math::matrix_inverse(&best_cov, 1e-12) {
                    for i in 0..n_samples {
                        let sample = data.row(i);
                        match mahalanobis_distance(&sample, &best_mean.view(), &inv_cov) {
                            Ok(dist) => {
                                // Tukey's biweight function
                                let c = 4.685;
                                if dist <= c {
                                    let u = dist / c;
                                    best_weights[i] = (1.0 - u * u).powi(2);
                                } else {
                                    best_weights[i] = 0.0;
                                }
                            },
                            Err(_) => {
                                best_weights[i] = 1.0 / n_samples as Float;
                            }
                        }
                    }
                }
            }
        }

        Ok((best_mean, best_cov, best_weights))
    }

    /// Compute multivariate normal log-likelihood
    pub fn multivariate_normal_logpdf(
        x: &ArrayView1<Float>,
        mean: &ArrayView1<Float>,
        cov: &Array2<Float>,
        regularization: Float,
    ) -> SklResult<Float> {
        let k = x.len() as Float;
        let diff = x - mean;

        // Compute log determinant
        let log_det = math::safe_log_determinant(cov, regularization);

        // Compute quadratic form (simplified using diagonal approximation)
        let mut quad_form = 0.0;
        for i in 0..diff.len() {
            let var = cov[[i, i]] + regularization;
            quad_form += diff[i] * diff[i] / var;
        }

        let log_pdf = -0.5 * (k * (2.0 * std::f64::consts::PI).ln() + log_det + quad_form);
        Ok(log_pdf)
    }

    /// Shrinkage covariance estimation (Ledoit-Wolf)
    pub fn shrinkage_covariance(data: &Array2<Float>, shrinkage: Option<Float>) -> SklResult<Array2<Float>> {
        let emp_cov = covariance_matrix(data, 1)?;
        let n_features = emp_cov.nrows();

        let shrinkage_param = if let Some(s) = shrinkage {
            s.clamp(0.0, 1.0)
        } else {
            // Simplified Ledoit-Wolf shrinkage estimation
            let trace = math::trace(&emp_cov);
            let target_var = trace / n_features as Float;

            // Estimate optimal shrinkage (simplified)
            let mut sum_sq_dev = 0.0;
            for i in 0..n_features {
                for j in 0..n_features {
                    if i != j {
                        sum_sq_dev += emp_cov[[i, j]].powi(2);
                    } else {
                        sum_sq_dev += (emp_cov[[i, j]] - target_var).powi(2);
                    }
                }
            }

            (sum_sq_dev / (n_features * n_features) as Float).clamp(0.0, 1.0)
        };

        // Apply shrinkage
        let trace = math::trace(&emp_cov);
        let target = Array2::eye(n_features) * (trace / n_features as Float);
        let shrunk_cov = &emp_cov * (1.0 - shrinkage_param) + &target * shrinkage_param;

        Ok(shrunk_cov)
    }
}

/// Data preprocessing utilities
pub mod preprocessing {
    use super::*;

    /// Standardize features (z-score normalization)
    pub fn standardize(data: &Array2<Float>) -> SklResult<(Array2<Float>, Array1<Float>, Array1<Float>)> {
        let (n_samples, n_features) = data.dim();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput("Cannot standardize empty data".to_string()));
        }

        let mean = data.mean_axis(Axis(0)).unwrap();
        let mut std_dev = Array1::zeros(n_features);

        // Compute standard deviation
        for i in 0..n_samples {
            let sample = data.row(i);
            let diff = &sample - &mean;
            for j in 0..n_features {
                std_dev[j] += diff[j] * diff[j];
            }
        }

        if n_samples > 1 {
            std_dev = std_dev.mapv(|x| (x / (n_samples - 1) as Float).sqrt());
        }

        // Avoid division by zero
        for j in 0..n_features {
            if std_dev[j] < 1e-15 {
                std_dev[j] = 1.0;
            }
        }

        // Standardize data
        let mut standardized = Array2::zeros((n_samples, n_features));
        for i in 0..n_samples {
            let sample = data.row(i);
            for j in 0..n_features {
                standardized[[i, j]] = (sample[j] - mean[j]) / std_dev[j];
            }
        }

        Ok((standardized, mean, std_dev))
    }

    /// Center data (subtract mean)
    pub fn center_data(data: &Array2<Float>) -> SklResult<(Array2<Float>, Array1<Float>)> {
        let (n_samples, n_features) = data.dim();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput("Cannot center empty data".to_string()));
        }

        let mean = data.mean_axis(Axis(0)).unwrap();
        let mut centered = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            let sample = data.row(i);
            for j in 0..n_features {
                centered[[i, j]] = sample[j] - mean[j];
            }
        }

        Ok((centered, mean))
    }

    /// Min-max normalization
    pub fn min_max_scale(data: &Array2<Float>, feature_range: (Float, Float)) -> SklResult<(Array2<Float>, Array1<Float>, Array1<Float>)> {
        let (n_samples, n_features) = data.dim();
        let (min_val, max_val) = feature_range;

        if min_val >= max_val {
            return Err(SklearsError::InvalidInput("Invalid feature range".to_string()));
        }

        let mut data_min = Array1::from_elem(n_features, Float::INFINITY);
        let mut data_max = Array1::from_elem(n_features, Float::NEG_INFINITY);

        // Find min and max for each feature
        for i in 0..n_samples {
            let sample = data.row(i);
            for j in 0..n_features {
                data_min[j] = data_min[j].min(sample[j]);
                data_max[j] = data_max[j].max(sample[j]);
            }
        }

        // Scale data
        let mut scaled = Array2::zeros((n_samples, n_features));
        for i in 0..n_samples {
            let sample = data.row(i);
            for j in 0..n_features {
                let range = data_max[j] - data_min[j];
                if range > 1e-15 {
                    scaled[[i, j]] = min_val + (max_val - min_val) * (sample[j] - data_min[j]) / range;
                } else {
                    scaled[[i, j]] = min_val;
                }
            }
        }

        Ok((scaled, data_min, data_max))
    }

    /// Robust scaling using median and MAD
    pub fn robust_scale(data: &Array2<Float>) -> SklResult<(Array2<Float>, Array1<Float>, Array1<Float>)> {
        let (n_samples, n_features) = data.dim();
        let mut medians = Array1::zeros(n_features);
        let mut mads = Array1::zeros(n_features);

        // Compute median and MAD for each feature
        for j in 0..n_features {
            let mut column: Vec<Float> = data.column(j).to_vec();
            column.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // Median
            let median = if n_samples % 2 == 0 {
                (column[n_samples / 2 - 1] + column[n_samples / 2]) / 2.0
            } else {
                column[n_samples / 2]
            };
            medians[j] = median;

            // Median Absolute Deviation (MAD)
            let mut abs_devs: Vec<Float> = column.iter()
                .map(|&x| (x - median).abs())
                .collect();
            abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mad = if n_samples % 2 == 0 {
                (abs_devs[n_samples / 2 - 1] + abs_devs[n_samples / 2]) / 2.0
            } else {
                abs_devs[n_samples / 2]
            };
            mads[j] = if mad > 1e-15 { mad } else { 1.0 };
        }

        // Scale data
        let mut scaled = Array2::zeros((n_samples, n_features));
        for i in 0..n_samples {
            let sample = data.row(i);
            for j in 0..n_features {
                scaled[[i, j]] = (sample[j] - medians[j]) / mads[j];
            }
        }

        Ok((scaled, medians, mads))
    }
}

/// Cross-validation utilities
pub mod cross_validation {
    use super::*;

    /// K-fold cross-validation split
    pub fn kfold_split(n_samples: usize, n_folds: usize, shuffle: bool, random_seed: Option<u64>) -> SklResult<Vec<(Vec<usize>, Vec<usize>)>> {
        if n_folds < 2 {
            return Err(SklearsError::InvalidInput("Number of folds must be at least 2".to_string()));
        }

        if n_folds > n_samples {
            return Err(SklearsError::InvalidInput("Number of folds cannot exceed number of samples".to_string()));
        }

        let mut indices: Vec<usize> = (0..n_samples).collect();

        if shuffle {
            let seed = random_seed.unwrap_or(42);
            // Simple pseudo-random shuffle
            for i in 0..n_samples {
                let j = (seed + i as u64 * 17) as usize % n_samples;
                indices.swap(i, j);
            }
        }

        let mut folds = Vec::new();
        let fold_size = n_samples / n_folds;
        let remainder = n_samples % n_folds;

        for fold in 0..n_folds {
            let start = fold * fold_size + fold.min(remainder);
            let end = start + fold_size + if fold < remainder { 1 } else { 0 };

            let test_indices = indices[start..end].to_vec();
            let train_indices: Vec<usize> = indices.iter()
                .enumerate()
                .filter(|(i, _)| *i < start || *i >= end)
                .map(|(_, &idx)| idx)
                .collect();

            folds.push((train_indices, test_indices));
        }

        Ok(folds)
    }

    /// Stratified K-fold split (maintains class distribution)
    pub fn stratified_kfold_split(
        y: &Array1<i32>,
        n_folds: usize,
        shuffle: bool,
        random_seed: Option<u64>,
    ) -> SklResult<Vec<(Vec<usize>, Vec<usize>)>> {
        let n_samples = y.len();

        // Group indices by class
        let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &class) in y.iter().enumerate() {
            class_indices.entry(class).or_insert_with(Vec::new).push(i);
        }

        // Shuffle indices within each class
        let seed = random_seed.unwrap_or(42);
        if shuffle {
            for (class, indices) in class_indices.iter_mut() {
                let class_seed = seed + *class as u64;
                for i in 0..indices.len() {
                    let j = (class_seed + i as u64 * 23) as usize % indices.len();
                    indices.swap(i, j);
                }
            }
        }

        // Create stratified folds
        let mut folds: Vec<(Vec<usize>, Vec<usize>)> = vec![(Vec::new(), Vec::new()); n_folds];

        for (_, indices) in class_indices {
            let class_size = indices.len();
            let fold_size = class_size / n_folds;
            let remainder = class_size % n_folds;

            for fold in 0..n_folds {
                let start = fold * fold_size + fold.min(remainder);
                let end = start + fold_size + if fold < remainder { 1 } else { 0 };

                for &idx in &indices[start..end] {
                    folds[fold].1.push(idx);
                }
            }
        }

        // Set training indices (complement of test indices)
        for fold in 0..n_folds {
            let test_set: HashSet<usize> = folds[fold].1.iter().cloned().collect();
            folds[fold].0 = (0..n_samples)
                .filter(|i| !test_set.contains(i))
                .collect();
        }

        Ok(folds)
    }
}

/// Distance computation utilities
pub mod distance {
    use super::*;

    /// Euclidean distance between two points
    pub fn euclidean_distance(x1: &ArrayView1<Float>, x2: &ArrayView1<Float>) -> SklResult<Float> {
        if x1.len() != x2.len() {
            return Err(SklearsError::InvalidInput("Vectors must have the same length".to_string()));
        }

        let dist_sq: Float = x1.iter()
            .zip(x2.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum();

        Ok(dist_sq.sqrt())
    }

    /// Manhattan distance between two points
    pub fn manhattan_distance(x1: &ArrayView1<Float>, x2: &ArrayView1<Float>) -> SklResult<Float> {
        if x1.len() != x2.len() {
            return Err(SklearsError::InvalidInput("Vectors must have the same length".to_string()));
        }

        let dist: Float = x1.iter()
            .zip(x2.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();

        Ok(dist)
    }

    /// Cosine distance between two points
    pub fn cosine_distance(x1: &ArrayView1<Float>, x2: &ArrayView1<Float>) -> SklResult<Float> {
        if x1.len() != x2.len() {
            return Err(SklearsError::InvalidInput("Vectors must have the same length".to_string()));
        }

        let dot_product: Float = x1.iter().zip(x2.iter()).map(|(&a, &b)| a * b).sum();
        let norm1: Float = x1.iter().map(|&x| x * x).sum::<Float>().sqrt();
        let norm2: Float = x2.iter().map(|&x| x * x).sum::<Float>().sqrt();

        if norm1 < 1e-15 || norm2 < 1e-15 {
            return Ok(1.0); // Maximum distance for zero vectors
        }

        let cosine_sim = dot_product / (norm1 * norm2);
        Ok(1.0 - cosine_sim)
    }

    /// Minkowski distance with parameter p
    pub fn minkowski_distance(x1: &ArrayView1<Float>, x2: &ArrayView1<Float>, p: Float) -> SklResult<Float> {
        if x1.len() != x2.len() {
            return Err(SklearsError::InvalidInput("Vectors must have the same length".to_string()));
        }

        if p <= 0.0 {
            return Err(SklearsError::InvalidInput("Minkowski parameter p must be positive".to_string()));
        }

        let dist: Float = x1.iter()
            .zip(x2.iter())
            .map(|(&a, &b)| (a - b).abs().powf(p))
            .sum();

        Ok(dist.powf(1.0 / p))
    }
}

/// Performance and benchmarking utilities
pub mod performance {
    use super::*;
    use std::time::{Duration, Instant};

    /// Simple timer for performance measurement
    pub struct Timer {
        start: Instant,
    }

    impl Timer {
        pub fn new() -> Self {
            Timer {
                start: Instant::now(),
            }
        }

        pub fn elapsed(&self) -> Duration {
            self.start.elapsed()
        }

        pub fn elapsed_ms(&self) -> u128 {
            self.elapsed().as_millis()
        }

        pub fn restart(&mut self) {
            self.start = Instant::now();
        }
    }

    /// Memory usage estimation for arrays
    pub fn estimate_memory_usage(shape: &[usize]) -> usize {
        let elements: usize = shape.iter().product();
        elements * std::mem::size_of::<Float>()
    }

    /// Check if computation is feasible given memory constraints
    pub fn check_memory_feasibility(shapes: &[&[usize]], max_memory_mb: usize) -> bool {
        let total_memory: usize = shapes.iter()
            .map(|shape| estimate_memory_usage(shape))
            .sum();

        total_memory <= max_memory_mb * 1024 * 1024
    }
}

/// Validation utilities
pub mod validation {
    use super::*;

    /// Check if arrays have consistent sample sizes
    pub fn check_consistent_samples(arrays: &[&Array2<Float>]) -> SklResult<()> {
        if arrays.is_empty() {
            return Ok(());
        }

        let first_samples = arrays[0].nrows();
        for (i, array) in arrays.iter().enumerate().skip(1) {
            if array.nrows() != first_samples {
                return Err(SklearsError::InvalidInput(format!(
                    "Array {} has {} samples, expected {}",
                    i, array.nrows(), first_samples
                )));
            }
        }

        Ok(())
    }

    /// Check if data contains any NaN or infinite values
    pub fn check_finite_data(data: &Array2<Float>) -> SklResult<()> {
        for &value in data.iter() {
            if !value.is_finite() {
                return Err(SklearsError::InvalidInput("Data contains non-finite values".to_string()));
            }
        }
        Ok(())
    }

    /// Check if class labels are valid
    pub fn check_valid_labels(y: &Array1<i32>) -> SklResult<()> {
        if y.is_empty() {
            return Err(SklearsError::InvalidInput("Labels array is empty".to_string()));
        }

        let unique_labels: HashSet<i32> = y.iter().cloned().collect();
        if unique_labels.len() < 2 {
            return Err(SklearsError::InvalidInput("Need at least 2 different class labels".to_string()));
        }

        Ok(())
    }

    /// Check if sample weights are valid
    pub fn check_sample_weights(weights: &Array1<Float>, n_samples: usize) -> SklResult<()> {
        if weights.len() != n_samples {
            return Err(SklearsError::InvalidInput("Sample weights length must match number of samples".to_string()));
        }

        for &weight in weights.iter() {
            if weight < 0.0 || !weight.is_finite() {
                return Err(SklearsError::InvalidInput("Sample weights must be non-negative and finite".to_string()));
            }
        }

        Ok(())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_matrix_inverse() {
        let matrix = array![[2.0, 1.0], [1.0, 1.0]];
        let inv = math::matrix_inverse(&matrix, 1e-12).unwrap();
        let identity = matrix.dot(&inv);

        // Check if result is close to identity matrix
        assert!((identity[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((identity[[1, 1]] - 1.0).abs() < 1e-10);
        assert!(identity[[0, 1]].abs() < 1e-10);
        assert!(identity[[1, 0]].abs() < 1e-10);
    }

    #[test]
    fn test_qr_decomposition() {
        let matrix = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let (q, r) = math::qr_decomposition(&matrix).unwrap();
        let reconstructed = q.dot(&r);

        // Check reconstruction accuracy
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!((matrix[[i, j]] - reconstructed[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_covariance_matrix() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let cov = stats::covariance_matrix(&data, 1).unwrap();

        // Check symmetry
        assert!((cov[[0, 1]] - cov[[1, 0]]).abs() < 1e-10);

        // Check positive diagonal
        assert!(cov[[0, 0]] > 0.0);
        assert!(cov[[1, 1]] > 0.0);
    }

    #[test]
    fn test_standardization() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let (standardized, mean, std_dev) = preprocessing::standardize(&data).unwrap();

        // Check that standardized data has zero mean
        let new_mean = standardized.mean_axis(Axis(0)).unwrap();
        assert!(new_mean[0].abs() < 1e-10);
        assert!(new_mean[1].abs() < 1e-10);

        // Check that original mean is correct
        assert!((mean[0] - 3.0).abs() < 1e-10);
        assert!((mean[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance_functions() {
        let x1 = array![1.0, 2.0, 3.0];
        let x2 = array![4.0, 5.0, 6.0];

        let euclidean = distance::euclidean_distance(&x1.view(), &x2.view()).unwrap();
        let manhattan = distance::manhattan_distance(&x1.view(), &x2.view()).unwrap();
        let cosine = distance::cosine_distance(&x1.view(), &x2.view()).unwrap();

        assert!(euclidean > 0.0);
        assert!(manhattan > 0.0);
        assert!(cosine >= 0.0 && cosine <= 2.0);

        // Manhattan distance should be larger than Euclidean for this case
        assert!(manhattan > euclidean);
    }

    #[test]
    fn test_cross_validation_split() {
        let folds = cross_validation::kfold_split(10, 5, false, None).unwrap();

        assert_eq!(folds.len(), 5);

        // Check that all indices are covered exactly once in test sets
        let mut all_test_indices = Vec::new();
        for (_, test) in &folds {
            all_test_indices.extend(test);
        }
        all_test_indices.sort();

        let expected: Vec<usize> = (0..10).collect();
        assert_eq!(all_test_indices, expected);
    }

    #[test]
    fn test_validation_functions() {
        let data1 = Array2::zeros((10, 5));
        let data2 = Array2::zeros((10, 3));
        let data3 = Array2::zeros((8, 5));

        // Same number of samples - should pass
        assert!(validation::check_consistent_samples(&[&data1, &data2]).is_ok());

        // Different number of samples - should fail
        assert!(validation::check_consistent_samples(&[&data1, &data3]).is_err());

        // Valid labels
        let valid_labels = array![0, 1, 0, 1, 2];
        assert!(validation::check_valid_labels(&valid_labels).is_ok());

        // Invalid labels (only one class)
        let invalid_labels = array![1, 1, 1, 1];
        assert!(validation::check_valid_labels(&invalid_labels).is_err());
    }

    #[test]
    fn test_performance_utilities() {
        let timer = performance::Timer::new();
        std::thread::sleep(std::time::Duration::from_millis(1));
        assert!(timer.elapsed_ms() >= 1);

        // Test memory estimation
        let memory = performance::estimate_memory_usage(&[100, 50]);
        assert_eq!(memory, 100 * 50 * std::mem::size_of::<Float>());

        // Test memory feasibility
        let small_arrays = vec![&[10usize, 10][..], &[5, 5][..]];
        assert!(performance::check_memory_feasibility(&small_arrays, 1)); // 1 MB should be enough
    }
}