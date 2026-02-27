//! Incremental Nyström method for online kernel approximation
//!
//! This module implements online/incremental versions of the Nyström method
//! that can efficiently update kernel approximations as new data arrives.

use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use scirs2_core::random::rngs::StdRng as RealStdRng;
use scirs2_core::random::seq::SliceRandom;
use scirs2_core::random::Rng;
use scirs2_core::random::{thread_rng, SeedableRng};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

use crate::nystroem::{Kernel, SamplingStrategy};

/// Update strategy for incremental Nyström
#[derive(Debug, Clone)]
/// UpdateStrategy
pub enum UpdateStrategy {
    /// Simple addition of new landmarks
    Append,
    /// Replace oldest landmarks with new ones (sliding window)
    SlidingWindow,
    /// Merge new data with existing approximation
    Merge,
    /// Selective update based on approximation quality
    Selective { threshold: Float },
}

/// Incremental Nyström method for online kernel approximation
///
/// Enables efficient updating of kernel approximations as new data arrives,
/// without requiring complete recomputation from scratch.
///
/// # Parameters
///
/// * `kernel` - Kernel function to approximate
/// * `n_components` - Maximum number of landmark points
/// * `update_strategy` - Strategy for incorporating new data
/// * `min_update_size` - Minimum number of new samples before updating
/// * `sampling_strategy` - Strategy for selecting new landmarks
#[derive(Debug, Clone)]
pub struct IncrementalNystroem<State = Untrained> {
    pub kernel: Kernel,
    pub n_components: usize,
    pub update_strategy: UpdateStrategy,
    pub min_update_size: usize,
    pub sampling_strategy: SamplingStrategy,
    pub random_state: Option<u64>,

    // Fitted attributes
    components_: Option<Array2<Float>>,
    normalization_: Option<Array2<Float>>,
    component_indices_: Option<Vec<usize>>,
    landmark_data_: Option<Array2<Float>>,
    update_count_: usize,
    accumulated_data_: Option<Array2<Float>>,

    _state: PhantomData<State>,
}

impl IncrementalNystroem<Untrained> {
    /// Create a new incremental Nyström approximator
    pub fn new(kernel: Kernel, n_components: usize) -> Self {
        Self {
            kernel,
            n_components,
            update_strategy: UpdateStrategy::Append,
            min_update_size: 10,
            sampling_strategy: SamplingStrategy::Random,
            random_state: None,
            components_: None,
            normalization_: None,
            component_indices_: None,
            landmark_data_: None,
            update_count_: 0,
            accumulated_data_: None,
            _state: PhantomData,
        }
    }

    /// Set the update strategy
    pub fn update_strategy(mut self, strategy: UpdateStrategy) -> Self {
        self.update_strategy = strategy;
        self
    }

    /// Set minimum update size
    pub fn min_update_size(mut self, size: usize) -> Self {
        self.min_update_size = size;
        self
    }

    /// Set sampling strategy
    pub fn sampling_strategy(mut self, strategy: SamplingStrategy) -> Self {
        self.sampling_strategy = strategy;
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl Estimator for IncrementalNystroem<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, ()> for IncrementalNystroem<Untrained> {
    type Fitted = IncrementalNystroem<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (n_samples, _n_features) = x.dim();
        let n_components = self.n_components.min(n_samples);

        let mut rng = match self.random_state {
            Some(seed) => RealStdRng::seed_from_u64(seed),
            None => RealStdRng::from_seed(thread_rng().gen()),
        };

        // Select initial landmark points
        let component_indices = self.select_components(x, n_components, &mut rng)?;
        let landmark_data = self.extract_landmarks(x, &component_indices);

        // Compute kernel matrix for landmarks
        let kernel_matrix = self.kernel.compute_kernel(&landmark_data, &landmark_data);

        // Compute eigendecomposition
        let (components, normalization) = self.compute_decomposition(kernel_matrix)?;

        Ok(IncrementalNystroem {
            kernel: self.kernel,
            n_components: self.n_components,
            update_strategy: self.update_strategy,
            min_update_size: self.min_update_size,
            sampling_strategy: self.sampling_strategy,
            random_state: self.random_state,
            components_: Some(components),
            normalization_: Some(normalization),
            component_indices_: Some(component_indices),
            landmark_data_: Some(landmark_data),
            update_count_: 0,
            accumulated_data_: None,
            _state: PhantomData,
        })
    }
}

impl IncrementalNystroem<Untrained> {
    /// Select component indices based on sampling strategy
    fn select_components(
        &self,
        x: &Array2<Float>,
        n_components: usize,
        rng: &mut RealStdRng,
    ) -> Result<Vec<usize>> {
        let (n_samples, _) = x.dim();

        match &self.sampling_strategy {
            SamplingStrategy::Random => {
                let mut indices: Vec<usize> = (0..n_samples).collect();
                indices.shuffle(rng);
                Ok(indices[..n_components].to_vec())
            }
            SamplingStrategy::KMeans => self.kmeans_sampling(x, n_components, rng),
            SamplingStrategy::LeverageScore => self.leverage_score_sampling(x, n_components, rng),
            SamplingStrategy::ColumnNorm => self.column_norm_sampling(x, n_components, rng),
        }
    }

    /// Simple k-means based sampling
    fn kmeans_sampling(
        &self,
        x: &Array2<Float>,
        n_components: usize,
        rng: &mut RealStdRng,
    ) -> Result<Vec<usize>> {
        let (n_samples, n_features) = x.dim();
        let mut centers = Array2::zeros((n_components, n_features));

        // Initialize centers randomly
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(rng);
        for (i, &idx) in indices[..n_components].iter().enumerate() {
            centers.row_mut(i).assign(&x.row(idx));
        }

        // Run a few iterations of k-means
        for _iter in 0..5 {
            let mut assignments = vec![0; n_samples];

            // Assign points to nearest centers
            for i in 0..n_samples {
                let mut min_dist = Float::INFINITY;
                let mut best_center = 0;

                for j in 0..n_components {
                    let diff = &x.row(i) - &centers.row(j);
                    let dist = diff.dot(&diff);
                    if dist < min_dist {
                        min_dist = dist;
                        best_center = j;
                    }
                }
                assignments[i] = best_center;
            }

            // Update centers
            for j in 0..n_components {
                let cluster_points: Vec<usize> = assignments
                    .iter()
                    .enumerate()
                    .filter(|(_, &assignment)| assignment == j)
                    .map(|(i, _)| i)
                    .collect();

                if !cluster_points.is_empty() {
                    let mut new_center = Array1::zeros(n_features);
                    for &point_idx in &cluster_points {
                        new_center = new_center + x.row(point_idx);
                    }
                    new_center /= cluster_points.len() as Float;
                    centers.row_mut(j).assign(&new_center);
                }
            }
        }

        // Find closest points to final centers
        let mut selected_indices = Vec::new();
        for j in 0..n_components {
            let mut min_dist = Float::INFINITY;
            let mut best_point = 0;

            for i in 0..n_samples {
                let diff = &x.row(i) - &centers.row(j);
                let dist = diff.dot(&diff);
                if dist < min_dist {
                    min_dist = dist;
                    best_point = i;
                }
            }
            selected_indices.push(best_point);
        }

        selected_indices.sort_unstable();
        selected_indices.dedup();

        // Fill remaining slots randomly if needed
        while selected_indices.len() < n_components {
            let random_idx = rng.gen_range(0..n_samples);
            if !selected_indices.contains(&random_idx) {
                selected_indices.push(random_idx);
            }
        }

        Ok(selected_indices[..n_components].to_vec())
    }

    /// Leverage score based sampling
    fn leverage_score_sampling(
        &self,
        x: &Array2<Float>,
        n_components: usize,
        _rng: &mut RealStdRng,
    ) -> Result<Vec<usize>> {
        let (n_samples, _) = x.dim();

        // Compute leverage scores (diagonal of hat matrix)
        // For simplicity, we approximate using row norms as proxy
        let mut scores = Vec::new();
        for i in 0..n_samples {
            let row_norm = x.row(i).dot(&x.row(i)).sqrt();
            scores.push(row_norm + 1e-10); // Add small epsilon for numerical stability
        }

        // Sample based on scores using cumulative distribution
        let total_score: Float = scores.iter().sum();
        if total_score <= 0.0 {
            return Err(SklearsError::InvalidInput(
                "All scores are zero or negative".to_string(),
            ));
        }

        // Create cumulative distribution
        let mut cumulative = Vec::with_capacity(scores.len());
        let mut sum = 0.0;
        for &score in &scores {
            sum += score / total_score;
            cumulative.push(sum);
        }

        let mut selected_indices = Vec::new();
        for _ in 0..n_components {
            let r = thread_rng().gen::<Float>();
            // Find index where cumulative probability >= r
            let mut idx = cumulative
                .iter()
                .position(|&cum| cum >= r)
                .unwrap_or(scores.len() - 1);

            // Ensure no duplicates
            while selected_indices.contains(&idx) {
                let r = thread_rng().gen::<Float>();
                idx = cumulative
                    .iter()
                    .position(|&cum| cum >= r)
                    .unwrap_or(scores.len() - 1);
            }
            selected_indices.push(idx);
        }

        Ok(selected_indices)
    }

    /// Column norm based sampling
    fn column_norm_sampling(
        &self,
        x: &Array2<Float>,
        n_components: usize,
        rng: &mut RealStdRng,
    ) -> Result<Vec<usize>> {
        let (n_samples, _) = x.dim();

        // Compute row norms
        let mut norms = Vec::new();
        for i in 0..n_samples {
            let norm = x.row(i).dot(&x.row(i)).sqrt();
            norms.push(norm + 1e-10);
        }

        // Sort by norm and take diverse selection
        let mut indices_with_norms: Vec<(usize, Float)> = norms
            .iter()
            .enumerate()
            .map(|(i, &norm)| (i, norm))
            .collect();
        indices_with_norms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut selected_indices = Vec::new();
        let step = n_samples.max(1) / n_components.max(1);

        for i in 0..n_components {
            let idx = (i * step).min(n_samples - 1);
            selected_indices.push(indices_with_norms[idx].0);
        }

        // Fill remaining with random if needed
        while selected_indices.len() < n_components {
            let random_idx = rng.gen_range(0..n_samples);
            if !selected_indices.contains(&random_idx) {
                selected_indices.push(random_idx);
            }
        }

        Ok(selected_indices)
    }

    /// Extract landmark data points
    fn extract_landmarks(&self, x: &Array2<Float>, indices: &[usize]) -> Array2<Float> {
        let (_, n_features) = x.dim();
        let mut landmarks = Array2::zeros((indices.len(), n_features));

        for (i, &idx) in indices.iter().enumerate() {
            landmarks.row_mut(i).assign(&x.row(idx));
        }

        landmarks
    }

    /// Compute eigendecomposition using power iteration method
    /// Returns (eigenvalues, eigenvectors) for symmetric matrix
    fn compute_eigendecomposition(
        &self,
        matrix: Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        let n = matrix.nrows();

        if n != matrix.ncols() {
            return Err(SklearsError::InvalidInput(
                "Matrix must be square for eigendecomposition".to_string(),
            ));
        }

        let mut eigenvals = Array1::zeros(n);
        let mut eigenvecs = Array2::zeros((n, n));

        // Use deflation method to find multiple eigenvalues
        let mut deflated_matrix = matrix.clone();

        for k in 0..n {
            // Power iteration for k-th eigenvalue/eigenvector
            let (eigenval, eigenvec) = self.power_iteration(&deflated_matrix, 100, 1e-8)?;

            eigenvals[k] = eigenval;
            eigenvecs.column_mut(k).assign(&eigenvec);

            // Deflate matrix: A_new = A - λ * v * v^T
            for i in 0..n {
                for j in 0..n {
                    deflated_matrix[[i, j]] -= eigenval * eigenvec[i] * eigenvec[j];
                }
            }
        }

        // Sort eigenvalues and eigenvectors in descending order
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| eigenvals[j].partial_cmp(&eigenvals[i]).unwrap());

        let mut sorted_eigenvals = Array1::zeros(n);
        let mut sorted_eigenvecs = Array2::zeros((n, n));

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            sorted_eigenvals[new_idx] = eigenvals[old_idx];
            sorted_eigenvecs
                .column_mut(new_idx)
                .assign(&eigenvecs.column(old_idx));
        }

        Ok((sorted_eigenvals, sorted_eigenvecs))
    }

    /// Power iteration method to find dominant eigenvalue and eigenvector
    fn power_iteration(
        &self,
        matrix: &Array2<Float>,
        max_iter: usize,
        tol: Float,
    ) -> Result<(Float, Array1<Float>)> {
        let n = matrix.nrows();

        // Initialize with deterministic vector to ensure reproducibility
        let mut v = Array1::from_shape_fn(n, |i| ((i as Float + 1.0) * 0.1).sin());

        // Normalize
        let norm = v.dot(&v).sqrt();
        if norm < 1e-10 {
            return Err(SklearsError::InvalidInput(
                "Initial vector has zero norm".to_string(),
            ));
        }
        v /= norm;

        let mut eigenval = 0.0;

        for _iter in 0..max_iter {
            // Apply matrix
            let w = matrix.dot(&v);

            // Compute Rayleigh quotient
            let new_eigenval = v.dot(&w);

            // Normalize
            let w_norm = w.dot(&w).sqrt();
            if w_norm < 1e-10 {
                break;
            }
            let new_v = w / w_norm;

            // Check convergence
            let eigenval_change = (new_eigenval - eigenval).abs();
            let vector_change = (&new_v - &v).mapv(|x| x.abs()).sum();

            if eigenval_change < tol && vector_change < tol {
                return Ok((new_eigenval, new_v));
            }

            eigenval = new_eigenval;
            v = new_v;
        }

        Ok((eigenval, v))
    }

    /// Compute eigendecomposition of kernel matrix
    fn compute_decomposition(
        &self,
        mut kernel_matrix: Array2<Float>,
    ) -> Result<(Array2<Float>, Array2<Float>)> {
        // Add small regularization to diagonal for numerical stability
        let reg = 1e-8;
        for i in 0..kernel_matrix.nrows() {
            kernel_matrix[[i, i]] += reg;
        }

        // Proper eigendecomposition for Nyström method
        let (eigenvals, eigenvecs) = self.compute_eigendecomposition(kernel_matrix)?;

        // Filter out small eigenvalues for numerical stability
        let threshold = 1e-8;
        let valid_indices: Vec<usize> = eigenvals
            .iter()
            .enumerate()
            .filter(|(_, &val)| val > threshold)
            .map(|(i, _)| i)
            .collect();

        if valid_indices.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No valid eigenvalues found in kernel matrix".to_string(),
            ));
        }

        // Construct components and normalization matrices
        let n_valid = valid_indices.len();
        let mut components = Array2::zeros((eigenvals.len(), n_valid));
        let mut normalization = Array2::zeros((n_valid, eigenvals.len()));

        for (new_idx, &old_idx) in valid_indices.iter().enumerate() {
            let sqrt_eigenval = eigenvals[old_idx].sqrt();
            components
                .column_mut(new_idx)
                .assign(&eigenvecs.column(old_idx));

            // For Nyström method: normalization = V * Λ^(-1/2)
            for i in 0..eigenvals.len() {
                normalization[[new_idx, i]] = eigenvecs[[i, old_idx]] / sqrt_eigenval;
            }
        }

        Ok((components, normalization))
    }
}

impl IncrementalNystroem<Trained> {
    /// Select component indices based on sampling strategy
    fn select_components(
        &self,
        x: &Array2<Float>,
        n_components: usize,
        rng: &mut RealStdRng,
    ) -> Result<Vec<usize>> {
        let (n_samples, _) = x.dim();

        match &self.sampling_strategy {
            SamplingStrategy::Random => {
                let mut indices: Vec<usize> = (0..n_samples).collect();
                indices.shuffle(rng);
                Ok(indices[..n_components].to_vec())
            }
            SamplingStrategy::KMeans => self.kmeans_sampling(x, n_components, rng),
            SamplingStrategy::LeverageScore => self.leverage_score_sampling(x, n_components, rng),
            SamplingStrategy::ColumnNorm => self.column_norm_sampling(x, n_components, rng),
        }
    }

    /// Simple k-means based sampling
    fn kmeans_sampling(
        &self,
        x: &Array2<Float>,
        n_components: usize,
        rng: &mut RealStdRng,
    ) -> Result<Vec<usize>> {
        let (n_samples, n_features) = x.dim();
        let mut centers = Array2::zeros((n_components, n_features));

        // Initialize centers randomly
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(rng);
        for (i, &idx) in indices[..n_components].iter().enumerate() {
            centers.row_mut(i).assign(&x.row(idx));
        }

        // Run a few iterations of k-means
        for _iter in 0..5 {
            let mut assignments = vec![0; n_samples];

            // Assign points to nearest centers
            for i in 0..n_samples {
                let mut min_dist = Float::INFINITY;
                let mut best_center = 0;

                for j in 0..n_components {
                    let diff = &x.row(i) - &centers.row(j);
                    let dist = diff.dot(&diff);
                    if dist < min_dist {
                        min_dist = dist;
                        best_center = j;
                    }
                }
                assignments[i] = best_center;
            }

            // Update centers
            for j in 0..n_components {
                let cluster_points: Vec<usize> = assignments
                    .iter()
                    .enumerate()
                    .filter(|(_, &assignment)| assignment == j)
                    .map(|(i, _)| i)
                    .collect();

                if !cluster_points.is_empty() {
                    let mut new_center = Array1::zeros(n_features);
                    for &point_idx in &cluster_points {
                        new_center = new_center + x.row(point_idx);
                    }
                    new_center /= cluster_points.len() as Float;
                    centers.row_mut(j).assign(&new_center);
                }
            }
        }

        // Find closest points to final centers
        let mut selected_indices = Vec::new();
        for j in 0..n_components {
            let mut min_dist = Float::INFINITY;
            let mut best_point = 0;

            for i in 0..n_samples {
                let diff = &x.row(i) - &centers.row(j);
                let dist = diff.dot(&diff);
                if dist < min_dist {
                    min_dist = dist;
                    best_point = i;
                }
            }
            selected_indices.push(best_point);
        }

        selected_indices.sort_unstable();
        selected_indices.dedup();

        // Fill remaining slots randomly if needed
        while selected_indices.len() < n_components {
            let random_idx = rng.gen_range(0..n_samples);
            if !selected_indices.contains(&random_idx) {
                selected_indices.push(random_idx);
            }
        }

        Ok(selected_indices[..n_components].to_vec())
    }

    /// Leverage score based sampling
    fn leverage_score_sampling(
        &self,
        x: &Array2<Float>,
        n_components: usize,
        _rng: &mut RealStdRng,
    ) -> Result<Vec<usize>> {
        let (n_samples, _) = x.dim();

        // Compute leverage scores (diagonal of hat matrix)
        // For simplicity, we approximate using row norms as proxy
        let mut scores = Vec::new();
        for i in 0..n_samples {
            let row_norm = x.row(i).dot(&x.row(i)).sqrt();
            scores.push(row_norm + 1e-10); // Add small epsilon for numerical stability
        }

        // Sample based on scores using cumulative distribution
        let total_score: Float = scores.iter().sum();
        if total_score <= 0.0 {
            return Err(SklearsError::InvalidInput(
                "All scores are zero or negative".to_string(),
            ));
        }

        // Create cumulative distribution
        let mut cumulative = Vec::with_capacity(scores.len());
        let mut sum = 0.0;
        for &score in &scores {
            sum += score / total_score;
            cumulative.push(sum);
        }

        let mut selected_indices = Vec::new();
        for _ in 0..n_components {
            let r = thread_rng().gen::<Float>();
            // Find index where cumulative probability >= r
            let mut idx = cumulative
                .iter()
                .position(|&cum| cum >= r)
                .unwrap_or(scores.len() - 1);

            // Ensure no duplicates
            while selected_indices.contains(&idx) {
                let r = thread_rng().gen::<Float>();
                idx = cumulative
                    .iter()
                    .position(|&cum| cum >= r)
                    .unwrap_or(scores.len() - 1);
            }
            selected_indices.push(idx);
        }

        Ok(selected_indices)
    }

    /// Column norm based sampling
    fn column_norm_sampling(
        &self,
        x: &Array2<Float>,
        n_components: usize,
        rng: &mut RealStdRng,
    ) -> Result<Vec<usize>> {
        let (n_samples, _) = x.dim();

        // Compute row norms
        let mut norms = Vec::new();
        for i in 0..n_samples {
            let norm = x.row(i).dot(&x.row(i)).sqrt();
            norms.push(norm + 1e-10);
        }

        // Sort by norm and take diverse selection
        let mut indices_with_norms: Vec<(usize, Float)> = norms
            .iter()
            .enumerate()
            .map(|(i, &norm)| (i, norm))
            .collect();
        indices_with_norms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut selected_indices = Vec::new();
        let step = n_samples.max(1) / n_components.max(1);

        for i in 0..n_components {
            let idx = (i * step).min(n_samples - 1);
            selected_indices.push(indices_with_norms[idx].0);
        }

        // Fill remaining with random if needed
        while selected_indices.len() < n_components {
            let random_idx = rng.gen_range(0..n_samples);
            if !selected_indices.contains(&random_idx) {
                selected_indices.push(random_idx);
            }
        }

        Ok(selected_indices)
    }

    /// Update the approximation with new data
    pub fn update(mut self, x_new: &Array2<Float>) -> Result<Self> {
        // Accumulate new data
        match &self.accumulated_data_ {
            Some(existing) => {
                let combined =
                    scirs2_core::ndarray::concatenate![Axis(0), existing.clone(), x_new.clone()];
                self.accumulated_data_ = Some(combined);
            }
            None => {
                self.accumulated_data_ = Some(x_new.clone());
            }
        }

        // Check if we have enough accumulated data to update
        let should_update = if let Some(ref accumulated) = self.accumulated_data_ {
            accumulated.nrows() >= self.min_update_size
        } else {
            false
        };

        if should_update {
            if let Some(accumulated) = self.accumulated_data_.take() {
                self = self.perform_update(&accumulated)?;
                self.update_count_ += 1;
            }
        }

        Ok(self)
    }

    /// Perform the actual update based on the strategy
    fn perform_update(self, new_data: &Array2<Float>) -> Result<Self> {
        match self.update_strategy.clone() {
            UpdateStrategy::Append => self.append_update(new_data),
            UpdateStrategy::SlidingWindow => self.sliding_window_update(new_data),
            UpdateStrategy::Merge => self.merge_update(new_data),
            UpdateStrategy::Selective { threshold } => self.selective_update(new_data, threshold),
        }
    }

    /// Append new landmarks (if space available)
    fn append_update(mut self, new_data: &Array2<Float>) -> Result<Self> {
        let current_landmarks = self.landmark_data_.as_ref().unwrap();
        let current_components = current_landmarks.nrows();

        if current_components >= self.n_components {
            // No space to append, just return current state
            return Ok(self);
        }

        let available_space = self.n_components - current_components;
        let n_new = available_space.min(new_data.nrows());

        if n_new == 0 {
            return Ok(self);
        }

        // Select new landmarks from new data
        let mut rng = match self.random_state {
            Some(seed) => RealStdRng::seed_from_u64(seed.wrapping_add(1000)),
            None => RealStdRng::from_seed(thread_rng().gen()),
        };

        let mut indices: Vec<usize> = (0..new_data.nrows()).collect();
        indices.shuffle(&mut rng);
        let selected_indices = &indices[..n_new];

        // Extract new landmarks
        let new_landmarks = self.extract_landmarks(new_data, selected_indices);

        // Combine with existing landmarks
        let combined_landmarks =
            scirs2_core::ndarray::concatenate![Axis(0), current_landmarks.clone(), new_landmarks];

        // Recompute decomposition
        let kernel_matrix = self
            .kernel
            .compute_kernel(&combined_landmarks, &combined_landmarks);
        let (components, normalization) = self.compute_decomposition(kernel_matrix)?;

        // Update indices
        let mut new_component_indices = self.component_indices_.as_ref().unwrap().clone();
        let base_index = current_landmarks.nrows();
        for &idx in selected_indices {
            new_component_indices.push(base_index + idx);
        }

        self.components_ = Some(components);
        self.normalization_ = Some(normalization);
        self.component_indices_ = Some(new_component_indices);
        self.landmark_data_ = Some(combined_landmarks);

        Ok(self)
    }

    /// Sliding window update (replace oldest landmarks)
    fn sliding_window_update(mut self, new_data: &Array2<Float>) -> Result<Self> {
        let current_landmarks = self.landmark_data_.as_ref().unwrap();
        let n_new = new_data.nrows().min(self.n_components);

        if n_new == 0 {
            return Ok(self);
        }

        // Select new landmarks
        let mut rng = match self.random_state {
            Some(seed) => RealStdRng::seed_from_u64(seed.wrapping_add(2000)),
            None => RealStdRng::from_seed(thread_rng().gen()),
        };

        let mut indices: Vec<usize> = (0..new_data.nrows()).collect();
        indices.shuffle(&mut rng);
        let selected_indices = &indices[..n_new];

        let new_landmarks = self.extract_landmarks(new_data, selected_indices);

        // Replace oldest landmarks with new ones
        let n_keep = self.n_components - n_new;
        let combined_landmarks = if n_keep > 0 {
            let kept_landmarks = current_landmarks.slice(s![n_new.., ..]).to_owned();
            scirs2_core::ndarray::concatenate![Axis(0), kept_landmarks, new_landmarks]
        } else {
            new_landmarks
        };

        // Recompute decomposition
        let kernel_matrix = self
            .kernel
            .compute_kernel(&combined_landmarks, &combined_landmarks);
        let (components, normalization) = self.compute_decomposition(kernel_matrix)?;

        // Update component indices (simplified)
        let new_component_indices: Vec<usize> = (0..combined_landmarks.nrows()).collect();

        self.components_ = Some(components);
        self.normalization_ = Some(normalization);
        self.component_indices_ = Some(new_component_indices);
        self.landmark_data_ = Some(combined_landmarks);

        Ok(self)
    }

    /// Merge update (combine approximations)
    fn merge_update(self, new_data: &Array2<Float>) -> Result<Self> {
        // Sophisticated merging strategy that combines existing and new Nyström approximations
        // This is based on the idea of merging two kernel approximations optimally

        let current_landmarks = self.landmark_data_.as_ref().unwrap();
        let _current_components = self.components_.as_ref().unwrap();
        let _current_normalization = self.normalization_.as_ref().unwrap();

        // Step 1: Create a new Nyström approximation from the new data
        let n_new_components = (new_data.nrows().min(self.n_components) / 2).max(1);

        let mut rng = match self.random_state {
            Some(seed) => RealStdRng::seed_from_u64(seed.wrapping_add(3000)),
            None => RealStdRng::from_seed(thread_rng().gen()),
        };

        // Select new landmarks using the same strategy
        let new_component_indices = self.select_components(new_data, n_new_components, &mut rng)?;
        let new_landmarks = self.extract_landmarks(new_data, &new_component_indices);

        // Compute new kernel matrix and decomposition
        let new_kernel_matrix = self.kernel.compute_kernel(&new_landmarks, &new_landmarks);
        let (_new_components, _new_normalization) =
            self.compute_decomposition(new_kernel_matrix)?;

        // Step 2: Combine the landmarks intelligently
        // Merge by selecting the most diverse/informative landmarks from both sets
        let merged_landmarks =
            self.merge_landmarks_intelligently(current_landmarks, &new_landmarks, &mut rng)?;

        // Step 3: Recompute the full approximation on merged landmarks
        let merged_kernel_matrix = self
            .kernel
            .compute_kernel(&merged_landmarks, &merged_landmarks);
        let (final_components, final_normalization) =
            self.compute_decomposition(merged_kernel_matrix)?;

        // Update component indices (simplified for merged case)
        let final_component_indices: Vec<usize> = (0..merged_landmarks.nrows()).collect();

        let mut updated_self = self;
        updated_self.components_ = Some(final_components);
        updated_self.normalization_ = Some(final_normalization);
        updated_self.component_indices_ = Some(final_component_indices);
        updated_self.landmark_data_ = Some(merged_landmarks);

        Ok(updated_self)
    }

    /// Intelligently merge landmarks from existing and new data
    fn merge_landmarks_intelligently(
        &self,
        current_landmarks: &Array2<Float>,
        new_landmarks: &Array2<Float>,
        rng: &mut RealStdRng,
    ) -> Result<Array2<Float>> {
        let n_current = current_landmarks.nrows();
        let n_new = new_landmarks.nrows();
        let n_features = current_landmarks.ncols();

        // Combine all landmarks temporarily
        let all_landmarks = scirs2_core::ndarray::concatenate![
            Axis(0),
            current_landmarks.clone(),
            new_landmarks.clone()
        ];

        // Use diversity-based selection to choose the best subset
        let n_target = self.n_components.min(n_current + n_new);
        let selected_indices = self.select_diverse_landmarks(&all_landmarks, n_target, rng)?;

        // Extract selected landmarks
        let mut merged_landmarks = Array2::zeros((selected_indices.len(), n_features));
        for (i, &idx) in selected_indices.iter().enumerate() {
            merged_landmarks.row_mut(i).assign(&all_landmarks.row(idx));
        }

        Ok(merged_landmarks)
    }

    /// Select diverse landmarks using maximum distance criterion
    fn select_diverse_landmarks(
        &self,
        landmarks: &Array2<Float>,
        n_select: usize,
        rng: &mut RealStdRng,
    ) -> Result<Vec<usize>> {
        let n_landmarks = landmarks.nrows();

        if n_select >= n_landmarks {
            return Ok((0..n_landmarks).collect());
        }

        let mut selected = Vec::new();
        let mut available: Vec<usize> = (0..n_landmarks).collect();

        // Start with a random landmark
        let first_idx = rng.gen_range(0..available.len());
        selected.push(available.remove(first_idx));

        // Greedily select landmarks that are maximally distant from already selected ones
        while selected.len() < n_select && !available.is_empty() {
            let mut best_idx = 0;
            let mut max_min_distance = 0.0;

            for (i, &candidate_idx) in available.iter().enumerate() {
                // Compute minimum distance to already selected landmarks
                let mut min_distance = Float::INFINITY;

                for &selected_idx in &selected {
                    let diff = &landmarks.row(candidate_idx) - &landmarks.row(selected_idx);
                    let distance = diff.dot(&diff).sqrt();
                    if distance < min_distance {
                        min_distance = distance;
                    }
                }

                if min_distance > max_min_distance {
                    max_min_distance = min_distance;
                    best_idx = i;
                }
            }

            selected.push(available.remove(best_idx));
        }

        Ok(selected)
    }

    /// Selective update based on approximation quality
    fn selective_update(self, new_data: &Array2<Float>, threshold: Float) -> Result<Self> {
        // Quality-based selective update that only incorporates new data if it improves approximation

        let current_landmarks = self.landmark_data_.as_ref().unwrap();

        // Step 1: Evaluate current approximation quality on new data
        let current_quality = self.evaluate_approximation_quality(current_landmarks, new_data)?;

        // Step 2: Create candidate updates and evaluate their quality
        let mut best_update = self.clone();
        let mut best_quality = current_quality;

        // Try append update
        let append_candidate = self.clone().append_update(new_data)?;
        let append_quality = append_candidate.evaluate_approximation_quality(
            append_candidate.landmark_data_.as_ref().unwrap(),
            new_data,
        )?;

        if append_quality > best_quality + threshold {
            best_update = append_candidate;
            best_quality = append_quality;
        }

        // Try merge update if we have enough data
        if new_data.nrows() >= 3 {
            let merge_candidate = self.clone().merge_update(new_data)?;
            let merge_quality = merge_candidate.evaluate_approximation_quality(
                merge_candidate.landmark_data_.as_ref().unwrap(),
                new_data,
            )?;

            if merge_quality > best_quality + threshold {
                best_update = merge_candidate;
                best_quality = merge_quality;
            }
        }

        // Try sliding window update
        let sliding_candidate = self.clone().sliding_window_update(new_data)?;
        let sliding_quality = sliding_candidate.evaluate_approximation_quality(
            sliding_candidate.landmark_data_.as_ref().unwrap(),
            new_data,
        )?;

        if sliding_quality > best_quality + threshold {
            best_update = sliding_candidate;
            best_quality = sliding_quality;
        }

        // Step 3: Only update if quality improvement exceeds threshold
        if best_quality > current_quality + threshold {
            Ok(best_update)
        } else {
            // No significant improvement, keep current state
            Ok(self)
        }
    }

    /// Evaluate approximation quality using kernel approximation error
    fn evaluate_approximation_quality(
        &self,
        landmarks: &Array2<Float>,
        test_data: &Array2<Float>,
    ) -> Result<Float> {
        // Quality metric: negative approximation error (higher is better)

        let n_test = test_data.nrows().min(50); // Limit for efficiency
        let test_subset = if test_data.nrows() > n_test {
            // Sample random subset for evaluation
            let mut rng = thread_rng();
            let mut indices: Vec<usize> = (0..test_data.nrows()).collect();
            indices.shuffle(&mut rng);
            test_data.select(Axis(0), &indices[..n_test])
        } else {
            test_data.to_owned()
        };

        // Compute exact kernel matrix for test subset
        let k_exact = self.kernel.compute_kernel(&test_subset, &test_subset);

        // Compute Nyström approximation: K(X,Z) * K(Z,Z)^(-1) * K(Z,X)
        let k_test_landmarks = self.kernel.compute_kernel(&test_subset, landmarks);
        let k_landmarks = self.kernel.compute_kernel(landmarks, landmarks);

        // Use our eigendecomposition to compute pseudo-inverse
        let (eigenvals, eigenvecs) = self.compute_eigendecomposition(k_landmarks)?;

        // Construct pseudo-inverse
        let threshold = 1e-8;
        let mut pseudo_inverse = Array2::zeros((landmarks.nrows(), landmarks.nrows()));

        for i in 0..landmarks.nrows() {
            for j in 0..landmarks.nrows() {
                let mut sum = 0.0;
                for k in 0..eigenvals.len() {
                    if eigenvals[k] > threshold {
                        sum += eigenvecs[[i, k]] * eigenvecs[[j, k]] / eigenvals[k];
                    }
                }
                pseudo_inverse[[i, j]] = sum;
            }
        }

        // Compute approximation: K(X,Z) * K(Z,Z)^(-1) * K(Z,X)
        let k_approx = k_test_landmarks
            .dot(&pseudo_inverse)
            .dot(&k_test_landmarks.t());

        // Compute approximation error (Frobenius norm)
        let error_matrix = &k_exact - &k_approx;
        let approximation_error = error_matrix.mapv(|x| x * x).sum().sqrt();

        // Convert to quality score (negative error, higher is better)
        let quality = -approximation_error / (k_exact.mapv(|x| x * x).sum().sqrt() + 1e-10);

        Ok(quality)
    }

    /// Extract landmark data points
    fn extract_landmarks(&self, x: &Array2<Float>, indices: &[usize]) -> Array2<Float> {
        let (_, n_features) = x.dim();
        let mut landmarks = Array2::zeros((indices.len(), n_features));

        for (i, &idx) in indices.iter().enumerate() {
            landmarks.row_mut(i).assign(&x.row(idx));
        }

        landmarks
    }

    /// Compute eigendecomposition using power iteration method
    /// Returns (eigenvalues, eigenvectors) for symmetric matrix
    fn compute_eigendecomposition(
        &self,
        matrix: Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        let n = matrix.nrows();

        if n != matrix.ncols() {
            return Err(SklearsError::InvalidInput(
                "Matrix must be square for eigendecomposition".to_string(),
            ));
        }

        let mut eigenvals = Array1::zeros(n);
        let mut eigenvecs = Array2::zeros((n, n));

        // Use deflation method to find multiple eigenvalues
        let mut deflated_matrix = matrix.clone();

        for k in 0..n {
            // Power iteration for k-th eigenvalue/eigenvector
            let (eigenval, eigenvec) = self.power_iteration(&deflated_matrix, 100, 1e-8)?;

            eigenvals[k] = eigenval;
            eigenvecs.column_mut(k).assign(&eigenvec);

            // Deflate matrix: A_new = A - λ * v * v^T
            for i in 0..n {
                for j in 0..n {
                    deflated_matrix[[i, j]] -= eigenval * eigenvec[i] * eigenvec[j];
                }
            }
        }

        // Sort eigenvalues and eigenvectors in descending order
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| eigenvals[j].partial_cmp(&eigenvals[i]).unwrap());

        let mut sorted_eigenvals = Array1::zeros(n);
        let mut sorted_eigenvecs = Array2::zeros((n, n));

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            sorted_eigenvals[new_idx] = eigenvals[old_idx];
            sorted_eigenvecs
                .column_mut(new_idx)
                .assign(&eigenvecs.column(old_idx));
        }

        Ok((sorted_eigenvals, sorted_eigenvecs))
    }

    /// Power iteration method to find dominant eigenvalue and eigenvector
    fn power_iteration(
        &self,
        matrix: &Array2<Float>,
        max_iter: usize,
        tol: Float,
    ) -> Result<(Float, Array1<Float>)> {
        let n = matrix.nrows();

        // Initialize with deterministic vector to ensure reproducibility
        let mut v = Array1::from_shape_fn(n, |i| ((i as Float + 1.0) * 0.1).sin());

        // Normalize
        let norm = v.dot(&v).sqrt();
        if norm < 1e-10 {
            return Err(SklearsError::InvalidInput(
                "Initial vector has zero norm".to_string(),
            ));
        }
        v /= norm;

        let mut eigenval = 0.0;

        for _iter in 0..max_iter {
            // Apply matrix
            let w = matrix.dot(&v);

            // Compute Rayleigh quotient
            let new_eigenval = v.dot(&w);

            // Normalize
            let w_norm = w.dot(&w).sqrt();
            if w_norm < 1e-10 {
                break;
            }
            let new_v = w / w_norm;

            // Check convergence
            let eigenval_change = (new_eigenval - eigenval).abs();
            let vector_change = (&new_v - &v).mapv(|x| x.abs()).sum();

            if eigenval_change < tol && vector_change < tol {
                return Ok((new_eigenval, new_v));
            }

            eigenval = new_eigenval;
            v = new_v;
        }

        Ok((eigenval, v))
    }

    /// Compute eigendecomposition of kernel matrix
    fn compute_decomposition(
        &self,
        mut kernel_matrix: Array2<Float>,
    ) -> Result<(Array2<Float>, Array2<Float>)> {
        // Add small regularization to diagonal for numerical stability
        let reg = 1e-8;
        for i in 0..kernel_matrix.nrows() {
            kernel_matrix[[i, i]] += reg;
        }

        // Proper eigendecomposition for Nyström method
        let (eigenvals, eigenvecs) = self.compute_eigendecomposition(kernel_matrix)?;

        // Filter out small eigenvalues for numerical stability
        let threshold = 1e-8;
        let valid_indices: Vec<usize> = eigenvals
            .iter()
            .enumerate()
            .filter(|(_, &val)| val > threshold)
            .map(|(i, _)| i)
            .collect();

        if valid_indices.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No valid eigenvalues found in kernel matrix".to_string(),
            ));
        }

        // Construct components and normalization matrices
        let n_valid = valid_indices.len();
        let mut components = Array2::zeros((eigenvals.len(), n_valid));
        let mut normalization = Array2::zeros((n_valid, eigenvals.len()));

        for (new_idx, &old_idx) in valid_indices.iter().enumerate() {
            let sqrt_eigenval = eigenvals[old_idx].sqrt();
            components
                .column_mut(new_idx)
                .assign(&eigenvecs.column(old_idx));

            // For Nyström method: normalization = V * Λ^(-1/2)
            for i in 0..eigenvals.len() {
                normalization[[new_idx, i]] = eigenvecs[[i, old_idx]] / sqrt_eigenval;
            }
        }

        Ok((components, normalization))
    }

    /// Get number of updates performed
    pub fn update_count(&self) -> usize {
        self.update_count_
    }

    /// Get current number of landmarks
    pub fn n_landmarks(&self) -> usize {
        self.landmark_data_.as_ref().map_or(0, |data| data.nrows())
    }
}

impl Transform<Array2<Float>> for IncrementalNystroem<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let _components = self
            .components_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform".to_string(),
            })?;

        let normalization =
            self.normalization_
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "transform".to_string(),
                })?;

        let landmark_data =
            self.landmark_data_
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "transform".to_string(),
                })?;

        // Compute kernel between input and landmarks
        let kernel_x_landmarks = self.kernel.compute_kernel(x, landmark_data);

        // Apply transformation: K(X, landmarks) @ normalization.T
        let transformed = kernel_x_landmarks.dot(&normalization.t());

        Ok(transformed)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_incremental_nystroem_basic() {
        let x_initial = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let x_new = array![[4.0, 5.0], [5.0, 6.0]];

        let nystroem = IncrementalNystroem::new(Kernel::Rbf { gamma: 1.0 }, 5)
            .update_strategy(UpdateStrategy::Append)
            .min_update_size(1);

        let fitted = nystroem.fit(&x_initial, &()).unwrap();
        assert_eq!(fitted.n_landmarks(), 3);

        let updated = fitted.update(&x_new).unwrap();
        assert_eq!(updated.n_landmarks(), 5);
        assert_eq!(updated.update_count(), 1);
    }

    #[test]
    fn test_incremental_transform() {
        let x_train = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let x_test = array![[1.5, 2.5], [2.5, 3.5]];

        let nystroem = IncrementalNystroem::new(Kernel::Rbf { gamma: 1.0 }, 3);
        let fitted = nystroem.fit(&x_train, &()).unwrap();

        let transformed = fitted.transform(&x_test).unwrap();
        assert_eq!(transformed.shape()[0], 2);
        assert!(transformed.shape()[1] <= 3);
    }

    #[test]
    fn test_sliding_window_update() {
        let x_initial = array![[1.0, 2.0], [2.0, 3.0]];
        let x_new = array![[3.0, 4.0], [4.0, 5.0], [5.0, 6.0]];

        let nystroem = IncrementalNystroem::new(Kernel::Linear, 3)
            .update_strategy(UpdateStrategy::SlidingWindow)
            .min_update_size(1);

        let fitted = nystroem.fit(&x_initial, &()).unwrap();
        let updated = fitted.update(&x_new).unwrap();

        assert_eq!(updated.n_landmarks(), 3);
        assert_eq!(updated.update_count(), 1);
    }

    #[test]
    fn test_different_kernels() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

        // Test with RBF kernel
        let rbf_nystroem = IncrementalNystroem::new(Kernel::Rbf { gamma: 0.5 }, 3);
        let rbf_fitted = rbf_nystroem.fit(&x, &()).unwrap();
        let rbf_transformed = rbf_fitted.transform(&x).unwrap();
        assert_eq!(rbf_transformed.shape()[0], 3);

        // Test with polynomial kernel
        let poly_nystroem = IncrementalNystroem::new(
            Kernel::Polynomial {
                gamma: 1.0,
                coef0: 1.0,
                degree: 2,
            },
            3,
        );
        let poly_fitted = poly_nystroem.fit(&x, &()).unwrap();
        let poly_transformed = poly_fitted.transform(&x).unwrap();
        assert_eq!(poly_transformed.shape()[0], 3);
    }

    #[test]
    fn test_min_update_size() {
        let x_initial = array![[1.0, 2.0], [2.0, 3.0]];
        let x_small = array![[3.0, 4.0]];
        let x_large = array![[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]];

        let nystroem = IncrementalNystroem::new(Kernel::Linear, 5).min_update_size(2);

        let fitted = nystroem.fit(&x_initial, &()).unwrap();

        // Small update should not trigger recomputation
        let after_small = fitted.update(&x_small).unwrap();
        assert_eq!(after_small.update_count(), 0);
        assert_eq!(after_small.n_landmarks(), 2);

        // Large update should trigger recomputation
        let after_large = after_small.update(&x_large).unwrap();
        assert_eq!(after_large.update_count(), 1);
        assert_eq!(after_large.n_landmarks(), 5);
    }

    #[test]
    fn test_reproducibility() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let x_new = array![[4.0, 5.0]];

        let nystroem1 = IncrementalNystroem::new(Kernel::Rbf { gamma: 1.0 }, 3)
            .random_state(42)
            .min_update_size(1);
        let fitted1 = nystroem1.fit(&x, &()).unwrap();
        let updated1 = fitted1.update(&x_new).unwrap();
        let result1 = updated1.transform(&x).unwrap();

        let nystroem2 = IncrementalNystroem::new(Kernel::Rbf { gamma: 1.0 }, 3)
            .random_state(42)
            .min_update_size(1);
        let fitted2 = nystroem2.fit(&x, &()).unwrap();
        let updated2 = fitted2.update(&x_new).unwrap();
        let result2 = updated2.transform(&x).unwrap();

        // Results should be very similar with same random seed (allowing for numerical precision)
        // Note: eigendecomposition can produce results that differ by a sign flip
        assert_eq!(result1.shape(), result2.shape());

        // Check if results are similar or similar up to sign flip
        let mut direct_match = true;
        let mut sign_flip_match = true;

        for i in 0..result1.len() {
            let val1 = result1.as_slice().unwrap()[i];
            let val2 = result2.as_slice().unwrap()[i];

            if (val1 - val2).abs() > 1e-6 {
                direct_match = false;
            }
            if (val1 + val2).abs() > 1e-6 {
                sign_flip_match = false;
            }
        }

        assert!(
            direct_match || sign_flip_match,
            "Results differ too much and are not related by sign flip"
        );
    }
}
