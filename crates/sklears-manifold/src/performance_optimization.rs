use scirs2_core::ndarray::{Array1, Array2};
use std::alloc::{alloc_zeroed, dealloc, Layout};

pub struct CacheFriendlyMatrix<T> {
    data: *mut T,
    layout: Layout,
    rows: usize,
    cols: usize,
    row_stride: usize,
}

impl<T> CacheFriendlyMatrix<T>
where
    T: Copy + Clone + Default,
{
    pub fn new(rows: usize, cols: usize) -> Result<Self, &'static str> {
        if rows == 0 || cols == 0 {
            return Err("Matrix dimensions must be positive");
        }

        let cache_line_size = 64; // bytes
        let element_size = std::mem::size_of::<T>();
        let elements_per_cache_line = cache_line_size / element_size;

        // Pad row stride to cache line boundary
        let row_stride = ((cols + elements_per_cache_line - 1) / elements_per_cache_line)
            * elements_per_cache_line;

        let total_elements = rows * row_stride;
        let layout = Layout::array::<T>(total_elements).map_err(|_| "Layout error")?;

        let data = unsafe { alloc_zeroed(layout) as *mut T };
        if data.is_null() {
            return Err("Allocation failed");
        }

        Ok(Self {
            data,
            layout,
            rows,
            cols,
            row_stride,
        })
    }

    pub unsafe fn get_unchecked(&self, row: usize, col: usize) -> &T {
        let ptr = self.data.add(row * self.row_stride + col);
        &*ptr
    }

    pub unsafe fn get_unchecked_mut(&mut self, row: usize, col: usize) -> &mut T {
        let ptr = self.data.add(row * self.row_stride + col);
        &mut *ptr
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.rows && col < self.cols {
            Some(unsafe { self.get_unchecked(row, col) })
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row < self.rows && col < self.cols {
            Some(unsafe { self.get_unchecked_mut(row, col) })
        } else {
            None
        }
    }

    pub fn set(&mut self, row: usize, col: usize, value: T) -> Result<(), &'static str> {
        if row >= self.rows || col >= self.cols {
            return Err("Index out of bounds");
        }
        unsafe {
            *self.get_unchecked_mut(row, col) = value;
        }
        Ok(())
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn to_ndarray(&self) -> Array2<T> {
        let mut result = Array2::default((self.rows, self.cols));
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[[i, j]] = *unsafe { self.get_unchecked(i, j) };
            }
        }
        result
    }

    pub fn from_ndarray(array: &Array2<T>) -> Result<Self, &'static str> {
        let mut result = Self::new(array.nrows(), array.ncols())?;
        for i in 0..array.nrows() {
            for j in 0..array.ncols() {
                result.set(i, j, array[[i, j]])?;
            }
        }
        Ok(result)
    }
}

impl<T> Drop for CacheFriendlyMatrix<T> {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.data as *mut u8, self.layout);
        }
    }
}

unsafe impl<T: Send> Send for CacheFriendlyMatrix<T> {}
unsafe impl<T: Sync> Sync for CacheFriendlyMatrix<T> {}

pub struct UnsafeDistanceComputer;

impl Default for UnsafeDistanceComputer {
    fn default() -> Self {
        Self::new()
    }
}

impl UnsafeDistanceComputer {
    pub fn new() -> Self {
        Self
    }

    pub unsafe fn euclidean_distance_unchecked(a: *const f64, b: *const f64, len: usize) -> f64 {
        let mut sum = 0.0;
        let mut i = 0;

        // Process 4 elements at a time for better performance
        while i + 4 <= len {
            let diff1 = *a.add(i) - *b.add(i);
            let diff2 = *a.add(i + 1) - *b.add(i + 1);
            let diff3 = *a.add(i + 2) - *b.add(i + 2);
            let diff4 = *a.add(i + 3) - *b.add(i + 3);

            sum += diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4;
            i += 4;
        }

        // Process remaining elements
        while i < len {
            let diff = *a.add(i) - *b.add(i);
            sum += diff * diff;
            i += 1;
        }

        sum.sqrt()
    }

    pub fn fast_pairwise_distances(&self, data: &Array2<f64>) -> Array2<f64> {
        let n_samples = data.nrows();
        let n_features = data.ncols();
        let mut distances: Array2<f64> = Array2::zeros((n_samples, n_samples));

        unsafe {
            for i in 0..n_samples {
                let row_i = data.as_ptr().add(i * n_features);
                for j in i + 1..n_samples {
                    let row_j = data.as_ptr().add(j * n_features);
                    let dist = Self::euclidean_distance_unchecked(row_i, row_j, n_features);

                    // Access matrix elements directly using raw pointers
                    let dist_ptr = distances.as_mut_ptr().add(i * n_samples + j);
                    *dist_ptr = dist;
                    let sym_dist_ptr = distances.as_mut_ptr().add(j * n_samples + i);
                    *sym_dist_ptr = dist;
                }
            }
        }

        distances
    }

    pub unsafe fn dot_product_unchecked(a: *const f64, b: *const f64, len: usize) -> f64 {
        let mut sum = 0.0;
        let mut i = 0;

        // Unroll loop for better performance
        while i + 8 <= len {
            sum += *a.add(i) * *b.add(i)
                + *a.add(i + 1) * *b.add(i + 1)
                + *a.add(i + 2) * *b.add(i + 2)
                + *a.add(i + 3) * *b.add(i + 3)
                + *a.add(i + 4) * *b.add(i + 4)
                + *a.add(i + 5) * *b.add(i + 5)
                + *a.add(i + 6) * *b.add(i + 6)
                + *a.add(i + 7) * *b.add(i + 7);
            i += 8;
        }

        while i < len {
            sum += *a.add(i) * *b.add(i);
            i += 1;
        }

        sum
    }

    pub fn fast_matrix_multiply(
        &self,
        a: &Array2<f64>,
        b: &Array2<f64>,
    ) -> Result<Array2<f64>, &'static str> {
        if a.ncols() != b.nrows() {
            return Err("Matrix dimensions do not match for multiplication");
        }

        let m = a.nrows();
        let n = a.ncols();
        let p = b.ncols();

        let mut result: Array2<f64> = Array2::zeros((m, p));

        unsafe {
            for i in 0..m {
                let a_row = a.as_ptr().add(i * n);
                for j in 0..p {
                    let mut sum = 0.0;
                    for k in 0..n {
                        let b_elem = *b.as_ptr().add(k * p + j);
                        sum += *a_row.add(k) * b_elem;
                    }
                    *result.as_mut_ptr().add(i * p + j) = sum;
                }
            }
        }

        Ok(result)
    }
}

pub struct BlockedMatrixMultiply {
    block_size: usize,
}

impl BlockedMatrixMultiply {
    pub fn new(block_size: usize) -> Self {
        Self { block_size }
    }

    pub fn multiply(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, &'static str> {
        if a.ncols() != b.nrows() {
            return Err("Matrix dimensions do not match");
        }

        let m = a.nrows();
        let n = a.ncols();
        let p = b.ncols();

        let mut result: Array2<f64> = Array2::zeros((m, p));

        // Block-based multiplication for better cache locality
        for i_block in (0..m).step_by(self.block_size) {
            for j_block in (0..p).step_by(self.block_size) {
                for k_block in (0..n).step_by(self.block_size) {
                    let i_end = (i_block + self.block_size).min(m);
                    let j_end = (j_block + self.block_size).min(p);
                    let k_end = (k_block + self.block_size).min(n);

                    for i in i_block..i_end {
                        for j in j_block..j_end {
                            let mut sum = result[[i, j]];
                            for k in k_block..k_end {
                                sum += a[[i, k]] * b[[k, j]];
                            }
                            result[[i, j]] = sum;
                        }
                    }
                }
            }
        }

        Ok(result)
    }
}

pub struct PrefetchOptimizedOperations;

impl PrefetchOptimizedOperations {
    pub fn optimized_vector_sum(data: &Array2<f64>) -> Array1<f64> {
        let n_rows = data.nrows();
        let n_cols = data.ncols();
        let mut result: Array1<f64> = Array1::zeros(n_cols);

        unsafe {
            for j in 0..n_cols {
                let mut sum = 0.0;
                let mut i = 0;

                while i + 4 < n_rows {
                    // Prefetch next cache line
                    if i + 8 < n_rows {
                        let prefetch_addr = data.as_ptr().add((i + 8) * n_cols + j);
                        #[cfg(target_arch = "x86_64")]
                        {
                            std::arch::x86_64::_mm_prefetch(
                                prefetch_addr as *const i8,
                                std::arch::x86_64::_MM_HINT_T0,
                            );
                        }
                    }

                    sum += *data.as_ptr().add(i * n_cols + j)
                        + *data.as_ptr().add((i + 1) * n_cols + j)
                        + *data.as_ptr().add((i + 2) * n_cols + j)
                        + *data.as_ptr().add((i + 3) * n_cols + j);
                    i += 4;
                }

                while i < n_rows {
                    sum += *data.as_ptr().add(i * n_cols + j);
                    i += 1;
                }

                *result.as_mut_ptr().add(j) = sum;
            }
        }

        result
    }
}

pub fn profile_guided_optimization_hint() {
    // This function would normally contain profile-guided optimization hints
    // For now, it's a placeholder for future PGO implementation
    #[cfg(target_feature = "likely")]
    {
        // Use likely/unlikely hints when available
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_cache_friendly_matrix_creation() {
        let matrix = CacheFriendlyMatrix::<f64>::new(10, 20);
        assert!(matrix.is_ok());

        let matrix = matrix.unwrap();
        assert_eq!(matrix.rows(), 10);
        assert_eq!(matrix.cols(), 20);
    }

    #[test]
    fn test_cache_friendly_matrix_access() {
        let mut matrix = CacheFriendlyMatrix::<f64>::new(5, 5).unwrap();

        matrix.set(2, 3, 42.0).unwrap();
        let value = matrix.get(2, 3).unwrap();
        assert_eq!(*value, 42.0);
    }

    #[test]
    fn test_cache_friendly_matrix_conversion() {
        let ndarray = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let matrix = CacheFriendlyMatrix::from_ndarray(&ndarray).unwrap();
        let converted_back = matrix.to_ndarray();

        assert_eq!(ndarray, converted_back);
    }

    #[test]
    fn test_unsafe_distance_computer() {
        let computer = UnsafeDistanceComputer::new();
        let data = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap();

        let distances = computer.fast_pairwise_distances(&data);

        assert_eq!(distances.shape(), &[3, 3]);
        assert!((distances[[0, 1]] - 1.0).abs() < 1e-10);
        assert!((distances[[0, 2]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_blocked_matrix_multiply() {
        let multiplier = BlockedMatrixMultiply::new(2);
        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let result = multiplier.multiply(&a, &b).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result[[0, 0]], 22.0);
        assert_eq!(result[[0, 1]], 28.0);
        assert_eq!(result[[1, 0]], 49.0);
        assert_eq!(result[[1, 1]], 64.0);
    }

    #[test]
    fn test_optimized_vector_sum() {
        let data = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();

        let result = PrefetchOptimizedOperations::optimized_vector_sum(&data);

        assert_eq!(result.len(), 4);
        assert_eq!(result[0], 15.0); // 1 + 5 + 9
        assert_eq!(result[1], 18.0); // 2 + 6 + 10
        assert_eq!(result[2], 21.0); // 3 + 7 + 11
        assert_eq!(result[3], 24.0); // 4 + 8 + 12
    }

    #[test]
    fn test_fast_matrix_multiply() {
        let computer = UnsafeDistanceComputer::new();
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![5.0, 6.0, 7.0, 8.0]).unwrap();

        let result = computer.fast_matrix_multiply(&a, &b).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result[[0, 0]], 19.0); // 1*5 + 2*7
        assert_eq!(result[[0, 1]], 22.0); // 1*6 + 2*8
        assert_eq!(result[[1, 0]], 43.0); // 3*5 + 4*7
        assert_eq!(result[[1, 1]], 50.0); // 3*6 + 4*8
    }
}
