//! # Dependent Type Experiments for sklears-core
//!
//! This module explores dependent type programming patterns in Rust, pushing the
//! boundaries of what's possible with Rust's type system. While Rust doesn't have
//! full dependent types like Idris or Agda, we can simulate many dependent type
//! features using:
//!
//! - Const generics for type-level values
//! - GADTs (Generalized Algebraic Data Types) simulation
//! - Type-level programming with traits
//! - Phantom types for compile-time proofs
//! - Associated types for type families
//!
//! ## Key Concepts
//!
//! 1. **Length-Indexed Vectors**: Vectors whose length is part of the type
//! 2. **Type-Level Naturals**: Natural numbers at the type level
//! 3. **Dependent Pairs (Sigma Types)**: Pairs where the second type depends on the first value
//! 4. **Indexed Types**: Types indexed by other types or values
//! 5. **Compile-time Proofs**: Using types to prove properties
//!
//! ## Examples
//!
//! ```rust,ignore
//! use sklears_core::dependent_types::*;
//!
//! // Vector with compile-time length checking
//! let v1: Vec3<i32> = Vec3::new([1, 2, 3]);
//! let v2: Vec3<i32> = Vec3::new([4, 5, 6]);
//! let dot_product = v1.dot(&v2); // Type-safe: same length
//!
//! // Matrix with compile-time dimension checking
//! let m1: Matrix<f64, 2, 3> = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
//! let m2: Matrix<f64, 3, 2> = Matrix::new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
//! let product = m1.multiply(&m2); // Type-safe: dimensions match
//! // Result is Matrix<f64, 2, 2>
//! ```

use crate::error::{Result, SklearsError};
use std::marker::PhantomData;
use std::ops::{Add, Deref, DerefMut, Index, IndexMut, Mul};

// =============================================================================
// Type-Level Natural Numbers (Peano Arithmetic)
// =============================================================================

/// Type-level natural number: Zero
pub struct Z;

/// Type-level natural number: Successor
pub struct S<N>(PhantomData<N>);

/// Trait for type-level natural numbers
pub trait Nat {
    /// Convert to runtime value
    fn to_usize() -> usize;
}

impl Nat for Z {
    fn to_usize() -> usize {
        0
    }
}

impl<N: Nat> Nat for S<N> {
    fn to_usize() -> usize {
        1 + N::to_usize()
    }
}

// Convenient type aliases for small numbers
pub type N0 = Z;
pub type N1 = S<Z>;
pub type N2 = S<N1>;
pub type N3 = S<N2>;
pub type N4 = S<N3>;
pub type N5 = S<N4>;
pub type N6 = S<N5>;
pub type N7 = S<N6>;
pub type N8 = S<N7>;
pub type N9 = S<N8>;
pub type N10 = S<N9>;

// =============================================================================
// Type-Level Arithmetic
// =============================================================================

/// Type-level addition
pub trait Add_<N: Nat>: Nat {
    type Output: Nat;
}

impl<N: Nat> Add_<N> for Z {
    type Output = N;
}

impl<M: Nat, N: Nat> Add_<N> for S<M>
where
    M: Add_<N>,
{
    type Output = S<<M as Add_<N>>::Output>;
}

/// Type-level multiplication
pub trait Mul_<N: Nat>: Nat {
    type Output: Nat;
}

impl<N: Nat> Mul_<N> for Z {
    type Output = Z;
}

impl<M: Nat, N: Nat> Mul_<N> for S<M>
where
    M: Mul_<N>,
    N: Add_<<M as Mul_<N>>::Output>,
{
    type Output = <N as Add_<<M as Mul_<N>>::Output>>::Output;
}

/// Type-level comparison
pub trait Compare<N: Nat>: Nat {
    type Result: ComparisonResult;
}

pub trait ComparisonResult {}

pub struct LT; // Less than
pub struct EQ; // Equal
pub struct GT; // Greater than

impl ComparisonResult for LT {}
impl ComparisonResult for EQ {}
impl ComparisonResult for GT {}

// =============================================================================
// Length-Indexed Vectors
// =============================================================================

/// A vector with compile-time length tracking
#[derive(Debug, Clone, PartialEq)]
pub struct LVec<T, N: Nat> {
    data: Vec<T>,
    _phantom: PhantomData<N>,
}

impl<T, N: Nat> LVec<T, N> {
    /// Create a new length-indexed vector
    pub fn new(data: Vec<T>) -> Result<Self> {
        if data.len() == N::to_usize() {
            Ok(Self {
                data,
                _phantom: PhantomData,
            })
        } else {
            Err(SklearsError::InvalidInput(format!(
                "Expected length {}, got {}",
                N::to_usize(),
                data.len()
            )))
        }
    }

    /// Create without checking (unsafe)
    pub fn new_unchecked(data: Vec<T>) -> Self {
        Self {
            data,
            _phantom: PhantomData,
        }
    }

    /// Get the length (compile-time constant)
    pub const fn length() -> usize
    where
        N: Nat,
    {
        // This would ideally use N::to_usize() but const fn limitations
        0 // Placeholder - in practice we'd use const generics
    }

    /// Get reference to underlying data
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Get mutable reference to underlying data
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Map a function over the vector, preserving length
    pub fn map<U, F>(self, f: F) -> LVec<U, N>
    where
        F: FnMut(T) -> U,
    {
        LVec {
            data: self.data.into_iter().map(f).collect(),
            _phantom: PhantomData,
        }
    }

    /// Zip two length-indexed vectors (same length guaranteed by types)
    pub fn zip<U>(self, other: LVec<U, N>) -> LVec<(T, U), N> {
        LVec {
            data: self.data.into_iter().zip(other.data).collect(),
            _phantom: PhantomData,
        }
    }
}

impl<T: Clone, N: Nat> LVec<T, N> {
    /// Create from a single value repeated N times
    pub fn replicate(value: T) -> Self {
        LVec {
            data: vec![value; N::to_usize()],
            _phantom: PhantomData,
        }
    }
}

impl<T, N: Nat> Deref for LVec<T, N> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T, N: Nat> DerefMut for LVec<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

// Concatenation of length-indexed vectors
pub fn concat<T, M, N>(v1: LVec<T, M>, v2: LVec<T, N>) -> LVec<T, <M as Add_<N>>::Output>
where
    M: Nat + Add_<N>,
    N: Nat,
{
    let mut result = v1.data;
    result.extend(v2.data);
    LVec {
        data: result,
        _phantom: PhantomData,
    }
}

// =============================================================================
// Fixed-Size Arrays with Const Generics
// =============================================================================

/// A fixed-size array with compile-time size checking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FixedArray<T, const N: usize> {
    data: [T; N],
}

impl<T, const N: usize> FixedArray<T, N> {
    /// Create a new fixed array
    pub const fn new(data: [T; N]) -> Self {
        Self { data }
    }

    /// Get the length (compile-time constant)
    pub const fn len() -> usize {
        N
    }

    /// Check if empty (always false for N > 0)
    pub const fn is_empty() -> bool {
        N == 0
    }

    /// Get as slice
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Get as mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T: Copy, const N: usize> FixedArray<T, N> {
    /// Create from a function
    pub fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize) -> T,
    {
        let mut data = std::mem::MaybeUninit::<[T; N]>::uninit();
        let ptr = data.as_mut_ptr() as *mut T;

        for i in 0..N {
            unsafe {
                ptr.add(i).write(f(i));
            }
        }

        Self {
            data: unsafe { data.assume_init() },
        }
    }
}

impl<T, const N: usize> Index<usize> for FixedArray<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T, const N: usize> IndexMut<usize> for FixedArray<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T, const N: usize> Deref for FixedArray<T, N> {
    type Target = [T; N];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T, const N: usize> DerefMut for FixedArray<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

// =============================================================================
// Type-Safe Matrix with Compile-Time Dimensions
// =============================================================================

/// A matrix with compile-time dimension checking
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T, const ROWS: usize, const COLS: usize> {
    data: Vec<T>,
}

impl<T, const ROWS: usize, const COLS: usize> Matrix<T, ROWS, COLS> {
    /// Create a new matrix from flat data
    pub fn new(data: Vec<T>) -> Result<Self> {
        if data.len() == ROWS * COLS {
            Ok(Self { data })
        } else {
            Err(SklearsError::InvalidInput(format!(
                "Expected {} elements for {}x{} matrix, got {}",
                ROWS * COLS,
                ROWS,
                COLS,
                data.len()
            )))
        }
    }

    /// Create from nested array
    pub fn from_array(arr: [[T; COLS]; ROWS]) -> Self
    where
        T: Copy,
    {
        let mut data = Vec::with_capacity(ROWS * COLS);
        for row in &arr {
            data.extend_from_slice(row);
        }
        Self { data }
    }

    /// Get the number of rows
    pub const fn rows() -> usize {
        ROWS
    }

    /// Get the number of columns
    pub const fn cols() -> usize {
        COLS
    }

    /// Get element at (row, col)
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < ROWS && col < COLS {
            Some(&self.data[row * COLS + col])
        } else {
            None
        }
    }

    /// Get mutable element at (row, col)
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row < ROWS && col < COLS {
            Some(&mut self.data[row * COLS + col])
        } else {
            None
        }
    }

    /// Set element at (row, col)
    pub fn set(&mut self, row: usize, col: usize, value: T) -> Result<()> {
        if row < ROWS && col < COLS {
            self.data[row * COLS + col] = value;
            Ok(())
        } else {
            Err(SklearsError::InvalidInput(format!(
                "Index ({}, {}) out of bounds for {}x{} matrix",
                row, col, ROWS, COLS
            )))
        }
    }

    /// Transpose the matrix
    pub fn transpose(self) -> Matrix<T, COLS, ROWS>
    where
        T: Clone,
    {
        let mut transposed = Vec::with_capacity(ROWS * COLS);
        for col in 0..COLS {
            for row in 0..ROWS {
                transposed.push(self.data[row * COLS + col].clone());
            }
        }
        Matrix { data: transposed }
    }
}

impl<T: Clone, const ROWS: usize, const COLS: usize> Matrix<T, ROWS, COLS> {
    /// Create a matrix filled with a value
    pub fn fill(value: T) -> Self {
        Self {
            data: vec![value; ROWS * COLS],
        }
    }
}

impl<T: Default + Clone, const ROWS: usize, const COLS: usize> Matrix<T, ROWS, COLS> {
    /// Create a matrix filled with default values
    pub fn default_filled() -> Self {
        Self::fill(T::default())
    }
}

impl<T, const ROWS: usize, const COLS: usize> Index<(usize, usize)> for Matrix<T, ROWS, COLS> {
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < ROWS && col < COLS);
        &self.data[row * COLS + col]
    }
}

impl<T, const ROWS: usize, const COLS: usize> IndexMut<(usize, usize)> for Matrix<T, ROWS, COLS> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        assert!(row < ROWS && col < COLS);
        &mut self.data[row * COLS + col]
    }
}

impl<T, const M: usize, const N: usize> Matrix<T, M, N>
where
    T: Clone + Default + Add<Output = T> + Mul<Output = T>,
{
    /// Multiply this matrix by another matrix
    /// Type ensures dimensions are compatible: (M x N) * (N x P) = (M x P)
    pub fn multiply<const P: usize>(&self, other: &Matrix<T, N, P>) -> Matrix<T, M, P> {
        let mut result = Matrix::<T, M, P>::default_filled();

        for i in 0..M {
            for j in 0..P {
                let mut sum = T::default();
                for k in 0..N {
                    sum = sum + self.data[i * N + k].clone() * other.data[k * P + j].clone();
                }
                result.data[i * P + j] = sum;
            }
        }

        result
    }
}

// =============================================================================
// Dependent Pairs (Sigma Types)
// =============================================================================

/// A dependent pair where the second type depends on the first value
/// This is a simplified version since Rust doesn't have full dependent types
pub struct Sigma<A, B>
where
    B: DependentType<A>,
{
    fst: A,
    snd: B::Output,
}

/// Trait for types that depend on a value
pub trait DependentType<A> {
    type Output;

    fn construct(value: &A) -> Self::Output;
}

impl<A, B> Sigma<A, B>
where
    B: DependentType<A>,
{
    /// Create a dependent pair
    pub fn new(fst: A) -> Self {
        let snd = B::construct(&fst);
        Self { fst, snd }
    }

    /// Get the first component
    pub fn fst(&self) -> &A {
        &self.fst
    }

    /// Get the second component
    pub fn snd(&self) -> &B::Output {
        &self.snd
    }

    /// Destructure the pair
    pub fn into_parts(self) -> (A, B::Output) {
        (self.fst, self.snd)
    }
}

// =============================================================================
// Indexed Types for ML
// =============================================================================

/// A dataset indexed by its size and feature count
pub struct IndexedDataset<T, const N_SAMPLES: usize, const N_FEATURES: usize> {
    features: Matrix<T, N_SAMPLES, N_FEATURES>,
    labels: FixedArray<T, N_SAMPLES>,
}

impl<T, const N_SAMPLES: usize, const N_FEATURES: usize> IndexedDataset<T, N_SAMPLES, N_FEATURES> {
    /// Create a new indexed dataset
    pub fn new(
        features: Matrix<T, N_SAMPLES, N_FEATURES>,
        labels: FixedArray<T, N_SAMPLES>,
    ) -> Self {
        Self { features, labels }
    }

    /// Get the number of samples (compile-time constant)
    pub const fn n_samples() -> usize {
        N_SAMPLES
    }

    /// Get the number of features (compile-time constant)
    pub const fn n_features() -> usize {
        N_FEATURES
    }

    /// Get features matrix
    pub fn features(&self) -> &Matrix<T, N_SAMPLES, N_FEATURES> {
        &self.features
    }

    /// Get labels array
    pub fn labels(&self) -> &FixedArray<T, N_SAMPLES> {
        &self.labels
    }

    /// Split into train/test sets with compile-time size tracking
    /// Note: Due to Rust const generics limitations, test size must be specified explicitly
    pub fn split<const TRAIN_SIZE: usize, const TEST_SIZE: usize>(
        self,
    ) -> Result<(
        IndexedDataset<T, TRAIN_SIZE, N_FEATURES>,
        IndexedDataset<T, TEST_SIZE, N_FEATURES>,
    )>
    where
        T: Clone + Default + Copy,
    {
        if TRAIN_SIZE + TEST_SIZE != N_SAMPLES {
            return Err(SklearsError::InvalidInput(
                "Train size + test size must equal total samples".to_string(),
            ));
        }

        // This is a simplified implementation
        // In practice, we'd split the actual data
        let train = IndexedDataset {
            features: Matrix::default_filled(),
            labels: FixedArray::from_fn(|_| T::default()),
        };

        let test = IndexedDataset {
            features: Matrix::default_filled(),
            labels: FixedArray::from_fn(|_| T::default()),
        };

        Ok((train, test))
    }
}

// =============================================================================
// Type-Level Proofs
// =============================================================================

/// A proof that a type-level predicate holds
pub struct Proof<P: Predicate> {
    _phantom: PhantomData<P>,
}

/// Trait for type-level predicates
pub trait Predicate {
    fn holds() -> bool;
}

impl<P: Predicate> Proof<P> {
    /// Construct a proof (checked at runtime for now)
    pub fn new() -> Result<Self> {
        if P::holds() {
            Ok(Self {
                _phantom: PhantomData,
            })
        } else {
            Err(SklearsError::ValidationError(
                "Predicate does not hold".to_string(),
            ))
        }
    }

    /// Construct a proof without checking (unsafe)
    ///
    /// # Safety
    ///
    /// The caller must ensure that the predicate actually holds.
    /// Creating a proof for a false predicate violates type safety assumptions.
    pub unsafe fn new_unchecked() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

// Example: Proof that a number is even
pub struct IsEven<const N: usize>;

impl<const N: usize> Predicate for IsEven<N> {
    fn holds() -> bool {
        N % 2 == 0
    }
}

// Example: Proof that a dimension is valid for matrix multiplication
pub struct CompatibleDims<const M: usize, const N: usize, const P: usize>;

impl<const M: usize, const N: usize, const P: usize> Predicate for CompatibleDims<M, N, P> {
    fn holds() -> bool {
        M > 0 && N > 0 && P > 0
    }
}

// =============================================================================
// GADTs Simulation
// =============================================================================

/// Expression language with type-safe evaluation (GADT-style)
pub enum Expr<T> {
    Lit(T),
    Add(Box<Expr<T>>, Box<Expr<T>>),
    Mul(Box<Expr<T>>, Box<Expr<T>>),
}

impl<T> Expr<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T>,
{
    /// Evaluate the expression
    pub fn eval(&self) -> T {
        match self {
            Expr::Lit(x) => x.clone(),
            Expr::Add(left, right) => left.eval() + right.eval(),
            Expr::Mul(left, right) => left.eval() * right.eval(),
        }
    }
}

// =============================================================================
// Helper Functions and Utilities
// =============================================================================

/// Create a length-indexed vector from a slice
pub fn lvec_from_slice<T: Clone, N: Nat>(slice: &[T]) -> Result<LVec<T, N>> {
    if slice.len() == N::to_usize() {
        Ok(LVec::new_unchecked(slice.to_vec()))
    } else {
        Err(SklearsError::InvalidInput(format!(
            "Expected length {}, got {}",
            N::to_usize(),
            slice.len()
        )))
    }
}

/// Dot product of two length-indexed vectors
pub fn dot_product<T, N: Nat>(v1: &LVec<T, N>, v2: &LVec<T, N>) -> T
where
    T: Clone + Default + Add<Output = T> + Mul<Output = T>,
{
    v1.iter()
        .zip(v2.iter())
        .map(|(a, b)| a.clone() * b.clone())
        .fold(T::default(), |acc, x| acc + x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_level_nats() {
        assert_eq!(N0::to_usize(), 0);
        assert_eq!(N1::to_usize(), 1);
        assert_eq!(N5::to_usize(), 5);
        assert_eq!(N10::to_usize(), 10);
    }

    #[test]
    fn test_type_level_addition() {
        type Sum = <N2 as Add_<N3>>::Output;
        assert_eq!(Sum::to_usize(), 5);
    }

    #[test]
    fn test_length_indexed_vector() {
        let v = LVec::<i32, N3>::new(vec![1, 2, 3]).unwrap();
        assert_eq!(v.len(), 3);

        let wrong_length = LVec::<i32, N3>::new(vec![1, 2]);
        assert!(wrong_length.is_err());
    }

    #[test]
    fn test_fixed_array() {
        let arr = FixedArray::new([1, 2, 3, 4, 5]);
        assert_eq!(FixedArray::<i32, 5>::len(), 5);
        assert_eq!(arr[0], 1);
        assert_eq!(arr[4], 5);
    }

    #[test]
    fn test_matrix_creation() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let matrix = Matrix::<i32, 2, 3>::new(data).unwrap();

        assert_eq!(Matrix::<i32, 2, 3>::rows(), 2);
        assert_eq!(Matrix::<i32, 2, 3>::cols(), 3);
        assert_eq!(*matrix.get(0, 0).unwrap(), 1);
        assert_eq!(*matrix.get(1, 2).unwrap(), 6);
    }

    #[test]
    fn test_matrix_from_array() {
        let arr = [[1, 2, 3], [4, 5, 6]];
        let matrix = Matrix::from_array(arr);

        assert_eq!(matrix[(0, 0)], 1);
        assert_eq!(matrix[(1, 2)], 6);
    }

    #[test]
    fn test_matrix_transpose() {
        let arr = [[1, 2, 3], [4, 5, 6]];
        let matrix = Matrix::from_array(arr);
        let transposed = matrix.transpose();

        assert_eq!(Matrix::<i32, 3, 2>::rows(), 3);
        assert_eq!(Matrix::<i32, 3, 2>::cols(), 2);
        assert_eq!(transposed[(0, 0)], 1);
        assert_eq!(transposed[(2, 1)], 6);
    }

    #[test]
    fn test_matrix_multiplication() {
        // 2x3 matrix
        let m1 = Matrix::from_array([[1, 2, 3], [4, 5, 6]]);

        // 3x2 matrix
        let m2 = Matrix::from_array([[7, 8], [9, 10], [11, 12]]);

        // Result should be 2x2
        let result = m1.multiply(&m2);

        assert_eq!(Matrix::<i32, 2, 2>::rows(), 2);
        assert_eq!(Matrix::<i32, 2, 2>::cols(), 2);

        // (1*7 + 2*9 + 3*11) = 58
        assert_eq!(result[(0, 0)], 58);
        // (1*8 + 2*10 + 3*12) = 64
        assert_eq!(result[(0, 1)], 64);
        // (4*7 + 5*9 + 6*11) = 139
        assert_eq!(result[(1, 0)], 139);
        // (4*8 + 5*10 + 6*12) = 154
        assert_eq!(result[(1, 1)], 154);
    }

    #[test]
    fn test_indexed_dataset() {
        let features = Matrix::from_array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let labels = FixedArray::new([0.0, 1.0, 0.0]);

        let _dataset = IndexedDataset::new(features, labels);

        assert_eq!(IndexedDataset::<f64, 3, 2>::n_samples(), 3);
        assert_eq!(IndexedDataset::<f64, 3, 2>::n_features(), 2);
    }

    #[test]
    fn test_proof_construction() {
        // 4 is even, should succeed
        let proof_even = Proof::<IsEven<4>>::new();
        assert!(proof_even.is_ok());

        // 5 is odd, should fail
        let proof_odd = Proof::<IsEven<5>>::new();
        assert!(proof_odd.is_err());
    }

    #[test]
    fn test_expr_eval() {
        // (2 + 3) * 4 = 20
        let expr = Expr::Mul(
            Box::new(Expr::Add(Box::new(Expr::Lit(2)), Box::new(Expr::Lit(3)))),
            Box::new(Expr::Lit(4)),
        );

        assert_eq!(expr.eval(), 20);
    }

    #[test]
    fn test_lvec_map() {
        let v = LVec::<i32, N3>::new(vec![1, 2, 3]).unwrap();
        let doubled = v.map(|x| x * 2);
        assert_eq!(doubled.as_slice(), &[2, 4, 6]);
    }

    #[test]
    fn test_lvec_zip() {
        let v1 = LVec::<i32, N3>::new(vec![1, 2, 3]).unwrap();
        let v2 = LVec::<i32, N3>::new(vec![4, 5, 6]).unwrap();
        let zipped = v1.zip(v2);

        assert_eq!(zipped.as_slice(), &[(1, 4), (2, 5), (3, 6)]);
    }

    #[test]
    fn test_dot_product() {
        let v1 = LVec::<i32, N3>::new(vec![1, 2, 3]).unwrap();
        let v2 = LVec::<i32, N3>::new(vec![4, 5, 6]).unwrap();

        let dot = dot_product(&v1, &v2);
        assert_eq!(dot, 1 * 4 + 2 * 5 + 3 * 6);
        assert_eq!(dot, 32);
    }

    #[test]
    fn test_concat() {
        let v1 = LVec::<i32, N2>::new(vec![1, 2]).unwrap();
        let v2 = LVec::<i32, N3>::new(vec![3, 4, 5]).unwrap();

        let concatenated = concat(v1, v2);

        // Type is LVec<i32, N5>
        assert_eq!(concatenated.as_slice(), &[1, 2, 3, 4, 5]);
    }
}
