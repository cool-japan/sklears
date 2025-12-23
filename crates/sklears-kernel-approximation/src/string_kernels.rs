//! String Kernel Approximations
//!
//! This module implements various string kernel approximation methods for
//! sequence and text analysis. String kernels measure similarity between
//! sequences of symbols (characters, words, etc.) by counting shared
//! subsequences or n-grams.
//!
//! # Key Features
//!
//! - **N-gram Kernels**: Count shared n-grams between sequences
//! - **Spectrum Kernels**: Fixed-length contiguous substring kernels
//! - **Subsequence Kernels**: Count all shared subsequences with gaps
//! - **Edit Distance Approximations**: Approximate edit distance kernels
//! - **Mismatch Kernels**: Allow for mismatches in n-gram comparisons
//! - **Weighted Subsequence Kernels**: Weight subsequences by length and gaps
//!
//! # Mathematical Background
//!
//! String kernel between sequences s and t:
//! K(s, t) = Σ φ(s)[u] * φ(t)[u]
//!
//! Where φ(s)[u] is the feature map that counts occurrences of substring u.
//!
//! # References
//!
//! - Shawe-Taylor, J., & Cristianini, N. (2004). Kernel methods for pattern analysis
//! - Lodhi, H., et al. (2002). Text classification using string kernels

use scirs2_core::ndarray::Array2;
use sklears_core::{
    error::Result,
    prelude::{Fit, Transform},
};
use std::collections::HashMap;

/// N-gram kernel for sequences
#[derive(Debug, Clone)]
/// NGramKernel
pub struct NGramKernel {
    /// N-gram size
    n: usize,
    /// Whether to normalize features
    normalize: bool,
    /// Whether to use binary features (presence/absence vs counts)
    binary: bool,
    /// Character-level vs word-level n-grams
    mode: NGramMode,
}

/// N-gram extraction mode
#[derive(Debug, Clone)]
/// NGramMode
pub enum NGramMode {
    Character,
    Word,
    Custom { delimiter: String },
}

/// Fitted n-gram kernel
#[derive(Debug, Clone)]
/// FittedNGramKernel
pub struct FittedNGramKernel {
    /// Vocabulary mapping from n-gram to index
    vocabulary: HashMap<String, usize>,
    /// N-gram size
    n: usize,
    /// Normalization flag
    normalize: bool,
    /// Binary flag
    binary: bool,
    /// N-gram mode
    mode: NGramMode,
}

impl NGramKernel {
    /// Create new n-gram kernel
    pub fn new(n: usize) -> Self {
        Self {
            n,
            normalize: true,
            binary: false,
            mode: NGramMode::Character,
        }
    }

    /// Set normalization
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set binary mode
    pub fn binary(mut self, binary: bool) -> Self {
        self.binary = binary;
        self
    }

    /// Set n-gram mode
    pub fn mode(mut self, mode: NGramMode) -> Self {
        self.mode = mode;
        self
    }

    /// Extract n-grams from a sequence
    fn extract_ngrams(&self, sequence: &str) -> Vec<String> {
        match &self.mode {
            NGramMode::Character => {
                let chars: Vec<char> = sequence.chars().collect();
                chars
                    .windows(self.n)
                    .map(|window| window.iter().collect())
                    .collect()
            }
            NGramMode::Word => {
                let words: Vec<&str> = sequence.split_whitespace().collect();
                words
                    .windows(self.n)
                    .map(|window| window.join(" "))
                    .collect()
            }
            NGramMode::Custom { delimiter } => {
                let tokens: Vec<&str> = sequence.split(delimiter).collect();
                tokens
                    .windows(self.n)
                    .map(|window| window.join(delimiter))
                    .collect()
            }
        }
    }
}

impl Fit<Vec<String>, ()> for NGramKernel {
    type Fitted = FittedNGramKernel;

    fn fit(self, sequences: &Vec<String>, _y: &()) -> Result<Self::Fitted> {
        let mut vocabulary = HashMap::new();
        let mut vocab_index = 0;

        // Build vocabulary from all sequences
        for sequence in sequences {
            let ngrams = self.extract_ngrams(sequence);
            for ngram in ngrams {
                if let std::collections::hash_map::Entry::Vacant(e) = vocabulary.entry(ngram) {
                    e.insert(vocab_index);
                    vocab_index += 1;
                }
            }
        }

        Ok(FittedNGramKernel {
            vocabulary,
            n: self.n,
            normalize: self.normalize,
            binary: self.binary,
            mode: self.mode.clone(),
        })
    }
}

impl Transform<Vec<String>, Array2<f64>> for FittedNGramKernel {
    fn transform(&self, sequences: &Vec<String>) -> Result<Array2<f64>> {
        let n_sequences = sequences.len();
        let vocab_size = self.vocabulary.len();
        let mut features = Array2::zeros((n_sequences, vocab_size));

        for (i, sequence) in sequences.iter().enumerate() {
            let ngrams = match &self.mode {
                NGramMode::Character => {
                    let chars: Vec<char> = sequence.chars().collect();
                    chars
                        .windows(self.n)
                        .map(|window| window.iter().collect::<String>())
                        .collect::<Vec<String>>()
                }
                NGramMode::Word => {
                    let words: Vec<&str> = sequence.split_whitespace().collect();
                    words
                        .windows(self.n)
                        .map(|window| window.join(" "))
                        .collect::<Vec<String>>()
                }
                NGramMode::Custom { delimiter } => {
                    let tokens: Vec<&str> = sequence.split(delimiter).collect();
                    tokens
                        .windows(self.n)
                        .map(|window| window.join(delimiter))
                        .collect::<Vec<String>>()
                }
            };

            // Count n-grams
            let mut ngram_counts = HashMap::new();
            for ngram in ngrams {
                if let Some(&vocab_idx) = self.vocabulary.get(&ngram) {
                    *ngram_counts.entry(vocab_idx).or_insert(0) += 1;
                }
            }

            // Fill feature vector
            for (vocab_idx, count) in ngram_counts {
                features[(i, vocab_idx)] = if self.binary { 1.0 } else { count as f64 };
            }

            // Normalize if requested
            if self.normalize {
                let norm = features.row(i).mapv(|x| x * x).sum().sqrt();
                if norm > 0.0 {
                    for j in 0..vocab_size {
                        features[(i, j)] /= norm;
                    }
                }
            }
        }

        Ok(features)
    }
}

/// Spectrum kernel for fixed-length contiguous substrings
#[derive(Debug, Clone)]
/// SpectrumKernel
pub struct SpectrumKernel {
    /// Substring length (k-mer size)
    k: usize,
    /// Whether to normalize features
    normalize: bool,
}

impl SpectrumKernel {
    /// Create new spectrum kernel
    pub fn new(k: usize) -> Self {
        Self { k, normalize: true }
    }

    /// Set normalization
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
}

/// Fitted spectrum kernel
#[derive(Debug, Clone)]
/// FittedSpectrumKernel
pub struct FittedSpectrumKernel {
    /// Vocabulary of k-mers
    vocabulary: HashMap<String, usize>,
    /// K-mer length
    k: usize,
    /// Normalization flag
    normalize: bool,
}

impl Fit<Vec<String>, ()> for SpectrumKernel {
    type Fitted = FittedSpectrumKernel;

    fn fit(self, sequences: &Vec<String>, _y: &()) -> Result<Self::Fitted> {
        let mut vocabulary = HashMap::new();
        let mut vocab_index = 0;

        // Build vocabulary of all k-mers
        for sequence in sequences {
            let chars: Vec<char> = sequence.chars().collect();
            for window in chars.windows(self.k) {
                let kmer: String = window.iter().collect();
                if let std::collections::hash_map::Entry::Vacant(e) = vocabulary.entry(kmer) {
                    e.insert(vocab_index);
                    vocab_index += 1;
                }
            }
        }

        Ok(FittedSpectrumKernel {
            vocabulary,
            k: self.k,
            normalize: self.normalize,
        })
    }
}

impl Transform<Vec<String>, Array2<f64>> for FittedSpectrumKernel {
    fn transform(&self, sequences: &Vec<String>) -> Result<Array2<f64>> {
        let n_sequences = sequences.len();
        let vocab_size = self.vocabulary.len();
        let mut features = Array2::zeros((n_sequences, vocab_size));

        for (i, sequence) in sequences.iter().enumerate() {
            let chars: Vec<char> = sequence.chars().collect();
            let mut kmer_counts = HashMap::new();

            // Count k-mers
            for window in chars.windows(self.k) {
                let kmer: String = window.iter().collect();
                if let Some(&vocab_idx) = self.vocabulary.get(&kmer) {
                    *kmer_counts.entry(vocab_idx).or_insert(0) += 1;
                }
            }

            // Fill feature vector
            for (vocab_idx, count) in kmer_counts {
                features[(i, vocab_idx)] = count as f64;
            }

            // Normalize if requested
            if self.normalize {
                let norm = features.row(i).mapv(|x| x * x).sum().sqrt();
                if norm > 0.0 {
                    for j in 0..vocab_size {
                        features[(i, j)] /= norm;
                    }
                }
            }
        }

        Ok(features)
    }
}

/// Subsequence kernel that counts all shared subsequences (with gaps)
#[derive(Debug, Clone)]
/// SubsequenceKernel
pub struct SubsequenceKernel {
    /// Maximum subsequence length
    max_length: usize,
    /// Gap penalty (lambda parameter, 0 < lambda <= 1)
    gap_penalty: f64,
    /// Normalize features
    normalize: bool,
}

impl SubsequenceKernel {
    /// Create new subsequence kernel
    pub fn new(max_length: usize, gap_penalty: f64) -> Self {
        Self {
            max_length,
            gap_penalty,
            normalize: true,
        }
    }

    /// Set normalization
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Compute subsequence kernel between two sequences using dynamic programming
    fn subsequence_kernel_value(&self, s1: &str, s2: &str) -> f64 {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();
        let n1 = chars1.len();
        let n2 = chars2.len();

        if n1 == 0 || n2 == 0 {
            return 0.0;
        }

        let mut dp = vec![vec![vec![0.0; n2 + 1]; n1 + 1]; self.max_length + 1];

        // Initialize base cases
        for i in 0..=n1 {
            for j in 0..=n2 {
                dp[0][i][j] = 1.0;
            }
        }

        // Fill DP table
        for k in 1..=self.max_length {
            for i in 1..=n1 {
                for j in 1..=n2 {
                    // Case 1: don't include chars1[i-1]
                    dp[k][i][j] = self.gap_penalty * dp[k][i - 1][j];

                    // Case 2: include chars1[i-1] if it matches chars2[j-1]
                    if chars1[i - 1] == chars2[j - 1] {
                        dp[k][i][j] += self.gap_penalty * dp[k - 1][i - 1][j - 1];
                    }

                    // Case 3: don't include chars2[j-1]
                    dp[k][i][j] += self.gap_penalty * dp[k][i][j - 1];

                    // Case 4: don't include both
                    if chars1[i - 1] == chars2[j - 1] {
                        dp[k][i][j] -=
                            self.gap_penalty * self.gap_penalty * dp[k - 1][i - 1][j - 1];
                    }
                }
            }
        }

        // Sum over all subsequence lengths
        let mut total = 0.0;
        for k in 1..=self.max_length {
            total += dp[k][n1][n2];
        }

        total
    }
}

/// Fitted subsequence kernel (computes full kernel matrix)
#[derive(Debug, Clone)]
/// FittedSubsequenceKernel
pub struct FittedSubsequenceKernel {
    /// Training sequences
    training_sequences: Vec<String>,
    /// Maximum subsequence length
    max_length: usize,
    /// Gap penalty
    gap_penalty: f64,
    /// Normalization flag
    normalize: bool,
}

impl Fit<Vec<String>, ()> for SubsequenceKernel {
    type Fitted = FittedSubsequenceKernel;

    fn fit(self, sequences: &Vec<String>, _y: &()) -> Result<Self::Fitted> {
        Ok(FittedSubsequenceKernel {
            training_sequences: sequences.clone(),
            max_length: self.max_length,
            gap_penalty: self.gap_penalty,
            normalize: self.normalize,
        })
    }
}

impl Transform<Vec<String>, Array2<f64>> for FittedSubsequenceKernel {
    fn transform(&self, sequences: &Vec<String>) -> Result<Array2<f64>> {
        let n_test = sequences.len();
        let n_train = self.training_sequences.len();
        let mut kernel_matrix = Array2::zeros((n_test, n_train));

        // Temporary kernel instance for computation
        let kernel = SubsequenceKernel {
            max_length: self.max_length,
            gap_penalty: self.gap_penalty,
            normalize: false, // Handle normalization separately
        };

        for i in 0..n_test {
            for j in 0..n_train {
                kernel_matrix[(i, j)] =
                    kernel.subsequence_kernel_value(&sequences[i], &self.training_sequences[j]);
            }

            // Normalize row if requested
            if self.normalize {
                let norm = kernel_matrix.row(i).mapv(|x| x * x).sum().sqrt();
                if norm > 0.0 {
                    for j in 0..n_train {
                        kernel_matrix[(i, j)] /= norm;
                    }
                }
            }
        }

        Ok(kernel_matrix)
    }
}

/// Edit distance approximation kernel
#[derive(Debug, Clone)]
/// EditDistanceKernel
pub struct EditDistanceKernel {
    /// Maximum edit distance to consider
    max_distance: usize,
    /// Kernel bandwidth parameter
    sigma: f64,
}

impl EditDistanceKernel {
    /// Create new edit distance kernel
    pub fn new(max_distance: usize, sigma: f64) -> Self {
        Self {
            max_distance,
            sigma,
        }
    }

    /// Compute edit distance between two strings
    fn edit_distance(&self, s1: &str, s2: &str) -> usize {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();
        let n1 = chars1.len();
        let n2 = chars2.len();

        let mut dp = vec![vec![0; n2 + 1]; n1 + 1];

        // Initialize first row and column
        for i in 0..=n1 {
            dp[i][0] = i;
        }
        for j in 0..=n2 {
            dp[0][j] = j;
        }

        // Fill DP table
        for i in 1..=n1 {
            for j in 1..=n2 {
                let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };
                dp[i][j] = (dp[i - 1][j] + 1)
                    .min(dp[i][j - 1] + 1)
                    .min(dp[i - 1][j - 1] + cost);
            }
        }

        dp[n1][n2]
    }

    /// Compute kernel value from edit distance
    fn kernel_value(&self, s1: &str, s2: &str) -> f64 {
        let distance = self.edit_distance(s1, s2);
        if distance > self.max_distance {
            0.0
        } else {
            (-(distance as f64) / self.sigma).exp()
        }
    }
}

/// Fitted edit distance kernel
#[derive(Debug, Clone)]
/// FittedEditDistanceKernel
pub struct FittedEditDistanceKernel {
    /// Training sequences
    training_sequences: Vec<String>,
    /// Maximum distance
    max_distance: usize,
    /// Sigma parameter
    sigma: f64,
}

impl Fit<Vec<String>, ()> for EditDistanceKernel {
    type Fitted = FittedEditDistanceKernel;

    fn fit(self, sequences: &Vec<String>, _y: &()) -> Result<Self::Fitted> {
        Ok(FittedEditDistanceKernel {
            training_sequences: sequences.clone(),
            max_distance: self.max_distance,
            sigma: self.sigma,
        })
    }
}

impl Transform<Vec<String>, Array2<f64>> for FittedEditDistanceKernel {
    fn transform(&self, sequences: &Vec<String>) -> Result<Array2<f64>> {
        let n_test = sequences.len();
        let n_train = self.training_sequences.len();
        let mut kernel_matrix = Array2::zeros((n_test, n_train));

        let kernel = EditDistanceKernel {
            max_distance: self.max_distance,
            sigma: self.sigma,
        };

        for i in 0..n_test {
            for j in 0..n_train {
                kernel_matrix[(i, j)] =
                    kernel.kernel_value(&sequences[i], &self.training_sequences[j]);
            }
        }

        Ok(kernel_matrix)
    }
}

/// Mismatch kernel that allows k mismatches in n-grams
#[derive(Debug, Clone)]
/// MismatchKernel
pub struct MismatchKernel {
    /// N-gram length
    k: usize,
    /// Number of allowed mismatches
    m: usize,
    /// Alphabet (set of allowed characters)
    alphabet: Vec<char>,
}

impl MismatchKernel {
    /// Create new mismatch kernel
    pub fn new(k: usize, m: usize) -> Self {
        // Default DNA alphabet
        let alphabet = vec!['A', 'C', 'G', 'T'];
        Self { k, m, alphabet }
    }

    /// Set custom alphabet
    pub fn alphabet(mut self, alphabet: Vec<char>) -> Self {
        self.alphabet = alphabet;
        self
    }

    /// Generate all possible k-mers with up to m mismatches from a given k-mer
    fn generate_neighborhood(&self, kmer: &str, mismatches: usize) -> Vec<String> {
        if mismatches == 0 {
            return vec![kmer.to_string()];
        }

        let chars: Vec<char> = kmer.chars().collect();
        let mut neighborhood = Vec::new();

        // Generate all combinations with exactly 'mismatches' positions changed
        self.generate_mismatches(&chars, 0, mismatches, &mut vec![], &mut neighborhood);

        neighborhood
    }

    /// Recursive helper for generating mismatches
    fn generate_mismatches(
        &self,
        original: &[char],
        pos: usize,
        mismatches_left: usize,
        current: &mut Vec<char>,
        result: &mut Vec<String>,
    ) {
        if pos == original.len() {
            if mismatches_left == 0 {
                result.push(current.iter().collect());
            }
            return;
        }

        // Option 1: Keep original character
        current.push(original[pos]);
        self.generate_mismatches(original, pos + 1, mismatches_left, current, result);
        current.pop();

        // Option 2: Try all alphabet characters (if we still have mismatches to use)
        if mismatches_left > 0 {
            for &c in &self.alphabet {
                if c != original[pos] {
                    current.push(c);
                    self.generate_mismatches(
                        original,
                        pos + 1,
                        mismatches_left - 1,
                        current,
                        result,
                    );
                    current.pop();
                }
            }
        }
    }
}

/// Fitted mismatch kernel
#[derive(Debug, Clone)]
/// FittedMismatchKernel
pub struct FittedMismatchKernel {
    /// Feature vocabulary
    vocabulary: HashMap<String, usize>,
    /// K-mer length
    k: usize,
    /// Number of mismatches
    m: usize,
    /// Alphabet
    alphabet: Vec<char>,
}

impl Fit<Vec<String>, ()> for MismatchKernel {
    type Fitted = FittedMismatchKernel;

    fn fit(self, sequences: &Vec<String>, _y: &()) -> Result<Self::Fitted> {
        let mut vocabulary = HashMap::new();
        let mut vocab_index = 0;

        // Build vocabulary of all possible k-mers with mismatches
        for sequence in sequences {
            let chars: Vec<char> = sequence.chars().collect();
            for window in chars.windows(self.k) {
                let kmer: String = window.iter().collect();

                // Generate neighborhood with up to m mismatches
                for mismatch_count in 0..=self.m {
                    let neighborhood = self.generate_neighborhood(&kmer, mismatch_count);
                    for neighbor in neighborhood {
                        if let std::collections::hash_map::Entry::Vacant(e) =
                            vocabulary.entry(neighbor)
                        {
                            e.insert(vocab_index);
                            vocab_index += 1;
                        }
                    }
                }
            }
        }

        Ok(FittedMismatchKernel {
            vocabulary,
            k: self.k,
            m: self.m,
            alphabet: self.alphabet.clone(),
        })
    }
}

impl Transform<Vec<String>, Array2<f64>> for FittedMismatchKernel {
    fn transform(&self, sequences: &Vec<String>) -> Result<Array2<f64>> {
        let n_sequences = sequences.len();
        let vocab_size = self.vocabulary.len();
        let mut features = Array2::zeros((n_sequences, vocab_size));

        let kernel = MismatchKernel {
            k: self.k,
            m: self.m,
            alphabet: self.alphabet.clone(),
        };

        for (i, sequence) in sequences.iter().enumerate() {
            let chars: Vec<char> = sequence.chars().collect();
            let mut feature_counts = HashMap::new();

            // Extract k-mers and generate their neighborhoods
            for window in chars.windows(self.k) {
                let kmer: String = window.iter().collect();

                for mismatch_count in 0..=self.m {
                    let neighborhood = kernel.generate_neighborhood(&kmer, mismatch_count);
                    for neighbor in neighborhood {
                        if let Some(&vocab_idx) = self.vocabulary.get(&neighbor) {
                            *feature_counts.entry(vocab_idx).or_insert(0) += 1;
                        }
                    }
                }
            }

            // Fill feature vector
            for (vocab_idx, count) in feature_counts {
                features[(i, vocab_idx)] = count as f64;
            }
        }

        Ok(features)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_ngram_kernel_character() {
        let kernel = NGramKernel::new(2).mode(NGramMode::Character);
        let sequences = vec!["hello".to_string(), "world".to_string(), "help".to_string()];

        let fitted = kernel.fit(&sequences, &()).unwrap();
        let features = fitted.transform(&sequences).unwrap();

        assert_eq!(features.nrows(), 3);
        assert!(features.ncols() > 0);
        assert!(features.iter().all(|&x| x >= 0.0 && x.is_finite()));
    }

    #[test]
    fn test_ngram_kernel_word() {
        let kernel = NGramKernel::new(2).mode(NGramMode::Word);
        let sequences = vec![
            "hello world".to_string(),
            "world peace".to_string(),
            "hello there".to_string(),
        ];

        let fitted = kernel.fit(&sequences, &()).unwrap();
        let features = fitted.transform(&sequences).unwrap();

        assert_eq!(features.nrows(), 3);
        assert!(features.ncols() > 0);
        assert!(features.iter().all(|&x| x >= 0.0 && x.is_finite()));
    }

    #[test]
    fn test_spectrum_kernel() {
        let kernel = SpectrumKernel::new(3);
        let sequences = vec![
            "ATCGATCG".to_string(),
            "GCTAGCTA".to_string(),
            "ATCGATCG".to_string(), // duplicate
        ];

        let fitted = kernel.fit(&sequences, &()).unwrap();
        let features = fitted.transform(&sequences).unwrap();

        assert_eq!(features.nrows(), 3);
        assert!(features.ncols() > 0);

        // First and third sequences should be identical
        for j in 0..features.ncols() {
            assert_abs_diff_eq!(features[(0, j)], features[(2, j)], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_subsequence_kernel() {
        let kernel = SubsequenceKernel::new(3, 0.5);
        let sequences = vec!["ABC".to_string(), "ACB".to_string(), "ABC".to_string()];

        let fitted = kernel.fit(&sequences, &()).unwrap();
        let features = fitted.transform(&sequences).unwrap();

        assert_eq!(features.nrows(), 3);
        assert_eq!(features.ncols(), 3);
        assert!(features.iter().all(|&x| x >= 0.0 && x.is_finite()));

        // Kernel should be symmetric for identical sequences
        assert!(features[(0, 0)] > 0.0);
        assert_abs_diff_eq!(features[(0, 0)], features[(2, 0)], epsilon = 1e-10);
    }

    #[test]
    fn test_edit_distance_kernel() {
        let kernel = EditDistanceKernel::new(5, 1.0);
        let sequences = vec![
            "cat".to_string(),
            "bat".to_string(),
            "rat".to_string(),
            "dog".to_string(),
        ];

        let fitted = kernel.fit(&sequences, &()).unwrap();
        let features = fitted.transform(&sequences).unwrap();

        assert_eq!(features.nrows(), 4);
        assert_eq!(features.ncols(), 4);
        assert!(features
            .iter()
            .all(|&x| x >= 0.0 && x <= 1.0 && x.is_finite()));

        // Self-similarity should be 1.0
        for i in 0..4 {
            assert_abs_diff_eq!(features[(i, i)], 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_mismatch_kernel() {
        let kernel = MismatchKernel::new(3, 1).alphabet(vec!['A', 'C', 'G', 'T']);
        let sequences = vec![
            "ATCG".to_string(),
            "ATCC".to_string(), // 1 mismatch from first
            "GCTA".to_string(),
        ];

        let fitted = kernel.fit(&sequences, &()).unwrap();
        let features = fitted.transform(&sequences).unwrap();

        assert_eq!(features.nrows(), 3);
        assert!(features.ncols() > 0);
        assert!(features.iter().all(|&x| x >= 0.0 && x.is_finite()));
    }

    #[test]
    fn test_edit_distance_computation() {
        let kernel = EditDistanceKernel::new(10, 1.0);

        assert_eq!(kernel.edit_distance("", ""), 0);
        assert_eq!(kernel.edit_distance("cat", "cat"), 0);
        assert_eq!(kernel.edit_distance("cat", "bat"), 1);
        assert_eq!(kernel.edit_distance("cat", "dog"), 3);
        assert_eq!(kernel.edit_distance("kitten", "sitting"), 3);
    }

    #[test]
    fn test_ngram_binary_mode() {
        let kernel = NGramKernel::new(2).binary(true).normalize(false);

        let sequences = vec![
            "aaa".to_string(), // "aa" appears twice
            "aba".to_string(), // "ab" and "ba" appear once each
        ];

        let fitted = kernel.fit(&sequences, &()).unwrap();
        let features = fitted.transform(&sequences).unwrap();

        // In binary mode, repeated n-grams should only count as 1
        assert!(features.iter().all(|&x| x == 0.0 || x == 1.0));
    }
}
