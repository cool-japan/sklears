//! Memory-efficient algorithms for high-performance dummy estimator operations

/// Calculate statistics with minimal memory allocation
pub fn streaming_stats(data: &[f64]) -> (f64, f64, f64) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    let std_dev = variance.sqrt();

    (mean, variance, std_dev)
}

/// Memory-efficient online statistics computation
#[derive(Debug, Clone)]
pub struct OnlineStats {
    count: usize,
    mean: f64,
    m2: f64, // Sum of squared deviations from mean
    min: f64,
    max: f64,
}

impl Default for OnlineStats {
    fn default() -> Self {
        Self::new()
    }
}

impl OnlineStats {
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;

        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }

    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }

    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    pub fn min(&self) -> f64 {
        self.min
    }

    pub fn max(&self) -> f64 {
        self.max
    }

    pub fn count(&self) -> usize {
        self.count
    }
}

/// Memory pool for efficient allocation/deallocation
pub struct MemoryPool<T> {
    pool: Vec<Vec<T>>,
    available: Vec<usize>,
}

impl<T: Clone + Default> MemoryPool<T> {
    pub fn new(pool_size: usize, buffer_size: usize) -> Self {
        let mut pool = Vec::with_capacity(pool_size);
        let mut available = Vec::with_capacity(pool_size);

        for i in 0..pool_size {
            pool.push(vec![T::default(); buffer_size]);
            available.push(i);
        }

        Self { pool, available }
    }

    pub fn acquire(&mut self) -> Option<Vec<T>> {
        self.available
            .pop()
            .map(|index| std::mem::take(&mut self.pool[index]))
    }

    pub fn release(&mut self, mut buffer: Vec<T>) {
        if let Some(index) = self.pool.iter().position(|v| v.is_empty()) {
            buffer.clear();
            self.pool[index] = buffer;
            self.available.push(index);
        }
    }
}

/// Efficient sparse vector representation
#[derive(Debug, Clone)]
pub struct SparseVector {
    indices: Vec<usize>,
    values: Vec<f64>,
    length: usize,
}

impl SparseVector {
    pub fn new(length: usize) -> Self {
        Self {
            indices: Vec::new(),
            values: Vec::new(),
            length,
        }
    }

    pub fn set(&mut self, index: usize, value: f64) {
        if index >= self.length {
            return;
        }

        if let Some(pos) = self.indices.iter().position(|&i| i == index) {
            self.values[pos] = value;
        } else {
            // Insert in sorted order
            let pos = self.indices.binary_search(&index).unwrap_or_else(|e| e);
            self.indices.insert(pos, index);
            self.values.insert(pos, value);
        }
    }

    pub fn get(&self, index: usize) -> f64 {
        if let Some(pos) = self.indices.iter().position(|&i| i == index) {
            self.values[pos]
        } else {
            0.0
        }
    }

    pub fn dot(&self, other: &[f64]) -> f64 {
        if other.len() != self.length {
            return 0.0;
        }

        self.indices
            .iter()
            .zip(self.values.iter())
            .map(|(&index, &value)| value * other[index])
            .sum()
    }

    pub fn norm_squared(&self) -> f64 {
        self.values.iter().map(|&v| v * v).sum()
    }
}
