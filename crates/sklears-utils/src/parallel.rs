//! Parallel computing utilities for machine learning workloads
//!
//! This module provides utilities for parallel processing, including thread pool
//! management, work-stealing algorithms, and parallel iterator utilities.

use crate::{UtilsError, UtilsResult};
use scirs2_core::numeric::Zero;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;
#[allow(non_snake_case)]
#[cfg(test)]
use std::time::Duration;

/// Thread pool for parallel task execution
#[derive(Debug)]
pub struct ThreadPool {
    workers: Vec<Worker>,
    sender: Option<std::sync::mpsc::Sender<Job>>,
    num_threads: usize,
}

type Job = Box<dyn FnOnce() + Send + 'static>;

impl ThreadPool {
    /// Create a new thread pool with the specified number of threads
    pub fn new(num_threads: usize) -> UtilsResult<Self> {
        if num_threads == 0 {
            return Err(UtilsError::InvalidParameter(
                "Thread pool size must be greater than 0".to_string(),
            ));
        }

        let (sender, receiver) = std::sync::mpsc::channel();
        let receiver = Arc::new(Mutex::new(receiver));
        let mut workers = Vec::with_capacity(num_threads);

        for id in 0..num_threads {
            workers.push(Worker::new(id, Arc::clone(&receiver))?);
        }

        Ok(ThreadPool {
            workers,
            sender: Some(sender),
            num_threads,
        })
    }

    /// Create a thread pool with number of threads equal to CPU cores
    pub fn with_cpu_cores() -> UtilsResult<Self> {
        let num_cores = num_cpus::get();
        Self::new(num_cores)
    }

    /// Submit a job to the thread pool
    pub fn execute<F>(&self, f: F) -> UtilsResult<()>
    where
        F: FnOnce() + Send + 'static,
    {
        let job = Box::new(f);
        self.sender
            .as_ref()
            .ok_or_else(|| {
                UtilsError::InvalidParameter("Thread pool is shutting down".to_string())
            })?
            .send(job)
            .map_err(|_| {
                UtilsError::InvalidParameter("Failed to send job to thread pool".to_string())
            })?;
        Ok(())
    }

    /// Get the number of worker threads
    pub fn thread_count(&self) -> usize {
        self.num_threads
    }

    /// Wait for all current jobs to complete
    pub fn join(&mut self) {
        drop(self.sender.take());
        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                thread.join().unwrap();
            }
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        drop(self.sender.take());
        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                thread.join().unwrap();
            }
        }
    }
}

#[derive(Debug)]
struct Worker {
    #[allow(dead_code)]
    id: usize,
    thread: Option<thread::JoinHandle<()>>,
}

impl Worker {
    fn new(id: usize, receiver: Arc<Mutex<std::sync::mpsc::Receiver<Job>>>) -> UtilsResult<Self> {
        let thread = thread::spawn(move || loop {
            let job = receiver.lock().unwrap().recv();
            match job {
                Ok(job) => {
                    job();
                }
                Err(_) => {
                    break;
                }
            }
        });

        Ok(Worker {
            id,
            thread: Some(thread),
        })
    }
}

/// Work-stealing deque for load balancing
#[derive(Debug)]
pub struct WorkStealingQueue<T> {
    local_queue: Arc<Mutex<VecDeque<T>>>,
    global_queue: Arc<Mutex<VecDeque<T>>>,
    workers: Vec<Arc<Mutex<VecDeque<T>>>>,
    worker_id: usize,
}

impl<T> WorkStealingQueue<T>
where
    T: Send + 'static + Clone,
{
    /// Create a new work-stealing queue system
    pub fn new(num_workers: usize) -> Self {
        let global_queue = Arc::new(Mutex::new(VecDeque::new()));
        let mut workers = Vec::with_capacity(num_workers);

        for _ in 0..num_workers {
            workers.push(Arc::new(Mutex::new(VecDeque::new())));
        }

        Self {
            local_queue: Arc::clone(&workers[0]),
            global_queue,
            workers,
            worker_id: 0,
        }
    }

    /// Push a task to the local queue
    pub fn push_local(&self, task: T) -> UtilsResult<()> {
        self.local_queue
            .lock()
            .map_err(|_| {
                UtilsError::InvalidParameter("Failed to acquire local queue lock".to_string())
            })?
            .push_back(task);
        Ok(())
    }

    /// Push a task to the global queue
    pub fn push_global(&self, task: T) -> UtilsResult<()> {
        self.global_queue
            .lock()
            .map_err(|_| {
                UtilsError::InvalidParameter("Failed to acquire global queue lock".to_string())
            })?
            .push_back(task);
        Ok(())
    }

    /// Pop a task from the local queue
    pub fn pop_local(&self) -> UtilsResult<Option<T>> {
        Ok(self
            .local_queue
            .lock()
            .map_err(|_| {
                UtilsError::InvalidParameter("Failed to acquire local queue lock".to_string())
            })?
            .pop_front())
    }

    /// Steal work from other workers' queues
    pub fn steal_work(&self) -> UtilsResult<Option<T>> {
        // Try to steal from other workers' queues
        for (i, worker_queue) in self.workers.iter().enumerate() {
            if i != self.worker_id {
                if let Ok(mut queue) = worker_queue.try_lock() {
                    if let Some(task) = queue.pop_back() {
                        return Ok(Some(task));
                    }
                }
            }
        }

        // If no work was stolen, try the global queue
        if let Ok(mut global) = self.global_queue.try_lock() {
            return Ok(global.pop_front());
        }

        Ok(None)
    }

    /// Get the next task, trying local queue first, then stealing
    pub fn get_task(&self) -> UtilsResult<Option<T>> {
        // Try local queue first
        if let Some(task) = self.pop_local()? {
            return Ok(Some(task));
        }

        // If local queue is empty, try stealing
        self.steal_work()
    }
}

/// Parallel iterator utilities
pub struct ParallelIterator<T> {
    items: Vec<T>,
    chunk_size: usize,
}

impl<T> ParallelIterator<T>
where
    T: Send + 'static + Clone,
{
    /// Create a new parallel iterator
    pub fn new(items: Vec<T>) -> Self {
        let chunk_size = (items.len() / num_cpus::get()).max(1);
        Self { items, chunk_size }
    }

    /// Set the chunk size for parallel processing
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size.max(1);
        self
    }

    /// Map function over items in parallel
    pub fn map<F, R>(self, f: F) -> UtilsResult<Vec<R>>
    where
        F: Fn(T) -> R + Send + Sync + 'static,
        R: Send + 'static + Clone,
    {
        let f = Arc::new(f);
        let results = Arc::new(Mutex::new(Vec::with_capacity(self.items.len())));
        let _thread_pool = ThreadPool::with_cpu_cores()?;

        // Split items into chunks
        let chunks: Vec<_> = self
            .items
            .into_iter()
            .collect::<Vec<_>>()
            .chunks(self.chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        let mut handles = Vec::new();

        for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
            let f_clone = Arc::clone(&f);
            let results_clone = Arc::clone(&results);
            let chunk_size = chunk.len();

            let handle = thread::spawn(move || {
                let mut chunk_results = Vec::with_capacity(chunk_size);
                for item in chunk {
                    chunk_results.push(f_clone(item));
                }

                let mut results_lock = results_clone.lock().unwrap();
                // Ensure we have enough space
                if results_lock.len() <= chunk_idx {
                    results_lock.resize_with(chunk_idx + 1, || Vec::new());
                }
                results_lock[chunk_idx] = chunk_results;
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().map_err(|_| {
                UtilsError::InvalidParameter(
                    "Thread panicked during parallel execution".to_string(),
                )
            })?;
        }

        // Collect results in order
        let results_lock = results.lock().unwrap();
        let mut final_results = Vec::new();
        for chunk_results in results_lock.iter() {
            final_results.extend_from_slice(chunk_results);
        }

        Ok(final_results)
    }

    /// Filter items in parallel
    pub fn filter<F>(self, predicate: F) -> UtilsResult<Vec<T>>
    where
        F: Fn(&T) -> bool + Send + Sync + 'static,
        T: Clone,
    {
        let predicate = Arc::new(predicate);
        let results = Arc::new(Mutex::new(Vec::new()));
        let _thread_pool = ThreadPool::with_cpu_cores()?;

        // Split items into chunks
        let chunks: Vec<_> = self
            .items
            .into_iter()
            .collect::<Vec<_>>()
            .chunks(self.chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        let mut handles = Vec::new();

        for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
            let predicate_clone = Arc::clone(&predicate);
            let results_clone = Arc::clone(&results);

            let handle = thread::spawn(move || {
                let filtered: Vec<T> = chunk
                    .into_iter()
                    .filter(|item| predicate_clone(item))
                    .collect();

                let mut results_lock = results_clone.lock().unwrap();
                // Ensure we have enough space
                if results_lock.len() <= chunk_idx {
                    results_lock.resize_with(chunk_idx + 1, || Vec::new());
                }
                results_lock[chunk_idx] = filtered;
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().map_err(|_| {
                UtilsError::InvalidParameter(
                    "Thread panicked during parallel execution".to_string(),
                )
            })?;
        }

        // Collect results in order
        let results_lock = results.lock().unwrap();
        let mut final_results = Vec::new();
        for chunk_results in results_lock.iter() {
            final_results.extend_from_slice(chunk_results);
        }

        Ok(final_results)
    }
}

/// Parallel reduction operations
pub struct ParallelReducer;

impl ParallelReducer {
    /// Reduce a vector in parallel using the given operation
    pub fn reduce<T, F>(items: Vec<T>, initial: T, op: F) -> UtilsResult<T>
    where
        T: Send + Sync + Clone + 'static,
        F: Fn(T, T) -> T + Send + Sync + 'static,
    {
        if items.is_empty() {
            return Ok(initial);
        }

        let op = Arc::new(op);
        let chunk_size = (items.len() / num_cpus::get()).max(1);

        // Split items into chunks
        let chunks: Vec<_> = items
            .chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        let mut handles = Vec::new();
        let mut partial_results = Vec::new();

        for chunk in chunks.into_iter() {
            let op_clone = Arc::clone(&op);
            let initial_clone = initial.clone();

            let handle = thread::spawn(move || {
                chunk
                    .into_iter()
                    .fold(initial_clone, |acc, item| op_clone(acc, item))
            });

            handles.push(handle);
        }

        // Collect partial results
        for handle in handles {
            let result = handle.join().map_err(|_| {
                UtilsError::InvalidParameter(
                    "Thread panicked during parallel reduction".to_string(),
                )
            })?;
            partial_results.push(result);
        }

        // Reduce partial results
        Ok(partial_results
            .into_iter()
            .fold(initial, |acc, partial| op(acc, partial)))
    }

    /// Sum elements in parallel
    pub fn sum<T>(items: Vec<T>) -> UtilsResult<T>
    where
        T: Send + Sync + Clone + std::ops::Add<Output = T> + Zero + 'static,
    {
        Self::reduce(items, T::zero(), |a, b| a + b)
    }

    /// Find minimum element in parallel
    pub fn min<T>(items: Vec<T>) -> UtilsResult<Option<T>>
    where
        T: Send + Sync + Clone + Ord + 'static,
    {
        if items.is_empty() {
            return Ok(None);
        }

        let first = items[0].clone();
        let result = Self::reduce(items, first, |a, b| if a < b { a } else { b })?;
        Ok(Some(result))
    }

    /// Find maximum element in parallel
    pub fn max<T>(items: Vec<T>) -> UtilsResult<Option<T>>
    where
        T: Send + Sync + Clone + Ord + 'static,
    {
        if items.is_empty() {
            return Ok(None);
        }

        let first = items[0].clone();
        let result = Self::reduce(items, first, |a, b| if a > b { a } else { b })?;
        Ok(Some(result))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_thread_pool_creation() {
        let pool = ThreadPool::new(4).unwrap();
        assert_eq!(pool.thread_count(), 4);
    }

    #[test]
    fn test_thread_pool_execution() {
        let pool = ThreadPool::new(2).unwrap();
        let counter = Arc::new(AtomicUsize::new(0));

        for _ in 0..10 {
            let counter_clone = Arc::clone(&counter);
            pool.execute(move || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            })
            .unwrap();
        }

        // Give threads time to complete
        thread::sleep(Duration::from_millis(100));

        assert_eq!(counter.load(Ordering::SeqCst), 10);
    }

    #[test]
    fn test_work_stealing_queue() {
        let queue = WorkStealingQueue::new(4);

        queue.push_local(42).unwrap();
        queue.push_global(24).unwrap();

        assert_eq!(queue.get_task().unwrap(), Some(42));
        assert_eq!(queue.get_task().unwrap(), Some(24));
        assert_eq!(queue.get_task().unwrap(), None);
    }

    #[test]
    fn test_parallel_iterator_map() {
        let items = vec![1, 2, 3, 4, 5];
        let iter = ParallelIterator::new(items);

        let results = iter.map(|x| x * 2).unwrap();
        assert_eq!(results, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_parallel_iterator_filter() {
        let items = vec![1, 2, 3, 4, 5, 6];
        let iter = ParallelIterator::new(items);

        let results = iter.filter(|&x| x % 2 == 0).unwrap();
        assert_eq!(results, vec![2, 4, 6]);
    }

    #[test]
    fn test_parallel_reducer_sum() {
        let items = vec![1, 2, 3, 4, 5];
        let result = ParallelReducer::sum(items).unwrap();
        assert_eq!(result, 15);
    }

    #[test]
    fn test_parallel_reducer_min_max() {
        let items = vec![5, 2, 8, 1, 9, 3];

        let min_result = ParallelReducer::min(items.clone()).unwrap();
        assert_eq!(min_result, Some(1));

        let max_result = ParallelReducer::max(items).unwrap();
        assert_eq!(max_result, Some(9));
    }

    #[test]
    fn test_parallel_reducer_empty() {
        let items: Vec<i32> = vec![];

        let min_result = ParallelReducer::min(items.clone()).unwrap();
        assert_eq!(min_result, None);

        let max_result = ParallelReducer::max(items).unwrap();
        assert_eq!(max_result, None);
    }

    #[test]
    fn test_thread_pool_with_cpu_cores() {
        let pool = ThreadPool::with_cpu_cores().unwrap();
        assert!(pool.thread_count() > 0);
        assert!(pool.thread_count() <= num_cpus::get());
    }
}
