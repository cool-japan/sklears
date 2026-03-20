//! # RingBufferIter - Trait Implementations
//!
//! This module contains trait implementations for `RingBufferIter`.
//!
//! ## Implemented Traits
//!
//! - `Iterator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::RingBufferIter;

impl<'a, T: Clone> Iterator for RingBufferIter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.buffer.size {
            return None;
        }
        let actual_index = (self.buffer.head + self.index) % self.buffer.capacity;
        self.index += 1;
        self.buffer.data[actual_index].as_ref()
    }
}
