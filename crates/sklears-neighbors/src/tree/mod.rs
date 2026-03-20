//! Tree-based data structures for efficient neighbor search

pub mod ball_tree;
pub mod cover_tree;
pub mod kd_tree;
pub mod vp_tree;

pub use ball_tree::BallTree;
pub use cover_tree::CoverTree;
pub use kd_tree::KdTree;
pub use vp_tree::VpTree;
