//! Integration tests for memory leak detection

use sklears_neural::memory_leak_tests::{MemoryLeakDetector, MemoryLeakTestSuite, MemorySnapshot};
use std::time::Duration;

#[test]
fn test_memory_snapshot_basic() {
    let snapshot = MemorySnapshot::take();
    // On supported platforms, we should get some memory info
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    assert!(snapshot.virtual_memory > 0 || snapshot.resident_memory > 0);
}

#[test]
fn test_memory_detector_basic() {
    let mut detector = MemoryLeakDetector::new();

    detector.snapshot();
    std::thread::sleep(Duration::from_millis(10));
    detector.snapshot();

    let stats = detector.get_statistics();
    assert_eq!(stats.total_snapshots, 2);
}

#[test]
#[ignore] // This test is slow, run manually with --ignored
fn test_memory_leak_test_suite() {
    // This is a comprehensive test that may take some time
    let result = MemoryLeakTestSuite::run_all_tests();
    assert!(
        result.is_ok(),
        "Memory leak test suite failed: {:?}",
        result.err()
    );
}

#[test]
fn test_memory_detector_threshold() {
    let detector = MemoryLeakDetector::new().with_threshold(1_000_000); // 1MB
    assert_eq!(detector.get_leak_threshold(), 1_000_000);
}

#[test]
fn test_memory_snapshot_labeling() {
    let snapshot = MemorySnapshot::take_with_label(Some("test_label".to_string()));
    assert_eq!(snapshot.label, Some("test_label".to_string()));
}

#[test]
fn test_memory_stats() {
    let mut detector = MemoryLeakDetector::new();

    // Take several snapshots
    for i in 0..5 {
        detector.snapshot_with_label(Some(format!("snapshot_{}", i)));
        std::thread::sleep(Duration::from_millis(5));
    }

    let stats = detector.get_statistics();
    assert_eq!(stats.total_snapshots, 5);
    assert!(stats.virtual_memory_avg >= 0.0);
    assert!(stats.resident_memory_avg >= 0.0);
}
