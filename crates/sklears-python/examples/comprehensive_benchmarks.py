#!/usr/bin/env python3
"""
Comprehensive Performance Benchmarking Suite for Sklears vs Scikit-learn

This script provides a detailed performance comparison between sklears and scikit-learn
across multiple algorithms, dataset sizes, and scenarios. It generates detailed reports
and visualizations to demonstrate the performance advantages of sklears.

Usage:
    python comprehensive_benchmarks.py [--output-dir results] [--run-small] [--run-large]

Requirements:
    - sklears (built with maturin develop)
    - scikit-learn
    - numpy
    - matplotlib (optional, for plots)
    - seaborn (optional, for better plots)
    - pandas (optional, for data handling)
"""

import argparse
import time
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import warnings

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Try to import optional dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Matplotlib/Seaborn not available. Plots will be skipped.")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Pandas not available. Some features will be limited.")

# Import libraries to benchmark
import sklears as skl

try:
    from sklearn.linear_model import LinearRegression as SklearnLR
    from sklearn.linear_model import Ridge as SklearnRidge
    from sklearn.linear_model import Lasso as SklearnLasso
    from sklearn.linear_model import LogisticRegression as SklearnLogistic
    from sklearn.cluster import KMeans as SklearnKMeans
    from sklearn.cluster import DBSCAN as SklearnDBSCAN
    # TODO: Coming soon - StandardScaler and MinMaxScaler not yet exposed via skl
    # from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
    # from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Scikit-learn not available. Only sklears benchmarks will be run.")

def _numpy_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R2 score using numpy (fallback when sklearn metrics unavailable)"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot != 0 else 0.0

@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    algorithm: str
    library: str
    dataset_size: Tuple[int, int]  # (n_samples, n_features)
    fit_time: float
    predict_time: float
    total_time: float
    memory_peak: Optional[float] = None
    accuracy_metric: Optional[float] = None
    error_message: Optional[str] = None

@dataclass
class BenchmarkSummary:
    """Summary of benchmark results"""
    results: List[BenchmarkResult]
    total_runtime: float
    system_info: Dict[str, Any]

class PerformanceBenchmarker:
    """Comprehensive performance benchmarking suite"""

    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = output_dir
        self.results: List[BenchmarkResult] = []

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Configure random seed for reproducibility
        self.random_state = 42
        np.random.seed(self.random_state)

    def _time_operation(self, operation, *args, **kwargs) -> Tuple[Any, float]:
        """Time a single operation and return result and elapsed time"""
        start_time = time.perf_counter()
        try:
            result = operation(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            return result, elapsed
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            return None, elapsed

    def _generate_regression_data(self, n_samples: int, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate regression dataset"""
        return datasets.make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=0.1,
            random_state=self.random_state
        )

    def _generate_classification_data(self, n_samples: int, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate classification dataset"""
        return datasets.make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=2,
            n_informative=n_features // 2,
            random_state=self.random_state
        )

    def _generate_clustering_data(self, n_samples: int, n_features: int) -> np.ndarray:
        """Generate clustering dataset"""
        X, _ = datasets.make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=4,
            cluster_std=1.0,
            random_state=self.random_state
        )
        return X

    def benchmark_linear_regression(self, dataset_sizes: List[Tuple[int, int]]):
        """Benchmark linear regression algorithms"""
        print("Benchmarking Linear Regression...")

        for n_samples, n_features in dataset_sizes:
            print(f"  Dataset size: {n_samples} x {n_features}")

            # Generate data
            X, y = self._generate_regression_data(n_samples, n_features)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)

            # Benchmark sklears
            try:
                model = skl.LinearRegression()
                fitted_model, fit_time = self._time_operation(model.fit, X_train, y_train)
                predictions, predict_time = self._time_operation(model.predict, X_test)

                if fitted_model is not None and predictions is not None:
                    # Use numpy fallback since skl.r2_score is not yet exposed
                    r2 = _numpy_r2_score(y_test, predictions)
                    self.results.append(BenchmarkResult(
                        algorithm="LinearRegression",
                        library="sklears",
                        dataset_size=(n_samples, n_features),
                        fit_time=fit_time,
                        predict_time=predict_time,
                        total_time=fit_time + predict_time,
                        accuracy_metric=r2
                    ))
            except Exception as e:
                self.results.append(BenchmarkResult(
                    algorithm="LinearRegression",
                    library="sklears",
                    dataset_size=(n_samples, n_features),
                    fit_time=0,
                    predict_time=0,
                    total_time=0,
                    error_message=str(e)
                ))

            # Benchmark scikit-learn
            if HAS_SKLEARN:
                try:
                    model = SklearnLR()
                    fitted_model, fit_time = self._time_operation(model.fit, X_train, y_train)
                    predictions, predict_time = self._time_operation(model.predict, X_test)

                    if fitted_model is not None and predictions is not None:
                        r2 = r2_score(y_test, predictions)
                        self.results.append(BenchmarkResult(
                            algorithm="LinearRegression",
                            library="sklearn",
                            dataset_size=(n_samples, n_features),
                            fit_time=fit_time,
                            predict_time=predict_time,
                            total_time=fit_time + predict_time,
                            accuracy_metric=r2
                        ))
                except Exception as e:
                    self.results.append(BenchmarkResult(
                        algorithm="LinearRegression",
                        library="sklearn",
                        dataset_size=(n_samples, n_features),
                        fit_time=0,
                        predict_time=0,
                        total_time=0,
                        error_message=str(e)
                    ))

    def benchmark_clustering(self, dataset_sizes: List[Tuple[int, int]]):
        """Benchmark clustering algorithms"""
        print("Benchmarking Clustering...")

        for n_samples, n_features in dataset_sizes:
            print(f"  Dataset size: {n_samples} x {n_features}")

            # Generate data
            X = self._generate_clustering_data(n_samples, n_features)

            # K-Means benchmarking
            algorithms = [
                ("KMeans", skl.KMeans(n_clusters=4, random_state=self.random_state), "sklears"),
            ]

            if HAS_SKLEARN:
                algorithms.append(("KMeans", SklearnKMeans(n_clusters=4, random_state=self.random_state, n_init=10), "sklearn"))

            for alg_name, model, library in algorithms:
                try:
                    labels, total_time = self._time_operation(model.fit_predict, X)

                    if labels is not None:
                        n_clusters = len(np.unique(labels))
                        self.results.append(BenchmarkResult(
                            algorithm=alg_name,
                            library=library,
                            dataset_size=(n_samples, n_features),
                            fit_time=total_time,  # fit_predict combines both
                            predict_time=0,
                            total_time=total_time,
                            accuracy_metric=float(n_clusters)
                        ))
                except Exception as e:
                    self.results.append(BenchmarkResult(
                        algorithm=alg_name,
                        library=library,
                        dataset_size=(n_samples, n_features),
                        fit_time=0,
                        predict_time=0,
                        total_time=0,
                        error_message=str(e)
                    ))

    # TODO: Coming soon - benchmark_preprocessing() requires StandardScaler/MinMaxScaler
    # which are not yet exposed in the sklears module.
    # def benchmark_preprocessing(self, dataset_sizes: List[Tuple[int, int]]):
    #     """Benchmark preprocessing algorithms"""
    #     print("Benchmarking Preprocessing...")
    #
    #     for n_samples, n_features in dataset_sizes:
    #         print(f"  Dataset size: {n_samples} x {n_features}")
    #
    #         X = np.random.randn(n_samples, n_features) * 10 + 5
    #
    #         scalers = [
    #             ("StandardScaler", skl.StandardScaler(), "sklears"),
    #         ]
    #
    #         if HAS_SKLEARN:
    #             scalers.append(("StandardScaler", SklearnStandardScaler(), "sklearn"))
    #
    #         for scaler_name, scaler, library in scalers:
    #             try:
    #                 fitted_scaler, fit_time = self._time_operation(scaler.fit, X)
    #                 X_scaled, transform_time = self._time_operation(scaler.transform, X)
    #
    #                 if fitted_scaler is not None and X_scaled is not None:
    #                     mean_close_to_zero = np.abs(X_scaled.mean()) < 0.01
    #                     std_close_to_one = np.abs(X_scaled.std() - 1.0) < 0.01
    #                     accuracy = float(mean_close_to_zero and std_close_to_one)
    #
    #                     self.results.append(BenchmarkResult(
    #                         algorithm=scaler_name,
    #                         library=library,
    #                         dataset_size=(n_samples, n_features),
    #                         fit_time=fit_time,
    #                         predict_time=transform_time,
    #                         total_time=fit_time + transform_time,
    #                         accuracy_metric=accuracy
    #                     ))
    #             except Exception as e:
    #                 self.results.append(BenchmarkResult(
    #                     algorithm=scaler_name,
    #                     library=library,
    #                     dataset_size=(n_samples, n_features),
    #                     fit_time=0,
    #                     predict_time=0,
    #                     total_time=0,
    #                     error_message=str(e)
    #                 ))

    def run_comprehensive_benchmark(self,
                                  small_datasets: bool = True,
                                  large_datasets: bool = True):
        """Run comprehensive benchmark suite"""
        print("Starting Comprehensive Benchmark Suite")
        print("=" * 50)

        start_time = time.time()

        # Define dataset sizes
        small_sizes = [(100, 5), (500, 10), (1000, 20)]
        medium_sizes = [(5000, 50), (10000, 100)]
        large_sizes = [(50000, 200), (100000, 500)]

        dataset_sizes = []
        if small_datasets:
            dataset_sizes.extend(small_sizes)
            dataset_sizes.extend(medium_sizes)
        if large_datasets:
            dataset_sizes.extend(large_sizes)

        # Run benchmarks
        self.benchmark_linear_regression(dataset_sizes)
        self.benchmark_clustering(dataset_sizes)
        # benchmark_preprocessing skipped: TODO: Coming soon (StandardScaler/MinMaxScaler not yet exposed)

        total_runtime = time.time() - start_time

        # Generate summary
        system_info = {
            "sklears_version": skl.get_version(),
            "sklears_build_info": skl.get_build_info(),
            # TODO: Coming soon - get_hardware_info() not yet exposed
            # "hardware_info": skl.get_hardware_info(),
        }

        if HAS_SKLEARN:
            import sklearn
            system_info["sklearn_version"] = sklearn.__version__

        summary = BenchmarkSummary(
            results=self.results,
            total_runtime=total_runtime,
            system_info=system_info
        )

        print(f"\nBenchmark completed in {total_runtime:.2f} seconds")
        print(f"Total results: {len(self.results)}")

        return summary

    def save_results(self, summary: BenchmarkSummary, filename: str = "benchmark_results.json"):
        """Save benchmark results to JSON file"""
        filepath = os.path.join(self.output_dir, filename)

        # Convert to JSON-serializable format
        data = {
            "results": [asdict(result) for result in summary.results],
            "total_runtime": summary.total_runtime,
            "system_info": summary.system_info
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"Results saved to {filepath}")

    def generate_report(self, summary: BenchmarkSummary) -> str:
        """Generate a detailed text report"""
        report = []
        report.append("SKLEARS PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 50)
        report.append("")

        # System information
        report.append("SYSTEM INFORMATION")
        report.append("-" * 30)
        report.append(f"Sklears version: {summary.system_info.get('sklears_version', 'Unknown')}")

        if 'sklearn_version' in summary.system_info:
            report.append(f"Scikit-learn version: {summary.system_info['sklearn_version']}")

        # TODO: Coming soon - hardware_info not yet in system_info (get_hardware_info not exposed)
        # if 'hardware_info' in summary.system_info:
        #     hw_info = summary.system_info['hardware_info']
        #     report.append(f"CPU cores: {hw_info.get('num_cpus', 'Unknown')}")
        #     report.append(f"SIMD support: {hw_info.get('avx2', False) or hw_info.get('neon', False)}")

        report.append(f"Total runtime: {summary.total_runtime:.2f} seconds")
        report.append("")

        # Performance results by algorithm
        algorithms = set(result.algorithm for result in summary.results if result.error_message is None)

        for algorithm in sorted(algorithms):
            report.append(f"ALGORITHM: {algorithm}")
            report.append("-" * 30)

            alg_results = [r for r in summary.results
                          if r.algorithm == algorithm and r.error_message is None]

            if not alg_results:
                report.append("No successful results found.")
                report.append("")
                continue

            # Group by dataset size
            by_size = {}
            for result in alg_results:
                size_key = f"{result.dataset_size[0]}x{result.dataset_size[1]}"
                if size_key not in by_size:
                    by_size[size_key] = {}
                by_size[size_key][result.library] = result

            for size_key in sorted(by_size.keys(), key=lambda x: int(x.split('x')[0])):
                report.append(f"  Dataset size: {size_key}")

                size_results = by_size[size_key]
                sklears_result = size_results.get('sklears')
                sklearn_result = size_results.get('sklearn')

                if sklears_result:
                    report.append(f"    Sklears: {sklears_result.total_time:.4f}s")

                if sklearn_result:
                    report.append(f"    Sklearn: {sklearn_result.total_time:.4f}s")

                    if sklears_result and sklearn_result.total_time > 0:
                        speedup = sklearn_result.total_time / sklears_result.total_time
                        report.append(f"    Speedup: {speedup:.2f}x")

                report.append("")

        # Error summary
        errors = [r for r in summary.results if r.error_message is not None]
        if errors:
            report.append("ERRORS ENCOUNTERED")
            report.append("-" * 30)
            for error in errors:
                report.append(f"{error.algorithm} ({error.library}): {error.error_message}")
            report.append("")

        # Overall statistics
        successful_results = [r for r in summary.results if r.error_message is None]
        if successful_results:
            report.append("OVERALL STATISTICS")
            report.append("-" * 30)

            sklears_results = [r for r in successful_results if r.library == 'sklears']
            sklearn_results = [r for r in successful_results if r.library == 'sklearn']

            if sklears_results:
                avg_sklears = sum(r.total_time for r in sklears_results) / len(sklears_results)
                report.append(f"Average Sklears time: {avg_sklears:.4f}s")

            if sklearn_results:
                avg_sklearn = sum(r.total_time for r in sklearn_results) / len(sklearn_results)
                report.append(f"Average Sklearn time: {avg_sklearn:.4f}s")

                if sklears_results and avg_sklearn > 0:
                    overall_speedup = avg_sklearn / avg_sklears
                    report.append(f"Overall average speedup: {overall_speedup:.2f}x")

        return "\n".join(report)

    def save_report(self, summary: BenchmarkSummary, filename: str = "benchmark_report.txt"):
        """Save the detailed report to a text file"""
        report = self.generate_report(summary)
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w') as f:
            f.write(report)

        print(f"Report saved to {filepath}")
        return report

    def plot_results(self, summary: BenchmarkSummary):
        """Generate performance comparison plots"""
        if not HAS_PLOTTING:
            print("Plotting libraries not available. Skipping plots.")
            return

        successful_results = [r for r in summary.results if r.error_message is None]

        if not successful_results:
            print("No successful results to plot.")
            return

        # Set up plotting style
        plt.style.use('default')
        if HAS_PLOTTING:
            sns.set_palette("husl")

        # Group results by algorithm
        algorithms = set(r.algorithm for r in successful_results)

        for algorithm in algorithms:
            alg_results = [r for r in successful_results if r.algorithm == algorithm]

            if len(alg_results) < 2:
                continue

            # Create subplot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'{algorithm} Performance Comparison', fontsize=16)

            # Prepare data for plotting
            sizes = []
            sklears_times = []
            sklearn_times = []
            speedups = []

            by_size = {}
            for result in alg_results:
                size_key = result.dataset_size[0] * result.dataset_size[1]  # Total elements
                if size_key not in by_size:
                    by_size[size_key] = {}
                by_size[size_key][result.library] = result

            for size_key in sorted(by_size.keys()):
                size_results = by_size[size_key]
                sklears_result = size_results.get('sklears')
                sklearn_result = size_results.get('sklearn')

                if sklears_result and sklearn_result:
                    sizes.append(size_key)
                    sklears_times.append(sklears_result.total_time)
                    sklearn_times.append(sklearn_result.total_time)
                    speedups.append(sklearn_result.total_time / sklears_result.total_time)

            if sizes:
                # Plot 1: Execution times
                ax1.loglog(sizes, sklears_times, 'o-', label='Sklears', linewidth=2, markersize=8)
                ax1.loglog(sizes, sklearn_times, 's-', label='Sklearn', linewidth=2, markersize=8)
                ax1.set_xlabel('Dataset Size (total elements)')
                ax1.set_ylabel('Execution Time (seconds)')
                ax1.set_title('Execution Time Comparison')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # Plot 2: Speedup
                ax2.semilogx(sizes, speedups, 'o-', color='green', linewidth=2, markersize=8)
                ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
                ax2.set_xlabel('Dataset Size (total elements)')
                ax2.set_ylabel('Speedup (sklearn time / sklears time)')
                ax2.set_title('Sklears Speedup over Sklearn')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                # Add speedup annotations
                for i, (size, speedup) in enumerate(zip(sizes, speedups)):
                    ax2.annotate(f'{speedup:.1f}x',
                               (size, speedup),
                               textcoords="offset points",
                               xytext=(0, 10),
                               ha='center')

            plt.tight_layout()

            # Save plot
            plot_filename = f"{algorithm.lower()}_performance.png"
            plot_filepath = os.path.join(self.output_dir, plot_filename)
            plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {plot_filepath}")

            plt.close()

def main():
    """Main benchmarking function"""
    parser = argparse.ArgumentParser(description='Comprehensive Sklears Performance Benchmarks')
    parser.add_argument('--output-dir', default='benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--run-small', action='store_true', default=True,
                       help='Run benchmarks on small to medium datasets')
    parser.add_argument('--run-large', action='store_true', default=False,
                       help='Run benchmarks on large datasets (warning: may take a long time)')
    parser.add_argument('--no-plots', action='store_true', default=False,
                       help='Skip generating plots')

    args = parser.parse_args()

    # Suppress sklearn warnings
    warnings.filterwarnings('ignore')

    # Initialize benchmarker
    benchmarker = PerformanceBenchmarker(args.output_dir)

    # Run benchmarks
    summary = benchmarker.run_comprehensive_benchmark(
        small_datasets=args.run_small,
        large_datasets=args.run_large
    )

    # Save results and generate report
    benchmarker.save_results(summary)
    report = benchmarker.save_report(summary)

    # Generate plots
    if not args.no_plots:
        benchmarker.plot_results(summary)

    # Print summary to console
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    print(report)

    print(f"\nAll results saved to: {args.output_dir}/")

if __name__ == "__main__":
    main()
