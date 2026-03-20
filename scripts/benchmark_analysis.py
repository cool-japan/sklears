#!/usr/bin/env python3
"""
Benchmark analysis script for continuous benchmarking.

This script analyzes benchmark results from criterion and generates
reports for performance tracking.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import statistics
import argparse

class BenchmarkAnalyzer:
    """Analyzes criterion benchmark results."""
    
    def __init__(self, target_dir: Path = Path("target/criterion")):
        self.target_dir = target_dir
        self.results = {}
        
    def load_results(self) -> Dict:
        """Load all benchmark results from criterion output."""
        if not self.target_dir.exists():
            print(f"Benchmark directory {self.target_dir} not found")
            return {}
            
        for group_dir in self.target_dir.iterdir():
            if not group_dir.is_dir():
                continue
                
            group_name = group_dir.name
            self.results[group_name] = {}
            
            for bench_dir in group_dir.iterdir():
                if not bench_dir.is_dir():
                    continue
                    
                bench_name = bench_dir.name
                estimates_file = bench_dir / "base" / "estimates.json"
                
                if estimates_file.exists():
                    with open(estimates_file) as f:
                        data = json.load(f)
                        self.results[group_name][bench_name] = {
                            'mean': data.get('mean', {}).get('point_estimate', 0),
                            'std_dev': data.get('std_dev', {}).get('point_estimate', 0),
                            'median': data.get('median', {}).get('point_estimate', 0),
                        }
                        
        return self.results
    
    def compare_with_baseline(self, baseline_dir: Path) -> Dict:
        """Compare current results with baseline."""
        baseline_analyzer = BenchmarkAnalyzer(baseline_dir)
        baseline_results = baseline_analyzer.load_results()
        
        comparison = {}
        
        for group, benchmarks in self.results.items():
            if group not in baseline_results:
                continue
                
            comparison[group] = {}
            
            for bench, current in benchmarks.items():
                if bench not in baseline_results[group]:
                    continue
                    
                baseline = baseline_results[group][bench]
                
                # Calculate percentage change
                if baseline['mean'] > 0:
                    change = ((current['mean'] - baseline['mean']) / baseline['mean']) * 100
                else:
                    change = 0
                    
                comparison[group][bench] = {
                    'baseline': baseline['mean'],
                    'current': current['mean'],
                    'change_percent': change,
                    'regression': change > 10,  # Consider >10% slower as regression
                }
                
        return comparison
    
    def generate_report(self, output_format: str = "markdown") -> str:
        """Generate a performance report."""
        if output_format == "markdown":
            return self._generate_markdown_report()
        elif output_format == "json":
            return json.dumps(self.results, indent=2)
        else:
            raise ValueError(f"Unknown format: {output_format}")
    
    def _generate_markdown_report(self) -> str:
        """Generate a markdown report."""
        lines = ["# Benchmark Results\n"]
        
        for group, benchmarks in sorted(self.results.items()):
            lines.append(f"\n## {group}\n")
            lines.append("| Benchmark | Mean Time | Std Dev | Median |")
            lines.append("|-----------|-----------|---------|--------|")
            
            for bench, results in sorted(benchmarks.items()):
                mean = self._format_time(results['mean'])
                std = self._format_time(results['std_dev'])
                median = self._format_time(results['median'])
                lines.append(f"| {bench} | {mean} | {std} | {median} |")
                
        return "\n".join(lines)
    
    def _format_time(self, nanoseconds: float) -> str:
        """Format time in appropriate units."""
        if nanoseconds < 1000:
            return f"{nanoseconds:.1f} ns"
        elif nanoseconds < 1_000_000:
            return f"{nanoseconds/1000:.1f} µs"
        elif nanoseconds < 1_000_000_000:
            return f"{nanoseconds/1_000_000:.1f} ms"
        else:
            return f"{nanoseconds/1_000_000_000:.2f} s"
    
    def check_for_regressions(self, threshold: float = 0.1) -> List[str]:
        """Check for performance regressions."""
        regressions = []
        
        # This would compare with historical data
        # For now, just check if any benchmark is suspiciously slow
        for group, benchmarks in self.results.items():
            for bench, results in benchmarks.items():
                # Simple heuristic: warn if std_dev is >20% of mean
                if results['std_dev'] > results['mean'] * 0.2:
                    regressions.append(
                        f"{group}/{bench}: High variance detected "
                        f"(std dev is {results['std_dev']/results['mean']*100:.1f}% of mean)"
                    )
                    
        return regressions

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("target/criterion"),
        help="Directory containing benchmark results"
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        help="Directory containing baseline results for comparison"
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format"
    )
    parser.add_argument(
        "--check-regressions",
        action="store_true",
        help="Check for performance regressions"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file (default: stdout)"
    )
    
    args = parser.parse_args()
    
    analyzer = BenchmarkAnalyzer(args.target_dir)
    analyzer.load_results()
    
    if args.baseline_dir:
        comparison = analyzer.compare_with_baseline(args.baseline_dir)
        print("## Performance Comparison\n")
        
        for group, benchmarks in comparison.items():
            print(f"\n### {group}\n")
            print("| Benchmark | Baseline | Current | Change | Status |")
            print("|-----------|----------|---------|--------|--------|")
            
            for bench, data in benchmarks.items():
                baseline = analyzer._format_time(data['baseline'])
                current = analyzer._format_time(data['current'])
                change = f"{data['change_percent']:+.1f}%"
                status = "⚠️ REGRESSION" if data['regression'] else "✅ OK"
                print(f"| {bench} | {baseline} | {current} | {change} | {status} |")
    
    report = analyzer.generate_report(args.format)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
    else:
        print(report)
    
    if args.check_regressions:
        regressions = analyzer.check_for_regressions()
        if regressions:
            print("\n## ⚠️ Potential Issues\n")
            for regression in regressions:
                print(f"- {regression}")
            sys.exit(1)

if __name__ == "__main__":
    main()