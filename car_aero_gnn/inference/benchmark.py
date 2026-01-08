"""
Benchmark GNN predictions against CFD results

Compares accuracy, speed, and aerodynamic coefficients
"""

import torch
import numpy as np
import time
from pathlib import Path
import json
from typing import Dict, List
from tqdm import tqdm


class AeroBenchmark:
    """
    Benchmark aerodynamic predictions

    Args:
        predictor: SteadyStatePredictor or UnsteadyRollout
        test_loader: Test data loader
    """

    def __init__(self, predictor, test_loader):
        self.predictor = predictor
        self.test_loader = test_loader

    def compute_errors(self, pred, target):
        """
        Compute various error metrics

        Args:
            pred: Predictions [N, D]
            target: Ground truth [N, D]

        Returns:
            errors: Dictionary of error metrics
        """
        # Mean Squared Error
        mse = np.mean((pred - target) ** 2)

        # Mean Absolute Error
        mae = np.mean(np.abs(pred - target))

        # Root Mean Squared Error
        rmse = np.sqrt(mse)

        # Normalized RMSE
        target_range = target.max() - target.min()
        nrmse = rmse / (target_range + 1e-8)

        # R² score
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - target.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)

        # Correlation coefficient
        correlation = np.corrcoef(pred.flatten(), target.flatten())[0, 1]

        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'nrmse': nrmse,
            'r2': r2,
            'correlation': correlation
        }

    def benchmark_accuracy(self):
        """
        Benchmark prediction accuracy

        Returns:
            results: Dictionary with accuracy metrics
        """
        print("=" * 70)
        print("Accuracy Benchmark")
        print("=" * 70)

        velocity_errors = []
        pressure_errors = []
        inference_times = []

        for batch in tqdm(self.test_loader, desc="Testing"):
            # Ground truth
            target_vel = batch.y[:, :3].numpy()
            target_p = batch.y[:, 3].numpy()

            # Predict
            start_time = time.time()
            pred_vel, pred_p = self.predictor.predict(batch)
            inference_time = time.time() - start_time

            # Compute errors
            vel_error = self.compute_errors(pred_vel, target_vel)
            p_error = self.compute_errors(pred_p[:, None], target_p[:, None])

            velocity_errors.append(vel_error)
            pressure_errors.append(p_error)
            inference_times.append(inference_time)

        # Aggregate results
        results = {
            'velocity': self._aggregate_errors(velocity_errors),
            'pressure': self._aggregate_errors(pressure_errors),
            'avg_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times)
        }

        # Print results
        self._print_accuracy_results(results)

        return results

    def _aggregate_errors(self, error_list):
        """Aggregate error metrics across all samples"""
        aggregated = {}
        for key in error_list[0].keys():
            values = [e[key] for e in error_list]
            aggregated[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        return aggregated

    def _print_accuracy_results(self, results):
        """Print accuracy results"""
        print("\nVelocity Field Accuracy:")
        print(f"  RMSE:        {results['velocity']['rmse']['mean']:.6f} ± {results['velocity']['rmse']['std']:.6f}")
        print(f"  MAE:         {results['velocity']['mae']['mean']:.6f} ± {results['velocity']['mae']['std']:.6f}")
        print(f"  R²:          {results['velocity']['r2']['mean']:.4f} ± {results['velocity']['r2']['std']:.4f}")
        print(f"  Correlation: {results['velocity']['correlation']['mean']:.4f}")

        print("\nPressure Field Accuracy:")
        print(f"  RMSE:        {results['pressure']['rmse']['mean']:.6f} ± {results['pressure']['rmse']['std']:.6f}")
        print(f"  MAE:         {results['pressure']['mae']['mean']:.6f} ± {results['pressure']['mae']['std']:.6f}")
        print(f"  R²:          {results['pressure']['r2']['mean']:.4f} ± {results['pressure']['r2']['std']:.4f}")

        print(f"\nInference Time: {results['avg_inference_time']*1000:.2f} ± {results['std_inference_time']*1000:.2f} ms")

    def benchmark_speed(self, cfd_times=None):
        """
        Benchmark inference speed

        Args:
            cfd_times: List of CFD solve times for comparison (optional)

        Returns:
            speed_results: Dictionary with speed metrics
        """
        print("\n" + "=" * 70)
        print("Speed Benchmark")
        print("=" * 70)

        # Warm-up
        for i, batch in enumerate(self.test_loader):
            if i >= 5:
                break
            _ = self.predictor.predict(batch)

        # Timed inference
        inference_times = []

        for batch in tqdm(self.test_loader, desc="Speed test"):
            start_time = time.time()
            _ = self.predictor.predict(batch)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

        results = {
            'mean_time_ms': np.mean(inference_times) * 1000,
            'std_time_ms': np.std(inference_times) * 1000,
            'min_time_ms': np.min(inference_times) * 1000,
            'max_time_ms': np.max(inference_times) * 1000,
            'throughput_samples_per_sec': 1.0 / np.mean(inference_times)
        }

        if cfd_times is not None:
            avg_cfd_time = np.mean(cfd_times)
            results['avg_cfd_time_s'] = avg_cfd_time
            results['speedup_factor'] = avg_cfd_time / np.mean(inference_times)

        # Print results
        self._print_speed_results(results)

        return results

    def _print_speed_results(self, results):
        """Print speed results"""
        print(f"\nGNN Inference Time: {results['mean_time_ms']:.2f} ± {results['std_time_ms']:.2f} ms")
        print(f"Throughput: {results['throughput_samples_per_sec']:.2f} samples/sec")

        if 'speedup_factor' in results:
            print(f"\nCFD Solve Time: {results['avg_cfd_time_s']:.2f} s")
            print(f"Speedup Factor: {results['speedup_factor']:.0f}x")

    def benchmark_drag_coefficient(self, reference_data=None):
        """
        Benchmark drag coefficient prediction

        Args:
            reference_data: Dictionary with reference C_D values

        Returns:
            drag_results: Dictionary with drag coefficient errors
        """
        print("\n" + "=" * 70)
        print("Drag Coefficient Benchmark")
        print("=" * 70)

        if reference_data is None:
            print("No reference data provided. Skipping drag benchmark.")
            return {}

        # This is a placeholder - actual implementation needs
        # proper surface integration and force computation

        print("Drag coefficient benchmark not fully implemented.")
        print("Requires proper surface mesh and pressure integration.")

        return {}

    def generate_report(self, output_path):
        """
        Generate comprehensive benchmark report

        Args:
            output_path: Path to save report
        """
        # Run all benchmarks
        accuracy_results = self.benchmark_accuracy()
        speed_results = self.benchmark_speed()

        # Combine results
        report = {
            'accuracy': accuracy_results,
            'speed': speed_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        # Save to JSON
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nBenchmark report saved to {output_path}")

        return report


def compare_gnn_vs_cfd(
    gnn_predictions,
    cfd_results,
    metrics=['velocity', 'pressure', 'drag']
):
    """
    Compare GNN predictions with CFD results

    Args:
        gnn_predictions: List of GNN predictions
        cfd_results: List of CFD results
        metrics: List of metrics to compare

    Returns:
        comparison: Dictionary with comparison results
    """
    comparison = {}

    for metric in metrics:
        if metric == 'velocity':
            # Compare velocity fields
            errors = []
            for gnn_pred, cfd_ref in zip(gnn_predictions, cfd_results):
                error = np.mean((gnn_pred['velocity'] - cfd_ref['velocity']) ** 2)
                errors.append(error)

            comparison['velocity_mse'] = {
                'mean': np.mean(errors),
                'std': np.std(errors)
            }

        elif metric == 'pressure':
            # Compare pressure fields
            errors = []
            for gnn_pred, cfd_ref in zip(gnn_predictions, cfd_results):
                error = np.mean((gnn_pred['pressure'] - cfd_ref['pressure']) ** 2)
                errors.append(error)

            comparison['pressure_mse'] = {
                'mean': np.mean(errors),
                'std': np.std(errors)
            }

    return comparison


if __name__ == '__main__':
    print("Benchmark module ready")
    print("Usage:")
    print("  benchmark = AeroBenchmark(predictor, test_loader)")
    print("  results = benchmark.benchmark_accuracy()")
    print("  speed_results = benchmark.benchmark_speed()")
