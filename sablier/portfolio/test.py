"""Test class for portfolio performance analysis"""

import json
import os
import sqlite3
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import numpy as np


class Test:
    """
    Represents a single portfolio test against a scenario
    
    A test contains all the performance metrics and analysis results
    from running a portfolio against a specific scenario.
    """
    
    def __init__(self, portfolio_id: str, test_data: dict):
        """
        Initialize Test instance
        
        Args:
            portfolio_id: ID of the portfolio this test belongs to
            test_data: Test data dictionary from SQLite
        """
        self.portfolio_id = portfolio_id
        self.id = test_data['id']
        self.scenario_id = test_data['scenario_id']
        self.scenario_name = test_data['scenario_name']
        self.test_date = test_data['test_date']
        
        # Parse JSON fields
        self.sample_results = json.loads(test_data['sample_results']) if isinstance(test_data['sample_results'], str) else test_data['sample_results']
        self.aggregated_results = json.loads(test_data['aggregated_results']) if isinstance(test_data['aggregated_results'], str) else test_data['aggregated_results']
        self.summary_stats = json.loads(test_data['summary_stats']) if isinstance(test_data['summary_stats'], str) else test_data['summary_stats']
        self.time_series_metrics = json.loads(test_data['time_series_metrics']) if test_data.get('time_series_metrics') and isinstance(test_data['time_series_metrics'], str) else test_data.get('time_series_metrics', {})
        self.n_days = test_data.get('n_days', 0)
    
    def __repr__(self) -> str:
        """Return a concise string representation of the test"""
        n_samples = self.aggregated_results.get('total_samples', len(self.sample_results) if self.sample_results else 0)
        return f"Test(scenario='{self.scenario_name}', n_samples={n_samples})"
    
    # ============================================
    # METRIC-FOCUSED REPORTING METHODS
    # ============================================
    
    def report_aggregated_metrics(self) -> Dict[str, Any]:
        """Report all aggregated static metrics across all samples"""
        return {
            'profitability_rate': self.aggregated_results['profitability_rate'],
            'tail_ratio': self.aggregated_results['tail_ratio'],
            'var_95': self.aggregated_results['var_95'],
            'var_99': self.aggregated_results['var_99'],
            'cvar_95': self.aggregated_results['cvar_95'],
            'cvar_99': self.aggregated_results['cvar_99'],
            'profitable_samples': self.aggregated_results['profitable_samples'],
            'total_samples': self.aggregated_results['total_samples'],
            'return_distribution': self.aggregated_results['return_distribution'],
            'sharpe_distribution': self.aggregated_results['sharpe_distribution'],
            'drawdown_distribution': self.aggregated_results['drawdown_distribution'],
            'average_drawdown_distribution': self.aggregated_results['average_drawdown_distribution'],
            'downside_deviation_distribution': self.aggregated_results['downside_deviation_distribution'],
            'volatility_distribution': self.aggregated_results['volatility_distribution']
        }
    
    def report_sample_metrics(self, sample_idx: int) -> Dict[str, Any]:
        """Report all static metrics for a specific sample"""
        if sample_idx >= len(self.sample_results):
            raise ValueError(f"Sample index {sample_idx} out of range. Available: 0-{len(self.sample_results)-1}")
        
        sample = self.sample_results[sample_idx]
        return {
            'sample_idx': sample['sample_idx'],
            'total_return': sample['total_return'],
            'pnl': sample['pnl'],
            'sharpe_ratio': sample['sharpe_ratio'],
            'sortino_ratio': sample['sortino_ratio'],
            'max_drawdown': sample['max_drawdown'],
            'average_drawdown': sample['average_drawdown'],
            'downside_deviation': sample['downside_deviation'],
            'is_profitable': sample['is_profitable'],
            'initial_value': sample['initial_value'],
            'final_value': sample['final_value']
        }
    
    # ============================================
    # STREAMLINED PLOTTING SYSTEM
    # ============================================
    
    def plot_distribution(self, metric: str, save: bool = False, save_dir: str = None, display: bool = True) -> List[str]:
        """Plot distribution of a metric across all samples (end-of-path metrics only)
        
        Args:
            metric: Metric name ('total_return', 'sharpe_ratio', 'sortino_ratio', 
                   'max_drawdown', 'average_drawdown', 'downside_deviation')
            save: Whether to save the plot
            save_dir: Directory to save plots (default: './portfolio_plots/')
            display: Whether to display the plot
            
        Returns:
            List of saved file paths
        """
        # Set default save directory and create it if needed
        if save:
            if save_dir is None:
                save_dir = './portfolio_plots/'
            os.makedirs(save_dir, exist_ok=True)
        
        # Extract metric values from sample results (end-of-path metrics only)
        end_of_path_metrics = ['total_return', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 
                              'average_drawdown', 'downside_deviation']
        
        if metric not in end_of_path_metrics:
            raise ValueError(f"Unsupported end-of-path metric: {metric}. Available: {end_of_path_metrics}")
        
        values = [s[metric] for s in self.sample_results]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.hist(values, bins=30, alpha=0.7, density=True, color='blue', edgecolor='black')
        
        # Add statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        
        # Add VaR lines for return metrics
        if metric == 'total_return':
            var_95 = np.percentile(values, 5)
            var_99 = np.percentile(values, 1)
            ax.axvline(var_95, color='orange', linestyle='--', linewidth=2, label=f'95% VaR: {var_95:.3f}')
            ax.axvline(var_99, color='darkred', linestyle='--', linewidth=2, label=f'99% VaR: {var_99:.3f}')
        
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.set_title(f'Portfolio {metric.replace("_", " ").title()} Distribution\n{self.scenario_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        saved_files = []
        if save:
            filename = f"{metric}_distribution_{self.scenario_name.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"✅ Saved: {filepath}")
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return saved_files
    
    def plot_evolution(self, metric: str, save: bool = False, save_dir: str = None, display: bool = True) -> List[str]:
        """Plot evolution of a metric over time (aggregated across all samples with confidence intervals)
        
        Args:
            metric: Metric name ('pnl', 'total_return', 'portfolio_value')
            save: Whether to save the plot
            save_dir: Directory to save plots (default: './portfolio_plots/')
            display: Whether to display the plot
            
        Returns:
            List of saved file paths
        """
        # Set default save directory and create it if needed
        if save:
            if save_dir is None:
                save_dir = './portfolio_plots/'
            os.makedirs(save_dir, exist_ok=True)
        
        # Extract time-series data (daily metrics aggregated across samples)
        time_series_metrics = ['pnl', 'returns', 'portfolio_value', 'drawdown']
        
        if metric not in time_series_metrics:
            raise ValueError(f"Unsupported time-series metric: {metric}. Available: {time_series_metrics}")
        
        # Map metric names to actual keys in the data
        metric_key_map = {
            'pnl': 'pnl',
            'total_return': 'returns',  # Map total_return to returns
            'portfolio_value': 'portfolio_value',
            'drawdown': 'drawdown'
        }
        
        actual_metric_key = metric_key_map.get(metric, metric)
        
        # Get time-series metrics from aggregated results
        time_series_data = self.time_series_metrics
        if not time_series_data:
            raise ValueError(f"Time-series data not available")
        
        # Extract daily aggregated data
        days = []
        mean_values = []
        std_values = []
        var_95_values = []
        var_99_values = []
        cvar_95_values = []
        cvar_99_values = []
        
        for day_key, day_data in time_series_data.items():
            if day_key.startswith('day_'):
                days.append(day_data['day'])
                
                if actual_metric_key in day_data:
                    metric_data = day_data[actual_metric_key]
                    mean_values.append(metric_data['mean'])
                    std_values.append(metric_data['std'])
                    
                    # Only add VaR/CVaR for metrics that have them
                    if 'var_95' in metric_data:
                        var_95_values.append(metric_data['var_95'])
                        var_99_values.append(metric_data['var_99'])
                        cvar_95_values.append(metric_data['cvar_95'])
                        cvar_99_values.append(metric_data['cvar_99'])
                    else:
                        var_95_values.append(None)
                        var_99_values.append(None)
                        cvar_95_values.append(None)
                        cvar_99_values.append(None)
        
        if not days:
            raise ValueError(f"No time-series data found for metric: {metric}")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Plot mean line
        ax.plot(days, mean_values, 'b-', linewidth=2, label=f'{metric.replace("_", " ").title()} (Mean)')
        
        # Add confidence intervals
        if metric == 'portfolio_value' and var_95_values[0] is not None:
            # For portfolio value, show VaR/CVaR bands
            ax.fill_between(days, var_95_values, var_99_values, alpha=0.2, color='red', label='95%-99% VaR Band')
            ax.fill_between(days, cvar_95_values, cvar_99_values, alpha=0.3, color='darkred', label='95%-99% CVaR Band')
        else:
            # For other metrics (including drawdown), show ±1σ bands
            upper_band = [m + s for m, s in zip(mean_values, std_values)]
            lower_band = [m - s for m, s in zip(mean_values, std_values)]
            ax.fill_between(days, lower_band, upper_band, alpha=0.3, color='blue', label='±1σ Band')
        
        ax.set_xlabel('Days')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Portfolio {metric.replace("_", " ").title()} Evolution (Aggregated)\n{self.scenario_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        saved_files = []
        if save:
            filename = f"{metric}_evolution_{self.scenario_name.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"✅ Saved: {filepath}")
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return saved_files
    
    def plot_sample_distribution(self, metric: str, sample_idx: int, save: bool = False, save_dir: str = None, display: bool = True) -> List[str]:
        """Plot distribution of daily metric values for a single sample
        
        Args:
            metric: Metric name ('pnl', 'total_return', 'daily_return', 'portfolio_value', 'drawdown')
            sample_idx: Index of the sample to plot
            save: Whether to save the plot
            save_dir: Directory to save plots (default: './portfolio_plots/')
            display: Whether to display the plot
            
        Returns:
            List of saved file paths
        """
        # Set default save directory and create it if needed
        if save:
            if save_dir is None:
                save_dir = './portfolio_plots/'
            os.makedirs(save_dir, exist_ok=True)
        
        if sample_idx >= len(self.sample_results):
            raise ValueError(f"Sample index {sample_idx} out of range (max: {len(self.sample_results)-1})")
        
        # Extract daily metrics for the sample
        sample_data = self.sample_results[sample_idx]
        daily_metrics = sample_data.get('daily_metrics', [])
        
        if not daily_metrics:
            raise ValueError(f"No daily metrics available for sample {sample_idx}")
        
        # Valid daily metrics
        daily_metric_names = ['pnl', 'cumulative_return', 'daily_return', 'portfolio_value', 'drawdown']
        if metric not in daily_metric_names:
            raise ValueError(f"Unsupported daily metric: {metric}. Available: {daily_metric_names}")
        
        # Map metric names to actual keys in the data
        metric_key_map = {
            'pnl': 'pnl',
            'total_return': 'cumulative_return',  # Map total_return to cumulative_return
            'daily_return': 'daily_return',
            'portfolio_value': 'portfolio_value',
            'drawdown': 'drawdown'
        }
        
        actual_metric_key = metric_key_map.get(metric, metric)
        
        # Extract values for the metric across all days
        values = [dm[actual_metric_key] for dm in daily_metrics]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.hist(values, bins=30, alpha=0.7, density=True, color='green', edgecolor='black')
        
        # Add statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1, label=f'+1σ: {mean_val + std_val:.3f}')
        ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1, label=f'-1σ: {mean_val - std_val:.3f}')
        
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.set_title(f'Sample {sample_idx} - {metric.replace("_", " ").title()} Distribution\n{self.scenario_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        saved_files = []
        if save:
            filename = f"sample_{sample_idx}_{metric}_distribution_{self.scenario_name.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"✅ Saved: {filepath}")
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return saved_files
    
    def plot_sample_evolution(self, metric: str, sample_idx: int, save: bool = False, save_dir: str = None, display: bool = True) -> List[str]:
        """Plot evolution of a metric over time for a single sample (no confidence intervals)
        
        Args:
            metric: Metric name ('pnl', 'total_return', 'daily_return', 'portfolio_value', 'drawdown')
            sample_idx: Index of the sample to plot
            save: Whether to save the plot
            save_dir: Directory to save plots (default: './portfolio_plots/')
            display: Whether to display the plot
            
        Returns:
            List of saved file paths
        """
        # Set default save directory and create it if needed
        if save:
            if save_dir is None:
                save_dir = './portfolio_plots/'
            os.makedirs(save_dir, exist_ok=True)
        
        if sample_idx >= len(self.sample_results):
            raise ValueError(f"Sample index {sample_idx} out of range (max: {len(self.sample_results)-1})")
        
        # Extract daily metrics for the sample
        sample_data = self.sample_results[sample_idx]
        daily_metrics = sample_data.get('daily_metrics', [])
        
        if not daily_metrics:
            raise ValueError(f"No daily metrics available for sample {sample_idx}")
        
        # Valid daily metrics
        daily_metric_names = ['pnl', 'cumulative_return', 'daily_return', 'portfolio_value', 'drawdown']
        if metric not in daily_metric_names:
            raise ValueError(f"Unsupported daily metric: {metric}. Available: {daily_metric_names}")
        
        # Map metric names to actual keys in the data
        metric_key_map = {
            'pnl': 'pnl',
            'total_return': 'cumulative_return',  # Map total_return to cumulative_return
            'daily_return': 'daily_return',
            'portfolio_value': 'portfolio_value',
            'drawdown': 'drawdown'
        }
        
        actual_metric_key = metric_key_map.get(metric, metric)
        
        # Extract values and days
        values = [dm[actual_metric_key] for dm in daily_metrics]
        days = list(range(len(values)))
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(days, values, 'b-', linewidth=2, label=f'Sample {sample_idx}')
        
        ax.set_xlabel('Days')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Sample {sample_idx} - {metric.replace("_", " ").title()} Evolution\n{self.scenario_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        saved_files = []
        if save:
            filename = f"sample_{sample_idx}_{metric}_evolution_{self.scenario_name.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"✅ Saved: {filepath}")
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return saved_files
    
    def delete(self) -> bool:
        """Delete this test from the database"""
        import sqlite3
        
        db_path = os.path.expanduser("~/.sablier/portfolios.db")
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("DELETE FROM portfolio_tests WHERE id = ?", (self.id,))
                if cursor.rowcount > 0:
                    conn.commit()
                    print(f"✅ Deleted test {self.id}")
                    return True
                else:
                    print(f"❌ Test {self.id} not found")
                    return False
        except Exception as e:
            print(f"❌ Failed to delete test: {e}")
            return False