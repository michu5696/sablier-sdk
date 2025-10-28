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
    
    # ============================================
    # METRIC-FOCUSED REPORTING METHODS
    # ============================================
    
    def report_aggregated_metrics(self) -> Dict[str, Any]:
        """Report all aggregated static metrics across all samples"""
        return {
            'survival_rate': self.aggregated_results['survival_rate'],
            'profit_probability': self.aggregated_results['profit_probability'],
            'tail_ratio': self.aggregated_results['tail_ratio'],
            'var_95': self.aggregated_results['var_95'],
            'var_99': self.aggregated_results['var_99'],
            'cvar_95': self.aggregated_results['cvar_95'],
            'cvar_99': self.aggregated_results['cvar_99'],
            'profitable_samples': self.aggregated_results['profitable_samples'],
            'surviving_samples': self.aggregated_results['surviving_samples'],
            'total_samples': self.aggregated_results['total_samples'],
            'return_distribution': self.aggregated_results['return_distribution'],
            'sharpe_distribution': self.aggregated_results['sharpe_distribution'],
            'drawdown_distribution': self.aggregated_results['drawdown_distribution'],
            'information_ratio_distribution': self.aggregated_results['information_ratio_distribution'],
            'average_drawdown_distribution': self.aggregated_results['average_drawdown_distribution'],
            'downside_deviation_distribution': self.aggregated_results['downside_deviation_distribution']
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
            'calmar_ratio': sample['calmar_ratio'],
            'information_ratio': sample['information_ratio'],
            'max_drawdown': sample['max_drawdown'],
            'average_drawdown': sample['average_drawdown'],
            'downside_deviation': sample['downside_deviation'],
            'is_profitable': sample['is_profitable'],
            'survives': sample['survives'],
            'initial_value': sample['initial_value'],
            'final_value': sample['final_value']
        }
    
    # ============================================
    # METRIC-FOCUSED DISTRIBUTION PLOTTING METHODS
    # ============================================
    
    def plot_return_distribution(self, save: bool = False, save_dir: str = None, display: bool = True) -> List[str]:
        """Plot distribution of total returns across all samples"""
        # Set default save directory and create it if needed
        if save:
            if save_dir is None:
                save_dir = './portfolio_plots/'
            os.makedirs(save_dir, exist_ok=True)
        
        returns = [s['total_return'] for s in self.sample_results]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.hist(returns, bins=30, alpha=0.7, density=True, color='blue', edgecolor='black')
        
        # Add statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        ax.axvline(mean_return, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_return:.3f}')
        ax.axvline(var_95, color='orange', linestyle='--', linewidth=2, label=f'95% VaR: {var_95:.3f}')
        ax.axvline(var_99, color='darkred', linestyle='--', linewidth=2, label=f'99% VaR: {var_99:.3f}')
        
        ax.set_xlabel('Total Return')
        ax.set_ylabel('Density')
        ax.set_title(f'Portfolio Return Distribution\n{self.scenario_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        saved_files = []
        if save:
            filename = f"return_distribution_{self.scenario_name.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"✅ Saved: {filepath}")
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return saved_files
    
    def plot_sharpe_distribution(self, save: bool = False, save_dir: str = None, display: bool = True) -> List[str]:
        """Plot distribution of Sharpe ratios across all samples"""
        # Set default save directory and create it if needed
        if save:
            if save_dir is None:
                save_dir = './portfolio_plots/'
            os.makedirs(save_dir, exist_ok=True)
        
        sharpe_ratios = [s['sharpe_ratio'] for s in self.sample_results]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.hist(sharpe_ratios, bins=30, alpha=0.7, density=True, color='green', edgecolor='black')
        
        # Add statistics
        mean_sharpe = np.mean(sharpe_ratios)
        std_sharpe = np.std(sharpe_ratios)
        
        ax.axvline(mean_sharpe, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_sharpe:.3f}')
        ax.axvline(mean_sharpe + std_sharpe, color='orange', linestyle=':', linewidth=1, label=f'+1σ: {mean_sharpe + std_sharpe:.3f}')
        ax.axvline(mean_sharpe - std_sharpe, color='orange', linestyle=':', linewidth=1, label=f'-1σ: {mean_sharpe - std_sharpe:.3f}')
        
        ax.set_xlabel('Sharpe Ratio')
        ax.set_ylabel('Density')
        ax.set_title(f'Portfolio Sharpe Ratio Distribution\n{self.scenario_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        saved_files = []
        if save:
            filename = f"sharpe_distribution_{self.scenario_name.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"✅ Saved: {filepath}")
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return saved_files
    
    def plot_information_ratio_distribution(self, save: bool = False, save_dir: str = None, display: bool = True) -> List[str]:
        """Plot distribution of Information ratios across all samples"""
        # Set default save directory and create it if needed
        if save:
            if save_dir is None:
                save_dir = './portfolio_plots/'
            os.makedirs(save_dir, exist_ok=True)
        
        info_ratios = [s['information_ratio'] for s in self.sample_results]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.hist(info_ratios, bins=30, alpha=0.7, density=True, color='purple', edgecolor='black')
        
        # Add statistics
        mean_info = np.mean(info_ratios)
        std_info = np.std(info_ratios)
        
        ax.axvline(mean_info, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_info:.3f}')
        ax.axvline(mean_info + std_info, color='orange', linestyle=':', linewidth=1, label=f'+1σ: {mean_info + std_info:.3f}')
        ax.axvline(mean_info - std_info, color='orange', linestyle=':', linewidth=1, label=f'-1σ: {mean_info - std_info:.3f}')
        
        ax.set_xlabel('Information Ratio')
        ax.set_ylabel('Density')
        ax.set_title(f'Portfolio Information Ratio Distribution\n{self.scenario_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        saved_files = []
        if save:
            filename = f"information_ratio_distribution_{self.scenario_name.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"✅ Saved: {filepath}")
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return saved_files
    
    def plot_max_drawdown_distribution(self, save: bool = False, save_dir: str = None, display: bool = True) -> List[str]:
        """Plot distribution of max drawdowns across all samples"""
        # Set default save directory and create it if needed
        if save:
            if save_dir is None:
                save_dir = './portfolio_plots/'
            os.makedirs(save_dir, exist_ok=True)
        
        max_drawdowns = [s['max_drawdown'] for s in self.sample_results]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.hist(max_drawdowns, bins=30, alpha=0.7, density=True, color='red', edgecolor='black')
        
        # Add statistics
        mean_dd = np.mean(max_drawdowns)
        std_dd = np.std(max_drawdowns)
        max_dd = np.max(max_drawdowns)
        
        ax.axvline(mean_dd, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_dd:.3f}')
        ax.axvline(max_dd, color='darkred', linestyle='--', linewidth=2, label=f'Max: {max_dd:.3f}')
        
        ax.set_xlabel('Max Drawdown')
        ax.set_ylabel('Density')
        ax.set_title(f'Portfolio Max Drawdown Distribution\n{self.scenario_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        saved_files = []
        if save:
            filename = f"max_drawdown_distribution_{self.scenario_name.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"✅ Saved: {filepath}")
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return saved_files
    
    def plot_average_drawdown_distribution(self, save: bool = False, save_dir: str = None, display: bool = True) -> List[str]:
        """Plot distribution of average drawdowns across all samples"""
        # Set default save directory and create it if needed
        if save:
            if save_dir is None:
                save_dir = './portfolio_plots/'
            os.makedirs(save_dir, exist_ok=True)
        
        avg_drawdowns = [s['average_drawdown'] for s in self.sample_results]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.hist(avg_drawdowns, bins=30, alpha=0.7, density=True, color='orange', edgecolor='black')
        
        # Add statistics
        mean_avg_dd = np.mean(avg_drawdowns)
        std_avg_dd = np.std(avg_drawdowns)
        
        ax.axvline(mean_avg_dd, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_avg_dd:.3f}')
        ax.axvline(mean_avg_dd + std_avg_dd, color='red', linestyle=':', linewidth=1, label=f'+1σ: {mean_avg_dd + std_avg_dd:.3f}')
        ax.axvline(mean_avg_dd - std_avg_dd, color='red', linestyle=':', linewidth=1, label=f'-1σ: {mean_avg_dd - std_avg_dd:.3f}')
        
        ax.set_xlabel('Average Drawdown')
        ax.set_ylabel('Density')
        ax.set_title(f'Portfolio Average Drawdown Distribution\n{self.scenario_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        saved_files = []
        if save:
            filename = f"average_drawdown_distribution_{self.scenario_name.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"✅ Saved: {filepath}")
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return saved_files
    
    def plot_downside_deviation_distribution(self, save: bool = False, save_dir: str = None, display: bool = True) -> List[str]:
        """Plot distribution of downside deviations across all samples"""
        # Set default save directory and create it if needed
        if save:
            if save_dir is None:
                save_dir = './portfolio_plots/'
            os.makedirs(save_dir, exist_ok=True)
        
        downside_deviations = [s['downside_deviation'] for s in self.sample_results]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.hist(downside_deviations, bins=30, alpha=0.7, density=True, color='brown', edgecolor='black')
        
        # Add statistics
        mean_dd_dev = np.mean(downside_deviations)
        std_dd_dev = np.std(downside_deviations)
        
        ax.axvline(mean_dd_dev, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_dd_dev:.3f}')
        ax.axvline(mean_dd_dev + std_dd_dev, color='red', linestyle=':', linewidth=1, label=f'+1σ: {mean_dd_dev + std_dd_dev:.3f}')
        ax.axvline(mean_dd_dev - std_dd_dev, color='red', linestyle=':', linewidth=1, label=f'-1σ: {mean_dd_dev - std_dd_dev:.3f}')
        
        ax.set_xlabel('Downside Deviation')
        ax.set_ylabel('Density')
        ax.set_title(f'Portfolio Downside Deviation Distribution\n{self.scenario_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        saved_files = []
        if save:
            filename = f"downside_deviation_distribution_{self.scenario_name.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"✅ Saved: {filepath}")
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return saved_files
    
    # ============================================
    # METRIC-FOCUSED TIME-SERIES PLOTTING METHODS (AGGREGATED)
    # ============================================
    
    def plot_pnl_evolution(self, save: bool = False, save_dir: str = None, display: bool = True) -> List[str]:
        """Plot PnL evolution over time (aggregated across samples)"""
        # Set default save directory and create it if needed
        if save:
            if save_dir is None:
                save_dir = './portfolio_plots/'
            os.makedirs(save_dir, exist_ok=True)
        
        if not self.time_series_metrics:
            raise ValueError("No time-series data found. This test may be from an older version.")
        
        days = []
        mean_pnl = []
        var_95_pnl = []
        var_99_pnl = []
        std_pnl = []
        
        for day_key, day_data in self.time_series_metrics.items():
            days.append(day_data['day'])
            mean_pnl.append(day_data['pnl']['mean'])
            var_95_pnl.append(day_data['pnl']['var_95'])
            var_99_pnl.append(day_data['pnl']['var_99'])
            std_pnl.append(day_data['pnl']['std'])
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Plot mean PnL
        ax.plot(days, mean_pnl, 'b-', linewidth=2, label='Mean PnL')
        
        # Plot confidence bands
        ax.fill_between(days, var_95_pnl, var_99_pnl, alpha=0.2, color='red', 
                       label='95%-99% VaR Band')
        
        # Plot standard deviation bands
        upper_std = np.array(mean_pnl) + np.array(std_pnl)
        lower_std = np.array(mean_pnl) - np.array(std_pnl)
        ax.fill_between(days, lower_std, upper_std, alpha=0.3, color='blue', 
                       label='±1 Std Dev')
        
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Days')
        ax.set_ylabel('PnL')
        ax.set_title(f'Portfolio PnL Evolution Over Time\n{self.scenario_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        saved_files = []
        if save:
            filename = f"pnl_evolution_{self.scenario_name.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"✅ Saved: {filepath}")
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return saved_files
    
    def plot_return_evolution(self, save: bool = False, save_dir: str = None, display: bool = True) -> List[str]:
        """Plot cumulative return evolution over time (aggregated across samples)"""
        # Set default save directory and create it if needed
        if save:
            if save_dir is None:
                save_dir = './portfolio_plots/'
            os.makedirs(save_dir, exist_ok=True)
        
        if not self.time_series_metrics:
            raise ValueError("No time-series data found. This test may be from an older version.")
        
        days = []
        mean_return = []
        var_95_return = []
        var_99_return = []
        std_return = []
        
        for day_key, day_data in self.time_series_metrics.items():
            days.append(day_data['day'])
            mean_return.append(day_data['returns']['mean'])
            var_95_return.append(day_data['returns']['var_95'])
            var_99_return.append(day_data['returns']['var_99'])
            std_return.append(day_data['returns']['std'])
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Plot mean return
        ax.plot(days, np.array(mean_return) * 100, 'g-', linewidth=2, label='Mean Return (%)')
        
        # Plot confidence bands
        ax.fill_between(days, np.array(var_95_return) * 100, np.array(var_99_return) * 100, 
                       alpha=0.2, color='red', label='95%-99% VaR Band')
        
        # Plot standard deviation bands
        upper_std = (np.array(mean_return) + np.array(std_return)) * 100
        lower_std = (np.array(mean_return) - np.array(std_return)) * 100
        ax.fill_between(days, lower_std, upper_std, alpha=0.3, color='green', 
                       label='±1 Std Dev')
        
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Days')
        ax.set_ylabel('Cumulative Return (%)')
        ax.set_title(f'Portfolio Cumulative Return Evolution\n{self.scenario_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        saved_files = []
        if save:
            filename = f"return_evolution_{self.scenario_name.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"✅ Saved: {filepath}")
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return saved_files
    
    def plot_var_evolution(self, save: bool = False, save_dir: str = None, display: bool = True) -> List[str]:
        """Plot VaR evolution over time (aggregated across samples)"""
        # Set default save directory and create it if needed
        if save:
            if save_dir is None:
                save_dir = './portfolio_plots/'
            os.makedirs(save_dir, exist_ok=True)
        
        if not self.time_series_metrics:
            raise ValueError("No time-series data found. This test may be from an older version.")
        
        days = []
        var_95_returns = []
        var_99_returns = []
        cvar_95_returns = []
        cvar_99_returns = []
        
        for day_key, day_data in self.time_series_metrics.items():
            days.append(day_data['day'])
            var_95_returns.append(day_data['returns']['var_95'])
            var_99_returns.append(day_data['returns']['var_99'])
            cvar_95_returns.append(day_data['returns']['cvar_95'])
            cvar_99_returns.append(day_data['returns']['cvar_99'])
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Plot VaR evolution
        ax.plot(days, np.array(var_95_returns) * 100, 'b-', linewidth=2, label='95% VaR (%)')
        ax.plot(days, np.array(var_99_returns) * 100, 'r-', linewidth=2, label='99% VaR (%)')
        
        # Plot CVaR evolution
        ax.plot(days, np.array(cvar_95_returns) * 100, 'b--', linewidth=2, label='95% CVaR (%)')
        ax.plot(days, np.array(cvar_99_returns) * 100, 'r--', linewidth=2, label='99% CVaR (%)')
        
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Days')
        ax.set_ylabel('Risk Level (%)')
        ax.set_title(f'Portfolio Risk Evolution (VaR/CVaR)\n{self.scenario_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        saved_files = []
        if save:
            filename = f"var_evolution_{self.scenario_name.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"✅ Saved: {filepath}")
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return saved_files
    
    # ============================================
    # METRIC-FOCUSED TIME-SERIES PLOTTING METHODS (SINGLE SAMPLE)
    # ============================================
    
    def plot_sample_pnl_evolution(self, sample_idx: int, save: bool = False, save_dir: str = None, display: bool = True) -> List[str]:
        """Plot PnL evolution for a specific sample"""
        # Set default save directory and create it if needed
        if save:
            if save_dir is None:
                save_dir = './portfolio_plots/'
            os.makedirs(save_dir, exist_ok=True)
        
        if sample_idx >= len(self.sample_results):
            raise ValueError(f"Sample index {sample_idx} out of range. Available: 0-{len(self.sample_results)-1}")
        
        sample = self.sample_results[sample_idx]
        daily_metrics = sample['daily_metrics']
        
        days = [dm['day'] for dm in daily_metrics]
        pnls = [dm['pnl'] for dm in daily_metrics]
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(days, pnls, 'b-', linewidth=2, label=f'Sample {sample_idx} PnL')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Days')
        ax.set_ylabel('PnL')
        ax.set_title(f'Portfolio PnL Evolution - Sample {sample_idx}\n{self.scenario_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        saved_files = []
        if save:
            filename = f"sample_{sample_idx}_pnl_evolution_{self.scenario_name.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"✅ Saved: {filepath}")
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return saved_files
    
    def plot_sample_return_evolution(self, sample_idx: int, save: bool = False, save_dir: str = None, display: bool = True) -> List[str]:
        """Plot cumulative return evolution for a specific sample"""
        # Set default save directory and create it if needed
        if save:
            if save_dir is None:
                save_dir = './portfolio_plots/'
            os.makedirs(save_dir, exist_ok=True)
        
        if sample_idx >= len(self.sample_results):
            raise ValueError(f"Sample index {sample_idx} out of range. Available: 0-{len(self.sample_results)-1}")
        
        sample = self.sample_results[sample_idx]
        daily_metrics = sample['daily_metrics']
        
        days = [dm['day'] for dm in daily_metrics]
        returns = [dm['cumulative_return'] for dm in daily_metrics]
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(days, np.array(returns) * 100, 'g-', linewidth=2, label=f'Sample {sample_idx} Return (%)')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Days')
        ax.set_ylabel('Cumulative Return (%)')
        ax.set_title(f'Portfolio Return Evolution - Sample {sample_idx}\n{self.scenario_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        saved_files = []
        if save:
            filename = f"sample_{sample_idx}_return_evolution_{self.scenario_name.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"✅ Saved: {filepath}")
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return saved_files
    
    def plot_sample_portfolio_value_evolution(self, sample_idx: int, save: bool = False, save_dir: str = None, display: bool = True) -> List[str]:
        """Plot portfolio value evolution for a specific sample"""
        # Set default save directory and create it if needed
        if save:
            if save_dir is None:
                save_dir = './portfolio_plots/'
            os.makedirs(save_dir, exist_ok=True)
        
        if sample_idx >= len(self.sample_results):
            raise ValueError(f"Sample index {sample_idx} out of range. Available: 0-{len(self.sample_results)-1}")
        
        sample = self.sample_results[sample_idx]
        daily_metrics = sample['daily_metrics']
        
        days = [dm['day'] for dm in daily_metrics]
        portfolio_values = [dm['portfolio_value'] for dm in daily_metrics]
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(days, portfolio_values, 'purple', linewidth=2, label=f'Sample {sample_idx} Portfolio Value')
        
        ax.set_xlabel('Days')
        ax.set_ylabel('Portfolio Value')
        ax.set_title(f'Portfolio Value Evolution - Sample {sample_idx}\n{self.scenario_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        saved_files = []
        if save:
            filename = f"sample_{sample_idx}_portfolio_value_evolution_{self.scenario_name.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"✅ Saved: {filepath}")
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return saved_files
    
    def plot_sample_daily_returns(self, sample_idx: int, save: bool = False, save_dir: str = None, display: bool = True) -> List[str]:
        """Plot daily returns for a specific sample"""
        # Set default save directory and create it if needed
        if save:
            if save_dir is None:
                save_dir = './portfolio_plots/'
            os.makedirs(save_dir, exist_ok=True)
        
        if sample_idx >= len(self.sample_results):
            raise ValueError(f"Sample index {sample_idx} out of range. Available: 0-{len(self.sample_results)-1}")
        
        sample = self.sample_results[sample_idx]
        daily_returns = np.array(sample['daily_returns'])
        daily_days = np.arange(1, len(daily_returns) + 1)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(daily_days, daily_returns * 100, 'r-', linewidth=1, alpha=0.7, label=f'Sample {sample_idx} Daily Returns (%)')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Days')
        ax.set_ylabel('Daily Return (%)')
        ax.set_title(f'Portfolio Daily Returns - Sample {sample_idx}\n{self.scenario_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        saved_files = []
        if save:
            filename = f"sample_{sample_idx}_daily_returns_{self.scenario_name.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"✅ Saved: {filepath}")
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return saved_files
    
    # ============================================
    # MISSING EVOLUTION PLOTS (Running + Aggregated)
    # ============================================
    
    def plot_cvar_evolution(self, save: bool = False, save_dir: str = None, display: bool = True) -> List[str]:
        """Plot CVaR evolution over time (aggregated across samples)"""
        if not self.time_series_metrics:
            raise ValueError("No time series metrics available")
        
        days = []
        cvar_95_values = []
        cvar_99_values = []
        
        for day_key, metrics in self.time_series_metrics.items():
            if day_key.startswith('day_'):
                day = int(day_key.split('_')[1])
                days.append(day)
                cvar_95_values.append(metrics['returns']['cvar_95'])
                cvar_99_values.append(metrics['returns']['cvar_99'])
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(days, np.array(cvar_95_values) * 100, 'b-', linewidth=2, label='CVaR 95%')
        ax.plot(days, np.array(cvar_99_values) * 100, 'r-', linewidth=2, label='CVaR 99%')
        ax.set_title(f'CVaR Evolution Over Time\n{self.scenario_name}')
        ax.set_xlabel('Day')
        ax.set_ylabel('CVaR (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        saved_files = []
        if save:
            if save_dir is None:
                save_dir = './portfolio_plots/'
            os.makedirs(save_dir, exist_ok=True)
            
            filename = f"cvar_evolution_{self.scenario_name.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"✅ Saved: {filepath}")
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return saved_files
    
    def plot_portfolio_value_evolution(self, save: bool = False, save_dir: str = None, display: bool = True) -> List[str]:
        """Plot portfolio value evolution over time (aggregated across samples)"""
        if not self.time_series_metrics:
            raise ValueError("No time series metrics available")
        
        days = []
        mean_values = []
        std_values = []
        
        for day_key, metrics in self.time_series_metrics.items():
            if day_key.startswith('day_'):
                day = int(day_key.split('_')[1])
                days.append(day)
                mean_values.append(metrics['pnl']['mean'])
                std_values.append(metrics['pnl']['std'])
        
        mean_values = np.array(mean_values)
        std_values = np.array(std_values)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(days, mean_values, 'b-', linewidth=2, label='Mean Portfolio Value')
        ax.fill_between(days, mean_values - std_values, mean_values + std_values, 
                       alpha=0.3, color='blue', label='±1 Std')
        ax.set_title(f'Portfolio Value Evolution Over Time\n{self.scenario_name}')
        ax.set_xlabel('Day')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        saved_files = []
        if save:
            if save_dir is None:
                save_dir = './portfolio_plots/'
            os.makedirs(save_dir, exist_ok=True)
            
            filename = f"portfolio_value_evolution_{self.scenario_name.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"✅ Saved: {filepath}")
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return saved_files
    
    # ============================================
    # MISSING SINGLE PATH EVOLUTION PLOTS
    # ============================================
    
    def plot_sample_var_evolution(self, sample_idx: int, save: bool = False, save_dir: str = None, display: bool = True) -> List[str]:
        """Plot VaR evolution for a specific sample"""
        if sample_idx >= len(self.sample_results):
            raise ValueError(f"Sample index {sample_idx} out of range. Available: 0-{len(self.sample_results)-1}")
        
        sample = self.sample_results[sample_idx]
        daily_metrics = sample['daily_metrics']
        
        days = [m['day'] for m in daily_metrics]
        returns = [m['cumulative_return'] for m in daily_metrics]
        
        # Compute rolling VaR (simplified - using historical returns)
        var_95_values = []
        var_99_values = []
        
        for i in range(len(returns)):
            if i < 10:  # Need at least 10 observations
                var_95_values.append(0)
                var_99_values.append(0)
            else:
                historical_returns = returns[:i+1]
                var_95_values.append(np.percentile(historical_returns, 5))
                var_99_values.append(np.percentile(historical_returns, 1))
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(days, np.array(var_95_values) * 100, 'b-', linewidth=2, label='VaR 95%')
        ax.plot(days, np.array(var_99_values) * 100, 'r-', linewidth=2, label='VaR 99%')
        ax.plot(days, np.array(returns) * 100, 'g-', linewidth=1, alpha=0.7, label='Cumulative Return')
        ax.set_title(f'VaR Evolution - Sample {sample_idx}\n{self.scenario_name}')
        ax.set_xlabel('Day')
        ax.set_ylabel('VaR / Return (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        saved_files = []
        if save:
            if save_dir is None:
                save_dir = './portfolio_plots/'
            os.makedirs(save_dir, exist_ok=True)
            
            filename = f"var_evolution_sample_{sample_idx}_{self.scenario_name.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"✅ Saved: {filepath}")
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return saved_files
    
    def plot_sample_cvar_evolution(self, sample_idx: int, save: bool = False, save_dir: str = None, display: bool = True) -> List[str]:
        """Plot CVaR evolution for a specific sample"""
        if sample_idx >= len(self.sample_results):
            raise ValueError(f"Sample index {sample_idx} out of range. Available: 0-{len(self.sample_results)-1}")
        
        sample = self.sample_results[sample_idx]
        daily_metrics = sample['daily_metrics']
        
        days = [m['day'] for m in daily_metrics]
        returns = [m['cumulative_return'] for m in daily_metrics]
        
        # Compute rolling CVaR
        cvar_95_values = []
        cvar_99_values = []
        
        for i in range(len(returns)):
            if i < 10:  # Need at least 10 observations
                cvar_95_values.append(0)
                cvar_99_values.append(0)
            else:
                historical_returns = np.array(returns[:i+1])
                var_95 = np.percentile(historical_returns, 5)
                var_99 = np.percentile(historical_returns, 1)
                
                cvar_95 = np.mean(historical_returns[historical_returns <= var_95]) if np.any(historical_returns <= var_95) else var_95
                cvar_99 = np.mean(historical_returns[historical_returns <= var_99]) if np.any(historical_returns <= var_99) else var_99
                
                cvar_95_values.append(cvar_95)
                cvar_99_values.append(cvar_99)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(days, np.array(cvar_95_values) * 100, 'b-', linewidth=2, label='CVaR 95%')
        ax.plot(days, np.array(cvar_99_values) * 100, 'r-', linewidth=2, label='CVaR 99%')
        ax.plot(days, np.array(returns) * 100, 'g-', linewidth=1, alpha=0.7, label='Cumulative Return')
        ax.set_title(f'CVaR Evolution - Sample {sample_idx}\n{self.scenario_name}')
        ax.set_xlabel('Day')
        ax.set_ylabel('CVaR / Return (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        saved_files = []
        if save:
            if save_dir is None:
                save_dir = './portfolio_plots/'
            os.makedirs(save_dir, exist_ok=True)
            
            filename = f"cvar_evolution_sample_{sample_idx}_{self.scenario_name.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"✅ Saved: {filepath}")
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return saved_files
    
    def __repr__(self) -> str:
        return f"Test(id='{self.id}', scenario='{self.scenario_name}', date='{self.test_date}')"
