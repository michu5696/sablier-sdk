"""
Visualization utilities for Sablier SDK.

Provides plotting functions for reconstructed samples, forecasts, and model diagnostics.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _check_matplotlib():
    """Check if matplotlib is available"""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install sablier-sdk[viz]"
        )


class TimeSeriesPlotter:
    """Utility class for creating time series visualizations"""
    
    @staticmethod
    def plot_reconstruction(
        dates: np.ndarray,
        original_values: np.ndarray,
        reconstructed_values: np.ndarray,
        feature_name: str,
        past_length: int,
        title: str = None,
        ax = None
    ):
        """
        Plot original vs reconstructed values for a single feature.
        
        Args:
            dates: Array of dates
            original_values: Original values
            reconstructed_values: Reconstructed values
            feature_name: Name of the feature
            past_length: Length of past window (for vertical line)
            title: Optional custom title
            ax: Optional matplotlib axis
        """
        _check_matplotlib()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 5))
        
        # Plot data
        ax.plot(dates, original_values, 'o-', label='Original', alpha=0.7, linewidth=2)
        ax.plot(dates, reconstructed_values, 's--', label='Reconstructed', alpha=0.7, linewidth=2)
        
        # Add vertical line at past/future boundary
        if past_length < len(dates):
            boundary_date = dates[past_length - 1]
            ax.axvline(x=boundary_date, color='red', linestyle=':', linewidth=2, alpha=0.5, label='Past/Future Boundary')
        
        # Formatting
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel(feature_name, fontsize=11)
        ax.set_title(title or f'Reconstruction: {feature_name}', fontsize=13, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        return ax
    
    @staticmethod
    def plot_forecasts(
        past_dates: np.ndarray,
        past_values: np.ndarray,
        future_dates: np.ndarray,
        forecast_paths: List[np.ndarray],
        feature_name: str,
        n_paths: int = 10,
        show_ci: bool = True,
        ci_levels: List[float] = [0.68, 0.95],
        title: str = None,
        ax = None
    ):
        """
        Plot forecast paths with optional confidence intervals.
        
        Args:
            past_dates: Array of past dates
            past_values: Array of past values
            future_dates: Array of future dates
            forecast_paths: List of forecast arrays (each is future values)
            feature_name: Name of the feature
            n_paths: Number of individual paths to show
            show_ci: Whether to show confidence interval bands
            ci_levels: Confidence interval levels (e.g., [0.68, 0.95] for 1σ and 2σ)
            title: Optional custom title
            ax: Optional matplotlib axis
        """
        _check_matplotlib()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot past data
        ax.plot(past_dates, past_values, 'o-', color='black', 
                label='Historical', linewidth=2, markersize=4, alpha=0.8)
        
        # Plot individual forecast paths
        n_to_plot = min(n_paths, len(forecast_paths))
        for i in range(n_to_plot):
            ax.plot(future_dates, forecast_paths[i], '-', 
                   alpha=0.3, linewidth=1, color='steelblue')
        
        # Add a single legend entry for forecast paths
        ax.plot([], [], '-', alpha=0.5, linewidth=1.5, color='steelblue', 
               label=f'Forecast Paths (n={len(forecast_paths)})')
        
        # Add confidence intervals
        if show_ci and len(forecast_paths) > 1:
            forecast_array = np.array(forecast_paths)  # Shape: (n_samples, time_steps)
            
            colors = ['#4A90E2', '#7CB342']  # Blue and green tones
            for idx, ci_level in enumerate(sorted(ci_levels, reverse=True)):
                lower_q = (1 - ci_level) / 2
                upper_q = 1 - lower_q
                
                lower = np.percentile(forecast_array, lower_q * 100, axis=0)
                upper = np.percentile(forecast_array, upper_q * 100, axis=0)
                
                ax.fill_between(future_dates, lower, upper, 
                               alpha=0.2, color=colors[idx % len(colors)],
                               label=f'{int(ci_level*100)}% CI')
        
        # Add median forecast
        if len(forecast_paths) > 1:
            median_forecast = np.median(np.array(forecast_paths), axis=0)
            ax.plot(future_dates, median_forecast, '-', color='darkred', 
                   linewidth=2.5, label='Median Forecast', alpha=0.9)
        
        # Add vertical line at past/future boundary
        if len(past_dates) > 0 and len(future_dates) > 0:
            boundary_date = past_dates[-1]
            ax.axvline(x=boundary_date, color='red', linestyle=':', 
                      linewidth=2, alpha=0.5, label='Forecast Start')
        
        # Formatting
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel(feature_name, fontsize=11)
        ax.set_title(title or f'Forecast: {feature_name}', fontsize=13, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9, fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        return ax
    
    @staticmethod
    def plot_feature_importance(
        feature_names: List[str],
        importance_values: List[float],
        title: str = "Feature Importance (SHAP Values)",
        top_n: int = None,
        ax = None
    ):
        """
        Plot feature importance as a horizontal bar chart.
        
        Args:
            feature_names: List of feature names
            importance_values: List of importance values (SHAP values)
            title: Plot title
            top_n: Show only top N features (None = show all)
            ax: Optional matplotlib axis
        """
        _check_matplotlib()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.3)))
        
        # Sort by importance
        sorted_indices = np.argsort(importance_values)[::-1]
        
        # Limit to top_n if specified
        if top_n is not None and top_n < len(sorted_indices):
            sorted_indices = sorted_indices[:top_n]
        
        sorted_names = [feature_names[i] for i in sorted_indices]
        sorted_values = [importance_values[i] for i in sorted_indices]
        
        # Create horizontal bar chart
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.8, len(sorted_values)))
        bars = ax.barh(range(len(sorted_values)), sorted_values, color=colors, alpha=0.8)
        
        # Formatting
        ax.set_yticks(range(len(sorted_values)))
        ax.set_yticklabels(sorted_names, fontsize=10)
        ax.set_xlabel('Mean Absolute SHAP Value', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, sorted_values)):
            ax.text(value, i, f' {value:.4f}', va='center', fontsize=9)
        
        # Invert y-axis to show highest at top
        ax.invert_yaxis()
        
        return ax


def create_date_array(start_date: str, length: int, freq: str = 'D') -> np.ndarray:
    """
    Create an array of dates.
    
    Args:
        start_date: Start date as string (YYYY-MM-DD)
        length: Number of dates
        freq: Frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
    
    Returns:
        Array of datetime objects
    """
    start = datetime.strptime(start_date, '%Y-%m-%d')
    
    if freq == 'D':
        delta = timedelta(days=1)
    elif freq == 'W':
        delta = timedelta(weeks=1)
    elif freq == 'M':
        delta = timedelta(days=30)  # Approximation
    else:
        raise ValueError(f"Unsupported frequency: {freq}")
    
    dates = [start + i * delta for i in range(length)]
    return np.array(dates)

