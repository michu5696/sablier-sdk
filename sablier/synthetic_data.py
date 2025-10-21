"""
SyntheticData class for forecast results

Contains:
- Conditioning data (past + future conditioning)
- Forecast paths (generated predictions)
- Ground truth (actual outcomes, if available)
"""

import logging
from typing import Optional, List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class SyntheticData:
    """
    Container for forecast results with conditioning data and ground truth
    
    Structure:
    - conditioning_past: Past observations {feature: values}
    - conditioning_future: Future conditioning {feature: values}
    - forecasts: Generated predictions {feature: [path1, path2, ...]}
    - ground_truth_target: Actual outcomes {feature: values} (optional)
    - metadata: Model info, dates, etc.
    
    Provides:
    - Easy access to all forecast components
    - Plotting methods (to be implemented)
    - pandas DataFrame export (to be implemented)
    """
    
    def __init__(self, 
                 reconstructed_windows: List[Dict[str, Any]],
                 forecast_metadata: Dict[str, Any] = None,
                 scenario=None, 
                 model=None):
        """
        Initialize SyntheticData from reconstructed windows
        
        Args:
            reconstructed_windows: List of reconstructed window dicts from backend
                Structure: [{'feature': 'US Treasury 10Y', 'temporal_tag': 'past', 
                             'reconstructed_values': [...], '_sample_idx': 0}]
            forecast_metadata: Metadata from forecast response
            scenario: Associated Scenario instance
            model: Associated Model instance
        """
        self.scenario = scenario
        self.model = model
        self.metadata = forecast_metadata or {}
        
        # Date information for plotting
        self.past_dates = self.metadata.get('past_dates', [])
        self.future_dates = self.metadata.get('future_dates', [])
        self.reference_date = self.metadata.get('reference_date', None)
        
        print(f"[DEBUG] Date info: past={len(self.past_dates)}, future={len(self.future_dates)}, ref={self.reference_date}")
        
        # Organize windows by type
        self.conditioning_past = {}      # {feature: values}
        self.conditioning_future = {}    # {feature: values}
        self.ground_truth_target = {}    # {feature: values}
        self.forecasts = {}              # {feature: [path1, path2, ...]}
        
        # Parse reconstructed windows
        # Windows with sample_idx=0 are conditioning/ground truth
        # Windows with sample_idx>0 are forecast paths
        
        # Debug: Count window types
        past_count = 0
        future_cond_count = 0
        future_gt_count = 0
        forecast_count = 0
        
        for window in reconstructed_windows:
            feature = window.get('feature')
            temporal_tag = window.get('temporal_tag')
            values = window.get('reconstructed_values', [])
            sample_idx = window.get('_sample_idx', 0)
            is_historical = window.get('_is_historical_pattern', False)
            data_type = window.get('data_type', '')
            
            if sample_idx == 0:
                # Ground truth / conditioning (sample_idx 0)
                if temporal_tag == 'past':
                    # Past conditioning: what we observed (today's data or sample's past)
                    self.conditioning_past[feature] = values
                    past_count += 1
                elif temporal_tag == 'future':
                    # Future windows with sample_idx=0 could be:
                    # 1. Ground truth target (for validation forecasts)
                    # 2. Future conditioning (observed but used for conditioning)
                    # 3. Historical pattern (for scenario reference)
                    
                    if is_historical or data_type == 'ground_truth':
                        # For historical patterns, check if it's a target feature
                        # Target features go to ground_truth_target, conditioning features go to conditioning_future
                        if hasattr(self.model, '_data') and self.model._data.get('target_features'):
                            # Handle both dict and string formats
                            target_features_raw = self.model._data['target_features']
                            if target_features_raw and isinstance(target_features_raw[0], dict):
                                target_features = [f.get('name') for f in target_features_raw]
                            else:
                                target_features = target_features_raw  # Already strings
                            
                            if feature in target_features:
                                # Historical target: actual outcome (validation) or historical pattern (scenario)
                                self.ground_truth_target[feature] = values
                                future_gt_count += 1
                                logger.debug(f"Added to ground_truth_target: {feature} (target feature)")
                            else:
                                # Historical conditioning: used to condition the forecast
                                self.conditioning_future[feature] = values
                                future_cond_count += 1
                                logger.debug(f"Added to conditioning_future: {feature} (conditioning feature)")
                        else:
                            # Fallback: treat as ground truth if we can't determine target features
                            self.ground_truth_target[feature] = values
                            future_gt_count += 1
                            logger.debug(f"Added to ground_truth_target: {feature} (fallback)")
                    else:
                        # Future conditioning: observed and used to condition the forecast
                        self.conditioning_future[feature] = values
                        future_cond_count += 1
            else:
                # Forecasts (sample_idx >= 1): generated predictions
                if feature not in self.forecasts:
                    self.forecasts[feature] = []
                self.forecasts[feature].append(values)
                forecast_count += 1
        
        self.n_paths = max(len(paths) for paths in self.forecasts.values()) if self.forecasts else 0
        self.features = list(set(list(self.conditioning_past.keys()) + 
                                 list(self.conditioning_future.keys()) + 
                                 list(self.forecasts.keys())))
        
        # Debug logging
        print(f"[DEBUG] Parsed {len(reconstructed_windows)} windows:")
        print(f"  Past: {past_count}, Future Cond: {future_cond_count}, Future GT: {future_gt_count}, Forecasts: {forecast_count}")
        
        print(f"✅ Forecast data loaded:")
        print(f"   Features: {len(self.features)}")
        print(f"   Forecast paths: {self.n_paths}")
        print(f"   Past conditioning: {len(self.conditioning_past)} features")
        print(f"   Future conditioning: {len(self.conditioning_future)} features")
        if self.ground_truth_target:
            print(f"   Ground truth: {len(self.ground_truth_target)} features ({list(self.ground_truth_target.keys())})")
        else:
            print(f"   Ground truth: None")
    
    def to_dataframe(self) -> 'pd.DataFrame':
        """
        Export to pandas DataFrame (to be implemented)
        
        Returns DataFrame with columns: date, path_idx, feature1, feature2, ...
        """
        if not HAS_PANDAS:
            raise ImportError("pandas required. Install with: pip install pandas")
        
        # TODO: Implement DataFrame export from new format
        raise NotImplementedError("to_dataframe() will be implemented for plotting needs")
    
    def get_quantiles(self, quantiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95]) -> Dict[str, Any]:
        """
        Get forecast quantiles for each feature
        
        Args:
            quantiles: List of quantile levels (default: [0.05, 0.25, 0.5, 0.75, 0.95])
        
        Returns:
            Dict: {feature: {quantile: values}}
        """
        result = {}
        for feature, paths in self.forecasts.items():
            paths_array = np.array(paths)  # Shape: (n_paths, n_timesteps)
            result[feature] = {}
            for q in quantiles:
                result[feature][q] = np.percentile(paths_array, q * 100, axis=0).tolist()
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for forecasts
        
        Returns:
            Dict with mean, std, min, max for each feature
        """
        stats = {}
        for feature, paths in self.forecasts.items():
            paths_array = np.array(paths)
            stats[feature] = {
                'mean': np.mean(paths_array, axis=0).tolist(),
                'std': np.std(paths_array, axis=0).tolist(),
                'min': np.min(paths_array, axis=0).tolist(),
                'max': np.max(paths_array, axis=0).tolist(),
                'median': np.median(paths_array, axis=0).tolist()
            }
        return stats
    
    def plot_forecasts(self, features: Optional[List[str]] = None, save_path: Optional[str] = None):
        """
        Plot forecast paths with conditioning and ground truth
        
        Shows:
        - Past trajectories (historical data)
        - Future ground truth (if available)
        - Confidence intervals (68% and 95%)
        - Limited number of individual forecast paths
        - Median forecast
        
        Args:
            features: List of features to plot (default: all forecast features)
            save_path: Path to save plot (default: display only)
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required. Install with: pip install matplotlib")
        
        # Select features to plot
        if features is None:
            features = list(self.forecasts.keys())
        
        # Filter to only features that have forecasts
        features = [f for f in features if f in self.forecasts]
        
        if not features:
            raise ValueError("No forecast features available to plot")
        
        n_features = len(features)
        fig, axes = plt.subplots(n_features, 1, figsize=(16, 5*n_features))
        if n_features == 1:
            axes = [axes]
        
        # Main title
        fig.suptitle('Conditional Forecasts', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        for idx, feature_name in enumerate(features):
            ax = axes[idx]
            
            # Get forecast data
            forecast_paths = self.forecasts[feature_name]
            forecasts_array = np.array(forecast_paths)
            n_samples, n_timesteps = forecasts_array.shape
            
            # Get ground truth data
            past_values = self.conditioning_past.get(feature_name)
            future_gt_values = self.ground_truth_target.get(feature_name)
            
            # Setup time axis with dates (if available)
            if self.past_dates and self.future_dates:
                # Use actual dates from forecast response
                past_t = self.past_dates
                future_t = self.future_dates
                use_dates = True
                print(f"[DEBUG] Using dates for {feature_name}: {len(past_t)} past + {len(future_t)} future")
            elif past_values:
                # Fallback to numeric indices if dates not available
                past_t = np.arange(len(past_values))
                future_t = np.arange(len(past_values), len(past_values) + n_timesteps)
                use_dates = False
            else:
                # No past data, just use future indices
                past_t = []
                future_t = np.arange(n_timesteps)
                use_dates = False
            
            if past_values and len(past_t) > 0:
                
                # Plot ground truth past (black line with markers)
                ax.plot(past_t, past_values, 'o-', color='black', linewidth=2, 
                       markersize=4, alpha=0.8, label='Historical', zorder=5)
                
                # Plot ground truth future (green line with markers)
                if future_gt_values:
                    ax.plot(future_t, future_gt_values, 'o-', color='green', linewidth=2.5, 
                           markersize=5, alpha=0.9, label='Ground Truth', zorder=6)
                
                # Vertical line at forecast start (red dotted)
                ax.axvline(x=len(past_values), color='red', linestyle=':', 
                          linewidth=2, alpha=0.5, label='Forecast Start', zorder=4)
            else:
                future_t = np.arange(n_timesteps)
            
            # Plot individual forecast paths (light blue, semi-transparent)
            n_to_plot = min(50, n_samples)  # Show up to 50 paths
            for i in range(n_to_plot):
                ax.plot(future_t, forecasts_array[i], '-', alpha=0.2,
                       linewidth=0.8, color='steelblue', zorder=2)
            
            # Add legend entry for forecast paths
            ax.plot([], [], '-', alpha=0.5, linewidth=1.5, color='steelblue',
                   label=f'Forecast Paths (n={n_samples})', zorder=2)
            
            # Add confidence intervals (68% and 95%)
            ci_levels = [0.68, 0.95]
            ci_colors = ['darkblue', 'steelblue']
            ci_alphas = [0.2, 0.15]
            
            for ci_idx, ci_level in enumerate(ci_levels):
                lower_q = (1 - ci_level) / 2
                upper_q = 1 - lower_q
                
                lower = np.percentile(forecasts_array, lower_q * 100, axis=0)
                upper = np.percentile(forecasts_array, upper_q * 100, axis=0)
                
                ax.fill_between(future_t, lower, upper, alpha=ci_alphas[ci_idx], 
                               color=ci_colors[ci_idx],
                               label=f'{int(ci_level*100)}% CI', zorder=3)
            
            # Plot median forecast (dark red line)
            median_forecast = np.median(forecasts_array, axis=0)
            ax.plot(future_t, median_forecast, '-', color='darkred', 
                   linewidth=2.5, alpha=0.9, label='Median Forecast', zorder=7)
            
            # Formatting
            ax.set_title(feature_name, fontsize=14, fontweight='bold')
            
            if use_dates:
                # Format x-axis for dates
                ax.set_xlabel('Date', fontsize=12)
                # Rotate labels and show every Nth date to avoid crowding
                n_dates = len(past_t) + len(future_t)
                tick_interval = max(1, n_dates // 10)  # Show ~10 ticks
                all_dates = list(past_t) + list(future_t)
                tick_indices = range(0, n_dates, tick_interval)
                ax.set_xticks([all_dates[i] for i in tick_indices if i < len(all_dates)])
                ax.tick_params(axis='x', rotation=45)
            else:
                # Numeric time steps
                ax.set_xlabel('Time Step', fontsize=12)
            
            ax.set_ylabel(f'{feature_name} Value', fontsize=12)
            ax.legend(loc='best', fontsize=10, framealpha=0.95)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add statistics text box
            stats_text = f'Min: {np.min(forecasts_array):.3f}%\n'
            stats_text += f'Max: {np.max(forecasts_array):.3f}%\n'
            stats_text += f'Median: {np.median(median_forecast):.3f}%'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Plot saved to: {save_path}")
        else:
            plt.show()
        
        return fig
    
    def plot_conditioning(self, features: Optional[List[str]] = None, save_path: Optional[str] = None):
        """
        Plot conditioning data (past and future conditioning windows)
        
        Shows:
        - Past conditioning (fetched recent data)
        - Future conditioning (from selected historical sample)
        - Boundary line separating past from future
        
        Args:
            features: List of features to plot (default: all conditioning features)
            save_path: Path to save plot (default: display only)
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required. Install with: pip install matplotlib")
        
        # Get only features that have future conditioning (not target features)
        # Target features should only appear in forecast plots, not conditioning plots
        future_conditioning_features = set(self.conditioning_future.keys())
        
        # Select features to plot
        if features is None:
            features = list(future_conditioning_features)
        
        # Filter to only features that have future conditioning data
        features = [f for f in features if f in future_conditioning_features]
        
        if not features:
            raise ValueError("No conditioning features available to plot")
        
        # Calculate grid layout (2x3 for 6 features, adjust as needed)
        n_features = len(features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten() if n_features > 1 else axes
        
        # Main title
        fig.suptitle('Conditioning Scenario', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        for idx, feature_name in enumerate(features):
            ax = axes_flat[idx]
            
            # Get conditioning data
            past_values = self.conditioning_past.get(feature_name, [])
            future_values = self.conditioning_future.get(feature_name, [])
            
            if not past_values and not future_values:
                continue
            
            # Create time axis with dates (if available)
            if self.past_dates and self.future_dates:
                past_t = self.past_dates if past_values else []
                future_t = self.future_dates if future_values else []
                use_dates = True
            else:
                past_t = np.arange(len(past_values)) if past_values else []
                future_t = np.arange(len(past_values), len(past_values) + len(future_values)) if future_values else []
                use_dates = False
            
            # Plot past conditioning (blue line)
            if past_values:
                ax.plot(past_t, past_values, '-', color='blue', linewidth=2, 
                       alpha=0.8, label='Past (Fetched)', zorder=3)
            
            # Plot future conditioning (orange line)
            if future_values:
                ax.plot(future_t, future_values, '-', color='orange', linewidth=2, 
                       alpha=0.8, label='Future (Conditioning)', zorder=3)
            
            # Add boundary line (dashed vertical line at reference date)
            if past_values and future_values:
                if use_dates and self.reference_date:
                    # Use actual reference date for boundary
                    boundary_x = self.reference_date
                else:
                    # Use numeric index
                    boundary_x = len(past_values)
                ax.axvline(x=boundary_x, color='red', linestyle='--', 
                          linewidth=2, alpha=0.7, label='Boundary', zorder=4)
            
            # Formatting
            ax.set_title(feature_name, fontsize=12, fontweight='bold')
            
            if use_dates:
                # Format x-axis for dates
                ax.set_xlabel('Date', fontsize=10)
                # Rotate labels and show every Nth date
                n_dates = len(past_t) + len(future_t)
                tick_interval = max(1, n_dates // 8)  # Show ~8 ticks
                all_dates = list(past_t) + list(future_t)
                tick_indices = range(0, n_dates, tick_interval)
                ax.set_xticks([all_dates[i] for i in tick_indices if i < len(all_dates)])
                ax.tick_params(axis='x', rotation=45, labelsize=8)
            else:
                # Numeric time steps
                ax.set_xlabel('Time Step', fontsize=10)
            
            ax.set_ylabel('Value', fontsize=10)
            ax.legend(loc='best', fontsize=9, framealpha=0.95)
            ax.grid(True, alpha=0.3, linestyle='--')
        
        # Hide unused subplots
        for idx in range(n_features, len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Conditioning plot saved to: {save_path}")
        else:
            plt.show()
        
        return fig
    
    def __repr__(self):
        """String representation"""
        return (f"SyntheticData(n_paths={self.n_paths}, features={len(self.features)}, "
                f"conditioning_past={len(self.conditioning_past)}, "
                f"conditioning_future={len(self.conditioning_future)}, "
                f"ground_truth={len(self.ground_truth_target)})")


# Legacy methods removed - all based on old nested format:
# - Old _to_dataframe, _extract_features, _extract_dates
# - Old plot_paths, validate_against_real_data methods
# - Will be re-implemented as needed for new format
