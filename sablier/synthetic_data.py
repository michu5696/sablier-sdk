"""SyntheticData class for managing and analyzing generated synthetic market paths"""

import logging
from typing import Optional, List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class SyntheticData:
    """
    Container for generated synthetic market paths with analysis capabilities
    
    Provides:
    - pandas DataFrame access to paths
    - Visualization methods
    - Statistical analysis
    - Export capabilities
    """
    
    def __init__(self, paths: List[Dict], scenario=None, model=None):
        """
        Initialize SyntheticData
        
        Args:
            paths: List of reconstructed path dictionaries
                   Each path: {feature_name: {'future': {'dates': [...], 'values': [...]}}}
            scenario: Associated Scenario instance
            model: Associated Model instance
        """
        if not HAS_PANDAS:
            raise ImportError(
                "pandas is required for SyntheticData. "
                "Install it with: pip install pandas"
            )
        
        self.scenario = scenario
        self.model = model
        self.n_paths = len(paths)
        self._raw_paths = paths
        
        # Convert to DataFrame
        self.paths = self._to_dataframe(paths)
        
        # Extract metadata
        self.features = self._extract_features(paths)
        self.dates = self._extract_dates(paths)
        
        print(f"[SyntheticData] Created dataset:")
        print(f"  Paths: {self.n_paths}")
        print(f"  Features: {len(self.features)}")
        print(f"  Time steps: {len(self.dates)}")
        print(f"  Total data points: {len(self.paths)}")
    
    def _to_dataframe(self, paths: List[Dict]) -> pd.DataFrame:
        """Convert paths to pandas DataFrame (long format)"""
        
        # Structure: Each row is (date, path_idx, feature1, feature2, ...)
        rows = []
        
        for path_idx, path in enumerate(paths):
            # Get dates from first feature
            dates = None
            for feature_data in path.values():
                if 'future' in feature_data and feature_data['future'].get('dates'):
                    dates = feature_data['future']['dates']
                    break
            
            if dates is None:
                logger.warning(f"No dates found in path {path_idx}, skipping")
                continue
            
            # Create a row for each timestep
            for t, date in enumerate(dates):
                row = {'date': date, 'path_idx': path_idx}
                
                # Add feature values
                for feature_name, feature_data in path.items():
                    if 'future' in feature_data:
                        values = feature_data['future'].get('values', [])
                        if t < len(values):
                            row[feature_name] = values[t]
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Convert date to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def _extract_features(self, paths: List[Dict]) -> List[str]:
        """Extract feature names from paths"""
        if paths:
            return list(paths[0].keys())
        return []
    
    def _extract_dates(self, paths: List[Dict]) -> List[str]:
        """Extract dates from paths"""
        if paths and paths[0]:
            first_feature = list(paths[0].values())[0]
            if 'future' in first_feature:
                return first_feature['future'].get('dates', [])
        return []
    
    # ============================================
    # DATA ACCESS METHODS
    # ============================================
    
    def get_path(self, path_idx: int) -> pd.DataFrame:
        """
        Get a single path as DataFrame
        
        Args:
            path_idx: Path index (0 to n_paths-1)
        
        Returns:
            DataFrame with columns [date, feature1, feature2, ...]
        """
        return self.paths[self.paths['path_idx'] == path_idx].drop(columns=['path_idx'])
    
    def get_feature(self, feature_name: str) -> pd.DataFrame:
        """
        Get all paths for a specific feature
        
        Args:
            feature_name: Feature name
        
        Returns:
            DataFrame with columns [date, path_idx, feature_name]
        """
        if feature_name not in self.features:
            raise ValueError(f"Feature {feature_name} not found. Available: {self.features}")
        
        return self.paths[['date', 'path_idx', feature_name]]
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics of the synthetic data"""
        summary = {
            'n_paths': self.n_paths,
            'n_features': len(self.features),
            'n_timesteps': len(self.dates),
            'features': self.features,
            'date_range': {
                'start': self.dates[0] if self.dates else None,
                'end': self.dates[-1] if self.dates else None
            },
            'statistics': {}
        }
        
        # Compute statistics per feature
        for feature in self.features:
            feature_data = self.paths[feature].dropna()
            if len(feature_data) > 0:
                summary['statistics'][feature] = {
                    'mean': float(feature_data.mean()),
                    'std': float(feature_data.std()),
                    'min': float(feature_data.min()),
                    'max': float(feature_data.max()),
                    'median': float(feature_data.median())
                }
        
        return summary
    
    # ============================================
    # EXPORT METHODS
    # ============================================
    
    def to_csv(self, path: str, **kwargs):
        """
        Export to CSV
        
        Args:
            path: File path
            **kwargs: Additional arguments to pandas.to_csv()
        """
        self.paths.to_csv(path, index=False, **kwargs)
        print(f"💾 Exported to {path}")
    
    def to_json(self, path: str, **kwargs):
        """
        Export to JSON
        
        Args:
            path: File path
            **kwargs: Additional arguments to pandas.to_json()
        """
        self.paths.to_json(path, orient='records', date_format='iso', **kwargs)
        print(f"💾 Exported to {path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'dates': self.dates,
            'features': self.features,
            'n_paths': self.n_paths,
            'paths': self._raw_paths,
            'summary': self.summary()
        }
    
    # ============================================
    # VISUALIZATION METHODS
    # ============================================
    
    def plot_paths(
        self,
        feature: str,
        n_paths: int = 10,
        show_ci: bool = True,
        ci_levels: List[float] = [0.68, 0.95],
        save_path: str = None,
        show: bool = True
    ):
        """
        Plot synthetic paths for a feature
        
        Args:
            feature: Feature name to plot
            n_paths: Number of individual paths to show
            show_ci: Whether to show confidence intervals
            ci_levels: Confidence interval levels
            save_path: Path to save plot
            show: Whether to display plot
        
        Example:
            >>> synthetic_data.plot_paths("Gold Price", n_paths=20, show_ci=True)
        """
        from .visualization import TimeSeriesPlotter, _check_matplotlib
        import matplotlib.pyplot as plt
        from datetime import datetime
        
        _check_matplotlib()
        
        if feature not in self.features:
            raise ValueError(f"Feature {feature} not found. Available: {self.features}")
        
        print(f"[Plotting] Plotting {self.n_paths} paths for '{feature}'...")
        
        # Extract paths for this feature
        feature_data = self.get_feature(feature)
        
        # Pivot to get paths as columns
        pivot = feature_data.pivot(index='date', columns='path_idx', values=feature)
        
        dates = pivot.index.to_list()
        forecast_paths = [pivot[col].values for col in pivot.columns]
        
        # Convert dates to datetime if needed
        if isinstance(dates[0], str):
            date_objects = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
        else:
            date_objects = dates
        
        # Create plot (no past data for scenarios)
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot individual paths
        n_to_plot = min(n_paths, len(forecast_paths))
        for i in range(n_to_plot):
            ax.plot(date_objects, forecast_paths[i], '-', 
                   alpha=0.3, linewidth=1, color='steelblue')
        
        # Add legend entry
        ax.plot([], [], '-', alpha=0.5, linewidth=1.5, color='steelblue', 
               label=f'Synthetic Paths (n={len(forecast_paths)})')
        
        # Add confidence intervals
        if show_ci and len(forecast_paths) > 1:
            forecast_array = np.array(forecast_paths)
            
            colors = ['#4A90E2', '#7CB342']
            for idx, ci_level in enumerate(sorted(ci_levels, reverse=True)):
                lower_q = (1 - ci_level) / 2
                upper_q = 1 - lower_q
                
                lower = np.percentile(forecast_array, lower_q * 100, axis=0)
                upper = np.percentile(forecast_array, upper_q * 100, axis=0)
                
                ax.fill_between(date_objects, lower, upper, 
                               alpha=0.2, color=colors[idx % len(colors)],
                               label=f'{int(ci_level*100)}% CI')
        
        # Add median
        if len(forecast_paths) > 1:
            median_path = np.median(np.array(forecast_paths), axis=0)
            ax.plot(date_objects, median_path, '-', color='darkred', 
                   linewidth=2.5, label='Median Path', alpha=0.9)
        
        # Formatting
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel(feature, fontsize=11)
        ax.set_title(f'Synthetic Paths: {feature}', fontsize=13, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9, fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Format dates
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def __repr__(self):
        return f"SyntheticData(n_paths={self.n_paths}, features={self.features}, timesteps={len(self.dates)})"
