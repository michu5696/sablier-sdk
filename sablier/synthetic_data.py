"""SyntheticData class for managing and analyzing generated synthetic market paths"""

import logging
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


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
    
    # ============================================
    # VALIDATION METHODS
    # ============================================
    
    def validate_against_real_data(
        self,
        real_validation_data: Dict[str, np.ndarray],
        real_test_data: Optional[Dict[str, np.ndarray]] = None,
        save_dir: Optional[str] = None,
        show_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive validation comparing synthetic data to real validation/test data.
        
        Args:
            real_validation_data: Dict with 'paths' (n_samples, n_timesteps, n_features) 
                                 and 'feature_names' list
            real_test_data: Optional test set data in same format
            save_dir: Directory to save plots and reports
            show_plots: Whether to display plots interactively
        
        Returns:
            Validation results dictionary with metrics and plot paths
        
        Example:
            >>> real_val = model.get_real_paths(split='validation')
            >>> real_test = model.get_real_paths(split='test')
            >>> results = synthetic_data.validate_against_real_data(
            ...     real_validation_data=real_val,
            ...     real_test_data=real_test,
            ...     save_dir='./validation_report'
            ... )
        """
        if not HAS_SCIPY:
            raise ImportError("scipy is required for validation. Install: pip install scipy")
        
        print("=" * 70)
        print("VALIDATION: Synthetic vs Real Data")
        print("=" * 70)
        print()
        
        # Prepare save directory
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / 'validation').mkdir(parents=True, exist_ok=True)
            (save_dir / 'validation' / 'distributions').mkdir(parents=True, exist_ok=True)
            (save_dir / 'validation' / 'correlations').mkdir(parents=True, exist_ok=True)
            (save_dir / 'test').mkdir(parents=True, exist_ok=True)
            (save_dir / 'test' / 'distributions').mkdir(parents=True, exist_ok=True)
            (save_dir / 'test' / 'correlations').mkdir(parents=True, exist_ok=True)
            (save_dir / 'metrics').mkdir(parents=True, exist_ok=True)
        
        results = {
            'unconditional': {},
            'validation_set': {},
            'test_set': {} if real_test_data else None
        }
        
        # Run validation on validation set
        print("📊 Validating against VALIDATION SET...")
        print("-" * 70)
        val_results = self._validate_unconditional(
            real_validation_data, 
            save_dir=save_dir / 'validation' if save_dir else None,
            show_plots=show_plots,
            dataset_name='Validation'
        )
        results['validation_set'] = val_results
        
        # Run validation on test set if provided
        if real_test_data:
            print("\n📊 Validating against TEST SET...")
            print("-" * 70)
            test_results = self._validate_unconditional(
                real_test_data,
                save_dir=save_dir / 'test' if save_dir else None,
                show_plots=show_plots,
                dataset_name='Test'
            )
            results['test_set'] = test_results
        
        # Summary
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        self._print_validation_summary(results)
        
        if save_dir:
            import json
            with open(save_dir / 'validation_results.json', 'w') as f:
                # Convert numpy types to python types for JSON
                json_results = self._convert_to_json_serializable(results)
                json.dump(json_results, f, indent=2)
            print(f"\n💾 Results saved to {save_dir}/validation_results.json")
        
        return results
    
    def _validate_unconditional(
        self,
        real_data: Dict[str, np.ndarray],
        save_dir: Optional[Path] = None,
        show_plots: bool = True,
        dataset_name: str = 'Real'
    ) -> Dict[str, Any]:
        """Run unconditional validation tests"""
        
        results = {}
        
        # 1. Distribution validation
        print("\n1️⃣  Distribution Tests...")
        dist_results = self.validate_distributions(
            real_data,
            save_dir=save_dir / 'distributions' if save_dir else None,
            show_plots=show_plots,
            dataset_name=dataset_name
        )
        results['distributions'] = dist_results
        
        # 2. Moments comparison
        print("\n2️⃣  Moment Comparison...")
        moment_results = self.validate_moments(real_data)
        results['moments'] = moment_results
        
        # 3. Correlation comparison
        print("\n3️⃣  Correlation Structure...")
        corr_results = self.compare_correlations(
            real_data,
            save_dir=save_dir / 'correlations' if save_dir else None,
            show_plots=show_plots,
            dataset_name=dataset_name
        )
        results['correlations'] = corr_results
        
        # 4. Tail validation
        print("\n4️⃣  Tail Behavior...")
        tail_results = self.validate_tails(
            real_data,
            save_dir=save_dir / 'distributions' if save_dir else None,
            show_plots=show_plots,
            dataset_name=dataset_name
        )
        results['tails'] = tail_results
        
        return results
    
    def validate_distributions(
        self,
        real_data: Dict[str, np.ndarray],
        save_dir: Optional[Path] = None,
        show_plots: bool = True,
        dataset_name: str = 'Real'
    ) -> Dict[str, Any]:
        """
        Validate distributions using statistical tests and visualizations.
        
        Returns KS test, AD test p-values and generates Q-Q plots, histograms.
        """
        import matplotlib.pyplot as plt
        
        # Extract synthetic and real data
        synthetic_paths = self._get_paths_array()  # (n_paths, n_timesteps, n_features)
        real_paths = real_data['paths']  # (n_samples, n_timesteps, n_features)
        
        # Flatten to get all values per feature
        synthetic_flat = synthetic_paths.reshape(-1, len(self.features))
        real_flat = real_paths.reshape(-1, len(self.features))
        
        results = {}
        
        for i, feature in enumerate(self.features):
            synthetic_values = synthetic_flat[:, i]
            real_values = real_flat[:, i]
            
            # Remove NaNs
            synthetic_values = synthetic_values[~np.isnan(synthetic_values)]
            real_values = real_values[~np.isnan(real_values)]
            
            # KS test
            ks_stat, ks_pval = stats.ks_2samp(synthetic_values, real_values)
            
            # Anderson-Darling test (using combined sample)
            combined = np.concatenate([synthetic_values, real_values])
            ad_result = stats.anderson_ksamp([synthetic_values, real_values])
            
            results[feature] = {
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pval),
                'ad_statistic': float(ad_result.statistic),
                'ad_pvalue': float(ad_result.pvalue) if hasattr(ad_result, 'pvalue') else None,
                'n_synthetic': len(synthetic_values),
                'n_real': len(real_values)
            }
            
            # Print results
            pass_ks = "✅" if ks_pval > 0.05 else "❌"
            print(f"  {feature}:")
            print(f"    KS test: p={ks_pval:.4f} {pass_ks}")
            print(f"    AD test: stat={ad_result.statistic:.4f}")
        
        # Generate plots
        if save_dir or show_plots:
            self._plot_distribution_comparison(
                synthetic_flat, real_flat, self.features,
                save_dir=save_dir, show_plots=show_plots, dataset_name=dataset_name
            )
        
        return results
    
    def validate_moments(self, real_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Compare statistical moments (mean, std, skewness, kurtosis).
        
        Returns table-ready comparison of moments.
        """
        # Extract data
        synthetic_paths = self._get_paths_array()
        real_paths = real_data['paths']
        
        synthetic_flat = synthetic_paths.reshape(-1, len(self.features))
        real_flat = real_paths.reshape(-1, len(self.features))
        
        results = {}
        
        print("\n  Feature                    | Metric    | Real      | Synthetic | Diff")
        print("  " + "-" * 75)
        
        for i, feature in enumerate(self.features):
            syn_vals = synthetic_flat[:, i][~np.isnan(synthetic_flat[:, i])]
            real_vals = real_flat[:, i][~np.isnan(real_flat[:, i])]
            
            moments = {
                'mean': {
                    'real': float(np.mean(real_vals)),
                    'synthetic': float(np.mean(syn_vals)),
                },
                'std': {
                    'real': float(np.std(real_vals)),
                    'synthetic': float(np.std(syn_vals)),
                },
                'skewness': {
                    'real': float(stats.skew(real_vals)),
                    'synthetic': float(stats.skew(syn_vals)),
                },
                'kurtosis': {
                    'real': float(stats.kurtosis(real_vals)),
                    'synthetic': float(stats.kurtosis(syn_vals)),
                }
            }
            
            # Add differences
            for metric in moments:
                moments[metric]['diff'] = moments[metric]['synthetic'] - moments[metric]['real']
                moments[metric]['pct_diff'] = (moments[metric]['diff'] / moments[metric]['real'] * 100) if moments[metric]['real'] != 0 else 0
            
            results[feature] = moments
            
            # Print compact table
            for metric in ['mean', 'std', 'skewness', 'kurtosis']:
                real_val = moments[metric]['real']
                syn_val = moments[metric]['synthetic']
                diff_pct = moments[metric]['pct_diff']
                
                print(f"  {feature:26s} | {metric:9s} | {real_val:9.4f} | {syn_val:9.4f} | {diff_pct:+6.1f}%")
        
        return results
    
    def compare_correlations(
        self,
        real_data: Dict[str, np.ndarray],
        save_dir: Optional[Path] = None,
        show_plots: bool = True,
        dataset_name: str = 'Real'
    ) -> Dict[str, Any]:
        """
        Compare correlation matrices between synthetic and real data.
        
        Returns correlation difference and generates heatmap visualizations.
        """
        import matplotlib.pyplot as plt
        
        # Extract data
        synthetic_paths = self._get_paths_array()
        real_paths = real_data['paths']
        
        # Flatten to (n_samples, n_features)
        synthetic_flat = synthetic_paths.reshape(-1, len(self.features))
        real_flat = real_paths.reshape(-1, len(self.features))
        
        # Compute correlation matrices
        synthetic_corr = np.corrcoef(synthetic_flat.T)
        real_corr = np.corrcoef(real_flat.T)
        
        # Compute difference
        corr_diff = synthetic_corr - real_corr
        frobenius_dist = np.linalg.norm(corr_diff, 'fro')
        max_diff = np.max(np.abs(corr_diff))
        
        results = {
            'synthetic_correlation': synthetic_corr.tolist(),
            'real_correlation': real_corr.tolist(),
            'difference': corr_diff.tolist(),
            'frobenius_distance': float(frobenius_dist),
            'max_absolute_difference': float(max_diff)
        }
        
        print(f"  Frobenius distance: {frobenius_dist:.4f}")
        print(f"  Max absolute diff:  {max_diff:.4f}")
        
        # Generate heatmap plots
        if save_dir or show_plots:
            self._plot_correlation_comparison(
                real_corr, synthetic_corr, self.features,
                save_dir=save_dir, show_plots=show_plots, dataset_name=dataset_name
            )
        
        return results
    
    def validate_tails(
        self,
        real_data: Dict[str, np.ndarray],
        quantiles: List[float] = [0.95, 0.99, 0.999],
        save_dir: Optional[Path] = None,
        show_plots: bool = True,
        dataset_name: str = 'Real'
    ) -> Dict[str, Any]:
        """
        Validate tail behavior using VaR, CVaR, and tail plots.
        """
        import matplotlib.pyplot as plt
        
        synthetic_paths = self._get_paths_array()
        real_paths = real_data['paths']
        
        synthetic_flat = synthetic_paths.reshape(-1, len(self.features))
        real_flat = real_paths.reshape(-1, len(self.features))
        
        results = {}
        
        print("\n  Feature                    | Quantile | Real VaR  | Syn VaR   | Diff")
        print("  " + "-" * 75)
        
        for i, feature in enumerate(self.features):
            syn_vals = synthetic_flat[:, i][~np.isnan(synthetic_flat[:, i])]
            real_vals = real_flat[:, i][~np.isnan(real_flat[:, i])]
            
            feature_results = {}
            
            for q in quantiles:
                real_var = np.percentile(real_vals, q * 100)
                syn_var = np.percentile(syn_vals, q * 100)
                
                # CVaR (expected shortfall beyond VaR)
                real_cvar = np.mean(real_vals[real_vals >= real_var])
                syn_cvar = np.mean(syn_vals[syn_vals >= syn_var])
                
                feature_results[f'q{q}'] = {
                    'real_var': float(real_var),
                    'synthetic_var': float(syn_var),
                    'real_cvar': float(real_cvar),
                    'synthetic_cvar': float(syn_cvar),
                    'var_diff': float(syn_var - real_var),
                    'cvar_diff': float(syn_cvar - real_cvar)
                }
                
                print(f"  {feature:26s} | {q:8.3f} | {real_var:9.4f} | {syn_var:9.4f} | {syn_var-real_var:+7.4f}")
            
            results[feature] = feature_results
        
        # Generate tail plots
        if save_dir or show_plots:
            self._plot_tail_comparison(
                synthetic_flat, real_flat, self.features,
                save_dir=save_dir, show_plots=show_plots, dataset_name=dataset_name
            )
        
        return results
    
    # ============================================
    # HELPER METHODS
    # ============================================
    
    def _get_paths_array(self) -> np.ndarray:
        """Convert paths DataFrame to numpy array (n_paths, n_timesteps, n_features)"""
        paths_list = []
        for path_idx in range(self.n_paths):
            path_df = self.get_path(path_idx)
            path_values = path_df[self.features].values  # (n_timesteps, n_features)
            paths_list.append(path_values)
        return np.array(paths_list)
    
    def _plot_distribution_comparison(
        self, synthetic_flat, real_flat, features,
        save_dir=None, show_plots=True, dataset_name='Real'
    ):
        """Generate Q-Q plots and histogram overlays"""
        import matplotlib.pyplot as plt
        
        n_features = len(features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        # Q-Q plots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, feature in enumerate(features):
            ax = axes[i]
            syn_vals = synthetic_flat[:, i][~np.isnan(synthetic_flat[:, i])]
            real_vals = real_flat[:, i][~np.isnan(real_flat[:, i])]
            
            # Q-Q plot
            percs = np.linspace(0, 100, 100)
            syn_quantiles = np.percentile(syn_vals, percs)
            real_quantiles = np.percentile(real_vals, percs)
            
            ax.scatter(real_quantiles, syn_quantiles, alpha=0.5, s=20)
            ax.plot([real_quantiles.min(), real_quantiles.max()],
                   [real_quantiles.min(), real_quantiles.max()],
                   'r--', lw=2, label='Perfect match')
            ax.set_xlabel(f'{dataset_name} Quantiles', fontsize=9)
            ax.set_ylabel('Synthetic Quantiles', fontsize=9)
            ax.set_title(f'Q-Q Plot: {feature}', fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / 'qq_plots.png', dpi=150, bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # Histogram overlays
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, feature in enumerate(features):
            ax = axes[i]
            syn_vals = synthetic_flat[:, i][~np.isnan(synthetic_flat[:, i])]
            real_vals = real_flat[:, i][~np.isnan(real_flat[:, i])]
            
            ax.hist(real_vals, bins=50, alpha=0.5, density=True, label=dataset_name, color='steelblue')
            ax.hist(syn_vals, bins=50, alpha=0.5, density=True, label='Synthetic', color='coral')
            ax.set_xlabel('Value', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.set_title(f'Distribution: {feature}', fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / 'histograms.png', dpi=150, bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    def _plot_correlation_comparison(
        self, real_corr, synthetic_corr, features,
        save_dir=None, show_plots=True, dataset_name='Real'
    ):
        """Generate side-by-side correlation heatmaps"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Real correlation
        im1 = axes[0].imshow(real_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        axes[0].set_title(f'{dataset_name} Data Correlation', fontsize=12, fontweight='bold')
        axes[0].set_xticks(range(len(features)))
        axes[0].set_yticks(range(len(features)))
        axes[0].set_xticklabels(features, rotation=45, ha='right', fontsize=8)
        axes[0].set_yticklabels(features, fontsize=8)
        plt.colorbar(im1, ax=axes[0], fraction=0.046)
        
        # Synthetic correlation
        im2 = axes[1].imshow(synthetic_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        axes[1].set_title('Synthetic Data Correlation', fontsize=12, fontweight='bold')
        axes[1].set_xticks(range(len(features)))
        axes[1].set_yticks(range(len(features)))
        axes[1].set_xticklabels(features, rotation=45, ha='right', fontsize=8)
        axes[1].set_yticklabels(features, fontsize=8)
        plt.colorbar(im2, ax=axes[1], fraction=0.046)
        
        # Difference
        diff = synthetic_corr - real_corr
        max_abs_diff = np.max(np.abs(diff))
        im3 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-max_abs_diff, vmax=max_abs_diff, aspect='auto')
        axes[2].set_title('Difference (Syn - Real)', fontsize=12, fontweight='bold')
        axes[2].set_xticks(range(len(features)))
        axes[2].set_yticks(range(len(features)))
        axes[2].set_xticklabels(features, rotation=45, ha='right', fontsize=8)
        axes[2].set_yticklabels(features, fontsize=8)
        plt.colorbar(im3, ax=axes[2], fraction=0.046)
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / 'correlation_heatmaps.png', dpi=150, bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    def _plot_tail_comparison(
        self, synthetic_flat, real_flat, features,
        save_dir=None, show_plots=True, dataset_name='Real'
    ):
        """Generate tail exceedance plots"""
        import matplotlib.pyplot as plt
        
        n_features = len(features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, feature in enumerate(features):
            ax = axes[i]
            syn_vals = synthetic_flat[:, i][~np.isnan(synthetic_flat[:, i])]
            real_vals = real_flat[:, i][~np.isnan(real_flat[:, i])]
            
            # Upper tail exceedance (probability plot)
            sorted_real = np.sort(real_vals)[::-1]
            sorted_syn = np.sort(syn_vals)[::-1]
            
            n_real = len(sorted_real)
            n_syn = len(sorted_syn)
            
            # Empirical exceedance probability
            real_exc_prob = np.arange(1, n_real + 1) / n_real
            syn_exc_prob = np.arange(1, n_syn + 1) / n_syn
            
            ax.loglog(real_exc_prob, sorted_real, 'o-', alpha=0.6, markersize=3, label=dataset_name)
            ax.loglog(syn_exc_prob, sorted_syn, 's-', alpha=0.6, markersize=3, label='Synthetic')
            ax.set_xlabel('Exceedance Probability', fontsize=9)
            ax.set_ylabel('Value', fontsize=9)
            ax.set_title(f'Tail Behavior: {feature}', fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, which='both')
        
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / 'tail_comparison.png', dpi=150, bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    def _print_validation_summary(self, results: Dict[str, Any]):
        """Print a summary of validation results"""
        
        print("\n📋 Key Metrics:")
        print("-" * 70)
        
        # Validation set summary
        if 'validation_set' in results and results['validation_set']:
            val_dist = results['validation_set'].get('distributions', {})
            val_corr = results['validation_set'].get('correlations', {})
            
            # Count features passing KS test
            ks_passes = sum(1 for f in val_dist.values() if f.get('ks_pvalue', 0) > 0.05)
            total_features = len(val_dist)
            
            print(f"Validation Set:")
            print(f"  Distribution match (KS p>0.05): {ks_passes}/{total_features} features")
            if val_corr:
                print(f"  Correlation Frobenius distance:  {val_corr.get('frobenius_distance', 'N/A'):.4f}")
        
        # Test set summary
        if results.get('test_set'):
            test_dist = results['test_set'].get('distributions', {})
            test_corr = results['test_set'].get('correlations', {})
            
            ks_passes = sum(1 for f in test_dist.values() if f.get('ks_pvalue', 0) > 0.05)
            total_features = len(test_dist)
            
            print(f"\nTest Set:")
            print(f"  Distribution match (KS p>0.05): {ks_passes}/{total_features} features")
            if test_corr:
                print(f"  Correlation Frobenius distance:  {test_corr.get('frobenius_distance', 'N/A'):.4f}")
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
