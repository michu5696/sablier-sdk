"""
SyntheticData class - Core data container with validation and analysis capabilities
"""

import pandas as pd
import numpy as np
from typing import Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .models.builder import Model
    from .scenarios.builder import Scenario


class SyntheticData:
    """
    Container for synthetic market data with built-in validation and analysis
    
    This class is returned by:
    - Model.generate_forecast()
    - Scenario.generate_paths()
    
    It provides methods for:
    - Statistical validation
    - Comparison to real data
    - Comparison to other synthetic datasets
    - Visualization
    - Export
    
    Example:
        >>> synthetic_data = scenario.generate_paths(n_samples=1000)
        >>> validation = synthetic_data.validate_realism()
        >>> synthetic_data.plot_distributions()
        >>> synthetic_data.to_csv("results.csv")
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        source_model: 'Model',
        source_scenario: Optional['Scenario'] = None,
        metadata: Optional[dict] = None
    ):
        """
        Initialize SyntheticData instance
        
        Args:
            data: DataFrame containing synthetic paths
            source_model: Model that generated this data
            source_scenario: Optional scenario that generated this data
            metadata: Additional metadata about generation
        """
        self.data = data
        self.source_model = source_model
        self.source_scenario = source_scenario
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        n_samples = len(self.data) if len(self.data.shape) > 1 else self.data.shape[0]
        n_features = len(self.data.columns) if hasattr(self.data, 'columns') else 0
        source = f"scenario='{self.source_scenario.name}'" if self.source_scenario else f"model='{self.source_model.name}'"
        return f"SyntheticData(samples={n_samples}, features={n_features}, {source})"
    
    def __len__(self) -> int:
        """Number of samples/paths"""
        return len(self.data)
    
    # ============================================
    # DATA ACCESS
    # ============================================
    
    def to_dataframe(self) -> pd.DataFrame:
        """Get data as pandas DataFrame"""
        return self.data.copy()
    
    def to_numpy(self) -> np.ndarray:
        """Get data as numpy array"""
        return self.data.values
    
    def get_feature(self, feature_name: str) -> pd.DataFrame:
        """
        Get all paths for a specific feature
        
        Args:
            feature_name: Feature name
            
        Returns:
            pd.DataFrame: All paths for the feature
        """
        if feature_name in self.data.columns:
            return self.data[feature_name]
        raise ValueError(f"Feature '{feature_name}' not found in synthetic data")
    
    @property
    def features(self) -> list[str]:
        """List of available features"""
        return list(self.data.columns)
    
    # ============================================
    # STATISTICAL PROPERTIES
    # ============================================
    
    def describe(self) -> pd.DataFrame:
        """Statistical summary of all features"""
        return self.data.describe()
    
    def correlation_matrix(self) -> pd.DataFrame:
        """Correlation matrix"""
        return self.data.corr()
    
    def moments(self, feature: Optional[str] = None) -> dict:
        """
        Calculate statistical moments
        
        Args:
            feature: Optional specific feature, otherwise all features
            
        Returns:
            dict: Moments (mean, variance, skewness, kurtosis)
        """
        from scipy.stats import skew, kurtosis
        
        if feature:
            data = self.data[feature]
            return {
                "mean": float(data.mean()),
                "variance": float(data.var()),
                "std": float(data.std()),
                "skewness": float(skew(data)),
                "kurtosis": float(kurtosis(data))
            }
        else:
            # Return for all features
            return {
                feat: self.moments(feat)
                for feat in self.features
            }
    
    # ============================================
    # VALIDATION
    # ============================================
    
    def validate_realism(
        self,
        historical_data: Optional[pd.DataFrame] = None,
        tests: Optional[list[str]] = None,
        significance_level: float = 0.05
    ) -> 'ValidationResults':
        """
        Validate synthetic data realism
        
        Args:
            historical_data: Optional real data for comparison
            tests: Specific tests to run (default: all)
            significance_level: Significance level for statistical tests
            
        Returns:
            ValidationResults: Comprehensive validation results
        """
        from .validators.validator import SyntheticDataValidator
        
        validator = SyntheticDataValidator(self)
        if historical_data is not None:
            validator.add_real_data(historical_data)
        
        return validator.run_tests(tests or ["all"], significance_level)
    
    def quick_validate(self) -> 'QuickValidationResults':
        """
        Quick validation check (basic statistics only)
        
        Returns:
            QuickValidationResults: Summary validation results
        """
        return self.validate_realism(tests=["moments", "distribution"])
    
    def compare_to_real(self, real_data: pd.DataFrame) -> 'ComparisonResults':
        """
        Detailed comparison to real data
        
        Args:
            real_data: Historical data for comparison
            
        Returns:
            ComparisonResults: Detailed comparison results
        """
        from .validators.validator import SyntheticDataValidator
        
        validator = SyntheticDataValidator(self)
        return validator.compare(real_data)
    
    # ============================================
    # COMPARISON
    # ============================================
    
    @staticmethod
    def compare(
        datasets: list['SyntheticData'],
        labels: Optional[list[str]] = None
    ) -> 'MultiDatasetComparison':
        """
        Compare multiple synthetic datasets
        
        Args:
            datasets: List of SyntheticData instances
            labels: Optional labels for each dataset
            
        Returns:
            MultiDatasetComparison: Comparison results
        """
        from .validators.comparator import MultiDatasetComparison
        return MultiDatasetComparison(datasets, labels)
    
    # ============================================
    # VISUALIZATION
    # ============================================
    
    def plot_distributions(self, features: Optional[list[str]] = None):
        """Plot distribution histograms for features"""
        # TODO: Implement visualization
        print(f"Plotting distributions...")
    
    def plot_time_series(
        self,
        feature: str,
        n_paths: int = 10,
        show_percentiles: bool = True
    ):
        """
        Plot time series paths
        
        Args:
            feature: Feature to plot
            n_paths: Number of individual paths to show
            show_percentiles: Whether to show percentile bands
        """
        # TODO: Implement time series plotting
        print(f"Plotting time series for {feature}...")
    
    def plot_correlation_heatmap(self):
        """Plot correlation matrix heatmap"""
        # TODO: Implement correlation heatmap
        print("Plotting correlation heatmap...")
    
    def plot_qq(self, feature: str, reference_data: Optional[pd.DataFrame] = None):
        """
        Q-Q plot for a feature
        
        Args:
            feature: Feature name
            reference_data: Optional reference data for comparison
        """
        # TODO: Implement Q-Q plot
        print(f"Plotting Q-Q plot for {feature}...")
    
    # ============================================
    # EXPORT
    # ============================================
    
    def to_csv(self, filepath: str, **kwargs):
        """
        Export to CSV
        
        Args:
            filepath: Output file path
            **kwargs: Additional arguments for pandas.to_csv
        """
        self.data.to_csv(filepath, index=False, **kwargs)
        print(f"✅ Exported to {filepath}")
    
    def to_parquet(self, filepath: str, **kwargs):
        """
        Export to Parquet (compressed, efficient)
        
        Args:
            filepath: Output file path
            **kwargs: Additional arguments for pandas.to_parquet
        """
        self.data.to_parquet(filepath, compression='gzip', **kwargs)
        print(f"✅ Exported to {filepath}")
    
    def to_excel(
        self,
        filepath: str,
        include_summary: bool = True,
        include_validation: bool = False
    ):
        """
        Export to Excel with optional summary sheets
        
        Args:
            filepath: Output file path
            include_summary: Include statistical summary sheet
            include_validation: Include validation results sheet
        """
        # TODO: Implement Excel export with multiple sheets
        print(f"✅ Exported to {filepath}")
