"""
Data Collector Wrapper for SDK

Provides a user-friendly interface for configuring data collectors
and adding features with source-specific parameters.
"""

from typing import List, Dict, Any, Optional


class DataCollectorWrapper:
    """
    Wrapper for data collectors in SDK
    
    Each collector knows what parameters it needs for adding features.
    Features are added to the parent model with standardized format.
    """
    
    def __init__(self, feature_set, source: str, api_key: Optional[str] = None, **config):
        """
        Initialize collector wrapper
        
        Args:
            feature_set: Parent FeatureSet instance
            source: Data source name ('fred', 'yahoo', etc.)
            api_key: API key for authentication
            **config: Additional configuration
        """
        self.feature_set = feature_set
        self.source = source.lower()
        self.api_key = api_key
        self.config = config
    
    def add_features(self, features: List[Dict[str, Any]]) -> 'DataCollectorWrapper':
        """
        Add features using this collector
        
        This is an abstract method - subclasses implement source-specific logic
        """
        raise NotImplementedError(f"Collector {self.source} must implement add_features()")
    
    def _add_to_feature_set(self, features: List[Dict[str, Any]]):
        """Add features to parent feature_set with source annotation"""
        # Annotate each feature with this collector's source
        for feature in features:
            feature['source'] = self.source
        
        # Add to feature_set
        current_features = self.feature_set.features.copy()
        current_features.extend(features)
        self.feature_set._update_features(current_features)


class FREDCollectorWrapper(DataCollectorWrapper):
    """FRED data collector wrapper"""
    
    def add_features(self, features: List[Dict[str, Any]]) -> 'FREDCollectorWrapper':
        """
        Add FRED features
        
        Args:
            features: List of dicts with:
                - series_id: FRED series ID (e.g., 'DGS30', 'FEDFUNDS')
                - display_name: User-friendly name (e.g., 'US Treasury 30Y')
        
        Note: The 'type' field is automatically determined from the FeatureSet type.
        
        Example:
            >>> fred.add_features([
            >>>     {'series_id': 'DGS30', 'display_name': 'US Treasury 30Y'},
            >>>     {'series_id': 'FEDFUNDS', 'display_name': 'Fed Funds Rate'}
            >>> ])
        """
        standardized_features = []
        
        for feature in features:
            if 'series_id' not in feature:
                raise ValueError(f"FRED features must have 'series_id': {feature}")
            if 'display_name' not in feature:
                raise ValueError(f"FRED features must have 'display_name': {feature}")
            
            standardized_features.append({
                'name': feature['series_id'],  # FRED uses series_id as name
                'display_name': feature['display_name'],
                'source': 'fred'
            })
        
        self._add_to_feature_set(standardized_features)
        print(f"✅ Added {len(features)} FRED features")
        
        return self


class YahooCollectorWrapper(DataCollectorWrapper):
    """Yahoo Finance data collector wrapper"""
    
    def add_features(self, features: List[Dict[str, Any]]) -> 'YahooCollectorWrapper':
        """
        Add Yahoo Finance features
        
        Args:
            features: List of dicts with:
                - symbol: Ticker symbol (e.g., '^GSPC', 'GLD', '^VIX')
                - display_name: User-friendly name (e.g., 'S&P 500', 'Gold ETF')
        
        Note: The 'type' field is automatically determined from the FeatureSet type.
        
        Example:
            >>> yahoo.add_features([
            >>>     {'symbol': '^GSPC', 'display_name': 'S&P 500'},
            >>>     {'symbol': 'GLD', 'display_name': 'Gold ETF'}
            >>> ])
        """
        standardized_features = []
        
        for feature in features:
            if 'symbol' not in feature:
                raise ValueError(f"Yahoo features must have 'symbol': {feature}")
            if 'display_name' not in feature:
                raise ValueError(f"Yahoo features must have 'display_name': {feature}")
            
            standardized_features.append({
                'name': feature['display_name'],  # Yahoo uses display_name as name
                'symbol': feature['symbol'],
                'display_name': feature['display_name'],
                'source': 'yahoo'
            })
        
        self._add_to_feature_set(standardized_features)
        print(f"✅ Added {len(features)} Yahoo features")
        
        return self


# Factory function for creating collector wrappers
def create_collector_wrapper(feature_set, source: str, api_key: Optional[str] = None, **config) -> DataCollectorWrapper:
    """Create appropriate collector wrapper based on source"""
    if source.lower() == 'fred':
        return FREDCollectorWrapper(feature_set, source, api_key, **config)
    elif source.lower() == 'yahoo':
        return YahooCollectorWrapper(feature_set, source, api_key, **config)
    else:
        # Generic wrapper for future collectors
        return DataCollectorWrapper(feature_set, source, api_key, **config)
