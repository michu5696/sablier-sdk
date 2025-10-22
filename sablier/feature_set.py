"""FeatureSet class for managing conditioning and target feature sets"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, date
from .http_client import HTTPClient

logger = logging.getLogger(__name__)


class FeatureSet:
    """
    Represents a set of features (conditioning or target) with data collectors
    
    A FeatureSet encapsulates:
    - Feature definitions (name, source, symbol, etc.)
    - Data collectors configuration (API keys, sources)
    - Computed feature groups (from correlation analysis)
    - Fetched data availability status
    """
    
    def __init__(self, 
                 http_client: HTTPClient, 
                 feature_set_data: dict, 
                 project_id: str,
                 interactive: bool = True):
        """
        Initialize FeatureSet instance
        
        Args:
            http_client: HTTP client for API requests
            feature_set_data: Feature set data from API
            project_id: Parent project ID
            interactive: Whether to prompt for confirmations
        """
        self.http = http_client
        self._data = feature_set_data
        self.project_id = project_id
        self.interactive = interactive
        
        # Core attributes
        self.id = feature_set_data.get('id')
        self.name = feature_set_data.get('name')
        self.description = feature_set_data.get('description', '')
        self.set_type = feature_set_data.get('set_type')  # 'conditioning' or 'target'
    
    def __repr__(self) -> str:
        return f"FeatureSet(id='{self.id}', name='{self.name}', type='{self.set_type}')"
    
    # ============================================
    # PROPERTIES
    # ============================================
    
    @property
    def features(self) -> List[Dict[str, Any]]:
        """Get feature definitions"""
        return self._data.get('features', [])
    
    @property
    def data_collectors(self) -> List[Dict[str, Any]]:
        """Get data collector configurations"""
        return self._data.get('data_collectors', [])
    
    @property
    def feature_groups(self) -> Optional[Dict[str, Any]]:
        """Get computed feature groups"""
        return self._data.get('feature_groups')
    
    @property
    def fetched_data_available(self) -> bool:
        """Check if data has been fetched for this feature set"""
        return self._data.get('fetched_data_available', False)
    
    # ============================================
    # FEATURE MANAGEMENT
    # ============================================
    
    def add_feature(self, 
                   name: str, 
                   source: str = "fred",
                   symbol: Optional[str] = None,
                   display_name: Optional[str] = None) -> 'FeatureSet':
        """
        Add a feature to this feature set
        
        Args:
            name: Feature name (e.g., "DGS10" for FRED)
            source: Data source ("fred", "yahoo")
            symbol: Symbol for Yahoo Finance (e.g., "^GSPC")
            display_name: Human-readable name (defaults to name)
            
        Returns:
            self (for chaining)
            
        Example:
            >>> conditioning_set.add_feature("DGS10", source="fred", display_name="10-Year Treasury")
            >>> conditioning_set.add_feature("S&P 500", source="yahoo", symbol="^GSPC")
        """
        feature = {
            "name": name,
            "source": source,
            "display_name": display_name or name
        }
        
        if symbol:
            feature["symbol"] = symbol
        
        # Add to local data
        current_features = self.features.copy()
        current_features.append(feature)
        
        # Update via API
        self._update_features(current_features)
        
        print(f"✅ Added feature: {display_name or name} ({source})")
        return self
    
    def add_features(self, features: List[Dict[str, Any]]) -> 'FeatureSet':
        """
        Add multiple features to this feature set
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            self (for chaining)
            
        Example:
            >>> features = [
            ...     {"name": "DGS10", "source": "fred", "display_name": "10-Year Treasury"},
            ...     {"name": "DGS2", "source": "fred", "display_name": "2-Year Treasury"}
            ... ]
            >>> target_set.add_features(features)
        """
        current_features = self.features.copy()
        current_features.extend(features)
        
        self._update_features(current_features)
        
        print(f"✅ Added {len(features)} features to {self.name}")
        return self
    
    def remove_feature(self, name: str) -> 'FeatureSet':
        """
        Remove a feature from this feature set
        
        Args:
            name: Feature name to remove
            
        Returns:
            self (for chaining)
        """
        current_features = [f for f in self.features if f.get('name') != name]
        
        if len(current_features) == len(self.features):
            print(f"⚠️  Feature '{name}' not found in {self.name}")
            return self
        
        self._update_features(current_features)
        print(f"✅ Removed feature: {name}")
        return self
    
    def _update_features(self, features: List[Dict[str, Any]]) -> None:
        """Update features via API"""
        response = self.http.patch(f'/api/v1/feature-sets/{self.id}', {
            "features": features
        })
        self._data = response
    
    # ============================================
    # DATA COLLECTOR MANAGEMENT
    # ============================================
    
    def add_data_collector(self, 
                          source: str, 
                          api_key: Optional[str] = None,
                          config: Optional[Dict[str, Any]] = None) -> 'DataCollectorWrapper':
        """
        Add a data collector to this feature set and return a collector wrapper
        
        The collector wrapper allows you to add features specific to that data source.
        This is the preferred flow:
        1. Add data collector: collector = feature_set.add_data_collector("fred", api_key="...")
        2. Add features to collector: collector.add_features([...])
        3. Fetch data: feature_set.fetch_data()
        
        Args:
            source: Data source ("fred", "yahoo")
            api_key: API key (required for FRED)
            config: Additional configuration
            
        Returns:
            DataCollectorWrapper for chaining feature additions
            
        Example:
            >>> # FRED collector
            >>> fred_collector = conditioning_set.add_data_collector("fred", api_key="your_key")
            >>> fred_collector.add_features([
            ...     {"series_id": "DGS10", "display_name": "10-Year Treasury"},
            ...     {"series_id": "UNRATE", "display_name": "Unemployment Rate"}
            ... ])
            >>> 
            >>> # Yahoo collector
            >>> yahoo_collector = conditioning_set.add_data_collector("yahoo")
            >>> yahoo_collector.add_features([
            ...     {"symbol": "^GSPC", "display_name": "S&P 500"}
            ... ])
        """
        collector = {
            "source": source,
            "api_key": api_key,
            "config": config or {}
        }
        
        current_collectors = self.data_collectors.copy()
        current_collectors.append(collector)
        
        self._update_data_collectors(current_collectors)
        
        print(f"✅ Added data collector: {source}")
        
        # Return a collector wrapper for adding features
        from .data_collector import create_collector_wrapper
        return create_collector_wrapper(self, source, api_key, **config or {})
    
    def _update_data_collectors(self, collectors: List[Dict[str, Any]]) -> None:
        """Update data collectors via API"""
        response = self.http.patch(f'/api/v1/feature-sets/{self.id}', {
            "data_collectors": collectors
        })
        self._data = response
    
    # ============================================
    # DATA FETCHING
    # ============================================
    
    def fetch_data(self, 
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  confirm: Optional[bool] = None) -> Dict[str, Any]:
        """
        Fetch data for all features in this feature set
        
        Args:
            start_date: Start date (uses project's training period if None)
            end_date: End date (uses project's training period if None)
            confirm: Skip confirmation prompt if True
            
        Returns:
            Dict with fetch results
            
        Example:
            >>> result = conditioning_set.fetch_data()
            >>> print(f"Fetched data for {result['features_processed']} features")
        """
        if not self.features:
            raise ValueError(f"No features defined in {self.name}")
        
        if not self.data_collectors:
            raise ValueError(f"No data collectors configured in {self.name}")
        
        # Confirmation
        if confirm is None and self.interactive:
            print(f"📥 Fetching data for {self.name} ({self.set_type} set)")
            print(f"   Features: {len(self.features)}")
            print(f"   Data collectors: {len(self.data_collectors)}")
            print()
            
            response = input("Continue? [y/N]: ")
            if response.lower() != 'y':
                return {"status": "cancelled"}
        
        # Fetch data via API
        response = self.http.post(f'/api/v1/feature-sets/{self.id}/fetch-data', {
            "start_date": start_date,
            "end_date": end_date
        })
        
        # Update local data
        self._data = response.get('feature_set', self._data)
        
        print(f"✅ Data fetched for {self.name}")
        print(f"   Features processed: {response.get('features_processed', 0)}")
        print(f"   Data points: {response.get('total_data_points', 0)}")
        
        return response
    
    # ============================================
    # FEATURE GROUPING
    # ============================================
    
    def compute_feature_groups(self, 
                              correlation_threshold: float = 0.75,
                              method: str = 'hierarchical',
                              confirm: Optional[bool] = None) -> Dict[str, Any]:
        """
        Compute feature groups based on correlation analysis
        
        Args:
            correlation_threshold: Minimum correlation for grouping
            method: Grouping method ('hierarchical', 'kmeans')
            confirm: Skip confirmation prompt if True
            
        Returns:
            Dict with grouping results
            
        Example:
            >>> groups = conditioning_set.compute_feature_groups(correlation_threshold=0.8)
            >>> print(f"Created {len(groups['groups'])} feature groups")
        """
        if not self.fetched_data_available:
            raise ValueError(f"Data must be fetched before computing feature groups")
        
        # Confirmation
        if confirm is None and self.interactive:
            print(f"🔗 Computing feature groups for {self.name}")
            print(f"   Correlation threshold: {correlation_threshold}")
            print(f"   Method: {method}")
            print()
            
            response = input("Continue? [y/N]: ")
            if response.lower() != 'y':
                return {"status": "cancelled"}
        
        # Compute groups via API
        response = self.http.post(f'/api/v1/feature-sets/{self.id}/compute-groups', {
            "correlation_threshold": correlation_threshold,
            "method": method
        })
        
        # Update local data
        self._data = response.get('feature_set', self._data)
        
        groups = response.get('feature_groups', {})
        n_groups = len(groups.get('groups', []))
        
        print(f"✅ Feature groups computed for {self.name}")
        print(f"   Groups created: {n_groups}")
        print(f"   Multivariate groups: {len([g for g in groups.get('groups', []) if g.get('is_multivariate', False)])}")
        
        return response
    
    # ============================================
    # UTILITY METHODS
    # ============================================
    
    def refresh(self) -> 'FeatureSet':
        """Refresh feature set data from API"""
        response = self.http.get(f'/api/v1/feature-sets/{self.id}')
        self._data = response
        return self
    
    def delete(self, confirm: Optional[bool] = None) -> Dict[str, Any]:
        """
        Delete this feature set
        
        Args:
            confirm: Skip confirmation prompt if True
            
        Returns:
            Dict with deletion status
        """
        # Always warn for deletion
        print("⚠️  WARNING: You are about to delete this feature set.")
        print(f"   Feature Set: {self.name} ({self.set_type})")
        print(f"   Features: {len(self.features)}")
        print()
        print("This will also delete:")
        print("  - All fetched data")
        print("  - All models using this feature set")
        print()
        
        # Get confirmation
        if confirm is None and self.interactive:
            response = input("Type the feature set name to confirm deletion: ")
            if response != self.name:
                print("❌ Feature set name doesn't match. Deletion cancelled.")
                return {"status": "cancelled"}
        elif confirm is None:
            print("❌ Deletion cancelled (interactive=False, no confirmation)")
            return {"status": "cancelled"}
        elif not confirm:
            print("❌ Deletion cancelled")
            return {"status": "cancelled"}
        
        # Delete via API
        print("🗑️  Deleting feature set...")
        response = self.http.delete(f'/api/v1/feature-sets/{self.id}')
        
        print(f"✅ Feature set '{self.name}' deleted")
        return response
