"""
Data collection for fetching and processing training data from multiple APIs
"""

from typing import List, Dict, Any, Optional
import pandas as pd


class DataSource:
    """Base class for data sources (FRED, Yahoo, etc.)"""
    
    def __init__(self, name: str, requires_api_key: bool = False):
        self.name = name
        self.requires_api_key = requires_api_key
    
    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search for available series"""
        raise NotImplementedError
    
    def get_info(self, series_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Get metadata about a series"""
        raise NotImplementedError


class FREDDataSource(DataSource):
    """FRED (Federal Reserve Economic Data) API"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("FRED", requires_api_key=True)
        self.api_key = api_key
    
    def search(self, query: str, api_key: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Search FRED for economic data series"""
        try:
            from fredapi import Fred
        except ImportError:
            raise ImportError("fredapi required. Install: pip install fredapi")
        
        key = api_key or self.api_key
        if not key:
            raise ValueError("FRED API key required. Get one at: https://fred.stlouisfed.org/docs/api/api_key.html")
        
        fred = Fred(api_key=key)
        results = fred.search(query, limit=limit)
        
        return [
            {
                "id": row['id'],
                "name": row.get('title', ''),
                "frequency": row.get('frequency', 'Unknown'),
                "units": row.get('units', ''),
                "source": "FRED"
            }
            for idx, row in results.iterrows()
        ]
    
    def get_info(self, series_id: str, api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get detailed FRED series metadata"""
        try:
            from fredapi import Fred
        except ImportError:
            raise ImportError("fredapi required. Install: pip install fredapi")
        
        key = api_key or self.api_key
        if not key:
            raise ValueError("FRED API key required")
        
        fred = Fred(api_key=key)
        
        try:
            info = fred.get_series_info(series_id)
            return {
                "id": series_id,
                "name": info.get('title', ''),
                "frequency": info.get('frequency', ''),
                "units": info.get('units', ''),
                "source": "FRED"
            }
        except:
            return None


class YahooDataSource(DataSource):
    """Yahoo Finance API"""
    
    def __init__(self):
        super().__init__("Yahoo Finance", requires_api_key=False)
        
        # Curated list of popular tickers
        self.popular_tickers = [
            # Indices
            {"symbol": "^GSPC", "name": "S&P 500", "category": "Indices"},
            {"symbol": "^DJI", "name": "Dow Jones", "category": "Indices"},
            {"symbol": "^IXIC", "name": "NASDAQ", "category": "Indices"},
            {"symbol": "^VIX", "name": "VIX", "category": "Indices"},
            
            # Commodities
            {"symbol": "GC=F", "name": "Gold Futures", "category": "Commodities"},
            {"symbol": "SI=F", "name": "Silver Futures", "category": "Commodities"},
            {"symbol": "CL=F", "name": "Crude Oil Futures", "category": "Commodities"},
            
            # Crypto
            {"symbol": "BTC-USD", "name": "Bitcoin", "category": "Crypto"},
            {"symbol": "ETH-USD", "name": "Ethereum", "category": "Crypto"},
        ]
    
    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search Yahoo Finance tickers"""
        query_lower = query.lower()
        results = [
            {**ticker, "source": "Yahoo"}
            for ticker in self.popular_tickers
            if query_lower in ticker['name'].lower() or 
               query_lower in ticker['symbol'].lower() or
               query_lower in ticker.get('category', '').lower()
        ]
        return results
    
    def get_info(self, symbol: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Get Yahoo Finance ticker info"""
        for ticker in self.popular_tickers:
            if ticker['symbol'] == symbol:
                return {**ticker, "source": "Yahoo"}
        return None


class DataCollector:
    """
    Data collection and management for a model
    
    Handles:
    - Searching available data from multiple APIs (FRED, Yahoo, etc.)
    - Adding features to the model
    - Fetching raw data from APIs
    - Processing (interpolation, gap filling)
    - Saving to database
    
    Usage:
        >>> data = DataCollection(model, fred_api_key="...")
        >>> 
        >>> # Search for data
        >>> results = data.search("10-year treasury", source="FRED")
        >>> 
        >>> # Add features
        >>> data.add("DGS10", source="FRED", name="10-Year Treasury")
        >>> data.add("GC=F", source="Yahoo", name="Gold")
        >>> 
        >>> # Set training period
        >>> data.set_period("2020-01-01", "2024-12-31")
        >>> 
        >>> # Fetch and process everything
        >>> data.fetch_and_process(max_gap_days=7, method="linear")
    """
    
    def __init__(self, model, fred_api_key: Optional[str] = None):
        """
        Initialize data collection for a model
        
        Args:
            model: Model instance to collect data for
            fred_api_key: Optional FRED API key
        """
        self.model = model
        self.http = model.http
        
        # Initialize data sources
        self.sources = {
            "FRED": FREDDataSource(api_key=fred_api_key),
            "Yahoo": YahooDataSource()
        }
        
        # Track features to fetch
        self.features = []
        self.start_date = None
        self.end_date = None
    
    # ============================================
    # SEARCH & DISCOVERY
    # ============================================
    
    def list_sources(self) -> Dict[str, Dict[str, Any]]:
        """List all available data sources"""
        return {
            name: {
                "name": source.name,
                "requires_api_key": source.requires_api_key
            }
            for name, source in self.sources.items()
        }
    
    def search(self, query: str, source: str = "FRED", limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search for available data
        
        Args:
            query: Search term (e.g., "10-year treasury", "gold", "unemployment")
            source: "FRED" or "Yahoo" (default: "FRED")
            limit: Max results (default: 20)
            
        Returns:
            List of matching series with metadata
            
        Example:
            >>> results = data.search("treasury", source="FRED")
            >>> for series in results:
            ...     print(f"{series['id']}: {series['name']}")
        """
        if source not in self.sources:
            raise ValueError(f"Unknown source: {source}. Available: {list(self.sources.keys())}")
        
        return self.sources[source].search(query, limit=limit)
    
    def get_info(self, identifier: str, source: str = "FRED") -> Optional[Dict[str, Any]]:
        """
        Get detailed info about a series
        
        Args:
            identifier: Series ID (FRED) or symbol (Yahoo)
            source: Data source (default: "FRED")
            
        Returns:
            Series metadata
            
        Example:
            >>> info = data.get_info("DGS10", source="FRED")
            >>> print(f"{info['name']} - Frequency: {info['frequency']}")
        """
        if source not in self.sources:
            raise ValueError(f"Unknown source: {source}")
        
        return self.sources[source].get_info(identifier)
    
    # ============================================
    # FEATURE MANAGEMENT
    # ============================================
    
    def add(self, identifier: str, source: str, name: str):
        """
        Add a feature to fetch
        
        Args:
            identifier: Series ID (FRED code like "DGS10") or symbol (Yahoo like "GC=F")
            source: "FRED" or "Yahoo"
            name: Display name for this feature (e.g., "10-Year Treasury", "Gold Price")
            
        Example:
            >>> # FRED: use series code + your name
            >>> data.add("DGS10", source="FRED", name="10-Year Treasury")
            >>> 
            >>> # Yahoo: use symbol + your name
            >>> data.add("GC=F", source="Yahoo", name="Gold Price")
        """
        feature = {
            "name": identifier if source == "FRED" else name,  # FRED uses series code as identifier
            "source": source,
            "display_name": name  # User-friendly name
        }
        
        if source == "Yahoo":
            feature["symbol"] = identifier
        
        self.features.append(feature)
        print(f"✅ Added: {name} ({source}: {identifier})")
    
    def list_features(self) -> List[Dict[str, Any]]:
        """List features that will be fetched"""
        return self.features.copy()
    
    def clear_features(self):
        """Clear all features"""
        self.features = []
        print("🗑️  Cleared all features")
    
    # ============================================
    # TRAINING PERIOD
    # ============================================
    
    def set_period(self, start_date: str, end_date: str):
        """
        Set training period
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Example:
            >>> data.set_period("2020-01-01", "2024-12-31")
        """
        self.start_date = start_date
        self.end_date = end_date
        print(f"📅 Training period: {start_date} to {end_date}")
    
    # ============================================
    # FETCH & PROCESS
    # ============================================
    
    def fetch_and_process(
        self,
        max_gap_days: int = 7,
        method: str = "linear",
        confirm: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Fetch raw data from APIs and process it
        
        This handles EVERYTHING:
        1. Fetches data from FRED/Yahoo APIs
        2. Applies interpolation with gap limits
        3. Saves raw and processed data to database
        4. Updates model with features and training period
        
        Args:
            max_gap_days: Maximum gap to fill (default: 7 days)
            method: "linear", "forward_fill", or "backward_fill" (default: "linear")
            confirm: Explicit confirmation for workflow conflicts
            
        Returns:
            Statistics about data fetched and processed
            
        Example:
            >>> data.fetch_and_process(max_gap_days=7, method="linear")
        """
        # Validation
        if not self.features:
            raise ValueError("No features added. Call data.add() first")
        
        if not self.start_date or not self.end_date:
            raise ValueError("Training period not set. Call data.set_period() first")
        
        print(f"🚀 Fetching {len(self.features)} features...")
        print(f"   Period: {self.start_date} to {self.end_date}")
        print(f"   Processing: {method} interpolation (max {max_gap_days} day gaps)")
        print()
        
        # First, add features to model and set period
        self.model.add_features(self.features, confirm=confirm)
        self.model.set_training_period(self.start_date, self.end_date, confirm=confirm)
        
        # Then fetch data via the model's fetch_data method
        # Pass the FRED API key from this DataCollection instance
        fred_key = self.sources["FRED"].api_key
        result = self.model.fetch_data(
            max_gap_days=max_gap_days,
            interpolation_method=method,
            fred_api_key=fred_key,
            confirm=confirm
        )
        
        return result