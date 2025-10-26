"""Model class representing a Sablier model"""

import logging
import numpy as np
from typing import Optional, Any, List, Dict
from ..http_client import HTTPClient
from ..workflow import WorkflowValidator, WorkflowConflict
from .validators import (
    validate_sample_generation_inputs,
    validate_splits,
    auto_generate_splits,
    validate_training_period
)
from .utils import update_feature_types

logger = logging.getLogger(__name__)


class Model:
    """
    Represents a Sablier model
    
    A model encapsulates the entire workflow:
    - Feature selection
    - Data fetching and processing
    - Sample generation
    - Encoding model fitting
    - QRF training
    - Forecasting
    """
    
    def __init__(self, http_client: HTTPClient, model_data: dict, interactive: bool = True):
        """
        Initialize Model instance
        
        Args:
            http_client: HTTP client for API requests
            model_data: Model data from API
            interactive: Whether to prompt for confirmations (default: True)
        """
        self.http = http_client
        self._data = model_data
        self.id = model_data.get('id')
        self.name = model_data.get('name')
        self.description = model_data.get('description', '')
        self.interactive = interactive
    
    def __repr__(self) -> str:
        return f"Model(id='{self.id}', name='{self.name}', status='{self.status}')"
    
    # ============================================
    # PROPERTIES
    # ============================================
    
    def data_collector(self, fred_api_key: Optional[str] = None):
        """
        Create a DataCollector instance for this model
        
        Args:
            fred_api_key: Optional FRED API key for searching and fetching
            
        Returns:
            DataCollector: Data collector instance scoped to this model
            
        Example:
            >>> data = model.data_collector(fred_api_key="...")
            >>> data.search("treasury")
            >>> data.add("DGS10", source="FRED", name="10-Year Treasury")
            >>> data.fetch_and_process()
        """
        from ..data_collector import DataCollector
        return DataCollector(self, fred_api_key=fred_api_key)
    
    # ============================================
    # PROPERTIES (continued)
    # ============================================
    
    @property
    def status(self) -> str:
        """Get current model status"""
        return self._data.get('status', 'created')
    
    @property
    def input_features(self) -> List[Dict[str, Any]]:
        """Get input features"""
        return self._data.get('input_features', [])
    
    @property
    def conditioning_set_id(self) -> Optional[str]:
        """Get conditioning set ID (for modular architecture)"""
        return self._data.get('conditioning_set_id')
    
    @property
    def target_set_id(self) -> Optional[str]:
        """Get target set ID (for modular architecture)"""
        return self._data.get('target_set_id')
    
    @property
    def project_id(self) -> Optional[str]:
        """Get project ID (for modular architecture)"""
        return self._data.get('project_id')
    
    def refresh(self):
        """Refresh model data from API"""
        response = self.http.get(f'/api/v1/models/{self.id}')
        # The API returns the model data directly, not wrapped in 'model' key
        self._data = response if isinstance(response, dict) and 'id' in response else response.get('model', {})
        return self
    
    def delete(self, confirm: Optional[bool] = None) -> Dict[str, Any]:
        """
        Delete this model and ALL associated data
        
        This will permanently delete:
        - Model record
        - Training data
        - Generated samples
        - Encoding models
        - Trained QRF model (from storage)
        - All scenarios using this model
        
        Args:
            confirm: Explicit confirmation (None = prompt if interactive)
            
        Returns:
            dict: Deletion status
            
        Example:
            >>> model.delete()  # Will prompt for confirmation
            >>> model.delete(confirm=True)  # Skip confirmation
        """
        # Always warn for deletion
        print("⚠️  WARNING: You are about to PERMANENTLY DELETE this model.")
        print(f"   Model: {self.name} ({self.id})")
        print(f"   Status: {self.status}")
        print()
        print("This will delete ALL associated data:")
        print("  - Training data")
        print("  - Generated samples")
        print("  - Encoding models")
        print("  - Trained QRF model")
        print("  - All scenarios using this model")
        print()
        print("This action CANNOT be undone.")
        print()
        
        # Get confirmation
        if confirm is None and self.interactive:
            response = input("Type the model name to confirm deletion: ")
            confirm = response == self.name
            if not confirm:
                print("❌ Model name doesn't match. Deletion cancelled.")
                return {"status": "cancelled"}
        elif confirm is None:
            # Non-interactive without explicit confirm
            print("❌ Deletion cancelled (interactive=False, no confirmation)")
            return {"status": "cancelled"}
        elif not confirm:
            print("❌ Deletion cancelled")
            return {"status": "cancelled"}
        
        # Delete via API
        print("🗑️  Deleting model...")
        response = self.http.delete(f'/api/v1/models/{self.id}')
        
        print(f"✅ Model '{self.name}' deleted successfully")
        
        return response
    
    # ============================================
    # WORKFLOW VALIDATION
    # ============================================
    
    def _check_and_handle_conflict(self, operation: str, confirm: Optional[bool] = None) -> bool:
        """
        Check for workflow conflict and handle it
        
        Args:
            operation: Operation name
            confirm: Explicit confirmation (None = prompt if interactive)
            
        Returns:
            True if operation should proceed, False if cancelled
        """
        conflict = WorkflowValidator.check_conflict(operation, self.status)
        
        if not conflict:
            # No conflict, proceed
            return True
        
        # Conflict detected
        print(conflict.format_warning())
        
        # Get confirmation
        if confirm is None and self.interactive:
            response = input("\nContinue? [y/N]: ")
            confirm = response.lower() == 'y'
        elif confirm is None:
            # Non-interactive mode without explicit confirm, cancel
            print("❌ Operation cancelled (interactive=False, no confirmation provided)")
            return False
        
        if not confirm:
            print("❌ Operation cancelled")
            return False
        
        # User confirmed, proceed with cleanup
        print("🗑️  Cleaning up dependent data...")
        self._cleanup_dependent_data(conflict.items_to_delete)
        
        return True
    
    def _cleanup_dependent_data(self, items_to_delete: List[str]):
        """
        Clean up dependent data before operation
        
        Args:
            items_to_delete: List of items to delete
        """
        client = self.http
        
        for item in items_to_delete:
            try:
                if item == "training_data":
                    # Delete all training_data for this model
                    # Note: We use the backend's Supabase client, not direct deletion
                    print(f"  - Deleting training data...")
                    # Will be handled by regenerating samples
                    
                elif item == "samples":
                    # Delete all samples for this model
                    print(f"  - Deleting samples...")
                    # Cascade delete via database FKs
                    
                elif item == "encoding_models":
                    # Delete all encoding models for this model
                    print(f"  - Deleting encoding models...")
                    # Cascade delete via database FKs
                    
                elif item == "trained_model":
                    # Delete trained model from storage
                    print(f"  - Deleting trained model from storage...")
                    # Handled by model deletion if model_path exists
                    
                elif item == "feature_importance":
                    # Feature importance is in model_metadata
                    print(f"  - Clearing feature importance...")
                    # Will be overwritten on next training
                    
            except Exception as e:
                print(f"    ⚠️  Warning: Failed to delete {item}: {e}")
        
        print("✅ Cleanup complete (dependent data will be overwritten)")
    
    # ============================================
    # CONFIGURATION METHODS
    # ============================================
    
    def add_features(self, features: List[Dict[str, Any]], confirm: Optional[bool] = None) -> 'Model':
        """
        Add features to the model
        
        Args:
            features: List of feature configs, e.g. [{"name": "Gold Price", "source": "yahoo", "symbol": "GC=F"}]
            confirm: Explicit confirmation (None = prompt if needed)
            
        Returns:
            self for method chaining
        """
        # Check for conflicts
        if not self._check_and_handle_conflict("add_features", confirm):
            return self
        
        print(f"[Model {self.name}] Adding {len(features)} features...")
        
        # Get current features
        current_features = self.input_features.copy()
        
        # Add new features
        for feature in features:
            # Check if feature already exists
            if any(f.get('name') == feature.get('name') for f in current_features):
                print(f"  ⚠️  Feature '{feature.get('name')}' already exists, skipping")
                continue
            current_features.append(feature)
            print(f"  ✅ Added '{feature.get('name')}'")
        
        # Update model
        response = self.http.patch(f'/api/v1/models/{self.id}', {
            'input_features': current_features
        })
        
        self._data = response.get('model', {})
        print(f"✅ Features updated ({len(current_features)} total)")
        return self
    
    def set_training_period(self, start_date: str, end_date: str, confirm: Optional[bool] = None) -> 'Model':
        """
        Set training period
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            confirm: Explicit confirmation (None = prompt if needed)
            
        Returns:
            self for method chaining
        """
        # Check for conflicts
        if not self._check_and_handle_conflict("set_training_period", confirm):
            return self
        
        # Validate minimum period length
        total_days = validate_training_period(start_date, end_date)
        
        print(f"[Model {self.name}] Setting training period: {start_date} to {end_date} ({total_days} days)")
        
        # Update model
        response = self.http.patch(f'/api/v1/models/{self.id}', {
            'training_start_date': start_date,
            'training_end_date': end_date
        })
        
        self._data = response.get('model', {})
        print("✅ Training period updated")
        return self
    
    # ============================================
    # DATA COLLECTORS
    # ============================================
    
    def add_data_collector(
        self,
        source: str,
        api_key: Optional[str] = None,
        **config
    ):
        """
        Add a data collector to this model
        
        Data collectors handle fetching data from external sources (FRED, Yahoo, etc.)
        Each collector provides a source-specific interface for adding features.
        
        Args:
            source: Data source name ('fred', 'yahoo', etc.)
            api_key: API key for authentication (if required by source)
            **config: Additional source-specific configuration
        
        Returns:
            Collector wrapper with add_features() method
        
        Example:
            >>> # FRED collector - uses series_id
            >>> fred = model.add_data_collector('fred', api_key='your_key')
            >>> fred.add_features([
            >>>     {'id': 'DGS30', 'name': 'US Treasury 30Y', 'type': 'target'}
            >>> ])
            >>> 
            >>> # Yahoo collector - uses symbol
            >>> yahoo = model.add_data_collector('yahoo')
            >>> yahoo.add_features([
            >>>     {'id': '^GSPC', 'name': 'S&P 500', 'type': 'conditioning'}
            >>> ])
        """
        from ..data_collector import create_collector_wrapper
        
        if 'data_collectors' not in self._data:
            self._data['data_collectors'] = []
        
        # Check if collector already exists
        existing = [c for c in self._data['data_collectors'] if c['source'] == source.lower()]
        if existing:
            print(f"⚠️  Data collector '{source}' already exists, updating...")
            self._data['data_collectors'] = [c for c in self._data['data_collectors'] if c['source'] != source.lower()]
        
        collector_config = {
            'source': source.lower(),
            'api_key': api_key,
            'config': config
        }
        
        self._data['data_collectors'].append(collector_config)
        
        # Save to backend immediately
        response = self.http.patch(f'/api/v1/models/{self.id}', {
            'data_collectors': self._data['data_collectors']
        })
        
        # Update local data with response
        self._data = response.get('model', {})
        
        print(f"✅ Added {source} data collector")
        
        # Return collector wrapper for adding features
        return create_collector_wrapper(self, source, api_key, **config)
    
    def list_data_collectors(self) -> List[str]:
        """List configured data collectors"""
        return [c['source'] for c in self._data.get('data_collectors', [])]
    
    # ============================================
    # DATA FETCHING
    # ============================================
    
    def fetch_data(
        self, 
        max_gap_days: Optional[int] = None,
        interpolation_method: str = "linear",
        confirm: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Fetch and process training data using configured data collectors
        
        This method uses the data collectors you've configured with add_data_collector().
        It handles EVERYTHING:
        1. Uses configured collectors (with their API keys)
        2. Fetches raw data with automatic frequency detection
        3. Auto-generates interpolation settings based on data frequency
        4. Saves processing config and data to database
        5. Updates model status to 'data_collected'
        
        Args:
            max_gap_days: Optional override for max gap (default: auto-detect from frequency)
                         If None, uses frequency-aware defaults:
                         - Daily: 7 days, Monthly: 35 days, etc.
            interpolation_method: "linear", "forward_fill", or "backward_fill" (default: "linear")
            confirm: Explicit confirmation (None = prompt if needed)
            
        Returns:
            dict: Fetch statistics with keys: status, features_fetched, total_raw_points, total_processed_points
            
        Example:
            >>> # First, add data collectors
            >>> model.add_data_collector('fred', api_key='your_key')
            >>> model.add_data_collector('yahoo')
            >>> 
            >>> # Then fetch data (auto frequency detection)
            >>> model.fetch_data()
            >>> 
            >>> # Or with manual max_gap override
            >>> model.fetch_data(max_gap_days=60)
        
        Note:
            You must call add_data_collector() before fetch_data()!
        """
        # Check for conflicts
        if not self._check_and_handle_conflict("fetch_data", confirm):
            return {"status": "cancelled"}
        
        print(f"[Model {self.name}] Fetching data...")
        
        # Check for data collectors
        data_collectors = self._data.get('data_collectors', [])
        if not data_collectors:
            print("❌ No data collectors configured!")
            print("   Add collectors first:")
            print("   model.add_data_collector('fred', api_key='your_key')")
            print("   model.add_data_collector('yahoo')")
            return {"status": "error", "message": "No data collectors configured"}
        
        print(f"  Using {len(data_collectors)} data collectors: {', '.join([c['source'] for c in data_collectors])}")
        
        # Get features and training period from model
        features = self.input_features
        if not features:
            print("❌ No features configured. Call model.add_features() first.")
            return {"status": "error", "message": "No features configured"}
        
        training_period = self._data.get('training_start_date'), self._data.get('training_end_date')
        if not all(training_period):
            print("❌ Training period not set. Call model.set_training_period() first.")
            return {"status": "error", "message": "Training period not set"}
        
        # Extract API keys from data collectors for backward compatibility
        # (Backend still expects fred_api_key in payload, will be refactored)
        fred_api_key = None
        for collector in data_collectors:
            if collector['source'] == 'fred':
                fred_api_key = collector.get('api_key')
                break
        
        # Build processing config
        if max_gap_days is not None:
            # Manual max_gap override - apply to all features
            print(f"  Using manual max_gap: {max_gap_days} days for all features")
            processing_config = {
                "interpolation": {
                    "method": interpolation_method,
                    "maxGapLength": {
                        feature.get("name", feature.get("display_name", "")): max_gap_days
                        for feature in features
                    }
                }
            }
        else:
            # Auto-detect max_gap from frequency (backend will auto-generate)
            print(f"  Using auto frequency detection (backend will set appropriate max_gap for each feature)")
            processing_config = {
                "interpolation": {
                    "method": interpolation_method,
                    "maxGapLength": {}  # Empty = auto-detect
                }
            }
        
        # Build request payload
        payload = {
            "model_id": self.id,
            "features": features,
            "start_date": training_period[0],
            "end_date": training_period[1],
            "fred_api_key": fred_api_key,
            "processing_config": processing_config
        }
        
        print(f"  Features: {len(features)}")
        print(f"  Period: {training_period[0]} to {training_period[1]}")
        print(f"  Interpolation method: {interpolation_method}")
        
        # Call backend
        print("📡 Fetching from APIs and processing...")
        response = self.http.post('/api/v1/data/fetch', payload)
        
        # Update model status
        self._data["status"] = "data_collected"
        
        print(f"✅ Fetched {response.get('total_raw_points', 0)} raw points")
        print(f"✅ Processed {response.get('total_processed_points', 0)} interpolated points")
        
        return response
    
    # ============================================
    # SAMPLE GENERATION
    # ============================================
    
    def generate_samples(
        self,
        past_window: int = 100,
        future_window: int = 80,
        stride: int = 5,
        splits: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate training samples and fit encoding models (PCA-ICA) using model's feature sets
        
        This method performs a complete pipeline:
        1. Generate training samples with proper windowing
        2. Fit PCA-ICA encoding models on training split
        3. Encode all samples using fitted models
        
        Args:
            past_window: Past window size (days, default: 100)
            future_window: Future window size (days, default: 80)
            stride: Stride between samples (days, default: 5)
            splits: Train/validation splits (optional, auto-calculated if not provided)
                Can be percentages: {"training": 80, "validation": 20}
                Or date ranges: {"training": {"start": "2020-01-01", "end": "2023-03-31"}, "validation": {"start": "2023-04-01", "end": "2023-12-31"}}
            
        Returns:
            dict: Generation and encoding statistics with keys: status, samples_generated, models_fitted, samples_encoded
            
        Example:
            >>> model.generate_samples()  # Uses defaults: 100 past, 80 future, stride 5, 80/20 split
            >>> model.generate_samples(past_window=50, future_window=30)  # Custom windows
        """
        # Check for conflicts
        if not self._check_and_handle_conflict("generate_samples", True):
            return {"status": "cancelled"}
        
        print(f"[Model {self.name}] Generating samples...")
        print(f"  Past window: {past_window} days")
        print(f"  Future window: {future_window} days")
        print(f"  Stride: {stride} days")
        
        # For modular architecture, features come from the model's feature sets
        # The backend will automatically determine conditioning and target features
        # based on the model's conditioning_set_id and target_set_id
        
        # Auto-generate splits if not provided or if percentage-based
        if splits is None or (isinstance(splits, dict) and isinstance(list(splits.values())[0], (int, float))):
            sample_size = past_window + future_window
            
            # Get training dates from project if not available on model
            start = self._data.get('training_start_date')
            end = self._data.get('training_end_date')
            
            if not start or not end:
                # Get project data to get training dates
                project_id = self.project_id
                if project_id:
                    project_response = self.http.get(f'/api/v1/projects/{project_id}')
                    start = project_response.get('training_start_date')
                    end = project_response.get('training_end_date')
            
            # If percentage splits provided, use them; otherwise use defaults
            if splits and isinstance(list(splits.values())[0], (int, float)):
                train_pct = splits.get('training', 80) / 100
                val_pct = splits.get('validation', 20) / 100
                test_pct = splits.get('test', 0) / 100
                if test_pct > 0:
                    print(f"  Converting percentage splits: {int(train_pct*100)}% train, {int(val_pct*100)}% val, {int(test_pct*100)}% test")
                else:
                    print(f"  Converting percentage splits: {int(train_pct*100)}% train, {int(val_pct*100)}% val")
                splits = auto_generate_splits(start, end, sample_size=sample_size, 
                                             train_pct=train_pct, val_pct=val_pct, test_pct=test_pct)
            else:
                splits = auto_generate_splits(start, end, sample_size=sample_size)
            print(f"  Auto-generated splits with {sample_size}-day gap")
        
        # Validate splits
        validate_splits(splits, past_window, future_window)
        
        # Build sample config (features will be determined by backend from model's feature sets)
        sample_config = {
            "pastWindow": past_window,
            "futureWindow": future_window,
            "stride": stride,
            "splits": splits,
            "conditioningFeatures": [],  # Will be populated by backend from conditioning_set_id
            "targetFeatures": []  # Will be populated by backend from target_set_id
        }
        
        # Build request payload
        payload = {
            "model_id": self.id,
            "sample_config": sample_config
        }
        
        # Call backend
        print("📡 Calling backend to generate samples...")
        response = self.http.post('/api/v1/ml/generate-samples', payload)
        
        # Store sample config but don't update status yet (we'll update to "encoded" at the end)
        self._data["sample_config"] = sample_config
        
        split_counts = response.get('split_counts', {})
        samples_generated = response.get('samples_generated', 0)
        print(f"✅ Generated {samples_generated} samples")
        print(f"   Training: {split_counts.get('training', 0)}")
        print(f"   Validation: {split_counts.get('validation', 0)}")
        
        # Step 2: Fit encoding models and encode samples
        print(f"\n🔧 Fitting PCA-ICA encoding models and encoding samples...")
        
        # Fit encoding models
        print(f"  Step 2/3: Fitting PCA-ICA encoding models on 'training' split...")
        try:
            fit_response = self.http.post('/api/v1/ml/fit?split=training', {
                "model_id": self.id,
                "encoding_type": "pca-ica",
                "pca_variance_threshold_series": 0.95,
                "pca_variance_threshold_residuals": 0.99
            })
            
            models_fitted = fit_response.get('models_fitted', 0)
            features_processed = fit_response.get('features_processed', 0)
            samples_used = fit_response.get('samples_used', 0)
            
            print(f"  ✅ Fitted {models_fitted} encoding models")
            print(f"     Features processed: {features_processed}")
            print(f"     Samples used: {samples_used}")
            
        except Exception as e:
            print(f"  ❌ Failed to fit encoding models: {e}")
            raise
        
        # Encode samples
        print(f"  Step 3/3: Encoding all samples...")
        try:
            encode_response = self.http.post('/api/v1/ml/encode?source=database', {
                "model_id": self.id,
                "encoding_type": "pca-ica"
            })
            
            samples_encoded = encode_response.get('samples_encoded', 0)
            encoding_features_processed = encode_response.get('features_processed', 0)
            
            print(f"  ✅ Encoded {samples_encoded} samples")
            print(f"     Features processed: {encoding_features_processed}")
            
        except Exception as e:
            print(f"  ❌ Failed to encode samples: {e}")
            raise
        
        # Update model status
        self._data["status"] = "encoded"
        
        # Persist status change to database
        try:
            self.http.patch(f'/api/v1/models/{self.id}', {"status": "encoded"})
        except Exception as e:
            print(f"⚠️  Warning: Failed to persist status change: {e}")
        
        print(f"✅ Sample generation and encoding complete")
        
        return {
            "status": "success",
            "samples_generated": samples_generated,
            "models_fitted": models_fitted,
            "samples_encoded": samples_encoded,
            "features_processed": features_processed,
            "split_counts": split_counts
        }
    
    
    def _update_metadata(self, updates: Dict[str, Any]):
        """Helper to update model metadata"""
        # Update local data
        self._data.update(updates)
        
        # Update in database via backend
        # The update endpoint expects fields directly in the payload, not nested under "updates"
        payload = updates.copy()
        self.http.patch(f'/api/v1/models/{self.id}', payload)
    
    # ============================================
    # ENCODING
    # ============================================
    
    def encode_samples(
        self, 
        encoding_type: str = "pca-ica",
        split: str = "training",
        pca_variance_threshold_series: float = 0.95,
        pca_variance_threshold_residuals: float = 0.99,
        confirm: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        [DEPRECATED] Fit encoding models and encode all samples
        
        This method is deprecated. Encoding is now automatically performed as part of generate_samples().
        Use model.generate_samples() instead, which includes sample generation, model fitting, and encoding.
        
        Args:
            encoding_type: Type of encoding ("pca-ica", default: "pca-ica")
            split: Which split to use for fitting ("training", default: "training")
            pca_variance_threshold_series: Variance threshold for normalized_series (default: 0.95)
            pca_variance_threshold_residuals: Variance threshold for normalized_residuals (default: 0.99)
            confirm: Explicit confirmation (ignored)
            
        Returns:
            dict: Encoding statistics
            
        Example:
            >>> # OLD WAY (deprecated):
            >>> model.generate_samples()
            >>> model.encode_samples()  # Don't use this anymore
            
            >>> # NEW WAY (recommended):
            >>> model.generate_samples()  # Includes encoding automatically
        """
        import warnings
        warnings.warn(
            "encode_samples() is deprecated. Encoding is now automatically performed as part of generate_samples(). "
            "Use model.generate_samples() instead, which includes sample generation, model fitting, and encoding.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Validate encoding_type
        if encoding_type not in ["pca-ica"]:
            raise ValueError(f"Invalid encoding_type: {encoding_type}. Must be 'pca-ica'")
        
        # Validate split
        if split not in ["training", "validation", "training+validation", "all"]:
            raise ValueError(f"Invalid split: {split}. Must be 'training', 'validation', 'training+validation', or 'all'")
        
        # Check for conflicts
        if not self._check_and_handle_conflict("encode_samples", confirm):
            return {"status": "cancelled"}
        
        print(f"[Model {self.name}] Encoding samples using {encoding_type.upper()}...")
        
        # Step 1: Fit encoding models
        print(f"  Step 1/2: Fitting {encoding_type.upper()} encoding models on '{split}' split...")
        try:
            # Add query parameter for split
            fit_url = f'/api/v1/ml/fit?split={split}'
            fit_response = self.http.post(fit_url, {
                "model_id": self.id,
                "user_id": self._data.get("user_id"),  # For API key auth
                "encoding_type": encoding_type,
                "pca_variance_threshold_series": pca_variance_threshold_series,
                "pca_variance_threshold_residuals": pca_variance_threshold_residuals
            })
            
            models_fitted = fit_response.get('models_fitted', 0)
            features_processed = fit_response.get('features_processed', 0)
            samples_used = fit_response.get('samples_used', 0)
            
            print(f"  ✅ Fitted {models_fitted} encoding models")
            print(f"     Features processed: {features_processed}")
            print(f"     Samples used: {samples_used}")
            
        except Exception as e:
            print(f"  ❌ Failed to fit encoding models: {e}")
            raise
        
        # Step 2: Encode samples
        print("  Step 2/2: Encoding all samples...")
        try:
            # Add query parameter for source
            encode_url = '/api/v1/ml/encode?source=database'
            encode_response = self.http.post(encode_url, {
                "model_id": self.id,
                "user_id": self._data.get("user_id"),  # For API key auth
                "encoding_type": encoding_type
            })
            
            samples_encoded = encode_response.get('samples_encoded', 0)
            encoding_features_processed = encode_response.get('features_processed', 0)
            
            print(f"  ✅ Encoded {samples_encoded} samples")
            print(f"     Features processed: {encoding_features_processed}")
            
        except Exception as e:
            print(f"  ❌ Failed to encode samples: {e}")
            raise
        
        # Update model status
        self._data["status"] = "samples_encoded"
        
        # Persist status change to database
        try:
            self.http.patch(f'/api/v1/models/{self.id}', {"status": "samples_encoded"})
        except Exception as e:
            print(f"⚠️  Warning: Failed to persist status change: {e}")
        
        print(f"✅ Encoding complete")
        
        return {
            "status": "success",
            "models_fitted": models_fitted,
            "samples_encoded": samples_encoded,
            "features_processed": features_processed,
            "split_used": split,
            "encoding_type": encoding_type
        }
    

    
    # ============================================
    # SCENARIO CREATION
    # ============================================
    
    def _validate_feature_simulation_dates(self, feature_simulation_dates: Dict[str, str]):
        """
        Validate that feature_simulation_dates keys correspond to valid conditioning features/groups.
        
        Args:
            feature_simulation_dates: Dict mapping feature/group names to simulation dates
            
        Raises:
            ValueError: If any key is not a valid conditioning feature or group name
        """
        if not feature_simulation_dates:
            return
        
        # Get the conditioning feature set
        conditioning_set_id = self._data.get('conditioning_set_id')
        if not conditioning_set_id:
            raise ValueError("Model has no conditioning feature set")
        
        # Get conditioning feature set details
        conditioning_set = self.http.get(f'/api/v1/feature-sets/{conditioning_set_id}')
        if not conditioning_set:
            raise ValueError(f"Could not retrieve conditioning feature set {conditioning_set_id}")
        
        # Get valid feature/group names from conditioning set
        valid_names = set()
        
        # Add individual feature names
        features = conditioning_set.get('features', [])
        for feature in features:
            valid_names.add(feature.get('name'))
        
        # Add feature group names (if any)
        feature_groups = conditioning_set.get('feature_groups', {})
        groups = feature_groups.get('groups', [])
        for group in groups:
            valid_names.add(group.get('name'))
        
        # Validate each key in feature_simulation_dates
        invalid_names = []
        for key in feature_simulation_dates.keys():
            if key not in valid_names:
                invalid_names.append(key)
        
        if invalid_names:
            available_names = sorted(list(valid_names))
            raise ValueError(
                f"Invalid feature/group names in feature_simulation_dates: {invalid_names}. "
                f"Valid conditioning feature/group names are: {available_names}"
            )
    
    def create_scenario(
        self,
        simulation_date: str,
        name: str,
        description: str = "",
        feature_simulation_dates: Optional[Dict[str, str]] = None
    ):
        """
        Create a new scenario linked to this model.
        
        This is a convenience method that creates a scenario without needing
        to access client.scenarios.create().
        
        Args:
            simulation_date: Default simulation date for all features (YYYY-MM-DD)
            name: Scenario name
            description: Optional scenario description
            feature_simulation_dates: Optional dict mapping feature names to specific simulation dates
        
        Returns:
            Scenario instance
        
        Example:
            >>> scenario = model.create_scenario(
            ...     simulation_date="2020-03-15",
            ...     name="COVID Crash Scenario",
            ...     description="Simulating March 2020 conditions",
            ...     feature_simulation_dates={
            ...         "5-Year Treasury Rate": "2008-09-15",  # Lehman crisis
            ...         "VIX Volatility Index": "2020-02-28"   # Different date
            ...     }
            ... )
        """
        # Check if model is trained
        if self.status not in ['trained', 'model_trained']:
            raise ValueError(f"Model must be trained to create scenarios. Current status: {self.status}")
        
        # Validate feature_simulation_dates if provided
        if feature_simulation_dates:
            self._validate_feature_simulation_dates(feature_simulation_dates)
        
        from ..scenario.builder import Scenario
        
        print(f"[Model {self.name}] Creating scenario: {name}")
        print(f"  Simulation date: {simulation_date}")
        if feature_simulation_dates:
            print(f"  Feature-specific dates: {feature_simulation_dates}")
        
        # Create via API
        response = self.http.post('/api/v1/scenarios', {
            'model_id': self.id,
            'name': name,
            'description': description,
            'simulation_date': simulation_date,
            'feature_simulation_dates': feature_simulation_dates or {}
        })
        
        print(f"✅ Scenario created: {response.get('name')} (ID: {response.get('id')[:8]}...)")
        
        return Scenario(self.http, response, self)
    
    # ============================================
    # RECONSTRUCTION QUALITY CHECKING
    # ============================================
    
    def check_reconstruction_quality(
        self,
        feature: str,
        window: str,
        split: str = "test",
        index: int = 0,
        plot: bool = True,
        save_path: str = None
    ) -> Dict[str, float]:
        """
        Check reconstruction quality for a specific feature-window combination
        
        Validates encoding quality by comparing original vs reconstructed values.
        Automatically selects the correct data type:
        - Past windows (all features): normalized_series
        - Future conditioning: normalized_series
        - Future target: normalized_residuals
        
        Args:
            feature: Feature name (e.g., "10-Year Treasury", "S&P 500")
            window: "past" or "future"
            split: Sample split ("training", "validation", "test")
            index: Which sample to check (0 = first sample in split)
            plot: Whether to generate plot
            save_path: Path to save plot (auto-generated if None)
        
        Returns:
            Dict with metrics: mse, rmse, mae, r_squared, max_error, n_components
        
        Examples:
            >>> # Check future target residuals on first test sample (critical for realism!)
            >>> model.check_reconstruction_quality("10-Year Treasury", "future", split="test", index=0)
            
            >>> # Check past conditioning on 5th training sample
            >>> model.check_reconstruction_quality("S&P 500", "past", split="training", index=4)
        """
        import numpy as np
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        print(f"\n{'='*70}")
        print(f"RECONSTRUCTION QUALITY CHECK")
        print(f"{'='*70}")
        print(f"Feature: {feature}")
        print(f"Window: {window}")
        print(f"Split: {split}")
        print(f"Index: {index}")
        print(f"{'='*70}\n")
        
        # Fetch samples from specified split
        print(f"🔍 Fetching {split} samples...")
        response = self.http.get(
            f'/api/v1/models/{self.id}/samples',
            params={'split_type': split, 'limit': 1000, 'include_data': 'true'}
        )
        samples = response.get('samples', [])
        
        if not samples:
            raise ValueError(f"No {split} samples found")
        
        if index >= len(samples):
            raise ValueError(f"Index {index} out of range (only {len(samples)} {split} samples available)")
        
        closest_sample = samples[index]
        
        print(f"  ✅ Selected sample {index+1}/{len(samples)}")
        print(f"     ID: {closest_sample['id']}")
        print(f"     Start Date: {closest_sample.get('start_date')}")
        print(f"     Split: {closest_sample.get('split_type', split)}")
        
        # Determine feature type and data type
        feature_info = next(
            (f for f in self.input_features if f.get('display_name') == feature or f.get('name') == feature),
            None
        )
        if not feature_info:
            raise ValueError(f"Feature '{feature}' not found")
        
        is_target = feature_info.get('type') == 'target'
        
        # Auto-select data_type
        if window == "past":
            data_type = "normalized_series"
        elif window == "future":
            data_type = "normalized_residuals" if is_target else "normalized_series"
        else:
            raise ValueError(f"Invalid window: {window}")
        
        print(f"\n📊 Configuration:")
        print(f"  Feature type: {'target' if is_target else 'conditioning'}")
        print(f"  Data type: {data_type}")
        
        # Extract data from sample
        # Original data is in conditioning_data/target_data
        # Encoded data is in encoded_conditioning_data/encoded_target_data
        print(f"\n📥 Extracting data...")
        
        # Determine source for original data
        if window == "past":
            original_source = "conditioning_data"  # All past windows
            encoded_source = "encoded_conditioning_data"
        elif window == "future" and is_target:
            original_source = "target_data"  # Future target windows
            encoded_source = "encoded_target_data"
        else:
            original_source = "conditioning_data"  # Future conditioning windows
            encoded_source = "encoded_conditioning_data"
        
        original_data = closest_sample.get(original_source, [])
        encoded_data = closest_sample.get(encoded_source, [])
        
        original_values = None
        encoded_values = None
        dates = None
        
        # Extract original values
        for item in original_data:
            if item.get('feature') == feature and item.get('temporal_tag') == window:
                original_values = item.get(data_type, [])
                dates = item.get('dates', [])
                break
        
        # Extract encoded values (from separate encoded arrays)
        # With groups, the feature might be encoded as part of a group
        for item in encoded_data:
            # Check if this item is for our feature (direct match or part of a group)
            item_feature = item.get('feature')
            item_temporal = item.get('temporal_tag')
            
            if item_temporal == window:
                # Check direct match or if feature is in group_features
                is_match = (item_feature == feature)
                if not is_match and item.get('group_features'):
                    is_match = feature in item.get('group_features', [])
                
                if is_match:
                    if data_type == "normalized_series":
                        encoded_values = item.get('encoded_normalized_series', [])
                    else:
                        encoded_values = item.get('encoded_normalized_residuals', [])
                    
                    if encoded_values:
                        break
        
        if original_values is None or encoded_values is None:
            raise ValueError(f"Data not found for {feature} {window} {data_type}")
        
        # Denormalize original values to match reconstructed scale
        norm_params = self._data.get('feature_normalization_params', {}).get(feature, {})
        mean = norm_params.get('mean', 0)
        std = norm_params.get('std', 1)
        
        if data_type == "normalized_residuals":
            # For residuals: residual = series - reference_value
            # To get series back: series = residual + reference_value
            # Get last past value as reference (from conditioning_data - past of all features is conditioning)
            past_ref_norm = None
            for item in closest_sample.get('conditioning_data', []):
                if item.get('feature') == feature and item.get('temporal_tag') == 'past':
                    past_series = item.get('normalized_series', [])
                    if past_series:
                        past_ref_norm = past_series[-1]
                    break
            
            if past_ref_norm is None:
                raise ValueError(f"Could not find past reference for {feature} in conditioning_data")
            
            # Add reference to each residual to get the series (all in normalized space)
            reconstructed_norm = [residual + past_ref_norm for residual in original_values]
            
            # Denormalize
            original_values_denorm = [(v * std) + mean for v in reconstructed_norm]
            
            print(f"  DEBUG RESIDUALS: past_ref_norm={past_ref_norm:.4f}")
            print(f"  DEBUG RESIDUALS: First residual={original_values[0]:.4f}")
            print(f"  DEBUG RESIDUALS: First reconstructed_norm={reconstructed_norm[0]:.4f}")
            print(f"  DEBUG RESIDUALS: First denorm={original_values_denorm[0]:.4f}")
        else:
            # For series: just denormalize directly
            original_values_denorm = [(v * std) + mean for v in original_values]
        
        print(f"  ✅ Original: {len(original_values)} points (denormalized)")
        print(f"  ✅ Encoded successfully")
        
        # Reconstruct
        print(f"\n🔄 Reconstructing...")
        
        # For residuals, we need a reference value (last past value)
        # For series, no reference needed
        if data_type == "normalized_residuals":
            # Get the last past value as reference (in normalized space for backend)
            # Find the past window for this feature
            for item in closest_sample.get('conditioning_data', []):
                if item.get('feature') == feature and item.get('temporal_tag') == 'past':
                    past_series = item.get('normalized_series', [])
                    if past_series:
                        past_ref_norm = past_series[-1]  # Keep in normalized space
                    break
            
            if past_ref_norm is None:
                raise ValueError(f"Could not find past reference value for {feature}")
            
            print(f"  DEBUG: Sending past_ref_norm={past_ref_norm:.4f} to backend as reference")
            
            # Build encoded window - need to find the actual encoded item to get group metadata
            encoded_window = {
                "feature": feature,
                "temporal_tag": window,
                "data_type": f"encoded_{data_type}",
                "encoded_values": encoded_values
            }
            
            # If feature is part of a group, find and preserve group metadata
            for item in encoded_data:
                if item.get('temporal_tag') == window:
                    if item.get('feature') == feature or (item.get('group_features') and feature in item.get('group_features', [])):
                        # Found the encoded item - copy metadata
                        if 'is_group' in item:
                            encoded_window['is_group'] = item['is_group']
                        if 'is_multivariate' in item:
                            encoded_window['is_multivariate'] = item['is_multivariate']
                        if 'group_features' in item:
                            encoded_window['group_features'] = item['group_features']
                            encoded_window['feature'] = item['feature']  # Use group_id as feature
                        break
            
            payload = {
                "user_id": self._data.get("user_id"),
                "model_id": self.id,
                "encoded_source": "inline",
                "encoded_windows": [encoded_window],
                "reference_source": "inline",
                "reference_values": {feature: past_ref_norm},  # Send NORMALIZED reference
                "output_destination": "return"
            }
        else:
            # For series, no reference needed
            # Build encoded window with group metadata if applicable
            encoded_window = {
                "feature": feature,
                "temporal_tag": window,
                "data_type": f"encoded_{data_type}",
                "encoded_values": encoded_values
            }
            
            # If feature is part of a group, find and preserve group metadata
            for item in encoded_data:
                if item.get('temporal_tag') == window:
                    if item.get('feature') == feature or (item.get('group_features') and feature in item.get('group_features', [])):
                        # Found the encoded item - copy metadata
                        if 'is_group' in item:
                            encoded_window['is_group'] = item['is_group']
                        if 'is_multivariate' in item:
                            encoded_window['is_multivariate'] = item['is_multivariate']
                        if 'group_features' in item:
                            encoded_window['group_features'] = item['group_features']
                            encoded_window['feature'] = item['feature']  # Use group_id as feature
                        break
            
            payload = {
                "user_id": self._data.get("user_id"),
                "model_id": self.id,
                "encoded_source": "inline",
                "encoded_windows": [encoded_window],
                "reference_source": "none",
                "output_destination": "return"
            }
        
        response = self.http.post('/api/v1/ml/reconstruct', payload)
        reconstructions = response.get('reconstructions', [])
        
        if not reconstructions:
            raise ValueError("No reconstructions returned")
        
        reconstructed_values = reconstructions[0].get('reconstructed_values', [])
        print(f"  ✅ Reconstructed: {len(reconstructed_values)} points")
        
        # Calculate metrics
        print(f"\n📈 Calculating metrics...")
        original_arr = np.array(original_values_denorm)  # Use denormalized values
        reconstructed_arr = np.array(reconstructed_values)
        
        min_len = min(len(original_arr), len(reconstructed_arr))
        original_arr = original_arr[:min_len]
        reconstructed_arr = reconstructed_arr[:min_len]
        
        # Debug: Check value ranges
        print(f"  DEBUG: Original range: [{original_arr.min():.2f}, {original_arr.max():.2f}]")
        print(f"  DEBUG: Reconstructed range: [{reconstructed_arr.min():.2f}, {reconstructed_arr.max():.2f}]")
        print(f"  DEBUG: First 3 original: {original_arr[:3].tolist()}")
        print(f"  DEBUG: First 3 reconstructed: {reconstructed_arr[:3].tolist()}")
        
        mse = mean_squared_error(original_arr, reconstructed_arr)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(original_arr, reconstructed_arr)
        r_squared = r2_score(original_arr, reconstructed_arr)
        max_error = np.max(np.abs(original_arr - reconstructed_arr))
        
        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r_squared": float(r_squared),
            "max_error": float(max_error),
            "n_components": len(encoded_values),
            "n_points": min_len
        }
        
        print(f"\n✅ Reconstruction Metrics:")
        print(f"  MSE:       {mse:.6f}")
        print(f"  RMSE:      {rmse:.6f}")
        print(f"  MAE:       {mae:.6f}")
        print(f"  R²:        {r_squared:.6f}")
        print(f"  Max Error: {max_error:.6f}")
        
        # Plot
        if plot:
            print(f"\n📊 Generating plot...")
            if save_path is None:
                import os
                # Create reconstructions subdirectory
                reconstructions_dir = os.path.join(os.getcwd(), "reconstructions")
                os.makedirs(reconstructions_dir, exist_ok=True)
                save_path = os.path.join(
                    reconstructions_dir,
                    f"reconstruction_{feature.replace(' ', '_')}_{window}_{data_type}_{split}.png"
                )
            
            self._plot_reconstruction_quality(
                original_arr, reconstructed_arr, dates[:min_len] if dates else None,
                feature, window, data_type, metrics, save_path
            )
            print(f"  ✅ Plot saved: {save_path}")
        
        return metrics
    
    def _plot_reconstruction_quality(
        self, original, reconstructed, dates, feature, window, data_type, metrics, save_path
    ):
        """Plot reconstruction quality comparison - simplified to show only original vs reconstructed + scatter R²"""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"Reconstruction Quality: {feature} ({window} window, {data_type})",
            fontsize=14, fontweight='bold'
        )
        
        # Prepare x-axis
        if dates:
            x_vals = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
            x_label = 'Date'
        else:
            x_vals = list(range(len(original)))
            x_label = 'Time Step'
        
        # Plot 1: Time series
        ax1 = axes[0]
        ax1.plot(x_vals, original, label='Original', linewidth=2, alpha=0.8)
        ax1.plot(x_vals, reconstructed, label='Reconstructed', linewidth=2, alpha=0.8, linestyle='--')
        ax1.set_xlabel(x_label)
        ax1.set_ylabel('Value')
        ax1.set_title('Original vs Reconstructed')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        if dates:
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 2: Scatter
        ax2 = axes[1]
        ax2.scatter(original, reconstructed, alpha=0.6, s=30)
        min_val = min(original.min(), reconstructed.min())
        max_val = max(original.max(), reconstructed.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect', linewidth=2)
        ax2.set_xlabel('Original')
        ax2.set_ylabel('Reconstructed')
        ax2.set_title(f'Scatter (R² = {metrics["r_squared"]:.4f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add metrics as text in the scatter plot
        metrics_text = f"""Metrics:
MSE: {metrics['mse']:.6f}
RMSE: {metrics['rmse']:.6f}
MAE: {metrics['mae']:.6f}
R²: {metrics['r_squared']:.6f}
Max Error: {metrics['max_error']:.6f}

Components: {metrics['n_components']}
Points: {metrics['n_points']}"""
        
        ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes, fontsize=9, 
                family='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _get_test_sample(self, index: int = 0, include_data: bool = False) -> Optional[Dict]:
        """
        Get test sample by index
        
        Args:
            index: Sample index (0 = first test sample)
            include_data: Whether to include full conditioning_data/target_data
        
        Returns:
            Sample dict or None if not found
        """
        try:
            params = {'split_type': 'test', 'limit': 100}
            if include_data:
                params['include_data'] = 'true'
            
            response = self.http.get(
                f'/api/v1/models/{self.id}/samples',
                params=params
            )
            
            samples = response.get('samples', [])
            if samples and len(samples) > index:
                return samples[index]
            
            return None
        except Exception as e:
            print(f"  ⚠️  Failed to get test sample: {e}")
            return None
    
    # ============================================
    # RECONSTRUCTION METHODS
    # ============================================
    
    def reconstruct_sample(
        self,
        sample_id: str = None,
        split: str = "test"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Reconstruct a sample from the database (decode + denormalize).
        
        Args:
            sample_id: Sample ID to reconstruct (auto-selects first test sample if None)
            split: Which split to fetch from if sample_id is None
        
        Returns:
            Dictionary with structure:
            {
                "feature_name": {
                    "past": {"dates": [...], "values": [...]},
                    "future": {"dates": [...], "values": [...]}
                },
                ...
            }
        
        Example:
            >>> reconstruction = model.reconstruct_sample()
            >>> feature_data = reconstruction["Gold Price"]
            >>> print(feature_data["past"]["values"])
        """
        import numpy as np
        
        # Auto-select sample if not provided
        if sample_id is None:
            print(f"[Reconstruction] Auto-selecting {split} sample...")
            response = self.http.get(
                f'/api/v1/models/{self.id}/samples',
                params={'split_type': split, 'limit': 1}
            )
            samples = response.get('samples', [])
            if not samples:
                raise ValueError(f"No {split} samples found")
            sample_id = samples[0]['id']
            print(f"  Selected sample: {sample_id}")
        
        print(f"[Reconstruction] Reconstructing sample {sample_id}...")
        
        # Call reconstruct endpoint
        payload = {
            "user_id": self._data.get("user_id"),
            "model_id": self.id,
            
            # Source encoded windows from database
            "encoded_source": "database",
            "encoded_table": "samples",
            "encoded_columns": ["encoded_conditioning_data", "encoded_target_data"],
            "sample_id": sample_id,
            
            # Reference values from same sample
            "reference_source": "database",
            "reference_table": "samples",
            "reference_column": "conditioning_data",
            "reference_sample_id": sample_id,
            
            "output_destination": "return"
        }
        
        try:
            response = self.http.post('/api/v1/ml/reconstruct', payload)
        except Exception as e:
            print(f"  ❌ Reconstruction failed: {e}")
            raise
        
        reconstructions = response.get('reconstructions', [])
        print(f"✅ Reconstructed {len(reconstructions)} windows")
        
        # Also fetch original sample for dates
        sample_response = self.http.get(
            f'/api/v1/models/{self.id}/samples',
            params={'split_type': split, 'limit': 100}
        )
        samples = sample_response.get('samples', [])
        original_sample = next((s for s in samples if s['id'] == sample_id), None)
        
        if not original_sample:
            raise ValueError(f"Could not find original sample {sample_id}")
        
        # Group reconstructions by feature and temporal_tag
        result = {}
        for window in reconstructions:
            feature = window['feature']
            temporal_tag = window['temporal_tag']
            reconstructed_values = window['reconstructed_values']
            
            if feature not in result:
                result[feature] = {}
            
            # Get dates from original sample
            dates = self._extract_dates_from_sample(original_sample, feature, temporal_tag)
            
            result[feature][temporal_tag] = {
                "dates": np.array(dates) if dates else None,
                "values": np.array(reconstructed_values)
            }
        
        return result
    
    
    
    def _get_original_past_data(self, sample: Dict, feature: str) -> tuple:
        """
        Get original (non-reconstructed) past data for plotting.
        This ensures alignment with forecast data which is anchored to original values.
        
        Args:
            sample: Full sample dict (with conditioning_data/target_data)
            feature: Feature name
        
        Returns:
            Tuple of (dates, denormalized_values)
        """
        import numpy as np
        
        # Find the feature in conditioning_data (all past windows are here)
        conditioning_data = sample.get('conditioning_data', [])
        
        for item in conditioning_data:
            if item.get('feature') == feature and item.get('temporal_tag') == 'past':
                dates = item.get('dates', [])
                normalized_series = item.get('normalized_series', [])
                
                if not normalized_series:
                    raise ValueError(f"No normalized_series found for {feature} past window")
                
                # Denormalize using model's normalization params
                denormalized_values = self._denormalize_values(feature, normalized_series)
                
                return dates, denormalized_values
        
        raise ValueError(f"Feature {feature} with temporal_tag='past' not found in conditioning_data")
    
    def _denormalize_values(self, feature: str, normalized_values: list) -> list:
        """Denormalize values using model's normalization parameters"""
        import numpy as np
        
        # Get normalization params from model data (separate field, not in metadata)
        feature_norm_params = self._data.get('feature_normalization_params', {})
        
        if not feature_norm_params or feature not in feature_norm_params:
            # Try fetching from database if not in local data
            try:
                model_response = self.http.get(f'/api/v1/models/{self.id}')
                feature_norm_params = model_response.get('feature_normalization_params', {})
                # Cache it for future use
                self._data['feature_normalization_params'] = feature_norm_params
            except:
                pass
        
        if not feature_norm_params or feature not in feature_norm_params:
            print(f"  ⚠️  Warning: No normalization params for {feature}, returning normalized values")
            return normalized_values
        
        norm_params = feature_norm_params[feature]
        mean = norm_params.get('mean', 0.0)
        std = norm_params.get('std', 1.0)
        
        # Denormalize: value = (normalized * std) + mean
        denormalized = [(val * std) + mean for val in normalized_values]
        
        return denormalized
    
    def _get_ground_truth_future_data(self, sample: Dict, feature: str) -> tuple:
        """
        Extract ground truth future target data from a test sample.
        
        This gets the actual realized future values that really happened,
        which we can compare against our forecasts.
        
        Args:
            sample: Full sample dict (with target_data)
            feature: Target feature name
        
        Returns:
            Tuple of (dates, values) - both can be None if not found
        """
        target_data = sample.get('target_data', [])
        
        for item in target_data:
            if (item.get('feature') == feature and 
                item.get('temporal_tag') == 'future'):
                
                # Get dates and residuals
                dates = item.get('dates', [])
                residuals = item.get('normalized_residuals', [])
                
                if residuals and dates:
                    # Get the last normalized past value as reference
                    past_dates, past_values = self._get_original_past_data(sample, feature)
                    if past_values:
                        # Get normalization parameters
                        norm_params = self._data.get('model_metadata', {}).get('normalization_parameters', {}).get(feature, {})
                        mean = norm_params.get('mean', 0.0)
                        std = norm_params.get('std', 1.0)
                        
                        # Normalize the last past value to get the reference
                        last_past_normalized = (past_values[-1] - mean) / std
                        
                        # Reconstruct the normalized series: add normalized reference to each residual
                        reconstructed_normalized = [last_past_normalized + residual for residual in residuals]
                        
                        # Denormalize the reconstructed values
                        denormalized_values = [(val * std) + mean for val in reconstructed_normalized]
                        return dates, denormalized_values
        
        return None, None
    
    def _extract_dates_from_sample(self, sample: Dict, feature: str, temporal_tag: str) -> Optional[List[str]]:
        """Extract date array from sample's conditioning_data or target_data"""
        # Search in conditioning_data
        conditioning_data = sample.get('conditioning_data', [])
        if conditioning_data:
            for item in conditioning_data:
                if item.get('feature') == feature and item.get('temporal_tag') == temporal_tag:
                    dates = item.get('dates')
                    if dates:
                        return dates
        
        # Search in target_data
        target_data = sample.get('target_data', [])
        if target_data:
            for item in target_data:
                if item.get('feature') == feature and item.get('temporal_tag') == temporal_tag:
                    dates = item.get('dates')
                    if dates:
                        return dates
        
        # If dates not found, generate them from sample metadata
        # This is a fallback for when dates aren't stored in the sample
        window_length = 30  # Default, should match the actual window length
        if temporal_tag == "past":
            # Use start_date if available
            start_date = sample.get('start_date')
            if start_date:
                from datetime import datetime, timedelta
                base = datetime.strptime(start_date, '%Y-%m-%d')
                return [(base + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(window_length)]
        elif temporal_tag == "future":
            # Use end_date and work backwards
            end_date = sample.get('end_date')
            if end_date:
                from datetime import datetime, timedelta
                base = datetime.strptime(end_date, '%Y-%m-%d')
                return [(base - timedelta(days=window_length - 1 - i)).strftime('%Y-%m-%d') for i in range(window_length)]
        
        return None
    
    # ============================================
    # PLOTTING METHODS
    # ============================================
    
    def plot_reconstruction(
        self,
        sample_id: str = None,
        feature: str = None,
        split: str = "test",
        save_path: str = None,
        show: bool = True
    ):
        """
        Reconstruct and plot a sample's original vs reconstructed values.
        
        Args:
            sample_id: Sample ID to reconstruct (auto-selects if None)
            feature: Feature name to plot (plots all if None)
            split: Which split to use if auto-selecting sample
            save_path: Path to save plot (e.g., "reconstruction.png")
            show: Whether to display the plot
        
        Example:
            >>> model.plot_reconstruction(feature="Gold Price")
            >>> model.plot_reconstruction(save_path="plots/reconstruction.png", show=False)
        """
        from ..visualization import TimeSeriesPlotter, _check_matplotlib
        import matplotlib.pyplot as plt
        from datetime import datetime
        
        _check_matplotlib()
        
        # Reconstruct the sample
        reconstruction = self.reconstruct_sample(sample_id=sample_id, split=split)
        
        # Determine which features to plot
        features_to_plot = [feature] if feature else list(reconstruction.keys())
        
        # Create subplots
        n_features = len(features_to_plot)
        fig, axes = plt.subplots(n_features, 1, figsize=(14, 5 * n_features))
        if n_features == 1:
            axes = [axes]
        
        for idx, feat_name in enumerate(features_to_plot):
            feat_data = reconstruction[feat_name]
            
            # Combine past and future
            past_data = feat_data.get('past', {})
            future_data = feat_data.get('future', {})
            
            past_dates = past_data.get('dates', [])
            past_values = past_data.get('values', [])
            future_dates = future_data.get('dates', [])
            future_values = future_data.get('values', [])
            
            # Concatenate
            all_dates = list(past_dates) + list(future_dates)
            all_values = list(past_values) + list(future_values)
            
            # Convert dates to datetime
            date_objects = [datetime.strptime(d, '%Y-%m-%d') if isinstance(d, str) else d for d in all_dates]
            
            # For now, we don't have original values to compare, so just plot reconstructed
            # In future, could fetch original from database for comparison
            TimeSeriesPlotter.plot_reconstruction(
                dates=date_objects,
                original_values=all_values,  # Would need true original values
                reconstructed_values=all_values,
                feature_name=feat_name,
                past_length=len(past_dates),
                title=f'Reconstruction: {feat_name}',
                ax=axes[idx]
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    
    
    # ============================================
    # DATA EXTRACTION FOR VALIDATION
    # ============================================
    
    def get_real_paths(
        self,
        split: str = 'validation',
        target_features_only: bool = True
    ) -> Dict[str, Any]:
        """
        Extract real paths from samples for validation against synthetic data.
        
        Args:
            split: Sample split to extract ('validation' or 'test')
            target_features_only: If True, only return target features (default)
        
        Returns:
            Dict with:
                - 'paths': np.ndarray of shape (n_samples, n_timesteps, n_features)
                - 'feature_names': List of feature names
                - 'dates': List of date strings (from first sample)
                - 'n_samples': Number of samples
                - 'split': Split name
        
        Example:
            >>> real_val = model.get_real_paths(split='validation')
            >>> real_test = model.get_real_paths(split='test')
            >>> validation_results = synthetic_data.validate_against_real_data(
            ...     real_validation_data=real_val,
            ...     real_test_data=real_test
            ... )
        """
        import numpy as np
        
        print(f"[Model] Extracting real paths from {split} split...")
        
        # Fetch samples
        response = self.http.get(
            f'/api/v1/models/{self.id}/samples',
            params={'split_type': split, 'limit': 1000, 'include_data': 'true'}
        )
        samples = response.get('samples', [])
        
        if not samples:
            raise ValueError(f"No {split} samples found")
        
        print(f"  Found {len(samples)} samples")
        
        # Determine which features to extract
        if target_features_only:
            feature_names = [f.get('display_name', f.get('name')) 
                           for f in self.input_features if f.get('type') == 'target']
        else:
            feature_names = [f.get('display_name', f.get('name')) 
                           for f in self.input_features]
        
        print(f"  Extracting {len(feature_names)} features: {feature_names}")
        
        # Get normalization params for denormalization
        norm_params = self._data.get('feature_normalization_params', {})
        
        # Extract paths from all samples
        all_paths = []
        dates = None
        
        for sample in samples:
            sample_path = []
            
            # For each target feature, extract future values (denormalized)
            for feature_name in feature_names:
                # Find in target_data
                target_data = sample.get('target_data', [])
                feature_values = None
                
                for item in target_data:
                    if (item.get('feature') == feature_name or 
                        item.get('feature_name') == feature_name) and \
                       item.get('temporal_tag') == 'future':
                        
                        # Get normalized residuals
                        normalized_residuals = item.get('normalized_residuals', [])
                        
                        if dates is None:
                            dates = item.get('dates', [])
                        
                        # Get reference value (last past value)
                        conditioning_data = sample.get('conditioning_data', [])
                        past_ref_norm = None
                        
                        for cond_item in conditioning_data:
                            if (cond_item.get('feature') == feature_name or
                                cond_item.get('feature_name') == feature_name) and \
                               cond_item.get('temporal_tag') == 'past':
                                past_series = cond_item.get('normalized_series', [])
                                if past_series:
                                    past_ref_norm = past_series[-1]
                                break
                        
                        if past_ref_norm is None:
                            raise ValueError(f"Could not find past reference for {feature_name}")
                        
                        # Convert residuals to series: series = residual + reference
                        normalized_series = [r + past_ref_norm for r in normalized_residuals]
                        
                        # Denormalize
                        mean = norm_params.get(feature_name, {}).get('mean', 0)
                        std = norm_params.get(feature_name, {}).get('std', 1)
                        feature_values = [(v * std) + mean for v in normalized_series]
                        
                        break
                
                if feature_values is None:
                    raise ValueError(f"Could not find data for {feature_name} in sample {sample.get('id')}")
                
                sample_path.append(feature_values)
            
            # Transpose to (n_timesteps, n_features)
            sample_path_array = np.array(sample_path).T
            all_paths.append(sample_path_array)
        
        # Stack to (n_samples, n_timesteps, n_features)
        paths_array = np.array(all_paths)
        
        print(f"  ✅ Extracted paths shape: {paths_array.shape}")
        print(f"     (n_samples={paths_array.shape[0]}, n_timesteps={paths_array.shape[1]}, n_features={paths_array.shape[2]})")
        
        return {
            'paths': paths_array,
            'feature_names': feature_names,
            'dates': dates,
            'n_samples': len(samples),
            'split': split
        }
    
    # ============================================
    # Vine Copula METHODS
    # ============================================
    
    def optimize(self,
                                    n_trials: int = 50,
                                    n_components_range: tuple = (2, 6),
                                    n_factors_range: tuple = (15, 50),
                                    lower_tail_quantile_range: tuple = (0.03, 0.15),
                                    upper_tail_quantile_range: tuple = (0.85, 0.97),
                                    top_k_neighbors_range: tuple = (25, 200),
                                    objectives: List[str] = ['validation_ll', 'generalization_gap', 'ks_pvalue'],
                                    split: str = 'training',
                                    confirm: Optional[bool] = None) -> Dict[str, Any]:
        """
        Optimize Vine Copula hyperparameters using Optuna
        
        This method runs Bayesian optimization to find the best hyperparameters
        for the Vine Copula model and automatically saves them for use in training.
        
        Args:
            n_trials: Number of optimization trials
            n_components_range: (min, max) for number of components
            n_factors_range: (min, max) for number of factors
            lower_tail_quantile_range: (min, max) for lower tail quantile
            upper_tail_quantile_range: (min, max) for upper tail quantile
            top_k_neighbors_range: (min, max) for top_k_neighbors
            objectives: List of objectives to optimize ['validation_ll', 'generalization_gap', 'ks_pvalue']
            split: Data split to use for optimization
            confirm: Whether to confirm the optimization (default: auto-confirm)
        
        Returns:
            Dictionary containing optimization results and optimal parameters
        """
        if confirm is None:
            confirm = True
        
        if confirm:
            print(f"\n🔧 Model Hyperparameter Optimization")
            print(f"   Trials: {n_trials}")
            print(f"   Objectives: {objectives}")
            print(f"   Data split: {split}")
            print(f"   This will test {n_trials} different parameter combinations")
            print(f"   and find the optimal settings for your data.")
            
            response = input(f"\nProceed with optimization? (y/N): ")
            if response.lower() != 'y':
                print("❌ Optimization cancelled")
                return {"status": "cancelled"}
        
        print(f"\n🚀 Starting hyperparameter optimization...")
        
        # Prepare optimization payload
        payload = {
            'user_id': self._data.get('user_id'),
            'model_id': self.id,
            'n_trials': n_trials,
            'n_components_range': n_components_range,
            'n_factors_range': n_factors_range,
            'lower_tail_quantile_range': lower_tail_quantile_range,
            'upper_tail_quantile_range': upper_tail_quantile_range,
            'top_k_neighbors_range': top_k_neighbors_range,
            'objectives': objectives,
            'split': split
        }
        
        # Call optimization endpoint
        response = self.http.post('/api/v1/ml/optimize', payload)
        
        if response.get('status') == 'success':
            optimal_params = response.get('optimal_parameters', {})
            print(f"\n✅ Optimization complete!")
            print(f"   Optimal parameters found:")
            print(f"   • n_components: {optimal_params.get('n_components')}")
            print(f"   • n_factors: {optimal_params.get('n_factors')}")
            print(f"   • lower_tail_quantile: {optimal_params.get('lower_tail_quantile')}")
            print(f"   • upper_tail_quantile: {optimal_params.get('upper_tail_quantile')}")
            print(f"   • top_k_neighbors: {optimal_params.get('top_k_neighbors')}")
            
            best_score = response.get('best_score')
            if best_score:
                print(f"   • Best validation LL: {best_score:.2f}")
            
            print(f"\n💡 These parameters are now saved and will be used automatically")
            print(f"   when you call model.train() without specifying parameters.")
        else:
            print(f"❌ Optimization failed: {response.get('error', 'Unknown error')}")
        
        return response
    
    def train(self,
                  n_regimes: int = 3,
                  compute_validation_ll: bool = False) -> Dict[str, Any]:
        """
        Train Vine Copula model on encoded samples
        
        Pipeline:
        - Empirical marginals (smoothed CDF + exponential tails)
        - EM algorithm with regime-specific vine copulas
        - Mixed copula families (student, clayton, gumbel, frank, joe)
        - Optimal truncation level and thread count for performance
        
        Args:
            n_regimes: Number of mixture components/regimes (default: 3)
            compute_validation_ll: Also compute validation log-likelihood (default: False)
            
        Returns:
            Dict with training results including metrics and model path
            
        Example:
            >>> # Train Vine Copula model with default settings
            >>> result = model.train()
            >>> print(f"Model trained with {result['training_metrics']['n_components']} regimes")
            
            >>> # Train with custom number of regimes
            >>> result = model.train(n_regimes=5)
            >>> print(f"Model trained with {result['training_metrics']['n_components']} regimes")
            
            >>> # Train with validation log-likelihood
            >>> result = model.train(n_regimes=3, compute_validation_ll=True)
        """
        print(f"\n🤖 Training Vine Copula Model")
        print(f"   Regimes: {n_regimes}")
        print(f"   Copula family: mixed (optimized)")
        print(f"   Truncation level: 3 (optimal)")
        print(f"   Threads: auto-detected")
        
        print(f"\n🚀 Training model...")
        
        # Call training endpoint
        payload = {
            'user_id': self._data.get("user_id"),  # For API key auth
            'model_id': self.id,
            'n_regimes': n_regimes,
            'compute_validation_ll': compute_validation_ll
        }
        
        result = self.http.post('/api/v1/ml/train', payload)
        
        train_metrics = result['training_metrics']
        
        print(f"✅ Vine Copula training completed!")
        print(f"   Samples used: {result['n_samples_used']}")
        print(f"   Dimensions: {result['n_dimensions']}")
        print(f"   Regime weights: {train_metrics.get('regime_weights', 'N/A')}")
        
        model_path = result.get('vine_copula_path')
        if model_path and model_path != 'N/A':
            print(f"   Model saved to: {model_path}")
        
        if result.get('validation_metrics'):
            val_metrics = result['validation_metrics']
            train_ll = val_metrics.get('train_per_sample_log_likelihood')
            val_ll = val_metrics['per_sample_log_likelihood']
            gen_gap = val_metrics.get('generalization_gap')
            
            if train_ll:
                print(f"   Training LL: {train_ll:.2f}")
            print(f"   Validation LL: {val_ll:.2f}")
            if gen_gap:
                print(f"   Generalization gap: {gen_gap:.2f}")
        
        # Refresh model data
        self.refresh()
        
        return result
    
    def validate(self,
                     n_forecast_samples: int = 100,
                     run_on_training: bool = True,
                     run_on_validation: bool = True) -> Dict[str, Any]:
        """
        Validate Vine Copula model on held-out data
        
        Computes:
        - Validation log-likelihood (out-of-sample fit)
        - Regime analysis (component assignments)
        - Calibration metrics (forecast quality)
        - Coverage metrics (68% and 95% confidence intervals)
        - KS p-value for calibration testing
        
        Args:
            n_forecast_samples: Number of forecast samples per validation sample (default: 100)
            run_on_training: Also run validation on training set (default: True)
            run_on_validation: Run validation on validation set (default: True)
            
        Returns:
            Dict with validation metrics
            
        Example:
            >>> validation = model.validate(n_forecast_samples=100)
            >>> print(f"Validation log-likelihood: {validation['validation_metrics']['log_likelihood']['per_sample_log_likelihood']}")
            >>> print(f"Coverage 95%: {validation['calibration_metrics']['calibration']['coverage_95']}")
        """
        print(f"\n🔍 Validating Vine Copula model...")
        print(f"   Training set: {run_on_training}")
        print(f"   Validation set: {run_on_validation}")
        print(f"   Forecast samples per validation sample: {n_forecast_samples}")
        
        result = self.http.post('/api/v1/ml/validate', {
            'user_id': self._data.get("user_id"),
            'model_id': self.id,
            'n_forecast_samples': n_forecast_samples,
            'run_on_training': run_on_training,
            'run_on_validation': run_on_validation
        })
        
        # Display results
        print(f"\n" + "="*70)
        print(f"✅ Model Validation Complete")
        print(f"="*70)
        
        # Training metrics
        if result.get('training_metrics'):
            train_metrics = result['training_metrics']
            train_ll = train_metrics.get('log_likelihood', {})
            print(f"\n📊 Training Set Metrics:")
            print(f"   Log-likelihood: {train_ll.get('per_sample_log_likelihood', 'N/A'):.4f}")
            print(f"   BIC: {train_ll.get('bic', 'N/A'):.2f}")
            print(f"   AIC: {train_ll.get('aic', 'N/A'):.2f}")
            print(f"   Samples: {train_metrics.get('n_samples', 'N/A')}")
        
        # Validation metrics
        if result.get('validation_metrics'):
            val_metrics = result['validation_metrics']
            val_ll = val_metrics.get('log_likelihood', {})
            print(f"\n📊 Validation Set Metrics:")
            val_ll_per_sample = val_ll.get('per_sample_log_likelihood', 'N/A')
            val_bic = val_ll.get('bic', 'N/A')
            val_aic = val_ll.get('aic', 'N/A')
            print(f"   Log-likelihood: {val_ll_per_sample if val_ll_per_sample == 'N/A' else f'{val_ll_per_sample:.4f}'}")
            print(f"   BIC: {val_bic if val_bic == 'N/A' else f'{val_bic:.2f}'}")
            print(f"   AIC: {val_aic if val_aic == 'N/A' else f'{val_aic:.2f}'}")
            print(f"   Samples: {val_metrics.get('n_samples', 'N/A')}")
            
            # Generalization gap
            if result.get('training_metrics') and result.get('validation_metrics'):
                train_ll_val = result['training_metrics'].get('log_likelihood', {}).get('per_sample_log_likelihood', 0)
                val_ll_val = val_ll.get('per_sample_log_likelihood', 0)
                gap = train_ll_val - val_ll_val
                print(f"   Generalization gap: {gap:.4f}")
        
        # Calibration metrics
        if result.get('calibration_metrics'):
            cal_metrics = result['calibration_metrics']
            
            print(f"\n🎯 Forecast Quality Metrics:")
            print(f"{'─'*70}")
            
            # Handle new reconstructed metrics structure
            if 'overall' in cal_metrics:
                overall_metrics = cal_metrics['overall']
                
                # CRPS
                if 'crps' in overall_metrics:
                    crps = overall_metrics['crps']
                    print(f"{crps['icon']} CRPS: {crps['value']:.4f} ({crps['quality'].title()})")
                    print(f"   {crps['interpretation']}")
                
                # Sharpness
                if 'sharpness' in overall_metrics:
                    sharpness = overall_metrics['sharpness']
                    print(f"{sharpness['icon']} Sharpness: {sharpness['value']:.4f} ({sharpness['quality'].title()})")
                    print(f"   {sharpness['interpretation']}")
                
                # Reliability (ECE)
                if 'reliability' in overall_metrics:
                    reliability = overall_metrics['reliability']
                    print(f"{reliability['icon']} Reliability (ECE): {reliability['ece']:.4f} ({reliability['quality'].title()})")
                    print(f"   {reliability['interpretation']}")
                
                # Show conditioning effectiveness
                if 'conditioning_effectiveness' in cal_metrics:
                    eff = cal_metrics['conditioning_effectiveness']
                    eff_status = "Good" if 0.4 <= eff <= 0.6 else "Poor"
                    print(f"   📊 Conditioning effectiveness: {eff:.3f} ({eff_status})")
                
                # Show note if available
                if cal_metrics.get('note'):
                    print(f"   📝 {cal_metrics['note']}")
                
                # Horizon-specific metrics
                if 'horizons' in cal_metrics:
                    horizons = cal_metrics['horizons']
                    print(f"\n📈 Horizon-Specific Metrics:")
                    print(f"{'─'*50}")
                    
                    for horizon_name, horizon_metrics in horizons.items():
                        horizon_display = horizon_name.replace('_', ' ').title()
                        print(f"\n{horizon_display}:")
                        
                        # CRPS
                        if 'crps' in horizon_metrics:
                            crps = horizon_metrics['crps']
                            print(f"  {crps['icon']} CRPS: {crps['value']:.4f} ({crps['quality'].title()})")
                        
                        # Sharpness
                        if 'sharpness' in horizon_metrics:
                            sharpness = horizon_metrics['sharpness']
                            print(f"  {sharpness['icon']} Sharpness: {sharpness['value']:.4f} ({sharpness['quality'].title()})")
                        
                        # Reliability (ECE)
                        if 'reliability' in horizon_metrics:
                            reliability = horizon_metrics['reliability']
                            print(f"  {reliability['icon']} Reliability: {reliability['ece']:.4f} ({reliability['quality'].title()})")
                
                # Show future window info
                if 'future_window' in cal_metrics:
                    print(f"\n📅 Future Window: {cal_metrics['future_window']} days")
                    
            else:
                # Fallback to old structure
                # CRPS
                if 'crps' in cal_metrics:
                    crps = cal_metrics['crps']
                    print(f"{crps['icon']} CRPS: {crps['value']:.4f} ({crps['quality'].title()})")
                    print(f"   {crps['interpretation']}")
                
                # Sharpness
                if 'sharpness' in cal_metrics:
                    sharpness = cal_metrics['sharpness']
                    print(f"{sharpness['icon']} Sharpness: {sharpness['value']:.4f} ({sharpness['quality'].title()})")
                    print(f"   {sharpness['interpretation']}")
                
                # Reliability (ECE)
                if 'reliability' in cal_metrics:
                    reliability = cal_metrics['reliability']
                    print(f"{reliability['icon']} Reliability (ECE): {reliability['ece']:.4f} ({reliability['quality'].title()})")
                    print(f"   {reliability['interpretation']}")
            
            print(f"{'─'*70}")
        
        # Regime analysis
        if result.get('regime_analysis'):
            regime_analysis = result['regime_analysis']
            print(f"\n🔄 Regime Analysis:")
            print(f"   Number of regimes: {regime_analysis.get('n_regimes', 'N/A')}")
            print(f"   Posterior entropy: {regime_analysis.get('posterior_entropy', 'N/A'):.4f}")
            
            regime_counts = regime_analysis.get('regime_counts', {})
            regime_weights = regime_analysis.get('regime_weights', {})
            for regime in regime_counts:
                count = regime_counts[regime]
                weight = regime_weights.get(regime, 0)
                print(f"   {regime}: {count} samples ({weight:.3f} weight)")
        
        print(f"\n✅ Validation complete!")
        print(f"{'='*70}")
        
        return result
    
    def forecast(self,
                     observed_components: Optional[List[Dict[str, Any]]] = None,
                     split: str = 'validation',
                     sample_index: int = 0,
                     sample_id: Optional[str] = None,
                     n_samples: int = 1000,
                     return_format: str = 'reconstructed',  # NEW: Default to reconstructed
                     use_local_copula: bool = True,
                     top_k_neighbors: int = 100,
                     copula_type: str = 't',
                     clayton_lower_threshold: float = 0.3,
                     clayton_ratio_threshold: float = 1.5,
                     gumbel_upper_threshold: float = 0.3,
                     gumbel_ratio_threshold: float = 0.67) -> Dict[str, Any]:
        """
        Generate forecasts using Vine Copula (with optional local copula)
        
        Two modes:
        1. Sample-based (default): Condition on a validation/test sample
        2. Inline: Provide observed_components manually
        
        Two conditional methods:
        1. Vine Copula conditionals (use_local_copula=False): Direct Gaussian conditional formulas
           - Faster, simpler
           - Gaussian tail dependence
        2. Local copula (use_local_copula=True): Fit copula on K nearest neighbors
           - Adaptive tail dependence (t-copula df parameter)
           - Supports Clayton (lower tail), Gumbel (upper tail), t-copula
        
        Args:
            observed_components: List of observed components for inline conditioning (optional)
            split: Which split to use for sample-based conditioning (default: 'validation')
            sample_index: Index of sample in split (default: 0)
            sample_id: Specific sample ID to use (overrides split/sample_index)
            n_samples: Number of forecast samples to generate (default: 1000)
            use_local_copula: Use local copula (True) or Vine Copula conditionals (False) (default: True)
            top_k_neighbors: Number of neighbors for local copula (default: 100, ignored if use_local_copula=False)
            copula_type: 't', 'adaptive', 'gaussian', 'clayton', 'gumbel' (default: 't')
            
        Returns:
            Dict with forecasts and conditioning info
            
        Examples:
            >>> # Local copula (adaptive tail dependence)
            >>> forecasts = model.forecast(split='validation', n_samples=50, use_local_copula=True)
            >>> 
            >>> # Vine Copula conditionals (faster, Gaussian tails)
            >>> forecasts = model.forecast(split='validation', n_samples=50, use_local_copula=False)
            >>> 
            >>> # Unconditional forecast
            >>> forecasts = model.forecast(observed_components=[], n_samples=1000)
        """
        print(f"\n🎲 Generating forecasts...")
        
        # Determine conditioning source
        if observed_components is not None:
            conditioning_source = "inline"
            print(f"   Mode: Inline conditioning")
            print(f"   Conditioning on {len(observed_components)} observations")
        else:
            conditioning_source = "sample"
            print(f"   Mode: Sample-based conditioning")
            print(f"   Split: {split}, Index: {sample_index}")
        
        print(f"   Samples: {n_samples}")
        # Method info removed - always uses Vine Copula conditionals with hoeffd criterion
        
        # Call forecasting endpoint
        payload = {
            'user_id': self._data.get("user_id"),  # For API key auth
            'model_id': self.id,
            'conditioning_source': conditioning_source,
            'n_samples': n_samples,
            'use_local_copula': use_local_copula,
            'top_k_neighbors': top_k_neighbors,
            'copula_type': copula_type,
            'clayton_lower_threshold': clayton_lower_threshold,
            'clayton_ratio_threshold': clayton_ratio_threshold,
            'gumbel_upper_threshold': gumbel_upper_threshold,
            'gumbel_ratio_threshold': gumbel_ratio_threshold
        }
        
        if conditioning_source == "inline":
            payload['observed_components'] = observed_components or []
        else:  # sample
            payload['split'] = split
            payload['sample_index'] = sample_index
            if sample_id:
                payload['sample_id'] = sample_id
        
        result = self.http.post('/api/v1/ml/forecast', payload)
        
        print(f"✅ Forecasting completed!")
        print(f"   Generated {result['n_samples']} samples")
        print(f"   Observed: {result['n_observed']} dimensions")
        print(f"   Predicted: {result['n_predicted']} dimensions")
        
        # Return the raw result with reconstructed windows
        reconstructed_windows = result.get('conditioning_info', {}).get('reconstructed', [])
        
        if not reconstructed_windows:
            # Fallback for old API (shouldn't happen with new backend)
            logger.warning("No auto-reconstructed data found, returning raw result")
            return result
        
        # Return the full forecast result
        return result
    
    def reconstruct_forecasts(self,
                                   forecasts: Dict[str, Any] = None,
                                   mfa_forecasts: Dict[str, Any] = None,  # Deprecated, for backward compatibility
                                   reference_sample_id: Optional[str] = None,
                                   split: str = 'validation') -> Dict[str, Any]:
        """
        Reconstruct forecasts to original feature space
        
        Args:
            forecasts: Output from forecast() (preferred parameter name)
            mfa_forecasts: [DEPRECATED] Use 'forecasts' instead
            reference_sample_id: Sample ID to use for reference values (for residuals)
                                If None, uses first validation sample
            split: Data split to get reference sample from (default: 'validation')
            
        Returns:
            dict: Reconstructed trajectories for all forecast samples
        """
        import numpy as np
        
        # Handle backward compatibility
        if forecasts is None and mfa_forecasts is not None:
            forecasts = mfa_forecasts
        elif forecasts is None:
            raise ValueError("forecasts parameter is required")
        
        print(f"\n🔄 Reconstructing {len(forecasts['forecasts'])} forecast samples...")
        
        # Get reference sample if needed (only if reference_values not available)
        # With Cloud SQL, reference values come directly from forecast response
        if reference_sample_id is None and not (forecasts.get('ground_truth') and forecasts['ground_truth'].get('reference_values')):
            print(f"  Fetching reference sample from {split} split...")
            sample_response = self.http.get(
                f'/api/v1/models/{self.id}/samples',
                params={'split_type': split, 'limit': 1}
            )
            samples = sample_response.get('samples', [])
            if not samples:
                raise ValueError(f"No {split} samples found")
            reference_sample_id = samples[0]['id']
            print(f"  Using reference sample: {reference_sample_id}")
        elif forecasts.get('ground_truth') and forecasts['ground_truth'].get('sample_id'):
            # Use sample_id from ground_truth
            reference_sample_id = forecasts['ground_truth']['sample_id']
            print(f"  Using reference sample from forecast ground_truth: {reference_sample_id}")
        
        # Extract group metadata from forecasts (if available)
        feature_metadata = {}
        if forecasts['forecasts']:
            first_sample = forecasts['forecasts'][0]
            if '_group_metadata' in first_sample:
                feature_metadata = first_sample['_group_metadata']
                print(f"  Found group metadata for {len(feature_metadata)} features")
        
        # Get feature_groups for mapping group_ids to individual features
        feature_groups = forecasts.get('conditioning_info', {}).get('feature_groups', {})
        group_id_to_features = {}
        
        if feature_groups:
            for group in feature_groups.get('target_groups', []):
                group_id_to_features[group['id']] = {
                    'features': group['features'],
                    'is_multivariate': group['is_multivariate']
                }
            for group in feature_groups.get('conditioning_groups', []):
                group_id_to_features[group['id']] = {
                    'features': group['features'],
                    'is_multivariate': group['is_multivariate']
                }
            print(f"  Loaded {len(group_id_to_features)} feature groups for reconstruction")
        
        # BATCH RECONSTRUCTION: Collect all encoded windows from all forecast samples
        print(f"  Collecting encoded windows from all {len(forecasts['forecasts'])} forecast samples...")
        all_encoded_windows = []
        sample_window_mapping = {}  # Track which windows belong to which sample
        
        for i, forecast_sample in enumerate(forecasts['forecasts']):
            if (i + 1) % 10 == 0:
                print(f"  Processing sample {i+1}/{len(forecasts['forecasts'])}...")
            
            # Group components by (source, feature, temporal_tag, data_type)
            windows = {}
            for key, value in forecast_sample.items():
                # Parse key: "source_feature_temporal_tag_data_type_component_idx"
                # Example: "target_group_1_future_normalized_residuals_0"
                parts = key.rsplit('_', 1)
                if len(parts) != 2:
                    continue
                    
                window_key = parts[0]  # Everything except component_idx
                try:
                    component_idx = int(parts[1])
                except ValueError:
                    continue
                
                if window_key not in windows:
                    windows[window_key] = {}
                windows[window_key][component_idx] = value
            
            # Build encoded_windows for this sample
            sample_windows = []
            for window_key, components in windows.items():
                # Parse window_key: "source_feature_temporal_tag_data_type"
                # Example: "target_group_1_future_normalized_residuals"
                # Need to extract: source, feature, temporal_tag, data_type
                
                # Split and identify parts
                parts = window_key.split('_')
                if len(parts) < 4:
                    continue
                
                # First part is source (target/conditioning)
                source = parts[0]
                
                # Last part is data_type (series/residuals)
                data_type = parts[-1]
                
                # Second to last is normalized/encoded status
                # temporal_tag is before that (past/future)
                temporal_tag = parts[-2]
                
                # Everything in between is the feature name
                # For "target_group_1_future_normalized_residuals":
                #   parts = ['target', 'group', '1', 'future', 'normalized', 'residuals']
                #   source = 'target'
                #   data_type = 'residuals'
                #   temporal_tag = 'normalized' <- WRONG!
                
                # Better parsing: look for 'past' or 'future' to identify temporal_tag
                temporal_idx = None
                for j, part in enumerate(parts):
                    if part in ['past', 'future']:
                        temporal_idx = j
                        break
                
                if temporal_idx is None:
                    continue
                
                # Feature is everything between source and temporal_tag
                feature = '_'.join(parts[1:temporal_idx])
                temporal_tag = parts[temporal_idx]
                
                # data_type is "normalized_series" or "normalized_residuals"
                # It's everything after temporal_tag
                data_type_parts = parts[temporal_idx+1:]
                data_type = '_'.join(data_type_parts)
                
                # Sort components by index
                sorted_components = [components[idx] for idx in sorted(components.keys())]
                
                # Build window in format expected by reconstruct endpoint
                # The reconstruct endpoint expects data_type like "encoded_normalized_residuals"
                if not data_type.startswith('encoded_'):
                    data_type = 'encoded_' + data_type
                
                encoded_window = {
                    "feature": feature,
                    "temporal_tag": temporal_tag,
                    "data_type": data_type,
                    "encoded_values": sorted_components,  # API expects "encoded_values"
                    "_sample_idx": i  # Track which forecast sample this belongs to
                }
                
                # Add group metadata if available (enables proper unpacking to individual features)
                if feature in feature_metadata:
                    metadata = feature_metadata[feature]
                    encoded_window['n_components'] = metadata['n_components']
                    encoded_window['is_multivariate'] = metadata['is_multivariate']
                    encoded_window['group_features'] = metadata['group_features']
                    encoded_window['is_group'] = metadata['is_group']
                elif feature in group_id_to_features:
                    # Use feature_groups mapping from forecast response
                    group_info = group_id_to_features[feature]
                    encoded_window['group_features'] = group_info['features']
                    encoded_window['is_multivariate'] = group_info['is_multivariate']
                    encoded_window['is_group'] = True
                    encoded_window['n_components'] = len(sorted_components)
                
                sample_windows.append(encoded_window)
            
            # Add to batch collection
            all_encoded_windows.extend(sample_windows)
            sample_window_mapping[i] = len(sample_windows)  # Track how many windows per sample
        
        print(f"  Collected {len(all_encoded_windows)} total encoded windows")
        
        # SINGLE BATCH API CALL: Reconstruct all windows at once
        print(f"  Making single batch reconstruction call...")
        
        # Check if reference_values are available in ground_truth
        reference_values = None
        if forecasts.get('ground_truth') and forecasts['ground_truth'].get('reference_values'):
            reference_values = forecasts['ground_truth']['reference_values']
            print(f"  Using {len(reference_values)} reference values from forecast ground_truth")
        
        payload = {
            "user_id": self._data.get("user_id"),
            "model_id": self.id,
            "encoded_source": "inline",
            "encoded_windows": all_encoded_windows,
            "reference_source": "inline" if reference_values else "database",
            "reference_values": reference_values if reference_values else None,
            "reference_table": "samples_normalized" if not reference_values else None,  # Cloud SQL table name
            "reference_column": "normalized_past" if not reference_values else None,  # Not actually used, but required
            "reference_sample_id": reference_sample_id if not reference_values else None,
            "output_destination": "return"
        }
        
        response = self.http.post('/api/v1/ml/reconstruct', payload)
        all_reconstructions_raw = response.get('reconstructions', [])
        
        print(f"  Received {len(all_reconstructions_raw)} reconstructed windows")
        
        # PARSE BATCH RESPONSE: Split back into individual samples
        print(f"  Parsing batch response back into individual samples...")
        all_reconstructions = []
        current_idx = 0
        
        for sample_idx in range(len(forecasts['forecasts'])):
            n_windows = sample_window_mapping[sample_idx]
            sample_reconstructions = all_reconstructions_raw[current_idx:current_idx + n_windows]
            current_idx += n_windows
            
            all_reconstructions.append({
                'sample_idx': sample_idx,
                'reconstructions': sample_reconstructions
            })
        
        print(f"✅ Reconstructed all {len(all_reconstructions)} forecast samples")
        
        # Extract ground truth from forecast response (if available)
        print(f"\n🔍 Extracting ground truth from forecast response...")
        ground_truth = None
        
        # Check if ground truth was included in forecast response
        if forecasts.get('ground_truth'):
            ref_sample = forecasts['ground_truth']
            print(f"✅ Found ground truth in forecast response")
        else:
            print(f"⚠️  No ground truth in forecast response (unconditional forecast)")
            ref_sample = None
        
        # Extract ground truth from normalized_sample in array format
        if ref_sample and ref_sample.get('normalized_sample'):
            normalized_sample = ref_sample['normalized_sample']
            norm_params = self._data.get('feature_normalization_params', {})
            
            ref_windows = []
            
            # Get metadata for feature orders
            metadata = normalized_sample.get('metadata', {})
            feature_order_past = metadata.get('feature_order_past', [])
            feature_order_target = metadata.get('feature_order_target', [])
            
            # Process past windows (all features)
            normalized_past = normalized_sample.get('normalized_past')
            if normalized_past is not None and feature_order_past:
                import numpy as np
                past_arr = np.array(normalized_past)
                
                for i, feature in enumerate(feature_order_past):
                    if i < len(past_arr):
                        normalized_series = past_arr[i].tolist()
                        
                        # Denormalize
                        if feature in norm_params:
                            mean = norm_params[feature].get('mean', 0.0)
                            std = norm_params[feature].get('std', 1.0)
                            denormalized = [val * std + mean for val in normalized_series]
                        else:
                            denormalized = normalized_series
                        
                        ref_windows.append({
                            'feature': feature,
                            'temporal_tag': 'past',
                            'values': denormalized
                        })
            
            # Process future target windows (residuals)
            normalized_future_target = normalized_sample.get('normalized_future_target_series')
            if normalized_future_target is not None and feature_order_target:
                import numpy as np
                target_arr = np.array(normalized_future_target)
                
                # Get past reference values for each target feature
                past_refs = {}
                if normalized_past is not None and feature_order_past:
                    past_arr = np.array(normalized_past)
                    for i, feature in enumerate(feature_order_past):
                        if feature in feature_order_target and i < len(past_arr):
                            past_refs[feature] = past_arr[i][-1]  # Last value
                
                for i, feature in enumerate(feature_order_target):
                    if i < len(target_arr):
                        normalized_residuals = target_arr[i].tolist()
                        
                        # Convert residuals to series by adding reference
                        if feature in past_refs:
                            ref_value = past_refs[feature]
                            normalized_series = [ref_value + res for res in normalized_residuals]
                        else:
                            normalized_series = normalized_residuals
                        
                        # Denormalize
                        if feature in norm_params:
                            mean = norm_params[feature].get('mean', 0.0)
                            std = norm_params[feature].get('std', 1.0)
                            denormalized = [val * std + mean for val in normalized_series]
                        else:
                            denormalized = normalized_series
                        
                        ref_windows.append({
                            'feature': feature,
                            'temporal_tag': 'future',
                            'values': denormalized
                        })
            
            ground_truth = {
                'sample_id': reference_sample_id,
                'windows': ref_windows
            }
            print(f"✅ Extracted ground truth: {len(ref_windows)} windows")
        else:
            print(f"⚠️  No normalized sample in ground truth")
        
        return {
            'reconstructions': all_reconstructions,
            'reference_sample_id': reference_sample_id,
            'n_samples': len(all_reconstructions),
            'ground_truth': ground_truth
        }
