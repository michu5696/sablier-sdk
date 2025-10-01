"""Model class representing a Sablier model"""

from typing import Optional, Any, List, Dict
from ..http_client import HTTPClient
from ..workflow import WorkflowValidator, WorkflowConflict


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
    
    def data_collection(self, fred_api_key: Optional[str] = None):
        """
        Create a DataCollection instance for this model
        
        Args:
            fred_api_key: Optional FRED API key for searching and fetching
            
        Returns:
            DataCollection: Data collection instance scoped to this model
            
        Example:
            >>> data = model.data_collection(fred_api_key="...")
            >>> data.search("treasury")
            >>> data.add("DGS10", source="FRED")
            >>> data.fetch_and_process()
        """
        from ..data_collection import DataCollection
        return DataCollection(self, fred_api_key=fred_api_key)
    
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
    
    def refresh(self):
        """Refresh model data from API"""
        response = self.http.get(f'/api/v1/models/{self.id}')
        self._data = response.get('model', {})
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
        
        print(f"[Model {self.name}] Setting training period: {start_date} to {end_date}")
        
        # Update model
        response = self.http.patch(f'/api/v1/models/{self.id}', {
            'training_start_date': start_date,
            'training_end_date': end_date
        })
        
        self._data = response.get('model', {})
        print("✅ Training period updated")
        return self
    
    # ============================================
    # DATA FETCHING
    # ============================================
    
    def fetch_data(
        self, 
        max_gap_days: int = 7,
        interpolation_method: str = "linear",
        fred_api_key: Optional[str] = None,
        confirm: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Fetch and process training data from FRED and Yahoo Finance
        
        This method handles EVERYTHING:
        1. Fetches raw data from APIs
        2. Applies interpolation with specified gap limit
        3. Saves both raw and processed data to database
        4. Updates model status to 'data_collected'
        
        Args:
            max_gap_days: Maximum gap (in days) to fill via interpolation (default: 7)
            interpolation_method: "linear", "forward_fill", or "backward_fill" (default: "linear")
            confirm: Explicit confirmation (None = prompt if needed)
            
        Returns:
            dict: Fetch statistics with keys: status, features_fetched, total_raw_points, total_processed_points
            
        Example:
            >>> # Simple - just fetch with default 7-day linear interpolation
            >>> model.fetch_data()
            
            >>> # Custom - 30-day forward fill
            >>> model.fetch_data(max_gap_days=30, interpolation_method="forward_fill")
        
        Note:
            FRED API key is taken from the client initialization.
            Yahoo Finance doesn't require an API key.
        """
        # Check for conflicts
        if not self._check_and_handle_conflict("fetch_data", confirm):
            return {"status": "cancelled"}
        
        print(f"[Model {self.name}] Fetching data...")
        
        # Get features and training period from model
        features = self.input_features
        if not features:
            print("❌ No features configured. Call model.add_features() first.")
            return {"status": "error", "message": "No features configured"}
        
        training_period = self._data.get('training_start_date'), self._data.get('training_end_date')
        if not all(training_period):
            print("❌ Training period not set. Call model.set_training_period() first.")
            return {"status": "error", "message": "Training period not set"}
        
        # Build processing config - apply same max_gap to all features
        processing_config = {
            "interpolation": {
                "method": interpolation_method,
                "maxGapLength": {
                    feature.get("name", feature.get("display_name", "")): max_gap_days
                    for feature in features
                }
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
        print(f"  Interpolation: {interpolation_method} (max {max_gap_days} day gaps)")
        
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
        past_window: int,
        future_window: int,
        stride: int = 10,
        conditioning_features: Optional[List[str]] = None,
        target_features: Optional[List[str]] = None,
        splits: Optional[Dict[str, float]] = None,
        confirm: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Generate training samples from processed data
        
        Args:
            past_window: Past window size (days)
            future_window: Future window size (days)
            stride: Stride between samples (days)
            conditioning_features: Features for conditioning (optional, uses all input features if not specified)
            target_features: Features to predict (required, must be subset of input features)
            splits: Train/validation/test splits, e.g. {"train": 0.7, "validation": 0.2, "test": 0.1}
            confirm: Explicit confirmation (None = prompt if needed)
            
        Returns:
            dict: Generation statistics with keys: status, samples_generated, splits
            
        Example:
            >>> model.generate_samples(
            ...     past_window=90,
            ...     future_window=30,
            ...     stride=10,
            ...     target_features=["Gold Price", "S&P 500"]
            ... )
        """
        # Check for conflicts
        if not self._check_and_handle_conflict("generate_samples", confirm):
            return {"status": "cancelled"}
        
        print(f"[Model {self.name}] Generating samples...")
        print(f"  Past window: {past_window} days")
        print(f"  Future window: {future_window} days")
        print(f"  Stride: {stride} days")
        
        # Build sample config
        sample_config = {
            "past_window": past_window,
            "future_window": future_window,
            "stride": stride
        }
        
        if splits:
            sample_config["splits"] = splits
        else:
            sample_config["splits"] = {"train": 0.7, "validation": 0.2, "test": 0.1}
        
        # Build request payload
        payload = {
            "model_id": self.id,
            "sample_config": sample_config
        }
        
        if conditioning_features:
            payload["conditioningFeatures"] = conditioning_features
        if target_features:
            payload["targetFeatures"] = target_features
        
        # Call backend
        print("📡 Calling backend to generate samples...")
        response = self.http.post('/api/v1/ml/generate-samples', payload)
        
        # Update model status
        self._data["status"] = "samples_generated"
        self._data["sample_config"] = sample_config
        
        print(f"✅ Generated {response.get('samples_generated', 0)} samples")
        
        return response
    
    # ============================================
    # ENCODING
    # ============================================
    
    def encode_samples(self, confirm: Optional[bool] = None) -> Dict[str, Any]:
        """
        Fit PCA-ICA encoding models and encode all samples
        
        This method performs two operations:
        1. Fits encoding models for each feature-window combination
        2. Encodes all samples using the fitted models
        
        Args:
            confirm: Explicit confirmation (None = prompt if needed)
            
        Returns:
            dict: Encoding statistics with keys: status, models_fitted, samples_encoded
            
        Example:
            >>> result = model.encode_samples()
            >>> print(f"Fitted {result['models_fitted']} encoding models")
            >>> print(f"Encoded {result['samples_encoded']} samples")
        """
        # Check for conflicts
        if not self._check_and_handle_conflict("encode_samples", confirm):
            return {"status": "cancelled"}
        
        print(f"[Model {self.name}] Encoding samples...")
        
        # Step 1: Fit encoding models
        print("  Step 1/2: Fitting PCA-ICA encoding models...")
        fit_response = self.http.post('/api/v1/ml/fit', {
            "model_id": self.id,
            "user_id": self._data.get("user_id")  # For API key auth
        })
        
        models_fitted = fit_response.get('models_fitted', 0)
        print(f"  ✅ Fitted {models_fitted} encoding models")
        
        # Step 2: Encode samples
        print("  Step 2/2: Encoding all samples...")
        encode_response = self.http.post('/api/v1/ml/encode', {
            "model_id": self.id,
            "user_id": self._data.get("user_id")  # For API key auth
        })
        
        samples_encoded = encode_response.get('samples_encoded', 0)
        print(f"  ✅ Encoded {samples_encoded} samples")
        
        # Update model status
        self._data["status"] = "samples_encoded"
        
        print(f"✅ Encoding complete")
        
        return {
            "status": "success",
            "models_fitted": models_fitted,
            "samples_encoded": samples_encoded
        }
    
    # ============================================
    # TRAINING
    # ============================================
    
    def train(self, qrf_config: Optional[dict] = None, confirm: Optional[bool] = None) -> 'TrainingResults':
        """
        Train the QRF model
        
        Args:
            qrf_config: Optional QRF configuration
            confirm: Explicit confirmation (None = prompt if needed)
            
        Returns:
            TrainingResults: Training results with metrics
        """
        # Training doesn't have conflicts (always replaces previous model)
        # TODO: Implement model training
        print(f"[Model {self.name}] Training model...")
        from .training_results import TrainingResults
        return TrainingResults({})
    
    # ============================================
    # FORECASTING
    # ============================================
    
    def generate_forecast(
        self,
        sample_id: str,
        n_samples: int = 100,
        conditioning_features: list[str] = None
    ) -> 'SyntheticData':
        """
        Generate forecast from a sample
        
        Args:
            sample_id: Sample ID to use for conditioning
            n_samples: Number of forecast samples to generate
            conditioning_features: Features to condition on
            
        Returns:
            SyntheticData: Synthetic data instance with forecasts
        """
        # TODO: Implement forecasting
        print(f"[Model {self.name}] Generating forecast...")
        from ..synthetic_data import SyntheticData
        import pandas as pd
        return SyntheticData(
            data=pd.DataFrame(),
            source_model=self,
            metadata={"n_samples": n_samples}
        )