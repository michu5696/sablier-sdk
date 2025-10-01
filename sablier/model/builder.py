"""Model class representing a Sablier model"""

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
        target_features: List[str],
        stride: int = 10,
        conditioning_features: Optional[List[str]] = None,
        splits: Optional[Dict[str, Dict[str, str]]] = None,
        confirm: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Generate training samples from processed data
        
        Args:
            past_window: Past window size (days)
            future_window: Future window size (days)
            target_features: Features to predict (REQUIRED, must be subset of input features)
            stride: Stride between samples (days, default: 10)
            conditioning_features: Features for conditioning (optional, defaults to all non-target features)
            splits: Train/validation/test date ranges (optional, auto-calculated if not provided)
                Example: {
                    "training": {"start": "2020-01-01", "end": "2023-12-31"},
                    "validation": {"start": "2024-01-01", "end": "2024-06-30"},
                    "test": {"start": "2024-07-01", "end": "2024-12-31"}
                }
            confirm: Explicit confirmation (None = prompt if needed)
            
        Returns:
            dict: Generation statistics with keys: status, samples_generated, split_counts
            
        Example:
            >>> model.generate_samples(
            ...     past_window=90,
            ...     future_window=30,
            ...     target_features=["Gold Price"],
            ...     stride=10
            ... )
        """
        # Check for conflicts
        if not self._check_and_handle_conflict("generate_samples", confirm):
            return {"status": "cancelled"}
        
        print(f"[Model {self.name}] Generating samples...")
        print(f"  Past window: {past_window} days")
        print(f"  Future window: {future_window} days")
        print(f"  Stride: {stride} days")
        
        # Validate inputs
        validate_sample_generation_inputs(
            self.input_features,
            past_window, 
            future_window, 
            target_features, 
            conditioning_features
        )
        
        # Auto-assign conditioning features if not provided
        all_feature_names = [f.get('display_name', f.get('name')) for f in self.input_features]
        if conditioning_features is None:
            conditioning_features = [f for f in all_feature_names if f not in target_features]
            print(f"  Auto-assigned {len(conditioning_features)} conditioning features")
        
        # Auto-generate splits if not provided
        if splits is None:
            sample_size = past_window + future_window
            start = self._data.get('training_start_date')
            end = self._data.get('training_end_date')
            splits = auto_generate_splits(start, end, sample_size=sample_size)
            print(f"  Auto-generated splits with {sample_size}-day gap")
        
        # Validate splits
        validate_splits(splits, past_window, future_window)
        
        # Build sample config
        sample_config = {
            "pastWindow": past_window,
            "futureWindow": future_window,
            "stride": stride,
            "splits": splits,
            "conditioningFeatures": conditioning_features,
            "targetFeatures": target_features
        }
        
        # Build request payload
        payload = {
            "user_id": self._data.get("user_id"),  # For backend
            "model_id": self.id,
            "sample_config": sample_config
        }
        
        # Call backend
        print("📡 Calling backend to generate samples...")
        response = self.http.post('/api/v1/ml/generate-samples', payload)
        
        # Update model status
        self._data["status"] = "samples_generated"
        self._data["sample_config"] = sample_config
        
        # Update input_features with types
        updated_features = update_feature_types(
            self.http, 
            self.id, 
            self.input_features, 
            conditioning_features, 
            target_features
        )
        self._data['input_features'] = updated_features
        
        split_counts = response.get('split_counts', {})
        print(f"✅ Generated {response.get('samples_generated', 0)} samples")
        print(f"   Training: {split_counts.get('training', 0)}")
        print(f"   Validation: {split_counts.get('validation', 0)}")
        print(f"   Test: {split_counts.get('test', 0)}")
        
        return response
    
    # ============================================
    # ENCODING
    # ============================================
    
    def encode_samples(
        self, 
        encoding_type: str = "pca-ica",
        split: str = "training",
        confirm: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Fit encoding models and encode all samples
        
        This performs a two-step process:
        1. Fits PCA-ICA (or UMAP) encoding models on specified split
        2. Encodes all samples using the fitted models
        
        Args:
            encoding_type: Type of encoding ("pca-ica" or "umap", default: "pca-ica")
            split: Which split to use for fitting ("training", "validation", 
                   "training+validation", or "all", default: "training")
            confirm: Explicit confirmation (None = prompt if needed)
            
        Returns:
            dict: {
                "status": "success",
                "models_fitted": int,
                "samples_encoded": int,
                "features_processed": int,
                "split_used": str
            }
            
        Example:
            >>> # Basic usage - fit and encode using PCA-ICA on training split
            >>> result = model.encode_samples()
            >>> print(f"Fitted {result['models_fitted']} encoding models")
            >>> print(f"Encoded {result['samples_encoded']} samples")
            
            >>> # Use UMAP encoding (requires PCA-ICA to be fitted first)
            >>> result = model.encode_samples(encoding_type="umap")
            
            >>> # Fit on combined training+validation data
            >>> result = model.encode_samples(split="training+validation")
        """
        # Validate encoding_type
        if encoding_type not in ["pca-ica", "umap"]:
            raise ValueError(f"Invalid encoding_type: {encoding_type}. Must be 'pca-ica' or 'umap'")
        
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
                "encoding_type": encoding_type
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
    # TRAINING
    # ============================================
    
    def train(self, config: Optional[Dict[str, Any]] = None, confirm: Optional[bool] = None) -> Dict[str, Any]:
        """
        Train the model on encoded samples
        
        This method:
        1. Trains a Quantile Regression Forest on encoded conditioning/target data
        2. Uses future feature augmentation for robustness
        3. Computes SHAP feature importance
        4. Saves trained model to storage
        
        Args:
            config: Optional model configuration. Defaults to:
                {
                    'n_estimators': 200,
                    'max_depth': 20,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                }
            confirm: Explicit confirmation (None = prompt if needed)
            
        Returns:
            dict: {
                "status": "success",
                "model_id": str,
                "training_metrics": {
                    "train_mse": float,
                    "val_mse": float,
                    "augmented_training_samples": int,
                    "augmented_validation_samples": int
                },
                "model_metadata": dict,
                "feature_importance": dict,
                "component_breakdown": dict,
                "categories": dict
            }
            
        Example:
            >>> # Basic usage - train with default config
            >>> result = model.train()
            
            >>> # Custom config
            >>> result = model.train(config={
            ...     'n_estimators': 500,
            ...     'max_depth': 30
            ... })
        """
        # Validate model status
        if self.status != "samples_encoded":
            print(f"❌ Model must be in 'samples_encoded' status to train (current: {self.status})")
            print("   Run model.encode_samples() first")
            return {"status": "error", "message": "Model not ready for training"}
        
        # Use default config if not provided
        default_config = {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        
        final_config = {**default_config, **(config or {})}
        
        print(f"[Model {self.name}] Training model...")
        print(f"  Config: n_estimators={final_config['n_estimators']}, max_depth={final_config['max_depth']}, "
              f"min_samples_split={final_config['min_samples_split']}")
        print()
        
        # Build payload
        payload = {
            "user_id": self._data.get("user_id"),
            "model_id": self.id,
            "qrf_config": final_config  # Backend expects qrf_config key
        }
        
        # Call backend
        print("📡 Step 1/2: Training model on encoded samples...")
        try:
            response = self.http.post('/api/v1/ml/train-model', payload)
        except Exception as e:
            print(f"  ❌ Training failed: {e}")
            raise
        
        # Extract metrics
        training_metrics = response.get('training_metrics', {})
        
        print("  ✅ Model training complete")
        print(f"     Training MSE: {training_metrics.get('train_mse', 0):.6f}")
        print(f"     Validation MSE: {training_metrics.get('val_mse', 0):.6f}")
        print(f"     Augmented training samples: {training_metrics.get('augmented_training_samples', 0)}")
        print(f"     Augmented validation samples: {training_metrics.get('augmented_validation_samples', 0)}")
        print()
        
        print("📊 Step 2/2: Computing feature importance (SHAP)...")
        feature_importance = response.get('feature_importance', {})
        categories = response.get('categories', {})
        
        print("  ✅ Feature importance computed")
        if categories:
            total_features = sum(len(v) for v in categories.values())
            print(f"     Total features analyzed: {total_features}")
            print(f"     Categories: {', '.join(categories.keys())}")
        print()
        
        # Update model status
        self._data["status"] = "trained"
        self._data["training_metrics"] = training_metrics
        self._data["model_metadata"] = response.get('model_metadata', {})
        
        print(f"✅ Training complete - model saved to storage")
        
        return {
            "status": "success",
            "model_id": response.get('model_id'),
            "training_metrics": training_metrics,
            "model_metadata": response.get('model_metadata', {}),
            "feature_importance": feature_importance,
            "component_breakdown": response.get('component_breakdown', {}),
            "categories": categories,
            "per_sample_importance": response.get('per_sample_importance', {})
        }
    
    # ============================================
    # FORECASTING
    # ============================================
    
    def generate_forecast(
        self,
        sample_id: Optional[str] = None,
        scenario = None,  # Scenario instance
        conditioning_features: Optional[List[str]] = None,
        n_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Generate forecast samples
        
        Two usage modes:
        
        1. Post-Training Validation (uses test samples):
           >>> # Use specific test sample
           >>> forecast = model.generate_forecast(
           ...     sample_id="test_sample_123",
           ...     conditioning_features=["Feature A"],  # Optional
           ...     n_samples=100
           ... )
           
           >>> # Auto-select first test sample
           >>> forecast = model.generate_forecast(n_samples=100)
        
        2. Scenario-Based Generation (called by Scenario):
           >>> forecast = model.generate_forecast(
           ...     scenario=scenario_instance,
           ...     n_samples=1000
           ... )
        
        Args:
            sample_id: Test sample ID (optional, auto-selects if None)
            scenario: Scenario instance for scenario-based generation
            conditioning_features: Which features to condition on (for sample mode only)
            n_samples: Number of forecast samples to generate
            
        Returns:
            dict: {
                "status": "success",
                "forecast_samples": List[Dict],
                "distribution_params": dict,
                "n_samples": int
            }
        """
        # Validate model status
        if self.status != "trained":
            print(f"❌ Model must be trained to generate forecasts (current status: {self.status})")
            print("   Run model.train() first")
            return {"status": "error", "message": "Model not trained"}
        
        print(f"[Model {self.name}] Generating forecast...")
        
        # Determine mode and build payload
        if scenario is not None:
            # Scenario mode
            print(f"  Mode: Scenario-based (scenario: {scenario.name})")
            print(f"  Generating {n_samples} synthetic paths...")
            
            payload = {
                "user_id": self._data.get("user_id"),
                "model_id": self.id,
                "conditioning_source": "scenario",
                "scenario_id": scenario.id,
                "n_samples": n_samples
            }
            
        else:
            # Sample validation mode
            # Auto-select test sample if not provided
            if sample_id is None:
                print("  Auto-selecting test sample...")
                sample_id = self._get_first_test_sample_id()
                if not sample_id:
                    print("❌ No test samples found")
                    return {"status": "error", "message": "No test samples available"}
                print(f"  Selected sample: {sample_id}")
            
            print(f"  Mode: Sample validation")
            print(f"  Sample: {sample_id}")
            if conditioning_features:
                print(f"  Conditioning on: {', '.join(conditioning_features)}")
            else:
                print(f"  Conditioning on: all future features")
            print(f"  Generating {n_samples} forecast samples...")
            
            payload = {
                "user_id": self._data.get("user_id"),
                "model_id": self.id,
                "conditioning_source": "sample",
                "sample_id": sample_id,
                "conditioning_features": conditioning_features,
                "n_samples": n_samples
            }
        
        # Call backend
        try:
            response = self.http.post('/api/v1/ml/forecast', payload)
        except Exception as e:
            print(f"  ❌ Forecast generation failed: {e}")
            raise
        
        forecast_samples = response.get('forecast_samples', [])
        
        print(f"✅ Generated {len(forecast_samples)} forecast samples")
        print(f"   Distribution: {response.get('distribution_params', {}).get('method', 'unknown')}")
        
        return {
            "status": "success",
            "forecast_samples": forecast_samples,
            "distribution_params": response.get('distribution_params', {}),
            "n_samples": len(forecast_samples)
        }
    
    def _get_first_test_sample_id(self) -> Optional[str]:
        """Get first test sample ID"""
        try:
            # Query backend for test samples
            response = self.http.get(
                f'/api/v1/models/{self.id}/samples',
                params={'split_type': 'test', 'limit': 1}
            )
            
            samples = response.get('samples', [])
            if samples and len(samples) > 0:
                return samples[0]['id']
            
            return None
        except Exception as e:
            print(f"  ⚠️  Failed to auto-select test sample: {e}")
            return None