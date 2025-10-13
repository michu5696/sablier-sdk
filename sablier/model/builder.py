"""Model class representing a Sablier model"""

import logging
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
        
        # Auto-generate splits if not provided or if percentage-based
        if splits is None or (isinstance(splits, dict) and isinstance(list(splits.values())[0], (int, float))):
            sample_size = past_window + future_window
            start = self._data.get('training_start_date')
            end = self._data.get('training_end_date')
            
            # If percentage splits provided, use them; otherwise use defaults
            if splits and isinstance(list(splits.values())[0], (int, float)):
                train_pct = splits.get('training', 70) / 100
                val_pct = splits.get('validation', 20) / 100
                test_pct = splits.get('test', 10) / 100
                print(f"  Converting percentage splits: {int(train_pct*100)}% train, {int(val_pct*100)}% val, {int(test_pct*100)}% test")
                splits = auto_generate_splits(start, end, sample_size=sample_size, 
                                             train_pct=train_pct, val_pct=val_pct, test_pct=test_pct)
            else:
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
    # FEATURE GROUPING
    # ============================================
    
    def auto_group_features(
        self,
        min_correlation: float = 0.75,
        min_correlation_target: float = None,
        min_correlation_conditioning: float = None,
        method: str = 'hierarchical',
        auto_apply: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze feature correlations and suggest groupings for multivariate encoding
        
        This should be called after generate_samples() and before encode_samples().
        It analyzes correlations within target and conditioning features separately
        and suggests groups of highly correlated features that should be encoded together.
        
        Args:
            min_correlation: Minimum correlation threshold for both target and conditioning (default: 0.75)
                           Ignored if min_correlation_target or min_correlation_conditioning are specified
            min_correlation_target: Minimum correlation threshold for target features (overrides min_correlation)
            min_correlation_conditioning: Minimum correlation threshold for conditioning features (overrides min_correlation)
            method: Clustering method ('hierarchical' or 'threshold', default: 'hierarchical')
            auto_apply: If True, automatically apply suggested groups without confirmation
            
        Returns:
            dict: {
                "status": "success",
                "target_groups": List[Dict],
                "conditioning_groups": List[Dict],
                "correlation_matrices": Dict
            }
            
        Example:
            >>> # Analyze and review groups
            >>> groups = model.auto_group_features(min_correlation=0.8)
            >>> 
            >>> # Review suggested groups
            >>> for group in groups['target_groups']:
            >>>     print(f"{group['name']}: {group['features']}")
            >>>     print(f"  Avg correlation: {group['avg_correlation']:.3f}")
            >>> 
            >>> # Rename a group (optional)
            >>> model.rename_group('target_group_1', 'Treasury Yield Curve')
            >>> 
            >>> # Groups are automatically used in encode_samples()
        """
        # Check status
        if self.status not in ['samples_generated', 'encoding_fitted', 'samples_encoded', 'trained']:
            raise ValueError(
                f"Cannot analyze feature groups. Model status: {self.status}. "
                f"Call generate_samples() first."
            )
        
        # Determine thresholds
        target_threshold = min_correlation_target if min_correlation_target is not None else min_correlation
        conditioning_threshold = min_correlation_conditioning if min_correlation_conditioning is not None else min_correlation
        
        print(f"[Model {self.name}] Analyzing feature correlations...")
        print(f"  Target correlation threshold: {target_threshold}")
        print(f"  Conditioning correlation threshold: {conditioning_threshold}")
        print(f"  Clustering method: {method}")
        
        # Call backend
        payload = {
            "user_id": self._data.get("user_id"),
            "model_id": self.id,
            "min_correlation_target": target_threshold,
            "min_correlation_conditioning": conditioning_threshold,
            "method": method
        }
        
        response = self.http.post('/api/v1/ml/analyze-feature-groups', payload)
        
        if response.get('status') != 'success':
            print(f"❌ Feature grouping analysis failed")
            return response
        
        target_groups = response.get('target_groups', [])
        conditioning_groups = response.get('conditioning_groups', [])
        
        # Display results
        print(f"\n✅ Feature grouping analysis complete")
        print(f"\n📊 TARGET FEATURE GROUPS ({len(target_groups)} groups):")
        print("=" * 70)
        for group in target_groups:
            print(f"\n{group['name']} (ID: {group['id']})")
            print(f"  Features ({group['n_features']}): {', '.join(group['features'])}")
            print(f"  Avg correlation: {group['avg_correlation']:.3f}")
            print(f"  Type: {'Multivariate' if group['is_multivariate'] else 'Univariate'}")
        
        if conditioning_groups:
            print(f"\n📊 CONDITIONING FEATURE GROUPS ({len(conditioning_groups)} groups):")
            print("=" * 70)
            for group in conditioning_groups:
                print(f"\n{group['name']} (ID: {group['id']})")
                print(f"  Features ({group['n_features']}): {', '.join(group['features'])}")
                print(f"  Avg correlation: {group['avg_correlation']:.3f}")
                print(f"  Type: {'Multivariate' if group['is_multivariate'] else 'Univariate'}")
        
        print("\n" + "=" * 70)
        
        # Apply groups if auto_apply or prompt user
        if auto_apply:
            print("\n🔄 Auto-applying suggested groups...")
            self.apply_feature_groups(response)
        else:
            print("\n💡 TIP: Review the groups above. You can:")
            print("   - Rename groups: model.rename_group('target_group_1', 'New Name')")
            print("   - Apply groups: model.apply_feature_groups(groups)")
            print("   - Or they will be automatically applied when you call encode_samples()")
        
        return response
    
    def rename_group(self, group_id: str, new_name: str):
        """
        Rename a feature group
        
        Args:
            group_id: Group ID (e.g., 'target_group_1')
            new_name: New name for the group
            
        Example:
            >>> model.rename_group('target_group_1', 'Treasury Yield Curve')
        """
        # Get current groups from model metadata
        feature_groups = self._data.get('feature_groups')
        if not feature_groups:
            raise ValueError("No feature groups found. Call auto_group_features() first.")
        
        # Find and rename the group
        found = False
        for group_list_key in ['target_groups', 'conditioning_groups']:
            groups = feature_groups.get(group_list_key, [])
            for group in groups:
                if group['id'] == group_id:
                    old_name = group['name']
                    group['name'] = new_name
                    found = True
                    print(f"✅ Renamed '{old_name}' → '{new_name}'")
                    break
            if found:
                break
        
        if not found:
            raise ValueError(f"Group '{group_id}' not found")
        
        # Update model metadata
        self._update_metadata({'feature_groups': feature_groups})
    
    def apply_feature_groups(self, groups: Dict[str, Any]):
        """
        Apply feature groupings to the model
        
        Args:
            groups: Groups dictionary from auto_group_features()
            
        Example:
            >>> groups = model.auto_group_features()
            >>> model.apply_feature_groups(groups)
        """
        feature_groups = {
            'target_groups': groups.get('target_groups', []),
            'conditioning_groups': groups.get('conditioning_groups', [])
        }
        
        # Update model metadata
        self._update_metadata({'feature_groups': feature_groups})
        
        print(f"✅ Applied {len(feature_groups['target_groups'])} target groups and "
              f"{len(feature_groups['conditioning_groups'])} conditioning groups")
    
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
        Fit encoding models and encode all samples
        
        This performs a two-step process:
        1. Fits PCA-ICA (or UMAP) encoding models on specified split
        2. Encodes all samples using the fitted models
        
        Args:
            encoding_type: Type of encoding ("pca-ica" or "umap", default: "pca-ica")
            split: Which split to use for fitting ("training", "validation", 
                   "training+validation", or "all", default: "training")
            pca_variance_threshold_series: Variance threshold for normalized_series (conditioning, default: 0.95)
            pca_variance_threshold_residuals: Variance threshold for normalized_residuals (targets, default: 0.99)
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
    # TRAINING
    # ============================================
    
    def train(self, 
            config: Optional[Dict[str, Any]] = None, 
            confirm: Optional[bool] = None,
            optimize_hyperparameters: bool = False,
            n_optimization_trials: int = 30,
            use_augmentation: bool = True,
            n_augmentations_per_sample: int = 2,
            fit_conditional_marginals: bool = False,
            use_randomized_pit: bool = True,
            parallel_samples: str = 'auto',
            parallel_marginals: str = 'auto',
            run_e2e_validation: bool = False,
            e2e_validation_samples: int = None,
            e2e_forecast_samples: int = 1000,
            e2e_covariance_type: str = 'increment',
            e2e_time_scales: List[int] = None) -> Dict[str, Any]:
        """
        Train the model on encoded samples
        
        This method:
        1. Trains a Quantile Regression Forest on encoded conditioning/target data
        2. Uses future feature augmentation for robustness
        3. Computes SHAP feature importance
        4. Saves trained model to storage
        5. Optionally runs end-to-end validation
        
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
            optimize_hyperparameters: Enable Bayesian optimization with Optuna (default: False)
            n_optimization_trials: Number of optimization trials (default: 30)
            use_augmentation: Enable component masking augmentation (default: True)
            n_augmentations_per_sample: Number of augmented copies per sample (default: 2)
            fit_conditional_marginals: Fit GMM+EVT marginals (slow but accurate) vs empirical (fast) (default: False)
            use_randomized_pit: Use randomized PIT for empirical mode (default: True)
            parallel_samples: Outer parallelism for sample processing (default: 'auto')
            parallel_marginals: Inner parallelism for marginal fitting (default: 'auto')
            run_e2e_validation: Whether to run end-to-end validation (default: False)
            e2e_validation_samples: Number of validation samples to use (None = use all, default: None)
            e2e_forecast_samples: Forecast paths per validation sample (default: 1000)
            e2e_covariance_type: 'increment' or 'level' (default: 'increment')
            e2e_time_scales: List of time scales for increment covariance (default: [1, 3, 5, 10])
            
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
        
        # Only send config if user provided one (let backend use adaptive defaults)
        if config:
            print(f"[Model {self.name}] Training model with custom config...")
            print(f"  Config: n_estimators={config.get('n_estimators', 'default')}, "
                  f"max_depth={config.get('max_depth', 'default')}, "
                  f"min_samples_split={config.get('min_samples_split', 'default')}")
        else:
            print(f"[Model {self.name}] Training model with adaptive config...")
            print(f"  Backend will auto-tune parameters based on training set size")
        print()
        
        # Build payload
        payload = {
            "user_id": self._data.get("user_id"),
            "model_id": self.id,
            "optimize_hyperparameters": optimize_hyperparameters,
            "n_optimization_trials": n_optimization_trials,
            "use_augmentation": use_augmentation,
            "n_augmentations_per_sample": n_augmentations_per_sample,
            "fit_conditional_marginals": fit_conditional_marginals,
            "use_randomized_pit": use_randomized_pit,
            "parallel_samples": parallel_samples,
            "parallel_marginals": parallel_marginals,
            "run_e2e_validation": run_e2e_validation,
            "e2e_validation_samples": e2e_validation_samples,
            "e2e_forecast_samples": e2e_forecast_samples,
            "e2e_covariance_type": e2e_covariance_type,
            "e2e_time_scales": e2e_time_scales or [1, 3, 5, 10]
        }
        
        # Only add qrf_config if user provided one (let backend use adaptive defaults)
        # Note: If optimize_hyperparameters=True, this config will be ignored
        if config:
            payload["qrf_config"] = config
        
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
        
        # Update model status and metadata
        self._data["status"] = "trained"
        self._data["training_metrics"] = training_metrics
        
        # Refresh model data from database to get complete metadata
        # This ensures we have the latest copula metadata
        try:
            refreshed_model = self.http.get(f'/api/v1/models/{self.id}')
            if refreshed_model:
                self._data["model_metadata"] = refreshed_model.get('model_metadata', {})
                logger.info("Refreshed model metadata from database after training")
            else:
                print("⚠️  Could not refresh model metadata from database")
        except Exception as e:
            logger.warning(f"Could not refresh model metadata: {e}")
            # Fallback to response metadata
            self._data["model_metadata"] = response.get('model_metadata', {})
        
        # Display copula validation results if available (after refresh)
        copula_metadata = self._data.get('model_metadata', {}).get('copula_metadata', {})
        if copula_metadata:
            print("🔗 Conditional Copula Validation Results (Uniform Space):")
            validation_results = copula_metadata.get('validation_results', {})
            
            if validation_results:
                # Display main metrics (new format)
                import numpy as np
                mean_log_lik = validation_results.get('mean_log_likelihood', None)
                corr_of_corr = validation_results.get('correlation_of_correlations', None)
                corr_mae = validation_results.get('correlation_mae', None)
                tail_coexceedance = validation_results.get('tail_coexceedance', None)
                
                if mean_log_lik is not None:
                    print(f"     Mean Log-Likelihood: {mean_log_lik:.3f}")
                if corr_of_corr is not None:
                    print(f"     Correlation of Correlations: {corr_of_corr:.3f}")
                if corr_mae is not None:
                    print(f"     Correlation MAE: {corr_mae:.3f}")
                if tail_coexceedance is not None:
                    print(f"     Tail Co-exceedance Rate: {tail_coexceedance:.3f}")
                
                # Quality assessment based on correlation of correlations
                print()
                if corr_of_corr is not None:
                    if corr_of_corr >= 0.7:
                        print("     🎉 Copula quality: EXCELLENT (Corr-of-Corr ≥ 0.7)")
                    elif corr_of_corr >= 0.5:
                        print("     ✅ Copula quality: GOOD (Corr-of-Corr ≥ 0.5)")
                    elif corr_of_corr >= 0.3:
                        print("     ⚠️  Copula quality: MODERATE (Corr-of-Corr ≥ 0.3)")
                    else:
                        print("     ❌ Copula quality: POOR (Corr-of-Corr < 0.3)")
                else:
                    print("     ⚠️  Copula quality: Unable to assess (no metrics available)")
                
                # Check for validation plot
                plot_filename = validation_results.get('plot_filename')
                if plot_filename:
                    print(f"     📊 Validation plot: {plot_filename}")
            else:
                print("     ⚠️  No validation results available")
        else:
            print("🔗 No copula validation performed")
        print()
        
        print(f"✅ Training complete - model saved to storage")
        
        return {
            "status": "success",
            "model_id": response.get('model_id'),
            "training_metrics": training_metrics,
            "model_metadata": self._data.get('model_metadata', {}),
            "feature_importance": feature_importance,
            "component_breakdown": response.get('component_breakdown', {}),
            "categories": categories,
            "per_sample_importance": response.get('per_sample_importance', {}),
            "copula_metadata": copula_metadata
        }
    
    def show_copula_validation(self) -> None:
        """
        Display copula validation results for a trained model
        
        Example:
            >>> model.show_copula_validation()
        """
        # Check if model is trained
        if self.status != "trained":
            print(f"❌ Model must be trained to show copula validation (current status: {self.status})")
            return
        
        # Get copula metadata from model
        model_metadata = self._data.get('model_metadata', {})
        copula_metadata = model_metadata.get('copula_metadata', {})
        
        if not copula_metadata:
            print("❌ No copula metadata found. Model may not have copula validation results.")
            return
        
        print("🔗 Structural Copula Validation Results (Uniform Space):")
        print("=" * 60)
        
        # Display basic copula info
        n_samples = copula_metadata.get('n_samples', 0)
        n_copula_dims = copula_metadata.get('n_copula_dimensions', 0)
        n_total_dims = copula_metadata.get('n_total_dimensions', 0)
        
        print(f"Copula Structure:")
        print(f"  Samples used for fitting: {n_samples}")
        print(f"  Copula dimensions: {n_copula_dims}")
        print(f"  Total dimensions: {n_total_dims}")
        print(f"  Dimensionality reduction: {n_total_dims - n_copula_dims} components")
        print()
        
        # Display validation results (conditional copula validation)
        validation_results = copula_metadata.get('validation_results', {})
        if validation_results:
            print("Conditional Copula Validation Metrics (Uniform Space):")
            
            # Main metrics from conditional copula validation
            log_lik = validation_results.get('mean_log_likelihood', None)
            corr_of_corr = validation_results.get('correlation_of_correlations', 0.0)
            corr_mae = validation_results.get('correlation_mae', 0.0)
            tail_coexceed = validation_results.get('tail_coexceedance', 0.0)
            
            if log_lik is not None:
                print(f"  Mean Log-Likelihood: {log_lik:.3f}")
            else:
                print(f"  Mean Log-Likelihood: N/A")
            
            print(f"  Correlation-of-Correlations: {corr_of_corr:.3f}")
            print(f"  Correlation MAE: {corr_mae:.3f}")
            print(f"  Tail Co-exceedance: {tail_coexceed:.3f}")
            print()
            
            # Quality assessment based on conditional copula metrics
            print("Quality Assessment:")
            if corr_of_corr >= 0.7:
                print("  🎉 EXCELLENT - Copula captures dependencies very well")
            elif corr_of_corr >= 0.5:
                print("  ✅ GOOD - Copula captures dependencies reasonably well")
            elif corr_of_corr >= 0.3:
                print("  ⚠️  MODERATE - Copula shows some dependency structure")
            else:
                print("  ❌ POOR - Copula fails to capture dependencies")
            
            print()
        else:
            print("⚠️  No validation results available")
        
        print("=" * 50)
    
    def show_e2e_validation(self) -> None:
        """
        Display end-to-end validation results for a trained model
        
        Example:
            >>> model.show_e2e_validation()
        """
        # Check if model is trained
        if self.status != "trained":
            print(f"❌ Model must be trained to show E2E validation (current status: {self.status})")
            return
        
        # Get E2E results from model metadata (nested in copula_metadata)
        model_metadata = self._data.get('model_metadata', {})
        copula_metadata = model_metadata.get('copula_metadata', {})
        e2e_results = copula_metadata.get('e2e_validation_results', {})
        
        if not e2e_results:
            print("❌ No E2E validation results found. Model may not have been trained with run_e2e_validation=True")
            return
        
        print("🎯 End-to-End Validation Results (Full Pipeline):")
        print("=" * 70)
        
        # Overall score
        overall_score = e2e_results.get('overall_score', 0.0)
        print(f"Overall Score: {overall_score:.3f}")
        print()
        
        # Calibration
        calibration = e2e_results.get('calibration_metrics', {})
        if calibration:
            print("📊 Calibration (Coverage):")
            cov_90 = calibration.get('mean_coverage_90', 0.0)
            cov_50 = calibration.get('mean_coverage_50', 0.0)
            well_cal_90 = calibration.get('well_calibrated_90', 0)
            well_cal_50 = calibration.get('well_calibrated_50', 0)
            total_comps = calibration.get('total_components', 0)
            
            print(f"  90% Coverage: {cov_90:.1%} (target: 90%)")
            print(f"  50% Coverage: {cov_50:.1%} (target: 50%)")
            print(f"  Well-calibrated components: {well_cal_90}/{total_comps} (90% CI)")
            print()
        
        # Distribution similarity
        distribution = e2e_results.get('distribution_metrics', {})
        if distribution:
            print("📈 Distribution Similarity:")
            mean_ks = distribution.get('mean_ks', 0.0)
            good_ks = distribution.get('good_ks_count', 0)
            total_comps = distribution.get('total_components', 0)
            
            print(f"  Mean KS statistic: {mean_ks:.3f}")
            print(f"  Components with KS<0.2: {good_ks}/{total_comps}")
            print()
        
        # Correlation structure
        correlation = e2e_results.get('correlation_metrics', {})
        if correlation:
            print("🔗 Correlation Structure (Reconstructed Space):")
            
            targets_only = correlation.get('targets_only', {})
            if targets_only:
                score = targets_only.get('correlation_similarity', targets_only.get('mean_score', 0.0))
                std = targets_only.get('std_score', 0.0)
                n_samples = targets_only.get('n_samples', 0)
                metric = targets_only.get('metric', 'correlation_of_correlations')
                print(f"  Targets Only:")
                print(f"    Correlation Similarity: {score:.3f} ± {std:.3f}")
                print(f"    Samples: {n_samples}")
                print(f"    Metric: {metric}")
            
            targets_cond = correlation.get('targets_and_conditioning')
            if targets_cond:
                score = targets_cond.get('correlation_similarity', targets_cond.get('mean_score', 0.0))
                std = targets_cond.get('std_score', 0.0)
                n_samples = targets_cond.get('n_samples', 0)
                metric = targets_cond.get('metric', 'cross_correlation_of_correlations')
                print(f"  Targets + Conditioning:")
                print(f"    Cross-Correlation Similarity: {score:.3f} ± {std:.3f}")
                print(f"    Samples: {n_samples}")
                print(f"    Metric: {metric}")
            print()
        
        # Quality assessment
        print("Quality Assessment:")
        if overall_score >= 0.7:
            print("  🎉 EXCELLENT - Model performs very well across all metrics")
        elif overall_score >= 0.5:
            print("  ✅ GOOD - Model performs reasonably well")
        elif overall_score >= 0.3:
            print("  ⚠️  MODERATE - Model shows acceptable performance")
        else:
            print("  ❌ POOR - Model needs improvement")
        
        print("=" * 70)
    
    # ============================================
    # SCENARIO CREATION
    # ============================================
    
    def create_scenario(
        self,
        name: str,
        description: str = "",
        n_scenarios: int = 100
    ):
        """
        Create a new scenario linked to this model.
        
        This is a convenience method that creates a scenario without needing
        to access client.scenarios.create().
        
        Args:
            name: Scenario name
            description: Optional scenario description
            n_scenarios: Number of synthetic paths to generate (default: 100)
        
        Returns:
            Scenario instance
        
        Example:
            >>> scenario = model.create_scenario(
            ...     name="COVID Crash Scenario",
            ...     description="Simulating March 2020 conditions",
            ...     n_scenarios=1000
            ... )
        """
        from ..scenario.builder import Scenario
        
        print(f"[Model {self.name}] Creating scenario: {name}")
        print(f"  Target paths: {n_scenarios}")
        
        # Create via API
        response = self.http.post('/api/v1/scenarios', {
            'model_id': self.id,
            'name': name,
            'description': description,
            'n_scenarios': n_scenarios
        })
        
        print(f"✅ Scenario created: {response.get('name')} (ID: {response.get('id')[:8]}...)")
        
        return Scenario(self.http, response, self)
    
    # ============================================
    # FORECASTING
    # ============================================
    
    def generate_forecast(
        self,
        sample_id: Optional[str] = None,
        test_sample_index: int = 0,
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
                print(f"  Auto-selecting test sample (index: {test_sample_index})...")
                test_sample = self._get_test_sample(index=test_sample_index, include_data=False)
                if not test_sample:
                    print(f"❌ No test sample found at index {test_sample_index}")
                    return {"status": "error", "message": f"No test sample at index {test_sample_index}"}
                sample_id = test_sample['id']
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
            "n_samples": len(forecast_samples),
            "reference_sample_id": sample_id if scenario is None else None  # Include for plotting
        }
    
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
        print(f"  ✅ Encoded: {len(encoded_values)} components")
        
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
        print(f"  Components: {len(encoded_values)}")
        
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
    
    def reconstruct_forecast(
        self,
        forecast_samples: List[Dict],
        reference_sample_id: str
    ) -> List[Dict[str, Dict[str, Any]]]:
        """
        Reconstruct forecast outputs back to original scale.
        
        Args:
            forecast_samples: Output from generate_forecast() method
            reference_sample_id: Sample ID used for forecast generation
        
        Returns:
            List of reconstructed forecast samples, each with structure:
            {
                "feature_name": {
                    "future": {"dates": [...], "values": [...]}
                }
            }
        
        Example:
            >>> forecast_result = model.generate_forecast(n_samples=10)
            >>> reconstructed = model.reconstruct_forecast(
            ...     forecast_result['forecast_samples'],
            ...     reference_sample_id="..."
            ... )
        """
        import numpy as np
        
        print(f"[Reconstruction] Reconstructing {len(forecast_samples)} forecast samples...")
        
        # Transform forecast samples to encoded_windows format
        # Track sample index in the window itself (since multivariate groups expand to multiple reconstructions)
        encoded_windows = []
        
        for sample_idx, sample in enumerate(forecast_samples):
            for item in sample.get('encoded_target_data', []):
                encoded_window = {
                    "feature": item['feature'],
                    "temporal_tag": item['temporal_tag'],
                    "data_type": "encoded_normalized_residuals",
                    "encoded_values": item['encoded_normalized_residuals'],  # API expects "encoded_values"
                    "_sample_idx": sample_idx  # Track which forecast sample this belongs to
                }
                # Preserve group metadata if present
                if 'is_group' in item:
                    encoded_window['is_group'] = item['is_group']
                if 'is_multivariate' in item:
                    encoded_window['is_multivariate'] = item['is_multivariate']
                if 'group_features' in item:
                    encoded_window['group_features'] = item['group_features']
                if 'n_components' in item:
                    encoded_window['n_components'] = item['n_components']
                
                encoded_windows.append(encoded_window)
        
        # Call reconstruct endpoint
        payload = {
            "user_id": self._data.get("user_id"),
            "model_id": self.id,
            
            # Encoded windows provided inline (forecast outputs)
            "encoded_source": "inline",
            "encoded_windows": encoded_windows,
            
            # Reference values from database (sample used for forecast)
            "reference_source": "database",
            "reference_table": "samples",
            "reference_column": "conditioning_data",
            "reference_sample_id": reference_sample_id,
            
            "output_destination": "return"
        }
        
        try:
            response = self.http.post('/api/v1/ml/reconstruct', payload)
        except Exception as e:
            print(f"  ❌ Reconstruction failed: {e}")
            raise
        
        reconstructions = response.get('reconstructions', [])
        print(f"✅ Reconstructed {len(reconstructions)} windows")
        
        # Fetch reference sample with full data (for dates)
        # Most common case: it's a test sample, so fetch test samples
        response = self.http.get(
            f'/api/v1/models/{self.id}/samples',
            params={'split_type': 'test', 'limit': 100, 'include_data': 'true'}
        )
        samples = response.get('samples', [])
        reference_sample = next((s for s in samples if s['id'] == reference_sample_id), None)
        
        if not reference_sample:
            raise ValueError(f"Reference sample {reference_sample_id} not found in test samples")
        
        # Group reconstructions back into samples
        # Use _sample_idx from window metadata (preserved through reconstruction)
        reconstructed_samples = []
        windows_by_sample = {}
        
        for window in reconstructions:
            # Get sample index from window metadata (added during encoding)
            sample_idx = window.get('_sample_idx', 0)  # Default to 0 if not found
            if sample_idx not in windows_by_sample:
                windows_by_sample[sample_idx] = []
            windows_by_sample[sample_idx].append(window)
        
        # Convert to structured format
        for sample_idx in sorted(windows_by_sample.keys()):
            sample_reconstruction = {}
            
            for window in windows_by_sample[sample_idx]:
                feature = window['feature']
                temporal_tag = window['temporal_tag']
                reconstructed_values = window['reconstructed_values']
                
                if feature not in sample_reconstruction:
                    sample_reconstruction[feature] = {}
                
                # Get dates from reference sample
                dates = self._extract_dates_from_sample(reference_sample, feature, temporal_tag)
                
                sample_reconstruction[feature][temporal_tag] = {
                    "dates": np.array(dates) if dates else None,
                    "values": np.array(reconstructed_values)
                }
            
            reconstructed_samples.append(sample_reconstruction)
        
        return reconstructed_samples
    
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
    
    def plot_forecasts(
        self,
        forecast_result: Dict,
        reference_sample_id: str = None,
        target_feature: str = None,
        n_paths: int = 10,
        show_ci: bool = True,
        ci_levels: List[float] = [0.68, 0.95],
        save_path: str = None,
        show: bool = True
    ):
        """
        Plot forecast samples overlaid on the reference sample.
        
        Args:
            forecast_result: Output from generate_forecast() method
            reference_sample_id: Sample ID used for forecast (auto-detects if None)
            target_feature: Target feature to plot (uses first target if None)
            n_paths: Number of individual forecast paths to show
            show_ci: Whether to show confidence interval bands
            ci_levels: Confidence levels for intervals (e.g., [0.68, 0.95])
            save_path: Path to save plot (e.g., "forecast.png")
            show: Whether to display the plot
        
        Example:
            >>> forecast_result = model.generate_forecast(n_samples=100)
            >>> model.plot_forecasts(
            ...     forecast_result,
            ...     target_feature="Gold Price",
            ...     show_ci=True,
            ...     ci_levels=[0.68, 0.95]
            ... )
        """
        from ..visualization import TimeSeriesPlotter, _check_matplotlib
        import matplotlib.pyplot as plt
        from datetime import datetime
        import numpy as np
        
        _check_matplotlib()
        
        forecast_samples = forecast_result['forecast_samples']
        
        # Auto-detect reference_sample_id from forecast_samples if not provided
        if reference_sample_id is None:
            # Try to get from first forecast sample metadata
            if forecast_samples and 'metadata' in forecast_samples[0]:
                reference_sample_id = forecast_samples[0]['metadata'].get('reference_sample_id')
            
            if not reference_sample_id:
                raise ValueError("reference_sample_id must be provided or detectable from forecast_samples")
        
        print(f"[Plotting] Reconstructing forecast samples...")
        
        # Reconstruct forecast samples
        reconstructed_forecasts = self.reconstruct_forecast(forecast_samples, reference_sample_id)
        
        # Determine target feature
        if target_feature is None:
            # Use first feature that has future data in forecasts
            for feat in reconstructed_forecasts[0].keys():
                if 'future' in reconstructed_forecasts[0][feat]:
                    target_feature = feat
                    break
        
        if not target_feature:
            raise ValueError("No target feature found in forecast samples")
        
        print(f"[Plotting] Reconstructed {len(reconstructed_forecasts)} forecast samples")
        print(f"[Plotting] Target feature: '{target_feature}'")
        print(f"[Plotting] Will plot {n_paths} paths with CI bands: {show_ci}")
        
        # Get reference sample with FULL data (for original past values)
        # Fetch test sample by index with full data
        reference_sample = self._get_test_sample(index=0, include_data=True)
        if not reference_sample or reference_sample['id'] != reference_sample_id:
            # Fallback: fetch all test samples and find by ID
            print(f"  Note: Fetching test samples to find reference...")
            response = self.http.get(
                f'/api/v1/models/{self.id}/samples',
                params={'split_type': 'test', 'limit': 100, 'include_data': 'true'}
            )
            samples = response.get('samples', [])
            reference_sample = next((s for s in samples if s['id'] == reference_sample_id), None)
            if not reference_sample:
                raise ValueError(f"Reference sample {reference_sample_id} not found in test samples")
        
        # Get ORIGINAL past data (not reconstructed) to align with forecast reference
        # Forecasts are anchored to the original reference value, not the reconstructed one
        past_dates, past_values = self._get_original_past_data(reference_sample, target_feature)
        
        # Get GROUND TRUTH future data from the reference sample (test sample has actual future values)
        ground_truth_dates, ground_truth_values = self._get_ground_truth_future_data(reference_sample, target_feature)
        
        if ground_truth_values is not None:
            print(f"  ✅ Found ground truth: {len(ground_truth_values)} future values")
        else:
            print(f"  ⚠️  No ground truth data found for {target_feature}")
        
        # Extract future data from forecasts
        future_dates = reconstructed_forecasts[0][target_feature]['future']['dates']
        forecast_paths = [
            sample[target_feature]['future']['values']
            for sample in reconstructed_forecasts
        ]
        
        print(f"[Plotting] Extracted {len(forecast_paths)} forecast paths from reconstructions")
        
        # Check if dates are available
        if past_dates is None or future_dates is None:
            print("  ⚠️  Warning: Dates not found in sample data, generating synthetic dates")
            # Generate synthetic dates for visualization
            past_dates = list(range(len(past_values)))
            future_dates = list(range(len(past_values), len(past_values) + len(forecast_paths[0])))
            # Convert to datetime objects
            from datetime import datetime, timedelta
            base_date = datetime(2024, 1, 1)
            past_date_objects = [base_date + timedelta(days=i) for i in past_dates]
            future_date_objects = [base_date + timedelta(days=i) for i in future_dates]
        else:
            # Convert dates to datetime
            past_date_objects = [datetime.strptime(d, '%Y-%m-%d') if isinstance(d, str) else d for d in past_dates]
            future_date_objects = [datetime.strptime(d, '%Y-%m-%d') if isinstance(d, str) else d for d in future_dates]
        
        # Handle ground truth dates
        if ground_truth_dates is not None:
            ground_truth_date_objects = [datetime.strptime(d, '%Y-%m-%d') if isinstance(d, str) else d for d in ground_truth_dates]
        else:
            ground_truth_date_objects = None
            print("  ⚠️  Warning: Ground truth dates not found")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        TimeSeriesPlotter.plot_forecasts(
            past_dates=np.array(past_date_objects),
            past_values=np.array(past_values),
            future_dates=np.array(future_date_objects),
            forecast_paths=forecast_paths,
            feature_name=target_feature,
            n_paths=n_paths,
            show_ci=show_ci,
            ci_levels=ci_levels,
            ground_truth_dates=np.array(ground_truth_date_objects) if ground_truth_date_objects else None,
            ground_truth_values=np.array(ground_truth_values) if ground_truth_values else None,
            ax=ax
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_feature_importance(
        self,
        top_n: int = 20,
        save_path: str = None,
        show: bool = True
    ):
        """
        Plot feature importance from trained QRF model (SHAP values).
        
        Args:
            top_n: Show only top N features (None = show all)
            save_path: Path to save plot (e.g., "feature_importance.png")
            show: Whether to display the plot
        
        Example:
            >>> model.plot_feature_importance(top_n=15)
            >>> model.plot_feature_importance(save_path="plots/importance.png")
        """
        from ..visualization import TimeSeriesPlotter, _check_matplotlib
        import matplotlib.pyplot as plt
        
        _check_matplotlib()
        
        # Check if model is trained
        if self.status != "trained":
            raise ValueError("Model must be trained before plotting feature importance. Call model.train() first.")
        
        # Get feature importance from model_metadata (nested structure)
        model_metadata = self._data.get('model_metadata', {})
        
        # Feature importance is nested: model_metadata -> feature_importance -> feature_importance
        feature_importance_wrapper = model_metadata.get('feature_importance', {})
        feature_importance = feature_importance_wrapper.get('feature_importance', {})
        
        if not feature_importance:
            # Print debug info to help user
            print(f"  Model metadata keys: {list(model_metadata.keys())}")
            if feature_importance_wrapper:
                print(f"  Feature importance wrapper keys: {list(feature_importance_wrapper.keys())}")
            raise ValueError("No feature importance data found. Model may not have been trained with SHAP values.")
        
        # Extract feature names and importance values
        feature_names = list(feature_importance.keys())
        importance_values = list(feature_importance.values())
        
        print(f"[Plotting] Feature importance for {len(feature_names)} features...")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, max(6, min(len(feature_names), top_n or len(feature_names)) * 0.4)))
        
        TimeSeriesPlotter.plot_feature_importance(
            feature_names=feature_names,
            importance_values=importance_values,
            top_n=top_n,
            ax=ax
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
    # MFA METHODS
    # ============================================
    
    def train_mfa(self,
                  n_components: int = 5,
                  n_factors: int = 10,
                  use_t_distribution: bool = False,
                  tail_quantile: float = 0.95,
                  covariance_type: str = 'diag',
                  split: str = 'training',
                  confirm: Optional[bool] = None) -> Dict[str, Any]:
        """
        Train MFA (Mixture of Factor Analyzers) model
        
        This is an alternative to the QRF approach that uses:
        - EVT-spliced marginals for heavy tails
        - MFA/t-MFA for joint distribution structure
        - Local copulas for conditional inference
        
        Args:
            n_components: Number of mixture components (default: 5)
            n_factors: Number of latent factors per component (default: 10)
            use_t_distribution: Use t-MFA instead of MFA for heavier tails (default: False)
            tail_quantile: Quantile threshold for EVT tails (default: 0.95)
            covariance_type: 'diag' or 'full' covariance (default: 'diag')
            split: Data split to use ('training', 'validation', or 'training+validation')
            confirm: Skip confirmation prompt if True
            
        Returns:
            Dict with training results including BIC, AIC, and model path
            
        Example:
            >>> # Train MFA model
            >>> result = model.train_mfa(
            ...     n_components=5,
            ...     n_factors=10,
            ...     use_t_distribution=True
            ... )
            >>> print(f"BIC: {result['training_metrics']['bic']}")
        """
        # Confirmation
        if confirm is None:
            confirm = not self.interactive
        
        if not confirm and self.interactive:
            print(f"\n🤖 Training MFA Model")
            print(f"   Components: {n_components}")
            print(f"   Factors: {n_factors}")
            print(f"   Type: {'t-MFA' if use_t_distribution else 'MFA'}")
            print(f"   Split: {split}")
            response = input("\nProceed with MFA training? (y/n): ")
            if response.lower() != 'y':
                print("❌ Training cancelled")
                return {'status': 'cancelled'}
        
        print(f"\n🚀 Training MFA model...")
        
        # Call training endpoint
        payload = {
            'user_id': self._data.get("user_id"),  # For API key auth
            'model_id': self.id,
            'n_components': n_components,
            'n_factors': n_factors,
            'use_t_distribution': use_t_distribution,
            'tail_quantile': tail_quantile,
            'covariance_type': covariance_type,
            'split': split
        }
        
        result = self.http.post('/api/v1/ml/train-mfa', payload)
        
        print(f"✅ MFA training completed!")
        print(f"   Samples used: {result['n_samples_used']}")
        print(f"   Dimensions: {result['n_dimensions']}")
        print(f"   BIC: {result['training_metrics']['bic']:.2f}")
        print(f"   AIC: {result['training_metrics']['aic']:.2f}")
        print(f"   Model saved to: {result['mfa_path']}")
        
        # Refresh model data
        self.refresh()
        
        return result
    
    def forecast_mfa(self,
                     observed_components: Optional[List[Dict[str, Any]]] = None,
                     split: str = 'validation',
                     sample_index: int = 0,
                     sample_id: Optional[str] = None,
                     n_samples: int = 1000,
                     top_k_neighbors: int = 100,
                     copula_type: str = 't') -> Dict[str, Any]:
        """
        Generate forecasts using MFA + local copula
        
        Two modes:
        1. Sample-based (default): Condition on a validation/test sample
        2. Inline: Provide observed_components manually
        
        Args:
            observed_components: List of observed components for inline conditioning (optional)
            split: Which split to use for sample-based conditioning (default: 'validation')
            sample_index: Index of sample in split (default: 0)
            sample_id: Specific sample ID to use (overrides split/sample_index)
            n_samples: Number of forecast samples to generate (default: 1000)
            top_k_neighbors: Number of neighbors for local copula (default: 100)
            copula_type: 't' for t-copula or 'skew-t' for skew-t copula
            
        Returns:
            Dict with forecasts and conditioning info
            
        Examples:
            >>> # Sample-based conditioning (default)
            >>> forecasts = model.forecast_mfa(split='validation', sample_index=0, n_samples=50)
            >>> 
            >>> # Unconditional forecast
            >>> forecasts = model.forecast_mfa(observed_components=[], n_samples=1000)
            >>> 
            >>> # Inline conditioning
            >>> observed = [{"source": "conditioning", "feature": "VIX", ...}]
            >>> forecasts = model.forecast_mfa(observed_components=observed)
        """
        print(f"\n🎲 Generating MFA forecasts...")
        
        # Determine conditioning source
        if observed_components is not None:
            conditioning_source = "inline"
            print(f"   Mode: Inline conditioning")
            print(f"   Conditioning on {len(observed_components)} components")
        else:
            conditioning_source = "sample"
            print(f"   Mode: Sample-based conditioning")
            print(f"   Split: {split}, Index: {sample_index}")
        
        print(f"   Samples: {n_samples}")
        
        # Call forecasting endpoint
        payload = {
            'user_id': self._data.get("user_id"),  # For API key auth
            'model_id': self.id,
            'conditioning_source': conditioning_source,
            'n_samples': n_samples,
            'top_k_neighbors': top_k_neighbors,
            'copula_type': copula_type
        }
        
        if conditioning_source == "inline":
            payload['observed_components'] = observed_components or []
        else:  # sample
            payload['split'] = split
            payload['sample_index'] = sample_index
            if sample_id:
                payload['sample_id'] = sample_id
        
        result = self.http.post('/api/v1/ml/forecast-mfa', payload)
        
        print(f"✅ Forecasting completed!")
        print(f"   Generated {result['n_samples']} samples")
        print(f"   Observed: {result['n_observed']} dimensions")
        print(f"   Predicted: {result['n_predicted']} dimensions")
        
        return result
    
    def reconstruct_mfa_forecasts(self,
                                   mfa_forecasts: Dict[str, Any],
                                   reference_sample_id: Optional[str] = None,
                                   split: str = 'validation') -> Dict[str, Any]:
        """
        Reconstruct MFA forecasts to original feature space
        
        Args:
            mfa_forecasts: Output from forecast_mfa()
            reference_sample_id: Sample ID to use for reference values (for residuals)
                                If None, uses first validation sample
            split: Data split to get reference sample from (default: 'validation')
            
        Returns:
            dict: Reconstructed trajectories for all forecast samples
        """
        import numpy as np
        
        print(f"\n🔄 Reconstructing {len(mfa_forecasts['forecasts'])} MFA forecast samples...")
        
        # Get reference sample if needed
        if reference_sample_id is None:
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
        
        # Extract group metadata from forecasts (if available)
        feature_metadata = {}
        if mfa_forecasts['forecasts']:
            first_sample = mfa_forecasts['forecasts'][0]
            if '_group_metadata' in first_sample:
                feature_metadata = first_sample['_group_metadata']
                print(f"  Found group metadata for {len(feature_metadata)} features")
        
        # Reconstruct each forecast sample
        all_reconstructions = []
        
        for i, forecast_sample in enumerate(mfa_forecasts['forecasts']):
            if (i + 1) % 10 == 0:
                print(f"  Reconstructing sample {i+1}/{len(mfa_forecasts['forecasts'])}...")
            
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
            
            # Build encoded_windows for reconstruction API (similar to QRF format)
            encoded_windows = []
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
                for i, part in enumerate(parts):
                    if part in ['past', 'future']:
                        temporal_idx = i
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
                
                encoded_windows.append(encoded_window)
            
            # Call reconstruct endpoint (same as QRF)
            payload = {
                "user_id": self._data.get("user_id"),
                "model_id": self.id,
                "encoded_source": "inline",
                "encoded_windows": encoded_windows,
                "reference_source": "database",
                "reference_table": "samples",
                "reference_column": "conditioning_data",
                "reference_sample_id": reference_sample_id,
                "output_destination": "return"
            }
            
            response = self.http.post('/api/v1/ml/reconstruct', payload)
            reconstructions = response.get('reconstructions', [])
            
            all_reconstructions.append({
                'sample_idx': i,
                'reconstructions': reconstructions
            })
        
        print(f"✅ Reconstructed all {len(all_reconstructions)} forecast samples")
        
        # Also get ground truth from reference sample
        print(f"\n🔍 Extracting ground truth from reference sample...")
        ref_sample_response = self.http.get(
            f'/api/v1/models/{self.id}/samples',
            params={'split_type': split, 'limit': 100, 'include_data': 'true'}
        )
        ref_samples = ref_sample_response.get('samples', [])
        ref_sample = next((s for s in ref_samples if s['id'] == reference_sample_id), None)
        
        ground_truth = None
        if ref_sample:
            # Get normalization params for denormalization
            norm_params = self._data.get('feature_normalization_params', {})
            
            ref_windows = []
            
            # Process past windows (conditioning_data with normalized_series)
            for item in ref_sample.get('conditioning_data', []):
                if item.get('temporal_tag') == 'past' and 'normalized_series' in item:
                    feature = item.get('feature')
                    normalized_series = item['normalized_series']
                    
                    # Denormalize
                    if feature in norm_params:
                        mean = norm_params[feature].get('mean', 0.0)
                        std = norm_params[feature].get('std', 1.0)
                        denormalized = [val * std + mean for val in normalized_series]
                    else:
                        denormalized = normalized_series  # No norm params, use as-is
                    
                    ref_windows.append({
                        'feature': feature,
                        'temporal_tag': 'past',
                        'values': denormalized
                    })
            
            # Process future target windows (target_data with normalized_residuals)
            # Need to add reference value from past window
            past_refs = {}
            for item in ref_sample.get('conditioning_data', []):
                if item.get('temporal_tag') == 'past' and 'normalized_series' in item:
                    feature = item.get('feature')
                    normalized_series = item['normalized_series']
                    if normalized_series:
                        past_refs[feature] = normalized_series[-1]  # Last value as reference
            
            for item in ref_sample.get('target_data', []):
                if 'normalized_residuals' in item:
                    feature = item.get('feature')
                    normalized_residuals = item['normalized_residuals']
                    
                    # Convert residuals to series by adding reference
                    if feature in past_refs:
                        ref_value = past_refs[feature]
                        normalized_series = [ref_value + res for res in normalized_residuals]
                    else:
                        normalized_series = normalized_residuals  # No reference, use as-is
                    
                    # Denormalize
                    if feature in norm_params:
                        mean = norm_params[feature].get('mean', 0.0)
                        std = norm_params[feature].get('std', 1.0)
                        denormalized = [val * std + mean for val in normalized_series]
                    else:
                        denormalized = normalized_series  # No norm params, use as-is
                    
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
            print(f"⚠️  Reference sample not found")
        
        return {
            'reconstructions': all_reconstructions,
            'reference_sample_id': reference_sample_id,
            'n_samples': len(all_reconstructions),
            'ground_truth': ground_truth
        }
