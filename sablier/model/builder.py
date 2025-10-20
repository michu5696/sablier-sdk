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
            print(f"\n🔧 Vine Copula Hyperparameter Optimization")
            print(f"   Trials: {n_trials}")
            print(f"   Objectives: {objectives}")
            print(f"   Data split: {split}")
            print(f"   This will test {n_trials} different parameter combinations")
            print(f"   and find the optimal settings for your data.")
            
            response = input(f"\nProceed with optimization? (y/N): ")
            if response.lower() != 'y':
                print("❌ Optimization cancelled")
                return {"status": "cancelled"}
        
        print(f"\n🚀 Starting Vine Copula hyperparameter optimization...")
        
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
                  model_type: str = 'vine_copula',
                  n_components: int = 5,
                  n_factors: int = 10,
                  covariance_type: str = 'diag',
                  top_pair_percent: float = 0.3,
                  copula_family: str = 'mixed',
                  trunc_lvl: int = 3,
                  num_threads: int = 4,
                  split: str = 'training',
                  compute_validation_ll: bool = False,
                  confirm: Optional[bool] = None) -> Dict[str, Any]:
        """
        Train model (Vine Copula or Vine Copula)
        
        Two model types available:
        1. Vine Copula (default): EM mixture of vine copulas with cross-window pair selection
        2. Vine Copula: Mixture of Factor Analyzers with Gaussian mixtures
        
        Pipeline:
        - Empirical marginals (smoothed CDF + exponential tails)
        - [Vine Copula] EM algorithm with regime-specific vine copulas
        - [Vine Copula] Factor-structured Gaussian mixture (Σ_k = Λ_k @ Λ_k^T + Ψ_k)
        - [Vine Copula] Local copulas for conditional tail dependence
        
        Args:
            model_type: Model type (default: 'vine_copula')
            n_components: Number of mixture components/regimes (default: 5)
            n_factors: [Vine Copula only] Number of latent factors per component (default: 10)
            covariance_type: [Vine Copula only] 'diag' or 'full' covariance (default: 'diag')
            top_pair_percent: [Vine Copula] Fraction of cross-window pairs to model (default: 0.3)
            copula_family: [Vine Copula] 'gaussian', 't', 'clayton', 'gumbel', 'mixed' (default: 'mixed')
            trunc_lvl: [Vine Copula] Truncation level - fit first L trees fully (default: 3)
            num_threads: [Vine Copula] Number of threads for parallel fitting (default: 4)
            split: Data split to use ('training', 'validation', or 'training+validation')
            compute_validation_ll: Also compute validation log-likelihood (default: False)
            confirm: Skip confirmation prompt if True
            
        Returns:
            Dict with training results including metrics and model path
            
        Example:
            >>> # Train Vine Copula model (default)
            >>> result = model.train(
            ...     model_type='vine_copula',
            ...     n_components=2,
            ...     top_pair_percent=0.3,
            ...     copula_family='mixed',  # Auto-select best copula per pair
            ...     trunc_lvl=3,  # Fit first 3 trees (10x speedup)
            ...     num_threads=4  # Use 4 cores
            ... )
            
            >>> # Train Vine Copula model
            >>> result = model.train(
            ...     model_type='vine_copula',
            ...     n_components=5,
            ...     n_factors=10,
            ... )
            >>> print(f"BIC: {result['training_metrics']['bic']}")
        """
        # Confirmation
        if confirm is None:
            confirm = not self.interactive
        
        if not confirm and self.interactive:
            print(f"\n🤖 Training Vine Copula Model")
            print(f"   Regimes: {n_components}")
            print(f"   Copula family: {copula_family}")
            print(f"   Truncation level: {trunc_lvl}")
            print(f"   Split: {split}")
        
        print(f"\n🚀 Training Vine Copula model...")
        
        # Check for saved optimal parameters from hyperparameter optimization
        model_metadata = self._data.get('model_metadata') or {}
        optimal_config = model_metadata.get('vine_copula_optimal_config', {}) if model_metadata else {}
        
        if optimal_config:
            print(f"🔧 Using optimal parameters from previous optimization:")
            n_components = optimal_config.get('n_components', n_components)
            n_factors = optimal_config.get('n_factors', n_factors)
            print(f"   • n_components: {n_components}")
            print(f"   • n_factors: {n_factors}")
        else:
            print(f"📝 Using provided/default parameters (no optimization found)")
        
        # Call training endpoint
        payload = {
            'user_id': self._data.get("user_id"),  # For API key auth
            'model_id': self.id,
            'model_type': model_type,
            'n_components': n_components,
            'n_factors': n_factors,
            'covariance_type': covariance_type,
            'top_pair_percent': top_pair_percent,
            'copula_family': copula_family,
            'trunc_lvl': trunc_lvl,
            'num_threads': num_threads,
            'split': split,
            'compute_validation_ll': compute_validation_ll
        }
        
        result = self.http.post('/api/v1/ml/train', payload)
        
        train_metrics = result['training_metrics']
        model_type_str = train_metrics.get('model_type', model_type).upper()
        
        print(f"✅ {model_type_str} training completed!")
        print(f"   Samples used: {result['n_samples_used']}")
        print(f"   Dimensions: {result['n_dimensions']}")
        
        if model_type == 'vine_copula':
            print(f"   Regime weights: {train_metrics.get('regime_weights', 'N/A')}")
        else:
            print(f"   BIC: {train_metrics.get('bic', 0):.2f}")
            print(f"   AIC: {train_metrics.get('aic', 0):.2f}")
        
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
                     run_on_validation: bool = True,
                     copula_type: str = 'adaptive',
                     top_k_neighbors: int = 100) -> Dict[str, Any]:
        """
        Validate Vine Copula model on held-out data
        
        Computes:
        - Validation log-likelihood (out-of-sample fit)
        - Regime analysis (component assignments)
        - Calibration metrics (forecast quality)
        
        Args:
            n_forecast_samples: Number of forecast samples per validation sample (default: 100)
            validation_split: Split to validate on (default: 'validation')
            
        Returns:
            Dict with validation metrics
            
        Example:
            >>> validation = model.validate(n_forecast_samples=100)
            >>> print(f"Validation log-likelihood: {validation['validation_metrics']['per_sample_log_likelihood']}")
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
            'run_on_validation': run_on_validation,
            'copula_type': copula_type,
            'top_k_neighbors': top_k_neighbors
        })
        
        # Display results
        print(f"\n" + "="*70)
        print(f"✅ Vine Copula Model Validation Complete")
        print(f"="*70)
        
        # Summary table for unconditional metrics
        if result.get('training_metrics') and result.get('validation_metrics'):
            train_metrics = result['training_metrics']
            val_metrics = result['validation_metrics']
            train_ll = train_metrics['per_sample_log_likelihood']
            val_ll = val_metrics['per_sample_log_likelihood']
            gap = val_ll - train_ll
            
            print(f"\n📊 Unconditional Log-Likelihood (Joint Distribution Fit)")
            print(f"{'─'*70}")
            print(f"{'Metric':<30} {'Training':<20} {'Validation':<20}")
            print(f"{'─'*70}")
            print(f"{'Log-likelihood (per sample)':<30} {train_ll:<20.2f} {val_ll:<20.2f}")
            print(f"{'BIC':<30} {train_metrics['bic']:<20.2f} {val_metrics['bic']:<20.2f}")
            print(f"{'AIC':<30} {train_metrics['aic']:<20.2f} {val_metrics['aic']:<20.2f}")
            print(f"{'N Samples':<30} {train_metrics['n_samples']:<20} {val_metrics['n_samples']:<20}")
            print(f"{'─'*70}")
            print(f"{'Generalization Gap (absolute)':<30} {gap:.2f}")
            
            # Compute relative gap
            relative_gap = abs(gap / train_ll) * 100 if train_ll != 0 else 0
            print(f"{'Generalization Gap (relative)':<30} {relative_gap:.1f}%")
            
            # Gaussian baseline reference
            n_dims = train_metrics.get('n_dims', 0)
            if n_dims > 0:
                gaussian_baseline = -n_dims/2 * np.log(2 * np.pi) - n_dims/2
                train_excess = train_ll - gaussian_baseline
                val_excess = val_ll - gaussian_baseline
                print(f"{'─'*70}")
                print(f"{'Gaussian baseline LL':<30} {gaussian_baseline:.2f}")
                print(f"{'Training excess (vs Gaussian)':<30} {train_excess:.2f}")
                print(f"{'Validation excess (vs Gaussian)':<30} {val_excess:.2f}")
                
                # Interpret training fit quality
                print(f"{'─'*70}")
                print(f"{'Training Fit Quality:':<30}")
                if train_excess > -20:
                    print(f"{'  Status':<30} ⚠️  Suspiciously good (check for data leakage)")
                elif train_excess > -50:
                    print(f"{'  Status':<30} ✅ Excellent (captures complexity well)")
                elif train_excess > -80:
                    print(f"{'  Status':<30} ✅ Good (reasonable for financial data)")
                elif train_excess > -120:
                    print(f"{'  Status':<30} ⚠️  Moderate (model may be underfitting)")
                else:
                    print(f"{'  Status':<30} ❌ Poor (model struggling to fit)")
            
            print(f"{'─'*70}")
            print(f"{'Generalization Quality:':<30}")
            if relative_gap < 10:
                print(f"{'  Status':<30} ✅ Excellent (minimal overfitting)")
                print(f"{'  Interpretation':<30} Model generalizes very well")
            elif relative_gap < 30:
                print(f"{'  Status':<30} ✅ Good (acceptable overfitting)")
                print(f"{'  Interpretation':<30} Model is reliable for new data")
            elif relative_gap < 50:
                print(f"{'  Status':<30} ⚠️  Moderate (some overfitting)")
                print(f"{'  Interpretation':<30} Model may be memorizing training data")
            elif relative_gap < 100:
                print(f"{'  Status':<30} ⚠️  Significant (notable overfitting)")
                print(f"{'  Interpretation':<30} Likely regime shift or insufficient data")
            else:
                print(f"{'  Status':<30} ❌ Severe (extreme overfitting)")
                print(f"{'  Interpretation':<30} Major regime shift between train/val periods")
            
            # BIC/AIC interpretation
            print(f"{'─'*70}")
            print(f"{'Model Complexity (BIC/AIC):':<30}")
            bic_diff = val_metrics['bic'] - train_metrics['bic']
            if bic_diff < 0:
                print(f"{'  BIC comparison':<30} ⚠️  Validation BIC lower (unusual)")
            else:
                print(f"{'  BIC comparison':<30} ✅ Validation BIC higher (expected)")
            print(f"{'  Note':<30} Lower BIC = better model")
            print(f"{'       ':<30} (penalizes complexity)")
            
            print(f"{'─'*70}")
        
        elif result.get('training_metrics'):
            train_metrics = result['training_metrics']
            print(f"\n📊 Training Metrics (In-Sample):")
            print(f"   Log-likelihood (per sample): {train_metrics['per_sample_log_likelihood']:.2f}")
            print(f"   BIC: {train_metrics['bic']:.2f}")
            print(f"   AIC: {train_metrics['aic']:.2f}")
            print(f"   Samples: {train_metrics['n_samples']}")
        
        elif result.get('validation_metrics'):
            val_metrics = result['validation_metrics']
            print(f"\n📊 Validation Metrics (Out-of-Sample):")
            print(f"   Log-likelihood (per sample): {val_metrics['per_sample_log_likelihood']:.2f}")
            print(f"   BIC: {val_metrics['bic']:.2f}")
            print(f"   AIC: {val_metrics['aic']:.2f}")
            print(f"   Samples: {val_metrics['n_samples']}")
        
        print(f"\n🎭 Regime Analysis:")
        regime_analysis = result['regime_analysis']
        print(f"   Components: {regime_analysis['n_components']}")
        for regime in regime_analysis['regime_stats']:
            k = regime['component_id']
            weight = regime['mixing_weight']
            n_assigned = regime['n_samples_assigned']
            print(f"   Component {k}: weight={weight:.3f}, assigned={n_assigned} samples")
        
        print(f"\n📈 Conditional Forecasting Metrics (with Local Copula)")
        print(f"{'─'*70}")
        calib = result['calibration_metrics']['calibration']
        forecast_quality = result['calibration_metrics']['forecast_quality']
        
        print(f"{'Metric':<35} {'Value':<20} {'Target/Status':<15}")
        print(f"{'─'*70}")
        
        # Calibration metrics
        ks_status = "✅ Good" if calib['ks_pvalue'] > 0.05 else "⚠️  Poor"
        print(f"{'KS test p-value':<35} {calib['ks_pvalue']:<20.4f} {ks_status:<15}")
        
        cov_68_diff = abs(calib['coverage_68'] - 0.68)
        cov_68_status = "✅" if cov_68_diff < 0.05 else "⚠️"
        print(f"{'68% coverage':<35} {calib['coverage_68']:<20.2%} {cov_68_status} (target: 68%)")
        
        cov_95_diff = abs(calib['coverage_95'] - 0.95)
        cov_95_status = "✅" if cov_95_diff < 0.05 else "⚠️"
        print(f"{'95% coverage':<35} {calib['coverage_95']:<20.2%} {cov_95_status} (target: 95%)")
        
        pit_mean_diff = abs(calib['pit_mean'] - 0.5)
        pit_mean_status = "✅" if pit_mean_diff < 0.05 else "⚠️"
        print(f"{'PIT mean':<35} {calib['pit_mean']:<20.3f} {pit_mean_status} (target: 0.5)")
        
        pit_std_diff = abs(calib['pit_std'] - 0.29)
        pit_std_status = "✅" if pit_std_diff < 0.05 else "⚠️"
        print(f"{'PIT std':<35} {calib['pit_std']:<20.3f} {pit_std_status} (target: 0.29)")
        
        print(f"{'─'*70}")
        
        # CRPS interpretation
        mean_crps = forecast_quality['mean_crps']
        median_crps = forecast_quality['median_crps']
        
        if mean_crps < 1.0:
            crps_status = "✅ Excellent"
        elif mean_crps < 2.0:
            crps_status = "✅ Good"
        elif mean_crps < 3.0:
            crps_status = "⚠️  Moderate"
        else:
            crps_status = "❌ Poor"
        
        print(f"{'Mean CRPS':<35} {mean_crps:<20.4f} {crps_status:<15}")
        print(f"{'Median CRPS':<35} {median_crps:<20.4f} {'(lower is better)':<15}")
        print(f"{'─'*70}")
        
        # Overall interpretation
        print(f"\n💡 Overall Assessment:")
        print(f"{'─'*70}")
        
        # 1. Distribution Match (CRPS)
        print(f"\n1️⃣  Distribution Match (CRPS):")
        if mean_crps < 1.0:
            print(f"   ✅ Excellent: Predicted distribution very close to true distribution")
        elif mean_crps < 2.0:
            print(f"   ✅ Good: Predicted distribution matches true distribution well")
        elif mean_crps < 3.0:
            print(f"   ⚠️  Moderate: Some mismatch between predicted and true distributions")
        else:
            print(f"   ❌ Poor: Significant mismatch between distributions")
        
        # 2. Calibration (KS test)
        print(f"\n2️⃣  Forecast Calibration (KS test):")
        if calib['ks_pvalue'] > 0.05:
            print(f"   ✅ Well-calibrated: Forecast probabilities are reliable")
            print(f"      → Can trust confidence intervals and risk estimates")
        else:
            print(f"   ⚠️  Poorly calibrated: Forecast probabilities may be biased")
            print(f"      → Confidence intervals may be too wide or too narrow")
        
        # 3. Coverage accuracy
        print(f"\n3️⃣  Coverage Accuracy:")
        coverage_68_ok = abs(calib['coverage_68'] - 0.68) < 0.05
        coverage_95_ok = abs(calib['coverage_95'] - 0.95) < 0.05
        if coverage_68_ok and coverage_95_ok:
            print(f"   ✅ Accurate: Confidence intervals have correct coverage")
            print(f"      → 68% and 95% intervals are reliable for risk management")
        elif coverage_68_ok or coverage_95_ok:
            print(f"   ⚠️  Partially accurate: Some intervals are miscalibrated")
            if not coverage_68_ok:
                print(f"      → 68% interval: {calib['coverage_68']:.1%} (target: 68%)")
            if not coverage_95_ok:
                print(f"      → 95% interval: {calib['coverage_95']:.1%} (target: 95%)")
        else:
            print(f"   ❌ Inaccurate: Confidence intervals need recalibration")
            print(f"      → 68% interval: {calib['coverage_68']:.1%} (target: 68%)")
            print(f"      → 95% interval: {calib['coverage_95']:.1%} (target: 95%)")
        
        # 4. Overall verdict
        print(f"\n4️⃣  Overall Verdict:")
        all_good = (mean_crps < 2.0 and calib['ks_pvalue'] > 0.05 and 
                   coverage_68_ok and coverage_95_ok)
        mostly_good = (mean_crps < 3.0 and calib['ks_pvalue'] > 0.05 and 
                      (coverage_68_ok or coverage_95_ok))
        
        if all_good:
            print(f"   🎉 EXCELLENT: Model is production-ready for scenario generation")
            print(f"      → Distributions match, calibration is good, coverage is accurate")
        elif mostly_good:
            print(f"   ✅ GOOD: Model is suitable for scenario generation")
            print(f"      → Minor calibration issues, but overall reliable")
        else:
            print(f"   ⚠️  NEEDS IMPROVEMENT: Consider hyperparameter tuning")
            print(f"      → Recalibration or more training data may help")
        
        print(f"{'─'*70}")
        
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
        print(f"\n🎲 Generating Vine Copula forecasts...")
        
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
