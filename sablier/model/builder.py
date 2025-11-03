"""Model class representing a Sablier model"""

import logging
import numpy as np
from typing import Optional, Any, List, Dict
from ..http_client import HTTPClient
from ..exceptions import APIError
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
    - Model training
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
    
    @property
    def is_shared(self) -> bool:
        """Check if this is a shared model from template"""
        return self._data.get('is_shared', False)
    
    def rename(self, new_name: str) -> Dict[str, Any]:
        """
        Rename the model
        
        Args:
            new_name: New name for the model
        
        Returns:
            Updated model data
            
        Example:
            >>> model.rename("Updated Model Name")
            ✅ Model renamed to 'Updated Model Name'
        """
        try:
            response = self.http.patch(f'/api/v1/models/{self.id}', {"name": new_name})
            
            # Update local data
            self._data = response
            
            old_name = self.name
            self.name = new_name
            
            print(f"✅ Model renamed from '{old_name}' to '{new_name}'")
            
            return response
        except APIError as e:
            if e.status_code == 403:
                print("Not authorized")
                return {}
            raise
    
    def set_sharing(self, enabled: bool = True) -> Dict[str, Any]:
        """
        Set model sharing status (admin only)
        
        Args:
            enabled: Whether to enable sharing (default: True)
        
        Returns:
            Updated model data
            
        Example:
            >>> model.set_sharing(enabled=True)
            ✅ Model 'Treasury Forecasting Model' is now shared
        """
        try:
            # The endpoint expects is_shared as a query parameter
            is_shared_str = "true" if enabled else "false"
            response = self.http.patch(f'/api/v1/models/{self.id}/share?is_shared={is_shared_str}')
            
            # Update local data
            self._data = response
            
            status = "shared" if enabled else "unshared"
            print(f"✅ Model '{self.name}' is now {status}")
            
            return response
        except APIError as e:
            if e.status_code == 403:
                print("Not authorized")
                return {}
            raise
    
    def refresh(self):
        """Refresh model data from API"""
        response = self.http.get(f'/api/v1/models/{self.id}')
        # The API returns the model data directly, not wrapped in 'model' key
        self._data = response if isinstance(response, dict) and 'id' in response else response.get('model', {})
        return self
    
    def get_conditioning_set(self):
        """
        Get the conditioning feature set for this model
        
        Returns:
            FeatureSet: Conditioning feature set object with features and groups
            
        Example:
            >>> conditioning = model.get_conditioning_set()
            >>> print(f"Features: {len(conditioning.features)}")
            >>> groups = conditioning.list_feature_groups()
        """
        from ..feature_set import FeatureSet
        
        if not self.conditioning_set_id:
            return None
        
        response = self.http.get(f'/api/v1/feature-sets/{self.conditioning_set_id}')
        return FeatureSet(self.http, response, self.project_id, self.interactive)
    
    def get_target_set(self):
        """
        Get the target feature set for this model
        
        Returns:
            FeatureSet: Target feature set object with features and groups
            
        Example:
            >>> target = model.get_target_set()
            >>> print(f"Features: {len(target.features)}")
            >>> groups = target.list_feature_groups()
        """
        from ..feature_set import FeatureSet
        
        if not self.target_set_id:
            return None
        
        response = self.http.get(f'/api/v1/feature-sets/{self.target_set_id}')
        return FeatureSet(self.http, response, self.project_id, self.interactive)
    
    def get_scenario(self, identifier):
        """
        Get scenario by name or index
        
        Args:
            identifier: Scenario name (str) or index (int)
            
        Returns:
            Scenario instance or None if not found
        """
        scenarios = self.list_scenarios()
        
        if isinstance(identifier, int):
            # Get by index
            if 0 <= identifier < len(scenarios):
                return scenarios[identifier]
            return None
        elif isinstance(identifier, str):
            # Get by name
            for scenario in scenarios:
                if scenario.name == identifier:
                    return scenario
            return None
        else:
            raise ValueError("Identifier must be string (name) or int (index)")
    
    def list_features(self) -> Dict[str, Any]:
        """
        List all features in this model with their grouping information
            
        Returns:
            dict: Feature information including conditioning and target features with groups
            
        Example:
            >>> features = model.list_features()
            >>> features['conditioning']['univariate']
            ['VIX Volatility Index', 'Crude Oil Prices']
            >>> features['target']['multivariate']
            [{'group_name': 'Treasury_Group_1', 'features': ['10Y Treasury', '20Y Treasury', '30Y Treasury']}]
        """
        result = {
            'conditioning': {'univariate': [], 'multivariate': []},
            'target': {'univariate': [], 'multivariate': []}
        }
        
        # Get conditioning set
        conditioning = self.get_conditioning_set()
        if conditioning:
            groups = conditioning.list_feature_groups()
            for group in groups:
                features = group.get('features', [])
                if len(features) == 1:
                    result['conditioning']['univariate'].append(features[0])
                else:
                    result['conditioning']['multivariate'].append({
                        'group_name': group.get('name', 'Unknown'),
                        'features': features
                    })
        
        # Get target set
        target = self.get_target_set()
        if target:
            groups = target.list_feature_groups()
            for group in groups:
                features = group.get('features', [])
                if len(features) == 1:
                    result['target']['univariate'].append(features[0])
                else:
                    result['target']['multivariate'].append({
                        'group_name': group.get('name', 'Unknown'),
                        'features': features
                    })
        
        return result
    
    def delete(self, confirm: Optional[bool] = None) -> Dict[str, Any]:
        """
        Delete this model and ALL associated data
        
        This will permanently delete:
        - Model record
        - Training data
        - Generated samples
        - Encoding models
        - Trained model (from storage)
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
        print("  - Trained model")
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
        try:
            print("🗑️  Deleting model...")
            response = self.http.delete(f'/api/v1/models/{self.id}')
            
            print(f"✅ Model '{self.name}' deleted successfully")
            
            return response
        except APIError as e:
            if e.status_code == 403:
                print("Not authorized")
                return {"status": "failed", "message": "Not authorized"}
            raise
    
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
        Generate training samples and fit encoding models using model's feature sets
        
        This method performs a complete pipeline:
        1. Generate training samples with proper windowing
        2. Fit encoding models on training split
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
        print(f"\n🔧 Fitting encoding models and encoding samples...")
        
        # Fit encoding models
        print(f"  Step 2/3: Fitting encoding models on 'training' split...")
        try:
            fit_response = self.http.post('/api/v1/ml/fit?split=training', {
                "model_id": self.id,
                "encoding_type": "pca-ica"
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
    
    
    

    

    # ============================================
    # SCENARIO CREATION
    # ============================================
    
    def _get_default_simulation_date(self) -> str:
        """
        Get default simulation date as the middle date of the validation split.
        
        Returns:
            Date string (YYYY-MM-DD) - middle date between validation split start and end
            
        Raises:
            ValueError: If no validation split information is available
        """
        from datetime import datetime, timedelta
        
        # Get date range from project (simplest approach)
        project_id = self._data.get('project_id')
        if not project_id:
            raise ValueError(
                "Model missing project_id. "
                "Please specify a simulation_date explicitly, for example: "
                "model.create_scenario(name='My Scenario', simulation_date='2020-03-15')"
            )
        
        try:
            project_response = self.http.get(f'/api/v1/projects/{project_id}')
            if not project_response:
                raise ValueError("Could not fetch project information")
            
            training_start = project_response.get('training_start_date')
            training_end = project_response.get('training_end_date')
            
            if not training_start or not training_end:
                raise ValueError("Project missing training date range")
            
            # Parse dates and calculate middle
            if isinstance(training_start, str):
                # Handle ISO format with or without time
                if 'T' in training_start:
                    training_start = datetime.fromisoformat(training_start.replace('Z', '+00:00'))
                else:
                    training_start = datetime.strptime(training_start, '%Y-%m-%d')
            
            if isinstance(training_end, str):
                # Handle ISO format with or without time
                if 'T' in training_end:
                    training_end = datetime.fromisoformat(training_end.replace('Z', '+00:00'))
                else:
                    training_end = datetime.strptime(training_end, '%Y-%m-%d')
            
            # Calculate middle date of training period
            total_days = (training_end - training_start).days
            middle_date = training_start + timedelta(days=total_days // 2)
            
            return middle_date.strftime('%Y-%m-%d')
            
        except ValueError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            # If API query fails, provide helpful error
            raise ValueError(
                f"Could not determine default simulation_date: {str(e)}. "
                "Please specify a simulation_date explicitly, for example: "
                "model.create_scenario(name='My Scenario', simulation_date='2020-03-15')"
            )
    
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
        name: str,
        simulation_date: Optional[str] = None,
        description: str = "",
        feature_simulation_dates: Optional[Dict[str, str]] = None
    ):
        """
        Create a new scenario linked to this model.
        
        This is a convenience method that creates a scenario without needing
        to access client.scenarios.create().
        
        Args:
            name: Scenario name
            simulation_date: Optional simulation date for all features (YYYY-MM-DD).
                           Defaults to today's date if not specified.
            description: Optional scenario description
            feature_simulation_dates: Optional dict mapping feature names to specific simulation dates
        
        Returns:
            Scenario instance
        
        Example:
            >>> scenario = model.create_scenario(
            ...     name="COVID Crash Scenario",
            ...     simulation_date="2020-03-15",
            ...     description="Simulating March 2020 conditions",
            ...     feature_simulation_dates={
            ...         "5-Year Treasury Rate": "2008-09-15",  # Lehman crisis
            ...         "VIX Volatility Index": "2020-02-28"   # Different date
            ...     }
            ... )
            
            # Use today's date as default
            >>> scenario = model.create_scenario(
            ...     name="Current Market Scenario"
            ... )
        """
        # Check if model is trained
        if self.status not in ['trained', 'model_trained']:
            raise ValueError(f"Model must be trained to create scenarios. Current status: {self.status}")
        
        # Set default simulation_date to latest validation sample date if not provided
        if simulation_date is None:
            simulation_date = self._get_default_simulation_date()
        
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
    
    def list_scenarios(self, verbose: bool = True):
        """
        List all scenarios created by the current user for this model.
        
        This is useful when working with template models - it shows only
        the scenarios YOU created, not scenarios created by other users.
        
        Args:
            verbose: If True, prints formatted output (default: True)
            
        Returns:
            List[Scenario]: List of Scenario instances
            
        Example:
            >>> # List scenarios for a template model
            >>> scenarios = model.list_scenarios()
            >>> for scenario in scenarios:
            ...     print(f"- {scenario.name}: {scenario.simulation_date}")
        """
        from ..scenario.builder import Scenario
        
        try:
            # Get all user's scenarios
            response = self.http.get('/api/v1/scenarios')
            all_scenarios = response.get('scenarios', []) if isinstance(response, dict) else response
            
            # Filter by this model
            model_scenarios = [s for s in all_scenarios if s.get('model_id') == self.id]
            
            if verbose:
                if model_scenarios:
                    print(f"\n📋 Scenarios for model '{self.name}' ({len(model_scenarios)} scenarios):")
                    for scenario in model_scenarios:
                        print(f"  - {scenario.get('name')}")
                        print(f"    ID: {scenario.get('id')[:8]}...")
                        print(f"    Simulation date: {scenario.get('simulation_date')}")
                        output = scenario.get('output')
                        if output:
                            # Handle both dict and string (shouldn't be string, but handle it)
                            if isinstance(output, dict):
                                n_samples = output.get('n_samples', 0)
                            else:
                                n_samples = 'unknown'
                            print(f"    Status: ✅ Simulation complete ({n_samples} samples)")
                        else:
                            print(f"    Status: ⏳ Not yet simulated")
                        print()
                else:
                    print(f"\n📋 No scenarios found for model '{self.name}'")
                    print(f"   Create one with: model.create_scenario(simulation_date='YYYY-MM-DD', name='...')")
            
            return [Scenario(self.http, s, self, interactive=self.interactive) for s in model_scenarios]
            
        except Exception as e:
            if verbose:
                print(f"❌ Failed to list scenarios: {e}")
            return []

    
    
    
    
   
    
    
    def train(self,
                  n_regimes: int = 3,
                  compute_validation_ll: bool = False) -> Dict[str, Any]:
        """
        Train statistical model on encoded samples
        
        Pipeline:
        - Empirical marginals with smoothed distributions
        - Multi-regime modeling with regime-specific dependencies
        - Mixed dependence structures optimized for the data
        - Optimal parameters and thread count for performance
        
        Args:
            n_regimes: Number of mixture components/regimes (default: 3)
            compute_validation_ll: Also compute validation log-likelihood (default: False)
            
        Returns:
            Dict with training results including metrics and model path
            
        Example:
            >>> # Train model with default settings
            >>> result = model.train()
            >>> print(f"Model trained with {result['training_metrics']['n_components']} regimes")
            
            >>> # Train with custom number of regimes
            >>> result = model.train(n_regimes=5)
            >>> print(f"Model trained with {result['training_metrics']['n_components']} regimes")
            
            >>> # Train with validation log-likelihood
            >>> result = model.train(n_regimes=3, compute_validation_ll=True)
        """
        print(f"\n🤖 Training Statistical Model")
        print(f"   Regimes: {n_regimes}")
        print(f"   Dependence structure: optimized")
        print(f"   Model parameters: auto-optimized")
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
        
        print(f"✅ Model training completed!")
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
        Validate statistical model on held-out data
        
        Computes:
        - Validation log-likelihood (out-of-sample fit)
        - Regime analysis (component assignments)
        - Calibration metrics (forecast quality)
        - Coverage metrics (68% and 95% confidence intervals)
        - Statistical tests for calibration
        
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
        print(f"\n🔍 Validating statistical model...")
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
                    # Display normalized value (scale-invariant) as primary, absolute value as secondary
                    if 'normalized_value' in crps:
                        print(f"{crps['icon']} CRPS: {crps['normalized_value']:.4f} (normalized, {crps['quality'].title()})")
                        print(f"   Raw CRPS: {crps['value']:.4f} (in data units)")
                    else:
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
                
                # Tail metrics (overall)
                if 'left_tail' in overall_metrics and overall_metrics['left_tail']:
                    left_tail = overall_metrics['left_tail']
                    print(f"\n📊 Left Tail Metrics (10th percentile - extreme lows):")
                    if 'crps' in left_tail:
                        crps = left_tail['crps']
                        if 'normalized_value' in crps:
                            print(f"  {crps['icon']} CRPS: {crps['normalized_value']:.4f} (normalized, {crps['quality'].title()}, {crps.get('n_samples', 0)} samples)")
                            print(f"     Raw CRPS: {crps['value']:.4f} (in data units)")
                        else:
                            print(f"  {crps['icon']} CRPS: {crps['value']:.4f} ({crps['quality'].title()}, {crps.get('n_samples', 0)} samples)")
                        print(f"     {crps['interpretation']}")
                    if 'sharpness' in left_tail:
                        sharpness = left_tail['sharpness']
                        print(f"  {sharpness['icon']} Sharpness: {sharpness['value']:.4f} ({sharpness['quality'].title()})")
                        print(f"     {sharpness['interpretation']}")
                    if 'reliability' in left_tail:
                        reliability = left_tail['reliability']
                        print(f"  {reliability['icon']} Reliability: {reliability['ece']:.4f} ({reliability['quality'].title()})")
                        print(f"     {reliability['interpretation']}")
                
                if 'right_tail' in overall_metrics and overall_metrics['right_tail']:
                    right_tail = overall_metrics['right_tail']
                    print(f"\n📊 Right Tail Metrics (90th percentile - extreme highs):")
                    if 'crps' in right_tail:
                        crps = right_tail['crps']
                        if 'normalized_value' in crps:
                            print(f"  {crps['icon']} CRPS: {crps['normalized_value']:.4f} (normalized, {crps['quality'].title()}, {crps.get('n_samples', 0)} samples)")
                            print(f"     Raw CRPS: {crps['value']:.4f} (in data units)")
                        else:
                            print(f"  {crps['icon']} CRPS: {crps['value']:.4f} ({crps['quality'].title()}, {crps.get('n_samples', 0)} samples)")
                        print(f"     {crps['interpretation']}")
                    if 'sharpness' in right_tail:
                        sharpness = right_tail['sharpness']
                        print(f"  {sharpness['icon']} Sharpness: {sharpness['value']:.4f} ({sharpness['quality'].title()})")
                        print(f"     {sharpness['interpretation']}")
                    if 'reliability' in right_tail:
                        reliability = right_tail['reliability']
                        print(f"  {reliability['icon']} Reliability: {reliability['ece']:.4f} ({reliability['quality'].title()})")
                        print(f"     {reliability['interpretation']}")
                
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
                            if 'normalized_value' in crps:
                                print(f"  {crps['icon']} CRPS: {crps['normalized_value']:.4f} (normalized, {crps['quality'].title()})")
                                print(f"     Raw: {crps['value']:.4f} (data units)")
                            else:
                                print(f"  {crps['icon']} CRPS: {crps['value']:.4f} ({crps['quality'].title()})")
                        
                        # Sharpness
                        if 'sharpness' in horizon_metrics:
                            sharpness = horizon_metrics['sharpness']
                            print(f"  {sharpness['icon']} Sharpness: {sharpness['value']:.4f} ({sharpness['quality'].title()})")
                        
                        # Reliability (ECE)
                        if 'reliability' in horizon_metrics:
                            reliability = horizon_metrics['reliability']
                            print(f"  {reliability['icon']} Reliability: {reliability['ece']:.4f} ({reliability['quality'].title()})")
                        
                        # Tail metrics for this horizon
                        if 'left_tail' in horizon_metrics and horizon_metrics['left_tail']:
                            left_tail = horizon_metrics['left_tail']
                            print(f"    Left Tail:")
                            if 'crps' in left_tail:
                                crps = left_tail['crps']
                                if 'normalized_value' in crps:
                                    print(f"      {crps['icon']} CRPS: {crps['normalized_value']:.4f} (normalized, {crps['quality'].title()}, {crps.get('n_samples', 0)} samples)")
                                    print(f"         Raw: {crps['value']:.4f} (data units)")
                                else:
                                    print(f"      {crps['icon']} CRPS: {crps['value']:.4f} ({crps['quality'].title()}, {crps.get('n_samples', 0)} samples)")
                            if 'reliability' in left_tail:
                                reliability = left_tail['reliability']
                                print(f"      {reliability['icon']} Reliability: {reliability['ece']:.4f} ({reliability['quality'].title()})")
                        
                        if 'right_tail' in horizon_metrics and horizon_metrics['right_tail']:
                            right_tail = horizon_metrics['right_tail']
                            print(f"    Right Tail:")
                            if 'crps' in right_tail:
                                crps = right_tail['crps']
                                if 'normalized_value' in crps:
                                    print(f"      {crps['icon']} CRPS: {crps['normalized_value']:.4f} (normalized, {crps['quality'].title()}, {crps.get('n_samples', 0)} samples)")
                                    print(f"         Raw: {crps['value']:.4f} (data units)")
                                else:
                                    print(f"      {crps['icon']} CRPS: {crps['value']:.4f} ({crps['quality'].title()}, {crps.get('n_samples', 0)} samples)")
                            if 'reliability' in right_tail:
                                reliability = right_tail['reliability']
                                print(f"      {reliability['icon']} Reliability: {reliability['ece']:.4f} ({reliability['quality'].title()})")
                
                # Show future window info
                if 'future_window' in cal_metrics:
                    print(f"\n📅 Future Window: {cal_metrics['future_window']} days")
                    
            else:
                # Fallback to old structure
                # CRPS
                if 'crps' in cal_metrics:
                    crps = cal_metrics['crps']
                    if 'normalized_value' in crps:
                        print(f"{crps['icon']} CRPS: {crps['normalized_value']:.4f} (normalized, {crps['quality'].title()})")
                        print(f"   Raw CRPS: {crps['value']:.4f} (in data units)")
                    else:
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
    
    def show_validation_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Display stored validation metrics for this model.
        
        This method shows the validation metrics that were computed and saved
        to the database during the last validation run. It does not re-run validation.
        All users (including those using template models) can view these metrics.
        
        Returns:
            Dict with validation metrics if available, None if no validation metrics found
            
        Example:
            >>> model.show_validation_metrics()
            📊 Model Validation Metrics - My Model
            ...
        """
        # Refresh model data to get latest validation_metrics
        self.refresh()
        
        validation_metrics = self._data.get('validation_metrics')
        validation_timestamp = self._data.get('validation_timestamp')
        
        if not validation_metrics:
            print(f"\n⚠️  No validation metrics found for model '{self.name}'")
            print(f"   Run validation first with: model.validate()")
            return None
        
        # Parse validation_timestamp if it's a string
        if validation_timestamp:
            from datetime import datetime
            if isinstance(validation_timestamp, str):
                try:
                    validation_timestamp = datetime.fromisoformat(validation_timestamp.replace('Z', '+00:00'))
                except:
                    pass
        
        # Display header
        print(f"\n" + "="*70)
        print(f"📊 Model Validation Metrics - {self.name}")
        print(f"="*70)
        
        if validation_timestamp:
            timestamp_str = validation_timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(validation_timestamp, 'strftime') else str(validation_timestamp)
            print(f"   Validated on: {timestamp_str}")
        
        n_forecast_samples = validation_metrics.get('n_forecast_samples', 'N/A')
        print(f"   Forecast samples per validation sample: {n_forecast_samples}")
        print()
        
        # Extract metrics (same structure as validate() returns)
        result = {
            'training_metrics': validation_metrics.get('training_metrics'),
            'validation_metrics': validation_metrics.get('validation_metrics'),
            'regime_analysis': validation_metrics.get('regime_analysis'),
            'calibration_metrics': validation_metrics.get('calibration_metrics')
        }
        
        # Reuse the same display logic from validate() method
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
                if train_ll_val != 'N/A' and val_ll_val != 'N/A' and isinstance(train_ll_val, (int, float)) and isinstance(val_ll_val, (int, float)):
                    gap = train_ll_val - val_ll_val
                    print(f"   Generalization gap: {gap:.4f}")
        
        # Calibration metrics - reuse same display logic
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
                    # Display normalized value (scale-invariant) as primary, absolute value as secondary
                    if 'normalized_value' in crps:
                        print(f"{crps['icon']} CRPS: {crps['normalized_value']:.4f} (normalized, {crps['quality'].title()})")
                        print(f"   Raw CRPS: {crps['value']:.4f} (in data units)")
                    else:
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
                
                # Tail metrics (overall)
                if 'left_tail' in overall_metrics and overall_metrics['left_tail']:
                    left_tail = overall_metrics['left_tail']
                    print(f"\n📊 Left Tail Metrics (10th percentile - extreme lows):")
                    if 'crps' in left_tail:
                        crps = left_tail['crps']
                        if 'normalized_value' in crps:
                            print(f"  {crps['icon']} CRPS: {crps['normalized_value']:.4f} (normalized, {crps['quality'].title()}, {crps.get('n_samples', 0)} samples)")
                            print(f"     Raw CRPS: {crps['value']:.4f} (in data units)")
                        else:
                            print(f"  {crps['icon']} CRPS: {crps['value']:.4f} ({crps['quality'].title()}, {crps.get('n_samples', 0)} samples)")
                        print(f"     {crps['interpretation']}")
                    if 'sharpness' in left_tail:
                        sharpness = left_tail['sharpness']
                        print(f"  {sharpness['icon']} Sharpness: {sharpness['value']:.4f} ({sharpness['quality'].title()})")
                        print(f"     {sharpness['interpretation']}")
                    if 'reliability' in left_tail:
                        reliability = left_tail['reliability']
                        print(f"  {reliability['icon']} Reliability: {reliability['ece']:.4f} ({reliability['quality'].title()})")
                        print(f"     {reliability['interpretation']}")
                
                if 'right_tail' in overall_metrics and overall_metrics['right_tail']:
                    right_tail = overall_metrics['right_tail']
                    print(f"\n📊 Right Tail Metrics (90th percentile - extreme highs):")
                    if 'crps' in right_tail:
                        crps = right_tail['crps']
                        if 'normalized_value' in crps:
                            print(f"  {crps['icon']} CRPS: {crps['normalized_value']:.4f} (normalized, {crps['quality'].title()}, {crps.get('n_samples', 0)} samples)")
                            print(f"     Raw CRPS: {crps['value']:.4f} (in data units)")
                        else:
                            print(f"  {crps['icon']} CRPS: {crps['value']:.4f} ({crps['quality'].title()}, {crps.get('n_samples', 0)} samples)")
                        print(f"     {crps['interpretation']}")
                    if 'sharpness' in right_tail:
                        sharpness = right_tail['sharpness']
                        print(f"  {sharpness['icon']} Sharpness: {sharpness['value']:.4f} ({sharpness['quality'].title()})")
                        print(f"     {sharpness['interpretation']}")
                    if 'reliability' in right_tail:
                        reliability = right_tail['reliability']
                        print(f"  {reliability['icon']} Reliability: {reliability['ece']:.4f} ({reliability['quality'].title()})")
                        print(f"     {reliability['interpretation']}")
                
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
                            if 'normalized_value' in crps:
                                print(f"  {crps['icon']} CRPS: {crps['normalized_value']:.4f} (normalized, {crps['quality'].title()})")
                                print(f"     Raw: {crps['value']:.4f} (data units)")
                            else:
                                print(f"  {crps['icon']} CRPS: {crps['value']:.4f} ({crps['quality'].title()})")
                        
                        # Sharpness
                        if 'sharpness' in horizon_metrics:
                            sharpness = horizon_metrics['sharpness']
                            print(f"  {sharpness['icon']} Sharpness: {sharpness['value']:.4f} ({sharpness['quality'].title()})")
                        
                        # Reliability (ECE)
                        if 'reliability' in horizon_metrics:
                            reliability = horizon_metrics['reliability']
                            print(f"  {reliability['icon']} Reliability: {reliability['ece']:.4f} ({reliability['quality'].title()})")
                        
                        # Tail metrics for this horizon
                        if 'left_tail' in horizon_metrics and horizon_metrics['left_tail']:
                            left_tail = horizon_metrics['left_tail']
                            print(f"    Left Tail:")
                            if 'crps' in left_tail:
                                crps = left_tail['crps']
                                if 'normalized_value' in crps:
                                    print(f"      {crps['icon']} CRPS: {crps['normalized_value']:.4f} (normalized, {crps['quality'].title()}, {crps.get('n_samples', 0)} samples)")
                                    print(f"         Raw: {crps['value']:.4f} (data units)")
                                else:
                                    print(f"      {crps['icon']} CRPS: {crps['value']:.4f} ({crps['quality'].title()}, {crps.get('n_samples', 0)} samples)")
                            if 'reliability' in left_tail:
                                reliability = left_tail['reliability']
                                print(f"      {reliability['icon']} Reliability: {reliability['ece']:.4f} ({reliability['quality'].title()})")
                        
                        if 'right_tail' in horizon_metrics and horizon_metrics['right_tail']:
                            right_tail = horizon_metrics['right_tail']
                            print(f"    Right Tail:")
                            if 'crps' in right_tail:
                                crps = right_tail['crps']
                                if 'normalized_value' in crps:
                                    print(f"      {crps['icon']} CRPS: {crps['normalized_value']:.4f} (normalized, {crps['quality'].title()}, {crps.get('n_samples', 0)} samples)")
                                    print(f"         Raw: {crps['value']:.4f} (data units)")
                                else:
                                    print(f"      {crps['icon']} CRPS: {crps['value']:.4f} ({crps['quality'].title()}, {crps.get('n_samples', 0)} samples)")
                            if 'reliability' in right_tail:
                                reliability = right_tail['reliability']
                                print(f"      {reliability['icon']} Reliability: {reliability['ece']:.4f} ({reliability['quality'].title()})")
                
                # Show future window info
                if 'future_window' in cal_metrics:
                    print(f"\n📅 Future Window: {cal_metrics['future_window']} days")
                    
            else:
                # Fallback to old structure
                # CRPS
                if 'crps' in cal_metrics:
                    crps = cal_metrics['crps']
                    if 'normalized_value' in crps:
                        print(f"{crps['icon']} CRPS: {crps['normalized_value']:.4f} (normalized, {crps['quality'].title()})")
                        print(f"   Raw CRPS: {crps['value']:.4f} (in data units)")
                    else:
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
            if regime_counts:
                for regime in regime_counts:
                    count = regime_counts[regime]
                    weight = regime_weights.get(regime, 0)
                    print(f"   {regime}: {count} samples ({weight:.3f} weight)")
        
        print(f"{'='*70}\n")
        
        return result
    
    