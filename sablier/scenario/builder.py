"""Scenario class representing a market scenario for conditional generation"""

import logging
from typing import Optional, Any, List, Dict
from datetime import datetime, timedelta
from ..http_client import HTTPClient

logger = logging.getLogger(__name__)


class Scenario:
    """
    Represents a market scenario for conditional synthetic data generation
    
    A scenario defines the conditioning context for generating synthetic market paths:
    - Past: Recent historical data (fetched from live sources)
    - Future: User-defined or sample-based conditioning
    
    Workflow:
    1. Create scenario (linked to trained model)
    2. Fetch recent past data
    3. Configure future conditioning
    4. Generate synthetic paths
    5. Analyze and validate results
    """
    
    def __init__(self, http_client: HTTPClient, scenario_data: dict, model, interactive: bool = True):
        """
        Initialize Scenario instance
        
        Args:
            http_client: HTTP client for API requests
            scenario_data: Scenario data from API
            model: Associated Model instance
            interactive: Whether to prompt for confirmations (default: True)
        """
        self.http = http_client
        self._data = scenario_data
        self.model = model
        self.interactive = interactive
        
        # Core attributes
        self.id = scenario_data.get('id')
        self.name = scenario_data.get('name')
        self.description = scenario_data.get('description', '')
        self.model_id = scenario_data.get('model_id')
        # n_scenarios removed - number of paths is specified per forecast request
        self.current_step = scenario_data.get('current_step', 'model-selection')
    
    @property
    def status(self) -> str:
        """Get scenario status based on current step"""
        step_to_status = {
            'model-selection': 'created',
            'past-data-fetched': 'past_fetched',
            'future-conditioning-configured': 'configured',
            'paths-generated': 'completed'
        }
        return step_to_status.get(self.current_step, 'unknown')
    
    def __repr__(self):
        return f"Scenario(id='{self.id}', name='{self.name}', status='{self.status}')"
    
    # ============================================
    # PAST DATA FETCHING
    # ============================================
    
    def fetch_recent_past(
        self,
        end_date: str = None,
        fred_api_key: str = None
    ) -> Dict[str, Any]:
        """
        Fetch recent past data from live sources (FRED, Yahoo Finance)
        
        This fetches the most recent historical data, normalizes it using the model's
        parameters, and encodes it with the model's PCA-ICA encoding models.
        
        The number of days fetched is always equal to the model's past_window
        (e.g., 100 days) to match the expected input dimensions for PCA encoding.
        
        Args:
            end_date: End date for fetching (default: today, format: YYYY-MM-DD)
            fred_api_key: FRED API key for data fetching
            
        Returns:
            Dictionary with fetching results
        
        Example:
            >>> scenario.fetch_recent_past()  # Fetches up to today
            >>> scenario.fetch_recent_past(end_date='2024-03-15')
        """
        if self.model is None:
            raise ValueError("Model not loaded. Cannot fetch past data.")
        
        # Determine date range
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Get the model's past window length (must match exactly for PCA encoding)
        # This is fixed by the model and cannot be changed
        past_window = None
        try:
            sample = self._get_test_sample(0, include_data=True)
            if sample and sample.get('conditioning_data'):
                # Get first conditioning item and count its dates
                first_item = sample['conditioning_data'][0]
                if 'dates' in first_item:
                    past_window = len(first_item['dates'])
        except:
            pass
        
        if past_window is None:
            past_window = 100  # Default fallback
        
        print(f"  Using model's past_window: {past_window} days")
        
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=past_window)).strftime('%Y-%m-%d')
        
        print(f"[Scenario] Fetching recent past data...")
        print(f"  Period: {start_date} to {end_date} ({past_window} days)")
        print(f"  Features: {len(self.model.input_features)}")
        
        # Step 1: Fetch data from FRED/Yahoo
        print("📡 Step 1/3: Fetching data from live sources...")
        
        # Call backend data fetch endpoint directly (without modifying model)
        # Match the payload structure from Model.fetch_data()
        processing_config = self.model._data.get('processing_config', {})
        
        fetch_payload = {
            'model_id': self.model_id,
            'features': self.model.input_features,
            'start_date': start_date,
            'end_date': end_date,
            'fred_api_key': fred_api_key,
            'processing_config': processing_config or {
                'interpolation': {
                    'method': 'linear',
                    'maxGapLength': {
                        feature.get('display_name') or feature.get('name'): 7
                        for feature in self.model.input_features
                    }
                }
            },
            'include_processed_data': True,  # Request data in response for scenarios
            'skip_database_save': True  # Don't modify model's training_data table
        }
        
        fetch_response = self.http.post('/api/v1/data/fetch', fetch_payload)
        
        # Get processed data (backend returns uncompressed for API responses)
        processed_data = fetch_response.get('processed_data', [])
        
        if not processed_data:
            raise ValueError("No processed data returned from backend. Check date range and features.")
        
        print(f"  Fetched {len(processed_data)} data points")
        
        # Data will be stored in recent_past_data table (separate from scenarios table)
        # No need to update scenario record here
        print(f"  ✓ Data fetched")
        
        # Step 2: Normalize using model's normalization parameters
        print("🔢 Step 2/3: Normalizing data...")
        norm_params = self.model._data.get('feature_normalization_params', {})
        
        # Restructure flat data into feature-based dict
        feature_data = {}
        for point in processed_data:
            feature_name = point['feature']
            if feature_name not in feature_data:
                feature_data[feature_name] = []
            feature_data[feature_name].append({
                'date': point['date'],
                'value': point['value']
            })
        
        # Build normalized conditioning_data structure
        # IMPORTANT: For PAST window, encode ALL features (both conditioning AND target)
        # Why? Recent past values of ALL features (including Treasury rates) are useful conditioning!
        # Only FUTURE target features are excluded (those are what we predict)
        conditioning_data = []
        normalized_data = {}  # For storing in scenario
        
        print(f"  Processing all {len(self.model.input_features)} features for past window")
        
        for feature in self.model.input_features:
            feature_name = feature.get('display_name') or feature['name']
            # Extract values from processed data
            if feature_name not in feature_data:
                print(f"  ⚠️  No data for {feature_name}, skipping")
                continue
            
            # Sort by date and extract values
            sorted_data = sorted(feature_data[feature_name], key=lambda x: x['date'])
            values = [item['value'] for item in sorted_data]
            dates = [item['date'] for item in sorted_data]
            
            # Normalize (handle None values)
            norm = norm_params.get(feature_name, {'mean': 0, 'std': 1})
            mean = norm.get('mean', 0)
            std = norm.get('std', 1)
            
            normalized_values = []
            for v in values:
                if v is not None:
                    normalized_values.append((v - mean) / std)
                else:
                    normalized_values.append(None)
            
            # Truncate to past_window length (most recent N days)
            truncated_values = normalized_values[-past_window:] if len(normalized_values) >= past_window else normalized_values
            truncated_dates = dates[-past_window:] if len(dates) >= past_window else dates
            
            # Check if this feature has all non-None values
            has_all_values = all(v is not None for v in truncated_values)
            
            conditioning_data.append({
                'feature_name': feature_name,  # Note: backend expects 'feature_name', not 'feature'
                'temporal_tag': 'past',
                'normalized_series': truncated_values,
                'has_complete_data': has_all_values  # Track availability
            })
            
            normalized_data[feature_name] = {
                'dates': truncated_dates,
                'values': truncated_values,
                'has_complete_data': has_all_values
            }
        
        print(f"  Normalized {len(conditioning_data)} features")
        
        # Note: We don't store normalized_data in the scenario table
        # It's just an intermediate step. We store raw (fetched_past_data) and encoded (encoded_fetched_past)
        
        # Step 3: Encode using model's encoding models
        print("🔐 Step 3/3: Encoding recent data...")
        
        sample_data = {
            'conditioning_data': conditioning_data,
            'target_data': []
        }
        
        encode_response = self.http.post('/api/v1/ml/encode?source=inline', {
            'user_id': self.model._data.get('user_id'),
            'model_id': self.model_id,
            'encoding_type': 'pca-ica',
            'sample_data': sample_data
        })
        
        encoded_sample = encode_response.get('encoded_sample', {})
        encoded_fetched_past = encoded_sample.get('encoded_conditioning_data', [])
        
        print(f"  Encoded {len(encoded_fetched_past)} feature windows")
        
        # Build feature availability map for past data
        past_feature_availability = {}
        for feature_name, data in normalized_data.items():
            past_feature_availability[feature_name] = {
                'past': data.get('has_complete_data', True)
            }
        
        # Count features with missing data
        missing_count = sum(1 for avail in past_feature_availability.values() if not avail['past'])
        if missing_count > 0:
            print(f"  ⚠️  {missing_count}/{len(past_feature_availability)} features have incomplete past data")
        
        # Don't update scenario in DB yet - data will be stored when forecast() is called
        # For now, just keep in memory
        print(f"  ✓ Data ready for forecasting")
        
        # Update local data
        self._data['encoded_fetched_past'] = encoded_fetched_past
        self._data['normalized_fetched_past'] = conditioning_data
        self._data['past_feature_availability'] = past_feature_availability
        self.current_step = 'past-data-fetched'
        
        print("✅ Past data fetched, normalized, and encoded")
        
        return {
            "status": "success",
            "start_date": start_date,
            "end_date": end_date,
            "features_fetched": len(self.model.input_features)
        }
    
    # ============================================
    # FUTURE CONDITIONING CONFIGURATION
    # ============================================
    
    def configure_future_conditioning(
        self,
        mode: str = 'from_sample',
        reference_date: str = None,
        reference_sample_id: str = None,
        manual_config: Dict[str, List[float]] = None,
        features: List[str] = None
    ) -> Dict[str, Any]:
        """
        Configure future conditioning for scenario
        
        Two modes:
        1. 'from_sample': Use encoded components from an existing sample
        2. 'manual': Manually specify component values
        
        Args:
            mode: 'from_sample' or 'manual'
            reference_date: Date to find closest sample (for from_sample mode)
            reference_sample_id: Specific sample ID (for from_sample mode)
            manual_config: Dict of {feature_name: [component_values]} (for manual mode)
            features: List of features to configure (default: all conditioning features)
        
        Returns:
            Configuration result
        
        Examples:
            >>> # Easy mode - from existing sample
            >>> scenario.configure_future_conditioning(
            ...     mode='from_sample',
            ...     reference_date='2024-01-15'
            ... )
            
            >>> # Advanced mode - manual components
            >>> scenario.configure_future_conditioning(
            ...     mode='manual',
            ...     manual_config={
            ...         "10-Year Treasury": [0.1, 0.2, ...],  # 20 components
            ...         "Gold Price": [0.0, 0.0, ...]
            ...     }
            ... )
        """
        if mode not in ['from_sample', 'manual']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'from_sample' or 'manual'")
        
        print(f"[Scenario] Configuring future conditioning (mode: {mode})...")
        
        if mode == 'from_sample':
            return self._configure_from_sample(reference_date, reference_sample_id, features)
        else:
            return self._configure_manual(manual_config)
    
    def _configure_from_sample(
        self,
        reference_date: str = None,
        reference_sample_id: str = None,
        features: List[str] = None
    ) -> Dict[str, Any]:
        """
        Configure future conditioning using components from an existing sample.
        
        IMPORTANT: This method now correctly processes conditioning data by:
        1. Extracting encoded_normalized_residuals (user's conditioning choice)
        2. Reconstructing them to normalized_series 
        3. Re-encoding to get CORRECT encoded_normalized_series for model input
        
        The historical encoded_normalized_series from the sample is NOT used directly
        because it represents what actually happened, not what we want to condition on.
        """
        
        # Get sample ID (either provided or from previous selection)
        if reference_sample_id is None:
            if hasattr(self, '_selected_sample_id'):
                reference_sample_id = self._selected_sample_id
                print(f"  Using previously selected sample: {reference_sample_id[:8]}...")
            elif reference_date is not None:
                print(f"  Finding closest sample to {reference_date}...")
                reference_sample_id = self.get_closest_sample(reference_date)
                print(f"  Selected sample: {reference_sample_id}")
            else:
                raise ValueError("Either reference_date or reference_sample_id must be provided, or call select_historical_scenario() first")
        
        # Get features (from previous selection or parameter)
        if features is None:
            if hasattr(self, '_selected_features') and self._selected_features:
                features = self._selected_features
            else:
                sample_config = self.model._data.get('sample_config', {})
                features = sample_config.get('conditioningFeatures', [])
        
        # Fetch full sample with encoded data
        print(f"  Fetching sample {reference_sample_id[:8]}... with encoded data...")
        sample_response = self.http.get(
            f'/api/v1/models/{self.model_id}/samples',
            params={'include_data': 'true', 'limit': 1000}
        )
        
        samples = sample_response.get('samples', [])
        sample = next((s for s in samples if s['id'] == reference_sample_id), None)
        
        if not sample:
            raise ValueError(f"Sample {reference_sample_id} not found")
        
        # Extract future conditioning from sample
        encoded_conditioning_data = sample.get('encoded_conditioning_data', [])
        
        # For PLOTTING: Extract encoded_normalized_residuals
        # Note: Always extract ALL features for plotting, ignoring the 'features' filter
        # because 'features' contains individual feature names but encoded data has group IDs
        encoded_residuals_for_plotting = {}
        all_feature_residuals = {}  # Track all features (selected or not)
        
        for item in encoded_conditioning_data:
            temporal_tag = item.get('temporal_tag')
            feature_name = item.get('feature')
            
            if temporal_tag == 'future':
                # For plotting, we specifically need encoded_normalized_residuals
                encoded_residuals = item.get('encoded_normalized_residuals', [])
                
                if encoded_residuals and len(encoded_residuals) > 0:
                    all_feature_residuals[feature_name] = encoded_residuals
                    # Always add to plotting (ignoring features filter since it has individual names not group IDs)
                    encoded_residuals_for_plotting[feature_name] = encoded_residuals
        
        if not all_feature_residuals:
            raise ValueError("No future conditioning residuals found in sample")
        
        print(f"  ✓ Extracted {len(encoded_residuals_for_plotting)} selected features for conditioning")
        print(f"  ✓ Extracted {len(all_feature_residuals)} total features from sample")
        
        # ====================================================================
        # RECONSTRUCT: encoded_normalized_residuals → normalized_series
        # ====================================================================
        print(f"  🔄 Reconstructing conditioning residuals to normalized series...")
        
        # Build encoded_windows for reconstruction
        encoded_windows = []
        for feature_name, encoded_residuals in all_feature_residuals.items():
            # Check if this is a group by looking in encoded_conditioning_data
            original_encoded_item = next((x for x in encoded_conditioning_data 
                                         if x.get('feature') == feature_name and x.get('temporal_tag') == 'future'), None)
            
            window = {
                "feature": feature_name,
                "temporal_tag": "future",
                "data_type": "encoded_normalized_residuals",
                "encoded_values": encoded_residuals
            }
            
            # Preserve group metadata if present
            if original_encoded_item:
                if original_encoded_item.get('is_group'):
                    window['is_group'] = True
                    window['is_multivariate'] = original_encoded_item.get('is_multivariate', False)
                    window['group_features'] = original_encoded_item.get('group_features', [])
                    window['n_components'] = original_encoded_item.get('n_components')
            
            encoded_windows.append(window)
        
        # Get reference values (last normalized value for each feature/group)
        reference_values = {}
        normalized_past = self._data.get('normalized_fetched_past', [])
        encoded_past = self._data.get('encoded_fetched_past', [])
        
        
        # Check if model uses feature groups
        feature_groups = self.model._data.get('feature_groups')
        
        for feature_or_group_name in all_feature_residuals.keys():
            # Check if this is a group ID by looking in encoded_conditioning_data (from sample)
            encoded_item = next((x for x in encoded_conditioning_data 
                               if x.get('feature') == feature_or_group_name and x.get('temporal_tag') == 'future'), None)
            
            # If not found in sample, try encoded_past (from fetch_recent_past)
            if not encoded_item:
                encoded_item = next((x for x in encoded_past if x.get('feature') == feature_or_group_name), None)
            
            if encoded_item and encoded_item.get('is_group'):
                # This is a group - get reference values for all features in the group
                group_features = encoded_item.get('group_features', [])
                is_multivariate = encoded_item.get('is_multivariate', False)
                
                
                if is_multivariate:
                    # For multivariate groups, we need all feature reference values
                    # The reconstruct endpoint expects them as individual entries, not nested dicts
                    for feat in group_features:
                        for item in normalized_past:
                            if item.get('feature_name') == feat or item.get('feature') == feat:
                                reference_values[feat] = item['normalized_series'][-1]
                                logger.debug(f"  Found reference for {feat}: {item['normalized_series'][-1]}")
                                break
                    
                    # Check if all features were found
                    missing = set(group_features) - set(reference_values.keys())
                    if missing:
                        raise ValueError(f"Missing reference values for features in group {feature_or_group_name}: {missing}")
                else:
                    # Univariate group - single reference value
                    # Store under feature name (not group ID) to match multivariate behavior
                    feat = group_features[0]
                    for item in normalized_past:
                        if item.get('feature_name') == feat or item.get('feature') == feat:
                            reference_values[feat] = item['normalized_series'][-1]
                            logger.debug(f"  Found reference for univariate group {feature_or_group_name}: {item['normalized_series'][-1]}")
                            break
            else:
                # Individual feature (no groups)
                for item in normalized_past:
                    if item.get('feature_name') == feature_or_group_name or item.get('feature') == feature_or_group_name:
                        reference_values[feature_or_group_name] = item['normalized_series'][-1]
                        break
        
        # Verify all reference values were found
        # For multivariate groups, individual features are in reference_values, not group IDs
        # So we need a smarter check
        missing_groups = []
        for feature_or_group_name in all_feature_residuals.keys():
            # Check if this is a group
            encoded_item = next((x for x in encoded_conditioning_data 
                               if x.get('feature') == feature_or_group_name and x.get('temporal_tag') == 'future'), None)
            if not encoded_item:
                encoded_item = next((x for x in encoded_past if x.get('feature') == feature_or_group_name), None)
            
            if encoded_item and encoded_item.get('is_group'):
                group_features = encoded_item.get('group_features', [])
                is_multivariate = encoded_item.get('is_multivariate', False)
                
                if is_multivariate:
                    # Check if all features in the group have reference values
                    if not all(feat in reference_values for feat in group_features):
                        missing_groups.append(feature_or_group_name)
                else:
                    # Univariate group - check if the single feature has a reference value
                    if group_features[0] not in reference_values:
                        missing_groups.append(feature_or_group_name)
            else:
                # Individual feature
                if feature_or_group_name not in reference_values:
                    missing_groups.append(feature_or_group_name)
        
        if missing_groups:
            raise ValueError(f"Missing reference values for features/groups: {missing_groups}")
        
        # Call reconstruct endpoint (ONE call for all features)
        reconstruct_payload = {
            "user_id": self.model._data.get("user_id"),
            "model_id": self.model_id,
            "encoded_source": "inline",
            "encoded_windows": encoded_windows,
            "reference_source": "inline",
            "reference_values": reference_values,
            "output_destination": "return"
        }
        
        reconstruct_response = self.http.post('/api/v1/ml/reconstruct', reconstruct_payload)
        reconstructions = reconstruct_response.get('reconstructions', [])
        
        if not reconstructions:
            raise ValueError("Reconstruction failed - no reconstructions returned")
        
        print(f"  ✓ Reconstructed {len(reconstructions)} features to normalized series")
        
        # ====================================================================
        # ENCODE: normalized_series → encoded_normalized_series
        # ====================================================================
        print(f"  🔄 Encoding reconstructed series...")
        
        # Get encoding type from model metadata
        model_metadata = self.model._data.get('model_metadata', {})
        encoding_type = model_metadata.get('encoding_type', 'pca-ica')
        
        # Build conditioning_data for encoding (mimics sample structure)
        conditioning_data_for_encoding = []
        reconstructed_series_by_feature = {}  # For caching/debugging
        
        for recon in reconstructions:
            feature_name = recon['feature']
            reconstructed_values = recon['reconstructed_values']
            
            # Cache for debugging
            reconstructed_series_by_feature[feature_name] = reconstructed_values
            
            conditioning_data_for_encoding.append({
                "feature_name": feature_name,  # encode endpoint expects "feature_name"
                "temporal_tag": "future",
                "normalized_series": reconstructed_values
            })
        
        # Call encode endpoint (ONE call for all features)
        encode_payload = {
            "user_id": self.model._data.get("user_id"),
            "model_id": self.model_id,
            "encoding_type": encoding_type,
            "sample_data": {
                "conditioning_data": conditioning_data_for_encoding,
                "target_data": []  # Empty for this use case
            }
        }
        
        encode_response = self.http.post('/api/v1/ml/encode?source=inline', encode_payload)
        encoded_sample = encode_response.get('encoded_sample', {})
        encoded_conditioning_data_new = encoded_sample.get('encoded_conditioning_data', [])
        
        if not encoded_conditioning_data_new:
            raise ValueError("Encoding failed - no encoded data returned")
        
        print(f"  ✓ Encoded {len(encoded_conditioning_data_new)} features")
        
        # ====================================================================
        # BUILD MODEL INPUT: Use NEW encodings (not historical!)
        # ====================================================================
        future_conditioning_for_model = []
        
        for encoded_item in encoded_conditioning_data_new:
            feature_name = encoded_item['feature']
            is_selected = not features or feature_name in features
            
            encoded_series = encoded_item.get('encoded_normalized_series', [])
            future_conditioning_for_model.append({
                'feature': feature_name,
                'temporal_tag': 'future',
                'encoded_normalized_series': encoded_series if is_selected else [0.0] * len(encoded_series)
            })
        
        print(f"  ✓ Built model input with {len(future_conditioning_for_model)} features")
        
        # Store all representations locally
        self._data['encoded_future_conditioning'] = future_conditioning_for_model
        self._data['encoded_residuals_for_plotting'] = encoded_residuals_for_plotting
        self._data['reconstructed_conditioning_series'] = reconstructed_series_by_feature  # For debugging
        self._data['reference_sample_id'] = reference_sample_id
        
        # Update scenario in database (only model input data)
        update_payload = {
            'encoded_future_conditioning': future_conditioning_for_model,
            'current_step': 'future-conditioning-configured'
        }
        
        self.http.patch(f'/api/v1/scenarios/{self.id}', update_payload)
        
        # Update local state
        self._data['current_step'] = 'future-conditioning-configured'
        self.current_step = 'future-conditioning-configured'
        
        print(f"✅ Future conditioning configured from sample {reference_sample_id[:8]}...")
        
        return {
            "status": "success",
            "mode": "from_sample",
            "reference_sample_id": reference_sample_id,
            "features_configured": len([f for f in future_conditioning_for_model if f['encoded_normalized_series'] != [0.0] * len(f['encoded_normalized_series'])])
        }
    
    def _configure_manual(self, manual_config: Dict[str, List[float]]) -> Dict[str, Any]:
        """Configure future conditioning manually"""
        
        if not manual_config:
            raise ValueError("manual_config must be provided for manual mode")
        
        print(f"  Configuring {len(manual_config)} features manually...")
        
        # Build encoded future conditioning
        future_conditioning = []
        for feature_name, components in manual_config.items():
            if not isinstance(components, list):
                raise ValueError(f"Components for {feature_name} must be a list")
            
            future_conditioning.append({
                'feature': feature_name,
                'temporal_tag': 'future',
                'encoded_normalized_series': components
            })
        
        # Update scenario in database
        update_payload = {
            'encoded_future_conditioning': future_conditioning,
            'current_step': 'future-conditioning-configured'
        }
        
        self.http.patch(f'/api/v1/scenarios/{self.id}', update_payload)
        
        # Update local data
        self._data['encoded_future_conditioning'] = future_conditioning
        self._data['current_step'] = 'future-conditioning-configured'
        self.current_step = 'future-conditioning-configured'
        
        print(f"✅ Future conditioning configured manually")
        
        return {
            "status": "success",
            "mode": "manual",
            "features_configured": len(manual_config)
        }
    
    # ============================================
    # PATH GENERATION
    # ============================================
    
    
    
    def generate_paths(self, 
                          n_samples: int = None, 
                          conditioning_features: List[str] = None) -> 'SyntheticData':
        """
        Generate synthetic market paths using Vine Copula model
        
        Args:
            n_samples: Number of paths to generate (default: scenario's n_scenarios)
            conditioning_features: Optional list of features to condition on (others marginalized)
        
        Returns:
            SyntheticData instance containing generated paths
        
        Example:
            >>> # Full conditioning
            >>> synthetic_data = scenario.generate_paths(n_samples=1000)
            
            >>> # Partial conditioning (marginalized inference)
            >>> synthetic_data = scenario.generate_paths(
            ...     n_samples=1000,
            ...     conditioning_features=['Fed Funds Rate', 'VIX']
            ... )
        """
        from ..synthetic_data import SyntheticData
        
        if n_samples is None:
            n_samples = 100  # Default number of forecast paths
        
        # Verify scenario is configured
        if self.current_step not in ['future-conditioning-configured', 'paths-generated']:
            raise ValueError(
                f"Scenario must be configured before generating paths. "
                f"Current step: {self.current_step}. "
                f"Call fetch_recent_past() and configure_future_conditioning() first."
            )
        
        print(f"[Scenario] Generating {n_samples} synthetic paths...")
        
        # Show conditioning mode
        if conditioning_features:
            print(f"🔬 Partial conditioning: {len(conditioning_features)} features")
            print(f"   Conditioned: {', '.join(conditioning_features)}")
            print(f"   Marginalized: others")
        else:
            print(f"📊 Full conditioning: all configured features")
        
        # Call forecast endpoint with scenario mode
        print("📊 Step 1/2: Generating forecast samples...")
        
        if not self.model:
            raise ValueError("Model not loaded. Cannot generate paths.")
        
        payload = {
            'user_id': self.model._data.get('user_id'),
            'model_id': self.model_id,
            'conditioning_source': 'scenario',
            'scenario_id': self.id,
            'n_samples': n_samples
        }
        
        # Add conditioning_features for partial conditioning
        if conditioning_features:
            payload['conditioning_features'] = conditioning_features
        
        # Call forecast endpoint
        forecast_response = self.http.post('/api/v1/ml/forecast', payload)
        
        forecast_samples = forecast_response.get('forecasts', [])
        print(f"  Generated {len(forecast_samples)} samples")
        
        # Reconstruct forecast samples
        print("🔄 Step 2/2: Reconstructing to original scale...")
        reconstructed_paths = self._reconstruct_scenario_forecast(forecast_samples)
        
        print(f"✅ Generated {len(reconstructed_paths)} synthetic paths")
        
        # Update scenario status
        self._update_step('paths-generated')
        
        # Create SyntheticData instance
        synthetic_data = SyntheticData(
            paths=reconstructed_paths,
            scenario=self,
            model=self.model
        )
        
        return synthetic_data
    
    
    
    # ============================================
    # UTILITY METHODS
    # ============================================
    
    
    
    def _reconstruct_scenario_forecast(self, forecast_samples: List[Dict]) -> List[Dict]:
        """
        Reconstruct forecast samples using scenario's fetched past as reference
        
        IMPORTANT: For scenarios, we use the REAL fetched past data as reference,
        not the historical sample. This simulates "what if the historical scenario
        happened starting from today".
        
        Args:
            forecast_samples: Forecast outputs (raw format with component keys)
        
        Returns:
            List of reconstructed paths with dates and values
        """
        # Get scenario data for reference values and dates
        scenario = self.http.get(f'/api/v1/scenarios/{self.id}')
        fetched_past = scenario.get('fetched_past_data', [])
        
        if not fetched_past:
            raise ValueError("No fetched_past_data in scenario")
        
        # Build reference values from FETCHED PAST (today's data), not historical sample
        # We need to normalize the fetched_past data to get reference values
        reference_values = {}
        norm_params = self.model._data.get('feature_normalization_params', {})
        
        # Group fetched_past by feature and get last value
        feature_last_values = {}
        for point in fetched_past:
            feature = point.get('feature')
            date = point.get('date')
            value = point.get('value')
            
            if feature and date and value is not None:
                if feature not in feature_last_values or date > feature_last_values[feature]['date']:
                    feature_last_values[feature] = {'date': date, 'value': value}
        
        # Normalize the last values to get reference values
        for feature, data in feature_last_values.items():
            value = data['value']
            mean = norm_params.get(feature, {}).get('mean', 0)
            std = norm_params.get(feature, {}).get('std', 1)
            if std == 0:
                std = 1  # Avoid division by zero
            normalized_value = (value - mean) / std
            reference_values[feature] = normalized_value
        
        # Parse forecast samples and group by sample_idx
        # Returns keys like "target_feature_name_future_normalized_residuals_0"
        samples_by_idx = {}
        
        for sample_idx, forecast_sample in enumerate(forecast_samples):
            # Group components by (source, feature, temporal_tag, data_type)
            windows = {}
            for key, value in forecast_sample.items():
                if key == '_group_metadata':
                    continue
                    
                # Parse key: "source_feature_temporal_tag_data_type_component_idx"
                parts = key.rsplit('_', 1)
                if len(parts) != 2:
                    continue
                    
                window_key = parts[0]
                try:
                    component_idx = int(parts[1])
                except ValueError:
                    continue
                
                if window_key not in windows:
                    windows[window_key] = {}
                windows[window_key][component_idx] = value
            
            # Build encoded_windows for this sample
            encoded_windows = []
            for window_key, components in windows.items():
                # Parse window_key to extract parts
                # Format: "source_feature_temporal_tag_data_type"
                # Example: "target_group_1_future_normalized_residuals"
                # Parts: ["target", "group", "1", "future", "normalized", "residuals"]
                parts = window_key.split('_')
                if len(parts) < 4:
                    continue
                
                source = parts[0]  # "target" or "conditioning"
                
                # Only reconstruct TARGET components, not conditioning components
                # Conditioning should already be reconstructed when scenario was created
                if source != "target":
                    continue
                
                # Find temporal_tag (it's either "past" or "future")
                temporal_idx = None
                for i, part in enumerate(parts):
                    if part in ['past', 'future']:
                        temporal_idx = i
                        break
                
                if temporal_idx is None:
                    continue
                
                # feature is everything between source and temporal_tag
                feature = '_'.join(parts[1:temporal_idx])
                temporal_tag = parts[temporal_idx]
                
                # data_type is everything after temporal_tag
                data_type = '_'.join(parts[temporal_idx+1:])
                
                # Convert components dict to list
                encoded_values = [components[i] for i in sorted(components.keys())]
                
                encoded_window = {
                    "feature": feature,
                    "temporal_tag": temporal_tag,
                    "data_type": "encoded_" + data_type,  # Prepend "encoded_" as expected by backend
                    "encoded_values": encoded_values,
                    "_sample_idx": sample_idx
                }
                
                # Add group metadata if available
                if '_group_metadata' in forecast_sample and feature in forecast_sample['_group_metadata']:
                    metadata = forecast_sample['_group_metadata'][feature]
                    encoded_window['is_group'] = metadata.get('is_group', False)
                    encoded_window['is_multivariate'] = metadata.get('is_multivariate', False)
                    encoded_window['group_features'] = metadata.get('group_features', [])
                
                encoded_windows.append(encoded_window)
            
            samples_by_idx[sample_idx] = encoded_windows
        
        # BATCH RECONSTRUCTION: Collect all encoded windows from all samples
        print(f"  Collecting {len(samples_by_idx)} samples for batch reconstruction...")
        all_encoded_windows = []
        sample_window_mapping = {}  # Track which windows belong to which sample
        
        for sample_idx in sorted(samples_by_idx.keys()):
            sample_windows = samples_by_idx[sample_idx]
            all_encoded_windows.extend(sample_windows)
            sample_window_mapping[sample_idx] = len(sample_windows)
        
        print(f"  Collected {len(all_encoded_windows)} total encoded windows")
        
        # SINGLE BATCH API CALL: Reconstruct all windows at once
        print(f"  Making single batch reconstruction call...")
        payload = {
            "user_id": self.model._data.get("user_id"),
            "model_id": self.model_id,
            "encoded_source": "inline",
            "encoded_windows": all_encoded_windows,
            "reference_source": "inline",
            "reference_values": reference_values,  # Use FETCHED PAST as reference
            "output_destination": "return"
        }
        
        response = self.http.post('/api/v1/ml/reconstruct', payload)
        all_reconstructions_raw = response.get('reconstructions', [])
        
        print(f"  Received {len(all_reconstructions_raw)} reconstructed windows")
        
        # PARSE BATCH RESPONSE: Split back into individual samples
        print(f"  Parsing batch response back into individual samples...")
        all_reconstructed = []
        current_idx = 0
        
        for sample_idx in sorted(samples_by_idx.keys()):
            n_windows = sample_window_mapping[sample_idx]
            sample_reconstructions = all_reconstructions_raw[current_idx:current_idx + n_windows]
            current_idx += n_windows
            all_reconstructed.append(sample_reconstructions)
        
        # Generate dates relative to FETCHED PAST (today's data)
        sample_config = self.model._data.get('sample_config', {})
        future_window = sample_config.get('futureWindow', 50)
        
        from datetime import datetime, timedelta
        last_date_str = max(point['date'] for point in fetched_past)
        start_date = datetime.strptime(last_date_str, '%Y-%m-%d') + timedelta(days=1)
        future_dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(future_window)]
        
        # Convert to scenario format and add past window for plotting
        # Extract past dates and values from fetched_past
        past_by_feature = {}
        for point in fetched_past:
            feature = point.get('feature')
            if feature not in past_by_feature:
                past_by_feature[feature] = {'dates': [], 'values': []}
            past_by_feature[feature]['dates'].append(point['date'])
            past_by_feature[feature]['values'].append(point['value'])
        
        # Sort by date
        for feature in past_by_feature:
            sorted_pairs = sorted(zip(past_by_feature[feature]['dates'], past_by_feature[feature]['values']))
            past_by_feature[feature]['dates'] = [d for d, v in sorted_pairs]
            past_by_feature[feature]['values'] = [v for d, v in sorted_pairs]
        
        reconstructed_samples = []
        
        for reconstructions in all_reconstructed:
            sample_dict = {}
            
            for window in reconstructions:
                feature = window['feature']
                values = window['reconstructed_values']
                
                # Include both past (fetched) and future (forecasted)
                sample_dict[feature] = {
                    'past': past_by_feature.get(feature, {'dates': [], 'values': []}),
                    'future': {
                        'dates': future_dates[:len(values)],
                        'values': values
                    }
                }
            
            reconstructed_samples.append(sample_dict)
        
        return reconstructed_samples
    
    def select_historical_scenario(
        self, 
        target_date: str, 
        features: List[str] = None
    ) -> Dict[str, Any]:
        """
        Select a historical scenario by finding the sample whose FUTURE window
        starts closest to the target date.
        
        This is the primary method for scenario definition. It finds a historical
        period that matches your target date and prepares it for conditioning.
        
        Args:
            target_date: Target date for scenario (YYYY-MM-DD)
            features: Conditioning features to use (default: all conditioning features)
        
        Returns:
            Dict with scenario info and selected sample details
        
        Example:
            >>> # Simulate COVID crash scenario
            >>> scenario.select_historical_scenario('2020-03-15', 
            ...     features=['VIX', 'S&P 500'])
        """
        print(f"[Scenario] Selecting historical scenario for {target_date}...")
        
        target = datetime.strptime(target_date, '%Y-%m-%d')
        
        # Fetch all samples with metadata
        response = self.http.get(
            f'/api/v1/models/{self.model_id}/samples',
            params={'limit': 1000}
        )
        
        samples = response.get('samples', [])
        
        if not samples:
            raise ValueError("No samples found for model")
        
        # Find sample where future_window_start is closest to target_date
        # future_start = end_date + 1 day
        closest_sample = None
        min_distance = float('inf')
        
        for sample in samples:
            if sample.get('end_date'):
                end_date = datetime.strptime(sample['end_date'], '%Y-%m-%d')
                future_start = end_date + timedelta(days=1)
                distance = abs((future_start - target).days)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_sample = sample
                    closest_future_start = future_start
        
        if closest_sample is None:
            raise ValueError("Could not find suitable sample")
        
        # Display scenario info
        print()
        print("=" * 70)
        print("📅 SELECTED HISTORICAL SCENARIO")
        print("=" * 70)
        print(f"  Target Date: {target_date}")
        print(f"  Sample ID: {closest_sample['id'][:8]}...")
        print(f"  Future Window Start: {closest_future_start.strftime('%Y-%m-%d')}")
        print(f"  Distance from Target: {min_distance} days")
        print(f"  Split: {closest_sample.get('split_type', 'unknown')}")
        print(f"  Sample Period: {closest_sample.get('start_date')} to {closest_sample.get('end_date')}")
        
        # Store selection for next step
        self._selected_sample_id = closest_sample['id']
        self._selected_features = features
        
        # Get conditioning features if not specified
        if features is None:
            sample_config = self.model._data.get('sample_config', {})
            features = sample_config.get('conditioningFeatures', [])
        
        print(f"  Conditioning Features: {len(features)}")
        for feat in features[:5]:  # Show first 5
            print(f"    • {feat}")
        if len(features) > 5:
            print(f"    ... and {len(features) - 5} more")
        print()
        
        return {
            'status': 'success',
            'sample_id': closest_sample['id'],
            'target_date': target_date,
            'future_start_date': closest_future_start.strftime('%Y-%m-%d'),
            'distance_days': min_distance,
            'split': closest_sample.get('split_type'),
            'features': features
        }
    
    def get_closest_sample(self, target_date: str, split: str = None) -> str:
        """
        Find sample with date closest to target_date
        
        DEPRECATED: Use select_historical_scenario() instead for better
        future window matching.
        
        Args:
            target_date: Target date (YYYY-MM-DD)
            split: Optional filter by split type
        
        Returns:
            Sample ID of closest sample
        
        Example:
            >>> sample_id = scenario.get_closest_sample('2024-01-15')
        """
        target = datetime.strptime(target_date, '%Y-%m-%d')
        
        # Fetch samples
        params = {'limit': 1000}
        if split:
            params['split_type'] = split
        
        response = self.http.get(
            f'/api/v1/models/{self.model_id}/samples',
            params=params
        )
        
        samples = response.get('samples', [])
        
        if not samples:
            raise ValueError("No samples found for model")
        
        # Find closest by either start_date or end_date
        closest_sample = None
        min_distance = float('inf')
        
        for sample in samples:
            # Try both start and end dates
            for date_field in ['start_date', 'end_date']:
                if sample.get(date_field):
                    sample_date = datetime.strptime(sample[date_field], '%Y-%m-%d')
                    distance = abs((sample_date - target).days)
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_sample = sample
        
        if closest_sample is None:
            raise ValueError("Could not find closest sample")
        
        print(f"  Closest sample: {closest_sample['id']} (distance: {min_distance} days)")
        
        return closest_sample['id']
    
    def plot_conditioning_scenario(
        self,
        features: List[str] = None,
        save_path: str = None,
        show: bool = True
    ):
        """
        Plot the conditioning scenario showing past (fetched) + future (selected).
        
        This visualizes what the model will be conditioned on before generating paths.
        Shows:
        - Past window: Recent fetched data (denormalized)
        - Future window: Selected historical scenario (decoded residuals → denormalized)
        - Smooth continuity at the boundary
        
        Args:
            features: Features to plot (default: all selected conditioning features)
            save_path: Path to save plot
            show: Whether to display plot
        
        Example:
            >>> scenario.select_historical_scenario('2020-03-15')
            >>> scenario.configure_future_conditioning(mode='from_sample')
            >>> scenario.plot_conditioning_scenario()
        """
        # Check if scenario is configured
        if 'encoded_residuals_for_plotting' not in self._data:
            raise ValueError(
                "Scenario must be configured before plotting. "
                "Call configure_future_conditioning() first."
            )
        
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import numpy as np
        
        # Get data - load from database if not in local cache
        fetched_past = self._data.get('fetched_past_data', [])
        if not fetched_past:
            # Load from database
            scenario_data = self.http.get(f'/api/v1/scenarios/{self.id}')
            fetched_past = scenario_data.get('fetched_past_data', [])
            self._data['fetched_past_data'] = fetched_past
        
        normalized_past = self._data.get('normalized_fetched_past', [])
        encoded_residuals = self._data.get('encoded_residuals_for_plotting', {})
        
        # Get encoding models and normalization params
        # Load from model metadata if not in local cache
        encoding_models = self.model._data.get('encoding_models', {})
        if not encoding_models:
            # Try to load from model metadata
            model_metadata = self.model._data.get('model_metadata', {})
            encoding_models = model_metadata.get('encoding_models', {})
            if encoding_models:
                self.model._data['encoding_models'] = encoding_models
        
        norm_params = self.model._data.get('feature_normalization_params', {})
        if not norm_params:
            # Try to load from model metadata
            model_metadata = self.model._data.get('model_metadata', {})
            norm_params = model_metadata.get('feature_normalization_params', {})
            if norm_params:
                self.model._data['feature_normalization_params'] = norm_params
        
        # Reconstruct ALL future conditioning features at once using the reconstruction endpoint
        if encoded_residuals:
            # Get the selected sample to access group metadata
            reference_sample_id = self._data.get('reference_sample_id')
            if reference_sample_id:
                sample_response = self.http.get(
                    f'/api/v1/models/{self.model_id}/samples',
                    params={'include_data': 'true', 'limit': 1000}
                )
                samples = sample_response.get('samples', [])
                sample = next((s for s in samples if s['id'] == reference_sample_id), None)
                encoded_conditioning_data = sample.get('encoded_conditioning_data', []) if sample else []
            else:
                encoded_conditioning_data = []
            
            # Prepare encoded windows for reconstruction (with group metadata)
            encoded_windows = []
            reference_values = {}
            
            for feat_or_group_name, encoded_vals in encoded_residuals.items():
                # Find the original encoded item to get group metadata
                encoded_item = next((x for x in encoded_conditioning_data 
                                    if x.get('feature') == feat_or_group_name and x.get('temporal_tag') == 'future'), None)
                
                window = {
                    "feature": feat_or_group_name,
                    "temporal_tag": "future",
                    "data_type": "encoded_normalized_residuals",
                    "encoded_values": encoded_vals
                }
                
                # Preserve group metadata if present
                if encoded_item:
                    if encoded_item.get('is_group'):
                        window['is_group'] = True
                        window['is_multivariate'] = encoded_item.get('is_multivariate', False)
                        window['group_features'] = encoded_item.get('group_features', [])
                        window['n_components'] = encoded_item.get('n_components')
                        
                        # Get reference values for this group's features
                        group_features = encoded_item.get('group_features', [])
                        for feat in group_features:
                            last_norm = None
                            if normalized_past:
                                for item in normalized_past:
                                    if item.get('feature_name') == feat or item.get('feature') == feat:
                                        last_norm = item['normalized_series'][-1]
                                        break
                            
                            if last_norm is None:
                                # Calculate from last past value
                                past_pts = [p for p in fetched_past if p['feature'] == feat or p.get('display_name') == feat]
                                if past_pts:
                                    past_pts = sorted(past_pts, key=lambda x: x['date'])
                                    mean = norm_params.get(feat, {}).get('mean', 0)
                                    std = norm_params.get(feat, {}).get('std', 1)
                                    last_norm = (past_pts[-1]['value'] - mean) / std
                            
                            if last_norm is not None:
                                reference_values[feat] = last_norm
                    else:
                        # Individual feature (no group)
                        last_norm = None
                        if normalized_past:
                            for item in normalized_past:
                                if item.get('feature_name') == feat_or_group_name or item.get('feature') == feat_or_group_name:
                                    last_norm = item['normalized_series'][-1]
                                    break
                        
                        if last_norm is None:
                            past_pts = [p for p in fetched_past if p['feature'] == feat_or_group_name or p.get('display_name') == feat_or_group_name]
                            if past_pts:
                                past_pts = sorted(past_pts, key=lambda x: x['date'])
                                mean = norm_params.get(feat_or_group_name, {}).get('mean', 0)
                                std = norm_params.get(feat_or_group_name, {}).get('std', 1)
                                last_norm = (past_pts[-1]['value'] - mean) / std
                        
                        if last_norm is not None:
                            reference_values[feat_or_group_name] = last_norm
                
                encoded_windows.append(window)
            
            # Call reconstruction endpoint
            payload = {
                "user_id": self.model._data.get("user_id"),
                "model_id": self.model_id,
                "encoded_source": "inline",
                "encoded_windows": encoded_windows,
                "reference_source": "inline",
                "reference_values": reference_values,
                "output_destination": "return"
            }
            
            try:
                response = self.http.post('/api/v1/ml/reconstruct', payload)
                reconstructions = response.get('reconstructions', [])
                
                # Index reconstructions by feature
                reconstructed_by_feature = {}
                for recon in reconstructions:
                    feat = recon['feature']
                    reconstructed_by_feature[feat] = recon['reconstructed_values']
            except Exception as e:
                logger.warning(f"Future conditioning reconstruction failed: {e}")
                reconstructed_by_feature = {}
        else:
            reconstructed_by_feature = {}
        
        # Determine features to plot after reconstruction
        if features is None:
            if reconstructed_by_feature:
                # Use individual feature names from reconstructed data (post-group expansion)
                features = list(reconstructed_by_feature.keys())[:6]  # Limit to 6 for readability
            else:
                features = list(encoded_residuals.keys())[:6]
        
        if not features:
            raise ValueError("No features to plot")
        
        print(f"[Scenario] Plotting conditioning scenario for {len(features)} features...")
        
        # Create subplots
        n_features = len(features)
        n_cols = min(2, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_features > 1 else [axes]
        
        # Now plot with the reconstructed future values
        for idx, feature_name in enumerate(features):
            ax = axes[idx]
            
            # Get past data again for this feature
            past_points = [p for p in fetched_past if p['feature'] == feature_name or p.get('display_name') == feature_name]
            past_points = sorted(past_points, key=lambda x: x['date'])
            
            if not past_points:
                continue  # Already handled above
            
            past_dates_str = [p['date'] for p in past_points]
            past_values = [p['value'] for p in past_points]
            from datetime import datetime
            past_dates = [datetime.strptime(d, '%Y-%m-%d') for d in past_dates_str]
            
            # Check if we have reconstructed future values
            if feature_name in reconstructed_by_feature:
                future_values = reconstructed_by_feature[feature_name]
                
                # Generate future dates
                last_date = past_dates[-1]
                future_dates = [
                    last_date + timedelta(days=i+1)
                    for i in range(len(future_values))
                ]
                
                # Plot
                ax.plot(past_dates, past_values, '-', color='steelblue', 
                       linewidth=2, label='Past (Fetched)', alpha=0.8)
                ax.plot(future_dates, future_values, '-', color='coral', 
                       linewidth=2, label='Future (Conditioning)', alpha=0.8)
                
                # Mark boundary
                ax.axvline(past_dates[-1], color='gray', linestyle='--', 
                          linewidth=1, alpha=0.5, label='Boundary')
                
                # Formatting
                ax.set_title(feature_name, fontsize=11, fontweight='bold')
                ax.set_xlabel('Date', fontsize=9)
                ax.set_ylabel('Value', fontsize=9)
                ax.legend(loc='best', fontsize=8, framealpha=0.9)
                ax.grid(True, alpha=0.3)
                
                # Format x-axis
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
            else:
                # Only past data available
                ax.plot(past_dates, past_values, '-', color='steelblue', 
                       linewidth=2, label='Past (Fetched)', alpha=0.8)
                ax.set_title(feature_name, fontsize=11, fontweight='bold')
                ax.set_xlabel('Date', fontsize=9)
                ax.set_ylabel('Value', fontsize=9)
                ax.legend(loc='best', fontsize=8, framealpha=0.9)
                ax.grid(True, alpha=0.3)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
        
        # Hide unused subplots
        for idx in range(len(features), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Conditioning Scenario: Past + Future', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Conditioning scenario plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def _decode_with_pca_ica(self, encoding_model: Dict, encoded_values: List[float]) -> List[float]:
        """
        Decode PCA-ICA encoded values back to original dimension.
        
        This performs the inverse transformation:
        1. ICA inverse (unmixing) 
        2. PCA inverse (to original space)
        
        Args:
            encoding_model: Dict with PCA/ICA parameters
            encoded_values: Encoded component values
        
        Returns:
            Decoded values in original dimension
        """
        import numpy as np
        
        # Extract PCA and ICA components
        pca_components = np.array(encoding_model.get('pca_components', []))
        pca_mean = np.array(encoding_model.get('pca_mean', []))
        ica_mixing = np.array(encoding_model.get('ica_mixing_matrix', []))
        
        # Handle empty model
        if pca_components.size == 0 or ica_mixing.size == 0:
            logger.warning("Empty encoding model, returning zeros")
            return [0.0] * len(encoded_values)
        
        # Decode: ICA inverse → PCA inverse
        encoded_array = np.array(encoded_values).reshape(1, -1)
        
        # ICA inverse (unmixing)
        pca_space = encoded_array @ ica_mixing.T
        
        # PCA inverse
        original_space = pca_space @ pca_components + pca_mean
        
        return original_space.flatten().tolist()
    
    def _update_step(self, step: str):
        """Update scenario step both locally and in database"""
        try:
            self.http.patch(f'/api/v1/scenarios/{self.id}', {'current_step': step})
            self._data['current_step'] = step
            self.current_step = step
        except Exception as e:
            logger.warning(f"Could not update scenario step: {e}")
    
    def forecast(self, 
                 simulation_date: str,
                 n_samples: int = 100):
        """
        Generate forecasts for this scenario (ONE-CALL SIMPLIFIED API)
        
        This method automatically:
        1. Fetches recent market data using model's configured data collectors
        2. Selects historical sample matching simulation_date
        3. Generates conditional forecasts
        4. Reconstructs everything to denormalized scale
        5. Returns ready-to-use SyntheticData
        
        Args:
            simulation_date: Historical date to simulate (e.g., '2020-03-15' for COVID)
            n_samples: Number of forecast paths to generate (default: 100)
        
        Returns:
            SyntheticData instance with conditioning, forecasts, and ground truth
        
        Example:
            >>> # Data collectors are already configured in the model
            >>> scenario = model.create_scenario("COVID Crisis")
            >>> forecasts = scenario.forecast(simulation_date='2020-03-15', n_samples=100)
            >>> forecasts.plot_forecasts()
        
        Note:
            API keys are stored with the model's data collectors.
            No need to pass them here!
        """
        from ..synthetic_data import SyntheticData
        
        print(f"\n🎯 Generating scenario forecast...")
        print(f"   Simulation: {simulation_date}")
        print(f"   Paths: {n_samples}")
        print(f"   (Using model's configured data collectors)")
        
        print(f"\n🎲 Generating forecasts...")
        print(f"   (Backend will auto-fetch recent data using model's collectors)")
        
        payload = {
            'user_id': self.model._data.get('user_id'),
            'model_id': self.model.id,
            'scenario_id': self.id,
            'conditioning_source': 'scenario',
            'simulation_date': simulation_date,
            'n_samples': n_samples
            # Note: API keys are stored with model's data_collectors, not passed in request
        }
        
        result = self.http.post('/api/v1/ml/forecast', payload)
        
        print(f"✅ Forecasting completed!")
        print(f"   Generated {result['n_samples']} samples")
        
        # Step 3: Return SyntheticData with all reconstructed windows
        reconstructed_windows = result.get('conditioning_info', {}).get('reconstructed', [])
        
        if not reconstructed_windows:
            logger.warning("No auto-reconstructed data found")
            raise ValueError("Backend did not return reconstructed data")
        
        return SyntheticData(
            reconstructed_windows=reconstructed_windows,
            forecast_metadata={
                'n_samples': result['n_samples'],
                'n_observed': result['n_observed'],
                'n_predicted': result['n_predicted'],
                'model_id': result['model_id'],
                'scenario_id': self.id,
                'simulation_date': simulation_date,
                'conditioning_info': result.get('conditioning_info', {}),
                # Date information for plotting
                'past_dates': result.get('past_dates', []),
                'future_dates': result.get('future_dates', []),
                'reference_date': result.get('reference_date')
            },
            scenario=self,
            model=self.model
        )
    
    def refresh(self):
        """Refresh scenario data from database"""
        response = self.http.get(f'/api/v1/scenarios/{self.id}')
        self._data = response
        self.current_step = response.get('current_step', 'model-selection')
        # n_scenarios removed - number of paths is specified per forecast request
    
    def delete(self, confirm: bool = False) -> Dict[str, Any]:
        """
        Delete this scenario
        
        Args:
            confirm: Skip confirmation prompt if True
        
        Returns:
            Deletion result
        
        Example:
            >>> scenario.delete(confirm=True)
        """
        if self.interactive and not confirm:
            response = input(f"Delete scenario '{self.name}'? [y/N]: ")
            if response.lower() != 'y':
                print("❌ Deletion cancelled")
                return {"status": "cancelled"}
        
        print(f"🗑️  Deleting scenario: {self.name}...")
        
        try:
            # Delete via API (backend handles cascade)
            self.http.delete(f'/api/v1/scenarios/{self.id}')
            print("✅ Scenario deleted")
            
            return {"status": "success"}
        except Exception as e:
            print(f"❌ Deletion failed: {e}")
            raise
