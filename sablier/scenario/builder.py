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
        self.n_scenarios = scenario_data.get('n_scenarios', 100)
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
        
        # Store fetched data in scenario (not in model's training_data)
        self.http.patch(f'/api/v1/scenarios/{self.id}', {
            'fetched_past_data': processed_data
        })
        print(f"  ✓ Stored in scenario")
        
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
            
            # Normalize
            norm = norm_params.get(feature_name, {'mean': 0, 'std': 1})
            mean = norm.get('mean', 0)
            std = norm.get('std', 1)
            
            normalized_values = [(v - mean) / std for v in values]
            
            # Truncate to past_window length (most recent N days)
            truncated_values = normalized_values[-past_window:] if len(normalized_values) >= past_window else normalized_values
            truncated_dates = dates[-past_window:] if len(dates) >= past_window else dates
            
            conditioning_data.append({
                'feature_name': feature_name,  # Note: backend expects 'feature_name', not 'feature'
                'temporal_tag': 'past',
                'normalized_series': truncated_values
            })
            
            normalized_data[feature_name] = {
                'dates': truncated_dates,
                'values': truncated_values
            }
        
        print(f"  Normalized {len(conditioning_data)} features")
        
        # Note: We don't store normalized_data in the scenario table
        # It's just an intermediate step. We store raw (fetched_past_data) and encoded (encoded_fetched_past)
        
        # Step 3: Encode using model's encoding models
        print("🔐 Step 3/3: Encoding with PCA-ICA...")
        
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
        
        # Update scenario in database with encoded data and final step
        self.http.patch(f'/api/v1/scenarios/{self.id}', {
            'encoded_fetched_past': encoded_fetched_past,
            'current_step': 'past-data-fetched'  # Valid DB constraint value
        })
        print(f"  ✓ Stored encoded data in scenario")
        
        # Update local data
        self._data['encoded_fetched_past'] = encoded_fetched_past
        self._data['normalized_fetched_past'] = conditioning_data
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
        """Configure future conditioning using components from an existing sample"""
        
        # Get sample ID (either provided or find closest)
        if reference_sample_id is None:
            if reference_date is None:
                raise ValueError("Either reference_date or reference_sample_id must be provided")
            
            print(f"  Finding closest sample to {reference_date}...")
            reference_sample_id = self.get_closest_sample(reference_date)
            print(f"  Selected sample: {reference_sample_id}")
        
        # Fetch full sample with encoded data
        print(f"  Fetching sample {reference_sample_id}...")
        sample_response = self.http.get(
            f'/api/v1/models/{self.model_id}/samples',
            params={'include_data': 'true', 'limit': 500}
        )
        
        samples = sample_response.get('samples', [])
        sample = next((s for s in samples if s['id'] == reference_sample_id), None)
        
        if not sample:
            # Try all splits if not found
            sample_response = self.http.get(
                f'/api/v1/models/{self.model_id}/samples',
                params={'include_data': 'true', 'limit': 1000}
            )
            samples = sample_response.get('samples', [])
            sample = next((s for s in samples if s['id'] == reference_sample_id), None)
        
        if not sample:
            raise ValueError(f"Sample {reference_sample_id} not found")
        
        # Extract conditioning features if not specified
        if features is None:
            sample_config = self.model._data.get('sample_config', {})
            features = sample_config.get('conditioningFeatures', [])
        
        # Extract future conditioning from sample
        # IMPORTANT: Only include CONDITIONING features' future windows
        # Target features are never part of the input - they're what we're predicting!
        encoded_conditioning_data = sample.get('encoded_conditioning_data', [])
        
        future_conditioning = []
        
        # Process ALL conditioning features' future windows (some masked, some unmasked)
        for item in encoded_conditioning_data:
            if item.get('temporal_tag') == 'future':
                feature_name = item.get('feature')
                encoded_series = item.get('encoded_normalized_series', [])
                
                if encoded_series:
                    # If this feature is in our scenario's conditioning list, use actual values (unmasked)
                    # Otherwise, mask it with zeros
                    is_selected = not features or feature_name in features
                    
                    future_conditioning.append({
                        'feature': feature_name,
                        'temporal_tag': 'future',
                        'encoded_normalized_series': encoded_series if is_selected else [0.0] * len(encoded_series)
                    })
        
        if not future_conditioning:
            raise ValueError("No future conditioning data found in sample")
        
        print(f"  Extracted conditioning for {len(future_conditioning)} features")
        
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
        
        print(f"✅ Future conditioning configured from sample {reference_sample_id}")
        
        return {
            "status": "success",
            "mode": "from_sample",
            "reference_sample_id": reference_sample_id,
            "features_configured": len(future_conditioning)
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
    
    def generate_paths(self, n_samples: int = None) -> 'SyntheticData':
        """
        Generate synthetic market paths using the configured scenario
        
        Args:
            n_samples: Number of paths to generate (default: scenario's n_scenarios)
        
        Returns:
            SyntheticData instance containing generated paths
        
        Example:
            >>> synthetic_data = scenario.generate_paths(n_samples=1000)
            >>> print(synthetic_data.paths.head())
            >>> synthetic_data.plot_paths("Gold Price")
        """
        from ..synthetic_data import SyntheticData
        
        if n_samples is None:
            n_samples = self.n_scenarios
        
        # Verify scenario is configured
        if self.current_step not in ['future-conditioning-configured', 'paths-generated']:
            raise ValueError(
                f"Scenario must be configured before generating paths. "
                f"Current step: {self.current_step}. "
                f"Call fetch_recent_past() and configure_future_conditioning() first."
            )
        
        print(f"[Scenario] Generating {n_samples} synthetic paths...")
        
        # Call forecast endpoint with scenario mode
        print("📊 Step 1/2: Generating forecast samples...")
        
        # Get user_id from model (always available)
        if not self.model:
            raise ValueError("Model not loaded. Cannot generate paths.")
        
        forecast_response = self.http.post('/api/v1/ml/forecast', {
            'user_id': self.model._data.get('user_id'),
            'model_id': self.model_id,
            'conditioning_source': 'scenario',
            'scenario_id': self.id,
            'n_samples': n_samples
        })
        
        forecast_samples = forecast_response.get('forecast_samples', [])
        print(f"  Generated {len(forecast_samples)} samples")
        
        # Reconstruct forecast samples using scenario's fetched past as reference
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
        Reconstruct forecast samples using the scenario's own fetched past as reference
        
        Args:
            forecast_samples: Forecast outputs from QRF
            
        Returns:
            List of reconstructed paths with dates and values
        """
        # Fetch the scenario's fetched_past_data for reference values
        scenario = self.http.get(f'/api/v1/scenarios/{self.id}')
        fetched_past = scenario.get('fetched_past_data', [])
        
        if not fetched_past:
            raise ValueError("No fetched_past_data in scenario. Call fetch_recent_past() first.")
        
        # Build reference values dict from fetched past (last value for each feature)
        reference_values = {}
        for point in fetched_past:
            feature = point.get('feature')
            if feature:
                # Keep updating - last one wins (most recent value)
                reference_values[feature] = point['value']
        
        # Transform forecast samples to encoded_windows format for reconstruction
        window_to_sample_index = []
        encoded_windows = []
        
        for sample_idx, sample in enumerate(forecast_samples):
            for item in sample.get('encoded_target_data', []):
                window_to_sample_index.append(sample_idx)
                encoded_windows.append({
                    "feature": item['feature'],
                    "temporal_tag": item['temporal_tag'],
                    "data_type": "encoded_normalized_residuals",
                    "encoded_values": item['encoded_normalized_residuals']
                })
        
        # Call reconstruct endpoint with inline reference
        payload = {
            "user_id": self.model._data.get("user_id"),
            "model_id": self.model_id,
            "encoded_source": "inline",
            "encoded_windows": encoded_windows,
            "reference_source": "inline",
            "reference_values": reference_values,
            "output_destination": "return"
        }
        
        response = self.http.post('/api/v1/ml/reconstruct', payload)
        reconstructions = response.get('reconstructions', [])
        
        # Group reconstructions by sample
        windows_by_sample = {}
        for idx, window in enumerate(reconstructions):
            sample_idx = window_to_sample_index[idx]
            if sample_idx not in windows_by_sample:
                windows_by_sample[sample_idx] = []
            windows_by_sample[sample_idx].append(window)
        
        # Generate dates for the forecast window
        # Get future_window from model's sample config
        sample_config = self.model._data.get('sample_config', {})
        future_window = sample_config.get('futureWindow', 80)
        
        # Get last date from fetched past as starting point
        from datetime import datetime, timedelta
        last_date_str = max(point['date'] for point in fetched_past)
        start_date = datetime.strptime(last_date_str, '%Y-%m-%d') + timedelta(days=1)
        dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(future_window)]
        
        # Convert to structured format
        reconstructed_samples = []
        for sample_idx in sorted(windows_by_sample.keys()):
            sample_reconstruction = {}
            
            for window in windows_by_sample[sample_idx]:
                feature = window['feature']
                values = window.get('reconstructed_values', [])
                
                sample_reconstruction[feature] = {
                    'future': {
                        'dates': dates[:len(values)],  # Match dates to values length
                        'values': values
                    }
                }
            
            reconstructed_samples.append(sample_reconstruction)
        
        return reconstructed_samples
    
    def get_closest_sample(self, target_date: str, split: str = None) -> str:
        """
        Find sample with date closest to target_date
        
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
    
    def _update_step(self, step: str):
        """Update scenario step both locally and in database"""
        try:
            self.http.patch(f'/api/v1/scenarios/{self.id}', {'current_step': step})
            self._data['current_step'] = step
            self.current_step = step
        except Exception as e:
            logger.warning(f"Could not update scenario step: {e}")
    
    def refresh(self):
        """Refresh scenario data from database"""
        response = self.http.get(f'/api/v1/scenarios/{self.id}')
        self._data = response
        self.current_step = response.get('current_step', 'model-selection')
        self.n_scenarios = response.get('n_scenarios', 100)
    
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
