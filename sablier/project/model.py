"""Model class for the new modular architecture"""

import logging
from typing import List, Dict, Any, Optional
from ..http_client import HTTPClient

logger = logging.getLogger(__name__)


class Model:
    """
    Model class for the new modular architecture
    
    A Model is created from a conditioning set and target set within a project.
    It handles:
    - Sample generation
    - Encoding model fitting
    - Vine copula training
    - Forecasting
    """
    
    def __init__(self, 
                 http_client: HTTPClient, 
                 model_data: dict, 
                 interactive: bool = True):
        """
        Initialize Model instance
        
        Args:
            http_client: HTTP client for API requests
            model_data: Model data from API
            interactive: Whether to prompt for confirmations
        """
        self.http = http_client
        self._data = model_data
        self.interactive = interactive
        
        # Core attributes
        self.id = model_data.get('id')
        self.name = model_data.get('name')
        self.description = model_data.get('description', '')
        self.project_id = model_data.get('project_id')
        self.conditioning_set_id = model_data.get('conditioning_set_id')
        self.target_set_id = model_data.get('target_set_id')
    
    def __repr__(self) -> str:
        return f"Model(id='{self.id}', name='{self.name}', status='{self.status}')"
    
    # ============================================
    # PROPERTIES
    # ============================================
    
    @property
    def status(self) -> str:
        """Get current model status"""
        return self._data.get('status', 'created')
    
    @property
    def current_stage(self) -> str:
        """Get current model stage"""
        return self._data.get('current_stage', 'created')
    
    @property
    def sample_config(self) -> Optional[Dict[str, Any]]:
        """Get sample generation configuration"""
        return self._data.get('sample_config')
    
    @property
    def vine_copula_path(self) -> Optional[str]:
        """Get path to trained vine copula model"""
        return self._data.get('vine_copula_path')
    
    # ============================================
    # SAMPLE GENERATION
    # ============================================
    
    def generate_samples(self,
                        past_window: int = 100,
                        future_window: int = 80,
                        stride: int = 5,
                        splits: Optional[Dict[str, int]] = None,
                        confirm: Optional[bool] = None) -> Dict[str, Any]:
        """
        Generate samples for this model
        
        Args:
            past_window: Number of days in past window
            future_window: Number of days in future window
            stride: Stride between samples
            splits: Training/validation splits (e.g., {"training": 80, "validation": 20})
            confirm: Skip confirmation prompt if True
            
        Returns:
            Dict with sample generation results
            
        Example:
            >>> result = model.generate_samples(
            ...     past_window=100,
            ...     future_window=80,
            ...     stride=5,
            ...     splits={"training": 80, "validation": 20}
            ... )
        """
        if self.status not in ['created', 'data_collected']:
            raise ValueError(f"Model must be in 'created' or 'data_collected' status, current: {self.status}")
        
        # Default splits
        if splits is None:
            splits = {"training": 80, "validation": 20}
        
        # Confirmation
        if confirm is None and self.interactive:
            print(f"📊 Generating samples for model: {self.name}")
            print(f"   Past window: {past_window} days")
            print(f"   Future window: {future_window} days")
            print(f"   Stride: {stride} days")
            print(f"   Splits: {splits}")
            print()
            
            response = input("Continue? [y/N]: ")
            if response.lower() != 'y':
                return {"status": "cancelled"}
        
        # Generate samples via API
        response = self.http.post(f'/api/v1/models/{self.id}/generate-samples', {
            "past_window": past_window,
            "future_window": future_window,
            "stride": stride,
            "splits": splits
        })
        
        # Update local data
        self._data = response.get('model', self._data)
        
        split_counts = response.get('split_counts', {})
        print(f"✅ Samples generated for {self.name}")
        print(f"   Total samples: {response.get('samples_generated', 0)}")
        for split, count in split_counts.items():
            print(f"   {split}: {count} samples")
        
        return response
    
    # ============================================
    # ENCODING
    # ============================================
    
    def encode_samples(self,
                      encoding_type: str = "pca-ica",
                      split: str = "training",
                      pca_variance_threshold_series: float = 0.95,
                      pca_variance_threshold_residuals: float = 0.99,
                      confirm: Optional[bool] = None) -> Dict[str, Any]:
        """
        Fit encoding models and encode all samples
        
        Args:
            encoding_type: Type of encoding ("pca-ica" or "umap")
            split: Which split to use for fitting
            pca_variance_threshold_series: Variance threshold for series
            pca_variance_threshold_residuals: Variance threshold for residuals
            confirm: Skip confirmation prompt if True
            
        Returns:
            Dict with encoding results
            
        Example:
            >>> result = model.encode_samples(
            ...     encoding_type="pca-ica",
            ...     split="training"
            ... )
        """
        if self.status != 'samples_created':
            raise ValueError(f"Model must be in 'samples_created' status, current: {self.status}")
        
        # Confirmation
        if confirm is None and self.interactive:
            print(f"🔍 Encoding samples for model: {self.name}")
            print(f"   Encoding type: {encoding_type}")
            print(f"   Fitting split: {split}")
            print(f"   PCA thresholds: series={pca_variance_threshold_series}, residuals={pca_variance_threshold_residuals}")
            print()
            
            response = input("Continue? [y/N]: ")
            if response.lower() != 'y':
                return {"status": "cancelled"}
        
        # Encode samples via API
        response = self.http.post(f'/api/v1/models/{self.id}/encode-samples', {
            "encoding_type": encoding_type,
            "split": split,
            "pca_variance_threshold_series": pca_variance_threshold_series,
            "pca_variance_threshold_residuals": pca_variance_threshold_residuals
        })
        
        # Update local data
        self._data = response.get('model', self._data)
        
        print(f"✅ Samples encoded for {self.name}")
        print(f"   Models fitted: {response.get('models_fitted', 0)}")
        print(f"   Samples encoded: {response.get('samples_encoded', 0)}")
        print(f"   Features processed: {response.get('features_processed', 0)}")
        
        return response
    
    # ============================================
    # TRAINING
    # ============================================
    
    def train(self,
              model_type: str = 'vine_copula',
              n_components: int = 3,
              copula_family: str = 'mixed',
              trunc_lvl: int = 3,
              num_threads: int = 4,
              split: str = 'training',
              confirm: Optional[bool] = None) -> Dict[str, Any]:
        """
        Train the vine copula model
        
        Args:
            model_type: Model type (default: 'vine_copula')
            n_components: Number of mixture components/regimes
            copula_family: Copula family ('gaussian', 'mixed', etc.)
            trunc_lvl: Truncation level for vine copula
            num_threads: Number of threads for parallel fitting
            split: Data split to use for training
            confirm: Skip confirmation prompt if True
            
        Returns:
            Dict with training results
            
        Example:
            >>> result = model.train(
            ...     n_components=3,
            ...     copula_family='mixed',
            ...     trunc_lvl=3
            ... )
        """
        if self.status != 'encoded':
            raise ValueError(f"Model must be in 'encoded' status, current: {self.status}")
        
        # Confirmation
        if confirm is None and self.interactive:
            print(f"🤖 Training vine copula model: {self.name}")
            print(f"   Model type: {model_type}")
            print(f"   Components: {n_components}")
            print(f"   Copula family: {copula_family}")
            print(f"   Truncation level: {trunc_lvl}")
            print(f"   Threads: {num_threads}")
            print(f"   Training split: {split}")
            print()
            
            response = input("Continue? [y/N]: ")
            if response.lower() != 'y':
                return {"status": "cancelled"}
        
        # Train model via API
        response = self.http.post(f'/api/v1/models/{self.id}/train', {
            "model_type": model_type,
            "n_components": n_components,
            "copula_family": copula_family,
            "trunc_lvl": trunc_lvl,
            "num_threads": num_threads,
            "split": split
        })
        
        # Update local data
        self._data = response.get('model', self._data)
        
        print(f"✅ Model trained: {self.name}")
        print(f"   Status: {self.status}")
        print(f"   Model path: {self.vine_copula_path}")
        
        return response
    
    # ============================================
    # FORECASTING
    # ============================================
    
    def forecast(self,
                 conditioning_source: str = "sample",
                 sample_id: Optional[str] = None,
                 n_samples: int = 100,
                 confirm: Optional[bool] = None) -> 'SyntheticData':
        """
        Generate forecasts using the trained model
        
        Args:
            conditioning_source: Source of conditioning data ("sample" or "scenario")
            sample_id: Sample ID to use for conditioning (if conditioning_source="sample")
            n_samples: Number of forecast samples to generate
            confirm: Skip confirmation prompt if True
            
        Returns:
            SyntheticData: Forecast results
            
        Example:
            >>> forecasts = model.forecast(
            ...     conditioning_source="sample",
            ...     n_samples=100
            ... )
        """
        if self.status != 'model_trained':
            raise ValueError(f"Model must be in 'model_trained' status, current: {self.status}")
        
        # Confirmation
        if confirm is None and self.interactive:
            print(f"🔮 Generating forecasts for model: {self.name}")
            print(f"   Conditioning source: {conditioning_source}")
            print(f"   Number of samples: {n_samples}")
            if sample_id:
                print(f"   Sample ID: {sample_id}")
            print()
            
            response = input("Continue? [y/N]: ")
            if response.lower() != 'y':
                return None
        
        # Generate forecasts via API
        response = self.http.post(f'/api/v1/models/{self.id}/forecast', {
            "conditioning_source": conditioning_source,
            "sample_id": sample_id,
            "n_samples": n_samples
        })
        
        # Create SyntheticData instance
        from ..synthetic_data import SyntheticData
        return SyntheticData(
            reconstructed_windows=response.get('reconstructed_windows', []),
            forecast_metadata=response.get('forecast_metadata', {}),
            model=self
        )
    
    # ============================================
    # SCENARIO CREATION
    # ============================================
    
    def create_scenario(self,
                       name: str,
                       description: str = "",
                       conditioning_config: Optional[Dict[str, Any]] = None) -> 'Scenario':
        """
        Create a scenario for this model
        
        Args:
            name: Scenario name
            description: Scenario description
            conditioning_config: Per-feature simulation dates for cross-sample conditioning
            
        Returns:
            Scenario: New scenario instance
            
        Example:
            >>> scenario = model.create_scenario(
            ...     name="COVID Market Crash",
            ...     conditioning_config={
            ...         "conditioning_group_1": {"simulation_date": "2020-03-15"},
            ...         "conditioning_group_2": {"simulation_date": "2020-03-15"}
            ...     }
            ... )
        """
        from ..scenario.builder import Scenario
        
        response = self.http.post('/api/v1/scenarios', {
            "model_id": self.id,
            "name": name,
            "description": description,
            "conditioning_config": conditioning_config or {}
        })
        
        print(f"✅ Created scenario: {name}")
        return Scenario(self.http, response, self, self.interactive)
    
    # ============================================
    # UTILITY METHODS
    # ============================================
    
    def refresh(self) -> 'Model':
        """Refresh model data from API"""
        response = self.http.get(f'/api/v1/models/{self.id}')
        self._data = response
        return self
    
    def delete(self, confirm: Optional[bool] = None) -> Dict[str, Any]:
        """
        Delete this model and ALL associated data
        
        Args:
            confirm: Skip confirmation prompt if True
            
        Returns:
            Dict with deletion status
        """
        # Always warn for deletion
        print("⚠️  WARNING: You are about to PERMANENTLY DELETE this model.")
        print(f"   Model: {self.name} ({self.id})")
        print(f"   Status: {self.status}")
        print()
        print("This will delete ALL associated data:")
        print("  - Generated samples")
        print("  - Encoding models")
        print("  - Trained vine copula model")
        print("  - All scenarios using this model")
        print()
        print("This action CANNOT be undone.")
        print()
        
        # Get confirmation
        if confirm is None and self.interactive:
            response = input("Type the model name to confirm deletion: ")
            if response != self.name:
                print("❌ Model name doesn't match. Deletion cancelled.")
                return {"status": "cancelled"}
        elif confirm is None:
            print("❌ Deletion cancelled (interactive=False, no confirmation)")
            return {"status": "cancelled"}
        elif not confirm:
            print("❌ Deletion cancelled")
            return {"status": "cancelled"}
        
        # Delete via API
        print("🗑️  Deleting model...")
        response = self.http.delete(f'/api/v1/models/{self.id}')
        
        print(f"✅ Model '{self.name}' deleted")
        return response
    
    def summary(self) -> None:
        """Print a summary of this model"""
        print(f"🤖 Model: {self.name}")
        print(f"   ID: {self.id}")
        print(f"   Status: {self.status}")
        print(f"   Stage: {self.current_stage}")
        print(f"   Description: {self.description}")
        print()
        
        if self.sample_config:
            print(f"📊 Sample Configuration:")
            print(f"   Past window: {self.sample_config.get('pastWindow', 'N/A')} days")
            print(f"   Future window: {self.sample_config.get('futureWindow', 'N/A')} days")
            print(f"   Stride: {self.sample_config.get('stride', 'N/A')} days")
            print()
        
        if self.vine_copula_path:
            print(f"🎯 Trained Model:")
            print(f"   Path: {self.vine_copula_path}")
            print()
