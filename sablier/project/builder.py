"""Project class - Master class for modular architecture"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, date
from ..http_client import HTTPClient

logger = logging.getLogger(__name__)


class Project:
    """
    Master class that defines training period and contains feature sets and models
    
    A Project encapsulates:
    - Training period (start_date, end_date) - shared across all feature sets
    - Conditioning sets (input features for prediction)
    - Target sets (output features to predict)
    - Models (created from conditioning_set + target_set combinations)
    """
    
    def __init__(self, 
                 http_client: HTTPClient, 
                 project_data: dict, 
                 interactive: bool = True):
        """
        Initialize Project instance
        
        Args:
            http_client: HTTP client for API requests
            project_data: Project data from API
            interactive: Whether to prompt for confirmations
        """
        self.http = http_client
        self._data = project_data
        self.interactive = interactive
        
        # Core attributes
        self.id = project_data.get('id')
        self.name = project_data.get('name')
        self.description = project_data.get('description', '')
        self.training_start_date = project_data.get('training_start_date')
        self.training_end_date = project_data.get('training_end_date')
    
    def __repr__(self) -> str:
        return f"Project(id='{self.id}', name='{self.name}', period='{self.training_start_date} to {self.training_end_date}')"
    
    # ============================================
    # PROPERTIES
    # ============================================
    
    @property
    def conditioning_sets(self) -> List[Dict[str, Any]]:
        """Get conditioning feature sets"""
        return self._data.get('conditioning_sets', [])
    
    @property
    def target_sets(self) -> List[Dict[str, Any]]:
        """Get target feature sets"""
        return self._data.get('target_sets', [])
    
    @property
    def models(self) -> List[Dict[str, Any]]:
        """Get models created in this project"""
        return self._data.get('models', [])
    
    # ============================================
    # FEATURE SET MANAGEMENT
    # ============================================
    
    def create_conditioning_set(self, 
                               name: str, 
                               description: str = "") -> 'FeatureSet':
        """
        Create a new conditioning feature set
        
        Args:
            name: Feature set name
            description: Feature set description
            
        Returns:
            FeatureSet: New conditioning feature set
            
        Example:
            >>> conditioning_set = project.create_conditioning_set(
            ...     name="Economic Indicators",
            ...     description="Key economic indicators for conditioning"
            ... )
        """
        from ..feature_set import FeatureSet
        
        response = self.http.post('/api/v1/feature-sets', {
            "project_id": self.id,
            "name": name,
            "description": description,
            "set_type": "conditioning"
        })
        
        print(f"✅ Created conditioning set: {name}")
        return FeatureSet(self.http, response, self.id, self.interactive)
    
    def create_target_set(self, 
                         name: str, 
                         description: str = "") -> 'FeatureSet':
        """
        Create a new target feature set
        
        Args:
            name: Feature set name
            description: Feature set description
            
        Returns:
            FeatureSet: New target feature set
            
        Example:
            >>> target_set = project.create_target_set(
            ...     name="Treasury Yields",
            ...     description="Treasury yield curve for prediction"
            ... )
        """
        from ..feature_set import FeatureSet
        
        response = self.http.post('/api/v1/feature-sets', {
            "project_id": self.id,
            "name": name,
            "description": description,
            "set_type": "target"
        })
        
        print(f"✅ Created target set: {name}")
        return FeatureSet(self.http, response, self.id, self.interactive)
    
    def get_conditioning_set(self, set_id: str) -> 'FeatureSet':
        """
        Get a conditioning feature set by ID
        
        Args:
            set_id: Feature set ID
            
        Returns:
            FeatureSet: Conditioning feature set
        """
        from ..feature_set import FeatureSet
        
        response = self.http.get(f'/api/v1/feature-sets/{set_id}')
        return FeatureSet(self.http, response, self.id, self.interactive)
    
    def get_target_set(self, set_id: str) -> 'FeatureSet':
        """
        Get a target feature set by ID
        
        Args:
            set_id: Feature set ID
            
        Returns:
            FeatureSet: Target feature set
        """
        from ..feature_set import FeatureSet
        
        response = self.http.get(f'/api/v1/feature-sets/{set_id}')
        return FeatureSet(self.http, response, self.id, self.interactive)
    
    def list_conditioning_sets(self) -> List['FeatureSet']:
        """
        List all conditioning feature sets in this project
        
        Returns:
            List[FeatureSet]: All conditioning feature sets
        """
        from ..feature_set import FeatureSet
        
        response = self.http.get(f'/api/v1/projects/{self.id}/conditioning-sets')
        return [FeatureSet(self.http, data, self.id, self.interactive) for data in response]
    
    def list_target_sets(self) -> List['FeatureSet']:
        """
        List all target feature sets in this project
        
        Returns:
            List[FeatureSet]: All target feature sets
        """
        from ..feature_set import FeatureSet
        
        response = self.http.get(f'/api/v1/projects/{self.id}/target-sets')
        return [FeatureSet(self.http, data, self.id, self.interactive) for data in response]
    
    # ============================================
    # MODEL MANAGEMENT
    # ============================================
    
    def create_model(self, 
                    conditioning_set: 'FeatureSet',
                    target_set: 'FeatureSet',
                    name: str,
                    description: str = "") -> 'Model':
        """
        Create a new model from a conditioning set and target set
        
        Args:
            conditioning_set: Conditioning feature set
            target_set: Target feature set
            name: Model name
            description: Model description
            
        Returns:
            Model: New model instance
            
        Example:
            >>> model = project.create_model(
            ...     conditioning_set=economic_indicators,
            ...     target_set=treasury_yields,
            ...     name="Treasury Yield Predictor",
            ...     description="Predicts treasury yields from economic indicators"
            ... )
        """
        from ..model.builder import Model
        
        # Validate feature sets
        if conditioning_set.set_type != 'conditioning':
            raise ValueError(f"First argument must be a conditioning set, got {conditioning_set.set_type}")
        
        if target_set.set_type != 'target':
            raise ValueError(f"Second argument must be a target set, got {target_set.set_type}")
        
        # Create model via API (no data validation required - data fetching happens later)
        response = self.http.post('/api/v1/models', {
            "project_id": self.id,
            "conditioning_set_id": conditioning_set.id,
            "target_set_id": target_set.id,
            "name": name,
            "description": description
        })
        
        print(f"✅ Created model: {name}")
        print(f"   Conditioning: {conditioning_set.name} ({len(conditioning_set.features)} features)")
        print(f"   Target: {target_set.name} ({len(target_set.features)} features)")
        
        return Model(self.http, response, self.interactive)
    
    def get_model(self, model_id: str) -> 'Model':
        """
        Get a model by ID
        
        Args:
            model_id: Model ID
            
        Returns:
            Model: Model instance
        """
        from ..model.builder import Model
        
        response = self.http.get(f'/api/v1/models/{model_id}')
        return Model(self.http, response, self.interactive)
    
    def list_models(self) -> List['Model']:
        """
        List all models in this project
        
        Returns:
            List[Model]: All models
        """
        from ..model.builder import Model
        
        response = self.http.get(f'/api/v1/projects/{self.id}/models')
        # Handle response format - could be list directly or wrapped in response object
        models_data = response.get('models', []) if isinstance(response, dict) else response
        return [Model(self.http, data, self.interactive) for data in models_data]
    
    # ============================================
    # UTILITY METHODS
    # ============================================
    
    def refresh(self) -> 'Project':
        """Refresh project data from API"""
        response = self.http.get(f'/api/v1/projects/{self.id}')
        self._data = response
        return self
    
    def update_training_period(self, 
                              start_date: str, 
                              end_date: str) -> 'Project':
        """
        Update the training period for this project
        
        Args:
            start_date: New start date (YYYY-MM-DD)
            end_date: New end date (YYYY-MM-DD)
            
        Returns:
            self (for chaining)
            
        Example:
            >>> project.update_training_period("2015-01-01", "2023-12-31")
        """
        # Validate dates
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        if start_dt >= end_dt:
            raise ValueError("Start date must be before end date")
        
        # Update via API
        response = self.http.patch(f'/api/v1/projects/{self.id}', {
            "training_start_date": start_date,
            "training_end_date": end_date
        })
        
        self._data = response
        self.training_start_date = start_date
        self.training_end_date = end_date
        
        print(f"✅ Updated training period: {start_date} to {end_date}")
        return self
    
    def delete(self, confirm: Optional[bool] = None) -> Dict[str, Any]:
        """
        Delete this project and ALL associated data
        
        Args:
            confirm: Skip confirmation prompt if True
            
        Returns:
            Dict with deletion status
        """
        # Always warn for deletion
        print("⚠️  WARNING: You are about to PERMANENTLY DELETE this project.")
        print(f"   Project: {self.name} ({self.id})")
        print(f"   Training period: {self.training_start_date} to {self.training_end_date}")
        print()
        print("This will delete ALL associated data:")
        print("  - All feature sets")
        print("  - All models")
        print("  - All training data")
        print("  - All samples")
        print("  - All scenarios")
        print()
        print("This action CANNOT be undone.")
        print()
        
        # Get confirmation
        if confirm is None and self.interactive:
            response = input("Type the project name to confirm deletion: ")
            if response != self.name:
                print("❌ Project name doesn't match. Deletion cancelled.")
                return {"status": "cancelled"}
        elif confirm is None:
            print("❌ Deletion cancelled (interactive=False, no confirmation)")
            return {"status": "cancelled"}
        elif not confirm:
            print("❌ Deletion cancelled")
            return {"status": "cancelled"}
        
        # Delete via API
        print("🗑️  Deleting project...")
        response = self.http.delete(f'/api/v1/projects/{self.id}')
        
        print(f"✅ Project '{self.name}' deleted")
        return response
    
    # ============================================
    # SUMMARY METHODS
    # ============================================
    
    def summary(self) -> None:
        """Print a summary of this project"""
        print(f"📊 Project: {self.name}")
        print(f"   ID: {self.id}")
        print(f"   Training period: {self.training_start_date} to {self.training_end_date}")
        print(f"   Description: {self.description}")
        print()
        
        print(f"📈 Feature Sets:")
        print(f"   Conditioning sets: {len(self.conditioning_sets)}")
        for cs in self.conditioning_sets:
            print(f"     - {cs['name']} ({len(cs.get('features', []))} features)")
        
        print(f"   Target sets: {len(self.target_sets)}")
        for ts in self.target_sets:
            print(f"     - {ts['name']} ({len(ts.get('features', []))} features)")
        
        print()
        print(f"🤖 Models: {len(self.models)}")
        for model in self.models:
            print(f"     - {model['name']} (status: {model.get('status', 'unknown')})")
        
        print()
