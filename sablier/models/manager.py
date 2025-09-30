"""Model manager for creating and retrieving models"""

from typing import Optional, Any
from ..http_client import HTTPClient
from .builder import Model


class ModelManager:
    """Manages model creation and retrieval"""
    
    def __init__(self, http_client: HTTPClient, interactive: bool = True):
        self.http = http_client
        self.interactive = interactive
    
    def create(
        self,
        name: str,
        description: str = "",
        **kwargs
    ) -> 'Model':
        """
        Create a new model
        
        Args:
            name: Model name
            description: Model description (optional)
            **kwargs: Additional model parameters
            
        Returns:
            Model: Model instance
            
        Example:
            >>> model = client.models.create(
            ...     name="Market Model",
            ...     description="My trading model"
            ... )
        
        Note:
            Features and training period are set later via:
            - model.add_features([...])
            - model.set_training_period(start="...", end="...")
        """
        # Call API to create model
        response = self.http.post('/api/v1/models', {
            "name": name,
            "description": description
        })
        
        print(f"✅ Model created: {response['name']} (ID: {response['id']})")
        
        return Model(self.http, response, interactive=self.interactive)
    
    def get(self, model_id: str) -> 'Model':
        """
        Retrieve an existing model
        
        Args:
            model_id: Model ID
            
        Returns:
            Model: Model instance
        """
        # Call API to get model
        response = self.http.get(f'/api/v1/models/{model_id}')
        
        return Model(self.http, response, interactive=self.interactive)
    
    def list(
        self,
        limit: int = 100,
        offset: int = 0,
        status: Optional[str] = None
    ) -> list['Model']:
        """
        List all models
        
        Args:
            limit: Maximum number of models to return (default: 100)
            offset: Number of models to skip (for pagination)
            status: Optional filter by status (e.g., "trained", "created")
            
        Returns:
            list[Model]: List of Model instances
            
        Example:
            >>> # List all models
            >>> models = client.models.list()
            >>> for model in models:
            ...     print(f"{model.name}: {model.id}")
            
            >>> # List only trained models
            >>> trained = client.models.list(status="trained")
        """
        # Build query params
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        
        # Call API
        response = self.http.get('/api/v1/models', params)
        
        # Convert to Model instances
        models = [Model(self.http, model_data, interactive=self.interactive) for model_data in response.get('models', [])]
        
        print(f"✅ Found {len(models)} models (total: {response.get('total', 0)})")
        
        return models
    
    def get_by_name(self, name: str) -> Optional['Model']:
        """
        Get a model by name (returns first match)
        
        Args:
            name: Model name
            
        Returns:
            Model: Model instance, or None if not found
            
        Example:
            >>> model = client.models.get_by_name("My Market Model")
            >>> if model:
            ...     print(f"Found: {model.id}")
        """
        # List all models and filter by name
        all_models = self.list(limit=1000)  # Reasonable limit
        
        for model in all_models:
            if model.name == name:
                return model
        
        return None
