"""Scenario Manager for handling scenario operations"""

from typing import List, Optional, Dict
from ..http_client import HTTPClient
from .builder import Scenario


class ScenarioManager:
    """
    Manages scenario creation, retrieval, and listing
    
    Scenarios are market conditions used to generate conditional synthetic data.
    Each scenario is linked to a trained model.
    """
    
    def __init__(self, http_client: HTTPClient, model_manager):
        """
        Initialize ScenarioManager
        
        Args:
            http_client: HTTP client for API requests
            model_manager: ModelManager instance for model references
        """
        self.http = http_client
        self.model_manager = model_manager
    
    def create(
        self,
        name: str,
        model = None,
        model_id: str = None,
        simulation_date: Optional[str] = None,
        description: str = "",
        feature_simulation_dates: Optional[Dict[str, str]] = None
    ) -> Scenario:
        """
        Create a new scenario
        
        Args:
            name: Scenario name
            model: Model instance to use (either this or model_id required)
            model_id: Model ID to use (either this or model required)
            simulation_date: Optional simulation date for all features (YYYY-MM-DD).
                           Defaults to today's date if not specified.
            description: Optional scenario description
            feature_simulation_dates: Optional dict mapping feature names to specific simulation dates
        Returns:
            Scenario instance
        
        Example:
            >>> model = client.models.get("model-id")
            >>> scenario = client.scenarios.create(
            ...     name="Bull Market 2025",
            ...     model=model
            ... )
            
            # With specific simulation date
            >>> scenario = client.scenarios.create(
            ...     name="COVID Crash",
            ...     model=model,
            ...     simulation_date="2020-03-15"
            ... )
        """
        # Determine model_id and model instance
        if model is not None:
            model_id = model.id
        elif model_id is None:
            raise ValueError("Either 'model' or 'model_id' must be provided")
        else:
            # Fetch the model if only model_id was provided
            model = self.model_manager.get(model_id)
        
        # Set default simulation_date from model if not provided
        if simulation_date is None:
            simulation_date = model._get_default_simulation_date()
        
        print(f"[Scenario] Creating scenario: {name}")
        print(f"  Model ID: {model_id}")
        print(f"  Simulation date: {simulation_date}")
        
        # Create via API
        response = self.http.post('/api/v1/scenarios', {
            'model_id': model_id,
            'name': name,
            'description': description,
            'simulation_date': simulation_date,
            'feature_simulation_dates': feature_simulation_dates or {}
        })
        
        print(f"✅ Scenario created: {response.get('id')}")
        
        return Scenario(self.http, response, model)
    
    def get(self, scenario_id: str) -> Scenario:
        """
        Get a scenario by ID
        
        Args:
            scenario_id: Scenario ID
        
        Returns:
            Scenario instance
        
        Example:
            >>> scenario = client.scenarios.get("scenario-id")
        """
        response = self.http.get(f'/api/v1/scenarios/{scenario_id}')
        
        # Also fetch the model
        model_id = response.get('model_id')
        model = self.model_manager.get(model_id) if model_id else None
        
        return Scenario(self.http, response, model)
    
    def list(self, model_id: str = None, limit: int = 100, offset: int = 0) -> List[Scenario]:
        """
        List scenarios
        
        Args:
            model_id: Optional filter by model ID
            limit: Maximum number of scenarios to return
            offset: Pagination offset
        
        Returns:
            List of Scenario instances
        
        Example:
            >>> scenarios = client.scenarios.list(model_id="model-id")
            >>> for scenario in scenarios:
            ...     print(scenario.name, scenario.current_step)
        """
        params = {'limit': limit, 'offset': offset}
        if model_id:
            params['model_id'] = model_id
        
        response = self.http.get('/api/v1/scenarios', params=params)
        scenarios_data = response.get('scenarios', [])
        
        # Create Scenario instances
        scenarios = []
        for scenario_data in scenarios_data:
            model_id = scenario_data.get('model_id')
            try:
                model = self.model_manager.get(model_id) if model_id else None
            except:
                model = None
            
            scenarios.append(Scenario(self.http, scenario_data, model))
        
        return scenarios
