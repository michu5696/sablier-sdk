"""
Main Sablier SDK client
"""

from typing import Optional
from .auth import AuthHandler
from .http_client import HTTPClient
from .models.manager import ModelManager
from .scenarios.manager import ScenarioManager
from .exceptions import AuthenticationError


class SablierClient:
    """
    Main client for interacting with the Sablier API
    
    Example:
        >>> client = SablierClient(
        ...     api_url="http://localhost:8000",
        ...     api_key="your-api-key"
        ... )
        >>> model = client.models.create(name="My Model", features=[...])
        >>> model.fetch_data()
        >>> model.train()
    """
    
    def __init__(
        self,
        api_url: str,
        api_key: str,
        supabase_url: str = "https://ttlahhqhtmqwyeqvbmbp.supabase.co"
    ):
        """
        Initialize Sablier client
        
        Args:
            api_url: Base URL of the Sablier backend API
            api_key: API key for authentication (starts with sk_)
            supabase_url: Supabase URL (default: optimized schema branch)
        """
        if not api_key:
            raise AuthenticationError("API key is required")
        
        # Initialize authentication
        self.auth = AuthHandler(api_url, api_key)
        self.auth.set_supabase_url(supabase_url)
        
        # Initialize HTTP client
        self.http = HTTPClient(api_url, self.auth)
        
        # Initialize managers
        self.models = ModelManager(self.http)
        self.scenarios = ScenarioManager(self.http)
    
    def health_check(self) -> dict:
        """
        Check if the API is reachable and healthy
        
        Returns:
            dict: Health status information
        """
        try:
            # TODO: Add dedicated health endpoint
            response = self.http.get('/api/v1/health')
            return response
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
