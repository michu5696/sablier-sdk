"""
Main Sablier SDK client
"""

from typing import Optional
from .auth import AuthHandler
from .http_client import HTTPClient
from .model.manager import ModelManager
from .scenario.manager import ScenarioManager
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
        supabase_url: str = "https://ttlahhqhtmqwyeqvbmbp.supabase.co",
        fred_api_key: Optional[str] = None,
        interactive: bool = True
    ):
        """
        Initialize Sablier client
        
        Args:
            api_url: Base URL of the Sablier backend API
            api_key: API key for authentication (starts with sk_)
            supabase_url: Supabase URL (default: optimized schema branch)
            fred_api_key: Optional FRED API key for data searching and fetching
            interactive: Enable interactive prompts for confirmations (default: True)
        """
        if not api_key:
            raise AuthenticationError("API key is required")
        
        # Initialize authentication
        self.auth = AuthHandler(api_url, api_key)
        self.auth.set_supabase_url(supabase_url)
        
        # Initialize HTTP client
        self.http = HTTPClient(api_url, self.auth)
        
        # Store FRED API key for passing to DataCollection instances
        self.fred_api_key = fred_api_key
        
        # Initialize managers
        self.models = ModelManager(self.http, interactive=interactive)
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
