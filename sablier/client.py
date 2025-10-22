"""
Main Sablier SDK client
"""

from typing import Optional
from .auth import AuthHandler
from .http_client import HTTPClient
from .project.manager import ProjectManager
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
        fred_api_key: Optional[str] = None,
        interactive: bool = True
    ):
        """
        Initialize Sablier client
        
        Args:
            api_url: Base URL of the Sablier backend API (e.g., https://api.sablier.ai or http://localhost:8000)
            api_key: API key for authentication (format: sk_live_...) or dummy key for registration
            fred_api_key: Optional FRED API key for data searching and fetching
            interactive: Enable interactive prompts for confirmations (default: True)
        """
        if not api_key:
            raise AuthenticationError("API key is required")
        
        # Allow dummy keys for registration
        if not api_key.startswith("sk_") and not api_key.startswith("dummy_"):
            raise AuthenticationError("Invalid API key format. API keys should start with 'sk_'")
        
        # Initialize authentication
        self.auth = AuthHandler(api_url, api_key)
        
        # Initialize HTTP client
        self.http = HTTPClient(api_url, self.auth)
        
        # Store FRED API key for passing to DataCollection instances
        self.fred_api_key = fred_api_key
        
        # Initialize managers
        self.projects = ProjectManager(self.http, interactive=interactive)
        self.models = ModelManager(self.http, interactive=interactive)
        self.scenarios = ScenarioManager(self.http, self.models)
    
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
    
    async def register_user(
        self,
        email: str,
        name: str,
        company: str,
        api_key_name: str = "Default API Key"
    ) -> dict:
        """
        Register a new user and get an API key
        
        Args:
            email: User's email address
            name: User's full name
            company: Company name
            api_key_name: Name for the API key
            
        Returns:
            dict: Registration response with user details and API key
            
        Example:
            >>> response = await client.register_user(
            ...     email="user@company.com",
            ...     name="John Doe",
            ...     company="Acme Corp"
            ... )
            >>> print(f"API Key: {response['api_key']}")
        """
        payload = {
            "email": email,
            "name": name,
            "company": company,
            "api_key_name": api_key_name
        }
        
        response = self.http.post('/api/v1/auth/register', payload)
        return response
