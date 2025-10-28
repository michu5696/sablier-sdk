"""
Main Sablier SDK client
"""

from typing import Optional, List, Dict, Union
from .auth import AuthHandler
from .http_client import HTTPClient
from .project.manager import ProjectManager
from .model.manager import ModelManager
from .scenario.manager import ScenarioManager
from .portfolio.manager import PortfolioManager
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
        api_key: Optional[str] = None,
        fred_api_key: Optional[str] = None,
        interactive: bool = True
    ):
        """
        Initialize Sablier client
        
        Args:
            api_url: Base URL of the Sablier backend API (e.g., https://api.sablier.ai or http://localhost:8000)
            api_key: Optional API key for authentication. If None, will register a new user and generate API key
            fred_api_key: Optional FRED API key for data searching and fetching
            interactive: Enable interactive prompts for confirmations (default: True)
        """
        # If no API key provided, register a new user and get an API key
        if not api_key:
            api_key = self._register_and_get_api_key(api_url, interactive)
        
        # Validate API key format
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
        
        # Initialize portfolio manager (local-only)
        self.portfolios = PortfolioManager(self.http)
    
    # ============================================
    # CONSISTENT API METHODS
    # ============================================
    
    def list_projects(self, include_templates: bool = True) -> List:
        """List all projects"""
        return self.projects.list(include_templates=include_templates)
    
    def get_project(self, identifier) -> Optional:
        """
        Get project by name or index
        
        Args:
            identifier: Project name (str) or index (int)
            
        Returns:
            Project instance or None if not found
        """
        projects = self.list_projects()
        
        if isinstance(identifier, int):
            # Get by index
            if 0 <= identifier < len(projects):
                return projects[identifier]
            return None
        elif isinstance(identifier, str):
            # Get by name
            for project in projects:
                if project.name == identifier:
                    return project
            return None
        else:
            raise ValueError("Identifier must be string (name) or int (index)")
    
    # ============================================
    # PORTFOLIO METHODS (CONSISTENT API)
    # ============================================
    
    def list_portfolios(self) -> List:
        """List all portfolios"""
        return self.portfolios.list()
    
    def get_portfolio(self, identifier):
        """
        Get portfolio by name or index
        
        Args:
            identifier: Portfolio name (str) or index (int)
            
        Returns:
            Portfolio instance or None if not found
        """
        portfolios = self.list_portfolios()
        
        if isinstance(identifier, int):
            # Get by index
            if 0 <= identifier < len(portfolios):
                return portfolios[identifier]
            return None
        elif isinstance(identifier, str):
            # Get by name
            for portfolio in portfolios:
                if portfolio.name == identifier:
                    return portfolio
            return None
        else:
            raise ValueError("Identifier must be string (name) or int (index)")
    
    def create_portfolio(self, name: str, target_set, weights: Optional[Union[Dict[str, float], List[float]]] = None, 
                        capital: float = 100000.0, description: str = "", constraint_type: str = "long_short"):
        """
        Create a new portfolio
        
        Args:
            name: Portfolio name
            target_set: TargetSet instance to create portfolio from
            weights: Either:
                - Dict[str, float]: Dictionary of asset weights (must sum to 1.0)
                - List[float]: List of weights assigned to assets in order (must sum to 1.0)
                - None: Random weights will be generated (sum to 1.0)
            capital: Total capital allocation (default $100k)
            description: Optional description
            constraint_type: Type of constraints ('long_short', 'long_only', 'custom')
            
        Returns:
            Portfolio instance
        """
        return self.portfolios.create(
            name=name,
            target_set=target_set,
            weights=weights,
            capital=capital,
            description=description,
            constraint_type=constraint_type
        )
    
    def _register_and_get_api_key(self, api_url: str, interactive: bool) -> str:
        """
        Register a new user and get an API key
        
        Args:
            api_url: Base URL of the Sablier backend API
            interactive: Whether to prompt for user input
            
        Returns:
            str: Generated API key
        """
        import requests
        import uuid
        import time
        
        # Generate a unique email for this session
        timestamp = int(time.time())
        email = f"auto_user_{timestamp}@example.com"
        
        if interactive:
            print(f"🔑 No API key provided. Registering new user: {email}")
        
        # Register the user
        try:
            response = requests.post(
                f"{api_url}/api/v1/auth/register", 
                json={
                    "email": email,
                    "password": str(uuid.uuid4()),  # Random password
                    "name": f"Auto User {timestamp}",
                    "company": "Test Company"
                }
            )
            
            if response.status_code != 200:
                raise AuthenticationError(f"Failed to register user: {response.text}")
            
            user_data = response.json()
            user_id = user_data.get("user_id")
            api_key = user_data.get("api_key")
            
            if interactive:
                print(f"✅ User registered successfully (ID: {user_id})")
                print(f"🔑 API key generated: {api_key}")
            
            return api_key
            
        except requests.RequestException as e:
            raise AuthenticationError(f"Failed to register user and get API key: {e}")
    
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
