"""
Main Sablier SDK client
"""

from typing import Optional, List, Dict, Union, Any
from .auth import AuthHandler
from .http_client import HTTPClient
from .project.manager import ProjectManager
from .model.manager import ModelManager
from .scenario.manager import ScenarioManager
from .portfolio.manager import PortfolioManager
from .user_settings import UserSettingsManager
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
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        fred_api_key: Optional[str] = None,
        interactive: bool = True
    ):
        """
        Initialize Sablier client
        
        Args:
            api_url: Base URL of the Sablier backend API. If None, uses saved default URL
            api_key: Optional API key for authentication. If None, will register a new user and generate API key
            fred_api_key: Optional FRED API key for data searching and fetching
            interactive: Enable interactive prompts for confirmations (default: True)
        """
        # Initialize user settings manager first
        self.user_settings = UserSettingsManager()
        
        # Use default URL if not provided
        if api_url is None:
            api_url = self.user_settings.get_default_api_url()
            if api_url is None:
                print("❌ No API URL provided and no default URL set.")
                print("💡 First-time setup required:")
                print("   1. Run setup script: python setup_sablier.py")
                print("   2. Or provide api_url: SablierClient(api_url='https://your-api-url.com')")
                print("   3. Or run migration: python migrate_api_key.py")
                raise ValueError("No API URL provided and no default URL set. Please see setup instructions above.")
        
        # If no API key provided, try to get saved one or register new user
        if not api_key:
            # Try to get default API key first
            saved_api_key = self.user_settings.get_default_api_key()
            if saved_api_key:
                if interactive:
                    print(f"🔑 Using default API key")
                api_key = saved_api_key
            else:
                # No default key, try to get saved API key for this URL
                saved_api_key = self.user_settings.get_active_api_key(api_url)
                if saved_api_key:
                    if interactive:
                        print(f"🔑 Using saved API key for {api_url}")
                    api_key = saved_api_key
                else:
                    # No saved key, register new user
                    if interactive:
                        print(f"🔑 No saved API key found for {api_url}")
                        print(f"🔄 Registering new user...")
                    api_key = self._register_and_get_api_key(api_url, interactive)
                    # Save the new API key as default
                    self.user_settings.save_api_key(api_key, api_url, description=None, is_default=True)
                    if interactive:
                        print(f"💾 API key saved as default")
        
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
    # API KEY MANAGEMENT METHODS
    # ============================================
    
    def save_api_key(self, api_key: str, api_url: Optional[str] = None, 
                     description: Optional[str] = None, is_default: bool = False) -> bool:
        """
        Save an API key for future use
        
        Args:
            api_key: The API key to save
            api_url: The API URL (defaults to current client URL)
            description: Optional name/description (e.g., "default", "template", "production")
            is_default: Whether this should be the default key
            
        Returns:
            bool: True if saved successfully
        """
        if api_url is None:
            api_url = self.http.base_url
        
        return self.user_settings.save_api_key(api_key, api_url, description=description, is_default=is_default)
    
    def get_api_key(self, name: Optional[str] = None) -> Optional[str]:
        """
        Get an API key by name, or return the default key
        
        Args:
            name: The name of the API key (e.g., "template", "production").
                  If None, returns the default key.
                  
        Returns:
            str: The API key, or None if not found
        """
        if name is None:
            return self.user_settings.get_default_api_key()
        return self.user_settings.get_api_key_by_name(name)
    
    def list_api_keys(self) -> List[Dict[str, Any]]:
        """
        List all saved API keys
        
        Returns:
            List of API key dictionaries
        """
        return self.user_settings.list_api_keys()
    
    def delete_api_key(self, api_key: str) -> bool:
        """
        Delete a saved API key
        
        Args:
            api_key: The API key to delete
            
        Returns:
            bool: True if deleted successfully
        """
        return self.user_settings.delete_api_key(api_key)
    
    def set_default_api_url(self, api_url: str) -> bool:
        """
        Set the default API URL
        
        Args:
            api_url: The default API URL
            
        Returns:
            bool: True if set successfully
        """
        return self.user_settings.set_default_api_url(api_url)
    
    def get_default_api_url(self) -> Optional[str]:
        """
        Get the default API URL
        
        Returns:
            str: The default API URL, or None if not set
        """
        return self.user_settings.get_default_api_url()
    
    def clear_all_user_data(self) -> bool:
        """
        Clear all user data (API keys and settings)
        
        Returns:
            bool: True if cleared successfully
        """
        return self.user_settings.clear_all_data()
    
    def clear_all_portfolios(self) -> int:
        """
        Delete all portfolios from local SQLite database
        
        Returns:
            int: Number of portfolios deleted
        """
        import sqlite3
        import os
        
        db_path = os.path.expanduser("~/.sablier/portfolios.db")
        
        try:
            with sqlite3.connect(db_path) as conn:
                # Count portfolios before deletion
                cursor = conn.execute("SELECT COUNT(*) FROM portfolios")
                count_before = cursor.fetchone()[0]
                
                if count_before == 0:
                    print("📭 No portfolios found in local database")
                    return 0
                
                # Delete all portfolios and related data
                conn.execute("DELETE FROM portfolio_tests")
                conn.execute("DELETE FROM portfolio_optimizations") 
                conn.execute("DELETE FROM portfolio_evaluations")
                conn.execute("DELETE FROM portfolios")
                
                conn.commit()
                
                print(f"🗑️ Deleted {count_before} portfolios from local database")
                print("✅ All portfolio data cleared")
                
                return count_before
                
        except Exception as e:
            print(f"❌ Failed to clear portfolios: {e}")
            return 0
    
    
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
                        capital: float = 100000.0, description: str = "", constraint_type: str = "long_short",
                        asset_configs: Optional[Dict[str, Dict[str, Any]]] = None):
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
            asset_configs: Optional dict mapping asset names to their return calculation config
                Example: {
                    "10-Year Treasury": {
                        "type": "treasury_bond",
                        "params": {
                            "coupon_rate": 0.025,
                            "face_value": 1000,
                            "issue_date": "2020-01-01",
                            "payment_frequency": 2
                        }
                    }
                }
            
        Returns:
            Portfolio instance
        """
        return self.portfolios.create(
            name=name,
            target_set=target_set,
            weights=weights,
            capital=capital,
            description=description,
            constraint_type=constraint_type,
            asset_configs=asset_configs
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
            # Health check endpoint not yet implemented
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
