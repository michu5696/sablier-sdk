"""
Main Sablier SDK client
"""

import logging
from typing import Optional, List, Dict, Union, Any
from .auth import AuthHandler
from .http_client import HTTPClient
from .project.manager import ProjectManager
from .model.manager import ModelManager
from .scenario.manager import ScenarioManager
from .portfolio.manager import PortfolioManager
from .user_settings import UserSettingsManager
from .exceptions import AuthenticationError

logger = logging.getLogger(__name__)


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
        
        # Store the API URL for use in registration/verification methods
        self.api_url = api_url
        
        # Persist the resolved API URL as default for future runs
        try:
            self.user_settings.set_default_api_url(api_url)
        except Exception:
            pass
        
        # If no API key provided, try to get saved one
        # Note: Automatic registration is no longer supported - email verification is required
        if not api_key:
            # Try to get default API key first
            saved_api_key = self.user_settings.get_default_api_key()
            if saved_api_key and saved_api_key.strip():
                if interactive:
                    print(f"🔑 Using default API key")
                api_key = saved_api_key.strip()
            else:
                # No default key, try to get saved API key for this URL
                saved_api_key = self.user_settings.get_active_api_key(api_url)
                if saved_api_key and saved_api_key.strip():
                    if interactive:
                        print(f"🔑 Using saved API key for {api_url}")
                    api_key = saved_api_key.strip()
                else:
                    # No saved key - inform user they need to register
                    if interactive:
                        print("\n" + "=" * 80)
                        print("🔑 No API key found")
                        print("=" * 80)
                        print("\nTo get started, please register:")
                        print("  1. Register: client.register_user(email='...', name='...', company='...', role='CTO')")
                        print("  2. Check your email and click the verification link")
                        print("  3. The verification link will provide your API key - save it securely")
                        print("\nOr provide an API key directly: SablierClient(api_url='...', api_key='sk_...')")
                        print("=" * 80 + "\n")
                    # Set api_key to None - it will be validated below
                    api_key = None
        
        # Validate API key format (if provided)
        # Ensure it's not empty after stripping whitespace
        if api_key:
            api_key = api_key.strip()
            if not api_key:
                api_key = None
            elif not api_key.startswith("sk_") and not api_key.startswith("dummy_"):
                raise AuthenticationError("Invalid API key format. API keys should start with 'sk_'")
        
        # Initialize authentication (only if API key is provided)
        # Client can be created without API key for registration/verification
        if api_key:
            self.auth = AuthHandler(api_url, api_key)
            # Initialize HTTP client
            self.http = HTTPClient(api_url, self.auth)
            
            # Automatically fetch and save user info when API key is used for the first time
            # This ensures the email and other user info are stored locally
            try:
                saved_keys = self.user_settings.list_api_keys()
                existing_key = next((k for k in saved_keys if k.get('api_key') == api_key), None)
                
                # If key not saved, or saved but missing email, fetch user info and save it
                if not existing_key or not existing_key.get('user_email'):
                    try:
                        user_info = self.http.get('/api/v1/account/user')
                        user_email = user_info.get('email')
                        
                        if user_email:
                            # Save/update API key with user info
                            is_default = existing_key.get('description') == 'default' if existing_key else False
                            description = existing_key.get('description') if existing_key else None
                            
                            self.user_settings.save_api_key(
                                api_key=api_key,
                                api_url=api_url,
                                user_email=user_email,
                                description=description,
                                is_default=is_default
                            )
                            
                            if interactive and not existing_key:
                                logger.info(f"💾 API key saved with user info: {user_email}")
                    except Exception as e:
                        # If we can't fetch user info (network error, invalid key, etc.), 
                        # continue without saving - don't block client initialization
                        if interactive:
                            logger.debug(f"Could not fetch user info for API key: {e}")
                        pass
            except Exception as e:
                # Don't block client initialization if there's an error
                if interactive:
                    logger.debug(f"Could not check/save API key info: {e}")
                pass
        else:
            # No API key - client can still be used for registration/verification
            self.auth = None
            self.http = None
        
        # Store FRED API key for passing to DataCollection instances
        self.fred_api_key = fred_api_key
        
        # Store interactive flag
        self.interactive = interactive
        
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
                     description: Optional[str] = None, is_default: bool = False) -> str:
        """
        Save an API key for future use
        
        Args:
            api_key: The API key to save
            api_url: The API URL (defaults to current client URL or saved default)
            description: Optional name/description (e.g., "default", "template", "production")
                        If None and this is the first key, will default to "default"
            is_default: Whether this should be the default key (defaults to True if first key)
            
        Returns:
            str: Success message
        """
        # Get API URL - use provided, then current client, then saved default
        if api_url is None:
            if hasattr(self, 'http') and self.http:
                api_url = self.http.base_url
            else:
                api_url = self.user_settings.get_default_api_url()
                if api_url is None:
                    raise ValueError("api_url is required when client is not initialized with an API key")
        
        # Check if this key already exists
        existing_keys = self.user_settings.list_api_keys()
        existing_key = next((k for k in existing_keys if k.get('api_key') == api_key), None)
        
        # Check if this is the first key (excluding this key if it already exists)
        keys_to_check = [k for k in existing_keys if k.get('api_key') != api_key]
        is_first_key = len(keys_to_check) == 0
        
        # Set defaults: first key is default, and gets "default" description ONLY if none provided
        if is_first_key:
            is_default = True
            if description is None:
                description = "default"
        elif description is None:
            # Not first key, but no description provided - use a generic one
            description = f"key_{len(keys_to_check) + 1}"
        
        saved = self.user_settings.save_api_key(
            api_key=api_key, 
            api_url=api_url, 
            description=description, 
            is_default=is_default
        )
        
        if saved:
            return f"Successfully saved API key locally (description: '{description}')"
        else:
            return "Failed to save API key"
    
    def update_api_key_description(self, api_key: str, description: str) -> bool:
        """
        Update the description/name of a saved API key
        
        Args:
            api_key: The API key to update
            description: New description/name for the key
            
        Returns:
            bool: True if updated successfully
            
        Example:
            >>> client.update_api_key_description("sk_...", "production")
        """
        return self.user_settings.update_api_key_description(api_key, description)
    
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
    
    # ============================================
    # ACCOUNT MANAGEMENT METHODS
    # ============================================
    
    def _check_auth_status(self) -> Dict[str, Any]:
        """
        Check the current authentication status (for debugging)
        
        Returns:
            Dict with authentication status information
        """
        status = {
            'has_http': self.http is not None,
            'has_auth': self.auth is not None,
            'has_api_key': self.auth.api_key is not None if self.auth else False,
            'api_key_length': len(self.auth.api_key) if (self.auth and self.auth.api_key) else 0,
            'api_key_prefix': self.auth.api_key[:10] + '...' if (self.auth and self.auth.api_key and len(self.auth.api_key) > 10) else None
        }
        
        # Try to get headers to see if it would work
        try:
            if self.auth:
                headers = self.auth.get_headers()
                status['can_get_headers'] = True
                status['has_auth_header'] = 'Authorization' in headers
            else:
                status['can_get_headers'] = False
                status['has_auth_header'] = False
        except Exception as e:
            status['can_get_headers'] = False
            status['get_headers_error'] = str(e)
        
        return status
    
    def _ensure_authenticated(self):
        """
        Ensure the client is properly authenticated before making API calls
        
        Raises:
            AuthenticationError: If authentication is not properly configured
        """
        if not self.http:
            raise AuthenticationError(
                "API key required. Create client with api_key parameter or save an API key first.\n"
                "To fix: Recreate the client: client = SablierClient(api_url='...', api_key='sk_...')"
            )
        if not self.auth:
            raise AuthenticationError(
                "Authentication handler not initialized. Please recreate the client: "
                "client = SablierClient(api_url='...', api_key='sk_...')"
            )
        if not self.auth.api_key:
            raise AuthenticationError(
                "API key is missing or invalid. Please recreate the client with a valid API key: "
                "client = SablierClient(api_url='...', api_key='sk_...')\n"
                "Or check your saved API keys: client.list_api_keys()"
            )
        
        # Additional check: ensure API key is not empty string
        api_key = self.auth.api_key.strip() if isinstance(self.auth.api_key, str) else self.auth.api_key
        if not api_key:
            raise AuthenticationError(
                "API key is empty. Please recreate the client with a valid API key: "
                "client = SablierClient(api_url='...', api_key='sk_...')\n"
                "Or check your saved API keys: client.list_api_keys()"
            )
    
    def get_limits_and_usage(self) -> None:
        """
        Display limits and current usage for the authenticated user in a formatted way
        
        Example:
            >>> client = SablierClient(api_url="...", api_key="sk_...")
            >>> client.get_limits_and_usage()
            📊 LIMITS & USAGE:
               Scenarios This Month: 1 out of 100
               Simulation Paths This Month: 200 out of 100,000
               Max Paths/Scenario: 1,000
            ...
        """
        self._ensure_authenticated()
        response = self.http.get('/api/v1/account/limits')
        
        limits = response.get('limits', {})
        usage = response.get('usage', {})
        remaining = usage.get('remaining', {})
        
        print("\n📊 LIMITS & USAGE:")
        
        # Scenarios: show "X out of Y" format
        scenarios_used = usage.get('scenarios_this_month', 0)
        scenarios_limit = limits.get('max_scenarios_per_month')
        if scenarios_limit is not None:
            print(f"   Scenarios This Month: {scenarios_used} out of {scenarios_limit}")
        else:
            print(f"   Scenarios This Month: {scenarios_used} (Unlimited)")
        
        # Simulation paths: show "X out of Y" format
        paths_used = usage.get('simulation_paths_this_month', 0)
        paths_limit = limits.get('max_simulation_paths')
        if paths_limit is not None:
            print(f"   Simulation Paths This Month: {paths_used:,} out of {paths_limit:,}")
        else:
            print(f"   Simulation Paths This Month: {paths_used:,} (Unlimited)")
        
        # Max paths per scenario
        max_paths_per_scenario = limits.get('max_simulation_paths_per_scenario')
        if max_paths_per_scenario is not None:
            print(f"   Max Paths/Scenario: {max_paths_per_scenario:,}")
        else:
            print(f"   Max Paths/Scenario: Unlimited")
        
        print(f"\n📅 Period: {response.get('period_start', 'N/A')} to {response.get('period_end', 'N/A')}\n")
    
    def get_balance(self) -> Dict[str, Any]:
        """
        Get credit balance for the authenticated user
        
        Returns:
            Dict containing:
            - balance: Current credit balance (float)
            - currency: Currency code (e.g., "USD")
            - api_key_id: Optional API key ID if balance is key-specific
            
        Example:
            >>> client = SablierClient(api_url="...", api_key="sk_...")
            >>> balance_info = client.get_balance()
            >>> print(f"Balance: {balance_info['balance']} {balance_info['currency']}")
        """
        self._ensure_authenticated()
        response = self.http.get('/api/v1/account/balance')
        return response
    
    def get_user_info(self) -> Dict[str, Any]:
        """
        Get user information for the authenticated user
        
        Returns:
            Dict containing user_id, email, name, company, role, email_verified
            
        Example:
            >>> client = SablierClient(api_url="...", api_key="sk_...")
            >>> user = client.get_user_info()
            >>> print(f"Email: {user['email']}")
        """
        self._ensure_authenticated()
        response = self.http.get('/api/v1/account/user')
        return response
    
    def get_api_key_usage(self, key_id: str, days: int = 7) -> Dict[str, Any]:
        """
        Get usage statistics for a specific API key
        
        Args:
            key_id: The API key ID
            days: Number of days to look back (default: 7)
            
        Returns:
            Dict containing usage statistics for the specified period
            
        Example:
            >>> client = SablierClient(api_url="...", api_key="sk_...")
            >>> usage = client.get_api_key_usage(key_id="...", days=30)
            >>> print(f"Total requests: {usage['usage']['total_requests']}")
        """
        self._ensure_authenticated()
        response = self.http.get(f'/api/v1/api-keys/{key_id}/usage', params={'days': days})
        return response
    
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
                        capital: float = 100000.0, description: str = "",
                        asset_configs: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Create a new portfolio
        
        Args:
            name: Portfolio name
            target_set: TargetSet instance to create portfolio from
            weights: Either:
                - Dict[str, float]: Dictionary of asset weights (sum of absolute values must equal 1.0)
                - List[float]: List of weights assigned to assets in order (sum of absolute values must equal 1.0)
                - None: Random weights will be generated (sum of absolute values = 1.0)
            capital: Total capital allocation (default $100k)
            description: Optional description
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
            
        Note:
            Portfolios support long-short positions (negative weights allowed).
            The sum of absolute values of weights must equal 1.0.
        """
        return self.portfolios.create(
            name=name,
            target_set=target_set,
            weights=weights,
            capital=capital,
            description=description,
            asset_configs=asset_configs
        )
    
    # Note: _register_and_get_api_key method removed
    # Automatic registration is no longer supported due to email verification requirement
    # Users must explicitly call register_user() and verify_email() methods
    
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
    
    def register_user(
        self,
        email: str,
        name: str,
        company: str,
        role: str,
        api_key_name: str = "Default"
    ) -> dict:
        """
        Register a new user (email verification required before API key creation)
        
        Args:
            email: User's email address
            name: User's full name
            company: Company name
            role: Optional - User's role at their company (e.g., 'CTO', 'Data Scientist', 'Portfolio Manager')
                  This is informational only, not used for permissions
            api_key_name: Name for the API key (not used until email is verified)
            
        Returns:
            dict: Registration response with user details and message to verify email
            
        Example:
            >>> response = client.register_user(
            ...     email="user@company.com",
            ...     name="John Doe",
            ...     company="Acme Corp",
            ...     role="CTO"
            ... )
            >>> print(f"Message: {response['message']}")
            >>> # User must verify email before receiving API key
        """
        payload = {
            "email": email,
            "name": name,
            "company": company,
            "role": role
        }
        
        # Registration doesn't require authentication, so we need to make a direct request
        # Use the stored API URL from client initialization
        base_url = getattr(self, 'api_url', None)
        if not base_url:
            # Fallback: try http client or user_settings
            if hasattr(self, 'http') and self.http:
                base_url = self.http.base_url
            if not base_url:
                base_url = self.user_settings.get_default_api_url()
                if not base_url:
                    raise ValueError("API URL required for registration. Please provide api_url when creating SablierClient.")
        
        import requests
        try:
            response = requests.post(
                f"{base_url}/api/v1/auth/register",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            from .exceptions import APIError
            raise APIError(
                message=f"Failed to register user: {str(e)}",
                status_code=getattr(e.response, 'status_code', 500),
                response_data={}
            )
    
    def verify_email(self, verification_token: str) -> dict:
        """
        Verify user's email address using a verification token
        
        Args:
            verification_token: Email verification token (from email link or database)
            
        Returns:
            dict: Verification response with user details and API key
            
        Example:
            >>> response = client.verify_email(verification_token="abc123...")
            >>> api_key = response['api_key']
            >>> # Save the API key for future use
            >>> client.save_api_key(api_key, api_url=client.http.base_url)
        """
        # Get the base URL (email verification doesn't require auth)
        # Use the stored API URL from client initialization
        base_url = getattr(self, 'api_url', None)
        if not base_url:
            # Fallback: try http client or user_settings
            if hasattr(self, 'http') and self.http:
                base_url = self.http.base_url
            if not base_url:
                base_url = self.user_settings.get_default_api_url()
                if not base_url:
                    raise ValueError("API URL required for email verification. Please provide api_url when creating SablierClient.")
        
        import requests
        try:
            response = requests.get(
                f"{base_url}/api/v1/auth/verify-email",
                params={"token": verification_token}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            from .exceptions import APIError
            raise APIError(
                message=f"Failed to verify email: {str(e)}",
                status_code=getattr(e.response, 'status_code', 500),
                response_data={}
            )
