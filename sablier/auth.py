"""
Authentication handling for Sablier SDK
"""

import requests
from typing import Optional
from .exceptions import AuthenticationError


class AuthHandler:
    """Handles authentication for API requests"""
    
    def __init__(self, api_url: str, api_key: Optional[str] = None):
        """
        Initialize authentication handler
        
        Args:
            api_url: Base URL of the Sablier API
            api_key: API key for authentication
        """
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self._token = None
        self._supabase_url = None
    
    def get_headers(self) -> dict:
        """Get authentication headers for API requests"""
        if not self.api_key:
            raise AuthenticationError(
                "No API key provided. Please create the client with an API key: "
                "SablierClient(api_url='...', api_key='sk_...')"
            )
        
        # Ensure API key is not just whitespace
        api_key = self.api_key.strip() if isinstance(self.api_key, str) else self.api_key
        if not api_key:
            raise AuthenticationError(
                "API key is empty. Please provide a valid API key: "
                "SablierClient(api_url='...', api_key='sk_...')"
            )
        
        # FastAPI's Header() dependency converts parameter names to header names
        # When you use `authorization: str = Header(...)`, FastAPI looks for the header 'authorization' (lowercase)
        # HTTP standard says headers are case-insensitive, but FastAPI's Header() does case-sensitive matching
        # by default based on the parameter name. Using lowercase to match FastAPI's expectation.
        headers = {
            'authorization': f'Bearer {api_key}',  # FastAPI expects lowercase to match parameter name
            'Content-Type': 'application/json'
        }
        
        return headers
    
    def validate(self) -> bool:
        """
        Validate that the API key is valid
        
        Returns:
            bool: True if authentication is valid
            
        Raises:
            AuthenticationError: If authentication fails
        """
        if not self.api_key:
            raise AuthenticationError("No API key provided")
        
        # Health check endpoint not yet implemented
        # For now, assume valid if key is provided
        return True
