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
            raise AuthenticationError("No API key provided")
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # Add Supabase URL if available
        if self._supabase_url:
            headers['x-supabase-url'] = self._supabase_url
        
        return headers
    
    def set_supabase_url(self, supabase_url: str):
        """Set Supabase URL for backend communication"""
        self._supabase_url = supabase_url
    
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
        
        # TODO: Add health check endpoint call
        # For now, assume valid if key is provided
        return True
