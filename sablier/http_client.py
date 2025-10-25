"""
HTTP client for making requests to the Sablier API
"""

import requests
from typing import Any, Optional
from .auth import AuthHandler
from .exceptions import APIError, AuthenticationError


class HTTPClient:
    """Low-level HTTP client for API requests"""
    
    def __init__(self, api_url: str, auth_handler: AuthHandler):
        """
        Initialize HTTP client
        
        Args:
            api_url: Base URL of the Sablier API
            auth_handler: Authentication handler
        """
        self.api_url = api_url.rstrip('/')
        self.auth_handler = auth_handler
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
    
    def _get_url(self, endpoint: str, add_trailing_slash: bool = False) -> str:
        """Construct full URL for endpoint"""
        endpoint = endpoint.lstrip('/')
        # Add trailing slash if requested (for Cloud Run POST endpoints)
        if add_trailing_slash and not endpoint.endswith('/'):
            endpoint = endpoint + '/'
        return f"{self.api_url}/{endpoint}"
    
    def _handle_response(self, response: requests.Response) -> dict:
        """
        Handle API response and raise appropriate exceptions
        
        Args:
            response: requests Response object
            
        Returns:
            dict: Response data
            
        Raises:
            APIError: If request failed
            AuthenticationError: If authentication failed
        """
        try:
            response_data = response.json()
        except ValueError:
            response_data = {"error": response.text}
        
        if response.status_code == 401:
            raise AuthenticationError("Authentication failed. Please check your API key.")
        
        if response.status_code == 404:
            from .exceptions import ResourceNotFoundError
            raise ResourceNotFoundError(
                response_data.get('detail', 'Resource not found')
            )
        
        if not response.ok:
            error_message = response_data.get('detail', f"API request failed with status {response.status_code}")
            raise APIError(
                message=error_message,
                status_code=response.status_code,
                response_data=response_data
            )
        
        return response_data
    
    def get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """
        Make a GET request
        
        Args:
            endpoint: API endpoint (e.g., '/api/v1/models')
            params: Query parameters
            
        Returns:
            dict: Response data
        """
        # Only add trailing slash for simple endpoints without IDs (Cloud Run requirement)
        has_id_in_path = any(part and len(part) > 20 for part in endpoint.split('/'))
        add_slash = not has_id_in_path
        
        url = self._get_url(endpoint, add_trailing_slash=add_slash)
        headers = self.auth_handler.get_headers()
        
        response = self.session.get(url, headers=headers, params=params)
        return self._handle_response(response)
    
    def post(self, endpoint: str, data: Optional[dict] = None) -> dict:
        """
        Make a POST request
        
        Args:
            endpoint: API endpoint
            data: Request body data
            
        Returns:
            dict: Response data
        """
        # Don't add trailing slash - Cloud Run handles the routing correctly without it
        # Exception: Only for endpoints with IDs and sub-actions (PATCH-style)
        url = self._get_url(endpoint, add_trailing_slash=False)
        headers = self.auth_handler.get_headers()
        
        response = self.session.post(url, headers=headers, json=data, allow_redirects=False)
        
        # Handle redirects manually (307 for Cloud Run POST, 302 for HTTP->HTTPS)
        # Allow up to 2 levels of redirects to handle HTTP->HTTPS->path/ scenarios
        for _ in range(2):
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get('Location')
                if location:
                    # Explicitly use POST method on redirect, don't let requests auto-change to GET
                    response = self.session.request('POST', location, headers=headers, json=data, allow_redirects=False)
                else:
                    break
            else:
                break
        
        return self._handle_response(response)
    
    def put(self, endpoint: str, data: Optional[dict] = None) -> dict:
        """Make a PUT request"""
        url = self._get_url(endpoint)
        headers = self.auth_handler.get_headers()
        
        response = self.session.put(url, headers=headers, json=data)
        return self._handle_response(response)
    
    def patch(self, endpoint: str, data: Optional[dict] = None) -> dict:
        """
        Make a PATCH request
        
        Args:
            endpoint: API endpoint
            data: Request body data (only fields to update)
            
        Returns:
            dict: Response data
        """
        url = self._get_url(endpoint)
        headers = self.auth_handler.get_headers()
        
        response = self.session.patch(url, headers=headers, json=data)
        return self._handle_response(response)
    
    def delete(self, endpoint: str) -> dict:
        """Make a DELETE request"""
        url = self._get_url(endpoint)
        headers = self.auth_handler.get_headers()
        
        response = self.session.delete(url, headers=headers)
        return self._handle_response(response)
