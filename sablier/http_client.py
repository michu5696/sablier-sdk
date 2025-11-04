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
    
    def _make_request_with_retry(self, method: str, url: str, headers: dict, **kwargs) -> requests.Response:
        """
        Make HTTP request with timeout and retry logic for long operations
        
        Args:
            method: HTTP method
            url: Request URL
            headers: Request headers
            **kwargs: Additional request parameters
            
        Returns:
            requests.Response: HTTP response
        """
        import time
        from requests.exceptions import Timeout, ConnectionError
        
        # Determine timeout based on endpoint
        timeout = kwargs.pop('timeout', None)
        if timeout is None:
            # Validation can take a very long time (reconstruction for multiple samples)
            if '/validate' in url:
                timeout = 1800  # 30 minutes for validation (can be very slow with reconstruction)
            # Other long operations
            elif any(endpoint in url for endpoint in ['/train', '/fetch-data', '/generate-samples']):
                timeout = 600  # 10 minutes for ML operations and data fetching
            else:
                timeout = 60   # 1 minute for regular operations
        
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                response = self.session.request(method, url, headers=headers, timeout=timeout, **kwargs)
                return response
            except Timeout:
                if attempt < max_retries - 1:
                    print(f"⚠️ Request timeout (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise APIError(f"Request timed out after {max_retries} attempts. The operation may still be running on the server.")
            except ConnectionError as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Connection error (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise APIError(f"Connection failed after {max_retries} attempts: {str(e)}")
        
        # This should never be reached, but just in case
        raise APIError("Unexpected error in request retry logic")
    
    def get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """
        Make a GET request
        
        Args:
            endpoint: API endpoint (e.g., '/api/v1/models')
            params: Query parameters
            
        Returns:
            dict: Response data
        """
        # Add trailing slash for list endpoints (Cloud Run + FastAPI router requirement)
        # Only add if endpoint doesn't have an ID (no long UUID-like segments)
        has_id_in_path = any(part and len(part) > 20 for part in endpoint.split('/'))
        add_slash = not has_id_in_path
        
        url = self._get_url(endpoint, add_trailing_slash=add_slash)
        headers = self.auth_handler.get_headers()
        
        response = self._make_request_with_retry('GET', url, headers, params=params)
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
        
        response = self._make_request_with_retry('POST', url, headers, json=data, allow_redirects=False)
        
        # Handle redirects manually (307 for Cloud Run POST, 302 for HTTP->HTTPS)
        # Allow up to 2 levels of redirects to handle HTTP->HTTPS->path/ scenarios
        for _ in range(2):
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get('Location')
                if location:
                    # Explicitly use POST method on redirect, don't let requests auto-change to GET
                    response = self._make_request_with_retry('POST', location, headers, json=data, allow_redirects=False)
                else:
                    break
            else:
                break
        
        return self._handle_response(response)
    
    def put(self, endpoint: str, data: Optional[dict] = None) -> dict:
        """Make a PUT request"""
        url = self._get_url(endpoint)
        headers = self.auth_handler.get_headers()
        
        response = self._make_request_with_retry('PUT', url, headers, json=data)
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
        
        response = self._make_request_with_retry('PATCH', url, headers, json=data)
        return self._handle_response(response)
    
    def delete(self, endpoint: str) -> dict:
        """Make a DELETE request"""
        url = self._get_url(endpoint)
        headers = self.auth_handler.get_headers()
        
        response = self._make_request_with_retry('DELETE', url, headers)
        return self._handle_response(response)
