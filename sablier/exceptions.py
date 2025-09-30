"""
Custom exceptions for the Sablier SDK
"""


class SablierError(Exception):
    """Base exception for all Sablier SDK errors"""
    pass


class AuthenticationError(SablierError):
    """Raised when authentication fails"""
    pass


class APIError(SablierError):
    """Raised when API request fails"""
    
    def __init__(self, message: str, status_code: int = None, response_data: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class ValidationError(SablierError):
    """Raised when data validation fails"""
    pass


class ResourceNotFoundError(SablierError):
    """Raised when a requested resource is not found"""
    pass


class JobTimeoutError(SablierError):
    """Raised when a job times out"""
    pass


class JobFailedError(SablierError):
    """Raised when a background job fails"""
    
    def __init__(self, message: str, job_id: str = None, error_details: dict = None):
        super().__init__(message)
        self.job_id = job_id
        self.error_details = error_details
