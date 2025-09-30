"""
Sablier SDK - Python SDK for Market Scenario Generation

Create scenario-conditioned synthetic financial data for portfolio testing and risk analysis.
"""

__version__ = "0.1.0"

from .client import SablierClient
from .exceptions import (
    SablierError,
    AuthenticationError,
    APIError,
    ValidationError,
    ResourceNotFoundError,
    JobTimeoutError,
    JobFailedError
)

__all__ = [
    "SablierClient",
    "SablierError",
    "AuthenticationError",
    "APIError",
    "ValidationError",
    "ResourceNotFoundError",
    "JobTimeoutError",
    "JobFailedError",
]
