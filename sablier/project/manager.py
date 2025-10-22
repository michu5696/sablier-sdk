"""Project manager for creating and retrieving projects"""

from typing import Optional, Any, List
from ..http_client import HTTPClient
from .builder import Project


class ProjectManager:
    """Manages project creation and retrieval"""
    
    def __init__(self, http_client: HTTPClient, interactive: bool = True):
        self.http = http_client
        self.interactive = interactive
    
    def create(self,
               name: str,
               description: str = "",
               training_start_date: str = "2015-01-01",
               training_end_date: str = "2023-12-31") -> Project:
        """
        Create a new project
        
        Args:
            name: Project name
            description: Project description
            training_start_date: Training period start date (YYYY-MM-DD)
            training_end_date: Training period end date (YYYY-MM-DD)
            
        Returns:
            Project: New project instance
            
        Example:
            >>> project = client.projects.create(
            ...     name="Treasury Yield Analysis",
            ...     description="Analysis of treasury yield predictions",
            ...     training_start_date="2015-01-01",
            ...     training_end_date="2023-12-31"
            ... )
        """
        # Call API to create project
        response = self.http.post('/api/v1/projects', {
            "name": name,
            "description": description,
            "training_start_date": training_start_date,
            "training_end_date": training_end_date
        })
        
        print(f"✅ Project created: {response['name']} (ID: {response['id']})")
        print(f"   Training period: {training_start_date} to {training_end_date}")
        
        return Project(self.http, response, interactive=self.interactive)
    
    def get(self, project_id: str) -> Project:
        """
        Retrieve an existing project
        
        Args:
            project_id: Project ID
            
        Returns:
            Project: Project instance
        """
        # Call API to get project
        response = self.http.get(f'/api/v1/projects/{project_id}')
        
        return Project(self.http, response, interactive=self.interactive)
    
    def list(self,
             limit: int = 100,
             offset: int = 0) -> List[Project]:
        """
        List all projects
        
        Args:
            limit: Maximum number of projects to return
            offset: Number of projects to skip
            
        Returns:
            List[Project]: List of project instances
        """
        # Call API to list projects
        response = self.http.get('/api/v1/projects', params={
            "limit": limit,
            "offset": offset
        })
        
        return [Project(self.http, data, interactive=self.interactive) for data in response]
