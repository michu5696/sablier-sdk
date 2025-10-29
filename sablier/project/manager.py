"""Project manager for creating and retrieving projects"""

from typing import Optional, Any, List
from datetime import datetime
from ..http_client import HTTPClient
from ..exceptions import APIError
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
               training_end_date: Optional[str] = None) -> Optional[Project]:
        """
        Create a new project
        
        Args:
            name: Project name
            description: Project description
            training_start_date: Training period start date (YYYY-MM-DD). Default: "2015-01-01"
            training_end_date: Training period end date (YYYY-MM-DD). Default: today's date
            
        Returns:
            Project: New project instance
            
        Example:
            >>> project = client.projects.create(
            ...     name="Treasury Yield Analysis",
            ...     description="Analysis of treasury yield predictions"
            ... )
        """
        # Default training_end_date to today if not provided
        if training_end_date is None:
            training_end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Call API to create project
        try:
            response = self.http.post('/api/v1/projects', {
                "name": name,
                "description": description,
                "training_start_date": training_start_date,
                "training_end_date": training_end_date
            })
            
            print(f"✅ Project created: {response['name']} (ID: {response['id']})")
            print(f"   Training period: {training_start_date} to {training_end_date}")
            
            return Project(self.http, response, interactive=self.interactive)
        except APIError as e:
            if e.status_code == 403:
                print("Not authorized")
                return None
            raise
    
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
             offset: int = 0,
             include_templates: bool = True) -> List[Project]:
        """
        List all projects (user's own + templates)
        
        Args:
            limit: Maximum number of projects to return
            offset: Number of projects to skip
            include_templates: Include template projects (default: True)
            
        Returns:
            List[Project]: List of project instances
        """
        # Call API to list projects
        response = self.http.get('/api/v1/projects', params={
            "limit": limit,
            "offset": offset,
            "include_templates": include_templates
        })
        
        # Handle the response format from backend: {"projects": [...], "total": ..., "limit": ..., "offset": ...}
        projects_data = response.get('projects', []) if isinstance(response, dict) else response
        return [Project(self.http, data, interactive=self.interactive) for data in projects_data]
