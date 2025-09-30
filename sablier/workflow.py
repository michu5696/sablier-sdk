"""
Workflow validation and conflict detection for Sablier SDK
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class WorkflowConflict:
    """Represents a workflow conflict"""
    operation: str
    current_status: str
    items_to_delete: List[str]
    warning_message: str
    
    def format_warning(self) -> str:
        """Format a user-friendly warning message"""
        msg = f"⚠️  WARNING: This model is in '{self.current_status}' status.\n\n"
        msg += f"Operation '{self.operation}' will DELETE:\n"
        
        # Map technical names to user-friendly names
        friendly_names = {
            "training_data": "Training data rows",
            "samples": "Generated samples",
            "encoding_models": "Encoding models (PCA-ICA)",
            "trained_model": "Trained QRF model (in storage)",
            "feature_importance": "Feature importance data"
        }
        
        for item in self.items_to_delete:
            display = friendly_names.get(item, item)
            msg += f"  - {display}\n"
        
        msg += "\nThis action cannot be undone."
        return msg


class WorkflowValidator:
    """
    Validates workflow operations and detects conflicts
    
    Workflow stages (in order):
    1. created -> model exists
    2. data_collected -> training_data fetched
    3. samples_generated -> samples created
    4. samples_encoded -> encoding done
    5. trained -> QRF model trained
    """
    
    # Define workflow rules
    WORKFLOW_RULES = {
        "add_features": {
            "conflicts_with": ["data_collected", "samples_generated", "samples_encoded", "trained"],
            "deletes": ["training_data", "samples", "encoding_models", "trained_model", "feature_importance"],
            "message": "Adding/removing features requires re-fetching data and regenerating all dependent artifacts."
        },
        "set_training_period": {
            "conflicts_with": ["data_collected", "samples_generated", "samples_encoded", "trained"],
            "deletes": ["training_data", "samples", "encoding_models", "trained_model", "feature_importance"],
            "message": "Changing the training period requires re-fetching data."
        },
        "fetch_data": {
            "conflicts_with": ["samples_generated", "samples_encoded", "trained"],
            "deletes": ["samples", "encoding_models", "trained_model", "feature_importance"],
            "message": "Re-fetching data will invalidate existing samples and models."
        },
        "generate_samples": {
            "conflicts_with": ["samples_encoded", "trained"],
            "deletes": ["encoding_models", "trained_model", "feature_importance"],
            "message": "Regenerating samples will invalidate encoding and training."
        },
        "encode_samples": {
            "conflicts_with": ["trained"],
            "deletes": ["trained_model", "feature_importance"],
            "message": "Re-encoding samples will invalidate the trained model."
        }
    }
    
    @classmethod
    def check_conflict(cls, operation: str, current_status: str) -> Optional[WorkflowConflict]:
        """
        Check if an operation will cause a workflow conflict
        
        Args:
            operation: Operation name (e.g., "add_features", "generate_samples")
            current_status: Current model status
            
        Returns:
            WorkflowConflict if conflict exists, None otherwise
        """
        rule = cls.WORKFLOW_RULES.get(operation)
        
        if not rule:
            # Operation not governed by workflow rules
            return None
        
        if current_status not in rule["conflicts_with"]:
            # No conflict - model hasn't progressed to conflicting stage
            return None
        
        # Conflict detected
        return WorkflowConflict(
            operation=operation,
            current_status=current_status,
            items_to_delete=rule["deletes"],
            warning_message=rule["message"]
        )
    
    @classmethod
    def get_expected_status_for_operation(cls, operation: str) -> Optional[str]:
        """Get the expected status after an operation completes"""
        status_map = {
            "fetch_data": "data_collected",
            "generate_samples": "samples_generated",
            "encode_samples": "samples_encoded",
            "train_model": "trained"
        }
        return status_map.get(operation)
