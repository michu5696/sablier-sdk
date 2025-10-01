"""
Utility functions for Model class
"""

from typing import List, Dict, Optional, Any


def update_feature_types(http_client, model_id: str, input_features: List[Dict], conditioning_features: List[str], target_features: List[str]):
    """
    Update input_features with type assignment
    
    Args:
        http_client: HTTP client for API calls
        model_id: Model ID
        input_features: Current input features
        conditioning_features: Conditioning feature names
        target_features: Target feature names
        
    Returns:
        Updated features list
    """
    updated_features = []
    for feature in input_features:
        feature_name = feature.get('display_name', feature.get('name'))
        feature_copy = feature.copy()
        
        if feature_name in target_features:
            feature_copy['type'] = 'target'
        elif feature_name in conditioning_features:
            feature_copy['type'] = 'conditioning'
        
        updated_features.append(feature_copy)
    
    # Update via API
    http_client.patch(f'/api/v1/models/{model_id}', {
        'input_features': updated_features
    })
    
    return updated_features
