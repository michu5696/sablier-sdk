"""Training results container"""


class TrainingResults:
    """Results from model training"""
    
    def __init__(self, results_data: dict):
        self.data = results_data
        self.oob_score = results_data.get('oob_score', 0.0)
        self.feature_importance = results_data.get('feature_importance', {})
    
    def __repr__(self) -> str:
        return f"TrainingResults(oob_score={self.oob_score:.4f})"
