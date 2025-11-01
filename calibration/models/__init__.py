"""
Models module with optimized implementations.
"""

from .registry import ModelRegistry, ModelFactory
from .linear_models import PLSRModel, RidgeModel, LassoModel, ElasticNetModel
from .neural_models import MLPModel
from .svm_models import SVRModel
from .ensemble_models import RandomForestModel, XGBoostModel

# Register default models
def register_default_models():
    """Register all default models with the global registry."""
    registry = ModelRegistry()
    
    # Linear models (pipeline-based, no data leakage)
    registry.register('plsr', PLSRModel)
    registry.register('ridge', RidgeModel)
    registry.register('lasso', LassoModel)
    registry.register('elastic_net', ElasticNetModel)
    
    # Neural models (pipeline-based, no data leakage)
    registry.register('mlp', MLPModel)
    
    # SVM models (pipeline-based, no data leakage)
    registry.register('svr', SVRModel)
    
    # Ensemble models (don't need scaling)
    registry.register('random_forest', RandomForestModel)
    registry.register('xgboost', XGBoostModel)
    
    return registry

# Auto-register on import
_default_registry = register_default_models()

__all__ = [
    'ModelRegistry',
    'ModelFactory',
    'PLSRModel',
    'RidgeModel',
    'LassoModel',
    'ElasticNetModel',
    'MLPModel',
    'SVRModel',
    'RandomForestModel',
    'XGBoostModel',
]
