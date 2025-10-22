"""
Models module with optimized implementations.
"""

from .registry import ModelRegistry, ModelFactory
from .linear_models import PLSRModel, RidgeModel, LassoModel
from .ensemble_models import RandomForestModel, XGBoostModel, GradientBoostingModel
from .neural_models import MLPModel, CNN1DModel
from .svm_models import SVRModel

# Register default models
def register_default_models():
    """Register all default models with the global registry."""
    registry = ModelRegistry()
    
    # Linear models
    registry.register('plsr', PLSRModel)
    registry.register('ridge', RidgeModel)
    registry.register('lasso', LassoModel)
    
    # Ensemble models
    registry.register('random_forest', RandomForestModel)
    registry.register('xgboost', XGBoostModel)
    registry.register('gradient_boosting', GradientBoostingModel)
    
    # Neural models
    registry.register('mlp', MLPModel)
    registry.register('cnn1d', CNN1DModel)
    
    # SVM models
    registry.register('svr', SVRModel)
    
    return registry

# Auto-register on import
_default_registry = register_default_models()

__all__ = [
    'ModelRegistry',
    'ModelFactory',
    'PLSRModel',
    'RidgeModel',
    'LassoModel',
    'RandomForestModel',
    'XGBoostModel',
    'GradientBoostingModel',
    'MLPModel',
    'CNN1DModel',
    'SVRModel'
]
