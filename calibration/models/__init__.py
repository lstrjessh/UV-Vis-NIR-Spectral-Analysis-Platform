"""
Models module with optimized implementations.
"""

from .registry import ModelRegistry, ModelFactory
from .pipeline_models import PipelinePLSRModel, PipelineRidgeModel, PipelineLassoModel, PipelineElasticNetModel
from .ensemble_models import RandomForestModel
from .neural_models import MLPModel
from .svm_models import SVRModel

# Register default models
def register_default_models():
    """Register all default models with the global registry."""
    registry = ModelRegistry()
    
    # Pipeline models (no data leakage)
    registry.register('plsr', PipelinePLSRModel)
    registry.register('ridge', PipelineRidgeModel)
    registry.register('lasso', PipelineLassoModel)
    registry.register('elastic_net', PipelineElasticNetModel)
    
    # Ensemble models
    registry.register('random_forest', RandomForestModel)
    
    # Neural models
    registry.register('mlp', MLPModel)
    
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
    'ElasticNetModel',
    'MLPModel',
    'SVRModel'
]
