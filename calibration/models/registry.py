"""
Model registry and factory for dynamic model management.
"""

from typing import Type, Dict, List, Optional, Any
from ..core.base_model import BaseModel, ModelConfig
import inspect


class ModelRegistry:
    """Registry for managing available models."""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern for global registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
            cls._instance._metadata = {}
        return cls._instance
    
    def register(self, 
                 name: str, 
                 model_class: Type[BaseModel],
                 description: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 **metadata):
        """
        Register a model class.
        
        Args:
            name: Unique model identifier
            model_class: Model class (must inherit from BaseModel)
            description: Model description
            tags: Tags for categorization
            **metadata: Additional metadata
        """
        if not issubclass(model_class, BaseModel):
            raise TypeError(f"{model_class} must inherit from BaseModel")
        
        self._models[name] = model_class
        
        # Extract and store metadata
        self._metadata[name] = {
            'class': model_class.__name__,
            'description': description or model_class.__doc__,
            'tags': tags or [],
            'module': model_class.__module__,
            **metadata
        }
    
    def unregister(self, name: str):
        """Remove a model from registry."""
        if name in self._models:
            del self._models[name]
            del self._metadata[name]
    
    def get_model_class(self, name: str) -> Type[BaseModel]:
        """Get model class by name."""
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found in registry")
        return self._models[name]
    
    def list_models(self, tag: Optional[str] = None) -> List[str]:
        """
        List available models.
        
        Args:
            tag: Filter by tag
            
        Returns:
            List of model names
        """
        if tag is None:
            return list(self._models.keys())
        
        return [name for name, meta in self._metadata.items() 
                if tag in meta.get('tags', [])]
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get model metadata."""
        return self._metadata.get(name, {})
    
    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all models."""
        return self._metadata.copy()
    
    def has_model(self, name: str) -> bool:
        """Check if model is registered."""
        return name in self._models
    
    def clear(self):
        """Clear all registered models."""
        self._models.clear()
        self._metadata.clear()


class ModelFactory:
    """Factory for creating model instances."""
    
    def __init__(self, registry: Optional[ModelRegistry] = None):
        """
        Initialize factory.
        
        Args:
            registry: Model registry (uses global if None)
        """
        self.registry = registry or ModelRegistry()
    
    def create(self, 
               model_name: str,
               config: Optional[ModelConfig] = None,
               **config_kwargs) -> BaseModel:
        """
        Create a model instance.
        
        Args:
            model_name: Name of registered model
            config: Model configuration
            **config_kwargs: Configuration parameters if config not provided
            
        Returns:
            Model instance
        """
        # Get model class
        model_class = self.registry.get_model_class(model_name)
        
        # Create or update configuration
        if config is None:
            config = ModelConfig(name=model_name, **config_kwargs)
        elif config_kwargs:
            # Update existing config with kwargs
            for key, value in config_kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Instantiate model
        return model_class(config)
    
    def create_multiple(self, 
                       model_names: List[str],
                       config: Optional[ModelConfig] = None,
                       **config_kwargs) -> Dict[str, BaseModel]:
        """
        Create multiple model instances.
        
        Args:
            model_names: List of model names
            config: Shared configuration
            **config_kwargs: Shared configuration parameters
            
        Returns:
            Dictionary of model_name -> model_instance
        """
        models = {}
        
        for name in model_names:
            # Create individual config for each model
            model_config = ModelConfig(name=name, **config_kwargs) if config is None else config
            models[name] = self.create(name, model_config)
        
        return models
    
    def create_ensemble(self,
                       model_specs: Dict[str, Dict[str, Any]]) -> Dict[str, BaseModel]:
        """
        Create ensemble of models with different configurations.
        
        Args:
            model_specs: Dictionary of model_name -> config_dict
            
        Returns:
            Dictionary of model_name -> model_instance
        """
        models = {}
        
        for name, config_dict in model_specs.items():
            config = ModelConfig(name=name, **config_dict)
            models[name] = self.create(name, config)
        
        return models
    
    def available_models(self, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get information about available models.
        
        Args:
            tag: Filter by tag
            
        Returns:
            List of model information dictionaries
        """
        model_names = self.registry.list_models(tag)
        
        models_info = []
        for name in model_names:
            metadata = self.registry.get_metadata(name)
            models_info.append({
                'name': name,
                'class': metadata.get('class', 'Unknown'),
                'description': metadata.get('description', ''),
                'tags': metadata.get('tags', []),
                'available': self._check_availability(name)
            })
        
        return models_info
    
    def _check_availability(self, model_name: str) -> bool:
        """Check if model dependencies are available."""
        try:
            model_class = self.registry.get_model_class(model_name)
            
            # Check for specific dependencies
            if 'cnn' in model_name.lower():
                try:
                    import torch
                    return True
                except ImportError:
                    return False
            
            # Check for sklearn dependencies (all our models use sklearn)
            try:
                import sklearn
                return True
            except ImportError:
                return False
            
        except:
            return False
