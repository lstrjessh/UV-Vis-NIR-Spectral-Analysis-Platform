"""
Hyperparameter optimization utilities.
"""

from typing import Dict, Any, Callable, Optional, List, Union, Tuple
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')


class BaseOptimizer(ABC):
    """Base class for optimizers."""
    
    @abstractmethod
    def optimize(self, 
                objective: Callable,
                search_space: Optional[Dict[str, Any]] = None,
                direction: str = 'minimize',
                **kwargs) -> Dict[str, Any]:
        """Optimize objective function."""
        pass
    
    @abstractmethod
    def get_history(self) -> pd.DataFrame:
        """Get optimization history."""
        pass


class OptunaOptimizer(BaseOptimizer):
    """Optuna-based Bayesian optimization."""
    
    def __init__(self,
                 n_trials: int = 50,
                 timeout: Optional[float] = None,
                 random_state: int = 42):
        """
        Initialize Optuna optimizer.
        
        Args:
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            random_state: Random seed
        """
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_state = random_state
        self.study = None
        self._history = []
        
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            self.optuna = optuna
        except ImportError:
            raise ImportError("Optuna is required for Bayesian optimization. Install with: pip install optuna")
    
    def optimize(self,
                objective: Callable,
                search_space: Optional[Dict[str, Any]] = None,
                direction: str = 'minimize',
                **kwargs) -> Dict[str, Any]:
        """
        Optimize using Optuna.
        
        Args:
            objective: Objective function that takes trial object
            search_space: Not used (defined in objective)
            direction: 'minimize' or 'maximize'
            
        Returns:
            Best parameters
        """
        # Create study
        self.study = self.optuna.create_study(
            direction=direction,
            sampler=self.optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # Optimize
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False
        )
        
        # Store history
        self._history = []
        for trial in self.study.trials:
            if trial.state == self.optuna.trial.TrialState.COMPLETE:
                self._history.append({
                    'trial': trial.number,
                    'value': trial.value,
                    **trial.params
                })
        
        return self.study.best_params
    
    def get_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame."""
        if not self._history:
            return pd.DataFrame()
        return pd.DataFrame(self._history)
    
    def get_importance(self) -> Dict[str, float]:
        """Get parameter importance."""
        if self.study is None:
            return {}
        
        try:
            importance = self.optuna.importance.get_param_importances(self.study)
            return importance
        except:
            return {}


class GridSearchOptimizer(BaseOptimizer):
    """Grid search optimization."""
    
    def __init__(self):
        """Initialize grid search optimizer."""
        self._history = []
    
    def optimize(self,
                objective: Callable,
                search_space: Dict[str, List[Any]],
                direction: str = 'minimize',
                **kwargs) -> Dict[str, Any]:
        """
        Perform grid search.
        
        Args:
            objective: Objective function that takes parameter dict
            search_space: Dictionary of parameter_name -> list of values
            direction: 'minimize' or 'maximize'
            
        Returns:
            Best parameters
        """
        from itertools import product
        
        # Generate all combinations
        param_names = list(search_space.keys())
        param_values = [search_space[name] for name in param_names]
        
        best_score = float('inf') if direction == 'minimize' else float('-inf')
        best_params = {}
        
        self._history = []
        
        # Evaluate all combinations
        for values in product(*param_values):
            params = dict(zip(param_names, values))
            
            try:
                score = objective(params)
                
                self._history.append({
                    'value': score,
                    **params
                })
                
                # Update best
                if direction == 'minimize':
                    if score < best_score:
                        best_score = score
                        best_params = params.copy()
                else:
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
            except Exception:
                continue
        
        return best_params
    
    def get_history(self) -> pd.DataFrame:
        """Get optimization history."""
        if not self._history:
            return pd.DataFrame()
        return pd.DataFrame(self._history)


class RandomSearchOptimizer(BaseOptimizer):
    """Random search optimization."""
    
    def __init__(self,
                 n_trials: int = 50,
                 random_state: int = 42):
        """
        Initialize random search optimizer.
        
        Args:
            n_trials: Number of random trials
            random_state: Random seed
        """
        self.n_trials = n_trials
        self.random_state = random_state
        self._history = []
        np.random.seed(random_state)
    
    def optimize(self,
                objective: Callable,
                search_space: Dict[str, Union[Tuple[float, float], List[Any]]],
                direction: str = 'minimize',
                **kwargs) -> Dict[str, Any]:
        """
        Perform random search.
        
        Args:
            objective: Objective function that takes parameter dict
            search_space: Dictionary of parameter_name -> (min, max) or list of values
            direction: 'minimize' or 'maximize'
            
        Returns:
            Best parameters
        """
        best_score = float('inf') if direction == 'minimize' else float('-inf')
        best_params = {}
        
        self._history = []
        
        for _ in range(self.n_trials):
            # Sample parameters
            params = {}
            for name, space in search_space.items():
                if isinstance(space, tuple) and len(space) == 2:
                    # Continuous parameter
                    if isinstance(space[0], (int, np.integer)):
                        params[name] = np.random.randint(space[0], space[1] + 1)
                    else:
                        params[name] = np.random.uniform(space[0], space[1])
                elif isinstance(space, list):
                    # Discrete parameter
                    params[name] = np.random.choice(space)
                else:
                    raise ValueError(f"Invalid search space for {name}")
            
            try:
                score = objective(params)
                
                self._history.append({
                    'value': score,
                    **params
                })
                
                # Update best
                if direction == 'minimize':
                    if score < best_score:
                        best_score = score
                        best_params = params.copy()
                else:
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
            except Exception:
                continue
        
        return best_params
    
    def get_history(self) -> pd.DataFrame:
        """Get optimization history."""
        if not self._history:
            return pd.DataFrame()
        return pd.DataFrame(self._history)


class BayesianOptimizer:
    """Simplified Bayesian optimization without external dependencies."""
    
    def __init__(self,
                 n_trials: int = 50,
                 random_state: int = 42):
        """
        Initialize Bayesian optimizer.
        
        Args:
            n_trials: Number of optimization trials
            random_state: Random seed
        """
        self.n_trials = n_trials
        self.random_state = random_state
        self._history = []
        np.random.seed(random_state)
        
        # Try to use scikit-optimize if available
        self.use_skopt = False
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            self.skopt = {'gp_minimize': gp_minimize, 'Real': Real, 'Integer': Integer, 'Categorical': Categorical}
            self.use_skopt = True
        except ImportError:
            pass
    
    def optimize(self,
                objective: Callable,
                search_space: Dict[str, Any],
                direction: str = 'minimize',
                **kwargs) -> Dict[str, Any]:
        """
        Perform Bayesian optimization.
        
        Args:
            objective: Objective function
            search_space: Search space definition
            direction: 'minimize' or 'maximize'
            
        Returns:
            Best parameters
        """
        if self.use_skopt:
            return self._optimize_skopt(objective, search_space, direction)
        else:
            # Fallback to random search
            optimizer = RandomSearchOptimizer(self.n_trials, self.random_state)
            return optimizer.optimize(objective, search_space, direction)
    
    def _optimize_skopt(self,
                       objective: Callable,
                       search_space: Dict[str, Any],
                       direction: str) -> Dict[str, Any]:
        """Optimize using scikit-optimize."""
        # Convert search space
        dimensions = []
        param_names = []
        
        for name, space in search_space.items():
            param_names.append(name)
            
            if isinstance(space, tuple) and len(space) == 2:
                if isinstance(space[0], (int, np.integer)):
                    dimensions.append(self.skopt['Integer'](space[0], space[1], name=name))
                else:
                    dimensions.append(self.skopt['Real'](space[0], space[1], name=name))
            elif isinstance(space, list):
                dimensions.append(self.skopt['Categorical'](space, name=name))
        
        # Define objective for skopt
        def skopt_objective(values):
            params = dict(zip(param_names, values))
            score = objective(params)
            return -score if direction == 'maximize' else score
        
        # Optimize
        result = self.skopt['gp_minimize'](
            skopt_objective,
            dimensions,
            n_calls=self.n_trials,
            random_state=self.random_state,
            acq_func='EI'
        )
        
        # Extract best parameters
        best_params = dict(zip(param_names, result.x))
        
        return best_params
    
    def get_history(self) -> pd.DataFrame:
        """Get optimization history."""
        return pd.DataFrame(self._history)
