from abc import ABC, abstractmethod
import numpy as np


class OptimizationMethod(ABC):
    def __init__(self, name, parameters):
        self.name = name
        self.parameters = parameters

    @abstractmethod
    def optimize(self, model, w0):
        pass

    @abstractmethod
    def iterate(self, model, w):
        """Perform a single iteration of the optimization algorithm.
         Args:
             model (OptimizationModel): The optimization model.
             w (np.ndarray): Current solution estimate.
         Returns:
             np.ndarray: Updated solution estimate after one iteration.
         """
        pass


#### 
#### Methods for Smooth Markowitz Model
####


class ProjectedGradientMethod(OptimizationMethod):
    def __init__(self, parameters):
        super().__init__("ProjectedGradient", parameters)
        self.step_size = parameters.get("step_size", 0.01)
        self.max_iter = parameters.get("max_iter", 1000)
        self.tol = parameters.get("tol", 1e-6)

    def optimize(self, model, w0):
        # TODO
        pass

    def iterate(self, model, w):
        # TODO
        pass

class ProjectedGradientDescentMomentum(OptimizationMethod):
    def __init__(self, parameters):
        super().__init__("ProjectedGradientDescentMomentum", parameters)
        self.step_size = parameters.get("step_size", 0.01)
        self.momentum = parameters.get("momentum", 0.9)
        self.max_iter = parameters.get("max_iter", 1000)
        self.tol = parameters.get("tol", 1e-6)

    def optimize(self, model, w0):
        # TODO
        pass

    def iterate(self, model, w):
        # TODO
        pass
