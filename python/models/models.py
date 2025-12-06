from abc import ABC, abstractmethod
import numpy as np


class OptimizationModel(ABC):
    def __init__(self, name, parameters):
        self.name = name
        self.parameters = parameters

    @abstractmethod
    def is_feasible(self, sol):
        pass

    @abstractmethod
    def f(self, w):
        pass

    @abstractmethod
    def is_feasible(self, sol):
        pass

    @abstractmethod
    def f(self, w):
        pass



class SmoothMarkowitzModel(OptimizationModel):
    def __init__(self, parameters):
        super().__init__("SmoothMarkowitz", parameters)

        self.sigma = parameters.get("sigma")
        self.mu = parameters.get("mu")
        self.lam = parameters.get("lam")


    def is_feasible(self, sol):
        # Implement feasibility check for Smooth Markowitz model
        pass

    def f(self, w):
        return 0.5 * w.T @ self.sigma @ w - self.lam * self.mu.T @ w
    
    def gradient(self, w):
        return self.sigma @ w - self.lam * self.mu

    def hessian(self, w):
        return self.sigma
    

class NonSmoothMarkowitzModel(OptimizationModel):
    def __init__(self, parameters):
        super().__init__("NonSmoothMarkowitz", parameters)

        self.sigma = parameters.get("sigma")
        self.mu = parameters.get("mu")
        self.lam = parameters.get("lam")
        self.w_prev = parameters.get("w_prev")
        self.c = parameters.get("c")


    def is_feasible(self, sol):
        # Implement feasibility check for Non-Smooth Markowitz model
        pass

    def f(self, w):
        return 0.5 * w.T @ self.sigma @ w - self.lam * w.T @ self.mu + self.c * np.linalg.norm(w - self.w_prev, 1)