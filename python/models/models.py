from abc import ABC, abstractmethod
import numpy as np


class OptimizationModel(ABC):
    def __init__(self, name, parameters):
        self.name = name
        self.parameters = parameters

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

    def f(self, w):
        return 0.5 * w.T @ self.sigma @ w - self.lam * self.mu.T @ w
    
    def gradient(self, w):
        return self.sigma @ w - self.lam * self.mu

    def gradient_coordinate(self, w, i):
        """Compute the i-th coordinate of the gradient.
        
        grad_i = (Sigma @ w)_i - lambda * mu_i = Sigma[i, :] @ w - lambda * mu[i]
        
        This is O(n) instead of O(n^2) for the full gradient.
        """
        return self.sigma[i, :] @ w - self.lam * self.mu[i]

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


    def f(self, w):
        return 0.5 * w.T @ self.sigma @ w - self.lam * w.T @ self.mu + self.c * np.linalg.norm(w - self.w_prev, 1)

    def smooth_gradient(self, w):
        """Gradient of the smooth part: 0.5 * w^T @ Sigma @ w - lambda * mu^T @ w"""
        return self.sigma @ w - self.lam * self.mu

    def subgradient(self, w):
        """Compute a subgradient of the full objective.
        
        The subgradient of ||w - w_prev||_1 is sign(w - w_prev),
        with any value in [-1, 1] when w_i = w_prev_i.
        """
        smooth_grad = self.smooth_gradient(w)
        
        # Subgradient of L1 norm: sign(w - w_prev)
        diff = w - self.w_prev
        l1_subgrad = np.sign(diff)
        # Handle the case where diff = 0 (choose 0 as subgradient element)
        l1_subgrad[diff == 0] = 0
        
        return smooth_grad + self.c * l1_subgrad

    def gradient(self, w):
        """Alias for smooth_gradient for compatibility with some methods."""
        return self.smooth_gradient(w)

    def hessian(self, w):
        """Hessian of the smooth part (same as SmoothMarkowitzModel)."""
        return self.sigma