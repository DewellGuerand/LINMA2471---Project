from abc import ABC, abstractmethod
import numpy as np
from utils import simplex_projection


class PerformanceIndicator(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, w_new, w_old, model):
        pass

class ValuePerformanceIndicator(PerformanceIndicator):
    def __init__(self):
        pass

    def evaluate(self, w_new, w_old, model):
        return np.abs(model.f(w_new) - model.f(w_old))
    
class IteratePerformanceIndicator(PerformanceIndicator):
    def __init__(self):
        pass

    def evaluate(self, w_new, w_old, model):
        return np.linalg.norm(w_new - w_old)




class OptimizationMethod(ABC):
    def __init__(self, name, parameters, performance_indicator: PerformanceIndicator):
        self.name = name
        self.parameters = parameters
        self.performance_indicator = performance_indicator

    @abstractmethod
    def optimize(self, model, w0):
        """Run the optimization algorithm.
         Args:
             model (OptimizationModel): The optimization model.
             w0 (np.ndarray): Initial solution estimate.
         Returns:
             (dict): Optimization results including final solution, objective value, number of iterations, etc.
                   `Keys: {"sol": Solution, "value": Objective value, "iterations": Number of iterations, "converged": Convergence status}`
         """
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
    def __init__(self, parameters, performance_indicator: PerformanceIndicator):
        super().__init__("ProjectedGradient", parameters, performance_indicator)
        self.step_size = parameters.get("step_size", 0.01)
        self.max_iter = parameters.get("max_iter", 1000)
        self.tol = parameters.get("tol", 1e-6)

    def optimize(self, model, w0):
        w_new = w0.copy()
        for iter in range(self.max_iter):
            w_old = w_new.copy()
            w_new = self.iterate(model, w_old)

            # Check convergence
            if self.performance_indicator.evaluate(w_new, w_old, model) < self.tol:
                return {
                    "sol": w_new,
                    "value": model.f(w_new),
                    "iterations": iter + 1,
                    "converged": True,
                }

        return {
            "sol": w_new,
            "value": model.f(w_new),
            "iterations": self.max_iter,
            "converged": False,
        }

    def iterate(self, model, w):
        w_new = w - self.step_size * model.gradient(w)
        # Project onto feasible set (simplex)
        w_new = simplex_projection(w_new)
        return w_new

class ProjectedGradientDescentMomentum(OptimizationMethod):
    def __init__(self, parameters, performance_indicator: PerformanceIndicator):
        super().__init__("ProjectedGradientDescentMomentum", parameters, performance_indicator)
        self.step_size = parameters.get("step_size", 0.01)
        self.momentum = parameters.get("momentum", 0.9)
        self.max_iter = parameters.get("max_iter", 1000)
        self.tol = parameters.get("tol", 1e-6)
        self._velocity = None  # Store velocity for momentum

    def optimize(self, model, w0):
        w_new = w0.copy()
        self._velocity = np.zeros_like(w0)
        
        for iter in range(self.max_iter):
            w_old = w_new.copy()
            w_new = self.iterate(model, w_old)

            if self.performance_indicator.evaluate(w_new, w_old, model) < self.tol:
                return {
                    "sol": w_new,
                    "value": model.f(w_new),
                    "iterations": iter + 1,
                    "converged": True,
                }

        return {
            "sol": w_new,
            "value": model.f(w_new),
            "iterations": self.max_iter,
            "converged": False,
        }

    def iterate(self, model, w):
        # Update velocity with momentum
        self._velocity = self.momentum * self._velocity - self.step_size * model.gradient(w)
        w_new = w + self._velocity
        # Project onto feasible set (simplex)
        w_new = simplex_projection(w_new)
        return w_new


class ProjectedRandomizedCoordinateDescent(OptimizationMethod):
    def __init__(self, parameters, performance_indicator: PerformanceIndicator):
        super().__init__("ProjectedRandomizedCoordinateDescent", parameters, performance_indicator)
        self.step_size = parameters.get("step_size", 0.01)
        self.max_iter = parameters.get("max_iter", 1000)
        self.tol = parameters.get("tol", 1e-6)

    def optimize(self, model, w0):
        w_new = w0.copy()
        
        for iter in range(self.max_iter):
            w_old = w_new.copy()
            w_new = self.iterate(model, w_old)

            if self.performance_indicator.evaluate(w_new, w_old, model) < self.tol:
                return {
                    "sol": w_new,
                    "value": model.f(w_new),
                    "iterations": iter + 1,
                    "converged": True,
                }

        return {
            "sol": w_new,
            "value": model.f(w_new),
            "iterations": self.max_iter,
            "converged": False,
        }

    def iterate(self, model, w):
        n = w.shape[0]
        # Randomly select a coordinate
        coord_idx = np.random.randint(0, n)
        
        # Compute partial gradient for the selected coordinate only (O(n) instead of O(n^2))
        grad_i = model.gradient_coordinate(w, coord_idx)

        # Update only the selected coordinate
        w_new = w.copy()
        w_new[coord_idx] = w[coord_idx] - self.step_size * grad_i

        # Project onto feasible set (simplex)
        w_new = simplex_projection(w_new)
        return w_new


#### 
#### Methods for Non-Smooth Markowitz Model
####

class ProjectedSubgradientMethod(OptimizationMethod):
    def __init__(self, parameters, performance_indicator: PerformanceIndicator):
        super().__init__("ProjectedSubgradient", parameters, performance_indicator)
        self.step_size = parameters.get("step_size", 0.01)
        self.max_iter = parameters.get("max_iter", 1000)
        self.tol = parameters.get("tol", 1e-6)
        self.step_size_rule = parameters.get("step_size_rule", "constant")  # "constant", "diminishing"
        self._iter_count = 0  # Track iteration for diminishing step size

    def optimize(self, model, w0):
        w_new = w0.copy()
        self._iter_count = 0
        
        # Track best solution (subgradient methods are not descent methods)
        best_w = w_new.copy()
        best_value = model.f(w_new)
        
        for iter in range(self.max_iter):
            w_old = w_new.copy()
            self._iter_count = iter + 1
            w_new = self.iterate(model, w_old)
            
            # Update best solution
            current_value = model.f(w_new)
            if current_value < best_value:
                best_value = current_value
                best_w = w_new.copy()

            if self.performance_indicator.evaluate(w_new, w_old, model) < self.tol:
                return {
                    "sol": best_w,
                    "value": best_value,
                    "iterations": iter + 1,
                    "converged": True,
                }

        return {
            "sol": best_w,
            "value": best_value,
            "iterations": self.max_iter,
            "converged": False,
        }

    def _get_step_size(self):
        """Get step size based on the rule."""
        if self.step_size_rule == "diminishing":
            return self.step_size / np.sqrt(self._iter_count)
        return self.step_size

    def iterate(self, model, w):
        subgrad = model.subgradient(w)
        step = self._get_step_size()
        w_new = w - step * subgrad
        # Project onto feasible set (simplex)
        w_new = simplex_projection(w_new)
        return w_new

class ProximalGradientMethod(OptimizationMethod):
    def __init__(self, parameters, performance_indicator: PerformanceIndicator):
        super().__init__("ProximalGradient", parameters, performance_indicator)
        self.step_size = parameters.get("step_size", 0.01)
        self.max_iter = parameters.get("max_iter", 1000)
        self.tol = parameters.get("tol", 1e-6)

    def optimize(self, model, w0):
        w_new = w0.copy()
        
        for iter in range(self.max_iter):
            w_old = w_new.copy()
            w_new = self.iterate(model, w_old)

            if self.performance_indicator.evaluate(w_new, w_old, model) < self.tol:
                return {
                    "sol": w_new,
                    "value": model.f(w_new),
                    "iterations": iter + 1,
                    "converged": True,
                }

        return {
            "sol": w_new,
            "value": model.f(w_new),
            "iterations": self.max_iter,
            "converged": False,
        }

    def iterate(self, model, w):
        # Gradient step on smooth part
        grad_smooth = model.smooth_gradient(w)
        w_half = w - self.step_size * grad_smooth
        
        # Proximal operator for L1 penalty: soft thresholding
        # prox_{t * c * ||. - w_prev||_1}(x) = w_prev + soft_threshold(x - w_prev, t * c)
        threshold = self.step_size * model.c
        w_new = model.w_prev + self._soft_threshold(w_half - model.w_prev, threshold)
        
        # Project onto feasible set (simplex)
        w_new = simplex_projection(w_new)
        return w_new

    def _soft_threshold(self, x, threshold):
        """Soft thresholding operator (proximal operator for L1 norm)."""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


class InteriorPointMethod(OptimizationMethod):
    def __init__(self, parameters, performance_indicator: PerformanceIndicator):
        super().__init__("InteriorPoint", parameters, performance_indicator)
        self.mu = parameters.get("mu", 10)  # Initial barrier parameter
        self.mu_decay = parameters.get("mu_decay", 0.5)  # Decay factor for mu
        self.max_iter = parameters.get("max_iter", 1000)
        self.max_inner_iter = parameters.get("max_inner_iter", 50)
        self.tol = parameters.get("tol", 1e-6)
        self.inner_tol = parameters.get("inner_tol", 1e-8)

    def optimize(self, model, w0):
        n = w0.shape[0]
        w_new = w0.copy()
        
        # Ensure initial point is strictly feasible (interior of simplex)
        w_new = np.clip(w_new, 1e-8, 1 - 1e-8)
        w_new = w_new / np.sum(w_new)
        
        mu = self.mu
        
        for iter in range(self.max_iter):
            w_old = w_new.copy()
            
            # Solve barrier subproblem with current mu using Newton's method
            for inner_iter in range(self.max_inner_iter):
                w_prev_inner = w_new.copy()
                w_new = self._newton_step(model, w_new, mu, n)
                
                if np.linalg.norm(w_new - w_prev_inner) < self.inner_tol:
                    break
            
            # Decrease barrier parameter
            mu = mu * self.mu_decay
            
            if self.performance_indicator.evaluate(w_new, w_old, model) < self.tol:
                return {
                    "sol": w_new,
                    "value": model.f(w_new),
                    "iterations": iter + 1,
                    "converged": True,
                }

        return {
            "sol": w_new,
            "value": model.f(w_new),
            "iterations": self.max_iter,
            "converged": False,
        }

    def _newton_step(self, model, w, mu, n):
        """Perform one Newton step for the barrier subproblem."""
        # Barrier function: f(w) - mu * sum(log(w_i))
        # Gradient: grad_f - mu * (1/w)
        # Hessian: H_f + mu * diag(1/w^2)
        
        grad_f = model.gradient(w)
        H_f = model.hessian(w)
        
        # Barrier gradient and Hessian
        barrier_grad = grad_f - mu / w
        barrier_hess = H_f + mu * np.diag(1 / (w ** 2))
        
        # Solve Newton system with equality constraint sum(w) = 1
        # Using KKT system: [H, A^T; A, 0] [dw; nu] = [-grad; 0]
        A = np.ones((1, n))
        
        # Build KKT matrix
        KKT = np.zeros((n + 1, n + 1))
        KKT[:n, :n] = barrier_hess
        KKT[:n, n] = A.T.flatten()
        KKT[n, :n] = A.flatten()
        
        # RHS
        rhs = np.zeros(n + 1)
        rhs[:n] = -barrier_grad
        rhs[n] = 0  # Already on simplex
        
        # Solve
        try:
            sol = np.linalg.solve(KKT, rhs)
            dw = sol[:n]
        except np.linalg.LinAlgError:
            # Fallback to gradient step if Newton fails
            dw = -0.01 * barrier_grad
        
        # Line search to stay feasible (w > 0)
        alpha = 1.0
        while np.any(w + alpha * dw <= 0) and alpha > 1e-10:
            alpha *= 0.5
        
        w_new = w + 0.9 * alpha * dw  # 0.9 to stay strictly interior
        w_new = np.clip(w_new, 1e-10, None)
        w_new = w_new / np.sum(w_new)  # Normalize to simplex
        
        return w_new

    def iterate(self, model, w):
        # Single iteration is not well-defined for interior point
        # Use one Newton step with current mu
        n = w.shape[0]
        return self._newton_step(model, w, self.mu, n)