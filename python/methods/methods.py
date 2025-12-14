from abc import ABC, abstractmethod
import numpy as np
from utils import simplex_projection
import time 


####
#### Step Size Strategies
####

class StepSizeStrategy(ABC):
    """Abstract base class for step size strategies."""
    @abstractmethod
    def get_step_size(self, model, w, grad, iteration, **kwargs):
        """Compute the step size for the current iteration."""
        pass
    
    def reset(self):
        """Reset any internal state (called at the start of optimization)."""
        pass


class ConstantStepSize(StepSizeStrategy):
    """Constant step size: α = 1/L or user-specified value."""
    def __init__(self, step_size=None):
        self.step_size = step_size
    
    def get_step_size(self, model, w, grad, iteration, **kwargs):
        if self.step_size is not None:
            return self.step_size
        # Default: 1/L where L is the Lipschitz constant
        return 1.0 / model.lipschitz_constant()


class BacktrackingLineSearch(StepSizeStrategy):
    """
    Armijo backtracking line search.
    
    Find α such that: f(x - α∇f(x)) ≤ f(x) - c·α·‖∇f(x)‖²
    """
    def __init__(self, alpha_init=1.0, c=1e-4, rho=0.5, max_iter=50):
        self.alpha_init = alpha_init  # Initial step size
        self.c = c                     # Armijo parameter (typically 1e-4)
        self.rho = rho                 # Reduction factor (typically 0.5)
        self.max_iter = max_iter       # Max backtracking iterations
    
    def get_step_size(self, model, w, grad, iteration, **kwargs):
        alpha = self.alpha_init
        f_w = model.f(w)
        grad_norm_sq = np.dot(grad, grad)
        
        for _ in range(self.max_iter):
            w_new = simplex_projection(w - alpha * grad)
            if model.f(w_new) <= f_w - self.c * alpha * grad_norm_sq:
                return alpha
            alpha *= self.rho
        
        return alpha  # Return smallest tried step size


class BarzilaiBorweinStepSize(StepSizeStrategy):
    """
    Barzilai-Borwein step size (spectral gradient method).
    
    Two variants:
    - BB1: α_k = (s_{k-1}^T s_{k-1}) / (s_{k-1}^T y_{k-1})
    
    where s_{k-1} = x_k - x_{k-1} and y_{k-1} = ∇f(x_k) - ∇f(x_{k-1})
    """
    def __init__(self, variant='BB1', alpha_min=1e-10, alpha_max=1e10, alpha_init=1.0):
        self.variant = variant
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_init = alpha_init
        self.w_prev = None
        self.grad_prev = None
    
    def reset(self):
        self.w_prev = None
        self.grad_prev = None
    
    def get_step_size(self, model, w, grad, iteration, **kwargs):
        if self.w_prev is None or self.grad_prev is None:
            # First iteration: use initial step size
            self.w_prev = w.copy()
            self.grad_prev = grad.copy()
            return self.alpha_init
        
        s = w - self.w_prev      # x_k - x_{k-1}
        y = grad - self.grad_prev  # ∇f(x_k) - ∇f(x_{k-1})
        
        s_dot_y = np.dot(s, y)
        

        s_dot_s = np.dot(s, s)
        if abs(s_dot_y) < 1e-14:
            alpha = self.alpha_init
        else:
            alpha = s_dot_s / s_dot_y
        
        # Clip to reasonable range
        alpha = np.clip(alpha, self.alpha_min, self.alpha_max)
        
        # Update history
        self.w_prev = w.copy()
        self.grad_prev = grad.copy()
        
        return alpha


class ExactLineSearchQuadratic(StepSizeStrategy):
    """
    Exact line search for quadratic functions.
    
    For f(w) = 0.5 * w^T Σ w - λ μ^T w:
    α_k = (∇f(x_k)^T ∇f(x_k)) / (∇f(x_k)^T Σ ∇f(x_k))
    
    Note: This is exact for unconstrained problems. With projection,
    it's an approximation but often works well.
    """
    def __init__(self):
        pass
    
    def get_step_size(self, model, w, grad, iteration, **kwargs):
        grad_dot_grad = np.dot(grad, grad)
        if grad_dot_grad < 1e-14:
            return 1.0 / model.lipschitz_constant()
        
        Sigma_grad = model.sigma @ grad
        grad_dot_Sigma_grad = np.dot(grad, Sigma_grad)
        
        if grad_dot_Sigma_grad < 1e-14:
            return 1.0 / model.lipschitz_constant()
        
        return grad_dot_grad / grad_dot_Sigma_grad


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
    
class ValuePerformanceIndicator_with_ref(PerformanceIndicator):
    def __init__(self, f_ref):
        self.f_ref = f_ref

    def evaluate(self, w_new, w_old, model):
        return np.abs(model.f(w_new) - self.f_ref)
    





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
    """
    Projected Gradient Descent with configurable step size strategy.
    
    Parameters:
        step_size: float or StepSizeStrategy - Either a constant value or a strategy object
        max_iter: int - Maximum number of iterations
        tol: float - Convergence tolerance
    """
    def __init__(self, parameters, performance_indicator: PerformanceIndicator):
        super().__init__("ProjectedGradient", parameters, performance_indicator)
        step_size_param = parameters.get("step_size", 0.01)
        # Support both constant step size and strategy objects
        if isinstance(step_size_param, StepSizeStrategy):
            self.step_size_strategy = step_size_param
        else:
            self.step_size_strategy = ConstantStepSize(step_size_param)
        self.max_iter = parameters.get("max_iter", 1000)
        self.tol = parameters.get("tol", 1e-6)
        self.metric = []
        self.time = []
        self.obj_value = []
        self.step_sizes = []  # Track step sizes used

    def optimize(self, model, w0):
        # Reset history for each optimization run
        self.metric = []
        self.time = []
        self.obj_value = []
        self.step_sizes = []
        self.step_size_strategy.reset()
        
        w_new = w0.copy()
        self.obj_value.append(model.f(w_new))
        
        t_start = time.time()
        for iter in range(self.max_iter):
            w_old = w_new.copy()
            grad = model.gradient(w_old)
            
            # Get step size from strategy
            step = self.step_size_strategy.get_step_size(model, w_old, grad, iter)
            self.step_sizes.append(step)
            
            # Gradient step + projection
            w_new = simplex_projection(w_old - step * grad)
            self.obj_value.append(model.f(w_new))
            
            # Record cumulative time since start
            self.time.append(time.time() - t_start)

            convergence_value = self.performance_indicator.evaluate(w_new, w_old, model)
            self.metric.append(convergence_value)
            # Check convergence
            if convergence_value < self.tol:
                return {
                    "sol": w_new,
                    "value": model.f(w_new),
                    "iterations": iter + 1,
                    "converged": True,
                    "metric": self.metric,
                    "time": self.time,
                    "obj_value": self.obj_value,
                    "step_sizes": self.step_sizes,
                }

        return {
            "sol": w_new,
            "value": model.f(w_new),
            "iterations": self.max_iter,
            "converged": False,
            "metric": self.metric,
            "time": self.time,
            "obj_value": self.obj_value,
            "step_sizes": self.step_sizes,
        }

    def iterate(self, model, w):
        grad = model.gradient(w)
        step = self.step_size_strategy.get_step_size(model, w, grad, 0)
        w_new = w - step * grad
        # Project onto feasible set (simplex)
        w_new = simplex_projection(w_new)
        return w_new

class ProjectedGradientDescentMomentum(OptimizationMethod):
    """
    Projected Gradient Descent with Momentum and configurable step size strategy.
    
    Parameters:
        step_size: float or StepSizeStrategy - Either a constant value or a strategy object
        momentum: float - Momentum coefficient (default 0.9)
        max_iter: int - Maximum number of iterations
        tol: float - Convergence tolerance
    """
    def __init__(self, parameters, performance_indicator: PerformanceIndicator):
        super().__init__("ProjectedGradientDescentMomentum", parameters, performance_indicator)
        step_size_param = parameters.get("step_size", 0.01)
        # Support both constant step size and strategy objects
        if isinstance(step_size_param, StepSizeStrategy):
            self.step_size_strategy = step_size_param
        else:
            self.step_size_strategy = ConstantStepSize(step_size_param)
        self.momentum = parameters.get("momentum", 0.9)
        self.max_iter = parameters.get("max_iter", 1000)
        self.tol = parameters.get("tol", 1e-6)
        self._velocity = None  # Store velocity for momentum
        self.metric = []
        self.time = []
        self.obj_value = []
        self.step_sizes = []

    def optimize(self, model, w0):
        # Reset history for each optimization run
        self.metric = []
        self.time = []
        self.obj_value = []
        self.step_sizes = []
        self.step_size_strategy.reset()
        
        w_new = w0.copy()
        self._velocity = np.zeros_like(w0)
        self.obj_value.append(model.f(w_new))
        
        t_start = time.time()
        for iter in range(self.max_iter):
            w_old = w_new.copy()
            w_new = self.iterate(model, w_old)
            self.obj_value.append(model.f(w_new))
            
            # Record cumulative time since start
            self.time.append(time.time() - t_start)

            convergence_value = self.performance_indicator.evaluate(w_new, w_old, model)
            self.metric.append(convergence_value)
            
            if convergence_value < self.tol:
                return {
                    "sol": w_new,
                    "value": model.f(w_new),
                    "iterations": iter + 1,
                    "converged": True,
                    "metric": self.metric,
                    "time": self.time,
                    "obj_value": self.obj_value,
                    "step_sizes": self.step_sizes,
                }

        return {
            "sol": w_new,
            "value": model.f(w_new),
            "iterations": self.max_iter,
            "converged": False,
            "metric": self.metric,
            "time": self.time,
            "obj_value": self.obj_value,
            "step_sizes": self.step_sizes,
        }
    def iterate(self, model, w):
        grad = model.gradient(w)
        step = self.step_size_strategy.get_step_size(model, w, grad, 0)
        # Update velocity with momentum
        self._velocity = self.momentum * self._velocity + (1 - self.momentum) * grad
        w_new = w - step * self._velocity
        # Project onto feasible set (simplex)
        w_new = simplex_projection(w_new)
        return w_new

    
    
class ProjectedGradientDescentMomentum_Nesterov(OptimizationMethod):
    """
    Nesterov's Accelerated Gradient Method (FISTA-style).
    
    Algorithm from report:
        Input: w_0, L
        Init: w_{-1} = w_0, α = 1/L, λ_0 = β_0 = 0
        
        for k = 0, 1, ..., N-1 do:
            y_k = w_k + β_k (w_k - w_{k-1})
            w_{k+1} = P_Δ(y_k - α ∇f(y_k))
            λ_{k+1} = (1 + sqrt(1 + 4λ_k²)) / 2
            β_{k+1} = (λ_k - 1) / λ_{k+1}
    
    Convergence rate: O(1/k²) for smooth convex functions.
    
    Parameters:
        max_iter: int - Maximum number of iterations
        tol: float - Convergence tolerance
    """
    def __init__(self, parameters, performance_indicator: PerformanceIndicator):
        super().__init__("ProjectedGradientDescentMomentum_Nesterov", parameters, performance_indicator)
        self.max_iter = parameters.get("max_iter", 1000)
        self.tol = parameters.get("tol" , 10e-20)
        # Internal state for Nesterov acceleration
        self._w_prev = None      # w_{k-1}
        self._lambda_k = 0.0     # λ_k
        self._beta_k = 0.0       # β_k
        self._alpha = None       # step size = 1/L
        self.metric = []
        self.time = []
        self.obj_value = []

    def optimize(self, model, w0):
        # Reset history for each optimization run
        self.metric = []
        self.time = []
        self.obj_value = []
        
        # Initialize according to algorithm:
        # w_{-1} = w_0, α = 1/L, λ_0 = β_0 = 0
        w_new = w0.copy()
        self._w_prev = w0.copy()  # w_{-1} = w_0
        self._alpha = 1.0 / model.lipschitz_constant()  # α = 1/L
        self._lambda_k = 0.0  # λ_0 = 0
        self._beta_k = 0.0    # β_0 = 0
        
        self.obj_value.append(model.f(w_new))
        
        t_start = time.time()
        counter = 0
        for iter in range(self.max_iter):
            w_old = w_new.copy()
            
            w_new = self.iterate(model, w_old)
            
            self.obj_value.append(model.f(w_new))
            
            # Record cumulative time since start
            self.time.append(time.time() - t_start)

            convergence_value = self.performance_indicator.evaluate(w_new, w_old, model)
            self.metric.append(convergence_value)
            
            if convergence_value < self.tol and counter >= 2:
                return {
                    "sol": w_new,
                    "value": model.f(w_new),
                    "iterations": iter + 1,
                    "converged": True,
                    "metric": self.metric,
                    "time": self.time,
                    "obj_value": self.obj_value,
                }
            counter +=1

        return {
            "sol": w_new,
            "value": model.f(w_new),
            "iterations": self.max_iter,
            "converged": False,
            "metric": self.metric,
            "time": self.time,
            "obj_value": self.obj_value,
        }

    def iterate(self, model, w):
        """Single iteration (for external use). Note: requires proper initialization."""
        if self._w_prev is None:
            self._w_prev = w.copy()
            self._alpha = 1.0 / model.lipschitz_constant()
            self._lambda_k = 0.0
            self._beta_k = 0.0
        
        # y_k = w_k + β_k (w_k - w_{k-1})
        y_k = w + self._beta_k * (w - self._w_prev)
        
        # w_{k+1} = P_Δ(y_k - α ∇f(y_k))
        grad_y = model.gradient(y_k)
        w_new = simplex_projection(y_k - self._alpha * grad_y)
        
        # Update λ and β
        lambda_new = (1.0 + np.sqrt(1.0 + 4.0 * self._lambda_k ** 2)) / 2.0
        self._beta_k = (self._lambda_k - 1.0) / lambda_new
        self._lambda_k = lambda_new
        self._w_prev = w.copy()
        
        return w_new




class ProjectedRandomizedCoordinateDescent(OptimizationMethod):
    def __init__(self, parameters, performance_indicator: PerformanceIndicator):
        super().__init__("ProjectedRandomizedCoordinateDescent", parameters, performance_indicator)
        self.step_size = parameters.get("step_size", 0.01)  # Fallback
        self.max_iter = parameters.get("max_iter", 1000)
        self.iterate_meth = parameters.get("iter_met")
        self.tol = parameters.get("tol", 1e-6)
        self.metric = []
        self.time = []
        self.obj_value = []
        self.idx_deja_vu = []

    def optimize(self, model, w0):
        # Reset history for each optimization run
        self.metric = []
        self.time = []
        self.obj_value = []
        self.idx_deja_vu = []
        w_new = w0.copy()
        self.obj_value.append(model.f(w_new))
        
        

        
        t_start = time.time()
        for iter in range(self.max_iter):
            w_old = w_new.copy()

            if self.iterate_meth == "Naive": 
                w_new = self.iterate(model, w_old)
            elif self.iterate_meth == "Naive2": 
                w_new = self.iterate_option1(model, w_old)
            elif self.iterate_meth == "Cleaver":
                w_new = self.iterate_option2(model, w_old)
            self.obj_value.append(model.f(w_new))
            
            # Record cumulative time since start
            self.time.append(time.time() - t_start)

            convergence_value = self.performance_indicator.evaluate(w_new, w_old, model)
            self.metric.append(np.linalg.norm(w_new - w_old))

            # if convergence_value < self.tol:
            #     return {
            #         "sol": w_new,
            #         "value": model.f(w_new),
            #         "iterations": iter + 1,
            #         "converged": True,
            #         "metric": self.metric,
            #         "time": self.time,
            #         "obj_value": self.obj_value,
            #     }

        return {
            "sol": w_new,
            "value": model.f(w_new),
            "iterations": self.max_iter,
            "converged": False,
            "metric": self.metric,
            "time": self.time,
            "obj_value": self.obj_value,
        }

    def iterate(self, model, w):
        n = w.shape[0]
        # Randomly select a coordinate (éviter de reprendre la même)
        coord_idx = np.random.randint(0, n)
        while coord_idx in self.idx_deja_vu:
            coord_idx = np.random.randint(0, n)
        
        self.idx_deja_vu.append(coord_idx)

        if len(self.idx_deja_vu) == n:
            self.idx_deja_vu = []
        
        # Compute partial gradient for the selected coordinate only
        grad_i = model.gradient_coordinate(w, coord_idx)

       

        # Update only the selected coordinate
        w_new = w.copy()
        w_new[coord_idx] = w[coord_idx] - self.step_size * grad_i
        

        # Project onto feasible set (simplex)
        w_new = simplex_projection(w_new)
        return w_new
    def iterate_option1(self, model, w):
        n = w.shape[0]
        # Randomly select a coordinate (éviter de reprendre la même)
        coord_idx = np.random.randint(0, n)
        while coord_idx in self.idx_deja_vu:
            coord_idx = np.random.randint(0, n)
        
        self.idx_deja_vu.append(coord_idx)

        if len(self.idx_deja_vu) == n:
            self.idx_deja_vu = []
        
        # Compute partial gradient for the selected coordinate only
        grad_i = model.gradient_coordinate(w, coord_idx)

       

        # Update only the selected coordinate
        w_new = w.copy()
        w_new[coord_idx] = w[coord_idx] - self.step_size * grad_i

        other_coord = np.random.randint(0,n)  
        while other_coord in self.idx_deja_vu:
            other_coord = np.random.randint(0, n)
        
        w_new[other_coord] = w[other_coord] + self.step_size *grad_i

        # Project onto feasible set (simplex)
        # w_new = simplex_projection(w_new)
        return w_new
    def iterate_option2(self, model, w):
        n = w.shape[0]
        i, j = np.random.choice(n, size=2, replace=False)

        grad_i = model.gradient_coordinate(w, i)
        grad_j = model.gradient_coordinate(w, j)

        d = grad_i - grad_j
        alpha = self.step_size

        w_new = w.copy()
        w_new[i] -= alpha * d
        w_new[j] += alpha * d

        w_new[i] = max(w_new[i], 0)
        w_new[j] = max(w_new[j], 0)

        s = w_new[i] + w_new[j]
        if s > 0:
            w_new[i] *= (w[i] + w[j]) / s
            w_new[j] *= (w[i] + w[j]) / s

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
        self.metric = []
        self.time = []
        self.obj_value = []

    def optimize(self, model, w0):
        # Reset history for each optimization run
        self.metric = []
        self.time = []
        self.obj_value = []
        w_new = w0.copy()
        self._iter_count = 0
        
        # Track best solution (subgradient methods are not descent methods)
        best_w = w_new.copy()
        best_value = model.f(w_new)
        self.obj_value.append(best_value)
        
        t_start = time.time()
        for iter in range(self.max_iter):
            w_old = w_new.copy()
            self._iter_count = iter + 1
            w_new = self.iterate(model, w_old)
            
            # Update best solution
            current_value = model.f(w_new)
            if current_value < best_value:
                best_value = current_value
                best_w = w_new.copy()
            
            self.obj_value.append(current_value)
            
            # Record cumulative time since start
            self.time.append(time.time() - t_start)

            convergence_value = self.performance_indicator.evaluate(w_new, w_old, model)
            self.metric.append(convergence_value)

            if convergence_value < self.tol:
                return {
                    "sol": w_new,           # Return LAST iterate
                    "value": current_value, # Return LAST value
                    "best_sol": best_w,     # Also provide best for reference
                    "best_value": best_value,
                    "iterations": iter + 1,
                    "converged": True,
                    "metric": self.metric,
                    "time": self.time,
                    "obj_value": self.obj_value,
                }

        return {
            "sol": w_new,               # Return LAST iterate
            "value": model.f(w_new),    # Return LAST value
            "best_sol": best_w,         # Also provide best for reference
            "best_value": best_value,
            "iterations": self.max_iter,
            "converged": False,
            "metric": self.metric,
            "time": self.time,
            "obj_value": self.obj_value,
        }

    def _get_step_size(self):
        """Get step size based on the rule."""
        if self.step_size_rule == "diminishing":
            return self.step_size / np.sqrt(self._iter_count)
        return self.step_size

    def iterate(self, model, w):
        # model.subgradient() returns the FULL subgradient (smooth + non-smooth parts)
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
        self.metric = []
        self.time = []
        self.obj_value = []

    def optimize(self, model, w0):
        # Reset history for each optimization run
        self.metric = []
        self.time = []
        self.obj_value = []
        w_new = w0.copy()
        self.obj_value.append(model.f(w_new))
        
        t_start = time.time()
        for iter in range(self.max_iter):
            w_old = w_new.copy()
            w_new = self.iterate(model, w_old)
            self.obj_value.append(model.f(w_new))
            
            # Record cumulative time since start
            self.time.append(time.time() - t_start)

            convergence_value = self.performance_indicator.evaluate(w_new, w_old, model)
            self.metric.append(convergence_value)
            
            if convergence_value < self.tol:
                return {
                    "sol": w_new,
                    "value": model.f(w_new),
                    "iterations": iter + 1,
                    "converged": True,
                    "metric": self.metric,
                    "time": self.time,
                    "obj_value": self.obj_value,
                }

        return {
            "sol": w_new,
            "value": model.f(w_new),
            "iterations": self.max_iter,
            "converged": False,
            "metric": self.metric,
            "time": self.time,
            "obj_value": self.obj_value,
        }

    def iterate(self, model, w):
        """
        Proximal gradient for: f(w) = smooth(w) + c * ||w - w_prev||_1  subject to w ∈ Δ
        
        Steps:
        1. Gradient step on smooth part
        2. Soft thresholding (proximal of L1)
        3. Project onto simplex to maintain feasibility
        """
        # Step 1: Gradient step on smooth part only
        grad_smooth = model.smooth_gradient(w)
        w_half = w - self.step_size * grad_smooth
        
        # Step 2: Soft thresholding for L1 penalty
        w_prox = self._soft_threshold(w_half - model.w_prev, self.step_size * model.c) + model.w_prev
        
        # Step 3: Project onto simplex (CRITICAL!)
        w_new = simplex_projection(w_prox)
        
        return w_new

    def _soft_threshold(self, x, threshold):
        """Soft thresholding operator (proximal operator for L1 norm)."""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    

class ProximalGradientMethod_fast(OptimizationMethod):
    def __init__(self, parameters, performance_indicator: PerformanceIndicator):
        super().__init__("ProximalGradient", parameters, performance_indicator)
        self.step_size = parameters.get("step_size", 0.01)
        self.max_iter = parameters.get("max_iter", 1000)
        self.tol = parameters.get("tol", 1e-6)
        self.metric = []
        self.time = []
        self.obj_value = []
        self.t = 1.0
        self.y = None
        self.w_prev = None

    def optimize(self, model, w0):
        # Reset history for each optimization run
        self.metric = []
        self.time = []
        self.obj_value = []
        w_new = w0.copy()
        self.obj_value.append(model.f(w_new))
        t_start = time.time()
        self.y = w0 
        self.w_prev = w0
        for iter in range(self.max_iter):
            w_old = w_new.copy()
            w_new = self.iterate(model, w_old)
            self.obj_value.append(model.f(w_new))
            
            # Record cumulative time since start
            self.time.append(time.time() - t_start)

            convergence_value = self.performance_indicator.evaluate(w_new, w_old, model)
            self.metric.append(convergence_value)
            
            if convergence_value < self.tol:
                return {
                    "sol": w_new,
                    "value": model.f(w_new),
                    "iterations": iter + 1,
                    "converged": True,
                    "metric": self.metric,
                    "time": self.time,
                    "obj_value": self.obj_value,
                }

        return {
            "sol": w_new,
            "value": model.f(w_new),
            "iterations": self.max_iter,
            "converged": False,
            "metric": self.metric,
            "time": self.time,
            "obj_value": self.obj_value,
        }

    def iterate(self, model, w):
        """
        FISTA (Fast Proximal Gradient) for: f(w) = smooth(w) + c * ||w - w_prev||_1  subject to w ∈ Δ
        
        Steps:
        1. Gradient step on smooth part (at extrapolated point y)
        2. Soft thresholding (proximal of L1)
        3. Project onto simplex
        4. Update momentum parameters
        """
        # Step 1: Gradient step on smooth part at extrapolated point y
        grad_smooth = model.smooth_gradient(self.y)
        w_half = self.y - self.step_size * grad_smooth
        
        # Step 2: Soft thresholding for L1 penalty
        w_prox = self._soft_threshold(w_half - model.w_prev, self.step_size * model.c) + model.w_prev
        
        # Step 3: Project onto simplex (CRITICAL!)
        w_prox = simplex_projection(w_prox)
        
        # Step 4: FISTA momentum update
        prev_t = self.t
        self.t = (1 + np.sqrt(1 + 4 * self.t ** 2)) / 2
        self.y = w_prox + (prev_t - 1) / self.t * (w_prox - self.w_prev)
        self.y = simplex_projection(self.y)  # Keep y feasible too
        self.w_prev = w_prox
        
        return w_prox

    def _soft_threshold(self, x, threshold):
        """Soft thresholding operator (proximal operator for L1 norm)."""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


class InteriorPointMethodLongStep(OptimizationMethod):
    """
    Long-step path-following interior-point method (Lecture 9).

    While mu > mu_final:
        mu <- (1-theta) mu
        perform damped Newton steps until delta_mu(w) <= tau
    """

    def __init__(self, parameters, performance_indicator: PerformanceIndicator):
        super().__init__("InteriorPointLongStep", parameters, performance_indicator)

        self.epsilon = parameters.get("epsilon", 1e-8)

        # Long-step parameters (flexible)
        self.tau = parameters.get("tau", 0.25)                 # target proximity
        self.theta = parameters.get("theta", 0.2)              # any (0,1)
        self.t0 = parameters.get("t0", 1.0)                   # initial barrier parameter
        self.tol = parameters.get("tol", 1e-6)

        self.max_outer_iter = parameters.get("max_iter", 200)
        self.max_inner_iter = parameters.get("max_inner_iter", 50)

        self.metric = []
        self.time = []
        self.obj_value = []

    def optimize(self, model, w0):
        # Dimensions and variables
        n = w0.shape[0]

        x = np.zeros(3*n)
    
        # Initialize x = (w, u, v) satisfying w - u + v = w_prev
        # This means: u - v = w - w_prev
        # We need w, u, v > 0 (strictly interior for barrier method)
        # 
        # Use small positive slack to start closer to the central path
        epsilon_init = 0.01  # Small initial slack 
        x[0:n] = w0.copy()
        
        diff = w0 - model.w_prev
        # u - v = diff, and both u, v > 0
        # Set u = max(diff, 0) + eps, v = max(-diff, 0) + eps
        # Then u - v = max(diff,0) - max(-diff,0) = diff (since one is always 0)
        x[n:2*n] = np.maximum(diff, 0) + epsilon_init
        x[2*n:3*n] = np.maximum(-diff, 0) + epsilon_init


        # Parameters
        t_final = 3*n / self.tol # t = m/epsilon
        t = self.t0


        timer = time.time()

        for iter in range(self.max_outer_iter):
            w_old = x[0:n]

            # Perform damped Newton iterations
            x = self._inner_iterations(model, x, t)

            w = x[0:n]
            self.time.append(time.time() - timer)
            self.obj_value.append(model.f(w))
            perf_measure = self.performance_indicator.evaluate(w, w_old, model)
            self.metric.append(perf_measure)

            if t > t_final:
                return {
                    "sol": x[0:n],
                    "value": model.f(x[0:n]),
                    "iterations": iter + 1,
                    "converged": True,
                    "metric": self.metric,
                    "time": self.time,
                    "obj_value": self.obj_value,
                }
            
            # Update barrier parameter (increase t for next iteration)
            t = t / (1 - self.theta)
        
        return {
                "sol": x[0:n],
                "value": model.f(x[0:n]),
                "iterations": iter + 1,
                "converged": False,
                "metric": self.metric,
                "time": self.time,
                "obj_value": self.obj_value,
            }

    def iterate(self, model, w):
        """Single iteration is not well-defined for interior point (uses inner loops).
        This is a placeholder to satisfy the abstract class requirement.
        """
        raise NotImplementedError("Interior point uses optimize() directly, not iterate()")

    # ---------------- Core long-step routines ----------------

    def _inner_iterations(self, model, x, t):
        """Perform many damped Newton-Steps until delta <= tau
        """

        x_new = x.copy()
        for i in range(self.max_inner_iter):
            # Find Newton-Step direction and local norm value delta
            dx, delta = self.solve_KKT(model, x_new, t)

            # If we reached the tolerance, stop
            if delta < self.tau:
                break
            
            # Else, we perform a damped Newton-Step and we loop
            step = 1 / (1 + delta)
            x_new = x_new + step * dx

        if i == self.max_inner_iter: print("[WARNING] Newton-Steps did not reach convergence")

        return x_new


    def solve_KKT(self, model, x, t):
        """Solve the KKT system for the Newton step at current w and barrier parameter t."""
        ### Dimensions and variables
        nx = x.shape[0] # Dimensions of x = (w, u, v) (u and v are the slack variables)
        n = nx // 3 # Dimensions of w, u and v individually
        m = n + 1  # Number of equality constraints
        
        w, u, v = x[0:n], x[n:2*n], x[2*n:3*n]

        ###  Define equality constraints Ax = b
        A = np.zeros((m, nx))
        b = np.zeros(m)

        # wi - ui + vi = (w_prev)_i forall i
        A[0:n, 0:n] = np.eye(n)
        A[0:n, n:2*n] = -np.eye(n)
        A[0:n, 2*n:3*n] = np.eye(n)
        b[:n] = model.w_prev
        
        # sum w_i = 1
        A[n,:n] = np.ones(n)
        b[-1] = 1
        

        ### Define gradient and hessian of barrier function

        # Gradient
        grad_w = t * (model.sigma @ w - model.lam * model.mu) - 1/w
        grad_u = t * model.c * np.ones(n) - 1/u
        grad_v = t * model.c * np.ones(n) - 1/v

        grad = np.concatenate([grad_w, grad_u, grad_v])

        # Hessian
        H_w = t * model.sigma + np.diag(1.0 / (w ** 2))
        H_u = np.diag(1.0 / (u ** 2))
        H_v = np.diag(1.0 / (v ** 2))

        H = np.block([
            [H_w, np.zeros((n, n)), np.zeros((n, n))],
            [np.zeros((n, n)), H_u, np.zeros((n, n))],
            [np.zeros((n, n)), np.zeros((n, n)), H_v]
        ])

        ### Define and solve Newton-KKT problem (ref: https://won-j.github.io/M1399_000200-2021fall/lectures/22-newton/newton_constr.html)

        # Number of equality constraints
        m = n + 1

        # Define system
        KKT = np.block([
            [H, A.T],
            [A, np.zeros((m, m))]
        ])

        b_KKT = -np.concatenate([grad, A @ x - b])

        # Solve
        try:
            sol = np.linalg.solve(KKT, b_KKT)
            dx = sol[:nx]
        except np.linalg.LinAlgError:
            # Fallback (rare): small projected gradient-like direction
            print("[WARNING] Singular matrix in KKT")
            dx = -0.01 * (grad - np.mean(grad))

        # Compute delta (local norm) which satisfies (-grad^T dx)^(1/2)
        delta = np.sqrt(np.maximum(-grad @ dx, 0))  # Clip to avoid sqrt of negative

        return dx, delta