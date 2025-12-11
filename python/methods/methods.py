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
    - BB2: α_k = (s_{k-1}^T y_{k-1}) / (y_{k-1}^T y_{k-1})
    
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
        
        if self.variant == 'BB1':
            s_dot_s = np.dot(s, s)
            if abs(s_dot_y) < 1e-14:
                alpha = self.alpha_init
            else:
                alpha = s_dot_s / s_dot_y
        else:  # BB2
            y_dot_y = np.dot(y, y)
            if y_dot_y < 1e-14:
                alpha = self.alpha_init
            else:
                alpha = s_dot_y / y_dot_y
        
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
            grad = model.gradient(w_old)
            
            # Get step size from strategy
            step = self.step_size_strategy.get_step_size(model, w_old, grad, iter)
            self.step_sizes.append(step)
            
            # Update velocity with momentum
            self._velocity = self.momentum * self._velocity - step * grad
            w_new = simplex_projection(w_old + self._velocity)
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
        self._velocity = self.momentum * self._velocity - step * grad
        w_new = w + self._velocity
        # Project onto feasible set (simplex)
        w_new = simplex_projection(w_new)
        return w_new


class ProjectedRandomizedCoordinateDescent(OptimizationMethod):
    def __init__(self, parameters, performance_indicator: PerformanceIndicator):
        super().__init__("ProjectedRandomizedCoordinateDescent", parameters, performance_indicator)
        self.step_size = parameters.get("step_size", 0.01)  # Fallback
        self.max_iter = parameters.get("max_iter", 1000)
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
            w_new = self.iterate(model, w_old)
            self.obj_value.append(model.f(w_new))
            
            # Record cumulative time since start
            self.time.append(time.time() - t_start)

            convergence_value = self.performance_indicator.evaluate(w_new, w_old, model)
            self.metric.append(np.linalg.norm(w_new - w_old))
            
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
                    "sol": best_w,
                    "value": best_value,
                    "iterations": iter + 1,
                    "converged": True,
                    "metric": self.metric,
                    "time": self.time,
                    "obj_value": self.obj_value,
                }

        return {
            "sol": best_w,
            "value": best_value,
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
        self.metric = []
        self.time = []
        self.obj_value = []

    def optimize(self, model, w0):
        # Reset history for each optimization run
        self.metric = []
        self.time = []
        self.obj_value = []
        
        n = w0.shape[0]
        w_new = w0.copy()
        
        # Ensure initial point is strictly feasible (interior of simplex)
        w_new = np.clip(w_new, 1e-8, 1 - 1e-8)
        w_new = w_new / np.sum(w_new)
        self.obj_value.append(model.f(w_new))
        
        mu = self.mu
        
        t_start = time.time()
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