import numpy as np
import matplotlib.pyplot as plt
import os

def simplex_projection(v):
    r"""Compute the projection of `v` on the simplex.

    Args:
        v (np.ndarray): Input vector.
    """

    # Some black magic for simplex projection (found by solving KKT conditions on the projection minimization problem)
    n = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = 1 - np.cumsum(u)
    ind = np.arange(n) + 1
    cond = u + cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    w = np.maximum(v + theta, 0)
    return w

def plot_convergence(result, theoretical_rate_exp=-0.5, rate_label=r'$O(k^{-1/2})$', method_name='method', save_dir=None , metric_used='iterate' , f_ref=None):
    """
    Plot convergence analysis with 3 separate figures.
    
    Args:
        result: dict with 'metric', 'time', 'obj_value' keys
        theoretical_rate_exp: exponent for theoretical rate (e.g., -0.5 for O(k^{-1/2}), -1 for O(1/k))
        rate_label: label for the theoretical rate in the legend
        method_name: name of the method for saving files (e.g., 'ProjectedGD', 'RCD')
        save_dir: directory to save figures. If None, figures are not saved.
    """
    iterations = np.arange(1, len(result['metric']) + 1, dtype=float)
    metric = np.array(result['metric'])
    obj_values = np.array(result['obj_value'])
    time_values = np.array(result['time'])
    
    time_per_iter = np.diff(np.concatenate([[0], time_values])) * 1000  # Convert to ms

    # Create save directory if specified
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # Plot 1: Convergence metric
    plt.figure(figsize=(8, 5))
    if metric_used == 'iterate':
        plt.loglog(iterations, metric, 'b-', linewidth=1.5, label=r'$\|x_{k+1} - x_k\|$ (empirical)')
    elif metric_used  == 'function':
        plt.loglog(iterations, metric, 'b-', linewidth=1.5, label=r'$f(x_{k+1}) - f(x_k)$ (empirical)')
    elif metric_used == 'function_with_ref':
        plt.loglog(iterations, metric, 'b-', linewidth=1.5, label=r'$f(x_{k}) - f^*$ (empirical)')
    else :
        raise ValueError("metric must be 'iterate' or 'function'")
    plt.xlabel('Iteration $k$', fontsize=12)
    plt.ylabel('Convergence metric', fontsize=12)
    plt.title(f'Convergence rate', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f'{method_name}_convergence_metric.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # Plot 2: Objective value gap
    plt.figure(figsize=(8, 5))
    f_star = f_ref  # Approximate f* with final value
    obj_gap = obj_values[:-1] - f_star  # Remove last element (it's 0)
    iterations_obj = np.arange(1, len(obj_gap) + 1, dtype=float)
    
    plt.loglog(iterations_obj, obj_gap, 'b-', linewidth=1.5, label=r'$f(x_k) - f^*$ (empirical)')
    
   
    plt.xlabel('Iteration $k$', fontsize=12)
    plt.ylabel(r'$f(x_k) - f^*$', fontsize=12)
    plt.title('Objective value convergence', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f'{method_name}_objective_gap.png'), dpi=150, bbox_inches='tight')
    plt.show()
    # Plot 4: Objective value 
    plt.figure(figsize=(8, 5))
    iterations_obj = np.arange(1, len(obj_values) + 1, dtype=float)
    
    plt.plot(iterations_obj, obj_values, 'b-', linewidth=1.5, label=r'$f(x_k)$ (empirical)')
    
   
    plt.xlabel('Iteration $k$', fontsize=12)
    plt.ylabel(r'$f(x_k)$', fontsize=12)
    plt.title('Objective value convergence', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f'{method_name}_objective_value.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # Plot 3: Time per iteration
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, time_per_iter, 'g-', linewidth=0.8, alpha=0.5, label='Per iteration')
    
    # Add moving average for smoothing
    window = min(20, len(time_per_iter) // 5) if len(time_per_iter) > 5 else 1
    if window > 1:
        moving_avg = np.convolve(time_per_iter, np.ones(window)/window, mode='valid')
        plt.plot(iterations[window-1:], moving_avg, 'r-', linewidth=2, label=f'Moving avg (window={window})')
    plt.axhline(y=np.mean(time_per_iter), color='k', linestyle='--', linewidth=1.5, label=f'Mean: {np.mean(time_per_iter):.3f} ms')
    
    plt.xlabel('Iteration $k$', fontsize=12)
    plt.ylabel('Time per iteration (ms)', fontsize=12)
    plt.title('Computational cost per iteration', fontsize=13)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f'{method_name}_time_per_iteration.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # Summary stats
    print(f"Total time: {result['time'][-1]:.4f} seconds")
    print(f"Total iterations: {len(result['metric'])}")
    print(f"Final objective value: {obj_values[-1]:.6f}")
    print(f"Average time per iteration: {np.mean(time_per_iter):.4f} ms")
    print(f"Std time per iteration: {np.std(time_per_iter):.4f} ms")





def measure_iteration_complexity(method_class, method_params, model, w0, tolerances, 
                                  theoretical_exp=2, rate_label=r'$O(\epsilon^{-2})$',
                                  method_name='method', save_dir=None, max_iter=10000 , metrics = None , f_ref = None):
    """
    Measure iteration complexity by re-running optimization for each tolerance.
    
    This is the correct way to verify theoretical complexity bounds like O(epsilon^{-2}).
    
    Args:
        method_class: The optimization method class (e.g., ProjectedGradientMethod)
        method_params: Base parameters dict for the method (will be modified for each tol)
        model: The optimization model
        w0: Initial point
        tolerances: List/array of tolerance values to test
        theoretical_exp: Exponent for theoretical complexity (e.g., 2 for O(epsilon^{-2}))
        rate_label: Label for the theoretical rate
        method_name: Name for saving files
        save_dir: Directory to save figures
        max_iter: Maximum iterations allowed
        
    Returns:
        dict with 'tolerances' and 'iterations' arrays
    """
    from methods import IteratePerformanceIndicator, ValuePerformanceIndicator , ValuePerformanceIndicator_with_ref
    
    
    iterations_list = []
    valid_tolerances = []
    
    for tol in tolerances:
        # Create method with this tolerance
        params = method_params.copy()
        params['tol'] = tol
        params['max_iter'] = max_iter
        if metrics == 'iterate': 
            performance_indicator = IteratePerformanceIndicator()
        elif metrics == 'function':
            performance_indicator = ValuePerformanceIndicator()
        elif metrics == 'function_with_ref':
            performance_indicator = ValuePerformanceIndicator_with_ref(f_ref=f_ref)
        else:
            performance_indicator = IteratePerformanceIndicator()  # default
        method = method_class(params, performance_indicator=performance_indicator)
        result = method.optimize(model, w0.copy())
        
        if result['converged']:
            iterations_list.append(result['iterations'])
            valid_tolerances.append(tol)
            print(f"  tol={tol:.2e}: {result['iterations']} iterations")
        else:
            print(f"  tol={tol:.2e}: did not converge in {max_iter} iterations")
    
    iterations_arr = np.array(iterations_list)
    valid_tolerances = np.array(valid_tolerances)
    
    if len(iterations_arr) == 0:
        print("No valid data points.")
        return None
    
    # Create save directory if specified
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.loglog(valid_tolerances, iterations_arr, 'bo-', linewidth=1.5, markersize=6, label='Empirical')
    
    # Fit theoretical curve
    C = iterations_arr[0] * (valid_tolerances[0] ** theoretical_exp)
    theoretical_iters = C * (valid_tolerances ** (-theoretical_exp))
    plt.loglog(valid_tolerances, theoretical_iters, 'r--', linewidth=2, label=f'{rate_label} (theoretical)')
    
    plt.xlabel(r'Tolerance $\epsilon$', fontsize=12)
    plt.ylabel(r'Iterations $K(\epsilon)$', fontsize=12)
    plt.title(f'Iteration Complexity (re-run per $\\epsilon$)', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, which='both')
    plt.gca().invert_xaxis()
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f'{method_name}_true_iteration_complexity.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return {'tolerances': valid_tolerances, 'iterations': iterations_arr}
    

def compare_methods(results_dict, save_dir=None , method_name='method' , metric_used='iterate'):
    """
    Compare multiple optimization methods on 4 different plots.
    
    Args:
        results_dict: dict of {method_name: result_dict}
            Each result_dict should have keys: 'metric', 'time', 'obj_value'
        save_dir: directory to save figures. If None, figures are not saved.
    
    Example:
        compare_methods({
            'Projected GD': result_ProjectedGradientMethod,
            'PGD + Momentum': result_ProjectedGradientDescentMomentum,
            'Randomized CD': result_ProjectedRandomizedCoordinateDescent,
        }, save_dir='figures')
    """
    colors = ['b', 'r', 'g', 'orange', 'purple', 'brown', 'pink']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.']

    # Create save directory if specified
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: Convergence metric comparison
    plt.figure(figsize=(10, 6))
    for i, (name, result) in enumerate(results_dict.items()):
        iterations = np.arange(1, len(result['metric']) + 1, dtype=float)
        metric = np.array(result['metric'])
        plt.loglog(iterations, metric, color=colors[i % len(colors)], 
                   linestyle=linestyles[i % len(linestyles)], linewidth=1.5, label=name)
    
    plt.xlabel('Iteration $k$', fontsize=12)
    if metric_used == 'iterate':
        plt.ylabel(r'$\|x_{k+1} - x_k\|$', fontsize=12)
    elif metric_used  == 'function':
        plt.ylabel(r'$f(x_{k+1}) - f(x_k)$', fontsize=12)
    else : 
        raise ValueError("metric must be 'iterate' or 'function'")
    plt.title('Convergence Metric Comparison', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'comparison_convergence_metric.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Objective value gap comparison
    plt.figure(figsize=(10, 6))
    
    # Find global f* (minimum across all methods)
    all_final_values = [np.array(result['obj_value'])[-1] for result in results_dict.values()]
    f_star_global = min(all_final_values)
    
    for i, (name, result) in enumerate(results_dict.items()):
        obj_values = np.array(result['obj_value'])
        obj_gap = obj_values - f_star_global
        # Remove zeros or negative values for log plot
        iterations_obj = np.arange(1, len(obj_gap) + 1, dtype=float)
        plt.loglog(iterations_obj, obj_gap, color=colors[i % len(colors)], 
                   linestyle=linestyles[i % len(linestyles)], linewidth=1.5, label=name)
    
    plt.xlabel('Iteration $k$', fontsize=12)
    plt.ylabel(r'$f(x_k) - f^*$', fontsize=12)
    plt.title('Objective Value Convergence Comparison', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f'{method_name}_comparison_objective_gap.png'), dpi=150, bbox_inches='tight')
    plt.show()


    
    # Plot 3: Computational cost comparison (bar chart)
    plt.figure(figsize=(10, 6))
    
    method_names = []
    mean_times = []
    std_times = []
    
    for name, result in results_dict.items():
        time_values = np.array(result['time'])
        time_per_iter = np.diff(np.concatenate([[0], time_values])) * 1000  # ms
        method_names.append(name)
        mean_times.append(np.mean(time_per_iter))
        std_times.append(np.std(time_per_iter))
    
    x_pos = np.arange(len(method_names))
    bars = plt.bar(x_pos, mean_times, yerr=std_times, capsize=5, 
                   color=[colors[i % len(colors)] for i in range(len(method_names))], alpha=0.7)
    plt.xticks(x_pos, method_names, rotation=15, ha='right')
    plt.ylabel('Time per iteration (ms)', fontsize=12)
    plt.title('Computational Cost per Iteration', fontsize=13)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f'{method_name}_comparison_computational_cost.png'), dpi=150, bbox_inches='tight')
    plt.show()

    #Plot 4:Objectif function value 
    plt.figure(figsize=(10, 6))
    
    # Find global f* (minimum across all methods)
    all_final_values = [np.array(result['obj_value'])[-1] for result in results_dict.values()]
    f_star_global = min(all_final_values)
    
    for i, (name, result) in enumerate(results_dict.items()):
        obj_values = np.array(result['obj_value'])
        obj_gap = obj_values 
        # Remove zeros or negative values for log plot
        iterations_obj = np.arange(1, len(obj_gap) + 1, dtype=float)
        plt.plot(iterations_obj, obj_gap, color=colors[i % len(colors)], 
                   linestyle=linestyles[i % len(linestyles)], linewidth=1.5, label=name)
    
    plt.xlabel('Iteration $k$', fontsize=12)
    plt.ylabel(r'$f(x_k)$', fontsize=12)
    plt.title('Objective Value Convergence Comparison', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f'{method_name}_objectif_value.png'), dpi=150, bbox_inches='tight')
    plt.show()
    




def plot_subgradient_complexity(model, w0, f_ref, M, epsilons, max_iter=100000,
                                 method_name='Subgradient', save_dir=None):
    """
    Plot iteration complexity vs precision ε for Projected Subgradient Method.
    
    For each ε, runs subgradient with α = ε/M² and counts iterations to reach ε-accuracy.
    
    Theoretical complexity: T ≥ D²M²/ε² to achieve ε-accuracy.
    
    Args:
        model: NonSmoothMarkowitzModel instance
        w0: initial point
        f_ref: reference optimal value f*
        D: diameter of feasible set
        M: bound on subgradient norm
        epsilons: array of target accuracies to test
        max_iter: maximum iterations per run
        method_name: name for saving files
        save_dir: directory to save figures
    """
    from methods import ProjectedSubgradientMethod, ValuePerformanceIndicator , ValuePerformanceIndicator_with_ref
    
    iterations_to_reach = []
    valid_epsilons = []
    
    print(f"=== Plot 2: Complexity vs precision ε ===")
    print(f"Parameters: D = {D:.4f}, M = {M:.4f}")
    
    for eps in epsilons:
        # Set step size α = ε/M²
        alpha = eps / (M**2)
        
        method = ProjectedSubgradientMethod({
            'step_size': alpha,
            'step_size_rule': 'constant',
            'max_iter': max_iter,
            'tol': eps,  # Don't stop early based on tol
        }, ValuePerformanceIndicator_with_ref(f_ref=f_ref))
        
        result = method.optimize(model, w0.copy())
        
        # Find first iteration where min_{k≤t} f(w_k) - f_ref ≤ ε
        if result['converged'] : 
            iterations_to_reach.append(result['iterations'])
            valid_epsilons.append(eps)
            print(f"  ε={eps:.2e}: reached in {result['iterations']} iterations")
    
    valid_epsilons = np.array(valid_epsilons)
    iterations_to_reach = np.array(iterations_to_reach)
    # Create save directory if specified
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    # ========== Plot 2: Complexity vs precision ε ==========
    plt.figure(figsize=(10, 6))
    
    # Empirical
    plt.loglog(valid_epsilons, iterations_to_reach, 'bo-', linewidth=2, markersize=8,
               label='Empirical iterations')
    
    # Theoretical: T ∝ 1/ε² - scaled to pass through first empirical point
    # Scale factor: iterations_to_reach[0] / (1/valid_epsilons[0]²)
    scale_factor = iterations_to_reach[0] * (valid_epsilons[0]**2)
    theoretical_complexity = scale_factor / (valid_epsilons**2)
    plt.loglog(valid_epsilons, theoretical_complexity, 'r--', linewidth=2,
               label=r'Theory: $O(1/\varepsilon^2)$')
    
    plt.xlabel(r'Target accuracy $\varepsilon$', fontsize=12)
    plt.ylabel(r'Iterations $T(\varepsilon)$', fontsize=12)
    plt.title('Subgradient Iteration Complexity', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, which='both')
    plt.gca().invert_xaxis()  # Smaller ε on the right
    plt.tight_layout()
    
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f'{method_name}_complexity_vs_epsilon.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return {'epsilons': valid_epsilons, 'iterations': iterations_to_reach}


