# Export all utility functions for easy importing
# Usage: from utils import my_nice_function

from .utils import (
    plot_convergence,
    simplex_projection,
    plot_iterations_vs_tolerance,
    compare_methods,
    
)

__all__ = [
    "simplex_projection",
    "plot_convergence",
    "plot_iterations_vs_tolerance",
    "compare_methods"
]
