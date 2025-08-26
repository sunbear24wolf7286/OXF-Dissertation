import numpy as np
import sympy as sp

def eval_with_err(func_expr, variables, values, errors):
    func_val = float(func_expr.evalf(subs=dict(zip(variables, values))))
    squared_terms = []
    for var, val, err in zip(variables, values, errors):
        partial = sp.diff(func_expr, var)
        partial_val = float(partial.evalf(subs=dict(zip(variables, values))))
        squared_terms.append((partial_val * err)**2)
    error = np.sqrt(sum(squared_terms))
    return func_val, error