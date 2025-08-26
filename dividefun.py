import math
import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from evalwitherrfun import eval_with_err

x, y = sp.symbols('x y')

def divide_fun(a, a_err, b, b_err):
    expr = x / y
    return eval_with_err(expr, [x, y], [a, b], [a_err, b_err])