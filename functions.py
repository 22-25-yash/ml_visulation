import sympy as sp
import numpy as np

x = sp.symbols('x')

def get_function(expr_str):
    expr = sp.sympify(expr_str)
    f = sp.lambdify(x, expr, "numpy")
    return expr, f

def get_derivative(expr):
    return sp.diff(expr, x)

def get_derivative_function(deriv_expr):
    return sp.lambdify(x, deriv_expr, "numpy")