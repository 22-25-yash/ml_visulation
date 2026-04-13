import numpy as np

def gradient_descent(start, lr, iterations):
    x = start
    history = []

    for _ in range(iterations):
        grad = 2 * x   # derivative of x^2
        x = x - lr * grad
        history.append(x)

    return history