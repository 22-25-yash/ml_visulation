import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from vector_similarity import *
from gradient_descent import *
from functions import *

st.title("ML Foundations Visualizer")

menu = st.sidebar.selectbox("Select Module",
    ["Vector Similarity", "Function & Derivative", "Gradient Descent"])

# ================= VECTOR =================
if menu == "Vector Similarity":
    st.header("Vector Similarity")

    v1 = np.array([
        st.number_input("v1 x", value=1.0),
        st.number_input("v1 y", value=2.0)
    ])

    v2 = np.array([
        st.number_input("v2 x", value=3.0),
        st.number_input("v2 y", value=4.0)
    ])

    if st.button("Compute"):
        st.write("Dot Product:", dot_product(v1, v2))
        st.write("Cosine Similarity:", cosine_similarity(v1, v2))
        st.write("Euclidean Distance:", euclidean_distance(v1, v2))

        fig = plot_vectors(v1, v2)
        st.pyplot(fig)


# ================= FUNCTION =================
elif menu == "Function & Derivative":
    st.header("Function & Derivative")

    func_input = st.text_input("Enter function (e.g. x**2, sin(x))", "x**2")

    try:
        expr, f = get_function(func_input)
        deriv_expr = get_derivative(expr)
        f_deriv = get_derivative_function(deriv_expr)

        st.write("Function:", expr)
        st.write("Derivative:", deriv_expr)

        x_vals = np.linspace(-10, 10, 100)
        y_vals = f(x_vals)
        y_deriv = f_deriv(x_vals)

        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals, label="f(x)")
        ax.plot(x_vals, y_deriv, label="f'(x)")
        ax.legend()
        ax.grid()

        st.pyplot(fig)

    except:
        st.error("Invalid function input")


# ================= GRADIENT DESCENT =================
elif menu == "Gradient Descent":
    st.header("Gradient Descent")

    start = st.number_input("Start value", value=5.0)
    lr = st.slider("Learning Rate", 0.01, 1.0, 0.1)
    iters = st.slider("Iterations", 1, 100, 20)

    if st.button("Run"):
        history = gradient_descent(start, lr, iters)

        st.write("Final value:", history[-1])

        fig, ax = plt.subplots()
        ax.plot(history)
        ax.set_title("Optimization Path")
        ax.grid()

        st.pyplot(fig)