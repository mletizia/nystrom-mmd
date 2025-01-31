import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


def RBFkernel(X1: np.ndarray, X2: np.ndarray, sigma: float) -> np.ndarray:
    """
    Computes the Radial Basis Function (RBF) kernel, also known as the Gaussian kernel.
    The function calculates the kernel matrix using the formula:
    k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))

    Args:
        X1 (np.ndarray): Input data of shape (n_samples_1, n_features).
        X2 (np.ndarray): Input data of shape (n_samples_2, n_features).
        sigma (float): The standard deviation parameter for the Gaussian kernel.

    Returns:
        np.ndarray: The computed kernel matrix of shape (n_samples_1, n_samples_2).
    """
    gamma = 1 / (2 * sigma**2)  # Convert sigma to gamma for rbf_kernel computation
    return rbf_kernel(X1, X2, gamma)