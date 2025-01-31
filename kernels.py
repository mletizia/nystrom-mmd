import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


def RBFkernel(X1: np.ndarray, X2: np.ndarray, sigma: float) -> np.ndarray:
    """
    Computes the RBF (Gaussian) kernel: k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))

    Args:
        X1 (np.ndarray): Shape (n_samples_1, n_features) input data.
        X2 (np.ndarray): Shape (n_samples_2, n_features) input data.
        sigma (float): The standard deviation of the Gaussian kernel.

    Returns:
        np.ndarray: Kernel matrix of shape (n_samples_1, n_samples_2).
    """
    gamma = 1 / (2 * sigma**2)  # Convert sigma to gamma for rbf_kernel
    return rbf_kernel(X1, X2, gamma)