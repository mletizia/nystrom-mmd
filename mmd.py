import numpy as np
from kernels import RBFkernel


def MMD(phiX1, phiX2):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two feature sets phiX1 and phiX2.

    Args:
        phiX1 (np.ndarray): Feature representation of first sample set (shape [n_samples, n_features]).
        phiX2 (np.ndarray): Feature representation of second sample set (shape [n_samples, n_features]).

    Returns:
        float: The squared norm of the mean difference (MMD statistic).
    """
    # Compute the mean difference vector between the two groups in the feature space
    t = phiX1.mean(axis=0) - phiX2.mean(axis=0)  # Compute the mean of each feature and subtract

    # Return the squared norm of the mean difference (MMD statistic)
    return np.inner(t, t)  # Equivalent to the dot product of t with itself


def MMD2b(x, y, sigma):
    """
    Compute the biased MMD^2 between distributions X and Y.
    
    Args:
        x (np.ndarray): Samples from distribution X (shape [n_samples_x, n_features]).
        y (np.ndarray): Samples from distribution Y (shape [n_samples_y, n_features]).
        sigma (float): Bandwidth parameter for the RBF kernel.
        
    Returns:
        float: The biased MMD^2 value.
    """
    # Compute kernel matrices using the RBF kernel function
    K_xx = RBFkernel(x, x, sigma)  # Compute kernel matrix for X
    K_yy = RBFkernel(y, y, sigma)  # Compute kernel matrix for Y
    K_xy = RBFkernel(x, y, sigma)  # Compute kernel matrix between X and Y

    # Compute biased MMD^2 using kernel mean values
    mmd2 = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    
    return mmd2
