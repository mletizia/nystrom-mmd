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


def MMD2b_from_K(K: np.ndarray, A: np.ndarray) -> float:
    """
    Compute biased MMD^2 given a full kernel matrix.

    Args:
        K (np.ndarray): Full kernel matrix of shape (N, N), with N = 2n.
        A (np.ndarray): Indices (size n) of the first sample group.
                        The complement indices are used for the second group.

    Returns:
        float: Biased MMD^2 statistic.
    """
    n = A.size
    N = K.shape[0]

    # Complement indices for group B
    mask = np.ones(N, dtype=bool)
    mask[A] = False
    B = np.nonzero(mask)[0]

    # Block sums
    sumAA = K[np.ix_(A, A)].sum()
    sumBB = K[np.ix_(B, B)].sum()
    sumAB = K[np.ix_(A, B)].sum()

    # Biased MMD^2 (includes diagonals)
    return (sumAA + sumBB) / (n**2) - 2.0 * sumAB / (n**2)
