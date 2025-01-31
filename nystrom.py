import numpy as np
from kernels import RBFkernel
from scipy.linalg import svd
from sklearn.gaussian_process.kernels import RBF

from recursive_nystrom import recursiveNystrom, gauss


def nystrom_features(X, nys_inds, sigma):
    """
    Compute Nyström approximation features for a dataset X.

    Parameters:
    X : ndarray
        The input data matrix (n samples x d features).
    nys_inds : ndarray
        Indices of selected Nyström centers in X.
    kernel_function : str, optional
        The kernel function to use (default is 'rbf').
    gamma : float, optional
        Kernel parameter (e.g., width of the RBF kernel, default is 1). -  note that some kernels take sigma: gamma = 1/(2 sigma^2)

    Returns:
    ndarray
        The Nyström feature representation of the dataset.
    """

    # Compute the kernel matrix between all points and the Nyström centers.
    Knm = RBFkernel(X, X[nys_inds], sigma)  # Shape: [n, k] where k = len(nys_inds)

    # Compute the kernel matrix for the Nyström centers.
    Km = Knm[nys_inds,:]  # Shape: [k, k]

    # Perform Singular Value Decomposition (SVD) on the center kernel matrix.
    U, S, Vt = svd(Km)  # U: [k, k], S: singular values, Vt: [k, k] (transposed)

    # Stabilize singular values by enforcing a minimum value (to avoid division by zero).
    S = np.maximum(S, 1e-12)  # Ensures numerical stability for SVD inversion.

    # Construct the Nyström mapping matrix using the inverse square root of S.
    # `U / sqrt(S)` scales the singular vectors by the inverse square roots of singular values.
    # `Vt` completes the mapping structure.
    map = np.dot(U / np.sqrt(S), Vt)  # Shape: [k, k]

    # Return the transformed features: multiply Knm with the mapping matrix.
    return Knm @ map.T  # Shape: [n, k] # check if tranposition is needed!
    #return np.dot(Knm, map.T)  # Shape: [n, k] # check if tranposition is needed!



def nys_inds(X, k, method, sigma, lam, seed):
    """
    Compute indices for Nyström approximation based on a specified method.
    
    Parameters:
    X : array-like, shape (n_samples, n_features)
        The input data matrix.
    k : int
        The number of samples or landmarks to select.
    method : str, optional, default='uniform'
        The method for selecting indices:
        - 'full_rank': Select all indices.
        - 'uniform': Select indices uniformly at random.
        - 'bless': Use the BLESS algorithm for sampling.
    sigma : float, optional, default=1
        Parameter for the RBF kernel in the 'bless' method. -  note that some kernels take sigma: gamma=1/(2 sigma^2)

    lam : float, optional, default=0.1
        Regularization parameter for the BLESS algorithm.
    seed : int, optional, default=0
        Random seed for reproducibility.

    Returns:
    inds : array-like
        The selected indices based on the method.
    count : int
        The number of selected indices.
    """

    # Get the number of samples in the dataset
    ntot = X.shape[0]


    gamma = 1 / (2 * sigma**2)

    if method == 'uniform':
        # For 'uniform', randomly select k indices without replacement
        rng = np.random.default_rng(seed=seed)  # Initialize a random number generator
        inds = rng.choice(ntot, k, replace=False)  # Shuffle all indices
        return inds, k  # Return the first k indices and their count
    
    elif method == 'rlss': # takes gamma
        inds = recursiveNystrom(X, n_components=k, kernel_func=lambda *args, **kwargs: gauss(*args, **kwargs, gamma = gamma), random_state=seed)
        assert len(inds) == k
        return inds, k  # Return the indices and their count


    else:
        # Handle invalid method input
        raise ValueError("Method not implemented. Choose between 'uniform', and 'rlss'.")