import numpy as np

from nystrom import nys_inds, nystrom_features

import numpy as np



def NysMMDtest(Z, n, m, seed=None, method='uniform', bandwidth=1, k=20):
    """
    Performs the Nyström approximation MMD test.
    Parameters:
    Z (array): Feature matrix of shape (ntot, d).
    n (int): number of data points for first sample
    m (int): number of data points for second sample
    seed (int, optional): Random seed for reproducibility.
    method (str): Nyström sampling method ('uniform' or 'rlss').
    bandwidth (float): Kernel bandwidth parameter.
    k (int): Number of Nyström features.
    Returns:
    MMD (float): value of MMD test statistic
    """
    ntot, d = Z.shape  # Total samples and feature dimension
    assert n + m == ntot, "n + m must be equal to the total size of the dataset"
    
    # Compute Nyström feature mapping
    inds, _ = nys_inds(Z, k, method, bandwidth, seed)  # Fixed: nys*inds -> nys_inds
    psi_Z = nystrom_features(Z, inds, bandwidth)
    
    psi_X = psi_Z[:n]  # First n samples (X group)
    psi_Y = psi_Z[n:]  # Last m samples (Y group)
    
    # Compute mean embeddings
    bar_psi_X = np.mean(psi_X, axis=0)
    bar_psi_Y = np.mean(psi_Y, axis=0)
    
    T = bar_psi_X - bar_psi_Y
    
    # Compute MMD statistic
    MMD = np.sum(T ** 2) 
    
    return MMD
