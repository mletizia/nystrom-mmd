"""
The design of the code for the permutation test is based on: https://github.com/ikjunchoi/rff-mmd from https://arxiv.org/pdf/2407.08976.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from stat_utils import decide, independent_permutation

from mmd import MMD2b

from nystrom import nys_inds, nystrom_features

def MMDb_test(Z, bw, seed=None, alpha=0.05, B=199, plot=False):

    start = time.time()  # Record computation time

    rng = np.random.default_rng(seed)

    ntot, d = Z.shape  # Get size of feature matrix
    n = int(ntot/2) # Get sample size

    # Compute observed test statistic
    t_obs = MMD2b(Z[:n],Z[n:], bw)

    # Generate permuted statistics
    H0 = np.empty(B+1)
    for i in tqdm(range(B)):
        Z_perm = rng.permutation(Z)  # Permute data
        H0[i] = MMD2b(Z_perm[:n], Z_perm[n:], bw)

    # add observed value
    H0[-1] = t_obs

    # compute output of the test
    output, thr = decide(H0, t_obs, B, alpha)

    if plot:
        values, bins, patches = plt.hist(H0, label=r"$H_0$")
        plt.vlines(thr, 0, max(values) * 0.8, color='red', label=f"{alpha}-level thr")  # Threshold line
        plt.vlines(t_obs, 0, max(values) * 0.8, color='black', label="Obs test statistic")  # Observed statistic line
        plt.legend()
        plt.show()

    dt = time.time() - start

    return output, dt, ntot


# converted to numpy from the implementation in https://github.com/ikjunchoi/rff-mmd

def rMMDtest(
    Z,
    seed=None,
    bandwidth=1,
    alpha=0.05,
    kernel='gaussian',
    R=20,
    B=199
):
    '''
    [Input and output]
        (Input)
            X: array_like
                The shape of X must be of the form (m, d) where m is the number
                of samples and d is the dimension.
            Y: array_like
                The shape of Y must be of the form (n, d) where n is the number
                of samples and d is the dimension.

        (Output)
            Delta: int
                The value of delta must be 0 or 1;
                    return 0 if the test ACCEPTS the null
                 or return 1 if the test REJECTS the null 


    [Parameters]
        key:
            Placeholder for compatibility with JAX version (unused in NumPy).
        seed:
            Random seed for reproducibility.
        alpha: scalar
            The value of alpha must be between 0 and 1.
        bandwidth: scalar:
            The value of bandwidth must be between 0 and 1.
        kernel: str
            The value of kernel must be "gaussian", ...
        R: int
            The number of random Fourier features.
        stat_type: str
            The value must be 'U' or 'V'; U-statistic or V-statistic.
        B: int
            The number of simulated test statistics to approximate the quantiles.
    '''
    
    start = time.time()

    # Ensure reproducibility
    rng = np.random.default_rng(seed)

    ntot, d = Z.shape

    n = m = int(ntot/2)

    #Rhalf = int(R//2)

    # Feature mapping
    if kernel == 'gaussian':
        omegas = np.sqrt(2) / bandwidth * rng.normal(size=(R, d))
    else:
        raise ValueError("Currently only 'gaussian' kernel is supported.")

    omegas_Z = np.dot(Z, omegas.T)  # (m+n) x R
    cos_feature = (1 / np.sqrt(R)) * np.cos(omegas_Z)  # (m+n) x R
    sin_feature = (1 / np.sqrt(R)) * np.sin(omegas_Z)  # (m+n) x R
    psi_Z = np.concatenate((cos_feature, sin_feature), axis=1)  # (m+n) x 2R

    # Permutation test
    I_1 = np.concatenate((np.ones(m), np.zeros(n)))
    I = np.tile(I_1, (B + 1, 1))  # (B+1) x (m+n)

    # Independent permutations along axis=1
    I_X = independent_permutation(I, rng, axis=1)
    I_X[0] = I_1  # Include the non-permuted case Z=(X,Y)
    I_Y = 1 - I_X

    bar_Z_B_piX = (1 / m) * I_X @ psi_Z  # (B+1) x R
    bar_Z_B_piY = (1 / n) * I_Y @ psi_Z  # (B+1) x R
    T = bar_Z_B_piX - bar_Z_B_piY  # (B+1) x R
    V = np.sum(T ** 2, axis=1)  # (B+1, )

    test_stats = V

    rMMD2 = test_stats[0]

    # take decision based on empirical threshold and level
    output, _ = decide(test_stats, rMMD2, B, alpha)

    dt = time.time() - start

    return output, dt, R


def NysMMDtestV2(
    Z,
    seed=None,
    bandwidth=1,
    alpha=0.05,
    method='uniform',
    k=20,
    B=199
):
    '''
    [Input and output]
        (Input)
            X: array_like
                The shape of X must be of the form (m, d) where m is the number
                of samples and d is the dimension.
            Y: array_like
                The shape of Y must be of the form (n, d) where n is the number
                of samples and d is the dimension.

        (Output)
            Delta: int
                The value of delta must be 0 or 1;
                    return 0 if the test ACCEPTS the null
                 or return 1 if the test REJECTS the null 


    [Parameters]
        key:
            Placeholder for compatibility with JAX version (unused in NumPy).
        seed:
            Random seed for reproducibility.
        alpha: scalar
            The value of alpha must be between 0 and 1.
        bandwidth: scalar:
            The value of bandwidth must be between 0 and 1.
        kernel: str
            The value of kernel must be "gaussian", ...
        R: int
            The number of random Fourier features.
        stat_type: str
            The value must be 'U' or 'V'; U-statistic or V-statistic.
        B: int
            The number of simulated test statistics to approximate the quantiles.
    '''

    start = time.time()  # Record computation time
    
    # Ensure reproducibility
    rng = np.random.default_rng(seed)

    ntot, d = Z.shape

    n = m = int(ntot/2)

    # Feature mapping
    inds, _ = nys_inds(Z, k, method, bandwidth, 0, seed)

    # Map data to Nyström feature space
    psi_Z = nystrom_features(Z, inds, bandwidth)

    # Permutation test
    I_1 = np.concatenate((np.ones(m), np.zeros(n)))
    I = np.tile(I_1, (B + 1, 1))  # (B+1) x (m+n)

    # Independent permutations along axis=1
    I_X = independent_permutation(I, rng, axis=1)
    I_X[0] = I_1  # Include the non-permuted case Z=(X,Y)
    I_Y = 1 - I_X

    bar_Z_B_piX = (1 / m) * I_X @ psi_Z  # (B+1) x 2R
    bar_Z_B_piY = (1 / n) * I_Y @ psi_Z  # (B+1) x 2R
    T = bar_Z_B_piX - bar_Z_B_piY  # (B+1) x 2R
    V = np.sum(T ** 2, axis=1)  # (B+1, )

    test_stats = V

    rMMD2 = test_stats[0]

    # take decision based on empirical threshold and level
    output, _ = decide(test_stats, rMMD2, B, alpha)

    dt = time.time() - start

    return output, dt, k