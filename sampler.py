
import numpy as np
import h5py, pickle


def generate_correlated_gaussians(n, d, rho1, rho2, seed=None):
    """
    Generate two samples from correlated Gaussian distributions.
    
    Parameters
    ----------
    n : int
        Number of samples in each distribution.
    d : int
        Dimensionality of the data.
    rho : float
        Correlation coefficient between dimensions (-1 to 1).
    seed : int, optional
        Random seed for reproducibility. Default is None.
    
    Returns
    -------
    X : array_like, shape (n, d)
        Samples from the first Gaussian distribution.
    Y : array_like, shape (n, d)
        Samples from the second Gaussian distribution.
    """

    rng = np.random.default_rng(seed)
    
    # Define covariance matrix for the correlated Gaussians
    cov1 = (1 - rho1) * np.eye(d) + rho1 * np.ones((d, d))
    cov2 = (1 - rho2) * np.eye(d) + rho2 * np.ones((d, d))

    
    # Means of the two distributions
    mean = np.zeros(d)
    
    # Generate samples
    X = np.zeros((2*n,d))
    X[:n] = rng.multivariate_normal(mean, cov1, size=n)
    X[n:] = rng.multivariate_normal(mean, cov2, size=n)
    
    return X


# def generate_correlated_gaussians_old(n, d, rho, delta_mu=0.0, seed=None):
#     """
#     Generate two samples from correlated Gaussian distributions.
    
#     Parameters
#     ----------
#     n : int
#         Number of samples in each distribution.
#     d : int
#         Dimensionality of the data.
#     rho : float
#         Correlation coefficient between dimensions (-1 to 1).
#     delta_mu : array_like, shape (d,)
#         Shift vector to add to the mean of the second Gaussian.
#     seed : int, optional
#         Random seed for reproducibility. Default is None.
    
#     Returns
#     -------
#     X : array_like, shape (n, d)
#         Samples from the first Gaussian distribution.
#     Y : array_like, shape (n, d)
#         Samples from the second Gaussian distribution.
#     """

#     rng = np.random.default_rng(seed)
    
#     # Define covariance matrix for the correlated Gaussians
#     cov = (1 - rho) * np.eye(d) + rho * np.ones((d, d))
    
#     # Means of the two distributions
#     mean_X = np.zeros(d)
#     mean_Y = np.add(np.zeros(d), delta_mu)
    
#     # Generate samples
#     X = np.zeros((2*n,d))
#     X[:n] = rng.multivariate_normal(mean_X, cov, size=n)
#     X[n:] = rng.multivariate_normal(mean_Y, cov, size=n)
    
#     return X

def generate_samples_gaussians(n=1000, d=1, mean_diff=0.2, std2_mult=1, seed = 0):
    """
    Generates two sets of multivariate Gaussians in dimension d, with identity covariances
      and means that are mean_diff apart
    """

    rng = np.random.default_rng(seed=seed)

    mean_1 = np.zeros(d)
    mean_2 = np.zeros(d)
    mean_1[d-1] = mean_diff/2
    mean_2[d-1] = -mean_diff/2
    cov_1 = np.eye(d)
    cov_2 = np.eye(d)*std2_mult

    # print(
    #      mean_1,
    #      mean_2,
    #      cov_1,
    #      cov_2
    # )

    X = np.zeros(shape=(2*n,d))
    
    X[:n] = rng.multivariate_normal(mean_1, cov_1, n) 
    X[n:] = rng.multivariate_normal(mean_2, cov_2, n)

    #Y = np.zeros(shape=(2*n,1))
    #Y[n:] = np.ones((n,1))

    return X

def gen_rff_gaussians_loc(n=1000, d=500, seed = 0):
    """
    Generates two sets of multivariate Gaussians in dimension d, with identity covariances
      and means that are mean_diff apart
    """

    rng = np.random.default_rng(seed=seed)

    mean_1 = np.zeros(d)
    mean_2 = np.zeros(d)
    mean_2[:20] = 0.1
    cov_1 = np.eye(d)
    cov_2 = np.eye(d)

    X = np.zeros(shape=(2*n,d))
    
    X[:n] = rng.multivariate_normal(mean_1, cov_1, n) 
    X[n:] = rng.multivariate_normal(mean_2, cov_2, n)

    #Y = np.zeros(shape=(2*n,1))
    #Y[n:] = np.ones((n,1))

    return X

def read_data_higgs(file_name, reduced=0):
        print(f"Loading Higgs data - reduced: {reduced}")
        with h5py.File(file_name, "r") as h5py_file:
            arr = np.array(h5py_file["X"], dtype=np.float64).T
        if reduced == 1: X = arr[:, [8,12,16,20]]
        elif reduced == 2: X = arr[:, [8,12]]
        else: X = arr[:, 1:15] # low-level only
        Y = arr[:, 0]
        print("Done")
        return X, Y
        

# def read_data_higgs(file_name, reduced=0):
#     print(f"Loading Higgs data from CSV - reduced: {reduced}")
    
#     # Load data from CSV using NumPy
#     arr = np.loadtxt(file_name, delimiter=",", dtype=np.float64)
    
#     # Selecting the features based on the 'reduced' parameter
#     if reduced == 1:
#         X = arr[:, [8, 12, 16, 20]]
#     elif reduced == 2:
#         X = arr[:, [8, 12]]
#     else:
#         X = arr[:, 1:15]  # Low-level features only
    
#     Y = arr[:, 0]  # Target variable
    
#     print("Done")
#     return X, Y


# def load_higgs_tst(file, reduced=0):
#     with open(file, 'rb') as f:
#         data = pickle.load(f)
    
#     if reduced==0:
#         X = np.vstack((data[0][:, :-1], data[1][:, :-1]))
#         Y = np.hstack((data[0][:, -1], data[1][:, -1]))
#     elif reduced==1:
#         X = np.vstack((data[0][:, 0:2], data[1][:, 0:2]))
#         Y = np.hstack((data[0][:, -1], data[1][:, -1]))
    
#     return X, Y


def read_data_susy(file_name):
        print("Loading Susy data")
        with h5py.File(file_name, "r") as h5py_file:
            arr = np.array(h5py_file["X"], dtype=np.float64).T
        X = arr[:, 1:9] # low-level only
        Y = arr[:, 0]
        print("Done")
        return X, Y


# def sample_higgs_dataset(X, Y, n, seed_0=None, seed_1=None, shuffle=False):
#     """
#     Samples n instances from class 0 and class 1 in the Higgs dataset.

#     Parameters:
#     - X: np.ndarray
#         Array of features.
#     - Y: np.ndarray
#         Array of labels (0 for background, 1 for signal).
#     - n: int
#         Number of instances to sample from each class.
#     - seed_0: int, optional
#         Random seed for sampling class 0.
#     - seed_1: int, optional
#         Random seed for sampling class 1.

#     Returns:
#     - sampled_X: np.ndarray
#         Sampled feature array.
#     - sampled_Y: np.ndarray
#         Sampled label array.
#     """

#     # Set random seeds if provided
#     if seed_0 is not None:
#         rng = np.random.default_rng(seed_0)
    
#     # Find indices of each class
#     class_0_indices = np.where(Y == 0)[0]
#     class_1_indices = np.where(Y == 1)[0]

#     # Sample n instances from class 0
#     if len(class_0_indices) < n:
#         raise ValueError(f"Not enough instances in class 0 to sample {n}. Available: {len(class_0_indices)}")
    
#     sampled_class_0_indices = rng.choice(class_0_indices, n, replace=False)

#     # Set random seed for class 1 sampling
#     if seed_1 is not None:
#         rng = np.random.default_rng(seed_1)

#     # Sample n instances from class 1
#     if len(class_1_indices) < n:
#         raise ValueError(f"Not enough instances in class 1 to sample {n}. Available: {len(class_1_indices)}")
    
#     sampled_class_1_indices = rng.choice(class_1_indices, n, replace=False)

#     # Prepare output arrays
#     sampled_X = np.empty((2 * n, X.shape[1]), dtype=X.dtype)
#     sampled_Y = np.empty(2 * n, dtype=Y.dtype)

#     # Fill sampled arrays directly
#     sampled_X[:n] = X[sampled_class_0_indices]
#     sampled_X[n:] = X[sampled_class_1_indices]
#     sampled_Y[:n] = Y[sampled_class_0_indices]
#     sampled_Y[n:] = Y[sampled_class_1_indices]

#     # Shuffle the combined dataset in-place
#     indices = np.arange(sampled_X.shape[0])
#     if shuffle:
#         np.random.default_rng().shuffle(indices)

#     return sampled_X[indices], sampled_Y[indices]

def sample_higgs_susy_dataset(Z, Y, n, alpha_mix=1, seed=None):
    """
    Samples n instances from class 0 and n instances from a mixture of class 0 and class 1.
    The first n examples will have labels 0, and the second n examples will have labels 1.

    Parameters:
    - Z: np.ndarray
        Array of features.
    - Y: np.ndarray
        Array of labels (0 for background, 1 for signal).
    - n: int
        Number of instances to sample.
    - alpha_mix: float, optional (default=0.5)
        Proportion of class 1 in the mixture. (1-alpha_mix) will be the proportion of class 0.
    - seed_0: int, optional
        Random seed for sampling class 0.
    - seed_mix: int, optional
        Random seed for sampling the mixed set.

    Returns:
    - sampled_Z: np.ndarray
        Sampled feature array.
    - sampled_Y: np.ndarray
        Sampled label array with overridden labels (first n are 0, second n are 1).
    """
    if not (0 <= alpha_mix <= 1):
        raise ValueError("alpha_mix must be between 0 and 1.")

    # Set random seeds if provided
    rng = np.random.default_rng(seed)

    # Find indices of each class
    class_0_indices = np.where(Y == 0)[0]
    class_1_indices = np.where(Y == 1)[0]

    # Sample n instances from class 0
    if len(class_0_indices) < n:
        raise ValueError(f"Not enough instances in class 0 to sample {n}. Available: {len(class_0_indices)}")
    sampled_class_0_indices = rng.choice(class_0_indices, n, replace=False)

    # Sample n instances for the mixed set
    num_class_1_in_mix = int(n * alpha_mix)
    num_class_0_in_mix = n - num_class_1_in_mix

    # remove indeces from class 0 that have been already sampled
    reduced_class_0_indices = np.setdiff1d(class_0_indices, sampled_class_0_indices, assume_unique=True)

    if len(reduced_class_0_indices) < num_class_0_in_mix:
        raise ValueError(f"Not enough instances in class 0 for the mixture. Available: {len(reduced_class_0_indices)}")
    if len(class_1_indices) < num_class_1_in_mix:
        raise ValueError(f"Not enough instances in class 1 for the mixture. Available: {len(class_1_indices)}")

    mixed_class_0_indices = rng.choice(reduced_class_0_indices, num_class_0_in_mix, replace=False)
    mixed_class_1_indices = rng.choice(class_1_indices, num_class_1_in_mix, replace=False)
    mixed_indices = np.concatenate([mixed_class_0_indices, mixed_class_1_indices])

    # Prepare output array
    sampled_Z = np.empty((2 * n, Z.shape[1]), dtype=Z.dtype)

    # Fill sampled arrays directly
    sampled_Z[:n] = Z[sampled_class_0_indices]
    sampled_Z[n:] = Z[mixed_indices]

    return sampled_Z







def load_mnist():
    """
    Returns P and Q_list where P consists of images of all digits 
    in mnist_7x7.data, and Q_list contains 5 elements each consisting
    of images of fewer digits.
    This function should only be run after download_mnist().
    """
    with open('./data/MNIST/mnist_7x7.data', 'rb') as handle:
        X = pickle.load(handle)
    P  = np.vstack(
        (X['0'], X['1'], X['2'], X['3'], X['4'], X['5'], X['6'], X['7'], X['8'], X['9'])
    )
    Q1 = np.vstack((X['1'], X['3'], X['5'], X['7'], X['9']))
    Q2 = np.vstack((X['0'], X['1'], X['3'], X['5'], X['7'], X['9']))
    Q3 = np.vstack((X['0'], X['1'], X['2'], X['3'], X['5'], X['7'], X['9']))
    Q4 = np.vstack((X['0'], X['1'], X['2'], X['3'], X['4'], X['5'], X['7'], X['9']))
    Q5 = np.vstack((X['0'], X['1'], X['2'], X['3'], X['4'], X['5'], X['6'], X['7'], X['9']))
    Q_list = [Q1, Q2, Q3, Q4, Q5]
    return P, Q_list

def generate_MNIST_samples(P, Q, n, seed=0):

    rng = np.random.default_rng(seed=seed)

    d = P.shape[1]
    
    X = np.zeros(shape=(2*n,d))
    
    X[:n] = rng.choice(P, n)
    X[n:] = rng.choice(Q, n)

    Y = np.zeros(shape=(2*n,1))
    Y[n:] = np.ones((n,1))

    return X, Y


def load_miniboone(file_path):
    """
    Loads the MiniBooNE dataset from a text file.
    
    Parameters:
        file_path (str): Path to the dataset file.
    
    Returns:
        X (numpy.ndarray): Feature matrix of shape (n, 50).
        Y (numpy.ndarray): Label vector of shape (n,), where 1 = signal, 0 = background.
    """

    print("Loading MiniBooNE data")

    # Read the first line to get the number of signal and background events
    with open(file_path, "r") as f:
        first_line = f.readline().strip().split()
        n_signal, n_background = int(first_line[0]), int(first_line[1])
    
    # Load the remaining data (50 features per event)
    X = np.loadtxt(file_path, skiprows=1)

    # Generate labels: 1 for signal (first n_signal rows), 0 for background (remaining rows)
    Y = np.hstack([np.ones(n_signal), np.zeros(n_background)])

    print("Done")


    return X, Y


def sample_minoboone(X, Y, m, alpha_mix, seed=None):
    """
    Creates a sample Z of shape (m, d) where:
      - The first m/2 points are only background (Y = 0).
      - The second m/2 points are a mixture of background and signal with mixing parameter alpha_mix.
        If alpha_mix = 1, the second half is only signal.
        If alpha_mix = 0, the second half is only background.
    
    Parameters:
        X (numpy.ndarray): Feature matrix of shape (n, d).
        Y (numpy.ndarray): Label vector of shape (n,).
        m (int): Number of points in the sample (assumed to be even).
        alpha_mix (float): Mixing ratio of signal in the second half (0 ≤ alpha_mix ≤ 1).
    
    Returns:
        Z (numpy.ndarray): Sampled dataset of shape (m, d).
    """

    rng = np.random.default_rng(seed)

    # Select background (Y = 0) and signal (Y = 1) events
    X_bkg = X[Y == 0]
    X_sig = X[Y == 1]

    # Sample first m/2 points from background
    idx_bkg = rng.choice(len(X_bkg), size=m, replace=False)
    Z_bkg = X_bkg[idx_bkg]

    # Handle the second half based on alpha_mix
    if alpha_mix == 1:
        # Second half is only signal
        Z_mixed = X_sig[rng.choice(len(X_sig), size=m, replace=False)]
    elif alpha_mix == 0:
        # Second half is only background, ensure no overlap with Z_bkg
        idx_bkg_mix = rng.choice(np.setdiff1d(np.arange(len(X_bkg)), idx_bkg), size=m, replace=False)
        Z_mixed = X_bkg[idx_bkg_mix]
    else:
        # Mix signal and background based on alpha_mix
        num_signal = int(m * alpha_mix)
        num_background = m - num_signal
        
        # Ensure no overlap in the background portion of the mixed sample
        idx_bkg_mix = rng.choice(np.setdiff1d(np.arange(len(X_bkg)), idx_bkg), size=num_background, replace=False)
        
        Z_sig = X_sig[rng.choice(len(X_sig), size=num_signal, replace=False)]
        Z_bkg_mix = X_bkg[idx_bkg_mix]
        
        # Stack and shuffle the mixed portion
        Z_mixed = np.vstack((Z_sig, Z_bkg_mix))
        rng.shuffle(Z_mixed)

    # Concatenate background-only and mixed samples
    Z = np.vstack((Z_bkg, Z_mixed))

    return Z