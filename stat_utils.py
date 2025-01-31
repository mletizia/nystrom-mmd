import re, os
from glob import glob
import numpy as np
from scipy.stats import beta, norm, binom
from scipy.spatial.distance import pdist

# np.array([wilson_score_interval(el, rff.shape[0], confidence_level=0.95) for el in rff_powers])


def median_pairwise(data):
    # this function estimates the width of the gaussian kernel.
    # use on a small sample (standardize first if necessary)
    pairw = pdist(data)
    return np.median(pairw)


def check_if_seeds_exist(output_folder, n_tests):
    """
    Checks if a seeds.npy file exists in the given output folder.
    - If it exists, loads and returns the seeds.
    - If it doesn't exist, generates new seeds, saves them, and returns them.

    Args:
        output_folder (str): Path to the folder where seeds.npy should be stored.
        n_tests (int): Number of test iterations.

    Returns:
        np.ndarray: Array of seeds.
    """
    seed_file = os.path.join(output_folder, "seeds.npy")
    
    if os.path.exists(seed_file):
        print(f"Loading existing seeds from {seed_file}")
        seeds = np.load(seed_file)
    else:
        print(f"Generating new seeds for {n_tests} test iterations")
        seeds = generate_seeds(n_tests)  
        np.save(seed_file, seeds)  # Save for reproducibility
        print(f"Saved new seeds to {seed_file}")

    return seeds



def generate_seeds(n, seed=None):
    rng = np.random.default_rng(seed)  # Create a random number generator with the initial seed
    seeds = rng.integers(0, 2**32, size=n, dtype=np.uint32)  # Generate n random seeds
    return seeds


def load_results(folder, method='uniform'):

    if method=='fullrank':
        files = []
        for file in glob(folder+"/*/"+method+"/results.npy"):
            files.append(file)
        files =  sorted(files, key=extract_ntot_fromstring)
        results = [np.load(el) for el in files]

        n_rep = np.shape(results)[1]

        time_pow_nfeat = [(el[:,1].mean(axis=0), el[:,0].mean(axis=0), el[:,2].mean(axis=0)) for el in results]

    else:
        files = []
        for file in glob(folder+"/*/"+method+"/results.npy"):
            files.append(file)
        files =  sorted(files, key=extract_ntot_fromstring)
        results = [np.load(el) for el in files]

        n_rep = np.shape(results)[1]

        time_pow_nfeat = [(el[:,:,1].mean(axis=0), el[:,:,0].mean(axis=0), el[:,:,2].mean(axis=0)) for el in results]

    return np.asarray(time_pow_nfeat)


def load_results_CG(folder, method='uniform'):

    if method=='fullrank':
        files = []
        for file in glob(folder+"/*/"+method+"/results.npy"):
            files.append(file)
        files =  sorted(files, key=extract_rho_fromstring)
        results = [np.load(el) for el in files]

        n_rep = np.shape(results)[1]

        time_pow_nfeat = [(el[:,1].mean(axis=0), el[:,0].mean(axis=0), el[:,2].mean(axis=0)) for el in results]

    else:
        files = []
        for file in glob(folder+"/*/"+method+"/results.npy"):
            files.append(file)
        files =  sorted(files, key=extract_rho_fromstring)
        results = [np.load(el) for el in files]

        n_rep = np.shape(results)[1]

        time_pow_nfeat = [(el[:,:,1].mean(axis=0), el[:,:,0].mean(axis=0), el[:,:,2].mean(axis=0)) for el in results]

    return np.asarray(time_pow_nfeat)

# def load_results(folder, method='uniform'):
#     files = []
#     for file in glob(folder+"/*/"+method+"/results.npy"):
#         files.append(file)
#     files =  sorted(files, key=extract_ntot_fromstring)

#     for file in files:
#         result = np.load(file)
#         n_rep = np.shape(result)[0]
#         power = power_interval(result[:,:,0].mean(axis=0), n_rep)

#     results = [np.load(el) for el in files]

#     n_rep = np.shape(results)[1]

#     time_pow_nfeat = [(el[:,:,1].mean(axis=0), el[:,:,0].mean(axis=0), el[:,:,2].mean(axis=0)) for el in results]

#     return np.asarray(time_pow_nfeat)


def extract_ntot_fromstring(s):
    match = re.search(r'ntot(\d+)_', s)  # Find digits between 'n' and '_'
    return int(match.group(1)) if match else 0  # Convert to integer for proper sorting

def extract_rho_fromstring(s):
    match = re.search(r'rho([\d\.]+)', s)  # Find digits between 'n' and '_'
    return float(match.group(1)) if match else 0.0  # Convert to integer for proper sorting

def list_num_features(n):
    # Calculate sqrt(n)
    sqrt_n = int(np.sqrt(n))

    # Generate three integers between 0 and sqrt(n) (exclusive)
    before_sqrt = np.linspace(0, sqrt_n, num=5, endpoint=False, dtype=int)[1:]

    # Combine all numbers into a single sorted list
    result = np.concatenate((before_sqrt, [sqrt_n], [2*sqrt_n], [5*sqrt_n]))

    return result


# def jeffreys_interval(p, n, confidence_level=0.95):
#     # Calculate the number of successes x from the estimated proportion p
#     x = int(p * n)
    
#     # Jeffrey's prior parameters
#     alpha = x + 0.5
#     beta_ = n - x + 0.5
    
#     # Calculate the quantiles of the Beta distribution for the confidence interval
#     lower_bound = beta.ppf((1 - confidence_level) / 2, alpha, beta_)
#     upper_bound = beta.ppf(1 - (1 - confidence_level) / 2, alpha, beta_)
    
#     # returning bar size rather than lower and upper bounds
#     return p-lower_bound, upper_bound-p

# def binomial_confidence_interval(p, n, confidence_level=0.95):
#     # Convert the confidence level to alpha
#     alpha = 1 - confidence_level
    
#     # Calculate the quantiles for the given confidence level based on the estimated p (p_hat)
#     lower_quantile = binom.ppf(alpha / 2, n, p)  # Lower quantile for p_hat
#     upper_quantile = binom.ppf(1 - alpha / 2, n, p)  # Upper quantile for p_hat
    
#     # Convert the quantiles to the confidence interval for p
#     p_lower = lower_quantile / n
#     p_upper = upper_quantile / n
    
#     # returning bar size rather than lower and upper bounds
#     return p-p_lower, p_upper-p

# def clopper_pearson_interval(p, n, confidence_level=0.95):
#     """
#     Calculate the Clopper-Pearson confidence interval for a binomial proportion.
    
#     Parameters:
#     p (float): Proportion of successes (rate of success).
#     n (int): Number of trials.
#     confidence_level (float): Confidence level for the interval (default is 0.95 for 95% confidence).
    
#     Returns:
#     tuple: Lower and upper bounds of the Clopper-Pearson interval.
#     """
#     if n == 0:
#         raise ValueError("Number of trials (n) must be greater than 0.")
    
#     if not (0 <= p <= 1):
#         raise ValueError("Proportion (p) must be between 0 and 1.")
    
#     # Calculate the alpha for the confidence level
#     alpha = 1 - confidence_level
    
#     # The quantiles of the Beta distribution
#     lower_bound = beta.ppf(alpha / 2, p * n, (1 - p) * n)
#     upper_bound = beta.ppf(1 - alpha / 2, p * n, (1 - p) * n)

#     # Ensure the bounds are within the [0, 1] range
#     lower_bound = max(0, lower_bound)
#     upper_bound = min(1, upper_bound)
    
#     # returning bar size rather than lower and upper bounds
#     return p-lower_bound, upper_bound-p



def wilson_score_interval(p, n, confidence_level=0.95):
    """
    Calculate the Wilson score confidence interval for a binomial proportion.

    Parameters:
    p (float): Proportion of successes (rate of success).
    n (int): Number of trials.
    confidence_level (float): Confidence level for the interval (default is 0.95 for 95% confidence).

    Returns:
    tuple: Lower and upper bounds of the Wilson score interval.
    """
    if n == 0:
        raise ValueError("Number of trials (n) must be greater than 0.")

    if not (0 <= p <= 1):
        raise ValueError("Proportion (p) must be between 0 and 1.")

    # Z-value for the given confidence level
    alpha = 1 - confidence_level
    z = norm.ppf(1 - alpha / 2)

    # Wilson score calculation
    denominator = 1 + (z ** 2 / n)
    center = (p + (z ** 2 / (2 * n))) / denominator
    margin = (z * np.sqrt((p * (1 - p) / n) + (z ** 2 / (4 * n ** 2)))) / denominator

    lower_bound = max(0, center - margin)
    upper_bound = min(1, center + margin)

    return lower_bound, upper_bound

def power_interval(p, n, confidence_level=0.95):
    """
    Calculate the Wilson score confidence interval for a binomial proportion.

    Parameters:
    p (float): Proportion of successes (rate of success).
    n (int): Number of trials.
    confidence_level (float): Confidence level for the interval (default is 0.95 for 95% confidence).

    Returns:
    tuple: Lower and upper bounds of the Wilson score interval.
    """
    if n == 0:
        raise ValueError("Number of trials (n) must be greater than 0.")

    if not (0 <= p <= 1):
        raise ValueError("Proportion (p) must be between 0 and 1.")

    # Z-value for the given confidence level
    alpha = 1 - confidence_level
    z = norm.ppf(1 - alpha / 2)

    # Wilson score calculation
    denominator = 1 + (z ** 2 / n)
    center = (p + (z ** 2 / (2 * n))) / denominator
    margin = (z * np.sqrt((p * (1 - p) / n) + (z ** 2 / (4 * n ** 2)))) / denominator

    lower_bound = max(0, center - margin)
    upper_bound = min(1, center + margin)

    return p, lower_bound, upper_bound



def decide(H0, t_obs, B, alpha):

    # Sort null distribution
    H0_sorted = np.sort(H0)
    
    # Determine threshold based on significance level
    thr_ind = int(np.ceil((B + 1) * (1 - alpha))) - 1  # Index of alpha-level threshold
    thr = H0_sorted[thr_ind]

    # Handle cases where observed test statistic equals the threshold
    if t_obs == thr:
        greater = np.sum(H0_sorted > t_obs)  # Count values greater than t_obs
        equal = np.sum(H0_sorted == t_obs)  # Count values equal to t_obs
        a_prob = (alpha * (B + 1) - greater) / equal  # Adjusted probability

        # Randomly decide outcome based on adjusted probability
        output = np.random.default_rng(seed=None).choice([1, 0], p=[a_prob, 1 - a_prob])
    else:
        # Return 1 (reject null) if t_obs > thr, otherwise 0 (fail to reject null)
        output = int(t_obs > thr)

    return output, thr

# def decide_simple(H0, t_obs, B, alpha):

#     # Sort null distribution
#     H0_sorted = np.sort(H0)
    
#     # Determine threshold based on significance level
#     thr_ind = int(np.ceil((B + 1) * (1 - alpha))) - 1  # Index of alpha-level threshold
#     thr = H0_sorted[thr_ind]

#     return int(t_obs > thr), thr


def independent_permutation(I, rng, axis=1):
    """
    Perform independent permutations along a specified axis.
    Parameters:
        I: array_like
            Input array to permute.
        rng: np.random.Generator
            Random number generator.
        axis: int
            Axis along which to apply independent permutations (0 or 1).
    Returns:
        permuted: array_like
            The array with independent permutations applied along the specified axis.
    """
    permuted = np.empty_like(I)
    if axis == 1:  # Row-wise independent permutation
        for i in range(I.shape[0]):
            permuted[i] = rng.permutation(I[i])
    elif axis == 0:  # Column-wise independent permutation
        for i in range(I.shape[1]):
            permuted[:, i] = rng.permutation(I[:, i])
    else:
        raise ValueError("Axis must be 0 or 1.")
    return permuted


def standardize_data(data):
    """
    Standardize the given data (zero mean and unit variance).
    
    Parameters:
        data (numpy.ndarray): The data to be standardized, with samples as rows and features as columns.
    
    Returns:
        numpy.ndarray: Standardized data with the same shape as input.
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    return standardized_data