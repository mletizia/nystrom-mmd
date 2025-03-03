import re, os
from glob import glob
import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt

def plot_powervsvars(results, vars, config):
    # plot power (at sqrt(n)) vs sample size or separation parameter

    label_dict = {
                'uniform': r'Nyström-uniform (ours, $\ell=\sqrt{n}$)',
                'rlss': r'Nyström-AKRLS (ours, $\ell=\sqrt{n}$)',
                'rff': r'RFF  ($\ell=\sqrt{n}$)',
                'full_rank': 'Exact MMD',
    }

    color_dict = {
                'uniform': '#377eb8',
                'rlss': '#ff7f00',
                'rff': '#4daf4a',
                'full_rank': '#984ea3'
    }

    xlabel_dict={
                'higgs': 'n',
                'susy': 'n',
                'cg': r'$\rho_2$',
    }

    niter = config['niter']
    dataset = config['dataset']

    methods = results.keys()

    n_feat = 4 # index for num of features in the list K (4 is sqrt(n))

    plt.figure(figsize=(8, 5))

    # compute power for each method
    powers = {}
    for method in methods:
        powers[method] = np.asarray([power_interval(el, niter) for el in results[method][:,1, n_feat]])

        plt.plot(vars, powers[method][:,0], '-v', markersize=8, label=label_dict[method], c=color_dict[method])
        plt.fill_between(vars, 
                 powers[method][:,1], 
                 powers[method][:,2], 
                 alpha=0.2, color=color_dict[method])
        
    plt.ylabel(r'Power ($\alpha=0.05$)', fontsize=20)
    plt.xlabel(xlabel_dict[dataset], fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.locator_params(nbins=6, axis='x')
    plt.legend(loc='best', fontsize=15)
    plt.grid()
    plt.show()


def plot_powervscomp(results, var, config):
    # plot power (at sqrt(n)) vs sample size or separation parameter

    label_dict = {
                'uniform': r'Nyström-uniform (ours, $\ell=\sqrt{n}$)',
                'rlss': r'Nyström-AKRLS (ours, $\ell=\sqrt{n}$)',
                'rff': r'RFF  ($\ell=\sqrt{n}$)',
                'full_rank': 'Exact MMD',
    }

    color_dict = {
                'uniform': '#377eb8',
                'rlss': '#ff7f00',
                'rff': '#4daf4a',
                'full_rank': '#984ea3'
    }

    niter = config['niter']
    dataset = config['dataset']

    methods = results.keys()

    plt.figure(figsize=(8, 5))

    # compute power for each method
    powers_time = {}
    for method in methods:
        powers_time[method] = np.asarray([power_interval(el, niter) for el in results[method][var,1, :]])

        plt.plot(results[method][var,0,:], powers_time[method][:,0], '-v', markersize=8, label=label_dict[method], c=color_dict[method])
        plt.fill_between(results[method][var,0,:], 
                 powers_time[method][:,1], 
                 powers_time[method][:,2], 
                 alpha=0.2, color=color_dict[method])
        
    plt.ylabel(r'Power ($\alpha=0.05$)', fontsize=20)
    plt.xlabel('Computation time (s)', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.locator_params(nbins=6, axis='x')
    plt.legend(loc='best', fontsize=15)
    plt.xscale('log')
    plt.grid()
    plt.show()


def plot_powervsnfeat(results, var, config):
    # plot power vs number of random features

    label_dict = {
                'uniform': r'Nyström-uniform (ours)',
                'rlss': r'Nyström-AKRLS (ours)',
                'rff': r'RFF',
                'full_rank': 'Exact MMD',
    }

    color_dict = {
                'uniform': '#377eb8',
                'rlss': '#ff7f00',
                'rff': '#4daf4a',
                'full_rank': '#984ea3'
    }

    niter = config['niter']
    dataset = config['dataset']

    methods = results.keys()

    plt.figure(figsize=(8, 5))

    # compute power for each method
    powers_time = {}
    for method in methods:
        powers_time[method] = np.asarray([power_interval(el, niter) for el in results[method][var,1, :]])

        plt.plot(results[method][var,2,:], powers_time[method][:,0], '-v', markersize=8, label=label_dict[method], c=color_dict[method])
        plt.fill_between(results[method][var,2,:], 
                 powers_time[method][:,1], 
                 powers_time[method][:,2], 
                 alpha=0.2, color=color_dict[method])
        
    plt.ylabel(r'Power ($\alpha=0.05$)', fontsize=20)
    plt.xlabel(r'$\ell$', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.locator_params(nbins=6, axis='x')
    plt.legend(loc='best', fontsize=15)
    plt.xscale('log')
    plt.grid()
    plt.show()

# Function to estimate the width of the Gaussian kernel using pairwise distances
def median_pairwise(data):
    # this function estimates the width of the gaussian kernel.
    # use on a small sample (standardize first if necessary)
    pairw = pdist(data)  # Pairwise distance calculation (Euclidean by default)
    return np.median(pairw)  # Return the median of pairwise distances

# Function to check if a 'seeds.npy' file exists and load or generate seeds
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
    
    # If seed file exists, load it
    if os.path.exists(seed_file):
        print(f"Loading existing seeds from {seed_file}")
        seeds = np.load(seed_file)
    else:
        # If seed file doesn't exist, generate new seeds and save them
        print(f"Generating new seeds for {n_tests} test iterations")
        seeds = generate_seeds(n_tests)  
        np.save(seed_file, seeds)  # Save for reproducibility
        print(f"Saved new seeds to {seed_file}")

    return seeds

# Function to generate 'n' random seeds using a given or default seed
def generate_seeds(n, seed=None):
    rng = np.random.default_rng(seed)  # Create a random number generator with the initial seed
    seeds = rng.integers(0, 2**32, size=n, dtype=np.uint32)  # Generate n random seeds
    return seeds

# Function to generate a list of feature counts based on the square root of n
def list_num_features(n):
    # Calculate sqrt(n)
    sqrt_n = int(np.sqrt(n))

    # Generate three integers between 0 and sqrt(n) (exclusive)
    before_sqrt = np.linspace(0, sqrt_n, num=5, endpoint=False, dtype=int)[1:]

    # Combine all numbers into a single sorted list
    result = np.concatenate((before_sqrt, [sqrt_n], [2*sqrt_n], [5*sqrt_n]))

    return result

# Function to calculate the Wilson score interval for binomial proportions
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

# Function to calculate the Wilson score interval with an additional power interval return
def power_interval(p, n, confidence_level=0.95):
    """
    Calculate the Wilson score confidence interval for a binomial proportion.

    Parameters:
    p (float): Proportion of successes (rate of success).
    n (int): Number of trials.
    confidence_level (float): Confidence level for the interval (default is 0.95 for 95% confidence).

    Returns:
    tuple: Proportion, lower and upper bounds of the Wilson score interval.
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

# Function to decide whether to reject the null hypothesis based on a test statistic and null distribution
def decide(H0, t_obs, B, alpha):
    """
    Perform hypothesis testing decision.

    Args:
    H0 (array): Null distribution of test statistics.
    t_obs (float): Observed test statistic.
    B (int): Number of bootstrap samples.
    alpha (float): Significance level.

    Returns:
    output (int): 1 if null hypothesis is rejected, 0 otherwise.
    thr (float): Threshold value for decision.
    """
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

# Function to perform independent permutations on a given array along a specified axis
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


# Function to standardize data (zero mean and unit variance)
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
