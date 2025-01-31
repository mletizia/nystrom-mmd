import numpy as np
import os

# Import custom utilities for Nyström permutation test, kernel parameter estimation, and dataset sampling
from tests import rMMDtest, MMDb_test, NysMMDtest
from sampler import sample_higgs_susy_dataset, read_data_higgs
from stat_utils import list_num_features, check_if_seeds_exist, median_pairwise

# Define constant for scaling
SQRT_2 = np.sqrt(2)

# Specify which tests to perform
which_tests = ["uniform", "rlss", "rff"]  # Test types to run

# Parameters for statistical testing
alpha = 0.05  # Significance level of the test
B = 199  # Number of permutations in the permutation test
n_tests = 400  # Number of tests to perform

# Parameters for dataset sampling
sample_sizes = [500, 2500, 5000, 10000, 20000, 30000, 40000]  # Sample sizes
lambda_mix = 0.2  # Proportion of class 1 in the mixture
reduced = 0  # Reduction mode for dataset

# Main execution block
if __name__ == "__main__":
    print("Higgs experiments")  # Log the start of experiments

    # Load dataset
    X_all, Y_all = read_data_higgs("/data/DATASETS/HIGGS_UCI/Higgs.mat", reduced=reduced)

    # Estimate kernel bandwidth
    X_tune = sample_higgs_susy_dataset(X_all, Y_all, 1000, alpha_mix=lambda_mix, seed=None)
    sigmahat = median_pairwise(X_tune)

    # Iterate over different sample sizes
    for n in sample_sizes:
        ntot = 2 * n
        K = list_num_features(ntot)
        print(f"Num. of features {K}")

        # Define output folder
        output_folder = f'./output_higgs/ntot{ntot}_B{B+1}_niter{n_tests}_mix{lambda_mix}_reduced{reduced}'
        os.makedirs(output_folder, exist_ok=True)

        # Initialize arrays for storing test results
        if "fullrank" in which_tests:
            os.makedirs(output_folder + '/fullrank/')
            output_full = np.zeros(shape=(n_tests, 3))

        if "uniform" in which_tests:
            os.makedirs(output_folder + '/uniform/')
            output_uni = np.zeros(shape=(n_tests, len(K), 3))

        if "rlss" in which_tests:
            os.makedirs(output_folder + '/rlss/')
            output_rlss = np.zeros(shape=(n_tests, len(K), 3))

        if "rff" in which_tests:
            os.makedirs(output_folder + '/rff/')
            output_rff = np.zeros(shape=(n_tests, len(K), 3))

        # Generate or retrieve seeds
        seeds = check_if_seeds_exist(output_folder, n_tests)

        # Run tests
        for test in range(n_tests):
            print(f"Test: {test + 1}/{n_tests} - ntot = {ntot}")
            test_seed = seeds[test]
            X = sample_higgs_susy_dataset(X_all, Y_all, n, alpha_mix=lambda_mix, seed=test_seed)

            if "fullrank" in which_tests:
                print("Fullrank test")
                output_full[test, :] = MMDb_test(X, sigmahat, seed=None, B=199, plot=False)

            if "uniform" in which_tests:
                print("Uniform test")
                for i, k in enumerate(K):
                    output_uni[test, i, :] = NysMMDtest(X, seed=test_seed, bandwidth=sigmahat, alpha=0.05, method='uniform', k=k, B=B)

            if "rlss" in which_tests:
                print("RLSS test")
                for i, k in enumerate(K):
                    output_rlss[test, i, :] = NysMMDtest(X, seed=test_seed, bandwidth=sigmahat, alpha=0.05, method='rlss', k=k, B=B)

            if "rff" in which_tests:
                print("RFF test")
                for i, k in enumerate(K):
                    output_rff[test, i, :] = rMMDtest(X, seed=test_seed, bandwidth=SQRT_2*sigmahat, alpha=alpha, R=k, B=B)

        # Save test results
        if "fullrank" in which_tests:
            np.save(output_folder + '/fullrank/results.npy', output_full)
        if "uniform" in which_tests:
            np.save(output_folder + '/uniform/results.npy', output_uni)
        if "rlss" in which_tests:
            np.save(output_folder + '/rlss/results.npy', output_rlss)
        if "rff" in which_tests:
            np.save(output_folder + '/rff/results.npy', output_rff)
