# Import necessary libraries
import numpy as np
import os

# Import custom utilities for Nyström permutation test, kernel parameter estimation, and dataset sampling
from tests import rMMDtest, MMDb_test, NysMMDtestV2
from sampler import generate_correlated_gaussians
from stat_utils import check_if_seeds_exist, standardize_data, median_pairwise

SQRT_2 = np.sqrt(2)

# Specify which tests to perform
which_tests = ["uniform",  "rlss", "rff"]  # Test types to run. Options: "fullrank", "uniform",  "rlss", "rff"

# Parameters for statistical testing
alpha = 0.05  # Significance level of the test
B = 199  # Number of permutations in the permutation test
n_tests = 400  # Number of tests to perform on different subsamples

# Parameters for dataset sampling
n = 2500  # Size of each sample. Total size = 2 * sample_size
d = 3
RHO2 = [0.51, 0.54, 0.57, 0.60, 0.63, 0.66]

ntot = 2*n


#K = list_num_features(ntot)
K = [14, 28 , 42, 56, 70, 140, 350]
print(f"Num. of features {K}")


# Main execution block
if __name__ == "__main__":
    print(f"CG experiments")  # Log the start of experiments


    for rho2 in RHO2:

        # Estimate the median pairwise distance for the RBF kernel parameter
        X_tune = generate_correlated_gaussians(500, d, rho1=.5, rho2=rho2, seed=None)
        X_tune = standardize_data(X_tune)
        sigmahat = median_pairwise(X_tune)  # Median pairwise distance as kernel bandwidth

        # Define output folder for storing results
        output_folder = f'./output_CG/ntot{ntot}_B{B+1}_niter{n_tests}_rho{rho2}'
        os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it does not exist
        
        # Initialize arrays to store test results for each method
        if "fullrank" in which_tests:
            os.makedirs(output_folder + '/fullrank/')
            output_full = np.zeros(shape=(n_tests, 3))  # For full-rank tests

        if "uniform" in which_tests:
            os.makedirs(output_folder + '/uniform/')
            output_uni = np.zeros(shape=(n_tests, len(K), 3))  # For uniform sampling Nyström tests

        if "rlss" in which_tests:
            os.makedirs(output_folder + '/rlss/')
            output_rlss = np.zeros(shape=(n_tests, len(K), 3))  # For uniform sampling Nyström tests

        if "rff" in which_tests:
            os.makedirs(output_folder + '/rff/')
            output_rff = np.zeros(shape=(n_tests, len(K), 3))  # For uniform sampling Nyström tests

        # Call function to check or generate seeds
        seeds = check_if_seeds_exist(output_folder, n_tests)

        # Perform tests over multiple iterations
        for test in range(n_tests):
            print(f"Test: {test + 1}/{n_tests} - ntot = {ntot}")  # Log the progress of tests

            # Assign the test seed
            test_seed = seeds[test]

            # Sample data subsets from the Higgs dataset for this iteration
            X = generate_correlated_gaussians(n, d, rho1=.5, rho2=rho2, seed=seeds[test])
            X = standardize_data(X)

            # Perform full-rank permutation test if specified
            if "fullrank" in which_tests:
                print("Fullrank test")
                output_full[test, :] = MMDb_test(X, sigmahat, seed=None, B=B, plot=False)

            # Perform uniform Nyström-based permutation test if specified
            if "uniform" in which_tests:
                print("Uniform test")
                for i, k in enumerate(K):
                    output_uni[test, i, :] = NysMMDtestV2(X, seed=test_seed, bandwidth=sigmahat, alpha=0.05, method='uniform', k=k, B=B)

            # Perform recursive LSS Nyström-based permutation test if specified
            if "rlss" in which_tests:
                print("RLSS test")
                for i, k in enumerate(K):
                    output_rlss[test, i, :] = NysMMDtestV2(X, seed=test_seed, bandwidth=sigmahat, alpha=0.05, method='rlss', k=k, B=B)

            if "rff" in which_tests:
                print("RFF test")
                for i, k in enumerate(K):
                    output_rff[test, i, :] = rMMDtest(X, seed=test_seed, bandwidth=SQRT_2*sigmahat, alpha=alpha, R=k, B=B)



        # Save results for each test type to the corresponding subdirectory
        if "fullrank" in which_tests:
            np.save(output_folder + f'/fullrank/results.npy', output_full)
        if "uniform" in which_tests:
            np.save(output_folder + f'/uniform/results.npy', output_uni)
        if "rlss" in which_tests:
            np.save(output_folder + f'/rlss/results.npy', output_rlss)
        if "rff" in which_tests:
            np.save(output_folder + f'/rff/results.npy', output_rff)