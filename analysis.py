import numpy as np
from loaders import load_results
from utils import power_interval

from matplotlib import pyplot as plt

import argparse


def main():

    parser = argparse.ArgumentParser()

    # Required named arguments
    parser.add_argument('--data', type=str, help='Input dataset (choose one): "higgs", "susy", "cg".')   
    parser.add_argument('--tests', nargs='+', type=str, default=["uniform", "rlss", "rff"], help='Input tests: "uniform", "rlss", "rff".')
    parser.add_argument('--N', default=400 , type=int, help='Number of repetitions')


    args = parser.parse_args()

    POWERS = []

    for test in args.tests:

        results, config = load_results(args.folder, data=args.data, method=test)

        POWERS.append(np.asarray([power_interval(el, args.N) for el in results[:,1, 4]])) # 4 are the results for sqrt(n)


    plt.figure(figsize=(8, 5))
    plt.plot(n, uniform_powers[:,0], '-v', markersize=8, label=r'Nyström-uniform (ours, $\ell=\sqrt{n}$)', c='#377eb8')
    plt.fill_between(n, 
                    uniform_powers[:,1], 
                    uniform_powers[:,2], 
                    alpha=0.2, color='#377eb8')
    plt.plot(n, rlss_powers[:,0], '-p', markersize=8, label=r'Nyström-AKRLS (ours, $\ell=\sqrt{n}$)', c='#ff7f00')
    plt.fill_between(n, 
                    rlss_powers[:,1], 
                    rlss_powers[:,2], 
                    alpha=0.2, color='#ff7f00')
    plt.plot(n, rff_powers[:,0], '-x', markersize=8, label=r'RFF  ($\ell=\sqrt{n}$)', c='#4daf4a')
    plt.fill_between(n, 
                    rff_powers[:,1], 
                    rff_powers[:,2], 
                    alpha=0.2, color='#4daf4a')
    # plt.plot(fullrank_time_pow_nft[:,-1], fullrank_power[:,0], '-o', markersize=8, label=r'Full rank', c='#984ea3')
    # plt.fill_between(fullrank_time_pow_nft[:,-1], 
    #                  fullrank_power[:,1], 
    #                  fullrank_power[:,2], 
    #                  alpha=0.5, color='#984ea3')
    plt.ylabel(r'Power ($\alpha=0.05$)', fontsize=20)
    plt.xlabel(r'$n$', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.locator_params(nbins=6, axis='x')
    #plt.xscale('log')
    plt.legend(loc=2, fontsize=15)
    #plt.title("Correlated Gaussians", fontsize=20)
    #plt.xlim(0.5,0.67)
    plt.grid()
    plt.show()