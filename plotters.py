import numpy as np
from utils import power_interval

import matplotlib.pyplot as plt

def plot_powervsvars(results, vars, config, file=False):
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
    if file: plt.savefig(file)
    plt.show()


def plot_powervscomp(results, var, config, file=False):
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
    if file: plt.savefig(file)
    plt.show()


def plot_powervsnfeat(results, var, config, file=False):
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
    if file: plt.savefig(file)
    plt.show()