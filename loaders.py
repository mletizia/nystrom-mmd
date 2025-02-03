from glob import glob
import numpy as np
import re

def load_results(folder, data, method='uniform'):
    if data == 'susy' or 'hiss':
        return load_results_higgs(folder, method=method)
    
    elif data == "cg":
        return load_results_CG(folder, method=method)
    


# Function to load results from a folder based on the method type
def load_results_higgs(folder, method='uniform'):

    # Handle 'fullrank' method
    if method == 'fullrank':
        files = []
        for file in glob(folder+"/*/"+method+"/results.npy"):
            files.append(file)
        files = sorted(files, key=extract_ntot_fromstring)  # Sort based on ntot in filename
        results = [np.load(el) for el in files]

        n_rep = np.shape(results)[1]  # Get the number of repetitions

        # Calculate average time, power, and number of features for each result
        time_pow_nfeat = [(el[:,1].mean(axis=0), el[:,0].mean(axis=0), el[:,2].mean(axis=0)) for el in results]

    else:  # Handle other methods
        files = []
        for file in glob(folder+"/*/"+method+"/results.npy"):
            files.append(file)
        files = sorted(files, key=extract_ntot_fromstring)  # Sort based on ntot in filename
        results = [np.load(el) for el in files]

        n_rep = np.shape(results)[1]  # Get the number of repetitions

        # Calculate average time, power, and number of features for each result
        time_pow_nfeat = [(el[:,:,1].mean(axis=0), el[:,:,0].mean(axis=0), el[:,:,2].mean(axis=0)) for el in results]

    return np.asarray(time_pow_nfeat)

# Similar function to load results, but specific for 'CG' method
def load_results_CG(folder, method='uniform'):

    if method == 'fullrank':
        files = []
        for file in glob(folder+"/*/"+method+"/results.npy"):
            files.append(file)
        files = sorted(files, key=extract_rho_fromstring)  # Sort based on rho in filename
        results = [np.load(el) for el in files]

        n_rep = np.shape(results)[1]  # Get the number of repetitions

        # Calculate average time, power, and number of features for each result
        time_pow_nfeat = [(el[:,1].mean(axis=0), el[:,0].mean(axis=0), el[:,2].mean(axis=0)) for el in results]

    else:  # Handle other methods
        files = []
        for file in glob(folder+"/*/"+method+"/results.npy"):
            files.append(file)
        files = sorted(files, key=extract_rho_fromstring)  # Sort based on rho in filename
        results = [np.load(el) for el in files]

        n_rep = np.shape(results)[1]  # Get the number of repetitions

        # Calculate average time, power, and number of features for each result
        time_pow_nfeat = [(el[:,:,1].mean(axis=0), el[:,:,0].mean(axis=0), el[:,:,2].mean(axis=0)) for el in results]

    return np.asarray(time_pow_nfeat)

# Helper function to extract 'ntot' value from filenames
def extract_ntot_fromstring(s):
    match = re.search(r'ntot(\d+)_', s)  # Find digits between 'n' and '_'
    return int(match.group(1)) if match else 0  # Convert to integer for proper sorting

# Helper function to extract 'rho' value from filenames
def extract_rho_fromstring(s):
    match = re.search(r'rho([\d\.]+)', s)  # Find digits between 'n' and '_'
    return float(match.group(1)) if match else 0.0  # Convert to float for proper sorting