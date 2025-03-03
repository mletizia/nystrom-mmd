from glob import glob
import numpy as np
import re, ast
from pathlib import Path



# Similar function to load results, but specific for 'CG' data
def load_results(folder, methods=['uniform','rff','rlss']):

    config = return_parameters(folder)

    results_dict = {}

    for method in methods:

        if method == 'fullrank':
            print(f"loading {method}")
            files = glob(folder+"/*/fullrank/results.npy", recursive=True)
            print(files)
            files = sorted(files, key=extract_var_fromstring)  # Sort based on var parameter in filename
            results = [np.load(el) for el in files]
            sorted_vars = [extract_var_fromstring(file) for file in files]

            # Calculate average time, power, and number of features for each result
            time_pow_nfeat = np.asarray([(el[:,1].mean(axis=0), el[:,0].mean(axis=0), el[:,2].mean(axis=0)) for el in results])

            results_dict[method] = time_pow_nfeat

        else:  # Handle other methods
            print(f"loading {method}")
            files = glob(folder+"/*/"+method+"/results.npy", recursive=True)
            print(files)
            files = sorted(files, key=extract_var_fromstring)  # Sort based on var parameter in filename
            results = [np.load(el) for el in files]
            sorted_vars = [extract_var_fromstring(file) for file in files]

            # Calculate average time, power, and number of features for each result
            time_pow_nfeat = np.asarray([(el[:,:,1].mean(axis=0), el[:,:,0].mean(axis=0), el[:,:,2].mean(axis=0)) for el in results])

            results_dict[method] = time_pow_nfeat

            

        # file_0_path = Path(files[0]) # read config from first file (they are all the same)
        # config = read_config_if_exists(file_0_path.parts[0]+'/'+file_0_path.parts[1]+'/'+'arguments.txt')

    return results_dict, config, sorted_vars


def return_parameters(s: str) -> dict:
    # Extract the last part after the last "/"
    s = s.split("/")[-1]
    
    # Split the string by underscores to handle dataset-style strings
    parts = s.split("_")
    dataset_name = parts[0]  # The first part is the dataset name
    params = {}

    # Iterate through the remaining parts, ensuring each part is a key-value pair
    for part in parts[1:]:
        # Try to split into key and value
        match = re.match(r"([a-zA-Z]+)([0-9\.]+)", part)
        if match:
            key, value = match.groups()
            try:
                # Convert numerical values to int or float
                if '.' in value:
                    params[key] = float(value)
                else:
                    params[key] = int(value)
            except ValueError:
                params[key] = value  # Keep as a string if conversion fails
        else:
            # If no match, we assume the part is a malformed key-value pair and keep it as a string
            params[part] = part
    
    return {"dataset": dataset_name, **params}


# Helper function to extract 'ntot' value from filenames
def extract_ntot_fromstring(s):
    match = re.search(r'ntot(\d+)_', s)  # Find digits between 'n' and '_'
    return int(match.group(1)) if match else 0  # Convert to integer for proper sorting

# Helper function to extract 'rho' value from filenames
def extract_rho_fromstring(s):
    match = re.search(r'rho([\d\.]+)', s)  # Find digits between 'n' and '_'
    return float(match.group(1)) if match else 0.0  # Convert to float for proper sorting

def extract_var_fromstring(s):
    match = re.search(r'var([\d\.]+)', s)  # Find digits after 'var'
    var_value = float(match.group(1)) if match else 0.0  # Convert to float for proper sorting
    return var_value  # Return just the var value for sorting


def read_config_if_exists(file_path):
    config = {}
    file = Path(file_path)
    if file.is_file():
        with open(file_path, 'r') as file:
            for line in file:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    try:
                        # Safely evaluate lists, numbers, etc.
                        config[key] = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        # Fallback for plain strings
                        config[key] = value
    return config