from loaders import load_results
from plotters import plot_powervsvars, plot_powervscomp, plot_powervsnfeat
import argparse



def main():

    parser = argparse.ArgumentParser() 

    # Required named arguments
    parser.add_argument('--folder', type=str, help='Folder where results are stored. E.g. "./results".')   
    parser.add_argument('--tests', nargs='+', type=str, default=["uniform", "rlss", "rff"], help='Input tests as a list.')

    args = parser.parse_args()

    methods = args.tests

    folder = args.folder

    results, config, vars = load_results(folder, methods=methods) 
    print(config)
    print(vars)
    print(results.keys())

    plot_powervsvars(results, vars, config, file=folder+'/'+config['dataset']+'power_vs_vars')

    plot_powervscomp(results, 4, config, file=folder+'/'+config['dataset']+'power_vs_comptime')

    if config['dataset'] == 'higgs':
        var = 5
    elif config['dataset'] == 'susy':
        var = 5
    elif config['dataset'] == 'cg':
            var = 4

    plot_powervsnfeat(results, var, config, file=folder+'/'+config['dataset']+'power_vs_feat')




# Main execution block
if __name__ == "__main__":

    main()