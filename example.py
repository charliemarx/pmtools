from scripts.flipped_training import *
from scripts.ex_create_dataset_files import create_dataset_file
from matplotlib import pyplot as plt

info = {
    'data_name': 'compas_arrest_small',  # adding '_small' reduces dataset to 500 points for testing purposes
    'fold_id': 'K05N01',
    'fold_num': 1
    }

#### settings for each type of training job
baseline_info = {
    'data_name': info['data_name'],
    'fold_id': info['fold_id'],
    'fold_num': info['fold_num'],
    'random_seed': 1337,
    'print_flag': True,
    'time_limit': 7200,
    'load_from_disk': False,  # load previous runs if they exist
    'equivalent_time_limit': 0,
    'populate_time_limit': 0,
    'error_constraint_type': 3,
    }


discrepancy_info = {
    'data_name': info['data_name'],
    'fold_id': info['fold_id'],
    'fold_num': info['fold_num'],
    'random_seed': 1337,
    'print_flag': True,
    #
    'time_limit': 600,
    'initialize': True,
    'load_from_disk': True,  # load previous runs if they exist
    'error_constraint_type': 3,
    'instance_time_limit': 30,
    }

output_dir = results_dir / info['data_name']
output_files = {
    'baseline': output_dir / get_baseline_file_name(info),
    'discrepancy': output_dir / get_discrepancy_file_name(info),
    }


#### data preprocessing
is_test = "_small" in info["data_name"]
raw_data_name = info["data_name"].replace("_small", "")
create_dataset_file(raw_data_name, test=is_test)


#### baseline
baseline_results = train_baseline_classifier(baseline_info)

with open(output_files['baseline'], 'wb') as outfile:
    dill.dump(baseline_results, outfile, protocol = dill.HIGHEST_PROTOCOL)
    print_log('saved results in %s' % output_files['baseline'])


#### discrepancy
discrepancy_results = train_discrepancy_classifier(discrepancy_info)

with open(output_files['discrepancy'], 'wb') as outfile:
    dill.dump(discrepancy_results, outfile, protocol = dill.HIGHEST_PROTOCOL)
    print_log('saved results in %s' % output_files['discrepancy'])


#### Look at the results
# the info returned from the discrepancy training process
print(discrepancy_results["results_df"].columns)

# access the coefficients for a particular discrepancy model
print(discrepancy_results["results_df"]["coefficients"][13])

# example plot
eps = discrepancy_results["results_df"]["epsilon"]
disc = discrepancy_results["results_df"]["total_discrepancy"]
plt.plot(eps, disc)
plt.xlabel("Epsilon (# additional errors allowed)")
plt.ylabel("Discrepancy (# predictions flipped)")
plt.show()

