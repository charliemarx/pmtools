from scripts.flipped_training import *

info = {
    'data_name': 'compas_violent_small',
    'fold_id': 'K05N01',
    'fold_num': 1
    }

save_files = True

# basic correction to prevent dumb bug
if 'test' in info['data_name']:
    info['fold_id'] = 'K01N01'
    info['fold_num'] = 0
else:
    info['fold_id'] = 'K05N01'
    info['fold_num'] = 1

#### settings for each type of training job
baseline_info = {
    'data_name': info['data_name'],
    'fold_id': info['fold_id'],
    'fold_num': info['fold_num'],
    'random_seed': 1337,
    'print_flag': True,
    'time_limit': 7200,
    'load_from_disk': False,
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
    'load_from_disk': True,
    'error_constraint_type': 3,
    'instance_time_limit': 30,
    }


flipped_info = {
    'data_name': info['data_name'],
    'fold_id': info['fold_id'],
    'part_id': TRIVIAL_PART_ID,
    'fold_num': info['fold_num'],
    'random_seed': 1337,
    'time_limit_flipped': 300,
    }


output_dir = results_dir / info['data_name']
output_files = {
    'baseline': output_dir / get_baseline_file_name(info),
    'discrepancy': output_dir / get_discrepancy_file_name(info),
    'flipped': output_dir / get_flipped_file_name(flipped_info),
    'processed': output_dir / get_processed_file_name(info),
    }


#### baseline
baseline_results = train_baseline_classifier(baseline_info)

if save_files:
    with open(output_files['baseline'], 'wb') as outfile:
        dill.dump(baseline_results, outfile, protocol = dill.HIGHEST_PROTOCOL)
        print_log('saved results in %s' % output_files['baseline'])

### discrepancy ####
discrepancy_results = train_discrepancy_classifier(discrepancy_info)

if save_files:
    with open(output_files['discrepancy'], 'wb') as outfile:
        dill.dump(discrepancy_results, outfile, protocol = dill.HIGHEST_PROTOCOL)
        print_log('saved results in %s' % output_files['discrepancy'])

    with open(output_files['discrepancy'], 'rb') as infile:
        loaded_results = dill.load(infile)

#### flipped ####
part, n_parts = parse_part_id(flipped_info['part_id'])
flipped_info['part_id'] = PART_ID_HANDLE(p = part, n = n_parts)
flipped_results = train_flipped_classifiers(flipped_info)

# save and load
if save_files:
    with open(output_files['flipped'], 'wb') as outfile:
        dill.dump(flipped_results, outfile, protocol = dill.HIGHEST_PROTOCOL)
        print_log('saved results in %s' % output_files['flipped'])

    with open(output_files['flipped'], 'rb') as infile:
        loaded_results = dill.load(infile)

#### processed #####

processed_results = aggregate_baseline_and_flipped_results(info)

if save_files:
    with open(output_files['processed'], 'wb') as outfile:
        dill.dump(processed_results, outfile, protocol = dill.HIGHEST_PROTOCOL)
        print_log('saved results in %s' % output_files['processed'])

    with open(output_files['processed'], 'rb') as infile:
        processed_results = dill.load(infile)
