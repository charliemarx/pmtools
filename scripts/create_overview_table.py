from scripts.reporting import *
from scripts.flipped_training import *
from eqm.cross_validation import filter_data_to_fold
from eqm.data import get_common_row_indices


# dashboard
output_file_name = paper_dir / 'overview_table.csv'

all_data_names = [
    'compas_arrest', 'compas_violent',
    'compas_arrest_small', 'compas_violent_small',
    'pretrial_CA_arrest', 'pretrial_CA_fta',
    'pretrial_NY_arrest', 'pretrial_NY_fta',
    'recidivism_CA_arrest', 'recidivism_CA_drug',
    'recidivism_NY_arrest', 'recidivism_NY_drug',
]

def get_overview_table_row(info):
    """
    creates a row of the overview data frame
    :param info: dictionary containing data_name, fold_id, fold_num
    :return: dictionary containing all fields for the overview data frame
    """

    ## setup file names
    output_dir = results_dir / info['data_name']

    # file names
    file_names = {
        'data': '%s/%s_processed.pickle' % (data_dir, info['data_name']),
        'baseline': output_dir / get_baseline_file_name(info),
        'discrepancy': output_dir / get_discrepancy_file_name(info),
        'flipped': output_dir / get_processed_file_name(info),
        }

    # load data
    data, cvindices = load_processed_data(file_name = file_names['data'])
    data = filter_data_to_fold(data, cvindices = cvindices, fold_id = info['fold_id'], fold_num = info['fold_num'], include_validation = True)

    XY = np.hstack([data['X'], data['Y'][:, None]])
    XY_test = np.hstack([data['X_validation'], data['Y_validation'][:, None]])
    U = np.unique(XY, axis = 0)
    U_test = np.unique(XY_test, axis = 0)
    observed_idx = get_common_row_indices(U, U_test)

    # data-related fields
    row = {
        'data_name': info['data_name'],
        'fold_id': info['fold_id'],
        'fold_num': info['fold_num'],
        #
        'n': data['X'].shape[0],
        'd': data['X'].shape[1] - 1,
        'n_pos': np.sum(data['Y'] == 1),
        'n_neg': np.sum(data['Y'] == -1),
        'n_xy_unique': U.shape[0],
        'n_xy_unobs_test': U_test.shape[0] - len(observed_idx),
        #
        'n_test': data['X_validation'].shape[0],
        'n_test_pos': np.sum(data['Y_validation'] == 1),
        'n_test_neg': np.sum(data['Y_validation'] == -1),
        #
        'has_baseline_results': file_names['baseline'].exists(),
        'has_flipped_results': file_names['flipped'].exists(),
        'has_discrepancy_results': file_names['discrepancy'].exists(),
        }

    # fields from baseline results
    baseline_fields = ['baseline_train_error',  'baseline_test_error', 'baseline_ub', 'baseline_lb', 'baseline_gap', 'baseline_n_equivalent', 'baseline_time_limit']
    row.update({k: float('nan') for k in baseline_fields})

    if file_names['baseline'].exists():

        with open(file_names['baseline'], 'rb') as infile:
            baseline_results = dill.load(infile)

        baseline_coefs = baseline_results['pool_df'].query('model_type=="baseline"')['coefficients'].values[0]
        out = baseline_results['baseline_output']
        err_test = np.sign(data['X_validation'].dot(baseline_coefs))

        row.update({
            'baseline_train_error': out['upperbound'],
            'baseline_test_error': np.not_equal(err_test, data['Y_validation']).sum(),
            'baseline_n_equivalent': len(baseline_results['equivalent_output']),
            'baseline_time_limit': baseline_results['info']['time_limit'],
            'baseline_ub': out['upperbound'],
            'baseline_lb': out['lowerbound'],
            'baseline_gap': out['gap'],
            })

    # initialize discrepancy as 'nan'
    disc_fields = [
        'disc_instances', 'disc_nnz_instances',
        'disc_instances_gap_eq_0', 'disc_instances_gap_leq_0.1', 'disc_instances_gap_leq_0.5', 'disc_instances_gap_gt_0.5',
        'disc_min_epsilon', 'disc_discrepancy_ratio_min', 'disc_discrepancy_ratio_med', 'disc_discrepancy_ratio_max',
        'disc_discrepancy_min', 'disc_discrepancy_med', 'disc_discrepancy_max'
        ]
    row.update({k: float('nan') for k in disc_fields})

    if file_names['discrepancy'].exists():
        with open(file_names['discrepancy'], 'rb') as infile:
            discrepancy_results = dill.load(infile)

        n_instances = len(discrepancy_results['epsilon_values'])
        disc_df = discrepancy_results['results_df']

        nnz_df = disc_df.query('total_discrepancy > 0')
        n_nnz = len(nnz_df)
        if len(nnz_df) > 0:
            ratio = nnz_df['total_discrepancy'] / nnz_df['epsilon']
            row.update({
                'disc_min_epsilon': nnz_df['epsilon'].idxmin(),
                'disc_discrepancy_ratio_min': np.nanmin(ratio),
                'disc_discrepancy_ratio_med': np.nanmedian(ratio),
                'disc_discrepancy_ratio_max': np.nanmax(ratio),
                })

        # stats
        row.update({
            'disc_instances': n_instances,
            'disc_nnz_instances': n_nnz,
            #
            'disc_instances_gap_eq_0': len(disc_df.query('gap == 0.0')),
            'disc_instances_gap_leq_0.1': len(disc_df.query('gap <= 0.1')),
            'disc_instances_gap_leq_0.5': len(disc_df.query('gap <= 0.5')),
            'disc_instances_gap_gt_0.5': len(disc_df.query('gap > 0.5')),
            #
            'disc_discrepancy_min': np.nanmin(disc_df['total_discrepancy']),
            'disc_discrepancy_med':np.nanmedian(disc_df['total_discrepancy']),
            'disc_discrepancy_max': np.nanmax(disc_df['total_discrepancy']),
            })

    #initialize flipped fields
    flipped_fields = [
        'flipped_instances',
        'flipped_instances_gap_eq_0', 'flipped_instances_gap_leq_0.1', 'flipped_instances_gap_leq_0.5', 'flipped_instances_gap_gt_0.5',
        'flipped_change_in_error_min', 'flipped_change_in_error_med', 'flipped_change_in_error_max',
        'flipped_change_in_test_error_min', 'flipped_change_in_test_error_med', 'flipped_change_in_test_error_max'
        ]
    row.update({k: float('nan') for k in flipped_fields})

    if file_names['flipped'].exists():

        with open(file_names['flipped'], 'rb') as infile:
            flipped_results = dill.load(infile)

        flip_df = flipped_results['results_df']
        baseline = flip_df.query('model_type == "baseline"')
        flipped = flip_df.query('model_type == "flipped"')
        change_in_train_error = flipped['train_error'] - baseline['train_error'].values[0]
        change_in_test_error = flipped['validation_error'] - baseline['validation_error'].values[0]

        row.update({
            #
            'flipped_instances': len(flipped),
            'flipped_instances_missing': flipped_results['n_missing'],
            'flipped_instances_gap_eq_0': len(flipped.query('gap == 0.0')),
            'flipped_instances_gap_leq_0.1': len(flipped.query('gap <= 0.1')),
            'flipped_instances_gap_leq_0.5': len(flipped.query('gap <= 0.5')),
            'flipped_instances_gap_gt_0.5': len(flipped.query('gap > 0.5')),
            #
            'flipped_change_in_error_min': np.nanmin(change_in_train_error),
            'flipped_change_in_error_med': np.nanmedian(change_in_train_error),
            'flipped_change_in_error_max': np.nanmax(change_in_train_error),
            #
            'flipped_change_in_test_error_min': np.nanmin(change_in_test_error),
            'flipped_change_in_test_error_med': np.nanmedian(change_in_test_error),
            'flipped_change_in_test_error_max': np.nanmax(change_in_test_error),
            })

    return row




# build overview table row by row
df_rows = []

for data_name in all_data_names:

    row_info = {
        'data_name': data_name,
        'fold_id': 'K05N01',
        'fold_num': 1
        }

    # identify key files
    output_dir = results_dir / row_info['data_name']
    baseline_file_name = output_dir / get_baseline_file_name(row_info)
    flipped_results_file = output_dir / get_processed_file_name(row_info)

    if baseline_file_name.exists():

        # create the flipped results file
        try:
            results = aggregate_baseline_and_flipped_results(row_info)
            with open(flipped_results_file, 'wb') as outfile:
                dill.dump(results, outfile, protocol = dill.HIGHEST_PROTOCOL)
                print_log('saved results in %s' % flipped_results_file)

        except Exception as e:
            print_log('failed to process flipped results for data_name: %s' % row_info['data_name'])


        # build overview table
        try:
            row = get_overview_table_row(row_info)
            df_rows.append(row)
        except Exception as e:
            print_log('ran into error while processing %s' % row_info['data_name'])
            print_log(str(e))
            pass

    else:
        print_log('could not find baseline file for %s' % row_info['data_name'])


df = pd.DataFrame(df_rows)

# reorder data frame
all_columns = set(df.columns.tolist())
ordered_columns = ['data_name', 'fold_id', 'fold_num',
                   'has_baseline_results', 'has_discrepancy_results', 'has_flipped_results',
                   'n', 'd', 'n_neg', 'n_pos', 'n_test', 'n_test_neg', 'n_test_pos', 'n_xy_unique', 'n_xy_unobs_test']

all_columns = all_columns - set(ordered_columns)
ordered_columns = ordered_columns + [n for n in all_columns if 'baseline' in n]
all_columns = all_columns - set(ordered_columns)
ordered_columns = ordered_columns + [n for n in all_columns if 'disc' in n]
all_columns = all_columns - set(ordered_columns)
ordered_columns = ordered_columns + [n for n in all_columns if 'flipped' in n]
all_columns = all_columns - set(ordered_columns)
ordered_columns = ordered_columns + list(all_columns)
df = df[ordered_columns]

# save as CSV
df.to_csv(output_file_name, index = False)