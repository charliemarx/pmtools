from eqm.disc_mip import DiscrepancyMIP
from scripts.flipped_training import *

# general things we want to import for experiments
pd.set_option('display.max_columns', 30)
np.set_printoptions(precision = 2)
np.set_printoptions(suppress = True)

# dashboard
info = {
    'data_name': 'pretrial_CA_arrest',
    'random_seed': 1337,
    'fold_id': 'K05N01',
    'fold_num': 0,
    'print_flag': True,
    #
    'time_limit': 100,
    'load_from_disk': False,
    'instance_time_limit': 100,
    'error_constraint_type': 3,
    }

# setup files
output_dir = results_dir / info['data_name']
results_file_name = output_dir / get_discrepancy_file_name(info)
baseline_file_name = output_dir / get_baseline_file_name(info)

# load baseline file
assert baseline_file_name.exists()
with open(baseline_file_name, 'rb') as infile:
    baseline_results = dill.load(infile)
print_log('loaded baseline file %s' % baseline_file_name)

# setup seed
np.random.seed(seed = info['random_seed'])

# load dataset
data_file_name = '%s_processed.pickle' % info['data_name']
data_file_name = data_dir / data_file_name
data, cvindices = load_processed_data(file_name = data_file_name.with_suffix('.pickle'))
data = filter_data_to_fold(data, cvindices = cvindices, fold_id = info['fold_id'], fold_num = info['fold_num'])
print_log('loaded data: %s' % data_file_name)

pool = baseline_results['pool_df']
baseline_stats = baseline_results['baseline_output']
baseline_coefs = pool.iloc[pool['objval'].idxmin()]['coefficients']

n_samples = data['X'].shape[0]
baseline_ub = baseline_stats['upperbound']
baseline_lb = baseline_stats['lowerbound']
loss_values = np.arange(baseline_ub, n_samples // 2)
epsilon_values = sorted(np.array(loss_values - baseline_ub, dtype = int).tolist())
print_log('%1.0f values of epsilon in {%1.0f,...,%1.0f}' % (len(epsilon_values), min(epsilon_values), max(epsilon_values)))

# load existing results file if it exists
if info['load_from_disk'] and results_file_name.exists():

    # load from disk
    with open(results_file_name, 'rb') as infile:
        results = dill.load(infile)

    for k in ['data_name', 'fold_id', 'fold_num']:
        assert info[k] == results[k], 'mismatch in loaded results'

    assert np.isclose(baseline_coefs, results['baseline_coefs']).all()
    assert np.isclose(epsilon_values, results['epsilon_values']).all()
    print_log('loaded existing results from disk %s' % results_file_name)

else:

    results = {
        'data_name': info['data_name'],
        'fold_id': info['fold_id'],
        'fold_num': info['fold_num'],
        'baseline_ub': baseline_ub,
        'baseline_lb': baseline_lb,
        'baseline_coefs': baseline_coefs,
        'epsilon_values': epsilon_values,
        }

    results['output'] = {e: None for e in results['epsilon_values']}
    print_log('no results file on disk to load: %s' % results_file_name)

# initialize discrepancy MIP
mip = DiscrepancyMIP(data, baseline_coefs, baseline_stats, print_flag = True, parallel_flag = True, random_seed = info['random_seed'])
mip.bound_error_gap(lb = baseline_lb - baseline_ub, ub = 0)
mip.mip = set_mip_parameters(mip.mip, CPX_MIP_PARAMETERS)
out = mip.solve(time_limit = 100)
mip.check_solution()

# update date
now = datetime.now()
results['date'] = now.strftime("%y_%m_%d_%H_%M"),

# compute baseline classifier grap
g = mip.get_classifier(coefs = mip.baseline_coefs)
G = g.predict(data['X'])
err_g = np.not_equal(data['Y'], G).sum()

# run training
epsilon_values = results['epsilon_values']
total_iterations = len(epsilon_values)
training_epsilon_values = sorted([e for (e, out) in results['output'].items() if out is None])
remaining_time = info['time_limit']

for e in training_epsilon_values:

    i = epsilon_values.index(e)

    # record start time
    start_time = time.time()
    print_log('=' * 70)
    print_log('started training for epsilon = %1.0f, iteration %d/%d' % (e, i, total_iterations))
    print("\n")

    # setup MIP
    mip.bound_error_gap(ub = e)
    train_time = min(remaining_time, info['instance_time_limit'])
    out = mip.solve(time_limit = train_time)

    # compute current classifier stats
    h = mip.get_classifier()
    H = h.predict(data['X'])
    err_h = np.not_equal(H, data['Y']).sum()

    # record stats
    out['epsilon'] = e
    out['coefficients'] = mip.coefficients
    out['total_error_gap'] = mip.solution.get_values(mip.names['total_error_gap'])
    out['total_discrepancy'] = np.not_equal(H, G).sum()
    out['total_agreement'] = mip.solution.get_values(mip.names['total_agreement'])

    # check solution
    print_log(('=' * 20) + ' WARNINGS ' + ('=' * 20))
    mip.check_solution()

    # store solution
    results['output'][e] = out

    # print details about solution
    msg = [('=' * 20) + ' SUMMARY ' + ('=' * 20),
           '-' * 70,
           'mistakes',
           'R(h): %1.0f' % err_h,
           'R(g): %1.0f' % err_g,
           'R(h)-R(g): %1.0f' % (err_h - err_g),
           '-' * 70,
           'alignment',
           '#[h(x)==g(x)]: %1.0f' % np.equal(H, G).sum(),
           '#[h(x)!=g(x)]: %1.0f' % np.not_equal(H, G).sum(),
           '-' * 70,
           'mip output',
           'max error gap (=epsilon): %1.0f' % e,
           'total error gap (=epsilon): %1.0f' % out['total_error_gap'],
           'total discrepancy (objval): %1.0f' % out['total_discrepancy'],
           'total agreement (objval): %1.0f' % out['total_agreement'],
           'objval ub: %1.0f' % out['upperbound'],
           'objval lb: %1.0f' % out['lowerbound'],
           'objval gap: %1.2f%%' % (100.0 * out['gap']),
           '=' * 70,
           ]
    msg = '\n'.join(msg)
    print_log(msg)

    # print completion message
    print_log('completed training for epsilon = %1.0f, iteration %d/%d' % (e, i, total_iterations))

    # check whether to stop
    time_elapsed = time.time() - start_time
    remaining_time = remaining_time - time_elapsed
    if remaining_time < 30.0:
        print_log('STOPPING TRAINING: out of time')
        break

df = [out for out in results['output'].values() if out is not None]
df = pd.DataFrame(df)
df.set_index('epsilon')
df[['epsilon', 'gap', 'total_discrepancy', 'total_agreement']]
print(df)


#save file to disk
with open(results_file_name, 'wb') as outfile:
    dill.dump(results, outfile, protocol = dill.HIGHEST_PROTOCOL)
    print_log('saved results in %s' % results_file_name)





