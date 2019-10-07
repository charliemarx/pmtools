from eqm.paths import *
from eqm.data import *
from eqm.debug import ipsh
from eqm.cross_validation import filter_data_to_fold
from eqm.classifier_helper import ClassificationModel
from eqm.experiment_helper import *
from eqm.disc_mip import DiscrepancyMIP
from scripts.reporting import *

from datetime import datetime
import dill
import pandas as pd

BASELINE_TRAINING_SETTINGS = {
    'random_seed': 1337,
    'part_id': 'P01N01',
    'fold_id': 'K05N01',
    'fold_num': 1,
    'print_flag': True,
    'load_from_disk': True,
    #
    'time_limit': 3600,
    'equivalent_time_limit': 300,
    'populate_time_limit': 300,
    'error_constraint_type': 3,
    }

FLIPPED_TRAINING_SETTINGS = {
    'random_seed': 1337,
    'fold_id': 'K05N01',
    'fold_num': 1,
    'load_from_disk': True,
    #
    'time_limit_flipped': 300,
    }

DISCREPANCY_TRAINING_SETTINGS = {
    'random_seed': 1337,
    'fold_id': 'K05N01',
    'fold_num': 1,
    'print_flag': True,
    #
    'time_limit': 100,
    'instance_time_limit': 10,
    'initialize': True,
    'load_from_disk': True,
    'error_constraint_type': 3,
    'epsilon_step': 5,
    }

#### file names

def get_glmnet_file_name(info):
    assert 'data_name' in info and 'fold_id' in info
    fname = '%s_%s_glmnet_coefficients.csv' % (info['data_name'], info['fold_id'])
    return fname


def get_coefficient_pool_file_name(info):
    assert 'data_name' in info and 'fold_id' in info
    fname = '%s_%s_coefficient_pool.pickle' % (info['data_name'], info['fold_id'])
    return fname


def get_baseline_file_name(info):
    assert 'data_name' in info and 'fold_id' in info
    fname = '%s_%s_baseline_raw_results.pickle' % (info['data_name'], info['fold_id'])
    return fname


def get_flipped_file_name(info):
    assert 'data_name' in info and 'fold_id' in info and 'part_id' in info
    fname = '%s_%s_%s_flipped_raw_results.pickle' % (info['data_name'], info['fold_id'], info['part_id'])
    return fname


def get_discrepancy_file_name(info):
    assert 'data_name' in info and 'fold_id' in info
    fname = '%s_%s_discrepancy_raw_results.pickle' % (info['data_name'], info['fold_id'])
    return fname


def get_processed_file_name(info):
    assert 'data_name' in info and 'fold_id' in info
    fname = '%s_%s_flipped_processed_results.pickle' % (info['data_name'], info['fold_id'])
    return fname


#### file IO

def load_and_filter_processed_data(info):
    """
    :param info:
    :return:
    """
    assert isinstance(info, dict)
    file_name = '%s_processed.pickle' % info['data_name']
    file_name = data_dir / file_name
    data, cvindices = load_processed_data(file_name = file_name.with_suffix('.pickle'))
    # file_name
    if 'fold_id' in info:
        if 'fold_num' in info:
            data = filter_data_to_fold(data, cvindices = cvindices, fold_id = info['fold_id'], fold_num = info['fold_num'])
        else:
            data = filter_data_to_fold(data, cvindices = cvindices, fold_id = info['fold_id'], fold_num = 0)

    print_log('loaded dataset %s' % file_name)
    return data


def save_results_to_disk(results, file_name):
    """
    :param file_name:
    :param results:
    :return:
    """
    results['results_file_name'] = Path(file_name).name
    saved_flag = False
    try:
        with open(file_name, 'wb') as outfile:
            dill.dump(results, outfile, protocol = dill.HIGHEST_PROTOCOL)
            print_log('saved results in %s' % file_name)
        saved_flag = True
    except:
        print_log('encountered error while saving %s' % file_name)

    return saved_flag


def load_coefficients_from_flipped_files(info):

    output_dir = results_dir / info['data_name']
    flipped_file_helper = get_part_id_helper([f for f in output_dir.iterdir() if f.suffix == '.pickle'])
    loaded_coefficients = []

    for n, file_info in flipped_file_helper.items():
        for f in file_info['matched_files']:
            try:
                with open(f, 'rb') as infile:
                    flip_results = dill.load(infile)

                flipped_info = flip_results['info']
                if info['data_name'] == info['data_name'] and info['fold_id'] == flipped_info['fold_id'] and info['fold_num'] == flipped_info['fold_num']:
                    loaded_coefficients += flip_results['flipped_df']['coefficients'].tolist()

            except Exception as e:
                print_log('error while loading coefficients for file %s' % f)
                print_log('%r' % str(e))

    if len(loaded_coefficients):
        loaded_coefficients = np.unique(np.vstack((loaded_coefficients)), axis = 0)

    return loaded_coefficients


def build_coefficient_pool(info):
    """
    :param info:
    :return:
    """


    output_dir = results_dir / info['data_name']
    data = load_and_filter_processed_data(info)
    coef_df = pd.DataFrame(columns = ['coefficients', 'error_lb', 'agree_lb', 'error_ub', 'agree_ub'])


    # load baseline file
    file_name = output_dir / get_baseline_file_name(info)
    baseline_is_optimal = False
    if file_name.exists():
        print_log('loading %s' % file_name)
        with open(file_name, 'rb') as infile:
            loaded = dill.load(infile)

        # update flag
        baseline_coefs = loaded['pool_df'].query('model_type == "baseline"')['coefficients'].values[0]

        # add coefficients
        new_df = loaded['pool_df'][['coefficients', 'lowerbound']].rename(columns = {'lowerbound': 'error_lb'})
        coef_df = coef_df.append(new_df, sort = False)

    # load discrepancy
    file_name = output_dir / get_discrepancy_file_name(info)
    if file_name.exists():
        print_log('loading %s' % file_name)
        with open(file_name, 'rb') as infile:
            loaded = dill.load(infile)
        new_df = loaded['results_df'][['coefficients', 'lowerbound']].rename(columns = {'lowerbound': 'agree_lb'})
        matching_baseline = np.isclose(baseline_coefs, loaded['baseline_coefs']).all()
        if not matching_baseline:
            new_df['agree_lb'] = float('nan')
        coef_df = coef_df.append(new_df, sort = False)

    # load flipped_coefficients
    flipped_coefs = load_coefficients_from_flipped_files(info)
    if len(flipped_coefs) > 0:
        new_df = pd.DataFrame({'coefficients': list(flipped_coefs)})
        coef_df = coef_df.append(new_df, sort = False)

    # load glmnet coefficients
    file_name = output_dir / get_glmnet_file_name(info)
    if file_name.exists():
        df = pd.read_csv(file_name)
        df.columns = [n.replace('weight__', '') for n in df.columns.tolist()]
        coefs = list(df[data['variable_names']].values)
        coef_df = coef_df.append(new_df, sort = False)

    if coef_df.shape[0]:
        # select a pair of coefficients with the largest ub
        coef_df['error_lb'] = coef_df['error_lb'].astype(dtype=float)
        coef_df['agree_lb'] = coef_df['agree_lb'].astype(dtype=float)
        coef_df = coef_df.sort_values(by=['error_lb', 'agree_lb'], ascending=False, na_position='last')

        W = np.vstack(list(coef_df['coefficients'].values))
        _, distinct_idx = np.unique(W, axis = 0, return_index = True)
        coef_df = coef_df.iloc[distinct_idx].reset_index(drop = True)

        # compute error and agreement
        W = np.vstack(list(coef_df['coefficients'].values))
        compute_error = lambda w: np.not_equal(data['Y'], np.sign(data['X'].dot(w))).sum()
        coef_df['error_ub'] = np.apply_along_axis(func1d = compute_error, arr = W, axis = 1)

        # compute agreement
        G = np.sign(data['X'].dot(baseline_coefs))
        compute_agreement = lambda w: np.equal(G, np.sign(data['X'].dot(w))).sum()
        coef_df['agree_ub'] = np.apply_along_axis(func1d = compute_agreement, arr = W, axis = 1)

    return coef_df


##### training procedures


def train_baseline_classifier(info):
    """
    trains baseline classifier via zero one loss minimization
    :param info:
    :return:
    """
    print_log('entered train_baseline_classifier')

    # dashboard
    for k, v in BASELINE_TRAINING_SETTINGS.items():
        if k not in info:
            info[k] = v

    print_log('settings')
    print_log('-' * 50)
    for k, v in info.items():
        print_log("info['%s'] = %r" % (k, info[k]))

    # name and create output directory
    output_dir = results_dir / info['data_name']
    output_dir.mkdir(exist_ok = True)

    # set output results file
    results_file_name = output_dir / get_baseline_file_name(info)
    print_log('-' * 50)
    print_log('saving results in %s' % results_file_name)

    # load dataset
    data_file_name = '%s_processed.pickle' % info['data_name']
    data_file_name = data_dir / data_file_name
    data, cvindices = load_processed_data(file_name = data_file_name.with_suffix('.pickle'))
    data = filter_data_to_fold(data, cvindices = cvindices, fold_id = info['fold_id'], fold_num = info['fold_num'])
    print_log('loaded dataset %s' % data_file_name)

    # solve zero-one loss MIP
    mip = ZeroOneLossMIP(data, print_flag = info['print_flag'], parallel_flag = True, random_seed = info['random_seed'], error_constraint_type = info['error_constraint_type'])
    mip.mip = set_mip_parameters(mip.mip, CPX_MIP_PARAMETERS)

    # load initializations from disk
    if info['load_from_disk']:

        # load saved coefficients from all methods
        initial_pool = build_coefficient_pool(info)
        if initial_pool.shape[0] > 0:

            print_log('initializing with %d solutions' % len(initial_pool))

            # use the best lowerbound we have
            max_error_lb = np.ceil(initial_pool['error_lb'].max())
            if np.isfinite(max_error_lb):
                mip.set_total_mistakes(lb=max_error_lb)

            # supply all possible initializations
            for k, row in initial_pool.iterrows():
                mip.add_initial_solution_with_coefficients(coefs=row['coefficients'])

            # report expected objective value
            print_log('upperbound should be at most %d' % min(initial_pool['error_ub']))

    out = mip.solve(time_limit = info['time_limit'])
    baseline_output = mip.solution_info
    print_log(str(baseline_output))
    mip.check_solution(debug_flag = True)

    base_df = {
        'solution': list(out['values']),
        'objval': out['objval'],
        'coefficients': mip.coefficients,
        'lowerbound': out['lowerbound'],
        'prediction_constraint': None,
        'model_type': 'baseline'
        }

    # initialize solution pool
    pool = SolutionPool(mip = mip)

    # search for equivalent models
    if info['equivalent_time_limit'] > 0:
        equivalent_output, pool = mip.enumerate_equivalent_solutions(pool, time_limit = info['equivalent_time_limit'])
        print_log('{} global equivalent models found.'.format(len(equivalent_output)))
    else:
        equivalent_output = []

    # generate additional solutions using populate
    if info['populate_time_limit'] > 0:
        mip.populate(max_gap = 0, time_limit = info['populate_time_limit'])
        mip.populate(max_gap = data['X'].shape[0] // 2, time_limit = info['populate_time_limit'])
        pool.add_from_mip(mip)

    # get alternative models
    pool_df = pool.get_df()
    pool_df['model_type'] = 'alternative'
    pool_df = pool_df.append(base_df, sort = False, ignore_index = True).reset_index(drop = True)

    # get time
    now = datetime.now()

    results = {
        'date': now.strftime("%y_%m_%d_%H_%M"),
        'info': info,
        'data_file': data_file_name,
        'results_file': results_file_name,
        'pool_df': pool_df,
        'baseline_output': baseline_output,
        'equivalent_output': equivalent_output,
        }

    print_log('leaving train_baseline_classifier')
    return results


def train_flipped_classifiers(info):
    """
    :param info:
    :return:
    """

    print_log('entered train_flipped_classifiers')
    print_log('-' * 50)

    # dashboard
    for k, v in FLIPPED_TRAINING_SETTINGS.items():
        if k not in info:
            info[k] = v

    for k, v in info.items():
        print("info['%s'] = %r" % (k, info[k]))

    assert "part_id" in info
    output_dir = results_dir / info['data_name']
    output_dir.mkdir(exist_ok = True)

    baseline_file_name = output_dir / get_baseline_file_name(info)
    results_file_name = output_dir / get_flipped_file_name(info)
    print_log('baseline_file_name: %s' % baseline_file_name)
    print_log('results_file_name: %s' % results_file_name)

    # load baseline file
    assert baseline_file_name.exists()
    print_log('loading results from %s' % baseline_file_name)
    with open(baseline_file_name, 'rb') as infile:
        baseline_results = dill.load(infile)
    print_log('loaded results from %s' % baseline_file_name)

    # setup pool
    pool = SolutionPool(df = baseline_results['pool_df'])

    # setup baseline classifier
    coefs = baseline_results['pool_df'].query('model_type == "baseline"')['coefficients'].values[0]
    h = ClassificationModel(predict_handle = lambda X: 1,
                            model_info = {'coefficients': coefs[1:], 'intercept': coefs[0]},
                            model_type = ClassificationModel.LINEAR_MODEL_TYPE)

    # load data from disk
    data_file_name = '%s_processed.csv' % info['data_name']
    data_file_name = data_dir / data_file_name
    data, cvindices = load_processed_data(file_name = data_file_name.with_suffix('.pickle'))
    data = filter_data_to_fold(data, cvindices = cvindices, fold_id = info['fold_id'], fold_num = info['fold_num'])
    print_log('loaded dataset %s' % data_file_name)

    # compress dataset into distinct feature vectors and counts
    compressed = compress_data(data)
    for k, v in compressed.items():
        compressed[k] = filter_indices_to_part(v, part_id = info['part_id'])
    U, N_pos, N_neg, x_to_u_idx, u_to_x_idx = tuple(compressed[var] for var in ('U', 'N_pos', 'N_neg', 'x_to_u_idx', 'u_to_x_idx'))

    # setup MIP
    mip = ZeroOneLossMIP(data, print_flag = True, parallel_flag = True, random_seed = info['random_seed'])
    mip.mip = set_mip_parameters(mip.mip, CPX_MIP_PARAMETERS)
    mip.mip.parameters.emphasis.mip.set(3)
    mip.set_total_mistakes(lb = baseline_results['baseline_output']['lowerbound'])

    # solve flipped versions
    results = []
    total_iterations = U.shape[0]
    start_time = time.process_time()

    #### START INITIALIZATION
    # pre-initialize the mip from disk
    if info['load_from_disk']:

        # load saved coefficients from all methods
        initial_pool = build_coefficient_pool(info)
        if initial_pool.shape[0] > 0:

            print_log('initializing with %d solutions' % len(initial_pool))

            # use the best lowerbound we have
            max_error_lb = np.ceil(initial_pool['error_lb'].max())
            if np.isfinite(max_error_lb):
                mip.set_total_mistakes(lb=max_error_lb)

            # supply all possible initializations
            for k, row in initial_pool.iterrows():
                mip.add_initial_solution_with_coefficients(coefs=row['coefficients'])

    #### END INITIALIZATION

    for k, x in enumerate(U):

        print_log('iteration %d/%d' % (k, total_iterations))
        print_log('solution pool size: %d' % pool.size)

        yhat = int(h.predict(x[None, :]))

        # adjust prediction constraints
        mip.clear_prediction_constraints()
        mip.add_prediction_constraint(x = x, yhat = -yhat, name = 'pred_constraint')


        # initialize model
        good_pool = pool.get_solutions_with_pred(x, -yhat)
        if good_pool.size > 0:
            s = good_pool.get_best_solution()
            mip.add_initial_solution(solution = s['solution'], objval = s['objval'], name = 'init_from_pred_cons')
            mip.add_initial_solution_with_coefficients(coefs = s['coefficients'])
            print_log('initialized\nobjval:{}'.format(s['objval']))

        # solve MIP
        out = mip.solve(time_limit = info['time_limit_flipped'])
        mip.check_solution()

        # update solution pool
        pool.add_from_mip(mip, prediction_constraint = (x, yhat))

        # update out
        out.update(
                {
                    'k': k,
                    'i': np.flatnonzero(u_to_x_idx == k).tolist(),
                    'x': x,
                    'n_pos': N_pos[k],
                    'n_neg': N_neg[k],
                    'coefficients': mip.coefficients,
                    'elapsed_time': time.process_time() - start_time,
                    'prediction_constraint': (x, yhat),
                    'solution': list(out['values']),
                    }
                )
        results.append(out)

    results_df = pd.DataFrame(results)
    results_df.drop(['upperbound', 'values'], axis=1, inplace=True)

    # create output dictionary
    now = datetime.now()
    results = {
        'date': now.strftime("%y_%m_%d_%H_%M"),
        'info': info,
        'part_id': info['part_id'],
        'data_file': data_file_name,
        'baseline_file': baseline_file_name,
        'results_file': results_file_name,
        'flipped_df': results_df,
        }

    print_log('leaving train_flipped_classifier')
    return results


def train_discrepancy_classifier(info):
    """
    :param info:
    :return:
    """

    print_log('entered train_discrepancy_classifier')

    # dashboard
    for k, v in DISCREPANCY_TRAINING_SETTINGS.items():
        if k not in info:
            info[k] = v

    for k, v in info.items():
        print("info['%s'] = %r" % (k, info[k]))

    # setup files
    output_dir = results_dir / info['data_name']
    output_dir.mkdir(exist_ok = True)

    results_file_name = output_dir / get_discrepancy_file_name(info)
    baseline_file_name = output_dir / get_baseline_file_name(info)
    processed_file_name = output_dir / get_processed_file_name(info)

    # print file names
    print_log('baseline_file_name: %s' % baseline_file_name)
    print_log('results_file_name: %s' % results_file_name)

    # load baseline file
    assert baseline_file_name.exists()
    with open(baseline_file_name, 'rb') as infile:
        baseline_results = dill.load(infile)
    print_log('loaded baseline file %s' % baseline_file_name)

    # setup seed
    np.random.seed(seed = info['random_seed'])

    # load data from disk
    data_file_name = '%s_processed.csv' % info['data_name']
    data_file_name = data_dir / data_file_name
    data, cvindices = load_processed_data(file_name = data_file_name.with_suffix('.pickle'))
    data = filter_data_to_fold(data, cvindices = cvindices, fold_id = info['fold_id'], fold_num = info['fold_num'])
    print_log('loaded dataset %s' % data_file_name)

    # load baseline
    initial_pool = baseline_results['pool_df']
    baseline_stats = baseline_results['baseline_output']
    baseline_coefs = initial_pool.query('model_type == "baseline"')['coefficients'].values[0]

    n_samples = data['X'].shape[0]
    baseline_ub = baseline_stats['upperbound']
    baseline_lb = baseline_stats['lowerbound']
    loss_values = np.arange(baseline_ub, n_samples // 2, step=DISCREPANCY_TRAINING_SETTINGS['epsilon_step'])
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
        print_log('did not find file with existing results on disk: %s' % results_file_name)

    initial_pool = build_coefficient_pool(info)  # overwrites initial pool above with superset (also different format)
    initial_coefs = np.vstack(initial_pool['coefficients'])
    initial_errors = initial_pool['error_ub']

    # build discrepancy MIP
    mip = DiscrepancyMIP(data, baseline_coefs, baseline_stats, print_flag = info['print_flag'], parallel_flag = True, random_seed = info['random_seed'])
    mip.mip = set_mip_parameters(mip.mip, CPX_MIP_PARAMETERS)
    mip.bound_error_gap(lb = int(np.floor(baseline_lb - baseline_ub)))

    # update date
    now = datetime.now()
    results['results_file'] = results_file_name

    # compute baseline classifier gap
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

        # add initial solutions that are valid
        if info['initialize']:
            keep_idx = np.less_equal(initial_errors, e + err_g)
            feasible_coefs = initial_coefs[keep_idx, :]
            print_log('initializing with %d solutions' % feasible_coefs.shape[0])
            for w in feasible_coefs:
                mip.add_initial_solution_with_coefficients(coefs = w)

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

        # save file
        results['date'] = now.strftime("%y_%m_%d_%H_%M")
        results_df = [out for out in results['output'].values() if out is not None]
        results_df = pd.DataFrame(results_df)
        results_df.set_index('epsilon')
        results['results_df'] = results_df
        save_results_to_disk(results, file_name = results_file_name)

        # check whether to stop
        time_elapsed = time.time() - start_time
        remaining_time = remaining_time - time_elapsed
        if remaining_time < 30.0 + info['instance_time_limit']:
            print_log('STOPPING TRAINING: out of time')
            print(results_df[['epsilon', 'gap', 'total_discrepancy', 'total_agreement']])
            break

    # print final results
    print_log('leaving train_discrepancy_classifier')
    return results


def select_partition_for_flipped_training(partitions):
    """
    :param partitions:
    :return:
    """
    assert isinstance(partitions, dict) and len(partitions) > 0
    n_partitions = len(partitions)
    all_part_counts = list(partitions.keys())

    if n_partitions == 1:
        return dict(partitions[all_part_counts[0]])

    print_log('found results files for %d different partitions' % len(partitions))
    complete_part_counts = [k for k in all_part_counts if partitions[k]['complete']]
    if len(complete_part_counts) > 0:
        all_part_counts = complete_part_counts
    else:
        print_log('could not find any partition without missing files')

    n = all_part_counts[0]
    if len(all_part_counts) > 1:
        nt = partitions[n]['last_modification_time']
        for p in all_part_counts:
            pt = partitions[p]['last_modification_time']
            if nt < pt:
                n, nt = p, pt
        print_log('choosing partition for N = %d since it has the most recently modified file' % n)
    else:
        print_log('choosing partition for N = %d since it does not have any missing files' % n)

    return dict(partitions[n])


def aggregate_baseline_and_flipped_results(info):
    """
    :param data_name:
    :param fold_id:
    :param fold_num:
    :return:
    """
    assert isinstance(info, dict)
    assert 'data_name' in info and 'fold_id' in info and 'fold_num' in info

    # setup file names
    output_dir = results_dir / info['data_name']
    baseline_results_file = output_dir / get_baseline_file_name(info)
    data_file_name = Path('%s/%s_processed.pickle' % (data_dir, info['data_name']))

    # load baseline results
    assert baseline_results_file.exists()
    with open(baseline_results_file, 'rb') as f:
        baseline = dill.load(f)

    # load data from disk
    assert baseline_results_file.exists()
    data, cvindices = load_processed_data(file_name = data_file_name)
    data = filter_data_to_fold(data, cvindices = cvindices, fold_id = info['fold_id'], fold_num = info['fold_num'], include_validation = True)

    # load flipped results
    flipped_raw_results_files = [f for f in output_dir.iterdir() if f.suffix == '.pickle' and 'flipped_raw_results' in f.name]
    all_partitions = get_part_id_helper(flipped_raw_results_files)
    flipped_partition = select_partition_for_flipped_training(all_partitions)
    flipped_results_files = flipped_partition['matched_files']
    print_log('partition contains %d raw results files.' % len(flipped_results_files))

    # load files
    flipped = []
    for fname in flipped_results_files:
        with open(fname, 'rb') as f:
            flipped.append(dill.load(f))

     # concatenate data frames
    if len(flipped):
        flipped_df = pd.concat([info['flipped_df'] for info in flipped], sort = False)
        flipped_df['model_type'] = 'flipped'
        results_df = pd.concat([baseline['pool_df'], flipped_df], sort=False).reset_index(drop=True)
    else:
        print("Warning: no flipped model files found during results aggregation.")
        results_df = baseline['pool_df']

    # combine all results in dataframe
    results_df['train_error'] = float('nan')
    results_df['validation_error'] = float('nan')

    # compute error metrics
    W = np.stack(results_df['coefficients'].values)
    results_df['train_error'] = compute_error_rate_from_coefficients(W = W, X = data['X'], Y = data['Y'])
    if 'X_validation' in data and 'Y_validation' in data:
        results_df['validation_error'] = compute_error_rate_from_coefficients(W = W, X = data['X_validation'], Y = data['Y_validation'])

    # info to return
    now = datetime.now()
    out = {
        #
        'date': now.strftime("%y_%m_%d_%H_%M"),
        'info': info,
        'training_info': {'baseline': baseline['info'], 'flipped_info': [f['info'] for f in flipped]},
        #
        'data_file': baseline['data_file'],
        'baseline_results_file': baseline_results_file,
        'flipped_results_files': flipped_results_files,
        'results_df': results_df,
        'n_missing': len(flipped_partition['missing_parts']),
        #
        }

    return out
