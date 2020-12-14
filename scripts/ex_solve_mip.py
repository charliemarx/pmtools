from eqm.paths import *
from eqm.data import *
from eqm.cross_validation import filter_data_to_fold
from eqm.mip import ZeroOneLossMIP
from eqm.solution_pool import SolutionPool
from eqm.cplex_mip_helper import *

# general things we want to import for experiments
import time
from datetime import datetime
import dill
import pandas as pd
pd.set_option('display.max_columns', 30)
np.set_printoptions(precision = 2)
np.set_printoptions(suppress = True)

# dashboard
data_name = 'compas_arrest'
random_seed = 1337
time_limit_global = 60
time_limit_instance = 60
initialize_mip = True
custom_mip_params = True
search_global_equivalent = False
max_iterations = 2  #shorten loop during development
fold_id = 'K01N01'
fold_num = 1

# setup seed
np.random.seed(seed = random_seed)

## load data from disk
data_file_name = '%s/%s_processed.csv' % (data_dir, data_name)
data_file_name = Path(data_file_name)
data, cvindices = load_processed_data(file_name = data_file_name.with_suffix('.pickle'))
data = filter_data_to_fold(data, cvindices = cvindices, fold_id = fold_id, fold_num = fold_num, include_validation=True)

# compress dataset into distinct feature vectors and counts
compressed = compress_data(data)
U, N_pos, N_neg, x_to_u_idx, u_to_x_idx = tuple(compressed[var] for var in ('U', 'N_pos', 'N_neg', 'x_to_u_idx', 'u_to_x_idx'))

selection = pd.DataFrame({'n_pos': N_pos, 'n_neg': N_neg, 'idx': x_to_u_idx})

# solve zero-one loss MIP
mip = ZeroOneLossMIP(data, print_flag = True, parallel_flag = True, random_seed = random_seed)
if custom_mip_params:
    mip.mip = set_mip_parameters(mip.mip, CPX_MIP_PARAMETERS)
out = mip.solve(time_limit = time_limit_global)
mip.check_solution()
global_coefs = mip.coefficients
global_lb = out['lowerbound']
global_objval = out['objval']
h = mip.get_classifier()

# initialize solution pool
pool = SolutionPool(mip = mip)

# search for equivalent models
if search_global_equivalent:
    equivalent_output, pool = mip.enumerate_equivalent_solutions(pool, time_limit = time_limit_global)
    print('{} global equivalent models found.'.format(len(equivalent_output)))

# generate additional solutions using populate
n_points = data['X'].shape[0]
mip.populate(max_gap = 0, time_limit = time_limit_global)
mip.populate(max_gap = n_points // 2, time_limit = time_limit_global)
pool.add_from_mip(mip)

mip.mip.parameters.emphasis.mip.set(3)
mip.set_total_mistakes(lb = global_lb)
results = []
total_iterations = U.shape[0]
start_time = time.process_time()
for k, x in enumerate(U):

    print('iteration %d/%d' % (k, total_iterations))
    print('solution pool size: %d' % pool.size)

    yhat = int(h.predict(x[None, :]))

    # adjust prediction constraints
    mip.clear_prediction_constraints()
    mip.add_prediction_constraint(x = x, yhat = -yhat, name = 'pred_constraint')

    # initialize model
    # todo: remove this once you can tell it doesn't do worse
    if initialize_mip:
        good_pool = pool.get_solutions_with_pred(x, -yhat)
        if good_pool.size > 0:
            s = good_pool.get_best_solution()
            mip.add_initial_solution(solution = s['solution'], objval = s['objval'], name= 'init_from_pred_cons')
            print('initialized\nobjval:{}'.format(s['objval']))

    # solve MIP
    out = mip.solve(time_limit = time_limit_instance)
    mip.check_solution()

    # update solution pool
    pool.add_from_mip(mip, prediction_constraint = (x, yhat))

    # update out
    out.update(
            {'k': k,  # k is the indices of "distinct" points
             'i': np.flatnonzero(u_to_x_idx == k).tolist(),  #{i = 0,...n s.t. x[i] == U[k]}
             'x': x, #x = U[k]
             'n_pos': N_pos[k], #n_pos = {i: X[i] = x, and y[i] = +}
             'n_neg': N_neg[k], #n_pos = {i: X[i] = x, and y[i] = -}
             'coefficients': mip.coefficients,
             'elapsed_time': time.process_time() - start_time}
            )

    results.append(out)

    if k > max_iterations:
        break

results_df = pd.DataFrame(results)

# todo: save key results as a pickle file
now = datetime.now()
time_str = now.strftime("%y_%m_%d_%H_%M")
output = {
    'date': str(now),
    'data_name': data_name,
    'data_file_name': data_file_name,
    'results_df': results_df,
    'fold_id': fold_id,
    #'baseline_model':
    #'solution_pool':
    #todo later: add script information here
    }

results_file_name = '%s_%s_results.pkl' % (data_name, time_str)
results_file_name = results_dir / results_file_name
with open(results_file_name, 'wb') as f:
    dill.dump(output, f)


def acc_from_coefs(W, X, Y):
    preds = np.sign(np.dot(W, X.T))
    accs = np.mean(preds == Y, axis=1)
    return accs


# get flipped models
csv_df = results_df[['n_pos', 'n_neg', 'i', 'upperbound', 'lowerbound', 'coefficients']].copy()
csv_df['model_type'] = 'flipped'

# get alternative models
alt_df = pool.get_df()
alt_df = alt_df[['objval', 'lowerbound', 'coefficients']]
alt_df.columns = ['upperbound', 'lowerbound', 'coefficients']
alt_df['model_type'] = 'alternative'

# get optimal model
opt_df = pd.DataFrame({'upperbound': global_objval,
          'lowerbound': global_lb,
          'coefficients': [global_coefs],
          'model_type': 'global'})

# group all the models
csv_df = csv_df.append([alt_df, opt_df], ignore_index=True, sort=False).reset_index(drop=True)

# reformat coefficients in df
W = np.stack(csv_df['coefficients'].values)
coef_df = pd.DataFrame({'coef_%02d' % j: W[:, j] for j in range(W.shape[1])})
csv_df = pd.concat((csv_df, coef_df), axis = 1, sort=True)
csv_df = csv_df.drop('coefficients', axis=1)

# add additional info
csv_df['train_acc'] = acc_from_coefs(W, data['X'], data['Y'])
# csv_df['validation_acc'] = acc_from_coefs(W, data['X_validation'], data['Y_validation'])
csv_df['data_name'] = data_name
csv_df['fold_id'] = fold_id
csv_df['n_points'] = n_points

# save results df
csv_file_name = '%s_%s_plot_results.csv' % (data_name, time_str)
csv_file_name = results_dir / csv_file_name
csv_df.to_csv(csv_file_name)

