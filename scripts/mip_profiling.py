from eqm.mip import ZeroOneLossMIP
from eqm.solution_pool import SolutionPool
from scripts.reporting import *
from eqm.data import compress_data
from eqm.cplex_mip_helper import *


# save experiment name + results_df + console output

######################################
# - using initialization
use_init = False  # best is True

# - changing prediction constraint ind_type from 3 to 1
### takes twice as long with ind_type=1. Much better to use ind_type=3
indtype = 3  # best is 3

# - implementing the prediction constraint as only z[i] + 1
# ?????

# - implementing the prediction constraint as "z[i] = +1" + current constraint on score
# - using Big-M constraints instead of indicator constraints

use_custom_params = False  # best is True

# - setting lowerbound / upperbound for cplex directly reduces runtime (without "set_total_mistakes")
# options: 'both', 'direct', 'indirect'
bound_strategy = 'indirect'  # 'indirect' better than 'both' better than 'direct'
######################################



# dashboard
data_name = 'test_adult_binarized'
random_seed = 1337
time_limit_global = 60
time_limit_instance = 60
initialize_mip = use_init
search_global_equivalent = True
max_iterations = 1e8  #shorten loop during development
np.random.seed(seed = random_seed)

## load data from disk
data_file_name = '%s_processed.csv' % data_name
data_file_name = data_dir / data_file_name
data, cvindices = load_processed_data(file_name = data_file_name.with_suffix('.pickle'))

# compress dataset into distinct feature vectors and counts
compressed = compress_data(data)
U, N_pos, N_neg, x_to_u_idx, u_to_x_idx = tuple(compressed[var] for var in ('U', 'N_pos', 'N_neg', 'x_to_u_idx', 'u_to_x_idx'))

# solve zero-one loss MIP
mip = ZeroOneLossMIP(data, print_flag = False, parallel_flag = False, ind_cons_type = indtype, random_seed = random_seed)
if use_custom_params:
    mip.mip = set_mip_parameters(mip.mip, CPX_MIP_PARAMETERS)
out = mip.solve(time_limit = time_limit_global)
mip.check_solution()
global_lb = out['lowerbound']
global_objval = out['objval']
h = mip.get_classifier()

# get mistake list
mistake_names = mip.names['mistakes_pos'] + mip.names['mistakes_neg']
z = mip.solution.get_values(mistake_names)

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
mip.set_total_mistakes(lb = global_lb, strategy=bound_strategy)
results = []
total_iterations = U.shape[0]

start_time = time.process_time()

for k, x in enumerate(U[1:2]):

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
    start_time = time.process_time()
    out = mip.solve(time_limit = time_limit_instance)
    runtime = start_time - time.process_time()

    mip.check_solution()


    # update solution pool
    pool.add_from_mip(mip, prediction_constraint = (x, yhat))

    # update out
    out.update({'runtime': runtime,
             'k': k,  # k is the indices of "distinct" points
             'i': np.flatnonzero(u_to_x_idx == k).tolist(),  #{i = 0,...n s.t. x[i] == U[k]}
             'n_pos': N_pos[k],
             'n_neg': N_neg[k],
             'x': x
             })

    results.append(out)

    if k > max_iterations:
        break

elapsed = time.process_time() - start_time
print('Time elapsed: %s' % elapsed)

results_df = pd.DataFrame(results)
csv_df = results_df[['runtime', 'gap', 'status_code', 'lowerbound', 'upperbound', 'iterations', 'nodes_processed', 'nodes_remaining', 'status']]

# to check
# mean runtime
# runtime >= 5
# max runtime
# % solved to optimality
# % gap >= 10%
