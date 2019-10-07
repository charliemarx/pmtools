from eqm.solution_pool import SolutionPool

from eqm.paths import *
from eqm.data import *
from eqm.cross_validation import filter_data_to_fold
from eqm.mip import ZeroOneLossMIP

def test_solution_pool():

    s_pool = SolutionPool()

    data_name = 'breastcancer'
    data_file_name = '%s/%s_processed.csv' % (data_dir, data_name)
    data_file_name = Path(data_file_name)

    random_seed = 1337
    np.random.seed(seed=random_seed)

    ## load data from disk
    data, cvindices = load_processed_data(file_name=data_file_name.with_suffix('.pickle'))
    data = filter_data_to_fold(data, cvindices=cvindices, fold_id='K05N01', fold_num=1)

    # about the folds
    # fold_id = K[total # of folds]N[replicate number]
    # 'K05N01' has 5 total folds, N01 means this is the first replicate
    # fold_num = 1 means that we use fold # 1/5 as the test set
    # fold_num = 2 means that we use fold # 2/5 as the test set
    # fold_num = 0 means that we don't use a fold as the test set (i.e., so filtering with fold_num = 0 just returns the full training dataset)

    mip = ZeroOneLossMIP(data)
    mip.print_flag = True
    mip.set_parallel(True)
    out = mip.solve(time_limit=10)

    # add solutions manually
    s_pool.add(solution=[1,2,3], coefs=[1,2], objval=5, lowerbound=4, prediction_constraint=('x', 'y'))
    s_pool.add(solution=[[0,0,0], [1,1,1]], coefs=[[1,0], [1,1]], objval=[0, 1], lowerbound=[2, 3])
    preds = s_pool.get_preds([-2, 3])
    small_pool = s_pool.get_solutions_with_pred([-2, 3], 1)

    assert all(preds == [1, -1, 1])
    assert small_pool._df.shape[0] == 2
    s_pool.clear()

    s_pool.add_from_mip(mip)
    print(s_pool)
    s_pool.clear()

    s_pool.add_from_mip(mip, add_full_solution_pool=True)
    print(s_pool)
