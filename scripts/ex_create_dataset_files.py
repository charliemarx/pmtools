from eqm.paths import *
from eqm.data import *
from eqm.cross_validation import generate_stratified_cvindices
import numpy as np


def create_dataset_file(data_name, test=False):
    ## load data from disk
    data_file_name = '%s/%s_processed.csv' % (data_dir, data_name)
    data_file_name = Path(data_file_name)
    random_seed = 1337
    np.random.seed(seed = random_seed)

    # load dataset
    data = load_data_from_csv(data_file_name)

    if test:
        data['X'] = data['X'][:500]
        data['Y'] = data['Y'][:500]
        data['sample_weights'] = data['sample_weights'][:500]

    # correct for class imbalance
    data = oversample_minority_class(data, random_state = random_seed)

    # if row_id in data, save to disk then drop from data
    if "recidivism_" in data_name or "pretrial_" in data_name:
        # the file where we will save the row_ids
        row_id_name = data_dir / ('%s_row_id' % data_name)

        idx = get_index_of(data, 'row_id')
        row_id = data['X'][:, idx]
        np.savetxt(fname=row_id_name.with_suffix('.csv'), X=row_id, delimiter=',', header='row_id')
        data = remove_variable(data, "row_id")

    # generate cv indices
    cvindices = generate_stratified_cvindices(X = data['X'],
                                              strata = data['Y'],
                                              total_folds_for_cv = [1, 3, 4, 5],
                                              total_folds_for_inner_cv = [5],
                                              replicates = 3,
                                              seed = random_seed)

    # todo: sanity check that stratified sampling works (# of Y = +1 ~= # Y = -1 within fold)
    #save data and cv indices to disk
    if test:
        outfile = '%s/%s_small_processed.pickle' % (data_dir, data_name)
    else:
        outfile = data_file_name.with_suffix('.pickle')
    save_data(file_name = outfile,
              data = data,
              cvindices = cvindices,
              overwrite = True,
              stratified = True,
              check_save = True)

    print('data_name: %s saved' % data_name)

if __name__ == "__main__":
    create_dataset_file("compas_arrest", test=True)