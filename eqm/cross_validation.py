import re
import numpy as np
from sklearn.model_selection import StratifiedKFold

INNER_CV_SEPARATOR = "_"
FOLD_ID_FORMAT = 'K%02dN%02d'
INNER_ID_FORMAT = 'F%02dK%02d'
INNER_FOLD_ID_FORMAT = '%s%s%s' % (FOLD_ID_FORMAT, INNER_CV_SEPARATOR, INNER_ID_FORMAT)
TRIVIAL_FOLD_ID = "K01N01"
OUTER_CV_PATTERN = "^K[0-9]{2}N[0-9]{2}$"
INNER_CV_PATTERN = "^K[0-9]{2}N[0-9]{2}_F[0-9]{2}K[0-9]{2}$"
OUTER_CV_PARSER = re.compile(OUTER_CV_PATTERN)
INNER_CV_PARSER = re.compile(INNER_CV_PATTERN)


#### filtering data ####

def filter_data_to_fold(data, cvindices, fold_id = TRIVIAL_FOLD_ID, fold_num = 0, include_validation = False, include_test = False):
    """
    :param data: data object
    :param cvindices: cvindices dict
    :param fold_id: fold ID
    :param fold_num: # of fold
    :param include_validation: add X_validation, Y_validation in filtered data
    :param include_test:add X_test, Y_test in filtered data
    :return:
    """

    data['fold_id'] = fold_id
    data['fold_num'] = fold_num

    if fold_id == TRIVIAL_FOLD_ID:
        return data

    fold_is_for_inner_cv = is_inner_fold_id(fold_id)

    # only include validation if the fold number is positive
    include_validation = include_validation and fold_num > 0

    # only include test set if we have inner cv
    include_test = include_test and fold_is_for_inner_cv


    if fold_is_for_inner_cv:

        total_folds, replicate_idx, fold_idx_inner_cv, total_folds_inner_cv = parse_fold_id(fold_id)
        outer_fold_id = to_fold_id(total_folds, replicate_idx)

        outer_fold_idx = cvindices[outer_fold_id]
        inner_fold_idx = cvindices[fold_id]

        test_idx = outer_fold_idx == fold_idx_inner_cv
        train_and_valid_idx = ~test_idx
        inner_cv_train_idx = fold_num == inner_fold_idx
        inner_cv_valid_idx = ~inner_cv_train_idx

        train_idx = ~test_idx
        train_idx[train_and_valid_idx] = inner_cv_train_idx

        valid_idx = ~test_idx
        valid_idx[train_and_valid_idx] = inner_cv_valid_idx

    else:

        valid_idx = fold_num == cvindices[fold_id]
        train_idx = ~valid_idx
        test_idx = False * train_idx

    N = data['X'].shape[0]
    assert len(test_idx) == N
    assert len(valid_idx) == N
    assert len(train_idx) == N
    assert np.all(np.logical_or(train_idx, valid_idx, test_idx))

    if include_validation:
        data['X_validation'] = data['X'][valid_idx, ]
        data['Y_validation'] = data['Y'][valid_idx]
        data['sample_weights_validation'] = data['sample_weights'][valid_idx]

    if include_test:
        data['X_test'] = data['X'][test_idx,]
        data['Y_test'] = data['Y'][test_idx]
        data['sample_weights_test'] = data['sample_weights'][test_idx]

    data['X'] = data['X'][train_idx,]
    data['Y'] = data['Y'][train_idx]
    data['sample_weights'] = data['sample_weights'][train_idx]

    return data


def split_data_by_cvindices(data, cvindices, fold_id = TRIVIAL_FOLD_ID, fold_num = 0, fold_num_test = -1):
    """
    :param data:
    :param cvindices:
    :param fold_id:
    :param fold_num: fold for test evaluation; set to 0 to not select anything
    :param fold_num_test: fold for internal tuning
    :return:
    """
    assert isinstance(fold_num, int)
    assert isinstance(fold_num_test, int)
    assert fold_num >= 0
    assert fold_num_test >= -1
    assert fold_num_test != fold_num

    N = data['X'].shape[0]
    data['fold_id'] = fold_id
    data['fold_num'] = fold_num
    data['fold_num_test'] = fold_num_test

    folds = cvindices[fold_id]
    valid_idx = np.isin(folds, fold_num)
    test_idx = np.isin(folds, fold_num_test)
    train_idx = np.isin(folds, [fold_num, fold_num_test], invert = True)
    assert len(test_idx) == N
    assert len(valid_idx) == N
    assert len(train_idx) == N
    assert np.all(train_idx + valid_idx + test_idx)

    if any(valid_idx):
        data['X_validation'] = data['X'][valid_idx, ]
        data['Y_validation'] = data['Y'][valid_idx]
        data['sample_weights_validation'] = data['sample_weights'][valid_idx]

    if any(test_idx):
        data['X_test'] = data['X'][test_idx,]
        data['Y_test'] = data['Y'][test_idx]
        data['sample_weights_test'] = data['sample_weights'][test_idx]

    data['X'] = data['X'][train_idx,]
    data['Y'] = data['Y'][train_idx]
    data['sample_weights'] = data['sample_weights'][train_idx]
    return data


#### fold id parsing / validation ####

def parse_fold_id(fold_id):

    """
    #todo add spec
    :param fold_id:
    :return:
    """
    fold_id_elements = fold_id.split(INNER_CV_SEPARATOR)
    outer_fold_id = fold_id_elements[0]
    total_folds = int(outer_fold_id[1:3])
    replicate_idx = 1
    fold_idx_inner_cv = None
    total_folds_inner_cv = None

    if len(outer_fold_id) >= 4:
        replicate_idx = int(outer_fold_id[4:6])

    if len(fold_id_elements) > 1:
        inner_fold_id = fold_id_elements[1]
        fold_idx_inner_cv = int(inner_fold_id[1:3])
        total_folds_inner_cv = int(inner_fold_id[4:6])
        error_msg = "inner cv fold_id %s is for fold # %d which does not exist for outer fold_id %s" % (fold_id, fold_idx_inner_cv, outer_fold_id)
        assert fold_idx_inner_cv <= total_folds, error_msg

    return total_folds, replicate_idx, fold_idx_inner_cv, total_folds_inner_cv


def validate_fold_id(fold_id):
    """
    #todo add spec
    :param fold_id:
    :return:
    """

    fold_id = fold_id.strip().upper()
    parsed = INNER_CV_PARSER.match(fold_id)

    if parsed is not None:
        return parsed.string

    #must be outer-cv
    parsed = OUTER_CV_PARSER.match(fold_id)
    assert parsed is not None, \
        'invalid fold_id: %s' % fold_id

    return parsed.string


def is_inner_fold_id(fold_id):
    """
    #todo add spec
    :param fold_id:
    :return:
    """
    parsed = INNER_CV_PARSER.match(fold_id)
    return parsed is not None


def to_fold_id(total_folds, replicate_idx = 1, fold_idx_inner_cv = None, total_folds_inner_cv = None):

    total_folds = int(total_folds)
    replicate_idx = int(replicate_idx)

    assert total_folds >= 1
    assert replicate_idx >= 1

    if fold_idx_inner_cv is None:
        fold_id = FOLD_ID_FORMAT % (total_folds, replicate_idx)
    else:
        fold_idx_inner_cv = int(fold_idx_inner_cv)
        total_folds_inner_cv = int(total_folds_inner_cv)
        assert total_folds_inner_cv >= 1
        assert fold_idx_inner_cv >= 1
        assert total_folds >= fold_idx_inner_cv
        fold_id = INNER_FOLD_ID_FORMAT % (total_folds, replicate_idx, fold_idx_inner_cv, total_folds_inner_cv)

    fold_id = validate_fold_id(fold_id)
    return fold_id


#### fold generation ####


def generate_folds(n_samples, n_folds = 5, seed = None):

    # set random seed
    if seed is not None:
        np.random.seed(seed)

    # generate unscrambled fold indices (tiled copies of (1,..,K))
    n_copies = np.ceil(float(n_samples) / float(n_folds))
    folds = np.tile(np.arange(1, n_folds + 1, dtype = int), int(n_copies))
    to_keep = np.arange(n_samples)
    folds = folds[to_keep]

    # defensive reshape
    folds = np.random.permutation(folds)
    folds = np.array(folds, dtype = int).flatten()

    # check output
    folds = validate_folds(folds = folds, fold_id = None, n_samples = n_samples)
    return folds


def generate_cvindices(n_samples, total_folds_for_cv = [1, 2, 3, 5, 10], total_folds_for_inner_cv = [2, 3, 5], replicates = 3, seed = None):

    assert n_samples >= 1
    assert replicates >= 1

    if seed is not None:
        np.random.seed(seed)

    cvindices = dict()
    if 1 in total_folds_for_cv:
        cvindices[TRIVIAL_FOLD_ID] = generate_folds(n_samples, n_folds = 1)
        total_folds_for_cv.remove(1)

    for k in total_folds_for_cv:
        for n in range(1, replicates + 1):

            fold_id = to_fold_id(total_folds = k, replicate_idx = n)
            cvindices[fold_id] = generate_folds(n_samples, n_folds = k)

            # generate inner folds for k-fold cv
            for f in range(1, k + 1):
                fold_idx = np.not_equal(cvindices[fold_id], f)
                n_samples_fold = np.sum(fold_idx)
                for l in total_folds_for_inner_cv:
                    inner_fold_id = to_fold_id(total_folds = k, replicate_idx = n, fold_idx_inner_cv = f, total_folds_inner_cv = l)
                    cvindices[inner_fold_id] = generate_folds(n_samples = n_samples_fold, n_folds = l)

    cvindices = validate_cvindices(cvindices)
    return cvindices


#### stratified fold generation ####

def generate_stratified_folds(X, strata, n_folds):

    strata = np.array(strata).flatten()
    n_samples = len(strata)
    assert n_folds >= 1
    assert n_samples >= 1
    assert X.shape[0] == n_samples

    folds = np.zeros(n_samples, dtype = np.int)
    fold_generator = StratifiedKFold(n_splits = n_folds, shuffle = True)
    k = 1
    for train_idx, test_idx in fold_generator.split(X, strata):
        folds[test_idx] = k
        if k < n_folds:
            k = k + 1

    assert np.all(np.greater(folds, 0))
    return folds


def generate_stratified_cvindices(X, strata, total_folds_for_cv = [1, 2, 3, 5, 10], total_folds_for_inner_cv = [2, 3, 5], replicates = 3, group_names = None, seed = None):


    n_samples = len(strata)
    assert n_samples >= 1
    assert replicates >= 1
    assert X.shape[0] == n_samples

    if seed is not None:
        np.random.seed(seed)

    cvindices = dict()
    if 1 in total_folds_for_cv:
        cvindices[TRIVIAL_FOLD_ID] = np.ones(n_samples)
        total_folds_for_cv.remove(1)

    for k in total_folds_for_cv:
        for n in range(1, replicates + 1):

            fold_id = to_fold_id(total_folds = k, replicate_idx = n)
            cvindices[fold_id] = generate_stratified_folds(X, strata, n_folds = k)
            # generate inner folds for k-fold cv
            for f in range(1, k + 1):

                fold_idx = np.not_equal(cvindices[fold_id], f)
                Xf = X[fold_idx, :]
                Sf = strata[fold_idx]

                for l in total_folds_for_inner_cv:
                    inner_fold_id = to_fold_id(total_folds = k, replicate_idx = n, fold_idx_inner_cv = f, total_folds_inner_cv = l)
                    cvindices[inner_fold_id] = generate_stratified_folds(Xf, Sf, n_folds = l)

    cvindices = validate_cvindices(cvindices, stratified = True)
    return cvindices


#### fold validation ####


def validate_folds(folds, fold_id = None, n_samples = None, stratified = True):
    """

    :param folds:
    :param fold_id:
    :param n_samples:
    :param stratified:
    :return:
    """

    # reshape folds
    folds = np.array(folds, dtype = 'int').flatten()
    assert folds.ndim == 1

    # check length
    if n_samples is None:
        assert len(folds) >= 1
    else:
        assert len(folds) == n_samples

    # check fold values
    fold_values_min = np.min(folds)
    fold_values_max = np.max(folds)
    assert fold_values_min == 1
    assert fold_values_max <= len(folds)

    fold_values, fold_counts = np.unique(folds, return_counts = True)
    fold_values_expected = np.arange(1, fold_values_max + 1)

    assert np.all(fold_values == fold_values_expected), \
        'fold indices %s are not consecutive' % str(fold_values)

    if not stratified:
        assert np.min(fold_counts) >= np.max(fold_counts) - 1, \
            'imbalanced folds: max (points/fold) must be within min (points/fold)'

    # check that fold id matches fold content
    if fold_id is not None:
        (total_folds, replicate_idx, fold_idx_inner_cv, total_folds_inner_cv) = parse_fold_id(fold_id)
        if is_inner_fold_id(fold_id):
            assert total_folds >= 1
            assert total_folds >= fold_idx_inner_cv
            assert np.isin(fold_idx_inner_cv, np.arange(1, total_folds + 1))
            assert total_folds_inner_cv == fold_values_max
        else:
            assert total_folds == fold_values_max
            assert replicate_idx >= 1
            assert fold_idx_inner_cv is None
            assert total_folds_inner_cv is None

    return folds


def validate_cvindices(cvindices, stratified = True):
    """
    will drop fold_ids for inner cv if the corresponding outer_cv fold_id does not exist
    :param cvindices:
    :return:
    """

    #check that fold_ids are valid
    all_fold_ids = list(cvindices.keys())
    for fold_id in all_fold_ids:
        try:
            validated_id = validate_fold_id(fold_id)
            if validated_id != fold_id:
                cvindices[validated_id] = cvindices.pop(fold_id)
        except AssertionError:
            cvindices.pop(fold_id)

    all_fold_ids = list(cvindices.keys())
    outer_ids = list(filter(OUTER_CV_PARSER.match, all_fold_ids))
    inner_ids = list(filter(INNER_CV_PARSER.match, all_fold_ids))

    if len(outer_ids) == 0:
        assert len(inner_ids) == 0
        return cvindices

    #at this point cvindices must have at least one outer id
    validated_indices = dict()
    n_samples = len(cvindices[outer_ids[0]])
    for fold_id in outer_ids:
        try:
            validated_indices[fold_id] = validate_folds(cvindices[fold_id], fold_id, n_samples, stratified)
        except AssertionError:
            print('could not validate fold: %s' % fold_id)
            pass


    for fold_id in inner_ids:
        outer_id, _ = fold_id.split(INNER_CV_SEPARATOR)
        if outer_id in outer_ids:
            try:
                validated_indices[fold_id] = validate_folds(cvindices[fold_id], fold_id, stratified = stratified)
            except AssertionError:
                print('could not validate fold: %s' % fold_id)
                pass


    return validated_indices


#### filtering cross-validation folds ####


def filter_cvindices(cvindices, total_folds_for_cv = [1, 5, 10], total_folds_for_inner_cv = [3, 5 ,10], replicates = float('inf')):
    """
    #todo add spec
    :param cvindices:
    :param total_folds:
    :param total_folds_inner_cv:
    :param n_replicates:
    :return:
    """
    all_fold_ids = list(cvindices.keys())
    n_fold_ids = len(all_fold_ids)
    replicates = int(min(replicates, n_fold_ids))
    assert replicates >= 1

    replicate_idx = np.arange(1, replicates + 1, dtype = int)

    to_keep = set()

    for k in total_folds_for_cv:

        fold_values = np.arange(1, k + 1, dtype = int)

        for n in replicate_idx:

            fold_id = to_fold_id(total_folds = k,
                                 replicate_idx = n)

            if fold_id in all_fold_ids:
                all_fold_ids.remove(fold_id)
                to_keep.add(fold_id)

            for inner_k in total_folds_for_inner_cv:

                for f in fold_values:

                    inner_fold_id = to_fold_id(total_folds = k,
                                               replicate_idx = n,
                                               fold_idx_inner_cv = f,
                                               total_folds_inner_cv = inner_k)

                    if inner_fold_id in all_fold_ids:
                        all_fold_ids.remove(inner_fold_id)
                        to_keep.add(inner_fold_id)

    filtered_cvindices = {fold_id:cvindices[fold_id] for fold_id in to_keep if fold_id in cvindices}
    return filtered_cvindices
